"""
Проверка утечек между сплитами (train/val/test):
- пересечения по имени файла (basename)
- пересечения по содержимому (SHA1)
- (опционально) поиск near-duplicates по p-hash (8x8) и Hamming

CSV формат: требуется колонка `path` (абсолютная или относительная).
Относительные пути считаются от --root.

Пример:
python -m src.analysis.check_leakage ^
  --splits train=data/processed/splits/train.csv val=data/processed/splits/val.csv test=data/processed/splits/test.csv ^
  --root . ^
  --phash --phash-threshold 4 --phash-samples 5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
try:
    import cv2
except Exception:
    cv2 = None
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SplitData:
    name: str
    paths: List[Path]
    basenames: List[str]
    sha1: Dict[str, str]  # str(path) -> sha1
    phash: np.ndarray | None  # uint64 array


def parse_thresholds(spec: str) -> List[int]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Пустой список порогов для --phash-sweep.")
    out: List[int] = []
    if "," in spec:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        for part in parts:
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s), int(end_s)
                if end < start:
                    start, end = end, start
                out.extend(range(start, end + 1))
            else:
                out.append(int(part))
    elif "-" in spec:
        start_s, end_s = spec.split("-", 1)
        start, end = int(start_s), int(end_s)
        if end < start:
            start, end = end, start
        out = list(range(start, end + 1))
    else:
        out = [int(spec)]
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Проверка утечек между сплитами CSV.")
    p.add_argument(
        "--splits",
        type=str,
        nargs="+",
        required=True,
        help="Формат split_name=csv_path, например: train=... val=... test=...",
    )
    p.add_argument("--root", type=Path, default=PROJECT_ROOT, help="База для относительных путей в CSV.")
    p.add_argument("--phash", action="store_true", help="Считать p-hash (8x8) и искать near-duplicates.")
    p.add_argument("--phash-threshold", type=int, default=0, help="Порог Hamming для p-hash (0 = только точные совпадения).")
    p.add_argument("--phash-samples", type=int, default=5, help="Сколько примеров near-duplicate показать для p-hash.")
    p.add_argument(
        "--phash-sweep",
        type=str,
        default=None,
        help="Подбор порогов p-hash. Пример: 0,1,2,3,4,5 или диапазон 0-6.",
    )
    p.add_argument(
        "--phash-sweep-chunk",
        type=int,
        default=512,
        help="Размер чанка при подсчете количества совпадений для sweep.",
    )
    p.add_argument(
        "--write-clean",
        type=Path,
        default=None,
        help=(
            "Записать очищенные CSV (удаление точных дубликатов по SHA1). "
            "Приоритет сохранения — порядок --splits."
        ),
    )
    return p.parse_args()


def read_csv_paths(csv_path: Path, root: Path) -> List[Path]:
    out: List[Path] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames:
            raise ValueError(f"{csv_path} не содержит колонку 'path'")
        for row in reader:
            p = Path(row["path"])
            if not p.is_absolute():
                p = (root / p).resolve()
            out.append(p)
    return out


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path: Path, size: int = 32, hash_size: int = 8) -> int:
    img = Image.open(path).convert("L").resize((size, size))
    pixels = np.asarray(img, dtype=np.float32)
    if cv2 is not None:
        dct = cv2.dct(pixels)
    else:
        # fallback: average hash (если нет cv2)
        small = img.resize((hash_size, hash_size))
        pixels = np.asarray(small, dtype=np.float32)
        median = np.median(pixels)
        bits = pixels > median
        flat = bits.flatten().astype(np.uint8)
        packed = 0
        for b in flat:
            packed = (packed << 1) | int(b)
        return packed

    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    bits = dct_low > median
    flat = bits.flatten().astype(np.uint8)
    packed = 0
    for b in flat:
        packed = (packed << 1) | int(b)
    return packed


def load_split(name: str, csv_path: Path, root: Path, do_phash: bool) -> SplitData:
    paths = read_csv_paths(csv_path, root)
    basenames = [p.name for p in paths]
    sha1_map = {str(p): sha1_file(p) for p in paths}
    phash_arr = None
    if do_phash:
        phash_arr = np.array([compute_phash(p) for p in paths], dtype=np.uint64)
    return SplitData(name=name, paths=paths, basenames=basenames, sha1=sha1_map, phash=phash_arr)


_BITCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(a[:, None], b[None, :])
    if hasattr(np, "bit_count"):
        return np.bit_count(xor).astype(np.int16)
    # fallback: суммируем биты по байтам
    bytes_view = xor.view(np.uint8)
    return _BITCOUNT_LUT[bytes_view].sum(axis=-1).astype(np.int16)


def phash_matches(split_a: SplitData, split_b: SplitData, threshold: int, sample_n: int) -> List[Tuple[str, str, int]]:
    if split_a.phash is None or split_b.phash is None:
        return []
    a_hash = split_a.phash
    b_hash = split_b.phash
    matches: List[Tuple[str, str, int]] = []

    if threshold <= 0:
        set_b = {}
        for h, p in zip(b_hash, split_b.paths):
            set_b.setdefault(int(h), []).append(p.name)
        for h, p in zip(a_hash, split_a.paths):
            if int(h) in set_b:
                for q in set_b[int(h)]:
                    matches.append((p.name, q, 0))
                    if len(matches) >= sample_n:
                        return matches
        return matches

    b_arr = b_hash
    b_names = np.array([p.name for p in split_b.paths])
    for idx, h in enumerate(a_hash):
        dist = hamming_vec(np.array([h], dtype=np.uint64), b_arr)[0]
        mask = dist <= threshold
        if mask.any():
            for d, q in zip(dist[mask], b_names[mask]):
                matches.append((split_a.paths[idx].name, str(q), int(d)))
                if len(matches) >= sample_n:
                    return matches
    return matches


def phash_counts_by_threshold(
    split_a: SplitData,
    split_b: SplitData,
    thresholds: List[int],
    chunk: int,
) -> Dict[int, int]:
    if split_a.phash is None or split_b.phash is None:
        return {t: 0 for t in thresholds}
    a_hash = split_a.phash
    b_hash = split_b.phash
    thresholds = sorted(set(thresholds))
    counts = np.zeros(len(thresholds), dtype=np.int64)

    for i in range(0, len(a_hash), chunk):
        dist = hamming_vec(a_hash[i : i + chunk], b_hash)
        for idx, t in enumerate(thresholds):
            counts[idx] += (dist <= t).sum()

    return {t: int(c) for t, c in zip(thresholds, counts)}


def write_clean_splits(split_map: Dict[str, SplitData], args: argparse.Namespace) -> None:
    out_dir: Path = args.write_clean
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes = set()

    for spec in args.splits:
        name, path = spec.split("=", 1)
        sd = split_map[name]
        csv_path = Path(path)
        keep_rows: List[Dict[str, str]] = []
        removed = 0
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                p = Path(row["path"])
                if not p.is_absolute():
                    p = (args.root / p).resolve()
                h = sd.sha1.get(str(p))
                if h is None:
                    h = sha1_file(p)
                if h in seen_hashes:
                    removed += 1
                    continue
                seen_hashes.add(h)
                keep_rows.append(row)
        out_path = out_dir / csv_path.name
        with out_path.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(keep_rows)
        print(f"[clean] {name}: сохранено {len(keep_rows)}, удалено {removed}, файл {out_path}")


def main() -> None:
    args = parse_args()
    do_phash = args.phash or (args.phash_sweep is not None)
    split_map: Dict[str, SplitData] = {}
    for spec in args.splits:
        if "=" not in spec:
            raise ValueError("Ожидается формат name=path.csv")
        name, path = spec.split("=", 1)
        split_map[name] = load_split(name, Path(path), args.root, do_phash)

    names = list(split_map.keys())
    print("Размеры сплитов:")
    for n in names:
        s = split_map[n]
        print(f"  {n}: {len(s.paths)} файлов, уникальных имён {len(set(s.basenames))}")

    print("\nПересечения по basename:")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa, sb = split_map[a], split_map[b]
            inter = set(sa.basenames) & set(sb.basenames)
            print(f"  {a} vs {b}: {len(inter)}")
            if inter:
                print("    пример:", list(sorted(inter))[:5])

    print("\nПересечения по содержимому (SHA1):")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa, sb = split_map[a], split_map[b]
            inter = set(sa.sha1.values()) & set(sb.sha1.values())
            print(f"  {a} vs {b}: {len(inter)}")
            if inter:
                sample = []
                for h in list(inter)[:5]:
                    pa = [k for k, v in sa.sha1.items() if v == h][0]
                    pb = [k for k, v in sb.sha1.items() if v == h][0]
                    sample.append((Path(pa).name, Path(pb).name))
                print("    пример:", sample)

    if args.phash_sweep:
        thresholds = parse_thresholds(args.phash_sweep)
        print("\nP-hash sweep (количество пар с dist <= threshold):")
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                sa, sb = split_map[a], split_map[b]
                counts = phash_counts_by_threshold(sa, sb, thresholds, args.phash_sweep_chunk)
                pairs = ", ".join([f"{t}: {counts[t]}" for t in thresholds])
                print(f"  {a} vs {b}: {pairs}")

    if args.phash:
        print(f"\nP-hash (порог {args.phash_threshold}):")
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                sa, sb = split_map[a], split_map[b]
                matches = phash_matches(sa, sb, args.phash_threshold, args.phash_samples)
                print(f"  {a} vs {b}: найдено {len(matches)} примеров (показано до {args.phash_samples})")
                if matches:
                    print("    пример:", matches)

    if args.write_clean:
        print(f"\nЗапись очищенных CSV в {args.write_clean} (приоритет по порядку --splits)...")
        write_clean_splits(split_map, args)


if __name__ == "__main__":
    main()
