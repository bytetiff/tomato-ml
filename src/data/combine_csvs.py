import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple


def read_csv(file_path: Path) -> Tuple[List[str], List[dict]]:
    with file_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    return header, rows


def write_csv(file_path: Path, header: List[str], rows: List[dict]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Объединить несколько CSV с колонками path,label (и одинаковым заголовком)."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Список входных CSV-файлов (например, data/processed/splits/train.csv outputs/plantdoc_train.csv)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Путь к выходному CSV-файлу",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Перемешать строки перед сохранением",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Зерно для перемешивания (если включен --shuffle)",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Удалить дубликаты по полю path после объединения",
    )
    args = parser.parse_args()

    all_rows: List[dict] = []
    header_ref: List[str] = []

    for inp in args.inputs:
        path = Path(inp)
        if not path.is_file():
            raise FileNotFoundError(f"Не найден входной CSV: {path}")

        header, rows = read_csv(path)
        if not header:
            raise ValueError(f"Пустой заголовок в {path}")

        if not header_ref:
            header_ref = header
        else:
            if header != header_ref:
                # Объединяем заголовки: сохраняем порядок первого, добавляем новые поля из текущего
                union = header_ref[:]
                for col in header:
                    if col not in union:
                        union.append(col)
                header_ref = union

        # Дополняем строки отсутствующими ключами пустыми значениями
        for row in rows:
            for col in header_ref:
                if col not in row:
                    row[col] = ""
        all_rows.extend(rows)
        print(f"[INFO] Добавлено {len(rows):,} строк из {path}")

    if args.drop_duplicates:
        before = len(all_rows)
        seen = set()
        unique_rows = []
        path_key = "path" if "path" in header_ref else header_ref[0]
        for row in all_rows:
            key = row.get(path_key)
            if key and key not in seen:
                seen.add(key)
                unique_rows.append(row)
        all_rows = unique_rows
        print(f"[INFO] Дубликаты по '{path_key}' удалены: {before - len(all_rows):,}")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(all_rows)
        print(f"[INFO] Перемешано {len(all_rows):,} строк (seed={args.seed})")

    out_path = Path(args.out)
    write_csv(out_path, header_ref, all_rows)
    print(f"[OK] Сохранено {len(all_rows):,} строк в {out_path}")


if __name__ == "__main__":
    main()
