## 11.12.2025 14:10
- Созданы скрипты подготовки PlantVillage, разбиение 70/15/15 в CSV.
- Среда: Python 3.11, Torch 2.8.0+xpu, IPEX 2.8.10+xpu для Intel Arc B580.

## 18.12.2025 12:30
- Проверка утечек и очистка сплитов PlantVillage (`check_leakage --write-clean` → outputs/clean_splits).
- Обучение ConvNeXt-Tiny на чистом PlantVillage, best ckpt: experiments/teacher/convnext_tiny_clean/convnext_tiny.fb_in22k_ft_in1k-xpu-best.pth. Тест Acc≈0.998.
- Grad-CAM и матрицы ошибок сохранены в outputs/confmat_clean, outputs/gradcam_clean.

## 24.12.2025 15:30
- Смешанный датасет (PlantVillage+PlantDoc): объединено и очищено от дубликатов → outputs/mix_clean/mix_train.csv, outputs/mix_clean/plantdoc_test.csv; общий тест outputs/combined_test.csv (2477 образцов).
- Новая утилита src/data/combine_csvs.py; в eval_classifier/eval_confusion_matrix добавлен флаг --drop-missing для тестов без всех классов.
- Fine-tune ConvNeXt-Tiny (XPU) lr=5e-5 на миксе: best ckpt experiments/teacher/convnext_tiny_mix_clean_lr5e5/convnext_tiny.fb_in22k_ft_in1k-xpu-best.pth.
  * PlantDoc-only (8 классов, drop-missing): Acc≈0.758, Macro F1≈0.749, PR-AUC≈0.854, ROC-AUC≈0.970. Частые ошибки: Bacterial_spot↔Septoria, Healthy↔Leaf_Mold, Late↔Early blight.
  * Combined test (10 классов, PV+PD): Acc=0.98385, Macro F1=0.98038. Метрики: outputs/metrics_combined; матрица: outputs/confmat_combined/confusion_matrix.*.
  * PlantDoc-only отчёты: outputs/metrics_plantdoc_only; outputs/confmat_plantdoc_only/confusion_matrix.*.
