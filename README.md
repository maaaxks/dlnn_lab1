# Диагностика обучения MLP

Лабораторная работа: бинарная классификация `make_moons` с помощью MLP на PyTorch. Исследуется влияние функции активации, инициализации весов, режима оптимизации и глубины сети на кривые loss и accuracy и на нормы градиентов скрытых слоёв.

Вся реализация находится в `notebook.ipynb`.

## Установка

Использовался Python 3.12.3.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Запуск

Открыть `notebook.ipynb` в VS Code или Jupyter, выбрать ядро из созданного окружения `venv` и выполнить все ячейки по порядку (Restart и Run All). Рабочая папка — корень проекта.

Устройство: `cpu`, `torch.set_num_threads(1)`.

## Создаваемые файлы

Таблицы в `results/`:

| Файл | Содержимое |
|---|---|
| `metrics.csv` | Метрики всех 12 запусков за эпохи 0–100: `config, seed, epoch, train_loss, train_accuracy, val_loss, val_accuracy`. Accuracy записана долями от 0 до 1 |
| `gradient_norms.csv` | Нормы градиентов для A, C, D с обоими seed на эпохах 0, 20, 100: `config, seed, epoch, layer, gradient_norm` |
| `comparison_epoch100.csv` | Сравнительная таблица четырёх метрик на эпохе 100 и средний `val_loss` по двум seed |
| `final_model.csv` | Loss и accuracy выбранного запуска на train, validation и test |

Графики в `figures/`:

| Файл | Содержимое |
|---|---|
| `train_scatter.png` | Исходные обучающие объекты с цветом по классу |
| `val_loss_seed11.png` | Validation loss A–F по эпохам, seed 11 |
| `val_accuracy_seed11.png` | Validation accuracy A–F по эпохам, seed 11 |
| `grad_norms_seed11.png` | Нормы градиентов по слоям для A, C, D на эпохах 0, 20, 100, seed 11 |
| `best_run_curves.png` | Train loss и validation loss выбранного запуска |

Сравнительная таблица и результаты выбора модели также выводятся в ноутбуке.

## Конфигурации

Каждая конфигурация запускается с seed 11 и 29, всего 12 запусков.

| Код | Скрытые слои | Активация | Инициализация скрытых слоёв | Оптимизатор | Learning rate |
|---|---:|---|---|---|---:|
| A | 6 | Sigmoid | Xavier normal | Adam | 0.003 |
| B | 6 | Tanh | Xavier normal | Adam | 0.003 |
| C | 6 | ReLU | Xavier normal | Adam | 0.003 |
| D | 6 | ReLU | He normal | Adam | 0.003 |
| E | 6 | ReLU | He normal | SGD (`momentum=0`) | 0.05 |
| F | 1 | ReLU | He normal | Adam | 0.003 |

Общие условия:

- **Модель:** ширина скрытых слоёв 32, два логита на выходе. Выходной слой инициализируется Xavier normal (`gain=1.0`), все смещения равны нулю.
- **Обучение:** 100 эпох, batch size 128, перемешивание train в каждой эпохе, `CrossEntropyLoss(reduction='mean')`.
- **Данные:** `make_moons(n_samples=2400, noise=0.25, random_state=2026)`, разбиение train/validation/test 1440/480/480 со стратификацией. `StandardScaler` обучается только на train.
- **Воспроизводимость:** перед каждым запуском вызывается `torch.manual_seed(seed)`, а `DataLoader` получает `torch.Generator().manual_seed(10000 + seed)`.
