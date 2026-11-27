# oligomatrix

Расчёт матрицы минимальных свободных энергий (ккал/моль) из Python или Rust. Алгоритм воспроизводит `mfe_dimer()` ViennaRNA 2.8.0 для двух линейных фрагментов A/C/G/T при `min_loop_size=0`. Ячейка учитывает как связанную структуру, так и две отдельно свернувшиеся цепи. Для ДНК задайте `model="mathews2004"`. По умолчанию используется модель РНК Turner 2004.

```python
from oligomatrix import EnergyCalculator

calc = EnergyCalculator(temperature=37.0, model="turner2004", threads=8)
matrix = calc.matrix(["GAT", "TACA"], ["AGC", "GACT", "ATCG"])
print(matrix)  # [[0., 0., -0.6], [0., 0., 0.]]
```

Для одного вызова есть `energy_matrix(rows, columns, **settings)`. Результат — независимый двумерный массив NumPy типа `float32` в исходном порядке. Калькулятор сохраняет энергии упорядоченных пар между вызовами. Rayon распределяет новые пары между потоками. Значение `threads=0` выбирает до восьми доступных логических процессоров.

## Расчёт на GPU

```python
from oligomatrix import energy_matrix

matrix = energy_matrix(rows, columns, model="mathews2004", backend="cuda")
```

`backend="cuda"` доступен и в `EnergyCalculator`. Один блок CUDA считает одну пару. Его потоки заполняют таблицу по диагоналям, начиная с коротких участков. Блоки для разных пар выполняются независимо. Rust готовит энергетические параметры, по одному разу считает энергию свободной цепи для каждого уникального фрагмента правой оси и собирает результат в NumPy. Параметр `threads` задаёт число потоков CPU при `backend="cpu"`.

CUDA поддерживает пары суммарной длиной до 128 нуклеотидов. Для более длинных пар выберите `backend="cpu"`. В wheel для Windows и Linux включено скомпилированное ядро PTX. Нужна видеокарта NVIDIA с compute capability 7.5 или выше и драйвер с поддержкой CUDA 12.9 или новее. Библиотека использует первое видимое устройство. При явно выбранном CUDA ошибки загрузки или выполнения возвращаются в Python. По умолчанию библиотека считает на CPU.

Для проверки на машине с GPU выполните `python scripts/check_cuda.py --benchmark --repeats 3`. Скрипт сверяет результат с ViennaRNA 2.8.0, отдельно показывает первый вызов CUDA и измеряет время до возврата NumPy-массива с пустым кешем пар для каждого повтора.

На RTX 5060 Ti и AMD Ryzen 7 255 матрица 128 × 193 содержала 24 704 разные пары фрагментов длиной 16–36 нуклеотидов каждый. Для модели Mathews 2004 при 37 °C получены следующие медианы пяти повторов, включая возврат результата в Python.

| Режим | Время |
| --- | ---: |
| Rust, 1 поток CPU | 4,110 с |
| Rust, 8 потоков CPU | 0,604 с |
| CUDA, RTX 5060 Ti | 0,132 с |

Первый вызов CUDA в процессе занял 0,348 с. После инициализации ускорение относительно восьми потоков составило 4,59 раза. При поиске внутренних петель ядро использует нижнюю границу энергии и пропускает варианты, которые уже не могут улучшить найденный минимум. Во всех замерах значения всех ячеек в точности совпали с Python-вызовом `mfe_dimer()`.

После изменения исходников CUDA пересоберите PTX. Для разработчика сборка использует пакеты `cuda-bindings>=12.9,<13` и `nvidia-cuda-nvrtc-cu12==12.9.86`, а также доступную загрузчику библиотеку NVRTC. Выполните `python scripts/build_cuda.py` и включите обновлённый `src/cuda/fold.ptx` в то же изменение. Команда `python scripts/build_cuda.py --check` проверяет соответствие исходников и настроек сборки сохранённой контрольной сумме.

## Apple Silicon и Metal

На Mac с процессором Apple и нативным Python для arm64 выберите `backend="metal"`. Один вызов возвращает обычную матрицу NumPy. Например, для списка из 20 фрагментов ДНК:

```python
from oligomatrix import EnergyCalculator

calc = EnergyCalculator(backend="metal", model="mathews2004", temperature=65.0)
matrix = calc.matrix(oligos, oligos)  # (20, 20), numpy.float32
```

Каждая группа потоков GPU обрабатывает одну упорядоченную пару. Потоки параллельно заполняют ячейки одной диагонали и синхронизируются перед переходом к следующей. Разные пары могут выполняться одновременно. Metal и CUDA используют один исходный код для энергетических формул и рекуррентных соотношений. Суммарная длина двух фрагментов ограничена 128 нуклеотидами. Если устройство Metal недоступно, Python получает исключение.

В GitHub Actions откройте **Metal validation → Run workflow**. Задание на macOS arm64 собирает wheel, проверяет его через Python и сохраняет wheel и протокол замера в артефакте `oligomatrix-metal-macos-arm64`. Для повтора на своём Mac из репозитория с установленным wheel и `ViennaRNA==2.8.0` выполните:

```sh
python scripts/check_cuda.py --backend metal
python scripts/benchmark_apple.py
```

Бенчмарк рассчитывает 1 000 разных наборов по 20 олигов длиной 40–60 нуклеотидов. В каждом наборе исходная цепь имеет длину 600 нуклеотидов, перекрытия соседних олигов — 20–30. Модель — DNA Mathews 2004 при 65 °C. Время включает создание калькулятора и получение всех 1 000 матриц в Python. Для сверки всех 400 000 значений скрипт использует SHA-256 массива `float32` с заданным порядком байтов. Эталон рассчитан вызовом `RNA.fold_compound(...).mfe_dimer()` для каждой ячейки. Его можно воспроизвести командой `python scripts/benchmark_apple.py --oracle-workers 8`.

## Устройство проекта

| Файл | Ответственность |
| --- | --- |
| `src/params.rs`, `data/*.par` | Таблицы Turner и Mathews, пересчёт для температуры |
| `src/folding.rs` | Динамическое программирование для одной и двух цепей |
| `src/gpu/` | Общие для GPU энергетические формулы, расчёт по диагоналям и упаковка входных данных |
| `src/cuda/`, `src/metal/` | Запуск ядра на NVIDIA или Apple GPU и передача результатов в Rust |
| `src/matrix.rs`, `src/workers.rs` | Индексация фрагментов, кеш, сборка матрицы и пул потоков |
| `src/python.rs` | Python-классы, исключения, освобождение GIL и передача буфера в NumPy |

По умолчанию Cargo собирает ядро без Python. В Rust используйте `MatrixCalculator::new` для CPU или `MatrixCalculator::with_backend` с `Backend::Cuda` или `Backend::Metal` для GPU. Сборка `cargo build --no-default-features` включает только CPU. Maturin включает адаптер Python при сборке пакета.

## Локальная сборка и проверка

Нужны Rust 1.88 или новее и стандартный CPython 3.10 или новее. Команды Python выполняются в активированном виртуальном окружении. В Windows используйте подходящий для выбранного Rust toolchain линкер.

```sh
cargo test --locked
python -m pip install "maturin>=1.16,<2" numpy
maturin develop --release --locked
python -m pip install ViennaRNA==2.8.0
python scripts/check_matrix.py
python scripts/check_matrix.py --stress
```

Для эталонного теста в CI используется CPython 3.12, для которого опубликованы бинарные пакеты ViennaRNA. Зависимость установленного модуля во время работы — NumPy. Локальные артефакты поставки собираются командами `maturin build --release --locked --out dist` и `maturin sdist --out dist`.

Термодинамические таблицы взяты из ViennaRNA Package, разработанного авторами пакета в Институте теоретической химии Венского университета. Условия использования и распространения приведены в [NOTICE-ViennaRNA.txt](NOTICE-ViennaRNA.txt).

## GitHub и публикация пакета

Корень этой папки — отдельный Git-репозиторий с веткой `main`. После создания пустого репозитория на GitHub подставьте свой адрес и выполните команды из этой папки.

```sh
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

[Build and test](.github/workflows/ci.yml) запускается для pull request и при отправке в `main`. Он проверяет ядро на Rust 1.88, собирает исходный архив и три бинарных wheel для стандартного CPython 3.10+ (Linux x86-64 manylinux 2.28, Windows x86-64, macOS ARM64). Wheel проверяется после установки с ViennaRNA на Python 3.12 и отдельно загружается на Python 3.10. Из исходного архива пакет собирается заново и проходит нагрузочный тест.

[Publish Python package](.github/workflows/release.yml) запускается вручную через **Actions → Publish Python package → Run workflow**. Выберите ветку или тег и назначение. Сценарий заново вызывает тот же CI, затем загружает прошедшие проверки wheel и исходный архив из этого запуска. Версия пакета берётся из `Cargo.toml`. Для следующего выпуска измените версию и обновите `Cargo.lock` командой `cargo check`.

| Назначение | Однократная настройка |
| --- | --- |
| `testpypi` | В TestPyPI добавьте GitHub Trusted Publisher для проекта `oligomatrix`. Укажите владельца, репозиторий, файл `release.yml` и окружение `testpypi`. Создайте одноимённое environment в GitHub. |
| `pypi` | Настройте Trusted Publisher на PyPI с теми же данными и окружением `pypi`. Для первого выпуска нового проекта можно создать pending publisher. Авторизация выполняется по OIDC. |
| `private-index` | В GitHub environment `private-index` задайте переменную `PACKAGE_REPOSITORY_URL` с HTTPS-адресом **для загрузки** и секреты `TWINE_USERNAME`, `TWINE_PASSWORD`. Для токена используйте имя пользователя, требуемое вашим реестром. Поддерживается протокол загрузки Twine. |

Через GitHub CLI тот же ручной выпуск можно запустить командой `gh workflow run release.yml --ref main -f registry=testpypi`. Для установки сборки из TestPyPI сначала установите NumPy из обычного индекса, затем выполните `python -m pip install --index-url https://test.pypi.org/simple/ --no-deps oligomatrix==0.4.0`.

Правила идентификации репозитория и окружения описаны в [инструкции PyPI по Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
