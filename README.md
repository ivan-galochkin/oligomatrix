# oligomatrix

Расчёт матрицы минимальных свободных энергий (ккал/моль) из Python или Rust. Алгоритм воспроизводит `mfe_dimer()` ViennaRNA 2.8.0 для двух линейных фрагментов A/C/G/T при `min_loop_size=0`. Ячейка учитывает как связанную структуру, так и две отдельно свернувшиеся цепи. Для ДНК задайте `model="mathews2004"`. По умолчанию используется модель РНК Turner 2004.

```python
from oligomatrix import EnergyCalculator

calc = EnergyCalculator(temperature=37.0, model="turner2004", threads=8)
matrix = calc.matrix(["GAT", "TACA"], ["AGC", "GACT", "ATCG"])
print(matrix)  # [[0., 0., -0.6], [0., 0., 0.]]
```

Для одного вызова есть `energy_matrix(rows, columns, **settings)`. Результат — независимый двумерный массив NumPy типа `float32` в исходном порядке. Калькулятор сохраняет энергии упорядоченных пар между вызовами. Rayon распределяет новые пары между потоками. Значение `threads=0` выбирает до восьми доступных логических процессоров.

## Устройство проекта

| Файл | Ответственность |
| --- | --- |
| `src/params.rs`, `data/*.par` | Таблицы Turner и Mathews, пересчёт для температуры |
| `src/folding.rs` | Динамическое программирование для одной и двух цепей |
| `src/matrix.rs`, `src/workers.rs` | Индексация фрагментов, кеш, сборка матрицы и пул потоков |
| `src/python.rs` | Python-классы, исключения, освобождение GIL и передача буфера в NumPy |

По умолчанию Cargo собирает ядро без Python. Публичный Rust API состоит из `MatrixCalculator`, `Model`, `EnergyMatrix` и `MatrixError`. Maturin включает адаптер Python при сборке пакета.

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

Через GitHub CLI тот же ручной выпуск можно запустить командой `gh workflow run release.yml --ref main -f registry=testpypi`. Для установки сборки из TestPyPI сначала установите NumPy из обычного индекса, затем выполните `python -m pip install --index-url https://test.pypi.org/simple/ --no-deps oligomatrix==0.3.0`.

Правила идентификации репозитория и окружения описаны в [инструкции PyPI по Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
