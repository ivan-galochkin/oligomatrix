# oligomatrix

oligomatrix computes pairwise minimum free energy (MFE) matrices for DNA and RNA oligonucleotides. The Rust implementation runs on the CPU, on NVIDIA GPUs with CUDA, or on Apple Silicon with Metal. The Python API returns a NumPy array.

Each entry gives the MFE in kcal/mol for one row strand and one column strand. It reproduces ViennaRNA 2.8.0’s `mfe_dimer()` for two linear strands written with A, C, G and T and with `min_loop_size=0`. The calculation considers both a bound structure and two independently folded strands. Use `model="mathews2004"` for DNA. The default is the Turner 2004 RNA model.

## Install from PyPI

```sh
python -m pip install oligomatrix
```

The package requires CPython 3.10 or later and uses NumPy at runtime. Wheels are available for Linux x86-64, Windows x64 and macOS arm64. [Package on PyPI](https://pypi.org/project/oligomatrix/)

## Compute an MFE matrix in Python

```python
from oligomatrix import EnergyCalculator

rows = ["GAT", "TACA"]
columns = ["AGC", "GACT", "ATCG"]
calc = EnergyCalculator(temperature=37.0, model="turner2004", threads=8)
matrix = calc.matrix(rows, columns)
print(matrix)
# [[ 0.   0.  -0.6]
#  [ 0.   0.   0. ]]
```

The result is an independent two-dimensional `float32` array. Entry `matrix[i, j]` corresponds to `rows[i]` and `columns[j]` in the supplied order. For a single call, use `energy_matrix(rows, columns, **settings)`. For repeated calls at the same temperature and with the same model, reuse an `EnergyCalculator`. It caches energies for ordered pairs. On the default CPU backend, Rayon distributes new pairs across worker threads. Setting `threads=0` uses up to eight available logical CPUs.

## NVIDIA GPU acceleration with CUDA

```python
from oligomatrix import energy_matrix

matrix = energy_matrix(rows, columns, model="mathews2004", backend="cuda")
```

You can also pass `backend="cuda"` to `EnergyCalculator`. Each CUDA block folds one ordered pair. Its threads fill the dynamic programming table one diagonal at a time, starting with the shortest intervals. Blocks for different pairs run independently. Rust prepares the energy parameters, computes the free-strand energy once for each distinct column strand and assembles the NumPy result. The `threads` setting controls the CPU backend.

CUDA supports a combined length of up to 128 nucleotides per pair. Use `backend="cpu"` for longer pairs. The Windows and Linux wheels include the compiled PTX kernel. CUDA requires an NVIDIA GPU with compute capability 7.5 or later and a driver that supports CUDA 12.9 or later. The library selects the first visible device. If CUDA initialization or execution fails, the Python call raises an exception.

### CUDA benchmark against ViennaRNA

This benchmark used a 128 × 193 matrix with 24,704 distinct ordered pairs and strands of 16–36 nucleotides. The table shows median times over five runs with the Mathews 2004 DNA model at 37 °C. Each run started with an empty pair cache and ended when Python received the NumPy array.

| Backend | Time |
| --- | ---: |
| Rust CPU, AMD Ryzen 7 255, 1 thread | 4.110 s |
| Rust CPU, AMD Ryzen 7 255, 8 threads | 0.604 s |
| CUDA, NVIDIA GeForce RTX 5060 Ti | 0.132 s |

The first CUDA call in the process took 0.348 s. After initialization, CUDA was 4.59× faster than the eight-thread CPU run. When evaluating internal loops, the kernel uses a lower bound on the energy to skip candidates that cannot improve the current minimum. Every matrix entry in every timed run matched the Python result from ViennaRNA’s `mfe_dimer()` exactly.

To repeat the comparison on an NVIDIA GPU, run `python scripts/check_cuda.py --benchmark --repeats 3`. The script reports the first CUDA call separately. Use `--repeats 5` to match the sample count in the table.

After changing the CUDA kernel or its shared headers, rebuild the PTX with `python scripts/build_cuda.py` and commit the updated `src/cuda/fold.ptx` alongside the source changes. The build requires `cuda-bindings>=12.9,<13`, `nvidia-cuda-nvrtc-cu12==12.9.86` and an NVRTC library visible to the system loader. Run `python scripts/build_cuda.py --check` to verify that the source and build options match the recorded checksum.

## Apple Silicon GPU acceleration with Metal

On an Apple Silicon Mac, use a native arm64 Python interpreter and select `backend="metal"`. For a list named `oligos` containing 20 DNA strands, the following call returns a 20 × 20 NumPy matrix.

```python
from oligomatrix import EnergyCalculator

calc = EnergyCalculator(backend="metal", model="mathews2004", temperature=65.0)
matrix = calc.matrix(oligos, oligos)  # (20, 20), numpy.float32
```

Each Metal threadgroup processes one ordered pair. Its threads compute cells on the same diagonal in parallel and synchronize before advancing to the next diagonal. Different threadgroups can process different pairs at the same time. Metal and CUDA share the source for the energy functions and folding recurrences. Metal supports a combined length of up to 128 nucleotides per pair and raises a Python exception if the device is unavailable.

### Benchmark for 1,000 sets of 20 oligonucleotides

Each set contained 20 oligos of 40–60 nucleotides drawn from a different 600-nucleotide template. Adjacent oligos overlapped by 20–30 nucleotides. The benchmark used the Mathews 2004 DNA model at 65 °C. Elapsed time includes creating the calculator and returning all 1,000 matrices to Python.

| Hardware | Backend | Time for 1,000 matrices |
| --- | --- | ---: |
| Apple M1 (virtual), GitHub Actions | Metal, Apple Paravirtual device | 90.786 s |
| Same Apple M1 virtual machine | Rust CPU, up to 8 threads, limited by available vCPUs | 340.872 s |
| NVIDIA GeForce RTX 5060 Ti, Windows | CUDA | 21.058 s |
| AMD Ryzen 7 255, same Windows machine | Rust CPU, 8 threads | 89.735 s |

Metal was 3.75× faster than the CPU on the same virtual M1 in this run. Timings on a physical Mac depend on the chip and the number of GPU cores. All four configurations matched ViennaRNA 2.8.0 bit for bit. The reference was computed by calling `RNA.fold_compound(...).mfe_dimer()` for each of the 400,000 entries. The benchmark checks the SHA-256 digest of all entries in little-endian `float32` format. [Virtual M1 benchmark log](https://github.com/ivan-galochkin/oligomatrix/actions/runs/46012530443)

To run the checks without owning a Mac, open **Actions → Metal validation → Run workflow** on GitHub. The macOS arm64 job builds and tests the wheel through Python, then saves the wheel and timing log in the `oligomatrix-metal-macos-arm64` artifact. On your own Mac, install the wheel and `ViennaRNA==2.8.0`, then run these commands from the repository.

```sh
python scripts/check_cuda.py --backend metal
python scripts/benchmark_apple.py
```

To run the same 1,000-set benchmark on NVIDIA, use `python scripts/benchmark_apple.py --gpu cuda`. To regenerate the independent ViennaRNA reference, use `python scripts/benchmark_apple.py --oracle-workers 8`.

## Rust API and project layout

Cargo builds the Rust library without Python bindings by default. Use `MatrixCalculator::new` for CPU execution or `MatrixCalculator::with_backend` with `Backend::Cuda` or `Backend::Metal` for GPU execution. For a CPU-only build, run `cargo build --no-default-features`. Maturin enables the Python bindings when building the Python package.

| Path | Purpose |
| --- | --- |
| `src/params.rs`, `data/*.par` | Turner and Mathews parameter tables and temperature adjustment |
| `src/folding.rs` | Dynamic programming for single-strand and two-strand folding |
| `src/gpu/` | Energy functions, diagonal folding recurrences and input packing shared by the GPU backends |
| `src/cuda/`, `src/metal/` | NVIDIA and Apple GPU kernel launches and result transfer to Rust |
| `src/matrix.rs`, `src/workers.rs` | Strand indexing, caching, matrix assembly and the CPU thread pool |
| `src/python.rs` | Python classes, exceptions, GIL release and transfer of the result buffer to NumPy |

## Build from source and test

Use Rust 1.88 or later and standard CPython 3.10 or later. Run the Python commands in an activated virtual environment. On Windows, use a linker compatible with your Rust toolchain.

```sh
cargo test --locked
python -m pip install "maturin>=1.16,<2" numpy
maturin develop --release --locked
python -m pip install ViennaRNA==2.8.0
python scripts/check_matrix.py
python scripts/check_matrix.py --stress
```

CI runs the reference comparison on CPython 3.12, for which ViennaRNA provides binary wheels. To build local distributions, run `maturin build --release --locked --out dist` for a wheel and `maturin sdist --out dist` for a source archive.

The thermodynamic parameter tables come from the ViennaRNA Package, developed at the Institute for Theoretical Chemistry, University of Vienna. [ViennaRNA attribution and distribution terms](https://github.com/ivan-galochkin/oligomatrix/blob/main/NOTICE-ViennaRNA.txt)

## GitHub Actions and package publishing

This directory is a standalone Git repository with `main` as its primary branch. To push a copy to an empty GitHub repository, substitute its remote URL and run the following commands from the repository root.

```sh
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

The **Build and test** workflow runs on pull requests and pushes to `main`. It tests the Rust core on Rust 1.88 and builds a source archive and three wheels for standard CPython 3.10 or later. The wheel targets are Linux x86-64 (manylinux 2.28), Windows x86-64 and macOS arm64. CI installs each wheel, compares its results with ViennaRNA on Python 3.12 and tests the same wheel on Python 3.10. It also rebuilds the package from the source archive and runs the stress test against that installation. [Build and test workflow](https://github.com/ivan-galochkin/oligomatrix/blob/main/.github/workflows/ci.yml)

To publish a package, open **Actions → Publish Python package → Run workflow** and select a branch or tag and a destination. The workflow reruns the same CI checks and uploads the wheels and source archive from that run. Cargo reads the version from `Cargo.toml`. For a new release, change the version there and run `cargo check` to update `Cargo.lock`. [Publish Python package workflow](https://github.com/ivan-galochkin/oligomatrix/blob/main/.github/workflows/release.yml)

| Destination | One-time setup |
| --- | --- |
| `testpypi` | Add a GitHub Trusted Publisher for `oligomatrix` on TestPyPI. Specify the owner, repository, `release.yml` workflow and `testpypi` environment. Create the matching environment on GitHub. |
| `pypi` | Configure a Trusted Publisher on PyPI with the same repository and workflow and the `pypi` environment. For a new project, you can create a pending publisher before the first release. Authentication uses OpenID Connect (OIDC). |
| `private-index` | In the `private-index` GitHub environment, set `PACKAGE_REPOSITORY_URL` to the registry’s HTTPS upload endpoint and add the `TWINE_USERNAME` and `TWINE_PASSWORD` secrets. For token authentication, use the username your registry requires. The registry must accept Twine uploads. |

To start a TestPyPI release from the GitHub CLI, run `gh workflow run release.yml --ref main -f registry=testpypi`. To install that build, first install NumPy from the regular index, then select the version you uploaded on TestPyPI. For example, for version 0.4.0, use `python -m pip install --index-url https://test.pypi.org/simple/ --no-deps oligomatrix==0.4.0`. [PyPI guide to using a Trusted Publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
