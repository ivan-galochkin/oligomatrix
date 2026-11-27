"""Check GPU energies against ViennaRNA 2.8.0 and optionally time full NumPy calls."""

import argparse
from random import Random
from statistics import median
from time import perf_counter

import numpy as np
import RNA

from oligomatrix import EnergyCalculator, energy_matrix
from check_matrix import MODELS, TARGETS, check_concurrent, check_matrix, model_details, reference, unique_axes


def check_model(model, temperature, backend):
    md = model_details(model, temperature)
    settings = dict(model=model, temperature=temperature)
    calc = EnergyCalculator(backend=backend, **settings)
    cells = 0

    def compare(actual, expected, rows, columns, label):
        nonlocal cells
        cells += check_matrix(actual, expected, rows, columns, model, temperature, label)

    def cached_matrix(rows, columns, label):
        expected = reference(rows, columns, md)
        compare(calc.matrix(rows, columns), expected, rows, columns, label)

    # Includes a partial final CPU batch.
    rows, columns = unique_axes(211, (23, 29), (16, 20, 24))
    expected = reference(rows, columns, md)
    for label, options in ((backend.upper(), {"backend": backend}),
                           ("CPU/1", {"threads": 1}), ("CPU/8", {"threads": 8})):
        actual = energy_matrix(rows, columns, **settings, **options)
        compare(actual, expected, rows, columns, label)

    # Both orientations of all curated structures, plus the off-diagonal pairs.
    rows = [a for _, a, _ in TARGETS]
    columns = [b for _, _, b in TARGETS]
    cached_matrix(rows, columns, "ViennaRNA structure fixtures")
    cached_matrix(columns, rows, "reversed structure fixtures")

    # Straddle combined lengths 32, 64 and 128, including highly unequal strands.
    rng = Random(5729)
    rows = ["".join(rng.choices("ACGT", k=n)) for n in (15, 16, 17, 31, 32, 33, 63, 64)]
    columns = ["".join(rng.choices("ACGT", k=n)) for n in (16, 32, 64)]
    cached_matrix(rows, columns, "balanced boundary pairs")
    long_axis = ["".join(rng.choices("ACGT", k=n)) for n in (30, 31, 32, 62, 63, 64, 126, 127)]
    cached_matrix(long_axis, ["C"], "strand break at the end")
    cached_matrix(["C"], long_axis, "strand break at the beginning")
    for total in (2, 32, 64, 96, 97, 128):
        left = "".join(rng.choices("ACGT", k=total // 2))
        right = "".join(rng.choices("ACGT", k=total - len(left)))
        expected = reference([left], [right], md)
        actual = energy_matrix([left], [right], backend=backend, **settings)
        compare(actual, expected, [left], [right], f"single GPU group with capacity {total}")

    rows, columns = ["GATTACA", "GGGGGG"], ["CCCCCC", "TGT", "ATCG"]
    cached_matrix(rows[:1], columns[:2], "prime partial cache")
    rows = [rows[1], rows[0], rows[1]]
    columns = [columns[2], columns[0], columns[1], columns[2]]
    expected = reference(rows, columns, md)
    result = calc.matrix(rows, columns)
    compare(result, expected, rows, columns, "reordered hits, misses and duplicate axes")
    result.fill(np.inf)
    compare(calc.matrix(rows, columns), expected, rows, columns, "independent cached result")
    return cells


def check_edges(backend):
    calc = EnergyCalculator(backend=backend)
    for matrix in (calc.matrix, lambda rows, cols: energy_matrix(rows, cols, backend=backend)):
        for rows, columns in (([], []), ([], ["A", "C"]), (["G", "T"], [])):
            expected = np.empty((len(rows), len(columns)))
            check_matrix(matrix(rows, columns), expected, rows, columns, "turner2004", 37, "empty axis")
        for rows, columns in (([""], ["A"]), ([], ["N"]), (["A" * 65], ["C" * 64]),
                              (["A"], ["C" * 128])):
            try:
                matrix(rows, columns)
            except ValueError:
                continue
            raise AssertionError(f"GPU accepted invalid fragment lengths or bases: {rows}, {columns}")

    # A pair beyond the CUDA limit is still valid on the default CPU backend.
    rows, columns = ["CG" * 32 + "A"], ["AT" * 32]
    md = model_details("turner2004", 37.0)
    check_matrix(energy_matrix(rows, columns), reference(rows, columns, md), rows, columns,
                 "turner2004", 37.0, "CPU pair of combined length 129")


def benchmark(repeats, backend):
    model, temperature = "mathews2004", 37.0
    rows, columns = unique_axes(391, (128, 193), (16, 20, 24, 28, 32, 36))
    print("128 x 193 unique pairs, Mathews 2004 at 37 C, seed 391. Computing ViennaRNA oracle.",
          flush=True)
    expected = reference(rows, columns, model_details(model, temperature))
    configurations = {backend.upper(): {"backend": backend}, "CPU/1": {"threads": 1}, "CPU/8": {"threads": 8}}

    def timed_call(label):
        # The one-shot public API creates a new calculator for every call. The
        # returned ndarray already resides on the host when the timer stops.
        start = perf_counter()
        actual = energy_matrix(rows, columns, model=model, temperature=temperature,
                               **configurations[label])
        elapsed = perf_counter() - start
        check_matrix(actual, expected, rows, columns, model, temperature, label)
        return elapsed

    for label in configurations:
        print(f"First {label:5} call {timed_call(label):.6f} s", flush=True)

    samples = {label: [] for label in configurations}
    rng = Random(480)
    order = list(configurations)
    for i in range(repeats):
        rng.shuffle(order)
        for label in order:
            elapsed = timed_call(label)
            samples[label].append(elapsed)
            print(f"Warm run {i + 1}, {label:5} {elapsed:.6f} s", flush=True)
    medians = {label: median(values) for label, values in samples.items()}
    for label, elapsed in medians.items():
        print(f"{label:5} warm median {elapsed:.6f} s over {repeats} run(s)")
    print(f"CPU/1 time divided by {backend.upper()} time {medians['CPU/1'] / medians[backend.upper()]:.3f}x")
    print(f"CPU/8 time divided by {backend.upper()} time {medians['CPU/8'] / medians[backend.upper()]:.3f}x")
    print(f"All timed calls matched all {expected.size} ViennaRNA energies exactly.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "metal"), default="cuda")
    parser.add_argument("--benchmark", action="store_true",
                        help="also time the seed-391 matrix used by benchmark_cofold.py")
    parser.add_argument("--repeats", type=int, default=3,
                        help="warm benchmark samples per backend (default: 3)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if RNA.__version__ != "2.8.0":
        raise RuntimeError(f"Expected ViennaRNA 2.8.0, found {RNA.__version__}")

    # Measure the first GPU use in this process before running the regressions.
    if args.benchmark:
        benchmark(args.repeats, args.backend)
    check_edges(args.backend)
    for model in MODELS:
        for temperature in (-5.0, 23.75, 37.0, 70.0):
            cells = check_model(model, temperature, args.backend)
            print(f"{model} at {temperature:g} C, {cells} cell comparisons matched ViennaRNA exactly.",
                  flush=True)
    print(f"{check_concurrent(args.backend)} cells matched with concurrent GPU calculators.")


if __name__ == "__main__":
    main()
