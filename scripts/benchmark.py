"""Time complete Python calls on the same ordered fragments and report medians."""

import argparse
from importlib.metadata import version
from random import Random
from statistics import median
from time import perf_counter_ns

import numpy as np
import RNA

from oligomatrix import EnergyCalculator, energy_matrix
from check_matrix import check_matrix, model_details, reference


def benchmark(name, rows, columns, repeats, legacy_lists):
    model, temperature = "turner2004", 37.0
    md = model_details(model, temperature)
    expected = reference(rows, columns, md)
    cached = EnergyCalculator(model=model, temperature=temperature, threads=1)
    # Each invocation of energy_matrix constructs a fresh calculator. Only the
    # last configuration keeps its pair cache between warmup and measurement.
    calls = {"Python + ViennaRNA": lambda: reference(rows, columns, md, dtype=np.float32)}
    for n in (1, 2, 4, 8):
        calls[f"fresh, {n} thread(s)"] = lambda n=n: energy_matrix(
            rows, columns, model=model, temperature=temperature, threads=n
        )
    calls["fully cached"] = lambda: cached.matrix(rows, columns)
    elapsed = {label: [] for label in calls}
    order = list(calls)
    rng = Random(42)
    for run in range(repeats + 1):
        rng.shuffle(order)
        for label in order:
            start = perf_counter_ns()
            values = calls[label]()
            end = perf_counter_ns()
            if legacy_lists and isinstance(values, list):
                values = np.asarray(values, dtype=np.float32)
            check_matrix(values, expected, rows, columns, model, temperature, label)
            if run:  # all configurations receive one untimed warmup
                elapsed[label].append((end - start) / 1e6)
            del values  # exclude disposal of the previous result from the next call
    medians = {label: median(times) for label, times in elapsed.items()}
    baseline = medians["Python + ViennaRNA"]
    print(f"{name}: {len(rows)} x {len(columns)}, {len(set(rows)) * len(set(columns))} distinct pairs")
    for label, ms in medians.items():
        print(f"  {label:20} {ms:9.3f} ms   {baseline / ms:7.2f}x vs Python")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7, help="samples per configuration after warmup")
    parser.add_argument("--legacy-lists", action="store_true",
                        help="accept list output to time an installed 0.2.x wheel on identical data")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if RNA.__version__ != "2.8.0":
        raise RuntimeError(f"Expected ViennaRNA 2.8.0, found {RNA.__version__}")
    print(f"oligomatrix {version('oligomatrix')}, ViennaRNA {RNA.__version__}, "
          f"Turner 2004 at 37 C, median of {args.repeats} runs")
    rng = Random(91)
    left, right = ("".join(rng.choices("ACGT", k=60)) for _ in range(2))
    rows = [left[i:i + 16] for i in range(24)]
    columns = [right[j:j + 18] for j in range(32)]
    assert len(set(rows)) == 24 and len(set(columns)) == 32
    benchmark("Unique", rows, columns, args.repeats, args.legacy_lists)
    benchmark("Repeated", rows[:8] * 3, columns[:8] * 4, args.repeats, args.legacy_lists)
