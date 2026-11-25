"""Compare RNAcofold --jobs and the Rust extension on the same 24,704 pairs."""

import argparse
import csv
from importlib.metadata import version
from io import StringIO
import os
from pathlib import Path
from random import Random
from statistics import median
import subprocess
from time import perf_counter

import numpy as np
import RNA
from oligomatrix import energy_matrix

from check_matrix import check_matrix, model_details, reference, unique_axes


def cofold(executable, threads, data, *, verbose=False):
    command = [str(executable), f"--jobs={threads}", "--noPS", "--noconv",
               "-P", "DNA", "-T", "37", "-d2", "--output-format=D", "--csv-noheader"]
    if verbose:
        command.extend(("--verbose", "--log-level=1"))
    run = subprocess.run(command, input=data, capture_output=True, check=True,
                         env=dict(os.environ, OMP_NUM_THREADS="1"))
    if b"without parallel input processing" in run.stderr:
        raise RuntimeError("RNAcofold was compiled without parallel input processing")
    if verbose and threads == 8 and b"Preparing 8 parallel computation slots" not in run.stderr:
        raise RuntimeError(f"RNAcofold did not confirm 8 slots: {run.stderr.decode(errors='replace')}")
    return run.stdout


def check_cli(output, pairs, expected):
    records = csv.reader(StringIO(output.decode("ascii")))
    differences = 0
    for row, pair, energy in zip(records, pairs, expected.flat, strict=True):
        if row[2] != pair:
            raise AssertionError(f"RNAcofold returned {row[2]} instead of {pair}")
        if row[4].strip() != f"{energy:.2f}":
            if differences == 0:
                print(f"First differing CLI energy for {pair}: expected {energy:.2f}, got {row[4].strip()}")
            differences += 1
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="RNAcofold 2.8.0 built with POSIX threads")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cli = args.executable.resolve(strict=True)
    cli_version = subprocess.check_output([str(cli), "--version"], text=True).strip()
    if cli_version != "RNAcofold 2.8.0" or RNA.__version__ != "2.8.0":
        raise RuntimeError(f"Version mismatch: {cli_version}, Python ViennaRNA {RNA.__version__}")
    print(f"{cli_version}, oligomatrix {version('oligomatrix')}; Mathews 2004, 37 C", flush=True)
    rows, columns = unique_axes(391, (128, 193), (16, 20, 24, 28, 32, 36))
    pairs = [f"{a}&{b}" for a in rows for b in columns]
    data = ("\n".join(pairs) + "\n").encode("ascii")
    warmup = ("\n".join(pairs[:16]) + "\n").encode("ascii")
    for n in (1, 8):
        cofold(cli, n, warmup, verbose=True)
        energy_matrix(rows[:1], columns[:16], model="mathews2004", threads=n)
    print(f"Verified native CLI worker configuration. Computing oracle for {len(pairs)} pairs.", flush=True)
    expected = reference(rows, columns, model_details("mathews2004", 37))
    order = [(name, n) for name in ("RNAcofold", "Rust") for n in (1, 8)]
    times = {key: [] for key in order}
    failures = 0
    rng = Random(480)
    for i in range(args.repeats):
        rng.shuffle(order)
        for name, n in order:
            if name == "RNAcofold":
                start = perf_counter()
                output = cofold(cli, n, data)
                elapsed = perf_counter() - start
                differences = check_cli(output, pairs, expected)
            else:
                start = perf_counter()
                result = energy_matrix(rows, columns, model="mathews2004", threads=n)
                elapsed = perf_counter() - start
                check_matrix(result, expected, rows, columns, "mathews2004", 37, f"threads={n}")
                differences = 0
            failures += differences
            times[name, n].append(elapsed)
            print(f"Run {i + 1}, {name}, {n} thread(s): {elapsed:.6f} s, "
                  f"{differences}/{len(pairs)} different energies", flush=True)
    for (name, n), samples in sorted(times.items()):
        print(f"Median {name} {n}: {median(samples):.6f} s")
    if failures:
        raise SystemExit(f"{failures} CLI energy comparisons differed across all timed runs.")


if __name__ == "__main__":
    main()
