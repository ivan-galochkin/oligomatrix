"""Reproduce the 1,000 independent 20-oligo matrices timed on the RTX 5060 Ti."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from random import Random
from time import perf_counter

import numpy as np
import RNA
from oligomatrix import EnergyCalculator
from check_matrix import model_details, reference


def inputs():
    rng = Random(60465)
    lengths = (40, 60, 56, 52, 59, 51, 55, 57, 48, 60) * 2
    overlaps = (20, 27, 28, 23, 24, 22, 26, 25, 30) * 2 + (26,)
    starts = [0]
    for length, overlap in zip(lengths[:-1], overlaps, strict=True):
        starts.append(starts[-1] + length - overlap)
    assert starts[-1] + lengths[-1] == 600
    complement = str.maketrans("ACGT", "TGCA")
    groups = []
    for _ in range(1000):
        template = "".join(rng.choices("ACGT", k=600))
        pieces = [template[s:s+n] for s, n in zip(starts, lengths, strict=True)]
        groups.append([p if i % 2 == 0 else p.translate(complement)[::-1] for i, p in enumerate(pieces)])
    return groups


def vienna(group):
    return reference(group, group, model_details("mathews2004", 65.0), dtype=np.float32)


def digest(matrices):
    return sha256(np.asarray(matrices, dtype="<f4").tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-workers", type=int, help="compute the independent ViennaRNA digest only")
    args = parser.parse_args()
    if RNA.__version__ != "2.8.0":
        raise RuntimeError("Expected ViennaRNA 2.8.0")
    groups = inputs()
    if args.oracle_workers:
        with ProcessPoolExecutor(args.oracle_workers) as pool:
            print("ViennaRNA SHA-256:", digest(list(pool.map(vienna, groups, chunksize=5))), flush=True)
        return
    results = {}
    for backend in ("metal", "cpu"):
        start = perf_counter()
        calc = EnergyCalculator(model="mathews2004", temperature=65, threads=8, backend=backend)
        values = [calc.matrix(group, group) for group in groups]
        elapsed = perf_counter() - start
        results[backend] = np.stack(values)
        print(f"{backend} timed at {elapsed:.3f} s for 1000 sets; SHA-256 {digest(values)}", flush=True)
    np.testing.assert_array_equal(results["metal"], results["cpu"])
    for i in (0, 499, 999):
        np.testing.assert_array_equal(results["metal"][i], vienna(groups[i]))
    print("400000 Metal/CPU energies matched; sets 1, 500 and 1000 matched ViennaRNA directly.", flush=True)


if __name__ == "__main__":
    main()
