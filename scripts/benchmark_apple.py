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


# All 400,000 values computed with the Python bindings for ViennaRNA 2.8.0.
# Regenerate with --oracle-workers 8. The byte order and dtype are set in digest().
VIENNA_SHA256 = "602e2075bebe552623041d3466afa0bd54dea77fd2df4872604e709ac83435fb"


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
        pieces = [template[start:start + length] for start, length in zip(starts, lengths, strict=True)]
        groups.append([
            piece if index % 2 == 0 else piece.translate(complement)[::-1]
            for index, piece in enumerate(pieces)
        ])
    return groups


def vienna(group):
    return reference(group, group, model_details("mathews2004", 65.0), dtype=np.float32)


def digest(matrices):
    return sha256(np.asarray(matrices, dtype="<f4").tobytes()).hexdigest()


def timed(groups, backend):
    start = perf_counter()
    calc = EnergyCalculator(model="mathews2004", temperature=65, threads=8, backend=backend)
    values = [calc.matrix(group, group) for group in groups]
    elapsed = perf_counter() - start
    print(f"{backend} timed at {elapsed:.3f} s for 1000 sets; SHA-256 {digest(values)}", flush=True)
    return np.stack(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("metal", "cuda"), default="metal")
    parser.add_argument("--oracle-workers", type=int, help="compute the independent ViennaRNA digest only")
    args = parser.parse_args()
    if args.oracle_workers is not None and args.oracle_workers < 1:
        parser.error("--oracle-workers must be positive")
    if RNA.__version__ != "2.8.0":
        raise RuntimeError("Expected ViennaRNA 2.8.0")
    groups = inputs()
    if args.oracle_workers:
        with ProcessPoolExecutor(args.oracle_workers) as pool:
            result = digest(list(pool.map(vienna, groups, chunksize=5)))
            print("ViennaRNA SHA-256:", result, flush=True)
            if result != VIENNA_SHA256:
                raise AssertionError(f"Unexpected ViennaRNA digest: {result}")
        return
    gpu = timed(groups, args.gpu)
    if digest(gpu) != VIENNA_SHA256:
        raise AssertionError(f"{args.gpu} matrices differ from the full ViennaRNA 2.8.0 reference")
    np.testing.assert_array_equal(gpu, timed(groups, "cpu"))
    for i in (0, 499, 999):
        np.testing.assert_array_equal(gpu[i], vienna(groups[i]))
    print(f"All 400000 {args.gpu}/CPU energies matched the independent ViennaRNA 2.8.0 digest.", flush=True)


if __name__ == "__main__":
    main()
