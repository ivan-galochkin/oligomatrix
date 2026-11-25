"""Compare the public Python API, cell by cell, with ViennaRNA 2.8.0."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from random import Random
from time import perf_counter

import numpy as np
import RNA

from oligomatrix import EnergyCalculator, energy_matrix


MODELS = {
    "turner2004": RNA.params_load_RNA_Turner2004,
    "mathews2004": RNA.params_load_DNA_Mathews2004,
}

# At 37 C with Turner parameters these pairs exercise the indicated structures.
# The oracle computes the expected values again for every temperature and model.
TARGETS = [
    ("separated hairpin", "GGGGAAAACCCC", "AAAAAAA"),
    ("two separated folds, float32 addition", "CCTGAGCCGCCGGGCTGCAGACTTAT",
     "GTATCCACGTGGTGATATCACAGGCAAAGCAGTCATT"),
    ("duplex up to the strand break", "GGGGGG", "CCCCCC"),
    ("dangling ends", "AGGGGGGT", "ACCCCCCT"),
    ("wobble stack", "GGGTGGG", "CCCGCCC"),
    ("unpaired", "A" * 60, "A" * 60),
    ("long hairpin", "GGGGG" + "A" * 31 + "CCCCC", "A"),
    ("triloop", "GCGCGCAACGCGCGC", "A"),
    ("tetraloop", "GCGCGCCAAGGCGCGC", "A"),
    ("tetraloop with literal T", "GCGCGCTTCGGCGCGC", "A"),
    ("hexaloop with literal T", "GCGCGACAGTACTCGCGC", "A"),
    ("bulge", "CTCTTCGCAAGGATTTCAGAAGATC", "GCTTCTAGGTCCTGGGGTCGCATATTGACCACGCAAG"),
    ("1x1 internal loop", "GTGCTAGTGAGGCTCCGACCTAAATTAGACCGTTGTCGTCTGCCAGA",
     "ATGTCGGCCTTCACTATCTTCCTTCACTGGACGCCTGACTGCTC"),
    ("1x2 internal loop", "ACGAACCTTAATCATTGTTTCCTCGTTGCGA",
     "GACCACTGTGCGAGCCGGGCCATGAAAAGAATCTC"),
    ("2x2 and 3x4 internal loops", "TCTAAGGCATATTATCTTTCGTCCGACACCGCAAGTTTCTCC",
     "AGGTGTGCTCACGGAATTTAACACCA"),
    ("2x3 internal loop", "GGATACGGCACTAGTTGGCGTTCGATCACGGAAGTC",
     "ACATGTCAAAACATGGAATACCACCACCGAGACGCAGGTAACGAGGCT"),
    ("multiloop within a strand", "GCGCGAGGGGAAAACCCCAGGGGAAAACCCCACGCGC", "A"),
    ("multiloop containing the break", "GTCCAAGTAACAAAGCGCTCAAGTACTGAGC",
     "CAGTAGGTACCTATATCAACACAAGAACTA"),
    ("multiloop with break inside a branch", "TTCGTACCTGGTAGCCACATGTGA",
     "ACACAGAGCTCCTCCGTTCGGCAATAGGAGTTCATGGTCAAGC"),
]


def model_details(model, temperature):
    # The reference library uses process-global parameter loading. Build and use
    # each reference on the main thread, including references for concurrent jobs.
    MODELS[model]()
    md = RNA.md()
    md.temperature = temperature
    md.min_loop_size = 0
    return md


def reference(rows, columns, md, dtype=np.float64):
    expected = np.empty((len(rows), len(columns)), dtype=dtype)
    for i, a in enumerate(rows):
        for j, b in enumerate(columns):
            expected[i, j] = RNA.fold_compound(f"{a}&{b}", md).mfe_dimer()[1]
    return expected


def check_matrix(actual, expected, rows, columns, model, temperature, label):
    context = f"{label}, {model}, {temperature:g} C"
    if not isinstance(actual, np.ndarray):
        raise AssertionError(f"{context}: expected numpy.ndarray, received {type(actual).__name__}")
    if actual.shape != expected.shape or actual.dtype != np.float32 or not actual.flags.c_contiguous:
        raise AssertionError(
            f"{context}: expected shape {expected.shape}, float32, C-contiguous; "
            f"received shape {actual.shape}, {actual.dtype}, C-contiguous={actual.flags.c_contiguous}"
        )
    bad = np.argwhere(actual != expected)
    if len(bad):
        i, j = bad[0]
        a, b = rows[i], columns[j]
        md = model_details(model, temperature)
        structure, _ = RNA.fold_compound(f"{a}&{b}", md).mfe_dimer()
        raise AssertionError(
            f"{context}: {len(bad)}/{actual.size} cells differ. First at [{i},{j}]\n"
            f"  strands    {a}&{b}\n"
            f"  structure  {structure[:len(a)]}&{structure[len(a):]}\n"
            f"  ViennaRNA  {expected[i,j].item()!r} kcal/mol\n"
            f"  Rust       {actual[i,j].item()!r} kcal/mol"
        )
    return actual.size


def random_batches(seed):
    # Both axes cover every length 1..60. Shuffle lengths independently, so
    # short strands are paired with long strands as well as with each other.
    rng = Random(seed)
    axes = []
    for _ in range(2):
        fragments = ["".join(rng.choices("ACGT", k=n)) for n in range(1, 61)]
        rng.shuffle(fragments)
        axes.append(fragments)
    rows, columns = axes
    return [(rows[i:i + 5], columns[i:i + 5]) for i in range(0, 60, 5)]


def unique_axes(seed, shape, lengths):
    rng = Random(seed)
    axes = []
    for size in shape:
        fragments = {}
        while len(fragments) < size:
            length = lengths[len(fragments) % len(lengths)]
            fragments["".join(rng.choices("ACGT", k=length))] = None
        axes.append(list(fragments))
    return axes


def check_load(model, rows, columns, label, *, expand_repeats=False):
    temperature = 37.0
    md = model_details(model, temperature)
    start = perf_counter()
    expected = reference(rows, columns, md)
    timings = [f"ViennaRNA {perf_counter() - start:.3f}s"]
    count = 0
    for workers in (1, 8):
        start = perf_counter()
        actual = energy_matrix(rows, columns, model=model, temperature=temperature, threads=workers)
        timings.append(f"Rust/{workers} {perf_counter() - start:.3f}s")
        count += check_matrix(actual, expected, rows, columns, model, temperature,
                              f"{label}, fresh calculator, threads={workers}")
    print(f"{model}, {label}, {len(rows)} x {len(columns)}: " + ", ".join(timings), flush=True)
    if expand_repeats:
        # Seed one quarter of the 32 x 48 distinct pairs; the next call mixes
        # hits and misses across multiple worker batches in shuffled output order.
        rows, columns = rows[:32], columns[:48]
        expected = expected[:len(rows), :len(columns)]
        calc = EnergyCalculator(model=model, temperature=temperature, threads=8)
        seed_rows, seed_columns = rows[::2], columns[::2]
        count += check_matrix(calc.matrix(seed_rows, seed_columns), expected[::2, ::2],
                              seed_rows, seed_columns, model, temperature, "prime partial cache")
        rng = Random(5729)
        row_ids = [i % len(rows) for i in range(1024)]
        col_ids = [j % len(columns) for j in range(1025)]
        rng.shuffle(row_ids)
        rng.shuffle(col_ids)
        rows, columns = [rows[i] for i in row_ids], [columns[j] for j in col_ids]
        # Expand independently computed ViennaRNA values to every output position.
        expanded = expected[np.ix_(row_ids, col_ids)]
        for cache_state in ("partially cached", "fully cached"):
            start = perf_counter()
            actual = calc.matrix(rows, columns)
            elapsed = perf_counter() - start
            count += check_matrix(actual, expanded, rows, columns, model, temperature, cache_state)
            print(f"{model}, {cache_state}, {len(rows)} x {len(columns)}: Rust/8 {elapsed:.3f}s",
                  flush=True)
    return count


def check_model(model, temperature, batches, extended):
    md = model_details(model, temperature)
    calc = EnergyCalculator(model=model, temperature=temperature, threads=4)
    count = 0

    def compare(rows, columns, label):
        expected = reference(rows, columns, md)
        actual = calc.matrix(rows, columns)
        return check_matrix(actual, expected, rows, columns, model, temperature, label)

    for label, a, b in TARGETS:
        count += compare([a], [b], label)
        count += compare([b], [a], label + " reversed")
    for i, (rows, columns) in enumerate(batches):
        count += compare(rows, columns, f"random batch {i}")
    if extended:
        short = ["".join(bases) for n in (1, 2) for bases in product("ACGT", repeat=n)]
        count += compare(short, short, "all ordered pairs of 1- and 2-mers")

    # Reuse pairs across reordered rectangular matrices and a shifted boundary.
    rows = ["GAT", "GAT", "TACA"]
    columns = ["AGC", "GACT", "ATCG", "GACT"]
    count += compare(rows, columns, "duplicate fragments")
    count += compare(rows[::-1], columns[::-1], "reordered cache hits")
    count += compare(["GATT", "ACA", rows[0]], ["AGC", "GACTA", "TCG"], "partial cache")
    expected = reference(rows, columns, md)
    returned = calc.matrix(rows, columns)
    returned[0, 0] = np.inf
    count += check_matrix(calc.matrix(rows, columns), expected, rows, columns,
                          model, temperature, "cache after modifying a returned array")
    for workers in (0, 1, 2, 4, 8):
        actual = energy_matrix(rows, columns, model=model, temperature=temperature, threads=workers)
        count += check_matrix(actual, expected, rows, columns, model, temperature, f"threads={workers}")
    return count


def check_api():
    calc = EnergyCalculator()
    for fn in (calc.matrix, energy_matrix):
        for rows, columns in (([], []), ([], ["A", "C"]), (["G", "T"], [])):
            shape = (len(rows), len(columns))
            check_matrix(fn(rows, columns), np.empty(shape), rows, columns,
                         "turner2004", 37.0, f"{fn.__name__} empty axis")
        for rows, columns in (([""], ["A"]), (["A"], ["A&U"]), (["ACNG"], []), ([], ["g"])):
            try:
                fn(rows, columns)
            except ValueError:
                continue
            raise AssertionError(f"{fn.__name__} accepted invalid fragments {rows}, {columns}")
    rows, columns = ["GGGGAAAACCCC"], ["AAAAAAA"]
    md = model_details("turner2004", 37.0)
    fc = RNA.fold_compound(f"{rows[0]}&{columns[0]}", md)
    _, energy = fc.mfe_dimer()
    assert energy < fc.mfe()[1], "the separated-chain fixture must favor the two monomers"
    for fn in (calc.matrix, energy_matrix):
        check_matrix(fn(rows, columns), np.array([[energy]]), rows, columns,
                     "turner2004", 37.0, f"{fn.__name__} defaults")


def check_concurrent():
    rows, columns = random_batches(73)[0]
    jobs = []
    for temperature in (16.25, 48.75):
        for model in MODELS:
            expected = reference(rows, columns, model_details(model, temperature))
            calc = EnergyCalculator(model=model, temperature=temperature, threads=2)
            jobs.append((calc, expected, model, temperature))
    count = 0
    with ThreadPoolExecutor(len(jobs)) as pool:
        calls = [pool.submit(calc.matrix, rows, columns) for calc, *_ in jobs]
        for call, (_, expected, model, temperature) in zip(calls, jobs):
            count += check_matrix(call.result(), expected, rows, columns,
                                  model, temperature, "concurrent independent calculators")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true",
                        help="also check three more random seeds, all 1- and 2-mers, and two more temperatures")
    parser.add_argument("--stress", action="store_true",
                        help="include --extended, dense 128 x 193 matrices, long strands and million-cell outputs")
    args = parser.parse_args()
    if RNA.__version__ != "2.8.0":
        raise RuntimeError(f"Expected ViennaRNA 2.8.0, found {RNA.__version__}")
    check_api()
    batches = random_batches(11)
    temperatures = [25.0, 37.0, 54.5]
    extended = args.extended or args.stress
    if extended:
        for seed in (41, 113, 293):
            batches.extend(random_batches(seed))
        temperatures.extend((-5.0, 70.0))
    total = 2
    for model in MODELS:
        for temperature in temperatures:
            cells = check_model(model, temperature, batches, extended)
            total += cells
            print(f"{model} at {temperature:g} C: {cells} cells matched", flush=True)
        # With 667 distinct pairs, even a default run crosses the 512-job batch boundary.
        total += check_load(model, *unique_axes(211, (23, 29), (16, 20, 24)), "multi-batch")
        if args.stress:
            total += check_load(model, *unique_axes(391, (128, 193), (16, 20, 24, 28, 32, 36)),
                                "dense", expand_repeats=True)
            total += check_load(model, *unique_axes(487, (8, 9), (80, 128, 192, 256)),
                                "long strands")
    total += check_concurrent()
    print(f"{total} cells matched ViennaRNA {RNA.__version__} exactly.")
