"""Regenerate the bundled PTX using NVRTC 12.9.86, or check source freshness."""

import argparse
from hashlib import sha256
from pathlib import Path


KERNEL = Path(__file__).resolve().parents[1] / "src" / "cuda"
OPTIONS = (b"--gpu-architecture=compute_75", b"--std=c++14", b"--fmad=false")


def compile_ptx(source, energy, fold):
    from cuda.bindings import nvrtc

    def checked(result):
        status, *values = result
        if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC returned {status}")
        return values

    (program,) = checked(nvrtc.nvrtcCreateProgram(source, b"fold.cu", 2, [energy, fold], [b"energy.h", b"fold.h"]))
    try:
        status, = nvrtc.nvrtcCompileProgram(program, len(OPTIONS), OPTIONS)
        if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            (size,) = checked(nvrtc.nvrtcGetProgramLogSize(program))
            log = bytearray(size)
            checked(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(log.rstrip(b"\0").decode())
        (size,) = checked(nvrtc.nvrtcGetPTXSize(program))
        ptx = bytearray(size)
        checked(nvrtc.nvrtcGetPTX(program, ptx))
        return ptx.rstrip(b"\0").decode()
    finally:
        checked(nvrtc.nvrtcDestroyProgram(program))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify sources without loading CUDA")
    args = parser.parse_args()
    source, energy, fold = [(KERNEL / name).read_text(encoding="utf-8").encode()
                      for name in ("fold.cu", "../gpu/energy.h", "../gpu/fold.h")]
    digest = sha256(b"\0".join((source, energy, fold, *OPTIONS))).hexdigest()
    stamp = f"// oligomatrix CUDA sources and compiler options: {digest}\n"
    path = KERNEL / "fold.ptx"
    if args.check:
        if not path.is_file() or not path.read_text(encoding="ascii").startswith(stamp):
            raise SystemExit("Bundled PTX is stale. Run python scripts/build_cuda.py")
    else:
        ptx = compile_ptx(source, energy, fold).rstrip("\r\n") + "\n"
        path.write_text(stamp + ptx, encoding="ascii", newline="\n")


if __name__ == "__main__":
    main()
