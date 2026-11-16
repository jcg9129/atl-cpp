#!/usr/bin/env python3
# Build every I/O variant of the prefix-sum bench, check they agree, and time
# them on a generated input. Env: CXX (default g++-15), N, Q, RUNS.
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = os.path.join(ROOT, "experimental", "io_bench")
CXX = os.environ.get("CXX", "g++-15")
INC = ["-I", str(ROOT / "include")]
COMPILER = [CXX]
IS_MSVC = Path(COMPILER[0]).stem.lower() in {"cl", "clang-cl"}
N = int(os.environ.get("N", "500000"))
Q = int(os.environ.get("Q", "500000"))
RUNS = int(os.environ.get("RUNS", "5"))

VARIANTS = [
    ("cin/cout", ["-DIO_CIN"]),
    ("println", ["-DIO_PRINTLN"]),
    ("getchar", ["-DIO_GETCHAR"]),
    ("atl-fread", ["-DIO_ATL"]),
    ("atl-mmap", ["-DIO_ATL", "-DATL_USE_MMAP"]),
]

if not 0 < N < 2**32 or RUNS <= 0 or not 0 <= Q < 2**32:
    raise SystemExit("N must be positive uint32, Q nonnegative uint32, RUNS positive")

with tempfile.TemporaryDirectory(prefix="io_bench_") as work:
    def compile(name, src, defs):
        out = os.path.join(work, name + (".exe" if os.name == "nt" else ""))
        if IS_MSVC:
            command = [*COMPILER, "/std:c++latest", "/O2", "/EHsc", f"/Fe{out}", f"/Fo{out}.obj", src, f"/I{ROOT / 'include'}", *["/D" + flag[2:] for flag in defs]]
        else:
            command = [*COMPILER, "-std=c++23", "-O2", "-o", out, src, *INC, *defs]
        subprocess.run(command, check=True, cwd=work)
        return out


    print(f"compiler={CXX}  N={N} Q={Q} runs={RUNS}")
    gen = compile("gen", os.path.join(BENCH, "gen.cpp"), [])
    inp = os.path.join(work, "input.txt")
    with open(inp, "w") as f:
        subprocess.run([gen, str(N), str(Q)], stdout=f, check=True)

    bins = {name: compile(name.replace("/", "_"), os.path.join(BENCH, "bench.cpp"), defs) for name, defs in VARIANTS}

    ref = refname = None
    for name, b in bins.items():
        with open(inp, "rb") as fi:
            out = subprocess.run([b], stdin=fi, capture_output=True, check=True).stdout
        if ref is None:
            ref, refname = out, name
        elif out != ref:
            print(f"  !! {name} disagrees with {refname}")
            sys.exit(1)
    print("correctness: all variants agree\n")


    def timeit(b):
        best = float("inf")
        for _ in range(RUNS):
            with open(inp, "rb") as fi, open(os.devnull, "wb") as fo:
                t = time.monotonic()
                subprocess.run([b], stdin=fi, stdout=fo, check=True)
                best = min(best, time.monotonic() - t)
        return best


    print(f"{'variant':12} {'min ms':>8}")
    base = None
    for name, b in bins.items():
        ms = timeit(b) * 1000
        if base is None:
            base = ms
        print(f"{name:12} {ms:8.1f}  {base / ms:4.1f}x vs cin")
