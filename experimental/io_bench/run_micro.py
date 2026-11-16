#!/usr/bin/env python3
# Isolate pure input (parse) and pure output (format) cost per backend.
# Env: CXX (default g++-15), N, RUNS.
import os
import random
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
N = int(os.environ.get("N", "2000000"))
RUNS = int(os.environ.get("RUNS", "5"))

if not 0 < N < 2**32 or RUNS <= 0:
    raise SystemExit("N must be positive uint32, Q nonnegative uint32, RUNS positive")

with tempfile.TemporaryDirectory(prefix="io_micro_") as work:
    def compile(name, defs):
        out = os.path.join(work, name + (".exe" if os.name == "nt" else ""))
        src = os.path.join(BENCH, "micro.cpp")
        if IS_MSVC:
            command = [*COMPILER, "/std:c++latest", "/O2", "/EHsc", f"/Fe{out}", f"/Fo{out}.obj", src, f"/I{ROOT / 'include'}", *["/D" + flag[2:] for flag in defs]]
        else:
            command = [*COMPILER, "-std=c++23", "-O2", "-o", out, src, *INC, *defs]
        subprocess.run(command, check=True, cwd=work)
        return out


    def timeit(b, inp):
        best = float("inf")
        for _ in range(RUNS):
            with open(inp, "rb") as fi, open(os.devnull, "wb") as fo:
                t = time.monotonic()
                subprocess.run([b], stdin=fi, stdout=fo, check=True)
                best = min(best, time.monotonic() - t)
        return best


    def run_group(title, mode_def, inp, variants):
        print(f"\n== {title} (N={N}) ==")
        bins = {name: compile(f"{mode_def}_{name}".replace("/", "_").replace("-", "_"), [f"-D{mode_def}", *defs]) for name, defs in variants}
        ref = refname = None
        for name, b in bins.items():
            with open(inp, "rb") as fi:
                out = subprocess.run([b], stdin=fi, capture_output=True, check=True).stdout
            if ref is None:
                ref, refname = out, name
            elif out != ref:
                print(f"  !! {name} disagrees with {refname}")
                sys.exit(1)
        base = None
        for name, b in bins.items():
            ms = timeit(b, inp) * 1000
            if base is None:
                base = ms
            print(f"  {name:12} {ms:8.1f} ms  {base / ms:4.1f}x")


    print(f"compiler={CXX}  N={N} runs={RUNS}")

    read_inp = os.path.join(work, "read.txt")
    rng = random.Random(1)
    with open(read_inp, "w") as f:
        f.write(f"{N}\n")
        f.write(" ".join(str(rng.randrange(1_000_000_000)) for _ in range(N)))
    write_inp = os.path.join(work, "write.txt")
    with open(write_inp, "w") as f:
        f.write(f"{N}\n")

    run_group("READ (pure parse)", "BENCH_READ", read_inp,
              [("cin", ["-DIO_CIN"]), ("println", ["-DIO_PRINTLN"]), ("getchar", ["-DIO_GETCHAR"]), ("atl", ["-DIO_ATL"])])

    run_group("WRITE (pure format)", "BENCH_WRITE", write_inp,
              [("cin", ["-DIO_CIN"]), ("println", ["-DIO_PRINTLN"]), ("getchar", ["-DIO_GETCHAR"]),
               ("atl (lut2)", ["-DIO_ATL"]), ("atl-lut4", ["-DIO_ATL", "-DATL_IO_BIG_LUT"])])
