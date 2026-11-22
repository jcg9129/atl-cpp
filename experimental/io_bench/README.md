# io_bench

Compares `atl::io`, `cin/cout`, C++23 `std::println`, and an unlocked-stdio
parser. `bench.cpp` builds prefix sums and answers half-open range-sum queries;
the compute is identical in every backend.
`micro.cpp` isolates parsing and formatting from that workload.

Run from any working directory with a plain Python 3 installation and a C++23
compiler with floating charconv and `<print>` support:

```sh
python3 experimental/io_bench/run.py
python3 experimental/io_bench/run_micro.py
```

Only Python's standard library is used. Set `CXX` to select the compiler
(default `g++-15`; GCC/Clang driver syntax and `cl`/`clang-cl` are supported).
MSVC requires a developer command prompt. `N`, `Q` (range-sum runner only), and
`RUNS` select the workload size and repetitions. `N` must be a positive uint32,
`Q` a nonnegative uint32, and `RUNS` positive. Defaults are 500000/500000/5 for
range sums and 2000000/5 for the microbenchmarks.

The runners build sequentially in a temporary directory, compare complete output
across variants, then report minimum wall times. Input comes from a real file;
output goes to the platform null device. Temporary artifacts are removed on exit.
`atl-mmap` exercises mappings on POSIX and the buffered fallback on Windows.
Unlocked stdio uses the POSIX names or the Windows CRT equivalents.

`gen.cpp` accepts `N Q [seed]` and generates values and valid `[l,r)` queries.
Backend definitions are `IO_CIN`, `IO_PRINTLN`, `IO_GETCHAR`, or `IO_ATL`, with
optional `ATL_USE_MMAP` / `ATL_IO_BIG_LUT` for atl. Microbenchmark modes are
`BENCH_READ` and `BENCH_WRITE`.

The top-level CMake option `ATL_BUILD_IO_HARNESS` (default OFF) builds and links
these programs independently of `BUILD_BENCHMARKS` and `BUILD_TESTS`. Enable it
with `-DATL_BUILD_IO_HARNESS=ON` and build `io-harness`. The CI `benchmarks`
preset enables the option and selects this target alongside `benchmarks`.
The executables are never registered with CTest, run by CI, or installed.
`ATL_BENCHMARK_ASSERTS` controls assertions as for the other benchmarks.

| Target | Source / definitions |
|---|---|
| `io-bench-generator` | `gen.cpp` |
| `io-bench-range-<backend>` | `bench.cpp`, one of `IO_CIN`, `IO_PRINTLN`, `IO_GETCHAR`, `IO_ATL` for `cin`, `println`, `getchar`, `atl` |
| `io-bench-range-atl-mmap` | `bench.cpp`, `IO_ATL ATL_USE_MMAP` |
| `io-bench-read-<backend>` | `micro.cpp`, each backend above plus `BENCH_READ` |
| `io-bench-write-<backend>` | `micro.cpp`, each backend above plus `BENCH_WRITE` |
| `io-bench-write-atl-lut4` | `micro.cpp`, `IO_ATL ATL_IO_BIG_LUT BENCH_WRITE` |

These are the same configurations built by `run.py` and `run_micro.py`.
CMake compile-and-link probes omit all `println` targets if `<print>` and
`std::println` are unavailable, and all `getchar` targets if the POSIX unlocked
stdio names or Windows CRT equivalents are unavailable. Configuration reports
any omission. Other backends do not include `<print>`. The Python runners
still require every listed backend; they perform comparisons and timings only
when explicitly invoked outside CI.

The programs make no recorded performance claim; results must be collected on
the target CPU with compiler flags and measurement metadata.
