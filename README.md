# atl

Header-only C++23 algorithms and data structures library. The focus is the **most
efficient implementation possible** — optimizing constant factors, not just
asymptotic complexity. Abstractions (notably std ranges/views) are used where
they aid clarity, but only when measured to add no overhead.

### How to build

Requires CMake 3.25+, Ninja, and C++23. Minimum development toolchains are
GCC 15 or Clang 22 (with libstdc++ 15), and AppleClang 17 / libc++ (Xcode 16.4).
CI selects these compiler majors and LLVM 22 formatting/lint tools; Linux
package updates may advance their patch versions. The full test suite runs on
Linux x86-64 (GCC, Clang, sanitizers, forced fallbacks), Linux ARM64 (GCC,
Clang), 32-bit Linux (GCC `-m32`), macOS ARM64 (AppleClang) and Windows x64,
x86 and ARM64 (MSVC 19.44, toolset 14.44, `/std:c++latest`); a change merges
only when every one of them passes. The installed target exports the `cxx_std_23` requirement,
leaving CMake to select the consumer language flag. MSVC CI uses
`/std:c++latest`; other MSVC language modes are unverified. Support claims
extend only to the configurations tested. The Windows jobs require toolset
14.44 on the selected images and fail if it is unavailable.

```sh
make test PRESET=gcc JOBS=2       # clang or appleclang also available
make sanitize JOBS=2
make headers JOBS=2              # each header alone, normal + ATL_NO_INT128; ODR
make install-check              # install to scratch, build and run external consumer
make check JOBS=2                # format, tidy, GCC tests, headers
make clean-build PRESET=gcc      # remove only build/gcc
```

The Makefile and CI share `CMakePresets.json`: equivalently run `cmake --preset
gcc`, `cmake --build --preset gcc`, and `ctest --preset gcc`. Each preset uses
`build/<preset>`; tests fail if none are discovered. ccache is enabled if found
(disable with `-DENABLE_CCACHE=OFF`); an existing compiler launcher is preserved.
CI requires ccache on Linux and installs ccache and Ninja on macOS. Windows
jobs have no compiler cache. Tests and benchmarks are independently controlled
by `BUILD_TESTS` and `BUILD_BENCHMARKS`. The top-level-only
`ATL_BUILD_IO_HARNESS` option (default OFF) builds the standalone experimental
I/O backend executables; they are never registered with CTest or installed.
Every discovered source is built; missing includes fail compilation. The external publication allowlist must
ship tests only when all their included folders are published.

With `add_subdirectory` or FetchContent, tests, benchmarks and `ATL_INSTALL`
default to OFF. Explicitly enabled development targets use an `atl_` prefix;
standalone builds retain `headers`, `check`, `tests` and `benchmarks`.

The `ENABLE_*SAN` options add instrumentation without changing optimization
or debug flags. The `sanitizers` preset supplies `-O1 -g`, frame pointers and
non-recovering UBSan checks in its Debug flags.

Install with `cmake --install build/gcc --prefix <prefix>`, then consume with
`find_package(atl CONFIG REQUIRED)` and `target_link_libraries(app PRIVATE
atl::atl)`. The package installs only public headers and the CMake interface.

CI only **builds** benchmarks in the `benchmarks` preset, which enables
`ATL_BUILD_IO_HARNESS` and builds its `io-harness` target too. Optional harness
backends are omitted when compile-and-link probes cannot find `std::println`
or the platform unlocked-stdio functions. See
the harness README (`experimental/io_bench/README.md`) for targets and backends.
Run and record benchmarks only on dedicated machines outside CI with
`ATL_BENCH_RESULTS=<dir> benchmarks/record.sh <area>-<name>` (e.g.
`utils-counting_sort`); records go to `<dir>`, which must exist outside the
repository. The recorder takes any configured build tree with
`BUILD_BENCHMARKS=ON` (Ninja or Makefiles), selected via `ATL_BENCH_TREE`, and
reads the compile flags from its `compile_commands.json`; the `benchmarks` preset
also provides a local `run_benchmarks` target. Benchmarks disable `ATL_ASSERT`
by default. Set
`ATL_BENCH_FLAGS="-march=native"` for machine-specific measurements; each flag
set gets its own build tree. No timings are recorded on shared CI runners.

`atl::int128` and `atl::uint128` are always available: native integers where
supported, portable software classes otherwise. `ATL_NO_INT128` selects software
aliases; `ATL_FORCE_FALLBACKS` also forces portable word arithmetic. Keep these
switches consistent across translation units. See [128-bit integers](docs/int128.md).

### Code style

-   Use C++23 and standard headers and functions.
-   Headers must survive single-translation-unit amalgamation and must not
    depend on transitive includes.
-   atl's own code is original; `external/catch2` is Catch2, Boost Software
    License 1.0.
-   Enforce constraints with concepts and traits wherever possible: constrain
    template parameters (`std::semiregular`, `atl::unsigned_integral`, …) so
    misuse fails at instantiation with a clear error instead of deep inside a
    template.
-   Give template parameters meaningful names when they have a clear role
    (`Base`, `Exponent`, `Mod`, `Compare`, …); `T`/`U` are fine only for
    genuinely generic types (a container element, a value type).
-   Prefer `constexpr` wherever code can be constant-evaluated, so functions and
    data structures work at compile time. The exceptions are stream operators
    and code that reads runtime-only state (e.g.
    `dynamic_mod_int`'s runtime modulus). When adding `constexpr` code, add a
    `static_assert` to the owning folder's
    `tests/<folder>/constexpr_checks.cpp` to prove it actually evaluates at compile time.
-   Comments should explain the non-obvious — the *why*, a subtlety, or an
    invariant — not restate what the code already says plainly.
-   Prefer `std::ranges` algorithms and `std::views` over iterator-pair `std::`
    calls — `std::ranges::sort(v)`, not `std::sort(v.begin(), v.end())`. It's a
    zero-cost readability win, enforced by clang-tidy (`modernize-use-ranges` is
    an error).
-   Prefer the fastest correct implementation, shaving constant factors. Reach
    for ranges/views where they aid clarity, but benchmark any new view against
    a hand-written loop and keep it only if it adds no overhead.
-   Avoid unnecessary copies. Take a sink parameter by value and `std::move` it
    to its destination; move (don't copy) a local into a container
    (`v.push_back(std::move(x))`); and thread a value through call layers with
    perfect forwarding (`Args&&...` + `std::forward`) where it is only passed
    along. Provide `emplace`-style members that construct the element in place
    from forwarded arguments wherever the equivalent `std` container does
    (`emplace_back`, `emplace_front`, `emplace`), so a caller can skip building a
    temporary. Some copies are unavoidable (e.g. path-copying a persistent node
    copies its untouched siblings) — eliminate only the avoidable ones.

### Documentation

Guide pages and the public API reference form one Doxygen site. Requires
Doxygen 1.15.0+, Python 3, and the build tools above:

```sh
make docs                         # plain Doxygen, build/docs/docs/html/index.html
make test PRESET=gcc JOBS=1        # includes executable guide examples and coverage
```

For the sidebar theme and dark-mode toggle, fetch the pinned theme into scratch
space (never vendor its attribution-licensed files into this repository):

```sh
theme_dir=$(mktemp -d)
curl -fL https://github.com/jothepro/doxygen-awesome-css/archive/46483f1e5a70ffb9ecd3b82d0a1cd1b24edf13da.tar.gz -o "$theme_dir/theme.tar.gz"
echo "65abe1d85bc2adf9d1a1b6ef90ae5e63f72425024196f89ec27b7bdc52d94267  $theme_dir/theme.tar.gz" | sha256sum -c -
tar -xzf "$theme_dir/theme.tar.gz" -C "$theme_dir" --strip-components=1
make docs ATL_DOXYGEN_AWESOME_DIR="$theme_dir"
rm -rf -- "$theme_dir"
```

Equivalently configure `cmake --preset docs -DATL_DOXYGEN_AWESOME_DIR=/path/to/theme`
and run `cmake --build --preset docs --parallel 1`. The theme path may be read-only.
The docs option (`ATL_BUILD_DOCS`, default OFF) is available only to the top-level
project. Documentation warnings fail the build; inspect `build/docs/docs/warnings.log`.
See the [guide](docs/README.md) and [documentation conventions](docs/conventions.md).

### License

atl is licensed under the [BSD Zero Clause License (0BSD)](LICENSE).
Vendored Catch2 in `external/catch2` is licensed under BSL-1.0; its license
notice is included in its source files.
