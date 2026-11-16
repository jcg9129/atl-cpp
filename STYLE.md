# Coding style

This project uses **standard-library style**: everything is `snake_case`, so the
code reads like `<algorithm>` or `<numeric>`. Formatting is handled by
`clang-format` and naming by `clang-tidy`; both run in CI.

## Formatting

Defined by [`.clang-format`](.clang-format) (LLVM base, tabs, no column limit,
`atl/` includes sorted first, **east `const`** — `T const&`, not `const T&`;
declaration specifiers like `static`/`inline`/`constexpr` stay leading). Format
tracked sources with:

```sh
make fmt-fix  # LLVM 22; tracked C++ sources only
```

CI fails if any tracked source is not clang-format clean.

## Naming

| Kind | Convention | Example |
|------|------------|---------|
| Namespaces | `snake_case` | `atl`, `traits`, `detail` |
| Classes / structs | `snake_case` | `mod_int`, `sieve`, `gcd_result` |
| Concepts | `snake_case` | `integral`, `commutative_monoid` |
| Functions (free & member) | `snake_case` | `extended_gcd`, `is_prime` |
| Variables / parameters | `snake_case` | `prime_exp`, `total_blocks_` |
| Private / protected members | `snake_case` + trailing `_` | `value_`, `info_`, `masks_` |
| Public data members | `snake_case` | `prime`, `exponent` |
| Global / static constants | `snake_case` | `largest_prime_64`, `num_bits` |
| Template parameters | `CamelCase` | `T`, `U`, `Mod`, `Callback` |

### Intentional exceptions (not machine-enforced)

- **Type aliases** are not gated, because they span two deliberate idioms:
  - std-mirroring traits keep the `_t` / `_v` suffix so they drop in for
    `std::` equivalents: `make_signed_t`, `is_integral_v`, `widen_mul_type_t`.
  - short template-like local aliases stay uppercase: `using S = ...`.
- **Macros** are not gated: `atl_assert` (function-like, lower case) and the
  `ATL_ASSERT` feature guard coexist by design.

## Enforcement scope

Naming is enforced on the **library only** (`include/atl`). Tests and
benchmarks are exempt (see their local `.clang-tidy` files) because they use
pragmatic, math-idiomatic names (`N`, `A`, `Point`, …). The quality checks
(`bugprone-*`, `performance-*`, `modernize-*`) still apply everywhere as
advisory warnings.

## Running the linter locally

`clang-tidy-22` reads [`.clang-tidy`](.clang-tidy). Naming and use-ranges are
errors; the remaining checks are advisory. To lint the whole library the way CI does:

```sh
make fmt
make tidy
make tidy-proof  # deliberately invalid naming must fail
make check JOBS=2
```

To lint a single translation unit against the build's compile database:

```sh
clang-tidy-22 -p build/gcc path/to/test.cpp
```

## Portability & feature fallbacks

Minimum toolchains and tested platform scopes are listed in [README.md](README.md).
GCC 15 is the intentional development floor, matching the GCC CI preset.
Where the library uses a recent C++23
standard-library feature, the use is guarded by its standard feature-test macro
so a fallback compiles when a target toolchain lacks it — see
[`include/atl/config.hpp`](include/atl/config.hpp) for the list.

In `types/` and `utils/`, the feature guards are exactly:

- Word primitive runtime fast paths: native int128 or MSVC architecture intrinsics,
  disabled by `ATL_PORTABLE_WIDE_ARITHMETIC` or `ATL_FORCE_FALLBACKS`.
- `ATL_HAS_INT128`: `__SIZEOF_INT128__`, disabled by `ATL_NO_INT128` or
  `ATL_FORCE_FALLBACKS` (`config.hpp`).
- `__cpp_lib_forward_like`, disabled by `ATL_NO_STD_FORWARD_LIKE` or
  `ATL_FORCE_FALLBACKS` (`types/forward_like.hpp`).
- `__GNUC__` / `__clang__`, disabled by `ATL_FORCE_FALLBACKS`
  (`utils/prefetch.hpp`, otherwise a portable no-op).

`ATL_FORCE_FALLBACKS` means use every portable fallback. Architecture and
numeric-limits checks in `utils/hash.hpp` select known floating object layouts;
unknown wide layouts and `ATL_FORCE_FALLBACKS` use arithmetic decomposition. Outside these directories,
`__cpp_lib_to_chars` gates floating-point I/O in `io/reader` and `io/writer`;
its absence disables floating-point I/O rather than selecting a portable algorithm.

Pattern:

```cpp
#include <atl/config.hpp>            // pulls in <version>
#include <utility>
#if defined(__cpp_lib_forward_like) && !defined(ATL_NO_STD_FORWARD_LIKE) && !defined(ATL_FORCE_FALLBACKS)
// … use std::forward_like …
#else
// … equivalent portable cast …
#endif
```

Rules:

- **Keep both branches valid.** The fallback is only compiled when forced or on
  an older toolchain, so it isn't checked by day-to-day builds — the CI
  `fallbacks` job (and `cmake -DENABLE_PORTABLE_FALLBACKS=ON`) forces every
  `#else` path so it stays working.
- **System `#include`s stay unconditional, at file scope** — never inside a
  `#if`/`#ifdef`. The *usage* is guarded; the include is not. During
  single-translation-unit amalgamation, a guarded include can otherwise remain
  inside `namespace atl`, nesting the standard library and breaking compilation.
  Unconditional includes also let `clang-include-cleaner` inspect inactive branches.
  Every guarded header (`<ranges>`, `<format>`, …) exists at the documented toolchain floor;
  only the feature within it may be missing.
- **Amalgamation.** Preserve feature guard macros and both implementation branches
  when generating a portable single translation unit.

### 128-bit integers

`__int128` is a GCC/Clang extension available on many 64-bit targets. When
`ATL_HAS_INT128` is 0, `int128`/`uint128` alias portable software classes. Traits,
64-bit widening and all published types and utilities remain available. The
concrete software classes are always defined for differential testing; see
[128-bit integers](docs/int128.md) for their semantics and class limitations.

`ATL_NO_INT128` selects software aliases. `ATL_PORTABLE_WIDE_ARITHMETIC` selects
portable word primitives independently of the aliases. `ATL_FORCE_FALLBACKS`
selects both, including on MSVC. Configuration switches must match across
translation units (inline definitions and, for alias selection, ABI).
Constant evaluation always uses portable primitive bodies;
runtime intrinsics are guarded by compiler and architecture (MSVC x64, ARM64,
or portable x86). The platform-only `<intrin.h>` include is necessarily guarded
at file scope; preserve that guard during amalgamation. Preserve both branches
of the portable implementation. The `no-int128` and `fallbacks` presets run the
full suite, including software/native differential tests where native integers exist.

### `atl::io` tuning knobs

The fast I/O (`include/atl/io/`) defaults are portable and safe; two opt-in
macros trade portability/size for a little speed, and were chosen by measuring
across target platforms. These two options are marginal and target-dependent:

- **`ATL_USE_MMAP`** — mmap a regular-file stdin instead of `fread`. POSIX only
  (Windows/MinGW has no `<sys/mman.h>`; the default fread path is used there
  regardless). A small speedup on some Linux workloads, but it makes the whole
  input resident in RAM (memory ≈ input size), so it is **off by default**.
- **`ATL_IO_BIG_LUT`** — a 4-wide (40 KB) output digit table instead of the
  2-wide (200 B) default. Faster on some x86 targets, a wash on others, slower
  on aarch64 — benchmark your target before enabling. **Off by default.**

Both branches are inactive unless defined. Enable an option with `-D` when a
benchmark on the target platform justifies it.
