# experimental/

Prototype and reference implementations that are **not** part of the public
`atl` library. Everything here lives in namespace **`atl::experimental`** and is
included as `<atl/experimental/...>`.

Use it for:

- alternative implementations you want to benchmark against the library implementation
  (e.g. `atl::experimental::factorization_by_trial_division` vs `atl::factorization`),
- prototypes not yet ready for `include/atl`,
- reference "obviously correct" versions used to cross-check the fast ones.

Guarantees are intentionally weaker than the main library:

- **Separate include root.** These headers are outside `include/atl`; callers
  opt in with `-Iexperimental`. They must also survive single-translation-unit
  amalgamation.
- **Loosely gated.** The strict naming CI (which lints only `include/atl`) does
  not cover this tree; keep it reasonable but don't sweat it.

## Using it

CMake exposes an `atl_experimental` interface target (adds this directory to the
include path and links `atl`). Benchmarks already link it; for another consumer,
add `atl_experimental` to its `target_link_libraries`.

```cpp
#include <atl/experimental/nt/factorization_trial.hpp>
auto f = atl::experimental::factorization_by_trial_division(n);   // vs atl::factorization(n)
```

## Layout

Mirror the `atl/` layout under `atl/experimental/`:

```
experimental/atl/experimental/nt/factorization_trial.hpp   // namespace atl::experimental
```
