# `atl::lyndon_factorization`, `atl::minimal_rotation_index`, `atl::runs`

Factor texts into non-increasing Lyndon words, select a least rotation, or
find every maximal periodic interval. Use these for ordered sequence
structure; matching alone needs only equality.

```cpp
#include <atl/strings/lyndon.hpp>
```

\include lyndon/quick.cpp

## API

Every function takes `R &&text, Proj proj = {}` and reads symbols in place.
`sortable_text<R, Proj>` refines `comparable_text`: the range is sized and
random-access, and projected symbols are totally ordered. Character types,
integers, proxy ranges and ordered tokens are accepted; ordering follows
the projected values, including the implementation's char signedness.

```cpp
template<typename R, typename Proj = std::identity>
    requires sortable_text<R, Proj>
[[nodiscard]] constexpr std::vector<uint32> lyndon_factorization(R &&text, Proj proj = {});
template<typename R, typename Proj = std::identity>
    requires sortable_text<R, Proj>
[[nodiscard]] constexpr uint32 minimal_rotation_index(R &&text, Proj proj = {});
struct run {
    uint32 period; // smallest positive period
    uint32 l;      // inclusive start
    uint32 r;      // exclusive end
    constexpr friend auto operator<=>(run const &, run const &) = default;
};
template<typename R, typename Proj = std::identity>
    requires sortable_text<R, Proj>
[[nodiscard]] constexpr std::vector<run> runs(R &&text, Proj proj = {});
```

`lyndon_factorization` returns boundaries `0 = b[0] < ... < b[k] = n`,
with factor i in `[b[i], b[i+1])`; empty input returns `{0}`.
`minimal_rotation_index` returns the start of a lexicographically least
rotation, or zero for empty input. `runs` returns unique records sorted
by `(period, l, r)`. Each interval has smallest period p and length at
least 2p, and cannot extend either way while preserving p. Comparison of
`run` records is lexicographic by those fields.

## Complexity

Factorization and rotation take O(n) comparisons. Factorization returns
O(n) boundaries; rotation uses O(1) extra space. Runs take O(n) time for
standard integral symbols no wider than uint64 with value span at most
min(n, 2^32 - 3); otherwise rank compression costs O(n log n). Runs use O(n)
storage. Suffix sorting recurses on geometrically smaller reduced texts,
with O(log n) stack depth; other passes are iterative. No input is retained.

## Preconditions

Asserted: factorization size is below 2^32, rotation size below 2^31,
and runs size below 2^32 - 2. Actual allocations must fit the platform's
containers, including on 32-bit targets.

Documented only: projections are stable, comparisons define a total order
consistent with equality. Wide integral spans and int128 use compression;
they are not truncated to uint64. Sparse integral alphabets also use
compression so suffix-sort buckets occupy O(n) space. All comparison APIs permit non-contiguous
random-access ranges. Ties between equal rotations need no caller tie rule.

## Measured performance

`benchmarks/strings/lyndon.cpp` races full runs construction on four
10^6-symbol texts. The library builds only inverted-order suffix ranks;
by full suffix arrays also computes and retains inverted-order LCP values.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | library -O2 | by full suffix arrays -O2 | library native | by full suffix arrays native |
|---|---:|---:|---:|---:|
| fib 10^6 | 234 (1.07x) | 247 (1.04x) | 252 (1.03x) | 259 (1.04x) |
| all-a 10^6 | 51.7 (1.02x) | 53 (1.02x) | 49.6 (1.06x) | 50.3 (1.06x) |
| random2 10^6 | 324 (1.05x) | 339 (1.04x) | 335 (1.07x) | 350 (1.06x) |
| random26 10^6 | 285 (1.04x) | 293 (1.07x) | 294 (1.05x) | 306 (1.04x) |

## Pitfalls and notes

A run's period is its smallest period, not an arbitrary repetition length.
Its two endpoints are half-open. Factor boundaries include both zero and n;
empty text therefore has one boundary and zero factors. Factorization and
minimal rotation are not measured yet.

## How it works

Duval grows a candidate while a trailing index follows repeated factors,
then emits the completed equal factors. Rotation compares two candidates in
virtual doubled coordinates and skips a span that cannot contain a minimum.

Runs use the Lyndon-root characterization under both alphabet orders.
Next-smaller-suffix stacks identify roots; forward and reversed suffix-array
LCP queries extend candidates in both directions. A bare suffix sort supplies
the inverted ranks without an unused LCP pass. Two counting passes on
`(period, l)` sort candidates; equal runs become adjacent and are removed.
Bucket positions use size_t so candidate counts cannot wrap uint32.
References: Duval (1983), “Factorizing words over an ordered alphabet”;
Bannai et al. (2017), “The Runs Theorem”.

## See also

[String matching](matching.md), [Range minimum queries](rmq.md),
[Suffix arrays](suffix-array.md).
