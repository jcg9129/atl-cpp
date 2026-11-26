# `atl::coordinate_compression`

The sorted distinct universe of values, mapping each to its lower-bound rank and
back. Use it to map sparse coordinates to dense array indices when all stored
coordinates are known before construction. Query endpoints may be absent; absent
values can have rank equal to `size()`, which is not a valid element index.

```cpp
#include <atl/ds/coordinate_compression.hpp>
```

\include coordinate-compression/basic.cpp

Expected stdout:

\verbinclude coordinate-compression/basic.out

## API

```cpp
template<std::movable T, typename Compare = std::ranges::less>
    requires std::strict_weak_order<Compare const&, T const&, T const&>
class coordinate_compression;

template<std::movable T, typename Compare = std::ranges::less>
    requires std::strict_weak_order<Compare const&, T const&, T const&>
struct compressed {
    coordinate_compression<T, Compare> universe;
    std::vector<uint32> ranks;
};
```

All operations are `constexpr`. `value_type` is `T`, `size_type` is `std::size_t`.
`coordinate_compression()` creates an empty universe. The range constructor and
`compress` take `template<std::ranges::input_range R, typename Proj = std::identity>`
(`compress` also has `typename Compare = std::ranges::less`). Projection must be
invocable on the forwarded reference. The range constructor checks
`std::constructible_from<T, std::invoke_result_t<Proj&, detail::forwarded_element_t<R>>>`,
matching its `static_cast<T>`; explicit conversions are accepted. Free `compress`
and the deduction guide check the same direct construction for their deduced,
movable stored value. Identity uses the range's value type, including for proxy
references. Lvalue ranges of move-only values are rejected when their const
references cannot construct stored values; owning rvalues can donate them.

| function | description | complexity |
|---|---|---|
| `coordinate_compression(R&& values, Proj proj = {}, Compare comp = {})` | build from a range, each element seen through `proj`; duplicates dropped; elements are moved only out of an owning rvalue range (never through a borrowed range or a view such as `views::filter`, whose source stays intact) and `proj` is checked against the element as forwarded | O(n log n) |
| `rank(T const& x)` → `std::size_t` | number of distinct values before `x` — the compressed index of `x` if present; lower-bound semantics if absent | O(log n) |
| `ranks(R&& queries)` → `std::vector<uint32>` | the `rank` of every query, in order; forward ranges with `q >= ceil(size()/64)` detect sortedness during a merge walk; at a descent, keep prefix ranks and sort and merge only the suffix for eligible trivially copyable query values, or search the suffix otherwise; smaller batches and non-forward inputs use binary searches | O(q + n) sorted; O(q log q + n) sort-merge; otherwise O(q log n) |
| `coordinate_compression(sorted_unique_t, std::vector<T> universe, Compare comp = {})` | adopt an already strictly increasing universe, no sort (order asserted) | O(1) |
| `rank_upper(T const& x)` → `std::size_t` | number of distinct values not after `x` — exclusive end for an *inclusive* right endpoint, no `rank(x + 1)` overflow trap | O(log n) |
| `operator[](std::size_t i)` → `T const&` | the `i`-th distinct value (inverse of `rank`) | O(1) |
| `contains(T const& x)` → `bool` | whether `x` is in the universe | O(log n) |
| `size()` / `empty()` | number of distinct values / emptiness | O(1) |
| `begin()` / `end()` | iterate the distinct values in order | — |

`T` is deduced from the projected element type, `Compare` (a
`std::strict_weak_order`, default `<`) defines the ordering — pass
`std::ranges::greater{}` for a descending universe. A default-constructed instance is
the empty universe (every rank 0). `sorted_unique` is the shared tag of
`<atl/types/tags.hpp>`. Single-value queries take `T const&`; bulk `ranks` accepts
mixed query types constrained by
`std::indirect_strict_weak_order<Compare const&, T const*, std::ranges::iterator_t<R>>`.
Both constructors, free `compress`, and the deduction guide check
`std::constructible_from<Compare, Compare&&>` for storing the comparator.
Ranges algorithms receive `std::cref(comp_)` (or a reference wrapper for a helper's
comparator), so they never copy the comparator. Const-callable move-only
comparators work for construction, single queries, and all bulk-rank paths.
The experimental pair/index baselines check their own direct stored-value
construction and comparator construction; the radix baseline checks its
projected-value cast without requiring a comparator.

\include coordinate-compression/move-only.cpp

\verbinclude coordinate-compression/move-only.out

Iterators are read-only; `operator[]` returns a reference. `size`, `empty`, `begin` and `end` are
`const noexcept`. Constructors invalidate any previous storage on assignment;
queries preserve references and iterators.

### Compressing and ranking in one pass — `compress`

```cpp
auto [universe, ranks] = atl::compress(xs);   // ranks[i] is xs[i]'s compressed index
```

`compress(R&& values, Proj proj = {}, Compare comp = {})` returns
`compressed<T, Compare>{universe, ranks}`: the universe plus every input value's rank
in input order (`std::vector<uint32>`), from fixed 11-bit LSD radix passes for integral keys under `<`
(including 128-bit integers, except `bool`) from 2^11 values, otherwise a comparison sort of (value, position) pairs, then a linear walk, with no per-value search. It is the "collect, compress, map everything"
step of an offline solution; moves elements out of an owning rvalue range like the
constructor. Recognized `<` comparator types are `std::ranges::less`,
`std::less<T>` and `std::less<>`; other comparators retain the comparison path.
Radix keys narrower than 32 bits use a 32-bit unsigned intermediate so an
11-bit digit fits, preserving the original value width when encoding and decoding.

## Complexity

The table counts comparisons and constant-cost moves. Construction sorts n values
in O(n log n) time with O(n) storage; its final universe occupies O(m), m distinct
values. Adopting a sorted vector is O(1), plus O(m) assertion checking. Individual
rank and membership queries use O(log m) comparisons and O(1) extra storage.
`compress` uses fixed-width radix passes for standard and atl integer types,
including native and software 128-bit keys; bool uses comparison sorting.

The final universe stores one `std::vector<T>` of distinct values. `compress` also
holds n keyed positions, radix scratch, and n ranks during construction. `ranks`
returns q 32-bit ranks and can use O(q) scratch for sort-merge.

## Preconditions

Asserted (`atl_assert`, active only under `-DATL_ASSERT`; the benchmarks run without it):

- the `sorted_unique` universe is strictly increasing under `Compare`;
- `operator[](i)` has `i < size()`;
- `ranks` and `compress` inputs have fewer than 2^32 elements (a non-sized range is
  checked as it is consumed); the universe queried by `ranks` also has fewer than 2^32 values.

Documented only: `Compare` is a strict weak order over `T`, and over `T` and the query
type for `ranks`; the queries of `ranks` are not moved from.

## Measured performance

Compression races the library's size dispatch against pair sort, index sort,
fixed keyed comparison sort, and radix digit widths. Bulk ranks race independent
searches, forced merge/sorted merge, rank loops, search/gallop dispatch, and a
separate sortedness pre-scan. Loops measure batch throughput, allocation and
rank checksums included, across random, duplicate-heavy, sorted, mostly sorted,
sparse and oversized inputs; rotated rows cycle through 16 vectors.

Source: [`benchmarks/ds/coordinate_compression.cpp`](../benchmarks/ds/coordinate_compression.cpp).


AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| n and input shape | library (recorded dispatch) | by keyed sort (radix11) | by keyed sort (comparison) | by pair sort |
|---|---|---|---|---|
| 1024 random | 0.011 / 0.01 | 0.011 / 0.011 | 0.01 / 0.01 | 0.01 / 0.01 |
| 1024, 1000 distinct | 0.01 / 0.01 | 0.012 / 0.011 | 0.01 / 0.01 | 0.009 / 0.009 |
| 2048 random | 0.02 / 0.02 | 0.02 / 0.019 | 0.023 / 0.022 | 0.022 / 0.021 |
| 2048, 1000 distinct | 0.021 / 0.02 | 0.021 / 0.02 | 0.023 / 0.022 | 0.022 / 0.021 |
| 3072 random | 0.03 / 0.028 | 0.029 / 0.027 | 0.042 / 0.04 | 0.039 / 0.039 |
| 3072, 1000 distinct | 0.03 / 0.029 | 0.03 / 0.028 | 0.038 / 0.039 | 0.036 / 0.037 |
| 4096 random | 0.038 / 0.037 | 0.038 / 0.036 | 0.062 / 0.058 | 0.055 / 0.054 |
| 4096, 1000 distinct | 0.039 / 0.039 | 0.039 / 0.038 | 0.058 / 0.056 | 0.052 / 0.053 |
| 16384 random | 0.146 / 0.143 | 0.146 / 0.138 | 0.853 / 0.828 | 0.76 / 0.75 |
| 16384, 1000 distinct | 0.175 / 0.169 | 0.173 / 0.165 | 0.674 / 0.661 | 0.627 / 0.622 |
| 32768 random | 0.371 / 0.377 | 0.366 / 0.364 | 1.8 / 1.8 | 1.6 / 1.6 |
| 32768, 1000 distinct | 0.301 / 0.291 | 0.299 / 0.283 | 1.4 / 1.3 | 1.3 / 1.3 |
| 65536 random | 0.779 / 0.782 | 0.777 / 0.758 | 3.8 / 3.7 | 3.4 / 3.4 |
| 65536, 1000 distinct | 0.54 / 0.527 | 0.54 / 0.51 | 2.8 / 2.8 | 2.7 / 2.7 |
| 131072 random | 1.6 / 1.6 | 1.6 / 1.5 | 8.1 / 7.9 | 7.3 / 7.2 |
| 131072, 1000 distinct | 1 / 0.996 | 1 / 0.96 | 5.5 / 5.5 | 5.2 / 5.3 |
| 1536 random, rotated | 0.064 / 0.062 | 0.017 / 0.016 | 0.065 / 0.062 | not measured yet |
| 1536, 1000 distinct, rotated | 0.068 / 0.065 | 0.018 / 0.018 | 0.067 / 0.065 | not measured yet |
| 8192 random, rotated | 0.081 / 0.077 | 0.079 / 0.076 | 0.407 / 0.397 | not measured yet |
| 8192, 1000 distinct, rotated | 0.092 / 0.09 | 0.092 / 0.089 | 0.369 / 0.36 | not measured yet |

Sizes 1536 and 8192 above use rotated inputs. The library dispatch changes
at 2048; the benchmark measures it at 2048 and 3072.
The library loses to fixed radix11 on rotated 1536 random inputs
(0.064/0.062 versus 0.017/0.016 ms), while 1024 inputs with 1000 distinct
values favor pair sort (0.009/0.009 versus 0.01/0.01 ms).

For bulk ranks, the columns distinguish the separate sortedness pre-scan
from the fused sortedness merge used by the library. “By forced merge” means
merge for sorted queries and sorted merge otherwise. Small-universe rows and
additional counts use rotated inputs in the source. Rounded zero does not
imply zero cost. Some small rows had unstable run spread.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| m | q | shape | by sortedness prescan | by search | by forced merge | by rank loop | library method (fused merge) |
|---|---|---|---|---|---|---|---|
| 1024 | 1024 | random | 0.01 / 0.01 | 0.01 / 0.01 | 0.01 / 0.01 | 0.01 / 0.01 | 0.011 / 0.01 |
| 1024 | 1024 | sorted | 0.002 / 0.002 | 0.008 / 0.008 | 0.002 / 0.002 | 0.008 / 0.008 | 0.002 / 0.002 |
| 1024 | 64 | random | 0.001 / 0.001 | 0.001 / 0.001 | 0.001 / 0.001 | 0.001 / 0.001 | 0.001 / 0.001 |
| 1024 | 64 | sorted | 0 / 0 | 0.001 / 0.001 | 0 / 0 | 0.001 / 0.001 | 0 / 0 |
| 4096 | 4096 | random | 0.063 / 0.061 | 0.128 / 0.112 | 0.065 / 0.06 | 0.126 / 0.086 | 0.057 / 0.062 |
| 4096 | 4096 | sorted | 0.008 / 0.008 | 0.058 / 0.054 | 0.008 / 0.007 | 0.054 / 0.05 | 0.01 / 0.008 |
| 4096 | 256 | random | 0.003 / 0.003 | 0.003 / 0.003 | 0.003 / 0.003 | 0.003 / 0.003 | 0.004 / 0.004 |
| 4096 | 256 | sorted | 0.002 / 0.002 | 0.003 / 0.003 | 0.002 / 0.001 | 0.003 / 0.003 | 0.002 / 0.002 |
| 65536 | 65536 | random | 3.7 / 3.7 | 5.9 / 5.9 | 3.7 / 3.7 | 5.9 / 5.9 | 3.8 / 3.8 |
| 65536 | 65536 | sorted | 0.443 / 0.418 | 1.9 / 1.9 | 0.428 / 0.406 | 1.9 / 1.9 | 0.468 / 0.438 |
| 1000000 | 15625 | random | 1.1 / 1.1 | 2.1 / 2.1 | 1.1 / 1.1 | 2.1 / 2.1 | 1.3 / 1.3 |
| 1000000 | 3906 | random | 0.495 / 0.498 | 0.487 / 0.494 | 0.366 / 0.364 | 0.47 / 0.495 | 0.498 / 0.472 |
| 1000000 | 976 | random | 0.022 / 0.022 | 0.022 / 0.022 | 0.295 / 0.295 | 0.021 / 0.022 | 0.023 / 0.025 |
| 100000 | 200000 | random | 12 / 12 | 16.7 / 16.6 | 12 / 12 | 16.7 / 16.7 | 11.8 / 11.8 |
| 100000 | 200000 | sorted | 0.924 / 0.844 | 4.7 / 4.6 | 0.914 / 0.857 | 4.7 / 4.6 | 1 / 0.975 |
| 100000 | 400000 | random | 24.5 / 24.4 | 33.5 / 33.3 | 24.5 / 24.4 | 33.4 / 33.3 | 23.9 / 23.7 |
| 100000 | 1600000 | random | 95.5 / 95.6 | 134 / 133 | 95.6 / 95.7 | 133 / 134 | 96.6 / 96.2 |
| 100000 | 1600000 | sorted | 2.6 / 2.2 | 22.4 / 21.7 | 2.8 / 2.3 | 22.4 / 22.2 | 3.5 / 3.1 |
| 1000000 | 2000000 | random | 138 / 139 | 268 / 268 | 138 / 138 | 268 / 268 | 138 / 137 |
| 1000000 | 2000000 | sorted | 9.3 / 8.8 | 50 / 49 | 9.3 / 8.8 | 49.8 / 49.1 | 10.6 / 9.8 |
| 1000000 | 2000000 | mostly sorted | 68.3 / 67.4 | 73.5 / 72.4 | 67.7 / 66.9 | 73.1 / 72.5 | 23 / 22.1 |
| 1000000 | 4000000 | random | 318 / 317 | 534 / 535 | 318 / 318 | 535 / 535 | 316 / 315 |
| 1000000 | 4000000 | sorted | 12.5 / 11.4 | 82.9 / 80.6 | 12.3 / 11.6 | 81.6 / 80.8 | 15.4 / 13.7 |
| 1000000 | 4000000 | mostly sorted | 164 / 162 | 131 / 129 | 163 / 161 | 130 / 129 | 41.6 / 39.6 |
| 1000000 | 16000000 | random | 1340 / 1340 | 2200 / 2200 | 1330 / 1340 | 2200 / 2190 | 1340 / 1340 |
| 1000000 | 16000000 | mostly sorted | 736 / 735 | 471 / 467 | 731 / 729 | 476 / 473 | 179 / 172 |
| 64 | 1 | random | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 64 | 1 | sorted | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 64 | 4 | random | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 64 | 4 | sorted | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 1024 | 16 | random | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.001 / 0.001 |
| 1024 | 16 | sorted | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 1024 | 256 | random | 0.003 / 0.003 | 0.005 / 0.004 | 0.003 / 0.003 | 0.004 / 0.003 | 0.004 / 0.003 |
| 1024 | 256 | sorted | 0.001 / 0.001 | 0.004 / 0.003 | 0.001 / 0.001 | 0.003 / 0.003 | 0.001 / 0.001 |
| 2048 | 32 | random | 0.001 / 0.001 | 0 / 0 | 0.001 / 0.001 | 0 / 0 | 0.001 / 0.001 |
| 2048 | 32 | sorted | 0.001 / 0.001 | 0 / 0 | 0.001 / 0.001 | 0 / 0 | 0.001 / 0.001 |
| 2048 | 128 | random | 0.002 / 0.002 | 0.002 / 0.002 | 0.002 / 0.002 | 0.002 / 0.001 | 0.002 / 0.002 |
| 2048 | 128 | sorted | 0.001 / 0.001 | 0.002 / 0.001 | 0.001 / 0.001 | 0.002 / 0.001 | 0.001 / 0.001 |
| 2048 | 512 | random | 0.019 / 0.018 | 0.023 / 0.023 | 0.019 / 0.017 | 0.024 / 0.023 | 0.019 / 0.017 |
| 2048 | 512 | sorted | 0.002 / 0.002 | 0.018 / 0.017 | 0.002 / 0.002 | 0.017 / 0.017 | 0.002 / 0.002 |
| 2048 | 2048 | random | 0.087 / 0.085 | 0.097 / 0.097 | 0.087 / 0.085 | 0.098 / 0.097 | 0.087 / 0.085 |
| 2048 | 2048 | sorted | 0.012 / 0.01 | 0.063 / 0.064 | 0.012 / 0.011 | 0.063 / 0.061 | 0.013 / 0.013 |
| 4096 | 64 | random | 0.002 / 0.002 | 0.001 / 0.001 | 0.002 / 0.002 | 0.001 / 0.001 | 0.003 / 0.003 |
| 4096 | 64 | sorted | 0.002 / 0.001 | 0.001 / 0.001 | 0.001 / 0.002 | 0.001 / 0.001 | 0.002 / 0.002 |
| 4096 | 1024 | random | 0.044 / 0.042 | 0.053 / 0.053 | 0.044 / 0.042 | 0.053 / 0.053 | 0.044 / 0.043 |
| 4096 | 1024 | sorted | 0.008 / 0.008 | 0.045 / 0.045 | 0.008 / 0.007 | 0.045 / 0.045 | 0.007 / 0.007 |

The benchmark also measures index sort, radix8/16, galloping and larger
build/rank workloads. Fused merge loses
to forced merge on sorted m=q=65536 (0.468/0.438 versus 0.428/0.406 ms),
while mostly sorted m=1e6, q=2e6 favors fused merge (23/22.1 versus
67.7/66.9 ms). The methods have no uniform winner.

The 2048 random compression row and several bulk-rank rows vary between
runs (up to 1.39x and 2.21x respectively). The sorted m=q=65536 comparison
is stable at this threshold, but small bulk-rank differences on the varying
rows do not establish a stable ranking.

## Pitfalls and notes

The `rank`/`rank_upper` distinction matters at query boundaries: for a half-open
range `[l, r)` use `[rank(l), rank(r))`; for an inclusive `[l, r]` use
`[rank(l), rank_upper(r))`.

`ranks()` detects sortedness during its merge walk for numerous forward queries.
At the first descent it keeps the prefix ranks and sorts copies of only the suffix,
restoring the suffix ranks to input order. Non-trivially-copyable and non-copyable
query types search the suffix; single-pass inputs always search per element. If you need the
rank of every input value as well as the universe, use `compress`.

## How it works

The universe is one `std::vector<T>`: the projected values are pushed, sorted with
`std::ranges::sort`, and `unique` plus `erase` drops the equivalents (neither of two
elements is before the other under `Compare`). `rank` and `rank_upper` are the lower and
upper bound in it, so they are meaningful for values outside the universe.

`ranks()` answers a batch either by one binary search per query or, for forward
ranges with `q >= ceil(size()/64)`, by a pointer moving forward through the
universe while comparing adjacent queries. No separate sortedness pass runs.
Sized ranges supply their count directly; non-sized forward ranges take one
counting pass before dispatch. At the first descent, ranks already written stay
intact. The remaining suffix of trivially copyable, copyable query values
constructible from their references sorts (query, position) pairs and merges from
the start of the universe, scattering after the prefix. Other query types search
the suffix. Single-pass inputs always search per element.

`compress` flips the sign bit of signed integral keys to obtain unsigned order,
then, from 2^11 values (smaller sizes keep a comparison sort because a
pass costs 2^11 counters whatever n is), stably sorts keyed positions with 11-bit LSD radix passes, skipping constant
digits. Other types and comparators keep the comparison sort; the walk over the sorted run starts
a new universe slot whenever the value strictly follows the previous one and writes that
slot's index to the pair's input position, so no search is made per value. The universe
is then adopted through the `sorted_unique` constructor.

## See also

- [Hierarchical bitset](hierarchical-bitset.md)
- [Types and utilities](types-and-utilities.md)
