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
CPU: Intel Xeon Platinum 8168; GCC 15.2; RelWithDebInfo, asserts off; 2026-11-03.
Flags: `-O2 -g -DNDEBUG -std=c++23` and
`-O2 -march=native -g -DNDEBUG -std=c++23`. Cells are minimum milliseconds
over three runs of 25 samples, plain / native.

| n and input shape | library (recorded dispatch) | by keyed sort (radix11) | by keyed sort (comparison) | by pair sort |
|---|---|---|---|---|
| 1024 random | 0.027/0.026 | 0.019/0.019 | 0.027/0.026 | 0.026/0.027 |
| 1024, 1000 distinct | 0.025/0.025 | 0.020/0.019 | 0.025/0.025 | 0.025/0.026 |
| 2048 random | 0.087/0.087 | 0.038/0.036 | 0.086/0.085 | 0.089/0.091 |
| 2048, 1000 distinct | 0.089/0.091 | 0.039/0.039 | 0.090/0.089 | 0.093/0.094 |
| 3072 random | 0.149/0.154 | 0.057/0.055 | 0.149/0.148 | 0.155/0.159 |
| 3072, 1000 distinct | 0.148/0.154 | 0.056/0.055 | 0.148/0.150 | 0.154/0.157 |
| 4096 random | 0.075/0.073 | 0.075/0.073 | 0.207/0.210 | 0.218/0.226 |
| 4096, 1000 distinct | 0.076/0.074 | 0.074/0.073 | 0.206/0.208 | 0.218/0.223 |
| 16384 random | 0.302/0.300 | 0.309/0.296 | 0.985/1.0 | 1.0/1.1 |
| 16384, 1000 distinct | 0.291/0.291 | 0.294/0.287 | 0.828/0.827 | 0.890/0.899 |
| 32768 random | 0.867/0.866 | 0.876/0.858 | 2.1/2.1 | 2.2/2.3 |
| 32768, 1000 distinct | 0.606/0.603 | 0.613/0.598 | 1.6/1.7 | 1.8/1.8 |
| 65536 random | 2.0/1.9 | 2.0/1.9 | 4.4/4.4 | 4.7/4.8 |
| 65536, 1000 distinct | 1.3/1.3 | 1.3/1.3 | 3.4/3.4 | 3.7/3.8 |
| 131072 random | 4.0/3.9 | 4.0/3.9 | 9.5/9.6 | 10.3/10.4 |
| 131072, 1000 distinct | 2.7/2.7 | 2.7/2.7 | 6.8/6.8 | 7.4/7.5 |
| 1536 random, rotated | 0.073/0.073 | 0.030/0.029 | 0.073/0.073 | not measured yet |
| 1536, 1000 distinct, rotated | 0.077/0.076 | 0.034/0.033 | 0.075/0.076 | not measured yet |
| 8192 random, rotated | 0.156/0.151 | 0.159/0.153 | 0.463/0.469 | not measured yet |
| 8192, 1000 distinct, rotated | 0.152/0.148 | 0.154/0.151 | 0.428/0.431 | not measured yet |

The record combines plain and rotated crossover test cases without tagging
repeated row names; sizes 1536 and 8192 identify rotated-only rows. The library's
current threshold is 2048, while the recorded dispatch changes at 4096. The
recorded dispatch timings therefore do not measure the current dispatch at
2048 and 3072. Fixed sorts still identify the measured algorithms.

For bulk ranks, the record's library rows used a separate sortedness pre-scan;
the “fused sortedness merge” rows describe the method now used by the library.
The following column labels identify those methods. “By forced merge” means
merge for sorted queries and sorted merge otherwise. Small-universe rows and
additional counts use rotated inputs in the source. Rounded zero is a recorded
value, not a claim of zero cost. Some small rows had unstable run spread.

| m | q | shape | by sortedness prescan | by search | by forced merge | by rank loop | library method (fused merge) |
|---|---|---|---|---|---|---|---|
| 1024 | 1024 | random | 0.032/0.036 | 0.050/0.053 | 0.031/0.033 | 0.051/0.055 | 0.034/0.036 |
| 1024 | 1024 | sorted | 0.003/0.003 | 0.033/0.036 | 0.002/0.002 | 0.034/0.037 | 0.003/0.002 |
| 1024 | 64 | random | 0.001/0.001 | 0.001/0.001 | 0.001/0.001 | 0.001/0.001 | 0.001/0.001 |
| 1024 | 64 | sorted | 0.001/0.001 | 0.001/0.001 | 0.000/0.000 | 0.001/0.001 | 0.001/0.000 |
| 4096 | 4096 | random | 0.245/0.252 | 0.248/0.265 | 0.236/0.238 | 0.254/0.292 | 0.239/0.248 |
| 4096 | 4096 | sorted | 0.034/0.033 | 0.152/0.162 | 0.032/0.030 | 0.153/0.168 | 0.035/0.032 |
| 4096 | 256 | random | 0.008/0.006 | 0.011/0.012 | 0.007/0.006 | 0.011/0.009 | 0.007/0.006 |
| 4096 | 256 | sorted | 0.004/0.004 | 0.010/0.011 | 0.003/0.003 | 0.011/0.009 | 0.004/0.003 |
| 65536 | 65536 | random | 5.1/5.2 | 6.6/7.0 | 5.0/5.0 | 6.7/7.5 | 5.1/5.2 |
| 65536 | 65536 | sorted | 0.614/0.598 | 2.8/3.1 | 0.568/0.550 | 2.9/3.0 | 0.632/0.574 |
| 1000000 | 15625 | random | 1.6/1.7 | 2.6/2.8 | 1.6/1.6 | 2.7/3.0 | 1.8/1.9 |
| 1000000 | 3906 | random | 0.680/0.679 | 0.664/0.699 | 0.638/0.662 | 0.680/0.753 | 0.678/0.685 |
| 1000000 | 976 | random | 0.153/0.152 | 0.155/0.158 | 0.432/0.464 | 0.153/0.171 | 0.151/0.155 |
| 100000 | 200000 | random | 16.9/17.4 | 19.0/20.2 | 16.6/16.8 | 19.3/22.0 | 16.3/16.8 |
| 100000 | 200000 | sorted | 1.3/1.3 | 7.7/8.3 | 1.2/1.1 | 7.8/8.0 | 1.4/1.2 |
| 100000 | 400000 | random | 35.1/36.0 | 38.0/40.3 | 34.4/34.7 | 38.5/43.9 | 33.3/35.0 |
| 100000 | 1600000 | random | 148/150 | 152/161 | 146/148 | 154/176 | 147/149 |
| 100000 | 1600000 | sorted | 5.3/4.9 | 33.7/39.7 | 4.0/3.7 | 38.7/34.4 | 5.3/4.0 |
| 1000000 | 2000000 | random | 206/213 | 350/357 | 204/207 | 356/382 | 206/210 |
| 1000000 | 2000000 | sorted | 14.8/14.1 | 86.9/93.8 | 12.9/12.0 | 89.1/88.9 | 14.7/12.5 |
| 1000000 | 2000000 | mostly sorted | 112/113 | 115/122 | 106/107 | 117/121 | 32.2/31.2 |
| 1000000 | 4000000 | random | 474/486 | 687/714 | 468/476 | 701/769 | 467/481 |
| 1000000 | 4000000 | sorted | 20.9/20.0 | 142/157 | 17.4/16.5 | 150/144 | 20.9/17.6 |
| 1000000 | 4000000 | mostly sorted | 260/266 | 202/218 | 254/254 | 212/212 | 58.5/56.0 |
| 1000000 | 16000000 | random | 2113/2146 | 2903/3016 | 2098/2115 | 2956/3218 | 2097/2129 |
| 1000000 | 16000000 | mostly sorted | 1292/1305 | 687/762 | 1271/1272 | 746/722 | 254/249 |
| 64 | 1 | random | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| 64 | 1 | sorted | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| 64 | 4 | random | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| 64 | 4 | sorted | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| 1024 | 16 | random | 0.001/0.001 | 0.001/0.001 | 0.001/0.001 | 0.000/0.000 | 0.001/0.001 |
| 1024 | 16 | sorted | 0.001/0.001 | 0.001/0.001 | 0.001/0.001 | 0.000/0.000 | 0.001/0.001 |
| 1024 | 256 | random | 0.013/0.013 | 0.014/0.014 | 0.012/0.012 | 0.016/0.016 | 0.012/0.012 |
| 1024 | 256 | sorted | 0.003/0.003 | 0.012/0.012 | 0.003/0.003 | 0.012/0.012 | 0.003/0.003 |
| 2048 | 32 | random | 0.001/0.001 | 0.002/0.002 | 0.001/0.001 | 0.002/0.002 | 0.002/0.002 |
| 2048 | 32 | sorted | 0.001/0.001 | 0.002/0.002 | 0.001/0.001 | 0.002/0.002 | 0.001/0.001 |
| 2048 | 128 | random | 0.006/0.006 | 0.008/0.008 | 0.006/0.006 | 0.008/0.008 | 0.006/0.006 |
| 2048 | 128 | sorted | 0.002/0.002 | 0.007/0.007 | 0.002/0.002 | 0.007/0.007 | 0.002/0.002 |
| 2048 | 512 | random | 0.028/0.028 | 0.031/0.031 | 0.027/0.027 | 0.034/0.034 | 0.028/0.028 |
| 2048 | 512 | sorted | 0.007/0.007 | 0.025/0.025 | 0.006/0.006 | 0.027/0.027 | 0.006/0.006 |
| 2048 | 2048 | random | 0.122/0.122 | 0.123/0.123 | 0.115/0.115 | 0.136/0.136 | 0.117/0.117 |
| 2048 | 2048 | sorted | 0.019/0.019 | 0.080/0.080 | 0.017/0.017 | 0.085/0.085 | 0.018/0.018 |
| 4096 | 64 | random | 0.003/0.003 | 0.004/0.004 | 0.003/0.003 | 0.004/0.004 | 0.004/0.004 |
| 4096 | 64 | sorted | 0.002/0.002 | 0.004/0.004 | 0.002/0.002 | 0.004/0.004 | 0.002/0.002 |
| 4096 | 1024 | random | 0.061/0.061 | 0.067/0.067 | 0.058/0.058 | 0.074/0.074 | 0.060/0.060 |
| 4096 | 1024 | sorted | 0.014/0.014 | 0.050/0.050 | 0.013/0.013 | 0.051/0.051 | 0.014/0.014 |

Index sort, radix8/16, galloping, and the larger build/rank shapes without a
supplied record: not measured yet. No timings here were collected on this machine.

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
