# `atl::counting_sort`

```cpp
#include <atl/utils/counting_sort.hpp>
```

A **stable counting sort**: it orders a range in place by an integer key
`proj(element)` in **O(n + span)**, where `span = max key − min key`. When the
keys pack into a small band (say ≤ 10⁶) this beats a comparison sort outright,
and it is *stable*, unlike `std::sort`. Measured speedups are under "Measured performance" below.

This complete example sorts signed keys and preserves the order of equal keys:

\include counting-sort/stable-records.cpp

Expected stdout:

\verbinclude counting-sort/stable-records.out

A free function, used like `std::ranges::sort` — nothing to construct:

```cpp
atl::counting_sort(points, &point::y);   // stable, in place, ascending by y
atl::counting_sort(depths);              // range of ints, identity projection
```

| function | description | complexity |
|---|---|---|
| `counting_sort(r, proj = {})` | stable in-place ascending sort — same result as `std::ranges::stable_sort(r, {}, proj)` but O(n + span) | O(n + span) |
| `counting_sort_clear()` | release the internal scratch on this thread | O(1) |

`r` is a sized random-access range; `proj` is a `std::ranges`-style projection
(default `std::identity`) whose result is integral — signed keys are handled by
offsetting by the minimum. The `counting_sortable` concept enforces this.

**Argsort / parallel arrays.** There is no separate index-returning overload —
sort an index range by an indexing projection, which gives the same stable
permutation as an owned `vector` you can keep and apply anywhere:

```cpp
#include <numeric>
std::vector<uint32> idx(n);
std::iota(idx.begin(), idx.end(), 0u); // Xcode 16.4 lacks ranges::iota.
atl::counting_sort(idx, [&](uint32 i){ return key[i]; });   // idx = stable order
// reorder any parallel array: for (uint32 i : idx) out.push_back(payload[i]);
```

**Scratch.** The `uint32` count / permutation buffers live in a thread-local
kept between calls, so a loop of sorts reallocates them only when one must grow;
`counting_sort_clear()` frees them if you want the memory back. The elements
themselves are gathered into one element-typed buffer per call (moved, not
copied) and moved back. Realising the permutation in place by cycle-walking its
scatter (`counting_sort_by_cycle_walk` in the benchmark) lost: its swaps land at
random positions, and the gather is 1.5× to 5.2× faster (see Measured
performance). Not `constexpr`, because of the thread-local scratch.

**Preconditions.** Asserted: the key span is `< 2³¹` (it sizes the bucket
array, so a key range that is not actually small is a bug, not a silent huge
allocation; reach for a comparison sort there instead) and the range holds fewer
than 2³² elements (indices are `uint32`). Documented only: `proj` must be
equality-preserving, as it is called three times per element (min/max scan,
count, scatter); and the sort is not reentrant on one thread, so neither `proj`
nor an element's move may itself call `counting_sort`.

**The bucket primitive underneath.** `counting_sort` is built on
`atl::detail::bucket_by_key(bucket_count, key, offsets, order)`, the lower layer
for callers that already know their bucket count (vertex ids, depths): keys
`key(i)` in `[0, bucket_count)` go in, and caller-owned spans come out holding
the CSR row starts (`offsets`, `bucket_count + 1` entries) and the stable gather
order (`order`, bucket `b` at `order[offsets[b] .. offsets[b + 1])`). It needs no
min/max scan, allocates nothing, and is `constexpr`. The hand-rolled
count/prefix-sum/scatter builds in the graph headers are candidates to move onto
it, each only once its own benchmark shows no regression.

Uses: a radix-sort pass, bucketing tree vertices by depth, sweep-line events by
coordinate, any stable "sort by a small integer key" where the log factor and
`std::stable_sort`'s allocation matter. Coordinate compression can first map
sparse keys into a dense `[0, m)` band.

## Measured performance

`benchmarks/utils/counting_sort.cpp` races `counting_sort` (the library) against
`counting_sort_by_cycle_walk`, `std::ranges::sort` and `std::ranges::stable_sort`.
The two counting sorts compute the same stable permutation and differ in how
they apply it: the library gathers the elements into a buffer and moves them
back; "by cycle walk" inverts the permutation to a scatter and walks its cycles
with swaps, which land at random positions. Each row sorts a fresh copy of 10⁶
elements, either `int32` keys or 16-byte records sorted by an `int32` field,
with key span 10³ or 10⁶, so it measures throughput. "Copy only" is the copy of
the input without sorting, the floor shared by every row.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | flags | copy only | library | by cycle walk | `std::ranges::sort` | `std::ranges::stable_sort` |
|---|---|---:|---:|---:|---:|---:|
| ints, span 10³ | `-O2` | 0.083 | **2.9** | 22 | 41 | 54.6 |
| ints, span 10³ | `-O2 -march=native` | 0.089 | **2.6** | 19.9 | 34.6 | 44.5 |
| ints, span 10⁶ | `-O2` | 0.087 | **4.5** | 23.5 | 69.4 | 81 |
| ints, span 10⁶ | `-O2 -march=native` | 0.088 | **4.5** | 21.6 | 57.1 | 64.5 |
| records, span 10³ | `-O2` | 0.442 | **4.5** | 23.3 | 45.6 | 61.6 |
| records, span 10³ | `-O2 -march=native` | 0.528 | **8.1** | 24.5 | 39.3 | 57.6 |
| records, span 10⁶ | `-O2` | 0.447 | **7.9** | 26.3 | 74.1 | 90.2 |
| records, span 10⁶ | `-O2 -march=native` | 0.522 | **9.3** | 25.9 | 62.3 | 82 |

The library is the fastest sorting contender on every shape in both modes.
Cycle walk takes 2.78 to 7.65 times as long, `std::ranges::sort` 4.85 to
15.42 times and `std::ranges::stable_sort` 7.11 to 18.83 times. Native flags
change some costs substantially: library records at span 10³ rise from
4.5 to 8.1 ms, while `std::ranges::sort` on those records falls from
45.6 to 39.3 ms.

The native int span-10³ library row varies between runs (1.39x spread);
its minimum alone does not establish a stable speedup.
