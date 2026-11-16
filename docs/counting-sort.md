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

Xeon Platinum 8280, GCC 15.2, RelWithDebInfo without assertions, 2026-10-15.
Minimum of 3 runs, milliseconds per sort including the copy:

| shape | flags | copy only | library | by cycle walk | `std::ranges::sort` | `std::ranges::stable_sort` |
|---|---|---:|---:|---:|---:|---:|
| ints, span 10³ | `-O2` | 0.722 | **7.3** | 37.1 | 43.7 | 60.8 |
| ints, span 10³ | `-O2 -march=native` | 0.719 | **7.1** | 36.6 | 42.2 | 61.1 |
| ints, span 10⁶ | `-O2` | 0.723 | **26.0** | 55.8 | 71.0 | 84.3 |
| ints, span 10⁶ | `-O2 -march=native` | 0.719 | **26.3** | 55.6 | 70.3 | 84.4 |
| records, span 10³ | `-O2` | 2.9 | **21.6** | 45.2 | 55.5 | 78.8 |
| records, span 10³ | `-O2 -march=native` | 2.9 | **22.7** | 45.8 | 49.4 | 74.4 |
| records, span 10⁶ | `-O2` | 2.9 | **43.1** | 66.2 | 83.6 | 104 |
| records, span 10⁶ | `-O2 -march=native` | 2.9 | **43.2** | 67.2 | 76.3 | 96.3 |

The library is fastest on every shape with both flag sets. By cycle walk is 1.5
to 5.2 times slower (ints, span 10³: 5.1 times; records, span 10⁶: 1.5 times),
`std::ranges::sort` 1.8 to 6.0 times and `std::ranges::stable_sort` 2.2 to 8.6
times. Native instructions change none of this by more than a few percent,
except that `std::ranges::sort` on records is faster with them (49.4 against
55.5 ms at span 10³).
