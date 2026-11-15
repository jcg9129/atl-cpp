# `atl::counting_sort`

```cpp
#include <atl/utils/counting_sort.hpp>
```

A **stable counting sort**: it orders a range in place by an integer key
`proj(element)` in **O(n + span)**, where `span = max key − min key`. When the
keys pack into a small band (say ≤ 10⁶) this beats a comparison sort outright,
and it is *stable*, unlike `std::sort`. Measured at n = 10⁶ on x86-64 (Xeon 8280,
plain -O2, benchmark source `benchmarks/utils/counting_sort.cpp`, 2026-10-15; each
row includes copying the input, 0.7 ms for ints and 2.9 ms for records):

| shape | `counting_sort` | `std::ranges::sort` | `std::ranges::stable_sort` |
|---|---|---|---|
| ints, key span 10³ | **7.3 ms** | 43.7 ms | 60.8 ms |
| ints, key span 10⁶ | **26.0 ms** | 71.0 ms | 84.3 ms |
| 16-byte records by an int field, span 10³ | **21.6 ms** | 55.5 ms | 78.8 ms |
| 16-byte records by an int field, span 10⁶ | **43.1 ms** | 83.6 ms | 104 ms |

The `-march=native` record agrees within a few percent.

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
scatter lost: its swaps land at random positions, and the gather is 1.5× to 5×
faster (int span 10³: 7.3 vs 37.1 ms; records span 10⁶: 43.1 vs 66.2 ms, same
record). Not `constexpr`, because of the thread-local scratch.

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
