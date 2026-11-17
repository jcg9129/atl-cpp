# `atl::direct_rmq`



Constant-time range-minimum queries over a **static** array. After O(n) construction it
answers, in O(1), the index (or value) of the smallest element in a half-open range,
where "smallest" is defined by `Compare` (default `std::less<>`). The build is a true
O(n) — a per-block bitmask scheme plus a sparse table over block minima — with no per-
query logarithm.

It holds a **non-owning** `std::span` of the array, so the array must outlive the
`direct_rmq` and must not be modified after construction. The constructor takes any
**borrowed** contiguous sized range whose `range_value_t` is `T`: an lvalue container, or
a `std::span` / `std::string_view`, even as a temporary. A temporary owning range (which
would leave a dangling view) or a mismatched element type is rejected by the constraint.

The queries are named like [`sparse_table`](sparse-table.md)'s: `position(l, r)` is the
index of the `Compare`-smallest element and `fold(l, r)` the element itself, so for
range-minimum the two structures are interchangeable.

```cpp
#include <atl/ds/direct_rmq.hpp>
```

\include rmq/basic.cpp

## API

### Template parameters

| parameter | meaning |
|---|---|
| `T` | element type |
| `Compare` | strict-weak-ordering comparator, as `std::sort` expects (default `std::less<>`) |

> **Note.** Supports up to 2³² − 1 elements; block-minima positions are stored as
> `atl::uint32` to halve the sparse table's footprint. A precondition on the caller,
> checked under `-DATL_ASSERT`.

### Member functions

| function | description | complexity |
|---|---|---|
| `direct_rmq()` | an empty structure (constexpr); assign a built one before querying | O(1) |
| `explicit direct_rmq(Range&& a, Compare comp = {})` | build over a **borrowed** contiguous range `a` of `T` (which must outlive the object) | O(n) |
| `position(std::size_t l, std::size_t r)` → `std::size_t` | index of the `Compare`-smallest element in the half-open range `[l, r)`; the leftmost among equals | O(1) |
| `fold(std::size_t l, std::size_t r)` → `T const&` | the `Compare`-smallest element in `[l, r)` (the maximum under `std::greater`) | O(1) |
| `size()`, `empty()` | number of elements viewed | O(1) |

Ranges are half-open. Under `-DATL_ASSERT`, both queries assert `l < r` and
`r <= size()` (so `[l, r)` is non-empty and in bounds). A deduction guide lets
`atl::direct_rmq(a)` deduce `T` and `Compare` from the range.

## Complexity

Writing m = ceil(n/64), construction takes O(n + m log m) operations and
O(n + m log m) words; each query is O(1). Since n is capped at 2^32-1,
the block table has at most 27 rows, giving bounded linear build and storage.
No query allocates.

## Preconditions

Asserted: n <= 2^32-1 before directory allocation, and l < r <= size().
Documented only: the underlying array outlives every query and remains
unchanged, and Compare is a strict weak order on its elements. The comparator
must be const-invocable; std::reference_wrapper can adapt a mutable callable.

## Measured performance

The min race in `benchmarks/ds/sparse_table.cpp` compares block-mask RMQ
with overlapping sparse windows, disjoint windows and mutable segment folds.
See [the complete tables](sparse-table.md) for all contenders and shapes.

Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-15;
minimum of three runs, milliseconds, -O2 / -O2 -march=native
(also -g -DNDEBUG -std=c++23).

| direct_rmq workload | time |
|---|---:|
| n=1e6 build | 11.3 / 10.9 |
| n=1e6, 1e7 uniform queries | 373 / 311 |
| n=1e6, 1e7 queries len<=64 | 194 / 169 |
| n=1e6, 1e7 queries len<=1000 | 318 / 275 |
| n=1e7, 1e7 uniform queries | 1196 / 1065 |

## Pitfalls and notes

A temporary owning range is rejected; temporary spans are safe only while
their underlying arrays remain alive. Equal minima return the leftmost
position. fold returns a reference into that same array. Use std::greater<>
for maxima. Rebuild after changing any element.

## How it works

A 64-bit monotone-candidate mask is maintained at each array position.
Shifting advances candidate distances; strictly worse candidates are
removed, while equal candidates remain to preserve leftmost ties. A query
within a block masks off candidates preceding l and selects the highest
remaining bit. Across blocks, partial boundary blocks are combined with
two overlapping entries of a sparse table over full-block minima. Their
positions are uint32, while masks retain all 64 bits even on 32-bit targets.

## See also

- [`binary_indexed_tree`](fenwick-tree.md) — Fenwick tree for prefix aggregates with updates
- [`sparse_table`](sparse-table.md) — O(1) folds of any idempotent operation (and `disjoint_sparse_table` for any associative one), at O(n log n) build time and memory
