# `atl::persistent_segment_tree`

A fixed-shape persistent segment tree for point updates, ordered folds and cross-version range copies. Prefix count versions also support range order statistics by lockstep descent.

```cpp
#include <atl/ds/persistent_segment_tree.hpp>
```

\include persistent-segment-tree/basic.cpp

## API

`template<std::copyable T, binary_operation<T> Op> class persistent_segment_tree`.
Aliases: value_type=T, size_type=std::size_t. Op must be associative with the supplied
identity. The default placeholder uses T{} and requires default-initializable
T and Op; this value must be a valid identity before folding it.

| signature | meaning / complexity |
|---|---|
| `persistent_segment_tree()` | empty placeholder; O(1) |
| `persistent_segment_tree(std::size_t n, T identity, Op op = Op{})` | n identity elements; O(n) |
| `persistent_segment_tree(R&& values, std::type_identity_t<T> identity, Op op = Op{})` | sized range; O(n), owning rvalues/prvalues may move, views copy |
| `set(std::size_t i, T x)` | new version; O(log n) time/new nodes |
| `update(std::size_t i, Update&& f)` | new version, f on T const& convertible to T; O(log n) |
| `copy_range(persistent_segment_tree const& source, std::size_t l, std::size_t r)` | same-pool source inside [l,r); O(log n) |
| `T at(std::size_t i) const`, `T fold(std::size_t l, std::size_t r) const` | allocation-free O(log n), empty fold returns identity |
| `T fold() const` | whole fold; O(1) |
| `static std::size_t descend(persistent_segment_tree const& a, persistent_segment_tree const& b, Choose&& choose)` | choose on corresponding left folds, result convertible to bool; O(log n) |
| `std::size_t size() const noexcept`, `bool empty() const noexcept` | O(1) |
| `std::size_t arena_size() const noexcept`, `void reserve(std::size_t ops)` | node footprint, capacity hint in mutations |

Versions use shared pool ownership and are runtime-only (`std::shared_ptr`).
Copying a version costs O(1). Version-returning members have `const &` and `&&`
overloads: the latter consumes the handle; the moved-from object may only be
destroyed or assigned to. Allocation failure propagates an exception; mutations
are not transactional if allocation, element operations or callables throw.

## Complexity

Construction makes 2n-1 tree nodes plus a sentinel. Each point mutation copies
one root-to-leaf path. Range copy rebuilds only boundary paths and shares fully
covered subtrees. Storage is O(n + q log n) after q mutations; reserve can
relocate the entire shared arena and is a capacity hint.

## Preconditions

Asserted under `ATL_ASSERT`: i < size, l <= r <= size; both versions share a
pool for range copy/descent; nonempty descent; n <= 2^31; available uint32 ids;
reserve products fit std::size_t. Op's monoid laws and user arithmetic bounds are
documented only. Choose is invoked as an lvalue and may keep search state.
It must choose the side containing the desired result; descent itself enforces
no order-statistic meaning. CTAD rejects narrowing from the range value type.

## Measured performance

`benchmarks/ds/persistent_segment_tree.cpp` compares prefix construction and kth
queries with parallel raw arrays, then child-id reads with whole-node copying
for point sets. The latter changes aggregate copy traffic, retaining the same tree.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| workload | library / by child ids | by parallel arrays / by whole-node copy |
|---|---:|---:|
| 2e5 prefix builds + 2e5 kth queries | 287 / 399 | 294 / 417 |
| 2e5 sets, 2x2 matrix | 203 / 252 | 213 / 268 |
| 2e5 sets, 24-character string maximum | 515 / 651 | 551 / 674 |
| 2e5 sets, uint32 sum | 78.8 / 113 | 82.4 / 121 |

## Pitfalls and notes

Two independently constructed trees cannot interact even if their sizes match.
Use copies/updates from one origin. A fold-only immutable dataset may fit a
[wavelet matrix](wavelet-matrix.md) better for static order statistics.

## How it works

Every version shares exact midpoint intervals on [0,n), without padding. Nodes
store an aggregate and two child ids. Rebuild reads child ids before recursive
allocation and consumes a leaf's update result before appending, avoiding
copies of unused aggregates. A range copy adopts whole subtrees by id.
Descent walks two aligned versions together: prefix-count differences in the
left children give the number of slice elements on that side. Return values
are copied out so later shared-arena reallocations cannot dangle them.

## See also

- [Wavelet matrix](wavelet-matrix.md)
- [Persistent lazy segment tree](persistent-lazy-segment-tree.md)
- [Persistent treap sequence](persistent-treap-sequence.md)
