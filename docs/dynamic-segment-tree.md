# `atl::dynamic_segment_tree`

A monoid tree for sparse point changes over `[0, n)`, useful when coordinates
arrive online. Untouched positions contain the supplied identity. If all
coordinates are available in advance, compression and a dense tree use less
memory and usually run faster.

```cpp
#include <atl/ds/dynamic_segment_tree.hpp>
```

\include dynamic-segment-tree/quick-start.cpp

Expected stdout:

\verbinclude dynamic-segment-tree/quick-start.out

## API and complexity

`dynamic_segment_tree<T, Op>` requires copyable `T` and
`atl::binary_operation<Op, T>`: the stored const operation accepts two const
elements and returns a result convertible to `T`. `value_type` is `T`;
`size_type` is `std::size_t`. Costs assume constant-time element operations.

| signature | behavior | cost |
|---|---|---|
| `constexpr dynamic_segment_tree()` | empty, value-initialized identity and operation; both must be default-initializable | O(1) |
| `constexpr dynamic_segment_tree(std::size_t n, T identity, Op op = Op{})` | domain `[0, n)`, only a null sentinel allocated | O(1) |
| `constexpr void set(std::size_t i, T x)` | assign one position | O(log n) time and new nodes, worst case |
| `template<update_function<T> Update> constexpr void update(std::size_t i, Update&& f)` | replace by `f(current)`; invoke exactly once, including at an untouched position | O(log n) time and new nodes, worst case |
| `constexpr T fold(std::size_t l, std::size_t r) const` | ordered fold; identity when empty | O(log n), no allocation |
| `constexpr T const& fold() const` | current whole-domain fold | O(1) |
| `constexpr T const& at(std::size_t i) const` | stored value, or identity if untouched | O(log n), no allocation |
| `template<std::predicate<T const&> Pred> constexpr std::size_t max_right(std::size_t l, Pred pred) const` | largest accepted right endpoint | O(log n) |
| `template<std::predicate<T const&> Pred> constexpr std::size_t min_left(std::size_t r, Pred pred) const` | smallest accepted left endpoint | O(log n) |
| `constexpr std::size_t size() const noexcept`, `constexpr bool empty() const noexcept` | domain size and emptiness | O(1) |
| `constexpr std::size_t node_count() const noexcept` | pool entries, including the null sentinel | O(1) |
| `constexpr void reserve(std::size_t ops)` | capacity hint: current entries + `3*ops + 64` | O(current entries) if reallocated |

The pool uses 32-bit child ids. Memory is O(q log n) in the worst case for
q distinct touched positions; scattered positions benefit from compressed
single-position subtrees. `reserve` is a heuristic, not an allocation bound.
Returned references may change or become invalid at the next mutation or tree
assignment. All operations are constexpr when their element and callable types
support constant evaluation.

## Preconditions

Asserted with `ATL_ASSERT`: `i < size()`, `l <= r <= size()`, valid search
endpoints and `pred(identity)`. New pool entries must fit the 32-bit id space; reserve arithmetic must fit `size_t`.

Documented only: `(T, op, identity)` is a monoid. Search predicates are monotone
in the searched direction. Untouched values must equal the monoid identity;
for non-identity implicit values use a dynamic lazy tree with a fill callable.
The domain fits `size_t`; 10^18-sized domains require a sufficiently wide
platform. Element arithmetic must stay within the operation's own bounds.
Allocation failure propagates the allocator exception; mutations offer no
transactional rollback if a callable or allocation throws.

## Measured performance

`benchmarks/ds/dynamic_segment_tree.cpp` measures a mixed affine-composition
stream. Materialized paths allocate a full descent per point, while the library
compresses a subtree containing one touched position. Offline compression uses
a dense tree; the treap map keeps only touched keys. The fold comparison visits
both children unconditionally.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2, RelWithDebInfo, assertions off,
2026-10-15. Milliseconds, plain `-O2` / `-O2 -march=native`; both also use
`-g -DNDEBUG -std=c++23`, minimum of three run means.

| workload | library | by materialized paths | by treap map | by offline compression |
|---|---:|---:|---:|---:|
| 500,000 mixed steps, domain 10^9 | 263 / 271 | 430 / 427 | 604 / 633 | 174 / 189 |

| 1,000,000 folds, domain 10^9 | library | by two-child fold |
|---|---:|---:|
| affine, lengths 1–64 | 404 / 445 | 508 / 539 |
| int64 sum, lengths 1–64 | 360 / 400 | 404 / 421 |
| affine, uniform ranges | 831 / 850 | 882 / 923 |

## Pitfalls and notes

Default construction supplies `T{}` as identity; it does not establish the
monoid laws for an arbitrary operation. Reads do not allocate, but const
callables with shared mutable state still need their own synchronization.

## How it works

Pool index zero represents every absent subtree and folds to identity. A
pending node stores one position and its value. When a different position
arrives, the pending element moves into a child and descent continues until
the positions diverge. Pulls restore ordered child folds. A query skips absent
and disjoint children, avoiding both identity copies and operation calls.
Each path has at most O(log n) levels; queries decompose two boundary paths,
so folds and searches retain that bound even in clustered data.

## See also

- [Segment tree](segment-tree.md)
- [Coordinate compression](coordinate-compression.md)
- [Dynamic lazy segment tree](dynamic-lazy-segment-tree.md)
