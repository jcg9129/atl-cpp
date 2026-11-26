# `atl::binary_indexed_tree`

A Binary Indexed Tree (Fenwick tree) over a commutative monoid, in **one or more dimensions**. It
represents a `Dim`-dimensional box of `T`, combining a value into a cell in
O(log^Dim n) and querying the aggregate over a prefix box `[0, i0) x [0, i1) x …` in
O(log^Dim n). The operation and identity are supplied at construction, so the same
class serves sums, products, min/max, xor, and any other commutative monoid — not just sum. All
indices are 0-based and values live in a single flat row-major vector.

Only **prefix** queries are provided. A box (range) query needs `op` to be invertible
(an abelian group) and is left to the caller via inclusion–exclusion.

```cpp
#include <atl/ds/binary_indexed_tree.hpp>
```

\include fenwick-tree/basic.cpp

## API

### Template parameters

| parameter | meaning |
|---|---|
| `T` | element type |
| `Dim` | number of dimensions (default 1) |
| `Op` | binary operation combining two elements, modelling `binary_operation<T>` (default `std::plus<>`) |

> **Note.** `(T, op, identity)` must form a commutative monoid: `op` associative and
> commutative, with `identity` as its neutral element. Each cell aggregates a range that
> extends past the updated index, and prefix folds combine cells out of order, so a
> non-commutative `op` (string concatenation, matrix product) gives wrong results. This is a
> precondition on the caller; it is not enforced.

### Constructors

| function | description | complexity |
|---|---|---|
| `binary_indexed_tree(std::array<std::size_t, Dim> sizes, T identity = T{}, Op op = Op{})` | empty box of the given extents, filled with `identity` | O(total) |
| `binary_indexed_tree(std::size_t n, T identity = T{}, Op op = Op{})` | 1-D convenience overload (`Dim == 1` only) | O(n) |
| `binary_indexed_tree(Range&& values, T identity = T{}, Op op = Op{})` | build from a nested range whose nesting depth equals `Dim` (e.g. `vector<vector<T>>` for `Dim == 2`); extents read from the range, which must be rectangular | O(total) |
| `binary_indexed_tree(std::array<std::size_t, Dim> sizes, Range&& values, T identity = T{}, Op op = Op{})` | build from values already laid out flat and row-major (`size == product of sizes`) | O(total) |

`total` is the number of cells (the product of the extents). The range
constructors move elements out of an owning rvalue range, along owning nested
range chains, or out of prvalue elements yielded by any range; a borrowed range or a view
(`std::span`, `views::transform`, …) is copied from and its source left intact.

### Member functions

| function | description | complexity |
|---|---|---|
| `update(T const& value, Indices... indices)` → `void` | combine `value` into the cell at `(indices…)` (`sizeof...(Indices) == Dim`) | O(log^Dim n) |
| `prefix_fold(Indices... indices)` → `T` | aggregate over the prefix box `[0, i0) x [0, i1) x …`; returns `identity` when the box is empty (`sizeof...(Indices) == Dim`) | O(log^Dim n) |
| `max_right(Pred pred)` → `std::size_t` | `Dim == 1`: the largest `r` in `[0, size()]` with `pred(prefix_fold(r))` true — k-th / lower bound on prefix folds as one top-down descent (precond: `pred` monotone, `pred(identity)`) | O(log n) |
| `size()` → `std::size_t` | `Dim == 1`: the number of cells | O(1) |
| `sizes()` → `std::array<std::size_t, Dim> const&` | `Dim > 1`: the extent of each axis | O(1) |

Under `-DATL_ASSERT`, `update` asserts each `indices[d] < sizes[d]` and `prefix_fold`
asserts each `indices[d] <= sizes[d]`.

```cpp
// k-th one (0-indexed) in a 0/1 multiset over [0, n): O(log n), not O(log² n)
std::size_t kth = bit.max_right([k](int ones) { return ones <= k; });
```

### Factory helpers

Deduce `T` and `Dim` without spelling the class template arguments:

| function | description |
|---|---|
| `make_binary_indexed_tree<Op = std::plus<>>(T identity, Sizes... sizes)` → `binary_indexed_tree<T, sizeof...(Sizes), Op>` | empty box; `Dim` deduced from the number of `sizes`, `T` from `identity` |
| `make_binary_indexed_tree_from<Op = std::plus<>>(Range&& values)` → tree | build from a nested range; both `T` and `Dim` deduced from its nesting depth. Sum trees only (the identity is `T{}`) |
| `make_binary_indexed_tree_from<Op = std::plus<>>(Range&& values, Sizes... sizes)` → tree | build from a flat row-major range; `Dim` is the number of `sizes` |
| `make_binary_indexed_tree_from<Op>(Identity identity, Range&& values, Sizes... sizes)` → tree | the same two builds for any other operation: the identity comes first (`make_binary_indexed_tree_from<std::multiplies<>>(1LL, values)`) |

Class-template-argument deduction guides are also provided, so
`atl::binary_indexed_tree(std::vector<std::vector<int>>{…})` deduces `<int, 2>`.

## Complexity

With fixed Dim, build and storage are O(total); counting dimensions, build
is O(Dim * total). An update or prefix query takes the product of the
logarithms of the extents. max_right takes O(log n).

## Preconditions

Asserted: nonnegative extents and indices with exact size_t conversions;
all strides and their products fit size_t; flat input size matches the box;
nested ranges are rectangular; indices are in bounds; pred(identity) is true.
Documented only: the commutative monoid laws, validity and representability
of every operation, and monotonicity of a max_right predicate. The default
T{} must be the identity of the selected operation; products need identity 1.

## Measured performance

`benchmarks/ds/binary_indexed_tree.cpp` races two linear build passes: by
division and modulo computes each axis index from its flat offset, while by
odometer carries axis counters. The library constructor uses the latter.
The passes include copying their input; the constructor builds the same box.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | flags | by division and modulo | by odometer | library |
|---|---|---:|---:|---:|
| 1-D, n=1e7 | -O2 | 75.9 | 55.6 | 54 |
| 2-D, 3000 x 3000 | -O2 | 118 | 57.8 | 56.8 |
| 3-D, 200 cubed | -O2 | 140 | 57.7 | 55.9 |
| 1-D, n=1e7 | -O2 -march=native | 75.3 | 55.3 | 53.7 |
| 2-D, 3000 x 3000 | -O2 -march=native | 118 | 57.9 | 57.2 |
| 3-D, 200 cubed | -O2 -march=native | 138 | 56.8 | 56.1 |

## Pitfalls and notes

Updates combine into a cell rather than assigning it. Prefix folds are
half-open; deriving arbitrary box folds requires inverses. Source views
are copied, including a view passed as an rvalue; yielded prvalues may move.

## How it works

The flat row-major layout records one stride per axis. For each dimension,
a linear build sends each partial aggregate to its Fenwick parent. The
odometer loops enumerate blocks, axis positions and inner offsets without
dividing each flat position. Updates visit the Cartesian product of Fenwick
ancestors; prefixes visit disjoint Fenwick blocks on each axis. Commutativity
permits their traversal order. max_right tests powers of two from largest
to smallest while carrying the accepted prefix aggregate.

## See also

- [`rmq`](rmq.md) — O(n) build, O(1) static range-minimum queries
