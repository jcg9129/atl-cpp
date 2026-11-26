# `atl::sparse_table` / `atl::disjoint_sparse_table`

Constant-time range folds over static data, using O(n log n) preprocessing
and storage. Choose sparse_table for an idempotent operation and
disjoint_sparse_table for any associative operation, including ordered
composition. Use a mutable tree when values change.

```cpp
#include <atl/ds/sparse_table.hpp>
```

\include sparse-table/basic.cpp

## API

Both templates have parameters `std::copyable T, binary_operation<T> Op`.
The operation must be const-invocable on two const T references with a
result convertible to T. The concepts check signatures, not algebraic laws.
For either `Table = sparse_table<T, Op>` or `disjoint_sparse_table<T, Op>`:

| signature | meaning |
|---|---|
| `constexpr Table()` | empty placeholder; requires default-constructible Op |
| `constexpr explicit Table(R &&values, Op op = {})` | sized range, with forwarded elements convertible to T; owns the copied or moved values |
| `constexpr T fold(std::size_t l, std::size_t r) const` | fold [l,r), returning T |
| `constexpr std::size_t size() const` | number of input elements |
| `constexpr bool empty() const` | whether size is zero |

Deduction from `(range, op)` uses range_value_t as T. Row zero moves from
owning rvalue ranges and prvalues and copies from lvalues and views,
including rvalue views referring to caller storage.

## Complexity

Both build in O(n log n) time and space and fold in O(1), counting
constant-time operation and value copies. Accessors take O(1).

## Preconditions

Asserted under ATL_ASSERT: `l < r <= size()` and the padded table's number
of elements fits `vector<T>::max_size()` before multiplication.
Documented only: every operation and conversion is valid and representable;
sparse_table's operation is associative, commutative and idempotent;
disjoint_sparse_table's operation is associative. Neither needs an identity.

## Measured performance

`benchmarks/ds/sparse_table.cpp` races overlapping windows (sparse_table),
disjoint suffix/prefix windows (disjoint_sparse_table), block masks
(direct_rmq) and logarithmic mutable folds (segment_tree). Construction is
timed separately from independent range-query throughput; min and gcd,
uniform and bounded-length ranges exercise different costs.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| workload | sparse_table min | disjoint_sparse_table min | direct_rmq | segment_tree min |
|---|---:|---:|---:|---:|
| n=1e6 build | 52.8 / 51.2 | 55 / 51.7 | 8.4 / 7.7 | 0.652 / 0.514 |
| n=1e6, 1e7 uniform queries | 83 / 69.1 | 83.1 / 63.6 | 146 / 143 | 1030 / 1070 |
| n=1e6, 1e7 queries len<=64 | 71.1 / 59.5 | 87.7 / 70.2 | 85 / 81.2 | — |
| n=1e6, 1e7 queries len<=1000 | 75.8 / 65.1 | 88.9 / 72.5 | 135 / 137 | — |
| n=1e7, 1e7 uniform queries | 168 / 146 | 155 / 140 | 701 / 687 | 1510 / 1450 |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| gcd workload, n=1e6 | disjoint_sparse_table | segment_tree |
|---|---:|---:|
| build | 309 / 310 | 21.5 / 21.7 |
| 1e7 uniform queries | 358 / 106 | 1600 / 1610 |

## Pitfalls and notes

Queries are nonempty: there is no identity for empty folds. Sum is invalid
for an overlapping sparse table because overlap counts elements twice.
The disjoint table preserves order, so affine or matrix composition works.
Choose direct_rmq for positions and lower build/storage costs, and a
segment tree for updates. There is no size cap independent of allocation
limits; padded allocation size is checked before arithmetic.

## How it works

Both use a single flat level-major vector, each row padded to n. The
sparse table combines adjacent half-windows to build doubling windows;
a query combines two windows covering its endpoints, possibly overlapping.
The disjoint table builds suffixes to, and prefixes from, every block
midpoint. The highest differing bit of l and r-1 identifies the unique
level where they straddle a midpoint; two entries combine each element
exactly once and in order. Row zero serves single-element queries.

## See also

- [Range minimum queries](rmq.md)
- [Segment tree](segment-tree.md)
