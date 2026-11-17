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

Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-15.
Minimum of three runs, milliseconds, plain -O2 / -O2 -march=native;
both also use -g -DNDEBUG -std=c++23.

| workload | sparse_table min | disjoint_sparse_table min | direct_rmq | segment_tree min |
|---|---:|---:|---:|---:|
| n=1e6 build | 58.7 / 57.7 | 64.7 / 60.2 | 11.3 / 10.9 | 2.0 / 1.9 |
| n=1e6, 1e7 uniform queries | 173 / 149 | 156 / 146 | 373 / 311 | 1200 / 1168 |
| n=1e6, 1e7 queries len<=64 | 142 / 129 | 143 / 143 | 194 / 169 | — |
| n=1e6, 1e7 queries len<=1000 | 152 / 123 | 164 / 147 | 318 / 275 | — |
| n=1e7, 1e7 uniform queries | 377 / 357 | 358 / 354 | 1196 / 1065 | 2034 / 1643 |

| gcd workload, n=1e6 | disjoint_sparse_table | segment_tree |
|---|---:|---:|
| build | 467 / 433 | 29.8 / 30.8 |
| 1e7 uniform queries | 515 / 173 | 2062 / 2152 |

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
