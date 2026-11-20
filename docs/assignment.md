# `atl::graph::assignment`

Dense assignment matches the smaller side of a rectangular cost matrix to
distinct vertices of the larger side, minimizing total cost. Use it for complete
dense weighted matching with an exact dual certificate. Sparse networks,
capacities and lower bounds belong to min-cost flow.

```cpp
#include <atl/graph/assignment.hpp>
```

\include assignment/quick.cpp

Expected stdout:
\include assignment/quick.out

## API

```cpp
template<typename Sum = void, cost_matrix Matrix>
    requires(assignment_sum<detail::sum_t<Sum, Matrix>>
             && detail::assignment_cost<
                 std::ranges::range_value_t<std::ranges::range_value_t<Matrix>>,
                 detail::sum_t<Sum, Matrix>>)
constexpr auto assignment(Matrix const &costs);
```

`cost_matrix<Matrix>` checks a const sized random-access range of sized
random-access rows with a numeric element type and the indexing operations
used by the solver. `assignment_sum<Sum>` requires signed integer arithmetic,
excluding character types and bool; portable `atl::int128` is supported.
The detail aliases select Sum, or the matrix element type when Sum is void,
and check numeric conversion from the matrix element type.

`assignment<int64>(int32_matrix)` keeps storage narrow while widening all
cost arithmetic. Unsigned matrix elements need an explicit signed Sum.
Floating elements can be converted to an explicitly selected integer Sum only
when that conversion is exact; floating Sum is rejected because plateau tests
rely on exact integer equality. Proxy cells, including packed bool rows with
an explicit integer Sum, are accepted.

```cpp
template<assignment_sum Sum>
struct assignment_result {
    Sum cost{};
    std::vector<uint32> row_to_col;
    std::vector<uint32> col_to_row;
    std::vector<Sum> row_potential;
    std::vector<Sum> col_potential;
};
```

Every row is matched when rows <= columns; every column is matched otherwise.
`none` appears only for unmatched vertices on the larger side. The matrix is
neither modified nor retained, and all returned vectors own their storage.
The matching has no promised tie order. An empty smaller dimension gives
zero cost, no matches and zero duals.

The dual certificate is `row_potential[i]+col_potential[j] <= costs[i][j]`,
with equality on matched pairs. For rectangular matrices, the larger side's
potentials are also nonpositive and exactly zero on unmatched vertices.
Together with a valid matching, these conditions prove optimality by LP duality.
Negate costs at the call site for maximization, using a type that can hold
the negated values.

## Complexity

For k=min(rows,columns), l=max(rows,columns), time is O(k²l+rows*columns)
in the worst case. Square Monge and anti-Monge inputs take O(n²), including
the detection scan. Working storage is O(rows*columns+rows+columns), retaining
the input element type for the flat matrix. The tall path transposes in
32-by-32 tiles. All alternating paths use loops and heap arrays with O(1)
call stack. The public entry is constexpr; allocation failures propagate.

## Preconditions

Asserted with `ATL_ASSERT`: equal row sizes, both dimensions below `none`,
and a representable product within the flat vector's `max_size`.
`none` never names a real row or column.

Documented-only: conversion of every cell to Sum is exact, and every sum,
difference and potential fits Sum. Let D=max(1,max absolute converted cost).
A conservative sufficient bound is `8*(k+1)*D < max(Sum)`. This covers
objective accumulation, telescoped Monge potentials, paired costs in Monge
detection, reduced-cost differences and shortest-path candidates below the
`max(Sum)/4` sentinel. A total objective fitting Sum alone is insufficient.
The test suite checks a mixed-sign 3-by-3 matrix with D=max(int64)/32, as well
as narrow matrices with 128-bit sums.

## Measured performance

`benchmarks/graph/assignment.cpp` compares the library's Jonker-Volgenant
preprocessing plus partitioned shortest augmenting paths with SAP alone,
epsilon-scaling auction, and the general min-cost-flow reduction. Separate
loops time column-reduction traversal, flattening copies and transpose order;
each sample includes its own working-storage allocation.

Intel Xeon Platinum 8280 at 2.70 GHz, GCC 15.2, RelWithDebInfo, assertions
disabled, 2026-10-15. Minimum of three per-run means, ten samples per row.
Plain flags: `-O2 -g -DNDEBUG -std=c++23`; native adds `-march=native`.
All times below are milliseconds; a dash means that operation is not in that
shape's race.

### Plain -O2

| contender | uniform 1e9 500² | uniform 0..9 500² | product mod 4096 500² | banded 500² | wide 100×5000 |
|---|---|---|---|---|---|
| library | 2.8 | 0.919 | 3.0 | 0.66 | 1.2 |
| by shortest augmenting paths | 7.1 | 0.74 | 9.2 | 0.46 | 1.2 |
| by auction | 7.7 | 4.4 | 6.6 | 5.3 | — |
| by min cost flow reduction | 33.7 | 12.1 | 33.6 | 10.1 | — |

### -O2 -march=native

| contender | uniform 1e9 500² | uniform 0..9 500² | product mod 4096 500² | banded 500² | wide 100×5000 |
|---|---|---|---|---|---|
| library | 3.3 | 0.926 | 3.5 | 0.651 | 1.2 |
| by shortest augmenting paths | 8.4 | 0.725 | 10.8 | 0.444 | 1.2 |
| by auction | 7.9 | 4.5 | 6.6 | 5.6 | — |
| by min cost flow reduction | 38.1 | 13.0 | 38.6 | 10.5 | — |

### Additional solves

| library shape | plain -O2 | -O2 -march=native |
|---|---|---|
| uniform 1e9 int32 500², int64 sums | 2.1 | 1.8 |
| Monge 500² | 0.644 | 0.527 |
| tall 5000×100 | 1.4 | 1.4 |
| same tall data stored wide 100×5000 | 1.1 | 1.0 |
| tall 2000×1000 int32 | 11.2 | 8.9 |
| uniform 1e9 int64 2000² | 98.8 | 113 |
| Monge 3000² | 62.2 | 60.3 |

### Matrix passes, plain -O2

| shape | column reduction by column major | by row major | flatten by push back | by row insert |
|---|---|---|---|---|
| uniform 1e9 500² | 0.331 | 0.295 | 0.209 | 0.203 |
| uniform 0..9 500² | 0.309 | 0.278 | 0.210 | 0.191 |
| uniform 1e9 int64 2000² | 32.8 | 5.0 | 6.0 | 6.8 |
| uniform 1e9 int32 2000² | 25.2 | 4.4 | 3.8 | 2.4 |
| uniform 1e9 int64 4000² | 200 | 21.6 | 77.8 | 81.1 |
| uniform 1e9 int32 4000² | 140 | 17.7 | 44.8 | 39.7 |
| Monge 3000² | — | — | 45.5 | 47.7 |

### Matrix passes, -O2 -march=native

| shape | column reduction by column major | by row major | flatten by push back | by row insert |
|---|---|---|---|---|
| uniform 1e9 500² | 0.343 | 0.125 | 0.210 | 0.202 |
| uniform 0..9 500² | 0.316 | 0.108 | 0.210 | 0.194 |
| product mod 4096 500² | 0.334 | 0.119 | 0.201 | 0.201 |
| banded 500² | 0.365 | 0.129 | 0.207 | 0.191 |
| uniform 1e9 int64 500² | 0.331 | 0.122 | 0.200 | 0.201 |
| uniform 1e9 int32 500² | 0.289 | 0.092 | 0.143 | 0.103 |
| uniform 1e9 int64 2000² | 32.3 | 2.9 | 6.0 | 6.9 |
| uniform 1e9 int32 2000² | 26.2 | 1.6 | 3.0 | 2.4 |
| uniform 1e9 int64 4000² | 200 | 15.2 | 78.1 | 80.9 |
| uniform 1e9 int32 4000² | 140 | 9.7 | 41.1 | 39.4 |
| Monge 3000² | — | — | 46.1 | 47.7 |

### Transpose passes

| shape | row order, plain | tiles, plain | row order, native | tiles, native |
|---|---|---|---|---|
| tall 5000×100 int64 | 0.788 | 0.841 | 0.787 | 0.806 |
| tall 2000×1000 int32 | 5.1 | 2.7 | 6.2 | 2.4 |
| tall 2000×1000 int64 | 8.1 | 5.9 | 8.1 | 6.0 |

JV preprocessing helps large-range costs but costs more than the tail it saves
on tie-heavy and banded matrices; SAP alone wins those shapes. The separate
matrix and sum types reduce the uniform 500² plain solve from 2.8 to 2.1 ms.
Row-major column minima cost 5.0 versus 32.8 ms on int64 2000² in plain mode.
Tiles help the 2000×1000 transpose, while 5000×100 sees a small regression.
Row inserts benefit several int32 copies but lose on large int64 copies, so
the solver keeps per-element push_back. Auction never strictly wins a recorded
shape. The general flow comparison remains useful for showing specialization
cost and as an independent oracle; its timings include the full reduction.

## Pitfalls and notes

The experimental header `atl/experimental/graph/assignment_variants.hpp`
provides `assignment_by_shortest_augmenting_paths<Sum=void>(costs)` and
`assignment_by_auction<Sum=void>(costs)`. Both return `assignment_result<S>`
with the same input vocabulary. SAP requires rows <= columns and keeps the
library's partitioned scan while disabling JV preprocessing. It includes the
exact dual. Auction requires a square integer matrix, scales costs by n+1,
then reduces epsilon to 1; only the matching and cost are returned, and its
dual vectors are empty. Tests certify SAP's dual, check auction's matching
and cost, and use brute force and min-cost flow as independent oracles.

Auction requires `abs(cost) <= max(S)/(4*(n+1))`; every scaled cost plus price,
bid increment, accumulated price and partial objective sum must also fit S.
These operations are asserted. Its epsilon phase count is O(log(n*max|cost|+1))
with O(n) work per bid; the bid count is data-dependent. All dimensions and
flat-storage sizes are checked. A separate textbook Hungarian engine is
excluded because SAP is already represented and independent oracles exist.

## How it works

For square matrices, Jonker-Volgenant preprocessing performs a row-major
column-minimum scan, reduction transfer for singly assigned rows, and two
augmenting-row-reduction passes. Strict minima preserve the first minimal row
within a column, without promising a particular full matching. Rectangular
instances use pure shortest augmenting paths: the square preprocessing needs
every column matched for its dual argument and can be suboptimal on wide input.

The shortest-path search partitions columns into scanned, labeled and pending
segments. It collects all columns at the next minimum in one pass and promotes
zero-slack columns immediately, avoiding repeated plateau scans. Only column
potentials are stored during the solve; a matched row's potential is implicit
from its tight matched edge. The result reconstructs both dual directions.

Square Monge detection checks adjacent 2-by-2 inequalities; they imply all
Monge inequalities. The exchange argument makes the diagonal optimal, or the
anti-diagonal for anti-Monge input. Column potentials telescope diagonal cost
steps in O(n), retaining a full certificate. Tall matrices are transposed into
the same flat storage and their matching and dual vectors are swapped back.

References: Jonker and Volgenant, 1987, *A Shortest Augmenting Path Algorithm
for Dense and Sparse Linear Assignment Problems*; Bertsekas, 1988,
*The Auction Algorithm: A Distributed Relaxation Method for the Assignment Problem*.

## See also

- [Graph vocabulary](graph.md)
- [Min-cost flow](min-cost-flow.md)
- [128-bit integers](int128.md)
