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

### Plain -O2

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| contender | uniform 1e9 500² | uniform 0..9 500² | product mod 4096 500² | banded 500² | wide 100×5000 |
|---|---|---|---|---|---|
| library | 2 | 0.523 | 2.1 | 0.385 | 0.494 |
| by shortest augmenting paths | 5.1 | 0.482 | 6.5 | 0.224 | 0.482 |
| by auction | 8.6 | 5 | 7.3 | 4.9 | — |
| by min cost flow reduction | 13.6 | 3 | 13.2 | 1.6 | — |

### -O2 -march=native

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| contender | uniform 1e9 500² | uniform 0..9 500² | product mod 4096 500² | banded 500² | wide 100×5000 |
|---|---|---|---|---|---|
| library | 2.1 | 0.563 | 2.2 | 0.386 | 0.468 |
| by shortest augmenting paths | 5.4 | 0.456 | 6.9 | 0.208 | 0.478 |
| by auction | 8.6 | 5.1 | 7.3 | 4.9 | — |
| by min cost flow reduction | 13.6 | 3 | 13.2 | 1.5 | — |

### Additional solves

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| library shape | plain -O2 | -O2 -march=native |
|---|---|---|
| uniform 1e9 int32 500², int64 sums | 1.6 | 1.5 |
| Monge 500² | 0.252 | 0.25 |
| tall 5000×100 | 0.546 | 0.539 |
| same tall data stored wide 100×5000 | 0.432 | 0.437 |
| tall 2000×1000 int32 | 6.9 | 6.7 |
| uniform 1e9 int64 2000² | 70.6 | 71.9 |
| Monge 3000² | 48 | 46.9 |

### Matrix passes, plain -O2

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | column reduction by column major | by row major | flatten by push back | by row insert |
|---|---|---|---|---|
| uniform 1e9 500² | 0.168 | 0.1 | 0.087 | 0.05 |
| uniform 0..9 500² | 0.165 | 0.08 | 0.087 | 0.048 |
| uniform 1e9 int64 2000² | 4.8 | 1.5 | 1.5 | 1.3 |
| uniform 1e9 int32 2000² | 2.8 | 1.3 | 1.3 | 0.582 |
| uniform 1e9 int64 4000² | 36 | 8.1 | 72.7 | 66.4 |
| uniform 1e9 int32 4000² | 21.6 | 6.3 | 38.1 | 33.8 |
| Monge 3000² | — | — | 41.5 | 38.1 |

### Matrix passes, -O2 -march=native

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | column reduction by column major | by row major | flatten by push back | by row insert |
|---|---|---|---|---|
| uniform 1e9 500² | 0.175 | 0.03 | 0.088 | 0.051 |
| uniform 0..9 500² | 0.165 | 0.027 | 0.088 | 0.05 |
| product mod 4096 500² | 0.167 | 0.028 | 0.087 | 0.05 |
| banded 500² | 0.151 | 0.038 | 0.087 | 0.05 |
| uniform 1e9 int64 500² | 0.173 | 0.03 | 0.088 | 0.05 |
| uniform 1e9 int32 500² | 0.151 | 0.026 | 0.085 | 0.027 |
| uniform 1e9 int64 2000² | 4.8 | 0.777 | 1.4 | 1.2 |
| uniform 1e9 int32 2000² | 2.6 | 0.462 | 1.3 | 0.559 |
| uniform 1e9 int64 4000² | 35.9 | 5.4 | 72.2 | 66 |
| uniform 1e9 int32 4000² | 22.1 | 4 | 37.8 | 33.4 |
| Monge 3000² | — | — | 40.7 | 37.3 |

### Transpose passes

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | row order, plain | tiles, plain | row order, native | tiles, native |
|---|---|---|---|---|
| tall 5000×100 int64 | 0.21 | 0.261 | 0.209 | 0.265 |
| tall 2000×1000 int32 | 1.3 | 0.836 | 1.3 | 0.822 |
| tall 2000×1000 int64 | 2.2 | 1.9 | 2.2 | 2.1 |

The library beats SAP alone on the large-range square matrices, but loses
on tie-heavy costs (0.523/0.563 versus 0.482/0.456 ms) and banded matrices
(0.385/0.386 versus 0.224/0.208 ms). The wide shape favors SAP plain
(0.482 versus 0.494 ms) and the library native (0.468 versus 0.478 ms).
Separate matrix and sum types reduce the uniform 500² solve from 2/2.1 to
1.6/1.5 ms. Row-major column minima take 1.5 versus 4.8 ms on int64 2000²
plain (0.777 versus 4.8 ms native). Tiles help both 2000×1000 transposes,
but slow 5000×100 from 0.21/0.209 to 0.261/0.265 ms, or 24.29%/26.79%.
Row insertion beats per-element `push_back` on every represented flattening
workload, including int64 4000² (66.4/66 versus 72.7/72.2 ms).
Auction loses every measured shape. The general flow timings include the
full reduction; that contender also serves as an independent oracle.

The int64 2000×1000 transpose rows vary between runs (spreads up to
1.76x), as do some matrix passes; the transpose minima alone do not
establish a stable tile advantage.

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
