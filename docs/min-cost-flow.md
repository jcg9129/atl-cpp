# `atl::graph::min_cost_max_flow` and `atl::graph::min_cost_feasible_flow`

Min-cost flow optimizes capacitated networks with edge costs, including negative
costs and negative cycles. Use the source-to-sink entry to maximize value then
minimize cost, or the supply entry for prescribed divergence and lower bounds.
For costless bounded flow, use the dedicated maximum-flow reductions.

```cpp
#include <atl/graph/min_cost_flow.hpp>
```

\include min-cost-flow/quick.cpp

Expected stdout:
\include min-cost-flow/quick.out

## API

Every entry has template parameters
`template<typename Flow = void, typename Cost = void, cost_capacitated_edge_list EdgeList>`.
`Flow=void` selects `capacity_t<EdgeList>`; `Cost=void` selects `cost_t<EdgeList>`.
The selected F and C satisfy `signed_flow_value` and the checked arithmetic and
construction requirements. Floating types are admitted under exact-arithmetic
preconditions below. `EdgeList` is a stable const forward range; both its values
and scanned references expose endpoints, capacity and cost, using named members
or `graph_traits`. Boolean flow storage is rejected. Plain maximum flow also
rejects lower bounds on either edge values or scanned references.

| declaration | result and meaning |
|---|---|
| `auto min_cost_max_flow(std::size_t n, EdgeList const &edges, vertex_id source, vertex_id sink, detail::flow_t<Flow, EdgeList> limit = std::numeric_limits<detail::flow_t<Flow, EdgeList>>::max())` | `min_cost_flow_result<F,C>`; maximum value up to limit, minimum cost among those flows |
| `auto min_cost_feasible_flow(std::size_t n, EdgeList const &edges, std::vector<detail::flow_t<Flow, EdgeList>> b) -> std::optional<min_cost_flow_result<detail::flow_t<Flow, EdgeList>, detail::cost_type<Cost, EdgeList>>>` | prescribed outgoing minus incoming flow b[v]; nullopt if infeasible or supplies do not balance |
| `std::vector<detail::cost_type<Cost, EdgeList>> min_cost_flow_potentials(std::size_t n, EdgeList const &edges, std::vector<detail::flow_t<Flow, EdgeList>> const &edge_flow)` | opt-in dual certificate for a cost-optimal flow |

The detail aliases in the declarations mean the selected F and C above.
All entries are runtime APIs: residual construction and the simplex engine are
not constexpr. Assignment has a constexpr specialized solver.

```cpp
template<signed_flow_value Flow, signed_flow_value Cost>
struct min_cost_flow_result {
    Flow value{};
    Cost cost{};
    std::vector<Flow> edge_flow;
};
```

`value` is source-to-sink value for maximum flow, and the sum of positive
original supplies for feasible flow. `cost` is the input-order sum of
`edge_flow[i]*cost(edges[i])`. The owned witness follows input iteration order,
including loops and parallel arcs; no input storage is changed or retained.
Optimal witnesses need not be unique, and no tie order is promised.

The edge vocabulary comes from `atl/graph/concepts.hpp` and
`atl/graph/max_flow.hpp`: `flow_arc<Capacity,Cost>{from,to,capacity,cost}` and
`bounded_flow_arc<Capacity,Cost>{from,to,lower,capacity,cost}`.
Plain edges have lower bound zero. The supply vector is taken by value and reused;
pass `std::move(b)` to avoid a copy. Positive b is supply, negative b is demand.

\include min-cost-flow/supplies.cpp

Expected stdout:
\include min-cost-flow/supplies.out

The dual satisfies `cost(a)+p[tail(a)]-p[head(a)] >= 0` for every residual arc.
Forward residual capacity is `capacity-flow`; backward capacity is `flow-lower`.
Bounds, conservation and this inequality prove minimum cost for the given
supplies. Maximum value additionally needs a maximum-flow or cut check, unless
the requested limit binds. The certificate excludes the solver's closure arc;
it certifies the returned value, rather than optimizing value again.

## Complexity

The primal network simplex uses O(n+m) heap memory. Each pivot walks a tree
cycle and a detached subtree in O(n); block pricing can scan O(n+m) arcs to
find an improving entry. This pivot rule has no polynomial bound on the number
of pivots. The certificate uses an O(nm+n+m) worst-case queue Bellman-Ford pass
and O(n+m) storage. All graph walks use heap queues or explicit paths, with
O(1) call stack even on million-vertex paths. Allocation failures propagate.

## Preconditions

Asserted with `ATL_ASSERT`:

- n < `none-1`, endpoints in [0,n), distinct valid source and sink for maximum
  flow, and a nonnegative limit. `none` never names a vertex.
- Capacities are nonnegative; bounded inputs require `0 <= lower <= capacity`.
  Supply vectors have length n. The feasible-flow solver rejects negative lower bounds. The certificate also
  accepts negative lower bounds when capacities are nonnegative and both
  residual differences fit F.
- The simplex has at most `UINT32_MAX-n` real arcs, including the maximum-flow
  closure arc. The certificate has at most `UINT32_MAX/2` input edges.
- For a simplex whose real arcs have largest absolute cost D, the checked
  sufficient cost bound is `D < max(C)/5/(n+1)` using integer division.
  Artificial arcs cost `(n+1)*(D+1)`; their potentials and reduced costs fit
  under this bound. A cost cannot be the lowest representable C.
- Maximum flow adds a closure arc of reward `n*max|input cost|+1`, also checked
  without overflowing. Apply the simplex bound to this enlarged D: internal
  cost headroom is of order n² times the largest input cost. Feasible flow
  needs order n times its largest cost.
- Partial supply sums, total positive supply, each lower-bound supply adjustment,
  negated adjusted demand, every objective product and every partial objective
  sum must fit their selected types; checks precede those operations.

Documented-only: conversions to F and C are exact; values and every intermediate
are finite and exactly representable. All vector lengths fit their `max_size`.
Widen C for objective sums as well as potentials: a sufficient objective bound
is `sum capacity(e)*abs(cost(e)) <= max(C)`. F must cover every individual
capacity and adjusted supply magnitude. The source capacity sum is saturated
at the requested limit, so multiple large source capacities need not sum in F.
The artificial capacity is `max(F)`, rather than a smaller pseudo-infinity.

For the certificate, `(n+1)*max|cost|` must fit C, including relaxation
candidates; evaluating the dual inequality externally needs up to
`(2n+1)*max|cost|`. The residual must have no negative-cost cycle; a widened
pop-count bound asserts on violations. Calling the certificate on arbitrary
suboptimal flows is outside its contract. No floating tolerance is used;
scaled integers give exact certificates most directly.

## Measured performance

`benchmarks/graph/min_cost_flow.cpp` races threaded network simplex against
SPFA SSP, Johnson-Dijkstra SSP, zkw admissible-subgraph DFS, capacity scaling,
and cost scaling with and without global price updates. Separate loops isolate
arc layout, adjacency-tree traversal and the residual certificate. Samples
measure complete operations with their local working-storage allocation.

Intel Xeon Platinum 8358 at 2.60 GHz, GCC 15.2, RelWithDebInfo, assertions
disabled, 2026-10-15. Minimum of three per-run means, ten samples per row.
Plain flags: `-O2 -g -DNDEBUG -std=c++23`; native adds `-march=native`.
All times below are milliseconds. The supplied complete-workload solver record
includes certificate work; this API exposes that work separately. Fresh
solve-only timings are not measured yet. The isolated rows retain the measured
core and certificate costs without subtracting independently measured minima.

### Plain -O2

| contender | random 2e4/1.2e5 | assignment 300² | grid 80² | parallel paths 4e3 | negative DAG 60×40 |
|---|---|---|---|---|---|
| library | 65.3 | 10.9 | 86.1 | 2.6 | 53.3 |
| by cost scaling price update | 107 | 33.7 | 206 | 5.7 | 71.5 |
| by cost scaling | 602 | 48.7 | 706 | 2.9 | 191 |
| by capacity scaling | 4003 | 72.7 | 290 | 844 | 464 |
| by dijkstra | 254 | 95.1 | 789 | 663 | 275 |
| by zkw | 336 | 73.7 | 888 | 815 | 339 |
| by spfa | 251 | 103 | 2363 | 97.2 | 391 |

### -O2 -march=native

| contender | random 2e4/1.2e5 | assignment 300² | grid 80² | parallel paths 4e3 | negative DAG 60×40 |
|---|---|---|---|---|---|
| library | 65.2 | 11.2 | 87.7 | 2.6 | 53.7 |
| by cost scaling price update | 106 | 33.4 | 208 | 5.7 | 71.7 |
| by cost scaling | 599 | 48.5 | 713 | 2.8 | 185 |
| by capacity scaling | 3918 | 73.8 | 286 | 831 | 458 |
| by dijkstra | 255 | 124 | 796 | 676 | 278 |
| by zkw | 337 | 75.6 | 888 | 808 | 337 |
| by spfa | 251 | 103 | 2278 | 92.6 | 390 |

### Isolated operations, plain -O2

| operation | random 2e4/1.2e5 | assignment 300² | grid 80² | paths 4e3 | negative DAG | random 5e4/3e5 |
|---|---|---|---|---|---|---|
| simplex core library | 42.9 | 7.1 | 53.8 | 2.4 | 50.5 | 177 |
| simplex core by adjacency | 64.1 | 11.3 | 123 | 4.0 | 118 | 279 |
| certificate | 21.0 | 3.2 | 30.7 | 0.178 | 2.7 | — |
| library complete workload | — | — | — | — | — | 238 |

### Isolated operations, -O2 -march=native

| operation | random 2e4/1.2e5 | assignment 300² | grid 80² | paths 4e3 | negative DAG | random 5e4/3e5 |
|---|---|---|---|---|---|---|
| simplex core library | 42.7 | 7.1 | 54.3 | 2.5 | 50.7 | 178 |
| simplex core by struct arcs | 46.6 | 7.6 | 57.2 | 2.8 | 53.1 | 192 |
| simplex core by flat arcs | 42.9 | 7.1 | 54.7 | 2.5 | 51.0 | 178 |
| simplex core by adjacency | 66.1 | 11.3 | 130 | 4.1 | 125 | 290 |
| certificate | 21.0 | 3.2 | 32.0 | 0.153 | 2.8 | — |
| library complete workload | — | — | — | — | — | 240 |

The library wins the five complete-workload shapes under both flag sets.
Global price updates improve cost scaling on random, assignment, grid and DAG
inputs, while adding work on parallel paths. Early Dijkstra termination avoids
labels beyond the sink; capacity scaling pushes the full bottleneck, using its
threshold only for eligibility. Parallel paths expose the augmentation count
of the SSP family. Flat pricing arrays and threaded subtree walks reduce the
core cost without changing the pivot rule.

## Pitfalls and notes

The experimental header `atl/experimental/graph/min_cost_flow_variants.hpp`
provides `min_cost_max_flow_by_spfa`, `min_cost_max_flow_by_dijkstra`,
`min_cost_max_flow_by_zkw`, `min_cost_max_flow_by_capacity_scaling`,
`min_cost_max_flow_by_cost_scaling` and
`min_cost_max_flow_by_cost_scaling_price_update`. They share the maximum-flow
entry's template defaults, arguments and witness. SPFA is the readable oracle;
all contenders are cross-checked for value and cost, and the oracle and global
price update implementation also receive dual checks in tests.

SSP and zkw require no negative-cost input cycles. Both scaling contenders
retain that documented input restriction; use the library for arbitrary
negative cycles. All experimental intermediate arithmetic must fit its types:
F covers incident-capacity excesses and injected supply; C covers path sums,
reduced costs, potentials, each objective product and sum. Capacity scaling
also forms `2*(1+sum|cost|)`; cost scaling forms `abs((n+1)*cost)`, epsilon-scaled
labels and price shifts. Exact epsilon scaling needs integer-valued costs.
Cost scaling also inherits `max_flow_value`'s n < 2³¹ and source-capacity
sum bounds. All experimental residuals have at most UINT32_MAX/2 edges,
including capacity scaling's closure arc. These bounds can exceed those of
the library solver. No contender recurses
per vertex.

Parallel arcs model convex piecewise costs; fixed divergence models flow repair.
Bounded maximum flow can be expressed by a closure arc of cost -1, but the
costless lower-bound reductions avoid simplex overhead on large instances.
A subtree walk can dominate at tens of thousands of vertices.

## How it works

The primal network simplex keeps a strongly feasible spanning tree rooted at
an artificial vertex. Wraparound block pricing selects a violating arc; the
induced cycle is pushed until the last blocking arc from the apex leaves.
That leaving rule prevents degenerate cycling. Each subtree is a contiguous
preorder thread segment with a stored size and final vertex. Reattachment
splices segments using heap-backed stem arrays, then shifts all potentials
in the detached subtree by one constant. Flat arrays store arc endpoints
and costs for pricing, including the artificial arcs.

Artificial costs exceed every real simple-path cost, so feasible supplies
eliminate all artificial flow. Lower bounds shift supplies, then get added to
the returned witness. Maximum flow adds a sink-to-source reward arc that makes
every augmenting path profitable; finite input capacities bound the optimum,
including profitable cycles. The separate Bellman-Ford pass builds bounded
potentials from a virtual source adjacent to every vertex. Feasibility and
nonnegative residual reduced costs give the LP duality certificate.

Reference: Ahuja, Magnanti and Orlin, 1993, *Network Flows*, chapter 11;
Goldberg, 1997, *An Efficient Implementation of a Scaling Minimum-Cost Flow Algorithm*.

## See also

- [Graph vocabulary](graph.md)
- [Maximum flow and lower bounds](max-flow.md)
- [Dense assignment](assignment.md)
- [128-bit integers](int128.md)
