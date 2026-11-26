# `atl::graph::max_flow`

Maximum flow routes capacity from a source to a sink and returns a feasible
per-edge witness with a minimum-cut certificate. Use `max_flow_value` when
only the value matters. Lower-bound wrappers solve circulations, supply/demand
feasibility and signed source-to-sink extremal flows without cost machinery.

```cpp
#include <atl/graph/max_flow.hpp>
```

\include max-flow/quick.cpp

Expected stdout:
\include max-flow/quick.out

## API

All entries are constexpr. `EdgeList` satisfies `capacitated_edge_list`: a
const forward range whose values and scanned references have endpoints and
capacities. `flow_arc<Capacity>{from, to, capacity}` is the canonical edge;
`bounded_flow_arc<Capacity>{from, to, lower, capacity}` adds bounds.
Accessors use named members or `graph_traits`; `capacity`, `lower` and `cost`
are distinct from path `weight`. Output positions follow input iteration
order, including parallel edges and loops. Input storage is neither modified
nor retained; returned vectors own their storage.

Every function has template parameters
`template<typename Flow = void, capacitated_edge_list EdgeList>`.
`Flow = void` selects `capacity_t<EdgeList>`; an explicit type widens arithmetic.
The selected type F must satisfy `flow_value`, be semiregular, accept implicit
integer zero and be constructible from scanned capacities. Boolean F is
rejected because packed vector storage cannot provide ordinary F references. Plain entries reject lower bounds on
both value and reference types. Bounded entries also require
`signed_flow_value<F>` and construction from scanned lower bounds when present.
Signedness comes from `numeric_limits`, including software `atl::int128`.

| constexpr declaration | result and meaning |
|---|---|
| `auto max_flow(std::size_t n, EdgeList const &edges, vertex_id source, vertex_id sink)` | `max_flow_result<F>`; full witness and cut |
| `auto max_flow_value(std::size_t n, EdgeList const &edges, vertex_id source, vertex_id sink)` | F; value without a witness |
| `auto feasible_flow(std::size_t n, EdgeList const &edges, std::vector<detail::flow_t<Flow, EdgeList>> b) -> std::optional<std::vector<detail::flow_t<Flow, EdgeList>>>` | flows with outgoing minus incoming flow equal to b[v]; nullopt if infeasible |
| `auto bounded_max_flow(std::size_t n, EdgeList const &edges, vertex_id source, vertex_id sink)` | `optional<max_flow_result<F>>`; maximum signed value |
| `auto bounded_min_flow(std::size_t n, EdgeList const &edges, vertex_id source, vertex_id sink)` | `optional<max_flow_result<F>>`; minimum signed value |

`detail::flow_t` in the supply parameter denotes the selected F described above.
The common result is:

```cpp
template<flow_value Flow>
struct max_flow_result {
    Flow value{};
    std::vector<Flow> edge_flow;
    std::vector<uint8> source_side;
};
```

For plain flow, `source_side[v]` is 0 or 1 according to residual reachability
from the source. Feasibility plus saturated forward and empty backward edges
crossing this cut certify optimality. Minimum-cut edges are input edges leaving
this set. Unit-capacity edges with positive flow recover a matching; positive
flow can also be decomposed into paths and cycles.

For bounded maximum flow the source set S certifies
`value = sum upper(S->T) - sum lower(T->S)`; for bounded minimum flow it certifies
`value = sum lower(S->T) - sum upper(T->S)`. Values can be negative: bounds can
force backward flow. A caller requiring nonnegative value checks it separately.

`feasible_flow` accepts ordinary edges with implicit lower bound zero. Positive
b[v] means supply, negative means demand; b=0 asks for a circulation. Unequal
total supply and demand returns nullopt. The vector b is taken by value and
reused; `std::move(b)` avoids its copy.

\include max-flow/bounds.cpp

Expected stdout:
\include max-flow/bounds.out

## Complexity

For n vertices and m input edges, plain entries take
O(n² sqrt(m) + n + m) time and O(n+m) heap storage. The lower-bound reductions
add two vertices and at most n supply arcs; their bound is
O((n+2)² sqrt(m+n) + n+m). Feasibility uses one maximum-flow call;
bounded extremals use two. All traversals use heap queues or explicit paths,
with O(1) call stack even on million-vertex paths. Allocation failures propagate.

## Preconditions

Asserted under `ATL_ASSERT`:

- Plain flow has distinct source and sink below n, n < 2³¹, endpoints in [0,n),
  nonnegative capacities and m <= UINT32_MAX/2. `none` is never a valid vertex.
- The sum of non-loop source-outgoing capacities fits F. Preflow can collect
  this entire amount at one vertex, even when the maximum flow is tiny.
- Lower-bound entries have n < 2³¹-2, 0 <= lower <= capacity and valid endpoints.
  Feasibility requires b.size()==n. Partial supply sums, lower-bound supply
  adjustments, negated demands and the total positive adjusted supply fit F.
- Bounded extremals require `1 + sum capacity` to fit F. This bounds closure
  shortcuts, supply adjustments, residual capacities and final signed values.
- Each actual augmented network has at most UINT32_MAX/2 edges. Feasibility
  has at most m+n edges; bounded closure at most m+2+n, and augmentation 2m.
  Reservation arithmetic must also fit the receiving vector's max_size.

Documented-only: conversions to F are exact, arithmetic and `numeric_limits`
model bounded real values, and edge iteration remains stable through a call.
Floating capacities and intermediate values must be finite. No tolerance is
applied: exact certificates require exactly representable intermediate
additions and subtractions. A sufficient condition for binary floating plain
flow is a common power-of-two quantum q, with every capacity and the source
sum at most 2^digits quanta, and all values within the exponent range. The
double test exercises the 2^53 boundary in eighths. Dyadic inputs exceeding
that significand bound can round; use scaled integers and widen F when needed.

## Measured performance

`benchmarks/graph/max_flow.cpp` compares two-phase highest-label push-relabel
(HLPP), its value-only phase, Dinic, capacity-scaling Dinic and ISAP. It also
isolates direct versus reverse-arc debit, height-first versus capacity-first
BFS tests, and residual bucket grouping versus scatter. Each sample includes
construction and one complete operation on the named shape. Edmonds-Karp and
capacity-scaling Edmonds-Karp remain readable certifying test oracles; their
O(nm²+n+m) and O((n+m)m log(U+1)+n+m) bounds exclude them from timing at these sizes.

### Plain -O2

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| contender | random 1e5 / 5e5 | bipartite 2e4+2e4 | grid 300×300 | dense 2e3 / 4e5 | layered 200×50 |
|---|---|---|---|---|---|
| by dinic | 169 | 26.3 | 3120 | 21.9 | 0.987 |
| by capacity scaling dinic | 439 | 28 | 7290 | 244 | 1.2 |
| by isap | 43.7 | 22.5 | 1080 | 16.5 | 0.991 |
| library | 105 | 18.8 | 129 | 24.6 | 1.8 |
| library value | 66.8 | 15.8 | 122 | 12.8 | 1.1 |
| by discharge order capacity first | 65.6 | — | — | 12.3 | — |
| by discharge order direct debit | 66.3 | — | — | 13.4 | — |
| by discharge order height first | 62.8 | — | — | 7 | — |
| by discharge order direct debit height first | 61 | — | — | 6.7 | — |
| residual library | 6.8 | — | — | 2.7 | 0.291 |
| residual by bucket grouping | 12.6 | — | — | 8.1 | 0.54 |

### -O2 -march=native

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| contender | random 1e5 / 5e5 | bipartite 2e4+2e4 | grid 300×300 | dense 2e3 / 4e5 | layered 200×50 |
|---|---|---|---|---|---|
| by dinic | 178 | 29.4 | 4250 | 23.1 | 0.999 |
| by capacity scaling dinic | 453 | 30.9 | 9440 | 218 | 1.1 |
| by isap | 51.7 | 26.2 | 1640 | 20.2 | 1 |
| library | 152 | 25.9 | 148 | 38.1 | 1.9 |
| library value | 96.4 | 21.9 | 137 | 20.7 | 1.2 |
| by discharge order capacity first | 95.7 | 23.1 | 134 | 19.8 | 1.2 |
| by discharge order direct debit | 88.7 | 21.5 | 135 | 19.3 | 1.2 |
| by discharge order height first | 86.3 | 23.1 | 130 | 9.3 | 1.1 |
| by discharge order direct debit height first | 82 | 21.4 | 130 | 9.2 | 1.1 |
| residual library | 12.3 | 0.866 | 1.3 | 6.6 | 0.338 |
| residual by bucket grouping | 14.3 | 1.3 | 2.2 | 11 | 0.639 |

HLPP takes 129/148 ms on the grid versus ISAP's 1080/1640 and Dinic's
3120/4250 ms. The library loses sparse random to ISAP (105/152 versus
43.7/51.7 ms), dense input to ISAP (24.6/38.1 versus 16.5/20.2 ms),
and layered input to Dinic (1.8/1.9 versus 0.987/0.999 ms). Value-only HLPP saves the second
phase and witness construction. Capacity scaling wins none of these shapes;
Dinic's phase bound is already independent of capacity values.

The micro comparisons favor height-first over capacity-first on the
native grid (130 versus 134 ms), and direct debit takes 135 ms there.
Bucket grouping loses to library scatter (2.2 versus 1.3 ms on that grid).
Sorting adjacency by target adds construction
work without an established end-to-end win here.

The native ISAP grid, complete library dense and complete library layered
rows vary between runs (1.30x, 1.31x and 1.37x spread); those minima alone
do not establish stable rankings. The plain random residual-scatter row
also varies (1.80x).

## Pitfalls and notes

Give conceptual infinite capacities finite, representable bounds. Plain max
flow rejects lower-bounded edge types at compile time; use the wrappers rather
than discarding lower bounds. Cut membership is a certificate, not a promise
of a unique cut or a particular flow decomposition.

The experimental header `atl/experimental/graph/max_flow_variants.hpp` provides
`max_flow_by_edmonds_karp`, `max_flow_by_dinic`, `max_flow_by_isap`,
`max_flow_by_capacity_scaling_edmonds_karp` and
`max_flow_by_capacity_scaling_dinic`. They use the same function parameters and template defaults as `max_flow`,
return `max_flow_result<F>`, and support heap-backed paths. Scaling accepts integer flow types, including
portable 128-bit integers; a halving ladder has no floating-point meaning.
Non-scaling variants accept floating capacities under the arithmetic conditions
above. Their value accumulation must fit F; they do not saturate all source
arcs simultaneously.

## How it works

The residual stores each vertex's arcs contiguously in CSR order and uses an
explicit mate index for reverse capacities. HLPP saturates outgoing source
arcs, discharges the highest active vertex, and periodically rebuilds exact
sink distances with reverse-residual BFS. A circular discharge scan pushes
along admissible arcs while collecting a relabel target. Intrusive active
buckets and per-height gap lists avoid allocation in the hot loop; an empty
height band retires exactly the stranded vertices above it. A global relabel
runs per n local relabels. The second phase roots the same engine at the source
to drain stranded excess, leaving a feasible witness. Heights cap at n in both
phases. `max_flow_value` ends with the first phase.

Lower bounds shift supplies by their forced divergence. A super-source and
super-sink reduce feasibility to saturating all positive adjusted supply.
Bounded extremals close the network with shortcuts in both directions, so
negative feasible values remain discoverable. They remove the shortcuts and
augment forward for a maximum or backward for a minimum; residual reachability
provides the respective Gale-Hoffman cut equality. Tests verify conservation,
bounds and cut equality and cross-check extremals with an independent cost
engine.

## See also

- [Min-cost flow](min-cost-flow.md) for costed capacities and prescribed supplies.

- [Graph vocabulary and representations](graph.md)
- [Matrices and matrix-tree counts](matrix.md)
- [128-bit integers](int128.md)
