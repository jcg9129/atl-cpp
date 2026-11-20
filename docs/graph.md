# `atl::graph`

Dense vertex ids, named edge fields and constrained algorithms let the same graph
code consume adjacency containers, compressed sparse rows or generated neighbors.
Use the traversal views for early exit and range composition, and the eager
visitors for complete traversals. This page covers vocabulary, representations,
traversal and the common partition result; algorithm guides are linked below.

```cpp
#include <atl/types/vertex.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/graph/partition.hpp>
```

\include graph/quick.cpp

## Vocabulary and API

`vertex_id` is `atl::uint32`. Vertices are `[0, vertex_count(g))`; `none` is
`2^32 - 1`, reserved for no vertex, no edge or unreached. The vertex count may
be at most `none`, so `none` never names a vertex. The lightweight
`<atl/types/vertex.hpp>` provides these names and `vertex_range<R>` for graph
and dynamic-forest data structures without including the graph algorithms.
`vertex_range` checks convertibility, including proxy ranges; values still
must fit the receiving operation's domain.

| Canonical element | Fields | Use |
|---|---|---|
| `edge<Weight = void>` | `to`, `weight` (omitted for `void`) | adjacency |
| `arc<Weight = void>` | `from`, `to`, optional `weight` | edge list |
| `flow_arc<Capacity, Cost = void>` | `from`, `to`, `capacity`, optional `cost` | capacity vocabulary |
| `bounded_flow_arc<Capacity, Cost = void>` | `from`, `to`, `lower`, `capacity`, optional `cost` | bounded-capacity vocabulary |

For each `Edge`, `target(Edge const &e)` returns `vertex_id`: a non-character,
non-boolean integral value itself, `graph_traits<Edge>::target(e)`, or `e.to`,
in that priority order. `source(Edge const &e)` uses traits then `e.from`.
The `decltype(auto)` accessors `weight`, `capacity`, `lower` and `cost` use
traits then their matching named field and preserve references. Each is
available only when its expression is valid. `lower` has no fallback accessor
for an absent field. Cost and path weight are separate fields.

A `graph_traits<Edge>` specialization supplies static accessors for types
whose fields cannot follow these names. Positional pairs are not edges unless
adapted explicitly. Boolean, character and floating endpoints are rejected by
the integral/member tiers; strong wrappers implicitly converting to `vertex_id`
are accepted as members. Traits explicitly control their own endpoint types.

`vertex_count(Graph const &g)` returns `std::size_t` and
`out_edges(Graph const &g, vertex_id u)` returns `decltype(auto)`.
Both prefer the member pair `g.vertex_count()` / `g.out_edges(u)`; otherwise a
const sized random-access range of edge ranges uses its size and indexed row.

| Concepts | Checked requirement |
|---|---|
| `adjacency_edge<Edge>`, `weighted_adjacency_edge<Edge>` | extractable target, optionally weight |
| `list_edge<Edge>`, `weighted_list_edge<Edge>` | extractable source and target, optionally weight |
| `adjacency_graph<Graph>`, `weighted_adjacency_graph<Graph>` | const shape access and input edge ranges with usable values and references, optionally weighted values and references |
| `edge_list<EdgeList>`, `weighted_edge_list<EdgeList>` | const forward range of usable list-edge values and references, optionally weighted values and references |
| `vertex_pair_range<R>` | sized forward range of two-element tuple-like values convertible to ids |
| `capacitated_edge<Edge>`, `bounded_capacitated_edge<Edge>`, `cost_capacitated_edge<Edge>` | list edge with capacity, optionally lower bound or cost |
| `capacitated_edge_list<EdgeList>`, `cost_capacitated_edge_list<EdgeList>` | edge list with capacity, optionally cost |
| `flow_value<Flow>`, `signed_flow_value<Flow>` | ordered additive value constructible from zero; signedness follows `numeric_limits`, and signed values support unary negation |

Type aliases `edge_weight_t<Edge>`, `edge_capacity_t<Edge>`, `edge_cost_t<Edge>`
strip cv/ref from accessor results. `out_edges_t<Graph>` strips cv/ref from
the row type; `edge_t<Graph>` is its value type; `weight_t<Graph>` is the edge
weight type. `capacity_t<EdgeList>` and `cost_t<EdgeList>` are the edge-list
field types. These aliases require the corresponding accessor to exist.

### Complexity and preconditions

Accessors add O(1) work to the underlying member/trait operation. Integral
endpoints outside `[0, 2^32)` violate an asserted precondition, including
integral trait results; strong-wrapper conversions must preserve the id.
`out_edges(g, u)` requires `u < vertex_count(g)`. Every actual adjacency target
is below the vertex count. Adjacency order must be deterministic across calls,
and returned references must remain valid while an algorithm scans the graph.
These graph laws are documented preconditions; concepts cannot inspect values.

## Representations

| Representation | Concept and contract |
|---|---|
| `vector<vector<int>>` | unweighted `adjacency_graph`, without wrapping |
| `vector<vector<edge<W>>>` | `weighted_adjacency_graph` |
| `csr<Edge>` | contiguous rows built by stable counting sort |
| `implicit_graph<Neighbors>` | rows generated by a const-callable neighbor function |
| `vector<arc<W>>` | const multipass `edge_list`; a consumer may count then fill |

`vector<string>` and `vector<vector<bool>>` are not adjacency graphs. A
vector of sets or lists can be an adjacency graph. Traversal's resumable scans
require const forward rows when returned by reference; by-value rows must be
const sized random-access ranges, default-initializable and assignable because
the scan owns them. This admits CSR spans and generated `static_vector`s while
avoiding iterators that would dangle when a frame moves.

### CSR API

For `template<adjacency_edge Edge> class csr`, `value_type` is `Edge`.
Constructors additionally require a default-initializable movable stored
edge and a compatible conversion from the list element.

| Signature | Result / semantics | Complexity |
|---|---|---|
| `constexpr csr(std::size_t n, R const &arcs)` | directed graph from `edge_list R`; CTAD selects `vertex_id` or `edge<edge_weight_t<A>>` | O(n + m) time and storage |
| `constexpr csr(std::size_t n, R const &arcs, Proj proj)` | `edge_list R`; const-invocable projection converts each list element to `Edge` | O(n + m), plus projection costs |
| `static constexpr csr undirected(std::size_t n, R const &arcs)` | each element contributes forward and reverse adjacency | O(n + m) |
| `constexpr auto undirected_csr(std::size_t n, R const &arcs)` | deduced undirected build | O(n + m) |
| `static constexpr csr adopt(std::vector<uint32> offsets, std::vector<Edge> edges)` | owns the supplied arrays | O(1), O(n + m) with assertions |
| `constexpr std::size_t vertex_count() const noexcept` | number of rows | O(1) |
| `constexpr std::size_t edge_count() const noexcept` | directed stored elements | O(1) |
| `constexpr std::span<Edge const> out_edges(vertex_id u) const` | read-only row | O(1) |
| `constexpr std::size_t out_degree(vertex_id u) const` | row length | O(1) |

An integral stored edge takes the target alone; a weighted edge is constructed
as `Edge{target, weight}`; another unweighted edge as `Edge{target}`. The
reverse direction uses `source(a)` directly, including trait-adapted arcs.
The input order within each source row is stable. In an undirected build, each
arc contributes its forward entry then reverse entry; a self-loop contributes
two consecutive entries. Parallel edges remain present.

Asserted: endpoints are below `n`, `n <= none`, `n + 1` fits `size_t`, and the
stored edge count fits `uint32` (twice the input count for undirected graphs).
The count is checked while counting, ahead of offset overflow. A narrower
integral stored type must represent every target. Projections must produce
valid targets; they are responsible for fields beyond the standard conversion.
`adopt` asserts a nonempty, sorted offsets array starting at zero, ending at
`edges.size()`, with a representable vertex count and valid targets.
An invalid precondition aborts only when `ATL_ASSERT` is active. Standard
allocation failures propagate. Spans last until their storage owner is destroyed
or assigned over; moving the owning vectors transfers the storage.

### Generated graph API

`constexpr implicit_graph(std::size_t n, Neighbors neighbors)` owns its callable;
`constexpr std::size_t vertex_count() const noexcept` returns `n`, and
`constexpr decltype(auto) out_edges(vertex_id u) const` calls `neighbors(u)`.
The callable must be constructible from its rvalue, accept `vertex_id` as a const object and return an input
range of adjacency edges. Construction and vertex count add O(1) work;
row cost and lifetime follow the callable. Asserted: `n <= none` and `u < n`.
Repeated generation must preserve adjacency order during a traversal. For a
grid, generating four moves can avoid loading a materialized edge array.

## Traversal views and visitors

`visit{vertex, parent, depth}` describes a discovery. A root has itself as
parent and depth zero. Traversals use heap queues or explicit heap stacks;
DFS never recurses per vertex, including on a million-vertex path.

| Entry point | Behavior |
|---|---|
| `constexpr auto bfs(Graph const &g, vertex_id source)` | `bfs_view<Graph>` in nondecreasing depth |
| `constexpr auto bfs(Graph const &g, Sources const &sources)` | one depth-zero frontier; nearest-source distances |
| `constexpr auto bfs(Graph const &g, vertex_id source, Marker &marker)` | persistent visited state |
| `constexpr auto bfs(Graph const &g, Sources const &sources, Marker &marker)` | multiple sources and persistent marks |
| `constexpr auto dfs(Graph const &g, vertex_id source)` | `dfs_view<Graph>` in recursive preorder, children in adjacency order |
| `constexpr auto dfs(Graph const &g, vertex_id source, Marker &marker)` | preorder with persistent visited state |
| `constexpr void bfs_visit(Graph const &g, vertex_id source, Step step)` | eager `step(v, parent, depth)` per discovery |
| `constexpr void bfs_visit(Graph const &g, vertex_id source, Marker &marker, Step step)` | eager with persistent marks |
| `constexpr void bfs_visit(Graph const &g, Sources const &sources, Step step)` | eager multiple sources |
| `constexpr void bfs_visit(Graph const &g, Sources const &sources, Marker &marker, Step step)` | eager multiple sources and marks |

All graph parameters satisfy the resumable-scan constraints described above;
`Sources const` satisfies `vertex_range`; `Marker` satisfies `visit_marker`;
`Step` satisfies `bfs_step_event` (`std::invocable<Step &, vertex_id, vertex_id, uint32>`).
The views may also be constructed directly with these source/marker combinations.
`begin() const` creates a cursor and `end() const` returns `std::default_sentinel_t`.
Their iterators have `value_type = visit`, `difference_type = std::ptrdiff_t`,
`iterator_concept = std::input_iterator_tag`, read-only dereference, prefix
increment, void postfix increment and sentinel equality. When marker storage permits default construction, the unbound cursor must
be assigned a bound cursor prior to use. Copying a cursor does not create a
multipass range.

### DFS events and folds

\include graph/callbacks.cpp

`skip` accepts any arguments and does nothing. Callback concepts are
`dfs_enter_event<F>` for `enter(v, parent, parent_edge, depth)`,
`dfs_seen_edge_event<F>` for `seen_edge(u, k, v)` and `dfs_exit_event<F>` for
`exit(v, parent)`. Their coordinates are `vertex_id` / `uint32`.
A root enters with `parent_edge == none`; every tree edge causes an enter and
every non-tree edge causes a seen-edge event. `k` is the iteration position in
u's row, including for forward-only adjacency. The reverse of an undirected
tree edge is a seen edge. A completed subtree causes one exit.

`dfs_visit` returns `void` with four explicit-callback forms:

```cpp
constexpr void dfs_visit(Graph const &g, vertex_id source, Enter enter, SeenEdge seen_edge, Exit exit);
constexpr void dfs_visit(Graph const &g, vertex_id source, Marker &marker, Enter enter, SeenEdge seen_edge, Exit exit);
constexpr void dfs_visit(Graph const &g, Enter enter, SeenEdge seen_edge, Exit exit);
constexpr void dfs_visit(Graph const &g, Marker &marker, Enter enter, SeenEdge seen_edge, Exit exit);
```

The same four forms accept `event_set<Enter, SeenEdge, Exit> bundle` in place
of the three callbacks. The public bundle members are `enter`, `seen_edge`,
`exit`. `constexpr auto events(Events &&...event)` infers roles for one to three callables,
using signature arity rather than argument position, and fills missing roles
with `skip`. It rejects a callable matching no role, multiple roles, or a role
already occupied. Capturing lambdas, const and mutable callables and
`std::reference_wrapper` are supported. Overloads forwarding callbacks into another overload require construction from
rvalues; the direct caller-marker overloads also accept immovable callback
prvalues without forwarding. Whole-graph DFS roots each unvisited
vertex in ascending id order.

```cpp
constexpr auto dfs_fold(Graph const &g, vertex_id root, Init init, Fold fold, Finalize finalize)
    -> std::invoke_result_t<Init &, visit>;
constexpr auto dfs_fold(Graph const &g, vertex_id root, Init init, Fold fold);
```

`dfs_fold_callbacks<Graph, Init, Fold, Finalize>` requires a movable accumulator
`Acc` returned by `init(visit)`, `fold(Acc &, Acc, visit parent, visit child,
Edge const &)`, and `finalize(Acc &, visit)`. The edge value is copyable,
default-initializable and constructible from a scanned edge reference. The
root's finalized accumulator is returned; non-tree edges are ignored. Without
`finalize`, sealing a node does nothing. Structural facts arrive through
`visit`, mutable computed state through `Acc`, and external per-node data can
be indexed by vertex id. The connecting edge is an element, so weighted or
forward-only adjacency works without retrieving an edge by index.

### Complexity, preconditions and lifetimes

With O(1) row access and O(1) callbacks/marker operations, a complete single-root
traversal scans O(r + e) reached vertices and their outgoing edges. An owned
marker adds O(n / 64 + 1) initialization; whole-graph DFS is O(n + m).
For K supplied sources, duplicates included, multi-source BFS takes
O(K + r + e) traversal time. Its view stores O(K + r) copied source ids,
pending seeds and queue records. Eager BFS stores O(r) queue ids with an
external marker, or reserves O(n) queue capacity with an owned marker.
Single-source BFS holds O(r) queue records; DFS holds O(depth) frames. `dfs_fold` also has
O(depth) live accumulators **plus** its O(n / 64 + 1) dense marker. Generating a
row repeatedly adds its generation cost; callbacks add their own cost.
`operator++` scans only until the next discovery, so a view permits early
exit without expanding an entire adjacency row at each yield. One increment
can still scan many edges; O(1) worst-case increment is not promised.
An immediate break after visit k performs no further increment. Fully consuming
`bfs(g, s) | std::views::take(k)` increments after the k-th yielded visit,
scanning until one additional vertex is discovered and marked, or the search
is exhausted. With k = 0, `begin()` still initializes the search and may mark
a source. The same increment caveat applies to DFS views.

Asserted: sources/roots are valid ids, vertex count is at most `none`, and
indexed row lengths fit `uint32`. Every row must have at most `none` edges,
including forward-only rows. Targets and source-range values must already fit
the graph's domain; source-range conversion narrows. Markers must cover every
vertex they may see. Graphs, their adjacency storage and external markers must
outlive iteration; callbacks must not mutate adjacency order or invalidate rows.
A yielded `visit const &` lasts only until cursor increment or destruction.
A cursor may outlive its originating view object,
but it still borrows the graph and any external marker.

## Markers

`visit_marker<M>` checks `m.try_visit(vertex_id) -> convertible_to<bool>`.
The operation marks a vertex and returns true exactly on its first visit in
the current run. Already-marked sources and targets are skipped; an external
marker is used as-is and retains all marks. Unmarked vertices behind an
already-marked vertex are not explored through it. Nearest-source distances
refer to the resulting search when marks are prepopulated.

| API | Storage and cost |
|---|---|
| `constexpr explicit dense_marker(std::size_t n)` | one bit per vertex; O(n / 64 + 1) initialization |
| `constexpr bool try_visit(vertex_id v)` / `constexpr bool visited(vertex_id v) const` | O(1), asserted `v < n` |
| `constexpr void reset()` | clear dense bits, O(n / 64 + 1) |
| `constexpr explicit epoch_marker(std::size_t n)` | one `uint32` stamp per vertex; O(n) initialization |
| epoch `try_visit` / `visited` | O(1), asserted `v < n` |
| `constexpr void next_run()` | O(1), with an O(n) clear on epoch wrap |
| `hash_marker` with `bool try_visit(vertex_id v)`, `bool visited(vertex_id v) const`, `void clear()` | O(marked vertices) storage; expected O(1) probes, amortized for insertion, worst-case O(marked vertices); clear O(capacity) |

Dense and epoch marker operations constant-evaluate. Hash marks use a runtime
hash table and are not constexpr. An epoch marker is useful for many short searches over a large graph; a hash
marker can explore a tiny reachable region of an implicit domain with up to
`none` states. Two lockstep BFS cursors with independent epoch markers can
stop when the smaller component exhausts, scanning only that side's reachable
vertices and the edges required by the interleaving.

## Vertex groups and partitions

`vertex_partition` owns public `std::vector<uint32> component`, with component
ids in `[0, count())`. The returning algorithm defines whether ids carry an
order beyond density. Hand-built partitions may leave component ids unused.

| Signature | Result | Complexity |
|---|---|---|
| `constexpr vertex_partition(std::vector<uint32> component_ids, std::size_t component_count)` | owns ids and stores the count | O(1), O(n) with assertions |
| `constexpr std::size_t count() const noexcept` | component count | O(1) |
| `constexpr bool same(vertex_id u, vertex_id v) const` | equal component ids | O(1) |
| `constexpr std::vector<std::vector<vertex_id>> groups() const` | nested groups in ascending component order | O(n + count()) time/storage |
| `constexpr vertex_groups grouped() const` | flat groups in ascending component order | O(n + count()) time/storage |

Both forms list vertices ascending within each group. `vertex_groups` owns
public `std::vector<uint32> offsets{0}` and `std::vector<vertex_id> members`;
`constexpr std::size_t count() const noexcept` gives the number of slices and
`constexpr std::span<vertex_id const> operator[](std::size_t g) const` reads a
slice in O(1). Flat groups use at most two allocations, independent of group
count. Nested groups may allocate once per nonempty group.

Asserted for partitions: vertex/component counts fit `uint32`, `count() + 1`
fits `size_t`, and each id is below `count()`. `same` asserts valid vertices;
flat indexing asserts a valid group. Public array mutations must preserve the
invariants: nonempty nondecreasing offsets starting at zero and ending at
`members.size()`, and valid component ids. Spans are invalidated when their
members storage is destroyed or reallocated. Results of `groups()` and
`grouped()` own independent storage.

## Measured performance

The tables use method names matching the contenders. Timing columns give
milliseconds, plain `-O2` / `-O2 -march=native`;
minimum of three run means, 25 samples per row. Spread is the slowest
run divided by the fastest run for the corresponding flag set. Benchmarks are compiled in CI
and measured separately.

### CSR construction

[benchmarks/graph/csr.cpp](../benchmarks/graph/csr.cpp) races library in-place
row-offset cursors against a separate cursor copy, bucket indices plus gather,
and nested vectors. Construction alone is timed, directed or doubled.
Intel Xeon Platinum 8358 @ 2.60 GHz, x86-64 Linux; GCC 15.2.0,
RelWithDebInfo, `-O2 -g -DNDEBUG -std=c++23`, also `-march=native`, 2026-10-15.

| Shape | Direction | Contender | ms plain / native | spread plain / native |
|---|---|---|---|---|
| n=1e6 m=2e6 `arc<>` | directed | library | 22.6 / 22.8 | 1.07 / 1.07 |
| same | directed | by cursor copy | 22.4 / 23.7 | 1.10 / 1.02 |
| same | directed | by bucket gather | 35.8 / 36.6 | 1.06 / 1.02 |
| same | undirected | library | 56.4 / 53.7 | 1.04 / 1.07 |
| same | undirected | by cursor copy | 54.9 / 54.7 | 1.06 / 1.05 |
| same | undirected | by bucket gather | 84.7 / 86.5 | 1.05 / 1.00 |
| same | directed | by nested vectors | 301 / 287 | 1.06 / 1.07 |
| same | undirected | by nested vectors | 535 / 535 | 1.01 / 1.05 |
| n=1e6 m=2e6 `arc<int64>` | directed | library | 43.4 / 43.3 | 1.02 / 1.03 |
| same | directed | by cursor copy | 44.2 / 43.7 | 1.02 / 1.03 |
| same | directed | by bucket gather | 51.8 / 52.2 | 1.04 / 1.04 |
| same | undirected | library | 112 / 111 | 1.01 / 1.01 |
| same | undirected | by cursor copy | 112 / 112 | 1.00 / 1.01 |
| same | undirected | by bucket gather | 143 / 147 | 1.01 / 1.01 |
| same | directed | by nested vectors | 345 / 335 | 1.01 / 1.04 |
| same | undirected | by nested vectors | 660 / 649 | 1.00 / 1.05 |
| n=1e7 m=1e7 `arc<>` | directed | library | 399 / 401 | 1.01 / 1.00 |
| same | directed | by cursor copy | 421 / 421 | 1.01 / 1.00 |
| same | directed | by bucket gather | 542 / 552 | 1.01 / 1.01 |
| same | undirected | library | 823 / 824 | 1.03 / 1.01 |
| same | undirected | by cursor copy | 845 / 845 | 1.01 / 1.00 |
| same | undirected | by bucket gather | 1201 / 1209 | 1.03 / 1.01 |

The cursor copy is close at n=1e6; at n=1e7 its extra n-entry allocation and
copy cost more. Bucket gather writes and rereads an m-entry index array absent
from direct scatter. These contenders remain in
`experimental/atl/experimental/graph/csr_variants.hpp` as
`csr_by_cursor_copy` and `csr_by_bucket_gather`, cross-checked against nested
vector rows, including their stable order.
### Traversal throughput

[benchmarks/graph/traversal.cpp](../benchmarks/graph/traversal.cpp) compares
complete traversal loops, lazy yields, eager event delivery, allocated seed
lists, full visit-record queues and byte/bit markers. Setup builds graphs
outside timing. Intel Xeon Platinum 8358 @ 2.60 GHz, x86-64 Linux;
GCC 15.2.0, RelWithDebInfo, `-O2 -g -DNDEBUG -std=c++23`, also
`-march=native`, 2026-10-15.

| Shape | Contender | ms plain / native | spread plain / native |
|---|---|---|---|
| n=1e6 m=4e6 | BFS by raw arrays | 61.0 / 60.5 | 1.04 / 1.03 |
| same | BFS library view, CSR | 61.0 / 60.3 | 1.13 / 1.01 |
| same | BFS by allocated seeds, CSR | 63.9 / 61.6 | 1.03 / 1.06 |
| same | bfs_visit library | 54.4 / 50.8 | 1.15 / 1.15 |
| same | bfs_visit library, caller marker, no reserve | 62.6 / 56.7 | 1.15 / 1.15 |
| same | bfs_visit by record queue | 60.0 / 55.5 | 1.09 / 1.11 |
| same | BFS library view, nested vectors | 104 / 103 | 1.07 / 1.05 |
| same | DFS by raw arrays | 122 / 126 | 1.11 / 1.07 |
| same | DFS library view | 140 / 143 | 1.13 / 1.05 |
| same | dfs_visit Euler intervals | 157 / 161 | 1.10 / 1.07 |
| 2000x2000 grid | BFS implicit | 96.2 / 95.6 | 1.00 / 1.01 |
| same | BFS materialized CSR | 124 / 119 | 1.00 / 1.02 |
| path n=1e6 | BFS by raw arrays | 7.3 / 5.5 | 1.01 / 1.00 |
| same | bfs_visit library | 4.9 / 5.0 | 1.00 / 1.00 |
| same | bfs_visit library, caller marker, no reserve | 5.3 / 6.8 | 1.00 / 1.00 |
| same | bfs_visit by record queue | 7.7 / 6.8 | 1.01 / 1.01 |
| same | BFS library view | 8.4 / 8.5 | 1.00 / 1.01 |
| random n=4e6 m=1.6e7 | BFS by raw arrays | 495 / 407 | 1.01 / 1.01 |
| same | bfs_visit library | 380 / 359 | 1.00 / 1.01 |
| same | bfs_visit library, caller marker, no reserve | 389 / 415 | 1.03 / 1.01 |
| same | bfs_visit by record queue | 448 / 410 | 1.00 / 1.01 |
| same | BFS library view | 444 / 405 | 1.03 / 1.02 |
| 1e6 runs, components of 1-4 | BFS library view | 56.7 / 57.8 | 1.04 / 1.03 |
| same | BFS by allocated seeds | 84.5 / 86.3 | 1.01 / 1.00 |
| same | bfs_visit library | 55.8 / 54.6 | 1.00 / 1.03 |
| same | bfs_visit by record queue | 53.6 / 57.1 | 1.01 / 1.02 |
| same | BFS by reused queue | 24.7 / 25.3 | 1.01 / 1.02 |
| n=1e5 m=4e5 | bfs_visit library bit marker | 3.0 / 3.9 | 1.00 / 1.01 |
| same | bfs_visit by byte marker | 4.0 / 4.3 | 1.01 / 1.00 |
| same | DFS library bit marker | 4.9 / 4.8 | 1.01 / 1.01 |
| same | DFS by byte marker | 4.9 / 4.9 | 1.01 / 1.01 |
| n=1e6 m=4e6 | bfs_visit library bit marker | 52.1 / 62.1 | 1.11 / 1.05 |
| same | bfs_visit by byte marker | 68.5 / 75.3 | 1.12 / 1.02 |
| same | DFS library bit marker | 136 / 138 | 1.02 / 1.05 |
| same | DFS by byte marker | 140 / 142 | 1.02 / 1.09 |
| n=4e6 m=1.6e7 | bfs_visit library bit marker | 379 / 413 | 1.02 / 1.01 |
| same | bfs_visit by byte marker | 462 / 477 | 1.02 / 1.04 |
| same | DFS library bit marker | 998 / 979 | 1.02 / 1.01 |
| same | DFS by byte marker | 981 / 994 | 1.02 / 1.02 |

Eager BFS holds bare vertex ids and tracks level boundaries, reducing queue
traffic. It reserves the queue for an owned marker; caller-owned markers avoid
an O(n) reserve when exploring tiny components. A single-source library view
holds its seed inline. Use lazy views for early exit and lockstep searches;
use eager visitors when consuming the entire traversal. Bit markers reduce the
working set; the DFS marker results show no consistent advantage. Generated
grid neighbors avoid loading explicit edges.
### Partition materialization

[benchmarks/graph/partition.cpp](../benchmarks/graph/partition.cpp) builds and
scans every group; nested vectors trade allocations against flat offsets and
members. Intel Xeon Platinum 8280 @ 2.70 GHz, x86-64 Linux; GCC 15.2.0,
RelWithDebInfo, `-O2 -g -DNDEBUG -std=c++23`, also `-march=native`, 2026-10-23.

| n=1e6 shape | `groups()` ms plain / native | `grouped()` ms plain / native | nested spread plain / native | flat spread plain / native |
|---|---|---|---|---|
| all singletons | 198 / 196 | 22.5 / 23.0 | 1.01 / 1.03 | 1.04 / 1.06 |
| components of 2-4 | 81.9 / 84.3 | 12.8 / 13.4 | 1.06 / 1.04 | 1.01 / 1.02 |
| one giant component | 3.5 / 3.3 | 3.7 / 3.5 | 1.01 / 1.03 | 1.01 / 1.01 |
| sqrt(n) components of sqrt(n) | 3.6 / 3.3 | 3.4 / 3.2 | 1.04 / 1.05 | 1.01 / 1.03 |

Flat groups help most when groups are numerous and tiny; either form is close
for a single large group.
## Pitfalls and notes

A view refers to its graph, including a temporary graph. The rvalue-graph
`bfs(g, s)` and `dfs(g, s)` calls are supported: in C++23, a range-for
over such a call extends the temporary graph's lifetime through the loop.
Storing the returned view beyond that full-expression leaves a dangling
graph reference. Keep a named graph alive for stored views and their cursors.
External markers preserve state; use `reset`, `next_run` or `clear` when
starting an independent traversal. Previsited vertices cut off exploration
through those vertices. Iterator increments can invalidate a retained visit
reference. CSR uses 32-bit offsets even on 64-bit systems, and a doubled input
must fit that stored-edge limit. On 32-bit systems the terminal row offset also
requires a representable `n + 1`. Forward-only adjacency retains stable
iterators; generated rows are owned and resumed by index.

## Graph pages

- [Graph components](graph-components.md): strongly connected, two-edge-connected and biconnected components.
- [Maximum flow](max-flow.md): capacity flow, cut certificates and lower-bound reductions.
- [Min-cost flow](min-cost-flow.md): costs, supplies, lower bounds and dual certificates.
- [Dense assignment](assignment.md): rectangular matching with exact dual potentials.
- [Bipartite matching](bipartite-matching.md): matching, vertex sets, edge roles and DAG covers.
- [Matrices and matrix-tree counts](matrix.md): spanning trees and directed arborescences.
- [Lowest common ancestors](lca.md).
- [Functional graphs](functional-graphs.md).
- [Virtual trees](virtual-tree.md).
- [Centroid decomposition](centroid-decomposition.md).
- [Offline connectivity](time-divide-and-conquer.md).

## How it works

Shape access prefers a graph's member functions, then const range indexing;
edge access prefers integral targets, explicit traits, then named members.
CSR counts each source degree, prefix-sums the offsets and scatters in input
order. During scatter the offsets serve as write cursors; shifting them right
restores row starts without allocating a separate cursor array.

BFS queues discoveries and resumes a paused edge scan at each yield. All seeds
are delivered at depth zero prior to expansion. Eager BFS queues ids and gets
depth from level boundaries. DFS frames retain only a vertex, depth and resume
position; rows are reloaded as a span or reference when a frame resumes.
A fold adds an accumulator and its incoming edge to each frame, merging the
child into the parent as it completes. Dense bits or generation stamps control
discovery without recursive calls. Partition grouping counts component sizes,
prefix-sums starts and fills members in ascending vertex order.

## See also

- [Maximum flow](max-flow.md) for capacitated networks and lower bounds.
- [Min-cost flow](min-cost-flow.md) for costed capacitated networks.
- [Assignment](assignment.md) for dense weighted matching.
- [Bipartite matching](bipartite-matching.md) for unweighted matching and DAG covers.
- [Persistent heaps](persistent-heap.md) for experimental k shortest walks.
- [Matrices](matrix.md) for matrix-tree counting.

- [Functional graphs](functional-graphs.md) for repeated successors and ordered edge folds.

- [Disjoint sets](dsu.md) for incremental component merging.
- [Time divide and conquer](time-divide-and-conquer.md) for offline add/remove schedules.
- [128-bit integers](int128.md) for portable wide numeric fields.
