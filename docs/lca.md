# `atl::graph::lca` and `atl::graph::offline_lca`

Build a static forest once to answer lowest-common-ancestor, ancestor, distance
and path queries. Heavy-first preorder lets the same structure drive subtree
and path operations on a segment tree. Use a dynamic forest when topology changes.

```cpp
#include <atl/graph/lca.hpp>
```

\include lca/quick.cpp

## API

The class `lca` accepts three constructor forms. `vertex_id` and `none` come
from the [graph vocabulary](graph.md); `path_segment_event<F>` checks
`std::invocable<F &, uint32, uint32, bool>`.

| Signature | Meaning |
|---|---|
| `template<adjacency_graph Graph> constexpr lca(Graph const &g, vertex_id root)` | root that component at `root`, every other component at its smallest id |
| `template<adjacency_graph Graph> constexpr explicit lca(Graph const &g)` | root every component at its smallest id |
| `template<typename Parents> requires vertex_range<Parents const> && std::convertible_to<std::ranges::range_reference_t<Parents const>, vertex_id> constexpr explicit lca(Parents const &parents)` | parent array; `none` or self marks a root |
| `constexpr std::size_t size() const` | indexed vertices |
| `constexpr vertex_id operator()(vertex_id u, vertex_id v) const` | LCA; `none` across components |
| `constexpr uint32 depth(vertex_id v) const` | hop distance from its root |
| `constexpr vertex_id parent(vertex_id v) const` | parent, or `none` at a root |
| `constexpr vertex_id root(vertex_id v) const` | component root |
| `constexpr uint32 distance(vertex_id u, vertex_id v) const` | path length in edges; `none` across components |
| `constexpr bool is_ancestor(vertex_id a, vertex_id b) const` | whether `a` lies on the root-to-`b` path, including `b` |
| `constexpr vertex_id ancestor(vertex_id v, uint32 k) const` | k-th ancestor, starting at v; `none` past the root |
| `constexpr vertex_id path_vertex(vertex_id u, vertex_id v, uint32 k) const` | k-th vertex from u toward v; `none` past v or across components |
| `template<path_segment_event F> constexpr void path(vertex_id u, vertex_id v, F f, bool includes_top = true) const` | emit nonempty `[l,r)` segments in path order, with a reverse-direction flag |
| `constexpr uint32 entry_time(vertex_id v) const` | position in heavy-first preorder |
| `constexpr uint32 exit_time(vertex_id v) const` | exclusive subtree end |
| `constexpr std::vector<uint32> const &order() const` | borrowed preorder, invalidated by assignment or destruction |

`path` includes every path vertex once. Set `includes_top = false` to omit
the LCA when storing edge values at child positions. For noncommutative folds,
store both segment directions and select the reverse fold when the flag is true.

\include lca/path-sum.cpp

For `Graph` satisfying traversal's resumable `detail::scannable_graph` and
`Queries` a const sized forward range of vertex pairs:

```cpp
template<detail::scannable_graph Graph, vertex_pair_range Queries>
    requires vertex_pair_range<Queries const>
                && requires(std::remove_reference_t<std::ranges::range_reference_t<Queries const>> const &q) {
                    { std::get<0>(q) } -> std::convertible_to<vertex_id>;
                    { std::get<1>(q) } -> std::convertible_to<vertex_id>;
                }
constexpr std::vector<vertex_id> offline_lca(Graph const &g, vertex_id root, Queries const &queries);
template<detail::scannable_graph Graph, vertex_pair_range Queries>
    requires vertex_pair_range<Queries const>
                && requires(std::remove_reference_t<std::ranges::range_reference_t<Queries const>> const &q) {
                    { std::get<0>(q) } -> std::convertible_to<vertex_id>;
                    { std::get<1>(q) } -> std::convertible_to<vertex_id>;
                }
constexpr std::vector<vertex_id> offline_lca(Graph const &g, Queries const &queries);
```

Answers retain query order. The rooted overload returns `none` for any query
with an endpoint unreachable from root, even when both belong to another
component. The rootless form agrees with rootless `lca`, including cross-tree
`none` answers. The query range must also be const-iterable.

## Complexity

`lca` builds in O(n) time and storage on a forest. LCA, depth, parent, root,
distance, ancestry and preorder access take O(1); ancestor, path_vertex and
path take O(log n), with O(log n) segments. Graph-access and callback costs
are additional. `offline_lca` takes O((n + q) alpha(n + q)) time and O(n + q)
space. Heap buffers replace per-vertex recursion; `path` uses a fixed 33-entry
segment buffer. Allocation failures propagate.

## Preconditions

Asserted under `ATL_ASSERT`: a nonempty input, n < 2^31, valid parent ids and
an acyclic parent array; query vertices below size(); path endpoints in the
same component. Offline LCA allows an empty forest in its rootless form;
n <= none, n + 1 fits size_t, q <= none/2, and endpoints are valid. The rooted
offline root must be valid.

Documented only: graph inputs are symmetric forests with valid targets and
stable adjacency during traversal. Parent ids must fit vertex_id when converted; signed -1 also denotes none.
Do not mutate the graph from a callback. Heavy-child ties are unspecified;
depend on subtree intervals and path order, not one constructor's exact preorder.

## Measured performance

The alternative engines are defined in
[`lca_variants.hpp`](../experimental/atl/experimental/graph/lca_variants.hpp).

[benchmarks/graph/lca.cpp](../benchmarks/graph/lca.cpp) measures builds and
throughput of 500000 independent queries on 500000 vertices, plus forests
and a four-million-vertex tree. Contenders change ancestor jumping, RMQ input,
RMQ method, graph-build passes or label layout. Binary lifting is the certifying
benchmark reference; tests also use a parent-walk oracle.

Intel Xeon Platinum 8280 @ 2.70 GHz, x86-64 Linux; GCC 15.2.0,
RelWithDebInfo, assertions off, 2026-10-15. Flags `-O2 -g -DNDEBUG -std=c++23`,
also with `-march=native`. Cells are minimum milliseconds over three runs,
plain / native, each run using 25 samples.

LCA query throughput:

| contender | random recursive | path | star | caterpillar | binary tree |
|---|---|---|---|---|---|
| library | 13.3 / 8.9 | 12 / 10.7 | 20.9 / 17.9 | 21.6 / 18.3 | 13.2 / 8.8 |
| by binary lifting | 37 / 37 | 108 / 98.4 | 4.2 / 4.2 | 434 / 403 | 29.4 / 29.7 |
| by head jumping | 67.8 / 68.6 | 4 / 4 | 7.7 / 7.6 | 10.8 / 10.7 | 109 / 110 |
| by Euler sparse table | 16.4 / 15.5 | 17 / 14.6 | 23 / 20.7 | 19.5 / 17.9 | 17.9 / 16.2 |
| by preorder sparse table | 10.1 / 9.2 | 13.9 / 12.2 | 10 / 9.1 | 13.8 / 12.4 | 10.6 / 9.9 |
| by Farach-Colton-Bender | 24.9 / 24.2 | 24.8 / 23.9 | 32.8 / 31.1 | 28.8 / 27.9 | 25 / 24.4 |
| by Euler direct RMQ | 27 / 25.4 | 28.3 / 25.9 | 36.5 / 34.2 | 32.5 / 30.6 | 27.2 / 25.6 |
| by preorder direct RMQ | 21.8 / 20 | 28.5 / 27 | 22.4 / 20.7 | 27.8 / 26.9 | 22.2 / 21 |
| by adjacency preorder | 13.2 / 9.3 | 12.6 / 10.9 | 22.7 / 19.9 | 24.2 / 20.8 | 12.9 / 9.2 |

Build latency:

| contender | random recursive | path | star | caterpillar | binary tree |
|---|---|---|---|---|---|
| library graph | 136 / 133 | 10.3 / 9.3 | 10.3 / 9.8 | 9.6 / 8.9 | 12.2 / 11.5 |
| by binary lifting | 62.8 / 60.5 | 34.5 / 34.1 | 7.9 / 7.5 | 32.6 / 32.3 | 9.5 / 9.1 |
| by Euler sparse table | 206 / 204 | 131 / 132 | 127 / 128 | 130 / 131 | 132 / 132 |
| by preorder sparse table | 104 / 101 | 37.1 / 36.6 | 33.3 / 32.2 | 36.2 / 35.7 | 38.2 / 36 |
| by Farach-Colton-Bender | 90.8 / 90.1 | 14.5 / 13.3 | 12.6 / 11.9 | 14.2 / 13.2 | 16.3 / 14.9 |
| by Euler direct RMQ | 89.3 / 90 | 13.4 / 13 | 12.2 / 11.3 | 13.3 / 12.6 | 16.1 / 15 |
| by preorder direct RMQ | 73.5 / 72.1 | 9.4 / 9.3 | 7.2 / 6.8 | 8.4 / 8.5 | 10.1 / 10 |
| by adjacency preorder | 86.9 / 81.8 | 10.9 / 10.4 | 8.1 / 7.2 | 9.6 / 8.9 | 19.1 / 17.6 |

The library uses interleaved label/ascendant pairs and keeps chains for
ancestor and path operations. The preorder sparse table exchanges more memory
for fast queries; head jumping is strong on paths but costly on binary trees.
Build plus query totals depend on shape; the random-tree graph build has a
substantial cost beyond the minimal adjacency-preorder contender.

Additional measured rows (same source, machine, compiler, flags and date):

| operation / shape | library plain / native ms | contender plain / native ms |
|---|---|---|
| graph build, random recursive | 136 / 133 | by children DFS 123 / 123 |
| graph build, singletons | 7.3 / 6.9 | by children DFS 32.2 / 32.2 |
| graph build, 1000 trees | 118 / 121 | by children DFS 120 / 124 |
| graph build, n=4e6 | 3264 / 3255 | by children DFS 3311 / 3372 |
| label layout queries, random recursive | 13.3 / 8.9 | by separate labels 12.6 / 9.3; by label pairs 10.2 / 8.6; by label depth triples 15.4 / 10.4 |
| label layout queries, n=4e6 | 30.0 / 20.1 | by separate labels 33.3 / 27.4; by label pairs 24.7 / 19.2; by label depth triples 42.7 / 25.3 |
| child CSR, random recursive | by counting cursors 4.3 / 4.2 | by bucket gather 3.9 / 3.9 |
| child CSR, n=4e6 | by counting cursors 245 / 230 | by bucket gather 207 / 204 |
| offline batch, random recursive | 199 / 199 | by query buckets 199 / 198 |
| offline batch, star | 71.2 / 70.3 | by query buckets 71.5 / 70.9 |
| offline batch, binary | 89.7 / 90.6 | by query buckets 89.8 / 90.6 |
| offline batch, singletons | 50.6 / 51.9 | by query buckets 50.1 / 51.1 |
| offline batch, 1000 trees | 130 / 128 | by query buckets 128 / 129 |
| offline batch, n=4e6 | 1429 / 1423 | by query buckets 1437 / 1429 |

## Pitfalls and notes

`none` is a sentinel, never a query vertex. A cross-component distance has the
same sentinel value; do not add it to a numeric path length. The rooted offline
overload visits only one component, while a rooted `lca` indexes every component.
Depth and finite distance are at most n - 1; entry times are below n and
exit times are at most n. Topology changes require reconstruction. A borrowed order reference lasts only
as long as its storage owner. Heavy-first preorder also supports tree differences
and offline distinct-subtree counting.

## How it works

Graph construction roots each component with one BFS. A reverse sweep computes
subtree sizes and heavy children; a forward sweep places heavy-first preorder
slots and chain heads without child lists. Parent-array construction instead
counts children into CSR, computes depths by BFS and emits preorder with an
explicit DFS stack. Ties need not produce identical orders between these builds.

For each subtree interval, the Schieber-Vishkin label is the one-indexed position
with the most trailing zero bits. Labels partition the tree into paths;
ascendant masks collect the labels on root paths. Intersecting masks above the
endpoint split bit identifies the common label path. Each endpoint enters it
through a path head, and the shallower entry is the answer. The n < 2^31 bound
keeps all shifts below 32. Heavy chains halve remaining subtree size at each
light edge, giving O(log n) ancestor jumps and path segments.

Offline LCA uses iterative DFS and directed union links: a finished subtree's
representative becomes its unfinished parent. The second entered endpoint can
then find the lowest unfinished ancestor of the first. Tree marks distinguish
components. Query incidences are bucketed by endpoint without keeping a query
index after the call.

References: Schieber and Vishkin (1988), *On Finding Lowest Common Ancestors:
Simplification and Parallelization*; Tarjan (1979), *Applications of Path
Compression on Balanced Trees*.

## See also

- [Graph vocabulary and traversal](graph.md).
- [Virtual trees](virtual-tree.md).
- [Centroid decomposition](centroid-decomposition.md).
- [Segment tree](segment-tree.md) and [lazy segment tree](lazy-segment-tree.md).
