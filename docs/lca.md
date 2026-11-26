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

LCA query throughput:

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| contender | random recursive | path | star | caterpillar | binary tree |
|---|---|---|---|---|---|
| library | 6.4 / 4.8 | 7.3 / 7.3 | 8.7 / 6.7 | 9.6 / 8.4 | 6.4 / 4.8 |
| by binary lifting | 21.5 / 17.2 | 71.2 / 58 | 2.1 / 1.6 | 260 / 236 | 17.7 / 14 |
| by head jumping | 41.7 / 39.2 | 1.5 / 1.1 | 4.8 / 3.7 | 7.4 / 7 | 81.7 / 80.2 |
| by Euler sparse table | 9.3 / 7.9 | 7.6 / 5.2 | 10.8 / 10.2 | 9.7 / 7.6 | 9.2 / 8.2 |
| by preorder sparse table | 5.7 / 4.3 | 6.3 / 4.8 | 5.5 / 4.5 | 6.3 / 4.8 | 5.7 / 4.5 |
| by Farach-Colton-Bender | 11 / 9.2 | 10.3 / 8.9 | 12.5 / 9.7 | 11 / 9.2 | 10.8 / 8.8 |
| by Euler direct RMQ | 14 / 9 | 11.9 / 9.5 | 18.6 / 10.3 | 15.6 / 9.5 | 13.8 / 9 |
| by preorder direct RMQ | 12 / 8.9 | 12.9 / 10.6 | 12.6 / 9 | 13.1 / 10.5 | 12.3 / 8.9 |
| by adjacency preorder | 6.7 / 4.8 | 7.4 / 7 | 8.9 / 6.9 | 10 / 8.5 | 6.7 / 4.7 |

Build latency:

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| contender | random recursive | path | star | caterpillar | binary tree |
|---|---|---|---|---|---|
| library graph | 37.9 / 35.5 | 5 / 4.6 | 4.9 / 3.6 | 5.5 / 5.2 | 5.7 / 4.8 |
| by binary lifting | 30.4 / 28.9 | 29.1 / 25.6 | 3.8 / 3 | 27.1 / 23.6 | 5 / 4 |
| by Euler sparse table | 166 / 153 | 115 / 99 | 114 / 99.1 | 115 / 98.8 | 117 / 100 |
| by preorder sparse table | 74.8 / 73.5 | 30 / 25.5 | 28.6 / 24.6 | 29.6 / 25.4 | 31.4 / 26.7 |
| by Farach-Colton-Bender | 68 / 61.6 | 8.5 / 6.9 | 7.6 / 6.1 | 8.2 / 6.7 | 9.7 / 7.9 |
| by Euler direct RMQ | 66.2 / 54.6 | 7.6 / 6.5 | 7.2 / 5.6 | 7.4 / 6.2 | 9.6 / 7.7 |
| by preorder direct RMQ | 47.4 / 44 | 5.3 / 4.7 | 4.2 / 3.5 | 4.8 / 4.2 | 6.1 / 4.9 |
| by adjacency preorder | 59.4 / 55.7 | 5.6 / 4.9 | 4.6 / 3.6 | 5.4 / 4.7 | 7.1 / 5.4 |

The library uses interleaved label/ascendant pairs and keeps chains for
ancestor and path operations. The preorder sparse table exchanges more memory
for fast queries; head jumping is strong on paths but costly on binary trees.
The library loses all five query shapes to preorder sparse tables; on
random trees the times are 6.4/4.8 versus 5.7/4.3 ms. Its random-tree
graph build is faster than adjacency preorder (37.9/35.5 versus
59.4/55.7 ms). Build plus query totals depend on shape and query count.

Additional measured rows (same source, machine, compiler, flags and date):

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| operation / shape | library plain / native ms | contender plain / native ms |
|---|---|---|
| graph build, random recursive | 37.9 / 35.5 | by children DFS 63 / 59.9 |
| graph build, singletons | 3.5 / 2.4 | by children DFS 23.2 / 18.3 |
| graph build, 1000 trees | 36.1 / 33 | by children DFS 61.9 / 56.8 |
| graph build, n=4e6 | 714 / 681 | by children DFS 1330 / 1290 |
| label layout queries, random recursive | 6.4 / 4.8 | by separate labels 7.3 / 4.8; by label pairs 6.8 / 4.3; by label depth triples 7.9 / 5.3 |
| label layout queries, n=4e6 | 18.6 / 10.5 | by separate labels 20.6 / 14.1; by label pairs 17.4 / 10.7; by label depth triples 28.5 / 19.5 |
| child CSR, random recursive | by counting cursors 1.4 / 1.1 | by bucket gather 1.3 / 1.1 |
| child CSR, n=4e6 | by counting cursors 38.9 / 33.1 | by bucket gather 24.9 / 24.1 |
| offline batch, random recursive | 88.2 / 82.7 | by query buckets 90.2 / 80.6 |
| offline batch, star | 20.9 / 19.1 | by query buckets 21 / 19.1 |
| offline batch, binary | 43.8 / 40.5 | by query buckets 43.8 / 40.1 |
| offline batch, singletons | 15.8 / 13 | by query buckets 16 / 12.9 |
| offline batch, 1000 trees | 67 / 62.8 | by query buckets 66.5 / 61.1 |
| offline batch, n=4e6 | 750 / 779 | by query buckets 763 / 779 |

The Euler sparse-table and preorder direct-RMQ query contenders include
rows that vary between runs (spreads up to 1.53x), as does the large
label-pair query row (1.43x). Those minima alone do not establish stable
rankings.

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
