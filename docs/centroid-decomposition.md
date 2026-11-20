# `atl::graph::centroid_visit`, `centroid_bisect` and `ternarization_of`

These static-tree tools support aggregation over paths. Ranking separators
expose all branches of a component; balanced bisection exposes two groups for
one balanced combine. Ternarization preserves real-vertex distances while
bounding degree for consumers that need binary fan-in.

```cpp
#include <atl/graph/centroid_decomposition.hpp>
#include <atl/graph/centroid_bisect.hpp>
#include <atl/graph/ternarization.hpp>
```

\include centroid-decomposition/quick.cpp

## Ranking separators

### API and complexity

```cpp
struct centroid_vertex {
    vertex_id vertex;
    vertex_id parent;
    uint32 depth;
    uint32 branch;
};
template<adjacency_graph Graph, typename Event>
    requires (centroid_event<Event> || centroid_edge_event<Event, edge_t<Graph>>)
        && (!centroid_edge_event<Event, edge_t<Graph>>
            || requires(edge_t<Graph> &stored, std::remove_reference_t<std::ranges::range_reference_t<out_edges_t<Graph>>> const &element) { stored = element; })
constexpr void centroid_visit(Graph const &g, Event f);
template<adjacency_graph Graph>
constexpr std::vector<vertex_id> centroid_decomposition(Graph const &g);
```

`centroid_visit` calls `f(center, parent_center, component)` once per vertex.
The span contains the center first (`parent == branch == none`, depth zero),
then contiguous branches. Every parent precedes its children, but subtree
records need not be contiguous. Every pair of distinct vertices in one tree
appears in exactly one event with distinct branches or with a center endpoint.
Events use decreasing rank, so each separator parent has already appeared;
no relative order is promised within one rank.

`centroid_event<F>` checks that three-argument call. `centroid_edge_event<F, Edge>`
also requires a default-initializable copyable edge and accepts
`f(center, parent_center, component, edges)`. The edge span is parallel to the
records, with a value-initialized center edge and parent-to-child elements.
A callback matching both forms takes the edge form. Propagate a weighted value
with `value[v] = value[parent] + weight(edges[i])`, seeded at the center.

`centroid_decomposition` returns each separator's parent, or `none` at each
component's top separator. This is an optimal vertex-ranking separator tree;
a separator need not halve its component. Its maximum depth is at most that
of any centroid tree of the same forest: a centroid tree gives a valid
ranking, while this ranking has minimum height. Use centroid_bisect for
branch-size guarantees.

A sweep takes O(n log n) time and O(n) memory, plus callback costs. Total span
length is at most n(floor(log2 n) + 1). The parent-only function takes
O(n + m) plus merges of lists containing at most 32 ranks, and O(n) memory.

### Preconditions

Asserted: n < 2^31. Documented only: symmetric, acyclic forest; valid targets;
stable graph during the call. The edge callback additionally requires storing
actual adjacency references in the edge value type. Spans expire at callback
return. Empty forests produce no events and an empty parent array. All
traversals use heap buffers without per-vertex recursion; allocation and
callback exceptions propagate.

### Measured performance

The alternative separator engines are defined in
[`centroid_variants.hpp`](../experimental/atl/experimental/graph/centroid_variants.hpp).

[benchmarks/graph/centroid_decomposition.cpp](../benchmarks/graph/centroid_decomposition.cpp)
measures full event sweeps, parent-only construction and ranking order.
Contenders change the component sweep, label bucketing, size recomputation or
BFS queue representation. Classic size-based decompositions can select
different separators; correctness tests certify the pair partition rather
than requiring equal depth sums.

Intel Xeon Platinum 8358 @ 2.60 GHz, x86-64 Linux; GCC 15.2.0,
RelWithDebInfo, assertions off, 2026-10-15. Flags `-O2 -g -DNDEBUG -std=c++23`,
also with `-march=native`. Minimum milliseconds over three runs of 25 samples,
plain / native.

| n=200000 shape | library ranking | by queue sweep | by bucket labels | by subtree sizes | by reused sizes |
|---|---|---|---|---|---|
| random | 47.8 / 47.2 | 41.1 / 40.3 | 48.0 / 46.8 | 67.8 / 66.6 | 51.3 / 50.3 |
| path | 26.9 / 25.0 | 28.6 / 26.4 | 26.8 / 24.9 | 58.5 / 56.1 | 44.8 / 42.6 |
| caterpillar | 20.8 / 19.5 | 20.6 / 17.8 | 20.8 / 19.4 | 41.3 / 39.4 | 32.3 / 30.2 |
| binary | 23.1 / 22.5 | 16.1 / 15.7 | 23.1 / 21.1 | 34.3 / 32.5 | 26.3 / 24.0 |
| broom | 21.5 / 20.5 | 21.8 / 20.9 | 21.5 / 20.2 | 33.1 / 31.9 | 26.2 / 25.1 |

| separator parent construction | library direct ranking | by component sweep |
|---|---|---|
| random 200000 | 9.9 / 9.5 | 46.5 / 46.3 |
| path 200000 | 2.9 / 2.8 | 25.4 / 23.7 |
| caterpillar 200000 | 2.5 / 2.3 | 19.5 / 18.5 |
| binary 200000 | 2.2 / 2.1 | 22.0 / 20.2 |
| broom 200000 | 2.6 / 2.5 | 20.3 / 19.3 |
| random 1000000 | 106 / 103 | 495 / 492 |
| path 1000000 | 15.4 / 14.8 | 146 / 135 |

| ranking order, n=1000000 | library direct queue | by view per vertex |
|---|---|---|
| random tree | 58.5 / 56.4 | 72.3 / 69.2 |
| singletons | 4.5 / 4.2 | 24.7 / 25.8 |
| pairs | 7.8 / 7.4 | 26.9 / 27.1 |

### Pitfalls and notes

Do not treat the span as true DFS preorder, or assume a minimum-ranking
separator halves the component size. For path counting, query accumulated
branches then insert the current branch; same-branch pairs belong to another
event. The parent-only entry point avoids building component spans.

## Balanced bisection

### API and complexity

```cpp
template<adjacency_graph Graph, typename Event>
    requires centroid_bisect_event<Event> || weighted_centroid_bisect_event<Event, Graph>
constexpr void centroid_bisect(Graph const &g, Event f);
```

The hop callback is `f(center, group_a, group_b)`, each group a
`std::span<visit const>`, excluding center. The five-argument edge form adds
parallel `edges_a, edges_b` spans and requires default-initializable copyable
edge elements. Edges retain their initial BFS orientation when components are rerooted,
so use symmetric payloads such as weights. The named concepts check invocability; matching both selects
the edge form.

Each group's records are parent-first, with hop depths from center. Their
sizes are each at least ceil((component_size - 1)/3). Every unordered pair at
distance at least two straddles groups in exactly one event. Self pairs and
adjacent pairs never appear; seed those separately. A crossing pair's distance
is the sum of depths, so histogram convolution can combine groups directly.
The engine takes O(n log n) time and total event size, excluding callback work,
and O(n) working memory. No ternarization is required.

### Preconditions

Asserted: n < 2^31. Documented only: symmetric forest, valid targets and no
graph mutation during callbacks. Empty forests and components of size at most
two emit nothing. Spans expire when the callback returns. Edge elements must
be storable from actual adjacency references. The initial BFS would choose a
spanning forest on invalid cyclic input; that is outside the contract.
Heap frames replace per-vertex recursion. Exceptions propagate.

### Measured performance

The single-span callback alternative is defined in
[`centroid_bisect_variants.hpp`](../experimental/atl/experimental/graph/centroid_bisect_variants.hpp).

[benchmarks/graph/centroid_bisect.cpp](../benchmarks/graph/centroid_bisect.cpp)
measures engine latency with a trivial mass checksum. The library reverses
only a centroid-to-root path, balances by thirds and writes child frames in
one layout sweep. The CSR/BFS contender builds children lists and uses
first-fit coloring; its bucket variant changes only CSR construction. They
need not emit the same mass, but must partition exactly the same path pairs.
Machine, compiler, build, date and plain/native flags are the ranking record
above: Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, 2026-10-15.

| hop shape | library | by CSR BFS | by bucket CSR BFS |
|---|---|---|---|
| random 200000 | 47.8 / 48.3 | 110 / 112 | 112 / 109 |
| star 200000 | 49.8 / 49.1 | 80.7 / 80.3 | 80.4 / 80.0 |
| caterpillar 200000 | 40.9 / 40.3 | 77.2 / 78.4 | 77.1 / 76.3 |
| path 200000 | 43.2 / 43.0 | 71.0 / 72.2 | 70.4 / 69.8 |
| broom 200000 | 46.6 / 46.5 | 75.8 / 76.8 | 75.7 / 75.0 |
| spider 1,h,h 200000 | 42.6 / 42.4 | 73.7 / 75.0 | 72.7 / 72.2 |
| random 1000000 | 266 / 264 | 644 / 650 | not measured yet |
| star 1000000 | 301 / 298 | 472 / 473 | not measured yet |

| edge-span shape | library | by CSR BFS |
|---|---|---|
| random 200000 | 58.5 / 58.9 | 125 / 124 |
| caterpillar 200000 | 50.4 / 49.9 | 93.2 / 93.9 |

Single-span callback shape: not measured yet in the supplied records.

### Pitfalls and notes

Keep the center available as the initial value when folding parent-edge
distances; it is absent from both groups. Group order is parent-first, not
necessarily depth-sorted. First-fit branch coloring is only a benchmark
alternative: it need not satisfy the library's thirds balance.

## Ternarization

### API and complexity

```cpp
template<typename Weight>
struct ternarization {
    csr<edge<Weight>> tree;
    vertex_id root;
    uint32 real_count;
    constexpr bool is_real(vertex_id v) const noexcept;
};
template<typename Weight = void, adjacency_graph Graph>
    requires ternarization_weight<Weight, Graph>
        && std::copyable<detail::ternarize_weight_t<Graph, Weight>>
        && std::default_initializable<detail::ternarize_weight_t<Graph, Weight>>
        && (weighted_adjacency_graph<Graph>
            || std::constructible_from<detail::ternarize_weight_t<Graph, Weight>, int>)
        && (!weighted_adjacency_graph<Graph>
            || requires(std::remove_reference_t<std::ranges::range_reference_t<out_edges_t<Graph>>> const &element) {
                static_cast<detail::ternarize_weight_t<Graph, Weight>>(graph::weight(element));
            })
constexpr auto ternarization_of(Graph const &g, vertex_id root);
```

The result's maximum degree is three, and root has at most two children.
Real vertices retain `[0, real_count)` ids; fake vertices use larger ids.
`is_real(v)` tests that boundary without accessing the graph. The selected
weight defaults to the input weight or int for unit hops. Fake edges carry
value-initialized zero; real edges retain converted input weights, or one
for unweighted input. Distances between real vertices are unchanged.
Explicit Weight must satisfy `ternarization_weight<Weight, Graph>`: default
initialization, copying and construction from int and the input weight.

Construction takes O(n log n) time and O(n) memory, producing fewer than 2n
vertices. `is_real` takes O(1). BFS and child arrays are on the heap; balanced
fan-out uses an explicit O(log n) frame stack and no recursive calls.

### Preconditions

Asserted: root is valid, n < 2^31, the input is connected and has exactly the
rejected-arc count of an acyclic symmetric tree, and output vertices <= none/2
so doubled CSR edge counts fit. Documented only: symmetric adjacency, valid
targets and zero-valued default weight initialization. Arbitrary asymmetric
graphs are outside the contract even if their rejection count happens to pass.
Weight conversion must preserve the distances the caller intends. Result
spans follow [CSR ownership rules](graph.md). Allocation failures propagate.

### Measured performance

[benchmarks/graph/ternarization.cpp](../benchmarks/graph/ternarization.cpp)
measures complete construction at n=1000000; unit-hop and int64 rows differ
in edge payload. No competing builder is timed here.
Intel Xeon Platinum 8280 @ 2.70 GHz, x86-64 Linux; GCC 15.2.0,
RelWithDebInfo, assertions off, 2026-10-23. Flags `-O2 -g -DNDEBUG -std=c++23`,
also with `-march=native`. Minimum milliseconds over three runs of 25 samples,
plain / native.

| shape | unit hops | int64 weights |
|---|---|---|
| star | 39.4 / 39.8 | 74.9 / 74.7 |
| path | 21.8 / 21.9 | 29.4 / 28.8 |
| caterpillar | 25.1 / 26.3 | 32.4 / 31.7 |
| random parent | 229 / 230 | 317 / 319 |

Explicit-frame fan-out requires remeasurement; the table retains the supplied
construction record.

### Pitfalls and notes

When composing with centroid_visit, propagate weighted depth across fake edges
and count only real endpoints. A fake center contributes no center-endpoint
pair. Bounded fan-in can simplify pairwise convolutions and sorted merges;
bisection avoids adding fake vertices. Compare complete pipelines with the
actual convolution or merge kernel: extra vertices can outweigh bounded degree.

## How it works

Ranking uses a reverse BFS mask DP. A vertex must hide ranks duplicated across
children and choose a rank absent from their visible union. One-hot labels make
rank comparisons integer comparisons; unprocessed parent masks remain zero.
A centroid argument bounds the number of ranks by floor(log2 n) + 1.
Sweeps visit only smaller labels, with branch-contiguous records and explicit
stacks. Every pair has one unique maximum-ranked vertex on its path. The direct
parent algorithm merges lists of visible ranks and finalizes a vertex as soon
as a stronger visible separator is known.

Bisection instead stores each pending component as a local parent array with
parents preceding children. A reverse size pass finds a centroid. Its branches
are divided into groups: a branch of at least a third stands alone; otherwise
branches accumulate until reaching a third. Only the centroid-to-root path
changes parent direction. One layout sweep writes both event groups and the
child frames, each retaining center. Induced parent arrays keep scanned degree
proportional to the component, even for stars; rescanning the original center's
full adjacency in every child would lose the O(n log n) bound. A frame pool
reuses vector storage.

Ternarization roots by BFS, buckets children by parent, and accumulates subtree
sizes. Each child list splits near half its subtree mass; internal fan-out nodes
have zero-weight edges. Real-node degrees and fake counts determine CSR row
sizes before edges are emitted, so both half-edges are written directly without
an intermediate arc list. Real edges keep their weights. Explicit fan-out frames preserve depth-first fake ids and edge emission order,
while avoiding recursive class-valued results in constant evaluation.

## See also

- [Graph vocabulary and traversal](graph.md).
- [Lowest common ancestors](lca.md).
- [Virtual trees](virtual-tree.md).
