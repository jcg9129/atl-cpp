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
BFS queue representation. Classic size-based decompositions can select different separators; correctness
tests certify the pair partition rather than requiring equal depth sums.
The library loses the random event sweep to queue sweep (41.9/36.9 versus
36.6/32.3 ms), and the binary tree likewise (16.3/14 versus 11.8/11.2 ms).

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| n=200000 shape | library ranking | by queue sweep | by bucket labels | by subtree sizes | by reused sizes |
|---|---|---|---|---|---|
| random | 41.9 / 36.9 | 36.6 / 32.3 | 41.9 / 37.2 | 53.9 / 51.9 | 38.6 / 35.6 |
| path | 29.6 / 23.3 | 33.2 / 25.2 | 29.6 / 23.3 | 49 / 41.7 | 38.8 / 32 |
| caterpillar | 19.2 / 15.7 | 19.7 / 15.7 | 19.2 / 15.8 | 36.9 / 32.5 | 28.2 / 24.5 |
| binary | 16.3 / 14 | 11.8 / 11.2 | 16.5 / 14.1 | 26.1 / 24 | 19.3 / 17.3 |
| broom | 20.5 / 16.7 | 21.8 / 18 | 20.8 / 17.3 | 26.7 / 22.8 | 21.5 / 18 |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| separator parent construction | library direct ranking | by component sweep |
|---|---|---|
| random 200000 | 8.5 / 8 | 41.5 / 35.8 |
| path 200000 | 2 / 2 | 28.7 / 22.2 |
| caterpillar 200000 | 2 / 2 | 18.1 / 15 |
| binary 200000 | 1.6 / 1.6 | 15.3 / 13.2 |
| broom 200000 | 1.7 / 1.8 | 18.4 / 16 |
| random 1000000 | 72.9 / 129 | 308 / 770 |
| path 1000000 | 10.3 / 10.7 | 163 / 131 |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| ranking order, n=1000000 | library direct queue | by view per vertex |
|---|---|---|
| random tree | 36.6 / 72.6 | 45.1 / 100 |
| singletons | 2.7 / 2.7 | 14.1 / 18.8 |
| pairs | 4.7 / 5.1 | 17.3 / 20.1 |

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


AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| hop shape | library | by CSR BFS | by bucket CSR BFS |
|---|---|---|---|
| random 200000 | 41.4 / 45.7 | 118 / 136 | 119 / 136 |
| star 200000 | 38.4 / 45.7 | 88.8 / 114 | 86.2 / 114 |
| caterpillar 200000 | 35.8 / 39.1 | 99.1 / 124 | 98 / 128 |
| path 200000 | 36.9 / 39.7 | 88.5 / 113 | 90.2 / 114 |
| broom 200000 | 38.2 / 42.4 | 89.7 / 113 | 89.3 / 115 |
| spider 1,h,h 200000 | 37.1 / 39.5 | 96.9 / 121 | 95.3 / 122 |
| random 1000000 | 238 / 269 | 692 / 810 | not measured yet |
| star 1000000 | 205 / 265 | 494 / 646 | not measured yet |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| edge-span shape | library | by CSR BFS |
|---|---|---|
| random 200000 | 53.8 / 63.9 | 128 / 151 |
| caterpillar 200000 | 46.6 / 55 | 107 / 132 |

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


AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | unit hops | int64 weights |
|---|---|---|
| star | 24 / 28.1 | 58.4 / 61.5 |
| path | 11.2 / 13.7 | 13.2 / 15.9 |
| caterpillar | 17.9 / 20.6 | 19.3 / 22.2 |
| random parent | 75.8 / 161 | 128 / 199 |

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
