# `atl::graph` components

Use `scc` for directed mutual reachability, `two_edge_connected_components`
for undirected bridge contraction, and `biconnected_components` for edge
blocks and articulation points. These are static whole-graph decompositions;
use disjoint sets when edges arrive incrementally and only connectivity matters.

```cpp
#include <atl/graph/scc.hpp>
#include <atl/graph/two_edge_connected_components.hpp>
#include <atl/graph/biconnected_components.hpp>
```

All three accept `template<detail::scannable_graph Graph>` and take
`Graph const &g`. This builds on `adjacency_graph`: targets come from
`target(edge)`; reference-returning rows must be const-iterable forward
ranges, while value-returning rows must be sized random-access ranges,
default-initializable and assignable. CSR, nested vectors, stable list rows,
and generated adjacency need no algorithm-specific wrapper. Write n for the
vertex count and m for the number of stored arcs (both directions count in an
undirected graph). See the
[graph vocabulary and representations](graph.md).

## Strongly connected components: `scc`

\include graph-components/scc.cpp

Expected stdout:

\include graph-components/scc.out

### API

```cpp
template<detail::scannable_graph Graph>
constexpr vertex_partition scc(Graph const &g);
```

Returns an owning `vertex_partition` from `<atl/graph/partition.hpp>`:
`component[v]` is dense in `[0, count())`, `same(u, v)` compares classes,
and `groups()` / `grouped()` materialize vertices in ascending order within
each component. See [partitions](graph.md#vertex-groups-and-partitions).
Component ids form a topological order of the condensation:
`component[u] <= component[v]` on every edge. Ascending ids support DAG
DP; unrelated components have no prescribed relative order.

### Complexity and preconditions

O(n + m) time and O(n) auxiliary storage, assuming O(1) graph access,
target extraction and iterator operations. Accessor and generated-row costs
are additional. Search state lives in heap vectors; path depth does not add
recursive calls, including on a path of a million vertices.

Asserted with `ATL_ASSERT`: `vertex_count(g) < none` (`none == 2^32 - 1`)
and every edge target is in `[0, vertex_count(g))`. Integral endpoint
conversion also checks representability in `vertex_id`. Index-resumed rows
contain at most `none` arcs (asserted by the shared scanner). Documented only:
adjacency remains unchanged, repeated access preserves row order, and
reference-returning rows remain valid during the call. Allocations may throw;
assertion-disabled calls still require these preconditions. An empty graph
returns zero components.

### Measured performance

The library single-array Pearce search races Tarjan with separate low-links,
Tarjan expressed through DFS events, Gabow path boundaries and Kosaraju's
transpose plus second traversal. Kosaraju uses O(n + m) heap storage for its
transpose and traversal buffers; the other SCC engines use O(n) auxiliary storage. All shapes use n=1e6. The nested-vector and forward-only cases
run the library algorithm; groups() materializes an already computed partition.

[benchmarks/graph/scc.cpp](../benchmarks/graph/scc.cpp).

Benchmarks are built in CI and
measured separately.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | Contender | ms plain / native | spread plain / native |
|---|---|---|---|
| random m=4e6 | library | 121 / 122 | 1.24 / 1.54 |
| random m=4e6 | by tarjan events | 202 / 205 | 1.29 / 1.34 |
| random m=4e6 | by tarjan | 164 / 182 | 1.03 / 1.18 |
| random m=4e6 | by gabow | 139 / 140 | 1.27 / 1.30 |
| random m=4e6 | by kosaraju | 190 / 209 | 1.53 / 1.33 |
| random m=4e6 | library on nested vectors | 267 / 285 | 1.21 / 1.10 |
| random m=4e6 | library on forward-only rows | 280 / 307 | 1.05 / 1.07 |
| random m=4e6 | groups() | 4.7 / 4.8 | 1.01 / 1.02 |
| giant cycle + chords | library | 17 / 18.1 | 1.03 / 1.05 |
| giant cycle + chords | by tarjan events | 20.8 / 24.3 | 1.06 / 1.19 |
| giant cycle + chords | by tarjan | 19.3 / 19.3 | 1.05 / 1.04 |
| giant cycle + chords | by gabow | 17.7 / 18.6 | 1.02 / 1.03 |
| giant cycle + chords | by kosaraju | 57.6 / 62 | 1.09 / 1.12 |
| giant cycle + chords | library on nested vectors | 20.9 / 20.9 | 1.03 / 1.02 |
| giant cycle + chords | library on forward-only rows | 44.1 / 47.1 | 1.02 / 1.04 |
| giant cycle + chords | groups() | 1.2 / 1.4 | 1.00 / 1.03 |
| deep DAG | library | 16.2 / 16.1 | 1.11 / 1.08 |
| deep DAG | by tarjan events | 17.6 / 20.9 | 1.33 / 1.08 |
| deep DAG | by tarjan | 17.5 / 19.1 | 1.31 / 1.06 |
| deep DAG | by gabow | 17.3 / 18.5 | 1.31 / 1.06 |
| deep DAG | by kosaraju | 37.2 / 39.9 | 1.02 / 1.05 |
| deep DAG | library on nested vectors | 17.5 / 18.6 | 1.04 / 1.04 |
| deep DAG | library on forward-only rows | 41.2 / 41.9 | 1.03 / 1.06 |
| deep DAG | groups() | 54.6 / 49.5 | 1.02 / 1.05 |

The library has the smallest algorithm-contender minimum on all three
shapes. The random rows vary between runs (library spread 1.24x/1.54x,
Gabow 1.27x/1.30x), so that row alone does not establish a stable ranking.
Kosaraju is slower on all three shapes and provides an independent oracle. All four contenders remain in
`experimental/atl/experimental/graph/scc_variants.hpp` as
`scc_by_tarjan_events`, `scc_by_tarjan`, `scc_by_gabow` and `scc_by_kosaraju`.

### Pitfalls and notes

Self-loops and parallel arcs are supported. Do not infer reachability merely
from ordered ids: the ordering is a necessary condition along edges.
`groups()` and `grouped()` own independent storage. SCC uses an index scan
position on random-access rows; forward-only rows retain iterators.
The row adapter allocates separate adjacency storage, so the timing comparison
also includes memory-layout effects and does not isolate scan-position cost.

## Two-edge-connected components: `two_edge_connected_components`

\include graph-components/bridges.cpp

Expected stdout:

\include graph-components/bridges.out

### API

```cpp
template<detail::scannable_graph Graph>
constexpr vertex_partition two_edge_connected_components(Graph const &g);
```

The return type and grouping operations match `scc`. Removing all bridges
leaves these vertex classes: an input edge `{u, v}` is a bridge iff
`!result.same(u, v)`. Contracting the classes gives a bridge forest. Root
each connected component's bridge tree at the class holding its lowest
vertex; parents have smaller ids than children. No order is promised across
connected components. Isolated vertices form singleton classes.

### Complexity and preconditions

O(n + m) time and O(n) auxiliary heap storage under the same access-cost
assumptions as SCC, with no recursion proportional to graph depth.
The vertex-count and target-range preconditions are asserted as for SCC.
Documented only: each undirected edge supplies both directed arcs, including
each parallel copy; row order and lifetimes satisfy the shared scan contract.
A self-loop may occur without affecting the result. Asymmetric input violates
the contract and is not checked. The empty graph has zero classes.
Allocation failures propagate.

### Measured performance

The single-array library search races low-link arrays, Gabow boundaries and
two-pass chain decomposition with parent walks. Each call computes a whole
partition at n=1e6; adjacency construction and correctness comparisons are outside timing.

[benchmarks/graph/two_edge_connected_components.cpp](../benchmarks/graph/two_edge_connected_components.cpp).

Benchmarks are built in CI and
measured separately.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | Contender | ms plain / native | spread plain / native |
|---|---|---|---|
| random m=4e6 | library | 392 / 258 | 1.07 / 1.29 |
| random m=4e6 | by low link | 478 / 316 | 1.06 / 1.21 |
| random m=4e6 | by gabow | 385 / 261 | 1.10 / 1.23 |
| random m=4e6 | by chain decomposition | 525 / 422 | 1.15 / 1.14 |
| random m=4e6 | library on nested vectors | 452 / 386 | 1.05 / 1.11 |
| random m=4e6 | library on forward-only rows | 450 / 377 | 1.03 / 1.11 |
| giant cycle + chords | library | 18.5 / 17.4 | 1.25 / 1.06 |
| giant cycle + chords | by low link | 22.3 / 20.2 | 1.14 / 1.06 |
| giant cycle + chords | by gabow | 20.3 / 19 | 1.09 / 1.03 |
| giant cycle + chords | by chain decomposition | 46.1 / 40.1 | 1.21 / 1.04 |
| giant cycle + chords | library on nested vectors | 59.8 / 54.4 | 1.09 / 1.06 |
| giant cycle + chords | library on forward-only rows | 22.3 / 21.3 | 1.24 / 1.08 |
| deep path + chords | library | 20.6 / 18.2 | 1.11 / 1.07 |
| deep path + chords | by low link | 23.3 / 20.3 | 1.17 / 1.06 |
| deep path + chords | by gabow | 20.5 / 18.9 | 1.13 / 1.04 |
| deep path + chords | by chain decomposition | 44.3 / 42 | 1.15 / 1.06 |
| deep path + chords | library on nested vectors | 58.2 / 50.9 | 1.08 / 1.12 |
| deep path + chords | library on forward-only rows | 25.1 / 21.8 | 1.08 / 1.07 |
| random tree | library | 95.1 / 27.6 | 1.12 / 2.80 |
| random tree | by low link | 132 / 61.9 | 1.06 / 1.82 |
| random tree | by gabow | 105 / 34.6 | 1.10 / 2.50 |
| random tree | by chain decomposition | 124 / 56.2 | 1.13 / 1.94 |
| random tree | library on nested vectors | 145 / 130 | 1.03 / 1.16 |
| random tree | library on forward-only rows | 154 / 129 | 1.02 / 1.17 |

Gabow beats the library plain on random graphs (385 versus 392 ms) and
deep paths (20.5 versus 20.6 ms); native the library wins those rows
(258 versus 261 ms, and 18.2 versus 18.9 ms). The library wins cycles
and trees in both modes. Native random graphs and trees vary between runs
(library spreads 1.29x and 2.80x), as does the plain cycle (1.25x); those
minima alone do not establish stable rankings. The library uses a single
discovery/low-link/component array. The
contenders are `two_edge_connected_components_by_low_link`,
`two_edge_connected_components_by_gabow` and
`two_edge_connected_components_by_chain_decomposition` in
`experimental/atl/experimental/graph/two_edge_connected_components_variants.hpp`.
Chain decomposition is the readable independent algorithm; tests certify
all variants against edge deletion. Experimental low-link and Gabow ids
promise the same partition, without the library's parent-order guarantee.

### Pitfalls and notes

An undirected edge list must be doubled; `undirected_csr` does this.
Exactly one arc to a DFS parent is skipped, so an additional parallel copy
is a back edge and prevents a false bridge. Self-loops alter no class.
Do not equate bridge components with vertex blocks: two cycles sharing
only a cut vertex are one bridge component and two blocks.

## Biconnected components: `biconnected_components`

\include graph-components/blocks.cpp

Expected stdout:

\include graph-components/blocks.out

### API

```cpp
template<detail::scannable_graph Graph>
constexpr block_decomposition biconnected_components(Graph const &g);

struct block_decomposition {
    std::vector<uint32> block;
    std::vector<vertex_id> top_vertices;
    constexpr std::size_t count() const;
    constexpr uint32 edge_block(vertex_id u, vertex_id v) const;
    constexpr std::vector<uint8> cut_vertex_mask() const;
    constexpr vertex_groups grouped() const;
    constexpr std::vector<std::vector<vertex_id>> groups() const;
};
```

Here `uint32` and `uint8` are atl's integer aliases. Blocks partition
non-self-loop edges; cut vertices can occur in several blocks. Tree edges
correspond to non-root vertices. `block[v]` names the block of the tree edge
above `v`, or `none` for a DFS root or isolated vertex. A non-isolated,
non-cut vertex has only one block, but its entry is still `none` when it is
a DFS root. For a cut vertex the entry describes this DFS tree, rather than
every incident block. `top_vertices[b]` supplies block b's shallowest vertex.
Block ids are dense with no ordering guarantee.

| Operation | Parameters and return | Time / additional storage |
|---|---|---|
| `count()` | Number of blocks | O(1) / O(1) |
| `edge_block(u, v)` | Block id of an existing non-self-loop edge, either orientation | O(1) / O(1) |
| `cut_vertex_mask()` | One 0/1 articulation flag per vertex | O(n + count()) / O(n) |
| `groups()` | Nested vertex lists: top first, then non-root members in ascending vertex order | O(n + count()) / O(n + count()) |
| `grouped()` | The same lists in owning flat `vertex_groups` storage | O(n + count()) / O(n + count()) |

The aggregate's public arrays must preserve a valid DFS block encoding:
vertex count is below `none`, block ids are valid or `none`, tops are valid
vertices outside their own non-root entries, and each block has a non-root
member. Groups and masks are independent copies. Spans obtained from flat
groups borrow that result's members and are invalidated by reallocation or
destruction of that storage.

### Complexity and preconditions

The search is O(n + m) time and O(n) auxiliary heap storage under the access
assumptions given for SCC. Depth uses explicit heap frames. The vertex-count,
valid-target, row-lifetime and symmetric-adjacency preconditions match
`two_edge_connected_components`; only the count and targets are asserted.
Allocation failures propagate. `edge_block` asserts in-range distinct
endpoints; existence of the edge is documented only, and cannot be inferred
from its structural assertions.

For `grouped()`, the total membership count must fit `uint32` (asserted),
and `count() + 1` must fit `size_t` (asserted). Total membership is the number
of non-root vertices plus the block count, at most twice the number of
non-root vertices. In particular a long path needs two entries per block,
so n alone does not bound the flat offsets. Nested `groups()` does not use
a single 32-bit cumulative offset.

### Measured performance

The library Gabow search uses bare discovery-index boundaries and reads the
parent from the adjacent frame. Contenders store paired vertex/index
boundaries plus frame parents, separate low-links, or single-array state
with larger frames. Chain decomposition computes only cuts; groups() is timed
on the library result independently. Unless a row states n=4e6, n=1e6.

[benchmarks/graph/biconnected_components.cpp](../benchmarks/graph/biconnected_components.cpp).

Benchmarks are built in CI and
measured separately.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | Contender | ms plain / native | spread plain / native |
|---|---|---|---|
| random m=4e6 | library | 236 / 197 | 1.25 / 1.05 |
| random m=4e6 | by single array | 252 / 210 | 1.37 / 1.04 |
| random m=4e6 | by low link | 271 / 210 | 1.27 / 1.08 |
| random m=4e6 | by chain decomposition (cuts only) | 398 / 337 | 1.23 / 1.09 |
| random m=4e6 | by paired boundaries | 253 / 195 | 1.25 / 1.04 |
| random m=4e6 | library on nested vectors | 375 / 341 | 1.13 / 1.05 |
| random m=4e6 | library on forward-only rows | 380 / 338 | 1.11 / 1.05 |
| random m=4e6 | by paired boundaries on forward-only rows | 382 / 336 | 1.09 / 1.02 |
| random m=4e6 | groups() | 1.7 / 1.7 | 1.00 / 1.01 |
| giant cycle + chords | library | 21 / 20 | 1.03 / 1.04 |
| giant cycle + chords | by single array | 25.2 / 27.1 | 1.05 / 1.03 |
| giant cycle + chords | by low link | 19.9 / 18.6 | 1.02 / 1.05 |
| giant cycle + chords | by chain decomposition (cuts only) | 48.3 / 48.3 | 1.04 / 1.05 |
| giant cycle + chords | by paired boundaries | 19.4 / 18.5 | 1.02 / 1.02 |
| giant cycle + chords | library on nested vectors | 49.9 / 45.6 | 1.12 / 1.14 |
| giant cycle + chords | library on forward-only rows | 19.1 / 19.3 | 1.03 / 1.06 |
| giant cycle + chords | by paired boundaries on forward-only rows | 18.5 / 19.2 | 1.04 / 1.03 |
| giant cycle + chords | groups() | 1.3 / 1.3 | 1.00 / 1.01 |
| deep path + chords | library | 21.1 / 20.3 | 1.02 / 1.05 |
| deep path + chords | by single array | 25.9 / 27.4 | 1.03 / 1.03 |
| deep path + chords | by low link | 19.4 / 18.9 | 1.07 / 1.03 |
| deep path + chords | by chain decomposition (cuts only) | 49.4 / 49.8 | 1.04 / 1.06 |
| deep path + chords | by paired boundaries | 20 / 18.8 | 1.04 / 1.06 |
| deep path + chords | library on nested vectors | 47.4 / 44.9 | 1.14 / 1.15 |
| deep path + chords | library on forward-only rows | 20 / 20.6 | 1.05 / 1.06 |
| deep path + chords | by paired boundaries on forward-only rows | 19.3 / 19.7 | 1.04 / 1.06 |
| deep path + chords | groups() | 1.3 / 1.3 | 1.01 / 1.01 |
| random tree | library | 61.4 / 43.8 | 1.89 / 1.41 |
| random tree | by single array | 56.5 / 44.6 | 1.82 / 1.44 |
| random tree | by low link | 79.3 / 49.4 | 1.63 / 2.09 |
| random tree | by chain decomposition (cuts only) | 82.2 / 66.9 | 1.60 / 1.27 |
| random tree | by paired boundaries | 54.7 / 41.5 | 2.08 / 1.91 |
| random tree | library on nested vectors | 152 / 140 | 1.12 / 1.05 |
| random tree | library on forward-only rows | 158 / 138 | 1.12 / 1.11 |
| random tree | by paired boundaries on forward-only rows | 151 / 131 | 1.11 / 1.05 |
| random tree | groups() | 43.3 / 42.8 | 1.02 / 1.02 |
| deep path + chords n=4e6 | library | 139 / 123 | 1.14 / 1.09 |
| deep path + chords n=4e6 | by single array | 226 / 210 | 1.15 / 1.07 |
| deep path + chords n=4e6 | by low link | 163 / 144 | 1.17 / 1.06 |
| deep path + chords n=4e6 | by chain decomposition (cuts only) | 299 / 269 | 1.08 / 1.15 |
| deep path + chords n=4e6 | by paired boundaries | 179 / 158 | 1.05 / 1.06 |
| deep path + chords n=4e6 | library on nested vectors | 312 / 298 | 1.07 / 1.09 |
| deep path + chords n=4e6 | library on forward-only rows | 188 / 186 | 1.12 / 1.06 |
| deep path + chords n=4e6 | by paired boundaries on forward-only rows | 206 / 200 | 1.09 / 1.07 |
| deep path + chords n=4e6 | groups() | 5 / 5 | 1.01 / 1.00 |

The library loses to separate low-links on cycles (21/20 versus
19.9/18.6 ms) and deep paths at n=1e6 (21.1/20.3 versus 19.4/18.9 ms).
Paired boundaries also beat it on those shapes and on random trees
(54.7/41.5 versus 61.4/43.8 ms); the tree rows vary between runs, with
library spread 1.89x/1.41x and paired-boundary spread 2.08x/1.91x.
On random graphs paired boundaries win native (195 versus 197 ms),
while the library wins plain (236 versus 253 ms); the plain rows vary
between runs. At n=4e6 the library beats paired boundaries
(139/123 versus 179/158 ms). Contenders remain in
`experimental/atl/experimental/graph/biconnected_components_variants.hpp` as
`biconnected_components_by_single_array`, `biconnected_components_by_low_link`,
`biconnected_components_by_paired_boundaries` and
`cut_vertex_mask_by_chain_decomposition`. Vertex-deletion tests certify
blocks and articulation points; chain decomposition is the readable cut oracle.

### Pitfalls and notes

Self-loops belong to no block and influence no articulation point. An
isolated or loop-only vertex has `block[v] == none` and appears in no group.
Parallel copies share a block. A bridge itself is a two-vertex block.
The top need not be the lowest-numbered member, so a whole group need not be
sorted; cut vertices occur in every group they touch. A root is an
articulation point when it is top of at least two blocks; a non-root needs
at least one such block. Saturating the count at two prevents overflow
on high-degree stars.

## How it works

### SCC

Pearce's search reuses one integer array for discovery indices, low-links
and final component ids. Live indices grow upward; final ids descend from
n, so the ranges remain separate as completed vertices release live indices.
Frames hold the current vertex, scan position and root-candidate flag;
completed non-roots wait on an open stack. Closing a root drains its suffix
and assigns a final id. A final shift makes ids dense and topological.
Each frame, vertex and arc is processed a constant number of times.

### Bridge components

The same single-array invariant closes a component whenever no path from its
subtree reaches a proper ancestor. The incoming tree edge is then a bridge.
Skipping one reverse parent arc admits every parallel back edge. Final ids
descend as components close, so shifting preserves parent-smaller-than-child
order in each bridge tree. Every stack entry is pushed and popped once.

### Blocks

Gabow's pending boundaries store parent discovery indices. A back edge to w
pops boundaries whose parent lies strictly below w; equality preserves a
parallel-pair block. The live indices increase up the stack, so at a child's
postorder equality with its parent's index identifies its surviving boundary.
The parent's frame is immediately below the child. A surviving boundary
closes a block and pops its still-open subtree vertices. Each boundary is
popped at most once, establishing linear time. Tree-edge labels plus each
block's top recover any non-self-loop edge's block in O(1): two blocks share
at most one vertex, so one endpoint's entry either contains the other or
that other endpoint must supply the block.

Algorithm references: Pearce (2005), “An Improved Algorithm for Finding the
Strongly Connected Components of a Directed Graph”; Gabow (2000),
“Path-based depth-first search for strong and biconnected components”;
Schmidt (2013), “A simple test on 2-vertex- and 2-edge-connectivity”.
These are algorithm references; implementations are independent.

## See also

- [Graph vocabulary, CSR and traversal](graph.md).
- [Disjoint sets](dsu.md) for incremental connectivity.
- [Time divide and conquer](time-divide-and-conquer.md) for offline connectivity.
