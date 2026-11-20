# `atl::graph::bipartite_matching`

Maximum bipartite matching pairs vertices from two separate ranges. The same
header constructs minimum vertex covers, maximum independent sets, edge roles
across all maximum matchings, and path and chain partitions of DAGs with
maximum-antichain certificates. Use [assignment](assignment.md) for a dense
weighted matching and [maximum flow](max-flow.md) for capacity constraints.

```cpp
#include <atl/graph/bipartite_matching.hpp>
```

\include bipartite-matching/quick.cpp

Expected stdout:
\include bipartite-matching/quick.out

## API

Every entry is constexpr and returns owning storage. `EdgeList` satisfies
`edge_list`: a const forward range whose values and scanned references expose
`source(e)` and `target(e)` convertible to `vertex_id`. This accepts arrays,
spans, nonsized forward lists, generated edge values and endpoint proxies.
`arc<>{l, r}` uses **separate** left and right ranges, each starting at zero.
The input is neither modified nor retained. Parallel edges are permitted.

```cpp
struct bipartite_matching_result {
    std::size_t size = 0;
    std::vector<uint32> left_pair;
    std::vector<uint32> right_pair;
};
struct bipartite_vertex_set {
    std::size_t size = 0;
    std::vector<uint8> left;
    std::vector<uint8> right;
};
enum class matching_edge : uint8 { forbidden, allowed, mandatory };
```

`left_pair[l]` is the paired right vertex, `right_pair[r]` the paired left
vertex; `none` denotes unmatched. No particular maximum matching is promised
when several exist. A vertex set holds one 0/1 mask per side and its size.

All the following declarations have `template<edge_list EdgeList>`:

| constexpr declaration | result |
|---|---|
| `bipartite_matching_result bipartite_matching(std::size_t left, std::size_t right, EdgeList const &edges)` | maximum matching |
| `bipartite_vertex_set minimum_vertex_cover(std::size_t left, std::size_t right, EdgeList const &edges, bipartite_matching_result const &matching)` | a minimum cover, touching every edge |
| `bipartite_vertex_set maximum_independent_set(std::size_t left, std::size_t right, EdgeList const &edges, bipartite_matching_result const &matching)` | the cover's complement, with no edge inside |
| `std::vector<matching_edge> matching_edge_roles(std::size_t left, std::size_t right, EdgeList const &edges, bipartite_matching_result const &matching)` | one role per input edge in iteration order |
| `path_cover minimum_path_cover(std::size_t n, EdgeList const &edges)` | fewest vertex-disjoint paths along DAG edges |
| `chain_cover minimum_chain_cover(std::size_t n, EdgeList const &edges)` | fewest chains in DAG reachability, plus a maximum antichain |

Cover, independent-set and edge-role functions take the graph again plus a
**valid maximum** matching. Cover size equals matching size, and the independent
set has `left + right - matching.size` vertices. These are pay-per-use operations.
A role is `forbidden` in no maximum matching, `allowed` in some but not all,
or `mandatory` in every one. Parallel copies have separate roles: a matched
pair with interchangeable copies makes each copy allowed.

```cpp
struct path_cover {
    std::size_t count = 0;
    std::vector<uint32> next, prev;
    constexpr std::vector<std::vector<vertex_id>> paths() const;
};
struct chain_cover {
    std::size_t count = 0;
    std::vector<uint32> next, prev;
    std::vector<uint8> antichain;
    constexpr std::vector<std::vector<vertex_id>> chains() const;
};
```

`next` and `prev` are successor and predecessor links, with `none` at the ends.
`paths()` and `chains()` reconstruct owning lists in increasing start-vertex
order, each start to end. Path links use input edges; chain links may skip
vertices through reachability. The chain-cover function computes transitive
closure internally. Its antichain mask contains exactly `count` vertices and
certifies optimality by Dilworth equality.

\include bipartite-matching/dag.cpp

Expected stdout:
\include bipartite-matching/dag.out

## Complexity

Write L and R for the side counts, m for input edges, and q for closure edges.
With V=L+R, the uncapped auction has a conservative O(V²(V+m)) worst-case
bound; its layered finish is O(V + m sqrt(V)). Storage is O(V+m). Each finite
bid increases one left level, global relabeling never lowers valid levels,
and finite levels remain O(V), bounding bids by O(V²). This conservative bound
does not claim the tighter bound of auction algorithms that cap their bidding phase.
The three matching companions take O(L + R + m) time and storage. Path cover
has matching's bound with L=R=n. Reconstruction takes O(n) time and storage.

Chain cover takes conservative O(n + m + (n+m) ceil(n/64) + n²(n+q)) time and
O(n ceil(n/64) + m + q) storage. Materializing q closure edges is part of the
cost, even when the bitset computation is cheap. Use this for modest n: a
chain has quadratic closure. All graph searches use heap queues or explicit
stacks, with O(1) call stack on million-vertex paths. Allocation failures propagate.

## Preconditions

Asserted under `ATL_ASSERT`:

- L and R are below `none` (UINT32_MAX), m is at most UINT32_MAX, and every
  source/target is in its respective range. Matching offsets are uint32.
- Matching companions check pair-array sizes. Cover construction checks that
  its cardinality equals the supplied matching's cardinality.
- Independent-set construction requires L+R to fit size_t.
- Edge roles require L+R+2 < UINT32_MAX and m+L+R < UINT32_MAX for their
  alternating residual CSR, including its source and sink.
- DAG covers require n < UINT32_MAX, m <= UINT32_MAX and valid endpoints.
  Kahn's algorithm checks acyclicity. Chain cover additionally checks that
  `n * ceil(n/64)` fits `vector<uint64>::max_size()` and q <= UINT32_MAX.

Documented only: iteration remains stable through a call, and a supplied
matching is valid, mutually consistent and maximum for that graph. Its size
field must agree with its pairs. The cover-size assertion detects a nonmaximum
**valid** matching; it does not validate arbitrary malformed arrays. Reconstruct
only unmodified links describing a partition. Auction levels and relabel
periods use uint64; between relabels levels are below 4L+2R+2. No arithmetic
assumes 64-bit size_t. Graphs and results otherwise use their own owning vectors.

## Measured performance

`benchmarks/graph/bipartite_matching.cpp` measures a complete matching including
adjacency construction. The library uses auction bidding plus a Hopcroft-Karp
finish; contenders use greedy Hopcroft-Karp, per-root Kuhn, pass-structured
Kuhn, a full bid scan, a scan stopping at level zero, or a max_flow reduction.
The max_flow comparison includes its network construction and certificate.

Intel Xeon Platinum 8358 @ 2.60 GHz, x86-64 Linux, GCC 15.2.0,
RelWithDebInfo, assertions off, 2026-10-15. Milliseconds, minimum of three
per-run means, ten samples per row. Each pair is plain `-O2` / `-O2 -march=native`,
both with `-g -DNDEBUG -std=c++23`.

| shape | library | by bid scan full | by bid scan level zero | by kuhn passes | by hopcroft karp | by kuhn | by max flow |
|---|---:|---:|---:|---:|---:|---:|---:|
| random L=R=1e5 m=2e5 | 15.9 / 16.8 | 15.7 / 15.5 | 18.2 / 18.1 | 18.4 / 18.1 | 18.8 / 18.9 | 12.8 / 12.7 | 77.1 / 79 |
| band L=R=66000 | 2.7 / 2.7 | 2.7 / 2.8 | 2.2 / 2.3 | 1.1 / 1.2 | 1 / 1 | 1.2 / 1.2 | 25.7 / 24.9 |
| line L=R=1e5 | 2.2 / 2.2 | 2.1 / 2 | 2 / 2 | 1 / 1.1 | 0.884 / 0.883 | 1.1 / 1.1 | 14.5 / 14.7 |
| augmented cycle L=R=66000 | 33.6 / 34.9 | 31.8 / 31.3 | 40.1 / 40.1 | 345 / 341 | 404 / 401 | 1385 / 1380 | 668 / 679 |
| skewed L=1e5 R=1e3 | 4.8 / 4.9 | 5 / 4.8 | 4.8 / 4.8 | 3.8 / 3.7 | 3.8 / 3.7 | 692 / 701 | 32.4 / 32.3 |
| dense L=R=3000 m=1e6 | 10.5 / 10.9 | 9.7 / 9.3 | 9.3 / 9.7 | 4 / 4.1 | 4.6 / 4.7 | 4.3 / 4.3 | 92.7 / 94.8 |
| dense L=R=3000 m=3e6 | 55.7 / 54.3 | 53.3 / 52.2 | 49.4 / 49.5 | 17.9 / 18.2 | 18.9 / 19.5 | 17.8 / 18 | 389 / 389 |
| complete L=R=2000 | 39.8 / 39.9 | 37 / 36.8 | 35.6 / 35.4 | 21 / 20.5 | 20.6 / 21 | 20.6 / 21.5 | 431 / 432 |

The plain augmented-cycle row is 404/33.6 = 12.02 times faster than pure
Hopcroft-Karp. Dense m=3e6 is 55.7/18.9 = 2.95 times slower. The level-zero
scan takes 18.2 versus 15.9 ms on random and 40.1 versus 33.6 on augmented
cycles; it helps the band and dense shapes. Per-pass Kuhn avoids repeated
failed searches on skewed sides (3.8 versus per-root Kuhn's 692 ms), while
augmented cycles still take 345 ms. The library retains the adversarial-path
win despite its adjacency-scan overhead on dense graphs.

### DAG closure and chain covers

`benchmarks/graph/dag_cover.cpp` times reverse-topological bitsets with CSR
output and reached-target skipping against arc output, arc output with
skipping, nearest-target-first ordering, Floyd-Warshall bitsets and per-source
DFS. It also isolates two-direction matching CSR construction, matching and
independent-set stages, and complete chain covers sharing CSR or rebuilding
from arc lists. Inputs are constructed outside the timed regions.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
2026-10-15; same flags and sampling as above. Milliseconds, plain / native.

| closure, plain -O2 ms | sparse 2000/10000 | dense 2000/200000 | chain 2000 | dense p=1/4 5000 | bare chain 5000 | complete 2000 |
|---|---:|---:|---:|---:|---:|---:|
| library | 1.2 | 3.5 | 2.2 | 61.3 | 40.0 | 13.1 |
| by topological bitsets | 1.6 | 8.5 | 3.7 | 271 | 70.5 | 53.1 |
| by topological bitsets skip | 1.5 | 5.2 | 3.7 | 91.5 | 70.7 | 15.2 |
| by topological bitsets nearest first | 1.6 | 6.0 | 3.6 | 139 | 70.1 | 37.8 |
| by floyd warshall | 16.4 | 36.1 | 36.0 | not measured yet | not measured yet | not measured yet |
| by dfs | 15.9 | 150 | 19.8 | not measured yet | not measured yet | not measured yet |

| closure, -O2 -march=native ms | library | by topological bitsets | by topological bitsets skip | by topological bitsets nearest first |
|---|---:|---:|---:|---:|
| sparse n=2000 m=10000 | 1.0 | 1.5 | 1.5 | 1.5 |
| dense n=2000 m=200000 | 3.4 | 8.0 | 5.2 | 6.0 |
| chain with chords n=2000 | 1.9 | 3.7 | 3.7 | 3.6 |
| dense p=1/4 n=2000 | 4.4 | 14.5 | 6.3 | 9.3 |
| sparse m=4n n=2000 | 0.859 | 1.2 | 1.2 | 1.2 |
| bare chain n=2000 | 1.8 | 3.7 | 3.7 | 3.6 |
| dense p=1/4 n=5000 | 60.3 | 253 | 92.7 | 141 |
| sparse m=4n n=5000 | 4.8 | 6.8 | 6.8 | 6.8 |
| bare chain n=5000 | 38.4 | 72.7 | 72.2 | 72.2 |
| complete n=2000 | 13.0 | 43.9 | 15.0 | 35.4 |

Floyd-Warshall native times are 19.5, 48.0 and 48.8 ms on the first three
shapes; DFS takes 15.4, 153 and 20.0 ms respectively.

| chain cover, plain / native ms | library | by arc lists |
|---|---:|---:|
| sparse n=2000 m=10000 | 6.5 / 6.4 | 10.3 / 10.3 |
| dense n=2000 m=200000 | 25.8 / 26.4 | 47.3 / 46.9 |
| chain with chords n=2000 | 13.7 / 13.5 | 30.5 / 30.6 |
| dense p=1/4 n=2000 | 53.2 / 53.6 | 82.1 / 79.5 |
| sparse m=4n n=2000 | 4.1 / 4.1 | 6.8 / 6.8 |
| bare chain n=2000 | 13.3 / 13.3 | 31.0 / 31.0 |
| dense p=1/4 n=5000 | 461 / 465 | 789 / 771 |
| sparse m=4n n=5000 | 27.8 / 26.5 | 49.0 / 49.4 |
| bare chain n=5000 | 233 / 233 | 377 / 382 |
| complete n=2000 | 24.8 / 24.2 | 80.7 / 72.8 |

| stages, -O2 -march=native ms | CSR by scatter | CSR by bucket grouping | matching | independent set |
|---|---:|---:|---:|---:|
| sparse n=2000 m=10000 | 3.9 | 7.0 | 6.6 | 2.2 |
| dense n=2000 m=200000 | 29.4 | 42.6 | 29.2 | 7.6 |
| chain with chords n=2000 | 31.2 | 46.1 | 20.5 | 6.8 |
| dense p=1/4 n=2000 | 30.8 | 44.8 | 55.9 | 6.7 |
| sparse m=4n n=2000 | 2.7 | 4.8 | 4.0 | 1.6 |
| bare chain n=2000 | 32.6 | 47.8 | 21.7 | 6.8 |
| dense p=1/4 n=5000 | 437 | 675 | 453 | 74.6 |
| bare chain n=5000 | 438 | 694 | 247 | 74.6 |
| complete n=2000 | 30.3 | 42.3 | 19.0 | 6.7 |

The Floyd-Warshall and DFS contenders on the remaining shapes are not measured
yet. Reached-target skipping saves a whole bitset-row OR when its target's
finished row is already contained. Shared CSR also avoids constructing the
same closure adjacency for matching and cover separately.

## Pitfalls and notes

A raw-edge path cover and a reachability chain cover answer different questions;
a chain's consecutive vertices need not have a direct edge. Enumerating all
maximum antichains can require exponential output: n/2 disjoint two-vertex
chains already have 2^(n/2) maximum antichains. Per-vertex membership can be
checked by deleting that vertex and everything comparable to it, recomputing,
and comparing sizes.

`atl/experimental/graph/bipartite_matching_variants.hpp` contains
`bipartite_matching_by_hopcroft_karp`, `bipartite_matching_by_kuhn` and
`bipartite_matching_by_kuhn_passes`. Their parameters and result match the
library entry, with the same endpoint/count bounds and unspecified ties.
They are runtime contenders, use explicit DFS stacks, and take O(L+R+m)
storage. The Kuhn forms have O(L+R+Lm) worst-case time; the layered contender
takes O(L+R+m sqrt(L+R)) time. Tests use an independent unit-capacity
max_flow reduction as the cardinality oracle, certify pairs and covers,
and classify edges by force/delete recomputation.

## How it works

The auction phase queues free right vertices. Each bids for a minimum-level
left neighbor, steals its partner and raises that left level by two. Exact
levels are periodically rebuilt from free left vertices. Hopcroft-Karp then
finishes from that partial matching, with full BFS labels and explicit-stack
augmentations. Adjacencies are stable CSR arrays in both directions.

An alternating BFS from free left vertices finds Z. The cover is L minus Z together
with R intersect Z; its complement is independent. Equality of the cover and
matching cardinalities certifies both. Edge roles use one SCC computation in
the alternating residual graph with source/sink arcs for free vertices. An
edge can change status exactly when its endpoints share an SCC; otherwise
its status is fixed by the supplied maximum matching.

DAG covers split each vertex into a left and right copy. A matching becomes
successor/predecessor links with n-matching.size paths. Chain cover applies
that construction to transitive closure: bitset rows fill in reverse
topological order, skipping edges to targets already reached, then emit a
CSR shared by matching and cover. Vertices with both split copies outside
the cover form the maximum antichain. No graph traversal recurses per vertex.

References: König (1931), “Graphen und Matrizen”; Dilworth (1950), “A
Decomposition Theorem for Partially Ordered Sets”; Hopcroft and Karp (1973),
“An n^(5/2) Algorithm for Maximum Matchings in Bipartite Graphs”; Cherkassky,
Goldberg, Martin, Setubal and Stolfi (1998), “Augment or push: a computational
study of bipartite matching and unit-capacity flow algorithms”.

## See also

- [Graphs](graph.md): edge vocabulary and traversal.
- [Maximum flow](max-flow.md): capacity constraints and matching reductions.
- [Assignment](assignment.md): dense weighted matching.
- [Graph components](graph-components.md): SCC decomposition.
