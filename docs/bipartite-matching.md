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

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | library | by bid scan full | by bid scan level zero | by kuhn passes | by hopcroft karp | by kuhn | by max flow |
|---|---:|---:|---:|---:|---:|---:|---:|
| random L=R=1e5 m=2e5 | 11.2 / 12.6 | 10.8 / 11.8 | 12.7 / 12.8 | 14 / 13.9 | 14.1 / 13.9 | 10.6 / 10.1 | 72.3 / 55.3 |
| band L=R=66000 | 1.7 / 2 | 1.5 / 1.8 | 1.1 / 1.6 | 0.528 / 0.767 | 0.5 / 0.703 | 0.513 / 0.705 | 15.1 / 18.1 |
| line L=R=1e5 | 1.3 / 1.8 | 1.2 / 1.6 | 1.1 / 1.7 | 0.504 / 0.698 | 0.453 / 0.684 | 0.467 / 0.666 | 7.8 / 9.9 |
| augmented cycle L=R=66000 | 26.1 / 31.5 | 24 / 28.5 | 32.3 / 34.4 | 305 / 320 | 303 / 338 | 1220 / 1280 | 547 / 509 |
| skewed L=1e5 R=1e3 | 3 / 3.3 | 3.1 / 3.2 | 2.9 / 3 | 2.3 / 2.4 | 2.4 / 2.4 | 351 / 445 | 22.9 / 23 |
| dense L=R=3000 m=1e6 | 4 / 4.4 | 3.7 / 3.8 | 3 / 3.3 | 1.2 / 1.7 | 1.8 / 2.3 | 1.4 / 1.7 | 56.5 / 56.2 |
| dense L=R=3000 m=3e6 | 20.4 / 23.2 | 19 / 20.2 | 17.8 / 18.7 | 8.2 / 8.4 | 9.3 / 9.6 | 8.3 / 7.8 | 199 / 209 |
| complete L=R=2000 | 14.7 / 18.8 | 12.7 / 16 | 11.8 / 15.3 | 4.7 / 6.9 | 4.8 / 6.4 | 4.8 / 5.4 | 274 / 326 |

The augmented-cycle library row is 303/26.1 = 11.61 times faster than
Hopcroft-Karp plain (338/31.5 = 10.73 native), but loses to the full bid
scan (26.1/31.5 versus 24/28.5 ms). On dense m=3e6 it loses to
Hopcroft-Karp by 20.4/9.3 = 2.19 times plain and 23.2/9.6 = 2.42 native.
The level-zero scan loses to the library on random inputs (12.7/12.8 versus
11.2/12.6 ms) and augmented cycles (32.3/34.4 versus 26.1/31.5 ms), but
beats it on band, line, skewed, dense and complete inputs. Per-pass Kuhn
takes 2.3/2.4 ms on skewed sides versus per-root Kuhn's 351/445 ms; the
library takes 3/3.3 ms. The library also loses to per-root Kuhn on random
inputs (11.2/12.6 versus 10.6/10.1 ms). Band and line favor the pure
matching contenders, as do dense and complete graphs.

### DAG closure and chain covers

`benchmarks/graph/dag_cover.cpp` times reverse-topological bitsets with CSR
output and reached-target skipping against arc output, arc output with
skipping, nearest-target-first ordering, Floyd-Warshall bitsets and per-source
DFS. It also isolates two-direction matching CSR construction, matching and
independent-set stages, and complete chain covers sharing CSR or rebuilding
from arc lists. Inputs are constructed outside the timed regions.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| closure, plain -O2 ms | sparse 2000/10000 | dense 2000/200000 | chain 2000 | dense p=1/4 5000 | bare chain 5000 | complete 2000 |
|---|---:|---:|---:|---:|---:|---:|
| library | 0.812 | 2 | 1.3 | 42.8 | 35 | 5 |
| by topological bitsets | 0.773 | 3.5 | 1.3 | 145 | 57.2 | 20.7 |
| by topological bitsets skip | 0.793 | 2 | 1.3 | 65.4 | 57.7 | 5.5 |
| by topological bitsets nearest first | 0.819 | 2.2 | 1.3 | 80.6 | 57.5 | 12.1 |
| by floyd warshall | 9.1 | 17.8 | 18 | not measured yet | not measured yet | not measured yet |
| by dfs | 9.2 | 92.6 | 7 | not measured yet | not measured yet | not measured yet |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| closure, -O2 -march=native ms | library | by topological bitsets | by topological bitsets skip | by topological bitsets nearest first |
|---|---:|---:|---:|---:|
| sparse n=2000 m=10000 | 0.628 | 0.68 | 0.701 | 0.716 |
| dense n=2000 m=200000 | 1.6 | 3.3 | 1.9 | 2.1 |
| chain with chords n=2000 | 0.94 | 1.2 | 1.1 | 1.2 |
| dense p=1/4 n=2000 | 2 | 6.7 | 2.3 | 3.2 |
| sparse m=4n n=2000 | 0.503 | 0.543 | 0.559 | 0.59 |
| bare chain n=2000 | 0.927 | 1.1 | 1.1 | 1.1 |
| dense p=1/4 n=5000 | 37.3 | 142 | 63.1 | 80.6 |
| sparse m=4n n=5000 | 2.8 | 3.1 | 3.2 | 3.2 |
| bare chain n=5000 | 30.1 | 55.3 | 55.6 | 55.3 |
| complete n=2000 | 4.8 | 20.6 | 5.3 | 12.6 |

Floyd-Warshall native times are 5.9, 9.7 and 9.7 ms on the first three
shapes; DFS takes 8.9, 89.5 and 6.9 ms respectively. Plain sparse closure
favors topological bitsets over the library (0.773 versus 0.812 ms); native
favors the library (0.628 versus 0.68 ms).

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| chain cover, plain / native ms | library | by arc lists |
|---|---:|---:|
| sparse n=2000 m=10000 | 3.8 / 3.6 | 4.7 / 4.6 |
| dense n=2000 m=200000 | 14.3 / 13.9 | 21.4 / 21.3 |
| chain with chords n=2000 | 5.9 / 5.5 | 12 / 12.2 |
| dense p=1/4 n=2000 | 33 / 33 | 42.9 / 43.4 |
| sparse m=4n n=2000 | 2.5 / 2.4 | 3.2 / 3.1 |
| bare chain n=2000 | 5.9 / 5.6 | 11.4 / 11.3 |
| dense p=1/4 n=5000 | 269 / 264 | 456 / 448 |
| sparse m=4n n=5000 | 15.1 / 13.9 | 22.9 / 23.2 |
| bare chain n=5000 | 115 / 108 | 222 / 217 |
| complete n=2000 | 9.5 / 9.4 | 31.1 / 31.9 |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| stages, -O2 -march=native ms | CSR by scatter | CSR by bucket grouping | matching | independent set |
|---|---:|---:|---:|---:|
| sparse n=2000 m=10000 | 1.1 | 1.9 | 3.2 | 0.701 |
| dense n=2000 m=200000 | 7.1 | 9.2 | 14.3 | 2.2 |
| chain with chords n=2000 | 7.6 | 15.9 | 8.2 | 1.6 |
| dense p=1/4 n=2000 | 7.8 | 12.9 | 34.1 | 1.6 |
| sparse m=4n n=2000 | 0.807 | 1.3 | 1.9 | 0.536 |
| bare chain n=2000 | 7.3 | 12.2 | 8.4 | 1.7 |
| dense p=1/4 n=5000 | 124 | 253 | 271 | 34.5 |
| bare chain n=5000 | 124 | 251 | 128 | 34 |
| complete n=2000 | 7.7 | 12.6 | 8.6 | 1.6 |

`benchmarks/graph/dag_cover.cpp` also measures Floyd-Warshall and DFS on the
additional n=2000 shapes; n=5000 remains unmeasured for these contenders.
Reached-target skipping saves a whole bitset-row OR when its target's
finished row is already contained. Shared CSR also avoids constructing the
same closure adjacency for matching and cover separately.

Random, skewed and dense matching comparisons include rows that vary
between runs (spreads up to 1.58x); the quoted minima alone do not establish
stable rankings there. Several chain-cover and stage rows also vary,
with spreads up to 1.60x.

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
