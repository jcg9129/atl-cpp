# `atl::top_tree`

A dynamic forest (link, cut, re-root) where each tree is summarized by a
cluster algebra you supply, so one structure answers path folds and lazy path
updates, subtree folds and lazy subtree updates **without an inverse** (min,
max, gcd), and whole-tree shape aggregates (diameter, farthest vertex, sum of
distances to marked vertices), each in amortized O(log n). Reach for it when
a problem needs two of these at once or a non-invertible subtree aggregate on
a changing tree, including dynamic diameter and marked-distance sums
under link and cut. For one kind of
query alone, a narrower structure is simpler and probably faster (chooser
below). Four ready-made cluster sets (`atl::top_clusters`) cover sums,
extremes, diameter and marked distances with named members. A client of the
public [`splay_engine`](splay-engine.md).

```cpp
#include <atl/ds/top_tree.hpp>
```

\include top-tree/basic.cpp

Output:

\include top-tree/basic.out

The ready-made algebras are declared in:

```cpp
#include <atl/ds/top_clusters.hpp>
```

## API

### The cluster algebra

A **path cluster** (`Path`) summarizes a connected piece of a tree with two
boundary vertices, top and bottom: its *spine* (the vertices from top to
bottom) and everything hanging off the spine. A **point cluster** (`Point`)
summarizes a set of subtrees hanging at one vertex, the vertex excluded.
Every tree is built from one-vertex clusters by two merges, so the user
supplies:

| Slot | Signature | Meaning | Law (documented, not checked) |
|---|---|---|---|
| `Value` | type | a vertex's own datum | |
| `Path` | type, identity passed | a path cluster | |
| `Point` | type, identity passed | a hanging set | |
| `Compress` | `Path(Path const &upper, Path const &lower)` | join two path clusters, `upper`'s bottom adjacent to `lower`'s top | associative; the path identity two-sided |
| `Rake` | `Point(Point const &, Point const &)` | merge two hanging sets at one vertex | associative AND commutative (light children are kept in no order); the point identity two-sided |
| `AddVertex` | `Path(Point const &light, Value const &v)` | v over its hanging set, as a one-vertex path | the result equals its own reversal |
| `AddEdge` | `Path` -> `Point` | a child's cluster (top = the child) seen from its parent | |
| `Reverse` | `void` / `commutative_fold_t` / `Path(Path const &)` | how a cluster's ends swap under re-rooting | a callable must reverse exactly |

`Reverse = void` (default) keeps a reversed cluster per node (two more
`compress` calls per pull; always correct). `commutative_fold_t` declares
clusters symmetric (sums, extremes: nothing kept, nothing done). A callable
swaps the ends in O(1) when the cluster keeps both directions anyway (the
diameter set swaps its farthest-from-top and farthest-from-bottom entries).

**Lazy tier** (a non-`void` `Tag`): `Mapping(Tag, Value) -> Value` as in
[`lazy_segment_tree`](lazy-segment-tree.md); `MapPath(Tag const &spine, Tag
const &light, Path) -> Path` applies `spine` to every spine vertex and
`light` to everything hanging off; `MapPoint(Tag, Point) -> Point`;
`Compose(f, g)` = "`g` first, then `f`". Each node keeps a pending (spine,
light) pair: a path update is `(f, identity)`, a subtree update `(f, f)`, so
path and subtree updates that do not commute (assign vs add, affine maps)
mix correctly. Laws: every map distributes over each merge,
`MapPath(s, l, add_vertex(P, v)) == add_vertex(MapPoint(l, P), Mapping(s, v))`
and `MapPoint(t, add_edge(C)) == add_edge(MapPath(t, t, C))`.

```cpp
template<typename Value, typename Path, typename Point, typename Compress, typename Rake, typename AddVertex,
         typename AddEdge, typename Reverse = void, typename Tag = void, typename Mapping = void,
         typename MapPath = void, typename MapPoint = void, typename Compose = void,
         typename ValuePlacement = void>    // void or edge_values_t
    requires detail::top_tree_algebra<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse,
                                     Tag, Mapping, MapPath, MapPoint, Compose, ValuePlacement>
class top_tree;
```

Value, Path, Point and enabled Tag types are copyable.
Concepts (atl/types/algebraic.hpp): `binary_operation` (`Compress`, `Rake`,
`Compose`), `top_vertex_operation<AddVertex, Point, Value, Path>`,
`top_edge_operation<AddEdge, Path, Point>`, `action` (`Mapping`, `MapPoint`),
`path_action<MapPath, Tag, Path>`; the lazy slots are all `void` or all set.

### Construction

| Constructor | |
|---|---|
| `top_tree(n, value, path_identity, point_identity, compress, rake, add_vertex, add_edge)` | n isolated vertices holding `value`; callables default-constructed when omitted |
| `... add_edge, reverse)` | with a callable `Reverse` |
| `... add_edge, [reverse,] tag_identity, mapping, map_path, map_point, compose)` | lazy tier |
| `top_tree(values, path_identity, point_identity, ...)` | one vertex per element (moved out of an owning rvalue range) |
| `top_tree(commutative_fold, ...)`, `top_tree(edge_values, ...)`, `top_tree(edge_values, commutative_fold, ...)` | leading tags deduce `Reverse` / `ValuePlacement` |
| `top_tree(n)`, `top_tree(values)` | ready-made sets only: identities from the types |
| `top_tree()` | the empty forest (default-constructible callables) |

Every form deduces from lambdas (CTAD). Vertex ids are `uint32` in
`[0, size())`; `none == graph::none`.

### Members (amortized O(log n) unless noted)

| Member | Meaning |
|---|---|
| `make_root(v)` | re-root v's tree at v |
| `link(u, v)` / `link(u, v, w) -> edge id` | add edge u - v (u's tree hangs below v; v's root stays); the edge tier takes the edge's value |
| `cut(u, v)`, `cut(e)`, `cut_from_parent(v)` | remove an edge; the side holding the root keeps it, the other is rooted at its endpoint |
| `connected(u, v)`, `root(v)`, `parent(v)`, `lca(u, v)`, `depth(v)`, `distance(u, v)` | as [`link_cut_tree`](link-cut-tree.md) (`none` across trees) |
| `at(id)`, `set(id, x)`, `update(id, f)` | a vertex value (an edge value on the edge tier); `at` is O(1) by reference without tags |
| `at_vertex(v)`, `set_vertex(v, x)`, `update_vertex(v, f)` | a vertex node's value (edge tier; on the vertex tier the same as above) |
| `Path fold(u, v)` | the tree as a path cluster with top u, bottom v: spine u..v plus everything hanging off it |
| `Path subtree_fold(v)` | v's subtree under the current root as a one-vertex cluster at v |
| `Path subtree_fold(v, p)` | v's side of the edge {v, p}, any root |
| `Path component_fold(v)` | v's whole tree with boundary (root, v) |
| `apply(u, v, f)`, `subtree_apply(v, f)`, `subtree_apply(v, p, f)`, `component_apply(v, f)` | lazy tier: the same regions |
| `assign_parents(parents[, weights])` | O(n) bulk link from a parent array (every vertex isolated) |
| `edge(u, v)`, `edge_count()` | edge tier: id of u - v (`none` if absent), expected O(1); edge count O(1) |
| `size()`, `empty()` | O(1) |

Path and edge-side operations keep the root without re-rooting: `fold(u, v)`
and `apply(u, v, f)` take two accesses that meet at the lca (as
[`link_cut_tree`](link-cut-tree.md) does), the root side entering the result
as a cluster hanging off the lca; the edge-side forms with p a child of v take
one access and read everything but p's side the same way. The by-rerooting contender instead re-roots at one end and restores it.

### Ready-made sets (`atl::top_clusters`, atl/ds/top_clusters.hpp)

Library value, cluster and tag types with their merges; identities are the
types' default values. A `top_tree` over a set gets its named members and
the `(n)` / `(values)` constructors.

| Alias | Set | Named members |
|---|---|---|
| `sum_top_tree<T>` | `top_clusters::sum<T>`: spine and hanging (sum, count) apart; tag = add | `path_sum`, `subtree_sum(v[, p])`, `component_sum`, `subtree_size(v[, p])`, `component_size`, `path_add`, `subtree_add(v[, p], x)` |
| `extremes_top_tree<T>` | `top_clusters::extremes<T>`: (sum, min, max, count) apart; tag = assign-then-add | the sum members plus `path_min/max`, `subtree_min/max(v[, p])`, `component_min/max`, `path_assign`, `subtree_assign(v[, p], x)` |
| `diameter_top_tree<W>` (edge tier) | `top_clusters::diameter<W>`: spine length, farthest vertex from each end, longest path; `Reverse` swaps the ends | `diameter(v)`, `diameter_endpoints(v)`, `eccentricity(v)`, `farthest(v)` (`farthest_vertex{distance, vertex}`), `path_length(u, v)` |
| `marked_distance_top_tree<W>` (edge tier) | `top_clusters::marked_distance<W>`: spine length, marked count, distance sums to each end | `set_marked(v, b)`, `is_marked(v)`, `marked_distance_sum(v)`, `marked_count(v)` |

The result and cluster records expose these fields. Unless
listed otherwise, `{}` is the identity; counts are `atl::uint32`.

| Type in `atl::top_clusters` | Fields / default identity |
|---|---|
| `sum_aggregate<T>` | `T sum{}`, `uint32 count{}` |
| `sum_path<T>` | `sum_aggregate<T> spine`, `light`; `total()` combines them |
| `sum_tag<T>` | `T add{}`; `add_of(T)` constructs an add tag |
| `extremes_aggregate<T>` | `T sum{}`, `min = numeric_limits<T>::max()`, `max = numeric_limits<T>::lowest()`, `uint32 count{}` |
| `extremes_path<T>` | `extremes_aggregate<T> spine`, `light`; `total()` combines them |
| `extremes_tag<T>` | `bool is_assign = false`, `T assigned{}`, `add{}`; `add_of(T)` and `assign_of(T)` construct tags |
| `farthest_vertex<W>` | `W distance{}`, `uint32 vertex = graph::none` |
| `diameter_span<W>` | `W length{}`, `uint32 first = graph::none`, `second = graph::none` |
| `diameter_value<W>` | `W length{}`, `uint32 vertex = graph::none`; implicit constructor from an edge length, `of_vertex(uint32)` for a zero-length vertex |
| `diameter_path<W>` | `W length{}`, `farthest_vertex<W> from_top`, `from_bottom`, `diameter_span<W> diameter` |
| `diameter_point<W>` | `farthest_vertex<W> deepest`, `diameter_span<W> diameter` |
| `marked_value<W>` | `W length{}`, `bool is_marked = false`; implicit constructor from an edge length |
| `marked_path<W>` | `W length{}`, `top_sum{}`, `bottom_sum{}`, `uint32 count{}` |
| `marked_point<W>` | `W sum{}`, `uint32 count{}` |

These records have defaulted equality. The sets expose `value_type`,
`path_type`, `point_type`, `tag_type` and `reverse`, and the function objects
`compress`, `rake`, `add_vertex`, `add_edge`; tagged sets also expose
`mapping`, `map_path`, `map_point`, `compose`. Their call signatures are the
slots above. `is_ready_made<Path>` recognizes their path types. The sum set
requires regular values constructible from a vertex count and closed under
addition and multiplication; extremes also requires ordering and numeric
limits; the distance sets require ordered summable lengths.

The sets also expose these static constexpr arithmetic helpers:

| Set | Signature | Result |
|---|---|---|
| `sum<T>` | `point_type merged(point_type const &a, point_type const &b)` | add sums and counts |
| `sum<T>` | `point_type mapped(T const &add, point_type const &a)` | add `add * count` to the sum |
| `extremes<T>` | `aggregate merged(aggregate const &a, aggregate const &b)` | combine sum, extrema and count |
| `extremes<T>` | `aggregate mapped(tag_type const &f, aggregate const &a)` | map assign/add, preserving the empty aggregate |
| `diameter<W>` | `farthest_vertex<W> farther(farthest_vertex<W> const &a, farthest_vertex<W> const &b)` | farther present candidate, choosing a on equal distances |
| `diameter<W>` | `farthest_vertex<W> shifted(farthest_vertex<W> const &c, W const &by)` | increase a present candidate's distance |
| `diameter<W>` | `diameter_span<W> longer(diameter_span<W> const &a, diameter_span<W> const &b)` | longer present path, choosing a on equal lengths |
| `diameter<W>` | `diameter_span<W> joined(farthest_vertex<W> const &a, farthest_vertex<W> const &b)` | join present candidates through their common attachment |

These take O(1) scalar operations and obey the same representability
preconditions as their enclosing sets.

On the edge-tier sets an edge length converts implicitly to the node value,
so `link(u, v, w)` and `set(e, w)` take a bare `W`.

## Complexity

Every operation is amortized O(log n) splay steps plus O(1) calls of the user
callables per step; `fold` / `apply` take two accesses, the edge-side forms
one access plus two splays. `assign_parents` is O(n) on the vertex tier and adds n expected O(1)
hash insertions on the edge tier. Memory: one
compress node per vertex (and per edge on the edge tier) holding links, the
value, one or two `Path`s and two tags; one rake node per light child
(fewer than the total number of vertex and live edge nodes) holding links, a `Point` and a tag.

Every parent-array overload requires
`std::ranges::forward_range<Parents const> && graph::vertex_range<Parents const>`.
Weighted overloads also require an input range of weights whose references
can construct `Value`.

## Preconditions

Asserted (`-DATL_ASSERT`): ids in range (`n < 2^31`, `2^30` on the edge
tier); `link`: u and v in different trees; `cut(u, v)`: an edge; `cut(e)`,
`at(e)`, `set(e, ...)`: a live edge id; `cut_from_parent(v)`: v is not a
root; `fold`, `apply`: u and v connected; `subtree_fold(v, p)`,
`subtree_apply(v, p, f)`: {v, p} is an edge; `assign_parents`: every vertex
isolated (no edges on the edge tier), `size()` entries, a forest (O(n)
check); weighted construction consumes the first `size()` weights and
ignores further entries.

Documented only: the laws in the algebra table (associativity, `rake`
commutativity, a one-vertex cluster equal to its reversal, distributivity of
the tag maps); a callable `Reverse` reverses exactly. Ready-made sets: sums
stay in range for `T` / `W` (an add tag multiplies by a vertex count, the
marked set multiplies lengths by marked counts); the distance sets keep
vertex nodes at length 0. Every intermediate sum, product and tag composition
must be representable; signed overflow is not checked.

## Measured performance

Measurements using `benchmarks/ds/top_tree.cpp`, 2026-10-25, Xeon Platinum 8168
(other records of the dynamic forests ran on a Xeon Platinum 8280: compare
rows only within one record), asserts off, min ms over 3 runs,
`-march=native -O2` / plain `-O2`, N = Q = 5e5 (the record also has 2e5).
Euler-tour measurements below use the by-splay contender. The benchmark
reference uses the library treap backend, so those rows need remeasurement.
Static contenders use [lca::path](lca.md), a lazy segment tree for path
addition and sums, and a direction-aware segment tree for affine composition.
They share the dynamic contenders' operation loops and verify answer agreement
outside timing. These restored contenders are **not measured yet** in the
supplied record for either flag set.

| shape | library top_tree | link_cut_tree | other |
|---|---|---|---|
| path add + path sum, random tree | 2175 / 2172 | 558 / 542 | |
| path add + path sum, path | 824 / 864 | 588 / 562 | |
| path add + path sum, star | 1672 / 1672 | 106 / 106 | |
| path add + path sum, caterpillar | 2172 / 2198 | 1106 / 1050 | |
| path add + path sum, binary | 2287 / 2321 | 781 / 781 | |
| set + affine path composite (library reversal; user `Reverse` 1576 / 1551) | 1514 / 1496 | 507 / 507 | |
| churn + vertex add + subtree sum (`sum_top_tree`) | 1896 / 1906 | | `subtree_link_cut_tree` 844 / 850; splay ETT 1280 / 1247 |
| churn + subtree add + subtree sum (`sum_top_tree`) | 2298 / 2344 | | splay ETT 2373 / 2341 |
| mixed path/subtree mix (`extremes_top_tree`) | 2656 / 2683 | | |
| weight set + diameter, random / path / star / caterpillar | 2952 / 2938, 1826 / 1806, 2100 / 2100, 2204 / 2160 | | |
| edge replacement churn + diameter (`diameter_top_tree`) | 4311 / 4288 | | |
| marked toggles + distance sums (`marked_distance_top_tree`) | 1719 / 1699 | | |
| structure only: assign_parents + churn | 886 / 886 | 419 / 419 | splay ETT 817 / 817 |
| structure only: N - 1 links | 465 / 465 | 3.9 / 3.9 | splay ETT 423 / 423 |

Native-flag contender comparison from the same source, CPU, compiler,
build and date, minimum milliseconds:

| workload | N = Q | library | by per-splay push | by edge leaves |
|---|---:|---:|---:|---:|
| diameter, random tree | 2e5 | 939 | 975 | 978 |
| diameter, random tree | 5e5 | 2952 | 3045 | 3053 |
| diameter, path | 2e5 | 558 | 603 | 771 |
| diameter, path | 5e5 | 1826 | 1937 | 2489 |
| diameter, star | 2e5 | 606 | 599 | 625 |
| diameter, star | 5e5 | 2100 | 2087 | 2057 |
| diameter, caterpillar | 2e5 | 685 | 732 | 789 |
| diameter, caterpillar | 5e5 | 2204 | 2321 | 2540 |
| diameter, edge churn | 2e5 | 1375 | | 1291 |
| diameter, edge churn | 5e5 | 4311 | | 4114 |
| marked toggles and distance sums | 2e5 | 476 | | 448 |
| marked toggles and distance sums | 5e5 | 1719 | | 1617 |

The `by per-splay push` contender loses the path-sum path at 5e5:
975 / 975 ms versus library 824 / 864 ms. The `by edge leaves` contender
loses that diameter path at 2489 / 2458 ms versus library 1826 / 1806 ms,
but wins edge-replacement churn at 4114 ms (`-march=native`) versus 4311 ms with the same flags,
and marked distances at 1617 / 1647 ms versus 1719 / 1699 ms.
Interleaved layout, rake append, rerooting and cached vertex clusters are
not measured in the supplied x86-64 record; the benchmark includes all of them.

## Pitfalls and choosing

| Need on a changing forest | Take |
|---|---|
| path folds / lazy path updates only | [`link_cut_tree`](link-cut-tree.md): 1.40-4.01x faster on non-star path sums, 15.77x on stars; static HLD comparison not measured yet |
| subtree folds over a commutative **group** (sum, xor), point updates, maybe path folds | [`subtree_link_cut_tree`](subtree-link-cut-tree.md): vertex add + subtree sum 850 ms vs 1906 for the top tree (and 1247 for the splay ETT) |
| subtree / component folds and lazy subtree updates, no paths | [`euler_tour_tree`](euler-tour-tree.md) for component and edge-side operations |
| path AND subtree updates together, a non-invertible subtree fold (min, max) with paths or re-rooting, or a whole-tree shape aggregate (diameter, farthest vertex, distance sums) | **`top_tree`** |

Use the top tree only for what the others cannot do. A top tree pull calls
`add_vertex` and two to four `compress`es, a rake pull two `rake`s and one
`add_edge`, so where a narrower structure applies it is slower: the table
above puts it 1.40-15.77x behind `link_cut_tree` on path sums and 2.24-2.25x behind
`subtree_link_cut_tree` on vertex add + subtree sum; plain connectivity
churn costs it 2.11x the link-cut tree's time.

- `fold(u, v)` returns the whole tree as a cluster: read the spine part for
  a path query (`spine.sum` in the sum set), the whole for a component one.
- `rake` must be commutative: light children move between rake trees in no
  particular order, so a non-commutative rake gives decomposition-dependent
  answers (the tests' algebra oracle detects it).
- `subtree_fold(v, p)` with p a child of v also splays p's path and its rake
  node; prefer the rooted `subtree_fold(v)` when the root is fixed.
- Untagged `at` references remain valid until assignment or destruction.
  Edge ids are recycled after cuts, so a reference to an edge slot can later
  observe a different edge. Mutating operations offer no rollback guarantee
  if allocation or an algebra callable throws.
- The ready-made distance sets break ties arbitrarily: `diameter_endpoints`
  and `farthest` return some optimal vertex.
- A static tree with these queries: a static top tree (rerooting DP with
  point updates) is a TODO; meanwhile this class works on a static tree too.

## How it works

The vertex-cluster top tree: a link-cut tree whose nodes also own **rake
trees**. Each tree is cut into preferred paths. A **compress tree** is a
splay tree of one preferred path's vertices in depth order; its subtree at a
node x summarizes x's segment of the path *and everything hanging off it*:
`path(x) = compress(compress(path(left), own(x)), path(right))` with
`own(x) = add_vertex(point(mid(x)), value(x))`. The non-preferred (light)
children of a vertex form its **rake tree**, a splay tree with one rake node
per light child, whose `mid` points at the compress root of that child's
path: `point(y) = rake(rake(point(left), add_edge(path(mid(y)))),
point(right))`. A compress root's parent pointer is the rake node holding it
(0 for a tree's root path), a rake root's parent the vertex owning the rake
tree. Rake ids carry the top bit, so `is_splay_root` is one bit test per
kind, and both kinds are separate `splay_store` views over their own pools
(split layout: links apart from payloads); index 0 of each pool holds the
identity clusters, so pulls never branch on empty children.

**Access.** `expose(x)` first pushes the whole ancestor chain top-down across
both kinds of trees through the engine's shared ancestor buffer (a compress
node's tags reach its rake tree, a rake node's its child's path, so the
engine's per-tree push pass cannot do it), then splays without re-pushing
(the engine's `splay_pushed`). x's lower path becomes a new light child (a
rake node inserted at x's rake root). Then, while x's compress tree hangs
from a rake node y owned by z: splay y in z's rake tree and z in its compress
tree; z's lower path takes y's place as a light child (or y is dropped and
its rake halves joined when z had none), and x's path becomes z's preferred
continuation. After the climb the root path's compress tree is the root..x
path and its root cluster is the whole tree with boundary (root, x). Re-rooting
flips that compress tree (a mark, swapping the stored clusters or calling
`Reverse`); a flip never enters rake trees, whose orientation is fixed by
their attachment. `link` appends u's re-rooted path as v's preferred
continuation (no rake node); `cut` splits a child's path off; lca, depth and
distance are the link-cut tree's two-exposure tricks on compress-tree sizes.

**Queries.** `component_fold(v)` is the root cluster after `access(v)`;
`subtree_fold(v)` is `own(v)` after `access(v)` (all of v's children are
then light). `fold(u, v)` exposes u, then v: the second climb meets u's
root path at w = lca(u, v), leaves w's right subtree the path down to v and
hangs u's segment as a light child of w (made w's rake root), so the answer is
`compress(compress(reverse(u segment), add_vertex(w's other light children +
add_edge(reverse(root side)), value(w))), path(below w))`, the root side being
w's left subtree read upwards. With p a child of v, `subtree_fold(v, p)` is
the same one-vertex cluster at v without p's rake node.

**Tags** are a (spine, light) pair per compress node and one tag per rake
node: a path update tags u's segment and w's lower path with `(f, identity)`
and maps w's value; a subtree update maps v's value and tags v's rake tree
(the edge-side complement also tags the root side with `(f, f)`). Since the
pair composes componentwise, any tag monoid works.

**Edge tier.** Edge e is compress node n + 1 + e, linked between its
endpoints (the [`link_cut_tree`](link-cut-tree.md) edge-as-vertex trick and
its `(u, v) -> id` registry, shared in detail/dynamic_forest.hpp); vertex
nodes hold user values that `add_vertex` sees.

**Complexity.** The self-adjusting top tree analysis
bounds the splays of an access, across compress and rake trees, by amortized
O(log n); `assign_parents` builds balanced rake trees in children-first
(topological) order. Nothing recurses: chains are pushed through a buffer, and a
1e6-vertex path and star are tests.

References: "Self-adjusting top trees" and "A data structure for dynamic trees".

## See also

[`link_cut_tree`](link-cut-tree.md), [`subtree_link_cut_tree`](subtree-link-cut-tree.md),
[`euler_tour_tree`](euler-tour-tree.md), [`splay_engine`](splay-engine.md),
[`lazy_segment_tree`](lazy-segment-tree.md).
