# `atl::link_cut_tree`

An online forest: link, cut and re-root trees, ask connectivity and rooted
questions (lca, parent, depth, k-th ancestor, distance, k-th path vertex), and
fold or lazily update paths over a monoid, each in amortized O(log n). Values
live on vertices, or on edges in the built-in edge tier (dynamic MST, path
max between endpoints). Reach for it when the tree changes (links, cuts or
re-rooting between queries); on a static tree, HLD over a segment tree
(heavy-light path decomposition + [`segment_tree`](segment-tree.md) /
[`lazy_segment_tree`](lazy-segment-tree.md)) is the simpler tool. A client of
the public [`splay_engine`](splay-engine.md).

```cpp
#include <atl/ds/link_cut_tree.hpp>
```

\include link-cut-tree/basic.cpp

Output:

\include link-cut-tree/basic.out

## API

```cpp
template<typename Value = void, typename Op = void, typename Tag = void, typename Mapping = void,
         typename Compose = void, typename FoldOrder = void, typename ValuePlacement = void>
    requires detail::link_cut_algebra<Value, Op, Tag, Mapping, Compose, FoldOrder, ValuePlacement>
          && (std::is_void_v<Value> || std::default_initializable<Value>)
          && (std::is_void_v<Tag> || std::default_initializable<Tag>)
class link_cut_tree;

template<typename Value, typename Op, typename Tag = void, typename Mapping = void,
         typename Compose = void, typename FoldOrder = void>
using edge_link_cut_tree = link_cut_tree<Value, Op, Tag, Mapping, Compose, FoldOrder, edge_values_t>;
```

Tiers are switched on by non-`void` arguments, in `treap_sequence` order with
the `lazy_segment_tree` names: structure only; `Value` + `Op` (`std::copyable`
Value, `binary_operation<Op, Value>`); + `Tag`, `Mapping`, `Compose`
(`action<Mapping, Tag, Value>`, `binary_operation<Compose, Tag>`). Two mode
selectors, each a tag type usable as a template argument and, through its
value, as the leading constructor argument that deduces it (the
`commutative_tags` pattern):

- `FoldOrder`: `void`, or `atl::commutative_fold_t` / `atl::commutative_fold`
  (in `types/algebraic.hpp`, with the `fold_order` concept): the caller
  promises `op(a, b) == op(b, a)`, and each node keeps one aggregate instead
  of one per direction (Measured performance below).
- `ValuePlacement`: `void` (vertex values), or `atl::edge_values_t` /
  `atl::edge_values` (concept `value_placement`): the edge tier, below.

Constructors (O(n)): `link_cut_tree()`; `explicit
link_cut_tree(n)` (structure only); `(n, identity, op = {})` (every vertex
holds `identity`); `(values, identity, op = {})` (vertex tier); `(n, identity,
tag_identity, op = {}, mapping = {}, compose = {})`; `(values, identity,
tag_identity, op, mapping, compose)`; each with a leading `commutative_fold`,
`edge_values` (the `n` forms) or `edge_values, commutative_fold`. Deduction
guides cover every form (`Value` from the identity; a values range must
convert to it without narrowing). `n < 2^32 - 1` (`2^31` on the edge tier) is
asserted.

| member | meaning | time (amortized) |
|---|---|---|
| `make_root(v)` | v becomes its tree's root | O(log n) |
| `link(u, v)` | add edge u - v: u's tree hangs below v (v's root stays) | O(log n) |
| `cut(u, v)` | remove edge u - v: the side with the root keeps it, the other is rooted at its endpoint | O(log n) |
| `cut_from_parent(v)` | remove v's parent edge under the current root; v roots its new tree | O(log n) |
| `connected(u, v)` | same tree? | O(log n) |
| `root(v)`, `parent(v)` | under the current root; `parent` is `none` for a root | O(log n) |
| `lca(u, v)` | lowest common ancestor under the current root, `none` across trees | O(log n) |
| `depth(v)`, `distance(u, v)` | edges from the root / on the path; `distance` is `none` across trees | O(log n) |
| `ancestor(v, k)` | k-th ancestor (`ancestor(v, 0) == v`), `none` past the root | O(log n) |
| `path_vertex(u, v, k)` | k-th vertex on the path u -> v (`k = 0` is u), `none` past v or across trees | O(log n) |
| `at(v)` | value; `Value const &` without tags (O(1)), `Value` with tags | O(1) / O(log n) |
| `set(v, x)`, `update(v, f)` | replace the value by `x` / `f(value)` | O(log n) |
| `fold(u, v)` | fold of the path's values in path order u -> v, endpoints included | O(log n) |
| `try_fold(u, v)` | `std::optional`: the fold when u and v are connected, `std::nullopt` otherwise; one pair of accesses where `connected` + `fold` costs two | O(log n) |
| `apply(u, v, f)` | apply tag f to every value on the path | O(log n) |
| `assign_parents(parents)` | link every v below `parents[v]` (`none` or `v` = root), a parent-array format | O(n) |
| `size()`, `empty()` | vertices | O(1) |

`none` is the class constant `link_cut_tree::none == atl::graph::none`. The
root changes only through `make_root`, `link` and `cut`: `fold` and `apply`
keep it, so rooted queries (lca under a root) and unrooted path queries
share one class. Structural and path queries mutate the auxiliary trees; untagged `at`,
`size`, `empty`, `edge` and `edge_count` are const. Memory per node:
16 bytes of links and size, a flag, and on the value tiers the value, one or
two aggregates and the tag.

### The edge tier

`ValuePlacement = edge_values_t` keeps values on edges with the edge-as-vertex
trick inside the class: each edge is a hidden node between its endpoints,
vertex nodes hold the identity, so a path fold is the fold of the path's edge
values. Edge ids are dense, `[0, size() - 1)`, recycled from a free list (the
most recently freed first); a flat hash map from the endpoint pair finds an
edge by its endpoints.

| member (edge tier) | meaning | time (amortized) |
|---|---|---|
| `link(u, v, w) -> uint32` | add edge u - v holding w (u's tree below v), return its id | O(log n) + expected O(1) |
| `cut(u, v)`, `cut(e)` | remove the edge by endpoints or by id; roots as in the vertex tier | O(log n) |
| `cut_from_parent(v)` | remove v's parent edge | O(log n) |
| `at(e)`, `set(e, w)`, `update(e, f)` | edge values, by id | as above |
| `fold(u, v)`, `apply(u, v, f)` | over the path's edges (the identity when u == v) | O(log n) |
| `edge(u, v)` | the id of edge u - v, `none` if absent | expected O(1) |
| `edge_count()` | live edges | O(1) |
| `assign_parents(parents, weights)` | as above, edge v - parents[v] holding `weights[v]`; ids 0, 1, ... in increasing v | O(n) plus n expected O(1) hash insertions |

The rooted queries (`depth`, `distance`, `ancestor`, `path_vertex`, `parent`,
`lca`) count vertices and edges of the forest, never the hidden nodes.

Non-void `Value` and `Tag` types must satisfy `std::default_initializable`.
Parent-array overloads require `std::ranges::forward_range<Parents const> &&
graph::vertex_range<Parents const>`; weighted overloads retain their input-range
and element-construction requirements for `Weights`.

## Preconditions

Asserted (`-DATL_ASSERT`): ids in range; `link`: u and v in different trees;
`cut(u, v)`: u - v is an edge; `cut(e)`, `at(e)`, `set(e, ...)`: e is a live
edge id; `cut_from_parent(v)`: v is not a root; `fold`, `apply`: u and v
connected; `assign_parents`: every vertex isolated (no edges at all on the
edge tier), exactly `size()` parent entries, and the array is a forest
(O(n) check). Weighted construction requires at least `size()` weights;
additional weights are ignored.

Documented only: the algebra's laws, as for `lazy_segment_tree`: `(Value, op,
identity)` a monoid, `compose(f, g)` = "`g` first, then `f`", `mapping`
distributing over `op`; a width-dependent tag (add to a sum) carries the
width inside `Value`. `commutative_fold` promises a commuting `op`. On the
edge tier with tags, `mapping(f, identity) == identity` (the hidden vertex
nodes hold the identity and receive the tag too): true for affine maps on a
(sum, width) pair, for adds on max with a true minus infinity, not for a
plain "assign c" on a value without width. All intermediate folds, mapped
values and tag compositions must be representable; arithmetic overflow is
not checked.

## Measured performance

`benchmarks/ds/link_cut_tree.cpp` measures complete dependent workloads,
including construction. Contenders change rerooting, size storage, ancestor
pushing or node layout; commutative folding is a public mode. Static contenders use [lca::path](lca.md) over a segment tree for point-add
path sums and direction-aware affine composition, or a lazy segment tree for
affine path updates. They share the dynamic contenders' operation loops and
check answer agreement outside timing. The supplied 2026-10-25 Xeon Platinum
8168 records cover churn and edge workloads; these static contenders are
**not measured yet** for either plain `-O2` or `-O2 -march=native`.

Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-25,
`-O2 -march=native`, N = Q = 5e5. Minimum milliseconds over three runs:

| shape | library | `commutative_fold` | by rerooting | by omitting size | by parity walk | by split layout | by columnar layout |
|---|---|---|---|---|---|---|---|
| random tree | **268** | 227 | 531 | 250 | 258 | 265 | 305 |
| path | **433** | 409 | 511 | 424 | 473 | 404 | 389 |
| star | **57.7** | 50.2 | 111 | 55.1 | 59.4 | 61.5 | 73.3 |
| caterpillar | **708** | 611 | 845 | 697 | 748 | 709 | 711 |
| random churn (cut / link / path sum) | **645** | 571 | 696 | 631 | 641 | 648 | 806 |
| `make_root` + path sum | **789** | 664 | 1052 | 741 | 764 | 792 | 949 |
| point set + affine path composite | **466** | | 790 | 438 | 460 | 447 | 570 |
| lazy affine path update + path sum | **913** | 736 | 1823 | 875 | | 950 | 1172 |
| structure-only churn | **463** | | | 434 | 458 | 469 | 515 |
| edge tier MST by path max | **753** (default order 828) | | 783 | | 757 | 817 | 1157 |

The newer churn and edge records use Xeon Platinum 8168, the same compiler,
build and date. Minimum milliseconds, `-O2 -march=native`, N = Q = 5e5:

| workload | library | commutative fold | by rerooting | by omitting size | by parity walk | by split layout | by columnar layout |
|---|---:|---:|---:|---:|---:|---:|---:|
| random churn | 701 | 606 | 734 | 696 | 674 | 682 | 908 |
| structure-only churn | 510 | | | 448 | 491 | 490 | 540 |
| edge-tier MST, commutative max | 838 | | 898 | | 843 | 910 | 1218 |
| edge-tier MST, default fold order | 908 | | | | | | |

The same native-flag record at N = Q = 2e5:

| workload | library | commutative fold | by rerooting | by omitting size | by parity walk | by split layout | by columnar layout |
|---|---:|---:|---:|---:|---:|---:|---:|
| random churn | 211 | 182 | 232 | 216 | 208 | 209 | 233 |
| structure-only churn | 165 | | | 148 | 158 | 160 | 177 |
| edge-tier MST, commutative max | 254 | | 271 | | 257 | 254 | 316 |
| edge-tier MST, default fold order | 273 | | | | | | |

Plain `-O2`, Xeon Platinum 8168, same compiler, build and date,
N = Q = 2e5, minimum milliseconds:

| workload | library | commutative fold | by rerooting | by omitting size | by parity walk | by split layout | by columnar layout |
|---|---:|---:|---:|---:|---:|---:|---:|
| random churn | 205 | 183 | 223 | 196 | 204 | 195 | 233 |

Commutative folding reduces aggregate work. Rerooting requires extra accesses.
The size field keeps depth and indexed path queries available. Split layout
has no consistent advantage; columnar layout loses the mutation-heavy shapes.
The ancestor stack remains the push strategy.

## Pitfalls and choosing

- On a static tree, contiguous range structures can avoid dynamic-forest restructuring.
- **Commuting folds:** pass `commutative_fold` (sum, min, max, xor) to drop
  the reversed aggregate, reduces aggregate storage and work; non-commuting folds
  (composition of maps, string hashes) need the default.
- **Connectivity-guarded folds:** `try_fold(u, v)` answers "fold if
  connected" with one pair of accesses; `connected(u, v)` then `fold(u, v)`
  costs two.
- `fold(u, v)` is in path order u -> v: `fold(v, u)` is the reversed fold.
- `link(u, v)` re-roots u's tree at u. To keep a designated root, link the
  non-root side first (`link(child, parent)`).
- `at` (untagged tiers) returns a reference into the node pool, which never
  reallocates after construction: it stays valid but follows later `set`,
  `update` and `apply` on that vertex or edge. A cut followed by a link may
  reuse an edge id and overwrite the slot observed by an existing reference.
  Mutations offer no rollback guarantee if allocation or an algebra callable throws.
- Offline edge lifetimes can be handled by [`time_divide_and_conquer`](time-divide-and-conquer.md) with a rollback DSU.
- Subtree folds and sizes (commutative groups): [`subtree_link_cut_tree`](subtree-link-cut-tree.md).

## How it works

Each tree is cut into preferred paths, each stored as a splay tree ordered by
depth; a splay tree's root keeps a path-parent pointer to the tree parent of
its path's top. The store's `is_splay_root(x)` hook is exactly that test (x's
parent does not list it as a child), so the engine's rotations stop at path
boundaries and carry the path-parent pointer along. `expose(x)` climbs from x:
splay each node on the way, make the preceding splay tree its right child (its
old right subtree becomes a separate path hanging by its pointer), and move
to the path-parent; root..x becomes one preferred path. The climb also
returns its top splay root and that root's former right subtree.

**Two-access path operations.** `fold(u, v)` exposes u, then exposes v: the
second climb meets u's path at their lca w, ends with w at the top of the
root path's splay tree with w's right subtree = the path below w to v, and
returns the former right subtree, which is exactly the path from below w down
to u. The fold is `reversed_aggregate(u side) · value(w) · aggregate(v side)`
(node 0 holds the identity, so an empty side folds through), and `apply`
tags the same three parts. No re-rooting happens, so the root and every
rooted query survive. `connected` and `lca` are the same two exposures: u and
v are connected iff u is w or u's splay tree, splayed, hangs below w by its
path-parent pointer. The sizes of the two sides give `distance` and the rank
of the k-th path vertex.

**Re-rooting** is `access(v)` then a reversal mark on v's splay tree. A mark
at x means x's children still owe a swap while x's own aggregates are already
swapped, the [`treap_sequence`](treap-sequence.md) scheme, so a subtree
root's aggregates are always current and `push` settles marks and lazy tags
one level before the engine reads children. `link(u, v)` re-roots at u and
points u at v; `cut` finds which endpoint is the child (the predecessor in
the exposed root path) and splits the child's path off, so the side holding
the root keeps it. `assign_parents` only writes path-parent pointers, O(n).

**Edge tier.** Edge e is node n + 1 + e, linked between its endpoints; vertex
nodes hold the identity, so the vertex-tier machinery folds edge values
unchanged, and the rooted queries divide node counts by two. The hash map's
key packs the sorted endpoint pair into 64 bits, hashed with the seeded
`atl::hash` at run time and an unseeded mix in constant evaluation (which has
no seed and no adversary), so the whole class, edge tier included, is
`constexpr` (proofs in tests/ds/constexpr_checks.cpp).

**Complexity.** The access lemma bounds each splay inside a climb by
3(r(top) − r(x)) + 1 in the virtual-tree potential, which telescopes along a
climb; heavy-light accounting bounds the number of preferred-path changes by
amortized O(log n). Nothing recurses: the engine's handle splays push their
path through a reused buffer, and a 1e6-vertex path is a test.

References: "A data structure for dynamic trees" and "Self-adjusting binary search trees".

## See also

[`splay_engine`](splay-engine.md), [`splay_sequence`](splay-sequence.md),
[`treap_sequence`](treap-sequence.md),
[`lazy_segment_tree`](lazy-segment-tree.md), [`dsu`](dsu.md).
