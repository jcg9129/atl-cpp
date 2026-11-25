# `atl::euler_tour_tree`

An online unrooted forest: link and cut edges, ask connectivity and component
sizes, and fold or lazily update over a commutative monoid either a whole
component or the subtree hanging off one side of an edge, each in expected
O(log n). Subtree *updates* are what it adds over
[`link_cut_tree`](link-cut-tree.md); it has no path queries, and its
`connected` is slower than the link-cut tree's (see the chooser below). It
is also the building block of fully dynamic graph connectivity. The tour
lives in a treap with parent links. The contender
`atl::experimental::euler_tour_tree_by_splay` stores tours in splay trees.
It loses churn, links and vertex-add shapes; subtree add and bulk build
are close (Measured performance).

```cpp
#include <atl/ds/euler_tour_tree.hpp>
```

The header `<atl/ds/euler_tour_tree_core.hpp>` provides the tree implementation with narrow dependencies.

\include euler-tour-tree/basic.cpp

Output:

\include euler-tour-tree/basic.out

## API

```cpp
template<typename Value = void, typename Op = void, typename Tag = void, typename Mapping = void,
         typename Compose = void>
    requires detail::euler_tour_algebra<Value, Op, Tag, Mapping, Compose>
          && (std::is_void_v<Value> || std::default_initializable<Value>)
          && (std::is_void_v<Tag> || std::default_initializable<Tag>)
class euler_tour_tree;
```

Tiers are switched on by non-`void` arguments, as in `link_cut_tree`:
structure only; `Value` + `Op` (`std::copyable` Value, `binary_operation<Op,
Value>`); + `Tag`, `Mapping`, `Compose` (`action<Mapping, Tag, Value>`,
`binary_operation<Compose, Tag>`). There is no fold-order selector: `op`
must commute in every tier (the tour's order is not the tree's).

Constructors (O(n)): `euler_tour_tree()`; `explicit euler_tour_tree(n)`
(structure only); `(n, identity, op = {})`; `(values, identity, op = {})`;
`(n, identity, tag_identity, op = {}, mapping = {}, compose = {})`;
`(values, identity, tag_identity, op, mapping, compose)`. Deduction guides
cover every form (`Value` from the identity; a values range must convert to
it without narrowing; an rvalue range that owns its elements is moved from).
`n <= (2^32 - 1) / 3` is asserted (the tour has up to `3n - 2` nodes).

| member | meaning | time (expected) |
|---|---|---|
| `link(u, v)` | add edge u - v | O(log n) + expected O(1) |
| `cut(u, v)` | remove edge u - v (either orientation) | O(log n) + expected O(1) |
| `connected(u, v)` | same tree? | O(log n) |
| `component_size(v)` | vertices in v's tree | O(log n) |
| `component_fold(v)` | fold of every value in v's tree | O(log n) |
| `component_apply(v, f)` | apply tag f to every value in v's tree | O(log n) |
| `subtree_fold(v, parent)` | fold over v's side of edge v - parent (v's subtree if the tree were rooted beyond `parent`) | O(log n) + expected O(1) |
| `subtree_apply(v, parent, f)` | apply tag f over that side | O(log n) + expected O(1) |
| `at(v)` | value; `Value const &` without tags (O(1)), `Value` with tags | O(1) / O(log n) |
| `set(v, x)`, `update(v, f)` | replace the value by `x` / `f(value)` | O(log n) |
| `assign_parents(parents)` | link every v to `parents[v]` (`none` or `v` = no edge), a parent-array format, building each tour as one treap | O(n) + n hash insertions |
| `size()`, `empty()` | vertices | O(1) |

The forest is unrooted: `parent` in `subtree_*` only names which side of
the edge is meant, nothing is re-rooted, and no query changes the trees.
Queries are not `const` (except `size`, `empty`): they reuse member scratch
for root paths and, on the lazy tier, settle the tags above the nodes they
read. Below the lazy tier a query writes no node. Memory: one node per
vertex and two per possible edge, 3n - 2 nodes allocated up front, each
16 bytes of child links, size and priority plus, on the value tiers, the
value, its subtree fold and the pending tag, and a 4-byte parent link per
node in an array of its own. The bounds are expected over the treap
priorities, drawn from a per-process seed (`atl::hash`'s), so an adversary
has no public priority seed to target; these are probabilistic, not worst-case bounds.

Non-void `Value` and `Tag` types must satisfy `std::default_initializable`.
Parent-array overloads require `std::ranges::forward_range<Parents const> &&
graph::vertex_range<Parents const>`; weighted overloads retain their input-range
and element-construction requirements for `Weights`.

## Preconditions

Asserted (`-DATL_ASSERT`): ids in range; `link`: u and v in different trees
(an O(log n) `connected` check); `cut(u, v)`, `subtree_*(v, parent)`: the
edge exists; `assign_parents`: no edges yet, exactly `size()` entries, and
the array is a forest (O(n) check).

Documented only:

- `(Value, op, identity)` is a **commutative** monoid. Sum, min, max, xor,
  gcd qualify; a composition of maps does not.
- `compose(f, g)` = "`g` first, then `f`", and `mapping` distributes over
  `op`, the `lazy_segment_tree` laws.
- **`mapping(f, identity) == identity` for every tag.** Arc nodes hold the
  identity and sit inside every tour segment, so they receive every tag.
  Hence the *width convention*: a tag whose effect depends on how many
  vertices it covers carries that count inside `Value`. Add on a sum is
  `{sum, count}` with identity `{0, 0}` and vertices `{a, 1}`,
  `mapping(add c, {s, k}) = {s + c k, k}`; assign on a max is `{max, count}`
  with `mapping(assign c, {m, k}) = k > 0 ? {c, k} : {m, k}`. A plain
  "assign c" on a value without width, or add on max with a finite minus
  infinity that overflows, breaks the law.
- Every intermediate aggregate, mapped value and tag composition must stay
  representable in the chosen types; arithmetic overflow is not checked.

## Measured performance

N = Q = 2e5. The contenders
use a parent-linked treap (library) or a splay tree (by splay). The static `subtree_sum_by_preorder_fenwick` contender uses [lca preorder](lca.md)
with a Fenwick tree instead of a dynamic tour. It uses the same operation loop
and untimed checksum agreement checks.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | library | by splay | reference |
|---|---|---|---|
| subtree add + subtree sum under edge replacement | 564 / 538 | 656 / 620 | |
| vertex add + subtree sum under edge replacement | 244 / 230 | 272 / 255 | `subtree_link_cut_tree` 205 / 206 |
| static vertex add + subtree sum | 119 / 117 | 188 / 176 | by preorder Fenwick 17.7 / 18.5 |
| build: `assign_parents` (random tree) | 27.6 / 26.2 | 22 / 22.3 | |
| build: N - 1 `link` calls | 62.7 / 63.8 | 92.4 / 94 | `link_cut_tree<>` 0.632 / 0.628 |
| `assign_parents` + churn (cut / link / connected thirds) | 122 / 121 | 150 / 151 | `link_cut_tree<>` 104 / 107 |

The treap beats splay on churn (122/121 versus 150/151 ms), but loses bulk
`assign_parents` construction (27.6/26.2 versus 22/22.3 ms). Its static
subtree workload is 188/119 = 1.58 times faster plain and 176/117 = 1.50
native than splay, but loses to preorder Fenwick (119/117 versus
17.7/18.5 ms). Under edge replacement, vertex-add/subtree-sum loses to
subtree link-cut (244/230 versus 205/206 ms). Link-cut also wins churn
(104/107 versus 122/121 ms). These comparisons measure different stores;
the tables do not isolate the cause of each difference.

The `assign_parents`, subtree and churn comparisons include rows that vary
between runs (up to 1.60x). Their ratios describe minima and do not by
themselves establish stable rankings.

## Pitfalls and choosing

- **euler_tour_tree vs link_cut_tree.** Take the Euler tour tree for
  subtree updates under link and cut, component-wide folds and updates, and
  component sizes. Take the link-cut tree for anything on paths (path folds
  and updates, lca, depth, distance), for non-commutative folds, for rooted
  problems, and for pure connectivity: an ETT tour has three nodes per
  vertex in one tree per component, while the link-cut tree's splay trees
  are short preferred paths, and its cut / link / connected churn takes 104/107 ms versus
  122/121 ms for the tour above, or 1.17/1.13 times as fast
  (plain/native). For vertex add + subtree sum (an
  invertible fold, no subtree update) take
  [`subtree_link_cut_tree`](subtree-link-cut-tree.md), 1.19/1.12 times as
  fast (244/230 versus 205/206 ms, plain/native); the ETT is the one for subtree and component *updates*. For mixed
  path and subtree work, see [`top_tree`](top-tree.md).
- **Static tree: Euler order + Fenwick or segment tree.** Without links and
  cuts, a preorder turns
  every subtree into an interval; that avoids restructuring a dynamic forest.
- **Initial trees: `assign_parents`, not n - 1 `link` calls.** Each `link`
  splits and inserts in tours that grow to 3n nodes; the bulk build is one
  DFS and one linear treap build per tree, 62.7/27.6 = 2.27 times as fast plain and 63.8/26.2 = 2.44 native.
- `subtree_fold(v, parent)` names a side, not a root: `subtree_fold(p, v)`
  is the other side, and the two fold to `component_fold(v)`.
- `at` (untagged tiers) returns a reference into the node pool, which never
  reallocates after construction: it stays valid but follows later `set`,
  `update` and tags on that vertex. Allocation and algebra exceptions
  propagate; mutations provide no rollback guarantee.

## How it works

Each tree is stored as its Euler tour, read cyclically: one node per vertex
and one per arc (u -> v and v -> u for each edge), in one treap whose nodes
also keep parent links. Going around the tour from arc p -> v to arc v -> p
visits exactly v's side of the edge, whatever the rotation of the cyclic
tour and wherever each vertex node sits among its corners (between an arc
into the vertex and the next arc out). Every operation keeps that
invariant, and it is the only one the structure needs.

**Finding a tree without a key.** A treap is normally entered from its root
by key or rank; a tour node has neither, only an id. The parent links (in an
array of their own, so a walk streams 4 bytes a node) give the root by an
upward walk, and `connected`, `component_size` and `component_fold` are
exactly that walk: no write. Splitting at a node is bottom-up
(`treap_base::split_at_node`): walking up from x, each ancestor joins the
part on its own side of x and adopts the part built so far as its child on
x's side. The parts keep the heap order, so the result is the tree a
top-down split would build, without comparisons or recursion. The tags on
x's root path are settled top-down first (lazy tier only).

**Link.** Split u's tour at u (C = before u, D = u and after), build the
block (v -> u), D, C, (u -> v) with three merges, and insert it just before
v's node (`treap_base::insert_before_node`): the block's root would settle
at the first node on the way to that gap that it outranks, so only the
subtree it outranks is split at v and rebuilt as left, block, right; the
nodes above are pulled once. The result is A, (v -> u), D, C, (u -> v), B:
u's tour rotated to start at u, spliced into a corner of v. **Cut.** Split
out arc u -> v; the tour is now L and R, the arc v -> u in one of them.
Splitting out v -> u leaves one side alone (the part between the arcs) and
two pieces of the other, which one merge rejoins (cyclically adjacent, so
their order is a rotation). The two arc nodes return to the free list (edge
e owns nodes n + 1 + 2e and the next).

**Subtree segments without splits.** Collect the root paths of arc p -> v
and arc v -> p and their lowest common ancestor. The segment strictly
between them is a set of whole subtrees hanging off the two paths below the
lca (the later sides of the first arc's path, the earlier sides of the
second's) plus the single nodes passed on the way, or, when the segment
wraps around the end of the tour, everything off both paths up to the root
on the outer sides. `subtree_fold` folds those parts (commutativity makes
the order irrelevant); `subtree_apply` tags them and pulls both paths
bottom-up. No split, no merge, O(depth).

**Tags** live at nodes already reflected in the node's own value and
aggregate (the [`treap_sequence`](treap-sequence.md) scheme); `push`
settles them one level and skips an identity tag when `Tag` is comparable.
There is no reversal: rotations of a cyclic tour need none.

**Edge lookup and constexpr.** A `flat_hash_map` from the sorted endpoint
pair (packed into 64 bits) to the edge id, with the seeded `atl::hash` at run
time and an unseeded mix in constant evaluation; edge ids come from a LIFO
free list. Priorities come from splitmix64 of the node id and the
per-process seed (a fixed seed in constant evaluation). The whole class is
`constexpr` (proofs in tests/ds/constexpr_checks.cpp).

**Bulk build.** `assign_parents` counting-sorts the children, writes each
tree's tour with one iterative DFS, and builds it as one treap in O(size)
by the rightmost-spine stack (the Cartesian tree of the priorities).

**Complexity.** Every operation is O(1) root walks, bottom-up splits and
merges, each O(depth), and a treap with random priorities has expected
depth O(log n) whatever the tree's shape. Merges recurse to the treap depth
(expected O(log n): a 1e6-vertex path is a test); nothing else recurses.

**Why not a splay tree.** The splay-engine tour (kept as
`atl::experimental::euler_tour_tree_by_splay`) has amortized bounds and no
parent-walk reads: every query splays, writing the access path, and a
subtree segment needs two splays. It lost churn, links and the vertex-add
shapes (table above).

References: "Dynamic trees as search trees via Euler tours" and "Randomized search trees".

## See also

[`link_cut_tree`](link-cut-tree.md), [`subtree_link_cut_tree`](subtree-link-cut-tree.md),
[`top_tree`](top-tree.md), [`treap_sequence`](treap-sequence.md),
[`fenwick-tree`](fenwick-tree.md), [`dsu`](dsu.md).
