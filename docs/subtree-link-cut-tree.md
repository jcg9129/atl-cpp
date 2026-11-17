# `atl::subtree_link_cut_tree`

A [`link_cut_tree`](link-cut-tree.md) that also folds subtrees: link, cut and
re-root trees, ask the same structural questions and path folds, and get the
fold or the vertex count of a subtree under the current root, of one side of
an edge, or the size of a whole component, each in amortized O(log n). The
fold must be a commutative group (sum, xor, addition mod p on tuples): an
access moves children in and out of a per-node fold of "everything hanging
off this node", which needs subtraction. Reach for it for vertex-add/subtree-sum
queries, component sizes in a changing forest, or the size of the side of an edge. Path updates, subtree updates and
non-invertible subtree folds (min, max) are not here: those are top-tree (or,
for subtree updates, Euler-tour-tree) territory.

```cpp
#include <atl/ds/subtree_link_cut_tree.hpp>
```

\include subtree-link-cut-tree/basic.cpp

Output:

\include subtree-link-cut-tree/basic.out

## API

```cpp
template<std::copyable Value, binary_operation<Value> Op, inverse_operation<Value> Inverse>
    requires (std::is_void_v<Value> || std::default_initializable<Value>)
class subtree_link_cut_tree;
```

`Op` folds, `Inverse` inverts: `(Value, op, identity)` must be a commutative
group. Constructors (O(n)): `subtree_link_cut_tree()` (empty);
`(n, identity, op = {}, inverse = {})` (every vertex holds `identity`);
`(values, identity, op = {}, inverse = {})` (one vertex per element, moved out
only of an owning rvalue range). Deduction guides: `Value` from the identity,
a values range converting without narrowing. `n < 2^32 - 1` is asserted.

| member | meaning | time (amortized) |
|---|---|---|
| `make_root`, `link`, `cut(u, v)`, `cut_from_parent` | as [`link_cut_tree`](link-cut-tree.md) | O(log n) |
| `connected`, `root`, `parent`, `lca`, `depth`, `distance`, `ancestor`, `path_vertex` | as `link_cut_tree` | O(log n) |
| `at(v)` | value (`Value const &`) | O(1) |
| `set(v, x)`, `update(v, f)` | replace the value by `x` / `f(value)` | O(log n) |
| `fold(u, v)` | fold of the path's values (order irrelevant: the fold commutes) | O(log n) |
| `subtree_fold(v)` | fold of v's subtree under the current root | O(log n) |
| `subtree_fold(v, p)` | fold of the side of edge {v, p} holding v: v's subtree if p were the root; the root is unchanged | O(log n) |
| `subtree_size(v)`, `subtree_size(v, p)` | the same as vertex counts | O(log n) |
| `component_size(v)` | vertices in v's tree | O(log n) |
| `assign_parents(parents)` | link every v below `parents[v]` (`none` or `v` = root) | O(n) |
| `size()`, `empty()` | vertices | O(1) |

`none` is the class constant (`== atl::graph::none`). There is no `apply`
(path update) and no edge tier. Memory per node: 16 bytes of links and splay
size, the value, the path aggregate, the virtual fold, the splay-subtree
total, two `uint32` counts and a flag: 64 bytes for `long long`.

`Value` must satisfy `std::default_initializable`. The inherited
`assign_parents` requires `std::ranges::forward_range<Parents const> &&
graph::vertex_range<Parents const>`.

## Preconditions

Asserted (`-DATL_ASSERT`): ids in range; `link`: different trees; `cut(u, v)`:
an edge; `cut_from_parent(v)`: v not a root; `fold`: connected;
`subtree_fold(v, p)`, `subtree_size(v, p)`: {v, p} is an edge;
`assign_parents`: every vertex isolated, `size()` entries forming a forest.

Documented only: `(Value, op, identity)` is a commutative group with
`op(a, inverse(a)) == identity` (so `inverse(identity) == identity`; the
structure folds through the null node with no branch). Overflow is the
group's: a `long long` sum must not overflow on any subtree, component or
intermediate fold.

## Measured performance

Measurements using `benchmarks/ds/subtree_link_cut_tree.cpp`, 2026-10-25, Xeon
Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, min ms over 3 runs, `-march=native -O2` / plain
`-O2`, N = Q = 2e5.

| shape | library | by split layout | by columnar layout | reference |
|---|---|---|---|---|
| edge swap + vertex add + subtree sum | **167 / 166** | 173 / 169 | 237 / 228 | |
| cut / link / `component_size` thirds | **165 / 166** | 174 / 171 | 239 / 233 | |
| cut / link / path sum thirds | **207 / 206** | | | `link_cut_tree`, `commutative_fold`: 164 / 172 |

Interleaved stays (split takes up to 5.5% longer, columnar takes 1.37-1.45x the time). The subtree
bookkeeping (virtual totals and counts exchanged on every preferred-child
change) costs 19.8-26.2% on path work against the plain link-cut tree, so use
this class only when subtree folds or sizes are asked for. Against the
Euler tour tree (the structure for subtree *updates*), the
`benchmarks/ds/euler_tour_tree.cpp` record (2026-10-25, Xeon Platinum 8168) races both on
the same vertex add + subtree sum shape: this class 290 / 283 ms, the
library treap-store `euler_tour_tree` 337 / 329, 1.16x slower.

## Pitfalls and notes

- `subtree_fold(v)` depends on the current root (`link` re-roots u's side,
  `make_root` moves it); `subtree_fold(v, p)` does not, which supports queries
  oriented by a parent without changing the current root.
- `set` and `update` cost an access here, not just a splay (the totals
  hanging above the vertex hold its value).
- Needs an inverse: min, max, gcd subtree folds do not fit. Subtree updates
  (add to a whole subtree) belong to the Euler tour tree, path updates to
  `link_cut_tree`.
- On a static tree, subtree folds are a range over the
  preorder entry times on a [`binary_indexed_tree`](fenwick-tree.md)
  or [`segment_tree`](segment-tree.md); use this class when the tree changes.

## How it works

The link-cut tree's preferred paths stay as they are (splay trees by depth
with path-parent pointers, see [`link_cut_tree`](link-cut-tree.md)). A
vertex's real children are its preferred child (inside its splay tree) and
its virtual children (the splay trees whose path-parent pointer names it).
Each node keeps `virtual`, the fold of its virtual children's whole subtrees,
and `total`, the fold over its splay subtree of every vertex's value and
`virtual`, so the total of a splay root covers its path and everything
hanging off it. `pull` recomputes `total` (and the counts) from the children
like any other aggregate; reversal leaves both untouched because the fold
commutes.

`virtual` only changes where a child switches between preferred and
virtual, which happens in exactly two places: the access climb, where each
step makes the old right subtree virtual (`virtual += total(old right)`) and
the splay tree arriving from below preferred (`virtual -= total(it)`); and
`link`, which accesses v first so that adding u's tree to `virtual(v)` leaves
nothing above v stale. `set` and `update` access the vertex for the same
reason. After `access(v)`, v is the deepest node of the root path, so its
subtree is its value plus `virtual(v)`, and its splay tree's `total` is the
whole component. For the side of edge {v, p}: access v; if p is v's
predecessor on the root path (its parent) the answer is v's subtree,
otherwise p is v's child, and the answer is the component total minus p's
subtree (one more access). `assign_parents` writes the path-parent pointers
and folds each parent's `virtual` over its children's totals in child-count
(topological) order: O(n), no recursion.

Implementation: one shared implementation with `link_cut_tree`; the subtree
variant is the node store's `Inverse` parameter (extra fields, the
`exchange_virtual` hook in the climb, `access` instead of `splay` before a
value change). The complexity argument is the link-cut tree's: each access
does O(1) extra group operations per preferred-path change.

References: "A data structure for dynamic trees"; the virtual-subtree augmentation uses group inverses.

## See also

[`link_cut_tree`](link-cut-tree.md), [`splay_engine`](splay-engine.md),
[`time_divide_and_conquer`](time-divide-and-conquer.md) (offline deletion).
