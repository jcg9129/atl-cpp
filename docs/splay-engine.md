# `atl::splay_engine`

Recursion-free structural operations for a splay tree over caller-owned node
storage. Use it for custom payloads and augmentations; use the map or sequence
wrapper when its value model already fits.

```cpp
#include <atl/ds/splay_engine.hpp>
```

\include splay-engine/quick-start.cpp

Expected stdout:

\verbinclude splay-engine/quick-start.out

## API

`atl::splay_store<Store>` checks const `child(x, side)` and `parent(x)`
accessors convertible to `uint32`, and mutable `set_child(x, side, id)` and
`set_parent(x, id)` setters. Side false means left, true means right.
`atl::sized_splay_store<Store>` also checks `size(x)`. Id zero is null.

`atl::splay_step` has `left`, `right`, `stop` enumerators.
`atl::splay_probe<Probe>` checks invocation of `Probe&` on `uint32` returning
exactly `splay_step`. `atl::splay_prefix` contains `uint32 root, length`:
the resulting tree root and accepted prefix length.

Every operation below is a static constexpr member except `ancestor_buffer`.
Store arguments are mutable references. A returned root replaces the caller's
root variable.

| signature | behavior |
|---|---|
| `void rotate(Store& s, uint32 x)` | rotate x above its parent, pushing both and pulling both |
| `void splay(Store& s, uint32 x)` | push the ancestor path then splay x to the top |
| `void splay_pushed(Store& s, uint32 x)` | splay an already-pushed path; [`top_tree`](top-tree.md) pushes its compress and rake chain once |
| `uint32 descend(Store& s, uint32 root, Probe&& probe)` | visit pushed nodes through one lvalue probe; return and splay the final node |
| `uint32 splay_at(Store& s, uint32 root, uint32 k)` | splay the node at zero-based rank k |
| `uint32 join(Store& s, uint32 a, uint32 b)` | concatenate disjoint trees in that order |
| `std::pair<uint32, uint32> split_at(Store& s, uint32 root, uint32 k)` | first k nodes, remaining nodes |
| `std::pair<uint32, uint32> split_before(Store& s, uint32 x)` | nodes preceding x, x and its suffix |
| `std::pair<uint32, uint32> split_after(Store& s, uint32 x)` | prefix ending at x, nodes following x |
| `uint32 insert_at(Store& s, uint32 root, uint32 k, uint32 x)` | insert a detached singleton at gap k, return x |
| `uint32 erase(Store& s, uint32 x)` | detach x, pull it as a singleton, return the remaining root |
| `void for_each(Store& s, uint32 root, Visit&& visit)` | visit each node id in order, pushing on descent |
| `splay_prefix max_right(Store& s, uint32 root, Value acc, Op&& op, Pred&& pred)` | longest accepted prefix fold from acc |
| `uint32 build_balanced(Store& s, R const& ids)` | overwrite links of distinct ids into a balanced in-order tree; [`top_tree`](top-tree.md) uses it for rake trees |
| `std::vector<uint32>& ancestor_buffer() noexcept` | current thread's reusable scratch, also used by [`top_tree`](top-tree.md) across compress and rake trees; runtime-only because it uses thread-local storage |

Positional methods and `max_right` need a sized store. Other operations need a
plain store. `for_each` checks `std::invocable<Visit&, uint32>`. Prefix search
checks store `value(x)` and `aggregate(x)`, an operation on const `Value`
operands with a result convertible to `Value`, and a predicate on const
`Value`; its accumulator must be move-assignable. `build_balanced` checks a
sized random-access range on `R const`, with ids explicitly castable to
`uint32`. Empty trees and ranges return zero roots.

The aliases `atl::splay_links` and `atl::sized_splay_links` provide reusable
link storage: `link` is the record type and `std::vector<link> links` contains
sentinel zero. All mixin methods are constexpr:

| signature | behavior |
|---|---|
| `uint32 child(uint32 x, bool side) const` / `uint32 parent(uint32 x) const` | read links |
| `void set_child(uint32 x, bool side, uint32 c)` / `void set_parent(uint32 x, uint32 p)` | write one link, with no other repair |
| `uint32 size(uint32 x) const` / `void pull_size(uint32 x)` | sized mixin only: read / recompute subtree size |
| `uint32 add_node()` | append a detached node of size one, return its id |
| `uint32 node_count() const` | allocated nodes, excluding sentinel zero |
| `void reserve(std::size_t nodes)` | reserve nodes besides the sentinel |

Optional store hooks are detected by expressions. `push(x)` settles pending
tags one level; `pull(x)` recomputes augmentation. A store without these hooks
gets no-op behavior. `is_splay_root(x)` replaces the default
`parent(x) == 0` stop test, allowing an external path-parent pointer. Rotating
a splay root upward preserves that external pointer on the resulting root.

## Complexity

Rotation is O(1). Splay, descent and composite tree operations are amortized
O(log n), plus callback/payload costs; one operation can be O(n). Full visit
and balanced build are O(n). Ancestor scratch grows to O(height) at runtime
and is reused per thread; balanced build uses O(n) temporary ids. Mixin
accessors are O(1), append is amortized O(1), and growing reserve is O(node
count). All traversal and structural operations avoid recursion.

## Preconditions

Asserted with `ATL_ASSERT`: nonzero handles in splay and rotation; rotation
must have a parent in its splay tree; positional ranks must be in range;
inserted nodes are nonzero with zero children and parent; balanced builds and
mixin growth fit the 32-bit id space. Reserve excludes the sentinel and must
leave room for it in `size_t`.

Documented only: all referenced ids exist and links agree in both directions;
root arguments name a splay root unless the documented subtree descent applies.
Join inputs are distinct trees with the intended concatenation order.
Balanced-build ids are nonzero, distinct, representable as `uint32`, in no
other tree, and carry settled payloads. The store supplies `size(0) == 0`.
`pull` really recomputes augmentation and `push` really settles one level.
For `splay_pushed`, every ancestor path node is already pushed. A proper-subtree
`descend` also requires its ancestors pushed and splays to the whole tree's
root. The store's hooks must never reenter a handle splay: scratch is shared.

Prefix search requires an associative ordered fold, an initial accumulator
accepted by `pred`, and monotonicity over growing prefixes. `for_each` callbacks
must not relink nodes. The engine does not read or write node zero's links;
a caller's `pull` may read its identity payload. Allocation or callable
exceptions propagate without restoring partially changed links.

## Measured performance

The engine is measured through `benchmarks/ds/splay_sequence.cpp` and
`benchmarks/ds/splay_map.cpp`: top-down descent, bottom-up descent, a second
push pass and semi-splay reads differ in restructuring. The complete sequence
and map tables are on their guide pages. Handle-splay measurements compare
ancestor scratch with a store-owned buffer, parent-pointer reversal and a
two-pass reversal-parity walk.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| flags | n | library | by store buffer | by pointer reversal | by parity walk |
|---|---:|---:|---:|---:|---:|
| `-O2` | 200,000 | 170 | 163 | 161 | 161 |
| `-O2` | 1,000,000 | 1870 | 1820 | 1900 | 2060 |
| `-O2 -march=native` | 200,000 | 137 | 136 | 137 | 143 |
| `-O2 -march=native` | 1,000,000 | 1540 | 1550 | 1570 | 1620 |


The [`link_cut_tree`](link-cut-tree.md) push-strategy race has no uniform
winner: at n=q=5e5 native the library stack loses random churn to parity
walk (473 versus 446 ms), but wins the path (287 versus 321 ms).
In this handle race the library loses at n=2e5 plain to pointer reversal
and parity walk (170 versus 161 ms), while n=1e6 native favors it
(1540 versus 1570/1620 ms).

The plain handle rows vary between runs (spreads up to 1.40x); their
small differences alone do not establish stable push-strategy rankings.

## Pitfalls and notes

A handle is a nonzero valid node id, not an arbitrary integer. Keep returned
roots. Splits keep the external parent pointer with the portion containing the
splayed root and detach the other portion; erase transfers the removed root's
external parent to the remaining tree. A callback should store its own search
state; probes see each visited node once through the same lvalue.
`ancestor_buffer` is scratch whose contents may be overwritten by engine calls.
Every other engine operation supports constant evaluation when its store and
callables do; constant-evaluated handle splays use a local vector.

## How it works

Handle splaying records and pushes the ancestor path, then performs bottom-up
zig-zig or zig-zag rotations. A rotation pulls the node that moves down;
the rising node is pulled when it stops. The stop test is evaluated while the
parent's original links are still intact.

A root descent splays top-down: a left and right side tree collect passed
nodes, and the final node adopts both trees. Parent pointers allow bottom-up
spine pulls without a recursion stack. A proper-subtree descent uses a pushed
walk followed by bottom-up splaying. Join splays the left tree's last node,
then attaches the right tree. Splits expose a rank or handle and detach one
side. Balanced construction uses an explicit range stack and reverse-order
pulls. Full visitation climbs by parent pointers.

The access lemma in "Self-Adjusting Binary Search Trees" bounds amortized
rotations by logarithmic changes of subtree-size ranks. Each composite here
performs a constant number of splays. This bounds structural work independently
of individual tree height.

## See also

 [Splay sequence](splay-sequence.md), [Splay map](splay-map.md),
[Treap sequence](treap-sequence.md).
