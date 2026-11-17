# `atl::splay_map` / `atl::splay_set`

A deterministic ordered map and set that moves accessed keys toward the root. Use it for successor sweeps and concentrated accesses; uniformly random traffic also pays for restructuring.

```cpp
#include <atl/ds/splay_map.hpp>
```

\include splay-map/quick-start.cpp

Expected stdout:

\verbinclude splay-map/quick-start.out

## API

```cpp
template<std::semiregular Key, typename Value = void,
         typename Compare = std::ranges::less, typename Op = void, typename Layout = atl::splay_layout_split>
class splay_map;
```

`splay_set<Key, Compare, Op, Layout>` aliases the class with
`Value = void`. `Compare const &` orders two const keys with a strict weak
order. `Value` is semiregular or void. With `Op`, `atl::binary_operation`
checks a const operation on two const mapped values (keys for a set), returning
a value convertible to that type. Fold methods require this tier. An
`entry` contains `Key const &key` and, for a map, `Value const &value`.
`key_type` is `Key`; `size_type` is `std::size_t`.

| signature | behavior |
|---|---|
| `splay_map()` / `explicit splay_map(Compare comp)` | empty, without a fold tier |
| `splay_map(fold_type identity, std::type_identity_t<OpArg> op, Compare comp = {})` | empty fold tier; `OpArg = Op` must equal `Op` |
| `splay_map(sorted_unique_t, R&& entries, Compare comp = {})` | plain tier, sorted unique keys or `(key, value)` tuples |
| `splay_map(sorted_unique_t, R&& entries, fold_type identity, std::type_identity_t<OpArg> op, Compare comp = {})` | sorted unique fold tier |
| `bool insert(Key key)` | set tier; true if new |
| `bool insert(Key key, std::type_identity_t<V> value)` | map tier; keep an existing value |
| `bool insert_or_assign(Key key, std::type_identity_t<V> value)` | map tier; true if new |
| `V const& update(Key key, Update&& f)` | map tier; replace with `f(value)` or insert `f(Value{})`; exactly one invocation |
| `bool contains(Lookup const& k)` | membership |
| `V const& at(Lookup const& k)` / `V const* find(Lookup const& k)` | map tier; assert presence / return null on a miss |
| `bool erase(Lookup const& k)` | true if a key was removed |
| `std::size_t erase(Lo const& lo, Hi const& hi)` | remove keys in `[lo, hi)`, return count |
| `splay_map split(Lookup const& k)` | retain keys ordered below k; return the remaining tree on the same pool |
| `std::size_t rank(Lookup const& k)` | number of keys ordered below k |
| `entry kth(std::size_t i)` | entry at zero-based rank i |
| `std::optional<entry> next(Lookup const& k)` / `prev(Lookup const& k)` | inclusive successor / predecessor under `Compare` |
| `fold_type fold(Lo const& lo, Hi const& hi)` / `fold_type fold() const` | ordered value fold; identity for an empty range |
| `void for_each(Visit visit) const` | in-order `visit(key)` or `visit(key, value)` |
| `void for_each(Lo const& lo, Hi const& hi, Visit visit)` | visit just `[lo, hi)` |
| `splay_map sibling() const` | empty handle sharing the pool |
| `void reserve(std::size_t elements)` | reserve room for this many additional keys, counting recycled slots |
| `std::size_t size() const noexcept` / `bool empty() const noexcept` | element count / emptiness |
| `void clear()` | recycle this handle's nodes |

`Layout` is `atl::splay_layout_split` or `atl::splay_layout_interleaved`.
Split storage separates the 16-byte link/size record from the payload.

`fold_type` is the mapped type or, for a set, `Key`. Map-tier `V` defaults to
`Value` and must equal it. Lookup templates use `ordered_lookup_key`: a
transparent comparator permits direct heterogeneous ordering; other arguments
convert to `Key`. Mixed arithmetic types always convert. Bounds are ordered
by `Compare`; an inverted or empty interval selects no keys. Sorted-range
constructors accept input ranges whose forwarded elements convert to keys or
whose tuple components construct the key and mapped type. Owning rvalues move;
views and borrowed/lvalue ranges copy; prvalues move. Update and visit
constraints check lvalue invocation and the update's convertible result.

Copy construction/assignment deep-clones nodes into the source pool. Moves
transfer the handle. Destruction recycles the owned nodes. A moved-from handle
may be destroyed, assigned, cleared or queried for emptiness; operations that
need its pool require an active handle. These containers are runtime-only
because their ownership uses `std::shared_ptr`.

## Complexity

Single-key operations, range folds and split take amortized O(log n).
Whole-tree folds, size, emptiness and sibling take O(1); range erasure and
visitation add O(number of selected keys). Full visitation, copying, clearing,
destruction and sorted construction take O(n). Move construction takes O(1);
move assignment first recycles the destination. A growing reserve takes O(pool
size). Memory is O(allocated pool slots), including recycled slots and nodes
owned by related handles. Costs assume constant-time payload operations.

A single splay can take O(n); the logarithmic bound is amortized over accesses. Full traversal and recycling use parent pointers without recursion.

## Preconditions

Asserted with `ATL_ASSERT`: `i < size()` for `kth`, presence for `at`, a
strictly increasing sorted build, a valid pool for mutating/pool-dependent
operations, and fewer than 2^32 pool slots when adding a node. Reserve growth
must fit the 32-bit slot space and `size_t`.

Documented only: comparator ordering laws; associativity and a two-sided fold
identity; arithmetic results within the payload type's bounds. Callbacks must
not mutate any tree sharing this pool while it is traversed or searched.
Returned references and entries are invalidated by pool reallocation, erasure
of their node, or destruction/assignment of the owning tree. A sibling's
allocation can invalidate a reference too. Allocation and callable exceptions propagate from non-noexcept operations;
recycling failures in destruction or noexcept move assignment terminate.
Mutation has no transactional rollback guarantee.

## Measured performance

`benchmarks/ds/splay_map.cpp` measures end-to-end throughput including setup.
The library uses top-down descent and split storage; other splay contenders
change descent or layout. The treap and standard map provide balancing baselines.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2, RelWithDebInfo, assertions off,
2026-10-24. Minimum of three run means, ms; both flag sets also use
`-g -DNDEBUG -std=c++23`. Larger timing spreads in the plain run mean small
differences should be interpreted cautiously.

Milliseconds, for every recorded workload and size. Columns use the benchmark labels.
The final column divides the standard-map time by the library time; values
above one mean that the library took less time.

| flags | workload | n | by std map | by treap map | library | by top down interleaved | by bottom up interleaved | by bottom up split | by std map / library |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `-O2` | uniform insert/find/erase | 2e5 | 69.8 | 128 | 111 | 112 | 132 | 141 | 0.63x |
| `-O2` | uniform insert/find/erase | 1e6 | 721 | 1171 | 1098 | 1071 | 1261 | 1319 | 0.66x |
| `-O2` | Zipf s=1.0 finds | 2e5 | 62.5 | 51.2 | 55.7 | 56.0 | 65.3 | 70.5 | 1.12x |
| `-O2` | Zipf s=1.5 finds | 2e5 | 30.2 | 21.5 | 19.9 | 20.3 | 21.3 | 22.4 | 1.52x |
| `-O2` | Zipf s=1.0 finds | 1e6 | 596 | 441 | 487 | 470 | 543 | 579 | 1.22x |
| `-O2` | Zipf s=1.5 finds | 1e6 | 207 | 141 | 116 | 130 | 137 | 128 | 1.78x |
| `-O2` | successor chains, length 1,000 | 2e5 | 29.3 | 24.3 | 13.8 | 14.6 | 14.2 | 14.0 | 2.12x |
| `-O2` | successor chains, length 1,000 | 1e6 | 205 | 141 | 80.1 | 96.9 | 100 | 82.0 | 2.56x |
| `-O2` | working-set finds, window 1,000 | 2e5 | 35.4 | 24.8 | 38.3 | 39.4 | 45.3 | 49.2 | 0.92x |
| `-O2` | working-set finds, window 1,000 | 1e6 | 218 | 147 | 212 | 233 | 267 | 269 | 1.03x |
| `-O2` | sorted inserts | 2e5 | 33.4 | 32.5 | 6.0 | 6.4 | 5.4 | 5.0 | 5.57x |
| `-O2` | sorted inserts | 1e6 | 222 | 218 | 33.6 | 57.2 | 52.3 | 28.4 | 6.61x |
| `-O2 -march=native` | uniform insert/find/erase | 2e5 | 70.4 | 129 | 114 | 112 | 135 | 142 | 0.62x |
| `-O2 -march=native` | uniform insert/find/erase | 1e6 | 762 | 1134 | 1065 | 1058 | 1258 | 1283 | 0.72x |
| `-O2 -march=native` | Zipf s=1.0 finds | 2e5 | 59.3 | 49.1 | 56.9 | 54.6 | 66.0 | 69.9 | 1.04x |
| `-O2 -march=native` | Zipf s=1.5 finds | 2e5 | 28.2 | 23.1 | 20.3 | 19.6 | 21.3 | 22.1 | 1.39x |
| `-O2 -march=native` | Zipf s=1.0 finds | 1e6 | 582 | 459 | 488 | 465 | 544 | 576 | 1.19x |
| `-O2 -march=native` | Zipf s=1.5 finds | 1e6 | 207 | 163 | 122 | 130 | 137 | 126 | 1.70x |
| `-O2 -march=native` | successor chains, length 1,000 | 2e5 | 29.4 | 24.2 | 14.2 | 14.1 | 14.5 | 14.3 | 2.07x |
| `-O2 -march=native` | successor chains, length 1,000 | 1e6 | 207 | 142 | 82.3 | 97.4 | 101 | 84.0 | 2.52x |
| `-O2 -march=native` | working-set finds, window 1,000 | 2e5 | 35.2 | 25.9 | 39.0 | 38.0 | 45.8 | 49.3 | 0.90x |
| `-O2 -march=native` | working-set finds, window 1,000 | 1e6 | 217 | 154 | 211 | 226 | 267 | 262 | 1.03x |
| `-O2 -march=native` | sorted inserts | 2e5 | 33.6 | 36.4 | 5.8 | 6.1 | 5.3 | 5.0 | 5.79x |
| `-O2 -march=native` | sorted inserts | 1e6 | 224 | 215 | 32.3 | 55.9 | 51.3 | 28.9 | 6.93x |

## Pitfalls and notes

`next` and `prev` are inclusive bounds. A fold needs an explicit identity;
`Value{}` is the starting value of an absent-key update and need not be that
identity. Reserve counts additional keys in the shared pool. Recycled nodes
remain allocated until the pool is released.

Keyed reads other than whole-tree folds and traversal restructure the tree and are non-const. Const traversal does not splay; `merge_union` is unavailable.

## How it works

The engine splays during key descent. A successful search moves its key to
the root; a miss moves its final neighbour. Insertion reuses that neighbour;
erasure joins the root's children. Two splays expose a key range for folds,
visits and erasure. Pulls restore ordered aggregates; node zero holds the
identity. Sorted construction and copies use the engine's balanced build.

## See also

 [Treap sequence](treap-sequence.md), [Splay engine](splay-engine.md),
[Treap map](treap-map.md),
[Hierarchical bitset](hierarchical-bitset.md).
