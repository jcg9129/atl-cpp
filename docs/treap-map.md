# `atl::treap_map` / `atl::treap_set`

An ordered map and set with unique keys, order statistics and ordered value folds. Use it for online keys and positional queries such as rank; a dense integer universe can use a hierarchical bitset with less overhead.

```cpp
#include <atl/ds/treap_map.hpp>
```

\include treap-map/quick-start.cpp

Expected stdout:

\verbinclude treap-map/quick-start.out

## API

```cpp
template<std::semiregular Key, typename Value = void,
         typename Compare = std::ranges::less, typename Op = void>
class treap_map;
```

`treap_set<Key, Compare, Op>` aliases the class with
`Value = void`. `Compare const &` orders two const keys with a strict weak
order. `Value` is semiregular or void. With `Op`, `atl::binary_operation`
checks a const operation on two const mapped values (keys for a set), returning
a value convertible to that type. Fold methods require this tier. An
`entry` contains `Key const &key` and, for a map, `Value const &value`.
`key_type` is `Key`; `size_type` is `std::size_t`.

| signature | behavior |
|---|---|
| `treap_map()` / `explicit treap_map(Compare comp)` | empty, without a fold tier |
| `treap_map(fold_type identity, std::type_identity_t<OpArg> op, Compare comp = {})` | empty fold tier; `OpArg = Op` must equal `Op` |
| `treap_map(sorted_unique_t, R&& entries, Compare comp = {})` | plain tier, sorted unique keys or `(key, value)` tuples |
| `treap_map(sorted_unique_t, R&& entries, fold_type identity, std::type_identity_t<OpArg> op, Compare comp = {})` | sorted unique fold tier |
| `bool insert(Key key)` | set tier; true if new |
| `bool insert(Key key, std::type_identity_t<V> value)` | map tier; keep an existing value |
| `bool insert_or_assign(Key key, std::type_identity_t<V> value)` | map tier; true if new |
| `V const& update(Key key, Update&& f)` | map tier; replace with `f(value)` or insert `f(Value{})`; exactly one invocation |
| `bool contains(Lookup const& k) const` | membership |
| `V const& at(Lookup const& k) const` / `V const* find(Lookup const& k) const` | map tier; assert presence / return null on a miss |
| `bool erase(Lookup const& k)` | true if a key was removed |
| `std::size_t erase(Lo const& lo, Hi const& hi)` | remove keys in `[lo, hi)`, return count |
| `treap_map split(Lookup const& k)` | retain keys ordered below k; return the remaining tree on the same pool |
| `std::size_t rank(Lookup const& k) const` | number of keys ordered below k |
| `entry kth(std::size_t i) const` | entry at zero-based rank i |
| `std::optional<entry> next(Lookup const& k) const` / `prev(Lookup const& k) const` | inclusive successor / predecessor under `Compare` |
| `fold_type fold(Lo const& lo, Hi const& hi) const` / `fold_type fold() const` | ordered value fold; identity for an empty range |
| `void for_each(Visit visit) const` | in-order `visit(key)` or `visit(key, value)` |
| `void for_each(Lo const& lo, Hi const& hi, Visit visit) const` | visit just `[lo, hi)` |
| `treap_map sibling() const` | empty handle sharing the pool |
| `void reserve(std::size_t n)` | reserve room for this many additional keys, counting recycled slots |
| `std::size_t size() const noexcept` / `bool empty() const noexcept` | element count / emptiness |
| `void clear()` | recycle this handle's nodes |
| `void merge_union(treap_map&& other)` | absorb a distinct same-pool tree, empty it; this map wins duplicate values |

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

Single-key operations, range folds and split take expected O(log n).
Whole-tree folds, size, emptiness and sibling take O(1); range erasure and
visitation add O(number of selected keys). Full visitation, copying, clearing,
destruction and sorted construction take O(n). Move construction takes O(1);
move assignment first recycles the destination. A growing reserve takes O(pool
size). Memory is O(allocated pool slots), including recycled slots and nodes
owned by related handles. Costs assume constant-time payload operations.

Union takes expected O(m log(1 + n/m)), for m <= n, including duplicate removal. Random-priority bounds assume unbiased priorities and inputs independent of the seed; worst-case depth and operation cost can be linear.

## Preconditions

Asserted with `ATL_ASSERT`: `i < size()` for `kth`, presence for `at`, a
strictly increasing sorted build, a valid pool for mutating/pool-dependent
operations, and fewer than 2^32 pool slots when adding a node. Reserve growth
must fit the 32-bit slot space and `size_t`.
`merge_union` additionally asserts a distinct source on the same pool.

Documented only: comparator ordering laws; associativity and a two-sided fold
identity; arithmetic results within the payload type's bounds. Callbacks must
not mutate any tree sharing this pool while it is traversed or searched.
Returned references and entries are invalidated by pool reallocation, erasure
of their node, or destruction/assignment of the owning tree. A sibling's
allocation can invalidate a reference too. Allocation and callable exceptions propagate from non-noexcept operations;
recycling failures in destruction or noexcept move assignment terminate.
Mutation has no transactional rollback guarantee.

## Measured performance

Measurements from `benchmarks/ds/treap_map.cpp`
(2026-11-22, AMD EPYC-Milan ccx23, GCC 15.2.0,
RelWithDebInfo, assertions off; plain `-O2` / `-O2 -march=native`,
five runs of 25 samples). It compares mixed
insert/erase/rank with a `std::map` membership stand-in (the query costs are
not equivalent), and tree union with inserting every incoming key separately.
Both union loops include the same deep-copy setup. The 1e4-key union
into a copy of 1e6 takes 106/65.1 ms for the library versus 116/73.5 ms
by element insertion. The mixed library stream takes 145/141 ms versus
101/78.7 ms for the standard-map stand-in, which omits rank queries;
that is not an equivalent-operation speed comparison.

## Pitfalls and notes

`next` and `prev` are inclusive bounds. A fold needs an explicit identity;
`Value{}` is the starting value of an absent-key update and need not be that
identity. Reserve counts additional keys in the shared pool. Recycled nodes
remain allocated until the pool is released.

## How it works

Pool id zero is the empty subtree. Each node stores a random priority,
size and optional ordered aggregate. Merge follows priority order and split
follows a monotone key boundary, pulling sizes and aggregates on the return
walk. Range folds descend along two boundaries without changing links.
Sorted construction maintains a Cartesian tree's right spine, pulling each
node once. Union splits the other tree around the higher-priority root and
recurses into matching sides, preserving this map's duplicate value.

## See also

 [Treap sequence](treap-sequence.md), [Splay engine](splay-engine.md),
[Splay map](splay-map.md),
[Hierarchical bitset](hierarchical-bitset.md).
