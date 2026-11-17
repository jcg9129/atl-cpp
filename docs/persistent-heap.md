# `atl::persistent_heap`, `atl::persistent_heap_forest`

**A fully persistent meldable priority queue.** `push`, `pop`, `meld` and
`push_range` return a new version and leave every old one intact, at
O(log n) expected new nodes per operation. Reach for it when a search or a DP
needs many heaps that share most of their elements: the sidetrack heaps of
k shortest walks (`H[v]` = `H[parent]` plus `v`'s own edges), the
k-th best of a combinatorial choice, tree DPs that meld the children's heaps
but must keep each child's heap. Two forms share one engine: the class
`persistent_heap` (every version a self-contained object, the persistent
family's shape) and `persistent_heap_forest` (one object owns the pool, a
version is a `uint32` id), which avoids shared-pointer traffic per version.
When no old version is ever read again, use
[`meldable_heap_forest`](meldable-heap-forest.md) instead: it melds in O(1)
and frees popped nodes.

```cpp
#include <atl/ds/persistent_heap.hpp>
```

\include persistent-heap/basic.cpp


## API

`template<std::semiregular T, typename Compare = std::less<>>` for both forms;
`std::strict_weak_order<Compare&, T const&, T const&>` constrains ordering.
The class is runtime-only (`std::shared_ptr`); the forest is constexpr.
Allocation failures propagate; mutations are not transactional when a callable,
element operation or allocation throws.

`T` is `std::semiregular`: the pool's null node holds a `T{}`, and path
copying copies elements. `top` is the comp-largest element, as in
`std::priority_queue`: `std::less<>` gives a max-heap, `std::greater<>` a
min-heap.

Default construction requires `std::default_initializable<Compare>`; supply
a capturing-lambda comparator explicitly.

### `persistent_heap`

| member | description | complexity (expected) |
|---|---|---|
| `persistent_heap() requires std::default_initializable<Compare>` | empty, with a new pool | O(1) |
| `explicit persistent_heap(Compare comp)` | empty, ordered by `comp`, with a new pool | O(1) |
| `explicit persistent_heap(R &&values, Compare comp = {})` | a heap of an input range, with a new pool; elements moved only out of an owning rvalue range | O(n) |
| `size()` → `std::size_t`, `empty()` → `bool` | element count | O(1) |
| `top()` → `T const &` | the comp-largest element, a reference into the pool | O(1) |
| `push(T value)` → `persistent_heap` | the heap with `value` added | O(log n) |
| `emplace(Args &&...args)` → `persistent_heap` | `push(T(args...))` | O(log n) |
| `pop()` → `persistent_heap` | the heap without its top | O(log n) |
| `meld(persistent_heap const &other)` → `persistent_heap` | the elements of both (`other` may be `*this`) | O(log n + log m) |
| `push_range(R &&values)` → `persistent_heap` | every element of `values` added: an O(k) array-built heap, then one meld | O(k + log n) |
| `root()` → `node_view` | the root of the heap tree | O(1) |
| `arena_size()` → `std::size_t` | nodes in the shared pool, the null node included | O(1) |
| `reserve(std::size_t operations)` | pool room for that many operations on versions of about this size | O(pool entries) if relocated |

Every version-returning member is `[[nodiscard]]` and has a `const &` and a
`&&` overload; the `&&` one moves the pool handle instead of copying it, so
the linear idiom `h = std::move(h).push(x)` skips the atomic refcount traffic
([conventions](conventions.md)). The moved-from heap may only be destroyed or
assigned to. CTAD: `persistent_heap(R &&)` and `persistent_heap(R &&, Compare)`
deduce `T` as the range's value type.

**`node_view`** is a read-only position in a version's heap tree: `empty()`,
`value()` (a reference into the pool), `left()`, `right()`. Each subtree is
heap-ordered, so a child's value is never comp-larger than its parent's.
Searches that treat heap nodes as graph nodes walk views instead of
popping, and make no node. A view holds a pointer to the pool, not a share of
it: it is valid while some version of that pool lives.

### `persistent_heap_forest`

A version is a `uint32` id, its root node; `empty_heap` (0) is the empty heap.
Every id stays valid while the forest lives. Every subtree of a version is a
version too, so `left` and `right` are the forest's `node_view`.

| member | description | complexity (expected) |
|---|---|---|
| `explicit persistent_heap_forest(Compare comp = {})` | an empty forest | O(1) |
| `static constexpr uint32 empty_heap` | the id of the empty heap, 0 | |
| `empty(uint32 heap)` → `bool` | whether `heap` is the empty heap | O(1) |
| `top(uint32 heap)` → `T const &` | its comp-largest element, a reference into the pool | O(1) |
| `left(uint32 heap)`, `right(uint32 heap)` → `uint32` | the versions formed by the root's two subtrees | O(1) |
| `push(uint32 heap, T value)` → `uint32` | `heap` with `value` added | O(log n) |
| `emplace(uint32 heap, Args &&...args)` → `uint32` | `push(heap, T(args...))` | O(log n) |
| `pop(uint32 heap)` → `uint32` | `heap` without its top | O(log n) |
| `meld(uint32 a, uint32 b)` → `uint32` | the elements of both (`a == b` allowed) | O(log n + log m) |
| `push_range(uint32 heap, R &&values)` → `uint32` | every element of `values` added; `push_range(empty_heap, values)` builds a heap | O(k + log n) |
| `arena_size()` → `std::size_t`, `reserve(std::size_t node_count)` | pool size; pool room for that many nodes | |

No element count is kept per version (it would need a record per id); keep one
beside the id if you need it.

## Complexity

Each `push`, `pop` and `meld` walks one coin-picked path and copies the nodes
on it: expected length at most `log2(n_a + 1) + log2(n_b + 1)` for any heap
shapes, so O(log n) expected time and new nodes per operation, for every
version, old ones included (each operation flips fresh coins; there is no
amortization to break). The bound is in expectation over the coins only, not
over the input. `push_range` and the range constructor build their block in
O(k) with no meld inside. Nodes are never freed while a version of the pool
lives: memory is the total over all operations. A node holds `T` and two
`uint32` links (16 bytes for `int64`, 24 for a 16-byte `T`).

## Preconditions

Asserted (`atl_assert`, active under `-DATL_ASSERT`):

- `top`, `pop`, `node_view::value` / `left` / `right`, the forest's `top` /
  `pop` / `left` / `right`: not empty.
- `persistent_heap::meld(other)`: both versions share one pool, that is, both
  derive from one heap. A default-constructed heap starts a new pool, so build
  a family of heaps from one empty heap: `std::vector<persistent_heap<T>> h(n,
  empty)`.
- Operations accessing nodes check forest ids (`id < arena_size()`).
  `empty(id)` only compares with zero; callers still supply a valid id.
- A class version holds fewer than 2^32-1 elements; the pool holds fewer than
  2^32-1 nodes. Forest logical sizes have no stored count.

Documented only: `Compare` is a strict weak order. References from `top` and
`node_view::value` are invalidated by a later operation on any version sharing
the pool (the pool is a `std::vector` that may grow); copy the value out first.

## Measured performance

`benchmarks/ds/persistent_heap.cpp` races randomized, rank-leftist and
weight-biased melds, class versus forest versions, lazy shifts and in-arena
versus temporary-vector heapify. Loops measure whole scripts including allocation.
The k-shortest-walk comparisons return with the graph pages.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2, RelWithDebInfo, assertions off.
Milliseconds, minimum of three run means; each pair is plain `-O2` /
`-O2 -march=native`, both also with `-g -DNDEBUG -std=c++23`. Heapify measured 2026-11-01.

| shape | by vector heapify | library in-arena heapify |
|---|---:|---:|
| range build, 1e6 | 11.3 / 11.2 | 10.0 / 9.2 |
| 1e5 push_range(3), class | 32.5 / 32.4 | 30.8 / 30.1 |
| 1e5 push_range(3), forest | 34.3 / 32.8 | 31.0 / 30.3 |

Generic shapes measured 2026-10-23 (plain) and 2026-10-24 (native), same CPU,
compiler, build type and flags. All experimental rows use vector heapify.

| shape | library class | library forest |
|---|---:|---:|
| DAG 5e5 | 135 / 134 | 88.4 / 86.5 |
| linear 1e6 push/pop 2:1 | 100 / 98.3 | — |
| keep-all 1e6 push + 2e5 pop | 1006 / 1009 | — |
| linear, by mutable rank-leftist forest | 34.4 / 34.3 | — |

| shape | by rank leftist | by weight biased | by randomized meld |
|---|---:|---:|---:|
| DAG class | 158 / 159 | 165 / 165 | 113 / 116 |
| DAG forest | 125 / 125 | 132 / 133 | 87.7 / 87.2 |
| DAG class, unused lazy tags | 161 / 165 | 168 / 168 | 171 / 172 |
| DAG class, 10% shifts | 178 / 184 | 189 / 189 | 188 / 189 |
| linear class | 124 / 125 | 122 / 126 | 101 / 98.2 |
| linear forest | 123 / 124 | 127 / 127 | 101 / 98.3 |
| linear class, unused lazy tags | 170 / 172 | 166 / 169 | 135 / 136 |
| keep-all class | 748 / 748 | 802 / 808 | 1014 / 1015 |
| keep-all forest | 710 / 714 | 773 / 773 | 959 / 951 |
| range build class | 17.5 / 17.3 | 16.8 / 17.3 | 12.5 / 12.0 |

`benchmarks/ds/persistent_heap_tuning.cpp` varies root ordering and whether
fresh nodes are reused. Measured 2026-10-23 with the same CPU, compiler,
build type and flags; its graph comparison returns with the graph pages.

| method / knob | DAG 5e5 | linear 1e6 | keep-all |
|---|---:|---:|---:|
| by rank leftist / branch, reuse fresh | 143 / 142 | 120 / 121 | 732 / 743 |
| by rank leftist / select, reuse fresh | 168 / 170 | 122 / 125 | 755 / 764 |
| by rank leftist / branch, copy fresh | 159 / 160 | 130 / 132 | 748 / 757 |
| by weight biased / branch, reuse fresh | 162 / 163 | 123 / 125 | 793 / 798 |
| by weight biased / select, reuse fresh | 178 / 177 | 120 / 123 | 807 / 820 |
| by weight biased / branch, copy fresh | 164 / 165 | 126 / 130 | 789 / 793 |
| by randomized meld / branch, reuse fresh | 113 / 115 | 95.9 / 100 | 960 / 958 |
| by randomized meld / select, reuse fresh | 120 / 119 | 95.9 / 93.6 | 1058 / 1058 |
| by randomized meld / branch, copy fresh | 116 / 111 | 100 / 98.4 | 962 / 959 |

## Pitfalls and notes

Construct same-pool versions from one origin; independently constructed heaps
cannot meld. A class handle keeps its pool alive; node_view is a borrowed
pointer plus id and does not own the pool. Views and forest ids stay meaningful
through pool growth; references returned by value()/top() may be invalidated.
No node is reclaimed until the owning pool dies. The forest keeps no per-version
element count, so logical size can exceed uint32 through repeated self-meld;
class versions check their uint32 element count. User arithmetic must fit.

## How it works

A **randomized meldable heap** is a heap-ordered
binary tree with no balance information. To meld two heaps, keep the
comp-larger root, then meld the other heap into one of its two children,
picked by a fair coin. `push` melds a one-node heap; `pop` melds the root's
two subtrees. A random walk down any binary tree of n nodes ends after at most
`log2(n + 1)` steps in expectation (each step halves the expected share of
nodes below, by the coin), so a meld costs O(log n) expected whatever the
trees look like.

**Path copying.** A meld changes only the nodes on the path it walks, so
copying them (and pointing the copies at the untouched subtrees) leaves the
old heaps intact. Because the bound needs only fresh coins and holds for any
tree shape, it holds per operation on every version: any version can be
pushed, popped or melded with any other of its pool, any number of times.
Amortized heaps (skew, pairing) do not survive this: their bounds charge an
expensive operation to earlier cheap ones, and a persistent user can repeat
the expensive one on the same old version forever. Leftist heaps survive it
with a worst-case bound, and lost the race on constant factors.

**This code.** Nodes live in one append-only `detail::persistent_arena`
(`std::vector`, `uint32` ids, id 0 the null node); a class version is a
`shared_ptr` to the pool plus its root id and size, a forest version just the
id. The meld is iterative and top-down: each step copies the kept root, hangs
the copy in the hole the preceding step left, and continues into the
coin-picked child; the coins come from one splitmix64 output spent a bit at a
time. A node made by the running operation (the pushed leaf, the block of a
`push_range`) is changed in place instead of copied again . `push_range` and the range constructor build their block
directly in consecutive arena nodes, then link the implicit complete binary
tree and sift values down by holes in a descending pass. No temporary value
vector is allocated; one meld joins the fresh block to the version.

Graph comparisons return with the graph pages.

## See also

- [`meldable_heap_forest`](meldable-heap-forest.md): many ephemeral heaps on
  one pool, O(1) meld, lazy add-to-all.
- [`persistent_list`](persistent-list.md), [`persistent_treap_sequence`](persistent-treap-sequence.md):
  the other members of the persistent family (shared pool, `const &` / `&&`
  pairs).
