# `atl::meldable_heap_forest`, `atl::addressable_meldable_heap_forest`

Many priority queues over one node pool, with O(1) destructive meld and optional
lazy shifts or addressable handles. Use this family when versions need not
coexist; use a persistent heap when each input must remain queryable.

```cpp
#include <atl/ds/meldable_heap_forest.hpp>
```

\include meldable-heap-forest/basic.cpp

## API

`template<typename T, typename Compare = std::less<>, typename Shift = void,
typename Mapping = shift_by_addition, typename Compose = compose_by_addition,
bool HasHandles = false> class meldable_heap_forest` requires
`meldable_heap_parameters<T, Compare, Shift, Mapping, Compose>` and
`!HasHandles || is_void_v<Shift>`.

Elements are movable; Compare is a strict weak order on const elements.
Shifts are void or copyable with an action and a closed composition. The
addressable alias fixes Shift=void, Mapping=shift_by_addition,
Compose=compose_by_addition and HasHandles=true. Aliases are value_type=T,
shift_type=Shift, compare_type=Compare, handle=uint32 and inherited heap_id=uint32.
All operations are constexpr when their element and callable operations permit.

| signature | meaning / complexity |
|---|---|
| `explicit meldable_heap_forest(std::size_t heap_count = 0, Compare comp = {})` | empty heaps, Shift{} identity and default callables when enabled; O(heap_count) |
| `meldable_heap_forest(std::size_t heap_count, Compare comp, tag_type shift_identity, Mapping mapping = {}, Compose compose = {})` | explicit lazy identity/action; O(heap_count), shift tier only |
| `heap_id add_heap()`, `std::size_t heap_count() const noexcept` | append an empty heap (amortized O(1)); count heaps (O(1)) |
| `std::size_t size(heap_id heap) const`, `bool empty(heap_id heap) const` | element count and emptiness; O(1) |
| `void reserve(std::size_t node_count)` | capacity in live elements; O(pool entries) if relocated |
| `T const& top(heap_id heap) const` | comp-largest element; O(1) |
| `handle push(heap_id heap, T value)`, `handle emplace(heap_id heap, Args&&... args)` | insert element; O(1) amortized |
| `T pop(heap_id heap)` | remove and return top; O(log n) amortized |
| `void meld(heap_id into, heap_id from)` | transfer from into into, leaving from empty; self-meld is a no-op; O(1) |
| `void release(heap_id heap)` | empty heap and recycle every node; O(n), O(1) extra space |
| `void add_to_all(heap_id heap, tag_type const& shift)` | shift existing elements, lazy tier only; O(1) |
| `T const& value(handle x) const` | live element, handles only; O(1) |
| `void promote(heap_id heap, handle x, T value)` | improve priority, handles only; O(1) actual plus later pop costs |
| `T erase(heap_id heap, handle x)` | remove live element, handles only; O(log n) amortized |

`tag_type` above is the stored Shift for lazy overloads. Emplace requires
constructible_from<T, Args...>. The default order is std::less<> (max heap);
std::greater<> makes a min heap.

## Complexity

Push/meld/add do constant structural work; vector growth makes push amortized.
Two-pass pairing gives logarithmic amortized pop. Promote has constant actual
cost; its deferred restructuring is charged to subsequent pops. The pool size
is bounded by peak live elements and keeps storage for recycled values until
reuse or destruction. Heap directory storage is O(heap_count).

## Preconditions

Asserted under `ATL_ASSERT`: heap ids are valid, top/pop require nonempty heaps,
counts and node ids remain below 2^32-1, handles are live, and promoted values
are not worse. Handle membership in the supplied heap is documented only.
A handle is invalid after erase/pop/release and can be reused by a later push.

Documented laws: comparator ordering, associative composition with the supplied
identity, identity mapping, mapping respecting composition, and shifts preserving
heap order. Default shift identity Shift{} must actually be neutral. User
arithmetic must remain representable. Later pushes are not affected by earlier
shifts; meld keeps each element's shifted value.

## Measured performance

`benchmarks/ds/meldable_heap_forest.cpp` compares pairing with/without predecessor
handles, iterative rank-leftist, skew and vector-backed binary heaps. The tree
loop pushes three values, pops two and melds upward; slope loops use min-heaps
and lazy shifts. Mixes replay one million operations. Times include allocation.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2, RelWithDebInfo, assertions off,
2026-10-16. Milliseconds, minimum of three run means, plain `-O2` /
`-O2 -march=native`; both also use `-g -DNDEBUG -std=c++23`.

| workload | library pairing | pairing + handles (addressable) | by skew | by rank leftist | by binary |
|---|---|---|---|---|---|
| tree path 2e5 | 12.9 / 13.3 | 16.7 / 17.5 | 17.4 / 16.9 | 19.4 / 19.7 | 37.9 / 38.0 |
| tree star 2e5 | 7.1 / 7.1 | 9.0 / 10.1 | 21.5 / 21.1 | 18.3 / 17.6 | 27.3 / 27.6 |
| tree caterpillar 2e5 | 11.1 / 11.1 | 14.0 / 14.4 | 14.9 / 14.5 | 17.0 / 17.2 | 33.1 / 33.0 |
| tree random 2e5 | 12.7 / 13.1 | 16.8 / 17.4 | 13.7 / 13.4 | 19.1 / 19.4 | 45.8 / 44.5 |
| slope path 2e5 | 13.5 / 13.8 | - / - | 17.5 / 18.7 | 21.6 / 21.5 | 20.9 / 21.7 |
| slope caterpillar 2e5 | 13.2 / 13.3 | - / - | 16.3 / 16.9 | 21.6 / 20.9 | 28.5 / 28.5 |
| slope random 2e5 | 20.9 / 21.2 | - / - | 23.0 / 23.8 | 31.0 / 31.3 | 46.1 / 48.1 |
| tree path 1e6 | 69.3 / 70.3 | 114 / 118 | 93.1 / 91.9 | 127 / 128 | 220 / 221 |
| tree star 1e6 | 35.8 / 36.2 | 70.2 / 75.7 | 152 / 151 | 125 / 121 | 147 / 148 |
| tree caterpillar 1e6 | 58.9 / 59.9 | 75.4 / 78.1 | 82.9 / 81.1 | 91.3 / 92.3 | 185 / 185 |
| tree random 1e6 | 117 / 119 | 167 / 175 | 131 / 134 | 154 / 157 | 408 / 406 |
| slope path 1e6 | 73.1 / 75.2 | - / - | 97.6 / 103 | 131 / 132 | 127 / 128 |
| slope caterpillar 1e6 | 72.3 / 72.9 | - / - | 92.5 / 94.5 | 131 / 128 | 164 / 163 |
| slope random 1e6 | 256 / 255 | - / - | 259 / 262 | 380 / 379 | 431 / 435 |
| mix 1 heap, push 50 / pop 50 | 17.2 / 17.5 | 21.2 / 20.1 | 14.5 / 15.2 | 21.6 / 23.3 | 25.6 / 25.4 |
| mix 1000 heaps, push 50 / pop 30 / meld 20 | 21.0 / 21.6 | 28.0 / 28.3 | 25.7 / 26.2 | 37.5 / 38.3 | 38.2 / 36.6 |
| mix 1e5 heaps, push 40 / pop 20 / meld 40 | 25.0 / 25.6 | 35.3 / 35.6 | 26.2 / 26.3 | 39.3 / 39.8 | 53.8 / 52.6 |

Pairing has the lowest time in 16 of 17 shapes in both flag sets; skew wins the
single-heap push/pop mix (14.5 / 15.2 versus 17.2 / 17.5). Rank-leftist pays for
rank maintenance; binary meld reinserts elements. Handles add a predecessor
link and update it on every link, which explains their extra traffic.

`benchmarks/ds/meldable_heap_tuning.cpp` isolates each knob on a 2e5-vertex random
tree and one million mixed operations over 1000 heaps. Same CPU/compiler/build
and flags, measured 2026-10-15; plain/native milliseconds:

| method / knob | tree | mix |
|---|---:|---:|
| by rank leftist / recursive, uint8 rank | 21.7 / 22.2 | 38.3 / 38.2 |
| by rank leftist / iterative, uint8 rank | 18.9 / 19.8 | 38.8 / 37.1 |
| by rank leftist / iterative, uint32 rank | 19.5 / 20.4 | 39.3 / 38.9 |
| by skew / branch | 14.2 / 14.2 | 28.2 / 29.1 |
| by skew / select | 13.2 / 13.3 | 27.4 / 27.3 |
| by pairing / branch, free list | 12.6 / 13.2 | 22.1 / 21.6 |
| by pairing / select, free list | 12.9 / 13.8 | 22.4 / 21.8 |
| by pairing / branch, no free list | 15.9 / 17.4 | 22.8 / 22.5 |
| by split pairing / split arrays | 15.9 / 15.1 | 22.1 / 22.5 |
| by pairing / unused lazy tags | 18.4 / 18.0 | 26.9 / 26.6 |
| by pairing / unused lazy tags, no identity skip | 23.6 / 23.6 | 33.1 / 32.8 |
| by rank leftist / unused lazy tags | 24.2 / 22.7 | 41.0 / 40.9 |
| by binary / push each | 44.0 / 47.1 | 37.7 / 39.7 |
| by binary / bulk rebuild | 54.1 / 54.5 | 43.0 / 43.1 |
| by pairing / branch, free list, reserved | 12.2 / 12.6 | — |
| by rank leftist / iterative, uint8 rank, reserved | 17.9 / 18.8 | — |

| slope method, 2e5 random tree | time |
|---|---:|
| by pairing | 22.5 / 22.6 |
| by pairing / no identity skip | 23.6 / 23.1 |
| by skew / select | 23.5 / 24.3 |
| by rank leftist | 33.7 / 31.8 |
| by binary | 46.6 / 47.4 |

`release` empties a heap in one walk over its links, with no pairing. The
breakpoint pool of
[`convex_piecewise_linear_forest`](convex-piecewise-linear-forest.md) uses it
for `clear` and for the discarded right heap of `take_all`;
[the forest benchmark](../benchmarks/dp/convex_piecewise_linear_forest.cpp)
races that pool against the same pool emptying by repeated pops. Same CPU,
compiler and build, measured 2026-11-01, plain/native milliseconds: clearing
one set of 1e6 takes 1357 / 1348 with repeated pops and 74.1 / 75.3 with
release; clearing and refilling 1000 sets of 1000 over three rounds takes
1724 / 1693 and 632 / 631. The
[forest page](convex-piecewise-linear-forest.md#measured-performance) has
every shape.

## Pitfalls and notes

Top/value references can be invalidated by pool growth or reserve. Handles remain
stable across melds, but their containing heap id changes. Lazy shifts and
handles cannot be combined: computing a handled node's shift would require an
unbounded ancestor walk. A released element's payload can remain stored until
its slot is reused; release is logical removal and recycling.

## How it works

Children form first-child/next-sibling lists. Linking roots attaches the worse
root below the better. Pop pairs children left to right and folds those pairs
right to left, threading temporary lists through sibling links. Freed nodes
form another list. Pending shifts cover the node's binary subtree; its own
value is already shifted. Identity equality allows shift propagation to be
skipped. The handle tier adds a predecessor link for constant-time cuts.
The iterative skew contender avoids recursion because a right spine can have
linear length; tests build that shape with one million elements.

## See also

- [Convex piecewise-linear forest](convex-piecewise-linear-forest.md): the slope trick on trees over two of these forests (a min-heap and a max-heap per function)
- [Persistent heap](persistent-heap.md)
- [Lazy segment tree](lazy-segment-tree.md)
