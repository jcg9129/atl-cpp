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

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| workload | library pairing | pairing + handles (addressable) | by skew | by rank leftist | by binary |
|---|---|---|---|---|---|
| tree path 2e5 | 10.7 / 11.6 | 13.7 / 16.1 | 13.2 / 14.1 | 14.4 / 15 | 30.6 / 29.5 |
| tree star 2e5 | 5.4 / 5.6 | 7.1 / 7.7 | 17.2 / 16.9 | 12.5 / 12.6 | 19.9 / 19.6 |
| tree caterpillar 2e5 | 9.1 / 9.4 | 11.9 / 12.7 | 11.3 / 11.6 | 12.1 / 12.2 | 24.8 / 24.2 |
| tree random 2e5 | 9.5 / 9.8 | 12.4 / 13 | 10.2 / 10.2 | 13.8 / 14 | 31.5 / 31.3 |
| slope path 2e5 | 11 / 11.7 | - / - | 14.5 / 15.2 | 16.4 / 17 | 17.1 / 16.6 |
| slope caterpillar 2e5 | 10.3 / 10.8 | - / - | 13.1 / 13.5 | 15.9 / 15.9 | 21.2 / 20.8 |
| slope random 2e5 | 14.7 / 15.6 | - / - | 16.7 / 17.5 | 21.7 / 21.9 | 31.3 / 31.8 |
| tree path 1e6 | 54.2 / 59.1 | 98.3 / 110 | 68 / 72.9 | 102 / 104 | 172 / 166 |
| tree star 1e6 | 25.7 / 26.8 | 34.2 / 37.2 | 116 / 117 | 67.2 / 68 | 102 / 101 |
| tree caterpillar 1e6 | 46.6 / 48.5 | 60.5 / 65 | 59.9 / 61.8 | 62.3 / 63.1 | 135 / 132 |
| tree random 1e6 | 62.7 / 64.9 | 93.6 / 97.7 | 67.3 / 69 | 96.2 / 99.2 | 270 / 275 |
| slope path 1e6 | 55.6 / 59.6 | - / - | 75.3 / 79 | 101 / 105 | 96.1 / 93.4 |
| slope caterpillar 1e6 | 52.6 / 55.7 | - / - | 68.7 / 70.6 | 98.7 / 98.7 | 117 / 115 |
| slope random 1e6 | 165 / 184 | - / - | 175 / 179 | 280 / 275 | 291 / 299 |
| mix 1 heap, push 50 / pop 50 | 11.7 / 13.1 | 15.6 / 15.4 | 11.6 / 12 | 15.9 / 16.2 | 19.9 / 19.8 |
| mix 1000 heaps, push 50 / pop 30 / meld 20 | 15.1 / 15.9 | 20.5 / 20.7 | 20.2 / 20.6 | 27.5 / 27.6 | 28.7 / 28.9 |
| mix 1e5 heaps, push 40 / pop 20 / meld 40 | 16.1 / 16.9 | 20.7 / 22.3 | 18.3 / 18.4 | 25.3 / 26.4 | 35.7 / 35.6 |

Pairing has the lowest minimum in 16 of 17 plain shapes and 15 of 17
native shapes. It loses the single-heap push/pop mix to skew
(11.7/13.1 versus 11.6/12 ms), and slope random 1e6 native
(184 versus 179 ms), while winning that row plain (165 versus 175 ms). Rank-leftist pays for
rank maintenance; binary meld reinserts elements. Handles add a predecessor
link and update it on every link, which explains their extra traffic.

`benchmarks/ds/meldable_heap_tuning.cpp` isolates each knob on a 2e5-vertex random
tree and one million mixed operations over 1000 heaps. Plain/native milliseconds, with measurement setup below:

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| method / knob | tree | mix |
|---|---:|---:|
| by rank leftist / recursive, uint8 rank | 14.9 / 15.7 | 30.3 / 28.9 |
| by rank leftist / iterative, uint8 rank | 14 / 13.6 | 29.8 / 28.1 |
| by rank leftist / iterative, uint32 rank | 14.7 / 13.9 | 30.5 / 28.6 |
| by skew / branch | 11 / 10.5 | 23.6 / 21.3 |
| by skew / select | 10.8 / 10.1 | 22.4 / 20.5 |
| by pairing / branch, free list | 9.1 / 9.6 | 17.5 / 15.7 |
| by pairing / select, free list | 9.3 / 9.8 | 17.7 / 17 |
| by pairing / branch, no free list | 10.4 / 11.2 | 16.7 / 16.2 |
| by split pairing / split arrays | 11.2 / 12 | 17.5 / 17.4 |
| by pairing / unused lazy tags | 12.1 / 13.8 | 19.7 / 19.1 |
| by pairing / unused lazy tags, no identity skip | 17.4 / 16.9 | 25.4 / 23.9 |
| by rank leftist / unused lazy tags | 17.8 / 16.5 | 33.1 / 30.7 |
| by binary / push each | 33.3 / 29.4 | 33.1 / 28.6 |
| by binary / bulk rebuild | 37.1 / 34.7 | 34.7 / 31.5 |
| by pairing / branch, free list, reserved | 8.7 / 9.6 | — |
| by rank leftist / iterative, uint8 rank, reserved | 14.1 / 13.5 | — |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| slope method, 2e5 random tree | time |
|---|---:|
| by pairing | 15.3 / 16.2 |
| by pairing / no identity skip | 15.6 / 16.7 |
| by skew / select | 17.3 / 17 |
| by rank leftist | 22.6 / 21.2 |
| by binary | 34 / 31.2 |

`release` empties a heap in one walk over its links, with no pairing. The
breakpoint pool of
[`convex_piecewise_linear_forest`](convex-piecewise-linear-forest.md) uses it
for `clear` and for the discarded right heap of `take_all`;
[the forest benchmark](../benchmarks/dp/convex_piecewise_linear_forest.cpp)
races that pool against the same pool emptying by repeated pops. On the same CPU, compiler, build and flags, measured 2026-11-22:
clearing one set of 1e6 takes 1820/1830 ms with repeated pops and
83.3/71.2 ms with release; clearing and refilling 1000 sets of 1000 over
three rounds takes 1330/1400 and 170/157 ms (plain/native). The
[forest page](convex-piecewise-linear-forest.md#measured-performance) has
every shape.

Random-tree, slope-random and many-heap mix rows include timings that vary
between runs (spreads up to 2.49x). In particular, the native slope-random
1e6 comparison varies, so its minimum alone does not establish a stable
skew win.

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
