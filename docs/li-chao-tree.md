# `atl::li_chao_tree`, `atl::dynamic_li_chao_tree`

**The lower (or upper) envelope of lines and line segments** (or of any
function family whose members cross at most once), with insertion and point
queries in any order: the Li Chao tree. It covers the convex hull trick
`dp[i] = min_j a_j * x_i + b_j` without needing monotone slopes or monotone
queries, and it accepts segments, meaning a function valid only on `[l, r)`.

```cpp
#include <atl/ds/li_chao_tree.hpp>
#include <atl/ds/dynamic_li_chao_tree.hpp>
#include <atl/ds/linear_function.hpp>
```

\include li-chao-tree/basic.cpp

## API

```cpp
template<typename Coordinate, typename Compare = std::less<>, typename Function = linear_function<Coordinate>>
class li_chao_tree;          // same parameters for dynamic_li_chao_tree (integral Coordinate)

atl::li_chao_tree<int64> t(query_xs);  // sorted distinct points (or t(lo, hi): every integer of [lo, hi))
t.insert(a, b);                        // y = a x + b everywhere          O(log n)
t.insert(a, b, l, r);                  // y = a x + b on l <= x < r only  O(log^2 n)
std::optional<int64> y = t.query(x);   // best at x, none if nothing covers x   O(log n)

atl::dynamic_li_chao_tree<int64> d(-1'000'000'000, 1'000'000'001);   // any [lo, hi), online
d.insert(a, b);  d.insert(a, b, l, r);  d.query(x);
```

### Members

| member | effect | cost |
|---|---|---|
| `insert(f)`, `insert(a, b)` | add a function (line) over the whole domain | O(log n) |
| `insert(f, l, r)`, `insert(a, b, l, r)` | add it on `l <= x < r` only; an empty range adds nothing | O(log^2 n) |
| `query(x)` | best value at `x`, `std::optional` (none if nothing covers `x`) | O(log n) |
| `size()` (array tree), `empty()` | query points; whether the domain is empty | O(1) |
| `node_count()` (dynamic tree) | number of materialized nodes | O(1) |
| `reserve(nodes)` (dynamic tree) | reserve room for that many nodes | O(node_count()) if allocation changes |

`Compare` orders the values: `std::less<>` (the default) keeps minima and
`std::greater<>` keeps maxima. `query` returns `std::optional`. A point that
no function covers has no value, and no infinity is ever stored.

**Any family that crosses at most once.** The algorithm needs only one
property: for any two members `f`, `g`, the points where `f` beats `g` form a
prefix or a suffix of the domain, so the two cross at most once. `Function`
is any [`envelope_function`](../include/atl/ds/linear_function.hpp): a
semiregular callable, `f(x)` at a `Coordinate`, whose result `Compare`
orders. The value type is what `f(x)` returns. Insert members with
`insert(f)` and `insert(f, l, r)`. The crossing property is a documented
precondition, not checked (no concept can check it). Lines are the default,
`atl::linear_function<Coordinate>{slope, intercept}`, and only lines get the
`insert(slope, intercept[, l, r])` shorthands and the exact overflow check
below. The code path is the same for every family.

```cpp
// (x - a)^2 + b: two members differ by a linear function, so they cross at most once
struct parabola {
	int64 a = 0, b = 0;
	constexpr int64 operator()(int64 x) const { return (x - a) * (x - a) + b; }
};
atl::li_chao_tree<int64, std::less<>, parabola> t(0, 1'000'000);
t.insert(parabola{a, b});
t.insert(parabola{a, b}, l, r);
```

Other members of the class: `a * sqrt(x) + b` on `x >= 0`, and any `c(x) + a
* g(x) + b` with a fixed `c` and a monotone `g`.

## Complexity

Whole-domain insertion and `query` are O(log n) (`n` the number of points;
`W = hi - lo` for the dynamic tree, with at most two new nodes per
insertion). A segment insertion is O(log^2 n): O(log n) canonical nodes, each
descending O(log n). The coordinate-list tree adds one binary search per
query. Memory: the array tree has `2 * bit_ceil(n)` slots of `sizeof(Function)`
bytes plus a used bit per slot.

**Dynamic tree memory.** On the measured x86-64 ABI, nodes are 32 bytes for `int64` lines and 64 bytes
for `int128` lines (read off the node struct and confirmed with `sizeof` on
the two instantiations on that ABI: a line, 16 / 32 bytes, the child ids and the kind,
and for `int128` a separate 16-byte bound, since a 64-bit bound is packed
into the unused child fields of a clipped leaf). A whole-domain insertion
creates at most two nodes, a segment about `2 log W` (one boundary path per
end, stopping at the first missing node), so budget `2 * segments * log2(W)`
nodes at most, and the nodes stay fewer than `2^32 - 1` (asserted).

## Preconditions

Asserted (`-DATL_ASSERT`): `lo <= hi`; the array tree's `hi - lo` representable in Coordinate and less than `2^31` and,
for a coordinate list, strictly increasing coordinates, fewer than `2^31`; padded slot count representable in size_t;
`query(x)` at a domain point (a listed coordinate for the coordinate-list
tree, `lo <= x < hi` otherwise); every line evaluates without overflow over
the points it covers (below); fewer than `2^32 - 1` dynamic nodes.

Documented only: any two inserted functions cross at most once over the
domain (no concept can check it); comparisons obey a strict weak order,
including for floating values, with no NaNs or infinite intermediates.

**Overflow.** Every evaluation, including intermediate multiplication, must
be valid on the covered domain points. For `linear_function<Coordinate>`
over an integral Coordinate, insertion checks both `a*x` and `a*x+b` at
the two covered endpoints before computing either operation. Endpoint
bounds suffice because a line and its product are monotone. Cancellation
in the final sum does not permit an overflowing product. These checks use
portable division bounds, including at the signed minimum, and work with
native and software `atl::int128`. Without ATL_ASSERT the same bounds are
caller preconditions. Calling linear_function directly does not check them.
For `|a|, |x| <= 1e9` and `|b| <= 1e18`, int64 suffices. Larger values can
opt into atl::int128, but its intermediates must still fit 128 bits.

A custom Function may return a wider type than Coordinate, for example
linear_function<atl::int128> over int64 query points. Then valid conversion
and all evaluation bounds are documented caller obligations. Comparisons
place lines by comparing values directly; no subtraction of two values or
cross multiplication adds another overflow bound. The dynamic tree forms
floor midpoints with bit arithmetic, never `first + last` or `last - first`,
so it supports domains spanning the signed range. `middle + 1` is evaluated
only while descending a non-singleton interval. Dense array-domain width
is checked with unsigned distance before narrowing. reserve requires
`nodes < SIZE_MAX`; allocation failure propagates from vector.

## Measured performance

`benchmarks/ds/li_chao_tree.cpp` replays the same insertion/query streams
on array slots, online node pools and experimental layout, descent and
query variants. Rows include construction and interleaved operation latency;
phase rows separate construction, insertion and query throughput. Every
variant is checked against the library checksum outside the timed loops.
The array tree knows the query coordinates up front; the online tree does not.

Xeon Platinum 8168, GCC 15.2, RelWithDebInfo, assertions off, 2026-11-03;
minimum of three runs, milliseconds, plain -O2 / -O2 -march=native
(also -g -DNDEBUG -std=c++23). int64 random lines, 2e5 initial inserts
followed by 2e5 mixed operations, except the 1e6-line row.

| workload | library | by materialized paths | by whole first query |
|---|---:|---:|---:|
| lines, absolute x <= 1e9 | 7.5 / 7.8 | 6.9 / 7.1 | 8.0 / 7.6 |
| lines, [0,1e6) | 4.2 / 4.3 | 3.6 / 3.8 | 4.4 / 4.1 |
| 1e6 lines, absolute x <= 1e9 | 37.0 / 38.1 | 33.6 / 34.7 | 39.1 / 36.7 |

| segment workload | library | by whole first query | by omitted dominated exit | by clip overwrite | by optional dominated exit (enabled) | by optional dominated exit (disabled) |
|---|---:|---:|---:|---:|---:|---:|
| absolute x <= 1e9 | 28.8 / 30.6 | 627 / 659 | 629 / 665 | 549 / 593 | 28.7 / 30.3 | 625 / 656 |
| suffix, absolute x <= 1e9 | 20.5 / 21.7 | 256 / 280 | 257 / 286 | 232 / 264 | 20.5 / 21.7 | 253 / 282 |
| [0,1e6) | 28.1 / 29.8 | 424 / 444 | 431 / 454 | 419 / 449 | 28.1 / 29.6 | 426 / 445 |
| suffix, [0,1e6) | 18.9 / 20.0 | 211 / 232 | 212 / 233 | 188 / 208 | 18.9 / 19.8 | 208 / 228 |
| improving, [0,1e6) | 122 / 131 | — | 109 / 134 | — | 121 / 130 | 111 / 124 |

The dominated exit removes most random segment work. Improving segments
show its cost when pruning cannot help; the controlled optional-exit copy
keeps the same layout and code apart from that condition. Whole-first
queries can win or lose depending on flags and domain; they do not give
a consistent improvement. Materialized paths remain competitive on lines
but allocate long boundary paths for segments.

Array coordinate contenders: Xeon Platinum 8280, GCC 15.2, RelWithDebInfo,
assertions off, 2026-11-01; same flags and units. By endpoint carry reuses
endpoint values during descent; by branchless search and by Eytzinger
search change only coordinate lookup. Both searches lose some workloads.

| array coordinate workload | library | by endpoint carry | by branchless search | by Eytzinger search |
|---|---:|---:|---:|---:|
| full segments, queries at lo | 8.7 / 8.5 | 9.1 / 8.9 | 7.7 / 7.9 | 8.5 / 8.0 |
| segments | 196 / 197 | 200 / 206 | 201 / 204 | 195 / 200 |
| lines | 27.6 / 27.3 | 27.8 / 27.8 | 27.5 / 27.7 | 28.7 / 28.6 |
| full segments | 42.0 / 42.0 | 45.5 / 45.4 | 44.1 / 43.9 | 61.4 / 60.8 |
| suffix segments | 140 / 142 | 146 / 147 | 142 / 142 | 149 / 150 |

## Pitfalls and choosing

**Which one.**

- `li_chao_tree(coordinates)`: a perfect binary tree over the query points,
  `2 * bit_ceil(n)` slots. A query asks at a listed point (asserted, one
  binary search). A segment's ends may be any values, matched to the points
  inside. Needs the query points up front, so it suits batch input.
- `li_chao_tree(lo, hi)` (integer `Coordinate`): the same over every integer of
  `[lo, hi)` with no coordinate array and no search. `hi - lo` representable in Coordinate and less than `2^31`, and memory is about
  `2 * bit_ceil(hi - lo) * sizeof(Function)` bytes plus a used bit per slot.
- `dynamic_li_chao_tree(lo, hi)`: integer `Coordinate` only. It bisects
  `[lo, hi)` at floor midpoints and creates a node only where a function
  settles (at most two per whole-domain insertion) or along a segment's
  boundary paths (O(log W) per segment, `W = hi - lo`). Use it when queries
  arrive online over a huge range.

**Not raced:** kinetic tournament trees require monotone query coordinates
and function reassignment. Their interface differs from arbitrary-coordinate
queries on an insert-only set, so they are not drop-in contenders here.

## How it works

On the measured dynamic path, a partial segment stops at an existing
whole node if it fails to beat the held line at the two ends and midpoint
of their overlap. Crossing at most once means it cannot win inside. Every
query there passes through that node; later replacements or descents can
only improve its envelope. Dropping the segment therefore preserves answers
and avoids allocating its boundary paths. Tests use the overlap, never
uncovered points of a segment.

Inserting a line walks one root-to-leaf path. Each node keeps the line that
is better at its midpoint, and the loser goes into the one half where it can
still win (decided by comparing at the node's first point). A function that
wins, or loses, at both ends of a node's range wins or loses on all of it
(the two cross at most once), so the loser is dropped there and the walk
ends. A query takes the best line on its leaf-to-root path, bottom-up in the
array tree. A segment is split into canonical nodes, each insertion then
descending from there.

In the array tree a segment that reaches the last point also takes the
padding leaves past it (they repeat the last point, and no query asks
there), so a whole-domain segment goes to the root alone and a suffix splits
into one node per level, as for a power-of-two size. In the dynamic tree a
segment's boundary path stops at the first node that does not exist yet: the
segment stays there as a clipped leaf, a childless node that holds it on
`[bound, last]` or `[first, bound]` of its interval only (the bound packed
into the unused child fields for a `Coordinate` of 64 bits or less). A later
insertion that passes through a clipped leaf pushes it one level down first.

## See also

- [128-bit integers](int128.md)
