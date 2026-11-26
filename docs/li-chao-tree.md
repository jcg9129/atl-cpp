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

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| workload | library | by materialized paths | by whole first query |
|---|---:|---:|---:|
| lines, absolute x <= 1e9 | 5.5 / 5.3 | 5.7 / 5 | 5.6 / 5.1 |
| lines, [0,1e6) | 2.5 / 2.4 | 2.6 / 2.2 | 2.6 / 2.3 |
| 1e6 lines, absolute x <= 1e9 | 25.8 / 24.9 | 26.4 / 23.6 | 26.4 / 24.1 |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| segment workload | library | by whole first query | by omitted dominated exit | by clip overwrite | by optional dominated exit (enabled) | by optional dominated exit (disabled) |
|---|---:|---:|---:|---:|---:|---:|
| absolute x <= 1e9 | 18.8 / 19.3 | 469 / 478 | 459 / 477 | 407 / 403 | 18.9 / 18.8 | 467 / 480 |
| suffix, absolute x <= 1e9 | 13.4 / 13.4 | 174 / 179 | 175 / 181 | 159 / 170 | 13.5 / 12.7 | 172 / 177 |
| [0,1e6) | 18.5 / 19.2 | 281 / 288 | 286 / 283 | 292 / 283 | 18.5 / 18.3 | 277 / 286 |
| suffix, [0,1e6) | 12.2 / 12.4 | 131 / 134 | 130 / 137 | 117 / 110 | 12.3 / 11.6 | 126 / 131 |
| improving, [0,1e6) | 78.1 / 72.4 | — | 72.4 / 67.2 | — | 78.7 / 69.9 | 70.6 / 67.8 |

The dominated exit removes most random segment work. Improving segments
show its cost when pruning cannot help; the controlled optional-exit copy
keeps the same layout and code apart from that condition. On lines, whole-first queries beat the library native (5.1 versus
5.3 ms over absolute x<=1e9) and lose plain (5.6 versus 5.5 ms).
On improving segments the library loses to omitted dominated exit
(78.1/72.4 versus 72.4/67.2 ms); the random segment rows favor keeping
the exit. Materialized paths remain competitive on lines
but allocate long boundary paths for segments.

Array coordinate contenders: By endpoint carry reuses
endpoint values during descent; by branchless search and by Eytzinger
search change only coordinate lookup. Both searches lose some workloads. The library loses array segments to
Eytzinger search (132/140 versus 127/126 ms), and full-segment queries at
lo to branchless search (4/3 versus 3.5/2.8 ms).

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-22.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| array coordinate workload | library | by endpoint carry | by branchless search | by Eytzinger search |
|---|---:|---:|---:|---:|
| full segments, queries at lo | 4 / 3 | 4.3 / 3.3 | 3.5 / 2.8 | 3.8 / 3.2 |
| segments | 132 / 140 | 134 / 143 | 133 / 144 | 127 / 126 |
| lines | 21.5 / 22.1 | 21.3 / 21.8 | 21.7 / 22.8 | 22.7 / 21.2 |
| full segments | 30.5 / 30.9 | 30.5 / 30.8 | 30.9 / 31.4 | 48.1 / 39.1 |
| suffix segments | 95.9 / 98.7 | 97 / 98.8 | 93.4 / 100 | 104 / 103 |

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
