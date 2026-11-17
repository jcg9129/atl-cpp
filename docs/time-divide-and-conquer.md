# `atl::time_divide_and_conquer`, `atl::lifetimes_of`

The segment tree over time applies items alive on half-open intervals to an
insert-only structure, using rollback or copies to discard them later. Visit
callbacks see exactly the active collection at each selected time, in increasing
time order. Use this for an offline log known in advance, such as dynamic
connectivity; it does not answer an unknown future log online.

```cpp
#include <atl/ds/time_divide_and_conquer.hpp>
```

\include time-divide-and-conquer/connectivity.cpp

## API

```cpp
struct item_lifetime { uint32 begin, end; };          // [begin, end)

enum class event_kind : uint8 { add, remove, query };
template<typename Key> struct item_event { event_kind kind; Key key; };

template<std::move_constructible Save, std::move_constructible Restore>
struct by_rollback { Save save; Restore restore; };
struct by_copy_t;
inline constexpr by_copy_t by_copy{};
struct every_time;                                    // the predicate that is always true

template<typename Structure, lifetime_range Lifetimes, item_insertion<Structure> Insert,
         time_visit<Structure> Visit, time_predicate ShouldVisit = every_time,
         restore_mode<Structure> Mode = default_restore_t<Structure>>
constexpr void time_divide_and_conquer(Structure &structure, std::size_t time_count, Lifetimes const &lifetimes,
                                       Insert insert, Visit visit, ShouldVisit should_visit = {},
                                       Mode mode = default_restore<Structure>());

// every time visited, explicit mode
template<typename Structure, lifetime_range Lifetimes, item_insertion<Structure> Insert,
         time_visit<Structure> Visit, restore_mode<Structure> Mode>
constexpr void time_divide_and_conquer(Structure &structure, std::size_t time_count, Lifetimes const &lifetimes,
                                       Insert insert, Visit visit, Mode mode);

template<item_event_range Events>
[[nodiscard]] std::vector<item_lifetime> lifetimes_of(Events const &events);
```

The interval's equality and the default mode/predicate helpers are:

```cpp
constexpr bool item_lifetime::operator==(item_lifetime const &) const = default;
constexpr bool every_time::operator()(uint32) const noexcept;

template<typename Structure>
using default_restore_t = typename detail::default_restore_mode<Structure>::type;

template<typename Structure>
[[nodiscard]] constexpr default_restore_t<Structure> default_restore();
```

`every_time` returns true, and `default_restore` returns the member-pointer rollback
policy or `by_copy`. `by_copy_t` has `explicit by_copy_t() = default;`.

| parameter | meaning |
|---|---|
| `structure` | the state with no items; rollback mode returns it to that state, copy mode never inserts into it |
| `time_count` | `T`: the times are `0 .. T-1` |
| `lifetimes` | `lifetime_range`: a sized forward range of `[begin, end)` pairs, with members `begin` / `end` (`item_lifetime`) or tuple-like (`std::pair<int, int>`); item `i` is the `i`-th element |
| `insert(s, i)` | `item_insertion`: invocable with `Structure &` and the item index (`uint32`); applies item `i` |
| `visit(s, t)` | `time_visit`: invocable with `Structure &` and the time (`uint32`); called in increasing `t`; it must leave the represented state unchanged |
| `should_visit(t)` | `time_predicate`: which times to visit, default `every_time{}`; subtrees with no visited time are pruned |
| `mode` | `restore_mode`: `by_rollback{save, restore}` or `by_copy`, next table |

| restore mode | how a tree node is left | requirement |
|---|---|---|
| `by_rollback{save, restore}` | `token = save(s)` before the node's inserts, `restore(s, std::move(token))` after its subtree | `save` invocable on `Structure &`, `restore` on `Structure &` and `save`'s result; both through `std::invoke`, so lambdas, free functions and member pointers work |
| `by_copy` | the node's inserts go into a copy of the parent's state, dropped after its subtree | `std::copyable<Structure>` |
| omitted (`default_restore_t<Structure>`) | `by_rollback{&Structure::time, &Structure::rollback}` if both members exist and are not overloaded, else `by_copy` | as above |

The token can be anything movable: `rollback_dsu::time()`'s `std::size_t`, or a pair
that also saves a counter, to restore extra state. In copy mode, `insert` is
written as for a persistent structure, `s = std::move(s).unite(u, v)` (see
[`persistent_dsu`](dsu.md)), or as a plain mutation of a small copyable structure.

`lifetimes_of(events)` returns one `item_lifetime` per `add` event, in log order, for
`T = events.size()`. **Time `t` is the state after event `t`**: an item added by event
`a` and removed by event `r` lives on `[a, r)`, one never removed on `[a, T)`. A `remove`
closes the **oldest** live copy of its key (keys form a multiset). `begin` is the add
event's index, so item `i`'s key is `events[lifetimes[i].begin].key`. Queries change
nothing: they are the times to select with `should_visit`. A log without queries works
too: visiting every time then sees the state after each change.
`lifetime_range` checks both the value shape and lifetime extraction on the
`auto const&` binding used by the loop, preserving volatile qualification.
`item_event_range` requires a key hashable without throwing by `atl::hash` (specialize it
for an own type), equality comparable, default constructible and copyable. A query's key
is ignored, so it can carry the query's own data. On the same const-bound event
reference the body reads, the concept checks the contextual Boolean comparisons
`event.kind == event_kind::add` and `event.kind == event_kind::remove`, plus
`map[event.key]` and `map.find(event.key)` on the actual flat hash map. Proxy event
references need these operations; conversion to an event value alone is insufficient.

## Complexity

For m items, T times and Q selected times, setup takes O(T + m log(T + 1)) time
and memory, with O(m log(T + 1)) insertions and Q visits. Rollback restoration costs
must be added; when restore costs at most the work it undoes, they have the same
aggregate bound as insertions. Copy mode adds at most O(T) copies/assignments,
with O(log(T + 1)) live copied states. Their storage and saved-token storage are
additional to the engine's arrays.

When `0 < Q <= T / 10`, endpoints are translated to query ranks: setup and memory
become O(T + m log(Q + 1)), with O(m log(Q + 1)) insertions and O(Q) copies. The O(T)
term includes original leaf flags and a rank buffer. With Q == 0 there are no
insertions, visits, saves, restores or copies. `should_visit` runs exactly T times,
in increasing order. T == 0 returns immediately.

`lifetimes_of` takes O(T) expected time for bounded-cost hashing and equality,
and O(T) memory. It iterates const input ranges, including unsized input ranges;
it is runtime-only because its seeded hash map uses runtime state. The engine itself
is `constexpr` when all supplied structures and callbacks support constant evaluation.

## Preconditions

Asserted under `ATL_ASSERT`: `time_count <= 2^30`; fewer than `2^32 - 1` items and
stored cover entries; `begin <= end <= time_count`. Lifetime bounds are checked
only when there is at least one selected time. `lifetimes_of` asserts that every
remove finds a live copy and that there are fewer than `2^32 - 1` events.
Endpoint conversions must preserve the intended nonnegative bounds.

Documented-only: insertion order is unspecified and may be reversed within a node;
insertions must represent an order-independent collection. Visits must preserve the
represented state. Copy mode needs independent represented mutations; copying a
pointer to mutable state alone is insufficient. Save/restore must undo every change
since a saved token, including auxiliary counters. The token must be movable.
Worst-case insertion costs are needed when quoting rollback bounds: undoing and
replaying operations can defeat path-compression amortization. Exceptions propagate;
the engine does not guarantee restoration after a throwing callback or allocation.

## Measured performance

Time divide and conquer races recursive vectors, unpruned/configurable CSR,
dense CSR, frame skipping, rank compression, and ranked frame skipping. Loops
measure whole-log throughput with rollback DSU or persistent DSU by copy. The
supplied record's library row is dense CSR without ranks; the current library
chooses sparse ranks automatically. Forced ranking in the record has no separate
entry point in the current benchmark; its integrated dispatch is not measured yet.

Source: [`benchmarks/ds/time_divide_and_conquer.cpp`](../benchmarks/ds/time_divide_and_conquer.cpp).
CPU: Intel Xeon Platinum 8280; GCC 15.2; RelWithDebInfo, asserts off; 2026-11-01.
Flags: `-O2 -g -DNDEBUG -std=c++23` and
`-O2 -march=native -g -DNDEBUG -std=c++23`. Cells are minimum milliseconds
over three runs of 25 samples, plain / native.

| Shape and restore mode | by dense CSR | by frame skipping | by rank compression | by ranked frame skipping |
|---|---|---|---|---|
| churn n=q=300000, 50% queries, rollback | 20.6 / 19.4 | 21.6 / 20.8 | 19.3 / 19.8 | 19.2 / 18.6 |
| churn n=q=300000, 10% queries, rollback | 29.6 / 28.5 | 29.0 / 29.0 | 17.6 / 17.7 | 17.2 / 16.5 |
| churn n=q=300000, 1% queries, rollback | 34.6 / 32.9 | 33.1 / 32.8 | 13.8 / 13.9 | 13.5 / 13.2 |
| churn n=q=300000, 50% queries, by copy | 487 / 497 | 490 / 500 | 492 / 500 | 496 / 500 |
| churn n=q=300000, 10% queries, by copy | 573 / 577 | 572 / 577 | 550 / 557 | 550 / 553 |
| churn n=q=300000, 1% queries, by copy | 548 / 552 | 546 / 555 | 510 / 517 | 509 / 518 |
| bipartite toggles n=q=100000, every time | 11.1 / 10.7 | 11.4 / 10.8 | 10.8 / 11.4 | not measured yet |

The bipartite row describes the recorded shape; the current source retains the
connectivity shapes. Recursive, unpruned, unstretched, long/short-lived and
lifetimes-only rows: not measured yet. Automatic ranking uses `Q <= T / 10`;
callbacks retain original indices. Frame skipping remains experimental.

## Pitfalls and notes

Normalize undirected keys: `(u, v)` and `(v, u)` are different keys. Repeated adds
form a multiset and each remove closes the oldest live copy. Query keys are ignored
by `lifetimes_of` and may carry query data. Times describe the state **after** that
event: add at a and remove at r gives `[a, r)`; an unremoved item ends at T.

A rollback DSU restores only successful merges. For a parity application, keep a
contradiction counter beside the weighted DSU and save both in a pair token; rejected
constraints do not enter DSU history. Copy mode supports persistent structures by
assigning the returned version back to the callback's local state.

## How it works

A perfect segment tree covers `[0, bit_ceil(T))`. Each lifetime decomposes into
O(log(T + 1)) canonical cover nodes. A DFS inserts a node's items before descending,
visits its leaves, and restores the incoming state when leaving. Thus each leaf
sees exactly its active items. The implementation walks iteratively and stores
items in flat CSR arrays; backward filling leaves row items in decreasing index
order. Padding leaves are never visited, allowing suffix intervals ending at T
to extend to the next power of two and use fewer cover nodes.

Leaf flags prune subtrees with no selected time. At Q <= T / 10, an additional
rank pass maps `[b, e)` to `[rank[b], rank[e])` and visits rank k at its original
time `times[k]`. The lifetime view copies no item values. Only nodes with items
open frames. Copy frames retain slots and copy-assign on reuse; rollback frames
save a token and restore it after the subtree. `lifetimes_of` links each key's
live additions into a FIFO using item ids and a hash map.

Frame skipping stays experimental: it avoids frames whose contents an enclosing
restore will discard, but did not improve copy-mode measurements. The recursive
and unpruned engines remain correctness and performance comparators. Tests check
all variants against explicit active-item sets and an ordered-map FIFO oracle.

## See also

- [DSUs](dsu.md)
- [Weighted DSUs](weighted-dsu.md)
- [Link-cut tree](link-cut-tree.md) — online link/cut and path queries.
