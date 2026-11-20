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

The stored token must be move-constructible and destructible; assignment is
not required. Examples include `rollback_dsu::time()`'s `std::size_t`, or a pair
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

### Experimental graph specializations

These opt-in headers follow the same contracts and test conventions as library
code; their promotion and interface stability are not promised. The offline
solvers are worked applications of the engine. Their names identify the
application APIs; they are not engine contenders.

```cpp
#include <atl/experimental/graph/offline_dynamic_connectivity.hpp>
#include <atl/experimental/graph/offline_bipartiteness.hpp>
#include <atl/experimental/graph/deletion_time_connectivity.hpp>
```

In `atl::experimental`:

```cpp
using vertex_pair = std::pair<vertex_id, vertex_id>;
using graph_event = atl::item_event<vertex_pair>;
using deletion_log = std::span<atl::item_event<std::pair<vertex_id, vertex_id>> const>;

[[nodiscard]] inline std::vector<uint32> component_counts(std::size_t n, std::span<graph_event const> events);
[[nodiscard]] inline std::vector<uint8> bipartite_after_each(
    std::size_t n, std::span<std::pair<vertex_id, vertex_id> const> toggles);

struct parity_state {
    atl::rollback_weighted_dsu<uint32, std::bit_xor<uint32>, std::identity> sides;
    uint32 odd_cycles = 0;
};

class deletion_time_connectivity {
public:
    explicit deletion_time_connectivity(std::size_t n);
    uint32 insert(vertex_id u, vertex_id v, uint32 deletion_time);
    void erase(uint32 handle);
    [[nodiscard]] bool connected(vertex_id u, vertex_id v);
    [[nodiscard]] uint32 component_count() const;
};

template<typename OnQuery>
    requires std::invocable<OnQuery &, deletion_time_connectivity &, uint32>
void replay_by_deletion_time(std::size_t n, deletion_log events, OnQuery &&on_query);
[[nodiscard]] inline std::vector<uint32> component_counts_by_deletion_time(std::size_t n, deletion_log events);
[[nodiscard]] inline std::vector<uint8> connected_by_deletion_time(std::size_t n, deletion_log events);
```

`component_counts` returns the number of components at each query, in log order.
Its graph is an undirected multigraph: parallel adds are separate copies, a remove
closes the oldest live copy, reversed endpoints name the same edge, and self-loops
are allowed. Query keys are ignored. The deletion-time count wrapper has the same
semantics. `connected_by_deletion_time` instead returns a 0/1 byte for connectivity
of the endpoints in each query key. All wrappers leave the input unchanged and
retain no input views.

`bipartite_after_each` uses a simple edge set: each toggle adds an absent edge or
removes a present one, independent of orientation. It returns one 0/1 byte per
toggle for the graph's bipartiteness at that time. A self-loop makes the answer 0
until toggled away. `parity_state::sides` stores xor differences in Z/2;
`odd_cycles` counts detected contradictions and may stop at one when further
insertions cannot change the answer. A rollback token restores both fields.

The online `deletion_time_connectivity` starts with n isolated vertices. `insert`
returns handles 0, 1, ... in insertion order, stable even when a forest edge is
replaced. `erase` removes a live handle; `connected(u, v)` tests live connectivity
(including `u == v`); `component_count()` includes isolated vertices. The replay
helper preprocesses a complete log to determine deletion times, then invokes
`on_query(structure, t)` at each query, in increasing original time order, through
`std::invoke`. Reference wrappers and capturing, mutable or const callables work.
The callback may query the structure but must preserve its represented graph; its
structure reference is borrowed only for the duration of the call.

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

For the experimental offline solvers, q events/toggles cost
O(n + q log(q + 1) log(n + 1)) time and O(n + q log(q + 1)) memory, including
expected O(q) hashing. The n term initializes the DSU even for an empty log.
Deletion-time construction costs O(n); insert and erase take expected amortized
O(log(n + 1)) including forest hash operations, connectivity queries take amortized
O(log(n + 1)), and the count takes O(1). I insertions retain O(n + I) storage,
including erased handles. Log wrappers take expected amortized
O(n + q log(n + 1)) time and O(n + q) memory, including preprocessing. All walks
are iterative; a path of a million vertices creates no per-vertex call stack.

The log wrappers and bipartite solver are runtime-only because their hash tables
use runtime seeds. The online class also currently exposes runtime-only members.

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
since a saved token, including auxiliary counters. The stored token must be move-constructible and destructible.
Worst-case insertion costs are needed when quoting rollback bounds: undoing and
replaying operations can defeat path-compression amortization. Exceptions propagate;
the engine does not guarantee restoration after a throwing callback or allocation.

For the experimental solvers, asserted bounds are `n <= 2^31 - 1`, all non-query
endpoints below n, and matching live copies for removes. The offline DSU wrappers
also inherit `q <= 2^30` and the engine's cover-entry limit; deletion-time log
wrappers require fewer than `2^32 - 1` events. Query endpoints are ignored for
counts; connectivity query endpoints must be below n (asserted by the forest).
`graph::none` is never a valid vertex or returned handle. Empty logs and n == 0
are valid when no edge or connectivity query needs a vertex.

Direct deletion-time insertion asserts fewer than `2^32 - 1` existing handles
and a deletion time below `2^32 - 1`. Documented-only: erase each live handle once,
in increasing `(deletion_time, handle)` order; newly inserted keys must exceed
all keys erased so far. `erase` asserts only that its handle was allocated.
Exceptions propagate, with no transactional restoration guarantee.

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

The source includes connectivity and bipartite-toggle shapes. Both fixed-seed
timed bipartite-toggle shapes stay forests. Untimed validation repeatedly
creates and removes odd cycles, compares every engine's complete answer vector
against fresh graph coloring, and checks dense and sparse selected times.
Recursive, unpruned,
unstretched, long/short-lived and lifetimes-only rows: not measured yet. Automatic ranking uses `Q <= T / 10`;
callbacks retain original indices. Frame skipping remains experimental.

The forced rank-compression row cuts rollback time at 1% queries by
60.1% / 57.8%, and at 10% by 40.5% / 37.9% (plain/native), relative to dense CSR.
Sparse copy-mode reductions range from 3.5% to 6.9%. At every time visited,
forced ranking is 2.7% faster / 6.5% slower. Frame skipping costs 4.9% / 7.2%
on balanced rollback churn and shows no copy-mode gain. Ranked frame skipping
remains experimental; these records do not establish an advantage over ranking
alone. Rollback is much cheaper than persistent copies on these workloads.

### Deletion-time connectivity

The connectivity race compares a link-cut maximum spanning forest keyed by deletion
time with the offline rollback-DSU specialization. Both timed solvers normalize and
pair the raw log and return a component count per query; the loop measures whole-log
throughput, including preprocessing.

Source: [`benchmarks/graph/deletion_time_connectivity.cpp`](../benchmarks/graph/deletion_time_connectivity.cpp).
CPU: Intel Xeon Platinum 8168; GCC 15.2; RelWithDebInfo, asserts off; 2026-10-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` and
`-O2 -march=native -g -DNDEBUG -std=c++23`. Minimum milliseconds over three
runs of ten samples, plain / native:

| Shape | component_counts by deletion time | component_counts by rollback DSU |
|---|---|---|
| churn n=q=300000, 50% queries | 40.1 / 40.0 | 32.0 / 32.0 |
| churn n=q=300000, 1% queries | 74.1 / 70.1 | 41.4 / 39.5 |
| long-lived n=q=300000, 90% never cut, 50% queries | 65.7 / 63.0 | 59.6 / 61.4 |
| short-lived n=q=300000, link-query-cut | 49.5 / 47.3 | 14.3 / 14.3 |
| dense churn n=1000, q=300000, 50% queries | 27.7 / 27.8 | 29.8 / 30.1 |

For these recorded logs, rollback is 1.25 / 1.25 times as fast on balanced churn,
1.79 / 1.77 times on sparse churn, 1.10 / 1.03 times on long-lived edges, and
3.46 / 3.31 times on short-lived edges. The deletion-time forest is 1.08 / 1.08
times as fast on dense churn. Prefer rollback when its measured workload fits;
use the forest when deletion times are known on insertion but queries arrive
online or depend on earlier answers. Replay wrappers still require the complete
log. These measurements do not isolate the effect of automatic rank dispatch.

## Pitfalls and notes

Normalize undirected keys: `(u, v)` and `(v, u)` are different keys. Repeated adds
form a multiset and each remove closes the oldest live copy. Query keys are ignored
by `lifetimes_of` and may carry query data. Times describe the state **after** that
event: add at a and remove at r gives `[a, r)`; an unremoved item ends at T.

### Worked graph applications

`offline_dynamic_connectivity.hpp` specializes the engine for component counts
at `query` events, using `rollback_dsu` and a query-time predicate. Its tests
rebuild a fresh DSU over the live multiset at each query as the certifying oracle.
`offline_bipartiteness.hpp` visits every toggle with no query events, uses
`rollback_weighted_dsu<uint32, std::bit_xor<uint32>, std::identity>`, and checks
answers against an iterative BFS 2-colouring. Insertion stops once a contradiction
exists because every time in that subtree is already non-bipartite.
`deletion_time_connectivity.hpp` is checked against both the offline engine and
fresh-DSU answers, including parallel edges, loops and equal deletion times.

A rollback DSU restores only successful merges. For a parity application, keep a
contradiction counter beside the weighted DSU and save both in a pair token; rejected
constraints do not enter DSU history. Copy mode supports persistent structures by
assigning the returned version back to the callback's local state. It also suits
small copyable states such as an xor basis or a bounded-capacity knapsack array;
rollback requires worst-case insertion bounds because replay defeats amortization.

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

The deletion-time solver maintains a maximum spanning forest of the live graph,
ordered by `(deletion_time, handle)`. An insertion joining two components links
its edge. An insertion closing a cycle replaces the path minimum only if its key
is larger; a self-loop folds to the identity and cannot enter the forest. When a
tree edge is erased, every remaining edge has a larger key. A remaining non-tree
edge crossing its cut would contradict the maximum-forest cycle rule, so no
replacement search is necessary. The edge tier of `link_cut_tree` supplies the
path minimum and iterative link/cut operations.

Connectivity-specific divide and conquer with graph contraction can remove the
DSU logarithm; it has not been benchmarked here, so its constant-factor advantage
is unknown. The generic engine also supports non-graph states under deletions.

## See also

- [DSUs](dsu.md)
- [Weighted DSUs](weighted-dsu.md)
- [Link-cut tree](link-cut-tree.md) — online link/cut and path queries.
- [XOR bases](xor-basis.md) — small copyable state under offline deletion.
