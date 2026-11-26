# `atl::suffix_automaton`

Recognize and count substrings across one word or a collection using an online
generalized suffix automaton. Use it for state-based dynamic programming.

```cpp
#include <atl/strings/suffix_automaton.hpp>
```

\include suffix-automaton/quick.cpp

## API

`template<std::size_t Alphabet> requires(Alphabet <= (uint64(1) << 32))`
selects the exclusive integral alphabet bound. `state_id` is `uint32`.
Word operations require `trie_word<R, Proj>`: an input range projected to
`std::integral` symbols. The callback requires `std::invocable<Visit &, uint32, state_id>`.

```cpp
using state_id = uint32;
suffix_automaton();
template<std::integral Sym>
state_id extend(Sym sym);
state_id finish_word();
[[nodiscard]] state_id last() const;
template<typename R, typename Proj = std::identity>
requires trie_word<R, Proj>
state_id insert(R &&word, Proj proj = {});
template<typename R, typename Proj = std::identity>
requires trie_word<R, Proj>
[[nodiscard]] bool contains(R &&word, Proj proj = {}) const;
[[nodiscard]] state_id root() const;
template<std::integral Sym>
[[nodiscard]] std::optional<state_id> walk(state_id s, Sym sym) const;
[[nodiscard]] state_id suffix_link(state_id s) const;
[[nodiscard]] uint32 length(state_id s) const;
[[nodiscard]] uint32 shortest_length(state_id s) const;
template<typename Visit>
requires std::invocable<Visit &, uint32, state_id>
void for_each_transition(state_id s, Visit visit) const;
[[nodiscard]] std::vector<state_id> states_by_length() const;
[[nodiscard]] std::vector<uint64> occurrence_counts() const;
[[nodiscard]] std::vector<bool> terminals() const;
[[nodiscard]] uint64 distinct_substring_count() const;
std::size_t state_count() const;
std::size_t transition_count() const;
void reserve(std::size_t symbols);
void clear();
```

`extend` returns the current word's state. `insert` extends each projected
symbol and finishes the word; `finish_word` records a nonempty word end and resets
`last` to `root`. `contains` recognizes substrings, including the empty word.
`walk` returns an optional transition target. Non-root suffix links shorten the
state class; its shortest and longest lengths delimit that class. State ids are
creation order; use `states_by_length` for topological iteration.
`occurrence_counts` includes all fed symbols; its root entry counts positions.
`terminals` marks suffix classes of finished words, always including the root.
`distinct_substring_count` excludes the empty substring. Duplicate words count
with multiplicity. Transitions are visited in unspecified hashed order.

## Complexity

For L fed symbols, storage and state/transition counts are O(L+1), with
at most 2L+1 states and 3L transitions. Expected build time is
O(L log(degree+1)); `extend` has this expected amortized per-symbol cost,
`contains` O(m log(degree+1)), and `walk` expected O(log(degree+1)).
`finish_word` is amortized O(1); scalar queries are O(1).
Length ordering and occurrence counts take O(states + maximum word length)
time and auxiliary storage. Terminals take O(states + finished nonempty words);
distinct counting takes O(states). Reserve is linear in requested capacity;
clear is O(states + transitions + finished words), retaining capacity.
Traversal and clone copying use loops or heap stacks, never recursion per state.

## Preconditions

Asserted under `ATL_ASSERT`: symbols supplied to `extend` are in
[0, Alphabet), handles are valid, non-root operations exclude the root,
state ids and word lengths stay below 2^32 - 1, and per-state ending-position
seeds fit `uint32`. Reserve multiplication and counting-sort sizes must fit
`size_t`; reserve capacity must also fit the state and pool id domain.
Documented only: projection is stable. Queries with negative, wider-than-64-bit
or out-of-alphabet symbols return absence (compared at the symbol's own width). All integral character types work; signed negative bytes
need an unsigned-byte projection if intended as byte values.
Handles remain valid until clear, assignment or destruction. No input views are
retained. The seeded child pool makes construction and operations runtime-only;
allocation failures propagate as standard exceptions.

## Measured performance

[The benchmark](../benchmarks/strings/suffix_automaton.cpp) reserves and inserts
whole texts. By transition reinsertion looks up split transitions and reinserts
clone edges; the library carries found entries and copies the sibling tree
structurally.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape / operation | Contender | -O2 ms | native ms |
|---|---|---|---|
| 1e6 sigma=2 | library | 196 | 273 |
| 1e6 sigma=2 | by transition reinsertion | 308 | 355 |
| 1e6 sigma=26 | library | 406 | 554 |
| 1e6 sigma=26 | by transition reinsertion | 386 | 671 |
| 1e6 sigma=1000000 | library | 511 | 734 |
| 1e6 sigma=1000000 | by transition reinsertion | 703 | 872 |
| 1e6 unary | library | 14.1 | 14.2 |
| 1e6 unary | by transition reinsertion | 28 | 25.9 |
| 1e5 words x10 sigma=26 | library | 279 | 390 |
| 1e5 words x10 sigma=26 | by transition reinsertion | 322 | 453 |

The library loses sigma=26 plain (406 versus 386 ms for transition
reinsertion), while winning native (554 versus 671 ms). Both contenders
on this shape vary between runs (spreads up to 1.75x), so minima alone
do not establish stable rankings.

## Pitfalls and notes

Transition order is unsuitable for lexicographic descent: collect and sort
outgoing symbols when needed. For pattern multiplicity, walk the pattern and
read its state's occurrence count. Empty-pattern occurrence conventions differ
from ordinary string matching: the root counts positions, not word boundaries.

## How it works

Each state represents an end-position equivalence class. Extending a new
symbol adds transitions along suffix links until a matching transition is found;
when its target class is too long, a clone splits the class and receives redirected
links. Structural clone copying uses an explicit heap stack. Restarting at the
root between words prevents cross-word substrings. Counting sort orders states
by length; descending suffix-link propagation adds occurrence seeds. The child
pool orders siblings by a seeded hash, giving expected logarithmic lookup rather
than a deterministic balancing guarantee.

## See also

[Suffix tree](suffix-tree.md), [Suffix array](suffix-array.md), [Trie](trie.md).
