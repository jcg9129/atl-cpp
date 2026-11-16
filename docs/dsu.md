# `atl::dsu`, `atl::rollback_dsu`, `atl::persistent_dsu`

Union-find maintains a partition of elements `[0, n)`. Use `dsu` for mutable
connectivity, `rollback_dsu` for a traversal that undoes merges in reverse order,
and `persistent_dsu` when several versions must remain queryable. All three use
union by size; only the mutable DSU compresses paths.

```cpp
#include <atl/ds/dsu.hpp>
#include <atl/ds/rollback_dsu.hpp>
#include <atl/ds/persistent_dsu.hpp>
```

\include dsu/history.cpp

## API

All three classes provide a default constructor for an empty partition and
`explicit dsu(std::size_t n)`, `explicit rollback_dsu(std::size_t n)`, or
`explicit persistent_dsu(std::size_t n)` for `n` singleton components.
The exact size constructor declarations are `constexpr explicit dsu(std::size_t n)`,
`constexpr explicit rollback_dsu(std::size_t n)` and
`explicit persistent_dsu(std::size_t n)`. Each default constructor is defaulted.

| Query signature (`dsu`, also `constexpr` for `rollback_dsu`) | Result |
|---|---|
| `constexpr std::size_t elements() const noexcept` | Number of elements |
| `constexpr std::size_t count() const noexcept` | Number of components |
| `constexpr std::size_t find(std::size_t x)` | Component representative |
| `constexpr bool same(std::size_t a, std::size_t b)` | Whether the elements share a component |
| `constexpr std::size_t size(std::size_t x)` | Component size |

`rollback_dsu` makes `find`, `same` and `size` `const`. `persistent_dsu` has the
same const queries without `constexpr`; its supplied backing arrays use runtime
shared ownership. Representatives are implementation choices and may change on a merge.

| Mutation signature | Effect |
|---|---|
| `constexpr bool dsu::unite(std::size_t a, std::size_t b)` | Return whether two components merged |
| `constexpr bool rollback_dsu::unite(std::size_t a, std::size_t b)` | Merge and log only successful merges |
| `constexpr std::size_t rollback_dsu::time() const noexcept` | Number of merges still in history |
| `constexpr void rollback_dsu::undo()` | Undo the last successful merge |
| `constexpr void rollback_dsu::rollback(std::size_t t)` | Undo until history length equals `t` |
| `persistent_dsu unite(std::size_t a, std::size_t b) const &` | Return a new version; copy the backing handle |
| `persistent_dsu unite(std::size_t a, std::size_t b) &&` | Return a new version; move the backing handle |

The persistent overloads are `[[nodiscard]]`. They preserve the existing version's
partition, including when the two elements are already connected.

```cpp
template<detail::persistent_signed_array Backing = persistent_array<int32>>
class persistent_dsu;
```

The backing concept checks copyability, default construction, a signed integral
indexed value type, construction from `(std::size_t, value_type)`, a size convertible
to `std::size_t`, and the actual chained update
`std::move(b).set(i, value_type()).set(j, value_type())` implicitly convertible to
`Backing`. Each element argument is a prvalue. A chainable proxy result is allowed;
a merely convertible result without the second `set` is rejected. Persistence,
read isolation and operation costs are semantic requirements. The default
`persistent_array<int32>` supports branching; `semi_persistent_array<int32>`
allows updates only on the newest version.

## Complexity

Construction takes O(n) time and space. `elements()` and `count()` take O(1).

| Operation | `dsu` | `rollback_dsu` | `persistent_dsu` with supplied arrays |
|---|---|---|---|
| `find`, `same`, `size`, `unite` | O(α(n)) amortized | O(log(n + 1)) worst case | O(log²(n + 1)) worst case |
| `undo` | — | O(1) | — |
| `rollback(t)` | — | O(time() − t) | — |
| Copy a version | O(n) | O(n) | O(1) |

Rollback keeps at most n−1 history entries, each three `uint32` words. A persistent
merge makes two backing-array updates and O(Radix · (1 + log_Radix n)) extra storage
with full persistence. Versions share append-only node pools: dropping a handle
reclaims no individual nodes; the last handle frees the family. Partial persistence
uses amortized O(1) extra nodes per merge for fixed Radix.

## Preconditions

Asserted under `ATL_ASSERT`: all element indices are less than `elements()`;
`n <= INT32_MAX` for the mutable/rollback classes and `n` fits the signed backing
value type for the persistent class; `undo()` requires `time() > 0`;
`rollback(t)` requires `t <= time()`.

Documented-only: a rollback token must describe a still-live prefix of this object's
history. It is a length, not a saved copy: after undoing and making different merges,
an old token cannot recover the abandoned branch. With a partially persistent
backing, only the newest version may be updated; the supplied array asserts this
when an actual backing update occurs. Shared persistent pools need external
synchronization for concurrent updates. The moved-from persistent object may only
be destroyed or assigned to. Allocation failures propagate; rollback restoration
on exceptional exit is not promised.

## Measured performance

The library DSU uses full compression; by path halving relinks to grandparents.
Rollback uses a 12-byte log entry; by compact log stores 8 bytes and recovers the
surviving root on undo. Persistent DSU path-copies its backing arrays. Linear
histories, random/binomial unions, batched rollback and version-tree traversal
measure throughput, including construction where it occurs in the timed body.

Source: [`benchmarks/ds/dsu.cpp`](../benchmarks/ds/dsu.cpp).
CPU: Intel Xeon Platinum 8280; GCC 15.2; RelWithDebInfo, asserts off; 2026-10-15.
Flags: `-O2 -g -DNDEBUG -std=c++23` and
`-O2 -march=native -g -DNDEBUG -std=c++23`. Cells are minimum milliseconds
over three runs of 25 samples, plain / native.

| Shape | Contender | ms |
|---|---|---|
| 200000 unions, n=200000 | library dsu | 4.0 / 3.7 |
| 200000 unions, n=200000 | library rollback_dsu | 4.7 / 5.1 |
| 200000 unions, n=200000 | library persistent_dsu | 210 / 209 |
| 1000000 same queries | library rollback_dsu | 23.3 / 24.6 |
| 1000000 same queries | library persistent_dsu | 110 / 93.8 |
| n=1e6, 1e6 random unions + 1e6 queries | library full compression | 40.4 / 39.8 |
| n=1e6, 1e6 random unions + 1e6 queries | by path halving | 39.1 / 40.7 |
| n=2^20, binomial unions + 1e6 queries | library full compression | 23.1 / 20.8 |
| n=2^20, binomial unions + 1e6 queries | by path halving | 25.4 / 26.6 |
| 1e7 unions, batches of 64 + rollback | library 12-byte log | 296 / 271 |
| 1e7 unions, batches of 64 + rollback | by compact log | 298 / 301 |
| version tree n=q=200000 | library rollback_dsu | 6.5 / 6.8 |
| version tree n=q=200000 | by compact log | 6.3 / 6.3 |
| version tree n=q=200000 | library persistent_dsu | 256 / 253 |

## Pitfalls and notes

A redundant union does not advance rollback time. Undo restores connectivity,
component sizes and component count. Pure reads require no undo entries because
rollback and persistent DSUs omit path compression. An offline walk over a version
tree often needs rollback alone, avoiding persistent-array lookup and allocation.

## How it works

One signed array holds parents (nonnegative) and negative component sizes at roots.
A merge attaches the smaller component's root to the larger, so each extra hop at
least doubles component size. Mutable `find` first finds the root and then relinks
the entire path. Rollback stores the child root, surviving root and old child size;
those restore both changed slots and the component count in O(1). Persistent union
copies the two modified array paths and carries the component count in each handle.
Its root walk has O(log n) hops, each a trie lookup, explaining the second log factor.

## See also

- [Persistent arrays](persistent-array.md)
- [Weighted DSUs](weighted-dsu.md)
- [Time divide and conquer](time-divide-and-conquer.md)
