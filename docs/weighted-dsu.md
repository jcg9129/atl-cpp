# `atl::weighted_dsu`, `atl::rollback_weighted_dsu`

Union-find with group potentials records `pot(b) = pot(a) · w` in
`unite(a, b, w)`, and returns `pot(a)⁻¹ · pot(b)` from `difference(a, b)`.
Use it for difference constraints and parity relations; use the rollback variant
when successful merges must be undone. The group may be noncommutative.
Plain connectivity needs only a DSU.

```cpp
#include <atl/ds/weighted_dsu.hpp>
#include <atl/ds/rollback_weighted_dsu.hpp>
```

\include weighted-dsu/potentials.cpp

## API

```cpp
enum class unite_outcome { merged, consistent, contradicts };

template<std::copyable Potential = int64, binary_operation<Potential> Op = std::plus<>,
         inverse_operation<Potential> Inverse = std::negate<>>
    requires std::equality_comparable<Potential>
class weighted_dsu;

template<std::copyable Potential = int64, binary_operation<Potential> Op = std::plus<>,
         inverse_operation<Potential> Inverse = std::negate<>>
    requires std::equality_comparable<Potential>
class rollback_weighted_dsu;
```

The concepts check copyability, equality syntax, const-callability of composition
and inversion, and conversion of their results to `Potential`. They cannot verify
any group law. Composition and inversion use `std::invoke`, including member
function pointers. Every size constructor checks
`std::constructible_from<Op, Op&&> && std::constructible_from<Inverse, Inverse&&>`
for direct construction into storage, accepting explicit move constructors.
Omitted operations are value-initialized by constrained delegating overloads:
`Op{}` and `Inverse{}` must be valid only when that argument is omitted.
The one-argument overload additionally checks `Potential{}`. These same rules
apply to the experimental weighted variants.

| `weighted_dsu` signature | Effect |
|---|---|
| `weighted_dsu()` | Empty partition |
| `constexpr explicit weighted_dsu(std::size_t n)` | `n` singletons with identity `Potential{}` |
| `constexpr weighted_dsu(std::size_t n, Potential identity)` | Value-initialize both operations |
| `constexpr weighted_dsu(std::size_t n, Potential identity, Op op)` | Value-initialize the inverse |
| `constexpr weighted_dsu(std::size_t n, Potential identity, Op op, Inverse inverse)` | `n` singletons over the supplied group |
| `constexpr std::size_t elements() const noexcept` | Number of elements |
| `constexpr std::size_t count() const noexcept` | Number of components |
| `constexpr std::size_t find(std::size_t x)` | Representative |
| `constexpr bool same(std::size_t a, std::size_t b)` | Whether elements share a component |
| `constexpr std::size_t size(std::size_t x)` | Component size |
| `constexpr unite_outcome unite(std::size_t a, std::size_t b, Potential w)` | Record a relation |
| `constexpr std::optional<Potential> difference(std::size_t a, std::size_t b)` | Relative potential or `std::nullopt` across components |

The exact constructor constraints are:

```cpp
constexpr explicit weighted_dsu(std::size_t n)
    requires std::constructible_from<Op, Op&&> && std::constructible_from<Inverse, Inverse&&>
          && requires { Potential{}; Op{}; Inverse{}; };
constexpr weighted_dsu(std::size_t n, Potential identity)
    requires std::constructible_from<Op, Op&&> && std::constructible_from<Inverse, Inverse&&>
          && requires { Op{}; Inverse{}; };
constexpr weighted_dsu(std::size_t n, Potential identity, Op op)
    requires std::constructible_from<Op, Op&&> && std::constructible_from<Inverse, Inverse&&>
          && requires { Inverse{}; };
constexpr weighted_dsu(std::size_t n, Potential identity, Op op, Inverse inverse)
    requires std::constructible_from<Op, Op&&> && std::constructible_from<Inverse, Inverse&&>;
```

Replace the constructor name with `rollback_weighted_dsu` for the rollback class;
its `find`, `same`, `size` and `difference` are `const`. It also supplies:

| Signature | Effect |
|---|---|
| `constexpr std::size_t time() const noexcept` | Number of successful merges still live |
| `constexpr void undo()` | Undo the latest successful merge |
| `constexpr void rollback(std::size_t t)` | Undo back to history length `t` |

`merged` joins different components. Within a component, `consistent` accepts an
already implied relation and `contradicts` rejects a conflicting one. Neither adds
a relation or rollback entry. The compressed class may still relink paths during
these calls; the represented constraints remain unchanged.

## Complexity

Construction and storage are O(n), counting one potential per element and, for
rollback, at most n−1 three-word log entries. `elements`, `count` and `time` take O(1).
Other queries and union take O(α(n)) amortized group operations with compression,
or O(log(n + 1)) worst-case group operations without it. Rollback `undo` takes O(1)
potential assignments; `rollback(t)` undoes `time() − t` merges. Costs of user-supplied
group operations, copies and assignments multiply these bounds.

## Preconditions

Asserted under `ATL_ASSERT`: `n <= INT32_MAX`; all indices are less than `elements()`;
`undo()` requires `time() > 0`; `rollback(t)` requires `t <= time()`.

Documented-only group laws: composition is associative; `identity` is neutral on
both sides; `op(x, inverse(x)) == op(inverse(x), x) == identity`; equality represents
group equality. Operations must preserve the group and have defined intermediates.
No algebraic law is asserted. The one-argument constructor needs `Potential{}` to
be the identity; a multiplicative group normally supplies its identity explicitly.

For signed integer addition, **every evaluated intermediate** must fit: root-relative
sums, inverse values, differences and root-lifting products. Merely keeping answers
in range is insufficient. Native signed integers and native `atl::int128` have the
language's signed overflow rules; the software int128 backend wraps, so callers
must respect the common range contract for portable additive results.

Rollback tokens belong to the same object's still-live history prefix. They cannot
recover a branch already undone and replaced by new merges. Undo restores parents,
sizes, component count and the detached child's identity weight. Rejected constraints
are never remembered as an inconsistency flag. Allocation and user-operation failures
propagate; transactional recovery after throwing group operations is not promised.

## Measured performance

Weighted DSU contenders vary compression (two-pass buffer, two-pass inverse,
halving, splitting, recursion), node layout, and root-child exit. Rollback
compares interleaved nodes with split links/potentials. The loops measure
throughput of constraint/query streams. The supplied rollback record calls the
interleaved contender the library and the split contender “split”; the current
library selects split storage for wide potentials and interleaved storage for
compact ones. The table identifies the measured methods, rather than attributing
these timings to the current width dispatch.

Source: [`benchmarks/ds/weighted_dsu.cpp`](../benchmarks/ds/weighted_dsu.cpp).
CPU: Intel Xeon Platinum 8280; GCC 15.2; RelWithDebInfo, asserts off; 2026-11-01.
Flags: `-O2 -g -DNDEBUG -std=c++23` and
`-O2 -march=native -g -DNDEBUG -std=c++23`. Cells are minimum milliseconds
over three runs of 25 samples, plain / native.

| Shape | by interleaved nodes | by split nodes |
|---|---|---|
| modular addition n=q=200000 | 3.9 / 3.9 | 3.6 / 3.7 |
| random mix n=1e6, 2e6 ops, int64 addition | 103 / 103 | 84.1 / 86.2 |
| binomial trees n=2^20 + 1e6 queries, int64 addition | 249 / 254 | 176 / 168 |
| noncommutative SL2 n=q=200000 | 9.1 / 9.4 | 8.5 / 8.6 |
| parity n=1e6, 2e6 ops | 79.9 / 80.7 | 83.0 / 79.8 |

Current width dispatch, compression-strategy contenders and root-child-exit
tunings: not measured yet. The modular and SL2 rows describe recorded shapes;
the current benchmark uses only the published integer and parity shapes.

## Pitfalls and notes

Direction matters: `unite(a, b, w)` implies `difference(a, b) == w`, while the
reverse difference is `inverse(w)`. With noncommutative groups, reversing an operand
order changes the answer. The test oracle builds a constraint forest and walks
products in both directions, including permutations of five elements. This checks
operand order without relying on modular arithmetic.

## How it works

Each node stores a parent-or-negative-size and a parent-relative weight:
`pot(x) = pot(parent(x)) · weight(x)`. A root's weight is the identity. Union by size
bounds depth by 30 for the supported element count. Recursive compression first
repairs the parent, then replaces a child's weight with `weight(parent) · weight(child)`.
Potentials at most four bytes wide share a node with the link; wider ones use separate
arrays so the descent reads dense links before touching weights on its return.

If `pa` and `pb` are root-relative potentials, consistency is `pa · w == pb`.
Attaching root `rb` below root `ra` needs weight `pa · w · inverse(pb)`; reversing
the attachment uses its inverse. A difference is `inverse(pa) · pb`.
Rollback omits compression and logs the two roots and old child size. Detaching the
child restores its identity weight without logging a potential. This gives bounded
undo work independent of the potential group's operation cost.

## See also

- [DSUs](dsu.md)
- [Time divide and conquer](time-divide-and-conquer.md)
