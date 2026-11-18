# `atl::weighted_dsu`, `atl::rollback_weighted_dsu`

Union-find with group potentials records `pot(b) = pot(a) · w` in
`unite(a, b, w)`, and returns `pot(a)⁻¹ · pot(b)` from `difference(a, b)`.
Use it for difference constraints, modular potentials and parity relations;
use the rollback variant when successful merges must be undone. The group may be noncommutative.
Plain connectivity needs only a DSU.

```cpp
#include <atl/ds/weighted_dsu.hpp>
#include <atl/ds/rollback_weighted_dsu.hpp>
```

\include weighted-dsu/potentials.cpp

The last result wraps a negative difference modulo 998244353 using `mod_int`.
For noncommutative potentials, 2x2 matrices with determinant one form SL2 over
the same field; supply matrix multiplication, the identity matrix and the adjugate:

\include weighted-dsu/matrix.cpp

Expected stdout:

\include weighted-dsu/matrix.out

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

The same source races all compression strategies and root-child exits on modular
addition and noncommutative SL2, each with n=q=200000 and modulus 998244353.
The remaining three shapes above also remain in the benchmark. The two-pass
buffer stores the path; two-pass inverse reconstructs weights using inverses;
halving and splitting shorten paths during descent; recursion composes weights
on return. “No root-child exit” disables the one-edge fast path. Plain DSU
measures connectivity on the same unions, without potentials.

Intel Xeon Platinum 8280; GCC 15.2; RelWithDebInfo, asserts off; 2026-10-24.
Flags: `-O2 -g -DNDEBUG -std=c++23` and
`-O2 -march=native -g -DNDEBUG -std=c++23`. Minimum milliseconds over three
runs of 25 samples, plain / native. Labels beginning “by compression and layout”
refer to `weighted_dsu_by_compression_and_layout`.

| Contender | modular addition | SL2 |
|---|---:|---:|
| by compression and layout (two-pass buffer), interleaved | 4.8 / 4.6 | 8.4 / 9.8 |
| by compression and layout (two-pass buffer), split | 5.2 / 5.0 | 8.1 / 8.5 |
| by compression and layout (two-pass inverse), interleaved | 4.4 / 4.3 | 9.7 / 9.7 |
| by compression and layout (two-pass inverse), split | 4.6 / 4.5 | 8.8 / 8.8 |
| by compression and layout (halving), interleaved | 4.4 / 4.4 | 9.2 / 9.4 |
| by compression and layout (halving), split | 4.6 / 4.2 | 8.9 / 8.9 |
| by compression and layout (recursive), interleaved | 3.7 / 3.5 | 7.9 / 7.9 |
| by compression and layout (recursive), split | 4.1 / 4.0 | 7.4 / 7.3 |
| by compression and layout (splitting), interleaved | 4.4 / 4.1 | 8.8 / 9.7 |
| by compression and layout (splitting), split | 4.6 / 4.3 | 8.0 / 8.3 |
| by compression and layout (two-pass buffer), interleaved, no root-child exit | 4.7 / 3.5 | not measured yet |
| by compression and layout (two-pass inverse), interleaved, no root-child exit | 4.3 / 3.5 | not measured yet |
| by compression and layout (halving), interleaved, no root-child exit | 4.2 / 4.1 | not measured yet |
| by compression and layout (splitting), interleaved, no root-child exit | 3.3 / 3.6 | not measured yet |
| by compression and layout (recursive), interleaved, no root-child exit | 3.6 / 3.5 | not measured yet |
| weighted_dsu (library, layout by potential width) | 3.6 / 3.7 | 7.5 / 7.4 |
| baseline: plain dsu, same unites | 2.9 / 2.8 | 3.0 / 3.0 |

Recursive compression with the width-selected layout is competitive on both
modular workloads. The supplied records have no SL2 root-child-exit tuning rows;
those contenders remain in the benchmark. The rollback
width dispatch itself has no separate measurement: the rollback table reports
explicit interleaved and split layouts.

## Pitfalls and notes

Direction matters: `unite(a, b, w)` implies `difference(a, b) == w`, while the
reverse difference is `inverse(w)`. With noncommutative groups, reversing an operand
order changes the answer. The test oracle builds a constraint forest and walks
products in both directions, including modular addition, SL2 matrices and
permutations of five elements.
The permutation group supplies an independent check of operand order.

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
- [Modular arithmetic](modular-arithmetic.md)
