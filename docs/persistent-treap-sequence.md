# `atl::persistent_treap_sequence`

A fully persistent implicit treap for branching sequences with insertion, erasure, split, concat and optional folds and lazy tags. Use a fixed-shape persistent segment tree when positions never change.

```cpp
#include <atl/ds/persistent_treap_sequence.hpp>
```

\include persistent-treap-sequence/basic.cpp

## API

`template<std::semiregular T, typename Op = void, typename F = void,
typename Mapping = void, typename Compose = void> class persistent_treap_sequence`.
`Op` is void or `binary_operation<Op,T>`; tags require a semiregular F,
`action<Mapping,F,T>` and `binary_operation<Compose,F>`. Folds require an
associative Op and identity; tags form a monoid, compose(newer,older) applies
older first, and mapping respects composition and distributes over Op.
Aliases: value_type=T, tag_type=F, size_type=std::size_t.

Construction is `persistent_treap_sequence()` for the plain tier, or from a
sized range: `(R&& values)`, `(R&& values, T s_identity, OpArg op)`, and
`(R&& values, T s_identity, FArg f_identity, OpArg op, MapArg mapping, CompArg compose)`.
Explicit class arguments put T in a non-deduced context; deduction rejects
narrowing from the range value type. Owning rvalues and yielded prvalues may
move; lvalues and views copy.

| operation | result / cost |
|---|---|
| `insert(std::size_t i, T x)`, `erase(std::size_t i)`, `erase(std::size_t l, std::size_t r)`, `set(std::size_t i, T x)` | new version; expected O(log n) time/new nodes |
| `push_back(T x)`, `push_front(T x)` | new version; expected O(log n) |
| `apply(std::size_t l, std::size_t r, FArg const& f)` | new version, tagged tier; expected O(log n) |
| `split(std::size_t i)` | pair of prefix/suffix versions; expected O(log n) |
| `split_first()`, `split_last()` | pair of singleton/rest or rest/singleton versions; end-path cost, valid at saturated sizes |
| `concat(persistent_treap_sequence const& other)` | new version, same arena; expected O(log n) below saturation |
| `T at(std::size_t i) const`, `T fold(std::size_t l, std::size_t r) const` | allocation-free expected O(log n); fold tier for fold |
| `T fold() const` | whole fold, fold tier; O(1) |
| `for_each(Visit visit) const` | visit every logical value; O(n), invocable on T const& |
| `compacted() const` | balanced copy in a new pool; active callables must be `std::copy_constructible`; O(n) |
| `size() const noexcept`, `empty() const noexcept` | O(1), size saturates at 2^32-1 |
| `arena_size() const noexcept`, `reserve(std::size_t ops)` | shared node count, capacity hint in mutations |

Versions use shared pool ownership and are runtime-only (`std::shared_ptr`).
Copying a version costs O(1). Version-returning members have `const &` and `&&`
overloads: the latter consumes the handle; the moved-from object may only be
destroyed or assigned to. Allocation failure propagates an exception; mutations
are not transactional if allocation, element operations or callables throw.

## Complexity

Bounds above are expected below the size cap. Saturated joins pin the shorter
opposing spine; end splits copy and balance the end path. These structural
operations cost their visited path length, rather than a logarithm inferred
from the capped size. Storage is all nodes allocated across versions; compacted
creates a fresh pool and leaves existing versions queryable.

Constructor callable parameters use `std::type_identity_t<OpArg>` (and
`FArg`, `MapArg`, `CompArg` on the tagged tier), with defaulted template
arguments equal to the class parameters. They are non-deduced, so const-qualified
callable types accept ordinary callable arguments.

`compacted()` requires `(!has_op || std::copy_constructible<Op>) &&
(!has_tags || (std::copy_constructible<Mapping> && std::copy_constructible<Compose>))`.
An operation with a deleted copy constructor, or with an explicitly deleted
move constructor that prevents `std::copy_constructible`, cannot be compacted.

## Preconditions

Asserted under `ATL_ASSERT`: valid positions/ranges, nonempty end splits,
same arena for concat, and unsaturated size for positional operations and
compacted. Range construction requires size <= 2^32-1; reserve products fit
std::size_t and pool ids fit uint32. Algebraic laws and representable user arithmetic
are documented only. At and fold return values; callbacks must not mutate the
shared pool while traversal holds node references.

## Measured performance

`benchmarks/ds/persistent_treap_sequence.cpp` compares a chain of persistent
versions with an in-place treap on the same insert/fold stream, including
construction. **Not measured yet.**

## Pitfalls and notes

A saturated size represents any logical length at or above 2^32-1. Only
structural end splits, concat, whole fold and traversal remain exact then.
Traversal still visits every logical occurrence, which may be enormous.
Give tags equality so identity tags can avoid otherwise necessary copies.
A compacted result cannot be concatenated with a version of its source pool.

## How it works

Joins use a **size-weighted random coin per call** rather than
stored priorities: version sharing means a tree can be concatenated with itself,
and stored priorities would then compare nodes against themselves, voiding the
expected-depth bound — per-call randomness cannot be correlated by sharing (and
the node drops its priority field: 20 bytes + payload). Verified by an
adversarial self-concat stress: lists doubled to ~5·10⁸ logical elements stay
O(log)-deep. Sizes are `uint32` and **saturate** above 2³² − 1 (self-concat doubles lengths
without bound): positional operations (`insert`, `split(i)`, `at`, `fold(l, r)`)
are exact only below the cap, while `split_first`/`split_last`/`concat`/`fold()`
remain exact at any logical size.

Past the cap the weighted coin is meaningless (~50/50), and an interleaving
merge of two giants would pay the *sum* of their facing spines — repeated giant
merges then grow spines without bound (thousands of copies per op). Saturated
joins therefore switch to a **spine-pinning merge** (`treap_base::merge_spines`):
the shorter of a's right spine and b's left spine — found by a lockstep walk, no
stored ranks — is spliced whole into the interior, where end operations never
look; each end spine of the result is an input's plus at most one node.
Complementing it, `split_first`/`split_last` rebuild the end path they must copy
anyway **balanced**, so a long spine costs its length once and O(log) on every
later end-split. Together these keep the adversarial giant workloads (cross-
merging lineages, giant ++ singleton, merge-storm-then-pop-hammer) at a few
dozen new nodes per operation, amortized — pinned by a regression test.

**Tags never settle on the hot paths.** `apply` is a direct descent (applies
change no structure, so — exactly like a persistent lazy segment tree — pending
tags just compose into the O(log n) path copies), and the tags tier's
split/merge carry the accumulated ancestor tag down instead of settling it:
one path copy per level, plus one for an off-path child under a live tag.
**Give tag types `operator==`** (`= default` suffices) so identity tags are
detectable and that second copy is skipped; without `==` every tag is treated
as live — correct, just heavier. The equality check avoids copying nodes
when a deferred update is the identity, especially during cross-version copies.

Handles are value-semantic and O(1) to copy (a copy *is* the same version —
contrast the mutable treap's deep clone). Reads accumulate pending tags instead
of settling them. Algebra tiers and laws are exactly `treap_sequence`'s; no
reversal or `max_right` yet (add on demand). No node recycling — versions share
structure, so the arena only grows (`reserve` for headroom; `compacted()` to shed
dead paths).

The plain/fold tiers use the shared treap engine through a path-copying store.
The tagged tier uses accumulator-carrying joins and splits to avoid settling
tags into shared children. Initial construction uses midpoint recursion; reads
carry ancestor tags and allocate no nodes. Callable objects initialize directly
in the pool, including capturing lambdas.

## See also

- [Treap sequence](treap-sequence.md) — mutable surgery.
- [Persistent lazy segment tree](persistent-lazy-segment-tree.md) — fixed positions.
