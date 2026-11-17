# `atl::persistent_lazy_segment_tree`


A **fully persistent** lazy segment tree: range apply, range fold, and
cross-version range copy, with every version kept. Mutators are `const` and return
a new version sharing all untouched structure (path copying); any past version
stays queryable *and* mutable, so histories branch freely. The algebra is the
[`lazy_segment_tree`](lazy-segment-tree.md) contract verbatim — element monoid,
tag monoid, distributive action, width carried in `Value` when a tag's effect depends
on it.

```cpp
#include <atl/ds/persistent_lazy_segment_tree.hpp>
```

\include persistent-lazy-segment-tree/basic.cpp

Versions derived from one origin share a single arena (flat node pool, `uint32`
ids); a version handle is a `shared_ptr` + id, O(1) to copy, and the arena is
freed when the last handle dies. Nodes cover exact midpoint intervals of
`[0, n)` — no power-of-two padding.

Two design points worth knowing:

- **Reads never write.** `fold`/`at` descend accumulating the composition of
  pending path tags and evaluate covered nodes as `mapping(acc, sum)` — `const`,
  allocation-free, safe on any version at any time. (An in-place lazy tree must
  *settle* tags on read; a persistent one must not, since nodes are shared.)
- **`copy_range` adopts subtrees.** "Paste `[l, r)` from version `y`" walks only
  the O(log n) boundary; any subtree fully inside (or outside) the range is
  adopted whole by id, with that side's accumulated tag folded into the one
  adopted root. This is the primitive that lets two live versions interact —
  and the reason problems with cross-version copies genuinely require
  persistence (no offline rollback walk can serve two arbitrary versions).

## API

### Construction

| constructor | description | complexity |
|---|---|---|
| `persistent_lazy_segment_tree()` | empty placeholder (when `Value`, `Tag` and the three callables are default-initializable), for a member or global assigned later | O(1) |
| `persistent_lazy_segment_tree(std::size_t n, Value identity, Tag tag_identity, Op = Op{}, Mapping = Mapping{}, Compose = Compose{})` | `n` elements, all `identity` | O(n) |
| `persistent_lazy_segment_tree(R&& values, Value identity, Tag tag_identity, Op = Op{}, Mapping = Mapping{}, Compose = Compose{})` | from a sized range (narrowing deduction rejected) | O(n) |
| `persistent_lazy_segment_tree(commutative_tags_t, std::size_t n, Value identity, Tag tag_identity, Op = Op{}, Mapping = Mapping{}, Compose = Compose{})` | the fill form in the [commutative mode](#commutative-mode) | O(n) |
| `persistent_lazy_segment_tree(commutative_tags_t, R&& values, Value identity, Tag tag_identity, Op = Op{}, Mapping = Mapping{}, Compose = Compose{})` | the range form in the commutative mode | O(n) |

The class is `persistent_lazy_segment_tree<Value, Tag, Op, Mapping, Compose, TagOrder = void>`;
`TagOrder` (concept `atl::tag_order`) is `void` or `atl::commutative_tags_t`, the same
spelling as [`dual_segment_tree`](dual-segment-tree.md#commutative-mode). The tagged
constructors exist only in the commutative mode, and deduction picks that mode from
the leading `atl::commutative_tags`.

The range constructor moves elements only out of an owning rvalue range (an
rvalue that is neither a `borrowed_range` nor a view); an lvalue, an rvalue
`std::span`, or an rvalue view over the caller's container is copied.

### Operations

All mutators are `const`, `[[nodiscard]]`, and return a new version.

Each version-returning member also has an rvalue (`&&`) overload that moves the pool instead of copying it, so `t = std::move(t).apply(...)` skips the atomic refcount traffic of the linear idiom; the moved-from object may only be destroyed or assigned to (see [conventions](conventions.md)).

| function | description | new nodes |
|---|---|---|
| `apply(l, r, Tag const& f)` → version | tag `f` applied to `[l, r)` | O(log n) |
| `set(i, Value x)` → version | element `i` replaced | O(log n) |
| `update(i, f)` → version | element `i` replaced by `f(current)` in one descent (`f` invoked on `Value const &`, result convertible to `Value`) | O(log n) |
| `copy_range(source, l, r)` → version | this version outside `[l, r)`, `source`'s values inside; both must share an arena | O(log n) |
| `fold(l, r)` → `Value` | fold of `[l, r)` in this version | 0 |
| `fold()` → `Value` | fold of the whole range | 0 |
| `at(i)` → `Value` | the value at `i` | 0 |
| `arena_size()` | nodes in the shared arena (all versions' footprint) | — |
| `reserve(ops)` | arena headroom for `ops` further mutations | — |
| `size()` / `empty()` | element count / emptiness | — |

Preconditions as usual (`l <= r <= size()`, `i < size()`, shared arena for
`copy_range`), checked under `-DATL_ASSERT`.

### Commutative mode

When every pair of tags commutes (`compose(f, g) == compose(g, f)`: range add,
chmin, chmax, xor where the aggregate supports it; not assignment, not affine maps),
`apply` can keep tags **permanent** instead of copying them down. Opt in with
`atl::commutative_tags`:

In the default mode a partially covered node settles its tag into fresh copies of
both children before the descent continues: up to three new nodes per partial node
plus two `compose`s. In the commutative mode a partially covered node becomes one
new node that keeps its tag, its aggregate rebuilt as
`mapping(tag, op(left, right))`; untouched children are shared, and fully covered
nodes take the new tag as before. An `apply` therefore allocates exactly one node
per node it visits (about 4 log n at most, against about 8 log n). `fold` and `at`
are unchanged, since they already accumulate the path tags; `set`, `update` and
`copy_range` still settle the tags on their own path, which is valid in either
representation. Commutativity is a documented precondition only, never checked.
`reserve(ops)` sizes by the mode's per-level bound for any one mutation: 4 nodes
per level in the commutative mode, 8 in the default mode (the default `apply`).
Either bound covers every mutation: `copy_range` allocates at most 4 per level and
`set` and `update` at most 3 in both modes; past the reserve the arena grows
geometrically.

## Complexity

The fill and range constructors take O(n) time. `apply`, `set` and `update`
take O(log n) time and create O(log n) new nodes. `copy_range`, `fold(l, r)`
and `at(i)` take O(log n) time; `fold()` takes O(1) time.

## Preconditions

Asserted (`-DATL_ASSERT`): `l <= r <= size()`, `i < size()`, and that
`copy_range`'s source shares this version's arena. Initial size is at most
2^31, node ids fit uint32, and reserve products fit size_t. User arithmetic
must remain representable. APIs are runtime-only because versions use
std::shared_ptr ownership. Allocation failures propagate; mutations are not
transactional if an allocation, element operation or callable throws.

Documented only:

- The `lazy_segment_tree` laws: `(Value, op, identity)` and `(Tag, compose,
  tag_identity)` are monoids, `compose(f, g)` = "`g` first, then `f`", and
  `mapping(f, op(a, b)) == op(mapping(f, a), mapping(f, b))`.
- In the [commutative mode](#commutative-mode), every pair of tags commutes
  (never checked).
- A version moved from by an rvalue overload may only be destroyed or assigned to.

## Measured performance

`benchmarks/ds/persistent_lazy_segment_tree.cpp` measures complete affine/add
streams, point reads, settled versus permanent commuting tags, and whole-node
versus partial copies (links and tag, avoiding unused aggregate copies).

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2, RelWithDebInfo, assertions off.
Milliseconds, minimum of three run means; each pair is plain `-O2` /
`-O2 -march=native`, both also with `-g -DNDEBUG -std=c++23`. Copy comparison measured 2026-11-01.

| shape (2e5 mixed operations) | library whole-node copy | by partial copy |
|---|---:|---:|
| affine n=100k linear | 934 / 961 | 937 / 959 |
| add n=100k linear | 408 / 419 | 404 / 410 |
| add n=100k branching | 472 / 492 | 471 / 474 |
| add n=200k linear | 452 / 467 | 451 / 457 |
| add n=200k branching | 518 / 546 | 519 / 527 |
| wide add n=100k linear, default mode | 1087 / 1102 | 1064 / 1086 |
| wide add n=100k linear, commutative mode | 594 / 613 | 591 / 615 |

The following workloads were measured 2026-10-24 with the same CPU, compiler,
build type and flags. Each commuting tag row also reports node count / MB.

| shape | library persistent | by in-place lazy tree |
|---|---:|---:|
| 2e5 mixed affine operations, linear versions | 567 / 587 | 147 / 142 |

| 1e6 affine point reads after 1e5 applies | time |
|---|---:|
| library direct descent | 444 / 446 |
| by singleton fold | 547 / 547 |

| add shape | default mode | by permanent tags | default nodes / MB | permanent nodes / MB |
|---|---:|---:|---:|---:|
| n=100k linear | 806 / 812 | 437 / 449 | 10524748 / 336.8 | 4624692 / 148.0 |
| n=100k branching | 914 / 924 | 517 / 542 | 10524748 / 336.8 | 4624692 / 148.0 |
| n=200k linear | 882 / 894 | 489 / 501 | 11428509 / 365.7 | 5126301 / 164.0 |
| n=200k branching | 987 / 991 | 568 / 597 | 11428509 / 365.7 | 5126301 / 164.0 |

## Pitfalls and notes

- **Cost of persistence.** Against the in-place [`lazy_segment_tree`](lazy-segment-tree.md)
  on an identical *linear* history (never branching, the in-place tree's home
  ground), the persistent tree measures **3.86× / 4.13× the time** (567 / 587 vs 147 / 142 ms on 2×10⁵
  mixed affine ops, x86-64 Xeon Platinum 8280, plain -O2 / -O2 -march=native, record of 2026-10-24):
  each mutation allocates its copied path, and traversal is top-down through
  `uint32` links rather than the iterative array walk. Reach for it only when
  versions must coexist; a single timeline that merely needs undo is
  [`rollback`](dsu.md) territory, and an offline version *tree* can often be
  DFS-walked with an in-place structure.
- **Memory.** A mutation allocates at most 8 nodes per level (a default-mode
  `apply`; a commutative `apply` and any `copy_range` at most 4, `set`/`update`
  at most 3),
  each node holding `sizeof(Value) + sizeof(Tag) + 8` bytes; `reserve` sizes for
  that per mode.

## How it works

Path copying over exact midpoint intervals of `[0, n)`, nodes in a shared
`uint32`-addressed arena. A node's aggregate already includes its own tag; the tag
is pending for the subtree below only. Mutations copy the nodes on their path and
share the rest. `fold`/`at` accumulate the composition of the pending path tags
instead of settling them, so reads allocate nothing. `copy_range` adopts whole
subtrees by id with that side's accumulated tag folded into the adopted root.
The default `apply` settles a partial node's tag into fresh copies of both children
before descending; the commutative `apply` keeps the tag on a single rebuilt node,
whose aggregate becomes `mapping(tag, op(left, right))`.

## See also

- [`lazy_segment_tree`](lazy-segment-tree.md) — the in-place original; same algebra
- [`persistent_array`](persistent-array.md) / [`persistent_vector`](persistent-vector.md) — persistence without range operations
