# `atl::dual_segment_tree`

```cpp
#include <atl/ds/dual_segment_tree.hpp>
```

The **tags-only** segment tree: apply a tag to a half-open range and read a single
position, both O(log n) — the tool for **update-range / query-point**. It completes
the family:

| structure | update | query | stores |
|---|---|---|---|
| [`segment_tree`](segment-tree.md) | point | range | aggregates |
| **`dual_segment_tree`** | range | point | tags |
| [`lazy_segment_tree`](lazy-segment-tree.md) | range | range | both |

There are no aggregates and therefore **no element operation at all**: the algebra is
just the tag monoid `(Tag, compose, tag_identity)` — `compose(f, g)` meaning "`g` first,
then `f`" — and an action `mapping(f, s)` applying a tag to a **single element**.
Crucially, `mapping` needs no distributivity law (tags never act on folds here), so
`Value` is the bare value with no width smuggling:

```cpp
// Range affine update, point read, over a prime field.
struct affine { mint b, c; };
atl::dual_segment_tree st(
    values, affine{1, 0},                                              // tag_identity
    [](affine const& f, mint s)             { return f.b * s + f.c; }, // mapping
    [](affine const& f, affine const& g)    { return affine{f.b * g.b, f.b * g.c + f.c}; });
st.apply(l, r, affine{b, c});
auto v = st.at(i);
```

On its home workload it measures **1.85–2.11× faster** than a `lazy_segment_tree`
forced into the same job with a width-carrying algebra (see
[Measured performance](#measured-performance)).

\include dual-segment-tree/quick-start.cpp

Expected stdout:

\verbinclude dual-segment-tree/quick-start.out

## Construction

`Value` is deduced from the range's elements (or the fill value), the rest from the
arguments. Elements are moved in only from an rvalue range that owns them (`std::move(vec)`); an lvalue, a borrowed range (`std::span`, `string_view`) or a view (`views::transform`, …) is copied from, so the caller's container is never gutted through a view over it.

| constructor | description | complexity |
|---|---|---|
| `dual_segment_tree()` | an empty tree with value-initialized tag identity and callables (when all are default-initializable): a placeholder assigned later | O(1) |
| `dual_segment_tree(std::size_t n, Value fill, Tag tag_identity, Mapping mapping = Mapping{}, Compose compose = Compose{})` | `n` copies of `fill` | O(n) |
| `dual_segment_tree(R&& values, Tag tag_identity, Mapping mapping = Mapping{}, Compose compose = Compose{})` | from a sized range of initial values | O(n) |
| `dual_segment_tree(commutative_tags_t, std::size_t n, Value fill, Tag tag_identity, Mapping = Mapping{}, Compose = Compose{})` | the fill form in the [commutative mode](#commutative-mode) | O(n) |
| `dual_segment_tree(commutative_tags_t, R&& values, Tag tag_identity, Mapping = Mapping{}, Compose = Compose{})` | the range form in the commutative mode | O(n) |

The class is `dual_segment_tree<Value, Tag, Mapping, Compose, TagOrder = void>`;
`TagOrder` (concept `atl::tag_order`) is `void` or `atl::commutative_tags_t`. The
tagged constructors exist only in the commutative mode, and class template
argument deduction picks that mode from the leading `atl::commutative_tags`.

## Operations

| function | description | complexity |
|---|---|---|
| `apply(std::size_t l, std::size_t r, Tag const& f)` | apply tag `f` to every element of `[l, r)` | O(log n) |
| `apply(std::size_t i, Tag const& f)` | apply tag `f` to the single element at `i` | O(log n) |
| `at(std::size_t i)` → `Value` | the value at `i` — **`const` and write-free**: path tags are folded on the fly, never settled | O(log n) |
| `set(std::size_t i, Value x)` | assign `x` to position `i` (settles the path tags first) | O(log n) |
| `values()` → `std::vector<Value>` | every element with its pending tags folded in, by one top-down pass composing each node's tag into its children's — `const`, nothing settled; use it instead of `n` calls to `at` to read everything back | O(n) |
| `size()` / `empty()` | element count / emptiness | O(1) |

Ranges are half-open with `l <= r <= size()`, indices `i < size()`; checked under
`-DATL_ASSERT`. Concurrent reads also require thread-safe user callables.

## Commutative mode

When every pair of tags commutes (`compose(f, g) == compose(g, f)`: range add, xor,
chmin, chmax; not assignment, not affine maps), the order of the tags on a path
does not matter, so the boundary push that keeps them in temporal order is wasted
work. Opt in with the tag `atl::commutative_tags`:

```cpp
auto add = [](long long f, long long s) { return f + s; };
atl::dual_segment_tree st(atl::commutative_tags, std::size_t(n), 0LL, 0LL, add, add);
st.apply(l, r, 5);  // composes onto the canonical nodes only, no pushes
st.apply(i, -2);    // composes onto the leaf only, no path push
auto v = st.at(i);  // unchanged: folds the path tags
```

`apply(l, r, f)` then skips the up to 2·log n boundary `push_down`s and `apply(i, f)`
skips the root-to-leaf push; `at`, `set` and `values()` are unchanged. Commutativity
is a documented precondition only: nothing checks it, and a non-commuting algebra
in this mode returns wrong values. The default mode (`TagOrder = void`) preserves composition order.

## Preconditions

Asserted (`-DATL_ASSERT`): ranges are half-open with `l <= r <= size()`, indices
`i < size()`.

Documented only:

- `(Tag, compose, tag_identity)` is a monoid, `compose(f, g)` = "`g` first, then
  `f`"; `mapping(tag_identity, s) == s` and `mapping(compose(f, g), s) == mapping(f, mapping(g, s))`. No distributivity law is needed.
- In the [commutative mode](#commutative-mode), every pair of tags commutes. It
  is never checked; a non-commuting algebra returns wrong values.

## Measured performance

`benchmarks/ds/dual_segment_tree.cpp`: n=1e6, 1e6 operations,
including construction, int64 values and tags.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.
Every timing pair is plain / native.

| workload | commutative dual tree | difference_fenwick |
| --- | --- | --- |
| range add / point get, 50/50 | 79.4 / 77.3† | 57.8 / 59.7 |
| range add / point get, 90/10 | 109 / 108 | 90.9 / 92.6 |
| range xor / point get, 50/50 | 79.0 / 74.8 | 56.9 / 60.4 |
| range xor / point get, 90/10 | 110 / 104 | 94.8 / 93.5 |

The same source also measures affine updates and tag-order modes:

| workload | library | comparison |
| --- | --- | --- |
| affine apply/read, 50/50 | 457 / 421 | 844 / 890, lazy aggregation |
| add/read, 50/50, ordered / commutative | 118† / 112† | 75.5 / 77.2†, commutative tags |
| add/read, 90/10, ordered / commutative | 170 / 159† | 106 / 108, commutative tags |

Affine apply/read takes 457 / 421 ms versus 844 / 890 for lazy aggregation,
a 1.85x / 2.11x speedup. † marks a timing with slowest/fastest-run spread
at least 1.25x (up to 1.33x here). The tag-order minima alone do not establish
stable ratios; the plain 90/10 row also supports skipping boundary pushes
for commutative tags (106 versus 170 ms).

The commutative dual tree loses all four shown invertible workloads to
[`difference_fenwick`](difference-fenwick.md): add takes 79.4 / 77.3 versus
57.8 / 59.7 ms at 50/50 and 109 / 108 versus 90.9 / 92.6 at 90/10;
xor takes 79.0 / 74.8 versus 56.9 / 60.4 ms at 50/50 and 110 / 104 versus
94.8 / 93.5 at 90/10. The native 50/50 add dual-tree timing has 1.25x spread;
the plain row and both 90/10 rows also favor Fenwick. The dual tree stays
available for other tag actions, `set` and `values()`. It does not automatically
dispatch to the Fenwick wrapper; there is no performance claim for other
sizes, types or operation mixes.

## Pitfalls and notes

- For range add/xor with point get, prefer [`difference_fenwick`](difference-fenwick.md) when the operation has an inverse.
- Use [`lazy_segment_tree`](lazy-segment-tree.md) when range *folds* are also
  needed; this tree cannot answer them.
- `set` settles the path tags first, so it costs the same O(log n) as
  `apply(i, f)`; reading every element is `values()` in O(n), not n calls of `at`.
- Do not opt into the commutative mode for assignment-like or affine tags: the
  result is silently wrong.

Memory is O(n) base values and tags. `values()` also uses O(n) scratch tags.
Construction asserts that the doubled power-of-two layout fits `size_t`;
callable arithmetic and composition must remain in range.

## How it works

The base values sit untouched in a flat array while tags live on the shared
iterative tree ([`detail::segment_tree_base`](segment-tree.md)). `apply(l, r, f)`
composes `f` onto the O(log n) canonical nodes of `[l, r)`; `at(i)` folds the tags
on the root-to-leaf path of `i` into the stored value, so a read is O(log n) and
never writes. There is no pull-up pass, since there are no aggregates to repair.

In the default mode the invariant is that along any root-to-leaf path, deeper
tags are older. `apply` keeps it by first pushing the partially-covered boundary
ancestors down onto their children (tag onto tag only), so a new tag always lands
on a node whose ancestors are all newer or empty; this preserves temporal order
for non-commutative tags like assign. `at` then applies the path's tags deepest
first. In the commutative mode the invariant does not hold and is not needed:
order on a path is irrelevant, so both applies skip the pushes entirely. `set`
and `values()` behave identically in both modes. `values()` makes one top-down
pass composing each node's tag into its children's, then maps each leaf once.

## See also

- [`difference_fenwick`](difference-fenwick.md) — invertible commutative updates and point reads
- [`lazy_segment_tree`](lazy-segment-tree.md) — when range *folds* are also needed
