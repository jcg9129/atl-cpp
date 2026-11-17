# `atl::segment_tree_beats`

Segment tree beats: a lazy segment tree whose tag application may *fail* on a
node, in which case the update descends ("beats") into the node's children.
That is what range `a[i] = min(a[i], x)` needs once a range **sum** (or count)
is queried: the new sum depends on how many elements exceed `x`, which a plain
lazy tree's aggregate cannot know, but a node holding its maximum, strict
second maximum and the maximum's count can, whenever `x` lies above the
second maximum. The everyday class `atl::beats_segment_tree` is range chmin /
chmax / add / assign with range sum / min / max, hand-written for that one algebra. The engine
`atl::segment_tree_beats` carries three more library algebras (range modulo,
floor sqrt / divide, bitwise and / or) and any user algebra in the
`lazy_segment_tree` style. Reach for either only when a sum or count is queried:
chmin with min / max queries only is an ordinary
[`lazy_segment_tree`](lazy-segment-tree.md) (chmin distributes over max), and
chmin with point reads only is a [`dual_segment_tree`](dual-segment-tree.md)
in commutative mode.

```cpp
#include <atl/ds/segment_tree_beats.hpp>
#include <vector>

int main() {
    std::vector<long long> a{5, 1, 4, 2, 3};
    atl::beats_segment_tree<long long> t(a);
    t.chmin(0, 5, 3);             // 3 1 3 2 3
    t.add(1, 3, 10);              // 3 11 13 2 3
    t.chmax(0, 5, 4);             // 4 11 13 4 4
    long long s = t.sum(0, 5);    // 36
    long long m = t.max(2, 4);    // 13
    t.assign(0, 2, 0);            // 0 0 13 4 4
    // the longest prefix holding only values up to 4; the predicate gets the prefix's summary
    auto r = t.max_right(0, [](auto const &v) { return v.max <= 4; });  // 2
    return s == 36 && m == 13 && r == 2 && t.element(4) == 4 ? 0 : 1;
}
```

See `tests/ds/segment_tree_beats.cpp` for library and custom algebras.

\include segment-tree-beats/quick-start.cpp

Expected stdout:

\verbinclude segment-tree-beats/quick-start.out

## API: `beats_segment_tree`

```cpp
template<signed_integral Element, signed_integral Sum = Element>
    requires(std::numeric_limits<Sum>::digits >= std::numeric_limits<Element>::digits)
class beats_segment_tree {
public:
    using value_type = beats::extremes<Element, Sum>;  // what the predicates receive
    using size_type = std::size_t;

    constexpr beats_segment_tree();
    constexpr explicit beats_segment_tree(std::size_t n);  // n zeros
    template<std::ranges::sized_range R> requires requires { static_cast<Element>(std::declval<atl::detail::forwarded_element_t<R>>()); }
    constexpr explicit beats_segment_tree(R &&elements);

    constexpr void chmin(std::size_t l, std::size_t r, Element x);
    constexpr void chmax(std::size_t l, std::size_t r, Element x);
    constexpr void add(std::size_t l, std::size_t r, Element x);
    constexpr void assign(std::size_t l, std::size_t r, Element x);
    constexpr void set(std::size_t i, Element x);
    [[nodiscard]] constexpr Element element(std::size_t i);
    [[nodiscard]] constexpr Sum sum(std::size_t l, std::size_t r);
    [[nodiscard]] constexpr Element min(std::size_t l, std::size_t r);
    [[nodiscard]] constexpr Element max(std::size_t l, std::size_t r);
    template<std::predicate<beats::extremes<Element, Sum> const &> Pred>
    [[nodiscard]] constexpr std::size_t max_right(std::size_t l, Pred pred);
    template<std::predicate<beats::extremes<Element, Sum> const &> Pred>
    [[nodiscard]] constexpr std::size_t min_left(std::size_t r, Pred pred);
    [[nodiscard]] constexpr value_type const &at(std::size_t i);
    [[nodiscard]] constexpr value_type fold(std::size_t l, std::size_t r);
    [[nodiscard]] constexpr value_type const &fold() const;
    constexpr std::size_t size() const noexcept;
    constexpr bool empty() const noexcept;
};
```

| member | meaning | time |
|---|---|---|
| `chmin(l, r, x)`, `chmax(l, r, x)` | `a[i] = min(a[i], x)`, `a[i] = max(a[i], x)` on `[l, r)` | O(log² n) amortized per update with `add` in the mix, O(log n) amortized without |
| `add(l, r, x)`, `assign(l, r, x)` | `a[i] += x`, `a[i] = x` on `[l, r)` | O(log n), never descends |
| `set(i, x)`, `element(i)` | point write, point read | O(log n) |
| `sum(l, r)`, `min(l, r)`, `max(l, r)` | range queries; `sum(l, l)` is 0, `min` / `max` need `l < r` | O(log n) |
| `max_right(l, pred)` | largest `r` in `[l, size()]` with `pred(summary of [l, r))`; `pred` monotone in `r`, `pred(identity)` true | O(log n) |
| `min_left(r, pred)` | smallest `l` in `[0, r]` with `pred(summary of [l, r))`; `pred` monotone in `l`, `pred(identity)` true | O(log n) |
| `at(i)`, `fold(l, r)`, `fold()` | the `beats::extremes` summary of a position, a range, everything | O(log n), O(1) for `fold()` |
| `size()`, `empty()` | | O(1) |

Reads other than `fold()`, `size` and `empty` are not `const`: they settle the boundary paths. The predicates see the
full `beats::extremes` summary (`max`, `second_max`, `min`, `second_min`, `sum`,
`max_count`, `min_count`, `width`), the same one the engine's predicates see.

The generic engine members `apply`, `apply_derived` and `set(i, summary)` are **not** on
this class. Anyone needing them on clamp-add spells the engine type, whose named
members answer identically:

```cpp
atl::segment_tree_beats<atl::beats::extremes<long long>, atl::beats::clamp_add<long long>> t(a);
t.apply(0, 5, {2, 0, 9});     // add 2, then clamp to [0, 9]
```

## API: the engine `segment_tree_beats`

```cpp
template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op = beats::merge,
         fallible_action<Tag, Value> Mapping = beats::mapping, binary_operation<Tag> Compose = beats::compose>
class segment_tree_beats;
```

The algebra is [`lazy_segment_tree`](lazy-segment-tree.md)'s (`op` with an
identity, `compose(f, g)` = "`g` first, then `f`" with a tag identity) with one
change: the mapping works in place and reports whether the summary alone could
absorb the tag.

```cpp
// atl/types/algebraic.hpp
template<typename Mapping, typename F, typename S>
concept fallible_action = std::regular_invocable<Mapping const &, F const &, S &>
    && std::same_as<std::invoke_result_t<Mapping const &, F const &, S &>, bool>;
```

`true`: the tag is applied to the summary. `false`: the summary is left
unspecified, the tree pushes the node, recurses into both children and
recomputes it. Node-*dependent* updates (floor sqrt: the amount a node drops
depends on its values) cannot be a fixed tag; they go through `apply_derived`,
whose rule `std::optional<Tag>(Value const &)` (concept `beats_tag_rule`)
derives the tag a node absorbs, or `std::nullopt` to descend.

```cpp
template<typename Rule, typename Value, typename Tag>
concept beats_tag_rule = std::regular_invocable<Rule const &, Value const &>
    && std::same_as<std::invoke_result_t<Rule const &, Value const &>, std::optional<Tag>>;

// engine members (every algebra)
constexpr segment_tree_beats();
constexpr segment_tree_beats(std::size_t n, Value identity, Tag tag_identity, Op op = Op{}, Mapping mapping = Mapping{},
                             Compose compose = Compose{});
template<std::ranges::sized_range R> requires requires { static_cast<Value>(std::declval<atl::detail::forwarded_element_t<R>>()); }
constexpr segment_tree_beats(R &&values, std::type_identity_t<Value> identity, Tag tag_identity, Op op = Op{},
                             Mapping mapping = Mapping{}, Compose compose = Compose{});
constexpr void set(std::size_t i, Value x);
[[nodiscard]] constexpr Value const &at(std::size_t i);
constexpr void apply(std::size_t i, Tag const &f);
constexpr void apply(std::size_t l, std::size_t r, Tag const &f);
template<beats_tag_rule<Value, Tag> Rule> constexpr void apply_derived(std::size_t l, std::size_t r, Rule const &rule);
[[nodiscard]] constexpr Value fold(std::size_t l, std::size_t r);
[[nodiscard]] constexpr Value const &fold() const;
template<std::predicate<Value const &> Pred> [[nodiscard]] constexpr std::size_t max_right(std::size_t l, Pred pred);
template<std::predicate<Value const &> Pred> [[nodiscard]] constexpr std::size_t min_left(std::size_t r, Pred pred);
constexpr std::size_t size() const noexcept;
constexpr bool empty() const noexcept;

// library types (tag_element / value_element: the tag's / summary's element_type)
constexpr explicit segment_tree_beats(std::size_t n);              // beats::summary<Value> && beats::tag<Tag>
template<std::ranges::sized_range R> constexpr explicit segment_tree_beats(R &&elements);
constexpr void set(std::size_t i, value_element x);                // beats::summary<Value>
[[nodiscard]] constexpr auto element(std::size_t i);               // the leaf's max
constexpr void chmin(std::size_t l, std::size_t r, tag_element x); // clamp_add
constexpr void chmax(std::size_t l, std::size_t r, tag_element x); // clamp_add
constexpr void add(std::size_t l, std::size_t r, tag_element x);   // clamp_add or add
constexpr void assign(std::size_t l, std::size_t r, tag_element x);// clamp_add or assign
constexpr void modulo(std::size_t l, std::size_t r, tag_element m);// mod_sum + assign
constexpr void floor_sqrt(std::size_t l, std::size_t r);           // sqrt_add_sum + add
constexpr void divide(std::size_t l, std::size_t r, tag_element d);// sqrt_add_sum + add
constexpr void bit_and(std::size_t l, std::size_t r, tag_element x);// bit_mask
constexpr void bit_or(std::size_t l, std::size_t r, tag_element x); // bit_mask
[[nodiscard]] constexpr auto sum(std::size_t l, std::size_t r);    // summaries with a sum: Sum
[[nodiscard]] constexpr auto min(std::size_t l, std::size_t r);    // summaries with a min: Element
[[nodiscard]] constexpr auto max(std::size_t l, std::size_t r);    // summaries with a max: Element
```

Constructors (O(n)): `segment_tree_beats()` (empty placeholder);
`(n, identity, tag_identity, op = {}, mapping = {}, compose = {})`;
`(values, identity, tag_identity, op, mapping, compose)` from summaries
(CTAD from lambdas, narrowing rejected, owning rvalue ranges moved from). With
the library types, whose identities come from the types: `explicit (n)` (n
zeros) and `explicit (elements)` from plain elements. `n < 2^32 - 1` is
asserted.

| member | meaning | time |
|---|---|---|
| `apply(l, r, f)` | apply tag `f` to `[l, r)` | O(log n) + descents, amortized by the algebra's potential |
| `apply_derived(l, r, rule)` | node-dependent update: each node takes the tag `rule` derives | same |
| `apply(i, f)`, `set(i, x)` | point update (`x` a summary, or an element for library types) | O(log n) |
| `at(i)` | the leaf's summary (`Value const &`) | O(log n) |
| `fold(l, r)`, `fold()` | fold of `[l, r)` / of everything | O(log n) / O(1) |
| `max_right(l, pred)`, `min_left(r, pred)` | the lazy tree's binary searches on monotone `pred` | O(log n) |
| `size()`, `empty()` | | O(1) |

Reads are not `const` (except `fold()`, `size`, `empty`): they settle tags on
the boundary paths.

**Library sets** (`atl::beats`). The summary type and the tag type switch on
named members through requires-clauses:

| summary | tag | members | amortized per update |
|---|---|---|---|
| `extremes<Element, Sum>` | `clamp_add<Element>` | `chmin`, `chmax`, `add`, `assign` (l, r, x); `sum`, `min`, `max` (l, r) | O(log² n); O(log n) without `add` |
| `mod_sum<Element, Sum>` | `assign<Element>` | `modulo(l, r, m)`, `assign(l, r, x)`; `sum`, `min`, `max` | O(log n · log V) |
| `sqrt_add_sum<Element, Sum>` | `add<Element>` | `floor_sqrt(l, r)`, `divide(l, r, d)` (floor), `add(l, r, x)`; `sum`, `min`, `max` | sqrt O(log n · log log V), divide O(log n · log V) |
| `bits_max<Element>` (unsigned) | `bit_mask<Element>` | `bit_and(l, r, x)`, `bit_or(l, r, x)`; `max` | O(log n · w), w the bit width |

Every library summary also enables `set(i, element)` and `element(i)`; `sum`
returns `Sum`, `min` / `max` need `l < r`. `extremes` holds `max`,
`second_max`, `min`, `second_min`, `sum`, `max_count`, `min_count`, `width`
(56 bytes for int64), so `fold(l, r)` also answers "how many elements equal
the maximum". `clamp_add` is `v -> min(max(v + add, lower), upper)`, closed
under composition: add, chmin, chmax and assign are one tag. The callables
`beats::merge`, `beats::mapping`, `beats::compose` are overload sets over all
four pairs, so a user algebra can start from a library one (the concepts
`beats::summary` and `beats::tag` say what makes a type a library-style one:
`Value::identity()`, `Value::of(element)`, `Tag::identity()`). The first row is
what `beats_segment_tree` is hand-written for: same answers, without the
generic members, and faster (Measured performance).

A user algebra for chmin, max and sum:

```cpp
struct node { long long max, second, count, sum; };      // second = -inf on a leaf
auto op = [](node a, node b) { /* merge the top two values and counts, add sums */ };
auto chmin = [](long long const &x, node &s) {
    if (x >= s.max) return true;                          // no-op
    if (x <= s.second) return false;                      // descend
    s.sum -= (s.max - x) * s.count, s.max = x;            // only the max class moves
    return true;
};
atl::segment_tree_beats t(leaves, node{LLONG_MIN, LLONG_MIN, 0, 0}, LLONG_MAX, op, chmin,
                          [](long long f, long long g) { return (std::min)(f, g); });
```

## Complexity

Build O(n); point updates, folds, queries and the binary searches O(log n).
A range update costs O(log n) plus its descents below failing nodes,
amortized per the algebra (the library-sets table): the descents are paid
for by a potential that starts at O(n · c), drops on every failing node an
update visits and rises by O(log n · c) per update; an algebra without one
has no bound better than O(n) per update. `beats_segment_tree` descends below
a node exactly where the engine's clamp-add mapping fails, so the same
potential bounds it. Memory: `2 · 2^⌈log2 n⌉` summaries plus `2^⌈log2 n⌉`
pending tags. At n = 2e5 (power-of-two padding to 2^18) with int64 and two
56-byte summaries per leaf slot, `beats_segment_tree` adds an 8-byte pending add,
about 31 MB, and the engine on clamp-add a 24-byte tag, about 36 MB.

## Preconditions

Asserted (`-DATL_ASSERT`): indices and ranges; `n < 2^32 - 1` and a doubled power-of-two layout that fits `size_t`; the engine's
laws where they can be observed (a single element never fails, a push never
fails, a rule never declines a single element); `clamp_add` bounds ordered;
`chmin` / `chmax` / `assign` arguments and every element value strictly inside
`Element`'s range (its two ends are the "no second value" sentinels);
`max_right` / `min_left`: the start in range and `pred(identity)`;
`modulo`: `m >= 1` and the range nonnegative; `floor_sqrt`: the range
nonnegative; `divide`: `d >= 1`; every sum, product and value shift checked for
overflow using portable range comparisons (free without asserts). Both trees of the
clamp-add algebra share all of these.

Documented only:
- Overflow bounds of the clamp-add algebra: let `M` bound every `|a[i]|` at
  all times plus the total `|x|` of the `add` calls (a node's pending add is
  the sum of the adds since its last push); `2M` must fit `Element` and
  `n · 2M` must fit `Sum`. `beats_segment_tree` needs less when a stream keeps
  `|a[i]| <= A` at all times, however large the adds sum up to: `3A` strictly
  inside `Element` and `4nA` inside `Sum`. For example,
  `|a[i]| <= 1e12` throughout with n = 2e5 gives `4 · 2e5 · 1e12` within
  int64, so no int128 is needed.
- User algebras: a leaf never fails; a tag that succeeded on a node succeeds
  on both children's current summaries (the push law); where it succeeds, the
  mapping agrees with applying the tag to every element. And **a successful
  mapping must be node-independent**: the same tag is composed into the
  node's pending tag and later applied to its children, so a mapping that
  "succeeds" by an effect depending on the node's values (turning a floor
  sqrt into an add) corrupts the children. Use `apply_derived` for those.

## Measured performance

`benchmarks/ds/segment_tree_beats.cpp` replays five complete operation streams
from fresh trees: balanced chmin/chmax/add/sum, chmin/max/sum without add,
add-recharge, whole-range collapse, and 90% short sum queries. The generic
engine stores clamp-add tags; the library uses extrema to imply clamps and
stores only an add. The other contenders change recursion, layout, mapping
protocol, or a fast path, as their names indicate.

Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2, RelWithDebInfo, assertions off.
2026-11-02, `-O2 -g -DNDEBUG -std=c++23`; 2026-11-03, the same flags plus
`-march=native`. Times are milliseconds, minima of three run means with
25 samples per row. Each sample includes construction and the entire stream.

| contender | balanced mix, n=q=2e5, plain / native |
|---|---:|
| generic engine | 341 / 356 |
| by compact recursion | 403 / 421 |
| by explicit stack | 345 / 348 |
| by fixed clamp-add | 317 / 335 |
| library | 190 / 194 |
| by aligned nodes | 202 / 203 |
| by separate arrays | 314 / 317 |
| by three adds | 319 / 322 |
| by optional mapping | 429 / 454 |
| by unconditional push | 359 / 387 |
| by no collapse | 423 / 432 |

| contender | chmin-only | add-recharge | collapse | short sums |
|---|---:|---:|---:|---:|
| generic engine | 239 | 125 | 99.8 | 180 |
| by compact recursion | 302 | 121 | 78.3 | 202 |
| by explicit stack | 237 | 113 | 81.3 | 185 |
| by fixed clamp-add | 134 | 121 | 71.2 | 174 |
| library | 122 | 91.7 | 53.2 | 101 |
| by aligned nodes | 128 | 91.6 | 54.9 | 111 |
| by separate arrays | 171 | 117 | 66.3 | 169 |
| by three adds | 161 | 131 | 92.4 | 160 |
| by optional mapping | 337 | 151 | 166 | 230 |
| by unconditional push | 260 | 176 | 133 | 220 |
| by no collapse | 240 | 115 | 101 | 184 |

The second table uses `-march=native`, n=q=2e5. The library takes 194 ms
on the balanced mix versus 356 ms for the generic engine, a 45.5% reduction
(`1 - 194/356`). The aligned layout is close on add-recharge (91.6 versus
91.7 ms); this does not establish a general win for padding nodes.

## Pitfalls and choosing

- **Class or engine.** `beats_segment_tree` unless a generic action is needed on
  the clamp-add algebra (`apply` with a general tag or `apply_derived`),
  or another algebra is. Past the bound `2M` above, the engine can abort under
  `ATL_ASSERT` and go wrong without it where the class is right: elements near
  `max()` flattened by a low chmin, then a large add, overflow in the engine's push
  and not in the class's.
- **Not beats.** Chmin / chmax with only min / max queries: `lazy_segment_tree`
  with a max (min) aggregate, because chmin distributes over max. Chmin with
  only point reads: `dual_segment_tree` with `commutative_tags` (chmin tags
  commute). A Fenwick tree cannot hold chmin at all. Beats pays off only when
  a sum or a count is asked.
- `min(l, r)` / `max(l, r)` need a nonempty range (an empty one has no value);
  `sum(l, l)` is 0.
- The sentinels: an `extremes` element may not be `numeric_limits<Element>::lowest()`
  or `max()`, and no add may push one there.
- `assign` on the clamp-add tag is a clamp to `[x, x]`: it never descends.
- `mod_sum` takes range `assign`, so a modulo on an assigned (single-valued)
  range is one assignment per node; without that fast path assignments
  could recharge the potential. `modulo` together with range add is not
  offered: it has no potential bound.
- Only current element extrema are stored; extrema across all updates are not tracked.

## How it works

**Engine.** The iterative layout of the shared segment-tree base: an update
pushes the partially covered ancestors of the range's two boundary leaves,
visits the O(log n) canonical nodes, and recomputes the boundary ancestors.
On a canonical node the tag goes through `mapping`; on success it is composed
into the node's pending tag (applied to the node, still owed to the
subtree), on failure the node is pushed, both children are visited
recursively the same way, and the node is recomputed. The recursion only runs
below failing nodes, at most log n deep. A push hands the pending tag to both
children through the same mapping and asserts that it succeeds (the push
law), and skips an identity tag (the lazy tree's measured skip). The named
queries `sum` / `min` / `max` read one field off each canonical node instead
of folding whole summaries.

**Clamp-add on extremes.** The add moves every field. The clamp then
succeeds in two cases. Either it is constant on `[min, max]` (the upper bound
at or below the minimum, the lower bound at or above the maximum, or
`lower == upper`): the node flattens into one value, and assignment is this
case. Or it moves only the minimum's class up (the lower bound below the
second minimum) and only the maximum's class down (the upper bound above the
second maximum), which changes the sum by `delta · count`. The flatten checks
come first, and that is a correctness matter, not a speed-up: a pushed
composite such as "chmin below everything, then chmax" is constant on the
parent, and its children may hold several values at either end, which the
one-class rule would reject; the oracle tests found this. Composition adds
the adds, shifts the older clamp bounds by the newer add (saturating at the
sentinels) and clamps them into the newer bounds; a constant composite drops
its add, so assignments do not accumulate one.

**`beats_segment_tree`.** The same layout and the same extremes per node, but
the only pending tag is an add: a clamp is implied by the node's own extremes. A
chmin that moves only the maximum's class just lowers the node's maximum; a later
push lowers each child's maximum to the parent's if it lies above, and raises each
child's minimum likewise. That is enough because a parent's extreme moves only
within its own class: a child holding values other than the parent's maximum keeps
them at or below the parent's second maximum, below any bound the parent absorbed,
so only the child's maximum class can lie above the parent's new maximum. Chmin and
chmax descend, and only descend, where the bound reaches the second extreme without
flattening the node, which is where the engine's mapping fails.

Two details keep it right at the ends of the value range, the engine also drops the add of a constant
composite and flattens a child. Under a *single-valued* parent the pending
add is void: the children take the parent's value directly, because their stale
values plus that add could overflow (a child at `max() - 1` under a parent since
assigned low, then added to), and the node discards its pending add. And lowering a single-valued child sets its
sum from `x · width`: its stale value may be further from `x` than `Sum` holds, after
several chmins the parent absorbed one at a time.

The add of a multi-valued node stays small: its minimum's class changes only by
adds and upward clamps and its maximum's by adds and downward ones, so with
`|a[i]| <= A` throughout the net add is at most `2A`. That gives the `3A` and `4nA`
of Preconditions: `3A` for a stale value plus its add, `4A` for what a lowered class
moves by.

**Why the bounds hold.** Without add, take as potential the number of
distinct values summed over all nodes: a failing chmin node has its maximum
and second maximum merged, losing at least one, and an update adds O(log n)
on the boundary paths, so the total is O((n + q) log n). With add, a boundary
node can regain a distinct value per add, and the accounting over the
log n levels gives O((n + q) log² n) under the beats potential argument. Modulo: a value at least `m`
falls below half, and an assignment makes a node single-valued, which the
modulo rule handles without descending. Floor sqrt and divide succeed when
`max - f(max) == min - f(min)`, as `v - f(v)` is nondecreasing, so the whole
node drops by the same amount: an add; each failure shrinks `max - min` to
about its square root (to a `d`-th). And / or succeed when every element of
the node agrees on every bit the tag changes (then it is one shift for all of
them), and each failure makes some bit agree.

**Overflow** is checked by portable comparisons before arithmetic
under `ATL_ASSERT`, which compiles to the plain operation otherwise. `floor
sqrt` uses the rounded double root, corrected by a step, at run time for
types up to 64 bits, and integer Newton from a power of two above the root
in constant evaluation and for 128-bit types. The dense classes are `constexpr`
(proofs in tests/ds/constexpr_checks.cpp).

Reference: the segment tree beats potential argument.

## See also

[`lazy_segment_tree`](lazy-segment-tree.md), [`dual_segment_tree`](dual-segment-tree.md),
[`segment_tree`](segment-tree.md), [`dynamic_lazy_segment_tree`](dynamic-lazy-segment-tree.md).
