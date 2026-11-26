# `atl::segment_tree`

A segment tree over a **monoid**: hold `n` values, assign one element, and fold any
half-open range `[l, r)` in O(log n), plus a tree-descending binary search
(`max_right` / `min_left`) and direct access to the canonical nodes of a range. Reach for
it when values change between queries and the operation is associative; if the data is
static use [`direct_rmq`](rmq.md) or a sparse table, if the operation is invertible and
you only need prefix-style folds a Fenwick tree is lighter, and if you need range
*assignment* use `lazy_segment_tree`.

The fold operation is any callable `op(a, b)` with a neutral `identity` —
`(T, op, identity)` must be associative with that neutral element; `T` is copyable, and `op` **need not be commutative** (matrix product, string
concatenation, min/max, ordered monoid operations all work). There is no algebra *struct* to
define: you pass the operation and its identity as plain constructor arguments, and
`T`/`Op` are deduced.

```cpp
#include <atl/ds/segment_tree.hpp>

// A range-sum tree over 8 zeros.
atl::segment_tree seg(std::vector<long long>(8, 0), 0LL, std::plus<>{});
seg.set(3, 10);
seg.set(5, 4);
auto s = seg.fold(0, 6);   // 14
```

\include segment-tree/quick-start.cpp

Expected stdout:

\verbinclude segment-tree/quick-start.out

## API

```cpp
template<std::copyable T, binary_operation<T> Op>
class segment_tree;
```

- `T`: the element type; `std::copyable`.
- `Op`: any callable with `op(T const&, T const&)` convertible to `T`, callable on a
  `const` object ([`binary_operation<T>`](../include/atl/types/algebraic.hpp): functor,
  function pointer, lambda). Stored `[[no_unique_address]]`, so a stateless `Op` is free.
- `value_type` = `T`, `size_type` = `std::size_t`.

**Constructors.** Class-template-argument deduction fills in `T` (from `identity`) and
`Op` (from `op`), so the class arguments are never spelled.

| constructor | description |
|---|---|
| `segment_tree()` | requires `T` and `Op` default-initializable: an empty tree with value-initialized `identity` and `op`; a placeholder for a member or global assigned later |
| `segment_tree(std::size_t n, T identity, Op op = Op{})` | `n` elements, all equal to `identity` |
| `segment_tree(R&& values, std::type_identity_t<T> identity, Op op = Op{})` | built from a `std::ranges::sized_range R` whose forwarded element can be explicitly converted to `T` |

**Operations.** Ranges are half-open; "pos" means `i < size()`, "range" means
`l <= r <= size()`.

| function | description | time |
|---|---|---|
| `void set(std::size_t i, T x)` | assign `x` to position `i` | O(log n) |
| `void update(std::size_t i, Update&& f)` | replace element `i` by `f(element)`, invoked once on the stored value, with no copy out as `set(i, f(at(i)))` makes; `Update` models `update_function<T>` | O(log n) |
| `T const& at(std::size_t i) const` | the value at `i`, by reference (reflects subsequent updates at `i`) | O(1) |
| `T fold(std::size_t l, std::size_t r) const` | fold of `[l, r)`, or `identity` if empty | O(log n) |
| `T const& fold() const` | fold of the whole range `[0, size())`, by reference | O(1) |
| `void for_each_segment(std::size_t l, std::size_t r, F f) const` | call `f(T const&)` on each of the O(log n) canonical nodes tiling `[l, r)`, strictly left to right; `F` models `std::invocable<T const&>` | O(log n) calls |
| `std::size_t max_right(std::size_t l, Pred pred) const` | largest `r` in `[l, size()]` with `pred(fold(l, r))` true; `Pred` models `std::predicate<T const&>` | O(log n) |
| `std::size_t min_left(std::size_t r, Pred pred) const` | smallest `l` in `[0, r]` with `pred(fold(l, r))` true | O(log n) |
| `std::size_t size() const noexcept` / `bool empty() const noexcept` | element count / emptiness | O(1) |

`identity` must be the neutral element of `op` (`op(identity, x) == op(x, identity) ==
x`). To fill with a non-neutral value, use the range constructor —
`segment_tree(std::vector<int>(5, 7), 0, std::plus<>{})` is five 7s, not five copies of
the identity. `op` defaults to `Op{}` when the operation type is spelled
(`segment_tree<long long, std::plus<>>(n, 0)`); deduction still needs it passed.

**Range constructor and deduction.** Deduction refuses to **narrow** the values:
`segment_tree(vec_of_long_long, 0, …)` is a compile error (the `int` identity would
deduce `T = int` and silently truncate) — write `0LL`, or spell `segment_tree<T, Op>`
explicitly if the conversion is intended. Widening is fine: `int` values with a `0LL`
identity deduce `T = long long`. Elements are moved in only from an rvalue range that
owns them (`std::move(vec)`), worth it when elements are heavy, e.g. the merge-sort tree
below; an lvalue, a borrowed range (`std::span`, `string_view`) or a view
(`views::transform`, …) is copied from, so the caller's container is never gutted
through a view over it.

**`max_right` / `min_left`** are the tree-descending binary search. `pred` must be
monotone (once it flips it stays flipped) and `pred(identity)` must hold.
`max_right(l, pred)` returns the farthest `r` such that the fold of `[l, r)` still
satisfies `pred`; `min_left(r, pred)` is the mirror from the right.

```cpp
// Farthest r such that the sum of [l, r) stays <= budget.
std::size_t r = seg.max_right(l, [&](long long s) { return s <= budget; });
```

**`for_each_segment`: the decompose-then-aggregate primitive.** `fold` folds a range with
`op`. When you need something `op` can't express — a binary search *inside* each
canonical node, or a merge-sort tree that stores a sorted vector per node and counts
elements `< x` across a range — iterate the canonical nodes directly. A merge-sort tree
is exactly this: a `segment_tree<std::vector<T>, …>` whose "op" merges two sorted
vectors, queried with `for_each_segment`. The nodes are visited strictly left to right,
so it is also correct for non-commutative aggregation done by hand.

```cpp
// Merge-sort tree: how many elements in [l, r) are < x?
auto merge = [](std::vector<int> const& a, std::vector<int> const& b) {
    std::vector<int> c;
    std::ranges::merge(a, b, std::back_inserter(c));
    return c;
};
std::vector<std::vector<int>> leaves;
for (int v : data) leaves.push_back({v});
atl::segment_tree tree(leaves, std::vector<int>{}, merge);

int count = 0;
tree.for_each_segment(l, r, [&](std::vector<int> const& sorted) {
    count += std::ranges::lower_bound(sorted, x) - sorted.begin();
});
```

## Complexity

| operation | time | memory |
|---|---|---|
| construct (`n`, or from a range) | O(n) `op` calls | `2 * bit_ceil(n)` values of `T` (at most `4n`) |
| `set`, `update` | O(log n) `op` calls | none extra |
| `fold(l, r)` | at most about 2 log n `op` calls (one per boundary block) | none extra |
| `max_right`, `min_left` | O(log n) `op` and `pred` calls | none extra |
| `at`, `fold()`, `size`, `empty` | O(1) | none |

## Preconditions

Asserted (`atl_assert`, active only under `-DATL_ASSERT`; the benchmarks run without it):

- `i < size()` for `set`, `update`, `at`.
- `n <= 2^(numeric_limits<size_t>::digits - 2)` for the doubled layout.
- `l <= r <= size()` for `fold` and `for_each_segment`; `l <= size()` for `max_right`
  and `r <= size()` for `min_left`.
- `pred(identity)` for `max_right` and `min_left`. The assert *calls* `pred` once, so a
  predicate with side effects sees one extra call in an assert build.

Documented only:

- `(T, op, identity)` is a monoid (associative, `identity` neutral). Not checkable.
- `pred` is monotone in the searched direction.
- `op` is const-callable and, when `T` holds large values, cheap enough to be called
  O(log n) times per operation. No value range is imposed beyond what `op` itself needs
  (a sum tree overflows when its sum does).

## Measured performance

Against a textbook recursive segment tree on 10⁶ elements, the library
is 145/135 = 1.07 times faster on point sets plain (147/109 = 1.35
native), and 274/110 = 2.49 times faster on range queries
(282/126 = 2.24 native), for 10⁶ operations each. Native changes library
sets from 135 to 109 ms and queries from 110 to 126 ms:

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| 10⁶ ops, n = 10⁶ | iterative (library) | recursive |
|---|---|---|
| set | 135 / 109 ms | 145 / 147 ms |
| range query | 110 / 126 ms | 274 / 282 ms |

`max_right`, 10⁶ calls reaching about 5·10⁴ elements each: for non-scalar `T` the ascent
keeps the fold it tested instead of recomputing it (`reuse_ascent_fold`); scalars keep
the recomputing form. The comparison uses full-capacity build and two op calls per accepted search node (`by full build and recomputed search` rows):

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| `max_right`, 10⁶ calls | library, `-O2` / `-march=native` | by full build and recomputed search, `-O2` / `-march=native` |
|---|---|---|
| sum + 2x2 matrix (aggregate `T`) | 347 / 332 ms | 629 / 525 ms |
| gcd (scalar `T`) | 183 / 205 ms | 183 / 206 ms |

Construction combines only ancestors of the first `n` leaves, while
storage allocation still includes padding. `benchmarks/ds/segment_tree.cpp` also measures `ll` sum builds at `n=2^20`:
the library loses plain (0.885 versus 0.835 ms) and wins native
(0.684 versus 0.698 ms).
At `n=2^20+1` it wins both modes (1.4/1.3 versus 2.1/1.9 ms).

## Pitfalls and notes

- **Identity is not a fill value.** `segment_tree(n, x)` makes `n` copies of the
  identity; to start from non-neutral values pass them as a range.
- **Narrowing deduction errors** (`0` against `long long` values) are intentional; write
  `0LL`.
- **`at(i)` is a reference**: a later `set(i, …)` changes what it shows. Copy it if you
  need the value from that read.
- **Wrong op, silent wrong answer.** A non-associative `op`, or an `identity` that is
  not neutral, is not detected; `fold` returns something, just not the fold.
- **`max_right` needs a monotone `pred`.** On a non-monotone one it returns *a*
  boundary where the predicate flips, not the largest satisfying `r`.
- **Related components.** Range apply: [`lazy_segment_tree`](lazy-segment-tree.md) (same
  iterative core), [`dual_segment_tree`](dual-segment-tree.md) (range apply, point get).
  Invertible op: [`binary_indexed_tree`](fenwick-tree.md). Static data:
  [`direct_rmq`](rmq.md). Huge index range, nodes on demand:
  [`dynamic_segment_tree`](dynamic-segment-tree.md).

## How it works

**The idea.** A segment tree is a complete binary tree whose leaves are the elements and
whose every internal node stores the fold of its two children, i.e. of the contiguous
block of leaves beneath it. A point change only alters the folds on one leaf-to-root
path, and any range is a disjoint union of O(log n) blocks, so both operations touch
O(log n) nodes.

**Layout.** One flat `std::vector<T>` of `2 * size_` nodes, where `size_` is the
smallest power of two `>= n` (`std::bit_ceil`). Node `k` has children `2k` and `2k + 1`,
element `i` is the leaf at `size_ + i`, the root is node 1 (so `fold()` is `nodes_[1]`)
and node 0 is unused. There is no pointer and no recursion; the geometry is shared with
`lazy_segment_tree` and `dual_segment_tree` through `detail::segment_tree_base`, which
knows nothing about the algebra. The padding leaves `[n, size_)` hold `identity`, so
every node's value is already the fold of its *real* leaves and the padding is invisible
to queries.

**Invariant.** For every internal node `k`, `nodes_[k] == op(nodes_[2k], nodes_[2k+1])`,
and the padding leaves equal `identity`. `set`/`update` write the leaf and recompute its
strict ancestors bottom-up (`recompute_up`: `while (k >>= 1) nodes_[k] = op(left,
right)`), restoring the invariant on the one path that could have broken it: log n
`op` calls.

**Range decomposition.** `decompose(l, r)` shifts both bounds to leaf indices and walks
up while `l < r`: if `l` is a right child (odd) its node is a canonical block and `l`
steps right; if `r` is odd, `r - 1` is a canonical block; then both halve. Each step
emits at most one node on each side, there are at most log n steps, and the blocks it
emits tile `[l, r)` exactly. Left-side blocks come out in ascending position and
right-side blocks in descending position. `fold` therefore keeps two accumulators,
`sl = op(sl, node)` for the left side and `sr = op(node, sr)` for the right side, and
joins them as `op(sl, sr)`: the operands are always combined in position order, which is
why `op` need not be commutative. `for_each_segment` runs the left side directly and
buffers the right-side nodes (at most the bit width of `size_t`) in a stack array to replay them reversed, so
its callback sees strictly left-to-right order.

**`max_right`, the descent.** Starting at leaf `l + size_`, climb: while the node is a
left child, move to the parent, so the node is the largest block starting at the current
position that fits. Test `pred(op(sm, node))` where `sm` is the fold accepted so far. If
it holds, absorb the node and step to the next block to the right; once the node index
becomes a power of two the climb has covered everything up to the end and `n` is
returned. If it fails, the answer lies inside this block, so descend: go to the left
child, absorb it if the predicate still holds (and move to its right sibling), otherwise
stay and refine. At a leaf the position is the answer. Both phases are at most log n
nodes, since the climb only goes up and the descent only goes down. Because padding
leaves are `identity`, a failing block always contains a real element that fails, so the
result never exceeds `n`. `min_left` is the mirror image with the operands swapped
(`op(node, sm)`).

**Tricks in this implementation.**

- *Reuse of the tested fold.* The predicate is evaluated on a candidate `t = op(sm,
  node)`; for non-scalar `T` that same `t` is moved into the accumulator when accepted,
  so each visited node costs one `op` call instead of two (`reuse_ascent_fold =
  !std::is_scalar_v<T>`). For scalars the kept fold measured slower
  (gcd), so scalars keep the recomputing form (see the measured table above).
- *Build touches only real ancestors.* `build` combines only the ancestors of the first
  `n` leaves (a shrinking `[lo, hi]` index window per level); the padding subtrees keep
  the `reset` fill, which is already their fold.
- *Leaf-to-root updates with no `fold` call.* `update` runs `f` once on the stored value
  in place, avoiding the copy that `set(i, f(at(i)))` makes, which matters for heavy `T`.
- *Fold without an `identity` branch.* Empty sides stay `identity` and the final
  `op(sl, sr)` absorbs them, so there are no special cases for an empty or one-sided
  range.
- *`for_each_segment` is the same `decompose`* as `fold`, so any aggregation done by
  hand pays exactly the cost of a fold plus the callback.

**Reference.** “Efficient and easy segment trees”, for the bottom-up segment tree and two-accumulator ordered fold;
the implementation is independently written.

## See also

- [`lazy_segment_tree`](lazy-segment-tree.md) — adds range apply (lazy propagation) on the same iterative core
- [`dual_segment_tree`](dual-segment-tree.md) — the mirror image: range apply, point get, tags only
- [`binary_indexed_tree`](fenwick-tree.md) — lighter Fenwick tree when the operation is an abelian group (invertible)
- [`direct_rmq`](rmq.md) — O(1) queries when the data is static (no updates)
