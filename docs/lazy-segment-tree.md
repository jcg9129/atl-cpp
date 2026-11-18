# `atl::lazy_segment_tree`

```cpp
#include <atl/ds/lazy_segment_tree.hpp>
```

A segment tree with **lazy propagation**: apply a tag to every element of a half-open
range and fold any range, both in O(log n). Like [`segment_tree`](segment-tree.md),
there is no algebra struct to define — the two monoids and the action connecting them
are plain constructor arguments; `Value` and `Tag` must be copyable:

- `(Value, op, identity)` — the **element monoid** being folded (min, sum, matrix, …);
- `(Tag, compose, tag_identity)` — the **tag monoid**, `compose(f, g)` meaning "`g` was
  applied first, then `f`" (the order matters for non-commutative tags like assign);
- `mapping(f, s)` — the **action** applying a tag to an element aggregate.

The algebra must satisfy `mapping(f, op(a, b)) == op(mapping(f, a), mapping(f, b))` —
a tag acts on a fold the same as on its parts. When a tag's effect depends on how many
elements an aggregate covers (add-on-**sum**, but not add-on-**min**), carry the width
inside `Value`:

```cpp
// Range add / range sum: Value = (sum, width), Tag = the added amount.
using node = std::pair<long long, long long>;
std::vector<node> init(n, {0, 1});                       // n zeros, width 1 each
atl::lazy_segment_tree st(
    init, node{0, 0}, 0LL,
    [](node const& x, node const& y) { return node{x.first + y.first, x.second + y.second}; },
    [](long long f, node const& s) { return node{s.first + f * s.second, s.second}; },
    [](long long f, long long g) { return f + g; });
st.apply(2, 7, 10);            // add 10 to every element of [2, 7)
auto s = st.fold(0, 5).first; // sum of [0, 5)
```

## Construction

The size and range constructors take `(…, identity, tag_identity, op, mapping, compose)`; deduction
fills in every template argument, and — as with `segment_tree` — the range form
refuses to **narrow** the values (`int` identity with `long long` elements is a
compile error; spell the class arguments explicitly if intended). Owned rvalue ranges move their elements in; views over lvalues copy.

| constructor | description | complexity |
|---|---|---|
| `lazy_segment_tree()` | an empty tree with value-initialized identities and callables (when all are default-initializable): a placeholder assigned later | O(1) |
| `lazy_segment_tree(std::size_t n, Value identity, Tag tag_identity, Op op = Op{}, Mapping mapping = Mapping{}, Compose compose = Compose{})` | `n` elements, all equal to `identity` | O(n) |
| `lazy_segment_tree(R&& values, std::type_identity_t<Value> identity, Tag tag_identity, Op op = Op{}, Mapping mapping = Mapping{}, Compose compose = Compose{})` | built from a sized range of initial values; Elements are moved in only from an rvalue range that owns them (`std::move(vec)`); an lvalue, a borrowed range (`std::span`, `string_view`) or a view (`views::transform`, …) is copied from, so the caller's container is never gutted through a view over it. | O(n) |

## Operations

| function | description | complexity |
|---|---|---|
| `apply(std::size_t l, std::size_t r, Tag const& f)` | apply tag `f` to every element of `[l, r)` | O(log n) |
| `apply(std::size_t i, Tag const& f)` | apply tag `f` to the single element at `i` | O(log n) |
| `fold(std::size_t l, std::size_t r)` → `Value` | fold of `[l, r)`, or `identity` if empty | O(log n) |
| `fold()` → `Value const&` | fold of the whole range (always current), by reference | O(1) |
| `set(std::size_t i, Value x)` | assign `x` to position `i` | O(log n) |
| `at(std::size_t i)` → `Value const&` | the value at `i` (settles the tags above it, hence non-const) | O(log n) |
| `max_right(std::size_t l, Pred pred)` → `std::size_t` | largest `r` in `[l, size()]` with `pred(fold(l, r))` true | O(log n) |
| `min_left(std::size_t r, Pred pred)` → `std::size_t` | smallest `l` in `[0, r]` with `pred(fold(l, r))` true | O(log n) |
| `size()` / `empty()` | element count / emptiness | O(1) |

Ranges are half-open with `l <= r <= size()`, indices `i < size()`; checked under
`-DATL_ASSERT`. `max_right` / `min_left` are the tree-descending binary search:
`pred` must be monotone and `pred(identity)` must hold. Most mutating-looking
*and* reading operations are non-const because reads settle pending tags; only
`fold()` (whole range) is const.

## Common algebras

| task | `Value` | `Tag` | `op` | `mapping(f, s)` | `compose(f, g)` |
|---|---|---|---|---|---|
| add / min | value | added amount | `min` | `s + f` | `f + g` |
| add / sum | `(sum, width)` | added amount | pairwise `+` | `{s.first + f·s.second, s.second}` | `f + g` |
| assign / sum | `(sum, width)` | value or `no_tag` | pairwise `+` | `f == no_tag ? s : {f·s.second, s.second}` | `f == no_tag ? g : f` |
| affine / sum mod p | `(sum, width)` of [`mod_int`](modular-arithmetic.md) | `(a, b)`: `s ↦ a·s + b` | pairwise `+` | `{a·sum + b·width, width}` | `(f.a·g.a, f.a·g.b + f.b)` |
| assign / composite | `(function, width)` | function or `no_tag` | composition | `f == no_tag ? s : f^width` | `f == no_tag ? g : f` |

The last row's `mapping` is not O(1): raising the assigned function to the segment
width costs O(log width) (lawful because `f^(wx+wy) == f^wx ∘ f^wy`), making
updates O(log² n).

## Stateful callables

The callables need only be invocable as `const` — they may carry `mutable` state.
That matters exactly for expensive mappings like `f^width` above: one `apply`
invokes `mapping` with the *same* tag on every canonical node, so a functor that
memoizes the last tag's `f^(2^j)` squaring chain rebuilds it once per apply
instead of once per node when applying a range assignment before composing
affine functions. This keeps the cache update independent of the number of
covered nodes.
Keep such state a pure cache: results must depend only on the arguments, or the
tree's folds become order-dependent.

## Preconditions

Asserted with `ATL_ASSERT`: point indices, half-open range endpoints,
`pred(identity)`, and a capacity whose doubled power-of-two layout fits
`size_t`. Documented only: the two monoid laws,
compatibility with composition, distributivity, and monotone search predicates.
Mapping need not preserve the element identity: valid updates never tag padding.
`Value` and `Tag` are copyable; all three operations are const-callable.
Arithmetic must fit the types chosen by the caller. Ranges copy lvalue elements
through views, move owned rvalue elements, and move prvalues. The constructor
checks the actual forwarded conversion, including explicit casts.

## Measured performance

`benchmarks/ds/lazy_segment_tree.cpp` compares iterative propagation with
recursion, then compares skipping identity pushes with unconditional pushes.
Each sample builds a fresh tree and replays a complete operation stream.

Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2, RelWithDebInfo, assertions off,
2026-10-15. Milliseconds, plain `-O2` / `-O2 -march=native`, both with
`-g -DNDEBUG -std=c++23`; minimum of three run means.

| 1,000,000 mixed operations at n=1e6 | library | by recursion |
|---|---:|---:|
| add/sum, 50/50 | 470 / 444 | 751 / 750 |

| operation stream | library | by unconditional push |
|---|---:|---:|
| add/sum, n=1e6, 50/50 | 497 / 488 | 497 / 485 |
| affine, n=1e6, 50/50 | 922 / 944 | 970 / 1003 |

Additional `-O2 -march=native` streams from the same record:

| stream | library | by unconditional push |
|---|---:|---:|
| add/sum, n=1e6, 10/90 | 427 | 437 |
| affine, n=1e6, 10/90 | 840 | 938 |
| add/sum, n=1e6, point operations + 10% range apply | 308 | 325 |
| add/sum, n=1e6, boundary search + 10% range apply | 313 | 326 |
| add/sum, n=1e3, 50/50 | 145 | 146 |

## Pitfalls and notes

`fold(l, r)`, `at` and searches settle pending tags and are non-const.
`fold()` reads the current root directly. Returned references reflect later
changes; copy a result that must survive mutation. Initialization by size uses
the element identity, so a width-carrying sum needs the range constructor with
width-one leaves. Complexity assumes constant-time callables; exponentiating
an assigned composite can add another logarithmic factor.

## How it works

The shared flat layout holds aggregates; a parallel array holds internal tags.
Before touching a range, only its partially covered boundary ancestors are
pushed root-first. Canonical nodes receive the tag, and boundary ancestors
are rebuilt leaf-first. Whole covered nodes retain their pending tags, so each
operation touches O(log n) nodes. An equality-comparable identity tag skips
unnecessary mappings and stores. Storage is O(n) values and tags, including
power-of-two padding.

## See also

- [`segment_tree`](segment-tree.md) — point update only: half the memory, no tag machinery
- [`dual_segment_tree`](dual-segment-tree.md) — point *queries* only: tags without aggregates, ~2× faster there
