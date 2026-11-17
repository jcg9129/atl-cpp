# `atl::splay_sequence`

A sequence ordered by position with amortized O(log n) insertion, splitting,
concatenation, range folds and optional lazy tags and reversal. Use it when
positions move; a fixed sequence can use a flat segment tree.

```cpp
#include <atl/ds/splay_sequence.hpp>
```

\include splay-sequence/quick-start.cpp

Expected stdout:

\verbinclude splay-sequence/quick-start.out

## API

```cpp
template<std::semiregular T, typename Op = void, typename F = void,
         typename Mapping = void, typename Compose = void,
         bool IsReversible = false, typename Layout = atl::splay_layout_interleaved>
class splay_sequence;
```

`reversible_splay_sequence<T, Op, F, Mapping, Compose, Layout>` enables reversal.
`value_type` is `T`; `size_type` is `std::size_t`. With `Op`,
`atl::binary_operation<Op, T>` checks the fold. Tags require non-void `Op`,
semiregular `F`, `atl::action<Mapping, F, T>` and
`atl::binary_operation<Compose, F>`; `F`, `Mapping` and `Compose` are either all
void or all present. Algebra callables accept const operands through a const
callable; reference wrappers can refer to non-const callables.

`Layout` selects `atl::splay_layout_interleaved` or `atl::splay_layout_split`; the split record holds two child ids, a parent id and size (16 bytes) separately from the payload.

| signature | behavior |
|---|---|
| `splay_sequence()` | empty plain tier |
| `splay_sequence(std::size_t n, T fill)` | n copies, plain tier |
| `explicit splay_sequence(R&& values)` | sized range, plain tier |
| `splay_sequence(std::size_t n, T s_identity, std::type_identity_t<OpArg> op)` | n identity elements, fold tier |
| `splay_sequence(R&& values, std::type_identity_t<T> s_identity, std::type_identity_t<OpArg> op)` | sized range, fold tier |
| `splay_sequence(std::size_t n, T s_identity, std::type_identity_t<FArg> f_identity, std::type_identity_t<OpArg> op, std::type_identity_t<MapArg> mapping, std::type_identity_t<CompArg> compose)` | n element identities, tagged tier |
| `splay_sequence(R&& values, std::type_identity_t<T> s_identity, std::type_identity_t<FArg> f_identity, std::type_identity_t<OpArg> op, std::type_identity_t<MapArg> mapping, std::type_identity_t<CompArg> compose)` | sized range, tagged tier |
| `void insert(std::size_t i, T x)` / `void set(std::size_t i, T x)` | insert at a gap / replace an element |
| `void erase(std::size_t i)` / `void erase(std::size_t l, std::size_t r)` | remove one element / `[l, r)` |
| `void push_back(T x)` / `void push_front(T x)` | append / prepend |
| `void pop_back()` / `void pop_front()` | remove an end |
| `void emplace(std::size_t i, Args&&... args)` / `emplace_back(Args&&... args)` / `emplace_front(Args&&... args)` | construct `T(args...)` then move it into the node |
| `T const& at(std::size_t i)` | untagged tier, reference to value |
| `T at(std::size_t i)` | tagged tier, current value |
| `T fold(std::size_t l, std::size_t r)` / `T fold() const` | ordered range / whole fold; identity for an empty range |
| `void apply(std::size_t l, std::size_t r, FArg const& f)` | apply one tag to `[l, r)` |
| `void reverse(std::size_t l, std::size_t r)` | reversible tier; reverse `[l, r)` |
| `std::size_t max_right(std::size_t l, Pred pred)` | largest r with `pred(fold(l, r))` |
| `splay_sequence split(std::size_t i)` | retain `[0, i)`, return the suffix on the same pool |
| `splay_sequence sibling() const` | empty handle on the same pool |
| `void concat(Derived&& other)` | append a distinct same-pool handle, emptying it |
| `void for_each(Visit visit) const` | visit current values in sequence order |
| `std::size_t size() const noexcept` / `bool empty() const noexcept` | element count / emptiness |
| `void clear()` | recycle this handle's nodes |

Constructor `OpArg`, `FArg`, `MapArg`, `CompArg` default to the corresponding
class parameters and must equal them. Constructor arguments use
`std::type_identity_t` so those parameters are non-deduced and accept
const-qualified callable types; `apply` uses `FArg = F` under the same
rule. `Derived` denotes the concrete splay class in inherited signatures.
Range constraints check the actual forwarded element's `static_cast<T>`;
only owning rvalues and prvalues move. Views, borrowed ranges and lvalues
copy. `emplace` requires `std::constructible_from<T, Args...>`. Searches require
`std::predicate<Pred&, T const&>`; visits require
`std::invocable<Visit&, T const&>`.

The six deduction guides cover range, count/fill, range/identity/fold,
count/identity/fold, and the two full tag forms. Range guides with an identity
reject narrowing from the range value type to that identity type. Deduction
enables the non-reversible tier; spell the reversible alias explicitly.

Copies deep-clone into the source pool; moves transfer the handle.
Copy/move assignment recycles the destination first; destruction recycles its
nodes. The containers are runtime-only because they use `std::shared_ptr`.

## Complexity

Single-element changes, indexed reads, range operations, split, concat and
prefix search take amortized O(log n); range erasure additionally takes O(r-l)
to recycle nodes. Construction, copies, full visitation, clear and destruction
take O(n). Whole-tree fold, size, emptiness and sibling take O(1). Moves take
O(1), with assignment additionally recycling the destination. Memory is
O(allocated pool slots), including recycled slots and related handles' nodes.
Costs assume constant-time payload operations. Reversal stores a flag and,
with folds, one additional aggregate per node. Tags add one `F` per node.

The logarithmic bound is amortized; one access can be linear. Traversal, copying, balanced construction and recycling use explicit stacks or parent pointers, avoiding recursive call-stack growth. Prepending is O(1) apart from amortized allocation; appending repeatedly is O(1) amortized per append.

## Preconditions

Asserted with `ATL_ASSERT`: `i < size()` for an element, `i <= size()` for an
insertion gap or split, `l <= r <= size()` for ranges, `pred(s_identity)` for
search, a distinct same-pool source for concat, and fewer than 2^32 pool slots
when allocating. Pool-dependent methods require an active handle. Empty end
removal is invalid. Counts and capacity arithmetic must fit `size_t`.

Documented only: `(T, op, s_identity)` is a monoid; tags form a monoid with
`f_identity`; `compose(f, g)` means g acts first, then f. Mapping respects
composition, `mapping(f_identity, x) == x`, and distribution over folds. Store widths in `T` if an action
needs segment length. A search predicate accepts a prefix of the growing
folds. Callback arithmetic must remain representable; callbacks must not
mutate related handles during traversal. Returned references are invalidated
by erasing their node or reallocating their pool, including via a sibling.
A moved-from handle supports destruction, assignment, clear and size/empty;
other methods may require its pool. Callable/allocation exceptions propagate from non-noexcept operations;
recycling failures in destruction or noexcept move assignment terminate.
Mutations provide no transactional rollback guarantee.

## Measured performance

`benchmarks/ds/splay_sequence.cpp` measures throughput including construction.
Contenders change descent (top-down, bottom-up, repush, semi-splay), node layout,
or use a treap. Handle splays separately change ancestor propagation.

Intel Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2, RelWithDebInfo, assertions off,
2026-10-24. Minimum of three run means, ms. Both flag sets also use
`-g -DNDEBUG -std=c++23`. Some million-element rows have substantial run spread;
small differences are not a reliable ranking.

Milliseconds, for every recorded workload and size. The columnar contender varies
the field layout; repush repeats ancestor propagation, while semi-splay changes
indexed reads. The final column divides the treap time by the library time;
values above one mean that the library took less time.

| flags | workload | n | by treap sequence | library | by top down split | by bottom up interleaved | by bottom up repush interleaved | by semi splay interleaved | by bottom up split | by columnar layout | by treap sequence / library |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `-O2` | uniform insert/erase/fold | 2e5 | 174 | 169 | 165 | 214 | 237 | 212 | 192 | 296 | 1.03x |
| `-O2` | uniform insert/erase/fold | 1e6 | 1512 | 1545 | 1586 | 1922 | 2065 | 1950 | 1885 | 3750 | 0.98x |
| `-O2` | random indexed reads | 2e5 | 58.0 | 83.8 | 80.3 | 102 | 131 | 88.0 | 119 | 132 | 0.69x |
| `-O2` | random indexed reads | 1e6 | 545 | 698 | 665 | 839 | 1005 | 753 | 776 | 1197 | 0.78x |
| `-O2` | sequential indexed reads | 2e5 | 25.8 | 8.7 | 9.0 | 9.2 | 12.9 | 9.1 | 9.8 | 12.4 | 2.97x |
| `-O2` | sequential indexed reads | 1e6 | 161 | 73.4 | 53.4 | 75.3 | 90.4 | 73.3 | 67.3 | 67.1 | 2.19x |
| `-O2` | cursor edits within +/-16 | 2e5 | 68.3 | 36.6 | 35.1 | 43.1 | 50.4 | 42.2 | 39.2 | 45.4 | 1.87x |
| `-O2` | cursor edits within +/-16 | 1e6 | 408 | 235 | 211 | 266 | 305 | 262 | 231 | 260 | 1.74x |
| `-O2` | hot range: 90% in 1% | 2e5 | 79.3 | 73.1 | 70.8 | 90.9 | 105 | 86.1 | 82.1 | 98.0 | 1.08x |
| `-O2` | hot range: 90% in 1% | 1e6 | 575 | 548 | 521 | 683 | 767 | 650 | 609 | 830 | 1.05x |
| `-O2` | reverse/fold, 50/50 | 2e5 | 335 | 287 | 289 | 329 | 384 | 333 | 348 | 607 | 1.17x |
| `-O2` | reverse/fold, 50/50 | 1e6 | 2992 | 2693 | 3039 | 3023 | 3347 | 3060 | 3382 | 8814 | 1.11x |
| `-O2` | insert/erase/reverse/apply/fold | 2e5 | 682 | 592 | 706 | 620 | 849 | 612 | 723 | 1010 | 1.15x |
| `-O2` | insert/erase/reverse/apply/fold | 1e6 | 5295 | 4963 | 6453 | 5079 | 6463 | 5063 | 6735 | 11207 | 1.07x |
| `-O2 -march=native` | uniform insert/erase/fold | 2e5 | 169 | 164 | 169 | 224 | 251 | 224 | 200 | 289 | 1.03x |
| `-O2 -march=native` | uniform insert/erase/fold | 1e6 | 1555 | 1543 | 1649 | 2007 | 2171 | 1982 | 1937 | 3590 | 1.01x |
| `-O2 -march=native` | random indexed reads | 2e5 | 58.1 | 84.3 | 82.9 | 102 | 130 | 86.3 | 98.2 | 135 | 0.69x |
| `-O2 -march=native` | random indexed reads | 1e6 | 570 | 707 | 688 | 849 | 1006 | 738 | 776 | 1223 | 0.81x |
| `-O2 -march=native` | sequential indexed reads | 2e5 | 20.6 | 8.8 | 9.1 | 9.0 | 13.0 | 9.1 | 9.8 | 12.3 | 2.34x |
| `-O2 -march=native` | sequential indexed reads | 1e6 | 138 | 74.0 | 54.0 | 75.7 | 94.5 | 74.7 | 67.3 | 67.3 | 1.86x |
| `-O2 -march=native` | cursor edits within +/-16 | 2e5 | 63.2 | 36.3 | 36.6 | 43.6 | 52.6 | 43.6 | 39.9 | 44.0 | 1.74x |
| `-O2 -march=native` | cursor edits within +/-16 | 1e6 | 448 | 232 | 217 | 270 | 315 | 269 | 234 | 253 | 1.93x |
| `-O2 -march=native` | hot range: 90% in 1% | 2e5 | 85.3 | 70.2 | 72.3 | 95.2 | 108 | 89.2 | 81.2 | 98.5 | 1.22x |
| `-O2 -march=native` | hot range: 90% in 1% | 1e6 | 629 | 534 | 533 | 705 | 790 | 671 | 616 | 819 | 1.18x |
| `-O2 -march=native` | reverse/fold, 50/50 | 2e5 | 342 | 297 | 290 | 333 | 400 | 336 | 335 | 600 | 1.15x |
| `-O2 -march=native` | reverse/fold, 50/50 | 1e6 | 3037 | 2789 | 3071 | 3030 | 3455 | 3100 | 3384 | 8465 | 1.09x |
| `-O2 -march=native` | insert/erase/reverse/apply/fold | 2e5 | 770 | 585 | 698 | 624 | 875 | 620 | 743 | 1045 | 1.32x |
| `-O2 -march=native` | insert/erase/reverse/apply/fold | 1e6 | 5828 | 4944 | 6449 | 5112 | 6647 | 5140 | 6851 | 11567 | 1.18x |

Handle splay + reverse prefix, same record and units. The three ratios divide
each contender's time by the library time, in column order.

| flags | n | library | by store buffer | by pointer reversal | by parity walk | contender / library |
|---|---:|---:|---:|---:|---:|---|
| `-O2` | 2e5 | 179 | 179 | 185 | 188 | 1.00x / 1.03x / 1.05x |
| `-O2` | 1e6 | 1942 | 1940 | 1969 | 1890 | 1.00x / 1.01x / 0.97x |
| `-O2 -march=native` | 2e5 | 173 | 174 | 179 | 178 | 1.01x / 1.03x / 1.03x |
| `-O2 -march=native` | 1e6 | 1932 | 1930 | 1943 | 1833 | 1.00x / 1.01x / 0.95x |

## Pitfalls and notes

`apply` accepts a tag value; mapping and composition can run many times while
pushing and pulling. There is no once-per-call mapping guarantee. Use an
equality-comparable tag to skip identity pushes when possible. Non-commutative
folds are supported through both forward and reverse aggregates. Split and
copies share allocation capacity without sharing live tree nodes.

Indexed reads, range folds and prefix search splay and are non-const. Whole folds and full visitation remain const; full visitation carries tags and reversal parity without changing nodes.

## How it works

Two splays isolate a non-empty interval between its boundary nodes. Fold
reads its aggregate, apply tags its root, reversal flips its two aggregates,
and erasure detaches it; the boundaries are then pulled. Node zero holds the
identity. Positional access uses subtree sizes during top-down splaying.
Bulk construction and copies use the engine's balanced build; full visitation
carries pending tags and reversal parity in an explicit stack.

## See also

 [Splay engine](splay-engine.md),
[Treap sequence](treap-sequence.md),
[Lazy segment tree](lazy-segment-tree.md).
