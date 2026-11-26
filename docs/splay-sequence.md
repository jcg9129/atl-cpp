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

Some million-element rows have substantial run spread;
small differences are not a reliable ranking.

Milliseconds. The columnar contender varies
the field layout; repush repeats ancestor propagation, while semi-splay changes
indexed reads. The final column divides the treap time by the library time;
values above one mean that the library took less time.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| flags | workload | n | by treap sequence | library | by top down split | by bottom up interleaved | by bottom up repush interleaved | by semi splay interleaved | by bottom up split | by columnar layout | by treap sequence / library |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `-O2` | uniform insert/erase/fold | 2e5 | 132 | 131 | 132 | 171 | 195 | 170 | 174 | 163 | 1.01x |
| `-O2` | uniform insert/erase/fold | 1e6 | 1320 | 1230 | 1250 | 1530 | 1660 | 1520 | 1550 | 1530 | 1.07x |
| `-O2` | random indexed reads | 2e5 | 44.2 | 71.5 | 66.6 | 94.5 | 109 | 76.7 | 113 | 81.2 | 0.62x |
| `-O2` | random indexed reads | 1e6 | 418 | 578 | 517 | 745 | 830 | 615 | 816 | 621 | 0.72x |
| `-O2` | sequential indexed reads | 2e5 | 15.8 | 7.8 | 8.2 | 9.9 | 10.4 | 9.5 | 9.9 | 9.9 | 2.03x |
| `-O2` | sequential indexed reads | 1e6 | 112 | 70.9 | 45.4 | 78.1 | 81.4 | 74.2 | 55.4 | 53.1 | 1.58x |
| `-O2` | cursor edits within +/-16 | 2e5 | 47.1 | 27.4 | 27.7 | 33.2 | 37.7 | 32.8 | 32.9 | 31.1 | 1.72x |
| `-O2` | cursor edits within +/-16 | 1e6 | 273 | 212 | 182 | 240 | 262 | 239 | 208 | 194 | 1.29x |
| `-O2` | hot range: 90% in 1% | 2e5 | 56.6 | 57.2 | 57.2 | 72.9 | 86.5 | 71.4 | 71.7 | 64.7 | 0.99x |
| `-O2` | hot range: 90% in 1% | 1e6 | 474 | 473 | 434 | 566 | 663 | 559 | 542 | 494 | 1.00x |
| `-O2` | reverse/fold, 50/50 | 2e5 | 236 | 214 | 220 | 275 | 319 | 274 | 271 | 287 | 1.10x |
| `-O2` | reverse/fold, 50/50 | 1e6 | 3010 | 2770 | 2820 | 3350 | 3630 | 3290 | 3630 | 3980 | 1.09x |
| `-O2` | insert/erase/reverse/apply/fold | 2e5 | 605 | 481 | 582 | 499 | 689 | 501 | 596 | 700 | 1.26x |
| `-O2` | insert/erase/reverse/apply/fold | 1e6 | 5000 | 4100 | 5210 | 4250 | 5350 | 4300 | 5440 | 7070 | 1.22x |
| `-O2 -march=native` | uniform insert/erase/fold | 2e5 | 130 | 129 | 129 | 170 | 196 | 169 | 170 | 160 | 1.01x |
| `-O2 -march=native` | uniform insert/erase/fold | 1e6 | 1220 | 1190 | 1210 | 1500 | 1700 | 1510 | 1510 | 1510 | 1.02x |
| `-O2 -march=native` | random indexed reads | 2e5 | 43.9 | 70.1 | 66.9 | 93.5 | 108 | 77 | 94.3 | 81 | 0.63x |
| `-O2 -march=native` | random indexed reads | 1e6 | 421 | 560 | 518 | 720 | 804 | 607 | 693 | 611 | 0.75x |
| `-O2 -march=native` | sequential indexed reads | 2e5 | 16 | 7.8 | 8.7 | 9.8 | 12.4 | 9.3 | 10.5 | 9.7 | 2.05x |
| `-O2 -march=native` | sequential indexed reads | 1e6 | 112 | 69 | 46.7 | 77.4 | 90.5 | 72.8 | 58 | 52.3 | 1.62x |
| `-O2 -march=native` | cursor edits within +/-16 | 2e5 | 46.1 | 27.8 | 27.6 | 33 | 38.9 | 32.6 | 32.8 | 30.8 | 1.66x |
| `-O2 -march=native` | cursor edits within +/-16 | 1e6 | 296 | 205 | 179 | 234 | 262 | 234 | 205 | 190 | 1.44x |
| `-O2 -march=native` | hot range: 90% in 1% | 2e5 | 57.3 | 56.5 | 55.9 | 72.6 | 86.7 | 72.7 | 71.8 | 62.4 | 1.01x |
| `-O2 -march=native` | hot range: 90% in 1% | 1e6 | 449 | 447 | 424 | 565 | 654 | 557 | 538 | 474 | 1.00x |
| `-O2 -march=native` | reverse/fold, 50/50 | 2e5 | 228 | 212 | 215 | 276 | 325 | 276 | 270 | 278 | 1.08x |
| `-O2 -march=native` | reverse/fold, 50/50 | 1e6 | 2450 | 2250 | 2240 | 2770 | 3050 | 2770 | 2830 | 3240 | 1.09x |
| `-O2 -march=native` | insert/erase/reverse/apply/fold | 2e5 | 545 | 399 | 439 | 427 | 607 | 430 | 478 | 558 | 1.37x |
| `-O2 -march=native` | insert/erase/reverse/apply/fold | 1e6 | 4250 | 3420 | 4280 | 3600 | 4640 | 3610 | 4530 | 5840 | 1.24x |

Handle splay + reverse prefix, milliseconds. The three ratios divide
each contender's time by the library time, in column order.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| flags | n | library | by store buffer | by pointer reversal | by parity walk | contender / library |
|---|---:|---:|---:|---:|---:|---|
| `-O2` | 2e5 | 170 | 163 | 161 | 161 | 0.96x / 0.95x / 0.95x |
| `-O2` | 1e6 | 1870 | 1820 | 1900 | 2060 | 0.98x / 1.01x / 1.10x |
| `-O2 -march=native` | 2e5 | 137 | 136 | 137 | 143 | 0.99x / 1.00x / 1.04x |
| `-O2 -march=native` | 1e6 | 1540 | 1550 | 1570 | 1620 | 1.01x / 1.02x / 1.05x |

The library loses random indexed reads to the treap at both sizes
(71.5/70.1 versus 44.2/43.9 ms at n=2e5; 578/560 versus 418/421 ms at
n=1e6). Million-element random-read rows vary between runs; those minima
alone do not establish a stable speedup. Plain handle rows also vary
(spreads up to 1.40x).

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
