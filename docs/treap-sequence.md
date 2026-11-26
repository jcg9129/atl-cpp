# `atl::treap_sequence`

A sequence ordered by position with expected O(log n) insertion, splitting,
concatenation, range folds and optional lazy tags and reversal. Use it when
positions move; a fixed sequence can use a flat segment tree.

```cpp
#include <atl/ds/treap_sequence.hpp>
```

\include treap-sequence/quick-start.cpp

Expected stdout:

\verbinclude treap-sequence/quick-start.out

## API

```cpp
template<std::semiregular T, typename Op = void, typename F = void,
         typename Mapping = void, typename Compose = void,
         bool IsReversible = false>
class treap_sequence;
```

`reversible_treap_sequence<T, Op, F, Mapping, Compose>` enables reversal.
`value_type` is `T`; `size_type` is `std::size_t`. With `Op`,
`atl::binary_operation<Op, T>` checks the fold. Tags require non-void `Op`,
semiregular `F`, `atl::action<Mapping, F, T>` and
`atl::binary_operation<Compose, F>`; `F`, `Mapping` and `Compose` are either all
void or all present. Algebra callables accept const operands through a const
callable; reference wrappers can refer to non-const callables.

| signature | behavior |
|---|---|
| `treap_sequence()` | empty plain tier |
| `treap_sequence(std::size_t n, T fill)` | n copies, plain tier |
| `explicit treap_sequence(R&& values)` | sized range, plain tier |
| `treap_sequence(std::size_t n, T s_identity, std::type_identity_t<OpArg> op)` | n identity elements, fold tier |
| `treap_sequence(R&& values, std::type_identity_t<T> s_identity, std::type_identity_t<OpArg> op)` | sized range, fold tier |
| `treap_sequence(std::size_t n, T s_identity, std::type_identity_t<FArg> f_identity, std::type_identity_t<OpArg> op, std::type_identity_t<MapArg> mapping, std::type_identity_t<CompArg> compose)` | n element identities, tagged tier |
| `treap_sequence(R&& values, std::type_identity_t<T> s_identity, std::type_identity_t<FArg> f_identity, std::type_identity_t<OpArg> op, std::type_identity_t<MapArg> mapping, std::type_identity_t<CompArg> compose)` | sized range, tagged tier |
| `void insert(std::size_t i, T x)` / `void set(std::size_t i, T x)` | insert at a gap / replace an element |
| `void erase(std::size_t i)` / `void erase(std::size_t l, std::size_t r)` | remove one element / `[l, r)` |
| `void push_back(T x)` / `void push_front(T x)` | append / prepend |
| `void pop_back()` / `void pop_front()` | remove an end |
| `void emplace(std::size_t i, Args&&... args)` / `emplace_back(Args&&... args)` / `emplace_front(Args&&... args)` | construct `T(args...)` then move it into the node |
| `T const& at(std::size_t i) const` | untagged tier, reference to value |
| `T at(std::size_t i) const` | tagged tier, current value |
| `T fold(std::size_t l, std::size_t r) const` / `T fold() const` | ordered range / whole fold; identity for an empty range |
| `void apply(std::size_t l, std::size_t r, FArg const& f)` | apply one tag to `[l, r)` |
| `void reverse(std::size_t l, std::size_t r)` | reversible tier; reverse `[l, r)` |
| `std::size_t max_right(std::size_t l, Pred pred)` | largest r with `pred(fold(l, r))` |
| `treap_sequence split(std::size_t i)` | retain `[0, i)`, return the suffix on the same pool |
| `treap_sequence sibling() const` | empty handle on the same pool |
| `void concat(treap_sequence&& other)` | append a distinct same-pool handle, emptying it |
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
prefix search take expected O(log n); range erasure additionally takes O(r-l)
to recycle nodes. Construction, copies, full visitation, clear and destruction
take O(n). Whole-tree fold, size, emptiness and sibling take O(1). Moves take
O(1), with assignment additionally recycling the destination. Memory is
O(allocated pool slots), including recycled slots and related handles' nodes.
Costs assume constant-time payload operations. Reversal stores a flag and,
with folds, one additional aggregate per node. Tags add one `F` per node.

Logarithmic bounds assume independent random priorities and inputs independent of the seed. Worst-case depth is linear; structural recursion can consume linear stack space.

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

Measurements from `benchmarks/ds/treap_sequence.cpp`
(2026-11-22, AMD EPYC-Milan ccx23,
GCC 15.2.0, RelWithDebInfo, assertions off; plain `-O2` /
`-O2 -march=native`, five runs of 25 samples). It compares
shared-engine and direct-field treaps on inserts and folds, and treap versus
lazy segment tree on fixed-size affine updates and sums. Setup is timed. The shared-engine library wins insert/fold
(423/269 versus 494/346 ms), but loses fixed-size apply/fold to the
lazy segment tree (886/710 versus 283/282 ms).

## Pitfalls and notes

`apply` accepts a tag value; mapping and composition can run many times while
pushing and pulling. There is no once-per-call mapping guarantee. Use an
equality-comparable tag to skip identity pushes when possible. Non-commutative
folds are supported through both forward and reverse aggregates. Split and
copies share allocation capacity without sharing live tree nodes.

Indexed and fold reads are const: they carry pending tags and reversal parity instead of settling nodes. Prefix search can settle tags and restructure.

## How it works

Subtree sizes turn an in-order rank into a split boundary. Each mutation
composes priority-ordered merge and positional split, pulling sizes and
aggregates on return. Null id zero supplies the fold identity. Reversal swaps
forward/reverse aggregates immediately and postpones child swapping. Lazy tags
update a node's value and aggregates immediately and compose for its children.
Bulk construction uses a right-spine Cartesian build, pulling each node once.

The [persistent variant](persistent-treap-sequence.md) shares the same structural
engine through a path-copying store. Version copies cost O(1); edits return a
new version, and untouched subtrees remain shared. Equality-comparable tags
allow identity updates to skip otherwise necessary child copies.

## See also

 [Splay engine](splay-engine.md),
[Splay sequence](splay-sequence.md),
[Lazy segment tree](lazy-segment-tree.md).

- [Persistent treap sequence](persistent-treap-sequence.md) — branching versions and cross-version surgery.
