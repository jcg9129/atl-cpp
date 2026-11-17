# `atl::difference_fenwick`

Range updates and point reads when values and updates form the same
commutative group, such as addition or xor. Use `dual_segment_tree` for
general tag actions, assignment or operations without inverses.

```cpp
#include <atl/ds/difference_fenwick.hpp>
```

\include difference-fenwick/basic.cpp

## API and complexity

```cpp
template<std::copyable Value, binary_operation<Value> Op,
         inverse_operation<Value> Inverse>
    requires std::copy_constructible<Op> && std::move_constructible<Inverse>
class difference_fenwick;
```

`binary_operation` and `inverse_operation` check const invocation and
conversion of results to `Value`. They cannot prove any group laws.
`value_type = Value`, `size_type = std::size_t`. All members are constexpr.

| signature | meaning | time |
|---|---|---|
| `constexpr difference_fenwick(std::size_t n, Value const &fill, Value identity, Op op = Op{}, Inverse inverse = Inverse{})` | n copies of fill, explicit group identity | O(n) |
| `constexpr difference_fenwick(R const &values, Value identity, Op op = Op{}, Inverse inverse = Inverse{})` | initial values from a const sized input range, references convertible to Value | O(n) |
| `constexpr void apply(std::size_t l, std::size_t r, Value const &f)` | combine f into each value in [l,r); an empty range is allowed | O(log n) |
| `constexpr void apply(std::size_t i, Value const &f)` | combine f into value i | O(log n) |
| `[[nodiscard]] constexpr Value at(std::size_t i) const` | return value i; does not modify the tree | O(log n) |
| `constexpr std::size_t size() const noexcept` | number of elements | O(1) |
| `constexpr bool empty() const noexcept` | whether size is zero | O(1) |

Both the range constructor and its private difference-array helper require
`std::ranges::sized_range<R const>`, `std::ranges::input_range<R const>`
and `std::convertible_to<std::ranges::range_reference_t<R const>, Value>`.
The persistent storage is O(n) values; range construction uses an O(n)
temporary difference array. Operation costs count constant-time group calls.

## Preconditions

Asserted with `atl_assert` when `ATL_ASSERT` is enabled:

- `l <= r <= size()` for range apply; `i < size()` for point apply and at.
- Built-in integer conversions from initial values or operation results must
  be representable in `Value`.

Documented only: op is associative and commutative, identity is its neutral
element, and inverse is a two-sided group inverse. All calls must be valid;
signed intermediates, differences and inverses must fit, including negation
of initial values. Integer conversion assertions cannot detect overflow
inside user operations. Floating arithmetic does not obey the group laws
exactly. No concept can verify those laws. Non-invertible min/max, assignment
and arbitrary action tags are outside this interface.

## Measured performance

Measured 2026-11-01 using `benchmarks/ds/dual_segment_tree.cpp`.
x86-64 Xeon Platinum 8280, g++ 15.2, asserts off, minimum over three runs;
ms, plain `-O2` / `-march=native`. n=1e6, 1e6 operations including construction,
int64 values, random ranges; add updates in [-1000,1000], xor in [0,1024).

| workload | commutative dual tree | difference_fenwick |
|---|---|---|
| add / point get, 50/50 | 93.0 / 98.0 | 40.1 / 39.9 |
| add / point get, 90/10 | 128 / 143 | 57.0 / 58.1 |
| xor / point get, 50/50 | 103 / 99.0 | 39.6 / 38.8 |
| xor / point get, 90/10 | 145 / 144 | 57.3 / 57.2 |

These workloads cost 55% to 62% less than the commutative dual tree. Other
sizes, non-trivial values and query mixes were not measured. This is an
explicit separate class; no existing dual tree call gains allocation or
value-copy costs through automatic dispatch.

## Pitfalls and notes

Supply the correct identity (zero for both add and xor), and use
`std::identity` as the xor inverse. There is no `set`, general mapping or
range fold. Repeated point reads each take O(log n).

## How it works

A difference array stores a[0] and `op(a[i], inverse(a[i-1]))` thereafter.
Its prefix fold telescopes to a[i]. Updating [l,r) combines f at l and
inverse(f) at r, omitting the latter when r equals size. Commutativity makes
these endpoint updates commute with the existing differences. The wrapper
reuses `binary_indexed_tree<Value, 1, Op>` for linear construction, point
updates and prefix folds; it adds no second Fenwick engine. A filled array
starts with identity differences and combines fill at the first cell.
Each traversal touches O(log n) Fenwick ancestors.

## See also

- [`binary_indexed_tree`](fenwick-tree.md) — point updates and prefix folds
- [`dual_segment_tree`](dual-segment-tree.md) — general range actions and point reads
