# `atl::static_vector`

A fixed-capacity sequence with inline storage. Use it for a small bounded number of
copyable elements; large capacities make construction and copying expensive. The
container allocates no storage dynamically, but operations on `T` may allocate.

```cpp
#include <atl/ds/static_vector.hpp>
```

\include static-vector/basic.cpp

Expected stdout:

\verbinclude static-vector/basic.out

## API

```cpp
template<std::semiregular T, std::size_t Capacity>
class static_vector;
```

`T` is default-initializable and copyable. Every one of the `Capacity` slots is
value-initialized at construction, and appends assign into existing objects.
All members are `constexpr`; observers listed as `noexcept` do not call element operations.

| signature | behavior |
|---|---|
| `static_vector()` | Empty sequence; value-initialize all slots |
| `static_vector(std::initializer_list<T> init)` | Copy elements in order |
| `template<std::ranges::input_range R> explicit static_vector(R&& values)` | Convert forwarded elements to `T`; exclude this specialization itself |
| `void push_back(T const& value)`, `void push_back(T&& value)` | Append by assignment |
| `template<typename... Args> void emplace_back(Args&&... args)` | Require `std::constructible_from<T, Args...>`; construct a temporary, then assign |
| `void pop_back()`, `void clear()` | Remove the last or all live elements |
| `void resize(std::size_t count)`, `void resize(std::size_t count, T const& value)` | Grow with `T{}` or copies of `value`, or shrink |
| `T& front()`, `T const& front() const`, `T& back()`, `T const& back() const` | First or last live element |
| `T& operator[](std::size_t index)`, `T const& operator[](std::size_t index) const` | Live element at `index` |
| `T* data() noexcept`, `T const* data() const noexcept` | Inline storage pointer |
| `T* begin()`, `T* end()`, `T const* begin() const`, `T const* end() const` | Contiguous live-element range |
| `std::size_t size() const noexcept`, `bool empty() const noexcept` | Live count and emptiness |
| `static std::size_t capacity() noexcept` | `Capacity` |
| `friend bool operator==(static_vector const& a, static_vector const& b)` | Equality of live elements; require `std::equality_comparable<T>` |
| `friend auto operator<=>(static_vector const& a, static_vector const& b)` | Lexicographic order; require `std::three_way_comparable<T>` |

The range constructor requires `std::convertible_to<detail::forwarded_element_t<R>, T>`.
Owning rvalue ranges move elements; lvalue ranges, borrowed ranges and views copy
lvalue elements. Elements yielded by value are forwarded as rvalues.

Member aliases are `value_type = T`, `size_type = std::size_t`,
`difference_type = std::ptrdiff_t`, `reference = T&`, `const_reference = T const&`,
`pointer = T*`, `const_pointer = T const*`, `iterator = T*`, and `const_iterator = T const*`.

## Complexity

Construction, copying and destruction cost O(Capacity) element operations, including
range and initializer-list construction. Appends, access and removal of one element
cost O(1) element operations; `clear` costs O(size()), and `resize` costs
O(1 + |count - size()|). Comparisons inspect at most the shorter live prefix plus
one length comparison. Storage is O(Capacity), regardless of the live count.

## Preconditions

Asserted under `ATL_ASSERT`: appends require `size() < Capacity`; construction
requires at most `Capacity` inputs; `resize` requires `count <= Capacity`;
`front`, `back` and `pop_back` require a nonempty sequence; indexing requires
`index < size()`. Zero capacity is valid for an empty sequence.

Documented only: element operations must satisfy their concepts' semantic laws.
A removed slot is reset to `T{}` only for nontrivially destructible types; resource
release follows `T`'s assignment semantics. Throwing assignments have no rollback guarantee.

## Measured performance

Not measured yet.

## Pitfalls and notes

All slots stay constructed throughout the container's lifetime. Mutations preserve
storage addresses, but references to removed elements no longer denote live sequence
elements. Moving or copying the container transfers the whole array. Prefer small,
cheap element types; a heap-owning `T` can allocate even though the container does not.

## How it works

A value-initialized `std::array<T, Capacity>` holds storage and a `size_t` records the
live prefix. Appends assign at its end, removals shorten it, and comparisons ignore
the unused suffix. The fixed array explains both address stability and O(Capacity)
construction and copying.

## See also

- [Types and utilities](types-and-utilities.md)
