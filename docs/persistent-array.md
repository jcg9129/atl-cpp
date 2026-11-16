# `atl::persistent_array`, `atl::semi_persistent_array`

Immutable fixed-size arrays return a new version for every point update. Use
`persistent_array` to branch from any version; use `semi_persistent_array` for a
single update timeline with readable old versions. Both share allocation across
versions and work well as the backing of a persistent DSU.

```cpp
#include <atl/ds/persistent_array.hpp>
```

\include persistent-array/versions.cpp

## API

```cpp
template<std::semiregular T, std::size_t Radix = 16>
    requires detail::valid_radix<Radix>
class persistent_array;

template<std::semiregular T, std::size_t Radix = 16>
    requires detail::valid_radix<Radix>
class semi_persistent_array;
```

`T` must be copyable and default-initializable. `Radix` is a power of two in
`[2, 65536]`. Below, replace the class name with `semi_persistent_array` for the
partial variant, except for iterators. These APIs use runtime shared ownership.

| Signature | Effect |
|---|---|
| `persistent_array()` | Empty array |
| `explicit persistent_array(std::size_t n, T value = T{})` | `n` copies of `value` |
| `std::size_t size() const noexcept` | Number of elements |
| `bool empty() const noexcept` | Whether size is zero |
| `T const &operator[](std::size_t index) const` | Read element; reference into the pool |
| `persistent_array set(std::size_t index, T value) const &` | New version, copying the pool handle |
| `persistent_array set(std::size_t index, T value) &&` | New version, moving the pool handle |
| `void reserve(std::size_t updates)` | Capacity for this many additional point updates |

Both `set` overloads are `[[nodiscard]]`. Both array types and their deduction
guides require sized input ranges, so sized output-only ranges are rejected.
The range constructor is:

```cpp
template<std::ranges::sized_range Range>
    requires std::ranges::input_range<Range>
          && std::convertible_to<detail::forwarded_element_t<Range>, T>
explicit persistent_array(Range &&values);
```

The range must also be readable as an input range. An owning rvalue donates its
elements; lvalues and borrowed ranges such as spans are copied. Deduction guides
`persistent_array(Range &&) -> persistent_array<std::ranges::range_value_t<Range>>`
and `semi_persistent_array(Range &&) -> semi_persistent_array<std::ranges::range_value_t<Range>>`
apply to sized ranges.

Only the full variant supplies `iterator begin() const` and `iterator end() const`.
Its `iterator` provides `iterator_category = std::forward_iterator_tag`,
`value_type = T`, `difference_type = std::ptrdiff_t`, `reference = T const &`,
`pointer = T const *`, `iterator()`, `reference operator*() const`,
`pointer operator->() const`, `iterator &operator++()`, `iterator operator++(int)`
and `bool operator==(iterator const &o) const`.

## Complexity

Let h = O(1 + log_Radix n). Construction takes O(n + Radix) time and storage;
for fixed Radix it is O(n + 1). Copying a version takes O(1). Reads take O(h).
A full update copies one leaf and h−1 branches, O(Radix · h) work and extra storage.
A partial update costs O(Radix · h) worst case and amortized O(Radix) extra storage;
for fixed Radix the latter is O(1).

`size`, `empty` and `end` take O(1); `begin` takes O(h). Iteration caches one leaf
but still descends from the root at each leaf boundary: a scan takes
O(n + ceil(n / Radix) · h), with O(h) worst-case increment at a boundary.
Reservation may relocate all existing nodes, taking time linear in their storage;
the number of requested updates determines the extra capacity, not its running time.

## Preconditions

Asserted with `ATL_ASSERT`: the source range size fits `size_t`; `index < size()`;
partial updates use the newest version;
a partial family has at most `INT32_MAX` successful updates; node ids fit `uint32`;
reservation counts fit `size_t` arithmetic and vector `max_size()`.

Documented-only: iterators compared with each other must belong to the same
version (or both be value-initialized); increment/dereference need a valid position.
Iterators do not own the pool, so a family handle must outlive them. References
and iterators may be invalidated by `set` **or `reserve` on any shared version**.
Copy a value out before updating. A moved-from version may only be destroyed or
assigned to. Concurrent updates require external synchronization.
Allocation and element-operation failures propagate; partial updates do not promise
transactional recovery after a throwing element operation.

## Measured performance

The fully persistent array path-copies trie nodes; semi-persistence uses bounded
modifications. Build models isolate up-front reservation from vector growth;
arena models isolate exact from geometric reservation. Loops measure throughput
of builds, dependent update chains, reads, and allocation policies.

Source: [`benchmarks/ds/persistent_array.cpp`](../benchmarks/ds/persistent_array.cpp).
CPU: Intel Xeon Platinum 8280; GCC 15.2; RelWithDebInfo, asserts off; 2026-10-15.
Flags: `-O2 -g -DNDEBUG -std=c++23` and
`-O2 -march=native -g -DNDEBUG -std=c++23`. Cells are minimum milliseconds
over three runs of 25 samples, plain / native.

| Shape, n=1048576 | library persistent_array | library semi_persistent_array |
|---|---|---|
| build | 1.8 / 1.5 | 1.9 / 1.7 |
| 200000 updates | 87.1 / 89.3 | 49.5 / 50.8 |
| 1000000 reads | 26.7 / 21.0 | 49.4 / 53.3 |

| 200000 updates, n=1048576 | ms |
|---|---|
| by per-update reserve(1) | 178 / 183 |
| by up-front reserve | 90.6 / 92.9 |
| by no reserve | 182 / 181 |

| Build size | library | by build reservation (reserved model) | by build reservation (geometric model) |
|---|---|---|---|
| 200000 | 0.324 / 0.278 | 0.250 / 0.176 | 0.402 / 0.334 |
| 1048576 | 1.8 / 1.6 | 1.4 / 1.0 | 4.1 / 3.7 |
| 4194304 | 25.0 / 24.4 | 24.6 / 21.9 | 57.1 / 55.6 |

| Arena operations, 7 nodes each | by exact growth | by geometric growth |
|---|---|---|
| 2000 | 48.9 / 49.5 | 0.163 / 0.158 |
| 200000 | not measured yet | 136 / 135 |

## Pitfalls and notes

A version is a shared pool handle plus a root id (and a timestamp in the partial
variant). All family nodes live until its final handle dies; deleting old versions
does not reclaim individual nodes. `a = std::move(a).set(i, value)` avoids copying
the shared handle. Full persistence tolerates forks; partial persistence tolerates
copies for reading but allows only the latest timeline to advance.

## How it works

A bit-partitioned radix trie stores values in leaves and child ids in a separate
branch arena. Construction packs a level bottom-up, reserving known node counts.
A full update copies the root-to-leaf path; unchanged siblings remain shared.
The partial variant gives each node one timestamped modification box. A first
change fills the box; a second materializes it into a copied node and redirects
its parent. Copies propagate upward only when a child id changes. Charging a
copy to the filled box that caused it bounds amortized new nodes geometrically.
The pool grows geometrically, keeping ids stable while vector reallocations can
move node objects. This distinction explains reference invalidation.

## See also

- [DSUs](dsu.md)
