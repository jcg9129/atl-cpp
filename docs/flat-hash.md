# `atl::flat_hash_set`, `atl::flat_hash_map`

Unique-key containers using flat storage and linear probing. Use them for membership
or key-value lookup when contiguous slots suit the workload; use another container
when you need stable references across growth or ordered traversal. These APIs return
insertion flags and have different lifetime rules from standard unordered containers.

```cpp
#include <atl/ds/flat_hash_set.hpp>
#include <atl/ds/flat_hash_map.hpp>
```

\include flat-hash/basic.cpp

Expected stdout:

\verbinclude flat-hash/basic.out

`<atl/utils/hash.hpp>` is the normal entry point for the shared seed and all
default hashes. The reduced alternative `<atl/utils/hash_core.hpp>` implements
integer and enum hashes and `identity_hash`, and declares complete interfaces
for every supported extension key. `<atl/utils/hash_extensions.hpp>` supplies
the floating-point, string, pointer and composite implementations.

## API

```cpp
template<typename Key, typename Hash = hash<Key>, typename Eq = std::equal_to<>>
    requires std::default_initializable<Key> && std::copyable<Key>
          && detail::nothrow_hasher_for<Hash, Key> && detail::key_eq_for<Eq, Key>
          && std::default_initializable<Hash> && std::default_initializable<Eq>
class flat_hash_set;

template<typename Key, typename T, typename Hash = hash<Key>, typename Eq = std::equal_to<>>
    requires std::default_initializable<Key> && std::copyable<Key>
          && std::default_initializable<T> && std::movable<T>
          && detail::nothrow_hasher_for<Hash, Key> && detail::key_eq_for<Eq, Key>
          && std::default_initializable<Hash> && std::default_initializable<Eq>
class flat_hash_map;
```

The detail concepts require invocation of const `Hash` and `Eq` on const keys,
with results implicitly convertible to `size_t` and `bool`, respectively.
`detail::nothrow_hasher_for<Hash, Key>` additionally checks that the hash call
and the exact home-slot conversion/mixing expression are `noexcept`. Throwing
hashers and throwing hash-result conversions are rejected by constraints;
`home()` remains `noexcept`. The default `atl::hash` specializations and
`atl::identity_hash` satisfy this requirement. All
members are declared `constexpr`; constant evaluation requires suitable custom
hash/equality functors because the default hash reads runtime process state.
Each operation below has the same name in both containers unless specified.

| signature | behavior |
|---|---|
| `flat_hash_set()`, `flat_hash_map()` | Empty, no slots allocated |
| `explicit flat_hash_set(std::size_t expected_elements)`, `explicit flat_hash_map(std::size_t expected_elements)` | Reserve for the total element count |
| `template<std::ranges::input_range R> explicit flat_hash_set(R&& keys)` | Build by range insertion |
| `template<std::ranges::input_range R> explicit flat_hash_map(R&& entries)` | Build by pair-like range insertion; first duplicate wins |
| set: `bool insert(Key const& key)`, `bool insert(Key&& key)` | Add if absent; return whether inserted |
| map: `template<typename V = T> bool insert(Key const& key, V&& value)`, `bool insert(Key&& key, V&& value)` | Require `std::convertible_to<V&&, T>`; keep existing value on a hit |
| `template<std::ranges::input_range R> void insert(R&& keys)` / map `insert(R&& entries)` | Reserve for sized input; insert in input order |
| `template<typename... Args> bool emplace(Args&&... args)` | Require construction of `Key` (set) or `value_type` (map); build a temporary even on a duplicate |
| map: `template<typename... Args> bool try_emplace(Key const& key, Args&&... args)`, `bool try_emplace(Key&& key, Args&&... args)` | Require `std::constructible_from<T, Args...>`; build the mapped temporary only on a miss |
| map: `bool insert_or_assign(Key const& key, T value)`, `bool insert_or_assign(Key&& key, T value)` | Insert or overwrite; return whether inserted |
| map: `T& operator[](Key const& key)`, `T& operator[](Key&& key)` | Insert `T{}` on a miss, return mapped reference |
| map: `T& at(Key const& key)`, `T const& at(Key const& key) const` | Reference to an existing value; presence asserted |
| map: `template<typename U = T> T at_or(Key const& key, U&& fallback) const` | Copy a hit or forward fallback on a miss; require `std::convertible_to<U&&, T> && std::convertible_to<T const&, T>` |
| `bool contains(Key const& key) const`, `std::size_t count(Key const& key) const` | Membership, or 0/1 count |
| set: `const_iterator find(Key const& key) const` | Read-only iterator, or `end()` |
| map: `iterator find(Key const& key)`, `const_iterator find(Key const& key) const` | Entry iterator, or `end()` |
| `bool erase(Key const& key)` | Tombstone the entry; return whether present |
| `void clear()` | Remove all live keys, retain slot objects and capacity |
| `void reserve(std::size_t n)` | Reserve for n total elements, never shrink |
| `std::size_t size() const noexcept`, `bool empty() const noexcept`, `std::size_t capacity() const noexcept` | Live count, emptiness, allocated slot count |
| set: `const_iterator begin() const`, `const_iterator end() const` | Live key iteration |
| map: `iterator begin()`, `iterator end()`, `const_iterator begin() const`, `const_iterator end() const` | Live entry iteration |

Both expose `key_type`, `value_type`, `size_type`, `iterator`, `const_iterator`;
the map also exposes `mapped_type`. Set iterators are read-only. The map's
`value_type` is `std::pair<Key, T>`, with mutable mapped values.

The set's range overloads exclude `R` equal to `Key` and require forwarded element
conversion to `Key`. Map ranges require `std::get<0>` and `std::get<1>` on forwarded
elements convertible to `Key` and `T`. Owning rvalue ranges move elements; lvalues,
borrowed ranges and views copy lvalue elements. Value-yielding iterators forward
temporaries as rvalues. A duplicate does not consume the mapped argument.

### Heterogeneous lookup

`template<typename K>` overloads of `contains`, `count`, `find`, `erase`, and map
`at` have the same return types and const qualifications as their keyed forms.
Map `at_or` additionally has `template<typename K, typename U = T>` with the same
fallback constraint. They require `Hash::is_transparent`, `Eq::is_transparent`,
`detail::nothrow_hasher_for<Hash, K>` (including home-slot conversion/mixing),
and `eq(stored_key, k)` convertible to `bool`.
Arithmetic pairs, including software 128-bit integers, use conversion to `Key`
instead, to avoid surprising mixed signed/unsigned comparisons. Heterogeneous
`erase` also excludes arguments convertible to the container's iterator types.

The default string hasher and equality support `string_view` and string literals
without building a string temporary. `atl::hash<char const*>` hashes an address;
use string keys for content equality.

### Hashing and seed lifetime

`<atl/utils/hash.hpp>` provides seeded `atl::hash<Key>` and `atl::identity_hash`.
The latter is intended for already well-distributed integer keys. Seeded mixing
reduces predictable clustering, but is not a cryptographic or collision-resistance
guarantee. In particular, the byte-buffer mix has an unproven design intent, not a
promise that adversarial collisions are impossible.

The process seed is drawn once during static initialization. Build seeded tables
from `main` or later. Calling `atl::set_hash_seed(uint64)` is allowed from `main`
onward, before building tables that use the seed, with no concurrent hashing or
seed changes. Earlier computed hashes and table placement become invalid after a
seed change. A table built during static initialization can observe seed zero
before the initializer runs. Hash values are not stable serialization.

The static-initialization seed wins over a guarded function-local
seed; `benchmarks/ds/hash.cpp` keeps both implementations for comparison. See the published
[hash contract](types-and-utilities.md#hash-customization-and-seed-lifetime) for
customization requirements. Sequence keys cost O(length) to hash.

## Complexity

For constant-cost, well-distributed hashes and constant-cost equality, lookup and
erasure are expected O(1), insertion expected amortized O(1). A worst-case probe
scans O(capacity) slots; rehash can also have quadratic collision-heavy reinsertion
cost. `find` constructs an iterator at a full slot, so no extra scan is required.
A full iteration, `begin()` in the worst case, and `clear()` cost O(capacity).
`end`, `size`, `empty`, and `capacity` cost O(1). Growing `reserve` costs expected
O(old capacity + new capacity); otherwise it costs O(1). Sized range insertion
also pays for reservation, which can scan the existing table; inserting n values
into an empty table is expected amortized O(n). Storage is O(capacity), including
default-constructed unused slots. Element construction and hashing costs are additional.

## Preconditions

Asserted under `ATL_ASSERT`: map `at` requires a present key. Reservation requires
`n <= size_t maximum / 4`; the next power of two of `n * 4 / 3 + 1` must be
representable. Growth's doubled live count must fit, with a representable next
power of two. Bulk reservation's existing plus input count must fit `size_t`.

Documented only: equality is an equivalence relation; equal keys, including mixed
probes, have equal hashes. Hash and equality stay stable while keys are stored.
Input references must survive any rehash they trigger.
The constraints on both `at_or` overloads require implicit conversion from
`T const&` and `U&&` to `T`, even if the actual lookup misses.
Backing allocations must be addressable; throwing mutations have no rollback guarantee.

## Measured performance

Flat hash containers race standard unordered containers and seeded, standard,
and identity hash choices. Transparent string-view lookup races a string
temporary per query. The hash benchmark isolates function-local seed guards
versus the library's process seed, and three seeded pair rounds versus the
library's single packed round. Loops measure batch throughput; string queries
address 100000 stored strings of 32–64 characters.

Source: [`benchmarks/ds/flat_hash.cpp`](../benchmarks/ds/flat_hash.cpp).


AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | library transparent lookup | by string temporary |
|---|---|---|
| 1e6 string-view lookups | 82.1 / 157 | 114 / 235 |

The native transparent-lookup row varies between runs (1.56x spread); its minimum
does not establish a stable speedup by itself.


Source: [`benchmarks/ds/hash.cpp`](../benchmarks/ds/hash.cpp).


AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | library | by function local seed |
|---|---|---|
| uint64 insert 1e6 | 19.9 / 12 | 20.4 / 12.6 |
| uint64 lookup 1e6, alternating hits/misses | 19.4 / 15.4 | 29.4 / 18.1 |
| uint64 hash only 1e6 | 0.829 / 0.895 | 0.978 / 1.1 |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Shape | library packed pair | by three seeded rounds |
|---|---|---|
| uint32 pair insert 1e6 | 19.9 / 12.8 | 26.2 / 17.4 |
| uint32 pair lookup 1e6, alternating hits/misses | 23.1 / 16.9 | 35 / 22.8 |

## Pitfalls and notes

Rehash invalidates all references and iterators. Erase invalidates the erased entry;
clear invalidates all live entries. Other insertions preserve references if no
rehash occurs. Do not modify a map iterator's `.first`: it would invalidate placement.
Iteration order is unspecified, and iterator increment can scan many unused slots.

Erase and clear change control bytes without destroying or resetting slot objects.
Resources can remain held until overwritten, rehashed, or destroyed. Reserve counts
elements, while `capacity()` counts slots. A hit never rehashes. Tombstone cleanup
can rehash after erasures even within an earlier reservation.

## How it works

Parallel slot and control arrays use a power-of-two capacity. Probes advance one
slot at a time, wrapping by a mask. Never-used slots stop misses; tombstones keep
probe chains reachable. Insertion remembers a tombstone but continues probing for
a duplicate before writing. Occupied-plus-deleted load triggers cleanup or growth
at 0.75; rebuilding keeps at least the existing reservation and twice the live count.
The threshold comparison avoids multiplying a potentially large capacity.

Default `atl::hash` and `identity_hash` use their low bits directly. Other hashers
pass through a Fibonacci multiply and high-bit extraction to disperse structured
integer outputs. This does not repair a constant hasher. Expected short probes and
geometric growth explain the expected amortized bound.

## See also

- [Types and utilities](types-and-utilities.md)

`<atl/ds/flat_hash_map.hpp>` remains the normal container entry point with full
key support. Its reduced alternative `<atl/ds/flat_hash_map_core.hpp>` can use
integer and enum default hashes, or a caller-supplied hash, without extension
implementations. Include `<atl/utils/hash_extensions.hpp>` before the first use
requiring an extension hash call body in each translation unit (including
explicit instantiation of that body or taking its address). Merely instantiating
the hasher/container class or querying its traits and constraints may precede
the extension: those answers are identical with either include path. A call
using the core alone fails at compile time with an include diagnostic.

As with any customization, declare a user specialization of `atl::hash<K>`
before any use that would instantiate it or query its availability in every
translation unit. The include-order guarantee assumes those user declarations
are consistent across translation units.

A query that needs a user-defined body (for example, to deduce a custom
hasher's return type) follows the body-use rule too if that body hashes an
extension key. The library-provided hash signatures do not require their bodies.
