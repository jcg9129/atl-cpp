# `atl::xor_basis`, `atl::stamped_xor_basis`

Linear spans over GF(2) of fixed-width unsigned words: addition is XOR, and a
span contains every subset XOR, including zero. Use `xor_basis` for membership,
maximum, minimum, order and merge queries. Use `stamped_xor_basis` for queries
restricted to words with stamps at or above a threshold, such as subarray or
offline deletion queries. Neither type supports arbitrary online deletion.

```cpp
#include <atl/bits/xor_basis.hpp>
```

\include xor-basis/queries.cpp

Expected stdout:

\verbinclude xor-basis/queries.out

## `xor_basis` API and complexity

The exact template and range constraint are:

```cpp
template<typename R, typename Word>
concept word_range =
    std::ranges::input_range<R> && std::convertible_to<std::ranges::range_reference_t<R>, Word>;

template<unsigned_integral Word>
    requires std::same_as<Word, std::remove_cv_t<Word>>
class xor_basis;
```

`Word` is cv-unqualified and satisfies `atl::unsigned_integral`, including
native and software `atl::uint128`. Const and volatile word types are rejected
by the template constraints, including in the experimental basis variants.
`word_range<R, Word>` accepts a single-pass range whose references implicitly convert to `Word`; conversion can narrow, so choose the
word type to retain the bits you need. The deduction guide is:

```cpp
template<std::ranges::input_range R>
    requires unsigned_integral<std::ranges::range_value_t<R>>
xor_basis(R &&) -> xor_basis<std::ranges::range_value_t<R>>;
```

The signatures below are members of `xor_basis<Word>`. Let `w = bits`,
`r = rank()` and `n` be the input size. Storage is O(w) words, with no dynamic
allocation by the basis. All operations are constexpr. Bounds count each
fixed-width word operation as O(1), including the software 128-bit backend.

| signature | parameters and result | time |
|---|---|---|
| `static constexpr std::size_t bits` | `sizeof(Word) * CHAR_BIT`, maximum rank | constant |
| `constexpr xor_basis() noexcept` | Construct the zero space | O(w) |
| `template<word_range<Word> R> constexpr explicit xor_basis(R &&values)` | Consume values once and retain their span | O(w + n(1 + r) + r²) |
| `constexpr bool insert(Word x) noexcept` | Add `x`; return whether rank grew (false for zero or dependent words) | O(1 + r) |
| `constexpr void merge(xor_basis const &other) noexcept` | Replace this span by the sum with `other`; self-merge is allowed | O(1 + other.rank() · w) |
| `constexpr Word reduce(Word x) const noexcept` | Minimum `x ^ v` over this span; zero iff `x` belongs | O(1 + r) |
| `constexpr bool contains(Word x) const noexcept` | Whether `x` belongs, including zero | O(1 + r) |
| `constexpr Word min_xor(Word start) const noexcept` | Same minimum as `reduce(start)` | O(1 + r) |
| `constexpr Word max_xor(Word start = 0) const noexcept` | Maximum `start ^ v`; default is maximum subset XOR | O(1 + r) |
| `constexpr Word min_nonzero() const noexcept` | Least nonzero representable word | O(1) |
| `constexpr Word kth_smallest(Word k) const noexcept` | Word at zero-based index `k` in increasing unsigned order | O(1 + r) |
| `constexpr Word count_below(Word bound) const noexcept` | Number of representable words strictly below `bound` | O(1 + r) |
| `constexpr uint64 representable_count() const noexcept` | `2^r`, including zero | O(1) |
| `constexpr std::size_t rank() const noexcept` | Dimension, in [0, w] | O(1) |
| `constexpr bool empty() const noexcept` | Whether rank is zero | O(1) |
| `constexpr Word pivot_mask() const noexcept` | Bit `b` set iff slot `b` is occupied | O(1) |
| `constexpr Word vector(std::size_t b) const noexcept` | Basis vector with highest bit `b`, or zero | O(1) |
| `constexpr void clear() noexcept` | Reset to the zero space | O(w) |
| `constexpr bool operator==(xor_basis const &) const noexcept = default` | Equality of spans via their unique reduced bases | O(w) |

A failed insertion costs O(1 + popcount(x & pivot_mask())) word operations.
A successful insertion additionally visits higher pivots, O(r), and there can
be at most `w` successful insertions between resets. Construction includes
array initialization and at most O(r²) total work clearing newly added pivots.

### Preconditions

Asserted with `ATL_ASSERT`:

- `min_nonzero()` requires `rank() > 0`.
- `kth_smallest(k)` requires `k < 2^rank()`. At full rank every `Word` is valid;
  the check avoids a shift by the word width.
- `representable_count()` requires `rank() < 64`, even for a 128-bit word,
  because its return type is `uint64`.
- `vector(b)` requires `b < bits`.

With assertions disabled, violating these preconditions is outside the contract.
There are no additional algebraic preconditions for the unsigned word types.
`count_below(bound)` always fits `Word`: its count is at most `bound`.
The full span may have `2^bits` values, but no representable exclusive bound
can ask for that count.

### Measured performance

`benchmarks/bits/xor_basis.cpp` races three bases that all keep one vector per
highest set bit. The library's reduced `xor_basis` has no vector containing
another pivot bit, so the XORs a word needs are selected by `x & pivot_mask()`
and do not depend on one another. "Echelon, top-bit jump" stores plain echelon
form and jumps to the next set bit, so each XOR decides which vector the next
step reads. "Echelon, scan every bit" is the same form but tests every bit from
the top. The two echelon contenders are in
`experimental/atl/experimental/bits/xor_basis_variants.hpp`. Each `insert` row
builds a basis from 1e6 words; each `max_xor(start)` and `contains` row runs 1e6
independent probes against a built basis, so they measure throughput.

Xeon Platinum 8358, GCC 15.2, RelWithDebInfo without assertions, 2026-10-15.
Minimum of 3 runs, milliseconds per 1e6 operations, plain `-O2` / `-O2
-march=native`:

| shape | operation | library reduced | echelon, top-bit jump | echelon, scan every bit |
|---|---|---:|---:|---:|
| random uint64, full rank | insert | 33.6 / 23.8 | 140 / 129 | 287 / 279 |
| random uint64, full rank | max_xor(start) | 33.6 / 22.5 | 267 / 268 | 268 / 268 |
| random uint64, full rank | contains | 32.9 / 22.4 | 129 / 121 | 267 / 267 |
| uint32, rank 20 of 30 bits | insert | 14.3 / 12.2 | 48.9 / 42.1 | 108 / 104 |
| uint32, rank 20 of 30 bits | max_xor(start) | 14.9 / 12.4 | 129 / 129 | 129 / 129 |
| uint32, rank 20 of 30 bits | contains | 14.7 / 12.4 | 46.9 / 41.2 | 117 / 119 |
| uint64, rank 40 | insert | 23.4 / 17.6 | 90.7 / 82.2 | 207 / 201 |
| uint64, rank 40 | max_xor(start) | 24.1 / 17.0 | 268 / 268 | 268 / 268 |
| uint64, rank 40 | contains | 24.2 / 17.1 | 82.0 / 76.8 | 268 / 268 |

The library form is fastest on every row: top-bit jump is 3.2-15.8x slower and
scan every bit 7.6-15.8x slower. The reduced form wins because its pivot XORs
do not depend on previous elimination results.

Order queries from the same benchmark, at rank 50, milliseconds per 1e6
queries (plain `-O2` / `-O2 -march=native`). The experimental contender is a
greedy descent over the pivots, each comparison waiting for the previous
choice:

| operation | plain / native |
|---|---|
| library `kth_smallest` (branch-free select) | 57.7 / 40.2 |
| library `count_below` (pivot pattern) | 107 / 48.2 |
| `count_below` by greedy descent (`greedy_count_below`) | 208 / 130 |

The greedy descent is 1.9x slower with `-O2` and 2.7x slower with
`-march=native`. The plain flag set has no hardware popcount instruction.

### Pitfalls and notes

Queries count distinct values, not subsets. If `n` words were inserted, each
representable value has `2^(n - rank())` subset representations; compute that
power in a suitable larger type or modulus. The empty span contains zero and
has `representable_count() == 1`. Merge and insertion do not preserve the
original generating words or return a subset witness. Accessors return values,
so later mutations do not change earlier results.

## `stamped_xor_basis` API and complexity

```cpp
template<unsigned_integral Word, typename Stamp = uint32>
    requires std::same_as<Word, std::remove_cv_t<Word>>
        && std::totally_ordered<Stamp> && std::semiregular<Stamp>
class stamped_xor_basis;
```

For every threshold `from`, the selected pivot slots span exactly the inserted
words with stamps `>= from`, in any insertion order and with ties allowed.
`Stamp` need not be an integer. Bounds assume O(1) stamp operations; storage
is O(w) words and stamps plus a rank counter and possible padding. General
stamp types can themselves allocate. The basis is trivially copyable when
`Stamp` is; it is not promised to be so for every accepted stamp type.

| member signature | parameters and result | time |
|---|---|---|
| `static constexpr std::size_t bits` | Word width and maximum rank | constant |
| `constexpr stamped_xor_basis()` | Construct the zero space | O(w) |
| `constexpr bool insert(Word x, Stamp stamp)` | Add a stamped word; whether the full span grew | O(w) |
| `constexpr bool contains(Word x, Stamp const &from) const` | Whether `x` belongs to the restricted span; zero always does | O(w) |
| `constexpr Word max_xor(Stamp const &from, Word start = 0) const` | Maximum `start ^ v` over the restricted span | O(w) |
| `constexpr std::size_t rank(Stamp const &from) const` | Dimension of the restricted span | O(w) |
| `constexpr std::size_t rank() const noexcept` | Dimension of the full span | O(1) |
| `constexpr xor_basis<Word> basis_from(Stamp const &from) const` | Independent reduced basis for order, count or minimum queries | O(w²) |
| `constexpr Word vector(std::size_t b) const noexcept` | Echelon vector at pivot `b`, or zero | O(1) |
| `constexpr Stamp const &stamp(std::size_t b) const noexcept` | Reference to the stamp at slot `b` | O(1) |
| `constexpr void clear()` | Reset to the zero space | O(w) |

### Preconditions

`vector(b)` and `stamp(b)` assert `b < bits` under `ATL_ASSERT`. A returned
stamp is meaningful only for an occupied slot. A stamp reference remains
attached to its slot, whose value may change on insertion, assignment or
clear; destroying the object invalidates the reference. No exception guarantee
is provided if stamp construction, assignment, comparison or swap throws.
The total-order and semiregular laws must hold, as required by their concepts;
syntax alone cannot establish those semantic laws.

### Measured performance

`benchmarks/bits/xor_basis.cpp` times offline subarray maximum XOR with
`stamped_xor_basis`: `n = q = 5e5`, 20-bit values, queries sorted by right end.
Xeon Platinum 8358, GCC 15.2, RelWithDebInfo without assertions, 2026-10-15,
minimum of 3 runs: 50.0 ms with `-O2` and 48.9 ms with `-O2 -march=native`.

### Pitfalls and notes

After inserting `a[i]` with stamp `i` for `i < r`, query with `from = l` for
the half-open subarray `a[l, r)`. Process right ends in order or keep a basis
copy per prefix. A dependent insertion can still replace stamps and change
restricted spans even when `insert` returns false. For offline deletion with
a stamp equal to deletion time, include only words whose deletion time is
strictly later than the query time; choose a representable threshold accordingly,
without overflowing a stamp at its maximum. There is no stored element count.

## How it works

A reduced `xor_basis` stores one vector per highest set bit. No vector contains
another pivot bit. Reducing `x` therefore XORs precisely the vectors selected
by `x & pivot_mask()`, clearing all pivots without introducing another. The
least pivot vector is the least nonzero representable value. Selecting pivot
vectors in increasing order according to the bits of `k` produces sorted
values; the branch-free selection masks out unchosen vectors. Counting uses
the pivot pattern of the bound and its first differing non-pivot bit, summing
powers of two for smaller prefixes. Each query visits at most the rank many
pivots. Bit indices and popcounts use native limbs for software 128-bit words.

A stamped basis keeps plain echelon form, because reducing older vectors with
newer ones would mix stamp thresholds. At a collision it retains the newer
stamp and eliminates the older word with that vector. Each step removes the
highest bit, so insertion and membership inspect at most `bits` pivots.
`basis_from` inserts qualifying slots into a fresh reduced basis.

References: Chen Xinyang, *On the Linear (XOR) Basis in OI*, sections 2 and
3.4 (2025 national team papers), and Chang Ruinian, *Linear algebra*, section
8.2 (2022).
The literal update direction in the former's section 3.4 is reversed: the
retained pivot must be XORed into the candidate, not vice versa. A regression
test compares both updates against subset closure. The experimental echelon
forms are also checked against that independent closure oracle.

## See also

- [Bit utilities](bit-utilities.md) for low-bit isolation and subset enumeration.
- [128-bit integers](int128.md) for the portable word backend.
