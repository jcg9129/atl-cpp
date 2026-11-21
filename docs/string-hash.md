# `atl::string_hash`

An online polynomial rolling hash modulo the prime 2^61 - 1. Use it for
constant-time substring fingerprints and logarithmic common-prefix estimates
when collisions are acceptable; use comparison-based matching for exact answers.

```cpp
#include <atl/strings/string_hash.hpp>
```

\include string-hash/quick.cpp

## API

`hashable_text<R, Proj>` accepts input ranges with projected integral symbols
recognized by atl's integer traits, including char types and int128, excluding
bool. Non-contiguous and single-pass ranges work. Use a projection to turn
boolean proxies into integer digits. Capturing lambdas and reference wrappers
are accepted. The range concept is:

```cpp
template<typename R, typename Proj = std::identity>
concept hashable_text = std::ranges::input_range<R> && detail::byte_hashable<text_symbol_t<R, Proj>>;
```

Member signatures and the free whole-range hash:

```cpp
using hash_type = uint64;
static constexpr uint64 mod = detail::string_hash_mod;
constexpr explicit string_hash(uint64 base = default_base());
template<typename R, typename Proj = std::identity>
    requires hashable_text<R, Proj>
constexpr explicit string_hash(R &&seq, uint64 base = default_base(), Proj proj = {});
constexpr std::size_t size() const noexcept;
constexpr void reserve(std::size_t n);
template<detail::byte_hashable Symbol> constexpr void push_back(Symbol sym);
[[nodiscard]] constexpr hash_type substring_hash(std::size_t l, std::size_t r) const;
[[nodiscard]] constexpr hash_type full_hash() const;
[[nodiscard]] constexpr bool equal(std::size_t l1, std::size_t r1, std::size_t l2, std::size_t r2) const;
[[nodiscard]] constexpr hash_type concat(hash_type left, hash_type right, std::size_t right_length) const;
[[nodiscard]] constexpr std::size_t lcp(std::size_t i, std::size_t j) const;
[[nodiscard]] constexpr std::size_t lcp(std::size_t i, string_hash const &other, std::size_t j) const;
[[nodiscard]] constexpr uint64 base() const noexcept;
[[nodiscard]] static uint64 default_base();
template<typename R, typename Proj = std::identity>
    requires hashable_text<R, Proj>
[[nodiscard]] constexpr string_hash::hash_type hash_of(R &&seq, uint64 base = string_hash::default_base(), Proj proj = {});
```

| Operation | Result / effect | Time |
|---|---|---|
| constructors | Empty sequence, or hash the projected seq. Base is reduced modulo mod. | O(1) / O(n) |
| `size()` / `base()` | Number of symbols / reduced polynomial base. | O(1) |
| `reserve(n)` | Reserve tables for n symbols. | O(n) worst case |
| `push_back(sym)` | Append an encoded integer symbol. | O(1) amortized |
| `substring_hash(l, r)` / `full_hash()` | Reduced hash of `[l, r)` / whole sequence; empty hash is zero. | O(1) |
| `equal(l1, r1, l2, r2)` | Equal lengths and hashes, with possible false positives. | O(1) |
| `concat(left, right, right_length)` | Hash of left·right using this base. | O(1) if right_length <= size(), otherwise O(log(right_length + 1)) |
| `lcp(i, j)` / `lcp(i, other, j)` | Hash-based common-prefix length estimate; collisions can overestimate. | O(log(minimum suffix length + 1)) |
| `default_base()` | Process-seeded base in [2, mod - 2]. Runtime-only because it reads the process seed. | O(1) |
| `hash_of(seq, base, proj)` | Whole hash without tables. | O(n), O(1) extra space |

## Complexity

Stored hashes use two uint64 tables, 16 bytes per symbol plus two entries
and container capacity overhead. Input references and projections are not
retained. All operations are iterative. Explicit bases support constant
evaluation; omitted bases call runtime-only default_base(). Allocation
failures propagate standard container exceptions. If an append allocation
fails, destroy or assign the instance; no recovery guarantee is provided.
Moved-from instances may only be destroyed or assigned to.

## Preconditions

Asserted: base modulo mod is at least two; reserve addition fits size_t;
all substring intervals satisfy l <= r <= size(); lcp endpoints are valid
and both bases match; concat inputs are reduced hashes below mod.

Documented only: table lengths fit both vectors' max_size(); compared
sequences use compatible symbol encodings; concat parts use this base and
the supplied right length. Projection results must be stable when comparing
results from separate calls.

Each symbol encodes as `1 + unsigned(symbol) % (mod - 1)`, using its own
unsigned type, without truncating int128. Equal nonnegative values encode
alike across widths. Negative values can differ across widths: int8(-1)
encodes as 256, while int16(-1) encodes as 65536. Plain char follows its
platform signedness; cast byte data to unsigned char for portable byte hashes.
The mapping is injective on nonnegative values `[0, mod - 2]`, but larger
values can collide even as single-symbol sequences.

## Measured performance

`benchmarks/strings/string_hash.cpp` measures construction and query throughput
with explicit base 131; there is no alternative contender in this benchmark.

Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
2026-10-15. Flags: `-O2 -g -DNDEBUG -std=c++23`, optionally `-march=native`.
Minimum per-run mean ms over three runs of 25 samples; parentheses are
slowest/fastest spread.

| Shape | library -O2 | library native |
|---|---:|---:|
| build 10^6 | 4.8 (1.00x) | 5.0 (1.00x) |
| 10^5 instances of length 10 | 9.9 (1.00x) | 9.7 (1.06x) |
| forward + reverse 10^6 | 19.9 (1.01x) | 19.6 (1.01x) |
| 10^6 substring queries | 14.4 (1.12x) | 12.5 (1.11x) |

## Pitfalls and notes

Reserved push_back tables are retained because presized indexed writes did
not win consistently across construction shapes; the supplied record contains
only library timings for that comparison.

Hash equality is not content equality. For distinct encoded sequences whose
difference polynomial has degree d, a uniformly random base from
[2, mod - 2] gives collision probability at most d / (mod - 3). That bound
requires a nonzero polynomial; identical encoded digits defeat it. The seeded
default is not a cryptographic guarantee, and no default base can exceed
every allowed symbol. A fixed base makes examples reproducible, not exact.

For reproducibility across instances, pin the process hash seed prior to
construction or pass one explicit base. Hashes do not retain length, so
external keys should also record it. lcp binary search can overestimate
when a collision disrupts prefix equality.

## How it works

Prefix hashes follow `H[i+1] = H[i] * base + digit[i]` modulo mod, with
parallel powers of base. A substring subtracts `H[l] * base^(r-l)` from
H[r], so its hash is independent of position. Concatenation multiplies the
left hash by the right length's power and adds the right hash. Powers beyond
the table use modular exponentiation. LCP binary-searches the suffix length.
Arithmetic reuses static_modulus's Mersenne reduction and works with atl's
software int128 configuration as well as native wide arithmetic.

## See also

[String matching](matching.md), [Palindrome radii](palindrome-lengths.md).
