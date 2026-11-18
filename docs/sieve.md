# `atl::sieve`

A linear sieve stores factor information on `[0, n)` for repeated primality,
factorization and divisor queries. Use `prime_range` for just a prime list,
and the per-number factorization functions for isolated large inputs.

```cpp
#include <atl/nt/sieve.hpp>
```

\include sieve/quick.cpp

Expected stdout:
\include sieve/quick.out

## API

```cpp
struct sieve_item_info {
    uint32 prime, exponent, prime_power;
    constexpr bool operator==(sieve_item_info const &) const = default;
};
struct prime_factor {
    uint64 prime, exponent;
    constexpr bool operator==(prime_factor const &) const = default;
};
constexpr std::vector<sieve_item_info> sieve_info(std::size_t n);
constexpr std::vector<uint32> prime_range(std::size_t n);
```

`prime_factor` describes a prime power. `sieve_item_info` records the least prime
factor, its exponent and its power. Index 0 is `{0,0,0}`; index 1 is `{1,0,1}`.
`sieve_info` returns `n` records; `prime_range` returns ascending primes below `n`.

All class operations are `constexpr`:

| signature | result |
|---|---|
| `explicit sieve(std::size_t n)` | build on `[0,n)` |
| `std::size_t size() const` | exclusive bound |
| `bool is_prime(std::size_t i) const` | primality of `i` |
| `uint32 min_prime_factor(std::size_t i) const` | least prime divisor |
| `std::vector<uint32> const &primes() const` | ascending primes |
| `static_vector<prime_factor, max_distinct_primes> factorization(std::size_t i) const` | ascending prime powers |
| `static_vector<uint32, max_distinct_primes> prime_factors(std::size_t i) const` | ascending distinct primes |
| `std::vector<uint32> divisors(std::size_t i) const` | ascending positive divisors |

The fixed result capacity `max_distinct_primes` is 9: a product of ten distinct
primes exceeds the 32-bit domain. Factorization of 1 is empty; its divisors are `{1}`.
The prime-list reference survives queries and expires on assignment, moving from,
or destruction of the owning sieve.

## Complexity

Construction and `sieve_info`: O(n) time and storage. Primality, minimum prime
factor and accessors: O(1). Factor walks: O(omega(i)), at most 9 steps, with
bounded result storage. Divisors: O(d(i) log d(i)) time and O(d(i)) output space.
`prime_range`: O(n log log n) time, O(sqrt(n)) scratch plus a 64 KiB block and output.

## Preconditions

Asserted: `n < 2^32`; primality requires `i < size()`; minimum prime factor requires
`2 <= i < size()`; factorization, prime factors and divisors require `1 <= i < size()`.
Zero-sized and one-sized tables are valid. Allocation failures propagate.

## Measured performance

Linear builds compare a modulo divisibility test with a least-prime comparison.
Factor walks compare index widths and omitting result construction; prime lists
compare segmented and unsegmented byte arrays.

Source: `benchmarks/nt/sieve.cpp`. Intel Xeon Platinum 8280, GCC 15.2,
RelWithDebInfo, assertions disabled, 2026-10-15. Columns give milliseconds for
`-O2 -g -DNDEBUG -std=c++23` and the same flags plus `-march=native`.
Each cell is minimum milliseconds (spread = slowest run / fastest run), over
three runs of 25 samples; each run contributes its mean.
These records measure native 128-bit arithmetic; software-wide timings are not measured yet.

| workload | method | -O2 | -O2 -march=native |
| --- | --- | --- | --- |
| build n=1e6 | library (modulo) | 5.7 (1.07x) | 6.5 (1.01x) |
| build n=1e6 | by least prime comparison | 5.7 (1.03x) | 5.8 (1.00x) |
| build n=1e7 | library (modulo) | 163 (1.02x) | 164 (1.01x) |
| build n=1e7 | by least prime comparison | 169 (1.03x) | 166 (1.01x) |
| factor every i<1e7 | by size type walk | 918 (1.02x) | 806 (1.05x) |
| factor every i<1e7 | library (uint32 walk) | 690 (1.02x) | 490 (1.07x) |
| exponent sum every i<1e7 | by uint32 exponent sum | 439 (1.03x) | 476 (1.05x) |
| prime factors every i<1e7 | library | 414 (1.04x) | 413 (1.04x) |
| primes n=1e6 | by unsegmented sieve | 1.5 (1.06x) | 1.6 (1.01x) |
| primes n=1e6 | library | 1.5 (1.02x) | 1.3 (1.01x) |
| primes n=1e7 | by unsegmented sieve | 22.3 (1.03x) | 23.7 (1.00x) |
| primes n=1e7 | library | 14.9 (1.01x) | 13.6 (1.01x) |
| primes n=1e8 | by unsegmented sieve | 617 (1.02x) | 646 (1.01x) |
| primes n=1e8 | library | 162 (1.00x) | 152 (1.01x) |


The modulo test has no consistent disadvantage across shapes. Segmentation keeps
strikes in cache as the range grows. Returning the fixed factor object costs more
than directly accumulating exponents; the latter computes a smaller result.

## Pitfalls and notes

The free `sieve_info` still builds a full table; it is not a per-number factorizer.
The class does not provide totients; use the multiplicative range functions.

## How it works

The linear pass generates each composite once, stopping at the least prime factor
of its cofactor. Repeated least-prime powers let factor queries divide out an
entire prime power in one step. Divisors expand those powers and sort the result.
Prime enumeration uses odd indices, base primes through sqrt(n), and saved strike
positions across 64 KiB blocks.

Reference: The Art of Computer Programming, Volume 2 (prime sieves and factorization).

## See also

- [Factorization](factorization.md)
- [Multiplicative functions](multiplicative-functions.md)
- [Static vector](static-vector.md)
