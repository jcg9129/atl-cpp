# `atl::is_prime`, `atl::factorization`

Test primality or factor a single unsigned 64-bit integer without a precomputed
table. For many queries within a small fixed bound, a sieve amortizes its build
across the queries.

```cpp
#include <atl/nt/basic.hpp>
```

\include factorization/quick.cpp

Expected stdout:
\include factorization/quick.out

`<atl/nt/basic.hpp>` is the normal entry point for primality and factorization.
The reduced alternative `<atl/nt/primality.hpp>` provides `is_prime` and its
Miller–Rabin kernel without sieve or factorization storage.

## API

```cpp
constexpr bool is_prime(uint64 n);
constexpr std::vector<prime_factor> factorization(uint64 n);
```

`is_prime` is deterministic for every `n < 2^64`, returning false for 0 and 1.
`factorization` returns distinct prime bases in ascending order with their positive
exponents; 0 and 1 return an empty vector. The result type is declared in
`#include <atl/nt/sieve.hpp>` as `struct prime_factor { uint64 prime, exponent; };`.
Both functions support constant evaluation, including software-wide arithmetic.

## Complexity

Primality takes O(log n) modular products and O(1) storage. Factorization has
expected O(n^(1/4) polylog n) time and O(log n) scratch/result storage. The
expected factor-extraction bound does not promise a maximum number of retries.

## Preconditions

Every uint64 value is accepted. Allocation failure propagates. Montgomery
products use `atl::uint128` and the same input domain on every target.
The factor search uses successive uint64 polynomial constants, independent of
`std::hash` and the width of `size_t`.

## Measured performance

Trial division is the readable factorization oracle. Other contenders use twelve
Miller-Rabin bases with remainder products, or Montgomery rho with trial bounds
0 and 64. The library uses three or seven primality bases and trial bound 256.

Source: `benchmarks/nt/factorization.cpp`. Intel Xeon Platinum 8280, GCC 15.2,
RelWithDebInfo, assertions disabled, 2026-10-15. Timings are milliseconds for
`-O2 -g -DNDEBUG -std=c++23` and the same flags plus `-march=native`.
Each cell is minimum milliseconds (spread = slowest run / fastest run), over
three runs of 25 samples; each run contributes its mean.
These records measure native 128-bit arithmetic; software-wide timings are not measured yet.

| workload | method | -O2 | -O2 -march=native |
| --- | --- | --- | --- |
| 100 inputs near 1e9 | library | 0.207 (1.02x) | 0.201 (1.06x) |
| 100 inputs near 1e9 | by trial division | 9.4 (1.01x) | 9.4 (1.00x) |
| 20 inputs near 1e12 | library | 0.076 (1.06x) | 0.073 (1.11x) |
| 20 inputs near 1e12 | by trial division | 43.4 (1.00x) | 43.5 (1.00x) |
| 5e6 dependent products | uint128 remainder | 134 (1.00x) | 134 (1.00x) |
| 5e6 dependent products | barrett64 | 30.8 (1.02x) | 34.4 (1.01x) |
| 5e6 dependent products | montgomery64 | 19.4 (1.00x) | 18.6 (1.01x) |
| 1e4 primes near 1e9 | by twelve bases | 51.5 (1.00x) | 51.1 (1.02x) |
| 1e4 primes near 1e9 | library | 6.1 (1.00x) | 5.6 (1.01x) |
| 1e4 primes near 1e12 | by twelve bases | 178 (1.00x) | 178 (1.01x) |
| 1e4 primes near 1e12 | library | 15.5 (1.00x) | 14.7 (1.02x) |
| 1e4 primes near 2^62 | by twelve bases | 266 (1.00x) | 266 (1.01x) |
| 1e4 primes near 2^62 | library | 22.2 (1.00x) | 21.0 (1.01x) |
| 1e5 random 64-bit odds | by twelve bases | 200 (1.01x) | 200 (1.00x) |
| 1e5 random 64-bit odds | library | 25.5 (1.00x) | 25.5 (1.00x) |
| 200 semiprimes, two primes near 2^31 | by remainder products | 501 (1.00x) | 501 (1.00x) |
| 200 semiprimes, two primes near 2^31 | by bounded trial 0 | 75.7 (1.00x) | 74.3 (1.00x) |
| 200 semiprimes, two primes near 2^31 | by bounded trial 64 | 75.7 (1.00x) | 74.2 (1.01x) |
| 200 semiprimes, two primes near 2^31 | library | 75.7 (1.00x) | 74.3 (1.01x) |
| 2000 random 64-bit values | by remainder products | 228 (1.00x) | 228 (1.00x) |
| 2000 random 64-bit values | by bounded trial 0 | 36.8 (1.01x) | 36.2 (1.00x) |
| 2000 random 64-bit values | by bounded trial 64 | 35.4 (1.01x) | 34.7 (1.01x) |
| 2000 random 64-bit values | library | 35.9 (1.00x) | 35.2 (1.00x) |
| 2000 smooth values >=2^40 | by remainder products | 36.2 (1.00x) | 35.3 (1.00x) |
| 2000 smooth values >=2^40 | by bounded trial 0 | 11.9 (1.00x) | 11.9 (1.01x) |
| 2000 smooth values >=2^40 | by bounded trial 64 | 11.5 (1.01x) | 11.8 (1.01x) |
| 2000 smooth values >=2^40 | library | 11.1 (1.01x) | 10.8 (1.03x) |


Trial division reduces rho work on smooth values. Bounds 0, 64 and 256 have
similar semiprime costs, and 256 has the smallest smooth-input time in both flag
sets. Montgomery avoids a wide division at each multiplication.

## Pitfalls and notes

`factorization(0)` returning empty is a convention, not a mathematical prime
factorization of zero. Large prime factors can make trial division impractical.

## How it works

Primality first tests divisibility by primes through 37. The remaining odd inputs
use Miller-Rabin with bases {2,7,61} below 4,759,123,141 and
{2,325,9375,28178,450775,9780504,1795265022} throughout the remaining uint64 range.
A base congruent to zero is skipped. The test runs in Montgomery form.
Factorization strips twos, divides odd factors below 256, then uses an iterative
stack of rho splits and primality tests. Brent cycle detection batches 128 gcd
steps; a colliding batch is replayed to isolate a proper factor. Sorting the
remaining prime leaves combines repeated factors into exponents.

Reference: The Art of Computer Programming, Volume 2 (factoring into primes).

## See also

- [Sieve](sieve.md)
- [Modular arithmetic](modular-arithmetic.md)
