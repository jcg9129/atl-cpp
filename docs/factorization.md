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

`<atl/nt/primality.hpp>` provides `is_prime` and its Miller–Rabin kernel
without the sieve or factorization storage. `<atl/nt/basic.hpp>` continues to
include primality and factorization together.

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

Source: `benchmarks/nt/factorization.cpp`. These measurements use native 128-bit arithmetic; software-wide timings are not measured yet.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| workload | method | -O2 | -O2 -march=native |
| --- | --- | --- | --- |
| 100 inputs near 1e9 | library | 0.084 (1.04x) | 0.084 (1.01x) |
| 100 inputs near 1e9 | by trial division | 2.9 (1.00x) | 2.9 (1.00x) |
| 20 inputs near 1e12 | library | 0.039 (1.04x) | 0.039 (1.02x) |
| 20 inputs near 1e12 | by trial division | 13.3 (1.00x) | 13.3 (1.00x) |
| 5e6 dependent products | uint128 remainder | 31.4 (1.00x) | 31.3 (1.00x) |
| 5e6 dependent products | barrett64 | 30.3 (1.00x) | 32.6 (1.00x) |
| 5e6 dependent products | montgomery64 | 16.5 (1.00x) | 16.5 (1.00x) |
| 1e4 primes near 1e9 | by twelve bases | 20.9 (1.00x) | 20.8 (1.00x) |
| 1e4 primes near 1e9 | library | 3.8 (1.01x) | 3.8 (1.00x) |
| 1e4 primes near 1e12 | by twelve bases | 31.7 (1.00x) | 31.6 (1.01x) |
| 1e4 primes near 1e12 | library | 10.7 (1.01x) | 10.7 (1.00x) |
| 1e4 primes near 2^62 | by twelve bases | 54.2 (1.01x) | 54.4 (1.00x) |
| 1e4 primes near 2^62 | library | 16 (1.01x) | 16 (1.00x) |
| 1e5 random 64-bit odds | by twelve bases | 40.6 (1.01x) | 40.5 (1.00x) |
| 1e5 random 64-bit odds | library | 16.4 (1.01x) | 16.4 (1.00x) |
| 200 semiprimes, two primes near 2^31 | by remainder products | 98.8 (1.01x) | 98.9 (1.00x) |
| 200 semiprimes, two primes near 2^31 | by bounded trial 0 | 59.1 (1.01x) | 59 (1.00x) |
| 200 semiprimes, two primes near 2^31 | by bounded trial 64 | 59.1 (1.06x) | 59 (1.00x) |
| 200 semiprimes, two primes near 2^31 | library | 59.1 (1.01x) | 59.1 (1.00x) |
| 2000 random 64-bit values | by remainder products | 47.3 (1.01x) | 47.4 (1.01x) |
| 2000 random 64-bit values | by bounded trial 0 | 27.7 (1.00x) | 27.7 (1.00x) |
| 2000 random 64-bit values | by bounded trial 64 | 26.6 (1.00x) | 26.6 (1.00x) |
| 2000 random 64-bit values | library | 26.4 (1.01x) | 26.4 (1.00x) |
| 2000 smooth values >=2^40 | by remainder products | 11 (1.01x) | 11 (1.00x) |
| 2000 smooth values >=2^40 | by bounded trial 0 | 7.3 (1.00x) | 7.3 (1.00x) |
| 2000 smooth values >=2^40 | by bounded trial 64 | 7 (1.01x) | 7 (1.00x) |
| 2000 smooth values >=2^40 | library | 5.9 (1.02x) | 5.9 (1.00x) |


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
