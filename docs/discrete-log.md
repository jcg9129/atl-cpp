# `atl::discrete_log`, `atl::multiplicative_order`, `atl::primitive_root`

Find the least exponent reaching a residue modulo any positive uint64 modulus,
or compute a unit's order and a prime modulus's least primitive root. Smooth
orders are efficient; a large prime factor of the order can require a large table.

```cpp
#include <atl/nt/discrete_log.hpp>
```

\include discrete-log/quick.cpp

Expected stdout:
\include discrete-log/quick.out

## API

```cpp
inline std::optional<uint64> discrete_log(uint64 a, uint64 b, uint64 m);
constexpr uint64 multiplicative_order(uint64 a, uint64 m);
constexpr uint64 primitive_root(uint64 p);
```

`discrete_log` returns the least `x>=0` with `a^x=b (mod m)`, or `nullopt` if
unreachable. The convention is `0^0=1`; modulo 1 the exponent is zero. Both a and
b are reduced internally, and a need not be a unit. `multiplicative_order`
returns the least positive exponent reaching 1, with value 1 modulo 1.
`primitive_root` returns the least positive generator, including 1 modulo 2.
Discrete log is not constexpr: its `atl::flat_hash_map` tables use a runtime seed.
Hash seeds affect table layout, while the least returned exponent is invariant.

## Complexity

Discrete log uses expected O(sum e*(sqrt(q)+log m)) modular steps over the prime
powers q^e of the reduced unit's order, plus factorizations of m and of p-1 for
primes p dividing m. Table memory is O(max sqrt(q)); factorization scratch is
O(log m). A unit order costs those factorizations plus O(log^2 m) products.
Primitive roots factor p-1 then use O(log p) products per distinct factor per
candidate; at most p-1 candidates are examined.

## Preconditions

Asserted: m is positive; order requires gcd(a,m)=1; primitive roots require prime p.
The full uint64 modulus range is available with software `atl::uint128` too.
Allocation failure propagates. A table count is checked against `size_t` capacity
prior to conversion; the flat hash container also checks its own capacity bounds.
Practical memory limits are much smaller: prime-order groups near 1e18 need
about 1e9 baby entries. Those inputs retain their mathematical domain but can
exhaust memory.

## Measured performance

The strategy race compares Pohlig-Hellman with whole-group baby-step giant-step;
the engine race compares Barrett32, Barrett64 and runtime Montgomery. Targets
are powers of the least primitive root, so every timed query is solvable.

Source: `benchmarks/nt/discrete_log.cpp`. Intel Xeon Platinum 8280, GCC 15.2,
RelWithDebInfo, assertions disabled, 2026-10-15. Timings are milliseconds for
`-O2 -g -DNDEBUG -std=c++23` and the same flags plus `-march=native`.
Each cell is minimum milliseconds (spread = slowest run / fastest run), over
three runs of 25 samples; each run contributes its mean.
These records measure native 128-bit arithmetic; software-wide timings are not measured yet.

| modulus / query count | method | -O2 | -O2 -march=native |
| --- | --- | --- | --- |
| 998244353 / 100 | library | 1.1 (1.03x) | 1.1 (1.01x) |
| 998244353 / 100 | by pohlig hellman barrett32 | 1.1 (1.01x) | 1.1 (1.04x) |
| 998244353 / 100 | by baby step giant step barrett32 | 76.3 (1.01x) | 79.8 (1.01x) |
| 998244353 / 100 | by pohlig hellman barrett64 | 1.4 (1.02x) | 1.6 (1.00x) |
| 998244353 / 100 | by pohlig hellman montgomery | 1.2 (1.00x) | 1.2 (1.02x) |
| 998244353 / 100 | by baby step giant step barrett64 | 95.1 (1.00x) | 107 (1.01x) |
| 998244353 / 100 | by baby step giant step montgomery | 79.5 (1.01x) | 79.0 (1.00x) |
| 1e9+7 / 20 | library | 13.1 (1.00x) | 13.7 (1.01x) |
| 1e9+7 / 20 | by pohlig hellman barrett32 | 13.1 (1.04x) | 13.6 (1.01x) |
| 1e9+7 / 20 | by baby step giant step barrett32 | 15.4 (1.02x) | 16.1 (1.01x) |
| 1e9+7 / 20 | by pohlig hellman barrett64 | 15.7 (1.01x) | 15.9 (1.01x) |
| 1e9+7 / 20 | by pohlig hellman montgomery | 13.2 (1.00x) | 14.1 (1.00x) |
| 1e9+7 / 20 | by baby step giant step barrett64 | 21.1 (1.01x) | 21.6 (1.01x) |
| 1e9+7 / 20 | by baby step giant step montgomery | 16.0 (1.01x) | 16.0 (1.00x) |
| 29*2^57+1 / 100 | library | 3.8 (1.00x) | 3.7 (1.01x) |
| 29*2^57+1 / 100 | by pohlig hellman barrett64 | 4.6 (1.00x) | 5.4 (1.00x) |
| 29*2^57+1 / 100 | by pohlig hellman montgomery | 3.8 (1.00x) | 3.7 (1.01x) |
| semi-smooth ~1e18, q~1e12 / 5 | library | 377 (1.04x) | 381 (1.07x) |
| semi-smooth ~1e18, q~1e12 / 5 | by pohlig hellman barrett64 | 430 (1.01x) | 452 (1.01x) |
| semi-smooth ~1e18, q~1e12 / 5 | by pohlig hellman montgomery | 371 (1.01x) | 384 (1.02x) |


Pohlig-Hellman wins strongly on smooth orders. Barrett32 ties its library dispatch
near 1e9, while Montgomery has the smallest wide-modulus times. Whole-group
BSGS near 1e18 was excluded for its billion-entry table requirement.

## Pitfalls and notes

An order is defined only for units; a nonunit base is valid for `discrete_log`.
A target outside the generated subgroup has no logarithm even when it is a unit.
The table algorithms remain in `experimental/atl/experimental/nt/discrete_log_variants.hpp`.

## How it works

The solver checks exponents below 64, then removes shared gcd factors, shifting
the exponent at each step. The coprime core factors Carmichael's exponent,
strips redundant prime factors to obtain the base order, and solves one base-q
digit at a time in each prime-power subgroup. A baby-step table is reused across
the digits for that prime. CRT joins the pieces and a final power rejects any
outside-subgroup target. Products dispatch to Barrett32 below 2^32, Montgomery
for wider odd moduli, and Barrett64 for wider even moduli.

Reference: Handbook of Applied Cryptography (discrete logarithm algorithms).

## See also

- [Factorization](factorization.md)
- [CRT](crt.md)
- [Flat hash containers](flat-hash.md)
