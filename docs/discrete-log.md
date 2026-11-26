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

Source: `benchmarks/nt/discrete_log.cpp`. These measurements use native 128-bit arithmetic; software-wide timings are not measured yet.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| modulus / query count | method | -O2 | -O2 -march=native |
| --- | --- | --- | --- |
| 998244353 / 100 | library | 0.389 (1.02x) | 0.369 (1.04x) |
| 998244353 / 100 | by pohlig hellman barrett32 | 0.392 (1.02x) | 0.37 (1.04x) |
| 998244353 / 100 | by baby step giant step barrett32 | 61 (1.02x) | 59.2 (1.00x) |
| 998244353 / 100 | by pohlig hellman barrett64 | 0.572 (1.02x) | 0.528 (1.05x) |
| 998244353 / 100 | by pohlig hellman montgomery | 0.384 (1.02x) | 0.366 (1.04x) |
| 998244353 / 100 | by baby step giant step barrett64 | 72.5 (1.03x) | 70 (1.02x) |
| 998244353 / 100 | by baby step giant step montgomery | 60.2 (1.03x) | 58.8 (1.02x) |
| 1e9+7 / 20 | library | 10 (1.03x) | 9.8 (1.05x) |
| 1e9+7 / 20 | by pohlig hellman barrett32 | 10.1 (1.02x) | 10 (1.02x) |
| 1e9+7 / 20 | by baby step giant step barrett32 | 12.2 (1.04x) | 11.8 (1.06x) |
| 1e9+7 / 20 | by pohlig hellman barrett64 | 12 (1.03x) | 11.4 (1.05x) |
| 1e9+7 / 20 | by pohlig hellman montgomery | 10.1 (1.02x) | 9.6 (1.06x) |
| 1e9+7 / 20 | by baby step giant step barrett64 | 14.7 (1.02x) | 14.1 (1.05x) |
| 1e9+7 / 20 | by baby step giant step montgomery | 12.2 (1.03x) | 11.8 (1.04x) |
| 29*2^57+1 / 100 | library | 1.3 (1.04x) | 1.2 (1.03x) |
| 29*2^57+1 / 100 | by pohlig hellman barrett64 | 1.9 (1.03x) | 1.8 (1.05x) |
| 29*2^57+1 / 100 | by pohlig hellman montgomery | 1.3 (1.02x) | 1.2 (1.05x) |
| semi-smooth ~1e18, q~1e12 / 5 | library | 334 (1.05x) | 255 (1.04x) |
| semi-smooth ~1e18, q~1e12 / 5 | by pohlig hellman barrett64 | 428 (1.04x) | 302 (1.02x) |
| semi-smooth ~1e18, q~1e12 / 5 | by pohlig hellman montgomery | 343 (1.02x) | 253 (1.04x) |


Pohlig-Hellman wins strongly on smooth orders. Near 1e9, Barrett32 and
the library dispatch are close. At 998244353 the library narrowly loses to
Pohlig-Hellman Montgomery (0.389/0.369 versus 0.384/0.366 ms). For the
semi-smooth wide modulus the library wins plain (334 versus 343 ms), but
loses native (255 versus 253 ms); neither engine wins every comparison. Whole-group
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
