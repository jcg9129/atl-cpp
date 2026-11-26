# `atl::chinese_remainder`

Solve a linear congruence, intersect residue classes, or reconstruct a solution
modulo a target without forming the product of its moduli. Use intersection for
possibly shared factors; use Garner reconstruction for pairwise-coprime moduli.

```cpp
#include <atl/nt/crt.hpp>
```

\include crt/quick.cpp

Expected stdout:
\include crt/quick.out

## API

```cpp
template<unsigned_integral U>
struct congruence {
    U residue, modulus;
    constexpr bool operator==(congruence const &) const = default;
};
template<typename R>
concept congruence_range =
    std::ranges::input_range<R> && detail::is_congruence_v<std::ranges::range_value_t<R>>
    && std::convertible_to<std::ranges::range_reference_t<R>, std::ranges::range_value_t<R>>;

template<integral Coefficient, integral Constant, unsigned_integral Modulus>
constexpr std::optional<congruence<Modulus>> solve_linear_congruence(Coefficient a, Constant c, Modulus m);
template<unsigned_integral U>
constexpr std::optional<congruence<U>> chinese_remainder(congruence<U> const &first, congruence<U> const &second);
template<congruence_range R>
constexpr std::optional<std::ranges::range_value_t<R>> chinese_remainder(R &&congruences);
template<congruence_range R>
constexpr auto chinese_remainder_mod(R &&congruences, decltype(std::ranges::range_value_t<R>::modulus) target);
```

A canonical class has positive `modulus` and `residue < modulus`. `{0,1}` contains
every integer. Coefficients may use signed or unsigned integral types; unsigned
class types include `atl::uint128` on every target. Ranges are consumed once and
may yield proxies convertible to `congruence<U>`.

The linear solver returns `{x0,m/gcd(a,m)}` or `nullopt` if inconsistent.
Intersection returns the class modulo the lcm or `nullopt` if inconsistent;
an empty range returns `{0,1}`. Garner returns the canonical solution in
`[0,product(moduli))`, reduced modulo `target`; an empty range returns zero.

## Complexity

Linear solving: O(log m). Pair intersection: O(log min(m1,m2)). A k-class fold:
O(k log M), where M is the largest input modulus, O(1) auxiliary class objects.
Garner: O(k^2) modular products plus k inverses, O(k) allocated storage.
These count arithmetic operations; wide modular products may take binary steps.

## Preconditions

Asserted: positive m and target, canonical input classes, each running lcm fits U,
and Garner's moduli are pairwise coprime (checked through its inverses). Garner
never forms their product. Use `congruence<atl::uint128>` if a merged modulus
exceeds uint64. Allocation failures propagate.

## Measured performance

The three rows measure library calls with shared-factor pairs, three-prime
reconstructions, and general linear equations; there are no contender algorithms.

Source: `benchmarks/nt/crt.cpp`. These measurements use native 128-bit arithmetic; software-wide timings are not measured yet.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| workload (1e5 calls) | -O2 | -O2 -march=native |
| --- | --- | --- |
| pair intersection, moduli<1e9 | 5.8 (1.02x) | 5.6 (1.07x) |
| Garner, three primes to 1e9+7 | 15.6 (1.02x) | 15.9 (1.03x) |
| linear congruence, m<2^63 | 13.9 (1.02x) | 14.1 (1.02x) |


## Pitfalls and notes

Garner allocates a copy of the input system and two scratch vectors per call.
The arbitrary-modulus
[convolution](convolution.md) keeps a three-prime Garner specialized to its fixed
primes, avoiding runtime-modulus setup and those allocations. The generic
runtime-moduli function does not replace that specialized loop.

## How it works

Extended gcd supplies the inverse needed by the linear solver and the two-class
merge. The residue difference must be divisible by the gcd. The merge computes
only the multiple of the first modulus modulo the second cofactor, keeping the
result below the checked lcm. Garner computes mixed-radix digits, maintaining a
partial sum and a product modulo each later modulus and the target; no full
integer reconstruction is needed.

References: "Applications of modular linear equations" (2003);
The Art of Computer Programming, Volume 2 (modular arithmetic).

## See also

- [Extended gcd](gcd.md)
- [Discrete logarithms](discrete-log.md)
- [Convolution](convolution.md) — fixed three-prime Garner for polynomial products.
