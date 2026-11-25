# `atl::series_inverse`, `atl::series_exponential`, `atl::series_power` and prime-field helpers

Formal power series over a prime field, as plain `std::vector`s of coefficients in
increasing degree order: reciprocal, division, logarithm, exponential, integer
powers, canonical square roots, polynomial division with remainder, and sparse
variants of each, all in `O(M(n))` through Newton iteration on the library's
convolution. Reach for them when a problem asks for `f^-1`, `log f`, `exp f`,
`f^k` or `sqrt f` modulo `x^n`. The support pieces are usable alone: factorial
tables, batch inversion, the derivative and integral of a coefficient list, and
square roots modulo a prime. Skip this page for series over other fields or moduli
that are not prime; the concepts reject them.

```cpp
#include <atl/nt/power_series.hpp>
#include <atl/nt/factorial_table.hpp>
#include <atl/nt/batch_inverse.hpp>
#include <atl/nt/series_calculus.hpp>
#include <atl/nt/square_root_mod.hpp>
#include <atl/nt/prime_mod_int.hpp>  // the concepts and series_coefficient_t; every header above includes it
```

\include power-series/quick.cpp

Expected stdout:
\include power-series/quick.out

`<atl/nt/power_series.hpp>` is the normal entry point for the full generic API
and its complexity contracts. The examples use the reduced alternative
`<atl/nt/power_series_native.hpp>` for prime moduli fitting in 32 bits, with
coefficient storage supporting double-width multiplication. Here “native” names
the modulus/widening policy; widened storage can use portable arithmetic. This
path retains direct NTT, FFT split and CRT products.
`<atl/nt/power_series_wide.hpp>` installs the wider-residue product policy:
16-bit limbs at runtime, schoolbook products in constant evaluation. Include it
before using wider coefficient types.

The helper example similarly uses `<atl/nt/factorial_table_core.hpp>`,
`<atl/nt/batch_inverse_core.hpp>`, `<atl/nt/series_calculus_core.hpp>` and
`<atl/nt/square_root_mod_core.hpp>` as reduced alternatives to the general
headers listed above. `<atl/nt/prime_mod_int_core.hpp>` provides the shared
prime-field concepts without factorization storage.

## Conventions and concepts

```cpp
template<typename Coefficient>
concept static_prime_mod_int = /* static_mod_int, prime modulus that fits in 64 bits */;

template<typename Range>
concept prime_mod_int_range = /* input_range<Range const> of static_prime_mod_int */;

template<std::ranges::input_range Range>
using series_coefficient_t = std::ranges::range_value_t<Range const>;

inline constexpr std::size_t maximum_series_length = std::size_t{1} << 22;
```

- Coefficients are ordered by increasing degree and results are never trimmed:
  a length is exactly the number of coefficients returned, trailing zeros included.
- Inputs are any `prime_mod_int_range`: vectors, arrays, spans, lists, views, even
  single-pass input ranges, read once through a const reference and never modified.
  Primality of the modulus is decided at compile time: `static_prime_mod_int`
  runs `is_prime` on the static `Coefficient::mod()` while the constraint is
  checked, so dynamic and composite moduli are rejected at the template boundary.
  The coefficient type is taken as written (a reference or cv-qualified type does
  not model it); a range may be a reference or const, and its element type is
  `range_value_t<Range const>`, so ranges of `const` coefficients qualify.
- `series_coefficient_t<Range>` is that element type; every series function returns
  coefficients of it, and two input ranges must have the same one.
- `length` is the number of output coefficients. Missing input coefficients are
  zero and input coefficients at indices `>= length` are not read.
- The transform-backed functions take an `Engine` (default `default_ntt_engine`)
  and assert `length <= maximum_series_length`; that bound keeps the
  convolution's three-prime reconstruction exact for every 32-bit modulus. The
  sparse functions and the support pieces have no such cap.
- Everything is `constexpr`. Constant evaluation uses scalar transforms, and for
  moduli without a transform-friendly structure the products are quadratic.

## Prime-field helpers

```cpp
template<static_prime_mod_int Coefficient>
class factorial_table {
public:
    explicit constexpr factorial_table(std::size_t maximum_index);
    constexpr void extend(std::size_t maximum_index);
    constexpr std::size_t maximum_index() const;
    constexpr Coefficient factorial(std::size_t i) const;
    constexpr Coefficient inverse_factorial(std::size_t i) const;
    constexpr Coefficient inverse_integer(std::size_t i) const;
    constexpr Coefficient binomial(std::size_t n, std::size_t k) const;
};

template<prime_mod_int_range Range>
constexpr auto batch_inverse(Range const &values) -> std::vector<std::ranges::range_value_t<Range const>>;

template<prime_mod_int_range Range>
constexpr auto series_derivative(Range const &values) -> std::vector<std::ranges::range_value_t<Range const>>;

template<prime_mod_int_range Range>
constexpr auto series_integral(Range const &values) -> std::vector<std::ranges::range_value_t<Range const>>;

constexpr std::optional<std::uint64_t> square_root_mod(std::uint64_t value, std::uint64_t prime);

template<static_prime_mod_int Coefficient>
constexpr std::optional<Coefficient> square_root_mod(Coefficient value);
```

| Entry | Result | Time |
|---|---|---|
| `factorial_table(m)` | tables of `i!`, `1/i!` and `1/i` for `0 <= i <= m` | `O(m)` field operations and storage, one inversion |
| `extend(m)` | grows the tables through index `m`; a smaller `m` does nothing; existing entries keep their values, on allocation failure the table keeps its previous size; accessors return values, so growth invalidates nothing a caller holds | `O(m - old m)` field operations and one inversion when it grows; geometric capacity, so repeated small extensions stay linear |
| `factorial(i)`, `inverse_factorial(i)`, `inverse_integer(i)` (`i >= 1`) | table lookup | `O(1)` |
| `binomial(n, k)` | `n choose k`, zero when `k > n` | `O(1)` |
| `batch_inverse(values)` | elementwise inverses in input order; empty input gives empty output | `O(m)` and one inversion |
| `series_derivative(values)` | `max(m - 1, 0)` coefficients; degrees are taken in the field | `O(m)` |
| `series_integral(values)` | `m + 1` coefficients with constant term zero; `{0}` for empty input | `O(m)`, no inversion per term |
| `square_root_mod(value, prime)` | smaller root (`<= prime / 2`), zero for zero, or `nullopt` for a non-residue; supports 2 and every 64-bit prime | one exponentiation for `prime = 3 mod 4`; otherwise `O(log p + s^2)` multiplications after the non-residue search, `s` the exponent of 2 in `p - 1` |

\include power-series/prime-field-helpers.cpp

Expected stdout:
\include power-series/prime-field-helpers.out

`atl::experimental::square_root_mod_by_cipolla` (header
`<atl/experimental/nt/square_root_mod_variants.hpp>`) is the contender: same result
and preconditions, `O(log p)` multiplications after its parameter search,
independent of `s`.

## Series functions

```cpp
template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
constexpr auto series_inverse(Range const &series, std::size_t length) -> std::vector<series_coefficient_t<Range>>;
// series_logarithm, series_exponential: same shape

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Numerator, prime_mod_int_range Denominator>
    requires std::same_as<series_coefficient_t<Numerator>, series_coefficient_t<Denominator>>
constexpr auto series_division(Numerator const &numerator, Denominator const &denominator, std::size_t length)
    -> std::vector<series_coefficient_t<Numerator>>;

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
constexpr auto series_power(Range const &series, std::uint64_t exponent, std::size_t length)
    -> std::vector<series_coefficient_t<Range>>;

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
constexpr auto series_square_root(Range const &series, std::size_t length)
    -> std::optional<std::vector<series_coefficient_t<Range>>>;

template<static_prime_mod_int Coefficient>
struct polynomial_division_result { std::vector<Coefficient> quotient, remainder; };

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Numerator, prime_mod_int_range Denominator>
    requires std::same_as<series_coefficient_t<Numerator>, series_coefficient_t<Denominator>>
constexpr auto polynomial_division(Numerator const &numerator, Denominator const &denominator)
    -> polynomial_division_result<series_coefficient_t<Numerator>>;
```

The sparse entries drop `Engine` and the transform cap:
`series_inverse_sparse`, `series_division_sparse`, `series_logarithm_sparse`,
`series_exponential_sparse`, `series_power_sparse` and `series_square_root_sparse`
have the signatures of their dense counterparts without the `Engine` parameter.

| Function | Result modulo `x^length` | Preconditions for `length > 0` |
|---|---|---|
| `series_inverse(f, n)` | `1/f` | `f[0] != 0` |
| `series_division(f, g, n)` | `f/g` | `g[0] != 0` |
| `series_logarithm(f, n)` | `log f`, constant term 0 | `f[0] == 1`, `n <= p` |
| `series_exponential(f, n)` | `exp f`, constant term 1 | `f[0] == 0`, `n <= p` |
| `series_power(f, e, n)` | `f^e` for any 64-bit `e` | none beyond the cap |
| `series_square_root(f, n)` | canonical `g` with `g^2 = f`, or `nullopt` | none beyond the cap |

At zero length every function returns an empty vector without checking constant
terms. An empty input has constant term zero, so `series_exponential({}, n)` is the
constant one. Inverse and division work beyond `p`; logarithm and exponential
need division by every integer below `n`, hence `n <= p`.

**Powers.** Exponent zero gives the constant one, also for the zero series; a zero
series to a positive exponent gives zeros. With valuation `v`, the shift `v * e`
is compared against the length before it is formed, so `10^18` and `2^64 - 1`
are safe. There is no bound by `p`: when the precision left after removing the
valuation exceeds `p`, binary powering replaces `exp(e log f)`.

**Square roots.** In odd characteristic a nonzero input needs an even valuation
`2s` and a leading coefficient that is a square; otherwise the result is `nullopt`.
The leading coefficient of the root is the smaller scalar root. The unit part is
determined through `n - 2s` coefficients, shifted by `s`; the last `s`
coefficients, which the input does not determine, are zero. An all-zero input gives
zeros. In characteristic two, `g[i] = f[2i]` and every odd-indexed coefficient of
the input below `n` must vanish.

**Polynomial division.** The whole inputs are consumed, trailing zeros ignored.
The result is the unique `f = q g + r` with `deg r < deg g`, both normalized (the
zero polynomial is an empty vector). The divisor must not be the zero polynomial,
even for a zero dividend.

**Sparse entries.** They solve the same problems by triangular recurrences that
visit only the `s` nonconstant nonzero terms of the input (of the denominator for
division). Dense entries never dispatch by sparsity. Inverse and division never
divide by an integer and have no bound by `p`. Logarithm and exponential divide by
every integer below `n` and need `n <= p`. The power and the root divide only over
the recurrence that is actually run. Write `f = x^v u` with `u(0) != 0`:

- `series_power_sparse(f, e, n)` shifts the unit power by `e * v` and runs the
  recurrence over the remaining `n - e * v` coefficients, so it needs
  `n - e * v <= p`. Length zero, exponent zero, the zero series and a shift
  `e * v >= n` (all zeros) have no bound.
- `series_square_root_sparse(f, n)` in odd characteristic shifts the root of `u` by
  `v / 2` and runs the recurrence over the `n - v` known coefficients of `u`, so it
  needs `n - v <= p`. The zero series and the `nullopt` cases (odd `v`, non-square
  leading coefficient) have no bound, and neither has characteristic two.

For example, modulo 7 the cube of `x^2 (1 + x)` to length 10 runs the recurrence
over 4 coefficients and is valid, while the square of `x (1 + x)` to length 10 runs
it over 8 and is rejected.

\include power-series/powers-and-roots.cpp

Expected stdout:
\include power-series/powers-and-roots.out

## Complexity

`M(n)` is the cost of a length-`n` product.

| Function | Time | Memory |
|---|---|---|
| `series_inverse`, `series_division`, `series_logarithm`, `series_exponential`, `series_square_root`, `polynomial_division` | `O(M(n))` | `O(n)` |
| `series_power` | `O(M(n) + log(e + 1))`, or `O(M(n) log(e + 1))` in the binary fallback | `O(n)` |
| sparse entries | `O(n (s + 1))`, plus `O(log(e + 1))` for the power and one scalar root for the root | `O(n + s)` |

Allocation failures propagate.

## Preconditions

Asserted with `atl_assert` (active only under `-DATL_ASSERT`):

- `length <= maximum_series_length` for every transform-backed function; for
  `polynomial_division`, the normalized input lengths (trailing zeros may exceed it).
- Constant terms and length bounds from the table above, and a nonzero polynomial
  divisor. Sparse logarithm and exponential assert `length <= p`; the sparse power
  asserts `length - e * v <= p` and the sparse root (odd characteristic)
  `length - v <= p`, only when the recurrence runs (see **Sparse entries**).
- Factorial tables: `maximum_index < p` for the constructor and for an `extend` that
  grows, every queried index within `maximum_index()`, `inverse_integer` of a positive index, and `binomial` checks `n` even when
  `k > n`.
- Batch inversion: every element nonzero. Integration: `m < p` for `m` input
  coefficients. Differentiation has no such bound.
- Square roots modulo a prime: a prime modulus and `value < prime`, also for zero.

Documented only: the coefficient type obeys the modular-integer laws. The concepts
check the interface and the primality of the modulus, not user-defined arithmetic.

## Measured performance

Both flag sets are measured; adoption decisions remain open. Every library choice below is a
provisional default and every contender is an `atl::experimental`
function named `<function>_by_<method>`. The recurrence/Newton crossover is provisionally 64 coefficients.

| Operation | Library default | Contenders (`<atl/experimental/nt/power_series_variants.hpp>`, `<atl/experimental/nt/square_root_mod_variants.hpp>`) |
|---|---|---|
| inverse | high-half Newton step reusing one transform | `series_inverse_by_recurrence`, `series_inverse_by_plain_newton`, `series_inverse_by_newton_driver` |
| division | inverse to half precision, fused last step | `series_division_by_inverse_product` |
| logarithm | fused division of the derivative by the series | `series_logarithm_by_inverse_product` |
| exponential | Newton carrying the reciprocal along | `series_exponential_by_recurrence`, `series_exponential_by_plain_newton`, `series_exponential_by_newton_driver` |
| power | valuation, then `exp(e log f)`; binary fallback | `series_power_by_binary_powering` |
| square root of a series | coupled Newton on the shared driver | `series_square_root_by_plain_newton`, `series_square_root_by_logarithm` |
| polynomial division | reversed inverse and a low product | `polynomial_division_by_schoolbook` |
| square root modulo a prime | Tonelli-Shanks | `square_root_mod_by_cipolla` |

Each contender has the template parameters, argument order, result and
preconditions of the function it races (the logarithm-based root additionally
asserts `n <= p` in odd characteristic). The sparse entries are raced against the
dense ones.

The sources are `benchmarks/nt/power_series.cpp` (lengths 2^10, 2^12, 2^16, 2^20 and
500000 for 998244353 and 2^16 for 1000000007; powers with exponents 2, 3, 32 and
10^18; polynomial shapes `(deg f, deg g)` = `(2n, n)`, `(n, n/2)`, `(n, 64)`; sparse
shapes `s` = 2, 8, 64) and `benchmarks/nt/square_root_mod.cpp` (998244353,
1000000007, the 62-bit prime `29 * 2^57 + 1` and seeded random 64-bit primes). The
quadratic contenders are timed only up to length 2^12. A winner needs x86-64
measurements under both plain `-O2` and `-march=native`.

Measurements (2026-11-23, AMD EPYC-Milan ccx23, GCC 15.2.0,
RelWithDebInfo, assertions off; plain `-O2` / `-O2 -march=native`, minimum
of five run means, 25 samples per row), in plain/native milliseconds:
At p=998244353, n=1024, library logarithm loses to inverse-product
logarithm (0.246/0.232 versus 0.231/0.217 ms). Library division also loses
(0.24/0.228 versus 0.225/0.212 ms), but the plain inverse-product division
row varies between runs (1.30x spread). These examples do not establish
one winner across all series operations.

## Pitfalls and notes

- A factorial table cannot reach index `p` (`p!` is zero), and integrating `p`
  coefficients would divide by zero. Trailing zeros count toward these bounds.
- A square root's sign is canonical (the smaller representative); negate a
  nonzero result for the other root.
- `polynomial_division` normalizes: compare against empty vectors, not zero vectors.
- Logarithm and exponential need `n <= p`, and the sparse power and the sparse root
  (in odd characteristic) need the recurrence length after the valuation shift to be
  at most `p`; inverse, division, `series_power` and the dense root have no such bound,
  so for a small prime compute a power through `series_power`, not through `exp(e log f)`.
- Products of moduli above 32 bits go through 16-bit limbs and cost about sixteen
  64-bit-exact convolutions; prefer a 32-bit prime when the problem allows.

## How it works

**Support.** Factorials accumulate upward; one inversion of the last factorial seeds
a downward pass for the inverse factorials, and multiplying adjacent entries gives
each integer inverse. Batch inversion stores the running prefix products, inverts
the total once and recovers each inverse going backwards. Integration uses
`1/i = -(p / i) * (1 / (p mod i))` for `1 < i < p`; the remainder is smaller than
`i`, so every needed inverse already exists. Tonelli-Shanks splits `p - 1` into an odd
part times a power of two and corrects the root with powers of a generator of the
2-power subgroup, each correction strictly lowering the order of the error;
Cipolla finds `a` with `a^2 - value` a non-residue and raises `a + u` to
`(p + 1) / 2` in `u^2 = a^2 - value`. Euler's criterion rejects non-residues first.

**Newton for the reciprocal.** The recurrence gives the first coefficients up to
the cutoff. A step from `m` to `k <= 2m` coefficients uses that `f g - 1` vanishes
below `m`: a cyclic product of size `2m` cannot alias in degrees `m..k-1`, so
extracting that interval and multiplying by `-g` gives exactly the new
coefficients. For transform-friendly moduli one transform of `g` serves both
products (three forward and two inverse transforms of size `2m`); other moduli use
two ordinary products.

**Division and logarithm.** Fused division inverts only to
`m = bit_floor(length - 1)`, then forms the low quotient, the high residual of
`f - g q_low` and the residual times the half-length inverse. The logarithm is the
integral of `f' / f` computed this way.

**Exponential.** The exponential `g` is seeded by its defining recurrence while
`h = 1/g` is maintained at the same precision. Each doubling extends `h`, forms
`log g` by derivative, product and integral, and takes `delta = f - log g` on the
new interval. Since `delta^2` vanishes at the new precision, `g` becomes
`g (1 + delta)` and `h` becomes `h (1 - delta)`.

**Driver.** `detail::series_newton_lift` doubles a state in place, calling
`step(state, middle, next)` with `next = min(length, 2 middle)`; each step writes
only `[middle, next)`. The square root and the driver contenders use it.

**Powers and roots.** The valuation and leading scalar are split off first. The unit
part of a power comes from `exp(e log f)`, or for the sparse entry from the equation
`f g' = e f' g`, i.e. `k g[k] = sum_i ((e + 1) i - k) f[i] g[k - i]`. The same
recurrence with `e = 1/2` gives the sparse root. The dense root keeps `y` and `1/y`:
it forms the high residual `(f - y^2) / 2`, multiplies it by the old reciprocal and
appends the result, which is Newton's `y <- (y + f / y) / 2` because the residual
vanishes below the old precision.

**Polynomial division.** Reverse both inputs, compute the quotient from the low
product with the reversed divisor's inverse, reverse it back, and subtract the low
quotient-times-divisor product to get the remainder. Small divisors or quotients
use schoolbook long division.

References by name: Newton iteration (von zur Gathen and Gerhard, *Modern Computer
Algebra*, chapter 9), and the Tonelli-Shanks and Cipolla square-root algorithms.

## See also

- [Modular arithmetic](modular-arithmetic.md)
- [Convolution and transforms](convolution.md)
- [Linear recurrences](linear-recurrence.md)
- [Polynomial points](polynomial-points.md): multipoint evaluation, interpolation and shifts.
- [Series composition](series-composition.md): composition, power projection and compositional inverse.
- [Polynomial gcd](polynomial-gcd.md): gcd, Pade approximants and modular inverses.
- [Online convolution](online-convolution.md): relaxed products.
