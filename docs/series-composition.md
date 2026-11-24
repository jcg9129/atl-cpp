# `atl::series_composition`, `atl::series_power_projection` and `atl::series_compositional_inverse`

Composition of formal power series over a prime field, with series as plain
`std::vector`s of coefficients in increasing degree order: `f(g(x))` modulo `x^n`, the
power projection `sum_j w_j [x^j] g(x)^i` for `i < count` (the transpose of
composition), and the compositional inverse `h` with `f(h(x)) = x`. Composition and
reversion take `O(M(n) log n)`, where the classical methods take `O(n^2)` products or
`O(sqrt(n) M(n))`. Reach for them when a problem substitutes one series into
another, counts through Lagrange inversion, or needs the functional inverse of a
series. For `exp`, `log`, powers and roots of a series use [Power series](power-series.md)
instead.

```cpp
#include <atl/nt/series_composition.hpp>
```

\include series-composition/quick.cpp

Expected stdout:
\include series-composition/quick.out

`<atl/nt/series_composition_native.hpp>` provides the default Horner and
Kinoshita–Li strategies for the native series coefficient domain described in
the power-series guide. `<atl/nt/series_composition_brent_kung.hpp>` installs
the separate Brent–Kung strategy used by the experimental comparison API.
`<atl/nt/series_composition.hpp>` includes both strategies and the wide-residue
product policy, preserving the full API and algorithm choices.

## Conventions and concepts

- Inputs are any `prime_mod_int_range` (vectors, arrays, spans, lists, views, even
  single-pass input ranges), read once through a const reference and never modified;
  paired ranges have the same coefficient type. Absent coefficients are zero and
  coefficients beyond an explicitly stated precision are ignored.
- Results are exactly the requested number of coefficients, trailing zeros included.
- Everything is `constexpr`. The free functions take a leading
  `ntt_engine Engine = default_ntt_engine`; an engine that does not support the modulus
  falls back to the convolution driver's tiers, and constant evaluation uses scalar
  transforms. Any static prime works, including the characteristic two and primes above 32
  bits (through limb products).
- `maximum_series_length` (`2^22`) bounds the lengths and counts; see
  [Power series](power-series.md).

## API

`Coefficient` is the common coefficient type; every function returns
`std::vector<Coefficient>`.

```cpp
template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Outer, prime_mod_int_range Inner>
    requires std::same_as<series_coefficient_t<Outer>, series_coefficient_t<Inner>>
constexpr auto series_composition(Outer const &outer, Inner const &inner, std::size_t length)
    -> std::vector<series_coefficient_t<Outer>>;

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Weights, prime_mod_int_range Series>
    requires std::same_as<series_coefficient_t<Weights>, series_coefficient_t<Series>>
constexpr auto series_power_projection(Weights const &weights, Series const &series, std::size_t count)
    -> std::vector<series_coefficient_t<Weights>>;

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Series>
constexpr auto series_compositional_inverse(Series const &series, std::size_t length)
    -> std::vector<series_coefficient_t<Series>>;
```

| Function | Meaning |
|---|---|
| `series_composition(outer, inner, length)` | `outer(inner(x))` modulo `x^length`; the **whole outer range is a finite polynomial** |
| `series_power_projection(weights, series, count)` | `b_i = sum_j weights[j] [x^j] series(x)^i` for `i < count`, with precision `n = weights.size()` |
| `series_compositional_inverse(series, length)` | the unique `g` with `g[0] = 0` and `f(g) = x` modulo `x^length` |

Composition allows `inner[0] != 0`: the finite outer polynomial is Taylor-shifted by
`inner[0]` (see [Polynomial points](polynomial-points.md)) and the series `inner - inner[0]`
is substituted; truncating the outer before shifting would be wrong. A constant
inner evaluates the outer by Horner's rule; a single nonzero term `c x^v` scatters
`outer[i] c^i` to degree `i v`, and with valuation `v > 0` only the first
`ceil(length / v)` shifted outer coefficients contribute. An empty outer gives zeros.

The projection places no restriction on the constant term of the series or on the
characteristic. `count = 0` gives the empty vector, empty weights give `count` zeros,
and `count` may be smaller or larger than `n`. The series is read modulo `x^n`.

Reversion needs `f[0] == 0` for a positive length and `f[1] != 0` from length two.
Precision zero gives the empty vector without looking at `f`; precision one gives `{0}`
without checking `f[1]`. There is no `length <= p` restriction: Lagrange inversion runs
when `length <= p`, and Newton's method with one composition per step otherwise
(including `p = 2`).

## Complexity

For outer length `m`, precision `n` and `M(n)` the cost of a length-`n` product:

| Function | Time | Memory |
|---|---|---|
| `series_composition` | `O(M(m) log(m + 1) + M(n) log(n + 1))` | `O(m + n log(n + 1))` |
| `series_power_projection` | `O(M(n) log(n + 1) + M(count))`; `O(count M(n))` for `count <= 8` | `O(n + count)` |
| `series_compositional_inverse` | `O(M(n) log(n + 1))` | `O(n log(n + 1))` |

The outer term of the composition bound covers a possible Taylor shift. Packed products
beyond the series cap are blocked into operands of half the cap. `M` depends on the
modulus: a transform-friendly prime gives `O(n log n)`; other primes use the convolution
fallbacks, and constant evaluation can be quadratic. Allocation failures propagate.

## Preconditions

Asserted with `atl_assert` (active only under `-DATL_ASSERT`):

- The whole outer length, the weight length, `length` and `count` are at most
  `maximum_series_length`. There is no joint bound on the outer length times the
  precision: every working array of a composition at precision `n` stays below `8n`
  coefficients whatever the outer length, so the per-argument caps are the whole contract.
- Reversion: `f[0] == 0` for a positive length, and `f[1] != 0` from length two.

Documented only: the coefficient type obeys the modular-integer laws.

## Measured performance

Not measured yet. Every library choice is a provisional default and every contender is an
`atl::experimental` function named `<function>_by_<method>`; the races are pending.
The provisional cutoffs are 8 for tiny schoolbook composition, 16 for the outer length
below which fast-product Horner replaces the transposed projection, and 8 for the count
up to which the projection runs repeated products; they are starting choices, not
measured winners.

| Operation | Library default | Contenders (`<atl/experimental/nt/series_composition_variants.hpp>`) |
|---|---|---|
| composition | transpose of the optimized projection (Kinoshita-Li); Horner for tiny shapes | `series_composition_by_brent_kung`, `series_composition_by_fast_horner`, `series_composition_by_schoolbook_horner`, `series_composition_by_cutoffs` |
| power projection | bivariate Bostan-Mori with paired spectra; repeated products for small counts | `series_power_projection_by_padded_bostan_mori`, `series_power_projection_by_bostan_mori`, `series_power_projection_by_repeated_products` |
| reversion | Lagrange inversion through one projection and a normalized power; Newton beyond `p` | `series_compositional_inverse_by_composition_newton`, `series_compositional_inverse_by_projection_lagrange`, `series_compositional_inverse_by_baby_giant_lagrange`, `series_compositional_inverse_by_coefficient_recurrence` |

Each contender has the template parameters, argument order, result and preconditions of the
function it races (the Lagrange contenders additionally require `length <= p`).
`series_composition_by_cutoffs(outer, inner, length, horner_cutoff, schoolbook_cutoff)`
exposes both Horner crossovers for the sweep; both must be positive and below the series cap.
Brent-Kung and the baby-step/giant-step Lagrange take `O(sqrt(n) M(n) + n^2)` time (for
equal sizes) and `O(n sqrt(n))` storage. The direct reversion recurrence is cubic, and
schoolbook Horner is cubic for two dense inputs of length `n`.

The source is `benchmarks/nt/series_composition.cpp`. It checks every contender against the
library result (and composition against projection through the transposition identity)
before timing, at `n` = 32, 64, 2^10, 2^12, 2^13, 2^14, 2^16 and 250000 modulo 998244353 and
2^13 modulo 1000000007. Brent-Kung and the baby-step/giant-step Lagrange stop at 2^14, the
quadratic contenders (fast Horner, repeated products) at 2^12, and the cubic ones
(schoolbook Horner, the recurrence) at 64; a sweep times the cutoff contenders at sizes
`c - 1`, `c` and `c + 1` for `c` = 4 to 64. A winner needs x86-64 records under both plain
`-O2` and `-march=native`.

## Pitfalls and notes

- The outer range is the whole polynomial, never truncated to `length` before use; pass
  only the coefficients you mean.
- Reversion of a series with `f[1] == 0` has no solution; the check is active only under
  `-DATL_ASSERT`.
- The projection's precision is the weight length, not `count`: to read `[x^j] g^i` for
  `j < n` pass `n` weights.
- Composition and projection are transposes of each other:
  `sum_i weights[i] * series_composition(outer, g, n)[i]` equals
  `sum_i outer[i] * series_power_projection(weights, g, n)[i]` when `g[0] = 0`.
- Products of moduli above 32 bits go through 16-bit limbs and cost about sixteen 64-bit-exact
  convolutions; prefer a 32-bit prime when the problem allows.

## How it works

**Projection.** Write `P(x, y) = reverse_n(weights)(x)` and `Q(x, y) = 1 - y g(x)`. The
required generating series is `[x^(n - 1)] P / Q`. Multiplying numerator and denominator by
`Q(-x, y)` makes the denominator even in `x`, and keeping only the parity of `n - 1` in the
numerator halves the `x` precision while doubling the `y` degree; the `y` direction is always
truncated to `count`. At width one an ordinary series division returns the moments. This is
the Graeffe step of Kinoshita and Li (2024), Algorithm 1, implemented independently from
the rational identity.

The packed univariate exponent is `x_degree + 2 width y_degree`; an empty `x` block between
`y` rows keeps products from carrying into the next row. On a supported transform size `N`,
`Q(-x, y)` is obtained by swapping adjacent bit-reversed spectral entries, so it costs no
transform. A halving uses two `N`-point forwards and two `N/2`-point inverses; when an
operand fits in `N/2` coefficients its forward is two `N/2`-point forwards instead (the
first at the even roots, the second on coefficients twisted by a primitive `N`-th root for
the odd roots), so a typical halving is four half-size forwards and two half-size inverses.
Odd-parity extraction scales by inverse roots generated multiplicatively in bit-reversed
order, with no per-entry inversions. Other fields use the same algebra with ordinary exact
products and no division by two. The plain padded contender uses two products per step.

**Composition.** Composition is the transpose of projection in the weights, so it runs the
projection's steps backwards. The denominator chain does not depend on the weights, so each
level's spectrum is kept. The terminal division transposes to a reversed low product with
`1 / Q`, each parity selection to inserting zeros, each multiplication by `Q(-x, y)` to a
middle product that reuses the stored reflected-denominator spectrum, and the initial weight
reversal to the final coefficient reversal. Nothing is cached globally; all spectra belong to
the call.

**Reversion.** For `d = length - 1`, Lagrange gives
`[x^d] f(x)^i = (i / d) [t^(d - i)] (t / g(t))^d`. A projection with the single weight at
degree `d` returns these moments; reversing and scaling them recovers `(t / g)^d`, whose
constant is normalized, raised to the power `-1 / d` in the field (valid because `d < p`) and
scaled by `1 / f[1]`. Beyond `p` Newton's method computes `h = f(g)` once per step and uses
`f'(g) = h' / g'` to avoid a second composition; the residual vanishes below the old
precision, so multiplying it by that quotient appends exactly the new coefficients.

**Contenders.** Brent and Kung (1978), first algorithm: baby powers `g^i` for `i` up to
`sqrt(m)`, a classical rectangular matrix product, and giant steps by `g^sqrt(m)`. Johansson
(2015), fast Lagrange inversion: baby and giant powers of `h = x / f` produce only the
needed diagonal coefficient `[x^(k - 1)] h^k` for each `k`.

## See also

- [Power series](power-series.md): products, inverses, powers and the Newton driver used here.
- [Polynomial points](polynomial-points.md): the Taylor shift that composition uses.
- [Convolution and transforms](convolution.md)
- [Modular arithmetic](modular-arithmetic.md)
