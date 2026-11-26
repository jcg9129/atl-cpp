# `atl::polynomial_point_tree`, `atl::polynomial_evaluation`, `atl::polynomial_interpolation` and polynomial shifts

Operations on a polynomial and many points at once, over a prime field, with the
polynomial as a plain `std::vector` of coefficients in increasing degree order:
multipoint evaluation, interpolation, products of many polynomials, the polynomial
with given roots, the Taylor shift `f(x + c)`, evaluation along a geometric
progression (chirp-z) and the shift of a polynomial's samples `f(0), ..., f(n - 1)` to
`f(c), ..., f(c + k - 1)`. Every operation costs a few products instead of the
`O(n m)` of Horner's rule at each point. Reach for them when a problem asks for
`f` at `n` points, the polynomial through `n` points, `f(x + c)` or the product of
many small polynomials. When the same points serve many polynomials, build one
`polynomial_point_tree`. Skip this page for exact rational or floating-point
interpolation: the concepts reject every coefficient type except a prime modulus.

```cpp
#include <atl/nt/polynomial_points.hpp>
```

\include polynomial-points/quick.cpp

Expected stdout:
\include polynomial-points/quick.out

`<atl/nt/polynomial_points.hpp>` is the normal entry point, including the
wide-residue product policy. The examples use the reduced alternative
`<atl/nt/polynomial_points_native.hpp>` for the native series coefficient domain
described in the power-series guide.

## Conventions and concepts

- Coefficients are ordered by increasing degree. Inputs are any
  `prime_mod_int_range` (vectors, arrays, spans, lists, views, even single-pass input
  ranges), read once through a const reference and never modified; paired ranges have
  the same coefficient type. `polynomial_product` takes a const input range whose
  elements are themselves such ranges.
- Everything is `constexpr`. The free functions take a leading
  `ntt_engine Engine = default_ntt_engine`; an engine that does not support the
  modulus falls back to the convolution driver's tiers, and constant evaluation uses
  scalar transforms. Any static prime works, including primes above 32 bits (through
  limb products) and the characteristic two; moduli with a transform-friendly
  structure are faster.
- The output-length rule is deliberate. Fixed lengths keep trailing zeros: the Taylor
  shift returns the input length, interpolation returns exactly `n` coefficients (empty
  for `n = 0`), `polynomial_from_roots` returns `n + 1` monic coefficients (`{1}` for no
  roots), evaluation returns one value per point, and the geometric evaluation and the
  sample shift return exactly `count` values. Only `polynomial_product`, whose length
  depends on the values, is normalized: trailing zeros are stripped, an empty list gives
  `{1}` and a zero factor gives the empty vector.
- `maximum_series_length` (`2^22`) bounds every input length and `count`; see
  [Power series](power-series.md).

## API

`Coefficient` is the common coefficient type and `M(n)` the cost of a length-`n`
product.

```cpp
template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial>
constexpr auto polynomial_taylor_shift(Polynomial const &polynomial, series_coefficient_t<Polynomial> shift);

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Roots>
constexpr auto polynomial_from_roots(Roots const &roots);

template<ntt_engine Engine = default_ntt_engine, detail::points_polynomial_range Polynomials>
constexpr auto polynomial_product(Polynomials const &polynomials);

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial, prime_mod_int_range Points>
    requires std::same_as<series_coefficient_t<Polynomial>, series_coefficient_t<Points>>
constexpr auto polynomial_evaluation(Polynomial const &polynomial, Points const &points);

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Points, prime_mod_int_range Values>
    requires std::same_as<series_coefficient_t<Points>, series_coefficient_t<Values>>
constexpr auto polynomial_interpolation(Points const &points, Values const &values);

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial>
constexpr auto polynomial_geometric_evaluation(Polynomial const &polynomial, Coefficient first, Coefficient ratio, std::size_t count);

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Samples>
constexpr auto sample_point_shift(Samples const &samples, Coefficient shift, std::size_t count);

template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
class polynomial_point_tree {
public:
    template<prime_mod_int_range Points>
        requires std::same_as<Coefficient, series_coefficient_t<Points>>
    constexpr explicit polynomial_point_tree(Points const &points);

    constexpr std::size_t point_count() const;
    constexpr std::span<Coefficient const> points() const &;
    std::span<Coefficient const> points() const && = delete;

    template<prime_mod_int_range Polynomial>
        requires std::same_as<Coefficient, series_coefficient_t<Polynomial>>
    constexpr std::vector<Coefficient> evaluate(Polynomial const &polynomial) const;

    template<prime_mod_int_range Values>
        requires std::same_as<Coefficient, series_coefficient_t<Values>>
    constexpr std::vector<Coefficient> interpolate(Values const &values) const;
};
```

Every free function returns `std::vector<Coefficient>`.

| Function | Result | Time |
|---|---|---|
| `polynomial_taylor_shift(f, c)` | coefficients of `f(x + c)`, input length | `O(M(n))`; `O(M(n) log n)` when `n > p` |
| `polynomial_from_roots(roots)` | monic product of `x - root` | `O(M(n) log n)` |
| `polynomial_product(factors)` | normalized product of all factors | `O(M(d) log k)` for `k` factors of total degree `d` |
| `polynomial_evaluation(f, points)` | `f` at each point, duplicates allowed | `O(M(n) log n + M(d))` |
| `polynomial_interpolation(points, values)` | the unique polynomial of degree below `n` | `O(M(n) log n)` |
| `polynomial_geometric_evaluation(f, first, ratio, count)` | `f(first * ratio^j)` for `j < count` | `O(M(d + count))` |
| `sample_point_shift(samples, shift, count)` | `f(shift + j)` for `j < count`, from `f(0), ..., f(n - 1)` | `O(M(n + count))` |

Here `n` is the number of roots, points or samples, and `d` the polynomial length.
`polynomial_point_tree` owns its points, their product tree and the transforms of every
child product. Building it costs `O(M(n) log n)` time and `O(n log n)` storage;
each `evaluate` costs `O(M(n) log n + M(d))` and each `interpolate` `O(M(n) log n)`.
The free evaluation and interpolation build such a tree on every call, so reuse one
tree when the points stay fixed. A polynomial longer than the point count is reduced
modulo the root product first; a shorter one is padded internally. `points()` views the
stored points exactly as given (caller order, duplicates included, no padding); the view
lives as long as the tree and is not available on a temporary tree.

\include polynomial-points/sampling.cpp

Expected stdout:
\include polynomial-points/sampling.out

Zero `first` and zero `ratio` of the geometric evaluation are handled directly, so
the chirp-z method only inverts a nonzero ratio. In `sample_point_shift` a target that
equals a source node, also after wrapping modulo `p`, returns the known sample, and
`count` may exceed `p` (the sequence repeats).

## Complexity

| Operation | Time | Memory |
|---|---|---|
| Taylor shift | `O(M(n))`, `O(M(n) log n)` beyond `p` | `O(n)` |
| from roots | `O(M(n) log n)` | `O(n)` |
| product of `k` polynomials of total degree `d` | `O(M(d) log k)` | `O(d)` |
| tree build | `O(M(n) log n)` | `O(n log n)` |
| tree `evaluate` / free evaluation | `O(M(n) log n + M(d))` | `O(n log n + d)` |
| tree `interpolate` / free interpolation | `O(M(n) log n)` | `O(n log n)` |
| geometric evaluation | `O(M(d + count))` | `O(d + count)` |
| sample shift | `O(M(n + count))` | `O(n + count)` |

`M` depends on the modulus: with a transform-friendly prime it is `O(n log n)`; other
primes go through the convolution fallbacks, and constant evaluation can be quadratic.
Allocation failures propagate.

## Preconditions

Asserted with `atl_assert` (active only under `-DATL_ASSERT`):

- Every whole input length and every `count` is at most `maximum_series_length`.
  `polynomial_from_roots` additionally needs fewer than `maximum_series_length` roots,
  and a nonzero `polynomial_product` a total degree below it.
- Interpolation: equal point and value counts, and distinct points, detected as a zero
  value of the derivative of the root product at a point (a repeated point gives zero).
  This also holds for `polynomial_point_tree::interpolate`.
- `sample_point_shift`: `n <= p`, also for `count = 0`.

Documented only: the degree of the sampled polynomial is below `n` (no check can see
it), and the coefficient type obeys the modular-integer laws. There is no `n <= p`
restriction for the Taylor shift or the evaluation. Evaluation allows repeated points.

## Measured performance

Both flag sets are measured; adoption decisions remain open. Every library choice is a
provisional default and every contender is an `atl::experimental`
function named `<function>_by_<method>`.
The provisional schoolbook leaf size is 32 points; the contender
`polynomial_evaluation_by_leaf_cutoff` sweeps it over 16, 24, 32, 48, 64, 96 and 128.

| Operation | Library default | Contenders (`<atl/experimental/nt/polynomial_points_variants.hpp>`) |
|---|---|---|
| Taylor shift | factorial-scaled convolution; divide and conquer beyond `p` | `polynomial_taylor_shift_by_divide_and_conquer`, `polynomial_taylor_shift_by_horner` |
| from roots | balanced products, schoolbook leaves | `polynomial_from_roots_by_sequential_product` |
| product of polynomials | balanced tree | `polynomial_product_by_smallest_first`, `polynomial_product_by_sequential_schoolbook` |
| evaluation | transposed middle-product descent, Horner leaves | `polynomial_evaluation_by_remainder_tree`, `polynomial_evaluation_by_horner`, `polynomial_evaluation_by_leaf_cutoff` |
| interpolation | derivative weights, batch inverse, upward combination; Newton base case | `polynomial_interpolation_by_remainder_tree`, `polynomial_interpolation_by_newton_differences` |
| geometric evaluation | chirp-z with a middle product | `polynomial_geometric_evaluation_by_horner` |
| sample shift | factorial-weighted barycentric middle product | `sample_point_shift_by_falling_factorials`, `sample_point_shift_by_direct_barycentric` |

Each contender has the template parameters, argument order, result and preconditions of
the function it races, and tests check all of them against independent oracles
(scalar Horner, schoolbook products, Pascal-triangle shifts and Lagrange interpolation).
The quadratic contenders take `O(n^2)`, or `O(n m)` for evaluation and sample shift.
`sample_point_shift_by_falling_factorials` uses three convolutions and repeats one field
period when `count > p`.

The source is `benchmarks/nt/polynomial_points.cpp`. It checks every contender against the
library result before timing, at `n = m` = 2^10, 2^12, 2^16, 2^17 and 500000 modulo
998244353 and 2^16 modulo 1000000007; the quadratic contenders run only up to 2^12.
Other shapes: unequal factors of degree 1 to 17 with total degree `n`, and a tree built
once and used for evaluation and interpolation. A winner needs x86-64 measurements under both
plain `-O2` and `-march=native`.

Measurements (2026-11-23, AMD EPYC-Milan ccx23, GCC 15.2.0,
RelWithDebInfo, assertions off; plain `-O2` / `-O2 -march=native`, minimum
of five run means, 25 samples per row), in plain/native milliseconds:
At p=998244353, n=m=1024, library evaluation loses to leaf cutoff 16
(0.68/0.703 versus 0.646/0.662 ms). The library polynomial product loses
to smallest-first products (0.272/0.284 versus 0.265/0.277 ms); its plain
row varies between runs (1.38x spread), so this minimum alone does not
establish a stable ranking.

## Pitfalls and notes

- Trailing zeros count toward lengths and the length cap: `polynomial_interpolation`
  returns `n` coefficients even when the polynomial has lower degree.
- `polynomial_interpolation` needs distinct points (checked only under `-DATL_ASSERT`),
  hence `n <= p`. Evaluation allows repeated points.
- `sample_point_shift` treats the samples as those of a polynomial of degree below `n`, and
  needs `n <= p`: the nodes `0, ..., n - 1` must be distinct modulo `p`.
- Compare against the empty vector, not a zero vector: `polynomial_product` of a list
  containing the zero polynomial is empty.
- The Taylor shift dispatches on the coefficient count `n` (the input length, trailing
  zeros included): `n <= p` takes the factorial-scaled convolution, which needs `1 / i!`
  for every `i < n`; `n > p` (small `p`) takes the slower divide and conquer path,
  since `p!` is not invertible. Trimming trailing zeros can move an input to the faster
  path.
- For many polynomials at the same points, build the tree once; the free functions rebuild
  it on every call.
- Products modulo a prime above 32 bits go through 16-bit limbs and cost about sixteen
  64-bit-exact convolutions; prefer a 32-bit prime when the problem allows.

## How it works

**Product tree.** Each node stores the product of `1 - x_i t` over its points (the
reversed monic product), so a node of `k` points has `k + 1` coefficients and constant
term one. Nodes with at most 32 points are products of linear factors by schoolbook.
At a larger node, when the field supports a cyclic transform of length `bit_ceil(k)`, the
tree keeps the transforms of both child products: construction needs two forward
transforms and one inverse, and when `k` is a power of two the single wrapped top
coefficient is restored from the children and the known constant one. Other fields use
ordinary products and keep no spectra.

**Evaluation.** For a polynomial of length at most `n`, the initial state is the reversed
polynomial divided by the root product modulo `t^n`. A child state is the middle interval
of its parent's state times the sibling's reversed product, which is the transposed
product (Tellegen's principle applied to the remainder tree); at a leaf, multiplying back
by the leaf product and reversing recovers the local remainder for Horner's rule. This
derivation works with repeated points and with zero points. Each descent step uses one
forward and two inverse transforms, reusing the stored spectra. A longer polynomial is
reduced at the root by polynomial division first. The contender
`polynomial_evaluation_by_remainder_tree` divides by each child product instead.

**Interpolation.** The weights are `v_i / P'(x_i)`, with `P'` evaluated at the points by the
same descent and the inverses taken with one batch inversion; the combination goes up the
tree as `F_L P_R + F_R P_L`, in reversed coordinates through the same stored spectra (two
forward transforms and one inverse per node). Small point counts use Newton divided
differences after the derivative check.

**Middle product.** The coefficients `[b - 1, a)` of an `a`-by-`b` product with `a >= b`
are all the chirp-z and sample-shift steps need. A cyclic transform padded only to
`bit_ceil(a)` cannot alias into that interval, so the transform is never larger than
needed; fields without such a transform split both operands into blocks within that bound
and accumulate only intersecting block products.

**Taylor shift.** For `n <= p` coefficients: `f(x + c) = sum_j x^j / j! * sum_i f_i i! c^(i - j) / (i - j)!`,
a convolution of the factorial-scaled reversed polynomial with the powers of `c`. For
`n > p` the polynomial is split in halves, shifted recursively (quadratic shifts at the
leaves), and combined as `lower + upper * (x + c)^m` with the power computed by
repeated squaring.

**Chirp-z.** With `T(k) = k (k - 1) / 2`, `r^(i j) = r^T(i + j) / (r^T(i) r^T(j))`, with the
exponent powers generated by successive multiplication (no square root and no integer
exponent that could overflow), so `f(first * r^j)` is a middle product of a chirp with the
scaled coefficients, then scaled again. Requests whose working lengths exceed the series
cap are processed in blocks.

**Sample shift.** The factorials give the barycentric weights
`w_i = (-1)^(n - 1 - i) f(i) / (i! (n - 1 - i)!)`. A middle product computes
`sum_i w_i / (c + j - i)` for all targets and a product of the `n` factors
`c + j - i` rescales it. Every zero denominator is replaced by one before the batch
inversion, prefix products and inverse prefixes form the sliding products, and targets
whose residue is below `n` are overwritten by the known sample, so neither a collision nor
a wrap past `p` divides by zero. For `n = p` every target is already a node. The
three-convolution contender converts samples to Newton differences, translates the falling
factorials by Vandermonde's identity and converts back, computing at most one field period.

**Products of many polynomials.** Normalize the factors (strip trailing zeros), return at
once on a zero factor, and multiply in a balanced tree so each level costs `O(M(d))`.

References by name: the product tree, remainder tree and transposed evaluation (von zur
Gathen and Gerhard, *Modern Computer Algebra*, chapter 10; Bostan, Lecerf and Schost,
*Tellegen's principle into practice*, for the transposed algorithms), and the chirp-z
transform.

## See also

- [Power series](power-series.md): the products, inverses and polynomial division used here.
- [Series composition](series-composition.md): composition, power projection and compositional inverse.
- [Convolution and transforms](convolution.md)
- [Modular arithmetic](modular-arithmetic.md)
