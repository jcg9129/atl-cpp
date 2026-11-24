# `atl::polynomial_gcd`, `atl::polynomial_extended_gcd`, `atl::polynomial_inverse_mod` and `atl::pade_approximant`

The Euclidean algorithm for polynomials over a prime field, with polynomials as plain
`std::vector`s of coefficients in increasing degree order: the monic gcd, the gcd with its
Bezout cofactors, the inverse of a polynomial modulo another, and the Pade approximant
(the rational function `P / Q` with bounded degrees that matches a power series to the
precision those degrees allow). All run in `O(M(n) log n)` field work through a recursive
half-gcd instead of the `O(n^2)` of the plain algorithm. Reach for them for gcds of long
polynomials, modular inverses in `F[x] / (m)`, and recovering a rational generating
function from series terms. For a short recurrence from few terms the quadratic
`berlekamp_massey` of [Linear recurrences](linear-recurrence.md) is simpler; division with
remainder and series inverses are on the [Power series](power-series.md) page.

```cpp
#include <atl/nt/polynomial_gcd.hpp>
```

\include polynomial-gcd/quick.cpp

Expected stdout:
\include polynomial-gcd/quick.out

`<atl/nt/polynomial_gcd_native.hpp>` provides the same API for the native
series coefficient domain described in the power-series guide.
`<atl/nt/polynomial_gcd.hpp>` also installs the wide-residue product policy.

## Conventions and concepts

- Inputs are any `prime_mod_int_range` (vectors, arrays, spans, lists, views, even
  single-pass input ranges), read once through a const reference and never modified;
  paired ranges have the same coefficient type. Trailing zeros are ignored: a polynomial
  is its normalized form.
- Every polynomial result is normalized: the zero polynomial is the empty vector and
  nonzero polynomials end with a nonzero coefficient. Gcds are monic.
- Everything is `constexpr`. The free functions take a leading
  `ntt_engine Engine = default_ntt_engine`; an engine that does not support the modulus
  falls back to the convolution driver's tiers, and constant evaluation uses scalar
  transforms. Any static prime works, including the characteristic two and primes above 32
  bits (through limb products); there is no bound by the characteristic.
- `maximum_series_length` (`2^22`) bounds every normalized input length; see
  [Power series](power-series.md).

## API

`Coefficient` is the common coefficient type.

```cpp
template<static_prime_mod_int Coefficient>
struct polynomial_extended_gcd_result {
    std::vector<Coefficient> gcd;
    std::vector<Coefficient> first_cofactor;
    std::vector<Coefficient> second_cofactor;
    // Member-wise comparison, not equivalence: other valid cofactors compare unequal.
    friend bool operator==(polynomial_extended_gcd_result const &, polynomial_extended_gcd_result const &) = default;
};

template<static_prime_mod_int Coefficient>
struct rational_function {
    std::vector<Coefficient> numerator;
    std::vector<Coefficient> denominator;
};

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
    requires std::same_as<series_coefficient_t<First>, series_coefficient_t<Second>>
constexpr auto polynomial_gcd(First const &first, Second const &second)
    -> std::vector<series_coefficient_t<First>>;

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
    requires std::same_as<series_coefficient_t<First>, series_coefficient_t<Second>>
constexpr auto polynomial_extended_gcd(First const &first, Second const &second)
    -> polynomial_extended_gcd_result<series_coefficient_t<First>>;

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Value, prime_mod_int_range Modulus>
    requires std::same_as<series_coefficient_t<Value>, series_coefficient_t<Modulus>>
constexpr auto polynomial_inverse_mod(Value const &value, Modulus const &modulus)
    -> std::optional<std::vector<series_coefficient_t<Value>>>;

template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
constexpr auto pade_approximant(Range const &series, std::size_t numerator_degree, std::size_t denominator_degree)
    -> std::optional<rational_function<series_coefficient_t<Range>>>;
```

| Function | Result |
|---|---|
| `polynomial_gcd(first, second)` | the monic gcd; empty for two zeros, the monic form of the other input for one zero |
| `polynomial_extended_gcd(first, second)` | `gcd`, `first_cofactor`, `second_cofactor` with `first * first_cofactor + second * second_cofactor == gcd` |
| `polynomial_inverse_mod(value, modulus)` | the inverse of `value` modulo `modulus`, of degree below the modulus degree, or `nullopt` when their gcd is not one (also for a zero residue) |
| `pade_approximant(series, m, d)` | `P / Q` in lowest terms with `deg P <= m`, `deg Q <= d`, `Q[0] = 1` and `P == Q * series` modulo `x^(m + d + 1)`, or `nullopt` when no such `Q` exists |

**Cofactors.** With `g` the gcd of degree `e`, the cofactor of each input has degree
strictly below the degree of the other input minus `e`, whenever that difference is
positive. When it is zero, one input divides the other and only the divisor has a nonzero
cofactor, the constant `1 / lc(g)`; for equal degrees that is the second input. `(f, 0)`
gives `(f / lc(f), {1 / lc(f)}, {})`, `(0, f)` the same with the cofactors interchanged
and `(0, 0)` three empty polynomials. The gcd-only function does not accumulate cofactors,
and `polynomial_inverse_mod` accumulates only the cofactor of the value, after reducing it
modulo the modulus.

**Pade.** Only the first `m + d + 1` series coefficients are read; absent ones are zero.
The approximant is unique when it exists, and a nonzero `Q[0]` certifies lowest terms,
since a common factor would divide `x^(m + d + 1)`. It can be missing even for a series
that a rational function of larger degrees matches (the second example below).

\include polynomial-gcd/pade.cpp

Expected stdout:
\include polynomial-gcd/pade.out

## Complexity

| Operation | Time | Memory |
|---|---|---|
| `polynomial_gcd`, `polynomial_extended_gcd` | `O(M(n) log n)` | `O(n)` |
| `polynomial_inverse_mod` | `O(M(n) log n)` | `O(n)` |
| `pade_approximant` | `O(M(n) log n)` for `n = m + d + 1` | `O(n)` |

`n` bounds the normalized input lengths. `M` depends on the modulus: with a
transform-friendly prime it is `O(n log n)`; other primes go through the convolution
fallbacks, and constant evaluation can be quadratic. The plain Euclidean algorithm is
`O(n^2)`, also with fast division (many small degree drops are the worst case). Allocation
failures propagate.

## Preconditions

Asserted with `atl_assert` (active only under `-DATL_ASSERT`):

- Every normalized input length is at most `maximum_series_length` (trailing zeros may
  exceed it).
- `polynomial_inverse_mod`: the modulus has degree at least one.
- `pade_approximant`: `m + d + 1 <= maximum_series_length`, checked without overflow. A
  precision exactly at the cap is supported.

Documented only: the coefficient type obeys the modular-integer laws; concepts check the
interface and the prime modulus, not the arithmetic.

## Measured performance

Not measured yet. Every library choice is a provisional default and every contender is an
`atl::experimental` function named `<function>_by_<method>`; the races are pending. The
provisional Euclidean leaf size of the half-gcd is 64; `polynomial_gcd_by_leaf_cutoff_plain_products`,
`polynomial_gcd_by_leaf_cutoff_fused_products` and their `polynomial_extended_gcd_by_...`
counterparts sweep it over 32, 48, 64, 96, 128, 192 and 256, with ordinary or with fused
matrix products.

| Operation | Library default | Contenders (`<atl/experimental/nt/polynomial_gcd_variants.hpp>`) |
|---|---|---|
| gcd, extended gcd | recursive half-gcd, ordinary matrix products | `polynomial_gcd_by_fused_matrix_products`, `polynomial_gcd_by_fast_division_euclid`, `polynomial_gcd_by_schoolbook_euclid`, `polynomial_gcd_by_leaf_cutoff_plain_products`, `polynomial_gcd_by_leaf_cutoff_fused_products`, and the `polynomial_extended_gcd_by_...` counterparts |
| Pade | half-gcd stopped at the degree bound | `pade_approximant_by_quadratic_euclid` |
| shortest recurrence (experimental only) | none; the library `berlekamp_massey` is quadratic | `berlekamp_massey_by_half_gcd` |

Each contender has the template parameters, argument order, result and preconditions of the
function it races, and the tests check all of them against an independent Euclidean oracle,
Bezout identities and, over tiny fields, an exhaustive search for Pade denominators. The
fused contender caches the spectrum of each matrix entry and sums the two products of an
output entry before its inverse transform; the Euclid contenders take `O(n^2)`.
`berlekamp_massey_by_half_gcd` runs half-gcd on the reversed sequence; as with
`berlekamp_massey`, a prefix too short to determine the recurrence can select a different
shortest recurrence.

The source is `benchmarks/nt/polynomial_gcd.cpp`. It checks every contender before timing,
at `n` = 2^10, 2^12, 2^14, 2^16, 2^17 and 500000 modulo 998244353 and 2^16 modulo
1000000007, with coprime inputs and inputs with a planted common factor of degree `n / 3`;
the quadratic contenders run only up to 2^12. The recurrence race uses lengths 2^10, 2^14
and 2^16 (the library `berlekamp_massey` up to 2^14) and the leaf-size sweep lengths 256,
1024 and 4096. A winner needs x86-64 records under both plain `-O2` and `-march=native`.

## Pitfalls and notes

- Compare against the empty vector for zero, not a vector of zeros: results are normalized.
- A gcd that is `{1}` means coprime; any longer gcd means a common factor. For
  `polynomial_inverse_mod` this is the `nullopt` case, including a value that is a
  multiple of the modulus.
- The cofactors are the minimal ones, not an arbitrary Bezout pair; their degree bounds
  and the divisibility convention above are part of the contract.
- A Pade numerator may be shorter than `m + 1` and the denominator shorter than `d + 1`;
  trailing zeros are stripped. `nullopt` means no denominator of degree at most `d` makes the
  product agree through degree `m + d`; it does not mean the series is not rational.
- Inputs shorter than the Euclidean leaf size are reduced by plain Euclidean divisions, so
  short polynomials pay no half-gcd overhead.

## How it works

**Half-gcd.** The transition matrix `T` of a sequence of Euclidean steps is a two-by-two
matrix of polynomials with `T (a, b) = (c, d)` for the remainders `c, d` reached. The
recursion follows the degree invariant of von zur Gathen and Gerhard (*Modern Computer
Algebra*, chapter 11). With `h = ceil(deg a / 2)`, it first reduces `(a div x^h, b div x^h)`
recursively, applies that transition to the full inputs, and checks the actual degree of
the second polynomial. If it is still too large, one exact division crosses the boundary;
the next high shift is `2h - deg(second)` computed from that actual degree, before the
second recursion and the composition of the two matrices. A quotient of degree above one
and a zero remainder therefore exit or shift at their real degree, without assuming a unit
degree drop. Below the leaf size, ordinary Euclidean divisions are used. The top level
alternates a half-gcd step with one exact division until the second remainder reaches the
requested degree bound (below one coefficient for a gcd).

**Extended gcd and inverse.** Zero columns of the matrix are never accumulated for the gcd;
one column tracks only the second input (modular inverse, Pade); two track both. Every
division step updates the columns with the quotient, every half-gcd by one matrix product.
The result is scaled by the inverse of the last remainder's leading coefficient.

**Pade.** The extended Euclid of `x^(m + d + 1)` and the series, stopped at the first
remainder of degree at most `m`, gives `P` as that remainder and `Q` as the series cofactor.
A zero or too long `Q` rejects the candidate. The initial `x^(m + d + 1)` can have length
`maximum_series_length + 1`: the first division is a dedicated monomial-sentinel path, so
the product kernels never see an input beyond the cap.

**Matrix products.** An ordinary two-by-two composition uses eight products (typically 16
forward and eight inverse transforms) and applying a matrix to a pair uses four (eight
forward, four inverse). The fused contender transforms each input entry once and sums the two
pointwise products of an output entry before its single inverse: eight forward and four
inverse transforms for a composition, six and two for an application. A nonterminal
recursive node does one application, one composition and two products for the exact
boundary step, so besides division and its children it costs typically 28 forward and 14
inverse transforms in the ordinary path and 18 and eight in the fused path. Early degree
drops, tiny products and unsupported transform sizes change these counts; the tests pin the
composition and application counts with a counting engine. Products are reduced modulo
`x^maximum_series_length`, which the transition identities make exact.

All transforms and caches belong to the operation, without global mutable storage.

## See also

- [Power series](power-series.md): products, inverses and polynomial division used here.
- [Online convolution](online-convolution.md): streaming products for online algorithms.
- [Linear recurrences](linear-recurrence.md): the quadratic Berlekamp-Massey and distant terms.
- [Polynomial points](polynomial-points.md): evaluation and interpolation of polynomials.
- [Convolution and transforms](convolution.md)
- [Modular arithmetic](modular-arithmetic.md)
