# `atl::berlekamp_massey`, `atl::linear_recurrence_term`, `atl::rational_series_coefficient`

Infer a shortest recurrence from a sequence prefix, evaluate a distant term from
its coefficients, or extract a rational series coefficient over an exact field.
Use modular values with a prime modulus or another exact field; floating-point
sequences and integral types with truncating division are excluded.

```cpp
#include <atl/nt/linear_recurrence.hpp>
```

\include linear-recurrence/quick.cpp

Expected stdout:
\include linear-recurrence/quick.out

The field contract is also available independently in `<atl/nt/exact_field.hpp>`;
it does not include matrix storage or Gaussian elimination.

## API

`field_range<R>` requires a sized random-access range of `exact_field` values,
with references convertible to that value type. Each entry also requires the
const-qualified input ranges to satisfy field_range. Inputs can be arrays,
vectors, spans or ranges with a distinct sentinel; they are read without moving.

```cpp
template<field_range Sequence>
    requires field_range<Sequence const>
constexpr auto berlekamp_massey(Sequence const &sequence);

template<field_range Numerator, field_range Denominator>
    requires field_range<Numerator const> && field_range<Denominator const>
          && std::same_as<std::ranges::range_value_t<Numerator>, std::ranges::range_value_t<Denominator>>
constexpr auto rational_series_coefficient(Numerator const &numerator,
                                           Denominator const &denominator, uint64 k);

template<field_range Coefficients, field_range Initial>
    requires field_range<Coefficients const> && field_range<Initial const>
          && std::same_as<std::ranges::range_value_t<Coefficients>, std::ranges::range_value_t<Initial>>
constexpr auto linear_recurrence_term(Coefficients const &coefficients,
                                      Initial const &initial, uint64 k);
```

Berlekamp-Massey returns `std::vector<T>` of coefficients c, length d, with
`s[i]=c[0]s[i-1]+…+c[d-1]s[i-d]` for every `d<=i<sequence.size()`.
It finds a shortest recurrence of the supplied prefix. At least twice the true
minimal order in samples determines a generating sequence's minimal recurrence;
a shorter prefix may not predict unseen terms. Empty or all-zero input gives an
empty recurrence.

`linear_recurrence_term` returns T, the term a[k] using the first d initial terms.
Extra initial terms are ignored. If `k<d`, it returns initial[k]; an empty
recurrence returns zero for every k. The uint64 index supports 2^64-1.

`rational_series_coefficient` returns T, `[x^k] P(x)/Q(x)`. Coefficients are in
ascending degree order; P may be empty, and need not have degree below Q.
A constant denominator is allowed.

## Complexity

Berlekamp-Massey takes O(N²) field operations and O(N) storage. Recurrence
lookup takes O(d² log(k+1)) time for schoolbook Kitamasa, and
O(M(d) log(k+1)) for Bostan-Mori, with O(d) storage; reading inputs costs O(d).
The transform route has M(d)=O(d log d). Rational extraction has the same
product bound with d=max(P.size(), Q.size()-1), at least one storage slot,
plus the cost of reading P and Q. Small k does not avoid reading rational inputs.

Static modular values use [convolution](convolution.md); other exact fields use
schoolbook O(d²) products. Static moduli without a suitable NTT can also use
schoolbook products during constant evaluation. Runtime transform/product costs
follow convolution's dispatch and input bounds. Allocation failures propagate.

## Preconditions

Asserted with `ATL_ASSERT`: initial.size() >= coefficients.size(); Q nonempty
and Q[0] nonzero. Recurrence order and the rational working d are at most
`SIZE_MAX/4-1`, so doubling and bit-ceil size arithmetic is representable.
Convolution's size and coefficient bounds apply to paths that call it.

Documented-only: the exact-field laws and valid arithmetic on every evaluated
value. In particular, modular fields require a prime modulus and rational Q[0]
must be invertible. The constraints check operations, not field laws. Storage
must fit the respective vectors and iterator differences. Dynamic modular values
and their arithmetic use runtime modulus state, so those instantiations cannot
be constant-evaluated.

## Measured performance

The race compares transform-domain Bostan-Mori, Bostan-Mori with convolution or
schoolbook products, Kitamasa with Newton/convolution division or schoolbook
reduction, and companion matrix power. Each sample computes a[k] at k=1e18,
including allocation; quadratic contenders run through d=1000 and matrix power
at d=10. Every contender is checked against the library result on every raced
shape. The certifying test oracle steps the recurrence definition.

Source: `benchmarks/nt/linear_recurrence.cpp`. Intel Xeon Platinum 8280,
GCC 15.2, RelWithDebInfo, assertions disabled, 2026-10-15. Minimum milliseconds
of three per-run means, 10 samples per row. Each cell is plain / native:
`-O2 -g -DNDEBUG -std=c++23`, with native adding `-march=native`.

| d | by transform bostan mori | by convolution bostan mori | by convolution kitamasa | by schoolbook bostan mori | by schoolbook kitamasa | by matrix power |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 0.079 / 0.073 | 0.026 / 0.026 | 0.035 / 0.032 | 0.025 / 0.024 | 0.018 / 0.018 | 0.082 / 0.065 |
| 32 | 0.325 / 0.290 | 0.197 / 0.203 | 0.268 / 0.262 | 0.212 / 0.197 | 0.192 / 0.169 | — |
| 64 | 0.692 / 0.634 | 0.941 / 0.922 | 0.817 / 0.800 | 0.768 / 0.771 | 0.678 / 0.683 | — |
| 100 | 0.688 / 0.643 | 1.3 / 1.3 | 1.7 / 1.7 | 1.9 / 1.9 | 1.7 / 1.7 | — |
| 1000 | 6.4 / 6.1 | 12.4 / 12.2 | 15.7 / 15.5 | 182 / 182 | 153 / 153 | — |
| 100000 | 1204 / 1162 | 2270 / 2243 | 2549 / 2513 | — | — | — |

| Berlekamp-Massey, 2d terms | plain ms | native ms |
|---|---:|---:|
| d=1000 | 4.0 | 3.6 |
| d=5000 | 91.9 | 80.1 |

Schoolbook Kitamasa wins at d=10 (0.018 / 0.018 ms); transform Bostan-Mori
and schoolbook Kitamasa are close at d=64 (0.692 / 0.634 versus 0.678 / 0.683).
At d=100000, transforms take 1204 / 1162 ms versus 2270 / 2243 for convolution
Bostan-Mori and 2549 / 2513 for convolution Kitamasa. The library uses cutoff
64 when a transform of the planned size is supported; otherwise it uses
convolution Bostan-Mori for static moduli at that order, schoolbook Kitamasa
for smaller orders and other fields. Experimental contenders are in
`experimental/atl/experimental/nt/linear_recurrence_variants.hpp`.

## Pitfalls and notes

Do not infer a recurrence from a prefix shorter than the data needed to identify
its order and assume it extrapolates. Composite modular rings satisfy the syntax
of exact_field but can fail division at a non-unit. Choosing a static modulus
alone does not ensure the transform route: its root capacity must cover
`bit_ceil(2d+2)`.

## How it works

Berlekamp-Massey maintains a connection polynomial and an anchor polynomial
for the last length-changing discrepancy. A nonzero discrepancy subtracts a
shifted multiple of that anchor; the degree grows exactly when the current
prefix requires it. Each step scans at most the current order.

Kitamasa computes `x^k` modulo the characteristic polynomial and dots the
remainder with the initial terms. Its schoolbook reduction uses
`x^d = sum c[j] x^(d-1-j)`.

Bostan-Mori writes Q=1-c[0]x-…-c[d-1]x^d and P=A Q mod x^d. Multiplying
P and Q by Q(-x) makes the denominator even; keeping numerator coefficients
of k's parity halves the index each round. The transform route pairs entries
j and j^1 in the NTT's bit-reversed layout, evaluating at w and -w. It needs
two length-N forward transforms and two length-N/2 inverse transforms per
round. The transform of x supplies roots independently of the chosen engine.

References: J. L. Massey (1969), Shift-register synthesis and BCH decoding;
A. Bostan and R. Mori (2021), A simple and fast algorithm for computing the N-th
term of a linearly recurrent sequence; Kitamasa's method (polynomial reduction
modulo the characteristic polynomial).

## See also

- [Convolution and transforms](convolution.md)
- [Matrices and elimination](matrix.md)
- [Modular arithmetic](modular-arithmetic.md)
