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

Source: `benchmarks/nt/linear_recurrence.cpp`.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| d | by transform bostan mori | by convolution bostan mori | by convolution kitamasa | by schoolbook bostan mori | by schoolbook kitamasa | by matrix power |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 0.058 / 0.064 | 0.018 / 0.023 | 0.022 / 0.028 | 0.017 / 0.022 | 0.014 / 0.017 | 0.049 / 0.051 |
| 32 | 0.225 / 0.253 | 0.151 / 0.189 | 0.195 / 0.242 | 0.15 / 0.186 | 0.127 / 0.158 | — |
| 64 | 0.472 / 0.536 | 0.646 / 0.767 | 0.508 / 0.602 | 0.574 / 0.72 | 0.495 / 0.612 | — |
| 100 | 0.476 / 0.54 | 0.893 / 1.1 | 1.1 / 1.3 | 1.4 / 1.7 | 1.2 / 1.5 | — |
| 1000 | 4.6 / 5.3 | 8.9 / 10.7 | 10.1 / 12.1 | 137 / 167 | 113 / 138 | — |
| 100000 | 853 / 1000 | 1630 / 1980 | 1630 / 1980 | — | — | — |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Berlekamp-Massey, 2d terms | plain ms | native ms |
|---|---:|---:|
| d=1000 | 2.6 | 3.1 |
| d=5000 | 59.7 | 75.3 |

Schoolbook Kitamasa wins at d=10 (0.014/0.017 ms) and d=32
(0.127/0.158 ms). At d=64 transform Bostan-Mori takes 0.472/0.536 ms
versus schoolbook Kitamasa's 0.495/0.612 ms. At d=100000 transforms take
853/1000 ms versus 1630/1980 ms for both convolution contenders. The library uses cutoff
64 when a transform of the planned size is supported; otherwise it uses
convolution Bostan-Mori for static moduli at that order, schoolbook Kitamasa
for smaller orders and other fields. Experimental contenders are in
`experimental/atl/experimental/nt/linear_recurrence_variants.hpp`.

The d=10 through d=1000 comparisons include contenders that vary between
runs (spreads up to 1.53x). Their quoted minima alone do not establish
stable rankings at those orders.

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
