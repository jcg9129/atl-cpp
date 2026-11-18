# `atl::extended_gcd`

The extended Euclidean algorithm returns a gcd and Bézout coefficients satisfying
`a*x + b*y == gcd`. Use it for coefficients and modular inverses; use a plain gcd
when only the divisor is needed. Signed and unsigned inputs include the portable
128-bit integer types.

```cpp
#include <atl/nt/gcd.hpp>
```

\include gcd/bezout.cpp

## API

```cpp
template<unsigned_integral T>
struct gcd_result {
    T gcd;
    traits::make_signed_t<T> x;
    traits::make_signed_t<T> y;
};

template<unsigned_integral U>
constexpr gcd_result<U> extended_gcd(U const &a, U const &b);

template<signed_integral S>
constexpr gcd_result<traits::make_unsigned_t<S>>
extended_gcd(S const &a, S const &b);
```

The operands have the same integral type. The result is structured-bindable as
`{gcd, x, y}`. Its gcd is unsigned; `x` and `y` use the signed counterpart of that
unsigned type. For `(0, 0)` the result is `{0, 1, 0}`. A negative operand changes the
sign of its coefficient, keeping the gcd nonnegative.

## Complexity

O(log min(|a|, |b|)) Euclidean iterations and O(1) storage, counting
fixed-width integer operations as constant time. Both overloads are constexpr.

## Preconditions

There are no value restrictions beyond the operand type. The signed minimum is
supported. For `g = gcd(a, b) > 0`, the coefficient bounds are
`|x| <= max(1, |b|/(2*g))` and `|y| <= max(1, |a|/(2*g))`, so the returned
coefficients fit even when unsigned inputs use their entire range.

## Measured performance

Not measured yet; there is no assigned gcd benchmark.

## Pitfalls and notes

The Bézout identity is mathematical: evaluating its products in the input type
can overflow. Widen 32-bit inputs to `atl::int64` and 64-bit inputs to
`atl::int128` when checking it. For 128-bit inputs, use an exact wider accumulator
or check the identity with suitable bounds. Mixed signedness and non-integral
arguments do not satisfy the overload constraints.

## How it works

The loop divides successive unsigned remainders. A lagged quotient updates two
pairs of signed coefficients, stopping without forming the final cofactor pair
`(b/g, a/g)`. That avoids coefficients outside the signed range at the top of the
unsigned input range. The signed overload obtains magnitudes by unsigned
subtraction, calls the unsigned loop and restores the coefficient signs. There
is no recursive result or recursive reference to a local accumulator.

## See also

- [Modular arithmetic](modular-arithmetic.md): modular inverses built on these coefficients.
- [128-bit integers](int128.md): native and software wide arithmetic.
