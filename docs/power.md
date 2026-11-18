# `atl::power`

Binary exponentiation raises a value to a nonnegative integer power. Use it for
integers, modular values and associative products; supply an identity explicitly
when it depends on runtime state or cannot be constructed as `Base(1)`.

```cpp
#include <atl/nt/power.hpp>
```

\include power/monoid.cpp

## API

```cpp
template<typename Base, unsigned_integral Exponent>
    requires requires(Base b) {
        { b * b } -> std::convertible_to<Base>;
        b = b * b;
        Base(std::move(b));
    }
constexpr Base power(Base base, Exponent exp, Base identity);

template<typename Base, unsigned_integral Exponent>
    requires requires(Base b) {
        { b * b } -> std::convertible_to<Base>;
        b = b * b;
        Base(std::move(b));
        Base(1);
    }
constexpr Base power(Base base, Exponent exp);
```

`base` is the element, `exp` its unsigned exponent, and `identity` the product's
neutral element. The result is the product of `exp` copies of `base`. Exponent
zero returns the supplied identity, or `Base(1)` in the two-argument overload.
Direct, explicit construction from `1` is accepted; move-only bases work when
their product supports the stated expressions.

## Complexity

O(log exp) multiplications and O(1) auxiliary `Base` objects. The cost
of one multiplication depends on `Base`.

## Preconditions

No runtime assertions are needed. Documented-only requirements: multiplication
is associative, the identity is a two-sided identity, and every evaluated
multiplication is valid. Signed integral products must fit `Base`; unsigned
arithmetic wraps. These laws and value bounds cannot be checked by a concept.

## Measured performance

Not measured yet; there is no assigned generic-power benchmark. Modular power
measurements are on the modular arithmetic page.

## Pitfalls and notes

Signed exponents are rejected, even when their value is positive. Multiplication
alone is insufficient: products must also be assignable back to `Base`. For a
stateful identity, use the three-argument overload. The integer convention gives
`power(0, 0u) == 1`.

## How it works

Each exponent bit conditionally multiplies an accumulator and halves the
remaining exponent. Squaring supplies the next power of two. The final unused
square is skipped, avoiding overflow when the requested integer power fits but
its square does not. Associativity makes the grouping valid without requiring
commutativity.

## See also

- [Modular arithmetic](modular-arithmetic.md): modular values and modular exponentiation.
