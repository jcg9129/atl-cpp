# `atl::mod_int` and modular arithmetic

Modular values keep a representative in a fixed-width unsigned type. Choose
`mod_int` for a compile-time modulus, `dynamic_mod_int` for a runtime modulus,
or the raw helpers when a wrapper would be inconvenient. The static type also
supports 128-bit moduli; native and software `atl::uint128` share the same API.

```cpp
#include <atl/nt/mod_int.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/static_modulus.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/constants.hpp>
```

\include modular-arithmetic/residues.cpp

`<atl/nt/mod_int.hpp>` is the normal entry point for arithmetic and streaming.
The reduced alternative `<atl/nt/mod_int_core.hpp>` implements arithmetic and
declares the same non-template hidden stream friends, including implicit
conversions. Their implementations come from `<atl/nt/mod_int_io.hpp>`. Include
that extension before any use requiring a friend body in each translation unit
(including taking its address). Class instantiation and unevaluated stream
queries may precede the extension. With the core alone, streaming fails at
compile time with an include diagnostic; it never leaves an unresolved symbol.

`<atl/nt/mod_int_static.hpp>` is a further reduced alternative for the same
static arithmetic types and stream friend declarations, without the runtime
64-bit Montgomery class. It keeps exactly the same `modulus_engine<Mod>`
selection, including `montgomery<Mod>` for eligible 64-bit moduli. The matching
`<atl/nt/montgomery_static.hpp>` provides the compile-time Montgomery engine
and engine alias alone; `<atl/nt/montgomery.hpp>` remains the normal complete
entry point. The modular reducers and portable integer types still include
wide arithmetic.

## Modular values

### API

```cpp
template<unsigned_integral T, T Mod>
    requires(Mod > 0)
class mod_int;
template<uint32 Mod> using mod_int32 = mod_int<uint32, Mod>;
template<uint64 Mod> using mod_int64 = mod_int<uint64, Mod>;

template<typename T = uint32, int32 Id = 0>
    requires traits::is_one_of_v<T, uint32, uint64>
class dynamic_mod_int;
template<int32 Id = 0> using dynamic_mod_int32 = dynamic_mod_int<uint32, Id>;
template<int32 Id = 0> using dynamic_mod_int64 = dynamic_mod_int<uint64, Id>;
```

`T` is the residue type. `Mod` must be positive and may use its entire type range.
Each `(T, Id)` dynamic specialization owns an independent shared modulus, initially
1. Dynamic value methods belong to the runtime modulus domain and are not constexpr;
the defaulted zero constructor can be constant-evaluated. Static
methods are constexpr except stream I/O.

| Static member signature | Meaning |
|---|---|
| `static constexpr inline T mod()` | Positive modulus |
| `constexpr mod_int()` | Zero |
| `template<integral I> constexpr mod_int(I v)` | Reduce any integral input, including negatives |
| `template<integral I> static constexpr mod_int unchecked(I x)` | Lift a plain representative without reducing |
| `constexpr T value() const` | Plain representative in `[0, Mod)` |
| `constexpr mod_int inverse() const` | Multiplicative inverse |
| `constexpr mod_int power(uint64 exp) const` | Nonnegative power |
| `constexpr mod_int &operator+=(mod_int const &rhs)` | Modular addition, return this object |
| `constexpr mod_int &operator-=(mod_int const &rhs)` | Modular subtraction, return this object |
| `constexpr mod_int &operator*=(mod_int const &rhs)` | Modular multiplication, return this object |
| `constexpr mod_int &operator/=(mod_int const &rhs)` | Multiply by the inverse, return this object |
| `friend constexpr mod_int operator+(mod_int lhs, mod_int const &rhs)` | Sum |
| `friend constexpr mod_int operator-(mod_int lhs, mod_int const &rhs)` | Difference |
| `friend constexpr mod_int operator*(mod_int lhs, mod_int const &rhs)` | Product |
| `friend constexpr mod_int operator/(mod_int lhs, mod_int const &rhs)` | Quotient |
| `constexpr mod_int operator+() const` | Unchanged value |
| `constexpr mod_int operator-() const` | Additive inverse |
| `constexpr bool operator==(mod_int const &rhs) const` | Equality; `!=` is rewritten by C++ |
| `constexpr mod_int &operator++()` / `constexpr mod_int &operator--()` | Increment/decrement and return this object |
| `constexpr mod_int operator++(int)` / `constexpr mod_int operator--(int)` | Increment/decrement and return the saved value |
| `friend std::ostream &operator<<(std::ostream &os, mod_int const &mi)` | Write the plain representative, including uint128; return stream with its status |
| `friend std::istream &operator>>(std::istream &is, mod_int &mi)` | Read a checked signed magnitude, reduce and assign on success; return stream with its status |

Static stream insertion has this additional requires-clause (ordinary standard
integer insertions cover residue types through 64 bits):

```cpp
requires (std::same_as<T, atl::uint128> || requires(std::ostream &s, T r) {
    { s << r } -> std::convertible_to<std::ostream &>;
})
```

For `dynamic_mod_int`, replace `mod_int` in these signatures by
`dynamic_mod_int` and omit `constexpr` except for `value()`, equality and unary
`operator+`, which read only the stored value. `mod()` is declared `static T mod()`.
Its output friend needs no extra constraint because T is uint32 or uint64.
It also has `static void set_mod(T m)`, which replaces the shared reducer. Its
default constructor is `dynamic_mod_int() = default` and initializes zero.

`atl::modular_integer<Z>` in `<atl/types/concepts.hpp>` checks a regular value type,
construction from `int64` and `uint64`, matching unsigned modulus/representative
types, arithmetic and compound operators, negation, equality, inverse and power.
`atl::static_mod_int<Z>` additionally requires a constant-expression modulus.
Both modular classes satisfy `modular_integer`; only `mod_int` satisfies `static_mod_int`.
[Convolution](convolution.md) accepts static modular integers; its NTT engines
further require supported prime moduli and sufficient transform order.
They also satisfy the addition/multiplication `binary_operation` and negation
`inverse_operation` concepts in `<atl/types/algebraic.hpp>`. Inversion and division
still require units; these types do not imply that the modulus is prime.

### Complexity

One residue of storage per value, with O(1) shared reducer storage for a dynamic
type. Operations through 64 bits take O(1) fixed-width arithmetic, except inverse
and division use O(log m) Euclidean steps and power uses O(log exp) modular
multiplications. Static 128-bit multiplication takes O(log m) additions, so its
power takes O(log exp * log m). Construction and form conversion use bounded
fixed-width operations. Stream operations depend on the number of characters.

### Preconditions

Asserted with `ATL_ASSERT`: modulus is positive on `set_mod`, `unchecked` inputs
are nonnegative and smaller than the modulus, and inverses/divisors are coprime
to the modulus. Static positivity is a template constraint. Modulus 1 is legal;
all its representatives, including its multiplicative identity, are zero.

Documented-only: changing a dynamic modulus invalidates every live value of that
specialization. Set it when no values are live. Shared modulus updates require
external synchronization with all reads.

Stream insertion writes the plain representative, including native and software
`atl::uint128`. It honors decimal/octal/hexadecimal base, `showbase`, `uppercase`,
locale grouping and thousands separator, width, fill and left/right/internal
alignment, then resets width. For uint128, internal padding follows the base prefix.

Stream extraction honors `skipws`/`noskipws` and accepts at most one optional
`+` or `-`, immediately followed by the magnitude. It uses the selected decimal,
octal or hexadecimal base (decimal when unset); hexadecimal accepts an optional
`0x`/`0X` prefix. Input grouping is unsupported: a separator ends the digit run.
The magnitude must fit uint64 for residue types through 64 bits, or uint128 for
128-bit residues. Thus a 32-bit residue still accepts magnitudes above `2^32`.
Overflow is checked before arithmetic; the remaining digit run is consumed and
`failbit` is set. Malformed input, no digits or overflow leaves the destination
unchanged. Valid digits ending at EOF succeed with `eofbit` only. Stream exception
masks are respected, and stream flags are unchanged. Successful extraction reduces
the unsigned magnitude first, then applies modular negation for a minus sign.
Use `atl::io::reader` and `atl::io::writer` as the fast path for bulk I/O.

### Measured performance

`benchmarks/nt/dynamic_mod_int.cpp` measures sums of 10^7 independent constructor
reductions at modulus 998244353. Library `reduce_mod` uses division; by reciprocal
uses a local reciprocal and high-word product; the library constructor uses its
shared reciprocal, reducing signed magnitude then sign.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Input and method | `-O2` | `-O2 -march=native` |
|---|---:|---:|
| uint64, library reduce_mod (division) | 19.1 | 19.3 |
| uint64, by reciprocal | 7.2 | 7.2 |
| uint64, library constructor | 7.3 | 7.3 |
| int64, library reduce_mod (division) | 41.6 | 41.3 |
| int64, library constructor (magnitude reciprocal) | 39.7 | 39.4 |

Software-backend performance is not measured yet.

### Pitfalls and notes

A stored `mod_int64` word may be in Montgomery form. `value()` and `unchecked`
operate on plain representatives; bypassing that conversion corrupts arithmetic.
Use distinct dynamic tags for simultaneous moduli. Generic fast I/O recognizes
these types through `modular_integer`.

## Compile-time engines

### API

```cpp
template<auto Mod>
    requires unsigned_integral<decltype(Mod)> && (Mod > 0)
struct static_modulus;

template<auto Mod>
    requires unsigned_integral<decltype(Mod)>
        && traits::has_widen_mul_v<decltype(Mod)> && ((Mod & 1) != 0)
struct montgomery;

template<auto Mod>
    requires unsigned_integral<decltype(Mod)> && (Mod > 0)
using modulus_engine = /* selected engine */;
```

Each engine exposes `using value_type = decltype(Mod)` and
`static constexpr value_type mod = Mod`. All the operations below are static
constexpr functions. `static_modulus` accepts every positive unsigned modulus,
including 128-bit ones; `montgomery` accepts odd moduli in types through 64 bits.
`modulus_engine` selects Montgomery for 64-bit odd non-Mersenne moduli and
`static_modulus` for all other supported moduli.

| Signature | Result |
|---|---|
| `value_type add(value_type a, value_type b)` | Sum in the engine form |
| `value_type sub(value_type a, value_type b)` | Difference in the engine form |
| `value_type neg(value_type a)` | Additive inverse |
| `value_type incr(value_type a)` / `value_type decr(value_type a)` | Add/subtract the engine identity |
| `value_type mul(value_type a, value_type b)` | Product in the engine form |
| `value_type power(value_type base, uint64 exp)` | Power in the engine form |
| `value_type inverse(value_type a)` | Inverse in the engine form |
| `value_type to_form(value_type a)` / `value_type from_form(value_type a)` | Convert plain and engine residues |
| `value_type one()` | Engine representation of `1 % Mod` |

`static_modulus::is_mersenne` is a static constexpr bool: it identifies `2^k-1`
strictly below the full width. Montgomery uses `wide = traits::widen_mul_type_t<value_type>`
for the following parameter descriptions; the alias itself is private.

| Montgomery signature | Range / requirement |
|---|---|
| `value_type reduce_lazy(wide x) requires(Mod <= (std::numeric_limits<value_type>::max)() / 4)` | `x < Mod * R`; result in `[0, 2*Mod)` |
| `value_type mul_lazy(value_type a, value_type b) requires(Mod <= (std::numeric_limits<value_type>::max)() / 4)` | Inputs in `[0, 2*Mod)`; result in that range |
| `value_type companion(value_type w)` | `w < Mod`; returns `w * Mod^-1 mod R` |
| `value_type mul_lazy_fixed(value_type x, value_type w, value_type wq) requires(Mod <= (std::numeric_limits<value_type>::max)() / 4)` | `x < 2*Mod`, `w < Mod`, `wq == companion(w)` |

Here `R = 2^(word width)`. Lazy results are congruent to the tight engine result;
reduce them modulo `Mod` to tighten. The requires-clause limits lazy moduli to
`2^30-1` for uint32 and `2^62-1` for uint64; the tight API has no such restriction.

### Complexity

O(1) storage and tight arithmetic through 64 bits. Static 128-bit multiplication
uses O(log Mod) additions; Montgomery lazy functions take O(1). Power takes
O(log exp) modular multiplications and inverse takes O(log Mod) Euclidean steps.

### Preconditions

Compile-time constraints enforce positivity, unsigned modulus types, Montgomery
oddness and the availability of a double-width product. Lazy methods constrain
their modulus range. Arithmetic inputs must already be smaller than `Mod` in the
engine form (documented-only); lazy inputs follow the table's wider ranges.
The inverse also asserts coprimality. Form conversions require reduced inputs.
All-ones full-width moduli are valid but are excluded from the Mersenne fold:
its sum needs an extra bit at full width.

### Measured performance

`benchmarks/nt/montgomery.cpp` compares dependent chains of 5*10^6 multiplications:
static reciprocal reduction, Montgomery REDC, and the modular wrapper. The
wrapper selects Montgomery at 10^18+3. Each form conversion surrounds the chain.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Modulus and method | `-O2` | `-O2 -march=native` |
|---|---:|---:|
| 10^18+3, library static_modulus (reciprocal) | 26.2 | 23.1 |
| 10^18+3, library montgomery (REDC) | 17 | 16.5 |
| 10^18+3, library mod_int64 (Montgomery) | 17 | 16.5 |
| 998244353, library static_modulus (reciprocal) | 16.5 | 16.4 |
| 998244353, library montgomery (REDC) | 16.5 | 16.4 |

`benchmarks/nt/static_modulus.cpp` compares the same chain length across
reciprocal, Mersenne-fold, runtime-modulus and by-wide-remainder paths. Two
additional rows sum 10^5 modular powers with exponent `e | (1 << 40)` to compare
64-bit integer spellings. Same CPU, compiler, build, date, sampling and common
flags; milliseconds:

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| Modulus and method | `-O2` | `-O2 -march=native` |
|---|---:|---:|
| 998244353, library static_modulus | 16.4 | 16.4 |
| 998244353, library mod_int | 16.4 | 16.5 |
| 998244353, library mul_mod (runtime modulus) | 19.1 | 19.1 |
| 2^31-1, library static_modulus (fold) | 10.9 | 10.9 |
| 2^31-1, library mod_int | 10.9 | 10.9 |
| 2^61-1, library static_modulus (fold) | 12.3 | 12.4 |
| 2^61-1, library mod_int64 | 12.3 | 12.3 |
| 10^18, by wide remainder | 29.7 | 29.7 |
| 10^18, library static_modulus (barrett64) | 23.5 | 23.4 |
| 10^18, library mod_int64 | 23.5 | 23.1 |
| 10^18+3, library pow_mod<unsigned long long>, 10^5 powers | 22 | 22 |
| 10^18+3, library pow_mod<uint64>, 10^5 powers | 21.9 | 22.2 |

The 10^18 reciprocal is 29.7/23.5 = 1.26 times as fast as wide remainder
plain and 29.7/23.4 = 1.27 times native. Software-backend performance is
not measured yet.

### Pitfalls and notes

Engine words carry a representation, unlike ordinary residues. Stay in one
engine form throughout a calculation. Montgomery lazy results require tightening
prior to a tight operation. A Mersenne modulus need not be prime.
The [convolution transform engines](convolution.md) use Montgomery twiddles with
plain lazy residues; their prime-modulus and range constraints are stricter than
those of general modular arithmetic.

## Raw arithmetic and runtime reducers

### API

Every free function is `constexpr inline`; `T` satisfies `unsigned_integral` and
`I` satisfies `integral`.

| Signature | Result |
|---|---|
| `T reduce_mod(I a, T m)` | Representative of any signed/unsigned input in `[0, m)` |
| `T add_mod_unchecked(T a, T b, T m)` | Sum of reduced inputs |
| `T sub_mod_unchecked(T a, T b, T m)` | Difference of reduced inputs |
| `T mul_mod_unchecked(T a, T b, T m)` | Product of reduced inputs |
| `T mul_mod_binary(T a, T b, T m)` | Product using double-and-add |
| `T add_mod(T a, T b, T m)` / `T sub_mod(T a, T b, T m)` / `T mul_mod(T a, T b, T m)` | Reduce both inputs, then operate |
| `T pow_mod(T base, uint64 exp, T m)` | Power, reducing base first |
| `T incr_mod(T a, T m)` / `T decr_mod(T a, T m)` | Increment/decrement with wraparound |
| `T neg_mod(T a, T m)` | Additive inverse of a reduced input |
| `T inverse_mod(T v, T m)` | Multiplicative inverse; v need not be reduced |

`class barrett` and `class barrett64` retain a reciprocal for a runtime modulus.
Their complete value API is:

```cpp
constexpr barrett() = default;
constexpr explicit barrett(uint32 m);
constexpr uint32 mod() const;
constexpr uint32 mul(uint32 a, uint32 b) const;

constexpr barrett64() = default;
constexpr explicit barrett64(uint64 m);
constexpr uint64 mod() const;
constexpr uint64 mul(uint64 a, uint64 b) const;
```

Default reducers use modulus 1. Explicit constructors accept every positive value
of their modulus type, including `2^32-1` and `2^64-1`.

### Complexity

O(1) storage. Reduction, addition, subtraction, negation and increment/decrement
use O(1) fixed-width operations. Widening multiplication is O(1) through 64 bits,
using native or software uint128. `mul_mod_binary` and 128-bit `mul_mod` take
O(log b) additions. Power takes O(log exp) modular multiplications; inverse takes
O(log m) Euclidean steps. Reducer setup and multiplication take O(1).

### Preconditions

Asserted: `m > 0` for reduction, plain operations, power and inverse;
`a, b < m` for unchecked operations and binary multiplication; `a < m` for
increment/decrement/negation; coprimality for inverse; and positive modulus and
reduced multiplication operands for Barrett reducers. These bounds admit the
entire positive range of the modulus type, with no half-range restriction.
Invalid calls with assertions disabled are outside the contract.

### Measured performance

The runtime uint32 multiplication and reduction paths appear in the tables
above. Barrett64 chains are measured above, and `benchmarks/types/int128.cpp`
measures raw native/software 128-bit arithmetic.
Standalone Barrett setup is not measured yet.

### Pitfalls and notes

A widened product must retain both halves. `widen_mul_type_t<uint64>` is
`uint128` on every target; only 128-bit operands lack a double-width integer
and need binary multiplication. Native compiler integer traits and `<bit>`
operations do not recognize the software classes; use atl traits.

## Constants

### API, complexity and preconditions

`inline constexpr uint32 largest_prime_32 = 4'294'967'291u` and
`inline constexpr uint64 largest_prime_64 = 18'446'744'073'709'551'557ull`
are the largest primes below `2^32` and `2^64`. Reading either is O(1), with no
preconditions. Compile-time checks verify primality and test every larger
representable candidate.

### Measured performance and notes

Not measured yet; these are compile-time values. Primality of these constants
does not make every supported modulus prime.

## How it works

Raw addition and subtraction compare against the modulus gap so no overflowing
sum is needed. Negative reduction forms an unsigned magnitude, which also
handles the signed minimum. Extended gcd supplies inverse coefficients.

Barrett stores `ceil(2^64/m)` or `ceil(2^128/m)`. A high-product estimate is the
true quotient or one larger, so a conditional correction recovers the remainder.
The 128-by-128 high product uses the shared limb helper, preserving both carries
from its middle terms. The primitive layer selects native arithmetic, MSVC x64
or ARM64 intrinsics, or portable 32-bit-limb arithmetic as appropriate.

A static Mersenne reducer folds high bits using `2^k == 1 (mod m)` when a
widened product exists. Other 64-bit moduli use a constant Barrett reciprocal;
128-bit moduli use double-and-add to avoid truncating the product.

Montgomery stores `a*R mod m`. Newton iteration computes `m^-1 mod R` and the
high halves of the product and its correction implement reduction without
forming an overflowing double-width sum. Subtracting their high words supports
moduli above half the word range. Lazy methods omit tightening when the modulus
bound proves their larger inputs still produce a valid reduction. Binary
exponentiation keeps values in their engine form until `value()` is requested.

## See also

- [Extended gcd](gcd.md): coefficients and modular inversion.
- [Power](power.md): generic binary exponentiation with an explicit identity.
- [128-bit integers](int128.md): portable arithmetic and platform backends.
- [Fast I/O](fast-io.md): buffered modular value input and output.
- [Types and utilities](types-and-utilities.md): aliases and concepts.
- [Convolution](convolution.md) — static-modulus products and transform engines.
