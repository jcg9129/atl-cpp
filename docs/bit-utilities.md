# `atl::lowest_one`, `atl::next_bits_combination`

Small unsigned-word operations for isolating a set bit and enumerating subsets
of a fixed size. Use them when the subset fits in one word; they do not provide
a dynamic bitset. All supported unsigned widths, including native and software
`atl::uint128`, work in constant evaluation.

```cpp
#include <atl/bits/bitlow.hpp>
#include <atl/bits/next_bits_combination.hpp>
```

\include bit-utilities/combinations.cpp

Expected stdout:

\verbinclude bit-utilities/combinations.out

## API and complexity

Both templates require `atl::unsigned_integral<T>`: the supported unsigned
integer types, excluding bool and the distinct character types. Their exact
signatures are:

```cpp
template<unsigned_integral T>
constexpr T lowest_one(T x) noexcept;

template<unsigned_integral T>
constexpr T next_bits_combination(T x) noexcept;
```

| function | input and result | time / extra space |
|---|---|---|
| `lowest_one(x)` | Only the least significant set bit of `x`, or zero for zero input | O(1) / O(1) |
| `next_bits_combination(x)` | Least word greater than `x` with the same popcount, or zero if none exists or `x == 0` | O(1) / O(1) |

## Preconditions

Neither function has asserted or documented-only input preconditions: every
value of the selected unsigned type is accepted. Enumeration setup has separate
bounds: for `w` word bits and `0 < k < w`, start with `(T(1) << k) - 1`.
For `k == w`, start with `T(~T(0))` instead, avoiding a shift by the width.
For `k == 0`, process the empty subset once yourself: zero is the termination
sentinel and its successor is zero.

## Measured performance

`benchmarks/bits/next_bits_combination.cpp` races three formulations of the
successor. All add the lowest set bit and then shift the rest of the moved run
back down; they differ in how they find that shift. The library counts the
trailing zeros of `x` before the addition, so the count runs in parallel with
it. "By run ctz" counts them on the carried run, after the addition. "By shift
loop" shifts the carried run down one bit at a time. Each step of the
benchmark depends on the previous one, so it measures per-step latency.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | flags | library | by run ctz | by shift loop |
|---|---|---:|---:|---:|
| C(28,14) | `-O2` | 99.5 | 122 | 99.5 |
| C(28,14) | `-O2 -march=native` | 77.2 | 88.3 | 78.9 |
| C(40,6) | `-O2` | 9.5 | 11.6 | 13.7 |
| C(40,6) | `-O2 -march=native` | 7.4 | 8.4 | 13 |

The library form is never slower. The shift loop ties it only on the plain C(28,14) row (99.5 ms each);
the library wins that shape native (77.2 versus 78.9 ms).

## Pitfalls and notes

A combination is an unsigned numeric mask. Enumeration covers every bit of
`T`, so stop at the application's smaller width as the example does. The zero
sentinel must also be checked when enumerating the full word width. Signed
integers are rejected by the constraint. `atl::uint128` is always available;
no native-int128 guard is needed.

## How it works

`lowest_one` intersects the word with its two's-complement negation, written
as complement plus one to avoid unsigned-negation diagnostics. Integer
promotions on narrow built-in types are explicitly converted back to `T`.

The fixed-popcount successor uses a carry and packs the displaced ones. Let `r` be the lowest set bit. Adding `r`
moves the lowest movable one to the next zero, and the cleared ones are packed
at the bottom. The code checks for zero and for carry out before adding.
The trailing-zero count comes from the original input, so it can overlap the
addition. Standard bit operations inspect native 64-bit limbs for the software
128-bit type. Two successive right shifts keep each count below the word
width, including a lone one in the highest position.

## See also

- [XOR bases](xor-basis.md) for linear spans of unsigned words.
- [Types and utilities](types-and-utilities.md) for integer aliases and constraints.
- [128-bit integers](int128.md) for native and software integer semantics.
