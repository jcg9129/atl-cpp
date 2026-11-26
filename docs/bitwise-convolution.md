# `atl::subset_zeta`, `atl::subset_convolution`

Transform functions on subsets and multiply arrays indexed by bitmasks. Use
OR/AND/XOR products to group pairs by the corresponding mask operation, and
subset convolution to sum disjoint partitions of a mask.

```cpp
#include <atl/nt/bitwise_convolution.hpp>
```

\include bitwise-convolution/quick.cpp

Expected stdout:
\include bitwise-convolution/quick.out

## API and complexity

All transforms have signature `template<set_function_range R> constexpr void
name(R &&f)`. `inverse_walsh_hadamard` additionally requires
`halvable_ring<std::ranges::range_value_t<R>>`. Each mutates f in place with
O(d 2^d) arithmetic operations and O(1) auxiliary storage for `|f|=2^d`.

Each transform applies a 2×2 map to `(f[S], f[S ∪ {i}])`, where `i ∉ S`,
along each bit in turn:

| function | 2×2 map on `(f[S], f[S ∪ {i}])` | result at mask S |
|---|---|---|
| `subset_zeta(f)` | `(a, b) → (a, a + b)` | `f[S] = Σ_{T ⊆ S} f[T]` |
| `subset_mobius(f)` | `(a, b) → (a, b − a)` | inverse of `subset_zeta` |
| `superset_zeta(f)` | `(a, b) → (a + b, b)` | `f[S] = Σ_{T ⊇ S} f[T]` |
| `superset_mobius(f)` | `(a, b) → (a − b, b)` | inverse of `superset_zeta` |
| `walsh_hadamard(f)` | `(a, b) → (a + b, a − b)` | `f[S] = Σ_T (−1)^{\|S ∩ T\|} f[T]` (unscaled) |
| `inverse_walsh_hadamard(f)` | the same, then `/ 2^d` | inverse of `walsh_hadamard` |

The products have this signature, replacing name with each row's function:

```cpp
template<std::ranges::random_access_range A, ring_range<std::ranges::range_value_t<A>> B>
    requires ring_range<A, std::ranges::range_value_t<A>>
          && commutative_ring<std::ranges::range_value_t<A>>
constexpr std::vector<std::ranges::range_value_t<A>> name(A const &a, B const &b);
```

`xor_convolution` replaces the `commutative_ring` requirement with
`halvable_ring`. Both const-readable input ranges contain the same value type.
They may have arbitrary lengths; inputs are copied and zero-padded to
`N = bit_ceil(max(|a|,|b|))`. The returned vector has N entries and is empty
when both inputs are empty. One empty input gives a zero vector of the other
input's padded size.

| function | result h[S] | time | auxiliary space |
|---|---|---|---|
| `or_convolution(a,b)` | `Σ_{L ∪ R = S} a[L] b[R]` | O(dN) | O(N) |
| `and_convolution(a,b)` | `Σ_{L ∩ R = S} a[L] b[R]` | O(dN) | O(N) |
| `xor_convolution(a,b)` | `Σ_{L ⊕ R = S} a[L] b[R]` | O(dN) | O(N) |
| `subset_convolution(a,b)` | `Σ_{L ⊆ S} a[L] b[S \ L]` (disjoint union) | O(d^2 N) | O(dN) |

For N=1 these bounds include constant work. Allocation failures propagate.

| concept | syntactic requirement | documented algebra |
|---|---|---|
| `additive_group<T>` | regular T, convertible `+`/`-`, compound forms returning T& | T{} is zero; addition/subtraction form an abelian group |
| `commutative_ring<T>` | additive_group plus multiplication and compound form | multiplication is associative/commutative and distributes |
| `halvable_ring<T>` | commutative_ring plus modular, signed integral, or floating type | division by powers of two is available |
| `set_function_range<R>` | writable sized random-access range yielding additive_group lvalue references | elements obey the selected transform's laws |
| `ring_range<R,T>` | const-readable sized random-access range with value type T, copyable into T | ring laws are imposed by the selected product |

## Preconditions

Asserted with `ATL_ASSERT`: transform length is a nonzero power of two; modular
inverse Walsh/XOR has an odd modulus; signed inverse Walsh entries divide
exactly by 2^d. Product padding must fit size_t, and ranked array size
`N*(d+1)` must fit size_t.

Documented-only: the algebraic laws in the table, and no signed arithmetic
overflow in any intermediate. Unsigned arithmetic wraps modulo 2^width; inverse
Walsh and XOR reject unsigned elements because two is not invertible there.
For signed XOR products the conservative intermediate bound is
`N^3 max|a| max|b|`; both transform-domain products and the unscaled inverse
must fit. Floating operations are approximate. Modular operations are exact,
with the selected modulus held constant throughout a call. Signed inverse
scaling divides in representable steps when 2^d exceeds the element type's range.

## Measured performance

The loop-order race copies an input then transforms it using blocked radix-2
(library), bit-outer radix-2, radix-4, or blocked radix-4; a copy-only row
isolates copying. The product race compares restricted mask-major ranks with
full rank-major arrays. By-definition experimental sums certify correctness.

Source: `benchmarks/nt/bitwise_convolution.cpp`.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| transform and shape | library | by bit outer radix2 | by radix4 | by blocked radix4 | copy only |
|---|---:|---:|---:|---:|---:|
| subset zeta, mod_int 998244353, d=20 | 6.5 / 6.2 | 6.5 / 6.3 | 6.3 / 6.2 | 7 / 6.8 | 0.086 / 0.089 |
| subset zeta, uint64, d=20 | 3.4 / 3.4 | 4 / 3.5 | 3.2 / 3.4 | 3.6 / 3.6 | 0.174 / 0.176 |
| Walsh, mod_int 998244353, d=20 | 11.5 / 12.2 | 11.2 / 12.2 | 10.6 / 11.9 | 11.9 / 12.8 | 0.085 / 0.087 |
| Walsh, uint64, d=20 | 5.6 / 5.3 | 5.8 / 5.6 | 5.8 / 5.6 | 6 / 5.9 | 0.179 / 0.179 |
| subset zeta, mod_int 998244353, d=16 | 0.327 / 0.32 | 0.323 / 0.306 | 0.296 / 0.28 | 0.33 / 0.313 | 0.002 / 0.003 |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| product, mod_int 998244353, d=20 | -O2 ms | -O2 -march=native ms |
|---|---:|---:|
| library or_convolution | 20.5 | 20.2 |
| library xor_convolution | 38 | 37 |
| library subset_convolution | 576 | 505 |
| subset_convolution by rank major | 856 | 826 |

Mask-major restricted ranks improve the subset product in both configurations.
Radix-4 has the smallest plain sum of the four d=20 transforms (25.9 versus
the library's 27 ms); native they tie at 27.1 ms at table precision.
The library loses the modular Walsh transform to radix-4 (11.5/12.2 versus
10.6/11.9 ms), but wins the uint64 Walsh row (5.6/5.3 versus 5.8/5.6 ms). At d=20 the two ranked arrays hold
`2*(d+1)*2^d` values, or 168 MiB for a four-byte modular integer.

## Pitfalls and notes

The transforms reject empty ranges; products handle empty inputs as described
above. Padding changes the mask universe. Unsigned forward Walsh is valid,
but cannot be inverted by division modulo a power of two. An inverse integer
Walsh input must belong to the transform image, with intermediates representable.

## How it works

Each transform applies the same two-by-two butterfly along each bit; operations
on different bits commute. The low 11 bits are processed inside blocks, then
one radix-2 sweep handles each remaining bit. Zeta turns OR multiplication
into pointwise multiplication; superset zeta does the same for AND. Walsh
handles XOR, with a final inverse-power-of-two scale.

Subset convolution embeds a[S] at rank popcount(S), applies ranked subset
zeta, multiplies the rank polynomials for each mask, and applies Möbius.
Reading rank popcount(S) selects disjoint unions. Rows are mask-major; zeta
skips ranks above the mask's popcount, and products/Möbius skip ranks below
those needed by the final read. These restrictions retain every contributing
term and give O(d^2 N) work with O(dN) storage.

References: Liu Chengao (2018), "DFT" (national team papers);
Lv Kaifeng (2015), "Set power series", §5.1.1.

## See also

- [Convolution and transforms](convolution.md)
- [Modular arithmetic](modular-arithmetic.md)
