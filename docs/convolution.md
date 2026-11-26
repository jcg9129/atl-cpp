# `atl::convolution`

Multiply polynomials over a static modulus, or multiply nonnegative integer
coefficient arrays exactly under a sufficient uint64 bound. Use the transform
pairs for cyclic products or complex signals; use the polynomial entries for
ordinary products without wraparound.

```cpp
#include <atl/nt/convolution.hpp>
#include <atl/nt/fft.hpp>
#include <atl/nt/ntt_engine.hpp>
```

\include convolution/quick.cpp

Expected stdout:
\include convolution/quick.out

## API

```cpp
template<ntt_engine Engine = default_ntt_engine, mod_int_range A, mod_int_range B>
    requires std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>
          && mod_int_range<A const> && mod_int_range<B const>
constexpr auto convolution(A const &a, B const &b);
template<ntt_engine Engine = default_ntt_engine, integral_range A, integral_range B>
    requires integral_range<A const> && integral_range<B const>
constexpr std::vector<uint64> convolution_uint64(A const &a, B const &b);
template<ntt_engine Engine = default_ntt_engine, mod_int_range R>
    requires detail::ntt_modulus<std::ranges::range_value_t<R>>
          && std::ranges::output_range<R, std::ranges::range_value_t<R>>
constexpr void ntt(R &&a);
template<ntt_engine Engine = default_ntt_engine, mod_int_range R>
    requires detail::ntt_modulus<std::ranges::range_value_t<R>>
          && std::ranges::output_range<R, std::ranges::range_value_t<R>>
constexpr void inverse_ntt(R &&a);
template<cplx_range R> void fft(R &&a);
template<cplx_range R> void inverse_fft(R &&a);
```

`convolution` returns a vector of the common modular value type;
`convolution_uint64` returns `vector<uint64>`. Either returns empty when one
input is empty, and otherwise returns `n+m-1` entries. Input ranges are read
without moving their elements. `mod_int_range` requires a sized random-access
range of `static_mod_int` values; `integral_range` requires integral values,
including software `int128` and `uint128`. Both polynomial entries also require
const-readable ranges.

`ntt` maps natural order to bit-reversed order; `inverse_ntt` consumes that
order and scales by `1/n`. Pointwise multiplication between them computes a
cyclic product. `fft` and `inverse_fft` use natural order at both ends, with
inverse scaling included. `cplx_range` is a writable sized random-access range
of `cplx`; that aggregate has `double re = 0, im = 0`, componentwise `+` and
`-`, complex `*`, and `constexpr cplx conj() const`. Its arithmetic is constexpr;
the FFTs use runtime trigonometry and a shared root cache.

### Engine contract

`ntt_engine<E>` checks `supports<Z>()`, `forward<Z>(U*, uint32 h)`,
`inverse<Z>(U*, uint32 h)` and `convolve<Z>(U*, U*, uint32 h)` against
`mod_int<uint32,998244353>`. `supports` is a constexpr bool query; the convolve
probe is skipped when that representative modulus is declined. The raw
buffers hold `2^h` residues: forward and unscaled inverse use `[0,2p)`, while
convolve accepts canonical inputs, writes a fully scaled canonical result to
its first buffer and overwrites its second. Two-buffer convolve requires
distinct buffers. `ntt_engine_base<Derived>` supplies convolve and
`square<Z>(U*, uint32 h)` using Derived's transforms. `scalar_ntt` also exposes
its per-modulus `twiddles<Z>` and `twiddle_table<Z>()`; the latter returns a
shared immutable table and cannot be constant-evaluated.

`default_ntt_engine` aliases `scalar_ntt` unless `ATL_NTT_ENGINE` names another
engine whose header is already included. An explicit template argument selects
an engine per call. The driver falls back to scalar for a direct-NTT modulus
unsupported by the requested engine. Constant evaluation always uses scalar
transforms, with naive multiplication for a modulus without a suitable direct NTT.

Optional engine includes:

```cpp
#include <atl/nt/detail/ntt_simd.hpp>
#include <atl/nt/detail/ntt_conv8.hpp>
```

`simd_ntt` processes eight residues at a time. Its forward and inverse accept
an optional `uint32 stop = 0` to leave the finest levels undone. `conv8_ntt`
provides the same raw transforms and fuses eight-point products in convolve
and square (which require `h >= 3`). Its public block helpers
`conv8_block_simd<Z>(uint32*, uint32 const*, uint32 w_mont)` and
`conv8_block_reference<Z>(uint32*, uint32*, uint32 w_mont)` compute the same
product modulo `x^8-w`, scaled by one Montgomery inverse factor; the former
returns lazy residues, the latter canonical ones. Their eight-entry inputs
are lazy residues, `w_mont` is canonical Montgomery form, and the inputs may alias.

Both vector engines require a 32-bit representation and the scalar modulus
contract. On x86-64 and i386, GCC/Clang compile their vector code with
`__attribute__((target("avx2")))`, including under plain `-O2`. Choosing
`simd_ntt` or `conv8_ntt` requires an AVX2-capable CPU; execution stops with an
illegal instruction otherwise. There is no runtime CPU dispatch. The default
`scalar_ntt` has no AVX2 requirement, and including an engine header does not
require AVX2. AArch64 uses baseline NEON. MSVC, other architectures, and
`ATL_SIMD_PORTABLE` / `ATL_FORCE_FALLBACKS` builds use standard C++ lanes.
Engine selection does not change the result. Raw vector operations are runtime-only.

Function target attributes work with both GCC and Clang; Clang does not honour
GCC's target-option pragmas. All includes precede the attributed definitions.
Vector values stay inside functions with the same target; entry points exchange
only pointers, sizes and scalars, with the hot loops inside targeted functions.
The target selection depends on compiler, architecture and configuration macros,
so translation units with the same configuration use the same definitions
regardless of `-march`.

## Complexity

Polynomial products take O((n+m) log(n+m)) time and O(n+m) scratch storage;
small inputs use O(nm). Lopsided inputs reuse the short spectrum across chunks.
Unfriendly moduli use O(nm) during constant evaluation. Transform pairs take
O(n log n) time; typed NTTs allocate O(n) scratch, raw engines use O(1) scratch
plus O(log p) cached per-modulus tables. Complex FFT roots use O(n) shared
cached storage, growing to the largest requested size. Allocation failures propagate.

## Preconditions

Asserted with `ATL_ASSERT`:

- Transforms have nonzero power-of-two lengths. NTT lengths divide `p-1`;
  raw h fits `size_t` and does not exceed `countr_zero(p-1)`. Partial transforms
  require `stop <= h`; fused convolve/square require `h >= 3`.
- Nonempty polynomial input lengths are at most `2^(size_t digits - 2)`.
- The three-prime Garner tier requires planned transform size at most 2^23 and
  `min(n,m) max(a) max(b) < 167772161*469762049*998244353`.
  The maxima are canonical representatives; the check uses exact portable uint128.
- Integer inputs are nonnegative, fit uint64, and satisfy the sufficient bound
  `B = min(n,m) max(a) max(b) <= UINT64_MAX`. This can reject sparse inputs
  even when their actual coefficients would fit. Both integer transform tiers
  are capped at 2^23 points. The bound applies even below the naive cutoff.
  Its exact calculation and path selection run independently of `ATL_ASSERT`.
- The internal balanced FFT split checks modulus <= 2^30 and planned size <= 2^19.

Constrained: NTT representations are 32- or 64-bit unsigned words, with odd
prime moduli below 2^30; arbitrary modular convolution
also accepts composite and even moduli. Mutating transforms require writable ranges.

Documented-only: valid raw buffers, canonical or lazy residue ranges as specified,
nonaliasing of the two raw convolve buffers, and consistent modular arithmetic.
For floating arithmetic use finite samples, round-to-nearest, and no fast-math.
Raw FFTs are approximate and have no general exact-integer rounding guarantee.
The integer split's supported envelope is modulus <= 2^30 and at most 2^19
points per transform. Centering and balanced splitting bound each signed half
by 2^14 in magnitude. The tests exercise the size and magnitude limits together
against exact triangular products, including opposite signs and squares, on every
target; they compare integer coefficients, never floating bit patterns. These are
numerical regression bounds, not a guarantee for arbitrary libm implementations
or floating rounding modes. Use exact NTT/CRT entries when such assumptions are
unacceptable. Serialize FFT calls that can grow the shared root cache.

The planned size is `bit_ceil(n+m-1)`, except that overlap-add uses
`bit_ceil(2*min(n,m))` when that is at most a quarter of the padded size.
A direct NTT is selected when the modulus and this planned size permit it;
then the split envelope is considered, then three-prime Garner.

## Measured performance

The race isolates radix-2 bit reversal, DIF/DIT incremental rates, lazy Montgomery
reduction, radix-4 fused levels, vector pointwise/scaling, per-call block rates
with scalar reduction, padded FFT splitting, and overlap-add prescaling. Library
rows call the entry points directly. Samples include result/scratch allocation;
batch rows sum 20000 products. The experimental by-definition product is the
certifying oracle.

Source: `benchmarks/nt/convolution.cpp`. Vector engines use targeted AVX2 in both flag
sets. Each cell is plain / native milliseconds; a dash means that shape has
no measurement for that contender. MSVC and software-wide performance are not
measured yet.

Full products through the library driver (the library column uses the scalar default):

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | library | lazy Montgomery engine | radix-4 engine | simd engine | conv8 engine |
|---|---:|---:|---:|---:|---:|
| balanced 2^19 × 2^19 | 59.5 / 62.6 | 60.1 / 64 | 57.2 / 58.3 | 24.4 / 24 | 10.2 / 10.4 |
| square 2^19 | 39.2 / 40.4 | — | 38.1 / 38.9 | 16.6 / 15.7 | 7.6 / 7.8 |
| lopsided 2^20 × 2^6 | 37.1 / 36.1 | 36.5 / 36.5 | 36.1 / 34 | 27.3 / 24.9 | — |
| batch 20000 × (64×64) | 68.9 / 70.4 | — | — | 49 / 47.8 | 14 / 14.1 |

Scalar algorithm contenders (the lazy contender pads the lopsided shape):

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | library | by radix2 | by DIF | by lazy Montgomery | by radix4 |
|---|---:|---:|---:|---:|---:|
| balanced 2^19 × 2^19 | 59.5 / 62.6 | 127 / 133 | 56.5 / 61.4 | 58.7 / 62.8 | 56 / 57.4 |
| lopsided 2^20 × 2^6 | 37.1 / 36.1 | — | — | 122 / 131 | — |

Vector pointwise/scaling and fused-block alternatives:

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | simd engine | by vector pointwise | conv8 engine | by scalar block reduce |
|---|---:|---:|---:|---:|
| balanced 2^19 × 2^19 | 24.4 / 24 | 22.9 / 22.9 | 10.2 / 10.4 | 12.1 / 12.5 |
| batch 20000 × (64×64) | 49 / 47.8 | — | 14 / 14.1 | 22.8 / 23.7 |

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | library | by prescaled spectrum | by definition |
|---|---:|---:|---:|
| lopsided 2^20 × 2^6 | 37.1 / 36.1 | 32.9 / 33.8 | — |
| batch 20000 × (48×48) | 53 / 61.5 | — | 52.5 / 55.4 |

Other moduli and exact integer products:

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| shape | library | by padded FFT split | Garner with conv8 engine |
|---|---:|---:|---:|
| lopsided 2^20 × 2^6, mod 7340033 | 36.5 / 35.5 | 303 / 280 | — |
| lopsided 2^20 × 2^6, mod 1e9+7 | 64.8 / 52.8 | 308 / 290 | — |
| 1e9+7, balanced 2^19 × 2^19 (now Garner) | not measured yet | — | — |
| 1e9+7, square 2^19 (now Garner) | not measured yet | — | — |
| 2^32−5, balanced 2^19 × 2^19 (Garner) | 192 / 205 | — | 45.6 / 45 |
| 2^32−5, square 2^19 (Garner) | 130 / 137 | — | 37.2 / 37.6 |
| uint64 three-prime, balanced 2^19 × 2^19 | 123 / 125 | — | — |
| uint64 three-prime, square 2^19 | 78.8 / 81.4 | — | — |

The 1e9+7 balanced and square 2^19 inputs have planned size 2^20, above the
FFT split's exact range, so they use three-prime Garner.
`benchmarks/nt/convolution.cpp` measures 183/191 ms balanced and 121/122 ms
square. FFT splitting is not exact there for worst-case inputs.

The dedicated large-coefficient uint64 three-prime shapes take
182/191 ms balanced and 119/123 ms square. The table's
123/125 and 78.8/81.4 ms uint64 rows are the separate small-coefficient
workloads; they do not measure the dedicated three-prime shapes.

On balanced products the scalar default loses to conv8 by
59.5/10.2 = 5.83 times plain and 62.6/10.4 = 6.02 native; simd is
2.44/2.61 times as fast. Conv8 reduces time against scalar block reduction
by 15.70%/16.80% balanced and 38.60%/40.51% on the 64×64 batch. Radix-4
beats the scalar default on every represented driver shape; lazy Montgomery
wins the lopsided driver plain (36.5 versus 37.1 ms) but loses it native
(36.5 versus 36.1 ms) and loses balanced products in both modes.

The simd engine loses to vector pointwise/scaling on balanced products
(24.4/24 versus 22.9/22.9 ms), a 6.15%/4.58% saving. The library's
lopsided product loses to prescaled spectrum (37.1/36.1 versus
32.9/33.8 ms). On the 48×48 batch it loses to by-definition multiplication
(53/61.5 versus 52.5/55.4 ms). These measurements do not support scalar
pointwise loops or per-chunk scaling.
Overlap-add is 303/36.5 = 8.30 times as fast as padded FFT split at mod
7340033 plain (280/35.5 = 7.89 native), and 4.75/5.49 times as fast at mod
1e9+7. Garner with conv8 is 4.21/4.56 times as fast as the scalar library
for balanced 2^32−5 products and 3.49/3.64 times for squares.


## Pitfalls and notes

Equal contents in separate buffers are treated as a product. Identical contiguous
storage and length, or the same range object, uses one forward transform for a
square. The integer entry's coefficient limit is smaller than UINT64_MAX.
Use modular convolution for signed values reduced modulo a static modulus.
For polynomial-series operations, see [formal power series](power-series.md).

## How it works

Scalar NTT uses DIF forward and DIT inverse, so pointwise multiplication never
needs a bit-reversal permutation. Tight Montgomery twiddles advance through
increment tables while data stays in plain lazy `[0,2p)` form. Multiplication
by a fixed twiddle uses its precomputed companion. A minimum over wrapped
unsigned sums/differences folds each butterfly without a branch.

The driver uses the measured short-side cutoff of 48, reuses a short spectrum
for overlap-add, and recognizes identical input storage. Garner obtains three
mixed-radix digits and combines them directly in the target modular type, using
published widening/modular arithmetic for wide targets. The integer entry computes
its sufficient bound exactly as a 192-bit value built from portable uint128
partial products. Below `469762049*998244353` (about 4.69e17), it retains the
two-prime reconstruction. Otherwise it adds prime 167772161 (2-adic order 25,
above the planned-size cap) and reconstructs three mixed-radix digits in
`atl::uint128` before narrowing the proven uint64 result. The largest Garner
intermediate is below the three-prime product, which is below 2^87. Small inputs
still use direct multiplication; constant evaluation uses the scalar engine
for both transform paths.

After direct NTT dispatch, the FFT tier is used only for modulus <= 2^30
and planned size <= 2^19; larger planned sizes use three-prime Garner,
which supports up to 2^23 points. The FFT tier packs balanced low/high halves
into real/imaginary components.
Conjugate symmetry separates their products; two inverse transforms and rounding
recover four integer products. Root generation anchors trigonometry every 64th
root to limit phase error from multiplication chains. Long double may have the
same precision as double; correctness tests round on x87 as well as SSE/NEON.
The fused engine stops three levels early, multiplies each block modulo
`x^8-w`, performs the coarse inverse, and scales by `1/(n/8)`. Eight-product
64-bit accumulators fit because `p < 2^30`; reduction tightens the result to
the lazy range expected by the inverse. Portable lanes use explicit low/high
word pairs, so no host-endianness assumption is needed.

References: Cooley–Tukey FFT; Harvey's lazy modular butterflies; Garner's
mixed-radix reconstruction.

## See also

- [Modular arithmetic](modular-arithmetic.md)
- [CRT](crt.md)
- [Bitwise convolutions](bitwise-convolution.md)
