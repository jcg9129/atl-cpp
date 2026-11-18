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

Source: `benchmarks/nt/convolution.cpp`. Intel Xeon Platinum 8280, GCC 15.2,
RelWithDebInfo, assertions disabled, 2026-10-15. Minimum milliseconds of three
per-run means (25 samples each), `-O2 -g -DNDEBUG -std=c++23` and the same
flags plus `-march=native`. Vector engines use targeted AVX2 in both flag
sets. Each cell is plain / native milliseconds; a dash means that shape was
not recorded for that contender. MSVC and software-wide performance are not
measured yet.

Full products through the library driver (the library column uses the scalar default):

| shape | library | lazy Montgomery engine | radix-4 engine | simd engine | conv8 engine |
|---|---:|---:|---:|---:|---:|
| balanced 2^19 × 2^19 | 82.8 / 81.0 | 82.2 / 83.7 | 84.4 / 83.1 | 40.3 / 38.5 | 21.3 / 21.1 |
| square 2^19 | 54.3 / 52.1 | — | 57.2 / 55.7 | 27.3 / 25.3 | 15.4 / 15.1 |
| lopsided 2^20 × 2^6 | 48.1 / 45.8 | 47.1 / 45.2 | 48.6 / 46.4 | 38.7 / 35.9 | — |
| batch 20000 × (64×64) | 97.4 / 92.8 | — | — | 78.4 / 70.8 | 27.4 / 26.0 |

Scalar algorithm contenders (the lazy contender pads the lopsided shape):

| shape | library | by radix2 | by DIF | by lazy Montgomery | by radix4 |
|---|---:|---:|---:|---:|---:|
| balanced 2^19 × 2^19 | 82.8 / 81.0 | 156 / 155 | 80.2 / 79.9 | 80.8 / 80.9 | 83.8 / 81.6 |
| lopsided 2^20 × 2^6 | 48.1 / 45.8 | — | — | 168 / 168 | — |

Vector pointwise/scaling and fused-block alternatives:

| shape | simd engine | by vector pointwise | conv8 engine | by scalar block reduce |
|---|---:|---:|---:|---:|
| balanced 2^19 × 2^19 | 40.3 / 38.5 | 38.8 / 38.1 | 21.3 / 21.1 | 24.2 / 22.8 |
| batch 20000 × (64×64) | 78.4 / 70.8 | — | 27.4 / 26.0 | 40.2 / 35.9 |

| shape | library | by prescaled spectrum | by definition |
|---|---:|---:|---:|
| lopsided 2^20 × 2^6 | 48.1 / 45.8 | 49.4 / 44.9 | — |
| batch 20000 × (48×48) | 71.0 / 70.6 | — | 75.5 / 70.7 |

Other moduli and exact integer products:

| shape | library | by padded FFT split | Garner with conv8 engine |
|---|---:|---:|---:|
| lopsided 2^20 × 2^6, mod 7340033 | 48.3 / 45.1 | 538 / 521 | — |
| lopsided 2^20 × 2^6, mod 1e9+7 | 108 / 99.7 | 540 / 517 | — |
| 1e9+7, balanced 2^19 × 2^19 (now Garner) | not measured yet | — | — |
| 1e9+7, square 2^19 (now Garner) | not measured yet | — | — |
| 2^32−5, balanced 2^19 × 2^19 (Garner) | 266 / 259 | — | 84.6 / 82.5 |
| 2^32−5, square 2^19 (Garner) | 178 / 171 | — | 66.1 / 63.7 |
| uint64 two-prime, balanced 2^19 × 2^19 | 168 / 164 | — | — |
| uint64 two-prime, square 2^19 | 110 / 104 | — | — |

The 1e9+7 balanced and square 2^19 inputs have planned size 2^20, above the
FFT split's exact range, so they now take three-prime Garner (not measured yet).
The FFT split measured 227 / 217 ms and 183 / 171 ms on them, but is not exact
there for worst-case inputs: at planned size 2^20 thousands of coefficients
round wrongly.

The uint64 three-prime balanced and square 2^19 shapes are not measured yet.
They use the same benchmark loop and include exact widened agreement checks.

On balanced products, conv8 is 3.89× / 3.84× as fast as the scalar default,
and simd is 2.05× / 2.10× as fast (plain / native). Conv8's tuned block path
reduces time against scalar block reduction by 12.0% / 7.5% balanced and
31.8% / 27.6% on the 64×64 batch. The lazy and radix-4 driver engines have no
consistent advantage over the scalar default, which stays portable.

Vectorized pointwise/scaling lost the adoption decision: it saves only
3.7% / 1.0% over simd (40.3 → 38.8 ms plain, 38.5 → 38.1 ms native), so the
base-class scalar loops stay. Prescaled overlap-add tied across flags:
49.4 versus 48.1 ms plain (2.7% slower), 44.9 versus 45.8 ms native (2.0%
faster), so per-chunk scaling stays. Planning overlap-add instead of a padded
FFT split is 11.14× / 11.55× as fast at mod 7340033 and 5.00× / 5.19× at
mod 1e9+7. Garner with conv8 is 3.14× / 3.14× as fast for balanced products
and 2.69× / 2.68× for squares.

## Pitfalls and notes

Equal contents in separate buffers are treated as a product. Identical contiguous
storage and length, or the same range object, uses one forward transform for a
square. The integer entry's coefficient limit is smaller than UINT64_MAX.
Use modular convolution for signed values reduced modulo a static modulus.
No polynomial-series comparison is available yet; that comparison returns with
the later pages.

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
