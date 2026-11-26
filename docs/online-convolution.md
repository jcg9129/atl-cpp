# `atl::online_convolution` and `atl::online_fixed_convolution`

Relaxed (online) products of coefficient streams over a prime field. After the `k`-th
pair of inputs is pushed, `online_convolution` returns the coefficient `k` of the Cauchy
product `sum(i + j = k) a[i] b[j]` at once, so the next inputs may depend on the values it
returned; `online_fixed_convolution` does the same for one stream against a whole,
known kernel. Reach for them when a series is defined by itself through a product, as in
`f = 1 + x f^2`, `n f[n] = sum k g[k] f[n - k]` (the exponential and many counting
recurrences), or a convolution recurrence that no offline Newton step solves. They cost
`O(log^2 n)` amortized per push in transform-friendly fields, a logarithmic factor over the
offline product but with the feedback. When all inputs are known up front, use
`convolution` from [Convolution and transforms](convolution.md) or the Newton series
operations of [Power series](power-series.md) instead.

```cpp
#include <atl/nt/online_convolution.hpp>
```

\include online-convolution/quick.cpp

Expected stdout:
\include online-convolution/quick.out

`<atl/nt/online_convolution.hpp>` is the normal entry point, including the
wide-residue product policy. The reduced alternative
`<atl/nt/online_convolution_native.hpp>` for the native series coefficient domain
described in the power-series guide.

The examples use `<atl/nt/online_convolution_ntt.hpp>` with
`<atl/nt/mod_int_static.hpp>`. This reduced surface shares the native definitions
and requires direct NTT order at least 2^23, including 998244353. The asserted
push count and normalized kernel length are at most `maximum_series_length`
(2^22); dyadic block operands are at most half that cap. Their padded products
fit within 2^22, and every shared series product fits within 2^23. No length
precondition is added. Other primes need `<atl/nt/online_convolution_native.hpp>`
or the umbrella before instantiation; those headers keep the full fallbacks.

## Conventions and concepts

- Both classes are templates over `static_prime_mod_int Coefficient` and
  `ntt_engine Engine = default_ntt_engine`; an engine that does not support the modulus
  falls back to the convolution driver's tiers, and constant evaluation uses scalar
  transforms. Any static prime works, including the characteristic two and primes above 32
  bits (through limb products). There is no bound by the characteristic.
- Everything is `constexpr`. All transforms and caches belong to the stream, without global
  mutable storage.
- The fixed kernel of `online_fixed_convolution` is any `prime_mod_int_range` of the same
  coefficient type (vectors, arrays, spans, lists, views, even single-pass input ranges),
  read once and copied; trailing zeros are ignored, kernel coefficients beyond its length
  are zero, also after the stream outgrows it.
- Streams are copyable value types that keep every pushed coefficient, so memory grows
  with the stream.
- `maximum_series_length` (`2^22`) bounds the number of pushes and the kernel length; see
  [Power series](power-series.md).

## API

```cpp
template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
class online_convolution {
public:
    constexpr Coefficient push(Coefficient left, Coefficient right);
    constexpr void reserve(std::size_t count);
    constexpr std::size_t size() const;
};

template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
class online_fixed_convolution {
public:
    template<prime_mod_int_range Range>
        requires std::same_as<series_coefficient_t<Range>, Coefficient>
    constexpr explicit online_fixed_convolution(Range const &kernel);

    constexpr Coefficient push(Coefficient value);
    constexpr void reserve(std::size_t count);
    constexpr std::size_t size() const;
};
```

| Member | Meaning | Time |
|---|---|---|
| `online_convolution::push(a, b)` | appends `a[k], b[k]` and returns `sum(i + j = k) a[i] b[j]` for `k = size() - 1` | `O(log^2 n)` amortized, `O(M(n))` worst case |
| `online_fixed_convolution(kernel)` | copies the kernel and transforms its stationary blocks where the field has a transform | `O(M(kernel length))` |
| `online_fixed_convolution::push(a)` | appends `a[k]` and returns `sum(i + j = k) a[i] kernel[j]` | as above |
| `reserve(count)` | capacity for `count` pushes in total: the streams, the kernel's zero padding and the pending outputs (at most `min(2 count - 1, maximum_series_length)`) do not reallocate until `size() == count`; results and `size()` are unchanged | one allocation and `O(size())` copies per buffer |
| `size()` | number of pushes so far | `O(1)` |

`reserve` does not cover the retained block spectra (one pair of length `2s` per dyadic
scale `s`, allocated when the scale is first used; the fixed kernel's are built by the
constructor) or the temporaries of each block product.

\include online-convolution/partitions.cpp

Expected stdout:
\include online-convolution/partitions.out

The example computes the partition numbers from `n p(n) = sum sigma(k) p(n - k)`. The kernel
is shifted by one, so that pushing `p(n - 1)` returns the sum that determines `p(n)`; the
same shift turns the exponential `f = exp(g)` into `n f[n] = sum k g[k] f[n - k]` with the
kernel `k g[k]` at index `k - 1`.

## Complexity

For `n` pushes in a field with a transform, a push costs `O(log^2 (n + 1))` amortized field
work and `O(n log (n + 1))` in the worst case; more generally the amortized bound is
`O(sum over dyadic s <= n of M(s) / s)` with `O(M(n))` worst case, at the boundary where
every scale fires at once. Storage is `O(n)`, plus the kernel of the fixed class. All
schedules can be used with feedback, since each push needs only the current inputs.
`M` depends on the modulus: other primes go through the convolution fallbacks, and constant
evaluation can be quadratic. Allocation failures propagate.

## Preconditions

Asserted with `atl_assert` (active only under `-DATL_ASSERT`):

- `push` requires `size() < maximum_series_length`: a stream accepts at most that many pushes;
  `reserve(count)` requires `count <= maximum_series_length`.
- The normalized kernel length is at most `maximum_series_length`.
- The experimental contenders take a direct block size `cutoff` with
  `0 < cutoff <= maximum_series_length / 2`.

Documented only: the coefficient type obeys the modular-integer laws; the concepts check the
interface and the prime modulus, not the arithmetic.

## Measured performance

Each contender is an `atl::experimental` class named `<function>_by_<method>`.
The library's direct block size is 32 (blocks up to that size are multiplied by the defining
sums); the `_by_block_cutoff` contenders sweep it over 16, 24, 32, 48, 64, 96 and 128.

| Class | Library default | Contenders (`<atl/experimental/nt/online_convolution_variants.hpp>`) |
|---|---|---|
| `online_convolution` | dyadic rectangles, stationary spectra kept, direct blocks up to 32 | `online_convolution_by_block_cutoff`, `online_convolution_by_uncached_blocks`, `online_convolution_by_karatsuba_blocks`, `online_convolution_by_direct_sums` |
| `online_fixed_convolution` | the same against the kernel, kernel spectra transformed at construction | the `online_fixed_convolution_by_...` counterparts |

Each contender has the template parameters, `push` and `size` of the class it races, plus a
constructor argument `cutoff` (unused by the direct sums). The tests compare every one of
them against the defining Cauchy sum at every push. By uncached blocks transforms the
stationary blocks again for every rectangle; by Karatsuba blocks multiplies the
rectangles by recursive Karatsuba instead of transforms, costing `O(n^(log2 3 - 1))` per push
amortized and `O(n^(log2 3))` worst case; by direct sums takes `O(n)` per push.

The source is `benchmarks/nt/online_convolution.cpp`. It checks every contender against the
offline product before timing and measures a whole stream of `n` pushes (including
construction), at `n` = 2^10, 2^12, 2^16, 2^19 and 500000 modulo 998244353 and 2^16 modulo
1000000007; the quadratic direct sums run only up to 2^12.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25 (plain) and 2026-11-26 (native).
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds, plain / native;
plain: 3 runs of 10 samples per row; native: 2 runs of 10 samples per row.

The family comparisons use cutoff 32 for all block contenders. An empty cell means the
quadratic contender does not run at that length.

`online_convolution`:

| modulus | n | library | by block cutoff (32) | by uncached blocks (32) | by Karatsuba blocks (32) | by direct sums |
|---|---:|---:|---:|---:|---:|---:|
| 998244353 | 1024 | 0.368 / 0.413 | 0.367 / 0.397 | 0.454 / 0.481 | 0.664 / 0.727 | 0.651 / 0.720 |
| 998244353 | 4096 | 2.1 / 2.3 | 2.1 / 2.2 | 2.8 / 2.9 | 6.4 / 6.9 | 10.3 / 11.4 |
| 998244353 | 65536 | 55.8 / 60.0 | 55.8 / 59.6 | 82.0 / 87.0 | 543 / 594 |  |
| 998244353 | 524288 | 609 / 638 | 609 / 647 | 920 / 968 | 14853 / 16021 |  |
| 998244353 | 500000 | 536 / 569 | 534 / 565 | 815 / 859 | 10691 / 11488 |  |
| 1000000007 | 65536 | 118 / 103 | 118 / 104 | 118 / 102 | 525 / 590 |  |

`online_fixed_convolution`:

| modulus | n | library | by block cutoff (32) | by uncached blocks (32) | by Karatsuba blocks (32) | by direct sums |
|---|---:|---:|---:|---:|---:|---:|
| 998244353 | 1024 | 0.367 / 0.396 | 0.367 / 0.411 | 0.452 / 0.483 | 0.661 / 0.727 | 0.650 / 0.698 |
| 998244353 | 4096 | 2.1 / 2.2 | 2.1 / 2.2 | 2.8 / 3.0 | 6.3 / 7.0 | 10.3 / 11.1 |
| 998244353 | 65536 | 55.6 / 58.9 | 55.5 / 60.2 | 81.0 / 86.7 | 542 / 594 |  |
| 998244353 | 524288 | 608 / 641 | 608 / 646 | 916 / 981 | 14853 / 15978 |  |
| 998244353 | 500000 | 533 / 579 | 537 / 569 | 813 / 869 | 10696 / 11947 |  |
| 1000000007 | 65536 | 119 / 105 | 119 / 106 | 119 / 104 | 523 / 596 |  |

At p = 998244353 and n = 524288, keeping stationary spectra beats recomputing them:
609 / 638 versus 920 / 968 ms for two streams, and 608 / 641 versus 916 / 981 ms for a
fixed kernel. Karatsuba blocks take 14853 / 16021 and 14853 / 15978 ms respectively.
At n = 4096, direct sums take 10.3 / 11.4 ms for two streams and 10.3 / 11.1 ms for a
fixed kernel, versus the library's 2.1 / 2.3 and 2.1 / 2.2 ms.

Direct block size sweep (`full` = `online_convolution`, `fixed` = `online_fixed_convolution`):

| class | modulus | n | cutoff 16 | cutoff 24 | cutoff 32 | cutoff 48 | cutoff 64 | cutoff 96 | cutoff 128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 998244353 | 1024 | 0.350 / 0.376 | 0.350 / 0.368 | 0.367 / 0.397 | 0.367 / 0.393 | 0.448 / 0.499 | 0.451 / 0.507 | 0.636 / 0.732 |
| fixed | 998244353 | 1024 | 0.350 / 0.372 | 0.351 / 0.384 | 0.367 / 0.411 | 0.365 / 0.399 | 0.446 / 0.499 | 0.446 / 0.506 | 0.639 / 0.746 |
| full | 998244353 | 4096 | 2.0 / 2.1 | 2.0 / 2.1 | 2.1 / 2.2 | 2.1 / 2.2 | 2.4 / 2.7 | 2.4 / 2.7 | 3.3 / 3.7 |
| fixed | 998244353 | 4096 | 2.0 / 2.1 | 2.0 / 2.1 | 2.1 / 2.2 | 2.1 / 2.2 | 2.4 / 2.7 | 2.4 / 2.7 | 3.3 / 3.8 |
| full | 998244353 | 65536 | 55.5 / 58.4 | 54.7 / 59.0 | 55.8 / 59.6 | 55.9 / 59.2 | 62.4 / 67.1 | 61.5 / 67.1 | 76.5 / 84.2 |
| fixed | 998244353 | 65536 | 54.6 / 57.6 | 54.6 / 57.8 | 55.5 / 60.2 | 55.8 / 59.3 | 61.4 / 66.8 | 61.5 / 68.2 | 76.9 / 84.6 |
| full | 998244353 | 524288 | 601 / 628 | 600 / 643 | 609 / 647 | 610 / 648 | 656 / 703 | 653 / 700 | 777 / 849 |
| fixed | 998244353 | 524288 | 599 / 628 | 600 / 626 | 608 / 646 | 608 / 655 | 656 / 711 | 656 / 728 | 778 / 852 |
| full | 998244353 | 500000 | 525 / 557 | 528 / 547 | 534 / 565 | 535 / 566 | 579 / 616 | 578 / 620 | 694 / 777 |
| fixed | 998244353 | 500000 | 525 / 561 | 529 / 585 | 537 / 569 | 536 / 581 | 578 / 624 | 578 / 619 | 696 / 765 |
| full | 1000000007 | 65536 | 118 / 104 | 119 / 103 | 118 / 104 | 118 / 104 | 118 / 108 | 118 / 107 | 127 / 120 |
| fixed | 1000000007 | 65536 | 119 / 105 | 119 / 105 | 119 / 106 | 118 / 104 | 118 / 110 | 118 / 113 | 127 / 127 |

Both library classes lose to smaller direct cutoffs at every shown length modulo
998244353. For example, at n = 65536, cutoff 16 takes 55.5 / 58.4 ms versus the library's
55.8 / 60.0 ms for two streams, and 54.6 / 57.6 versus 55.6 / 58.9 ms for a fixed kernel.
The best cutoff varies with the length, class and flags; these timings do not establish
one uniform replacement for 32.

Modulo 1000000007 at n = 65536, the library loses to uncached blocks with cutoff 32 under
native flags (103 versus 102 ms for two streams, 105 versus 104 ms for a fixed kernel);
plain ties at 118 and 119 ms respectively. The fixed-kernel library also loses to cutoff
48 under both flags (119 / 105 versus 118 / 104 ms), and to cutoffs 64 and 96 under plain
flags (119 versus 118 ms).

The plain fixed-kernel block-cutoff-32 row at n = 4096 varies by 1.50x between runs;
its minimum alone does not establish a stable cutoff ranking. The plain fixed-kernel
uncached-blocks row at n = 1024 and cutoff 24 varies by 1.48x; the comparisons above use
cutoff 32 for uncached blocks.

## Pitfalls and notes

- The returned coefficient is that of index `size() - 1`, so it already contains the values
  just pushed (`a[k] b[0]` and `a[0] b[k]`). A recurrence therefore pushes the previous
  coefficient and uses the shifted kernel, as in the partition example.
- `online_fixed_convolution` needs the whole kernel before the first push; to stream two
  unknown series use `online_convolution`.
- Memory is `O(n)` for a stream of `n` pushes; discard the stream when done.
- Dividing by the index (as in the partition example or the exponential) needs `n < p`;
  the product itself has no such bound.
- The relaxed product is a logarithmic factor slower than the offline product of the same
  length. When the unknown series satisfies a Newton-friendly equation, the Newton operations
  of [Power series](power-series.md) are faster.

## How it works

**Schedule.** The positive index pairs `(i, j)` of the Cauchy product are partitioned by the
unique dyadic scale `s` with `min(i, j)` in `[s, 2s)`. At a time `t` divisible by `s` with
`t >= 2s`, the newly completed block `[t - s, t)` multiplies the stationary block `[s, 2s)`;
the first output the product touches is `t`, after every input it needs has arrived. The
square at `t = 2s` is counted once, later rectangles combine both orientations, and pairs
involving index zero are added at once. Summing `O(M(s))` per `s` pushes over all scales
gives the amortized bound; one push can trigger every scale. Zero stationary blocks skip
their rectangles, and blocks up to the direct size are multiplied by their defining sums.

**Transforms.** A supported block uses transforms of length `2s`. The diagonal costs two
forward and one inverse transform and retains both stationary spectra. Each later cached
rectangle costs two forward and one inverse transform, as the two orientations are added
before the inverse; the uncached contender takes four forward. A fixed kernel transforms its
complete stationary blocks in the constructor (one forward per block), so its diagonal
costs one new forward and one inverse and later rectangles two and one. Other fields use one
ordinary product on the diagonal and two on a later rectangle through the exact fallback
tiers. The tests pin the cached, uncached and pre-transformed schedules with a counting
engine.

**Karatsuba.** The contender keeps the same causal partition but splits each block product
into the low, the high and the summed-half products and reconstructs the middle by
subtraction, with a quadratic base case for tiny blocks.

References by name: relaxed multiplication (van der Hoeven, *Relax, but don't be too lazy*;
Fischer and Stockmeyer for the online schedule) and the semi-relaxed product with a known
operand.

## See also

- [Power series](power-series.md): the offline Newton operations and the series length cap.
- [Polynomial gcd](polynomial-gcd.md): Euclid, extended gcd, modular inverses and Pade approximants.
- [Convolution and transforms](convolution.md): the offline products used for the blocks.
- [Modular arithmetic](modular-arithmetic.md)
