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

Not measured yet. Every library choice is a provisional default and every contender is an
`atl::experimental` class named `<function>_by_<method>`; the races are pending. The
provisional direct block size is 32 (blocks up to that size are multiplied by the defining
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
1000000007; the quadratic direct sums run only up to 2^12. A winner needs x86-64 records
under both plain `-O2` and `-march=native`.

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
