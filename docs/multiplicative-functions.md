# `atl::multiplicative_fn_range`

Tabulate a multiplicative function on `[1,n)` from its values at prime powers.
Use the supplied totient and divisor functions for common tables; use the callback
when the prime-power rule depends on an exponent or a smaller table index.

```cpp
#include <atl/nt/multiplicative.hpp>
```

\include multiplicative-functions/quick.cpp

Expected stdout:
\include multiplicative-functions/quick.out

## API

```cpp
template<typename Fn, typename R>
concept partial_multiplicative_function =
    std::invocable<Fn &, uint64, uint64, uint64, std::vector<R> const &>
    && std::convertible_to<std::invoke_result_t<Fn &, uint64, uint64, uint64, std::vector<R> const &>, R>;

template<detail::multiplicative_value R, partial_multiplicative_function<R> Fn>
constexpr std::vector<R> multiplicative_fn_range(std::size_t n, Fn &&fn);
constexpr std::vector<uint32> euler_totient_range(std::size_t n);
constexpr std::vector<uint32> divisor_count_range(std::size_t n);
constexpr std::vector<uint64> divisor_sum_range(std::size_t n);
```

`fn(p,e,v,pre)` returns `f(p^e)` with `v=p^e`. `pre` has size `n`, but only
entries below `v` may be read. Callbacks are invoked as lvalues through `std::invoke_r`:
mutable/captured callables, const callables and reference wrappers are accepted.
`R` is default-initializable and copyable, directly constructible from 1, with
multiplication convertible to `R` (checked by `detail::multiplicative_value`).

```cpp
template<detail::multiplicative_value R>
    requires requires(R const &a, uint64 v) {
        R(v);
        { a + 1 } -> std::convertible_to<R>;
        { a - 1 } -> std::convertible_to<R>;
        power(R(v), uint64(0));
        requires requires(decltype(a - 1) &prime_power_step) {
            { a * prime_power_step + 1 } -> std::convertible_to<R>;
        };
    }
constexpr std::vector<R> divisor_power_sum_range(std::size_t n, uint64 k);
```

The named functions return phi(i), the divisor count, the divisor sum, and
sigma_k(i)=sum over divisors d of d^k. Every table has size n; index 0 is
value-initialized and index 1 is `R(1)` when present.

## Complexity

O(n) products plus the sum of callback costs, O(n) storage. The fixed totient,
divisor-count and divisor-sum rules take O(n) time. Divisor-power sums take
O(n log(k+2)) arithmetic operations.

## Preconditions

Asserted through the sieve: `n < 2^32`. Documented: the rule is multiplicative,
`f(1)=R(1)`, and every evaluated arithmetic operation is valid in R. For exact
integer answers all intermediate products and powers must fit; modular value
types deliberately compute in their ring. Increasing k can overflow even a
64-bit value. Empty tables do not invoke the callback.

## Measured performance

The library builds factor records then fills values. The fused-sieve contender
keeps four per-entry tables to avoid divisions, at the cost of scattered writes.

Source: `benchmarks/nt/sieve.cpp`. These measurements use native 128-bit arithmetic; software-wide timings are not measured yet.

AMD EPYC-Milan (Hetzner ccx23, 4 dedicated vCPU), x86-64 Linux,
GCC 15.2.0, RelWithDebInfo, assertions off, 2026-11-25.
Flags: `-O2 -g -DNDEBUG -std=c++23` (plain),
`-march=native -O2 -g -DNDEBUG -std=c++23` (native).
Minimum of the per-run means, milliseconds; 5 runs of 25 samples per row.

| table / n | library -O2 | by fused sieve -O2 | library native | by fused sieve native |
| --- | --- | --- | --- | --- |
| totient / 1e6 | 5.6 (1.03x) | 4.3 (1.14x) | 5 (1.16x) | 4 (1.63x) |
| divisor count / 1e6 | 5.5 (1.04x) | 4.3 (1.15x) | 4.9 (1.15x) | 4 (1.61x) |
| divisor sum / 1e6 | 5.8 (1.02x) | 5.6 (1.11x) | 5.2 (1.15x) | 4.6 (1.85x) |
| totient / 1e7 | 159 (1.01x) | 218 (1.16x) | 140 (1.07x) | 156 (1.09x) |
| divisor count / 1e7 | 158 (1.01x) | 219 (1.01x) | 139 (1.05x) | 160 (1.04x) |
| divisor sum / 1e7 | 189 (1.01x) | 254 (1.02x) | 164 (1.07x) | 190 (1.06x) |


The separate pass loses all three n=1e6 shapes to the fused sieve:
totient 5.6/5 versus 4.3/4 ms, divisor count 5.5/4.9 versus 4.3/4 ms,
and divisor sum 5.8/5.2 versus 5.6/4.6 ms. Those native fused rows vary
between runs (1.63x, 1.61x and 1.85x spread), so their minima alone do not
establish stable rankings. The separate pass wins all three n=1e7 shapes
in both modes. The fused implementation
remains a tested contender in `experimental/atl/experimental/nt/sieve_variants.hpp`.
Its callbacks have a narrower read contract: at a prime `p`, entries below `p`
are complete; at `v=p^e` with `e>1`, only entries through `v/p` (inclusive) are
guaranteed complete. For example, it evaluates `f(4)` while processing 2, when
`f(3)` is still unfilled. A recurrence using `pre[v/p]` is valid, but one using
`pre[v-1]` for higher prime powers is not. The library's separate ascending pass
supports the entire prefix below `v`.

## Pitfalls and notes

The callback sees a partially filled vector. Reading the current entry or retaining
references to a destroyed result is invalid. The sieve record table and value
table coexist, so peak storage includes both.

## How it works

For each i, the sieve supplies its least-prime power v. If i=v the callback supplies
f(i); otherwise v and i/v are coprime and f(i)=f(v)*f(i/v). Both indices are smaller
than i. Divisor-power sums use the recurrence sigma_k(p^e)=p^k*sigma_k(p^(e-1))+1.

Reference: An Introduction to the Theory of Numbers (multiplicative functions).

## See also

- [Sieve](sieve.md)
- [Modular arithmetic](modular-arithmetic.md)
