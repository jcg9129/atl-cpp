# `atl::floor_sum`, `atl::floor_staircase_product`, `atl::first_residue_in_range`

Compute sums of floors, products along a rational line's staircase, or the first
index whose affine residue reaches an interval. Euclidean reductions avoid
iterating through the entire index range.

```cpp
#include <atl/nt/floor_sum.hpp>
```

\include floor-sum/quick.cpp

Expected stdout:
\include floor-sum/quick.out

## API

```cpp
constexpr int128 floor_sum(int64 n, int64 m, int64 a, int64 b);
template<typename Monoid, typename Op>
    requires detail::staircase_monoid<Monoid, Op>
constexpr Monoid floor_staircase_product(uint64 n, uint64 m, uint64 a, uint64 b,
    Monoid up, Monoid right, Monoid const &identity, Op op);
constexpr std::optional<uint64> first_residue_in_range(uint64 m, uint64 a,
    uint64 b, uint64 lo, uint64 hi);
```

`floor_sum` returns sum over `0<=i<n` of floor((a*i+b)/m), rounding toward
negative infinity. Its int128 result and uint128 intermediates are available
on every target.

With f(x)=floor((a*x+b)/m), the staircase word is `up^f(0)`, followed for x=1..n by
`up^(f(x)-f(x-1))` and one `right`. The product follows this order; op need not
commute. Its checked concept requires a copy-constructible, move-assignable,
swappable Monoid and a const-invocable binary operation whose result converts to
Monoid. Calls use `std::invoke_r`, including reference wrappers and member pointers.

`first_residue_in_range` returns the least nonnegative x with
`lo <= (a*x+b) mod m <= hi`, or nullopt; a and b are reduced internally.
Any solution is below m, since residues repeat with period m/gcd(a,m).

## Complexity

Floor sum: O(log m) wide-arithmetic steps and O(1) storage. Staircase products:
O(log(2+max(m,a,n))+log(2+b/m)) applications of op and O(1) Monoid objects.
First residue: O(log m) wide-arithmetic steps and recursion storage. These bounds
count fixed-width wide arithmetic as constant cost, including software uint128.

## Preconditions

Asserted for floor sum: n>=0, m>=1, and the exact result fits signed int128.
Let A=floor(a/m), B=floor(b/m), P=n*(n-1)/2, and R be the reduced nonnegative
floor sum. Then 0<=R<=P and the exact result is A*P+B*n+R. A sufficient
input bound is `|A|*P+|B|*n+P <= 2^127-1`. In particular, `|a|,|b|<m` is
safe throughout the signed 64-bit n domain. The implementation checks the exact
range including cancellation, using unsigned magnitudes prior to multiplication.
It permits the negative endpoint -2^127 and never evaluates an overflowing signed
partial result. An unrepresentable answer violates the contract when assertions
are disabled too.

Asserted for staircase products: m>=1. Documented algebraic laws: op is
associative and identity is two-sided. All evaluated operations must be valid,
including arithmetic in the caller's Monoid. Unsigned line arguments cover their
full 64-bit ranges; wide products fit uint128.
Asserted for first residue: m>=1 and lo<=hi<m.

## Measured performance

Reduced-loop contenders retain 128-bit divisions throughout or choose 64/128 bits
once at entry. The library narrows once the running values shrink below 2^32.
Additional rows measure first-residue searches and a staircase sum monoid.

Source: `benchmarks/nt/floor_sum.cpp`. Intel Xeon Platinum 8358, GCC 15.2,
RelWithDebInfo, assertions disabled, 2026-10-15. Timings are milliseconds for
`-O2 -g -DNDEBUG -std=c++23` and the same flags plus `-march=native`.
Each cell is minimum milliseconds (spread = slowest run / fastest run), over
three runs of 25 samples; each run contributes its mean.
These records measure native 128-bit arithmetic; software-wide timings are not measured yet.

| workload (1e5 queries) | method | -O2 | -O2 -march=native |
| --- | --- | --- | --- |
| n,m<1e9 | library | 28.4 (1.01x) | 28.5 (1.00x) |
| n,m<1e9 | by wide divisions | 45.4 (1.05x) | 45.8 (1.00x) |
| n,m<1e9 | by entry dispatch | 26.6 (1.05x) | 26.8 (1.00x) |
| n,m<1e18 | library | 64.2 (1.01x) | 64.2 (1.00x) |
| n,m<1e18 | by wide divisions | 92.3 (1.00x) | 93.1 (1.00x) |
| n,m<1e18 | by entry dispatch | 107 (1.00x) | 108 (1.00x) |
| n<1e18,m<1e9 | library | 36.6 (1.00x) | 36.8 (1.00x) |
| n<1e18,m<1e9 | by wide divisions | 46.2 (1.00x) | 46.6 (1.00x) |
| n<1e18,m<1e9 | by entry dispatch | 53.7 (1.00x) | 53.9 (1.00x) |
| first residue m<1e18 | library | 61.8 (1.01x) | 70.7 (1.00x) |
| staircase sum n,m<1e9 | library | 112 (1.00x) | 122 (1.00x) |


Entry dispatch has the smallest small-input time. Narrowing wins both wide-input
shapes by benefiting from shrinking values. The generic staircase monoid costs
more than the dedicated floor-sum loop, and expresses aggregates the scalar loop
cannot. These records do not measure the current overflow-check implementation.

## Pitfalls and notes

The argument order is n,m,a,b for sums/products, and m,a,b,lo,hi for residues.
Floor sum uses signed line coefficients; the staircase product uses unsigned
ones. Its word spans x=1..n plus the initial climb, while floor sum spans i=0..n-1.

## How it works

Floor sum separates signed quotient contributions, then counts lattice points
rowwise instead of columnwise. Swapping step and denominator follows Euclid;
when both dimensions fall below 2^32 the remaining work fits uint64.
The staircase algorithm maintains prefix * current-word * suffix and transforms
the current line by swapping axes; binary powers compress runs of equal letters.
For first residues, reflecting the step keeps it at most half the modulus, then
recursion finds the first row with a reachable residue.

References: Jin Bin, "Applications of the Euclidean Algorithm" (2009), §4.2 (the
floor-sum identity); Dai Jiangqi (2022), §3.3 (the staircase word); Yang Xinhe, "On a
Class of Linear Congruence Inequality Problems" (2025), §3 (the least solution of
`L <= a·x mod m <= R`); Graham, Knuth and Patashnik, *Concrete Mathematics*, chapter 3
(integer functions and floor sums).

## See also

- [Extended gcd](gcd.md)
- [CRT](crt.md)
