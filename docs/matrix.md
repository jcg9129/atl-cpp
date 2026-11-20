# `atl::matrix`, Gaussian elimination and `atl::bit_matrix`

Dense matrices compose linear maps and semiring walks; elimination finds ranks,
determinants, inverses and affine solution spaces. Use packed `bit_matrix` for
XOR systems. Dense cubic algorithms suit dimensions of a few hundred; packed
systems can be much wider.

```cpp
#include <atl/nt/matrix.hpp>
#include <atl/nt/gaussian_elimination.hpp>
#include <atl/nt/bit_matrix.hpp>
```

\include matrix/quick.cpp

Expected stdout:
\include matrix/quick.out

## API

### Matrices and policies

`template<typename T, semiring Semiring = plus_times_semiring<T>> requires
std::same_as<typename Semiring::value_type, T> && (!std::same_as<T, bool>) class matrix` stores entries
row-major. `value_type` aliases T; `semiring_type` aliases Semiring.

`semiring<Semiring>` requires a semiregular `value_type` and static
`zero()`, `one()`, `add(a,b)` and `multiply(a,b)` with results convertible to it.
The complete policy is required at matrix construction, so every arithmetic
operation has both identities and both operations available. Element arithmetic
operators are unnecessary with a custom policy. Equality additionally requires
`std::equality_comparable<T>`.

Each policy has `using value_type = T` and static constexpr functions
`T zero()`, `T one()`, `T add(T const &a, T const &b)` and
`T multiply(T const &a, T const &b)`:

| policy template | constraint | add / multiply | zero / one |
|---|---|---|---|
| `template<typename T> struct plus_times_semiring` | `a+a`, `a*a` convertible to T; `T(0)`, `T(1)` valid | + / * | 0 / 1 |
| `template<typename T> struct min_plus_semiring` | `atl::integral<T> \|\| std::floating_point<T>` | min / sum preserving the sentinel | largest finite value / 0 |
| `template<typename T> struct max_plus_semiring` | `atl::signed_integral<T> \|\| std::floating_point<T>` | max / sum preserving the sentinel | lowest value / 0 |
| `template<typename T> struct max_min_semiring` | `atl::integral<T> \|\| std::floating_point<T>` | max / min | lowest value / largest value |

The numeric policies accept native and software `atl::int128` / `atl::uint128`
through atl's integral concepts and reject bool. Max-plus rejects unsigned
integers: their lowest value is 0, so `zero() == one()` and multiplication by
`one()` would fail to preserve ordinary values.

All following members and free functions are constexpr. Shapes use `std::size_t`
on construction and `uint32` for stored dimensions and indexed access.

| signature | result / meaning |
|---|---|
| `matrix()` | empty 0×0 |
| `matrix(std::size_t rows, std::size_t columns)` | every entry is policy zero |
| `matrix(std::size_t rows, std::size_t columns, T const &fill)` | every entry is fill |
| `matrix(std::initializer_list<std::initializer_list<T>> rows)` | copies rectangular rows |
| `static matrix identity(std::size_t n)` | policy one on diagonal, zero elsewhere |
| `uint32 rows() const`, `uint32 columns() const` | shape |
| `T &operator()(uint32 i, uint32 j)`, `T const &operator()(uint32 i, uint32 j) const` | indexed reference |
| `std::span<T> row(uint32 i)`, `std::span<T const> row(uint32 i) const` | borrowed contiguous row |
| `std::span<T> entries()`, `std::span<T const> entries() const` | borrowed row-major storage |
| `matrix &operator+=(matrix const &rhs)`; `matrix operator+(matrix lhs, matrix const &rhs)` | entrywise policy addition |
| `matrix operator*(matrix const &lhs, matrix const &rhs)`; `matrix &operator*=(matrix const &rhs)` | policy product |
| `std::vector<T> operator*(matrix const &lhs, std::span<T const> v)` | matrix-vector product |
| `matrix transpose() const` | swapped dimensions |
| `bool operator==(matrix const &, matrix const &) requires std::equality_comparable<T>` | shape and entries equal |

`template<typename T, semiring Semiring, unsigned_integral Exponent> constexpr
matrix<T, Semiring> power(matrix<T, Semiring> base, Exponent exp)` seeds binary
exponentiation with the dimension's identity. For min-plus, exponent k means
shortest walks of exactly k edges; for max-min it means widest such walks.

### Gaussian elimination

`exact_field<T>` checks regularity, `+ - * /` convertible to T and construction
from 0 and 1, excluding integral and floating types. The field laws, including
invertibility of every nonzero element, remain preconditions. Modular types need
a prime modulus for field elimination. Floating overloads use partial pivoting.

```cpp
template<typename T> requires(exact_field<T> || std::floating_point<T>)
struct linear_system_solution {
    bool is_solvable = false;
    std::vector<T> particular;
    matrix<T> null_space;
    constexpr bool is_unique() const;
};
```

The null-space basis has `columns(A)-rank(A)` rows of `columns(A)` entries,
including when the system is inconsistent. A solvable system has solutions
`particular + span(null_space rows)`; free variables of particular are zero.
An inconsistent system has an empty particular vector. `is_unique()` means
solvable with no null-space basis rows.

| exact-field signature (`template<exact_field T>`, constexpr) | floating signature (`template<std::floating_point T>`, constexpr) |
|---|---|
| `uint32 rank(matrix<T> a)` | `uint32 rank(matrix<T> a, T tolerance)` |
| `T determinant(matrix<T> a)` | `T determinant(matrix<T> a)` |
| `std::optional<matrix<T>> inverse(matrix<T> const &a)` | `std::optional<matrix<T>> inverse(matrix<T> const &a, T tolerance)` |
| `linear_system_solution<T> solve_linear_system(matrix<T> const &a, std::vector<T> const &b)` | `linear_system_solution<T> solve_linear_system(matrix<T> const &a, std::vector<T> const &b, T tolerance)` |

Rank counts pivots. Determinant returns T(1) for 0×0. Inverse returns nullopt
when no pivot can be found; the empty matrix has an empty inverse. Floating
pivots and consistency residuals of magnitude at most tolerance count as zero;
determinant uses zero tolerance. Underflow can also produce a zero determinant.

`template<modular_integer Z> constexpr Z determinant_any_modulus(matrix<Z> a)`
uses division-free modular row operations with integer Euclidean quotients.
It accepts composite moduli, prime powers and modulus 1. Modulus 1 returns its
only residue, including for 0×0. Field elimination on a composite modulus can
fail at a non-unit pivot; use this entry for that case.

### Packed binary matrices

`bit_matrix` stores explicit `uint64` words: bit `j%64` in word `j/64`.
Unused bits of the last word must remain zero when modifying a borrowed span.

| constexpr signature | meaning |
|---|---|
| `bit_matrix()`; `bit_matrix(std::size_t rows, std::size_t columns)` | empty / zero-filled |
| `uint32 rows() const`; `uint32 columns() const` | shape |
| `bool test(uint32 i, uint32 j) const` | read one bit |
| `void set(uint32 i, uint32 j, bool is_one = true)`; `void flip(uint32 i, uint32 j)` | modify one bit |
| `std::span<uint64> row_words(uint32 i)`; `std::span<uint64 const> row_words(uint32 i) const` | borrowed row words |
| `bool operator==(bit_matrix const &, bit_matrix const &)` | shape and packed words equal |
| `uint32 rank(bit_matrix a)` | GF(2) rank |
| `bit_linear_system_solution solve_linear_system(bit_matrix const &a, std::vector<bool> const &b)` | XOR solutions |

`bit_linear_system_solution` has `bool is_solvable = false`,
`std::vector<bool> particular`, `bit_matrix null_space` and
`constexpr bool is_unique() const`. It follows the field result's conventions;
solutions combine the basis rows by XOR.

## Complexity

For r×c matrices, dimension construction and addition cost O(rc) time;
initializer-list construction, transpose and matrix-vector product cost O(r+rc),
including zero-column shapes. Access costs O(1). A product r×k times k×c costs
O(rkc+rk+kc+rc) time including conversion and output initialization, O(rc+rk+kc+c) auxiliary storage in the modular lazy kernel and O(rc)
in the generic kernel. Power costs O(n³ log(exp+1)) time and O(n²) storage.
Matrix spans survive entry writes, but assignment, destruction and `*=` can
invalidate them. Allocation failures propagate.

Elimination costs O(rc(rank+1)) time including the copied work matrix.
Determinant and inverse cost O(n³+n²); Euclidean determinant costs
O(n³+n² log(mod+1)). Solve additionally constructs a c×(c-rank) basis,
so costs O(r(c+1)(rank+1)+c(c-rank)) time and
O(r(c+1)+c(c-rank)) storage. Inverse uses O(n²), rank/determinant O(rc).

Packed storage costs O(r ceil(c/64)) words. Bit access is O(1).
Binary rank costs O(rc+r rank ceil(c/64)) time including pivot scans;
solve costs O(r(c+1)+r rank ceil((c+1)/64)+c²) time and
O(r ceil((c+1)/64)+c ceil(c/64)) words plus O(r+c) bookkeeping.
These bounds include empty and rank-zero shapes.

## Preconditions

Asserted with `ATL_ASSERT`: dimensions below 2^32, storage products fitting the
respective vector's max_size, rectangular initializer rows, valid indices, equal
addition shapes, matching product inner dimensions, square powers/determinants/
inverses, and `b.size()==a.rows()`. Output product storage must also fit. Inverse
augmentation requires n <= UINT32_MAX/2; solve requires columns < UINT32_MAX.
All augmented and result allocations obey the same storage limits.

Documented-only: semiring laws (associative operations, commutative addition,
distributivity, additive zero annihilating multiplication), field laws, and valid
borrowed spans. Ordinary numeric-policy values and results must avoid the value
reserved for additive zero: they stay below `numeric_limits<T>::max()` for
min-plus and above `numeric_limits<T>::lowest()` for max-plus and max-min.
Max-min values are at most `numeric_limits<T>::max()`. Sentinel identities may
be used explicitly; max-min's maximum value is its multiplicative identity.
Min/max-plus sums must fit T without signed overflow and avoid their sentinel;
sentinel propagation does not clamp other sums. Floating policy values must not
be NaN. These domain restrictions are preconditions, not checked. Floating
elimination values and tolerance must be finite, with tolerance nonnegative.
Packed padding bits must stay zero.

`matrix<bool>` is rejected: `vector<bool>` cannot provide the element references,
contiguous row spans and pointers promised by the matrix API. Use `bit_matrix`
for GF(2) (XOR/AND) arithmetic, or `matrix<uint8>` with a user-defined Boolean
OR/AND semiring for Boolean reachability.

## Measured performance

The race compares lazy reduction, Montgomery arithmetic, transposing the right
factor, 256-column tiles and conditional subtraction. Each sample measures one
operation including result allocation. Shapes and agreement checks live in
`benchmarks/nt/matrix.cpp`.

Intel Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions disabled,
2026-10-14. Minimum milliseconds of three per-run means, 25 samples per row.
Plain flags: `-O2 -g -DNDEBUG -std=c++23`; native adds `-march=native`.

| contender | n=200 -O2 | n=300 -O2 | n=500 -O2 | n=200 native | n=300 native | n=500 native |
|---|---|---|---|---|---|---|
| library | 5.3 | 17.4 | 80.4 | 2.1 | 7.1 | 32.1 |
| by montgomery ikj | 12.1 | 40.9 | 190 | 12.2 | 41.0 | 189 |
| by montgomery transpose | 12.0 | 40.4 | 188 | 12.1 | 41.0 | 189 |
| by lazy transpose | 5.3 | 17.5 | 80.3 | 3.5 | 11.6 | 55.4 |
| by lazy tiles 256 | 5.2 | 18.2 | 81.5 | 1.9 | 7.0 | 32.1 |
| by lazy subtract | 7.2 | 24.0 | 110 | 2.5 | 9.5 | 42.7 |

The library loop keeps the simplest implementation: tiles give 81.5 versus
80.4 ms at n=500 with plain flags and tie at 32.1 ms with native flags.
The readable certifying oracle uses Montgomery arithmetic in i-k-j order.
The experimental implementations live in
`experimental/atl/experimental/nt/matrix_variants.hpp`.

Other operations (plain -O2 / -march=native):

| operation | -O2 | native |
|---|---|---|
| `power`, n=100, exp=1e18 | 58.6 | 23.4 |
| min-plus product, n=200 (`int64`) | 11.0 | 9.2 |
| `determinant` n=200 / n=500 | 4.1 / 62.5 | 4.1 / 62.5 |
| `determinant_any_modulus` n=200 / n=500 | 10.5 / 151 | 9.0 / 129 |
| `inverse` n=200 / n=500 | 18.8 / 297 | 18.9 / 295 |
| `bit_matrix` `solve_linear_system`, n=2000 | 33.3 | 35.2 |

The spanning-tree comparison is not published yet; its measured rows are
omitted until it is.

## Pitfalls and notes

Tolerance is absolute, so choose it for the scale of the input; floating rank
is numerical, and ill-conditioned inverses need independent residual checks.
Packed words have 64 bits even where size_t has 32. Modular products and
Euclidean determinants support 64-bit moduli using portable wide arithmetic.

## How it works

A generic i-k-j product calls the policy. Static modular values with modulus
above 1 and at most UINT32_MAX use plain residues and uint64 accumulators.
A batch contains at most `(UINT64_MAX-(p-1))/(p-1)²` products, keeping the
accumulator in range at each reduction boundary. Batches are capped by the
remaining inner dimension, including tiny moduli with very large safe batches.

Field elimination tracks pivot columns and swap parity. Unscaled forward
elimination preserves a diagonal product for determinants; Gauss-Jordan scales
pivots and clears their columns for inverses and solutions. Euclidean row
subtraction and swaps avoid modular inverses for arbitrary moduli. Within a
column the pivot decreases, giving the logarithmic term in its bound.
Binary elimination replaces scaled row subtraction with whole-word XORs;
nonpivot columns generate the null-space basis.

## See also

- [Power](power.md)
- [Modular arithmetic](modular-arithmetic.md)
- [Linear recurrences](linear-recurrence.md)
