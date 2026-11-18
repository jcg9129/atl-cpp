#pragma once

#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief A semiring policy: a stateless type naming the element type
	 *        (`value_type`) and its two operations and their identities as
	 *        static members — `add` with identity `zero()`, `multiply` with
	 *        identity `one()`.
	 *
	 * The laws (both associative, `add` commutative, `multiply` distributing
	 * over `add`, `zero()` annihilating under `multiply`) are preconditions
	 * on the caller, not checkable by a concept. A policy rather than the
	 * element's operators, so min-plus over a plain `int64` needs no wrapper
	 * type and the ordinary `+`/`*` of the same type stay usable.
	 */
	template<typename Semiring>
	concept semiring = requires { typename Semiring::value_type; }
					&& std::semiregular<typename Semiring::value_type>
					&& requires(typename Semiring::value_type const &a) {
						   { Semiring::zero() } -> std::convertible_to<typename Semiring::value_type>;
						   { Semiring::one() } -> std::convertible_to<typename Semiring::value_type>;
						   { Semiring::add(a, a) } -> std::convertible_to<typename Semiring::value_type>;
						   { Semiring::multiply(a, a) } -> std::convertible_to<typename Semiring::value_type>;
					   };

	/**
	 * @brief The ordinary ring operations of `T`: `+`, `*`, `T(0)`, `T(1)`
	 *        (integers, `mod_int`, floating point). The default semiring of
	 *        @ref atl::matrix, and the one Gaussian elimination requires.
	 */
	template<typename T>
		requires requires(T const &a) {
			{ a + a } -> std::convertible_to<T>;
			{ a * a } -> std::convertible_to<T>;
			T(0);
			T(1);
		}
	struct plus_times_semiring {
		/** @brief The policy element type. */
		using value_type = T;

		/** @brief The identity for addition. */
		static constexpr T zero() { return T(0); }

		/** @brief The identity for multiplication. */
		static constexpr T one() { return T(1); }

		/** @brief Combine two alternatives under the policy. */
		static constexpr T add(T const &a, T const &b) { return a + b; }

		/** @brief Compose two factors under the policy. */
		static constexpr T multiply(T const &a, T const &b) { return a * b; }
	};

	/**
	 * @brief The (min, +) semiring over an integer or floating `T`: shortest
	 *        walks. `zero()` is `+infinity` — `numeric_limits<T>::max()` —
	 *        which `multiply` preserves, so an unreachable entry stays
	 *        unreachable.
	 * @tparam T An atl integral or floating-point type; bool is excluded.
	 * @pre Ordinary values and sums are below the maximum sentinel; sums fit `T`
	 *      without signed overflow, and floating values are not NaN (not checked).
	 */
	template<typename T>
		requires(integral<T> || std::floating_point<T>)
	struct min_plus_semiring {
		/** @brief The policy element type. */
		using value_type = T;

		/** @brief The identity for addition. */
		static constexpr T zero() { return (std::numeric_limits<T>::max)(); }

		/** @brief The identity for multiplication. */
		static constexpr T one() { return T(0); }

		/** @brief Combine two alternatives under the policy. */
		static constexpr T add(T const &a, T const &b) { return (std::min)(a, b); }

		/** @brief Compose two factors under the policy. */
		static constexpr T multiply(T const &a, T const &b) {
			return a == zero() || b == zero() ? zero() : T(a + b);
		}
	};

	/**
	 * @brief The (max, +) semiring over a signed integer or floating `T`: longest
	 *        walks. `zero()` is `-infinity` — `numeric_limits<T>::lowest()` —
	 *        which `multiply` preserves when either operand is the sentinel.
	 * @tparam T An atl signed integral or floating-point type.
	 * @pre Ordinary values and sums are above the lowest sentinel; sums fit `T`
	 *      without signed overflow, and floating values are not NaN (not checked).
	 */
	template<typename T>
		requires(signed_integral<T> || std::floating_point<T>)
	struct max_plus_semiring {
		/** @brief The policy element type. */
		using value_type = T;

		/** @brief The identity for addition. */
		static constexpr T zero() { return std::numeric_limits<T>::lowest(); }

		/** @brief The identity for multiplication. */
		static constexpr T one() { return T(0); }

		/** @brief Combine two alternatives under the policy. */
		static constexpr T add(T const &a, T const &b) { return (std::max)(a, b); }

		/** @brief Compose two factors under the policy. */
		static constexpr T multiply(T const &a, T const &b) {
			return a == zero() || b == zero() ? zero() : T(a + b);
		}
	};

	/**
	 * @brief The (max, min) semiring over an integer or floating `T`: bottleneck (widest)
	 *        walks. `zero()` is `numeric_limits<T>::lowest()`, `one()` is
	 *        `numeric_limits<T>::max()`.
	 * @tparam T An atl integral or floating-point type; bool is excluded.
	 * @pre Ordinary values are above the lowest sentinel and at most the maximum
	 *      identity; floating values are not NaN (not checked).
	 */
	template<typename T>
		requires(integral<T> || std::floating_point<T>)
	struct max_min_semiring {
		/** @brief The policy element type. */
		using value_type = T;

		/** @brief The identity for addition. */
		static constexpr T zero() { return std::numeric_limits<T>::lowest(); }

		/** @brief The identity for multiplication. */
		static constexpr T one() { return (std::numeric_limits<T>::max)(); }

		/** @brief Combine two alternatives under the policy. */
		static constexpr T add(T const &a, T const &b) { return (std::max)(a, b); }

		/** @brief Compose two factors under the policy. */
		static constexpr T multiply(T const &a, T const &b) { return (std::min)(a, b); }
	};

	template<typename T, semiring Semiring>
		requires std::same_as<typename Semiring::value_type, T> && (!std::same_as<T, bool>)
	class matrix;

	namespace detail {
		/**
		 * @brief The element types the lazily reduced product kernel serves:
		 *        a static modulus of at most 32 bits (so a product of two
		 *        residues fits 64 bits) and above 1.
		 */
		template<typename T, typename Semiring>
		concept lazy_matrix_product = static_mod_int<T> && std::same_as<Semiring, plus_times_semiring<T>>
								   && (T::mod() <= (std::numeric_limits<uint32>::max)()) && (T::mod() > 1);

		/**
		 * @brief `c = a * b` over a modulus of at most 32 bits, i-k-j order on
		 *        plain residues with 64-bit accumulators reduced only once every
		 *        `batch` products (the largest count that cannot overflow).
		 *
		 * The residues are read out once (O(n²)) so the O(n³) loop is a plain
		 * 32x32->64 multiply-add the compiler vectorizes; Montgomery-form
		 * products per term were the raced alternative (see
		 * experimental/nt/matrix_variants.hpp): 80.4 vs 190 ms at n = 500
		 * with plain -O2, 32.1 vs 189 with -march=native
		 * (benchmarks/nt/matrix.cpp, 2026-10-14).
		 *
		 * @pre `b.size() == inner * columns`, `a.size() == rows * inner`.
		 */
		template<static_mod_int Z>
			requires lazy_matrix_product<Z, plus_times_semiring<Z>>
		constexpr std::vector<Z> lazy_product(std::span<Z const> a, std::span<Z const> b, uint32 rows,
											  uint32 inner, uint32 columns) {
			constexpr uint64 mod = Z::mod();
			constexpr uint64 square = (mod - 1) * (mod - 1);
			// accumulators stay below mod at each reduction boundary, so `batch` more
			// products are safe while (mod - 1) + batch * square <= 2^64 - 1
			constexpr uint64 batch = square == 0 ? (std::numeric_limits<uint64>::max)()
												 : ((std::numeric_limits<uint64>::max)() - (mod - 1)) / square;
			std::vector<uint32> lhs(a.size()), rhs(b.size());
			for (std::size_t i = 0; i < a.size(); ++i) lhs[i] = uint32(a[i].value());
			for (std::size_t i = 0; i < b.size(); ++i) rhs[i] = uint32(b[i].value());
			atl_assert(columns == 0 || std::size_t(rows) <= std::vector<Z>().max_size() / columns);
			atl_assert(a.size() == std::size_t(rows) * inner && b.size() == std::size_t(inner) * columns);
			std::vector<Z> c(std::size_t(rows) * columns);
			if (rows == 0 || inner == 0 || columns == 0) return c;
			std::vector<uint64> accumulator(columns);
			for (uint32 i = 0; i < rows; ++i) {
				std::ranges::fill(accumulator, uint64(0));
				uint32 const *lhs_row = lhs.data() + std::size_t(i) * inner;
				uint64 *acc = accumulator.data();
				for (uint32 k0 = 0; k0 < inner;) {
					uint32 const k1 = k0 + uint32((std::min<uint64>)(batch, inner - k0));
					for (uint32 k = k0; k < k1; ++k) {
						uint64 const x = lhs_row[k];
						uint32 const *rhs_row = rhs.data() + std::size_t(k) * columns;
						for (uint32 j = 0; j < columns; ++j) acc[j] += x * rhs_row[j];
					}
					k0 = k1;
					if (k0 < inner)
						for (uint32 j = 0; j < columns; ++j) acc[j] %= mod;
				}
				Z *out = c.data() + std::size_t(i) * columns;
				for (uint32 j = 0; j < columns; ++j) out[j] = Z::unchecked(acc[j] % mod);
			}
			return c;
		}
	}

	/**
	 * @brief A dense `rows x columns` matrix over a semiring, stored row-major
	 *        in one flat array.
	 *
	 * The entries are ordinary values of `T`; the semiring decides what the
	 * product means: `matrix<mod_int32<P>>` is the usual ring product,
	 * `matrix<int64, min_plus_semiring<int64>>` composes shortest walks
	 * (`power(m, k)` = shortest walks of exactly k edges). `atl::power`
	 * works on it through its identity-seeded form (see @ref atl::power).
	 * The two-argument dimension constructor fills with `Semiring::zero()`.
	 *
	 * @tparam T A semiregular stored value other than bool; equality is needed only for matrix equality.
	 * @note Borrowed spans/references survive entry writes; assignment, destruction and `*=` may invalidate them.
	 * Allocation failures propagate.
	 * @note Dynamic-modulus arithmetic uses runtime state and cannot be constant-evaluated.
	 * @tparam Semiring The semiring policy (see @ref atl::semiring); defaults to
	 *                  `T`'s own `+` and `*`.
	 */
	template<typename T, semiring Semiring = plus_times_semiring<T>>
		requires std::same_as<typename Semiring::value_type, T> && (!std::same_as<T, bool>)
	class matrix {
		uint32 rows_ = 0, columns_ = 0;
		std::vector<T> entries_;

		static constexpr uint32 checked_dimension(std::size_t n) {
			atl_assert(n <= (std::numeric_limits<uint32>::max)());
			return uint32(n);
		}

		static constexpr std::size_t checked_entry_count(std::size_t rows, std::size_t columns) {
			atl_assert(columns == 0 || rows <= std::vector<T>().max_size() / columns);
			return rows * columns;
		}

	public:
		/** @brief The stored element type. */
		using value_type = T;
		/** @brief The policy defining both operations and identities. */
		using semiring_type = Semiring;

		/** @brief The empty 0 x 0 matrix. */
		constexpr matrix() = default;

		/**
		 * @brief A `rows x columns` matrix of `Semiring::zero()`.
		 * @pre `rows, columns < 2^32` and their product fits `vector<T>::max_size()` (asserted).
		 */
		constexpr matrix(std::size_t rows, std::size_t columns)
			: matrix(rows, columns, T(Semiring::zero())) {}

		/**
		 * @brief A `rows x columns` matrix with every entry `fill`.
		 * @pre `rows, columns < 2^32` and their product fits `vector<T>::max_size()` (asserted).
		 */
		constexpr matrix(std::size_t rows, std::size_t columns, T const &fill)
			: rows_(checked_dimension(rows)), columns_(checked_dimension(columns)),
			  entries_(checked_entry_count(rows, columns), fill) {}

		/**
		 * @brief A matrix from nested rows: `{{1, 2}, {3, 4}}`.
		 * @pre Every row has the same length; dimensions and storage fit (asserted).
		 */
		constexpr matrix(std::initializer_list<std::initializer_list<T>> rows)
			: rows_(checked_dimension(rows.size())),
			  columns_(checked_dimension(rows.size() == 0 ? 0 : rows.begin()->size())) {
			entries_.reserve(checked_entry_count(rows_, columns_));
			for (auto const &row : rows) {
				atl_assert(row.size() == columns_);
				entries_.insert(entries_.end(), row.begin(), row.end());
			}
		}

		/** @brief The `n x n` identity: `one()` on the diagonal, `zero()` elsewhere.
		 * @param n The number of rows and columns.
		 * @pre n fits uint32 and n² fits vector storage (asserted).
		 * @return An independently owned identity matrix.
		 * @complexity O(n²) time and storage.
		 */
		[[nodiscard]] static constexpr matrix identity(std::size_t n) {
			matrix m(n, n);
			for (uint32 i = 0; i < m.rows_; ++i) m(i, i) = T(Semiring::one());
			return m;
		}

		/** @brief The number of rows. */
		constexpr uint32 rows() const { return rows_; }

		/** @brief The number of columns. */
		constexpr uint32 columns() const { return columns_; }

		/** @brief Entry (i, j). @pre `i < rows(), j < columns()` (asserted). */
		constexpr T &operator()(uint32 i, uint32 j) {
			atl_assert(i < rows_ && j < columns_);
			return entries_[std::size_t(i) * columns_ + j];
		}

		/** @brief Entry (i, j). @pre `i < rows(), j < columns()` (asserted). */
		constexpr T const &operator()(uint32 i, uint32 j) const {
			atl_assert(i < rows_ && j < columns_);
			return entries_[std::size_t(i) * columns_ + j];
		}

		/** @brief Row i as a contiguous span. @pre `i < rows()` (asserted). */
		constexpr std::span<T> row(uint32 i) {
			atl_assert(i < rows_);
			return std::span<T>(entries_).subspan(std::size_t(i) * columns_, columns_);
		}

		/** @brief Row i as a contiguous span. @pre `i < rows()` (asserted). */
		constexpr std::span<T const> row(uint32 i) const {
			atl_assert(i < rows_);
			return std::span<T const>(entries_).subspan(std::size_t(i) * columns_, columns_);
		}

		/** @brief All entries, row-major. */
		constexpr std::span<T> entries() { return entries_; }

		/** @brief All entries, row-major. */
		constexpr std::span<T const> entries() const { return entries_; }

		/** @brief Entrywise semiring addition. @pre Same shape (asserted). */
		constexpr matrix &operator+=(matrix const &rhs) {
			atl_assert(rows_ == rhs.rows_ && columns_ == rhs.columns_);
			for (std::size_t i = 0; i < entries_.size(); ++i)
				entries_[i] = T(Semiring::add(entries_[i], rhs.entries_[i]));
			return *this;
		}

		/** @brief Entrywise semiring addition. @pre Same shape (asserted). */
		friend constexpr matrix operator+(matrix lhs, matrix const &rhs) { return lhs += rhs; }

		/**
		 * @brief The semiring product. A modular `T` of at most 32 bits under
		 *        the default semiring takes a lazily reduced kernel; everything
		 *        else the generic i-k-j loop.
		 * @pre `lhs.columns() == rhs.rows()` and output storage fits (asserted).
		 * @complexity O(rows · inner · columns + rows · inner + inner · columns + rows · columns)
		 */
		friend constexpr matrix operator*(matrix const &lhs, matrix const &rhs) {
			atl_assert(lhs.columns_ == rhs.rows_);
			matrix c;
			(void)checked_entry_count(lhs.rows_, rhs.columns_);
			c.rows_ = lhs.rows_;
			c.columns_ = rhs.columns_;
			if constexpr (detail::lazy_matrix_product<T, Semiring>) {
				c.entries_ = detail::lazy_product<T>(lhs.entries_, rhs.entries_, lhs.rows_, lhs.columns_,
													 rhs.columns_);
			} else {
				uint32 const inner = lhs.columns_, columns = rhs.columns_;
				c.entries_.assign(checked_entry_count(c.rows_, columns), T(Semiring::zero()));
				if (c.rows_ == 0 || inner == 0 || columns == 0) return c;
				for (uint32 i = 0; i < c.rows_; ++i) {
					T *out = c.entries_.data() + std::size_t(i) * columns;
					for (uint32 k = 0; k < inner; ++k) {
						T const x = lhs.entries_[std::size_t(i) * inner + k];
						T const *rhs_row = rhs.entries_.data() + std::size_t(k) * columns;
						for (uint32 j = 0; j < columns; ++j)
							out[j] = T(Semiring::add(out[j], Semiring::multiply(x, rhs_row[j])));
					}
				}
			}
			return c;
		}

		/** @brief `*this = *this * rhs`. @pre `columns() == rhs.rows()` (asserted). */
		constexpr matrix &operator*=(matrix const &rhs) { return *this = *this * rhs; }

		/**
		 * @brief The matrix-vector product `lhs · v`.
		 * @pre `v.size() == lhs.columns()` (asserted).
		 * @complexity O(rows · (columns + 1))
		 */
		friend constexpr std::vector<T> operator*(matrix const &lhs, std::span<T const> v) {
			atl_assert(v.size() == lhs.columns_);
			std::vector<T> out(lhs.rows_, T(Semiring::zero()));
			if (lhs.columns_ == 0) return out;
			for (uint32 i = 0; i < lhs.rows_; ++i) {
				T const *row = lhs.entries_.data() + std::size_t(i) * lhs.columns_;
				T sum = T(Semiring::zero());
				for (uint32 j = 0; j < lhs.columns_; ++j) sum = T(Semiring::add(sum, Semiring::multiply(row[j], v[j])));
				out[i] = sum;
			}
			return out;
		}

		/** @brief The transpose. @complexity O(rows + rows · columns) */
		[[nodiscard]] constexpr matrix transpose() const {
			matrix t(columns_, rows_);
			for (uint32 i = 0; i < rows_; ++i)
				for (uint32 j = 0; j < columns_; ++j)
					t.entries_[std::size_t(j) * rows_ + i] = entries_[std::size_t(i) * columns_ + j];
			return t;
		}

		/** @brief Whether shapes and all corresponding entries are equal. */
		friend constexpr bool operator==(matrix const &, matrix const &)
			requires requires(T const &x) {
				{ x == x } -> std::convertible_to<bool>;
			}
		= default;
	};

	/**
	 * @brief `base` raised to `exp` under its semiring's product, by
	 *        `atl::power` seeded with the identity of `base`'s dimension.
	 * @pre `base` is square (asserted).
	 * @complexity O(n³ log exp)
	 */
	template<typename T, semiring Semiring, unsigned_integral Exponent>
	[[nodiscard]] constexpr matrix<T, Semiring> power(matrix<T, Semiring> base, Exponent exp) {
		atl_assert(base.rows() == base.columns());
		auto identity = matrix<T, Semiring>::identity(base.rows());
		return power(std::move(base), exp, std::move(identity));
	}
}
