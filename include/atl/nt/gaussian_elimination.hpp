#pragma once

#include <atl/nt/matrix.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief A field whose arithmetic is exact: `+ - * /`, `T(0)`, `T(1)`,
	 *        equality — `mod_int` / `dynamic_mod_int` with a PRIME modulus, or
	 *        a rational type. Dynamic-modulus arithmetic is runtime-only. Integers (whose `/` truncates) and floating
	 *        point (which needs pivoting and a tolerance, and has its own
	 *        overloads) are excluded.
	 *
	 * That every nonzero element is invertible is a precondition: for a
	 * modular type, a prime modulus. A composite one trips `inverse()`'s
	 * assert on the first non-unit pivot; use @ref atl::determinant_any_modulus.
	 */
	template<typename T>
	concept exact_field = std::regular<T> && !integral<T> && !std::floating_point<T>
					   && requires(T const a) {
							  { a + a } -> std::convertible_to<T>;
							  { a - a } -> std::convertible_to<T>;
							  { a * a } -> std::convertible_to<T>;
							  { a / a } -> std::convertible_to<T>;
							  T(0);
							  T(1);
						  };

	/**
	 * @brief The solutions of `A x = b`: none, or `particular + span(rows of
	 *        null_space)`.
	 *
	 * @tparam T The field element type: an @ref atl::exact_field or floating point.
	 */
	template<typename T>
		requires(exact_field<T> || std::floating_point<T>)
	struct linear_system_solution {
		/** @brief Whether any `x` satisfies `A x = b`. */
		bool is_solvable = false;
		/** @brief One solution (free variables 0); empty when unsolvable. */
		std::vector<T> particular;
		/**
		 * @brief A basis of the null space of `A`, one vector per row
		 *        (`columns(A) - rank(A)` rows of `columns(A)` entries), whether
		 *        or not the system is solvable.
		 */
		matrix<T> null_space;

		/** @brief Whether exactly one `x` satisfies `A x = b`. */
		[[nodiscard]] constexpr bool is_unique() const { return is_solvable && null_space.rows() == 0; }
	};

	namespace detail {
		/**
		 * @brief A pivot rule: `find(a, r, c)` names the row in `[r, rows)` to
		 *        pivot on in column `c` (or `rows` for none), and `is_zero(x)`
		 *        decides whether a leftover right-hand side is zero.
		 */
		template<typename Rule, typename T>
		concept pivot_rule = (exact_field<T> || std::floating_point<T>) && requires(Rule const &rule, matrix<T> const &a, uint32 i, T const &x) {
			{ rule.find(a, i, i) } -> std::same_as<uint32>;
			{ rule.is_zero(x) } -> std::same_as<bool>;
		};

		struct first_nonzero_pivot {
			/** @brief The first row in `[r, rows)` with a nonzero entry in column `c`, else `rows`. */
			template<exact_field T>
			constexpr uint32 find(matrix<T> const &a, uint32 r, uint32 c) const {
				for (uint32 i = r; i < a.rows(); ++i)
					if (!(a(i, c) == T(0))) return i;
				return a.rows();
			}

			template<exact_field T>
			constexpr bool is_zero(T const &x) const {
				return x == T(0);
			}
		};

		/**
		 * @brief Floating point: partial pivoting, the largest magnitude, with
		 *        magnitudes up to `tolerance` treated as zero.
		 */
		template<std::floating_point T>
		struct largest_magnitude_pivot {
			T tolerance;

			static constexpr T magnitude(T x) { return x < T(0) ? -x : x; }

			/** @brief The row in `[r, rows)` of largest magnitude in column `c` above `tolerance`, else `rows`. */
			constexpr uint32 find(matrix<T> const &a, uint32 r, uint32 c) const {
				uint32 best = a.rows();
				T best_magnitude = tolerance;
				for (uint32 i = r; i < a.rows(); ++i)
					if (magnitude(a(i, c)) > best_magnitude) best = i, best_magnitude = magnitude(a(i, c));
				return best;
			}

			constexpr bool is_zero(T x) const { return magnitude(x) <= tolerance; }
		};

		struct echelon_form {
			/** @brief The pivot column of each of the first `rank` rows. */
			std::vector<uint32> pivot_columns;
			/** @brief Whether an odd number of row swaps happened. */
			bool is_negated = false;
		};

		/**
		 * @brief Gaussian elimination of `a` in place, pivoting only in the first
		 *        `pivot_limit` columns (the rest ride along: right-hand sides,
		 *        an identity being inverted).
		 *
		 * @tparam IsReduced Gauss–Jordan: each pivot row is scaled to a leading
		 *         1 and its column cleared above as well as below (reduced row
		 *         echelon form). Otherwise only below, rows unscaled, so the
		 *         diagonal product is the determinant up to `is_negated`.
		 * @complexity O(rows · columns · (rank + 1))
		 */
		template<bool IsReduced, typename T, pivot_rule<T> Rule>
		constexpr echelon_form eliminate(matrix<T> &a, uint32 pivot_limit, Rule const &rule) {
			echelon_form form;
			uint32 const rows = a.rows(), columns = a.columns();
			uint32 r = 0;
			for (uint32 c = 0; c < pivot_limit && r < rows; ++c) {
				uint32 const p = rule.find(a, r, c);
				if (p == rows) continue;
				if (p != r) {
					// rows at or below r are zero left of c
					std::ranges::swap_ranges(a.row(p).subspan(c), a.row(r).subspan(c));
					form.is_negated = !form.is_negated;
				}
				T *pivot_row = a.row(r).data();
				T const inverse = T(1) / pivot_row[c];
				if constexpr (IsReduced) {
					pivot_row[c] = T(1);
					for (uint32 j = c + 1; j < columns; ++j) pivot_row[j] = pivot_row[j] * inverse;
				}
				for (uint32 i = IsReduced ? 0 : r + 1; i < rows; ++i) {
					if (i == r) continue;
					T *row = a.row(i).data();
					if (row[c] == T(0)) continue;
					T const factor = IsReduced ? row[c] : row[c] * inverse;
					row[c] = T(0);
					for (uint32 j = c + 1; j < columns; ++j) row[j] = row[j] - factor * pivot_row[j];
				}
				form.pivot_columns.push_back(c);
				++r;
			}
			return form;
		}

		template<typename T, pivot_rule<T> Rule>
		constexpr T determinant(matrix<T> &a, Rule const &rule) {
			atl_assert(a.rows() == a.columns());
			auto const form = eliminate<false>(a, a.columns(), rule);
			if (form.pivot_columns.size() < a.rows()) return T(0);
			T det = T(1);
			for (uint32 i = 0; i < a.rows(); ++i) det = det * a(i, i);
			return form.is_negated ? T(0) - det : det;
		}

		template<typename T, pivot_rule<T> Rule>
		constexpr std::optional<matrix<T>> inverse(matrix<T> const &a, Rule const &rule) {
			atl_assert(a.rows() == a.columns());
			uint32 const n = a.rows();
			atl_assert(n <= (std::numeric_limits<uint32>::max)() / 2);
			matrix<T> augmented(n, 2 * std::size_t(n));
			for (uint32 i = 0; i < n; ++i) {
				std::ranges::copy(a.row(i), augmented.row(i).begin());
				augmented(i, n + i) = T(1);
			}
			if (eliminate<true>(augmented, n, rule).pivot_columns.size() < n) return std::nullopt;
			matrix<T> result(n, n);
			for (uint32 i = 0; i < n; ++i) std::ranges::copy(augmented.row(i).subspan(n), result.row(i).begin());
			return result;
		}

		template<typename T, pivot_rule<T> Rule>
		constexpr linear_system_solution<T> solve(matrix<T> const &a, std::vector<T> const &b, Rule const &rule) {
			atl_assert(b.size() == a.rows());
			uint32 const rows = a.rows(), n = a.columns();
			atl_assert(n < (std::numeric_limits<uint32>::max)());
			matrix<T> augmented(rows, std::size_t(n) + 1);
			for (uint32 i = 0; i < rows; ++i) {
				std::ranges::copy(a.row(i), augmented.row(i).begin());
				augmented(i, n) = b[i];
			}
			auto const form = eliminate<true>(augmented, n, rule);
			uint32 const rank = uint32(form.pivot_columns.size());

			linear_system_solution<T> solution;
			std::vector<bool> is_pivot(n, false);
			for (uint32 const c : form.pivot_columns) is_pivot[c] = true;
			solution.null_space = matrix<T>(n - rank, n, T(0));
			for (uint32 free = 0, k = 0; free < n; ++free) {
				if (is_pivot[free]) continue;
				auto basis = solution.null_space.row(k++);
				basis[free] = T(1);
				for (uint32 t = 0; t < rank; ++t) basis[form.pivot_columns[t]] = T(0) - augmented(t, free);
			}

			for (uint32 i = rank; i < rows; ++i)
				if (!rule.is_zero(augmented(i, n))) return solution;
			solution.is_solvable = true;
			solution.particular.assign(n, T(0));
			for (uint32 t = 0; t < rank; ++t) solution.particular[form.pivot_columns[t]] = augmented(t, n);
			return solution;
		}
	}

	/**
	 * @brief The rank of `a` over an exact field.
	 * @complexity O(rows · columns · (rank + 1))
	 */
	template<exact_field T>
	[[nodiscard]] constexpr uint32 rank(matrix<T> a) {
		return uint32(detail::eliminate<false>(a, a.columns(), detail::first_nonzero_pivot{}).pivot_columns.size());
	}

	/**
	 * @brief The numerical rank of a floating-point `a`: partial pivoting,
	 *        with pivots of magnitude at most `tolerance` counted as zero.
	 * @pre Entries and tolerance are finite; tolerance is nonnegative.
	 * @complexity O(rows · columns · (rank + 1))
	 */
	template<std::floating_point T>
	[[nodiscard]] constexpr uint32 rank(matrix<T> a, T tolerance) {
		return uint32(
			detail::eliminate<false>(a, a.columns(), detail::largest_magnitude_pivot<T>{tolerance}).pivot_columns.size());
	}

	/**
	 * @brief The determinant of a square `a` over an exact field
	 *        (`T(1)` for the 0 x 0 matrix).
	 * @pre `a` is square (asserted).
	 * @complexity O(n³)
	 */
	template<exact_field T>
	[[nodiscard]] constexpr T determinant(matrix<T> a) {
		return detail::determinant(a, detail::first_nonzero_pivot{});
	}

	/**
	 * @brief The determinant of a square floating-point `a`, by elimination
	 *        with partial pivoting (zero pivot tolerance; determinant underflow may also produce zero).
	 * @pre Entries are finite.
	 * @pre `a` is square (asserted).
	 * @complexity O(n³)
	 */
	template<std::floating_point T>
	[[nodiscard]] constexpr T determinant(matrix<T> a) {
		return detail::determinant(a, detail::largest_magnitude_pivot<T>{T(0)});
	}

	/**
	 * @brief The inverse of a square `a` over an exact field, or `nullopt` when
	 *        `a` is singular.
	 * @pre `a` is square with dimension at most UINT32_MAX / 2 (asserted).
	 * @complexity O(n³) time and O(n²) storage.
	 */
	template<exact_field T>
	[[nodiscard]] constexpr std::optional<matrix<T>> inverse(matrix<T> const &a) {
		return detail::inverse(a, detail::first_nonzero_pivot{});
	}

	/**
	 * @brief The inverse of a square floating-point `a` (Gauss–Jordan with
	 *        partial pivoting), or `nullopt` when a pivot of magnitude above
	 *        `tolerance` cannot be found.
	 * @pre `a` is square with dimension at most UINT32_MAX / 2 (asserted).
	 * @pre Entries and tolerance are finite; tolerance is nonnegative.
	 * @complexity O(n³) time and O(n²) storage.
	 */
	template<std::floating_point T>
	[[nodiscard]] constexpr std::optional<matrix<T>> inverse(matrix<T> const &a, T tolerance) {
		return detail::inverse(a, detail::largest_magnitude_pivot<T>{tolerance});
	}

	/**
	 * @brief All solutions of `A x = b` over an exact field, for any shape of
	 *        `A`: whether one exists, one particular solution, and a basis of
	 *        the null space (so the solution set is `particular + span(basis)`).
	 * @pre `b.size() == a.rows()`, `a.columns() < UINT32_MAX`, and augmented storage fits (asserted).
	 * @complexity O(rows · (columns + 1) · (rank + 1) + columns · (columns - rank)) time;
	 * O(rows · (columns + 1) + columns · (columns - rank)) storage.
	 */
	template<exact_field T>
	[[nodiscard]] constexpr linear_system_solution<T> solve_linear_system(matrix<T> const &a, std::vector<T> const &b) {
		return detail::solve(a, b, detail::first_nonzero_pivot{});
	}

	/**
	 * @brief @ref atl::solve_linear_system over floating point: Gauss–Jordan with
	 *        partial pivoting, magnitudes up to `tolerance` treated as zero
	 *        (both for pivots and for the consistency check of leftover rows).
	 * @pre `b.size() == a.rows()`, `a.columns() < UINT32_MAX`, and augmented storage fits (asserted).
	 * @pre Entries, right-hand side and tolerance are finite; tolerance is nonnegative.
	 * @complexity O(rows · (columns + 1) · (rank + 1) + columns · (columns - rank)) time;
	 * O(rows · (columns + 1) + columns · (columns - rank)) storage.
	 */
	template<std::floating_point T>
	[[nodiscard]] constexpr linear_system_solution<T> solve_linear_system(matrix<T> const &a, std::vector<T> const &b,
																		  T tolerance) {
		return detail::solve(a, b, detail::largest_magnitude_pivot<T>{tolerance});
	}

	/**
	 * @brief The determinant of a square matrix over ANY modulus — composite,
	 *        a prime power, 1 — where elimination cannot divide by a non-unit.
	 *
	 * Euclid's algorithm on rows: within a column, the pivot row and a lower
	 * row trade `row_c -= floor(a_cc / a_rc) · row_r` and swap until the lower
	 * entry is zero; the pivot entry only shrinks, so each column costs
	 * O(n + log mod) such steps across all its rows (Wang Zhanpeng, IOI China
	 * national team papers 2020).
	 *
	 * @pre `a` is square (asserted).
	 * @complexity O(n³ + n² log mod)
	 */
	template<modular_integer Z>
	[[nodiscard]] constexpr Z determinant_any_modulus(matrix<Z> a) {
		atl_assert(a.rows() == a.columns());
		uint32 const n = a.rows();
		Z det = Z(uint64(1));
		for (uint32 c = 0; c < n; ++c) {
			for (uint32 r = c + 1; r < n; ++r) {
				while (!(a(r, c) == Z(uint64(0)))) {
					auto const quotient = a(c, c).value() / a(r, c).value();
					if (quotient != 0) {
						Z const factor = [&] {
							using Word = decltype(Z::mod());
							if constexpr (std::numeric_limits<Word>::digits <= 64) {
								return Z(uint64(quotient));
							} else {
								// The modular concept only requires construction from uint64.
								Z const radix = Z((std::numeric_limits<uint64>::max)()) + Z(uint64(1));
								return Z(uint64(quotient >> 64)) * radix + Z(uint64(quotient));
							}
						}();
						Z *pivot_row = a.row(c).data();
						Z const *row = a.row(r).data();
						for (uint32 j = c; j < n; ++j) pivot_row[j] -= factor * row[j];
					}
					std::ranges::swap_ranges(a.row(c).subspan(c), a.row(r).subspan(c));
					det = -det;
				}
			}
			if (a(c, c) == Z(uint64(0))) return Z(uint64(0));
			det *= a(c, c);
		}
		return det;
	}
}
