#pragma once

#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace atl {
	/**
	 * @brief A dense `rows x columns` matrix over GF(2), each row packed into
	 *        64-bit words (bit `j % 64` of word `j / 64`), for XOR systems.
	 *
	 * Run-time width rather than `std::bitset<N>`: the column count is the
	 * input's, and one elimination sweeps whole words either way.
	 */
	class bit_matrix {
		uint32 rows_ = 0, columns_ = 0, words_per_row_ = 0;
		std::vector<uint64> words_;

		static constexpr uint32 checked_dimension(std::size_t n) {
			atl_assert(n <= (std::numeric_limits<uint32>::max)());
			return uint32(n);
		}

		static constexpr std::size_t checked_word_count(uint32 rows, uint32 words) {
			atl_assert(words == 0 || std::size_t(rows) <= std::vector<uint64>().max_size() / words);
			return std::size_t(rows) * words;
		}

	public:
		/** @brief The empty 0 x 0 matrix. */
		constexpr bit_matrix() = default;

		/**
		 * @brief An all-zero `rows x columns` matrix.
		 * @pre `rows, columns < 2^32`; the packed word count fits `vector<uint64>::max_size()` (asserted).
		 */
		constexpr bit_matrix(std::size_t rows, std::size_t columns)
			: rows_(checked_dimension(rows)), columns_(checked_dimension(columns)),
			  words_per_row_(uint32((uint64(columns_) + 63) / 64)), words_(checked_word_count(rows_, words_per_row_)) {}

		/** @brief The number of rows. */
		constexpr uint32 rows() const { return rows_; }

		/** @brief The number of columns. */
		constexpr uint32 columns() const { return columns_; }

		/** @brief Entry (i, j). @pre `i < rows(), j < columns()` (asserted). */
		constexpr bool test(uint32 i, uint32 j) const {
			atl_assert(i < rows_ && j < columns_);
			return (words_[std::size_t(i) * words_per_row_ + j / 64] >> (j % 64)) & 1;
		}

		/** @brief Set entry (i, j) to `is_one`. @pre `i < rows(), j < columns()` (asserted). */
		constexpr void set(uint32 i, uint32 j, bool is_one = true) {
			atl_assert(i < rows_ && j < columns_);
			uint64 &word = words_[std::size_t(i) * words_per_row_ + j / 64];
			word = (word & ~(uint64(1) << (j % 64))) | (uint64(is_one) << (j % 64));
		}

		/** @brief Flip entry (i, j). @pre `i < rows(), j < columns()` (asserted). */
		constexpr void flip(uint32 i, uint32 j) {
			atl_assert(i < rows_ && j < columns_);
			words_[std::size_t(i) * words_per_row_ + j / 64] ^= uint64(1) << (j % 64);
		}

		/**
		 * @brief Row i's packed words; bits at or past `columns()` in the last
		 *        word must stay zero.
		 * @pre `i < rows()` (asserted).
		 */
		constexpr std::span<uint64> row_words(uint32 i) {
			atl_assert(i < rows_);
			return std::span<uint64>(words_).subspan(std::size_t(i) * words_per_row_, words_per_row_);
		}

		/** @brief Row i's packed words. @pre `i < rows()` (asserted). */
		constexpr std::span<uint64 const> row_words(uint32 i) const {
			atl_assert(i < rows_);
			return std::span<uint64 const>(words_).subspan(std::size_t(i) * words_per_row_, words_per_row_);
		}

		/** @brief Whether the shapes and packed words are equal. */
		friend constexpr bool operator==(bit_matrix const &, bit_matrix const &) = default;
	};

	/**
	 * @brief The solutions of an XOR system `A x = b` over GF(2): none, or
	 *        `particular` XOR any combination of the rows of `null_space`.
	 */
	struct bit_linear_system_solution {
		/** @brief Whether any `x` satisfies `A x = b`. */
		bool is_solvable = false;
		/** @brief One solution (free variables 0); empty when unsolvable. */
		std::vector<bool> particular;
		/** @brief A basis of the null space of `A`, one vector per row, whether or not solvable. */
		bit_matrix null_space;

		/** @brief Whether exactly one `x` satisfies `A x = b`. */
		[[nodiscard]] constexpr bool is_unique() const { return is_solvable && null_space.rows() == 0; }
	};

	namespace detail {
		/**
		 * @brief Gaussian elimination over GF(2) in place, pivoting in the first
		 *        `pivot_limit` columns; row operations are word-wide XORs from
		 *        the pivot's word on.
		 *
		 * @tparam IsReduced Clear each pivot column above as well as below
		 *         (reduced row echelon form).
		 * @return The pivot column of each of the first `rank` rows.
		 * @complexity O(rows · columns + rows · rank · ceil(columns / 64))
		 */
		template<bool IsReduced>
		constexpr std::vector<uint32> eliminate(bit_matrix &a, uint32 pivot_limit) {
			std::vector<uint32> pivot_columns;
			uint32 const rows = a.rows();
			uint32 r = 0;
			for (uint32 c = 0; c < pivot_limit && r < rows; ++c) {
				uint32 const word = c / 64;
				uint64 const bit = uint64(1) << (c % 64);
				uint32 p = r;
				while (p < rows && !(a.row_words(p)[word] & bit)) ++p;
				if (p == rows) continue;
				if (p != r) std::ranges::swap_ranges(a.row_words(p).subspan(word), a.row_words(r).subspan(word));
				auto const pivot = a.row_words(r).subspan(word);
				for (uint32 i = IsReduced ? 0 : r + 1; i < rows; ++i) {
					auto const row = a.row_words(i).subspan(word);
					if (i == r || !(row[0] & bit)) continue;
					for (std::size_t w = 0; w < row.size(); ++w) row[w] ^= pivot[w];
				}
				pivot_columns.push_back(c);
				++r;
			}
			return pivot_columns;
		}
	}

	/**
	 * @brief The rank of `a` over GF(2).
	 * @complexity O(rows · columns + rows · rank · ceil(columns / 64))
	 */
	[[nodiscard]] constexpr uint32 rank(bit_matrix a) {
		return uint32(detail::eliminate<false>(a, a.columns()).size());
	}

	/**
	 * @brief All solutions of the XOR system `A x = b`: whether one exists,
	 *        one particular solution, and a basis of the null space.
	 * @pre `b.size() == a.rows()`, `a.columns() < UINT32_MAX`, and augmented storage fits (asserted).
	 * @complexity O(rows · columns + rows · rank · ceil((columns + 1) / 64) + columns²)
	 */
	[[nodiscard]] constexpr bit_linear_system_solution solve_linear_system(bit_matrix const &a, std::vector<bool> const &b) {
		atl_assert(b.size() == a.rows());
		uint32 const rows = a.rows(), n = a.columns();
		atl_assert(n < (std::numeric_limits<uint32>::max)());
		bit_matrix augmented(rows, std::size_t(n) + 1);
		for (uint32 i = 0; i < rows; ++i) {
			std::ranges::copy(a.row_words(i), augmented.row_words(i).begin());
			augmented.set(i, n, b[i]);
		}
		auto const pivot_columns = detail::eliminate<true>(augmented, n);
		uint32 const rank = uint32(pivot_columns.size());

		bit_linear_system_solution solution;
		std::vector<bool> is_pivot(n, false);
		for (uint32 const c : pivot_columns) is_pivot[c] = true;
		solution.null_space = bit_matrix(n - rank, n);
		for (uint32 free = 0, k = 0; free < n; ++free) {
			if (is_pivot[free]) continue;
			solution.null_space.set(k, free);
			for (uint32 t = 0; t < rank; ++t)
				if (augmented.test(t, free)) solution.null_space.set(k, pivot_columns[t]);
			++k;
		}

		for (uint32 i = rank; i < rows; ++i)
			if (augmented.test(i, n)) return solution;
		solution.is_solvable = true;
		solution.particular.assign(n, false);
		for (uint32 t = 0; t < rank; ++t) solution.particular[pivot_columns[t]] = augmented.test(t, n);
		return solution;
	}
}
