#pragma once

/**
 * @file
 * @brief Row minima of an implicit matrix whose minima move monotonically:
 *        @ref atl::monotone_row_minima (divide and conquer, monotone matrices) and
 *        @ref atl::totally_monotone_row_minima (SMAWK, totally monotone
 *        matrices).
 */

#include <atl/dp/concepts.hpp>
#include <atl/dp/detail/wide.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Each row's best entry: `column[i]` is the leftmost column whose
	 *        value is Compare-best in row `i`, and `value[i]` that value.
	 */
	template<std::semiregular Value>
	struct row_minima_result {
		/** @brief `column[i]`: the leftmost Compare-best column of row `i`. */
		std::vector<uint32> column;
		/** @brief `value[i]`: the entry of row `i` at `column[i]`. */
		std::vector<Value> value;
	};

	namespace detail {
		/**
		 * @brief Whether an index count fits the family's `uint32` indices
		 *        with one value to spare (`uint32(-1)` stays free as a
		 *        sentinel).
		 */
		constexpr bool fits_index(std::size_t count) {
			return count < std::size_t(std::numeric_limits<uint32>::max());
		}

		/**
		 * @brief A per-row column bound for `monotone_minima_recurse`:
		 *        `last(row)` is the largest column row `row` may take.
		 */
		template<typename Last>
		concept column_bound = std::regular_invocable<Last const &, uint32>
							&& std::convertible_to<std::invoke_result_t<Last const &, uint32>, uint32>;

		/**
		 * @brief The sink `monotone_minima_recurse` reports each row's
		 *        leftmost best column and value to.
		 */
		template<typename Store, typename Value>
		concept minimum_sink = std::invocable<Store &, uint32, uint32, Value &&>;

		/**
		 * @brief The divide-and-conquer core: rows `[row_lo, row_hi)` with
		 *        candidate columns `[col_lo, col_hi]`, each row `i` further
		 *        capped at `last(i)`. The middle row is scanned left to right
		 *        (keeping the leftmost best), and its winner splits the column
		 *        range for the two halves.
		 *
		 * The cap lets a caller run a staircase matrix -- row `i` defined only
		 * on columns `<= last(i)`, `last` nondecreasing -- without padding the
		 * undefined entries with an infinity: the scanned range always stays
		 * non-empty, as the left bound is a smaller row's winner, which is at
		 * most that row's cap.
		 *
		 * @pre the leftmost best columns are nondecreasing over the rows;
		 *      `last` nondecreasing with `last(i) >= col_lo` on every row
		 * @complexity O((rows + columns) log rows) evaluations
		 */
		template<typename Cost, typename Compare, column_bound Last, minimum_sink<cost_value_t<Cost>> Store>
			requires matrix_cost<Cost, Compare>
		constexpr void monotone_minima_recurse(uint32 row_lo, uint32 row_hi, uint32 col_lo, uint32 col_hi,
											   Cost &cost, Compare const &comp, Last const &last, Store &store) {
			using Value = cost_value_t<Cost>;
			while (row_lo < row_hi) {
				uint32 const mid = row_lo + (row_hi - row_lo) / 2;
				uint32 const hi = (std::min)(col_hi, uint32(last(mid)));
				uint32 best = col_lo;
				Value best_value = cost(mid, col_lo);
				for (uint32 c = col_lo + 1; c <= hi; ++c) {
					Value v = cost(mid, c);
					if (std::invoke(comp, v, best_value)) {
						best_value = std::move(v);
						best = c;
					}
				}
				store(mid, best, std::move(best_value));
				// the smaller half recurses and the larger one loops, so the
				// depth stays below log2(rows) + 1
				if (mid - row_lo < row_hi - mid - 1) {
					monotone_minima_recurse(row_lo, mid, col_lo, best, cost, comp, last, store);
					row_lo = mid + 1;
					col_lo = best;
				} else {
					monotone_minima_recurse(mid + 1, row_hi, best, col_hi, cost, comp, last, store);
					row_hi = mid;
					col_hi = best;
				}
			}
		}

		/**
		 * @brief An evaluator for `smawk`: `eval(row, column)` on two
		 *        `uint32` ids yields a semiregular value.
		 */
		template<typename Eval>
		concept smawk_evaluator =
			std::invocable<Eval &, uint32, uint32>
			&& std::semiregular<std::remove_cvref_t<std::invoke_result_t<Eval &, uint32, uint32>>>;

		template<smawk_evaluator Eval>
		using smawk_value_t = std::remove_cvref_t<std::invoke_result_t<Eval &, uint32, uint32>>;

		/**
		 * @brief SMAWK over an implicit totally monotone matrix, shared by
		 *        @ref atl::totally_monotone_row_minima, the experimental 1D/1D
		 *        contender `optimal_segmentation_by_smawk_blocks` and the
		 *        experimental `optimal_partition_by_smawk`.
		 *
		 * Rows are the positions `0 .. rows-1`; columns are the ids in the
		 * initial column list (increasing, any ids). `eval(row, column)`
		 * returns a value and `better(a, b)` says whether `a` is strictly
		 * better; `answer[row]` / `value[row]` receive each row's leftmost
		 * best column and its value.
		 *
		 * One level, on the rows `first, first + step, ...` (`count` of them):
		 * - REDUCE, when there are more columns than rows: a left-to-right
		 *   stack pass keeps at most one column per row. A column that is
		 *   strictly better than the stack top in the top's row kills the top
		 *   (total monotonicity: the top can then be best in no row at or
		 *   below), otherwise it is pushed if the stack has room (it can be
		 *   best only below the top's row). Ties keep the left column, so the
		 *   survivors include every row's LEFTMOST best.
		 * - recurse on the odd rows with the survivors;
		 * - each even row scans the survivors between its neighbours' answers.
		 * The stack caches each entry's value in its own row, so a comparison
		 * evaluates one new entry. The recursion halves the rows, so its depth
		 * is at most log2(rows) + 1.
		 *
		 * @complexity O(rows + columns) evaluations
		 */
		template<smawk_evaluator Eval, typename Better>
			requires std::strict_weak_order<Better const &, smawk_value_t<Eval> const &, smawk_value_t<Eval> const &>
		class smawk {
			using value_type = smawk_value_t<Eval>;

			Eval &eval_;
			Better const &better_;
			uint32 *answer_;
			value_type *value_;
			std::vector<uint32> columns_;
			std::vector<value_type> stack_values_;

			/**
			 * @brief One SMAWK level on the rows `first + k * step`, `k <
			 *        count`, over the column ids `columns_[col_begin, col_end)`;
			 *        REDUCE appends its survivors past the end and truncates
			 *        them on return.
			 * @pre the listed columns hold every listed row's leftmost best
			 * @complexity O(count + (col_end - col_begin)) evaluations, plus
			 *             the recursion on half the rows
			 */
			constexpr void solve(uint32 first, uint32 step, uint32 count, std::size_t col_begin, std::size_t col_end) {
				if (count == 0) return;
				std::size_t const mark = columns_.size();
				if (col_end - col_begin > count) {
					for (std::size_t at = col_begin; at < col_end; ++at) {
						uint32 const c = columns_[at];
						std::size_t top = columns_.size() - mark;
						while (top > 0) {
							value_type v = eval_(first + uint32(top - 1) * step, c);
							if (!std::invoke(better_, v, stack_values_[top - 1])) break;
							columns_.pop_back();
							--top;
						}
						if (top < count) {
							stack_values_[top] = eval_(first + uint32(top) * step, c);
							columns_.push_back(c);
						}
					}
					col_begin = mark;
					col_end = columns_.size();
				}
				// odd rows first; the step only doubles when the next level has
				// two rows or more, so it stays below 2^32
				if (count >= 2) solve(first + step, count >= 4 ? step * 2 : step, count / 2, col_begin, col_end);
				std::size_t at = col_begin;
				for (uint32 k = 0; k < count; k += 2) {
					uint32 const row = first + k * step;
					uint32 const stop = k + 1 < count ? answer_[row + step] : columns_[col_end - 1];
					uint32 best = columns_[at];
					value_type best_value = eval_(row, best);
					while (columns_[at] != stop) {
						++at;
						value_type v = eval_(row, columns_[at]);
						if (std::invoke(better_, v, best_value)) {
							best_value = std::move(v);
							best = columns_[at];
						}
					}
					answer_[row] = best;
					value_[row] = std::move(best_value);
				}
				columns_.resize(mark);
			}

		public:
			/**
			 * @brief Bind the solver to its evaluator, order and outputs
			 *        (`answer` and `value` hold one slot per row).
			 */
			constexpr smawk(Eval &eval, Better const &better, uint32 *answer, value_type *value)
				: eval_(eval), better_(better), answer_(answer), value_(value) {}

			/**
			 * @brief Row minima of rows `0 .. rows-1` over the increasing
			 *        column ids in @p columns.
			 * @pre `rows < 2^32`, `columns` non-empty when `rows > 0`, the
			 *      matrix totally monotone
			 */
			constexpr void run(uint32 rows, std::vector<uint32> columns) {
				if (rows == 0) return;
				atl_assert(!columns.empty());
				std::size_t const n = columns.size();
				columns_ = std::move(columns);
				columns_.reserve(n + 2 * std::size_t(rows) + 2);
				stack_values_.resize(rows);
				solve(0, 1, rows, 0, n);
			}
		};
	}

	/**
	 * @brief Row minima of a monotone matrix by divide and conquer: for each
	 *        row of the implicit `rows x cols` matrix `cost(i, j)`, its
	 *        leftmost Compare-best column and value.
	 *
	 * The precondition is the weakest of the family: the matrix is
	 * MONOTONE -- each row's leftmost best column is at least the previous
	 * row's. Total monotonicity (and so the quadrangle inequality) implies
	 * it; this is the engine of the divide-and-conquer optimization for
	 * layered DPs (`dp_k[i] = min_j dp_{k-1}[j] + w(j, i)`, one call per
	 * layer). The middle row of a range is scanned over its column window
	 * left to right, so a two-pointer cost moves O((rows + cols) log rows)
	 * steps in total. The recursion descends into the smaller half and
	 * loops on the larger one, so its depth is at most log2(rows) + 1.
	 *
	 * Ties: each row gets its LEFTMOST best column. Only the leftmost best
	 * columns need to be nondecreasing; other best columns may move freely.
	 *
	 * With `Compare = std::greater<>` the result is the row maxima, and the
	 * precondition reads with "best" = largest.
	 *
	 * @param rows, cols The matrix shape (`< 2^32 - 1` each, asserted;
	 *        `cols >= 1` when `rows >= 1`, asserted).
	 * @param cost `cost(i, j)` for `i < rows`, `j < cols`.
	 * @param comp The order: `comp(a, b)` is "a is better than b".
	 * @pre the leftmost best column is nondecreasing over the rows (not
	 *      checkable; documented)
	 * @complexity O((rows + cols) log rows) evaluations of `cost`
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires matrix_cost<Cost, Compare>
	[[nodiscard]] constexpr row_minima_result<cost_value_t<Cost>> monotone_row_minima(std::size_t rows, std::size_t cols,
																					  Cost &&cost, Compare comp = {}) {
		using value_type = cost_value_t<Cost>;
		atl_assert(detail::fits_index(rows) && detail::fits_index(cols));
		atl_assert(rows == 0 || cols > 0);
		row_minima_result<value_type> result;
		result.column.resize(rows);
		result.value.resize(rows);
		if (rows == 0) return result;
		auto const last = [last_column = uint32(cols - 1)](uint32) { return last_column; };
		auto store = [&result](uint32 row, uint32 column, value_type &&value) {
			result.column[row] = column;
			result.value[row] = std::move(value);
		};
		detail::monotone_minima_recurse(0, uint32(rows), 0, uint32(cols - 1), cost, comp, last, store);
		return result;
	}

	/**
	 * @brief Row minima of a totally monotone matrix by SMAWK: for each row
	 *        of the implicit `rows x cols` matrix `cost(i, j)`, its leftmost
	 *        Compare-best column and value, in a linear number of
	 *        evaluations.
	 *
	 * The precondition is TOTAL monotonicity: in every 2x2 submatrix (rows
	 * `i < i'`, columns `j < j'`), if column `j'` is strictly better than
	 * `j` in row `i` then it is strictly better in row `i'`, and if it is at
	 * least as good in row `i` then it is at least as good in row `i'`. A
	 * Monge matrix (`A[i][j] + A[i'][j'] <= A[i][j'] + A[i'][j]` -- the
	 * quadrangle inequality) satisfies it. Monotone argmins alone are not
	 * enough: SMAWK discards columns by 2x2 comparisons that only total
	 * monotonicity makes safe. Entries are queried in no locality-friendly
	 * order, so a stateful cost gets no amortization here.
	 *
	 * Ties: each row gets its LEFTMOST best column, as with
	 * @ref atl::monotone_row_minima, so the two agree on every totally monotone
	 * matrix. The recursion halves the rows: depth at most log2(rows) + 1.
	 *
	 * With `Compare = std::greater<>` the result is the row maxima, and the
	 * precondition reads with "better" = larger (an inverse-Monge matrix).
	 *
	 * @param rows, cols The matrix shape (`< 2^32 - 1` each, asserted;
	 *        `cols >= 1` when `rows >= 1`, asserted).
	 * @param cost `cost(i, j)` for `i < rows`, `j < cols`.
	 * @param comp The order: `comp(a, b)` is "a is better than b" (called
	 *        through `std::invoke`, as everywhere in the family).
	 * @pre the matrix is totally monotone (not checkable; documented); the
	 *      value type is not `bool` (rejected: the minima are written through
	 *      a raw pointer into the result)
	 * @complexity O(rows + cols) evaluations of `cost`, O(rows + cols) memory
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires matrix_cost<Cost, Compare> && detail::contiguously_storable<cost_value_t<Cost>>
	[[nodiscard]] constexpr row_minima_result<cost_value_t<Cost>> totally_monotone_row_minima(std::size_t rows, std::size_t cols,
																							  Cost &&cost, Compare comp = {}) {
		atl_assert(detail::fits_index(rows) && detail::fits_index(cols));
		atl_assert(rows == 0 || cols > 0);
		row_minima_result<cost_value_t<Cost>> result;
		result.column.resize(rows);
		result.value.resize(rows);
		if (rows == 0) return result;
		std::vector<uint32> columns(cols);
		for (uint32 j = 0; j < cols; ++j) columns[j] = j;
		detail::smawk<std::remove_reference_t<Cost>, Compare> solver(cost, comp, result.column.data(),
																	 result.value.data());
		solver.run(uint32(rows), std::move(columns));
		return result;
	}
}
