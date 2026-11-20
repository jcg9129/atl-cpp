#pragma once
#include <atl/graph/assignment.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Assignment-problem contenders.
 *
 * Shared output contract: an `atl::graph::assignment_result`. The tests
 * run both against the library solver on the random, mid-size and canonical
 * cases: `assignment_by_shortest_augmenting_paths` through the full dual certificate, the
 * auction by cost and matching validity only (its duals are not returned).
 *
 * A self-contained textbook O(n^3) Hungarian is excluded: it would be one more SAP engine of the library kind, and the
 * independent oracles the tests need already exist (`min_cost_feasible_flow`
 * on the bipartite reduction, plus brute force at small n).
 * `assignment_by_shortest_augmenting_paths` below stays labelled an ablation.
 */
namespace atl::experimental {
	namespace detail {
		template<typename Sum>
		concept auction_sum = graph::assignment_sum<Sum> && requires(Sum a, Sum b) {
			{ a * b } -> std::convertible_to<Sum>;
			{ a / b } -> std::convertible_to<Sum>;
			a /= 4;
		};
	}

	/**
	 * @brief The library shortest-augmenting-path core
	 *        (`graph::detail::dense_assignment`) with the Jonker-Volgenant
	 *        preprocessing disabled: an ablation, not an independent engine.
	 *
	 * It still carries the library JV scan partition, and on wide matrices
	 * (rows < cols) the library `assignment` makes this exact call, so the
	 * two are the same code path there. Against the full JV solver
	 * (record 2026-10-15, x86-64 Xeon Platinum 8280, plain -O2, 500x500):
	 * ahead on costs 0..9 (0.74 vs 0.92 ms) and banded (0.46 vs 0.66),
	 * behind on uniform 1e9 (7.1 vs 2.8) and product mod 4096 (9.2 vs 3.0).
	 * @pre rows <= cols
	 * @complexity O(rows^2 cols + rows cols) time, O(rows cols + rows + cols) heap storage.
	 * @tparam Sum Signed integer arithmetic, or void for the element type.
	 * @tparam Matrix Const sized random-access numeric matrix.
	 * @param costs Equal-sized rows; storage is not modified or retained.
	 * @return Owned matching, cost and exact dual certificate.
	 * @pre Dimensions fit uint32 below none and the flat vector fits max_size (asserted).
	 * SAP arithmetic obeys graph::assignment's bound. Allocation failures propagate.
	 * @note Not constexpr (the experimental entry is a runtime benchmark adapter).
	 */
	template<typename Sum = void, graph::cost_matrix Matrix>
		requires(graph::assignment_sum<graph::detail::sum_t<Sum, Matrix>>
				 && graph::detail::assignment_cost<std::ranges::range_value_t<std::ranges::range_value_t<Matrix>>,
												   graph::detail::sum_t<Sum, Matrix>>)
	[[nodiscard]] auto assignment_by_shortest_augmenting_paths(Matrix const &costs) {
		using C = std::ranges::range_value_t<std::ranges::range_value_t<Matrix>>;
		std::size_t const rows = std::ranges::size(costs);
		std::size_t const cols = rows == 0 ? 0 : std::ranges::size(costs[0]);
		atl_assert(rows <= cols && rows < graph::none && cols < graph::none);
		std::vector<C> flat;
		atl_assert(cols == 0 || rows <= flat.max_size() / cols);
		flat.reserve(rows * cols);
		for (auto const &row : costs) {
			atl_assert(std::ranges::size(row) == cols);
			for (auto const &x : row) flat.push_back(x);
		}
		return graph::detail::dense_assignment<graph::detail::sum_t<Sum, Matrix>>(rows, cols, flat, false);
	}

	/**
	 * @brief Bertsekas' auction with eps-scaling — the weighted twin of the
	 *        library bipartite-matching auction: free rows bid their best
	 *        column up by (second-best - best + eps), displacing the owner.
	 *
	 * eps shrinks geometrically, and eps = 1 on integer costs scaled by n+1 is exact.
	 * Scaled costs, prices and the total run in `Sum` (default: the matrix's
	 * element type), as in the library `assignment`; pass a wider `Sum` for
	 * an int32 matrix at the 1e9 cost range.
	 *
	 * Never wins (record 2026-10-15, x86-64 Xeon Platinum 8280, plain -O2,
	 * 500x500, auction vs library JV): uniform 1e9 7.7 vs 2.8 ms, 0..9 4.4 vs
	 * 0.92, product mod 4096 6.6 vs 3.0, banded 5.3 vs 0.66.
	 * @pre square matrix, n < graph::none; every |cost| <= max(Sum) /
	 *      (4 (n+1)); every scaled cost + price, bid increment, price and
	 *      partial objective sum must fit Sum (asserted); conversions are exact
	 * @complexity pseudo-polynomial: O(log(n · max|cost| + 1)) eps phases, O(n)
	 *             per bid
	 * @tparam Sum Signed integer arithmetic, or void for the element type.
	 * @tparam Matrix Const sized random-access numeric matrix.
	 * @param costs Equal-sized rows; storage is not modified or retained.
	 * @return Owned matching and cost, with empty dual vectors.
	 * @pre Dimensions fit uint32 below none and the flat vector fits max_size (asserted).
	 * Allocation failures propagate; all walks use O(1) call stack.
	 * @note Not constexpr (the experimental entry is a runtime benchmark adapter).
	 */
	template<typename Sum = void, graph::cost_matrix Matrix>
		requires(detail::auction_sum<graph::detail::sum_t<Sum, Matrix>>
				 && graph::detail::assignment_cost<std::ranges::range_value_t<std::ranges::range_value_t<Matrix>>,
												   graph::detail::sum_t<Sum, Matrix>>)
			 && std::numeric_limits<std::ranges::range_value_t<std::ranges::range_value_t<Matrix>>>::is_integer
	[[nodiscard]] auto assignment_by_auction(Matrix const &costs) {
		using C = graph::detail::sum_t<Sum, Matrix>;
		std::size_t const n = std::ranges::size(costs);
		graph::assignment_result<C> r;
		r.row_to_col.assign(n, graph::none);
		r.col_to_row.assign(n, graph::none);
		if (n == 0) return r;
		atl_assert(std::size_t(std::ranges::size(costs[0])) == n && n < graph::none);

		if constexpr (std::numeric_limits<C>::digits < std::numeric_limits<std::size_t>::digits)
			atl_assert(n < std::size_t(std::numeric_limits<C>::max()));
		C const scale = C(C(n) + 1);
		[[maybe_unused]] C const bound = C(std::numeric_limits<C>::max() / 4 / scale);
		std::vector<C> priced;
		atl_assert(n <= priced.max_size() / n);
		priced.resize(n * n);
		C top = 1;
		for (std::size_t i = 0; i < n; ++i) {
			atl_assert(std::ranges::size(costs[i]) == n);
			for (std::size_t j = 0; j < n; ++j) {
				atl_assert(C(costs[i][j]) <= bound && C(costs[i][j]) >= -bound);
				priced[i * n + j] = C(C(costs[i][j]) * scale);
				top = std::max(top, C(priced[i * n + j] < 0 ? -priced[i * n + j] : priced[i * n + j]));
			}
		}
		std::vector<C> price(n, 0);
		std::vector<uint32> ring(n + 1);
		for (C eps = top;; eps = C(eps / 4)) {
			eps = std::max(eps, C(1));
			std::ranges::fill(r.row_to_col, graph::none);
			std::ranges::fill(r.col_to_row, graph::none);
			std::size_t hd = 0, tl = 0;
			for (uint32 i = 0; i < n; ++i) ring[tl++] = i;
			while (hd != tl) {
				uint32 const i = ring[hd];
				if (++hd == ring.size()) hd = 0;
				std::size_t j1 = 0;
				C best = std::numeric_limits<C>::max(), second = best;
				for (std::size_t j = 0; j < n; ++j) {
					atl_assert(priced[i * n + j] <= 0 || price[j] <= std::numeric_limits<C>::max() - priced[i * n + j]);
					C const value = C(priced[i * n + j] + price[j]);
					if (value < best) {
						second = best;
						best = value;
						j1 = j;
					} else {
						second = std::min(second, value);
					}
				}
				C increment = eps;
				if (second != std::numeric_limits<C>::max()) {
					atl_assert(best >= 0 || second <= std::numeric_limits<C>::max() + best);
					C const gap = C(second - best);
					atl_assert(gap <= std::numeric_limits<C>::max() - eps);
					increment = C(increment + gap);
				}
				atl_assert(price[j1] <= std::numeric_limits<C>::max() - increment);
				price[j1] = C(price[j1] + increment);
				uint32 const loser = r.col_to_row[j1];
				r.row_to_col[i] = uint32(j1);
				r.col_to_row[j1] = i;
				if (loser != graph::none) {
					r.row_to_col[loser] = graph::none;
					ring[tl] = loser;
					if (++tl == ring.size()) tl = 0;
				}
			}
			if (eps == C(1)) break;
		}
		for (std::size_t i = 0; i < n; ++i) {
			C const value = C(costs[i][r.row_to_col[i]]);
			atl_assert(value >= 0 ? r.cost <= std::numeric_limits<C>::max() - value
								  : r.cost >= std::numeric_limits<C>::lowest() - value);
			r.cost = C(r.cost + value);
		}
		return r;
	}
}
