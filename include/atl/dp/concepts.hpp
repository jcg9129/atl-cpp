#pragma once

/**
 * @file
 * @brief The callable shapes the DP-optimization family (`atl/dp/`) takes:
 *        a matrix or a segment cost given as `cost(i, j)` on `uint32`
 *        indices, ordered by a `Compare`.
 *
 * Every algorithm in the family rests on a structural law of the cost --
 * monotone row minima, total monotonicity, the quadrangle (Monge)
 * inequality -- and no concept can check such a law: these concepts pin the
 * call shape only, and each function states its law as a documented
 * precondition (just as std::sort cannot verify that its comparator is a
 * strict weak ordering).
 */

#include <atl/types/alias.hpp>
#include <concepts>
#include <type_traits>

namespace atl {
	/**
	 * @brief The value type of a cost callable: what `cost(i, j)` returns on
	 *        two `uint32` indices, with references and cv-qualifiers dropped.
	 */
	template<typename Cost>
	using cost_value_t = std::remove_cvref_t<std::invoke_result_t<Cost &, uint32, uint32>>;

	/**
	 * @brief A matrix (or a two-index cost) given as a callable: `cost(i, j)`
	 *        on two `uint32` indices returns a semiregular value that
	 *        `Compare` strictly weakly orders.
	 *
	 * Checked on `Cost &`, not `Cost const &`: a stateful cost -- a
	 * two-pointer window that moves to the queried segment, a memo -- is
	 * accepted. Each algorithm documents the order it queries entries in, so
	 * such a cost can rely on the access pattern for its amortized bound.
	 *
	 * @tparam Cost    The callable type.
	 * @tparam Compare The order on the values: `comp(a, b)` reads "a is
	 *                 better than b" (std::less<> for minima, std::greater<>
	 *                 for maxima).
	 */
	template<typename Cost, typename Compare>
	concept matrix_cost =
		std::invocable<Cost &, uint32, uint32> && std::semiregular<cost_value_t<Cost>>
		&& std::strict_weak_order<Compare const &, cost_value_t<Cost> const &, cost_value_t<Cost> const &>;

	/**
	 * @brief A @ref atl::matrix_cost whose values add: the DP solvers sum a
	 *        stored state and a transition cost (`dp[j] + cost(j, i)`), so
	 *        `a + b` on two values must yield a value. A value-initialized
	 *        `Value{}` is the empty sum (the base state's value).
	 */
	template<typename Cost, typename Compare>
	concept additive_cost = matrix_cost<Cost, Compare> && requires(cost_value_t<Cost> const a) {
		{ a + a } -> std::convertible_to<cost_value_t<Cost>>;
	};
}
