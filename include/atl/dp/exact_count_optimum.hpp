#pragma once

/**
 * @file
 * @brief @ref atl::exact_count_optimum -- the exact optimum of a problem constrained
 *        to use exactly `k` items (segments, chosen elements, ...) by
 *        Lagrangian relaxation -- WQS binary search --
 *        given a solver of the unconstrained problem with a price per item.
 */

#include <atl/dp/detail/wide.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/floor_midpoint.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace atl {
	namespace detail {
		/** @brief Whether `T` is a two-element tuple-like type (`std::pair`, a two-element `std::tuple`). */
		template<typename T>
		concept pair_like = requires(T const &t) {
			typename std::tuple_size<T>::type;
			requires std::tuple_size_v<T> == 2;
			std::get<0>(t);
			std::get<1>(t);
		};
	}

	/**
	 * @brief A penalized solver for @ref atl::exact_count_optimum -- `solve(price)`
	 *        on a `Price` returns a pair-like `{value, count}` -- the best
	 *        penalized value and the item count of the solution achieving it,
	 *        an integer.
	 *
	 * Checked on `Solve &`, so a solver that reuses buffers across calls is
	 * accepted.
	 */
	template<typename Solve, typename Price>
	concept penalized_solver =
		integral<Price> && std::invocable<Solve &, Price>
		&& detail::pair_like<std::remove_cvref_t<std::invoke_result_t<Solve &, Price>>>
		&& integral<std::remove_cvref_t<std::tuple_element_t<1, std::remove_cvref_t<std::invoke_result_t<Solve &, Price>>>>>;

	/** @brief The value type a @ref atl::penalized_solver returns (the pair's first element). */
	template<typename Solve, typename Price>
	using penalized_value_t =
		std::remove_cvref_t<std::tuple_element_t<0, std::remove_cvref_t<std::invoke_result_t<Solve &, Price>>>>;

	/** @brief The count type a @ref atl::penalized_solver returns (the pair's second element). */
	template<typename Solve, typename Price>
	using penalized_count_t =
		std::remove_cvref_t<std::tuple_element_t<1, std::remove_cvref_t<std::invoke_result_t<Solve &, Price>>>>;

	namespace detail {
		/**
		 * @brief What @ref atl::exact_count_optimum computes on the solver's
		 *        values: a default (overwritten) value, `Value(price)`,
		 *        `Value(count)`, `Value(0)` and `Value(1)` (to read the
		 *        direction off the comparator), and `+ - *`.
		 */
		template<typename Value, typename Price, typename Count>
		concept penalty_arithmetic =
			std::default_initializable<Value> && std::movable<Value> && std::constructible_from<Value, Price>
			&& std::constructible_from<Value, Count> && std::constructible_from<Value, int>
			&& requires(Value const a) {
				   { a + a } -> std::convertible_to<Value>;
				   { a - a } -> std::convertible_to<Value>;
				   { a * a } -> std::convertible_to<Value>;
			   };
	}

	/**
	 * @brief The result of @ref atl::exact_count_optimum.
	 */
	template<std::movable Value, integral Price, integral Count>
	struct exact_count_result {
		/** @brief The best value over solutions with exactly the asked count. */
		Value value;
		/**
		 * @brief The smallest price in the searched range at which the asked
		 *        count is optimal for the penalized problem. The slope of the
		 *        optimum curve `f` just right of the count (when the count is
		 *        not the range's last) is `-price` when minimizing and `+price`
		 *        when maximizing. Re-running the solver at this price
		 *        reproduces an optimal solution's penalized value.
		 */
		Price price;
		/**
		 * @brief The count the solver reported at `price`: at most the asked
		 *        count, and equal to it unless the asked count lies inside a
		 *        collinear stretch of `f` (several counts tied for the
		 *        penalized optimum), where the solver's fewest-items rule
		 *        reports the stretch's left end. So `count < asked` detects a
		 *        tie without another solver call; a solution with exactly the
		 *        asked count must then be built from the tied solutions (see
		 *        the guide).
		 */
		Count count;
	};

	/**
	 * @brief The exact best value of a problem constrained to use exactly
	 *        `count` items, given `solve(price)`, a solver of the same problem
	 *        with the count constraint dropped and every item priced at
	 *        `price` instead (Lagrangian relaxation; WQS
	 *        binary search).
	 *
	 * Let `f(c)` be the best value with exactly `c` items. When minimizing
	 * (`Compare = std::less<>`), `solve(price)` returns the minimum over all
	 * solutions of `value + price * count`, and the count of a solution
	 * achieving it; when maximizing (`std::greater<>`) it returns the maximum
	 * of `value - price * count`. Either way a higher price makes items
	 * dearer, and the best count can only fall. This binary-searches the
	 * smallest integer price whose returned count is at most `count`, then
	 * returns `penalized - price * count` (minimizing; `penalized + price *
	 * count` maximizing), where `penalized` is the solver's value at that
	 * price.
	 *
	 * The returned value is computed from the penalized optimum, not from the
	 * solution the solver found, and that is what makes the answer exact on a
	 * collinear stretch of `f`: there several counts tie at every price, no
	 * price makes the solver report `count` itself, but at the found price
	 * `count` ties for the penalized optimum, so `penalized = f(count) + price
	 * * count` (minimizing).
	 *
	 * Why the search lands there (minimizing; maximizing is the mirror
	 * image): with integer slopes `s_c = f(c + 1) - f(c)`, nondecreasing by
	 * convexity, the counts optimal at price `p` are those `c` with
	 * `s_{c-1} <= -p <= s_c`, and the fewest-items rule reports the
	 * smallest of them. That is at most `count` exactly when `p >=
	 * -s_count`, so the search finds `p = -s_count` (clamped to `[lo, hi]`),
	 * and `count` is optimal there because `s_{count-1} <= s_count`. A
	 * solver that reported the MOST items on ties would make the search
	 * stop one price past `-s_count`, where `count` is no longer optimal
	 * when `s_{count-1} = s_count` (a collinear stretch), and the value
	 * would be wrong; so would a real-valued `f` searched with integer
	 * prices, as its slopes need not be integers.
	 *
	 * Overflow: the solver forms its own penalized values, which it returns
	 * as `Value`s. This function forms one product, `price * count` at
	 * the final price, and one sum with the solver's value there; for an
	 * integral `Value` up to 64 bits both are formed in a wider integer
	 * (`int128` for `int64`) and only the answer is narrowed, so the
	 * product need not fit `Value` (`Value = int64`, `Count = uint64`,
	 * `count = 2^63`, price `-1` is exact). A 128-bit or non-integral
	 * `Value` has no wider type and must hold the product, `Value(count)`
	 * and `Value(price)`.
	 *
	 * @param count The exact item count wanted.
	 * @param lo, hi The searched price range, `lo <= hi` (asserted). `hi`
	 *        must make the solver's count at most `count` (asserted on the
	 *        solver's answer at `hi`); `lo` must make it at least `count`
	 *        (asserted when the search ends at `lo`). For an `f` with all
	 *        slopes `f(c + 1) - f(c)` within `[-M, M]`, `[-M - 1, M + 1]`
	 *        always works when `count` is feasible. An infeasible `count`
	 *        (below every feasible count, or above them) fails one of the
	 *        two assertions.
	 * @param solve `solve(price) -> {value, count}`, a @ref atl::penalized_solver.
	 * @param comp `std::less<>` to minimize or `std::greater<>` to maximize:
	 *        the direction is read off `comp(Value(1), Value(0))`, so any
	 *        comparator ordering the values numerically works.
	 * @pre `f` is convex (minimizing) or concave (maximizing) over the
	 *      feasible counts, which lie in one interval, and integer-valued, so
	 *      its slopes are integers and integer prices reach every one (not
	 *      checkable; documented). Among the solutions tied for the penalized
	 *      optimum, `solve` reports the one with the FEWEST items, e.g. by
	 *      comparing `(penalized value, count)` lexicographically. `Value`
	 *      holds the solver's penalized value at every probed price and the
	 *      answer `f(count)`, and, without a wider type (see Overflow),
	 *      `price * count` at the final price. `Price` itself never
	 *      overflows: the midpoint is formed without `lo + hi`, so
	 *      `[INT64_MIN, INT64_MAX]` is a valid range.
	 * @complexity `ceil(log2(hi - lo + 1)) + 1` calls of `solve` at most
	 */
	template<typename Price, typename Solve, typename Compare = std::less<>>
		requires penalized_solver<Solve, Price>
			  && detail::penalty_arithmetic<penalized_value_t<Solve, Price>, Price, penalized_count_t<Solve, Price>>
			  && std::strict_weak_order<Compare const &, penalized_value_t<Solve, Price> const &,
										penalized_value_t<Solve, Price> const &>
	[[nodiscard]] constexpr exact_count_result<penalized_value_t<Solve, Price>, Price, penalized_count_t<Solve, Price>>
	exact_count_optimum(std::type_identity_t<penalized_count_t<Solve, Price>> count, Price lo, Price hi, Solve &&solve,
						Compare comp = {}) {
		using value_type = penalized_value_t<Solve, Price>;
		atl_assert(lo <= hi);
		Price const first = lo;
		value_type penalized{};
		penalized_count_t<Solve, Price> reached{};
		bool has_answer = false;
		while (lo < hi) {
			Price const middle = detail::floor_midpoint(lo, hi);
			auto [value, found] = solve(middle);
			if (found <= count) {
				hi = middle;
				penalized = std::move(value);
				reached = found;
				has_answer = true;
			} else {
				lo = Price(middle + 1);
			}
		}
		if (!has_answer) {
			auto [value, found] = solve(hi);
			atl_assert(found <= count);
			penalized = std::move(value);
			reached = found;
		}
		// a search ending at the range's first price has no probe below it
		// that reported more than `count`, so the answer there is the only
		// evidence that `count` is optimal: it must be `count` exactly
		if (hi == first) atl_assert(reached == count);
		bool const is_maximizing = std::invoke(comp, value_type(1), value_type(0));
		// the product and the sum are wide: only the answer has to fit `Value`
		using wide = detail::dp_wide_t<value_type>;
		auto const total = wide(wide(hi) * wide(count));
		auto const answer = is_maximizing ? wide(wide(penalized) + total) : wide(wide(penalized) - total);
		return {value_type(answer), hi, reached};
	}
}
