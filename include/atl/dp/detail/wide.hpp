#pragma once

/**
 * @file
 * @brief The wide arithmetic of the DP-optimization family (`atl/dp/`): the
 *        type intermediate sums, differences and products of an integral
 *        value are formed in, and the order that compares candidate totals
 *        in it.
 */

#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <concepts>
#include <functional>
#include <type_traits>

namespace atl::detail {
	/**
	 * @brief The type the family forms intermediates of a `T` in: an integer
	 *        wide enough for a sum, difference or product of two values of
	 *        the next narrower width (`int8`, `int16` -> `int32`, `int32` ->
	 *        `int64`, `int64` -> `int128`, the unsigned types alike); `T`
	 *        itself for a 128-bit integer (nothing wider: there the
	 *        intermediates must fit, a documented precondition), a floating
	 *        type or any other value type.
	 */
	template<typename T>
	struct dp_wide {
		using type = T;
	};

	template<typename T>
		requires(signed_integral<T> && sizeof(T) <= 2)
	struct dp_wide<T> {
		using type = int32;
	};

	template<typename T>
		requires(signed_integral<T> && sizeof(T) == 4)
	struct dp_wide<T> {
		using type = int64;
	};

	template<typename T>
		requires(signed_integral<T> && sizeof(T) == 8)
	struct dp_wide<T> {
		using type = int128;
	};

	template<typename T>
		requires(unsigned_integral<T> && sizeof(T) <= 2)
	struct dp_wide<T> {
		using type = uint32;
	};

	template<typename T>
		requires(unsigned_integral<T> && sizeof(T) == 4)
	struct dp_wide<T> {
		using type = uint64;
	};

	template<typename T>
		requires(unsigned_integral<T> && sizeof(T) == 8)
	struct dp_wide<T> {
		using type = uint128;
	};

	template<typename T>
	using dp_wide_t = typename dp_wide<std::remove_cv_t<T>>::type;

	/** @brief Whether `Compare` is the numeric "less" on `Value` (`std::less`, `std::ranges::less`). */
	template<typename Compare, typename Value>
	inline constexpr bool is_numeric_less_v = std::same_as<Compare, std::less<>> || std::same_as<Compare, std::less<Value>>
										   || std::same_as<Compare, std::ranges::less>;

	/** @brief Whether `Compare` is the numeric "greater" on `Value` (`std::greater`, `std::ranges::greater`). */
	template<typename Compare, typename Value>
	inline constexpr bool is_numeric_greater_v = std::same_as<Compare, std::greater<>>
											  || std::same_as<Compare, std::greater<Value>>
											  || std::same_as<Compare, std::ranges::greater>;

	/**
	 * @brief The type the solvers form and compare candidate totals of a
	 *        `Value` ordered by `Compare` in: @ref dp_wide_t when `Compare` is
	 *        a numeric order (`std::less`, `std::greater`, their `ranges`
	 *        forms), whose meaning carries over to the wide type; `Value`
	 *        itself for any other comparator, which is only known to order
	 *        `Value`s.
	 */
	template<typename Value, typename Compare>
	using dp_total_t = std::conditional_t<is_numeric_less_v<Compare, Value> || is_numeric_greater_v<Compare, Value>,
										  dp_wide_t<Value>, Value>;

	/**
	 * @brief `Compare` on @ref dp_total_t: `<` or `>` for a numeric order,
	 *        else the comparator itself (through `std::invoke`).
	 */
	template<typename Value, typename Compare>
	struct total_order {
		using total = dp_total_t<Value, Compare>;

		Compare const &comp;

		[[nodiscard]] constexpr bool operator()(total const &a, total const &b) const {
			if constexpr (is_numeric_less_v<Compare, Value>)
				return a < b;
			else if constexpr (is_numeric_greater_v<Compare, Value>)
				return b < a;
			else
				return std::invoke(comp, a, b);
		}
	};

	/**
	 * @brief Whether `std::vector<Value>` stores its values contiguously
	 *        (has `data()`), as the solvers that index raw rows need: every
	 *        type but `bool`.
	 */
	template<typename Value>
	concept contiguously_storable = !std::same_as<std::remove_cv_t<Value>, bool>;
}
