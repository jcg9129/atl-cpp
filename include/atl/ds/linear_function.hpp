#pragma once

/**
 * @file
 * @brief The element types of the Li Chao trees (@ref atl::li_chao_tree,
 *        @ref atl::dynamic_li_chao_tree): @ref atl::envelope_function, any function
 *        family evaluable at a coordinate, and its default and fast case
 *        @ref atl::linear_function over a @ref atl::line_value.
 */

#include <atl/types/concepts.hpp>
#include <atl/types/detail/checked_arithmetic.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace atl {
	/**
	 * @brief A coordinate, coefficient and value type for a line container:
	 *        totally ordered and closed under `+`, `-` and `*` (any signed
	 *        integer or floating-point type).
	 */
	template<typename Value>
	concept line_value = std::semiregular<Value> && std::totally_ordered<Value> && requires(Value const a) {
		{ a * a + a } -> std::convertible_to<Value>;
		{ a - a } -> std::convertible_to<Value>;
	};

	/**
	 * @brief The line `y = slope * x + intercept` evaluated in Value.
	 * @tparam Value Ordered arithmetic value type, including atl::int128.
	 * @pre Every multiplication and addition performed by operator() is representable
	 *      in Value, even when the final result would fit after cancellation.
	 *      Floating values and intermediates must be finite and comparisons ordered.
	 */
	template<line_value Value>
	struct linear_function {
		/** @brief The coefficient of `x`. */
		Value slope{};
		/** @brief The value at `x = 0`. */
		Value intercept{};

		/** @brief Evaluate at @p x. @pre `slope * x` and its sum with `intercept` fit Value. @return The line value. */
		[[nodiscard]] constexpr Value operator()(Value const &x) const { return slope * x + intercept; }

		/** @brief Compare both coefficients for equality. */
		friend constexpr bool operator==(linear_function const &, linear_function const &) = default;
	};

	/** @brief The value type of a function `f` evaluated at a `Coordinate`. */
	template<typename Function, typename Coordinate>
	using envelope_value_t = std::remove_cvref_t<std::invoke_result_t<Function const &, Coordinate const &>>;

	/**
	 * @brief A member of a function family a Li Chao tree can hold: a
	 *        semiregular value (stored in the tree's slots) that `f(x)`
	 *        evaluates at a `Coordinate`, with a result `Compare` strictly
	 *        weakly orders.
	 *
	 * The tree also needs any two members to cross at most once over the
	 * domain: `f(x)` better than `g(x)` holds on a prefix or a suffix of the
	 * domain points (equal values in between allowed). Lines, `(x - a)^2 + b`
	 * with a shared leading coefficient, and `a * sqrt(x) + b` qualify. No
	 * concept can check that; each tree states it as a documented
	 * precondition.
	 */
	template<typename Function, typename Coordinate, typename Compare>
	concept envelope_function =
		std::semiregular<Function> && std::regular_invocable<Function const &, Coordinate const &>
		&& std::semiregular<envelope_value_t<Function, Coordinate>>
		&& std::strict_weak_order<Compare const &, envelope_value_t<Function, Coordinate> const &,
								  envelope_value_t<Function, Coordinate> const &>;

	namespace detail {
		/**
		 * @brief Whether `f` evaluates without overflow at every `x` in
		 *        `[first, last]`: for an integral `Value`, both product and sum checked exactly at
		 *        the two ends (a line's extremes over an interval); always true
		 *        otherwise. For assertions.
		 */
		template<line_value Value>
		constexpr bool fits_on(linear_function<Value> const &f, Value const &first, Value const &last) {
			if constexpr (integral<Value>) {
				for (Value const &x : {first, last}) {
					Value product{}, sum{};
					if (mul_overflows(f.slope, x, product) || add_overflows(product, f.intercept, sum)) return false;
				}
			}
			return true;
		}

		/** @brief Dense-domain width without signed subtraction overflow, checked before narrowing. */
		template<integral Coordinate>
		constexpr std::size_t li_chao_width(Coordinate lo, Coordinate hi) {
			atl_assert(lo <= hi);
			using Unsigned = traits::make_unsigned_t<Coordinate>;
			Coordinate difference{};
			atl_assert(!sub_overflows(hi, lo, difference));
			Unsigned const width = Unsigned(Unsigned(hi) - Unsigned(lo));
			if constexpr (std::numeric_limits<Unsigned>::digits >= 31)
				atl_assert(width <= Unsigned(2147483647));
			return std::size_t(width);
		}

		/** @brief Whether `Function` is `linear_function<Coordinate>` (without forming it for other coordinates). */
		template<typename Function, typename Coordinate>
		inline constexpr bool is_linear_function_of = false;

		/** @brief A line over its own coordinate type. */
		template<typename Coordinate>
		inline constexpr bool is_linear_function_of<linear_function<Coordinate>, Coordinate> = true;

		/**
		 * @brief Whether `f` evaluates safely at every domain point in
		 *        `[first, last]`, as far as can be checked: exactly for a line
		 *        over integers (endpoint product and sum checks), trivially true for any other
		 *        family, whose fit is the caller's documented precondition.
		 */
		template<typename Function, typename Coordinate>
			requires std::regular_invocable<Function const &, Coordinate const &>
		constexpr bool evaluates_on(Function const &f, Coordinate const &first, Coordinate const &last) {
			if constexpr (is_linear_function_of<Function, Coordinate>)
				return fits_on(f, first, last);
			else {
				(void)f, (void)first, (void)last;
				return true;
			}
		}
	}
}
