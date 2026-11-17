#pragma once

#include <atl/config.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/integral.hpp>
#include <limits>
#include <type_traits>

namespace atl::detail {
	/** @brief Store `a + b` modulo the integer width; return whether the mathematical result is unrepresentable. */
	template<integral Integer>
	constexpr bool add_overflows(Integer a, Integer b, Integer &result) noexcept {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(ATL_FORCE_FALLBACKS)
		if constexpr (!std::is_class_v<Integer>) return __builtin_add_overflow(a, b, &result);
#endif
		using Unsigned = traits::make_unsigned_t<Integer>;
		using Working = std::conditional_t<(sizeof(Unsigned) < sizeof(unsigned)), unsigned, Unsigned>;
		result = Integer(Unsigned(Working(Unsigned(a)) + Working(Unsigned(b))));
		constexpr Integer low = (std::numeric_limits<Integer>::lowest)(), high = (std::numeric_limits<Integer>::max)();
		if constexpr (signed_integral<Integer>)
			return b >= 0 ? a > high - b : a < low - b;
		else
			return a > high - b;
	}

	/** @brief Store `a - b` modulo the integer width; return whether the mathematical result is unrepresentable. */
	template<integral Integer>
	constexpr bool sub_overflows(Integer a, Integer b, Integer &result) noexcept {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(ATL_FORCE_FALLBACKS)
		if constexpr (!std::is_class_v<Integer>) return __builtin_sub_overflow(a, b, &result);
#endif
		using Unsigned = traits::make_unsigned_t<Integer>;
		using Working = std::conditional_t<(sizeof(Unsigned) < sizeof(unsigned)), unsigned, Unsigned>;
		result = Integer(Unsigned(Working(Unsigned(a)) - Working(Unsigned(b))));
		constexpr Integer low = (std::numeric_limits<Integer>::lowest)(), high = (std::numeric_limits<Integer>::max)();
		if constexpr (signed_integral<Integer>)
			return b >= 0 ? a < low + b : a > high + b;
		else
			return a < b;
	}

	/** @brief Store `a * b` modulo the integer width; return whether the mathematical result is unrepresentable. */
	template<integral Integer>
	constexpr bool mul_overflows(Integer a, Integer b, Integer &result) noexcept {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(ATL_FORCE_FALLBACKS)
		if constexpr (!std::is_class_v<Integer>) return __builtin_mul_overflow(a, b, &result);
#endif
		using Unsigned = traits::make_unsigned_t<Integer>;
		// Small unsigned types promote to int, whose product could overflow before narrowing.
		using Working = std::conditional_t<(sizeof(Unsigned) < sizeof(unsigned)), unsigned, Unsigned>;
		result = Integer(Unsigned(Working(Unsigned(a)) * Working(Unsigned(b))));
		constexpr Integer low = (std::numeric_limits<Integer>::lowest)(), high = (std::numeric_limits<Integer>::max)();
		if constexpr (signed_integral<Integer>)
			return a != 0 && b != 0 && (a > 0 ? (b > 0 ? a > high / b : b < low / a) : (b > 0 ? a < low / b : a < high / b));
		else
			return b != 0 && a > high / b;
	}
}
