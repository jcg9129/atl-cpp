#pragma once

#include <type_traits>

namespace atl::traits {
	/**
	 * @brief Whether @p Tp is exactly one of @p Types (std::is_same, so cv- and
	 *        ref-qualifiers count).
	 */
	template<typename Tp, typename... Types>
	struct is_one_of : public std::disjunction<std::is_same<Tp, Types>...> {};

	/** @brief Shorthand for `is_one_of<Tp, Types...>::value`. */
	template<typename Tp, typename... Types>
	constexpr bool is_one_of_v = is_one_of<Tp, Types...>::value;
}
