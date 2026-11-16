#pragma once
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace atl::detail {
	/**
	 * @brief An element of a map's bulk-load range: a pair-like (`std::pair`,
	 *        `std::tuple`, `std::array`) whose first member converts to @p Key and
	 *        second to @p Mapped, checked as the forwarded @p Entry so a move-only
	 *        mapped type is accepted from an owning rvalue range.
	 */
	template<typename Entry, typename Key, typename Mapped>
	concept key_value_entry = requires { typename std::tuple_size<std::remove_cvref_t<Entry>>::type; }
						   && (std::tuple_size_v<std::remove_cvref_t<Entry>> >= 2) && requires(Entry &&e) {
								  { std::get<0>(std::forward<Entry>(e)) } -> std::convertible_to<Key>;
								  { std::get<1>(std::forward<Entry>(e)) } -> std::convertible_to<Mapped>;
							  };
}
