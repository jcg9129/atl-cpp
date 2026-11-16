#pragma once
#include <ranges>
#include <type_traits>
#include <utility>

namespace atl::detail {
	/**
	 * @brief Whether a range constructor taking `R &&` may move elements out of
	 *        the range: only when `R` is an rvalue that owns its elements.
	 *
	 * A `borrowed_range` (`std::span`, `std::string_view`, `views::take` over a
	 * borrowed range) and a `std::ranges::view` (`views::filter`, which is not
	 * borrowed yet still refers to the caller's container) are excluded even as
	 * rvalues: moving through them would gut a container the caller still owns.
	 * Everything else is copied.
	 *
	 * @tparam R The range type as deduced from a forwarding parameter.
	 */
	template<std::ranges::range R>
	inline constexpr bool moves_range_elements =
		!std::is_lvalue_reference_v<R> && !std::ranges::borrowed_range<R> && !std::ranges::view<std::remove_cvref_t<R>>;

	/**
	 * @brief The library-wide element-forwarding rule for range constructors:
	 *        @p element (obtained by iterating a range passed as `R &&`) as an
	 *        rvalue when @ref moves_range_elements<R> holds, as a const lvalue
	 *        (so it is copied) otherwise.
	 *
	 * An element the range yields by value (`range_reference_t<R>` is not a
	 * reference, e.g. from `views::transform` or `views::iota`) is a temporary the
	 * constructor already owns, so it is always forwarded as an rvalue. An element
	 * yielded as `T &&` (e.g. `views::as_rvalue`, or a transform returning
	 * `std::move(x)`) still refers to the caller's storage, so it follows the rule
	 * for its range like an lvalue element.
	 *
	 * @tparam R The range type as deduced from the constructor's forwarding parameter.
	 */
	template<std::ranges::range R, typename Element>
	[[nodiscard]] constexpr auto &&forward_element(Element &&element) noexcept {
		if constexpr (moves_range_elements<R> || !std::is_reference_v<std::ranges::range_reference_t<R>>)
			return static_cast<std::remove_reference_t<Element> &&>(element);
		else
			return std::as_const(element);
	}

	/**
	 * @brief The type @ref forward_element yields for an element of a range passed
	 *        as `R &&` — what a constraint on a range constructor should check.
	 */
	template<std::ranges::range R>
	using forwarded_element_t = decltype(forward_element<R>(std::declval<std::ranges::range_reference_t<R>>()));
}
