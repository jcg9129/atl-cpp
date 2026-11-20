#pragma once
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <ranges>

namespace atl::graph::detail {
	/** @brief Count edges without accumulating in a signed iterator difference type.
	 * @pre The count is at most limit (asserted before increment or conversion).
	 */
	template<typename EdgeList>
		requires std::ranges::forward_range<EdgeList const>
	[[nodiscard]] constexpr std::size_t checked_edge_count(EdgeList const &edges, std::size_t limit) {
		if constexpr (std::ranges::sized_range<EdgeList const>) {
			auto const count = std::ranges::size(edges);
			atl_assert(count <= limit);
			return std::size_t(count);
		} else {
			std::size_t count = 0;
			for (auto it = std::ranges::begin(edges); it != std::ranges::end(edges); ++it) {
				atl_assert(count < limit);
				++count;
			}
			return count;
		}
	}
}
