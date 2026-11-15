#pragma once

/**
 * @file
 * @brief Argument tags shared by several containers.
 */

namespace atl {
	/**
	 * @brief Tag selecting the constructor that adopts or builds from a range already sorted
	 *        strictly increasing under the container's comparator (so without duplicates);
	 *        the order is asserted, not sorted. Named after C++23's `std::sorted_unique`.
	 */
	struct sorted_unique_t {
		explicit sorted_unique_t() = default;
	};

	/** @brief The @ref sorted_unique_t tag value. */
	inline constexpr sorted_unique_t sorted_unique{};
}
