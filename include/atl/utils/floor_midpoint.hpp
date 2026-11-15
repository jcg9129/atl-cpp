#pragma once

/**
 * @file
 * @brief `atl::detail::floor_midpoint`: the overflow-free floor midpoint of two
 *        integers, for bisections over a whole integer type's range.
 */

#include <atl/types/concepts.hpp>

namespace atl::detail {
	/**
	 * @brief `floor((lo + hi) / 2)`, without the overflow of `lo + hi`: the
	 *        bits both share plus half the bits only one has (`>>` on a
	 *        negative value is an arithmetic shift since C++20, so the floor
	 *        holds for signed types too). @p lo and @p hi may come in either
	 *        order; the result always lies between them.
	 */
	template<integral Integer>
	constexpr Integer floor_midpoint(Integer lo, Integer hi) {
		return Integer((lo & hi) + ((lo ^ hi) >> 1));
	}
}
