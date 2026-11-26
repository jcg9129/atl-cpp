#pragma once

#include <atl/ds/detail/dynamic_forest_core.hpp>
#include <atl/ds/detail/forest_edge_registry.hpp>
#include <atl/ds/flat_hash_map.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <cstddef>

namespace atl::detail {
	/**
	 * @brief The (u, v) -> edge id hash of the dynamic forests (edge tier of @ref atl::link_cut_tree,
	 *        @ref atl::euler_tour_tree): the seeded @ref atl::hash at run
	 *        time; an unseeded mix in constant evaluation, which has no seed to
	 *        read and no adversary.
	 */
	struct forest_edge_hash {
		/** @brief The hash of a packed endpoint pair. */
		constexpr std::size_t operator()(uint64 key) const noexcept {
			if consteval {
				return std::size_t(splitmix64(key));
			} else {
				return hash<uint64>{}(key);
			}
		}
	};

}
