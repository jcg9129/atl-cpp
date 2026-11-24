#pragma once

#include <atl/ds/detail/dynamic_forest_core.hpp>
#include <atl/ds/flat_hash_map_core.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash_core.hpp>
#include <cstddef>
#include <utility>
#include <vector>

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

	/**
	 * @brief The edge-tier bookkeeping of the edge-as-vertex forests
	 *        (@ref atl::link_cut_tree, @ref atl::top_tree): (min, max) endpoint pair ->
	 *        edge id, the endpoints per id (`graph::none` when free), and the
	 *        free ids, handed out lowest first and reused most recently freed
	 *        first (dense LIFO ids).
	 */
	struct forest_edge_registry {
		flat_hash_map<uint64, uint32, forest_edge_hash> ids;
		std::vector<std::pair<uint32, uint32>> endpoints;
		std::vector<uint32> free_ids;

		/** @brief @p slots free ids, no edges. @complexity O(slots) */
		constexpr void reset(std::size_t slots) {
			atl_assert(slots < graph::none); // ids are 32-bit with none reserved
			ids.clear();
			ids.reserve(slots);
			endpoints.assign(slots, {graph::none, graph::none});
			free_ids.resize(slots);
			// popped from the back: the lowest id first
			for (std::size_t i = 0; i < slots; ++i) free_ids[i] = uint32(slots - 1 - i);
		}

		/** @brief Register the edge u - v under a free id and return it. @pre a free id exists */
		constexpr uint32 acquire(uint32 u, uint32 v) {
			uint32 const e = free_ids.back();
			free_ids.pop_back();
			ids.insert(forest_edge_key(u, v), e);
			endpoints[e] = {u, v};
			return e;
		}

		/** @brief Forget the live edge @p e; its id becomes free. */
		constexpr void release(uint32 e) {
			auto const [a, b] = endpoints[e];
			ids.erase(forest_edge_key(a, b));
			endpoints[e] = {graph::none, graph::none};
			free_ids.push_back(e);
		}

		/** @brief The id of edge u - v, `graph::none` when there is none. @complexity expected O(1) */
		[[nodiscard]] constexpr uint32 find(uint32 u, uint32 v) const { return ids.at_or(forest_edge_key(u, v), graph::none); }

		/** @brief Whether @p e is a live edge id. */
		[[nodiscard]] constexpr bool is_live(uint32 e) const { return e < endpoints.size() && endpoints[e].first != graph::none; }
	};

}
