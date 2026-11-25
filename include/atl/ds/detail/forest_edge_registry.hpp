#pragma once

#include <atl/ds/detail/dynamic_forest_core.hpp>
#include <atl/ds/detail/edge_id_table.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <utility>
#include <vector>

namespace atl::detail {

	/**
	 * @brief The edge-tier bookkeeping of the edge-as-vertex forests
	 *        (@ref atl::link_cut_tree, @ref atl::top_tree): (min, max) endpoint pair ->
	 *        edge id, the endpoints per id (`graph::none` when free), and the
	 *        free ids, handed out lowest first and reused most recently freed
	 *        first (dense LIFO ids).
	 */
	struct forest_edge_registry {
		edge_id_table ids;
		std::vector<std::pair<uint32, uint32>> endpoints;
		std::vector<uint32> free_ids;

		/** @brief @p slots free ids, no edges. @complexity O(slots) */
		constexpr void reset(std::size_t slots) {
			atl_assert(slots < graph::none); // ids are 32-bit with none reserved
			ids.reset_entries();
			ids.reserve_entries(slots);
			endpoints.assign(slots, {graph::none, graph::none});
			free_ids.resize(slots);
			// popped from the back: the lowest id first
			for (std::size_t i = 0; i < slots; ++i) free_ids[i] = uint32(slots - 1 - i);
		}

		/** @brief Register the edge u - v under a free id and return it. @pre a free id exists */
		constexpr uint32 acquire(uint32 u, uint32 v) {
			uint32 const e = free_ids.back();
			free_ids.pop_back();
			ids.insert_entry(forest_edge_key(u, v), e);
			endpoints[e] = {u, v};
			return e;
		}

		/** @brief Forget the live edge @p e; its id becomes free. */
		constexpr void release(uint32 e) {
			auto const [a, b] = endpoints[e];
			ids.erase_entry(forest_edge_key(a, b));
			endpoints[e] = {graph::none, graph::none};
			free_ids.push_back(e);
		}

		/** @brief The id of edge u - v, `graph::none` when there is none. @complexity expected O(1) */
		[[nodiscard]] constexpr uint32 find(uint32 u, uint32 v) const { return ids.lookup_id(forest_edge_key(u, v), graph::none); }

		/** @brief Whether @p e is a live edge id. */
		[[nodiscard]] constexpr bool is_live(uint32 e) const { return e < endpoints.size() && endpoints[e].first != graph::none; }
	};

}
