#pragma once

/**
 * @file
 * @brief @ref atl::experimental::component_counts: offline dynamic connectivity, the number of connected components at each
 *        query of an edge add / remove log, by @ref atl::time_divide_and_conquer
 *        over a @ref atl::rollback_dsu. An experimental specialization of the offline engine.
 */

#include <atl/ds/rollback_dsu.hpp>
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace atl::experimental {
	using atl::uint32;
	using atl::graph::vertex_id;

	/** @brief An undirected edge as an event key; the solvers normalize `{u, v}` and `{v, u}` to one key. */
	using vertex_pair = std::pair<vertex_id, vertex_id>;

	/** @brief One entry of the log: add edge `key`, remove one copy of it, or a query. */
	using graph_event = atl::item_event<vertex_pair>;

	/**
	 * @brief Component counts at query events, in log order, for an undirected multigraph.
	 * Parallel edges are separate copies; a remove closes the oldest live copy.
	 * Self-loops are allowed. Query keys are ignored, including their endpoints.
	 * @param n Number of vertices, with ids in `[0, n)`; `graph::none` is never a vertex.
	 * @param events Add / remove / query log; the input is neither modified nor retained.
	 * @return One count per query, describing the state at that event.
	 * @pre `n <= 2^31 - 1`, `events.size() <= 2^30`, fewer than `2^32 - 1` cover entries
	 *      (asserted by the DSU and engine).
	 * @pre Every non-query endpoint is below `n`; every remove matches a live copy (asserted).
	 * @complexity O(n + T log(T + 1) log(n + 1)) time, O(n + T log(T + 1)) memory,
	 *             with expected O(T) event pairing, for T events. All walks are iterative.
	 * @note Not constexpr (event pairing uses a runtime-seeded hash map). Exceptions propagate.
	 */
	[[nodiscard]] inline std::vector<uint32> component_counts(std::size_t n, std::span<graph_event const> events) {
		std::vector<graph_event> normalized(events.begin(), events.end());
		for (auto &[kind, key] : normalized) {
			if (kind == atl::event_kind::query) continue;
			atl_assert(key.first < n && key.second < n);
			if (key.first > key.second) std::swap(key.first, key.second);
		}
		auto const lifetimes = atl::lifetimes_of(normalized);

		std::vector<uint32> counts;
		atl::rollback_dsu components(n);
		atl::time_divide_and_conquer(
			components, normalized.size(), lifetimes,
			[&](atl::rollback_dsu &d, uint32 item) {
				auto const [u, v] = normalized[lifetimes[item].begin].key;
				d.unite(u, v);
			},
			[&](atl::rollback_dsu &d, uint32) { counts.push_back(uint32(d.count())); },
			[&](uint32 t) { return normalized[t].kind == atl::event_kind::query; });
		return counts;
	}
}
