#pragma once

/**
 * @file
 * @brief @ref atl::experimental::bipartite_after_each: offline dynamic bipartiteness, whether the graph is bipartite after
 *        each edge toggle, by @ref atl::time_divide_and_conquer over a
 *        @ref atl::rollback_weighted_dsu over Z/2. A worked example of a custom
 *        @ref atl::by_rollback: the state is the DSU plus an odd-cycle counter,
 *        saved and restored together through a pair token.
 */

#include <atl/ds/flat_hash_set.hpp>
#include <atl/ds/rollback_weighted_dsu.hpp>
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace atl::experimental {
	using atl::uint32;
	using atl::uint8;
	using atl::graph::vertex_id;

	/**
	 * @brief A rollback weighted DSU over Z/2 (side parities, xor) plus the number
	 *        of detected parity contradictions (insertion may stop at the first).
	 */
	struct parity_state {
		/** @brief `difference(u, v)` is 1 when u and v sit on opposite sides. */
		atl::rollback_weighted_dsu<uint32, std::bit_xor<uint32>, std::identity> sides;
		/** @brief Detected contradictions; zero certifies bipartiteness of the represented edges. */
		uint32 odd_cycles = 0;
	};

	/**
	 * @brief Per toggle, in order, whether the graph on `n` vertices is bipartite
	 *        after it; a toggle adds the edge `{u, v}` if absent and removes it
	 *        otherwise. A self-loop is an odd cycle.
	 * @param n Number of vertices with ids in `[0, n)`; `graph::none` is excluded.
	 * @param toggles Undirected endpoint pairs; the input is neither modified nor retained.
	 * @return One byte per toggle, 1 for bipartite and 0 otherwise; orientations are equivalent.
	 * @pre `n <= 2^31 - 1`, `toggles.size() <= 2^30`, fewer than `2^32 - 1` cover entries
	 *      (asserted by the DSU and engine); every endpoint is below `n` (asserted).
	 * @complexity O(n + q log(q + 1) log(n + 1)) time, O(n + q log(q + 1)) memory,
	 *             with expected O(q) hashing, for q toggles. All walks are iterative.
	 * @note Not constexpr (toggle pairing uses runtime-seeded hash tables). Exceptions propagate.
	 */
	[[nodiscard]] inline std::vector<uint8> bipartite_after_each(std::size_t n,
																 std::span<std::pair<vertex_id, vertex_id> const> toggles) {
		using vertex_pair = std::pair<vertex_id, vertex_id>;
		std::vector<atl::item_event<vertex_pair>> events;
		events.reserve(toggles.size());
		atl::flat_hash_set<vertex_pair> present;
		for (auto [u, v] : toggles) {
			atl_assert(u < n && v < n);
			vertex_pair const key{(std::min)(u, v), (std::max)(u, v)};
			bool const was_present = present.erase(key);
			if (!was_present) present.insert(key);
			events.push_back({was_present ? atl::event_kind::remove : atl::event_kind::add, key});
		}
		auto const lifetimes = atl::lifetimes_of(events);

		std::vector<uint8> is_bipartite(toggles.size());
		parity_state state{decltype(parity_state::sides)(n)};
		atl::time_divide_and_conquer(
			state, events.size(), lifetimes,
			[&](parity_state &s, uint32 item) {
				if (s.odd_cycles) return; // the whole subtree answers "no" already
				auto const [u, v] = events[lifetimes[item].begin].key;
				if (s.sides.unite(u, v, 1) == atl::unite_outcome::contradicts) ++s.odd_cycles;
			},
			[&](parity_state &s, uint32 t) { is_bipartite[t] = s.odd_cycles == 0; },
			atl::by_rollback{[](parity_state &s) { return std::pair{s.sides.time(), s.odd_cycles}; },
							 [](parity_state &s, std::pair<std::size_t, uint32> token) {
								 s.sides.rollback(token.first);
								 s.odd_cycles = token.second;
							 }});
		return is_bipartite;
	}
}
