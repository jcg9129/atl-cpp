#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/partition.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <utility>
#include <vector>

namespace atl::graph {
	/**
	 * @brief Bridge-connected vertex classes from an iterative, single-array
	 * Pearce-style search. An edge is a bridge iff its endpoints differ in class.
	 * @tparam Graph Scannable adjacency graph, as for atl::graph::scc.
	 * @param g Undirected graph represented by symmetric adjacency, unchanged
	 * throughout the call.
	 * @pre `vertex_count(g) < none`; every target is below `vertex_count(g)`
	 * (asserted). Index-resumed rows have at most `none` arcs (asserted).
	 * Each undirected edge contributes one arc in each direction,
	 * including every parallel copy (documented only). Repeated row access
	 * preserves order and reference-returning rows remain valid during the call.
	 * @return An owning partition with dense ids in `[0, count())`. Root each
	 * bridge tree at the class containing its connected component's lowest
	 * vertex: parent ids are smaller than child ids. No order is promised
	 * across connected components. Isolated vertices form singleton classes;
	 * self-loops change no class, and parallel edges are supported by skipping
	 * exactly one arc to the DFS parent. The empty graph has zero components.
	 * @complexity O(n + m) time and O(n) auxiliary storage, assuming O(1)
	 * graph access, target extraction and iterator operations. Explicit heap stacks avoid recursion
	 * proportional to path depth. Allocation failures propagate.
	 */
	template<detail::scannable_graph Graph>
	[[nodiscard]] constexpr vertex_partition two_edge_connected_components(Graph const &g) {
		atl_assert(graph::vertex_count(g) < none);
		std::vector<uint32> component(graph::vertex_count(g), none);
		// component[] doubles as the discovery index during the run: indices
		// grow upward from 0 for in-progress vertices, final ids are assigned
		// from n downward — the ranges cannot meet, so a plain `<` comparison
		// ignores finished vertices for free.
		auto const n = uint32(component.size());
		using scan = detail::edge_scan<Graph>;

		struct frame {
			vertex_id v, parent;
			[[no_unique_address]] typename scan::position at;
			uint8 is_root_candidate;
			uint8 has_skipped_parent;
		};

		std::vector<frame> frames;
		std::vector<vertex_id> open;
		uint32 timer = 0, next_component = n;
		for (vertex_id s = 0; s < n; ++s) {
			if (component[s] != none) continue;
			component[s] = timer++;
			frames.push_back({s, none, scan::initial(g, s), 1, 0});
			while (!frames.empty()) {
				frame &f = frames.back();
				auto es = scan(g, f.v, f.at);
				bool has_descended = false;
				while (!es.exhausted()) {
					vertex_id const w = graph::target(es.current());
					atl_assert(w < n);
					es.advance();
					if (w == f.parent && !f.has_skipped_parent) {
						f.has_skipped_parent = 1;
						continue;
					}
					if (component[w] == none) {
						f.at = es.pos(); // Stack growth can invalidate f; save the resume position first.
						component[w] = timer++;
						frames.push_back({w, f.v, scan::initial(g, w), 1, 0});
						has_descended = true;
						break;
					}
					if (component[w] < component[f.v]) {
						component[f.v] = component[w];
						f.is_root_candidate = 0;
					}
				}
				if (has_descended) continue;
				frame const done = frames.back();
				frames.pop_back();
				if (done.is_root_candidate) {
					// the tree edge into done.v is a bridge (or done.v roots a
					// DFS tree): close the component — everything on `open`
					// with an index at or above the root's belongs to it
					--next_component;
					--timer;
					while (!open.empty() && component[open.back()] >= component[done.v]) {
						component[open.back()] = next_component;
						open.pop_back();
						--timer;
					}
					component[done.v] = next_component;
				} else {
					open.push_back(done.v);
					if (!frames.empty() && component[done.v] < component[frames.back().v]) {
						component[frames.back().v] = component[done.v];
						frames.back().is_root_candidate = 0;
					}
				}
			}
		}
		for (auto &c : component) c -= next_component;
		return vertex_partition{std::move(component), std::size_t(n - next_component)};
	}
}
