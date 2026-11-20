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
	 * @brief Strongly connected components in topological order, computed by an
	 * iterative Pearce search with one discovery/low-link/component array.
	 * @tparam Graph Scannable adjacency graph; reference-returning rows are
	 * forward ranges, value-returning rows are sized random-access ranges.
	 * @param g Directed graph, unchanged throughout the call.
	 * @pre `vertex_count(g) < none`; every target is below `vertex_count(g)`
	 * (asserted). Index-resumed rows have at most `none` arcs (asserted).
	 * Repeated row access preserves order and reference-returning
	 * rows remain valid throughout the call (documented only).
	 * @return An owning partition with dense ids in `[0, count())`. For every
	 * edge `u -> v`, `component[u] <= component[v]`. No order is promised
	 * between unrelated components. The empty graph has zero components.
	 * @complexity O(n + m) time and O(n) auxiliary storage, assuming O(1)
	 * graph access, target extraction and iterator operations. Explicit heap stacks avoid recursion
	 * proportional to path depth. Allocation failures propagate.
	 */
	template<detail::scannable_graph Graph>
	[[nodiscard]] constexpr vertex_partition scc(Graph const &g) {
		atl_assert(graph::vertex_count(g) < none);
		std::vector<uint32> component(graph::vertex_count(g), none);
		// component[] doubles as Pearce's rindex during the run: discovery
		// indices grow upward from 0 for in-progress vertices, final ids
		// are assigned from n downward — the ranges cannot meet, so a
		// plain `<` comparison ignores finished vertices for free.
		auto const n = uint32(component.size());
		using scan = detail::edge_scan<Graph, true>;

		struct frame {
			vertex_id v;
			[[no_unique_address]] typename scan::position at;
			uint32 is_root_candidate;
		};

		std::vector<frame> frames;
		std::vector<vertex_id> open;
		uint32 timer = 0, next_component = n;
		for (vertex_id s = 0; s < n; ++s) {
			if (component[s] != none) continue;
			component[s] = timer++;
			frames.push_back({s, scan::initial(g, s), 1});
			while (!frames.empty()) {
				frame &f = frames.back();
				auto es = scan(g, f.v, f.at);
				bool has_descended = false;
				while (!es.exhausted()) {
					vertex_id const w = graph::target(es.current());
					atl_assert(w < n);
					es.advance();
					if (component[w] == none) {
						f.at = es.pos(); // Stack growth can invalidate f; save the resume position first.
						component[w] = timer++;
						frames.push_back({w, scan::initial(g, w), 1});
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
					// close the component: everything on `open` with an
					// index at or above the root's belongs to it
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
