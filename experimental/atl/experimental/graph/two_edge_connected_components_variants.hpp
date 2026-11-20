#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/partition.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

/** @file
 * @brief Bridge-component comparisons using low-links, Gabow boundaries or
 * chain decomposition. All use heap stacks and symmetric multigraph input,
 * skipping one parent arc (one tree arc for chains). The partition agrees
 * with the library; low-link and Gabow ids have no parent-order guarantee.
 * A brute-force edge-deletion oracle in the tests certifies every variant.
 * See docs/graph-components.md for complete measurements.
 */
namespace atl::experimental {
	namespace detail {
		struct two_edge_skip_frame {
			uint32 v, idx, parent;
			uint32 has_skipped_parent;
		};

		struct two_edge_bare_frame {
			uint32 v, idx;
		};
	}

	/**
	 * @brief Textbook iterative low-link: discovery + low arrays and one open
	 *        stack; a vertex with `low == disc` closes a component (the tree
	 *        edge above it is a bridge, or it roots a DFS tree).
	 * @pre net is a symmetric digraph, net.vertex_count() < graph::none
	 * @complexity O(n + m) time, O(n) heap storage; no depth-proportional recursion.
	 *
	 * Separate low-links cost 93.4 vs library 57.4 ms on a random tree.
	 * Xeon Platinum 8280, GCC 15.2.0, -O2, 2026-10-15.
	 * @param net Input graph; valid targets, unchanged adjacency.
	 * @return A dense vertex partition.
	 * @note Not constexpr (runtime comparison; constant evaluation is not enabled).
	 */
	[[nodiscard]] inline graph::vertex_partition two_edge_connected_components_by_low_link(graph::csr<graph::vertex_id> const &net) {
		std::vector<uint32> component(net.vertex_count(), graph::none);
		std::size_t count = 0;
		atl_assert(component.size() < graph::none);
		auto const n = uint32(component.size());
		std::vector<uint32> disc(n, graph::none), low(n);
		std::vector<uint32> open;
		open.reserve(n);
		std::vector<detail::two_edge_skip_frame> frames;
		uint32 timer = 0;
		for (uint32 s = 0; s < n; ++s) {
			if (disc[s] != graph::none) continue;
			disc[s] = low[s] = timer++;
			open.push_back(s);
			frames.push_back({s, 0, graph::none, 0});
			while (!frames.empty()) {
				detail::two_edge_skip_frame &f = frames.back();
				auto const es = net.out_edges(f.v);
				bool has_descended = false;
				while (f.idx < es.size()) {
					uint32 const w = es[f.idx++];
					if (w == f.parent && !f.has_skipped_parent) {
						f.has_skipped_parent = 1;
						continue;
					}
					if (disc[w] == graph::none) {
						disc[w] = low[w] = timer++;
						open.push_back(w);
						frames.push_back({w, 0, f.v, 0});
						has_descended = true;
						break;
					}
					low[f.v] = (std::min)(low[f.v], disc[w]);
				}
				if (has_descended) continue;
				uint32 const v = frames.back().v;
				frames.pop_back();
				if (low[v] == disc[v]) {
					auto const id = uint32(count++);
					uint32 w = 0;
					do {
						w = open.back();
						open.pop_back();
						component[w] = id;
					} while (w != v);
				}
				if (!frames.empty()) low[frames.back().v] = (std::min)(low[frames.back().v], low[v]);
			}
		}
		return graph::vertex_partition{std::move(component), count};
	}

	/**
	 * @brief Gabow's path-based variant: no low array; a second stack of
	 *        path-segment boundaries, contracted whenever a back edge closes
	 *        a cycle.
	 *
	 * Each undirected edge is contracted at its deeper endpoint (the
	 * shallower endpoint's later sighting is a no-op), so cycles are never
	 * missed.
	 * @pre net is a symmetric digraph, net.vertex_count() < graph::none
	 * @complexity O(n + m) time, O(n) heap storage; no depth-proportional recursion.
	 *
	 * Path boundaries take 252 vs library 263 ms on random m=4e6, but 68.4 vs 57.4 ms on a random tree.
	 * Xeon Platinum 8280, GCC 15.2.0, -O2, 2026-10-15.
	 * @param net Input graph; valid targets, unchanged adjacency.
	 * @return A dense vertex partition.
	 * @note Not constexpr (runtime comparison; constant evaluation is not enabled).
	 */
	[[nodiscard]] inline graph::vertex_partition two_edge_connected_components_by_gabow(graph::csr<graph::vertex_id> const &net) {
		std::vector<uint32> component(net.vertex_count(), graph::none);
		std::size_t count = 0;
		atl_assert(component.size() < graph::none);
		auto const n = uint32(component.size());
		std::vector<uint32> disc(n, graph::none);
		std::vector<uint32> open, bounds;
		std::vector<detail::two_edge_skip_frame> frames;
		uint32 timer = 0;
		for (uint32 s = 0; s < n; ++s) {
			if (disc[s] != graph::none) continue;
			disc[s] = timer++;
			open.push_back(s);
			bounds.push_back(s);
			frames.push_back({s, 0, graph::none, 0});
			while (!frames.empty()) {
				detail::two_edge_skip_frame &f = frames.back();
				auto const es = net.out_edges(f.v);
				bool has_descended = false;
				while (f.idx < es.size()) {
					uint32 const w = es[f.idx++];
					if (w == f.parent && !f.has_skipped_parent) {
						f.has_skipped_parent = 1;
						continue;
					}
					if (disc[w] == graph::none) {
						disc[w] = timer++;
						open.push_back(w);
						bounds.push_back(w);
						frames.push_back({w, 0, f.v, 0});
						has_descended = true;
						break;
					}
					while (disc[bounds.back()] > disc[w]) bounds.pop_back();
				}
				if (has_descended) continue;
				uint32 const v = frames.back().v;
				frames.pop_back();
				if (bounds.back() == v) {
					bounds.pop_back();
					auto const id = uint32(count++);
					uint32 w = 0;
					do {
						w = open.back();
						open.pop_back();
						component[w] = id;
					} while (w != v);
				}
			}
		}
		return graph::vertex_partition{std::move(component), count};
	}

	/**
	 * @brief Schmidt's chain decomposition — a two-pass algorithm:
	 *        build the DFS tree and preorder, then every back edge opens a
	 *        chain walked upward via parent pointers until it meets an
	 *        earlier chain.
	 *
	 * Tree edges covered by no chain are exactly the bridges; components are
	 * then labeled down the tree (each component's tree edges form a
	 * connected subtree, so a vertex joins its parent unless the edge above
	 * it is a bridge).
	 * @pre net is a symmetric digraph, net.vertex_count() < graph::none
	 * @complexity O(n + m) time, O(n) heap storage; no depth-proportional recursion.
	 *
	 * Two passes and parent walks cost 476 vs library 263 ms on random m=4e6.
	 * Xeon Platinum 8280, GCC 15.2.0, -O2, 2026-10-15.
	 * @param net Input graph; valid targets, unchanged adjacency.
	 * @return A dense vertex partition.
	 * @note Not constexpr (runtime comparison; constant evaluation is not enabled).
	 */
	[[nodiscard]] inline graph::vertex_partition two_edge_connected_components_by_chain_decomposition(graph::csr<graph::vertex_id> const &net) {
		std::vector<uint32> component(net.vertex_count(), graph::none);
		std::size_t count = 0;
		atl_assert(component.size() < graph::none);
		auto const n = uint32(component.size());
		std::vector<uint32> disc(n, graph::none), parent(n, graph::none);
		std::vector<uint32> order;
		order.reserve(n);
		{
			std::vector<detail::two_edge_bare_frame> frames;
			uint32 timer = 0;
			for (uint32 s = 0; s < n; ++s) {
				if (disc[s] != graph::none) continue;
				disc[s] = timer++;
				order.push_back(s);
				frames.push_back({s, 0});
				while (!frames.empty()) {
					detail::two_edge_bare_frame &f = frames.back();
					auto const es = net.out_edges(f.v);
					bool has_descended = false;
					while (f.idx < es.size()) {
						uint32 const w = es[f.idx++];
						if (disc[w] != graph::none) continue;
						disc[w] = timer++;
						parent[w] = f.v;
						order.push_back(w);
						frames.push_back({w, 0});
						has_descended = true;
						break;
					}
					if (!has_descended) frames.pop_back();
				}
			}
		}
		// visited: vertex lies on some chain (or already started its own);
		// covered: the tree edge parent[x] -> x lies on some chain;
		// tree_arc_seen: the one arc u -> w that *is* the tree edge, so a
		// second parallel copy counts as the back edge it is
		std::vector<uint8> visited(n, 0), covered(n, 0), tree_arc_seen(n, 0);
		for (uint32 const u : order) {
			visited[u] = 1;
			for (uint32 const w : net.out_edges(u)) {
				// the deeper endpoint owns the edge; `<=` also drops self-loops
				if (disc[w] <= disc[u]) continue;
				if (parent[w] == u && !tree_arc_seen[w]) {
					tree_arc_seen[w] = 1;
					continue;
				}
				for (uint32 x = w; !visited[x]; x = parent[x]) {
					visited[x] = 1;
					covered[x] = 1;
				}
			}
		}
		for (uint32 const v : order)
			component[v] = parent[v] == graph::none || !covered[v] ? uint32(count++) : component[parent[v]];
		return graph::vertex_partition{std::move(component), count};
	}

}
