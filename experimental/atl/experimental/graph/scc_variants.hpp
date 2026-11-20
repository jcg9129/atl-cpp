#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/partition.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

/** @file
 * @brief SCC comparisons with topological component ids. All searches use
 * heap stacks. Kosaraju builds a transpose and performs two traversals;
 * it is the designated readable oracle. Other variants use Tarjan low-links,
 * DFS event delivery or Gabow path boundaries. See docs/graph-components.md
 * for complete measurements and input contracts.
 */
namespace atl::experimental {
	/**
	 * @brief Tarjan expressed over the dfs_visit events — the textbook
	 *        algorithm as three lambdas.
	 *
	 * @tparam Graph Scannable adjacency graph.
	 * @pre Same input preconditions as atl::graph::scc.
	 * @complexity O(n + m) time, O(n) heap storage; no depth-proportional recursion.
	 *
	 * DFS event delivery costs 259 vs library 163 ms on random m=4e6.
	 * Xeon Platinum 8280, GCC 15.2.0, -O2, 2026-10-15.
	 * @param g Input graph; valid targets, unchanged adjacency.
	 * @return A dense vertex partition.
	 * @note Not constexpr (runtime comparison; constant evaluation is not enabled).
	 */
	template<graph::detail::scannable_graph Graph>
	[[nodiscard]] graph::vertex_partition scc_by_tarjan_events(Graph const &g) {
		std::vector<uint32> component(graph::vertex_count(g), graph::none);
		std::size_t count = 0;
		std::size_t const n = component.size();
		atl_assert(n < graph::none);
		std::vector<uint32> low(n), disc(n);
		std::vector<graph::vertex_id> open;
		open.reserve(n);
		uint32 timer = 0;
		graph::dfs_visit(
			g,
			[&](graph::vertex_id v, graph::vertex_id, uint32, uint32) {
				disc[v] = low[v] = timer++;
				open.push_back(v);
			},
			[&](graph::vertex_id u, uint32, graph::vertex_id v) {
				// on Tarjan's stack iff visited and not yet assigned a component
				if (component[v] == graph::none) low[u] = (std::min)(low[u], disc[v]);
			},
			[&](graph::vertex_id v, graph::vertex_id parent) {
				if (low[v] == disc[v]) {
					auto const id = uint32(count++); // reverse topological; flipped below
					graph::vertex_id w = 0;
					do {
						w = open.back();
						open.pop_back();
						component[w] = id;
					} while (w != v);
				}
				if (parent != v) low[parent] = (std::min)(low[parent], low[v]);
			});
		for (auto &c : component) c = uint32(count) - 1 - c;
		return graph::vertex_partition{std::move(component), count};
	}

	namespace detail {
		struct scc_frame {
			uint32 v, idx;
		};
	}

	/**
	 * @brief Textbook iterative Tarjan: discovery + low-link arrays, one open
	 *        stack, on-stack tested via component-unassigned (no separate
	 *        flag array).
	 * @pre net.vertex_count() < graph::none
	 * @complexity O(n + m) time, O(n) heap storage; no depth-proportional recursion.
	 *
	 * Separate discovery and low-link arrays cost 231 vs library 163 ms on random m=4e6.
	 * Xeon Platinum 8280, GCC 15.2.0, -O2, 2026-10-15.
	 * @param net Input graph; valid targets, unchanged adjacency.
	 * @return A dense vertex partition.
	 * @note Not constexpr (runtime comparison; constant evaluation is not enabled).
	 */
	// GCC 15 -O2 misattributes the `open` vector's teardown to an offset
	// pointer it invented from the back()/pop_back() arithmetic — a
	// -Wfree-nonheap-object false positive on plain std::vector use.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
	[[nodiscard]] inline graph::vertex_partition scc_by_tarjan(graph::csr<graph::vertex_id> const &net) {
		std::vector<uint32> component(net.vertex_count(), graph::none);
		std::size_t count = 0;
		std::size_t const n = component.size();
		atl_assert(n < graph::none);
		std::vector<uint32> disc(n, graph::none), low(n);
		std::vector<uint32> open;
		open.reserve(n);
		std::vector<detail::scc_frame> frames;
		uint32 timer = 0;
		for (uint32 s = 0; s < n; ++s) {
			if (disc[s] != graph::none) continue;
			frames.push_back({s, 0});
			disc[s] = low[s] = timer++;
			open.push_back(s);
			while (!frames.empty()) {
				detail::scc_frame &f = frames.back();
				auto const es = net.out_edges(f.v);
				bool has_descended = false;
				while (f.idx < es.size()) {
					uint32 const w = es[f.idx++];
					if (disc[w] == graph::none) {
						disc[w] = low[w] = timer++;
						open.push_back(w);
						frames.push_back({w, 0});
						has_descended = true;
						break;
					}
					if (component[w] == graph::none) low[f.v] = (std::min)(low[f.v], disc[w]);
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
		for (auto &c : component) c = uint32(count) - 1 - c;
		return graph::vertex_partition{std::move(component), count};
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

	/**
	 * @brief Gabow's path-based variant: low-links replaced by a second stack
	 *        of path-segment boundaries; an edge into an open vertex contracts
	 *        the segments above it.
	 *
	 * @pre net.vertex_count() < graph::none
	 * @complexity O(n + m) time, O(n) heap storage; no depth-proportional recursion.
	 *
	 * Path boundaries cost 181 vs library 163 ms on random m=4e6.
	 * Xeon Platinum 8280, GCC 15.2.0, -O2, 2026-10-15.
	 * @param net Input graph; valid targets, unchanged adjacency.
	 * @return A dense vertex partition.
	 * @note Not constexpr (runtime comparison; constant evaluation is not enabled).
	 */
	[[nodiscard]] inline graph::vertex_partition scc_by_gabow(graph::csr<graph::vertex_id> const &net) {
		std::vector<uint32> component(net.vertex_count(), graph::none);
		std::size_t count = 0;
		std::size_t const n = component.size();
		atl_assert(n < graph::none);
		std::vector<uint32> disc(n, graph::none);
		std::vector<uint32> open, bounds;
		std::vector<detail::scc_frame> frames;
		uint32 timer = 0;
		for (uint32 s = 0; s < n; ++s) {
			if (disc[s] != graph::none) continue;
			disc[s] = timer++;
			open.push_back(s);
			bounds.push_back(s);
			frames.push_back({s, 0});
			while (!frames.empty()) {
				detail::scc_frame &f = frames.back();
				auto const es = net.out_edges(f.v);
				bool has_descended = false;
				while (f.idx < es.size()) {
					uint32 const w = es[f.idx++];
					if (disc[w] == graph::none) {
						disc[w] = timer++;
						open.push_back(w);
						bounds.push_back(w);
						frames.push_back({w, 0});
						has_descended = true;
						break;
					}
					if (component[w] == graph::none)
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
		for (auto &c : component) c = uint32(count) - 1 - c;
		return graph::vertex_partition{std::move(component), count};
	}

	/**
	 * @brief Kosaraju–Sharir: finish order on G, then collection on the
	 *        transpose in reverse finish order — components emerge directly in
	 *        topological order.
	 *
	 * @pre net.vertex_count() < graph::none, net.edge_count() < 2^32
	 * @complexity O(n + m) time, O(n + m) heap storage; no depth-proportional recursion.
	 *
	 * The transpose and second traversal cost 457 vs library 163 ms on random m=4e6; this is the designated readable oracle.
	 * Xeon Platinum 8280, GCC 15.2.0, -O2, 2026-10-15.
	 * @param net Input graph; valid targets, unchanged adjacency.
	 * @return A dense vertex partition.
	 * @note Not constexpr (runtime comparison; constant evaluation is not enabled).
	 */
	[[nodiscard]] inline graph::vertex_partition scc_by_kosaraju(graph::csr<graph::vertex_id> const &net) {
		std::vector<uint32> component(net.vertex_count(), graph::none);
		std::size_t count = 0;
		std::size_t const n = component.size();
		atl_assert(n < graph::none);
		atl_assert(net.edge_count() < (uint64(1) << 32)); // transpose offsets are uint32
		std::vector<uint32> finish;
		finish.reserve(n);
		{
			std::vector<uint8> seen(n, 0);
			std::vector<detail::scc_frame> frames;
			for (uint32 s = 0; s < n; ++s) {
				if (seen[s]) continue;
				seen[s] = 1;
				frames.push_back({s, 0});
				while (!frames.empty()) {
					detail::scc_frame &f = frames.back();
					auto const es = net.out_edges(f.v);
					bool has_descended = false;
					while (f.idx < es.size()) {
						uint32 const w = es[f.idx++];
						if (!seen[w]) {
							seen[w] = 1;
							frames.push_back({w, 0});
							has_descended = true;
							break;
						}
					}
					if (!has_descended) {
						finish.push_back(frames.back().v);
						frames.pop_back();
					}
				}
			}
		}
		std::vector<uint32> offset(n + 1, 0), heads(net.edge_count());
		for (uint32 v = 0; v < n; ++v)
			for (uint32 const w : net.out_edges(v)) ++offset[w + 1];
		std::partial_sum(offset.begin(), offset.end(), offset.begin());
		{
			auto cursor = offset;
			for (uint32 v = 0; v < n; ++v)
				for (uint32 const w : net.out_edges(v)) heads[cursor[w]++] = v;
		}
		std::vector<uint32> stack;
		for (std::size_t i = finish.size(); i-- > 0;) {
			if (component[finish[i]] != graph::none) continue;
			auto const id = uint32(count++);
			stack.push_back(finish[i]);
			component[finish[i]] = id;
			while (!stack.empty()) {
				uint32 const v = stack.back();
				stack.pop_back();
				for (uint32 j = offset[v]; j < offset[v + 1]; ++j)
					if (component[heads[j]] == graph::none) {
						component[heads[j]] = id;
						stack.push_back(heads[j]);
					}
			}
		}
		return graph::vertex_partition{std::move(component), count};
	}

}
