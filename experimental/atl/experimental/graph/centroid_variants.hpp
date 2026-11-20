#pragma once
#include <atl/graph/centroid_decomposition.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

/**
 * @file
 * @brief The classic recursive centroid engine, racing the library
 *        vertex-ranking engine behind the three-argument event contract.
 *
 * Only the three-argument `graph::centroid_event` form is mirrored: there is
 * no four-argument edge-span callback and no classic twin of
 * `centroid_decomposition`. Also here: the classic engine with its per-component
 * size pass reused from the parent's emit sweep, and the sweep-wrapper
 * `centroid_decomposition` for comparison with the direct ranking read-off.
 */
namespace atl::experimental {
	/**
	 * @brief Textbook centroid decomposition: per component, sizes by DFS,
	 *        the centroid by the max-subtree walk, emit, block, recurse on
	 *        branches. Loses to the ranking engine by 1.4-2.2x (record
	 *        graph-centroid_decomposition 2026-10-15, x86-64 Xeon 8358, plain
	 *        -O2, n = 2e5 full sweep: random 67.8 vs 47.8 ms, path 58.5 vs
	 *        26.9, caterpillar 41.3 vs 20.8, binary 34.3 vs 23.1, broom 33.1 vs
	 *        21.5).
	 * @pre g is a forest (symmetric digraph), vertex_count(g) < 2^31
	 * @complexity O(n log n)
	 */
	template<graph::detail::scannable_graph Graph, graph::centroid_event Event>
	constexpr void centroid_visit_by_subtree_sizes(Graph const &g, Event f) {
		std::size_t const n = graph::vertex_count(g);
		if (n == 0) return;
		atl_assert(n < (uint64(1) << 31)); // uint32 sizes, doubled in the balance test
		std::vector<uint8> blocked(n, 0);
		std::vector<uint32> size(n, 0), stack, dfs;
		std::vector<graph::centroid_vertex> scratch(n);
		// (component root, parent center) work list
		std::vector<std::pair<uint32, graph::vertex_id>> todo;
		std::vector<uint8> seen(n, 0);
		for (uint32 v = 0; v < n; ++v)
			if (!seen[v]) {
				todo.emplace_back(v, graph::none);
				for (auto const &[w, p, d] : graph::bfs(g, v)) seen[w] = 1;
			}

		std::vector<uint32> parent(n);
		while (!todo.empty()) {
			auto const [root, up] = todo.back();
			todo.pop_back();
			dfs.clear();
			dfs.push_back(root);
			parent[root] = graph::none;
			for (std::size_t at = 0; at < dfs.size(); ++at) {
				uint32 const v = dfs[at];
				size[v] = 1;
				for (auto const &e : graph::out_edges(g, v)) {
					uint32 const w = graph::target(e);
					if (w == parent[v] || blocked[w]) continue;
					parent[w] = v;
					dfs.push_back(w);
				}
			}
			for (std::size_t at = dfs.size(); at-- > 1;) size[parent[dfs[at]]] += size[dfs[at]];
			uint32 const total = size[root];
			uint32 c = root;
			for (;;) {
				uint32 heavy = graph::none;
				for (auto const &e : graph::out_edges(g, c)) {
					uint32 const w = graph::target(e);
					if (w == parent[c] || blocked[w]) continue;
					if (size[w] * 2 > total && (heavy == graph::none || size[w] > size[heavy]))
						heavy = w;
				}
				if (heavy == graph::none) break;
				c = heavy;
			}
			std::size_t count = 0;
			scratch[count++] = {c, graph::none, 0, graph::none};
			uint32 branch = 0;
			for (auto const &e : graph::out_edges(g, c)) {
				uint32 const seed = graph::target(e);
				if (blocked[seed]) continue;
				stack.push_back(uint32(count));
				scratch[count++] = {seed, c, 1, branch};
				while (!stack.empty()) {
					auto const [v, p, depth, br] = scratch[stack.back()];
					stack.pop_back();
					for (auto const &e2 : graph::out_edges(g, v)) {
						uint32 const w = graph::target(e2);
						if (w == p || blocked[w]) continue;
						stack.push_back(uint32(count));
						scratch[count++] = {w, v, depth + 1, br};
					}
				}
				todo.emplace_back(seed, c);
				++branch;
			}
			f(c, up, std::span<graph::centroid_vertex const>(scratch.data(), count));
			blocked[c] = 1;
		}
	}

	/**
	 * @brief @ref atl::experimental::centroid_visit_by_subtree_sizes without the per-component size DFS:
	 *        the emit sweep from a centroid lists each branch parent-first,
	 *        so one reverse scan of that branch's records yields the subtree sizes
	 *        and parents its child component needs. Only each tree's first
	 *        component runs the size DFS. reduces elapsed time by about 21-26% versus the plain classic
	 *        engine, still behind the ranking engine at every shape (record
	 *        graph-centroid_decomposition 2026-10-15, x86-64 Xeon 8358, plain
	 *        -O2, n = 2e5 full sweep: random 51.3 vs ranking 47.8 ms, path 44.8
	 *        vs 26.9, caterpillar 32.3 vs 20.8, binary 26.3 vs 23.1, broom 26.2
	 *        vs 21.5).
	 * @pre g is a forest (symmetric digraph), vertex_count(g) < 2^31
	 * @complexity O(n log n)
	 */
	template<graph::detail::scannable_graph Graph, graph::centroid_event Event>
	constexpr void centroid_visit_by_reused_sizes(Graph const &g, Event f) {
		std::size_t const n = graph::vertex_count(g);
		if (n == 0) return;
		atl_assert(n < (uint64(1) << 31)); // uint32 sizes, doubled in the balance test
		std::vector<uint8> blocked(n, 0);
		std::vector<uint32> size(n, 0), stack, dfs;
		std::vector<graph::centroid_vertex> scratch(n);

		struct task {
			uint32 root;
			graph::vertex_id up;
			bool has_sizes; ///< whether size/parent of the component are already filled in
		};

		std::vector<task> todo;
		std::vector<uint8> seen(n, 0);
		for (uint32 v = 0; v < n; ++v)
			if (!seen[v]) {
				todo.push_back({v, graph::none, false});
				for (auto const &[w, p, d] : graph::bfs(g, v)) seen[w] = 1;
			}

		std::vector<uint32> parent(n);
		while (!todo.empty()) {
			auto const [root, up, has_sizes] = todo.back();
			todo.pop_back();
			if (!has_sizes) {
				dfs.clear();
				dfs.push_back(root);
				parent[root] = graph::none;
				for (std::size_t at = 0; at < dfs.size(); ++at) {
					uint32 const v = dfs[at];
					size[v] = 1;
					for (auto const &e : graph::out_edges(g, v)) {
						uint32 const w = graph::target(e);
						if (w == parent[v] || blocked[w]) continue;
						parent[w] = v;
						dfs.push_back(w);
					}
				}
				for (std::size_t at = dfs.size(); at-- > 1;) size[parent[dfs[at]]] += size[dfs[at]];
			}
			uint32 const total = size[root];
			uint32 c = root;
			for (;;) {
				uint32 heavy = graph::none;
				for (auto const &e : graph::out_edges(g, c)) {
					uint32 const w = graph::target(e);
					if (w == parent[c] || blocked[w]) continue;
					if (size[w] * 2 > total && (heavy == graph::none || size[w] > size[heavy]))
						heavy = w;
				}
				if (heavy == graph::none) break;
				c = heavy;
			}
			std::size_t count = 0;
			scratch[count++] = {c, graph::none, 0, graph::none};
			uint32 branch = 0;
			for (auto const &e : graph::out_edges(g, c)) {
				uint32 const seed = graph::target(e);
				if (blocked[seed]) continue;
				std::size_t const first = count;
				stack.push_back(uint32(count));
				scratch[count++] = {seed, c, 1, branch};
				while (!stack.empty()) {
					auto const [v, p, depth, br] = scratch[stack.back()];
					stack.pop_back();
					for (auto const &e2 : graph::out_edges(g, v)) {
						uint32 const w = graph::target(e2);
						if (w == p || blocked[w]) continue;
						stack.push_back(uint32(count));
						scratch[count++] = {w, v, depth + 1, br};
					}
				}
				// the branch is the seed-rooted child component, parent-first:
				// a reverse scan accumulates its subtree sizes
				for (std::size_t i = first; i < count; ++i) {
					size[scratch[i].vertex] = 1;
					parent[scratch[i].vertex] = scratch[i].parent;
				}
				parent[seed] = graph::none;
				for (std::size_t i = count; i-- > first + 1;) size[scratch[i].parent] += size[scratch[i].vertex];
				todo.push_back({seed, c, true});
				++branch;
			}
			f(c, up, std::span<graph::centroid_vertex const>(scratch.data(), count));
			blocked[c] = 1;
		}
	}

	/**
	 * @brief `graph::centroid_decomposition` as a thin wrapper over the full
	 *        `graph::centroid_visit` sweep, keeping each event's
	 *        `parent_center`. Lost to the direct read-off from the ranking DP
	 *        by 4.7-10x (record graph-centroid_decomposition 2026-10-15, x86-64
	 *        Xeon 8358, plain -O2: n = 2e5 random 46.5 vs 9.9 ms, path 25.4 vs
	 *        2.9, binary 22.0 vs 2.2; n = 1e6 random 495 vs 106, path 146 vs
	 *        15.4).
	 * @pre @p g is a forest given as a symmetric adjacency (not asserted)
	 * @complexity O(n log n) time, O(n) space
	 */
	template<graph::adjacency_graph Graph>
	[[nodiscard]] constexpr std::vector<graph::vertex_id> centroid_decomposition_by_component_sweep(Graph const &g) {
		std::vector<graph::vertex_id> parent(graph::vertex_count(g), graph::none);
		graph::centroid_visit(g, [&](graph::vertex_id c, graph::vertex_id up, std::span<graph::centroid_vertex const>) {
			parent[c] = up;
		});
		return parent;
	}
}
