#pragma once
#include <atl/graph/biconnected_components.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <vector>

/** @file
 * @brief Block comparisons using single-array state, low-links or paired
 * boundaries; chain decomposition computes only the articulation mask.
 * Searches use heap stacks, symmetric multigraph input and one skipped
 * parent/tree arc. Tests certify blocks by vertex deletion and cuts by an
 * independent deletion oracle; the chain method is the readable cut oracle.
 * See docs/graph-components.md for complete measurements.
 */
namespace atl::experimental {
	namespace detail {
		struct bcc_lowlink_frame {
			uint32 v, idx, parent;
			uint8 has_skipped_parent;
		};

		/**
		 * @brief Frame of the single-array variant.
		 *
		 * The evolving low-link lives here (the array must keep the immutable
		 * discovery index while a vertex is on the path — see the algorithm's
		 * doc), plus the open stack's size at descent (block membership is
		 * popped by position, not by value — low-links of pending sibling
		 * blocks can tie).
		 */
		struct bcc_state_frame {
			uint32 v, idx, parent, low, mark;
			uint8 has_skipped_parent;
		};

		struct bcc_chain_frame {
			uint32 v, idx;
		};
	}

	/**
	 * @brief Pearce-inspired single-array variant: one per-vertex integer
	 *        holds the discovery index while the vertex is on the DFS path,
	 *        its final low-link once it moves to the open stack, then the
	 *        block id (encoded downward from `none - 1` so live values, final
	 *        ids and `none` can never collide). The evolving low lives in the
	 *        frame — it is path-local anyway.
	 *
	 * The array must keep the *discovery index* while a vertex is on the
	 * path: a back edge into ancestor `w` must contribute `disc(w)` — `w`'s
	 * evolving low counts paths through `w` itself, which are no escape when
	 * `w` is the close point under test (the classic disc-not-low rule of
	 * articulation searches). A finished open vertex may contribute its final
	 * low: every such value is the disc of a target genuinely reachable
	 * without touching any close point above. Block membership pops by the
	 * frame's saved open-stack mark, not by value — low-links of pending
	 * sibling blocks can tie with the closing block's threshold.
	 * @pre net is a symmetric digraph, net.vertex_count() < 2^31
	 * @complexity O(n + m) time, O(n) heap storage; no depth-proportional recursion.
	 *
	 * Larger frames cost 369 vs library 260 ms on random m=4e6.
	 * Xeon Platinum 8280, GCC 15.2.0, -O2, 2026-10-15.
	 * @param net Input graph; valid targets, unchanged adjacency.
	 * @return Dense block ids with the library encoding.
	 * @note Not constexpr (runtime comparison; constant evaluation is not enabled).
	 */
	[[nodiscard]] inline graph::block_decomposition biconnected_components_by_single_array(graph::csr<graph::vertex_id> const &net) {
		atl_assert(net.vertex_count() < (uint64(1) << 31)); // block ids count down from none - 1
		auto const n = uint32(net.vertex_count());
		graph::block_decomposition r{std::vector<uint32>(n, graph::none), {}};
		auto &state = r.block; // disc on path, low on open, block id when done
		std::vector<uint32> open;
		std::vector<detail::bcc_state_frame> frames;
		uint32 timer = 0;
		for (uint32 s = 0; s < n; ++s) {
			if (state[s] != graph::none) continue;
			state[s] = timer++;
			frames.push_back({s, 0, graph::none, state[s], uint32(open.size()), 0});
			while (!frames.empty()) {
				detail::bcc_state_frame &f = frames.back();
				auto const es = net.out_edges(f.v);
				bool has_descended = false;
				while (f.idx < es.size()) {
					uint32 const w = es[f.idx++];
					if (w == f.parent && !f.has_skipped_parent) {
						f.has_skipped_parent = 1;
						continue;
					}
					if (state[w] == graph::none) {
						state[w] = timer++;
						frames.push_back({w, 0, f.v, state[w], uint32(open.size()), 0});
						has_descended = true;
						break;
					}
					f.low = (std::min)(f.low, state[w]);
				}
				if (has_descended) continue;
				detail::bcc_state_frame const done = frames.back();
				frames.pop_back();
				if (frames.empty()) {
					state[done.v] = graph::none; // DFS root: no tree edge above
					continue;
				}
				detail::bcc_state_frame &pf = frames.back();
				if (done.low >= state[pf.v]) {
					auto const id = uint32(r.top_vertices.size());
					r.top_vertices.push_back(pf.v);
					while (open.size() > done.mark) {
						state[open.back()] = (graph::none - 1) - id;
						open.pop_back();
					}
					state[done.v] = (graph::none - 1) - id;
				} else {
					open.push_back(done.v);
					state[done.v] = done.low;
					pf.low = (std::min)(pf.low, done.low);
				}
			}
		}
		for (auto &c : state)
			if (c != graph::none) c = (graph::none - 1) - c;
		return r;
	}

	/**
	 * @brief Textbook iterative Hopcroft–Tarjan low-link search with a vertex
	 *        stack: a block closes at the tree edge `parent -> v` whenever
	 *        `low[v] >= disc[parent]`, popping exactly the block's still-open
	 *        subtree vertices. One more n-word array than the library
	 *        path-based search, which is where it loses the benchmark.
	 * @pre net is a symmetric digraph, net.vertex_count() < graph::none
	 * @complexity O(n + m) time, O(n) heap storage; no depth-proportional recursion.
	 *
	 * A separate low-link array costs 308 vs library 260 ms on random m=4e6.
	 * Xeon Platinum 8280, GCC 15.2.0, -O2, 2026-10-15.
	 * @param net Input graph; valid targets, unchanged adjacency.
	 * @return Dense block ids with the library encoding.
	 * @note Not constexpr (runtime comparison; constant evaluation is not enabled).
	 */
	[[nodiscard]] inline graph::block_decomposition biconnected_components_by_low_link(graph::csr<graph::vertex_id> const &net) {
		atl_assert(net.vertex_count() < graph::none);
		auto const n = uint32(net.vertex_count());
		graph::block_decomposition r{std::vector<uint32>(n, graph::none), {}};
		std::vector<uint32> disc(n, graph::none), low(n);
		std::vector<uint32> open;
		std::vector<detail::bcc_lowlink_frame> frames;
		uint32 timer = 0;
		for (uint32 s = 0; s < n; ++s) {
			if (disc[s] != graph::none) continue;
			disc[s] = low[s] = timer++;
			frames.push_back({s, 0, graph::none, 0});
			while (!frames.empty()) {
				detail::bcc_lowlink_frame &f = frames.back();
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
				if (frames.empty()) continue; // DFS root: no tree edge above
				uint32 const parent = frames.back().v;
				low[parent] = (std::min)(low[parent], low[v]);
				if (low[v] >= disc[parent]) {
					auto const id = uint32(r.top_vertices.size());
					r.top_vertices.push_back(parent);
					uint32 w = 0;
					do {
						w = open.back();
						open.pop_back();
						r.block[w] = id;
					} while (w != v);
				}
			}
		}
		return r;
	}

	/**
	 * @brief Articulation points via Schmidt's chain decomposition — the
	 *        readable reference (it emits no certificate): a vertex is an articulation point iff it
	 *        starts a cycle chain other than its component's first chain, or
	 *        it is incident to a bridge and has another (non-self-loop) edge.
	 *        Blocks are not produced; this is the readable cut-vertex oracle.
	 * @pre net is a symmetric digraph, net.vertex_count() < graph::none
	 * @complexity O(n + m) time, O(n) heap storage; no depth-proportional recursion.
	 *
	 * Two passes and parent walks take 505 vs library 260 ms on random m=4e6, although only cuts are computed.
	 * Xeon Platinum 8280, GCC 15.2.0, -O2, 2026-10-15.
	 * @param net Input graph; valid targets, unchanged adjacency.
	 * @return A 0/1 articulation mask.
	 * @note Not constexpr (runtime comparison; constant evaluation is not enabled).
	 */
	[[nodiscard]] inline std::vector<uint8> cut_vertex_mask_by_chain_decomposition(graph::csr<graph::vertex_id> const &net) {
		atl_assert(net.vertex_count() < graph::none);
		auto const n = uint32(net.vertex_count());
		std::vector<uint32> disc(n, graph::none), parent(n, graph::none);
		std::vector<uint32> order;
		order.reserve(n);
		{
			std::vector<detail::bcc_chain_frame> frames;
			uint32 timer = 0;
			for (uint32 s = 0; s < n; ++s) {
				if (disc[s] != graph::none) continue;
				disc[s] = timer++;
				order.push_back(s);
				frames.push_back({s, 0});
				while (!frames.empty()) {
					detail::bcc_chain_frame &f = frames.back();
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
		std::vector<uint8> cut(n, 0), visited(n, 0), covered(n, 0), tree_arc_seen(n, 0);
		bool is_first_pending = false; // the component's first chain is always a cycle and never marks
		for (uint32 const u : order) {
			if (parent[u] == graph::none) is_first_pending = true;
			visited[u] = 1;
			for (uint32 const w : net.out_edges(u)) {
				// the deeper endpoint owns the edge; `<=` also drops self-loops
				if (disc[w] <= disc[u]) continue;
				if (parent[w] == u && !tree_arc_seen[w]) {
					tree_arc_seen[w] = 1;
					continue;
				}
				uint32 x = w;
				while (!visited[x]) {
					visited[x] = 1;
					covered[x] = 1;
					x = parent[x];
				}
				if (x == u && !is_first_pending) cut[u] = 1;
				is_first_pending = false;
			}
		}
		// bridges: uncovered tree edges; an endpoint is a cut vertex iff it
		// still has another non-self-loop edge
		std::vector<uint32> degree(n, 0);
		for (uint32 v = 0; v < n; ++v)
			for (uint32 const w : net.out_edges(v)) degree[v] += w != v;
		for (uint32 const v : order)
			if (parent[v] != graph::none && !covered[v]) {
				if (degree[v] >= 2) cut[v] = 1;
				if (degree[parent[v]] >= 2) cut[parent[v]] = 1;
			}
		return cut;
	}

	/**
	 * @brief Gabow search storing both child vertex and parent discovery index
	 * in each boundary, and the DFS parent in every frame.
	 * The extra state costs 254 vs library 213 ms on a 4e6-vertex path with
	 * chords (Xeon Platinum 8280, GCC 15.2.0, -O2, 2026-10-15);
	 * random trees take 80.8 vs 83.8 ms. See docs/graph-components.md.
	 * @tparam Graph Scannable adjacency graph.
	 * @param g Symmetric adjacency, unchanged during the call.
	 * @pre Same input preconditions as atl::graph::biconnected_components.
	 * @return Dense blocks with the library encoding; ids have no ordering guarantee.
	 * @complexity O(n + m) time, O(n) heap storage; no depth-proportional recursion.
	 */
	template<graph::detail::scannable_graph Graph>
	[[nodiscard]] constexpr graph::block_decomposition biconnected_components_by_paired_boundaries(Graph const &g) {
		atl_assert(graph::vertex_count(g) < graph::none);
		auto const n = uint32(graph::vertex_count(g));
		graph::block_decomposition r{std::vector<uint32>(n, graph::none), {}};
		std::vector<uint32> disc(n, graph::none);
		std::vector<graph::vertex_id> open;

		struct boundary {
			graph::vertex_id v;
			uint32 parent_discovery;
		};

		std::vector<boundary> bounds;
		using scan = graph::detail::edge_scan<Graph>;

		struct frame {
			graph::vertex_id v, parent;
			[[no_unique_address]] typename scan::position at;
			uint8 has_skipped_parent;
		};

		std::vector<frame> frames;
		uint32 timer = 0;
		for (graph::vertex_id s = 0; s < n; ++s) {
			if (disc[s] != graph::none) continue;
			disc[s] = timer++;
			frames.push_back({s, graph::none, scan::initial(g, s), 0});
			while (!frames.empty()) {
				frame &f = frames.back();
				auto es = scan(g, f.v, f.at);
				bool has_descended = false;
				while (!es.exhausted()) {
					graph::vertex_id const w = graph::target(es.current());
					atl_assert(w < n);
					es.advance();
					if (w == f.parent && !f.has_skipped_parent) {
						f.has_skipped_parent = 1;
						continue;
					}
					if (disc[w] == graph::none) {
						f.at = es.pos(); // Stack growth can invalidate f; save the resume position first.
						disc[w] = timer++;
						open.push_back(w);
						bounds.push_back({w, disc[f.v]});
						frames.push_back({w, f.v, scan::initial(g, w), 0});
						has_descended = true;
						break;
					}
					while (!bounds.empty() && bounds.back().parent_discovery > disc[w]) bounds.pop_back();
				}
				if (has_descended) continue;
				graph::vertex_id const v = frames.back().v;
				frames.pop_back();
				if (frames.empty()) continue; // v roots a DFS tree: no tree edge above
				if (!bounds.empty() && bounds.back().v == v) {
					// v's boundary survived: nothing in its subtree reaches
					// above its parent, the block through `parent -> v`
					// closes; its still-open vertices are exactly the top
					// run of `open` down through v
					bounds.pop_back();
					auto const id = uint32(r.top_vertices.size());
					r.top_vertices.push_back(frames.back().v);
					graph::vertex_id w = 0;
					do {
						w = open.back();
						open.pop_back();
						r.block[w] = id;
					} while (w != v);
				}
			}
		}
		return r;
	}
}
