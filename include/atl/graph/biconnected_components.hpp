#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/partition.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <limits>
#include <vector>

namespace atl::graph {
	/**
	 * @brief Edge blocks encoded by tree-edge labels and shallowest vertices,
	 * without edge identities. Produced by @ref atl::graph::biconnected_components.
	 *
	 * Blocks partition *edges*, but tree edges are in bijection with non-root
	 * vertices and every block is determined by its tree edges — so `block[v]`
	 * stores the block of the tree edge above `v`. A non-isolated, non-cut vertex belongs to
	 * exactly one block; its entry names it unless it is a DFS root. A cut
	 * vertex belongs to several and its entry is the one above it (or `none` for a root) in this run's
	 * DFS tree. The one shallowest vertex of each block (an articulation point
	 * or a DFS root) closes the vertex set: see `top_vertices` and @ref atl::graph::block_decomposition::groups.
	 *
	 * Self-loops belong to no block and never influence cuts; a vertex with
	 * only self-loops is treated as isolated (`block[v] == none`, no group).
	 * @pre Public arrays encode a valid DFS block decomposition on fewer than
	 * `none` vertices: every non-sentinel block id is below `count()`, each top
	 * is a valid vertex outside its block's non-root entries, and every block
	 * has at least one non-root entry. Preserve these invariants on mutation.
	 * Returned masks and groups own independent storage; spans from flat groups
	 * are invalidated by reallocation or destruction of that group's members.
	 */
	struct block_decomposition {
		/**
		 * @brief Per vertex: id of the block containing the tree edge above it;
		 * `none` for DFS roots and isolated vertices.
		 */
		std::vector<uint32> block;
		/**
		 * @brief Per block id: its shallowest vertex — the articulation point (or
		 * DFS root) through which the block attaches upward.
		 */
		std::vector<vertex_id> top_vertices;

		/** @brief Number of edge blocks. @return `top_vertices.size()`. @complexity O(1) */
		[[nodiscard]] constexpr std::size_t count() const { return top_vertices.size(); }

		/**
		 * @brief The block containing edge `{u, v}`. Two blocks share at most
		 *        one vertex, so the edge's block is `block[u]` when that block
		 *        also holds v (as v's own entry or as its top), else `block[v]`.
		 * @param u First endpoint. @param v Second endpoint.
		 * @pre Both endpoints are valid and distinct (asserted), and `{u, v}` is
		 * an edge of the decomposed graph (documented only).
		 * @return The edge's block id in `[0, count())`.
		 * @complexity O(1)
		 */
		[[nodiscard]] constexpr uint32 edge_block(vertex_id u, vertex_id v) const {
			atl_assert(u < block.size() && v < block.size() && u != v);
			uint32 const bu = block[u];
			if (bu != none && (block[v] == bu || top_vertices[bu] == v)) return bu;
			atl_assert(block[v] != none && top_vertices[block[v]] == u);
			return block[v];
		}

		/**
		 * @brief 0/1 mask of articulation points: a non-root vertex is one iff it is
		 * the top of at least one block, a root iff it is the top of at least two.
		 * @return One 0/1 entry per vertex.
		 * @complexity O(n + count()) time, O(n) storage.
		 */
		[[nodiscard]] constexpr std::vector<uint8> cut_vertex_mask() const {
			std::vector<uint8> cut(block.size(), 0);
			for (auto const t : top_vertices)
				if (cut[t] < 2) ++cut[t]; // saturating: only 0/1/2+ matters
			for (vertex_id v = 0; v < block.size(); ++v)
				cut[v] = cut[v] >= (block[v] == none ? 2 : 1);
			return cut;
		}

		/**
		 * @brief The vertex lists of @ref atl::graph::block_decomposition::groups as one flat array (no per-block allocation): the
		 *        flat form. The order matches groups().
		 * @pre `count() + 1` fits `size_t`, and total membership fits `uint32`
		 * (both asserted); total membership can be
		 * as large as twice the number of non-root vertices.
		 * @return Independent flat storage.
		 * @complexity O(n + count()) time and storage.
		 */
		[[nodiscard]] constexpr vertex_groups grouped() const {
			std::size_t const blocks = top_vertices.size();
			atl_assert(blocks < (std::numeric_limits<std::size_t>::max)());
			vertex_groups out{std::vector<uint32>(blocks + 1, 0), {}};
			for (auto const b : block)
				if (b != none) ++out.offsets[b + 1];
			uint64 total = 0;
			for (std::size_t b = 0; b < blocks; ++b) {
				total += uint64(out.offsets[b + 1]) + 1;
				atl_assert(total <= (std::numeric_limits<uint32>::max)());
				out.offsets[b + 1] = uint32(total);
			}
			out.members.resize(out.offsets[blocks]);
			std::vector<uint32> cursor(out.offsets.begin(), out.offsets.end() - 1);
			for (std::size_t b = 0; b < blocks; ++b) out.members[cursor[b]++] = top_vertices[b];
			for (vertex_id v = 0; v < block.size(); ++v)
				if (block[v] != none) out.members[cursor[block[v]]++] = v;
			return out;
		}

		/**
		 * @brief The vertices of every block: its top first, then the vertices whose
		 * tree edge lies in the block, ascending. Cut vertices appear in each
		 * block they touch.
		 * @return Independent nested storage, with no group for isolated vertices.
		 * @complexity O(n + count()) time and storage.
		 */
		[[nodiscard]] constexpr std::vector<std::vector<vertex_id>> groups() const {
			std::vector<std::vector<vertex_id>> out(top_vertices.size());
			std::vector<uint32> sizes(top_vertices.size(), 1);
			for (auto const b : block)
				if (b != none) ++sizes[b];
			for (std::size_t b = 0; b < top_vertices.size(); ++b) {
				out[b].reserve(sizes[b]);
				out[b].push_back(top_vertices[b]);
			}
			for (vertex_id v = 0; v < block.size(); ++v)
				if (block[v] != none) out[block[v]].push_back(v);
			return out;
		}
	};

	/**
	 * @brief Edge blocks and articulation points from an iterative Gabow search
	 * with pending boundaries instead of a low-link array.
	 * @tparam Graph Scannable adjacency graph, as for atl::graph::scc.
	 * @param g Undirected graph represented by symmetric adjacency, unchanged
	 * throughout the call.
	 * @pre `vertex_count(g) < none`; every target is below `vertex_count(g)`
	 * (asserted). Index-resumed rows have at most `none` arcs (asserted).
	 * Each undirected edge contributes one arc in each direction,
	 * including every parallel copy (documented only). Repeated row access
	 * preserves order and reference-returning rows remain valid during the call.
	 * @return An owning atl::graph::block_decomposition. Block ids are dense
	 * in `[0, count())`, with no ordering guarantee. Self-loops belong to no
	 * block; isolated and loop-only vertices have no group. Parallel copies
	 * share a block, using exactly one skipped parent arc per DFS frame.
	 * @complexity O(n + m) time and O(n) auxiliary storage, assuming O(1)
	 * graph access, target extraction and iterator operations. Explicit heap stacks avoid recursion
	 * proportional to path depth. Allocation failures propagate.
	 */
	template<detail::scannable_graph Graph>
	[[nodiscard]] constexpr block_decomposition biconnected_components(Graph const &g) {
		atl_assert(graph::vertex_count(g) < none);
		auto const n = uint32(graph::vertex_count(g));
		block_decomposition r{std::vector<uint32>(n, none), {}};
		std::vector<uint32> disc(n, none);
		std::vector<vertex_id> open;
		// a pending block boundary is its child's parent's discovery index:
		// live boundaries belong to path vertices, whose parents' indices
		// strictly increase up the stack, so v's own boundary is on top at
		// v's postorder iff `bounds.back() == disc[parent(v)]`
		std::vector<uint32> bounds;
		using scan = detail::edge_scan<Graph>;

		// the DFS parent is not stored: it is always the vertex of the frame
		// below, avoiding a redundant field
		struct frame {
			vertex_id v;
			[[no_unique_address]] typename scan::position at;
			uint8 has_skipped_parent;
		};

		std::vector<frame> frames;
		uint32 timer = 0;
		for (vertex_id s = 0; s < n; ++s) {
			if (disc[s] != none) continue;
			disc[s] = timer++;
			frames.push_back({s, scan::initial(g, s), 0});
			while (!frames.empty()) {
				frame &f = frames.back();
				vertex_id const parent = frames.size() > 1 ? frames[frames.size() - 2].v : none;
				auto es = scan(g, f.v, f.at);
				bool has_descended = false;
				while (!es.exhausted()) {
					vertex_id const w = graph::target(es.current());
					atl_assert(w < n);
					es.advance();
					if (w == parent && !f.has_skipped_parent) {
						f.has_skipped_parent = 1;
						continue;
					}
					if (disc[w] == none) {
						f.at = es.pos(); // Stack growth can invalidate f; save the resume position first.
						disc[w] = timer++;
						open.push_back(w);
						bounds.push_back(disc[f.v]);
						frames.push_back({w, scan::initial(g, w), 0});
						has_descended = true;
						break;
					}
					while (!bounds.empty() && bounds.back() > disc[w]) bounds.pop_back();
				}
				if (has_descended) continue;
				vertex_id const v = frames.back().v;
				frames.pop_back();
				if (frames.empty()) continue; // v roots a DFS tree: no tree edge above
				vertex_id const up = frames.back().v;
				if (!bounds.empty() && bounds.back() == disc[up]) {
					// v's boundary survived: nothing in its subtree reaches
					// above its parent, the block through `up -> v` closes;
					// its still-open vertices are exactly the top run of
					// `open` down through v
					bounds.pop_back();
					auto const id = uint32(r.top_vertices.size());
					r.top_vertices.push_back(up);
					vertex_id w = 0;
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
