#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/counting_sort.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::graph {
	/**
	 * @brief A distance-preserving ternarization of a rooted tree: every
	 *        vertex has degree at most three — rooted, that means at most two
	 *        children — with a vertex's extra children fanned out through
	 *        zero-weight "fake" vertices.
	 *
	 * Real edges keep their weight, so the distance between any two
	 * *original* vertices is unchanged; the payoff is bounded fan-in for
	 * balanced tree divide-and-conquer (convolution trees, sorted-merge
	 * distance counting, top trees). Real vertices keep their original ids
	 * `[0, real_count)`; fakes take ids `>= real_count`, so telling them
	 * apart is one compare (`is_real`), no lookup table.
	 * @tparam Weight Stored edge payload type, selected by the producer.
	 */
	template<typename Weight>
	struct ternarization {
		csr<edge<Weight>> tree; ///< Binarized weighted tree, degree <= 3.
		vertex_id root;			///< Real root, retaining its input id.
		uint32 real_count;		///< Input vertex count; smaller ids are real.

		/** @brief Whether vertex @p v of `tree` is an original (not a fake) vertex. */
		[[nodiscard]] constexpr bool is_real(vertex_id v) const noexcept { return v < real_count; }
	};

	namespace detail {
		template<adjacency_graph Graph, bool = weighted_adjacency_graph<Graph>>
		struct ternarize_weight {
			using type = int;
		};

		template<adjacency_graph Graph>
		struct ternarize_weight<Graph, true> {
			using type = weight_t<Graph>;
		};

		template<adjacency_graph Graph, typename Override>
		using ternarize_weight_t =
			std::conditional_t<std::is_void_v<Override>, typename ternarize_weight<Graph>::type,
							   Override>;
	}

	/**
	 * @brief An explicit weight type for @ref atl::graph::ternarization_of over graph
	 *        @p Graph: `void` (keep the default), or a value-initialisable
	 *        type (the zero weight of fake edges) constructible from `1` (unit
	 *        hops) and from the graph's edge weight when it has one.
	 */
	template<typename Weight, typename Graph>
	concept ternarization_weight =
		adjacency_graph<Graph>
		&& (std::is_void_v<Weight>
			|| (std::default_initializable<Weight> && std::copyable<Weight> && std::constructible_from<Weight, int>
				&& (!weighted_adjacency_graph<Graph> || std::constructible_from<Weight, weight_t<Graph>>)));

	/**
	 * @brief The ternarization of the tree @p g rooted at @p root: return a
	 *        distance-preserving tree of maximum degree three (see @ref atl::graph::ternarization). Extra children are fanned through a **size-balanced**
	 *        binary tree of fake vertices, so each convolution / merge up the
	 *        tree combines comparable-size operands — the property that keeps a
	 *        binary convolution tree at O(n log n) rather than degrading on
	 *        high-degree vertices.
	 *
	 * `Weight` defaults to the graph's edge weight (unit `int` hops when
	 * unweighted); fake structural edges get weight 0. Real vertices keep their
	 * ids (`is_real(v)` iff `v < n`), so no csr_bfs-to-new mapping is needed.
	 * @tparam Weight The result's weight type (@ref atl::graph::ternarization_weight);
	 *         `void` (the default) picks the graph's edge weight, or `int`
	 *         when unweighted.
	 * @tparam Graph Adjacency graph of the input tree.
	 * @param g Symmetric adjacency of the input tree.
	 * @param root The root, keeping its id; must be a vertex of @p g.
	 * @pre @p g is a tree, given symmetrically (every edge in both
	 * directions, as @ref atl::graph::csr::undirected builds): connected and acyclic
	 * (both asserted — a forest or an extra edge is rejected)
	 * @pre Fewer than 2^31 input vertices; output vertices at most none / 2
	 * (asserted). Value initialization of Weight represents zero. Allocation
	 * failures propagate. Traversals use heap buffers; fan-out uses an
	 * explicit O(log n) frame stack; there is no per-vertex call-stack recursion.
	 * @return A weighted tree preserving distances between real vertices.
	 * @complexity O(n log n) time and O(n) storage — the size-balanced fan-out is the only superlinear
	 * part; the result has fewer than 2n vertices
	 */
	template<typename Weight = void, adjacency_graph Graph>
		requires ternarization_weight<Weight, Graph>
			  && std::copyable<detail::ternarize_weight_t<Graph, Weight>>
			  && std::default_initializable<detail::ternarize_weight_t<Graph, Weight>>
			  && (weighted_adjacency_graph<Graph> || std::constructible_from<detail::ternarize_weight_t<Graph, Weight>, int>)
			  && (!weighted_adjacency_graph<Graph> || requires(std::remove_reference_t<std::ranges::range_reference_t<out_edges_t<Graph>>> const &element) {
					 static_cast<detail::ternarize_weight_t<Graph, Weight>>(graph::weight(element));
				 })
	[[nodiscard]] constexpr auto ternarization_of(Graph const &g, vertex_id root) {
		using W = detail::ternarize_weight_t<Graph, Weight>;
		std::size_t const n = graph::vertex_count(g);
		atl_assert(n < (uint64(1) << 31));
		atl_assert(root < n);

		std::vector<vertex_id> parent(n, root);
		std::vector<W> parent_weight(n, W{});
		std::vector<uint32> order;
		order.reserve(n);
		dense_marker seen(n);
		order.push_back(root);
		seen.try_visit(root);
		// a symmetric tree rejects exactly its n - 1 child-to-parent arcs;
		// any extra edge adds two more rejections
		[[maybe_unused]] std::size_t rejected = 0;
		for (std::size_t at = 0; at < order.size(); ++at) {
			uint32 const v = order[at];
			for (auto const &e : graph::out_edges(g, v)) {
				uint32 const w = graph::target(e);
				if (!seen.try_visit(w)) {
					atl_assert(++rejected < n); // see @pre: acyclic
					continue;
				}
				parent[w] = v;
				if constexpr (weighted_adjacency_graph<Graph>)
					parent_weight[w] = W(graph::weight(e));
				else
					parent_weight[w] = W(1);
				order.push_back(w);
			}
		}
		atl_assert(order.size() == n && rejected == n - 1); // see @pre: connected, acyclic

		// children CSR (sorted by vertex id: the fan-out loop below then reads
		// `child` sequentially — measured faster than using the contiguous
		// child runs BFS leaves in `order`, whose runs are id-scattered) +
		// subtree sizes, bottom-up over the BFS order (<= n < 2^32: uint32).
		// The root goes to the extra bucket n, so `child` ends with it; on a
		// single tree atl::detail::bucket_by_key ties or beats the hand-rolled
		// count/cursor passes (benchmarks/graph/lca.cpp, "children CSR")
		std::vector<uint32> child_offset(n + 2), child(n);
		atl::detail::bucket_by_key(
			n + 1, [&](std::size_t v) { return v == root ? n : std::size_t(parent[v]); }, child_offset, child);
		std::vector<uint32> subtree_size(n, 1);
		for (std::size_t i = n; i-- > 0;) {
			uint32 const v = order[i];
			if (v != root) subtree_size[parent[v]] += subtree_size[v];
		}

		// Every output degree is known without creating edges: a real vertex has
		// its parent (unless root) plus min(children, 2) children, and a vertex
		// with k >= 2 children spawns k - 2 fakes of degree 3 (parent + two
		// children). So the row table is two sequential passes and the edges are
		// placed directly, both half-edges at emission time, in the order a
		// counting-sort build of the same arcs would give.
		std::size_t fakes = 0;
		for (std::size_t v = 0; v < n; ++v) {
			std::size_t const k = child_offset[v + 1] - child_offset[v];
			if (k > 2) fakes += k - 2;
		}
		std::size_t const total = n + fakes;
		atl_assert(total <= (std::numeric_limits<uint32>::max)() / 2); // the stored edge count is below 2 * total
		std::vector<uint32> offset(total + 1);
		offset[0] = 0;
		for (std::size_t v = 0; v < n; ++v) {
			std::size_t const k = child_offset[v + 1] - child_offset[v];
			offset[v + 1] = offset[v] + (v != root) + uint32(std::min<std::size_t>(k, 2));
		}
		for (std::size_t f = n; f < total; ++f) offset[f + 1] = offset[f] + 3;
		std::vector<edge<W>> edges(offset[total]);
		// offset[u] doubles as u's write cursor and ends at u's slice end, as in csr's own build
		auto const link = [&](uint32 u, uint32 x, W const &w) {
			edges[offset[u]++] = {x, w};
			edges[offset[x]++] = {u, w};
		};

		uint32 next_fake = uint32(n);
		std::vector<uint32> prefix_size; // prefix subtree sizes of the current child list
		auto const edge_weight = [&](uint32 v) -> W { return v < n ? parent_weight[v] : W{}; };

		struct fanout_frame {
			uint32 lo = 0, hi = 0, mid = 0, fake = 0, left = 0;
			unsigned state = 0;
		};

		std::vector<fanout_frame> fanout;
		auto const build = [&](uint32 const *ch, uint32 lo, uint32 hi) -> uint32 {
			fanout.clear();
			fanout.push_back({lo, hi});
			uint32 result = 0;
			while (!fanout.empty()) {
				auto &frame = fanout.back();
				if (frame.hi - frame.lo == 1) {
					result = ch[frame.lo];
					fanout.pop_back();
				} else if (frame.state == 0) {
					uint32 const target = prefix_size[frame.lo] + (prefix_size[frame.hi] - prefix_size[frame.lo]) / 2;
					frame.mid = uint32(std::ranges::lower_bound(prefix_size.begin() + frame.lo + 1,
																prefix_size.begin() + frame.hi, target)
									   - prefix_size.begin());
					frame.mid = std::clamp(frame.mid, frame.lo + 1, frame.hi - 1);
					frame.fake = next_fake++;
					frame.state = 1;
					fanout.push_back({frame.lo, frame.mid});
				} else if (frame.state == 1) {
					frame.left = result;
					frame.state = 2;
					fanout.push_back({frame.mid, frame.hi});
				} else {
					link(frame.fake, frame.left, edge_weight(frame.left));
					link(frame.fake, result, edge_weight(result));
					result = frame.fake;
					fanout.pop_back();
				}
			}
			return result;
		};
		for (uint32 v = 0; v < n; ++v) {
			uint32 const *ch = child.data() + child_offset[v];
			uint32 const k = child_offset[v + 1] - child_offset[v];
			if (k == 0) continue;
			if (k == 1) {
				link(v, ch[0], parent_weight[ch[0]]);
				continue;
			}
			prefix_size.assign(1, 0); // prefix_size[0] = 0, prefix_size[i] = sizes of ch[0..i)
			for (uint32 i = 0; i < k; ++i) prefix_size.push_back(prefix_size.back() + subtree_size[ch[i]]);
			uint32 const target = prefix_size[k] / 2;
			uint32 mid = uint32(std::ranges::lower_bound(prefix_size.begin() + 1,
														 prefix_size.begin() + k, target)
								- prefix_size.begin());
			mid = std::clamp(mid, uint32(1), k - 1);
			uint32 const a = build(ch, 0, mid);
			uint32 const b = build(ch, mid, k);
			link(v, a, edge_weight(a));
			link(v, b, edge_weight(b));
		}

		atl_assert(next_fake == total);
		std::ranges::copy_backward(offset.begin(), offset.end() - 1, offset.end()); // shift right by one; std::ranges::shift_right needs GCC 14
		offset[0] = 0;
		return ternarization<W>{csr<edge<W>>::adopt(std::move(offset), std::move(edges)), root, uint32(n)};
	}
}
