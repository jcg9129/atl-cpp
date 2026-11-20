#pragma once

/**
 * @file
 * @brief Experimental Eppstein-style k shortest walks using persistent sidetrack heaps.
 * Binary-heap Dijkstra constructs the target tree; forest heap nodes are the
 * states of a best-first search. See docs/persistent-heap.md for contracts and measurements.
 */

#include <atl/ds/persistent_heap.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/graph/detail/edge_count.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental {
	using atl::int64;
	using atl::uint32;
	using atl::graph::vertex_id;

	/** @brief The distance of a vertex that cannot reach the target. */
	inline constexpr int64 unreachable = (std::numeric_limits<int64>::max)();

	namespace detail {
		template<typename EdgeList>
		concept sidetrack_edge_list = graph::weighted_edge_list<EdgeList> && requires(std::ranges::range_reference_t<EdgeList const> e) {
			int64(graph::weight(e));
			{ graph::weight(e) >= 0 } -> std::convertible_to<bool>;
			{ graph::weight(e) < unreachable } -> std::convertible_to<bool>;
		};

		/** @brief Add nonnegative walk costs without colliding with the unreachable sentinel. */
		[[nodiscard]] constexpr int64 walk_sum(int64 a, int64 b) {
			atl_assert(a >= 0 && b >= 0 && b < unreachable - a);
			return a + b;
		}
	}

	/** @brief "No edge": the tree edge of the target and of vertices that cannot reach it. */
	inline constexpr uint32 no_edge = (std::numeric_limits<uint32>::max)();

	/** @brief The shortest-path tree towards one target (the result of @ref atl::experimental::shortest_paths_to). */
	struct shortest_path_tree {
		/** @brief Per vertex, the length of a shortest walk from it to the target, or @ref atl::experimental::unreachable. */
		std::vector<int64> distance;
		/** @brief Per vertex, the index of the first edge of one shortest walk to the target, or @ref atl::experimental::no_edge. */
		std::vector<uint32> tree_edge;
		/** @brief Per vertex, the head of its tree edge (the next vertex of that walk); unset without a tree edge. */
		std::vector<vertex_id> next;
		/** @brief The vertices that reach the target, by nondecreasing distance (the target first). */
		std::vector<vertex_id> order;
	};

	/**
	 * @brief Dijkstra towards `target` over the reversed edges, with a binary
	 *        heap and lazy deletion.
	 * @pre `target < n < no_edge`, fewer than `no_edge` edges, endpoints in range,
	 * and weights nonnegative (asserted). Weights convert exactly to int64 in
	 * `[0, unreachable)`. Every evaluated `distance + weight` is below
	 * `unreachable` (asserted); this includes relaxations that do not improve a label.
	 * @tparam EdgeList Const forward weighted edge range with int64-convertible weights.
	 * @param n Number of vertices.
	 * @param edges Stable edge sequence; tree-edge ids follow its iteration order.
	 * @param target Destination vertex.
	 * @return Owning distances, tree links and settlement order; ties are unspecified.
	 * @note Not constexpr (C++23 std::priority_queue). Uses O(n + m) heap storage,
	 * with O(1) call stack even on million-vertex paths. Allocation failures propagate.
	 * @complexity O(n + m log(m + 2))
	 */
	template<detail::sidetrack_edge_list EdgeList>
	[[nodiscard]] shortest_path_tree shortest_paths_to(std::size_t n, EdgeList const &edges, vertex_id target) {
		atl_assert(n < (std::numeric_limits<uint32>::max)() && target < n);
		std::vector<uint32> offset(n + 1, 0);
		std::size_t m = 0;
		for (auto const &e : edges) {
			atl_assert(atl::graph::source(e) < n && atl::graph::target(e) < n && atl::graph::weight(e) >= 0);
			using weight_type = std::remove_cvref_t<decltype(graph::weight(e))>;
			if constexpr (std::floating_point<weight_type>)
				atl_assert(graph::weight(e) < weight_type(unreachable));
			else
				atl_assert(graph::weight(e) < unreachable);
			atl_assert(m + 1 < no_edge);
			++offset[atl::graph::target(e) + 1];
			++m;
		}
		atl_assert(m < (std::numeric_limits<uint32>::max)());
		for (std::size_t v = 0; v < n; ++v) offset[v + 1] += offset[v];
		std::vector<std::pair<uint32, int64>> incoming(m);
		{
			std::vector<uint32> fill(offset.begin(), offset.end() - 1);
			uint32 index = 0;
			for (auto const &e : edges) incoming[fill[atl::graph::target(e)]++] = {index++, int64(atl::graph::weight(e))};
		}
		std::vector<vertex_id> tail(m);
		{
			uint32 index = 0;
			for (auto const &e : edges) tail[index++] = atl::graph::source(e);
		}

		shortest_path_tree tree{std::vector<int64>(n, unreachable), std::vector<uint32>(n, no_edge), std::vector<vertex_id>(n, 0), {}};
		using entry = std::pair<int64, vertex_id>;
		std::priority_queue<entry, std::vector<entry>, std::greater<>> queue;
		tree.distance[target] = 0;
		queue.emplace(0, target);
		while (!queue.empty()) {
			auto const [d, v] = queue.top();
			queue.pop();
			if (d != tree.distance[v]) continue;
			tree.order.push_back(v);
			for (uint32 i = offset[v]; i < offset[v + 1]; ++i) {
				auto const [edge, weight] = incoming[i];
				vertex_id const u = tail[edge];
				int64 const candidate = detail::walk_sum(d, weight);
				if (candidate < tree.distance[u]) {
					tree.distance[u] = candidate;
					tree.tree_edge[u] = edge;
					tree.next[u] = v;
					queue.emplace(candidate, u);
				}
			}
		}
		return tree;
	}

	/**
	 * @brief A sidetrack: leaving the shortest-path tree by one edge into
	 *        `head`, which makes the walk longer by `extra`.
	 */
	struct sidetrack {
		int64 extra;	///< nonnegative excess over the tree route
		vertex_id head; ///< destination of the sidetrack edge
	};

	/** @brief The heap order of sidetracks: `top` is the cheapest. */
	struct cheaper_sidetrack {
		/** @brief Compare extras for a min-heap. @param a First sidetrack. @param b Second sidetrack. @return Whether a costs more than b. @complexity O(1) */
		[[nodiscard]] constexpr bool operator()(sidetrack const &a, sidetrack const &b) const { return a.extra > b.extra; }
	};

	/** @brief Every vertex's own sidetracks: those of vertex `v` are `items[offset[v] .. offset[v + 1])`. */
	struct sidetrack_lists {
		/** @brief Per vertex, where its sidetracks start in `items` (`n + 1` entries). */
		std::vector<uint32> offset;
		/** @brief Every sidetrack, grouped by tail vertex. */
		std::vector<sidetrack> items;

		/** @brief Borrow the sidetracks leaving v. @param v Vertex below offset.size()-1. @return A span invalidated by items reallocation or destruction. @pre Offsets describe items; v is in range (asserted). @complexity O(1) */
		[[nodiscard]] constexpr std::span<sidetrack const> of(vertex_id v) const {
			atl_assert(std::size_t(v) + 1 < offset.size());
			return std::span<sidetrack const>(items).subspan(offset[v], offset[v + 1] - offset[v]);
		}
	};

	/**
	 * @brief The sidetracks of every vertex: each edge `(u, v, w)` between
	 *        vertices that reach the target, other than `u`'s tree edge, costs
	 *        `w + distance[v] - distance[u] >= 0` extra.
	 * @tparam EdgeList Const forward weighted edge range with int64-convertible weights.
	 * @param n Number of vertices.
	 * @param edges Exactly the stable edge sequence used for tree construction.
	 * @param tree The shortest_paths_to result for these edges and n.
	 * @return Owning sidetracks grouped stably by source vertex.
	 * @pre The shortest_paths_to input bounds hold; distance and tree_edge have n
	 * entries (asserted). The tree is valid for this edge sequence (documented only).
	 * Each computed extra lies in `[0, unreachable)` (asserted).
	 * @note Not constexpr (runtime staging companion to Dijkstra).
	 * @complexity O(n + m) time and heap storage; O(1) call stack.
	 */
	template<detail::sidetrack_edge_list EdgeList>
	[[nodiscard]] sidetrack_lists sidetracks_of(std::size_t n, EdgeList const &edges, shortest_path_tree const &tree) {
		atl_assert(n < no_edge && tree.distance.size() == n && tree.tree_edge.size() == n);
		atl_assert(graph::detail::checked_edge_count(edges, no_edge - 1) < no_edge);
		using scanned_edge = std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>;
		auto const is_sidetrack = [&](scanned_edge const &e, uint32 index) {
			atl_assert(graph::source(e) < n && graph::target(e) < n);
			atl_assert(graph::weight(e) >= 0);
			using weight_type = std::remove_cvref_t<decltype(graph::weight(e))>;
			if constexpr (std::floating_point<weight_type>)
				atl_assert(graph::weight(e) < weight_type(unreachable));
			else
				atl_assert(graph::weight(e) < unreachable);
			vertex_id const u = atl::graph::source(e), v = atl::graph::target(e);
			return tree.distance[u] != unreachable && tree.distance[v] != unreachable && tree.tree_edge[u] != index;
		};
		sidetrack_lists lists{std::vector<uint32>(n + 1, 0), {}};
		uint32 index = 0;
		for (auto const &e : edges)
			if (is_sidetrack(e, index++)) ++lists.offset[atl::graph::source(e) + 1];
		for (std::size_t v = 0; v < n; ++v) lists.offset[v + 1] += lists.offset[v];
		lists.items.resize(lists.offset[n]);
		std::vector<uint32> fill(lists.offset.begin(), lists.offset.end() - 1);
		index = 0;
		for (auto const &e : edges)
			if (is_sidetrack(e, index++)) {
				vertex_id const u = atl::graph::source(e), v = atl::graph::target(e);
				int64 const w = int64(graph::weight(e));
				int64 extra = 0;
				if (tree.distance[v] >= tree.distance[u]) {
					extra = detail::walk_sum(w, tree.distance[v] - tree.distance[u]);
				} else {
					atl_assert(w >= tree.distance[u] - tree.distance[v]);
					extra = w - (tree.distance[u] - tree.distance[v]);
				}
				lists.items[fill[u]++] = {extra, v};
			}
		return lists;
	}

	/**
	 * @brief The lengths of the `k` shortest walks from `source` to `target`
	 *        (with repeats, nondecreasing; fewer when fewer walks exist).
	 *
	 * Eppstein's construction. A walk is the shortest-path tree route from
	 * `source`, deviated by a sequence of sidetracks (@ref atl::experimental::sidetracks_of),
	 * each taken from a vertex the tree route of the preceding sidetrack passes, and
	 * its length is `distance[source]` plus their extras. `H[v]`, the heap of
	 * every sidetrack leaving the tree route from `v`, is `H[parent of v]`
	 * plus `v`'s own sidetracks: one `push_range` on a persistent heap, in
	 * Dijkstra order, so the n heaps share their nodes (O(m log m) expected
	 * in all).
	 *
	 * The walks are then the paths of an implicit graph whose nodes are the
	 * heap nodes: from heap node `x`, go to its two children (swap the last
	 * sidetrack for a costlier one of the same heap) or to the root of
	 * `H[head of x]` (append a sidetrack). Every arc costs a nonnegative
	 * amount, and each walk is one path, so a best-first search over this
	 * graph pops the `k` answers in order. It walks the heap trees by id
	 * (a subtree of a forest version is a version) and makes no node: the
	 * heaps are a @ref atl::persistent_heap_forest, whose 4-byte versions avoid
	 * a pool handle per version (see docs/persistent-heap.md).
	 *
	 * @tparam EdgeList Const forward weighted edge range with int64-convertible weights.
	 * @param n Number of vertices.
	 * @param edges Stable directed edges; parallel copies count as distinct choices.
	 * @param source Starting vertex.
	 * @param target Destination vertex; walks may leave and revisit it.
	 * @param k Maximum number of answers, including the empty walk when source == target.
	 * @return Owning nondecreasing lengths, retaining multiplicity; fewer if exhausted.
	 * @pre The shortest_paths_to and sidetracks_of bounds hold; source < n
	 * (asserted). Every candidate walk sum formed by the search must stay below INT64_MAX
	 * (asserted),
	 * including candidates beyond the requested prefix. Heap pools have fewer
	 * than 2^32-1 nodes. Zero-cost cycles are permitted.
	 * @note Not constexpr (C++23 std::priority_queue). Graph traversals are iterative;
	 * the persistent heap meld is iterative too. Allocation failures propagate.
	 * @complexity O(n + m log(m + 2) + k log(k + 2)) expected time and O(n + m log(m + 2) + k) heap storage.
	 */
	template<detail::sidetrack_edge_list EdgeList>
	[[nodiscard]] std::vector<int64> k_shortest_walks(std::size_t n, EdgeList const &edges, vertex_id source, vertex_id target,
													  std::size_t k) {
		atl_assert(source < n);
		shortest_path_tree const tree = shortest_paths_to(n, edges, target);
		std::vector<int64> lengths;
		if (k == 0 || tree.distance[source] == unreachable) return lengths;
		sidetrack_lists const lists = sidetracks_of(n, edges, tree);

		atl::persistent_heap_forest<sidetrack, cheaper_sidetrack> heaps;
		auto constexpr none = decltype(heaps)::empty_heap;
		std::vector<uint32> sidetracks(n, none);
		for (vertex_id v : tree.order) sidetracks[v] = heaps.push_range(v == target ? none : sidetracks[tree.next[v]], lists.of(v));

		struct candidate {
			int64 length;
			uint32 at;
		};

		auto const longer = [](candidate const &a, candidate const &b) { return a.length > b.length; };
		std::priority_queue<candidate, std::vector<candidate>, decltype(longer)> queue(longer);
		lengths.push_back(tree.distance[source]);
		if (uint32 const h = sidetracks[source]; !heaps.empty(h))
			queue.push({detail::walk_sum(tree.distance[source], heaps.top(h).extra), h});
		while (lengths.size() < k && !queue.empty()) {
			candidate const c = queue.top();
			queue.pop();
			lengths.push_back(c.length);
			sidetrack const here = heaps.top(c.at);
			for (uint32 child : {heaps.left(c.at), heaps.right(c.at)})
				if (!heaps.empty(child)) queue.push({detail::walk_sum(c.length - here.extra, heaps.top(child).extra), child});
			if (uint32 const next = sidetracks[here.head]; !heaps.empty(next))
				queue.push({detail::walk_sum(c.length, heaps.top(next).extra), next});
		}
		return lengths;
	}
}
