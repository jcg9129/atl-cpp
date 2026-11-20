#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::graph {
	/**
	 * @brief One vertex of a separator's component, as seen from that
	 *        separator: hop distance, parent within the component, and
	 *        which branch (child subtree of the separator) it belongs to.
	 */
	struct centroid_vertex {
		vertex_id vertex; ///< the vertex itself
		vertex_id parent; ///< its parent on the path to the separator (`none` for the separator)
		uint32 depth;	  ///< hop distance to the separator
		uint32 branch;	  ///< index of the separator's child subtree holding it, `none` for the separator
	};

	/**
	 * @brief The @ref atl::graph::centroid_visit event: `f(center, parent_center,
	 *        component)`. A three-argument callback pays nothing for edges
	 *        (the edge scratch is never built); see @ref atl::graph::centroid_edge_event
	 *        for the four-argument form.
	 */
	template<typename F>
	concept centroid_event =
		std::invocable<F &, vertex_id, vertex_id, std::span<centroid_vertex const>>;

	/**
	 * @brief The four-argument @ref atl::graph::centroid_visit event over adjacency
	 *        element @p Edge: `f(center, parent_center, component, edges)`,
	 *        where `edges` is a `span<Edge const>` parallel to `component`
	 *        and `edges[i]` is the adjacency element on the parent ->
	 *        component[i] edge (value-initialized at the center) — for
	 *        weighted/custom top-down values, since every record's parent
	 *        appears earlier in the span. The engine stores the elements by
	 *        value, so @p Edge must be default-initializable and copyable.
	 *        A callback modeling both forms takes this one.
	 */
	template<typename F, typename Edge>
	concept centroid_edge_event =
		std::default_initializable<Edge> && std::copyable<Edge>
		&& std::invocable<F &, vertex_id, vertex_id, std::span<centroid_vertex const>, std::span<Edge const>>;

	namespace detail {
		/**
		 * @brief BFS order of a forest, one tree after another, with `order`
		 *        itself as the queue: no per-root traversal object, so a forest
		 *        of tiny trees costs O(n) with no allocation per tree (measured:
		 *        docs/centroid-decomposition.md).
		 * @complexity O(n + m)
		 */
		template<adjacency_graph Graph>
		[[nodiscard]] constexpr std::vector<uint32> forest_bfs_order(Graph const &g) {
			std::size_t const n = graph::vertex_count(g);
			atl_assert(n < (uint64(1) << 31));
			std::vector<uint32> order(n);
			dense_marker marks(n);
			std::size_t tail = 0;
			for (uint32 s = 0; s < n; ++s) {
				if (!marks.try_visit(s)) continue;
				std::size_t head = tail;
				order[tail++] = s;
				for (; head < tail; ++head)
					for (auto const &e : graph::out_edges(g, order[head])) {
						uint32 const w = graph::target(e);
						if (marks.try_visit(w)) order[tail++] = w;
					}
			}
			return order;
		}

		/**
		 * @brief One step of the minimum-ranking DP: the one-hot label of a
		 *        vertex whose children's visible-rank masks OR to @p used, with
		 *        the ranks seen in two or more children in @p duplicated.
		 *
		 * The label must exceed every duplicated rank and be absent from
		 * @p used: smear-and-increment turns the duplicate set into the minimum
		 * allowed bit, and `possible & -possible` is the label. Ranks are
		 * bounded by the centroid argument at floor(log2 n) + 1 <= 31 (asserted).
		 */
		[[nodiscard]] constexpr uint32 next_rank(uint32 used, uint32 duplicated) {
			// one past the highest duplicated bit, branchless
			// (bit_width(0) == 0 covers the no-duplicates case); ranks
			// are < 32, so the shift never reaches bit 32
			uint32 const floor_bit = uint32(1) << std::bit_width(duplicated);
			uint32 const possible = ~used & (~floor_bit + 1);
			uint32 const one_hot = possible & (~possible + 1);
			atl_assert(one_hot != 0);
			return one_hot;
		}

		/**
		 * @brief Minimum vertex ranking (= tree-depth labeling) of a forest,
		 *        bottom-up over one BFS order, as one-hot labels.
		 *
		 * Each subtree is summarized by the mask of its *visible* ranks (not
		 * shadowed by a larger rank on the way up); at v, ranks duplicated
		 * across children must fall below v's rank and v's rank must be
		 * absent from their union (`next_rank`). Everything stays in one-hot
		 * form, so rank comparisons are plain compares of the one-hot values.
		 * The reverse BFS order makes the parent test vanish — a parent's mask
		 * is still zero when its children fold it in.
		 * @pre @p g is a forest given as a symmetric adjacency (documented only),
		 * with fewer than 2^31 vertices (asserted). Targets are valid vertices.
		 * Callbacks must not mutate the graph; spans expire when the callback returns.
		 * Traversals use heap buffers and never recurse per vertex.
		 * @complexity O(n + m)
		 */
		template<adjacency_graph Graph>
		[[nodiscard]] constexpr std::vector<uint32> vertex_ranking(Graph const &g) {
			std::size_t const n = graph::vertex_count(g);
			atl_assert(n < (uint64(1) << 31));
			auto const order = forest_bfs_order(g);
			std::vector<uint32> label(n), mask(n, 0);
			for (std::size_t i = n; i-- > 0;) {
				uint32 const v = order[i];
				uint32 used = 0, duplicated = 0;
				for (auto const &e : graph::out_edges(g, v)) {
					uint32 const m = mask[graph::target(e)]; // parent still 0: no-op
					duplicated |= used & m;
					used |= m;
				}
				uint32 const one_hot = next_rank(used, duplicated);
				label[v] = one_hot;
				mask[v] = (used | one_hot) & (~one_hot + 1);
			}
			return label;
		}
	}

	/**
	 * @brief Centroid-style divide and conquer over a tree or forest,
	 *        event-driven: `f(center, parent_center, component)` fires once
	 *        per vertex acting as a separator. `component` is the span of
	 *        vertices whose path to the center crosses no stronger
	 *        separator — the center itself first (depth 0, branch `none`),
	 *        then branch-contiguous groups with hop depths and in-component
	 *        parents. Every unordered vertex pair appears together in
	 *        **exactly one** event with distinct branches (or one of them
	 *        the center): the canonical way to count or aggregate over all
	 *        paths. The span lives in shared scratch — valid only during
	 *        the callback.
	 *
	 * The engine is minimum vertex ranking (see docs/centroid-decomposition.md; the classic
	 * recursive centroid engine lost the race and lives in
	 * atl/experimental/graph/centroid_variants.hpp): one O(n) bottom-up
	 * mask DP labels the tree so that any path's maximum label is unique,
	 * then each vertex's component is a plain DFS over smaller labels — no
	 * blocked markers, no size recomputation and no recursion. A vertex joins
	 * at most one component per rank, so total event size is at most
	 * n·(floor(log2 n) + 1) — and the ranking is *optimal*, so usually
	 * less. Events fire in decreasing label order (each tree's top
	 * separator first); `parent_center` is the weakest stronger separator
	 * whose component contains the center (`none` at the top).
	 *
	 * Within a component the order is parent-first (every record's
	 * parent appears earlier in the span) with branches contiguous — not
	 * a true preorder: a subtree is not a contiguous range.
	 * @pre @p g is a forest given as a symmetric adjacency (documented only),
	 * with fewer than 2^31 vertices (asserted). Targets are valid vertices.
	 * Callbacks must not mutate the graph; spans expire when the callback returns.
	 * Traversals use heap buffers and never recurse per vertex.
	 * @tparam Graph Adjacency graph of a forest.
	 * @tparam Event Callback satisfying centroid_event or centroid_edge_event.
	 * @param g Forest to decompose.
	 * @param f Receives each separator and its temporary component records.
	 * @complexity O(n log n) time, O(n) space
	 */
	template<adjacency_graph Graph, typename Event>
		requires(centroid_event<Event> || centroid_edge_event<Event, edge_t<Graph>>)
			 && (!centroid_edge_event<Event, edge_t<Graph>> || requires(edge_t<Graph> &stored, std::remove_reference_t<std::ranges::range_reference_t<out_edges_t<Graph>>> const &element) { stored = element; })
	constexpr void centroid_visit(Graph const &g, Event f) {
		using Edge = edge_t<Graph>;
		constexpr bool wants_edges = centroid_edge_event<Event, Edge>;
		std::size_t const n = graph::vertex_count(g);
		atl_assert(n < (uint64(1) << 31));
		if (n == 0) return;
		auto const label = detail::vertex_ranking(g);
		// decreasing labels: every stronger separator sweeps ahead of v's own
		// event, so v's parent_center is final by the time v emits — and
		// events come top-separator-first. Labels are one-hot (<= 32 distinct
		// ranks) and nothing orders vertices within a rank, so a direct bucket
		// fill replaces a comparison sort (bucket_by_key ties with it: record
		// graph-centroid_decomposition 2026-10-15)
		std::vector<uint32> by_label(n);
		{
			std::array<uint32, 34> pos{};
			for (uint32 v = 0; v < n; ++v) ++pos[std::bit_width(label[v])];
			uint32 at = 0;
			for (uint32 r = 34; r-- > 0;) { // high rank first
				uint32 const c = pos[r];
				pos[r] = at;
				at += c;
			}
			for (uint32 v = 0; v < n; ++v) by_label[pos[std::bit_width(label[v])]++] = v;
		}

		std::vector<centroid_vertex> scratch(n);
		std::vector<Edge> edge_scratch;					   // parallel to scratch; only when requested
		if constexpr (wants_edges) edge_scratch.resize(n); // slot 0 (the center) stays default
		// indices into scratch (regions are subtrees: parent-avoidance
		// suffices, no visited marks). Using scratch itself as a BFS queue reduces elapsed time by about
		// 14-30% on random and binary trees but adds about 6% on paths, so the
		// stack stays (record graph-centroid_decomposition 2026-10-15)
		std::vector<uint32> stack;
		std::vector<vertex_id> parent_center(n, none);
		for (uint32 const c : by_label) {
			std::size_t count = 0;
			scratch[count++] = {c, none, 0, none};
			uint32 branch = 0;
			for (auto const &e : graph::out_edges(g, c)) {
				uint32 const seed = graph::target(e);
				if (label[seed] >= label[c]) continue;
				if constexpr (wants_edges) edge_scratch[count] = e; // center -> seed
				stack.push_back(uint32(count));
				scratch[count++] = {seed, c, 1, branch};
				while (!stack.empty()) {
					auto const [v, p, depth, br] = scratch[stack.back()];
					stack.pop_back();
					// each sweep that reaches v is a stronger separator;
					// sweeping strongest-first, the last one preceding v's own
					// event is the weakest — v's centroid parent
					parent_center[v] = c;
					for (auto const &e2 : graph::out_edges(g, v)) {
						uint32 const w = graph::target(e2);
						if (w == p || label[w] >= label[c]) continue;
						if constexpr (wants_edges) edge_scratch[count] = e2; // v -> w
						stack.push_back(uint32(count));
						scratch[count++] = {w, v, depth + 1, br};
					}
				}
				++branch;
			}
			std::span<centroid_vertex const> const comp(scratch.data(), count);
			if constexpr (wants_edges)
				f(c, parent_center[c], comp, std::span<Edge const>(edge_scratch.data(), count));
			else
				f(c, parent_center[c], comp);
		}
	}

	/**
	 * @brief The separator tree alone: `parent[v]` = the weakest stronger
	 *        separator whose component contains v (`none` for each tree's
	 *        top separator), i.e. the `parent_center` of v reported by @ref atl::graph::centroid_visit —
	 *        for the query-by-walking-ancestors problem family.
	 *
	 * This is the separator tree of an optimal vertex ranking. Separators need
	 * not halve their components. Its maximum depth is at most the maximum
	 * depth of any centroid tree of the same forest: a centroid tree induces a
	 * valid vertex ranking, and the ranking used here minimizes its height.
	 *
	 * Read straight off the ranking DP, with no component sweep: the parent of
	 * x is the smallest-label vertex with a larger label that sees x (every
	 * vertex strictly between them has a smaller label). Bottom-up, each
	 * subtree keeps its *visible* vertices (no larger label above them in the
	 * subtree) as a list ascending by label, headed by the subtree root. When v
	 * takes label r, each child's visible vertices below r are finalized: their
	 * parent is the next one in their list if that is still below r, else v.
	 * The rest, all above r and pairwise distinct across children by the DP,
	 * merge into v's list behind v; a tree root finalizes its whole list, the
	 * top getting `none`. 4.7-10x faster than reading `parent_center` off the
	 * O(n log n) sweep (record graph-centroid_decomposition 2026-10-15; the
	 * sweep wrapper lives in atl/experimental/graph/centroid_variants.hpp).
	 * @pre @p g is a forest given as a symmetric adjacency (documented only),
	 * with fewer than 2^31 vertices (asserted). Targets are valid vertices.
	 * Callbacks must not mutate the graph; spans expire when the callback returns.
	 * Traversals use heap buffers and never recurse per vertex.
	 * @tparam Graph Adjacency graph of a forest.
	 * @param g Forest to decompose.
	 * @return Separator parent per vertex, or none at each top separator.
	 * @complexity O(n + m) plus the list merges, each over at most 32 ranks
	 */
	template<adjacency_graph Graph>
	[[nodiscard]] constexpr std::vector<vertex_id> centroid_decomposition(Graph const &g) {
		std::size_t const n = graph::vertex_count(g);
		atl_assert(n < (uint64(1) << 31));
		auto const order = detail::forest_bfs_order(g);
		std::vector<uint32> label(n, 0), mask(n, 0);
		std::vector<vertex_id> parent(n, none), next(n, none);
		auto const merge = [&](vertex_id a, vertex_id b) {
			if (a == none) return b;
			if (b == none) return a;
			if (label[b] < label[a]) std::swap(a, b);
			vertex_id const head = a;
			while (b != none) {
				while (next[a] != none && label[next[a]] < label[b]) a = next[a];
				vertex_id const rest = next[a];
				next[a] = b;
				a = b;
				b = rest;
			}
			return head;
		};
		for (std::size_t i = n; i-- > 0;) {
			uint32 const v = order[i];
			uint32 used = 0, duplicated = 0;
			for (auto const &e : graph::out_edges(g, v)) {
				uint32 const m = mask[graph::target(e)];
				duplicated |= used & m;
				used |= m;
			}
			uint32 const one_hot = detail::next_rank(used, duplicated);
			label[v] = one_hot;
			mask[v] = (used | one_hot) & (~one_hot + 1);
			vertex_id merged = none;
			bool has_parent = false;
			for (auto const &e : graph::out_edges(g, v)) {
				vertex_id x = graph::target(e);
				if (label[x] == 0) { // the BFS parent, not processed yet
					has_parent = true;
					continue;
				}
				for (; x != none && label[x] < one_hot; x = next[x])
					parent[x] = next[x] != none && label[next[x]] < one_hot ? next[x] : v;
				merged = merge(merged, x);
			}
			next[v] = merged;
			if (!has_parent) // a tree root: nothing above, each links to the next
				for (vertex_id x = v; x != none; x = next[x]) parent[x] = next[x];
		}
		return parent;
	}
}
