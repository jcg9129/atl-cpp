#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::graph {
	namespace detail {
		/**
		 * @brief The adjacency element a list edge naturally compresses to: its
		 *        target alone when unweighted, `edge<W>` when weighted. (A
		 *        partial specialization, not conditional_t — the weighted
		 *        branch must not be instantiated for weightless arcs.)
		 */
		template<list_edge A>
		struct adjacent_of {
			using type = vertex_id;
		};

		template<weighted_list_edge A>
		struct adjacent_of<A> {
			using type = edge<edge_weight_t<A>>;
		};

		template<typename A>
		using adjacent_of_t = typename adjacent_of<A>::type;

		template<typename R>
		using list_reference_t = std::add_lvalue_reference_t<std::add_const_t<
			std::remove_reference_t<std::ranges::range_reference_t<R const>>>>;

		/**
		 * @brief Whether adjacency element @p Edge can be built from list edge
		 *        @p A: an integral @p Edge from the target alone, a weighted
		 *        @p Edge as `Edge{target, weight}` (so @p A must be weighted),
		 *        any other @p Edge as `Edge{target}`.
		 */
		template<typename Edge, typename A>
		concept adjacent_buildable =
			adjacency_edge<Edge> && std::default_initializable<Edge> && std::movable<Edge> && list_edge<A>
			&& (std::integral<Edge>
				|| (weighted_adjacency_edge<Edge>
						? requires(A const &a, vertex_id v) { Edge{v, graph::weight(a)}; }
						: requires(vertex_id v) { Edge{v}; }));

		/**
		 * @brief The adjacency element @p Edge pointing at @p to and carrying
		 *        list edge @p a's weight when @p Edge is weighted (see
		 *        `adjacent_buildable`). Undirected builds pass the source
		 *        here for the reverse direction, so no reversed copy of @p a is
		 *        ever made.
		 */
		template<typename Edge, list_edge A>
			requires adjacent_buildable<Edge, A>
		[[nodiscard]] constexpr Edge adjacent_from(A const &a, vertex_id to) {
			if constexpr (std::integral<Edge>) {
				atl_assert(uint64(to) <= uint64((std::numeric_limits<Edge>::max)()));
				return Edge(to);
			} else if constexpr (weighted_adjacency_edge<Edge>)
				return Edge{to, graph::weight(a)};
			else
				return Edge{to};
		}

		template<typename Edge, list_edge A>
			requires adjacent_buildable<Edge, A>
		[[nodiscard]] constexpr Edge adjacent_from(A const &a) {
			return detail::adjacent_from<Edge>(a, graph::target(a));
		}
	}

	/**
	 * @brief Compressed sparse row adjacency: all edges in one flat array,
	 *        per-vertex slices via an offset table. The cache-friendly static
	 *        representation — build once from an edge list, traverse fast.
	 *
	 * Models @ref atl::graph::adjacency_graph. Edges of a vertex keep the input's relative
	 * order (the counting sort is stable); self-loops and parallel edges are retained.
	 * @pre `n <= none`, `n + 1` fits `size_t`, and stored edges fit `uint32`.
	 * Every endpoint is below `n` (asserted). The projection must preserve valid targets.
	 * Allocation failure propagates the standard container exception.
	 * @complexity O(n + m) time and storage for construction; `out_edges` is a contiguous span.
	 *
	 * @tparam Edge The stored adjacency element (`vertex_id`, `edge<W>`, or any
	 *           @ref atl::graph::adjacency_edge).
	 */
	template<adjacency_edge Edge>
	class csr {
		/** @brief `n + 1` entries; the edges of `u` live in `[offset_[u], offset_[u + 1])`. */
		std::vector<uint32> offset_;
		std::vector<Edge> edges_;

		/**
		 * @brief Counting-sort build shared by every constructor: one stored
		 *        edge per element (two when @p HasBothDirections), projected
		 *        through `proj(element, head)`, in stable input order per source.
		 * @pre `n <= 2^32 - 1` and the stored edge count fits `uint32` (both asserted).
		 */
		template<bool HasBothDirections, edge_list R, typename Proj>
			requires std::convertible_to<
				std::invoke_result_t<Proj const &, detail::list_reference_t<R>, vertex_id>, Edge>
		constexpr void build(std::size_t n, R const &arcs, Proj const &proj) {
			atl_assert(n <= (std::numeric_limits<vertex_id>::max)() && n < (std::numeric_limits<std::size_t>::max)());
			offset_.assign(n + 1, 0);
			[[maybe_unused]] std::size_t stored = 0;
			for (auto const &a : arcs) {
				atl_assert(graph::source(a) < n && graph::target(a) < n);
				atl_assert(stored <= (std::numeric_limits<uint32>::max)() - (HasBothDirections ? 2U : 1U));
				++offset_[graph::source(a) + 1];
				if constexpr (HasBothDirections) ++offset_[graph::target(a) + 1];
				stored += HasBothDirections ? 2 : 1;
			}
			atl_assert(stored <= (std::numeric_limits<uint32>::max)());
			std::partial_sum(offset_.begin(), offset_.end(), offset_.begin());
			edges_.resize(offset_.back());
			// offset_[u] doubles as u's write cursor, ending at u's slice end =
			// the start of u + 1; one shift right restores the starts (cheaper
			// than a separate cursor copy, measured in benchmarks/graph/csr.cpp)
			for (auto const &a : arcs) {
				vertex_id const s = graph::source(a), t = graph::target(a);
				edges_[offset_[s]++] = std::invoke(proj, a, t);
				if constexpr (HasBothDirections) edges_[offset_[t]++] = std::invoke(proj, a, s);
			}
			std::ranges::copy_backward(offset_.begin(), offset_.end() - 1, offset_.end());
			offset_[0] = 0;
		}

		// user-provided, not defaulted: MSVC rejects constant evaluation of a
		// private defaulted constructor called from a static member (C2248)
		constexpr csr(std::vector<uint32> offsets, std::vector<Edge> edges) noexcept
			: offset_(std::move(offsets)), edges_(std::move(edges)) {}

	public:
		/** @brief Stored adjacency element type. */
		using value_type = Edge;

		/**
		 * @brief Directed graph from an edge list: each element contributes one edge
		 * `source -> target`.
		 * @param n Vertex count. @param arcs Multipass list of valid endpoints.
		 * @complexity O(n + m)
		 */
		template<edge_list R>
			requires detail::adjacent_buildable<Edge, std::ranges::range_value_t<R>>
				  && detail::adjacent_buildable<Edge, std::remove_cvref_t<detail::list_reference_t<R>>>
		constexpr csr(std::size_t n, R const &arcs) {
			build<false>(n, arcs, []<list_edge A>(A const &a, vertex_id to)
							 requires detail::adjacent_buildable<Edge, A>
						 { return detail::adjacent_from<Edge>(a, to); });
		}

		/**
		 * @brief As above with an explicit projection `proj(list_edge) -> Edge` — for
		 * stored elements richer than target + weight.
		 * @param n Vertex count. @param arcs Multipass list of valid endpoints.
		 * @param proj Const-invocable mapping to a valid stored adjacency element.
		 * @complexity O(n + m) plus projection costs.
		 */
		template<edge_list R, typename Proj>
			requires std::default_initializable<Edge> && std::movable<Edge>
				  && std::convertible_to<std::invoke_result_t<Proj const &, detail::list_reference_t<R>>, Edge>
		constexpr csr(std::size_t n, R const &arcs, Proj proj) {
			build<false>(n, arcs, [&proj]<list_edge A>(A const &a, vertex_id)
							 requires std::is_invocable_r_v<Edge, Proj const &, A const &>
						 { return std::invoke(std::as_const(proj), a); });
		}

		/**
		 * @brief Undirected graph: every element contributes both directions, the
		 * reverse one built from `source(a)` and `weight(a)` (so any
		 * @ref atl::graph::list_edge works, `graph_traits`-adapted ones included).
		 * @complexity O(n + m)
		 */
		template<edge_list R>
			requires detail::adjacent_buildable<Edge, std::ranges::range_value_t<R>>
				  && detail::adjacent_buildable<Edge, std::remove_cvref_t<detail::list_reference_t<R>>>
		[[nodiscard]] static constexpr csr undirected(std::size_t n, R const &arcs) {
			csr g(std::vector<uint32>{}, std::vector<Edge>{});
			g.build<true>(n, arcs, []<list_edge A>(A const &a, vertex_id to)
							  requires detail::adjacent_buildable<Edge, A>
						  { return detail::adjacent_from<Edge>(a, to); });
			return g;
		}

		/**
		 * @brief Take ownership of an already built row table and edge array, for
		 *        callers that know every vertex's degree in advance and can place
		 *        edges directly: no edge list, no counting sort, no copy.
		 *
		 * @param offsets `n + 1` entries; the edges of `u` are `edges[offsets[u] .. offsets[u + 1])`.
		 * @param edges   All adjacency elements, grouped by source vertex.
		 * @pre `offsets` is non-empty, starts at 0, is non-decreasing and ends at
		 *      `edges.size()`; every edge targets a vertex below `n` (all asserted,
		 *      the O(n + m) scans only under `ATL_ASSERT`); the vertex count must fit `vertex_id`.
		 * @complexity O(1), O(n + m) under `ATL_ASSERT`
		 */
		[[nodiscard]] static constexpr csr adopt(std::vector<uint32> offsets, std::vector<Edge> edges) {
			atl_assert(!offsets.empty() && offsets.size() - 1 <= none);
			atl_assert(!offsets.empty() && offsets.front() == 0 && offsets.back() == edges.size());
			atl_assert(std::ranges::is_sorted(offsets));
			atl_assert(std::ranges::all_of(
				edges, [n = offsets.size() - 1](Edge const &e) { return std::size_t(graph::target(e)) < n; }));
			return csr(std::move(offsets), std::move(edges));
		}

		/** @brief Number of row slices. @complexity O(1) */
		[[nodiscard]] constexpr std::size_t vertex_count() const noexcept { return offset_.size() - 1; }

		/** @brief Number of stored directed adjacency elements. @complexity O(1) */
		[[nodiscard]] constexpr std::size_t edge_count() const noexcept { return edges_.size(); }

		/** @brief Stable input-order row. @pre `u < vertex_count()` (asserted).
		 * @return A span invalidated by destruction or assignment of its storage owner. @complexity O(1)
		 */
		[[nodiscard]] constexpr std::span<Edge const> out_edges(vertex_id u) const {
			atl_assert(u < vertex_count());
			return std::span<Edge const>(edges_).subspan(offset_[u], offset_[u + 1] - offset_[u]);
		}

		/** @brief Number of elements in row @p u. @pre `u < vertex_count()` (asserted). @complexity O(1) */
		[[nodiscard]] constexpr std::size_t out_degree(vertex_id u) const {
			atl_assert(u < vertex_count());
			return offset_[u + 1] - offset_[u];
		}
	};

	/** @brief Deduce the adjacency element from an edge list: `csr rows(n, arcs)`. */
	template<edge_list R>
	csr(std::size_t, R &&) -> csr<detail::adjacent_of_t<std::ranges::range_value_t<R>>>;

	/** @brief Deduced undirected build: `undirected_csr(n, arcs)`. */
	template<edge_list R>
		requires detail::adjacent_buildable<detail::adjacent_of_t<std::ranges::range_value_t<R>>, std::ranges::range_value_t<R>>
			  && detail::adjacent_buildable<detail::adjacent_of_t<std::ranges::range_value_t<R>>,
											std::remove_cvref_t<detail::list_reference_t<R>>>
	[[nodiscard]] constexpr auto undirected_csr(std::size_t n, R const &arcs) {
		return csr<detail::adjacent_of_t<std::ranges::range_value_t<R>>>::undirected(n, arcs);
	}
}
