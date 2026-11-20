#pragma once
#include <atl/graph/centroid_bisect.hpp>
#include <atl/graph/concepts.hpp>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>

/**
 * @file
 * @brief The single-span callback shape for mode-1 balanced-bisection
 *        centroid decomposition, racing the library two-span
 *        `centroid_bisect` behind the same engine and contract.
 *
 * It hands the caller one parent-first span split at n0. Callback-shape
 * comparison: not measured yet in the supplied records.
 */
namespace atl::experimental {
	/**
	 * @brief @ref atl::graph::centroid_bisect with a single-span callback:
	 *
	 *            f(center, component, n0) // hops
	 *            f(center, component, n0, edges) // weighted
	 *
	 * `component[0]` is the center (depth 0), `component[1 .. n0]` is group 0
	 * and `component[n0+1 ..]` is group 1 — one parent-first span with the parallel
	 * `edges` span (weighted form) letting the caller fold depth in a single
	 * sweep including the center. Same distance-≥2 contract and engine as
	 * @ref atl::graph::centroid_bisect; only the hand-off differs.
	 * @pre g is a forest (symmetric digraph); the spans are valid only for
	 *      the duration of the callback
	 * @complexity O(n log n) total span length
	 */
	template<graph::adjacency_graph Graph, typename Event>
		requires std::invocable<Event &, graph::vertex_id, std::span<graph::visit const>, std::size_t>
			  || (std::default_initializable<graph::edge_t<Graph>> && std::copyable<graph::edge_t<Graph>>
				  && std::constructible_from<graph::edge_t<Graph>, std::remove_reference_t<std::ranges::range_reference_t<graph::out_edges_t<Graph>>> const &>
				  && requires(graph::edge_t<Graph> &stored, std::remove_reference_t<std::ranges::range_reference_t<graph::out_edges_t<Graph>>> const &element) { stored = element; }
				  && std::invocable<Event &, graph::vertex_id, std::span<graph::visit const>, std::size_t,
									std::span<std::ranges::range_value_t<graph::out_edges_t<Graph>> const>>)
	constexpr void centroid_bisect_by_single_span(Graph const &g, Event f) {
		using Edge = std::ranges::range_value_t<graph::out_edges_t<Graph>>;
		using cpv = graph::visit;
		constexpr bool is_weighted = std::default_initializable<Edge> && std::copyable<Edge>
								  && std::constructible_from<Edge, std::remove_reference_t<std::ranges::range_reference_t<graph::out_edges_t<Graph>>> const &>
								  && requires(Edge &stored, std::remove_reference_t<std::ranges::range_reference_t<graph::out_edges_t<Graph>>> const &element) { stored = element; }
								  && std::invocable<Event &, graph::vertex_id, std::span<cpv const>,
													std::size_t, std::span<Edge const>>;
		graph::detail::centroid_bisect_engine<is_weighted>(
			g, [&](graph::vertex_id c, std::span<cpv const> comp, std::size_t n0,
				   [[maybe_unused]] std::span<Edge const> ec) {
				if constexpr (is_weighted)
					f(c, comp, n0, ec);
				else
					f(c, comp, n0);
			});
	}
}
