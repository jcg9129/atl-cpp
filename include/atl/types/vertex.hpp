#pragma once
#include <atl/types/alias.hpp>
#include <concepts>
#include <ranges>

/**
 * @file
 * @brief The vertex vocabulary shared by graph algorithms and the dynamic-forest data structures: the vertex id type,
 *        the "no vertex" sentinel and ranges of vertex ids. The full graph concepts live in graph/concepts.hpp.
 */
namespace atl::graph {
	/**
	 * @brief Vertices are dense integral ids in `[0, vertex_count(g))` — a deliberate
	 * simplification over descriptor-based designs: no handle abstraction, no
	 * map-backed vertex storage, ids index straight into arrays.
	 */
	using vertex_id = uint32;

	/** @brief Sentinel for "no vertex / no edge / unreached" in graph results. */
	inline constexpr uint32 none = uint32(-1);

	/**
	 * @brief A range of vertex ids — parent arrays, vertex subsets, orderings.
	 *
	 * Any element type convertible to @ref atl::graph::vertex_id qualifies, so a range of
	 * `int` or `int64` is accepted and converted: its values must already be
	 * valid ids (or @ref atl::graph::none), as the conversion itself narrows silently.
	 */
	template<typename R>
	concept vertex_range = std::ranges::input_range<R>
						&& std::convertible_to<std::ranges::range_value_t<R>, vertex_id>;
}
