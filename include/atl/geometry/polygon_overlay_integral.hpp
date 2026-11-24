#pragma once

/** @file
 * @brief Polygon overlays for integral coordinates without adaptive floating kernels.
 */
#include <atl/geometry/arrangement_core.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/segment_core.hpp>
#include <atl/geometry/segment_intersections_core.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

namespace atl::geometry {
	/**
	 * @brief The OVERLAY of two simple polygons: the arrangement of both
	 *        boundaries plus, for every face, whether its open interior lies
	 *        inside each input. Every boolean operation reads off it — the
	 *        area overloads below sum faces, and the faces/half-edges of
	 *        `arr` give the full boundary structure of any combination
	 *        (disjoint result pieces and holes included).
	 *
	 * `segments` is the exact segment list `arr` was built from, so
	 * `point_location(o.arr, o.segments)` locates points in the overlay
	 * directly (the overlay must outlive that structure).
	 */
	template<coordinate T>
	struct polygon_overlay {
		arrangement arr;				  /**< of both boundaries; segment indices
											   `< first_edge_count` came from the
											   first polygon */
		std::vector<segment<T>> segments; /**< `arr`'s input: edge `i` of the first
											   polygon (`v[i] -> v[i + 1]`, cyclic) at
											   index `i`, then the second's edges
											   from `first_edge_count` on */
		uint32 first_edge_count;		  /**< the first polygon's edge count */
		std::vector<char> in_first;		  /**< per face: interior inside polygon 1 */
		std::vector<char> in_second;	  /**< per face: interior inside polygon 2 */
	};

	/**
	 * @brief Builds the @ref atl::geometry::polygon_overlay of two SIMPLE polygons, either
	 *        winding. Face membership is decided EXACTLY by Jordan parity
	 *        propagation over the face-adjacency graph: the unbounded face
	 *        is outside both, and crossing an arrangement edge toggles
	 *        membership in each input whose boundary supports that edge —
	 *        no point-in-polygon construction anywhere. Shared boundary
	 *        pieces (collinear overlaps between the inputs) are single
	 *        edges supporting both, so they toggle both memberships.
	 *
	 * @tparam R1 First vertex range.
	 * @tparam R2 Second vertex range, with the same coordinate type.
	 * @param first_polygon First cyclic boundary.
	 * @param second_polygon Second cyclic boundary.
	 * @return Owned arrangement, original segments and per-face memberships.
	 * @pre Both inputs are simple polygons (asserted through
	 *      @ref atl::geometry::simplicity_of under `ATL_ASSERT`).
	 * @pre Integral coordinates: `|x|, |y| < 2^32` (the arrangement's
	 *      bound); floating inputs best-effort.
	 * @complexity Expected O((n + k) log(n + k + 1)) time, O(n + k) storage, k = boundary crossings.
	 * @pre int32 accepts its full range; combined edge count below 2^32 (asserted).
	 * @note Not constexpr (runtime arrangement construction). Parity propagation is iterative.
	 */
	template<vertex_range R1, vertex_range R2>
		requires std::same_as<vertex_coordinate_t<R1>, vertex_coordinate_t<R2>>
	[[nodiscard]] polygon_overlay<vertex_coordinate_t<R1>> overlay_of(R1 &&first_polygon, R2 &&second_polygon) {
		using T = vertex_coordinate_t<R1>;
		atl_assert(simplicity_of(first_polygon).is_simple);
		atl_assert(simplicity_of(second_polygon).is_simple);
		auto const edges_of = []<vertex_range R>(R &poly, std::vector<segment<T>> &out) {
			auto const n = std::ranges::size(poly);
			auto const v = std::ranges::begin(poly);
			for (std::size_t i = 0; i < n; ++i)
				out.push_back({point<T>{v[i].x, v[i].y},
							   point<T>{v[(i + 1) % n].x, v[(i + 1) % n].y}});
		};
		std::vector<segment<T>> segments;
		auto const first_size = std::ranges::size(first_polygon);
		auto const second_size = std::ranges::size(second_polygon);
		atl_assert(uint64(first_size) < (uint64(1) << 32) && uint64(second_size) < (uint64(1) << 32));
		atl_assert(uint64(first_size) + uint64(second_size) < (uint64(1) << 32));
		segments.reserve(std::size_t(first_size) + std::size_t(second_size));
		edges_of(first_polygon, segments);
		uint32 const split = uint32(segments.size());
		edges_of(second_polygon, segments);

		auto arr_built = arrangement_of(segments);
		polygon_overlay<T> out{std::move(arr_built), std::move(segments), split, {}, {}};
		auto const &arr = out.arr;
		std::size_t const faces = arr.faces.size();
		out.in_first.assign(faces, 0);
		out.in_second.assign(faces, 0);

		// Jordan parity BFS from the unbounded face (outside both): each
		// crossed edge toggles the memberships its supports belong to
		std::vector<char> visited(faces, 0);
		std::vector<uint32> queue{arrangement::unbounded_face};
		visited[arrangement::unbounded_face] = 1;
		std::vector<std::vector<uint32>> halves_of(faces);
		for (uint32 h = 0; h < uint32(arr.next.size()); ++h)
			halves_of[arr.face[h]].push_back(h);
		for (std::size_t at = 0; at < queue.size(); ++at) {
			uint32 const f = queue[at];
			for (uint32 const h : halves_of[f]) {
				uint32 const g = arr.face[arrangement::twin(h)];
				bool should_flip_first = false, should_flip_second = false;
				for (uint32 const s : arr.supporting[arrangement::edge_of(h)])
					(s < split ? should_flip_first : should_flip_second) = true;
				char const want_first = char(out.in_first[f] ^ char(should_flip_first));
				char const want_second = char(out.in_second[f] ^ char(should_flip_second));
				if (visited[g]) { // parity must agree on every route
					atl_assert(out.in_first[g] == want_first
							   && out.in_second[g] == want_second);
					continue;
				}
				visited[g] = 1;
				out.in_first[g] = want_first;
				out.in_second[g] = want_second;
				queue.push_back(g);
			}
		}
		return out;
	}

	namespace detail {
		/**
		 * @brief A face selector for `overlay_area`: called as
		 *        `keep(is_in_first, is_in_second)`, true to count the face.
		 */
		template<typename Keep>
		concept overlay_keep_predicate = std::predicate<Keep &, bool, bool>;

		/**
		 * @brief Sum of the areas of the overlay's faces selected by
		 *        @p keep, called as `keep(is_in_first, is_in_second)` per
		 *        face — a rounded `float64` construction.
		 */
		template<coordinate T, overlay_keep_predicate Keep>
		[[nodiscard]] constexpr float64 overlay_area(polygon_overlay<T> const &overlay, Keep &&keep) {
			float64 total = 0;
			for (std::size_t f = 0; f < overlay.arr.faces.size(); ++f)
				if (keep(overlay.in_first[f] != 0, overlay.in_second[f] != 0))
					total += overlay.arr.faces[f].area;
			return total;
		}
	}

	/**
	 * @brief Area of the intersection of the two polygons of @p overlay — a
	 * rounded `float64` construction. Reading several areas off one
	 * overlay costs one sweep instead of one per area.
	 * @note The face sum is constexpr; overlay_of uses runtime arrangement construction.
	 * @complexity O(faces)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr float64 intersection_area(polygon_overlay<T> const &overlay) {
		return detail::overlay_area(overlay, [](bool const is_in_first, bool const is_in_second) {
			return is_in_first && is_in_second;
		});
	}

	/**
	 * @brief Area of the union of the two polygons of @p overlay.
	 * @note The face sum is constexpr; overlay_of uses runtime arrangement construction.
	 * @complexity O(faces)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr float64 union_area(polygon_overlay<T> const &overlay) {
		return detail::overlay_area(overlay, [](bool const is_in_first, bool const is_in_second) {
			return is_in_first || is_in_second;
		});
	}

	/**
	 * @brief Area of the difference `first minus second` of the polygons of @p overlay.
	 * @note The face sum is constexpr; overlay_of uses runtime arrangement construction.
	 * @complexity O(faces)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr float64 difference_area(polygon_overlay<T> const &overlay) {
		return detail::overlay_area(overlay, [](bool const is_in_first, bool const is_in_second) {
			return is_in_first && !is_in_second;
		});
	}

	/**
	 * @brief Area of the symmetric difference of the polygons of @p overlay.
	 * @note The face sum is constexpr; overlay_of uses runtime arrangement construction.
	 * @complexity O(faces)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr float64 symmetric_difference_area(polygon_overlay<T> const &overlay) {
		return detail::overlay_area(overlay, [](bool const is_in_first, bool const is_in_second) {
			return is_in_first != is_in_second;
		});
	}

	/**
	 * @brief Area of the intersection of two SIMPLE polygons (see @ref atl::geometry::overlay_of) — a rounded `float64` construction.
	 * @note Not constexpr (runtime arrangement construction).
	 */
	template<vertex_range R1, vertex_range R2>
		requires std::same_as<vertex_coordinate_t<R1>, vertex_coordinate_t<R2>>
	[[nodiscard]] float64 intersection_area(R1 &&a, R2 &&b) {
		return intersection_area(overlay_of(std::forward<R1>(a), std::forward<R2>(b)));
	}

	/** @brief Area of the union of two SIMPLE polygons.
	 * @pre As atl::geometry::overlay_of.
	 * @note Not constexpr (runtime arrangement construction). */
	template<vertex_range R1, vertex_range R2>
		requires std::same_as<vertex_coordinate_t<R1>, vertex_coordinate_t<R2>>
	[[nodiscard]] float64 union_area(R1 &&a, R2 &&b) {
		return union_area(overlay_of(std::forward<R1>(a), std::forward<R2>(b)));
	}

	/** @brief Area of the difference `a minus b` of two SIMPLE polygons.
	 * @pre As atl::geometry::overlay_of.
	 * @note Not constexpr (runtime arrangement construction). */
	template<vertex_range R1, vertex_range R2>
		requires std::same_as<vertex_coordinate_t<R1>, vertex_coordinate_t<R2>>
	[[nodiscard]] float64 difference_area(R1 &&a, R2 &&b) {
		return difference_area(overlay_of(std::forward<R1>(a), std::forward<R2>(b)));
	}

	/** @brief Area of the symmetric difference of two SIMPLE polygons.
	 * @pre As atl::geometry::overlay_of.
	 * @note Not constexpr (runtime arrangement construction). */
	template<vertex_range R1, vertex_range R2>
		requires std::same_as<vertex_coordinate_t<R1>, vertex_coordinate_t<R2>>
	[[nodiscard]] float64 symmetric_difference_area(R1 &&a, R2 &&b) {
		return symmetric_difference_area(overlay_of(std::forward<R1>(a), std::forward<R2>(b)));
	}
}
