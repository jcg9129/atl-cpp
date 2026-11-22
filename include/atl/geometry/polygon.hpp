#pragma once
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

namespace atl::geometry {
	/**
	 * @brief Where a point sits relative to a polygon; the values follow the sign
	 * convention (`int(c)` is usable like an orientation result).
	 */
	enum class containment : int { outside = -1, ///< Strictly outside.
								   boundary = 0, ///< On a vertex or an edge.
								   inside = 1 }; ///< Strictly inside.

	/**
	 * @brief Traversal direction of a polygon; the values follow the sign
	 * convention (`int(w)` matches an orientation sign).
	 */
	enum class winding : int { clockwise = -1,		   ///< Negative orientation, negative signed area.
							   none = 0,			   ///< Degenerate: no direction (or, for @ref atl::geometry::convexity, not convex).
							   counterclockwise = 1 }; ///< Positive orientation, positive signed area.

	/** @brief Result of @ref atl::geometry::convexity_of. */
	struct convexity {
		bool is_convex;	   ///< A convex SIMPLE polygon (collinear vertices allowed).
		bool is_strict;	   ///< Additionally no collinear vertices; false when not convex.
		winding direction; ///< The winding; `winding::none` when not convex.

		/** @brief Memberwise equality. */
		friend constexpr bool operator==(convexity const &, convexity const &) = default;
	};

	/**
	 * @note For general simple-polygon validation use atl::geometry::simplicity_of
	 * from `atl/geometry/segment_intersections.hpp`.
	 * @brief Classifies the vertex sequence @p poly: is it a convex simple
	 *        polygon, does it have collinear (straight) vertices, and which
	 *        way does it wind. Every decision is a sign of the policy or an
	 *        exact comparison, so the answer is exact for integral and
	 *        floating coordinates under the default policy.
	 *
	 * Rejected (`{false, false, winding::none}`): fewer than 3 vertices,
	 * consecutive equal vertices (compared with `==`, whatever the policy),
	 * mixed turn signs, collinear reversals (an edge doubling back, which
	 * also rejects every zero-area, all-collinear sequence), and sequences
	 * that wind around more than once. The last are the gap of a bare
	 * same-turn-sign test (a pentagram turns left at every vertex); they are
	 * caught by counting how often the edge directions wrap past angle 0,
	 * exactly once for a convex simple polygon.
	 * @pre `size(poly) < 2^32` (asserted); for `int64` coordinates
	 *      `|x|, |y| < 2^62` (asserted).
	 * @complexity O(n)
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr convexity convexity_of(R &&poly, Policy const p = {}) {
		using T = vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(poly) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(poly));
		constexpr convexity rejected{false, false, winding::none};
		if (n < 3) return rejected;
		auto const first = std::ranges::begin(poly);
		auto const next = [&](uint32 const i) { return i == n - 1 ? uint32(0) : i + 1; };
		uint32 lefts = 0, rights = 0, straights = 0;
		for (uint32 i = 0; i < n; ++i) {
			point<T> const a = first[i], b = first[next(i)], c = first[next(next(i))];
			if (a == b) return rejected;
			int const turn = p.orientation(a, b, c);
			if (turn > 0)
				++lefts;
			else if (turn < 0)
				++rights;
			else {
				// collinear: opposed edge directions are a reversal, which no
				// simple polygon has; equal directions make a straight vertex.
				// Component signs decide it exactly, doubles included
				auto const sgn_of = [](auto const lo, auto const hi) {
					return int(hi > lo) - int(hi < lo);
				};
				if (sgn_of(a.x, b.x) != sgn_of(b.x, c.x) || sgn_of(a.y, b.y) != sgn_of(b.y, c.y))
					return rejected;
				++straights;
			}
		}
		if (lefts != 0 && rights != 0) return rejected;
		bool const is_counterclockwise = lefts != 0; // all straight was rejected as a reversal
		// k wraps of the edge directions past angle 0 is winding number k;
		// a doubly wound sequence passes every test above
		uint32 wraps = 0;
		for (uint32 i = 0; i < n; ++i) {
			point<T> const a = first[i], b = first[next(i)], c = first[next(next(i))];
			wraps += is_counterclockwise ? detail::sweep_before(b, c, a, b, p) : detail::sweep_before(a, b, b, c, p);
		}
		if (wraps != 1) return rejected;
		return {true, straights == 0, is_counterclockwise ? winding::counterclockwise : winding::clockwise};
	}

	/**
	 * @brief Twice the SIGNED area of the polygon @p poly (the shoelace
	 *        formula): positive when the vertices run counterclockwise, 0
	 *        for fewer than 3 vertices.
	 *
	 * Exact for integral coordinates whenever the result fits
	 * `difference_product_t`, which every simple polygon within the
	 * coordinate bounds does (for `int64` the doubled area of a box with
	 * sides below 2^63 is below 2^127). The terms are the triangles of a fan
	 * from the first vertex, each exact, accumulated modulo the type's width,
	 * so partial sums that exceed the result (the prefixes of a spiral wind
	 * several times) cannot overflow. For floating coordinates the value is a
	 * rounded construction.
	 *
	 * Defined on ANY closed vertex sequence: by Green's theorem the value is
	 * the winding-number weighted signed area, each region counted with how
	 * often the traversal winds around it (a symmetric bowtie gives 0, a
	 * doubly wound triangle twice its area, a hole traversed through a cut
	 * is subtracted).
	 * @pre For `int64` coordinates `|x|, |y| < 2^62` (asserted).
	 * @complexity O(n)
	 */
	template<vertex_range R>
	[[nodiscard]] constexpr difference_product_t<vertex_coordinate_t<R>> polygon_area2(R &&poly) {
		using T = vertex_coordinate_t<R>;
		using W = difference_product_t<T>;
		auto const n = std::ranges::size(poly);
		if (n < 3) return W{};
		auto const first = std::ranges::begin(poly);
		point<T> const o = first[0];
		if constexpr (std::floating_point<T>) {
			W sum = 0;
			for (std::size_t i = 1; i + 1 < n; ++i) sum += cross(o, point<T>(first[i]), point<T>(first[i + 1]));
			return sum;
		} else {
			// unsigned: wraparound is defined, and the final value is exact
			using U = traits::make_unsigned_t<W>;
			U sum = 0;
			for (std::size_t i = 1; i + 1 < n; ++i) sum += U(cross(o, point<T>(first[i]), point<T>(first[i + 1])));
			return W(sum);
		}
	}

	/**
	 * @brief Result of @ref atl::geometry::lattice_count_of.
	 * @tparam Count The signed integer type of the counts.
	 */
	template<signed_integral Count>
	struct lattice_count {
		Count boundary; ///< Lattice points on the polygon's boundary.
		Count interior; ///< Lattice points strictly inside.

		/** @brief Memberwise equality. */
		friend constexpr bool operator==(lattice_count const &, lattice_count const &) = default;
	};

	/**
	 * @brief Lattice-point counts of a SIMPLE polygon with integer vertices,
	 *        either winding: the boundary count is the sum over the edges of
	 *        `gcd(|dx|, |dy|)`, the interior count follows from Pick's
	 *        theorem `A = I + B/2 - 1` with the exact doubled area. Exact.
	 * @pre At least 3 vertices and nonzero area (both asserted); the polygon
	 *      is simple (Pick's theorem needs it; not checked).
	 * @pre For `int64` coordinates `|x|, |y| < 2^62` (asserted by the area
	 *      sum), so every edge difference fits `int64`.
	 * @complexity O(n log C) for coordinate magnitude C
	 */
	template<vertex_range R>
		requires std::signed_integral<vertex_coordinate_t<R>>
	[[nodiscard]] constexpr lattice_count<difference_product_t<vertex_coordinate_t<R>>> lattice_count_of(R &&poly) {
		using W = difference_product_t<vertex_coordinate_t<R>>;
		auto const n = std::ranges::size(poly);
		atl_assert(n >= 3);
		// first: its bound assert guards the int64 differences below
		W const area2 = polygon_area2(poly);
		atl_assert(area2 != 0);
		auto const first = std::ranges::begin(poly);
		W boundary = 0;
		point<vertex_coordinate_t<R>> p = first[n - 1];
		for (std::size_t i = 0; i < n; ++i) {
			point<vertex_coordinate_t<R>> const q = first[i];
			int64 const dx = int64(q.x) - int64(p.x), dy = int64(q.y) - int64(p.y);
			boundary += W(std::gcd(dx, dy)); // the edge's lattice points, one endpoint counted
			p = q;
		}
		W const doubled = area2 < 0 ? W(-area2) : area2;
		return {boundary, W((doubled - boundary + 2) / 2)};
	}

	/**
	 * @brief The area centroid (the center of mass of a uniform lamina) of a
	 *        SIMPLE polygon, either winding, as a rounded `float64`
	 *        construction. Not the vertex average: a dense run of vertices
	 *        does not move it.
	 *
	 * The triangles of a fan from the first vertex contribute their exact
	 * doubled areas (rounded once to `double`) and their centroids from
	 * coordinate differences to that vertex, so a polygon far from the
	 * origin keeps its precision.
	 * @see atl::geometry::triangulation_of in `atl/geometry/triangulation.hpp` for
	 * an independent triangle decomposition and area-weighted centroid.
	 * @pre At least 3 vertices (asserted) and nonzero area (asserted on the
	 *      rounded doubled area, which the division needs).
	 * @pre For `int64` coordinates `|x|, |y| < 2^62` (asserted).
	 * @pre Floating squares and pairwise products of coordinates and offsets,
	 *      and the area-weighted moments formed here, are finite and normal
	 *      unless a factor is zero. Moment products are asserted.
	 * @complexity O(n)
	 */
	template<vertex_range R>
	[[nodiscard]] constexpr point<float64> polygon_centroid(R &&poly) {
		auto const n = std::ranges::size(poly);
		atl_assert(n >= 3);
		auto const first = std::ranges::begin(poly);
		using T = vertex_coordinate_t<R>;
		point<T> const o = first[0];
		for (std::size_t i = 0; i < n; ++i) detail::check_floating_magnitude(point<T>(first[i]));
		float64 cx = 0, cy = 0, doubled = 0;
		for (std::size_t i = 1; i + 1 < n; ++i) {
			// triangle (o, p_i, p_i+1): doubled area cr, centroid o + (u + v) / 3
			float64 const cr = float64(cross(o, point<T>(first[i]), point<T>(first[i + 1])));
			auto const u = detail::float64_difference(o, point<T>(first[i]));
			auto const v = detail::float64_difference(o, point<T>(first[i + 1]));
			atl_assert(detail::normal_product(cr, u.x + v.x));
			atl_assert(detail::normal_product(cr, u.y + v.y));
			cx += cr * (u.x + v.x);
			cy += cr * (u.y + v.y);
			doubled += cr;
		}
		atl_assert(doubled != 0);
		return {float64(o.x) + cx / (3 * doubled), float64(o.y) + cy / (3 * doubled)};
	}

	/**
	 * @brief The winding direction of a SIMPLE polygon, from one orientation
	 *        at its lowest-then-rightmost vertex, a strictly convex corner,
	 *        instead of an area sign: exact for floating coordinates too
	 *        (the shoelace sign of `double` vertices is a rounded
	 *        construction), with no widened accumulation.
	 *
	 * Copies of the corner (consecutive vertices `policy.equal` to it) are
	 * skipped, so the nearest distinct neighbors decide. `winding::none`
	 * means a degenerate sequence: fewer than 3 vertices, or a corner whose
	 * neighbors are collinear with it (zero area).
	 * @pre The polygon is simple (not checked: a self-intersecting sequence
	 *      gets its corner's verdict, which need not match any area
	 *      convention); for `int64` coordinates `|x|, |y| < 2^62` (asserted).
	 * @complexity O(n)
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr winding winding_of(R &&poly, Policy const p = {}) {
		auto const n = std::ranges::size(poly);
		if (n < 3) return winding::none;
		auto const first = std::ranges::begin(poly);
		std::size_t low = 0;
		for (std::size_t i = 1; i < n; ++i) {
			point<vertex_coordinate_t<R>> const c = first[i], b = first[low];
			if (c.y != b.y ? c.y < b.y : c.x > b.x) low = i;
		}
		std::size_t predecessor = low == 0 ? n - 1 : low - 1;
		std::size_t successor = low + 1 == n ? 0 : low + 1;
		for (std::size_t step = 1; step < n && p.equal(point<vertex_coordinate_t<R>>(first[predecessor]), point<vertex_coordinate_t<R>>(first[low])); ++step)
			predecessor = predecessor == 0 ? n - 1 : predecessor - 1;
		for (std::size_t step = 1; step < n && p.equal(point<vertex_coordinate_t<R>>(first[successor]), point<vertex_coordinate_t<R>>(first[low])); ++step)
			successor = successor + 1 == n ? 0 : successor + 1;
		return winding(p.orientation(point<vertex_coordinate_t<R>>(first[predecessor]), point<vertex_coordinate_t<R>>(first[low]), point<vertex_coordinate_t<R>>(first[successor])));
	}

	/**
	 * @brief The SIMPLE polygon @p poly with consecutive duplicates and
	 *        straight vertices removed (a vertex collinear with its cyclic
	 *        neighbors, whole collinear runs included), oriented
	 *        COUNTERCLOCKWISE. The first surviving input vertex stays first;
	 *        the boundary as a point set, and so the area, is unchanged; the
	 *        operation is idempotent.
	 *
	 * Duplicates (consecutive vertices `policy.equal` to each other, the
	 * cyclic wrap included) go first: a duplicated corner makes the neighbor
	 * triples of both copies degenerate, so testing collinearity directly
	 * would delete the corner. No cascade is needed afterwards: in a simple
	 * polygon a straight vertex lies strictly between its neighbors (a spike
	 * would overlap itself), so removing one never straightens a survivor.
	 *
	 * Degenerate inputs, with fewer than 3 distinct consecutive vertices
	 * after deduplication (an empty input, one point, a back-and-forth
	 * `{p, q, q, p}`), have nothing to straighten and no orientation: the
	 * deduplicated sequence is returned as it is.
	 * @pre The polygon is simple (not checked); nonzero area once 3 distinct
	 *      vertices remain (asserted); for `int64` coordinates
	 *      `|x|, |y| < 2^62` (asserted).
	 * @complexity O(n) time and memory
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr std::vector<point<vertex_coordinate_t<R>>> normalized_polygon(R &&poly, Policy const p = {}) {
		using T = vertex_coordinate_t<R>;
		auto const n = std::ranges::size(poly);
		auto const first = std::ranges::begin(poly);
		std::vector<point<T>> distinct;
		distinct.reserve(n);
		for (std::size_t i = 0; i < n; ++i)
			if (distinct.empty() || !p.equal(point<T>(first[i]), distinct.back())) distinct.push_back(first[i]);
		while (distinct.size() > 1 && p.equal(distinct.front(), distinct.back())) distinct.pop_back();
		std::size_t const m = distinct.size();
		if (m <= 2) return distinct;
		std::vector<point<T>> out;
		out.reserve(m);
		point<T> predecessor = distinct[m - 1];
		for (std::size_t i = 0; i < m; ++i) {
			point<T> const successor = distinct[i + 1 == m ? 0 : i + 1];
			if (p.orientation(predecessor, distinct[i], successor) != 0) out.push_back(distinct[i]);
			predecessor = distinct[i];
		}
		atl_assert(out.size() >= 3); // fewer survivors: zero area
		if (winding_of(out, p) == winding::clockwise)
			std::ranges::reverse(out.begin() + 1, out.end()); // keeps out[0] first
		return out;
	}

	/**
	 * @brief The absolute area of @p poly as `double`, a construction: half
	 *        the magnitude of @ref atl::geometry::polygon_area2.
	 * @pre As @ref atl::geometry::polygon_area2.
	 * @complexity O(n)
	 */
	template<vertex_range R>
	[[nodiscard]] constexpr float64 polygon_area(R &&poly) {
		float64 const doubled = float64(polygon_area2(poly));
		return (doubled < 0 ? -doubled : doubled) * 0.5;
	}

	/**
	 * @brief Locates @p q relative to the SIMPLE polygon @p poly (vertices in
	 *        order, either orientation) by the crossing parity of a
	 *        rightward ray, with the boundary detected first: a query on a
	 *        vertex, inside an edge or on a collinear run of edges is
	 *        `boundary`, never miscounted. Signs come from the policy, so
	 *        under the default the answer is exact for integral and floating
	 *        coordinates.
	 *
	 * Fewer than 3 vertices are accepted: an empty range has every point
	 * outside, one or two vertices report `boundary` on the point or the
	 * segment they span and `outside` elsewhere.
	 * @pre The polygon is simple (not checked); for `int64` coordinates
	 *      `|x|, |y| < 2^62` (asserted).
	 * @note For a strict CCW hull from atl::geometry::convex_hull, use
	 * atl::geometry::convex_contains for O(log n) queries.
	 * @complexity O(n)
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr containment contains(R &&poly, point<vertex_coordinate_t<R>> const q, Policy const p = {}) {
		using T = vertex_coordinate_t<R>;
		auto const n = std::ranges::size(poly);
		if (n == 0) return containment::outside;
		auto const first = std::ranges::begin(poly);
		bool is_inside = false;
		for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
			point<T> const u = first[j], v = first[i];
			if (contains(segment<T>{u, v}, q, p)) return containment::boundary;
			// half-open straddle: horizontal edges never count and a shared
			// vertex counts once; the orientation is nonzero here, since a
			// zero with a straddle is a boundary point
			if ((u.y > q.y) != (v.y > q.y)) {
				int const turn = p.orientation(u, v, q);
				is_inside ^= v.y > u.y ? turn > 0 : turn < 0;
			}
		}
		return is_inside ? containment::inside : containment::outside;
	}

	/**
	 * @brief Point-in-polygon query with losslessly widened integer query coordinates.
	 * @pre As the polygon-coordinate overload of `contains`.
	 * @complexity O(n)
	 */
	template<vertex_range R, coordinate Query, sign_policy<vertex_coordinate_t<R>> Policy = exact>
		requires std::signed_integral<Query> && std::signed_integral<vertex_coordinate_t<R>>
			  && (!std::same_as<Query, vertex_coordinate_t<R>>)
			  && (sizeof(Query) <= sizeof(vertex_coordinate_t<R>))
	[[nodiscard]] constexpr containment contains(R &&poly, point<Query> const q, Policy const p = {}) {
		using T = vertex_coordinate_t<R>;
		return contains(std::forward<R>(poly), point<T>{T(q.x), T(q.y)}, p);
	}

}
