#pragma once
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numbers>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::geometry {
	/**
	 * @brief A CLOSED arc of directions: from angle `from` counterclockwise
	 *        to angle `to`, in radians. Any finite radians are accepted; the
	 *        span is the counterclockwise distance from `from` to `to`
	 *        reduced into `[0, 2 pi)`, so `from == to` is a single direction
	 *        and a full circle is NOT representable (count elements that
	 *        cover every direction outside the sweep instead).
	 */
	struct arc {
		float64 from{}; ///< Start angle in radians.
		float64 to{};	///< End angle in radians, reached counterclockwise from `from`.
	};

	/** @brief A sized random-access range of @ref atl::geometry::arc. */
	template<typename R>
	concept arc_range = std::ranges::random_access_range<R> && std::ranges::sized_range<R>
					 && std::same_as<std::remove_cvref_t<std::ranges::range_value_t<R>>, arc>;

	/** @brief Result of @ref atl::geometry::max_arc_overlap. */
	struct arc_overlap {
		uint32 count{};	 ///< The most arcs that contain one direction.
		float64 angle{}; ///< A direction they all contain, in `[0, 2 pi)`; 0 when there are no arcs.
	};

	namespace detail {
		/**
		 * @brief @p a reduced modulo `2 pi` into `[0, 2 pi)`. A small negative
		 *        @p a rounds up to `2 pi` when shifted: as an angle that is 0,
		 *        as a span (@p is_span) the largest value below `2 pi`, a
		 *        nearly full arc.
		 */
		inline float64 reduced_angle(float64 const a, bool const is_span) {
			constexpr float64 turn = 2 * std::numbers::pi;
			float64 const m = std::fmod(a, turn);
			float64 const reduced = m < 0 ? m + turn : m;
			if (reduced < turn) return reduced;
			return is_span ? std::nextafter(turn, 0.0) : 0.0;
		}

		/** @brief Sweep reduced closed endpoints with the count already open across zero. */
		inline arc_overlap max_arc_overlap_events(std::vector<float64> &entries, std::vector<float64> &exits, uint32 open) {
			std::ranges::sort(entries);
			std::ranges::sort(exits);
			arc_overlap best{0, 0.0};
			std::size_t closed = 0;
			for (float64 const from : entries) {
				// exits strictly ahead of this entry; ties keep the arc open
				for (; closed < exits.size() && exits[closed] < from; ++closed) --open;
				++open;
				if (open > best.count) best = {open, from};
			}
			return best;
		}

		/**
		 * @brief The sweep of @ref atl::geometry::max_arc_overlap over arcs given as entry
		 *        angles in @p entries and exit angles `from + span` in
		 *        @p exits, index-aligned, on caller-owned scratch: both are
		 *        sorted as plain arrays and merged, entries first on ties.
		 *
		 * An exit at or past `2 pi` wraps: the arc is open at angle 0 and
		 * closes at the wrapped exit. Rounding can carry a wrapped exit to
		 * its own entry or beyond, which would count the arc twice there; it
		 * is clamped just below the entry instead.
		 * @pre `entries.size() == exits.size() < 2^32`; each entry in
		 *      `[0, 2 pi)` and each span in `[0, 2 pi)`
		 * @complexity O(n log n)
		 */
		inline arc_overlap max_arc_overlap_sorted(std::vector<float64> &entries, std::vector<float64> &exits) {
			constexpr float64 turn = 2 * std::numbers::pi;
			if (entries.empty()) return {0, 0.0};
			uint32 open = 0; // arcs open at angle 0: those wrapping past 2 pi
			for (std::size_t i = 0; i < exits.size(); ++i)
				if (exits[i] >= turn) {
					exits[i] -= turn;
					if (exits[i] >= entries[i]) exits[i] = std::nextafter(entries[i], 0.0);
					++open;
				}
			return max_arc_overlap_events(entries, exits, open);
		}
	}

	/**
	 * @brief The maximum number of arcs that contain one common direction,
	 *        with such a direction: the RADIAL SWEEP. Entry and exit events
	 *        are sorted by angle, entries first on ties, so arcs that only
	 *        touch at an endpoint do overlap (they are closed). The maximum
	 *        is attained at the `from` endpoint of some arc, which is the
	 *        witness returned.
	 *
	 * Angles are rounded `float64` values and compare as such, like every
	 * angular quantity in the library: an arc ending within rounding of
	 * another's start may or may not meet it. An empty range gives
	 * `{0, 0.0}`.
	 * @pre `size(arcs) < 2^32` (asserted: the counts are `uint32`); finite
	 *      angles.
	 * @note Not constexpr (`std::fmod`, and sorts of heap buffers).
	 * @complexity O(n log n) time, O(n) memory
	 */
	template<arc_range R>
	[[nodiscard]] arc_overlap max_arc_overlap(R &&arcs) {
		atl_assert(std::ranges::size(arcs) < (uint64(1) << 32));
		std::vector<float64> entries, exits;
		entries.reserve(std::ranges::size(arcs));
		exits.reserve(std::ranges::size(arcs));
		uint32 open = 0;
		constexpr float64 turn = 2 * std::numbers::pi;
		for (arc const &a : std::forward<R>(arcs)) {
			atl_assert(std::isfinite(a.from) && std::isfinite(a.to));
			float64 const from = detail::reduced_angle(a.from, false);
			entries.push_back(from);
			// Reduce signed endpoints before subtracting: huge angles must not
			// absorb the other endpoint or overflow their difference.
			float64 const span = detail::reduced_angle(std::fmod(a.to, turn) - std::fmod(a.from, turn), true);
			float64 to = detail::reduced_angle(a.to, false);
			if (from > to) {
				++open;
			} else if (from == to && span > 0) {
				// At zero no reduced exit can precede the entry. Represent
				// the nearly full turn by the last direction below 2 pi.
				if (from == 0)
					to = std::nextafter(turn, 0.0);
				else {
					++open;
					to = std::nextafter(from, 0.0);
				}
			}
			exits.push_back(to);
		}
		return detail::max_arc_overlap_events(entries, exits, open);
	}

	/** @brief Result of @ref atl::geometry::max_points_in_circle. */
	struct circle_cover {
		uint32 count{};			 ///< The most points one closed disk of the radius covers.
		point<float64> center{}; ///< The center of such a disk, a rounded construction.
	};

	/**
	 * @brief The maximum number of points a circle of the given FIXED radius
	 *        can cover (a closed disk: the boundary counts), with a witness
	 *        center. Some optimal disk has an input point on its boundary, so
	 *        for each point the candidate centers form the circle of radius
	 *        @p radius around it, every other point within `2 radius` cuts a
	 *        coverage arc into that circle, and @ref atl::geometry::max_arc_overlap does the
	 *        rest.
	 *
	 * Coverage tests are `float64` constructions (distances and angles are
	 * irrational): the count is right except for points within rounding of
	 * the boundary of an optimal disk, where it is best-effort; there is no
	 * exact tier for this quantity. Points at equal positions all count.
	 * `radius == 0` gives the size of the largest group of equal points.
	 * Coordinate differences are taken exactly in the widened type before
	 * rounding, so there is no coordinate bound: a cloud far from the origin
	 * gets the same arcs as near it.
	 * @pre `1 <= size(points) < 2^32` and `radius >= 0` (asserted); finite coordinates and radius.
	 *      For positive radius, all formed coordinate/offset and radius squares
	 *      and pairwise products must be finite and normal unless a factor is
	 *      zero (squares asserted); their sums must also be finite.
	 * @note Not constexpr (`std::atan2`, `std::acos`, `std::sqrt`).
	 * @complexity O(n log n + n k log(k + 1)) for at most k points within `2 radius` of
	 *             each point: a grid of cell side a little over `2 radius`
	 *             finds the neighbors. When the bounding box spans `2^30`
	 *             cells or more on a side, every pair is scanned instead,
	 *             O(n^2 + n k log(k + 1)). Radius zero uses sorting and run
	 *             lengths in O(n log n). O(n) memory.
	 */
	template<vertex_range R>
	[[nodiscard]] circle_cover max_points_in_circle(R &&points, float64 const radius) {
		using T = vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(points) >= 1 && radius >= 0);
		atl_assert(std::ranges::size(points) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(points));
		auto const first = std::ranges::begin(points);
		atl_assert(std::isfinite(radius));
		if (radius == 0) {
			std::vector<point<T>> sorted;
			sorted.reserve(n);
			for (uint32 i = 0; i < n; ++i) {
				point<T> const q = first[i];
				atl_assert(std::isfinite(float64(q.x)) && std::isfinite(float64(q.y)));
				sorted.push_back(q);
			}
			std::ranges::sort(sorted);
			circle_cover answer{1, to_float64(sorted[0])};
			uint32 run = 1;
			for (uint32 i = 1; i < n; ++i) {
				run = sorted[i] == sorted[i - 1] ? run + 1 : 1;
				if (run > answer.count) answer = {run, to_float64(sorted[i])};
			}
			return answer;
		}
		atl_assert(detail::normal_product(radius, radius));
		atl_assert(detail::normal_product(2 * radius, 2 * radius));
		for (uint32 i = 0; i < n; ++i) detail::check_floating_magnitude(point<T>(first[i]));
		constexpr float64 turn = 2 * std::numbers::pi;
		float64 const reach2 = 4 * radius * radius;
		circle_cover best{0, {}};
		std::vector<float64> entries, exits;
		uint32 coincident = 1; // the anchor and its copies
		auto const visit = [&](uint32 const i, uint32 const j) {
			if (first[i] == first[j]) {
				++coincident;
				return;
			}
			auto const towards = detail::float64_difference(point<T>(first[i]), point<T>(first[j]));
			atl_assert(detail::normal_product(towards.x, towards.x));
			atl_assert(detail::normal_product(towards.y, towards.y));
			float64 const d2 = towards.x * towards.x + towards.y * towards.y;
			if (d2 > reach2) return;
			float64 const theta = std::atan2(towards.y, towards.x);
			float64 const alpha = std::acos((std::min)(std::sqrt(d2) / (2 * radius), 1.0));
			// theta in [-pi, pi] and alpha in [0, pi/2]: one wrap reduces it,
			// and a sum that rounds up to 2 pi is angle 0
			float64 const lowest = theta - alpha < 0 ? theta - alpha + turn : theta - alpha;
			float64 const from = lowest < turn ? lowest : 0.0;
			entries.push_back(from);
			exits.push_back(from + 2 * alpha);
		};
		auto const finish_anchor = [&](uint32 const i) {
			if (coincident + entries.size() <= best.count) return; // cannot beat the incumbent
			auto const overlap = detail::max_arc_overlap_sorted(entries, exits);
			if (coincident + overlap.count <= best.count) return;
			point<float64> center = to_float64(first[i]);
			if (overlap.count > 0) center += {radius * std::cos(overlap.angle), radius * std::sin(overlap.angle)};
			best = {coincident + overlap.count, center};
		};
		auto const start_anchor = [&] {
			entries.clear();
			exits.clear();
			coincident = 1;
		};
		// neighbors within 2r come from the 3x3 block of a grid of cell side
		// a little over 2r, cells taken from exact differences to the lower
		// left corner. Cell quotients stay below 2^30, so their rounding is
		// far below the side's 2^-16 slack and a true neighbor is never more
		// than one cell away
		point<T> low = first[0];
		for (uint32 i = 1; i < n; ++i) low = {(std::min)(low.x, first[i].x), (std::min)(low.y, first[i].y)};
		float64 const side = 2 * radius * (1 + 0x1p-16);
		float64 span = 0;
		for (uint32 i = 0; i < n; ++i) {
			auto const offset = detail::float64_difference(low, first[i]);
			span = (std::max)({span, offset.x, offset.y});
		}
		if (radius > 0 && span / side < 0x1p30) {
			struct cell_entry {
				int64 cx, cy;
				uint32 index;
			};

			auto const cell_less = [](cell_entry const &l, cell_entry const &r) {
				return l.cx != r.cx ? l.cx < r.cx : l.cy < r.cy;
			};
			std::vector<cell_entry> cells(n);
			for (uint32 i = 0; i < n; ++i) {
				auto const offset = detail::float64_difference(low, first[i]);
				cells[i] = {int64(offset.x / side), int64(offset.y / side), i};
			}
			std::vector<cell_entry> sorted = cells;
			std::ranges::sort(sorted, cell_less);
			for (uint32 i = 0; i < n; ++i) {
				start_anchor();
				for (int64 row = cells[i].cx - 1; row <= cells[i].cx + 1; ++row) {
					auto it = std::ranges::lower_bound(sorted, cell_entry{row, cells[i].cy - 1, 0}, cell_less);
					for (; it != sorted.end() && it->cx == row && it->cy <= cells[i].cy + 1; ++it)
						if (it->index != i) visit(i, it->index);
				}
				finish_anchor(i);
			}
		} else {
			for (uint32 i = 0; i < n; ++i) {
				start_anchor();
				for (uint32 j = 0; j < n; ++j)
					if (j != i) visit(i, j);
				finish_anchor(i);
			}
		}
		return best;
	}
}
