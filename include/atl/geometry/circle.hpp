#pragma once

#include <atl/ds/static_vector.hpp>
#include <atl/geometry/circle_core.hpp>
#include <atl/geometry/line.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/predicates_floating.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>

namespace atl::geometry {
	/**
	 * @brief The minimum enclosing circle of a non-empty point set —
	 *        Randomized incremental construction, expected O(n) with a
	 *        deterministic seeded shuffle (`atl::set_hash_seed` pins it).
	 *        The result is a `double` construction; containment in the
	 *        nested loops uses squared-distance slack `1e-12 * radius^2 + 1e-30`.
	 *        This is best effort: there is no uniform relative error bound for
	 *        ill-conditioned inputs or scales near the absolute slack. No recursion.
	 * @pre Floating intermediates must remain finite and nonzero products normal.
	 *      Integer inputs are converted to `double` first; differences may lose bits.
	 * @pre `size(points) >= 1` (asserted).
	 * @complexity Expected O(n) time with random order, O(n^3) worst case, O(n) heap storage.
	 *
	 * @note Not constexpr (uses runtime math or seeded random state).
	 * @ingroup circle_geometry
	 */
	template<vertex_range R>
	[[nodiscard]] circle<float64> minimum_enclosing_circle(R &&points) {
		atl_assert(std::ranges::size(points) >= 1);
		auto const first = std::ranges::begin(points);
		std::vector<point<float64>> pts;
		pts.reserve(std::ranges::size(points));
		for (auto const &q : std::ranges::subrange(first, std::ranges::end(points)))
			pts.push_back(to_float64(q));
		std::mt19937_64 rng(atl::detail::hash_seed() ^ 0x0c1cc1e5eedULL);
		std::ranges::shuffle(pts, rng);

		auto const covers = [](circle<float64> const &c, point<float64> const q) {
			float64 const dx = q.x - c.center.x, dy = q.y - c.center.y;
			return dx * dx + dy * dy <= c.radius * c.radius * (1 + 1e-12) + 1e-30;
		};
		circle<float64> best{pts[0], 0};
		for (std::size_t i = 1; i < pts.size(); ++i) {
			if (covers(best, pts[i])) continue;
			best = {pts[i], 0};
			for (std::size_t j = 0; j < i; ++j) {
				if (covers(best, pts[j])) continue;
				best = diametral_circle(pts[i], pts[j]);
				for (std::size_t k = 0; k < j; ++k) {
					if (covers(best, pts[k])) continue;
					if (exact{}.orientation(pts[i], pts[j], pts[k]) == 0) {
						// exactly collinear boundary triple (possible only at
						// the slack boundary): the farthest pair's diametral
						// circle covers the third
						auto const dij = squared_distance(pts[i], pts[j]);
						auto const dik = squared_distance(pts[i], pts[k]);
						auto const djk = squared_distance(pts[j], pts[k]);
						if (dij >= dik && dij >= djk)
							best = diametral_circle(pts[i], pts[j]);
						else if (dik >= djk)
							best = diametral_circle(pts[i], pts[k]);
						else
							best = diametral_circle(pts[j], pts[k]);
					} else
						best = circumcircle(pts[i], pts[j], pts[k]);
				}
			}
		}
		return best;
	}
}
