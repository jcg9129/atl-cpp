#pragma once
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <iterator>
#include <numeric>
#include <ranges>
#include <vector>

namespace atl::geometry {
	namespace detail {
		/**
		 * @brief Coarse polar-angle class in `atan2` order: 0 = lower half plane
		 * `(-pi, 0)`, 1 = the positive x axis (angle 0; the zero vector by
		 * convention too), 2 = upper half plane `(0, pi)`, 3 = the negative
		 * x axis (angle `pi`, the atan2 maximum). Each half plane spans
		 * less than pi, so one cross-product sign orders points inside it.
		 */
		template<coordinate T>
		constexpr int polar_angle_group(point<T> const p) {
			if (p.y < T(0)) return 0;
			if (p.y > T(0)) return 2;
			return p.x < T(0) ? 3 : 1;
		}

		/**
		 * @brief The orientation of `(0, a, b)` under @p policy: for integral
		 * coordinates the policy's sign of the two-point cross, which fits
		 * @ref product_t for every input (`int64` needs no 2^62 bound,
		 * `int32` no 128-bit products).
		 */
		template<coordinate T, sign_policy<T> Policy>
		constexpr int polar_turn(Policy const &policy, point<T> const a, point<T> const b) {
			if constexpr (std::signed_integral<T>)
				return policy.sign(cross(a, b));
			else
				return policy.orientation(point<T>{}, a, b);
		}
	}

	/**
	 * @brief Comparator ordering directions (vectors from the origin) by
	 *        polar angle, the `atan2(y, x)` order of the angle in
	 *        `(-pi, pi]`, from signs only: no `atan2`, no epsilon, exact for
	 *        integral and floating coordinates under the default policy.
	 *        Points on the SAME ray from the origin compare equivalent
	 *        (magnitude is no tiebreak), duplicates included; the zero vector
	 *        is equivalent to the positive x axis (angle 0).
	 *
	 * A comparison object, so it drops into the standard algorithms and
	 * containers directly: `std::ranges::stable_sort(pts, polar_angle_less{})`,
	 * `std::set<point<int64>, polar_angle_less<>>`. One-off comparisons read
	 * `polar_angle_less{}(a, b)`. With @ref exact (any coordinate type), and
	 * with @ref fast on integral coordinates, it is a strict weak ordering on
	 * every input.
	 * @tparam Policy The sign policy (@ref exact by default), checked as
	 *         `sign_policy<Policy, T>` for the compared coordinate type.
	 * @pre For floating coordinates: the floating @ref orientation
	 *      precondition. Integral coordinates take the two-point cross,
	 *      exact over the whole range of the type.
	 * @complexity O(1) per comparison
	 */
	template<std::copyable Policy = exact>
	struct polar_angle_less {
		[[no_unique_address]] Policy policy{}; ///< The sign policy deciding turns within a half plane.

		/** @brief Whether direction @p a has a strictly smaller polar angle than @p b. */
		template<coordinate T>
			requires sign_policy<Policy, T>
		[[nodiscard]] constexpr bool operator()(point<T> const a, point<T> const b) const {
			int const ga = detail::polar_angle_group(a), gb = detail::polar_angle_group(b);
			if (ga != gb) return ga < gb;
			if (ga == 1 || ga == 3) return false; // same axis ray: equivalent
			return detail::polar_turn(policy, a, b) > 0;
		}
	};

	/** @brief `polar_angle_less{p}` deduces the policy type from @p p. */
	template<std::copyable Policy>
	polar_angle_less(Policy) -> polar_angle_less<Policy>;

	/**
	 * @brief Sorts @p points in place by polar angle (see @ref
	 *        polar_angle_less) with `std::ranges::sort`: no allocation. The
	 *        order among same-direction points (duplicates and zero vectors
	 *        included) is unspecified; use `std::ranges::stable_sort(points,
	 *        polar_angle_less{})` if input order must be kept within a ray,
	 *        or @ref polar_angle_order for the stable argsort.
	 * @pre The policy induces a strict weak ordering on the input (the sort
	 *      is undefined otherwise): @ref exact always does; @ref fast on
	 *      floating coordinates only when rounding misclassifies no pair of
	 *      directions; @ref approx in general does not (its dead zone is not
	 *      transitive). Coordinate bounds as @ref polar_angle_less.
	 * @complexity O(n log n) comparisons
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
		requires std::sortable<std::ranges::iterator_t<R>, polar_angle_less<Policy>>
	constexpr void sort_by_polar_angle(R &&points, Policy const p = {}) {
		std::ranges::sort(points, polar_angle_less<Policy>{p});
	}

	/**
	 * @brief Indices of @p points in polar-angle order (see @ref
	 *        polar_angle_less): the argsort counterpart of
	 *        @ref sort_by_polar_angle, and STABLE: same-direction points
	 *        (duplicates and zero vectors included) keep their input order,
	 *        the `lexicographic_order` convention. An empty input gives an
	 *        empty result.
	 * @pre `size(points) < 2^32` (asserted); coordinate bounds as
	 *      @ref polar_angle_less; the policy induces a strict weak ordering
	 *      on the input, as for @ref sort_by_polar_angle.
	 * @complexity O(n log n) time, O(n) memory
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr std::vector<vertex_index> polar_angle_order(R &&points, Policy const p = {}) {
		using T = vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(points) < (uint64(1) << 32));
		auto const first = std::ranges::begin(points);
		uint32 const n = uint32(std::ranges::size(points));

		// Keeping points beside indices avoids gathering coordinates on every
		// comparison; index ties provide stability in constexpr-capable sort.

		struct record {
			point<T> position;
			vertex_index index;
		};

		std::vector<record> records(n);
		for (uint32 i = 0; i < n; ++i) records[i] = {first[i], i};
		// the index breaks same-ray ties, which makes the order stable
		std::ranges::sort(records, [&](record const &l, record const &r) {
			int const gl = detail::polar_angle_group(l.position), gr = detail::polar_angle_group(r.position);
			if (gl != gr) return gl < gr;
			if (gl == 1 || gl == 3) return l.index < r.index; // same axis ray
			int const s = detail::polar_turn(p, l.position, r.position);
			return s > 0 || (s == 0 && l.index < r.index);
		});
		std::vector<vertex_index> order(n);
		for (uint32 k = 0; k < n; ++k) order[k] = records[k].index;
		return order;
	}
}
