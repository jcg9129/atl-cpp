#pragma once
#include <atl/ds/static_vector.hpp>
#include <atl/geometry/line.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <random>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::geometry {
	/**
	 * @defgroup circle_geometry Circle geometry
	 * @brief Integral circle predicates are exact within their asserted bounds;
	 * floating circle predicates evaluate rounded expressions, even with `exact`.
	 * Constructions have no uniform error bound near tangency or collinearity.
	 * @pre Floating inputs are finite, nonzero products normal, and sums finite.
	 * @complexity O(1) time and space unless stated otherwise.
	 * @{
	 */
	/**
	 * @brief A circle as center and radius (`radius >= 0`; a point circle
	 *        is legal). Storing the RADIUS rather than its square is what
	 *        keeps the circle-circle relation exact for integers:
	 *        `(r1 + r2)^2` and `(r1 - r2)^2` are computable in the widened
	 *        types, while recovering `r1 * r2` from stored squares is not.
	 *        Circles with irrational radii (circumcircles) are
	 *        `circle<float64>` constructions.
	 */
	template<coordinate T>
	struct circle {
		using coordinate_type = T; ///< Coordinate and radius type.

		point<T> center{}; ///< Center of the closed disk.
		T radius{};		   ///< Non-negative radius.

		/** @brief Representational equality of all stored fields. */
		[[nodiscard]] friend constexpr bool operator==(circle const &, circle const &) = default;
	};

	namespace detail {
		/**
		 * @brief Preconditions shared by the exact circle predicates: non-negative
		 * radius, and for 64-bit coordinates the usual `< 2^62` bound so
		 * radius sums square inside `int128`.
		 */
		template<coordinate T>
		constexpr void check_radius([[maybe_unused]] T const radius) {
			atl_assert(radius >= T(0));
			if constexpr (std::floating_point<T>)
				atl_assert(normal_product(float64(radius), float64(radius)));
			if constexpr (std::signed_integral<T> && sizeof(T) == 8)
				atl_assert(radius < T(1) << 62);
		}
	}

	/**
	 * @brief Where @p q sits relative to the circle: the sign of
	 *        `radius^2 - |q - center|^2`, exact for integral coordinates
	 *        (both squares in the widened types).
	 * @pre `radius >= 0`; for `int64`: `|x|, |y|, radius < 2^62` (asserted).
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr containment contains(circle<T> const c, point<T> const q, Policy const p = {}) {
		detail::check_radius(c.radius);
		using P2 = difference_product_t<T>;
		P2 const gap = P2(c.radius) * P2(c.radius) - squared_distance(c.center, q);
		return containment(p.sign(gap));
	}

	/** @brief How two circles relate; ordered from far apart to coinciding. */
	enum class circle_relation {
		disjoint,			///< no common point, neither inside the other
		externally_tangent, ///< one common point, touching from outside
		overlapping,		///< two common points
		internally_tangent, ///< one common point, one circle inside the other
		nested,				///< no common point, one strictly inside the other
		coincident			///< the same circle (infinitely many common points)
	};

	/**
	 * @brief Classifies two circles by comparing the squared center
	 *        distance with `(r1 + r2)^2` and `(r1 - r2)^2` — exact for
	 *        integral coordinates.
	 * @pre As @ref atl::geometry::contains, for both circles.
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr circle_relation relation(circle<T> const a, circle<T> const b, Policy const p = {}) {
		detail::check_radius(a.radius);
		detail::check_radius(b.radius);
		detail::check_difference_bounds(a.center, b.center);
		using P2 = difference_product_t<T>;
		if (p.equal(a.center, b.center) && p.sign(P2(P2(a.radius) - P2(b.radius))) == 0)
			return circle_relation::coincident;
		P2 const d2 = squared_distance(a.center, b.center);
		P2 const sum = P2(a.radius) + P2(b.radius);
		P2 const diff = a.radius >= b.radius ? P2(a.radius) - P2(b.radius)
											 : P2(b.radius) - P2(a.radius);
		int const versus_outer = p.sign(d2 - sum * sum);
		if (versus_outer > 0) return circle_relation::disjoint;
		if (versus_outer == 0) return circle_relation::externally_tangent;
		int const versus_inner = p.sign(d2 - diff * diff);
		if (versus_inner < 0) return circle_relation::nested;
		if (versus_inner == 0) return circle_relation::internally_tangent;
		return circle_relation::overlapping;
	}

	/** @brief The circle with segment `a b` as diameter — a `double` construction.
	 * @note Not constexpr (uses runtime math or seeded random state).
	 */
	template<coordinate T>
	[[nodiscard]] circle<float64> diametral_circle(point<T> const a, point<T> const b) {
		point<float64> const center{(float64(a.x) + float64(b.x)) / 2,
									(float64(a.y) + float64(b.y)) / 2};
		return {center, distance(a, b) / 2};
	}

	/**
	 * @brief The circle through three non-collinear points — a `double`
	 *        construction (the true circumcenter is rational; certified
	 *        against a rational test oracle).
	 * @pre `orientation(a, b, c) != 0` (asserted with the exact policy).
	 *
	 * @note Not constexpr (uses runtime math or seeded random state).
	 */
	template<coordinate T>
	[[nodiscard]] circle<float64> circumcircle(point<T> const a, point<T> const b, point<T> const c) {
		atl_assert(exact{}.orientation(a, b, c) != 0);
		auto const u = detail::float64_difference(a, b), v = detail::float64_difference(a, c);
		float64 const u2 = u.x * u.x + u.y * u.y, v2 = v.x * v.x + v.y * v.y;
		float64 const d = 2 * (u.x * v.y - u.y * v.x);
		// the center's offset from a; the radius is its length, taken before
		// the offset is absorbed into a large absolute coordinate
		float64 const ox = (v.y * u2 - u.y * v2) / d, oy = (u.x * v2 - v.x * u2) / d;
		return {{float64(a.x) + ox, float64(a.y) + oy}, std::sqrt(ox * ox + oy * oy)};
	}

	/** @brief How a circle and a line relate. */
	enum class tangency {
		disjoint, ///< no common point
		tangent,  ///< exactly one common point
		secant	  ///< two common points
	};

	/**
	 * @brief Classifies a circle against a line by the sign of the
	 *        discriminant `radius^2 * |direction|^2 - cross^2` — EXACT for
	 *        integral coordinates within `|x|, |y|, radius < 2^30` for
	 *        `int32`/`int64`, `2^14` for `int16` (the
	 *        squared products then fit @ref atl::geometry::difference_product_t; asserted).
	 *        Floating inputs classify on rounded values.
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr tangency relation(circle<T> const c, line<T> const l, Policy const p = {}) {
		atl_assert(l.a != l.b);
		detail::check_radius(c.radius);
		using W = std::conditional_t<std::floating_point<T>, float64, difference_product_t<T>>;
		if constexpr (std::signed_integral<T>) {
			[[maybe_unused]] constexpr T bound = T(1) << (sizeof(T) == 8 ? 30 : sizeof(T) * 8 - 2);
			[[maybe_unused]] auto const is_bounded = [&](point<T> const q) {
				return -bound < q.x && q.x < bound && -bound < q.y && q.y < bound;
			};
			atl_assert(is_bounded(l.a) && is_bounded(l.b) && is_bounded(c.center) && c.radius < bound);
		}
		W const doubled = W(cross(l.a, l.b, c.center));
		W const disc = W(c.radius) * W(c.radius) * W(squared_distance(l.a, l.b))
					 - doubled * doubled;
		int const sign = p.sign(disc);
		return sign < 0 ? tangency::disjoint : sign == 0 ? tangency::tangent
														 : tangency::secant;
	}

	/** @brief Symmetric line-circle relation; same integral bounds and floating rounding. */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr tangency relation(line<T> const l, circle<T> const c, Policy const p = {}) {
		return relation(c, l, p);
	}

	/**
	 * @brief Intersection points of a circle and a line, 0 to 2 of them.
	 *        The COUNT comes from @ref atl::geometry::relation — exact for integral
	 *        coordinates (within its bound); the points are `double`
	 *        constructions. A point circle on the line yields its center.
	 *
	 * @note Not constexpr (uses runtime math or seeded random state).
	 */
	template<coordinate T>
	[[nodiscard]] static_vector<point<float64>, 2> intersection(circle<T> const c, line<T> const l) {
		auto const rel = relation(c, l);
		static_vector<point<float64>, 2> out;
		if (rel == tangency::disjoint) return out;
		// everything relative to the center: the foot of the perpendicular
		// is `center + (fx, fy)`, formed from exact coordinate differences
		auto const d = detail::float64_difference(l.a, l.b), w = detail::float64_difference(l.a, c.center);
		float64 const len2 = d.x * d.x + d.y * d.y;
		float64 const t = (w.x * d.x + w.y * d.y) / len2;
		float64 const fx = t * d.x - w.x, fy = t * d.y - w.y;
		point<float64> const foot{float64(c.center.x) + fx, float64(c.center.y) + fy};
		if (rel == tangency::tangent) {
			out.push_back(foot);
			return out;
		}
		float64 const gap2 =
			(std::max)(0.0, float64(c.radius) * float64(c.radius) - (fx * fx + fy * fy));
		float64 const scale = std::sqrt(gap2 / len2);
		out.push_back({foot.x - scale * d.x, foot.y - scale * d.y});
		out.push_back({foot.x + scale * d.x, foot.y + scale * d.y});
		return out;
	}

	/** @brief Symmetric line-circle boundary crossings; same bounds and point order.
	 * @note Not constexpr (uses runtime math or seeded random state).
	 */
	template<coordinate T>
	[[nodiscard]] static_vector<point<float64>, 2> intersection(line<T> const l, circle<T> const c) {
		return intersection(c, l);
	}

	/**
	 * @brief Intersection points of the circle's BOUNDARY and a CLOSED
	 *        segment, 0 to 2, ordered along the segment. The COUNT is exact
	 *        for integral coordinates (root-counting of the quadratic
	 *        `|s(t) - center|^2 - r^2` on `[0, 1]` from exact signs: the
	 *        line tangency, endpoint containments, and the vertex-parameter
	 *        position); the points are `double` constructions. Disk queries
	 *        are @ref atl::geometry::contains on the endpoints — this is the curve.
	 * @pre As the circle-line relation for integral exactness.
	 *
	 * @note Not constexpr (uses runtime math or seeded random state).
	 */
	template<coordinate T>
	[[nodiscard]] static_vector<point<float64>, 2> intersection(circle<T> const c, segment<T> const s) {
		static_vector<point<float64>, 2> out;
		detail::check_radius(c.radius);
		if (s.a == s.b) { // a point segment has no spanning line
			if (contains(c, s.a) == containment::boundary)
				out.push_back(to_float64(s.a));
			return out;
		}
		line<T> const spanned{s.a, s.b};
		auto const line_rel = relation(c, spanned);
		if (line_rel == tangency::disjoint) return out;
		exact const ex{};
		// sign of dot(s.b - s.a, to - from) via dot(u, w) = -cross(perp(u), w)
		auto const dot_sign = [&](point<T> const from, point<T> const to) {
			return -ex.cross_sign(perp(s.a), perp(s.b), from, to);
		};
		int const after_start = dot_sign(s.a, c.center); // sign of the vertex parameter t*
		int const before_end = -dot_sign(s.b, c.center); // sign of 1 - t*
		std::size_t count = 0;
		if (line_rel == tangency::tangent)
			count = after_start >= 0 && before_end >= 0 ? 1 : 0;
		else {
			int const start_gap = -int(contains(c, s.a)); // sign of f(0)
			int const end_gap = -int(contains(c, s.b));	  // sign of f(1)
			if (start_gap < 0 && end_gap < 0)
				count = 0; // both endpoints strictly inside: convexity keeps f < 0
			else if (start_gap * end_gap < 0)
				count = 1; // one sign change
			else if (start_gap == 0 && end_gap == 0)
				count = 2; // the endpoints themselves are the roots
			else if (start_gap == 0)
				count = end_gap < 0 ? 1 : 1 + std::size_t(after_start > 0);
			else if (end_gap == 0)
				count = start_gap < 0 ? 1 : 1 + std::size_t(before_end > 0);
			else // both strictly outside: two roots iff the vertex is interior
				count = after_start > 0 && before_end > 0 ? 2 : 0;
		}
		if (count == 0) return out;
		auto const on_line = intersection(c, spanned);
		if (on_line.size() == std::size_t(1) || count == 2) {
			for (auto const q : on_line) out.push_back(q);
			return out;
		}
		// one of two candidates: keep the one whose parameter lies in [0, 1]
		auto const d = detail::float64_difference(s.a, s.b);
		float64 const len2 = d.x * d.x + d.y * d.y;
		auto const off_range = [&](point<float64> const q) {
			float64 const t =
				((q.x - float64(s.a.x)) * d.x + (q.y - float64(s.a.y)) * d.y) / len2;
			return t < 0 ? -t : t > 1 ? t - 1
									  : 0.0;
		};
		out.push_back(off_range(on_line[0]) <= off_range(on_line[1]) ? on_line[0] : on_line[1]);
		return out;
	}

	/** @brief Symmetric segment-circle boundary crossings; same bounds and point order.
	 * @note Not constexpr (uses runtime math or seeded random state).
	 */
	template<coordinate T>
	[[nodiscard]] static_vector<point<float64>, 2> intersection(segment<T> const s, circle<T> const c) {
		return intersection(c, s);
	}

	/**
	 * @brief Intersection points of two circles, 0 to 2. The COUNT comes
	 *        from @ref atl::geometry::relation — exact for integral coordinates; the
	 *        points are `double` constructions via the radical line.
	 *        Coincident circles return none (the intersection is not a
	 *        finite point set — classify first).
	 *
	 * @note Not constexpr (uses runtime math or seeded random state).
	 */
	template<coordinate T>
	[[nodiscard]] static_vector<point<float64>, 2> intersection(circle<T> const a, circle<T> const b) {
		auto const rel = relation(a, b);
		static_vector<point<float64>, 2> out;
		if (rel == circle_relation::disjoint || rel == circle_relation::nested
			|| rel == circle_relation::coincident)
			return out;
		auto const [dx, dy] = detail::float64_difference(a.center, b.center);
		float64 const d2 = dx * dx + dy * dy;
		float64 const ra = float64(a.radius), rb = float64(b.radius);
		// distance from a.center to the radical line, times d
		float64 const along = (d2 + ra * ra - rb * rb) / 2;
		point<float64> const foot{float64(a.center.x) + along / d2 * dx,
								  float64(a.center.y) + along / d2 * dy};
		if (rel != circle_relation::overlapping) { // either tangency
			out.push_back(foot);
			return out;
		}
		float64 const gap2 = (std::max)(0.0, ra * ra * d2 - along * along);
		float64 const scale = std::sqrt(gap2) / d2;
		out.push_back({foot.x - scale * dy, foot.y + scale * dx});
		out.push_back({foot.x + scale * dy, foot.y - scale * dx});
		return out;
	}

	/**
	 * @brief The points where the tangents from @p q touch the circle —
	 *        empty when @p q is strictly inside, @p q itself when it lies
	 *        ON the boundary (the tangent touches there), and two points
	 *        when outside, the one on the LEFT of the directed line
	 *        `c.center -> q` first (the circle-circle convention). The
	 *        COUNT is exact for integral inputs (the power of the point);
	 *        the points are rounded constructions.
	 * @pre `radius >= 0`; for `int64`: `|x|, |y|, radius < 2^62` (asserted).
	 *
	 * @note Not constexpr (uses runtime math or seeded random state).
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] static_vector<point<float64>, 2> tangent_points(circle<T> const c, point<T> const q,
																  Policy const p = {}) {
		static_vector<point<float64>, 2> out;
		containment const at = contains(c, q, p);
		if (at == containment::inside) return out;
		if (at == containment::boundary) {
			out.push_back(to_float64(q));
			return out;
		}
		auto const [dx, dy] = detail::float64_difference(c.center, q);
		float64 const d2 = float64(squared_distance(c.center, q));
		float64 const r = float64(c.radius);
		// the tangent chord's foot sits r^2/d2 along the center line; the
		// half-chord is r * tangent_length / d
		float64 const foot = r * r / d2;
		float64 const half = r * std::sqrt((std::max)(d2 - r * r, 0.0)) / d2;
		float64 const fx = float64(c.center.x) + foot * dx;
		float64 const fy = float64(c.center.y) + foot * dy;
		out.push_back({fx - half * dy, fy + half * dx}); // left of center -> q
		out.push_back({fx + half * dy, fy - half * dx});
		return out;
	}

	/**
	 * @brief The common tangent LINES of two circles: 4 for disjoint
	 *        circles, emitted in side-sign order `(+,+), (-,-), (+,-),
	 *        (-,+)` — so the two OUTER tangents (both centers on one side)
	 *        first, then the two INNER ones; 3 when
	 *        externally tangent (the inner pair collapses into the tangent
	 *        at the touch point, last), 2 when overlapping (outer only), 1 when
	 *        internally tangent, 0 nested, and 0 for COINCIDENT circles
	 *        (infinitely many — the documented representational limit).
	 *        Counts are decided EXACTLY for integral inputs (the
	 *        discriminant `d^2 - (s2 r_b - s1 r_a)^2` per sign pair in the
	 *        widened type); the lines are rounded constructions. Point
	 *        circles are legal (tangents through the point).
	 * @pre Radii `>= 0`; for `int64`: `|x|, |y|, radius < 2^62` (asserted).
	 *
	 * @note Not constexpr (uses runtime math or seeded random state).
	 */
	template<coordinate T>
	[[nodiscard]] static_vector<line<float64>, 4> common_tangents(circle<T> const a, circle<T> const b) {
		detail::check_radius(a.radius);
		detail::check_radius(b.radius);
		detail::check_difference_bounds(a.center, b.center);
		static_vector<line<float64>, 4> out;
		if (a == b) return out; // coincident: infinitely many tangents
		using P2 = difference_product_t<T>;
		P2 const d2 = squared_distance(a.center, b.center);
		if (a.radius == T(0) && b.radius == T(0)) {
			out.push_back({to_float64(a.center),
						   to_float64(b.center)});
			return out;
		}
		auto const [dx, dy] = detail::float64_difference(a.center, b.center);
		// a tangent A x + B y + C = 0 with A^2 + B^2 = 1 signs its distance
		// s1 r_a at a and s2 r_b at b; subtracting pins the direction. A
		// ZERO radius makes its sign vacuous: only one sign is enumerated,
		// or every line would appear twice
		constexpr std::array<std::array<int, 2>, 4> sign_pairs{{{1, 1}, {-1, -1}, {1, -1}, {-1, 1}}};
		for (auto const [s1, s2] : sign_pairs) {
			if ((s1 < 0 && a.radius == T(0)) || (s2 < 0 && b.radius == T(0))) continue;
			P2 const rr = P2(s2) * P2(b.radius) - P2(s1) * P2(a.radius);
			P2 const z = d2 - rr * rr; // exact discriminant
			if (z < 0) continue;
			// a tangency line repeats under the global sign flip; keep the
			// first enumerated of each pair (a point circle enumerates
			// only s1 = 1, so there the flip is the other s2)
			if (z == 0 && (s1 != 1 || (a.radius == T(0) && s2 != 1))) continue;
			float64 const sq = std::sqrt((std::max)(float64(z), 0.0));
			float64 const big_a = (dx * float64(rr) + dy * sq) / float64(d2);
			float64 const big_b = (dy * float64(rr) - dx * sq) / float64(d2);
			float64 const big_c = float64(s1) * float64(a.radius)
								- big_a * float64(a.center.x)
								- big_b * float64(a.center.y);
			out.push_back(line_from_coefficients(big_a, big_b, big_c));
		}
		return out;
	}

	namespace detail {
		/**
		 * @brief Signed area of the intersection of triangle (origin, a, b) with
		 * the disk of radius @p r about the origin: split the chord a-b at
		 * its circle crossings; pieces whose midpoint is inside contribute
		 * plain triangles, pieces outside contribute circular SECTORS
		 * (half r^2 times the signed subtended angle — always below pi,
		 * since a triangle's apex angle is).
		 */
		inline float64 disk_triangle_area(point<float64> const a, point<float64> const b,
										  float64 const r) {
			auto const inside = [&](point<float64> const p) {
				return p.x * p.x + p.y * p.y < r * r;
			};
			auto const sector = [&](point<float64> const u, point<float64> const v) {
				return r * r * std::atan2(u.x * v.y - u.y * v.x, u.x * v.x + u.y * v.y)
					 / 2;
			};
			point<float64> const d{b.x - a.x, b.y - a.y};
			float64 const qa = d.x * d.x + d.y * d.y;
			float64 const qb = a.x * d.x + a.y * d.y; // halved linear term
			float64 const qc = a.x * a.x + a.y * a.y - r * r;
			float64 ts[2]{};
			int crossings = 0;
			if (float64 const disc = qb * qb - qa * qc; disc > 0 && qa > 0) {
				float64 const sq = std::sqrt(disc);
				for (float64 const t : {(-qb - sq) / qa, (-qb + sq) / qa})
					if (t > 0 && t < 1) ts[crossings++] = t;
			}
			float64 total = 0, t_prev = 0;
			point<float64> prev = a;
			for (int k = 0; k <= crossings; ++k) {
				float64 const t = k < crossings ? ts[k] : 1;
				point<float64> const cur =
					k < crossings ? point<float64>{a.x + t * d.x, a.y + t * d.y} : b;
				float64 const tm = (t_prev + t) / 2;
				point<float64> const mid{a.x + tm * d.x, a.y + tm * d.y};
				total += inside(mid) ? (prev.x * cur.y - prev.y * cur.x) / 2
									 : sector(prev, cur);
				prev = cur;
				t_prev = t;
			}
			return total;
		}
	}

	/**
	 * @brief The area of the intersection of a disk and a SIMPLE polygon,
	 *        either winding — a rounded `float64` construction: the
	 *        polygon's signed triangle fan about the circle's center, each
	 *        triangle clipped against the disk into straight and sector
	 *        pieces.
	 * @pre A simple polygon (documented only); `radius >= 0` (asserted). Fewer than 3
	 *      vertices or a zero radius give 0.
	 * @complexity O(n)
	 *
	 * @note Not constexpr (uses runtime math or seeded random state).
	 */
	template<coordinate T, vertex_range R>
		requires std::same_as<vertex_coordinate_t<R>, T>
	[[nodiscard]] float64 intersection_area(circle<T> const c, R &&poly) {
		detail::check_radius(c.radius);
		auto const n = std::ranges::size(poly);
		if (n < 3 || c.radius == T(0)) return 0;
		auto const first = std::ranges::begin(poly);
		auto const rel = [&](point<T> const p) { return detail::float64_difference(c.center, p); };
		float64 sum = 0;
		for (std::size_t i = 0; i < n; ++i)
			sum += detail::disk_triangle_area(rel(first[i]), rel(first[(i + 1) % n]),
											  float64(c.radius));
		return std::abs(sum);
	}

	/** @brief Symmetric polygon-disk intersection area; same simple-polygon precondition.
	 * @note Not constexpr (uses runtime math or seeded random state).
	 */
	template<coordinate T, vertex_range R>
		requires std::same_as<vertex_coordinate_t<R>, T>
	[[nodiscard]] float64 intersection_area(R &&poly, circle<T> const c) {
		return intersection_area(c, std::forward<R>(poly));
	}

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

	/** @} */
}
