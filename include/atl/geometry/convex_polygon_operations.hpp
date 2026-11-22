#pragma once
#include <atl/geometry/calipers.hpp>
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/detail/compare_products.hpp>
#include <atl/geometry/line.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <deque>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::geometry {
	namespace detail {
		/**
		 * @brief `normalized_polygon` for a CONSTRUCTED (rounded `double`)
		 *        polygon: merges consecutive vertices that coincide
		 *        and drops vertices lying on the chord of their neighbors,
		 *        both up to the rounding of the constructions. The tolerance
		 *        is RELATIVE — `2^-44` of the largest coordinate magnitude,
		 *        a few hundred ulps. Translation increases this tolerance, so
		 *        thin shapes can lose corners; this is a construction cleanup.
		 *        The winding is made counterclockwise, keeping the first
		 *        vertex first.
		 *
		 * When fewer than 3 vertices would survive (a result thinner than
		 * the rounding itself) the input is returned unchanged.
		 * @complexity O(n)
		 */
		[[nodiscard]] inline std::vector<point<float64>> normalized_constructed_polygon(std::vector<point<float64>> const &boundary) {
			float64 magnitude = 0;
			for (auto const q : boundary) magnitude = (std::max)({magnitude, std::abs(q.x), std::abs(q.y)});
			float64 const tolerance = magnitude * 0x1p-44;
			auto const apart = [&](point<float64> const a, point<float64> const b) {
				return std::hypot(a.x - b.x, a.y - b.y) > tolerance;
			};
			std::vector<point<float64>> distinct;
			distinct.reserve(boundary.size());
			for (auto const q : boundary)
				if (distinct.empty() || apart(q, distinct.back())) distinct.push_back(q);
			while (distinct.size() > 1 && !apart(distinct.front(), distinct.back())) distinct.pop_back();
			std::size_t const m = distinct.size();
			if (m < 3) return boundary;
			std::vector<point<float64>> out;
			out.reserve(m);
			float64 doubled_area = 0;
			for (std::size_t i = 0; i < m; ++i) {
				auto const before = distinct[i == 0 ? m - 1 : i - 1], after = distinct[i + 1 == m ? 0 : i + 1];
				auto const chord = after - before, offset = distinct[i] - before;
				float64 const turn = chord.x * offset.y - chord.y * offset.x;
				if (std::abs(turn) > tolerance * std::hypot(chord.x, chord.y)) out.push_back(distinct[i]);
				doubled_area += (distinct[i].x - distinct[0].x) * (after.y - distinct[0].y)
							  - (distinct[i].y - distinct[0].y) * (after.x - distinct[0].x);
			}
			if (out.size() < 3) return boundary;
			if (doubled_area < 0) std::ranges::reverse(out.begin() + 1, out.end());
			return out;
		}
	}

	namespace detail {
		/**
		 * @brief Check raw input bounds before any Minkowski sums in T, and
		 * select wide quartic predicates if any coordinate reaches 2^30.
		 * Int64 inputs stay below 2^61; Minkowski vertices stay below 2^62,
		 * their differences below 2^63, and quadratic predicates fit int128.
		 * Narrower types stay below 2^(bits - 2) so those sums fit T.
		 */
		template<vertex_range R>
		constexpr bool check_operation_bounds(R &&points) {
			using T = vertex_coordinate_t<R>;
			bool uses_wide = false;
			if constexpr (std::signed_integral<T>) {
				[[maybe_unused]] constexpr T bound = T(1) << (sizeof(T) == 8 ? 61 : sizeof(T) * 8 - 2);
				for (point<T> const q : points) {
					atl_assert(-bound < q.x && q.x < bound && -bound < q.y && q.y < bound);
					if constexpr (sizeof(T) == 8) {
						constexpr T fast_bound = T(1) << 30;
						uses_wide |= q.x <= -fast_bound || q.x >= fast_bound || q.y <= -fast_bound || q.y >= fast_bound;
					}
				}
			}
			return uses_wide;
		}

		template<typename Product>
			requires std::same_as<Product, int128> || std::same_as<Product, float64>
		[[nodiscard]] constexpr int compare_operation_products(Product const a, Product const b, Product const c, Product const d,
															   [[maybe_unused]] bool const uses_wide) {
			if constexpr (std::same_as<Product, int128>)
				if (uses_wide) return compare_wide_products(a, b, c, d);
			Product const left = a * b, right = c * d;
			return left > right ? 1 : left < right ? -1
												   : 0;
		}

		/**
		 * @brief Exact side of the crossing point of @p l1 and @p l2 relative to
		 * @p l3 (positive = left side of `l3`). Integral coordinates: an
		 * int128 fast path below 2^30, exact wide products below 2^61;
		 * floating coordinates evaluate rounded.
		 * @pre `l1`, `l2` not parallel; raw integral coordinates below 2^61 when
		 * uses_wide, below 2^30 otherwise.
		 */
		template<coordinate T>
		[[nodiscard]] constexpr int crossing_side(line<T> const l1, line<T> const l2, line<T> const l3, [[maybe_unused]] bool const uses_wide = false) {
			if constexpr (std::signed_integral<T>) {
				if (uses_wide) {
					// D = u x v, N = (q-p) x v, E = w x (p-r), F = w x u.
					// Each quadratic factor is below 2^126 at the raw 2^61 bound.
					int128 const d = cross_determinant(l1.a, l1.b, l2.a, l2.b);
					int128 const n = cross_determinant(l1.a, l2.a, l2.a, l2.b);
					int128 const e = cross_determinant(l3.a, l3.b, l3.a, l1.a);
					int128 const f = cross_determinant(l3.a, l3.b, l1.a, l1.b);
					return compare_wide_products(e, d, -n, f) * (d < 0 ? -1 : 1);
				}
			}
			auto const [a1, b1, c1] = line_coefficients(l1);
			auto const [a2, b2, c2] = line_coefficients(l2);
			auto const [a3, b3, c3] = line_coefficients(l3);
			using W = std::conditional_t<std::floating_point<T>, float64, int128>;
			W const d = W(a1) * W(b2) - W(a2) * W(b1);
			W const v = W(a3) * (W(b1) * W(c2) - W(b2) * W(c1))
					  + W(b3) * (W(c1) * W(a2) - W(c2) * W(a1)) + W(c3) * d;
			W const signed_value = d < 0 ? -v : v;
			return signed_value > 0 ? 1 : signed_value < 0 ? -1
														   : 0;
		}

		/**
		 * @brief A hull edge as a half-plane boundary (interior = left side),
		 * remembering which polygon and which edge produced it.
		 */
		template<coordinate T>
		struct tagged_halfplane {
			line<T> boundary;
			bool is_from_second;
			uint32 edge;
		};

		/**
		 * @brief Rotate a CCW hull to start at its (y, x)-minimal vertex, making
		 * the edge-direction sequence sorted in the [0, 2pi) sweep order.
		 */
		template<coordinate T>
		void rotate_hull_to_low(std::vector<point<T>> &hull) {
			auto const it = std::ranges::min_element(hull, [](point<T> const l, point<T> const r) {
				return l.y != r.y ? l.y < r.y : l.x < r.x;
			});
			std::ranges::rotate(hull, it);
		}

		/**
		 * @brief Angular merge of both hulls' edge half-planes (hulls must already
		 * start at their (y, x)-minimal vertex); same-direction parallel
		 * pairs keep only the MORE RESTRICTIVE line.
		 */
		template<coordinate T, sign_policy<T> Policy>
		[[nodiscard]] std::vector<tagged_halfplane<T>> merged_halfplanes(std::vector<point<T>> const &a,
																		 std::vector<point<T>> const &b,
																		 Policy const p) {
			auto const edges_of = [](std::vector<point<T>> const &v, bool const is_second) {
				std::vector<tagged_halfplane<T>> e;
				e.reserve(v.size());
				for (uint32 i = 0; i < uint32(v.size()); ++i)
					e.push_back({{v[i], v[i + 1 == v.size() ? 0 : i + 1]}, is_second, i});
				return e;
			};
			auto const ea = edges_of(a, false), eb = edges_of(b, true);
			std::vector<tagged_halfplane<T>> merged;
			merged.reserve(ea.size() + eb.size());
			std::size_t i = 0, j = 0;
			auto const before = [&](line<T> const &l, line<T> const &m) {
				return detail::sweep_before(l.a, l.b, m.a, m.b, p);
			};
			while (i < ea.size() || j < eb.size()) {
				bool takes_a = j == eb.size(), takes_b = i == ea.size();
				if (!takes_a && !takes_b) {
					takes_a = !before(eb[j].boundary, ea[i].boundary);
					takes_b = !before(ea[i].boundary, eb[j].boundary);
				}
				if (takes_a && takes_b) { // same direction: the more restrictive
					bool const a_restricts =
						p.orientation(eb[j].boundary.a, eb[j].boundary.b, ea[i].boundary.a) >= 0;
					merged.push_back(a_restricts ? ea[i] : eb[j]);
					++i;
					++j;
				} else
					merged.push_back(takes_a ? ea[i++] : eb[j++]);
			}
			return merged;
		}

		/**
		 * @brief Half-plane intersection over angularly sorted constraints — the
		 * classic deque with integral-exact crossing-side pops. Callers guarantee a
		 * positive-area result (classified beforehand), so adjacent
		 * survivors always properly cross; exactly-parallel adjacencies
		 * (transient, mid-processing) are conservatively never popped.
		 */
		template<coordinate T, sign_policy<T> Policy>
		[[nodiscard]] std::vector<tagged_halfplane<T>>
		halfplane_intersection(std::vector<tagged_halfplane<T>> const &constraints, Policy const p, bool const uses_wide) {
			std::deque<tagged_halfplane<T>> active;
			// is `candidate` redundant, given its surviving neighbor and the
			// incoming constraint? Non-parallel neighbors: the classic
			// their-crossing-right-of-incoming test. ANTI-parallel neighbors
			// (same-direction pairs were merged) have no crossing — the
			// candidate is implied iff the keeper-incoming corner already
			// satisfies it.
			auto const redundant = [&](tagged_halfplane<T> const &keeper,
									   tagged_halfplane<T> const &candidate,
									   tagged_halfplane<T> const &incoming) {
				if (p.cross_sign(keeper.boundary.a, keeper.boundary.b, candidate.boundary.a,
								 candidate.boundary.b)
					== 0)
					return crossing_side(keeper.boundary, incoming.boundary,
										 candidate.boundary, uses_wide)
						>= 0;
				return crossing_side(keeper.boundary, candidate.boundary, incoming.boundary, uses_wide) < 0;
			};
			auto const bad_back = [&](tagged_halfplane<T> const &incoming) {
				return active.size() >= 2
					&& redundant(active[active.size() - 2], active.back(), incoming);
			};
			auto const bad_front = [&](tagged_halfplane<T> const &incoming) {
				return active.size() >= 2 && redundant(active[1], active[0], incoming);
			};
			for (auto const &c : constraints) {
				while (bad_back(c)) active.pop_back();
				while (bad_front(c)) active.pop_front();
				active.push_back(c);
			}
			for (bool has_changed = true; has_changed;) {
				has_changed = false;
				while (active.size() >= 3 && bad_back(active.front())) {
					active.pop_back();
					has_changed = true;
				}
				while (active.size() >= 3 && bad_front(active.back())) {
					active.pop_front();
					has_changed = true;
				}
			}
			return {active.begin(), active.end()};
		}

		/**
		 * @brief Recover the contact set of two hulls whose interiors do not overlap
		 * but whose distance is zero. The Minkowski-difference feature
		 * through the origin gives the contact normal; each hull's EXACT
		 * extreme face along it (one edge or one vertex — strict hulls) is
		 * scanned directly, and the contact is the faces' overlap.
		 */
		template<coordinate T, sign_policy<T> Policy>
		[[nodiscard]] segment_intersection<T> boundary_contact(minkowski_difference<T> const &d, Policy const p) {
			point<T> const origin{};
			// the feature direction: a boundary edge through the origin, or
			// any incident edge when the origin is a vertex
			point<T> direction{};
			std::size_t const k = d.boundary.size();
			if (k == 1) { // point-vs-point difference: vertex contact
				segment_intersection<T> out;
				out.kind = intersection_kind::at_point;
				out.at = to_float64(d.a_hull[0]);
				return out;
			}
			for (std::size_t i = 0, j = k - 1; i < k; j = i++) {
				segment<T> const edge{d.boundary[j], d.boundary[i]};
				if (edge.a != edge.b && contains(edge, origin, p)) {
					if constexpr (std::signed_integral<T>) {
						// A supporting input edge is parallel to this difference edge.
						// Its magnitude stays below 2^62 (the predicate coordinate bound),
						// even when the difference edge approaches 2^63.
						auto const from = d.supports[j], to = d.supports[i];
						direction = from[0] != to[0] ? d.a_hull[to[0]] - d.a_hull[from[0]]
													 : d.negated_b_hull[to[1]] - d.negated_b_hull[from[1]];
					} else {
						// Floating coordinates keep the difference edge itself: a tolerant
						// policy's ties depend on the direction's length, and the shorter input
						// edge would tie faces the difference edge separates.
						direction = edge.b - edge.a;
					}
					break;
				}
			}
			atl_assert(direction.x != T(0) || direction.y != T(0));
			point<T> const outward{direction.y, T(-direction.x)}; // CCW: right side
			// the exact extreme face of a hull along `u`: an edge or vertex
			auto const face_along = [&](std::vector<point<T>> const &hull, point<T> const u) {
				point<T> const anchor{};
				std::size_t best = 0;
				for (std::size_t v = 1; v < hull.size(); ++v)
					if (-p.cross_sign(perp(anchor), perp(u), hull[best], hull[v]) > 0) best = v;
				std::size_t other = best; // a strict hull's face has <= 2 vertices
				for (std::size_t v = 0; v < hull.size(); ++v)
					if (v != best
						&& -p.cross_sign(perp(anchor), perp(u), hull[best], hull[v]) == 0)
						other = v;
				return segment<T>{hull[best], hull[other]};
			};
			segment<T> const a_face = face_along(d.a_hull, outward);
			segment<T> const negated_b_face = face_along(d.negated_b_hull, outward);
			segment<T> const b_face{-negated_b_face.a, -negated_b_face.b};
			return intersection(a_face, b_face, p);
		}

		template<coordinate T>
		[[nodiscard]] std::vector<point<float64>> as_float_polygon(std::vector<point<T>> const &hull) {
			std::vector<point<float64>> out;
			out.reserve(hull.size());
			for (auto const q : hull) out.push_back(to_float64(q));
			return out;
		}

		/**
		 * @brief Clip a segment to a convex hull using an integral-exact rational
		 * parameter interval (rounded for floating T), with constructed endpoints. Returns 0-2 points.
		 * @pre Nonempty hull; raw integral coordinates below 2^61 when uses_wide,
		 * below 2^30 otherwise. Narrower storage limits as check_operation_bounds.
		 */
		template<coordinate T, sign_policy<T> Policy>
		[[nodiscard]] std::vector<point<float64>>
		segment_clipped_to_hull(segment<T> const s, std::vector<point<T>> const &hull, Policy const p, bool const uses_wide) {
			using W = std::conditional_t<std::floating_point<T>, float64, int128>;
			W low_n = 0, low_d = 1, high_n = 1, high_d = 1; // t in [0, 1]
			for (std::size_t i = 0, j = hull.size() - 1; i < hull.size(); j = i++) {
				int const side_a = p.orientation(hull[j], hull[i], s.a);
				int const side_b = p.orientation(hull[j], hull[i], s.b);
				if (side_a >= 0 && side_b >= 0) continue;
				if (side_a < 0 && side_b < 0) return {};
				W numerator = W(cross(hull[j], hull[i], s.a));
				W denominator = numerator - W(cross(hull[j], hull[i], s.b));
				if (denominator < 0) {
					numerator = -numerator;
					denominator = -denominator;
				}
				if (side_a < 0) { // entering: raise the lower bound
					if (compare_operation_products(numerator, low_d, low_n, denominator, uses_wide) > 0) {
						low_n = numerator;
						low_d = denominator;
					}
				} else { // leaving: lower the upper bound
					if (compare_operation_products(numerator, high_d, high_n, denominator, uses_wide) < 0) {
						high_n = numerator;
						high_d = denominator;
					}
				}
			}
			if (compare_operation_products(low_n, high_d, high_n, low_d, uses_wide) > 0) return {};
			auto const at = [&](W const n, W const den) {
				float64 const t = float64(n) / float64(den);
				return point<float64>{float64(s.a.x) + t * (float64(s.b.x) - float64(s.a.x)),
									  float64(s.a.y) + t * (float64(s.b.y) - float64(s.a.y))};
			};
			std::vector<point<float64>> out{at(low_n, low_d)};
			if (compare_operation_products(low_n, high_d, high_n, low_d, uses_wide) != 0) out.push_back(at(high_n, high_d));
			if (out.size() == 2 && !(out[0] < out[1])) std::swap(out[0], out[1]); // lex order
			return out;
		}

		/** @brief Exact rational order on a host edge; both cuts must cross it.
		 * @pre Raw integral coordinates below 2^61 when uses_wide, below 2^30 otherwise.
		 */
		template<coordinate T>
		[[nodiscard]] constexpr bool param_before(line<T> const host, line<T> const cut_u,
												  line<T> const cut_v, bool const uses_wide) {
			using W = std::conditional_t<std::floating_point<T>, float64, int128>;
			auto const rational = [&](line<T> const cut) {
				W n = W(cross(cut.a, cut.b, host.a));
				W d = n - W(cross(cut.a, cut.b, host.b));
				if (d < 0) {
					n = -n;
					d = -d;
				}
				return std::pair<W, W>{n, d};
			};
			auto const [un, ud] = rational(cut_u);
			auto const [vn, vd] = rational(cut_v);
			return compare_operation_products(un, vd, vn, ud, uses_wide) < 0;
		}

		/**
		 * @brief Classify the hulls' overlap through the Minkowski
		 * difference: outside = disjoint, boundary = zero-measure contact,
		 * inside = interiors properly overlap. Integral decisions are exact under
		 * exact/fast; floating differences and non-exact policies are construction-grade.
		 */
		template<coordinate T, sign_policy<T> Policy>
		[[nodiscard]] containment overlap_of(minkowski_difference<T> const &d, Policy const p) {
			point<T> const origin{};
			if (d.boundary.size() >= 3) return contains(d.boundary, origin, p);
			return contains(segment<T>{d.boundary.front(), d.boundary.back()}, origin, p)
					 ? containment::boundary
					 : containment::outside;
		}

		/**
		 * @brief Position of `q` (known to lie on the hull boundary) as the edge
		 * index `e` with `q` in the half-open span `[hull[e], hull[e+1])`.
		 */
		template<coordinate T, sign_policy<T> Policy>
		[[nodiscard]] std::size_t boundary_position(std::vector<point<T>> const &hull, point<T> const q,
													Policy const p) {
			std::size_t const h = hull.size();
			for (std::size_t e = 0; e < h; ++e) {
				point<T> const v0 = hull[e], v1 = hull[e + 1 == h ? 0 : e + 1];
				if (p.equal(q, v1)) continue; // belongs to the next span
				if (contains(segment<T>{v0, v1}, q, p)) return e;
			}
			atl_assert(false);
			return 0;
		}

		/**
		 * @brief Append `from` and every hull vertex strictly between `from` and
		 * `to` (both on the boundary), walking CCW; `to` itself is NOT
		 * appended. Same-span inputs with `to` ahead of `from` contribute
		 * only `from`; `to` behind `from` wraps the whole polygon.
		 */
		template<coordinate T, sign_policy<T> Policy>
		void append_arc(std::vector<point<T>> const &hull, point<T> const from, point<T> const to,
						Policy const p, std::vector<point<float64>> &sink) {
			std::size_t const h = hull.size();
			std::size_t const e_from = boundary_position(hull, from, p);
			std::size_t const e_to = boundary_position(hull, to, p);
			sink.push_back(to_float64(from));
			std::size_t count = 0; // vertices to emit, starting at e_from + 1
			if (e_from != e_to)
				count = (e_to + h - e_from) % h;
			else {
				// same span: exact order along the edge direction decides
				point<T> const v0 = hull[e_from];
				point<T> const v1 = hull[e_from + 1 == h ? 0 : e_from + 1];
				int const ahead = -p.cross_sign(perp(v0), perp(v1), from, to); // dot sign
				count = ahead > 0 ? 0 : h;									   // to ahead: nothing; behind: wrap
			}
			for (std::size_t s = 1; s <= count; ++s) {
				point<T> const v = hull[(e_from + s) % h];
				if (p.equal(v, from) || p.equal(v, to)) continue;
				sink.push_back(to_float64(v));
			}
		}
	}

	/**
	 * @brief The intersection of the convex hulls of two point sets, as a
	 *        convex CCW boundary of `point<float64>`: empty (0
	 *        vertices), a point (1), a segment (2), or a polygon. Under exact/fast with integral input the
	 *        overlap classification is exact (Minkowski difference):
	 *        disjoint returns empty; touching selects the exact contact
	 *        point/segment on the input grid, then rounds its coordinates; proper overlap runs a
	 *        half-plane intersection over the angularly merged edge
	 *        constraints with exact integral crossing-side pops. Crossing
	 *        vertex coordinates are rounded constructions (coinciding
	 *        constructions of concurrent constraints are merged by a
	 *        cleanup using 2^-44 times the largest coordinate magnitude).
	 * @param a_points First point set; not modified.
	 * @param b_points Second point set; not modified.
	 * @param p Sign policy for input predicates.
	 * @return Constructed intersection boundary; the closing edge is implicit.
	 * @note Not constexpr (uses runtime deque and construction cleanup).
	 * @tparam RA First vertex range.
	 * @tparam RB Second vertex range with the same coordinate type.
	 * @tparam Policy Sign policy for that coordinate type.
	 * @pre Both inputs non-empty (asserted); integral coordinates:
	 *      `|x|, |y| < 2^61` for int64, `< 2^30` for int32, `< 2^14` for int16,
	 *      `< 2^6` for 8-bit storage (asserted; character aliases such as int8 are
	 *      rejected by coordinate). Integral quartic predicates use int128
	 *      below 2^30 and unsigned 256-bit magnitudes otherwise, selected in
	 *      the entry bounds scan. Integral topology decisions are exact under exact/fast.
	 *      Floating Minkowski differences and crossing-side tests are rounded,
	 *      including under exact; topology is construction-grade. Finite normal
	 *      products and finite intermediate sums/constructions are required.
	 * @note Cleanup uses 2^-44 times the largest coordinate magnitude; thin or
	 * translated results can retain duplicate or straight vertices, so strictness
	 * is not guaranteed. Sorting uses O(log(n + m)) stack depth; other passes are iterative.
	 * @complexity O((n + m) log(n + m)) time and O(n + m) extra memory.
	 */
	template<vertex_range RA, vertex_range RB,
			 sign_policy<vertex_coordinate_t<RA>> Policy = exact>
		requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
	[[nodiscard]] std::vector<point<float64>> intersection_of_convex_polygons(RA &&a_points, RB &&b_points,
																			  Policy const p = {}) {
		using T = vertex_coordinate_t<RA>;
		bool const uses_wide_a = detail::check_operation_bounds(a_points);
		bool const uses_wide_b = detail::check_operation_bounds(b_points);
		bool const uses_wide = uses_wide_a || uses_wide_b;
		auto const d = detail::difference_of(a_points, b_points, p);
		std::vector<point<float64>> out;
		containment const overlap = detail::overlap_of(d, p);
		if (overlap == containment::outside) return out;
		if (overlap == containment::boundary) { // measure zero: exact contact
			auto const contact = detail::boundary_contact(d, p);
			if (contact.kind == intersection_kind::at_point) out.push_back(contact.at);
			if (contact.kind == intersection_kind::along_segment) {
				out.push_back(to_float64(contact.along.a));
				out.push_back(to_float64(contact.along.b));
			}
			return out;
		}
		std::vector<point<T>> real_b;
		real_b.reserve(d.negated_b_hull.size());
		for (auto const q : d.negated_b_hull) real_b.push_back(-q);
		// negation is a 180-degree ROTATION: orientation is preserved, so
		// real_b is already CCW (its start vertex is B's lex-max)
		if (d.a_hull.size() < 3 || real_b.size() < 3) { // degenerate summand
			auto const &thin = d.a_hull.size() < 3 ? d.a_hull : real_b;
			auto const &other = d.a_hull.size() < 3 ? real_b : d.a_hull;
			if (thin.size() == 1) return detail::as_float_polygon(thin);
			if (other.size() < 3) { // segment against segment-or-point
				auto const si = intersection(segment<T>{thin[0], thin[1]},
											 segment<T>{other.front(), other.back()}, p);
				if (si.kind == intersection_kind::at_point) out.push_back(si.at);
				if (si.kind == intersection_kind::along_segment) {
					out.push_back(to_float64(si.along.a));
					out.push_back(to_float64(si.along.b));
				}
				return out;
			}
			return detail::segment_clipped_to_hull(segment<T>{thin[0], thin[1]}, other, p, uses_wide);
		}
		std::vector<point<T>> a_rot = d.a_hull, b_rot = real_b;
		detail::rotate_hull_to_low(a_rot);
		detail::rotate_hull_to_low(b_rot);
		auto const active =
			detail::halfplane_intersection(detail::merged_halfplanes(a_rot, b_rot, p), p, uses_wide);
		out.reserve(active.size());
		for (std::size_t i = 0, j = active.size() - 1; i < active.size(); j = i++)
			out.push_back(intersection(active[j].boundary, active[i].boundary));
		return detail::normalized_constructed_polygon(out);
	}

	/**
	 * @brief The union of the convex hulls of two point sets, as CCW
	 *        boundary components of `point<float64>`: ONE component when the
	 *        union is a simple polygon (proper overlap, containment, or
	 *        contact along a shared segment), TWO when the hulls are
	 *        disjoint or touch at a single point (a pinched boundary is not
	 *        a simple polygon; the shared point appears in both components).
	 *        A degenerate hull (point/segment) not fully inside the other is
	 *        returned as its own component (a region with a whisker is not a
	 *        polygon). Integral classification and stitching decisions are exact under exact/fast;
	 *        crossing vertices are rounded constructions.
	 *
	 *        Proper overlap stitches on the half-plane deque: consecutive
	 *        active constraints from DIFFERENT polygons are exactly the
	 *        transversal boundary crossings (with edge tags); the walk
	 *        alternates outside arcs, switching polygons at each crossing.
	 * @param a_points First point set; not modified.
	 * @param b_points Second point set; not modified.
	 * @param p Sign policy.
	 * @return One or two boundary components; collinear touching degenerate hulls merge.
	 * @note Not constexpr (uses runtime deque and construction cleanup).
	 * @tparam RA First vertex range.
	 * @tparam RB Second vertex range with the same coordinate type.
	 * @tparam Policy Sign policy for that coordinate type.
	 * @pre As `intersection_of_convex_polygons`.
	 * @complexity O((n + m) log(n + m)) time and O(n + m) extra memory.
	 */
	template<vertex_range RA, vertex_range RB,
			 sign_policy<vertex_coordinate_t<RA>> Policy = exact>
		requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
	[[nodiscard]] std::vector<std::vector<point<float64>>>
	union_of_convex_polygons(RA &&a_points, RB &&b_points, Policy const p = {}) {
		using T = vertex_coordinate_t<RA>;
		bool const uses_wide_a = detail::check_operation_bounds(a_points);
		bool const uses_wide_b = detail::check_operation_bounds(b_points);
		bool const uses_wide = uses_wide_a || uses_wide_b;
		auto const d = detail::difference_of(a_points, b_points, p);
		std::vector<point<T>> real_b;
		real_b.reserve(d.negated_b_hull.size());
		for (auto const q : d.negated_b_hull) real_b.push_back(-q);
		// negation preserves orientation: real_b is already CCW
		auto const &a_hull = d.a_hull;
		auto const both = [&] {
			return std::vector<std::vector<point<float64>>>{detail::as_float_polygon(a_hull),
															detail::as_float_polygon(real_b)};
		};
		containment const overlap = detail::overlap_of(d, p);
		if (overlap == containment::outside) return both();
		// containment shortcut — also the only polygonal answer possible
		// for a degenerate summand
		auto const covered = [&](std::vector<point<T>> const &inner,
								 std::vector<point<T>> const &outer) {
			if (outer.size() < 3) return false;
			for (auto const q : inner)
				if (convex_contains(outer, q, p) == containment::outside) return false;
			return true;
		};
		if (covered(a_hull, real_b)) return {detail::as_float_polygon(real_b)};
		if (covered(real_b, a_hull)) return {detail::as_float_polygon(a_hull)};
		if (a_hull.size() < 3 && real_b.size() < 3) {
			// two degenerate hulls, touching (overlap != outside here): if
			// every point is collinear the union IS one point/segment —
			// the lex extremes; otherwise (X-crossings, bent chains) it is
			// not representable as a polygon: two components
			std::vector<point<T>> pts(a_hull);
			pts.insert(pts.end(), real_b.begin(), real_b.end());
			point<T> anchor0 = pts[0], anchor1 = pts[0];
			for (auto const q : pts)
				if (!p.equal(q, anchor0)) anchor1 = q;
			bool is_collinear = true;
			for (auto const q : pts)
				is_collinear &= p.equal(anchor0, anchor1)
							 || p.orientation(anchor0, anchor1, q) == 0;
			if (is_collinear) {
				auto const [lo, hi] = std::ranges::minmax(pts);
				std::vector<point<float64>> part{to_float64(lo)};
				if (!p.equal(lo, hi)) part.push_back(to_float64(hi));
				return {part};
			}
			return both();
		}
		if (a_hull.size() < 3 || real_b.size() < 3) return both(); // whisker
		if (overlap == containment::boundary) {
			auto const contact = detail::boundary_contact(d, p);
			if (contact.kind != intersection_kind::along_segment) return both(); // point touch
			// shared-segment touch: A's outside arc + B's outside arc joined
			// at the overlap endpoints, each arc walked CCW in its own hull
			point<T> const lo = contact.along.a, hi = contact.along.b;
			// A's supporting face: the edge containing BOTH contact
			// endpoints (boundary_position(lo) alone can pick the wrong
			// incident edge when lo is a vertex)
			std::size_t e_a = a_hull.size();
			for (std::size_t e = 0, prev = a_hull.size() - 1; e < a_hull.size(); prev = e++) {
				segment<T> const span{a_hull[prev], a_hull[e]};
				if (contains(span, lo, p) && contains(span, hi, p)) {
					e_a = prev;
					break;
				}
			}
			atl_assert(e_a < a_hull.size());
			point<T> const a0 = a_hull[e_a];
			point<T> const a1 = a_hull[e_a + 1 == a_hull.size() ? 0 : e_a + 1];
			int const lo_first = -p.cross_sign(perp(a0), perp(a1), lo, hi); // dot sign
			point<T> const a_end = lo_first > 0 ? hi : lo;					// A traverses lo->hi?
			point<T> const a_start = lo_first > 0 ? lo : hi;				// arc resumes at the far end
			std::vector<point<float64>> boundary;
			detail::append_arc(a_hull, a_end, a_start, p, boundary);
			detail::append_arc(real_b, a_start, a_end, p, boundary);
			return {boundary};
		}
		// proper overlap of two full polygons: around the intersection the
		// active constraints form maximal A-runs and B-runs, so the A|B
		// TRANSITIONS strictly alternate — the union walk is simply the
		// transitions in deque cyclic order, with the exiting polygon's
		// outside arc emitted between consecutive ones. Grazing contacts
		// are zero-length runs and stitch through naturally (a pinch).
		std::vector<point<T>> a_rot = a_hull, b_rot = real_b;
		detail::rotate_hull_to_low(a_rot);
		detail::rotate_hull_to_low(b_rot);
		auto const active =
			detail::halfplane_intersection(detail::merged_halfplanes(a_rot, b_rot, p), p, uses_wide);
		bool is_mixed = false;
		for (auto const &c : active) is_mixed |= c.is_from_second != active.front().is_from_second;
		if (!is_mixed) // single-polygon constraint set: containment near-tie
			return {detail::as_float_polygon(active.front().is_from_second ? b_rot : a_rot)};
		auto const edge_line = [](std::vector<point<T>> const &hull, uint32 const e) {
			return line<T>{hull[e], hull[e + 1 == hull.size() ? 0 : e + 1]};
		};

		struct transition {
			point<float64> at;
			bool a_exits; // the deque switches A -> B here
			uint32 a_edge, b_edge;
		};

		std::vector<transition> transitions;
		for (std::size_t i = 0, j = active.size() - 1; i < active.size(); j = i++) {
			if (active[j].is_from_second == active[i].is_from_second) continue;
			auto const &al = active[j].is_from_second ? active[i] : active[j];
			auto const &bl = active[j].is_from_second ? active[j] : active[i];
			transitions.push_back({intersection(al.boundary, bl.boundary),
								   !active[j].is_from_second, al.edge, bl.edge});
		}

		std::vector<point<float64>> boundary;
		for (std::size_t k = 0; k < transitions.size(); ++k) {
			auto const &from = transitions[k];
			auto const &to = transitions[k + 1 == transitions.size() ? 0 : k + 1];
			atl_assert(from.a_exits != to.a_exits); // strict alternation
			boundary.push_back(from.at);
			// the exiting polygon's outside arc: its vertices strictly
			// between the two transition points
			auto const &hull = from.a_exits ? a_rot : b_rot;
			uint32 const from_edge = from.a_exits ? from.a_edge : from.b_edge;
			uint32 const to_edge = from.a_exits ? to.a_edge : to.b_edge;
			std::size_t count = 0;
			if (from_edge != to_edge)
				count = (std::size_t(to_edge) + hull.size() - from_edge) % hull.size();
			else {
				// same host edge: wraps unless `to` sits ahead of `from`
				line<T> const host = from.a_exits ? edge_line(a_rot, from_edge)
												  : edge_line(b_rot, from_edge);
				line<T> const cut_from = from.a_exits ? edge_line(b_rot, from.b_edge)
													  : edge_line(a_rot, from.a_edge);
				line<T> const cut_to = from.a_exits ? edge_line(b_rot, to.b_edge)
													: edge_line(a_rot, to.a_edge);
				// wrap only when `to` sits STRICTLY behind; equal = pinch, 0
				count = detail::param_before(host, cut_to, cut_from, uses_wide) ? hull.size() : 0;
			}
			for (std::size_t step = 1; step <= count; ++step) {
				point<T> const v = hull[(from_edge + step) % hull.size()];
				boundary.push_back(to_float64(v));
			}
		}
		return {detail::normalized_constructed_polygon(boundary)};
	}

	/**
	 * @brief The pair of points ACHIEVING `distance_between_convex_hulls` — one on
	 *        each hull (boundary points when disjoint), `double` constructions (the closest point
	 *        is usually interior to an edge, so no input-index form can
	 *        exist). Touching hulls give the shared boundary point twice.
	 *        Recovery: the closest point at parameter `t` on a Minkowski
	 *        difference edge splits as `a(t) - b(t)` along the source edges
	 *        (each difference edge advances A's edge, B's edge, or — on
	 *        parallel ties — both, and the same `t` parameterizes each part).
	 * @param a_points First point set; not modified.
	 * @param b_points Second point set; not modified.
	 * @param p Sign policy.
	 * @return One constructed point per hull; ties resolve arbitrarily. Projection,
	 * clamping and minimum-distance selection are floating constructions, even
	 * for integral inputs; only overlap classification uses exact predicates.
	 * @note Not constexpr (uses runtime square roots and intersection cleanup).
	 * @tparam RA First vertex range.
	 * @tparam RB Second vertex range with the same coordinate type.
	 * @tparam Policy Sign policy for that coordinate type.
	 * @pre As `intersection_of_convex_polygons`: both inputs non-empty;
	 *      integral coordinates `|x|, |y| < 2^61` for int64, with the narrower
	 *      storage limits stated there (asserted at entry). Properly overlapping hulls
	 *      return a common point twice, via the intersection machinery.
	 * @complexity O((n + m) log(n + m)) time and O(n + m) extra memory.
	 */
	template<vertex_range RA, vertex_range RB,
			 sign_policy<vertex_coordinate_t<RA>> Policy = exact>
		requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
	[[nodiscard]] std::array<point<float64>, 2> closest_points_between_convex_hulls(RA &&a_points, RB &&b_points,
																					Policy const p = {}) {
		using T = vertex_coordinate_t<RA>;
		detail::check_operation_bounds(a_points);
		detail::check_operation_bounds(b_points);
		auto const d = detail::difference_of(a_points, b_points, p);
		point<T> const origin{};
		if (d.boundary.size() >= 3 && contains(d.boundary, origin, p) == containment::inside) {
			// properly overlapping hulls: any common point, twice — the
			// intersection machinery provides one
			auto const common = intersection_of_convex_polygons(a_points, b_points, p);
			atl_assert(!common.empty());
			return {common.front(), common.front()};
		}
		float64 best = std::numeric_limits<float64>::infinity();
		std::size_t best_from = 0, best_to = 0;
		float64 best_t = 0;
		for (std::size_t i = 0, j = d.boundary.size() - 1; i < d.boundary.size(); j = i++) {
			point<float64> const from = to_float64(d.boundary[j]);
			point<float64> const step = detail::float64_difference(d.boundary[j], d.boundary[i]);
			float64 const len2 = step.x * step.x + step.y * step.y;
			float64 t = len2 == 0 ? 0 : -(from.x * step.x + from.y * step.y) / len2;
			t = t < 0 ? 0 : t > 1 ? 1
								  : t;
			point<float64> const at{from.x + t * step.x, from.y + t * step.y};
			float64 const gap = std::sqrt(at.x * at.x + at.y * at.y);
			if (gap < best) {
				best = gap;
				best_from = j;
				best_to = i;
				best_t = t;
			}
		}
		auto const &[pa_from, qb_from] = d.supports[best_from];
		auto const &[pa_to, qb_to] = d.supports[best_to];
		auto const lerp = [](point<T> const u, point<T> const v, float64 const t) {
			auto const step = detail::float64_difference(u, v);
			return point<float64>{float64(u.x) + t * step.x, float64(u.y) + t * step.y};
		};
		point<float64> const on_a = lerp(d.a_hull[pa_from], d.a_hull[pa_to], best_t);
		point<float64> const negated_on_b =
			lerp(d.negated_b_hull[qb_from], d.negated_b_hull[qb_to], best_t);
		return {on_a, point<float64>{-negated_on_b.x, -negated_on_b.y}};
	}
}
