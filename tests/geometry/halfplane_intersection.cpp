#include <atl/experimental/geometry/halfplane_intersection_variants.hpp>
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/convex_polygon_operations.hpp>
#include <atl/geometry/halfplane.hpp>
#include <atl/geometry/halfplane_intersection.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/detail/int256.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <ranges>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::int128;
using atl::int64;
using atl::uint32;
using atl::geometry::containment;
using atl::geometry::halfplane;
using atl::geometry::halfplane_intersection;
using atl::geometry::halfplane_region;
using atl::geometry::point;
using atl::geometry::region_kind;

namespace {
	bool same_vertices(std::vector<point<double>> const &a, std::vector<point<double>> const &b) {
		if (a.size() != b.size()) return false;
		for (std::size_t i = 0; i < a.size(); ++i) {
			double const scale = 1 + (std::max)(std::abs(b[i].x), std::abs(b[i].y));
			if (!std::isfinite(a[i].x) || !std::isfinite(a[i].y)
				|| !std::isfinite(b[i].x) || !std::isfinite(b[i].y)
				|| !(std::hypot(a[i].x - b[i].x, a[i].y - b[i].y) <= 0x1p-48 * scale)) return false;
		}
		return true;
	}

	bool same_region(halfplane_region const &a, halfplane_region const &b) {
		return a.kind == b.kind && a.boundary == b.boundary && same_vertices(a.vertices, b.vertices);
	}

	using hp = halfplane<int64>;

	/**
	 * @brief The CERTIFYING ORACLE of the half-plane family (library and
	 *        every experimental variant): O(n^3) exact, written independently
	 *        of the library's predicates — plain int128 / int256 algebra and
	 *        its own angular order. Kind from the feasible pairwise crossings
	 *        (and a recession-direction test); boundary from the constraints
	 *        whose line carries a feasible part of positive length.
	 */

	struct rational_point {
		int128 x, y, d; // (x / d, y / d), d > 0
	};

	int128 normal_cross(hp const &a, hp const &b) { return int128(a.p) * b.q - int128(b.p) * a.q; }

	bool is_upper(hp const &a) { return a.p > 0 || (a.p == 0 && a.q < 0); }

	// [0, 2pi) order of the boundary directions (-q, p)
	bool angle_less(hp const &a, hp const &b) {
		if (is_upper(a) != is_upper(b)) return is_upper(a);
		return normal_cross(a, b) > 0;
	}

	int64 scale_of(hp const &h, bool const uses_p) { return uses_p ? (h.p < 0 ? -h.p : h.p) : (h.q < 0 ? -h.q : h.q); }

	// same set: same direction and the same offset relative to the normal
	bool same_set(hp const &a, hp const &b) {
		if (normal_cross(a, b) != 0) return false;
		if (int128(a.p) * b.p + int128(a.q) * b.q <= 0) return false;
		bool const uses_p = a.p != 0;
		return int128(a.r) * scale_of(b, uses_p) == int128(b.r) * scale_of(a, uses_p);
	}

	rational_point crossing(hp const &a, hp const &b) {
		int128 x = int128(a.r) * b.q - int128(b.r) * a.q;
		int128 y = int128(a.p) * b.r - int128(b.p) * a.r;
		int128 d = normal_cross(a, b);
		if (d < 0) {
			x = -x;
			y = -y;
			d = -d;
		}
		return {x, y, d};
	}

	bool satisfies(hp const &k, rational_point const &v) {
		return int128(k.r) * v.d - int128(k.p) * v.x - int128(k.q) * v.y >= 0;
	}

	bool on_line(hp const &k, rational_point const &v) {
		return int128(k.r) * v.d - int128(k.p) * v.x - int128(k.q) * v.y == 0;
	}

	bool same_point(rational_point const &a, rational_point const &b) {
		return atl::detail::compare_products(a.x, b.d, b.x, a.d) == 0
			&& atl::detail::compare_products(a.y, b.d, b.y, a.d) == 0;
	}

	point<double> approx(rational_point const &v) { return {double(v.x) / double(v.d), double(v.y) / double(v.d)}; }

	struct expected_region {
		region_kind kind = region_kind::empty;
		std::vector<uint32> boundary;
		std::vector<rational_point> vertices;
		std::optional<rational_point> single; // the point of a point result
	};

	// the feasible part of constraint i's line, parameterized along its
	// direction from the foot of the perpendicular: positive length?
	// (is_lo_open: unbounded below, is_hi_open: unbounded above)
	struct line_part {
		bool is_edge = false, is_lo_open = true, is_hi_open = true;
	};

	line_part feasible_part(std::vector<hp> const &h, std::size_t const i) {
		hp const &a = h[i];
		int128 const norm = int128(a.p) * a.p + int128(a.q) * a.q;
		bool has_lo = false, has_hi = false;
		int128 lo_n = 0, lo_d = 1, hi_n = 0, hi_d = 1;
		for (std::size_t k = 0; k < h.size(); ++k) {
			if (k == i) continue;
			hp const &c = h[k];
			// t-coefficient of n_c . X(t), X(t) = foot + t (-q_a, p_a)
			int128 const slope = int128(a.p) * c.q - int128(c.p) * a.q;
			int128 const num = int128(c.r) * norm - (int128(c.p) * a.p + int128(c.q) * a.q) * a.r;
			if (slope == 0) {
				if (num < 0) return {}; // the whole line is outside c
				continue;
			}
			int128 const den = slope * norm;
			if (den > 0) { // t <= num / den
				if (!has_hi || atl::detail::compare_products(num, hi_d, hi_n, den) < 0) {
					hi_n = num;
					hi_d = den;
					has_hi = true;
				}
			} else { // t >= num / den = (-num) / (-den)
				if (!has_lo || atl::detail::compare_products(-num, lo_d, lo_n, -den) > 0) {
					lo_n = -num;
					lo_d = -den;
					has_lo = true;
				}
			}
		}
		line_part out;
		out.is_lo_open = !has_lo;
		out.is_hi_open = !has_hi;
		out.is_edge = !has_lo || !has_hi || atl::detail::compare_products(lo_n, hi_d, hi_n, lo_d) < 0;
		return out;
	}

	expected_region oracle(std::vector<hp> const &h) {
		expected_region out;
		std::size_t const n = h.size();
		if (n == 0) {
			out.kind = region_kind::unbounded;
			return out;
		}
		auto const representative = [&](std::size_t const i) {
			for (std::size_t j = 0; j < i; ++j)
				if (same_set(h[j], h[i])) return false;
			return true;
		};
		bool is_all_parallel = true;
		for (std::size_t i = 1; i < n; ++i) is_all_parallel &= normal_cross(h[0], h[i]) == 0;
		std::vector<rational_point> kept;
		for (std::size_t i = 0; i < n; ++i)
			for (std::size_t j = i + 1; j < n; ++j) {
				if (normal_cross(h[i], h[j]) == 0) continue;
				rational_point const v = crossing(h[i], h[j]);
				bool is_feasible = true;
				for (std::size_t k = 0; k < n && is_feasible; ++k) is_feasible = satisfies(h[k], v);
				if (!is_feasible) continue;
				bool is_new = true;
				for (auto const &w : kept) is_new &= !same_point(v, w);
				if (is_new) kept.push_back(v);
			}
		std::vector<std::size_t> edges;
		std::vector<line_part> parts;
		auto const collect_edges = [&] {
			for (std::size_t i = 0; i < n; ++i) {
				if (!representative(i)) continue;
				line_part const part = feasible_part(h, i);
				if (!part.is_edge) continue;
				edges.push_back(i);
				parts.push_back(part);
			}
		};
		auto const by_angle = [&](std::vector<std::size_t> &v) {
			std::ranges::sort(v, [&](std::size_t const a, std::size_t const b) { return angle_less(h[a], h[b]); });
		};
		if (is_all_parallel) {
			collect_edges();
			if (edges.empty()) return out; // the two sides cross over: empty
			out.kind = region_kind::unbounded;
			by_angle(edges);
			for (auto const e : edges) out.boundary.push_back(uint32(e));
			return out;
		}
		if (kept.empty()) return out;
		bool is_unbounded = false;
		for (std::size_t i = 0; i < n && !is_unbounded; ++i) {
			bool is_recession_forward = true, is_recession_backward = true;
			for (std::size_t j = 0; j < n; ++j) {
				int128 const c = normal_cross(h[i], h[j]);
				is_recession_forward &= c <= 0;
				is_recession_backward &= c >= 0;
			}
			is_unbounded = is_recession_forward || is_recession_backward;
		}
		collect_edges();
		if (is_unbounded) {
			out.kind = region_kind::unbounded;
			std::vector<std::size_t> order(edges.size());
			for (std::size_t t = 0; t < order.size(); ++t) order[t] = t;
			std::ranges::sort(order,
							  [&](std::size_t const a, std::size_t const b) { return angle_less(h[edges[a]], h[edges[b]]); });
			std::size_t start = 0;
			for (std::size_t t = 0; t < order.size(); ++t)
				if (parts[order[t]].is_lo_open && !parts[order[t]].is_hi_open) start = t;
			std::ranges::rotate(order, order.begin() + std::ptrdiff_t(start));
			for (auto const t : order) out.boundary.push_back(uint32(edges[t]));
			if (out.boundary.size() == 2 && normal_cross(h[out.boundary[0]], h[out.boundary[1]]) == 0) {
				if (!kept.empty()) out.vertices = {kept[0]}; // a ray's apex
			} else
				for (std::size_t t = 0; t + 1 < out.boundary.size(); ++t)
					out.vertices.push_back(crossing(h[out.boundary[t]], h[out.boundary[t + 1]]));
			return out;
		}
		if (kept.size() == 1) {
			out.kind = region_kind::point;
			out.single = kept[0];
			return out;
		}
		by_angle(edges);
		for (auto const e : edges) out.boundary.push_back(uint32(e));
		if (kept.size() == 2) {
			out.kind = region_kind::segment;
			REQUIRE(edges.size() == 2);
			// order the endpoints along boundary[0]'s direction (-q, p)
			hp const &b0 = h[edges[0]];
			int const order = b0.q != 0
								? atl::detail::compare_products(kept[0].x, kept[1].d, kept[1].x, kept[0].d) * (b0.q < 0 ? 1 : -1)
								: atl::detail::compare_products(kept[0].y, kept[1].d, kept[1].y, kept[0].d) * (b0.p > 0 ? 1 : -1);
			out.vertices = order < 0 ? std::vector{kept[0], kept[1]} : std::vector{kept[1], kept[0]};
			return out;
		}
		out.kind = region_kind::polygon;
		REQUIRE(edges.size() == kept.size());
		for (std::size_t t = 0; t < edges.size(); ++t)
			out.vertices.push_back(crossing(h[edges[t == 0 ? edges.size() - 1 : t - 1]], h[edges[t]]));
		return out;
	}

	bool close(double const v, double const ref) { return std::abs(v - ref) <= 0x1p-48 * (std::max)(1.0, std::abs(ref)); }

	bool close(point<double> const v, rational_point const &ref) {
		point<double> const r = approx(ref);
		return close(v.x, r.x) && close(v.y, r.y);
	}

	void check_against(halfplane_region const &got, expected_region const &want, std::vector<hp> const &h) {
		REQUIRE(int(got.kind) == int(want.kind));
		if (want.kind == region_kind::point) {
			REQUIRE(got.boundary.size() == 2);
			REQUIRE(got.vertices.size() == 1);
			REQUIRE(want.single.has_value());
			CHECK(normal_cross(h[got.boundary[0]], h[got.boundary[1]]) != 0);
			// A point result has an engaged exact witness; keep the check visible to optional-access analysis.
			if (want.single) {
				CHECK(on_line(h[got.boundary[0]], *want.single));
				CHECK(on_line(h[got.boundary[1]], *want.single));
				CHECK(close(got.vertices[0], *want.single));
			}
			return;
		}
		REQUIRE(got.boundary == want.boundary);
		REQUIRE(got.vertices.size() == want.vertices.size());
		for (std::size_t i = 0; i < want.vertices.size(); ++i) CHECK(close(got.vertices[i], want.vertices[i]));
	}

	// each half-plane scaled by its own power of two: the same sets, the
	// same answer, coefficients of very different magnitudes for the filter
	std::vector<halfplane<double>> as_doubles(std::vector<hp> const &h, std::mt19937_64 &rng) {
		std::vector<halfplane<double>> out;
		for (auto const &c : h) {
			double const s = std::ldexp(1.0, int(rng() % 41) - 20) * double(1 + 2 * (rng() % 3));
			out.push_back({double(c.p) * s, double(c.q) * s, double(c.r) * s});
		}
		return out;
	}

	// ---- generators ----

	struct generator {
		std::mt19937_64 rng;
		int64 normal_range, offset_range;

		int64 uniform(int64 const lo, int64 const hi) { return lo + int64(rng() % uint64_t(hi - lo + 1)); }

		std::pair<int64, int64> normal() {
			for (;;) {
				int64 const p = uniform(-normal_range, normal_range), q = uniform(-normal_range, normal_range);
				if (p != 0 || q != 0) return {p, q};
			}
		}

		hp random() {
			auto const [p, q] = normal();
			return {p, q, uniform(-offset_range, offset_range)};
		}

		// through or around a lattice point
		hp around(int64 const x, int64 const y, int64 const slack) {
			auto const [p, q] = normal();
			return {p, q, p * x + q * y + uniform(0, slack)};
		}

		// a random mix of the interesting shapes
		std::vector<hp> instance(std::size_t const n) {
			std::vector<hp> h;
			int const shape = int(rng() % 9);
			int64 const cx = uniform(-5, 5), cy = uniform(-5, 5);
			auto const [up, uq] = normal(); // the half-plane of normals, for unbounded
			for (std::size_t i = 0; i < n; ++i) {
				switch (shape) {
				case 0:
					h.push_back(random());
					break;
				case 1:
					h.push_back(around(cx, cy, offset_range / 4));
					break;
				case 2:
					h.push_back(around(cx, cy, 0));
					break; // all through one point
				case 3:
					h.push_back(around(cx, cy, rng() % 2 == 0 ? 0 : 2));
					break;
				case 4: { // normals in a closed half-plane: unbounded or empty
					hp c = rng() % 2 == 0 ? around(cx, cy, 3) : random();
					if (int128(c.p) * up + int128(c.q) * uq < 0) {
						c.p = -c.p;
						c.q = -c.q;
						c.r = -c.r + uniform(-2, 6);
					}
					h.push_back(c);
					break;
				}
				case 5: { // a thin band: anti-parallel pairs of small width
					hp const c = around(cx, cy, 1);
					h.push_back(c);
					if (rng() % 2 == 0) h.push_back({-c.p, -c.q, -c.r + uniform(-1, 1)});
					break;
				}
				case 6:
					h.push_back(around(cx, cy, rng() % 3 == 0 ? 1 : offset_range / 8));
					break;
				case 7: { // near-tangents of a circle: large polygons, few redundant
					auto const [p, q] = normal();
					double const reach = double(offset_range) * std::sqrt(double(p * p + q * q)) / double(normal_range);
					h.push_back({p, q, int64(std::llround(reach)) + uniform(0, 1)});
					break;
				}
				default:
					h.push_back(rng() % 2 == 0 ? around(cx, cy, 1) : random());
					break;
				}
			}
			// duplicates, scaled copies and anti-parallel partners
			std::size_t const extras = std::size_t(rng() % 4);
			for (std::size_t e = 0; e < extras && !h.empty(); ++e) {
				hp const c = h[std::size_t(rng() % h.size())];
				switch (rng() % 3) {
				case 0:
					h.push_back(c);
					break;
				case 1: {
					int64 const k = uniform(2, 3);
					h.push_back({c.p * k, c.q * k, c.r * k});
					break;
				}
				default:
					h.push_back({-c.p, -c.q, -c.r + uniform(-1, 1)});
					break;
				}
			}
			std::ranges::shuffle(h, rng);
			return h;
		}
	};

	void check_instance(std::vector<hp> const &h, std::mt19937_64 &rng) {
		INFO("n = " << h.size());
		expected_region const want = oracle(h);
		check_against(halfplane_intersection(h), want, h);
		check_against(halfplane_intersection(as_doubles(h, rng)), want, h);
	}
}

TEST_CASE("halfplane: factories, contains, boundary", "[geometry][halfplane]") {
	hp const h = hp::left_of({0, 0}, {2, 1}); // interior: left of (0,0) -> (2,1)
	CHECK(h == hp{1, -2, 0});
	CHECK(h.contains({0, 1}) == containment::inside);
	CHECK(h.contains({4, 2}) == containment::boundary);
	CHECK(h.contains({1, 0}) == containment::outside);
	auto const l = h.boundary();
	// the boundary line runs along (2, 1) with the interior on its left
	CHECK(atl::geometry::side_of(l, point<double>{0, 1}) == atl::geometry::side::left);
	CHECK(std::abs((l.b.x - l.a.x) * 1 - (l.b.y - l.a.y) * 2) < 1e-12);

	auto const full_range_line = hp{(std::numeric_limits<int64>::min)(), 0, 1}.boundary();
	CHECK(full_range_line.a != full_range_line.b);
	hp const m = hp::at_most(3, 4, 5);
	CHECK(m.contains({-1, 2}) == containment::boundary);
	CHECK(m.contains({0, 0}) == containment::inside);
	// Residual evaluation supports points beyond the intersection-coordinate bound.
	CHECK(m.contains({int64(1) << 62, 0}) == containment::outside);

	// floating: exact with respect to the stored coefficients
	using fh = halfplane<double>;
	fh const third = fh::at_most(3, 0, 1);
	double const x = 1.0 / 3.0; // below 1/3: 3x < 1 exactly, though fl(3x) == 1
	CHECK(third.contains({x, 0}) == containment::inside);
	CHECK(third.contains({std::nextafter(x, 1.0), 0}) == containment::outside);
	fh const diagonal = fh::at_most(1, 1, 1);
	CHECK(diagonal.contains({0.375, 0.625}) == containment::boundary);
	CHECK(diagonal.contains({0.375, std::nextafter(0.625, 1.0)}) == containment::outside);
	fh const lf = fh::left_of({0.0, 0.0}, {2.0, 1.0});
	CHECK(lf == fh{1.0, -2.0, 0.0});

	// random lattice points against the plain formula
	std::mt19937_64 rng(7);
	for (int iter = 0; iter < 20000; ++iter) {
		auto const coordinate = [&] { return int64(rng() % 2001) - 1000; };
		point<int64> const a{coordinate(), coordinate()}, b{coordinate(), coordinate()}, v{coordinate(), coordinate()};
		if (a == b) continue;
		hp const c = hp::left_of(a, b);
		int128 const cross = int128(b.x - a.x) * (v.y - a.y) - int128(b.y - a.y) * (v.x - a.x);
		CHECK(int(c.contains(v)) == (cross > 0) - (cross < 0));
		halfplane<double> const cd = halfplane<double>::left_of({double(a.x), double(a.y)}, {double(b.x), double(b.y)});
		CHECK(int(cd.contains({double(v.x), double(v.y)})) == (cross > 0) - (cross < 0));
	}
}

TEST_CASE("halfplane: asserted bounds", "[geometry][halfplane]") {
	constexpr int64 big = int64(1) << 31, huge = int64(1) << 62, point_bound = int64(1) << 30;
	CHECK(atl_test::aborts([] { (void)hp::at_most(0, 0, 1); }));
	CHECK(atl_test::aborts([&] { (void)hp::at_most(big, 0, 0); }));
	CHECK(atl_test::aborts([&] { (void)hp::at_most(0, -big, 0); }));
	CHECK(atl_test::aborts([&] { (void)hp::at_most(1, 0, huge); }));
	CHECK(atl_test::aborts([&] { (void)hp::at_most(1, 0, -huge); }));
	CHECK(hp::at_most(big - 1, -(big - 1), huge - 1).r == huge - 1);
	CHECK(atl_test::aborts([] { (void)hp::left_of({1, 1}, {1, 1}); }));
	CHECK(atl_test::aborts([&] { (void)hp::left_of({point_bound, 0}, {0, 0}); }));
	CHECK(hp::left_of({point_bound - 1, -(point_bound - 1)}, {-(point_bound - 1), point_bound - 1}).r == 0);
	CHECK(atl_test::aborts([] {
		double const inf = std::numeric_limits<double>::infinity();
		(void)halfplane<double>::at_most(1, 0, inf);
	}));
	// an aggregate filled directly is checked at the algorithm's entry
	CHECK(atl_test::aborts([&] { (void)halfplane_intersection(std::vector<hp>{{big, 0, 0}}); }));
	CHECK(atl_test::aborts([] { (void)halfplane_intersection(std::vector<hp>{{0, 0, 0}}); }));
}

TEST_CASE("halfplane_intersection: handcrafted shapes", "[geometry][halfplane_intersection]") {
	std::mt19937_64 rng(11);
	auto const kind_of = [](std::vector<hp> const &h) { return halfplane_intersection(h).kind; };
	std::vector<std::vector<hp>> const zoo = {
		{},																	  // the plane
		{{1, 0, 1}},														  // a half-plane
		{{1, 0, 1}, {-1, 0, 1}},											  // a strip
		{{1, 0, 1}, {-1, 0, -1}},											  // a line
		{{1, 0, 1}, {-1, 0, -2}},											  // empty strip
		{{1, 0, 0}, {0, 1, 0}},												  // a wedge
		{{0, 1, 0}, {0, -1, 0}, {-1, 0, 0}},								  // a ray from the origin along +x
		{{0, 1, 0}, {0, -1, 0}, {-1, 0, 0}, {-1, 1, 0}, {-2, -3, 0}},		  // a ray, concurrent cuts
		{{0, 1, 1}, {0, -1, 1}, {-1, 0, 0}},								  // a half-strip
		{{0, 1, 1}, {0, -1, -1}, {-1, 0, 0}, {1, 0, 2}},					  // a segment
		{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}},						  // a point
		{{1, 1, 0}, {-1, 1, 0}, {0, -1, 0}},								  // a point, three lines
		{{1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1}},						  // a square
		{{1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1}, {1, 1, 2}},			  // + a zero-length diagonal
		{{100, 0, 100}, {0, 100, 100}, {1, 1, 2}, {-1, 0, 0}, {0, -1, 0}},	  // diagonal cuts at the shrink
		{{1, 0, 1}, {1, 0, 1}, {2, 0, 2}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1}}, // duplicates
		{{1, 0, -1}, {-1, 0, -1}, {0, 1, 1}, {0, -1, 1}},					  // empty, all bounded
		{{1, 0, 0}, {0, 1, 0}, {-1, -1, -1}},								  // empty triangle
		{{1, 0, 0}, {0, 1, 0}, {-1, -1, 0}},								  // a point corner
		{{1, 0, 0}, {0, 1, 0}, {-1, 2, 0}, {-1, -1, 0}, {2, -1, 0}},		  // concurrent, perturbation tie
		{{0, -1, 0}, {1, 0, 0}, {0, 1, 1}},									  // half-strip open to -x
	};
	std::vector<region_kind> const kinds = {
		region_kind::unbounded,
		region_kind::unbounded,
		region_kind::unbounded,
		region_kind::unbounded,
		region_kind::empty,
		region_kind::unbounded,
		region_kind::unbounded,
		region_kind::unbounded,
		region_kind::unbounded,
		region_kind::segment,
		region_kind::point,
		region_kind::point,
		region_kind::polygon,
		region_kind::polygon,
		region_kind::polygon,
		region_kind::polygon,
		region_kind::empty,
		region_kind::empty,
		region_kind::point,
		region_kind::point,
		region_kind::unbounded,
	};
	REQUIRE(zoo.size() == kinds.size());
	for (std::size_t i = 0; i < zoo.size(); ++i) {
		INFO("zoo case " << i);
		CHECK(int(kind_of(zoo[i])) == int(kinds[i]));
		check_instance(zoo[i], rng);
	}
	// layouts the oracle does not spell out
	auto const ray = halfplane_intersection(zoo[6]);
	CHECK(ray.boundary == std::vector<uint32>{0, 1});
	REQUIRE(ray.vertices.size() == 1);
	CHECK(ray.vertices[0] == point<double>{0, 0});
	auto const square = halfplane_intersection(zoo[13]);
	CHECK(square.boundary == std::vector<uint32>{3, 0, 2, 1}); // starts at the +x direction
	CHECK(same_vertices(square.vertices, std::vector<point<double>>{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}));
	CHECK(halfplane_intersection(zoo[15]).boundary == std::vector<uint32>{5, 0, 4, 3});
	auto const segment = halfplane_intersection(zoo[9]);
	CHECK(segment.boundary == std::vector<uint32>{1, 0});
	CHECK(same_vertices(segment.vertices, std::vector<point<double>>{{0, 1}, {2, 1}}));
}

TEST_CASE("halfplane_intersection: random instances vs the O(n^3) oracle", "[geometry][halfplane_intersection]") {
	std::mt19937_64 rng(2026);
	for (int round = 0; round < 3; ++round) {
		int64 const normal_range = round == 0 ? 2 : round == 1 ? 6
															   : 40;
		int64 const offset_range = round == 0 ? 6 : round == 1 ? 40
															   : 1000;
		generator g{std::mt19937_64(uint64_t(round) + 1), normal_range, offset_range};
		for (int iter = 0; iter < 6000; ++iter) {
			std::size_t const n = 1 + std::size_t(g.rng() % (iter % 10 == 0 ? 40 : 10));
			check_instance(g.instance(n), rng);
		}
	}
}

TEST_CASE("halfplane_intersection: coefficients at the bounds", "[geometry][halfplane_intersection]") {
	std::mt19937_64 rng(5);
	constexpr int64 normal_max = (int64(1) << 31) - 1, offset_max = (int64(1) << 62) - 1;
	auto const uniform = [&](int64 const lo, int64 const hi) { return lo + int64(rng() % uint64_t(hi - lo + 1)); };
	for (int iter = 0; iter < 3000; ++iter) {
		std::vector<hp> h;
		std::size_t const n = 1 + std::size_t(rng() % 12);
		int const shape = int(rng() % 3);
		for (std::size_t i = 0; i < n; ++i) {
			int64 p = uniform(-normal_max, normal_max), q = uniform(-normal_max, normal_max);
			if (shape == 1) { // few distinct directions: parallels at extreme offsets
				p = rng() % 2 == 0 ? normal_max : -normal_max;
				q = rng() % 3 == 0 ? 0 : (rng() % 2 == 0 ? normal_max : -normal_max);
			}
			int64 r = uniform(-offset_max, offset_max);
			if (shape == 2) r = uniform(offset_max / 2, offset_max); // contains a big box around the origin
			if (p == 0 && q == 0) p = 1;
			h.push_back(hp::at_most(p, q, r));
		}
		INFO("iteration " << iter);
		auto const got = halfplane_intersection(h);
		check_against(got, oracle(h), h);
		// the cached corners' int128 products at the bounds
		auto const cached = atl::experimental::halfplane_intersection_by_composition<
			atl::geometry::detail::filtered_determinant, atl::experimental::detail::corner_cached_pass,
			atl::experimental::halfplane_sort::key>(h);
		CHECK(same_region(cached, got));
	}
	// left_of at the point bound: tangents of a big circle
	constexpr double radius = double((int64(1) << 30) - 2);
	std::vector<point<int64>> ring;
	for (int i = 0; i < 64; ++i) {
		double const a = 6.283185307179586 * i / 64;
		ring.push_back({int64(std::llround(radius * std::cos(a))), int64(std::llround(radius * std::sin(a)))});
	}
	std::vector<hp> h;
	for (std::size_t i = 0; i < ring.size(); ++i) h.push_back(hp::left_of(ring[i], ring[(i + 1) % ring.size()]));
	auto const got = halfplane_intersection(h);
	check_against(got, oracle(h), h);
	CHECK(got.kind == region_kind::polygon);
	CHECK(got.boundary.size() == 64);
}

TEST_CASE("halfplane_intersection: matches intersection_of_convex_polygons", "[geometry][halfplane_intersection]") {
	std::mt19937_64 rng(99);
	for (int iter = 0; iter < 4000; ++iter) {
		int64 const span = iter % 2 == 0 ? 12 : (int64(1) << 29);
		auto const cloud = [&](int64 const cx, int64 const cy) {
			std::vector<point<int64>> pts(3 + std::size_t(rng() % 12));
			for (auto &v : pts)
				v = {cx + int64(rng() % uint64_t(span)) - span / 2, cy + int64(rng() % uint64_t(span)) - span / 2};
			return pts;
		};
		int64 const shift = iter % 3 == 0 ? span : span / 3;
		auto const a = cloud(0, 0), b = cloud(shift, iter % 5 == 0 ? 0 : shift / 2);
		auto const ha = atl::geometry::convex_hull(a), hb = atl::geometry::convex_hull(b);
		if (ha.size() < 3 || hb.size() < 3) continue;
		std::vector<hp> h;
		for (auto const *hull : {&ha, &hb})
			for (std::size_t i = 0; i < hull->size(); ++i)
				h.push_back(hp::left_of((*hull)[i], (*hull)[(i + 1) % hull->size()]));
		auto const got = halfplane_intersection(h);
		auto const reference = atl::geometry::intersection_of_convex_polygons(a, b);
		INFO("iteration " << iter);
		std::size_t const count = got.kind == region_kind::empty   ? 0
								: got.kind == region_kind::point   ? 1
								: got.kind == region_kind::segment ? 2
																   : got.vertices.size();
		REQUIRE(got.kind != region_kind::unbounded);
		REQUIRE(count == reference.size());
		double const tolerance = 0x1p-30 * double(span);
		for (auto const v : reference) {
			bool is_matched = false;
			for (auto const w : got.vertices) is_matched |= std::hypot(v.x - w.x, v.y - w.y) <= tolerance;
			CHECK(is_matched);
		}
	}
}

TEST_CASE("halfplane_intersection: int32 coordinates and other ranges", "[geometry][halfplane_intersection]") {
	using h32 = halfplane<atl::int32>;
	std::vector<h32> const square = {h32::left_of({0, 0}, {2, 0}), h32::left_of({2, 0}, {2, 2}),
									 h32::left_of({2, 2}, {0, 2}), h32::left_of({0, 2}, {0, 0})};
	auto const got = halfplane_intersection(square);
	CHECK(got.kind == region_kind::polygon);
	CHECK(got.boundary == std::vector<uint32>{0, 1, 2, 3});
	CHECK(same_vertices(got.vertices, std::vector<point<double>>{{0, 0}, {2, 0}, {2, 2}, {0, 2}}));
	// any sized random-access range
	std::array<hp, 2> const wedge = {hp{1, 0, 0}, hp{0, 1, 0}};
	CHECK(halfplane_intersection(wedge).kind == region_kind::unbounded);
}

TEST_CASE("halfplane_intersection: near-identical directions share sort keys", "[geometry][halfplane_intersection]") {
	// directions (a, a + 1) for a near 2^30 differ by about 2^-61 radians:
	// their rounded slopes collide, and the exact comparator settles them
	std::mt19937_64 rng(3);
	constexpr int64 base = int64(1) << 30;
	for (int iter = 0; iter < 200; ++iter) {
		std::vector<hp> h;
		for (int i = 0; i < 12; ++i) {
			int64 const a = base + int64(rng() % 64);
			int64 const sign = rng() % 2 == 0 ? 1 : -1;
			h.push_back(hp::at_most(sign * a, sign * (a + 1), int64(rng() % (uint64_t(1) << 61))));
		}
		h.push_back(hp::at_most(1, -1, int64(1) << 40));
		h.push_back(hp::at_most(-1, 1, int64(1) << 40));
		std::ranges::shuffle(h, rng);
		check_against(halfplane_intersection(h), oracle(h), h);
	}
}

TEST_CASE("halfplane_intersection: every variant agrees with the library composition",
		  "[geometry][halfplane_intersection]") {
	namespace ex = atl::experimental;
	namespace gd = atl::geometry::detail;
	using exd = ex::detail::always_exact_determinant;
	std::mt19937_64 rng(17);
	auto const same = [](halfplane_region const &a, halfplane_region const &b) { return same_region(a, b); };
	for (int iter = 0; iter < 6000; ++iter) {
		int64 const normal_range = iter % 3 == 0 ? 3 : 30, offset_range = iter % 3 == 0 ? 10 : 500;
		generator g{std::mt19937_64(uint64_t(iter) + 1000), normal_range, offset_range};
		auto const h = g.instance(1 + std::size_t(g.rng() % 30));
		auto const hd = as_doubles(h, rng);
		auto const library = halfplane_intersection(h);
		auto const library_double = halfplane_intersection(hd);
		check_against(library, oracle(h), h);
		check_against(library_double, oracle(h), h);
		INFO("iteration " << iter);
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::key>(h), library));
		CHECK(same(ex::halfplane_intersection_by_composition<exd, gd::deque_pass, ex::halfplane_sort::key>(hd), library_double));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, ex::detail::envelope_pass, ex::halfplane_sort::key>(h), library));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, ex::detail::envelope_pass, ex::halfplane_sort::key>(hd), library_double));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, ex::detail::deque_pass_testing_every_front, ex::halfplane_sort::key>(h), library));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::comparator>(hd), library_double));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::key_std_sort>(h), library));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::key_radix_16>(hd), library_double));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::key_64_std_sort>(h), library));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::key_64_radix_16>(hd), library_double));
		CHECK(same(ex::halfplane_intersection_by_composition<ex::detail::product_filter_determinant, gd::deque_pass, ex::halfplane_sort::key>(hd), library_double));
		CHECK(same(ex::halfplane_intersection_by_composition<ex::detail::dekker_determinant, gd::deque_pass, ex::halfplane_sort::key>(hd), library_double));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::key, true>(h), library));
		// coefficients this small make even the rounded determinant exact
		CHECK(same(ex::halfplane_intersection_by_composition<ex::detail::rounded_determinant, gd::deque_pass, ex::halfplane_sort::key>(hd), library_double));
		// Corner-cache losers are checked against the library composition.
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, ex::detail::corner_cached_pass, ex::halfplane_sort::key>(h), library));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, ex::detail::corner_cached_pass, ex::halfplane_sort::key>(hd), library_double));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::rounded_determinant, ex::detail::corner_cached_pass, ex::halfplane_sort::key>(h), library));
		auto runs = h;
		std::ranges::sort(runs, atl::geometry::halfplane_sweep_less{});
		std::size_t const pieces = 1 + std::size_t(g.rng() % 12); // more than 8 runs takes the fallback sort
		for (std::size_t k = 0; k < pieces; ++k) {
			std::size_t const from = std::size_t(g.rng() % (runs.size() + 1)), to = from + std::size_t(g.rng() % (runs.size() - from + 1));
			std::ranges::rotate(runs.begin() + std::ptrdiff_t(from), runs.begin() + std::ptrdiff_t(from + (to - from) / 2),
								runs.begin() + std::ptrdiff_t(to));
		}
		auto const library_runs = halfplane_intersection(runs); // the boundary indices are positions in runs
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, ex::detail::corner_cached_pass, ex::halfplane_sort::rotated_runs>(runs), library_runs));
	}
}

TEST_CASE("halfplane_intersection: large inputs take the radix sort", "[geometry][halfplane_intersection]") {
	// above 2048 constraints the library sort is a radix sort on the
	// monotone key: checked against the comparator sort, which shares no
	// code with it but the exact comparator
	namespace ex = atl::experimental;
	namespace gd = atl::geometry::detail;
	std::mt19937_64 rng(23);
	auto const same = [](halfplane_region const &a, halfplane_region const &b) { return same_region(a, b); };
	for (int iter = 0; iter < 40; ++iter) {
		int64 const normal_range = iter % 2 == 0 ? 1000 : (int64(1) << 29);
		int64 const offset_range = iter % 2 == 0 ? 1'000'000 : (int64(1) << 59);
		generator g{std::mt19937_64(uint64_t(iter) + 77), normal_range, offset_range};
		auto h = g.instance(2048 + std::size_t(g.rng() % 6000));
		if (iter % 4 == 1) // pre-sorted input skips the sort
			std::ranges::sort(h, atl::geometry::halfplane_sweep_less{});
		if (iter % 4 == 3) { // one rotated sorted run, as from a polygon's edges
			std::ranges::sort(h, atl::geometry::halfplane_sweep_less{});
			std::ranges::rotate(h, h.begin() + std::ptrdiff_t(g.rng() % h.size()));
		}
		auto const reference =
			ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::comparator>(h);
		INFO("iteration " << iter << ", n = " << h.size());
		CHECK(same(halfplane_intersection(h), reference));
		CHECK(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, ex::detail::corner_cached_pass, ex::halfplane_sort::key>(h), reference));
		auto const hd = as_doubles(h, rng);
		CHECK(same(halfplane_intersection(hd),
				   ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::comparator>(hd)));
	}
}

TEST_CASE("halfplane_intersection: the docs example", "[geometry][halfplane_intersection]") {
	using atl::geometry::halfplane;
	std::vector<halfplane<long long>> h = {
		halfplane<long long>::left_of({0, 0}, {4, 0}), // y >= 0
		halfplane<long long>::left_of({4, 0}, {0, 4}), // x + y <= 4
		halfplane<long long>::at_most(-1, 0, 0),	   // x >= 0
	};
	auto region = atl::geometry::halfplane_intersection(h);
	CHECK(region.kind == region_kind::polygon);
	CHECK(region.boundary == std::vector<uint32>{0, 1, 2});
	CHECK(same_vertices(region.vertices, std::vector<point<double>>{{0, 0}, {4, 0}, {0, 4}}));
}

TEST_CASE("halfplane: the free contains under each policy", "[geometry][halfplane]") {
	using atl::geometry::approx;
	using atl::geometry::exact;
	using atl::geometry::fast;
	std::mt19937_64 rng(31);
	for (int iter = 0; iter < 20000; ++iter) {
		auto const coordinate = [&] { return int64(rng() % 2001) - 1000; };
		point<int64> const a{coordinate(), coordinate()}, b{coordinate(), coordinate()}, v{coordinate(), coordinate()};
		if (a == b) continue;
		hp const c = hp::left_of(a, b);
		CHECK(contains(c, v) == c.contains(v));
		CHECK(contains(c, v, exact{}) == c.contains(v));
		CHECK(contains(c, v, fast{}) == c.contains(v)); // integral: exact either way
		// small integers: the rounded double residual is exact too
		halfplane<double> const cd = halfplane<double>::left_of({double(a.x), double(a.y)}, {double(b.x), double(b.y)});
		point<double> const vd{double(v.x), double(v.y)};
		CHECK(contains(cd, vd, fast{}) == c.contains(v));
		CHECK(contains(cd, vd) == c.contains(v));
	}
	// where rounding loses it, exact still decides: 3x < 1 exactly though fl(3x) == 1
	halfplane<double> const third = halfplane<double>::at_most(3, 0, 1);
	point<double> const just_below{1.0 / 3.0, 0};
	CHECK(contains(third, just_below) == containment::inside);
	CHECK(contains(third, point<double>{0.3, 0}, approx{0.5}) == containment::boundary);
}

TEST_CASE("halfplane_intersection: fast agrees with exact on well-conditioned input",
		  "[geometry][halfplane_intersection]") {
	using atl::geometry::exact;
	using atl::geometry::fast;
	std::mt19937_64 rng(41);
	std::uniform_real_distribution<double> unit(-1.0, 1.0), offset(0.05, 1.0);
	auto const same = [](halfplane_region const &a, halfplane_region const &b) { return same_region(a, b); };
	for (int iter = 0; iter < 3000; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % (iter % 10 == 0 ? 200 : 20));
		std::vector<halfplane<double>> h;
		for (std::size_t i = 0; i < n; ++i) {
			double p = 0, q = 0;
			while (std::abs(p) + std::abs(q) < 0.1) p = unit(rng), q = unit(rng);
			// a third of the instances move some offsets past the origin (empty or thin regions)
			double const r = iter % 3 == 0 && rng() % 4 == 0 ? -offset(rng) : offset(rng);
			h.push_back({p, q, r});
		}
		INFO("iteration " << iter << ", n = " << n);
		auto const want = halfplane_intersection(h, exact{});
		CHECK(same(halfplane_intersection(h), want));
		CHECK(same(halfplane_intersection(h, fast{}), want));
	}
}

TEST_CASE("halfplane_intersection: fast never aborts on near-degenerate input",
		  "[geometry][halfplane_intersection]") {
	using atl::geometry::fast;
	using atl::geometry::region_kind;
	// boundaries through (1/3, 1/7) up to a rounding of their offsets: the
	// rounded signs misclassify such triples, and the debug checks (exact
	// signs only) must not take that for a bug
	std::mt19937_64 rng(43);
	std::uniform_real_distribution<double> angle(0.0, 6.283185307179586), wobble(-1e-15, 1e-15);
	for (int iter = 0; iter < 2000; ++iter) {
		std::size_t const n = 3 + std::size_t(rng() % 40);
		std::vector<halfplane<double>> h;
		for (std::size_t i = 0; i < n; ++i) {
			double const a = angle(rng), p = std::cos(a), q = std::sin(a);
			h.push_back({p, q, p / 3 + q / 7 + (iter % 2 == 0 ? wobble(rng) : 0.0)});
		}
		INFO("iteration " << iter << ", n = " << n);
		auto const got = halfplane_intersection(h, fast{});
		CHECK(got.boundary.size() <= n);
		if (got.kind == region_kind::point) CHECK(got.boundary.size() == 2);
	}
}

TEST_CASE("halfplane_intersection: explicit policies on the degenerate shapes",
		  "[geometry][halfplane_intersection]") {
	using atl::geometry::exact;
	using atl::geometry::fast;
	std::mt19937_64 rng(43);
	auto const same = [](halfplane_region const &a, halfplane_region const &b) { return same_region(a, b); };
	std::vector<std::vector<hp>> const zoo = {
		{{1, 0, 1}, {-1, 0, -1}},									  // a line
		{{0, 1, 0}, {0, -1, 0}, {-1, 0, 0}, {-1, 1, 0}, {-2, -3, 0}}, // a ray, concurrent cuts
		{{0, 1, 1}, {0, -1, -1}, {-1, 0, 0}, {1, 0, 2}},			  // a segment
		{{1, 1, 0}, {-1, 1, 0}, {0, -1, 0}},						  // a point, three lines
		{{1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1}, {1, 1, 2}},	  // a zero-length diagonal
		{{1, 0, 0}, {0, 1, 0}, {-1, 2, 0}, {-1, -1, 0}, {2, -1, 0}},  // concurrent, perturbation tie
		{{1, 0, 0}, {0, 1, 0}, {-1, -1, -1}},						  // empty triangle
	};
	for (std::size_t i = 0; i < zoo.size(); ++i) {
		INFO("zoo case " << i);
		auto const want = oracle(zoo[i]);
		check_against(halfplane_intersection(zoo[i], exact{}), want, zoo[i]);
		check_against(halfplane_intersection(zoo[i], fast{}), want, zoo[i]); // integral: exact either way
		auto const hd = as_doubles(zoo[i], rng);
		check_against(halfplane_intersection(hd, exact{}), want, zoo[i]);
		// coefficients this small make even the rounded determinant exact
		CHECK(same(halfplane_intersection(hd, fast{}), halfplane_intersection(hd)));
	}
}

TEST_CASE("halfplane_sweep_less: the sweep order, reusable across calls", "[geometry][halfplane_intersection]") {
	using atl::geometry::halfplane_sweep_less;
	std::mt19937_64 rng(47);
	for (int iter = 0; iter < 2000; ++iter) {
		generator g{std::mt19937_64(uint64_t(iter) + 3000), 6, 40};
		auto h = g.instance(1 + std::size_t(g.rng() % 30));
		std::ranges::sort(h, halfplane_sweep_less{});
		INFO("iteration " << iter);
		// the oracle's independent angle order agrees, ties being parallel
		for (std::size_t i = 0; i + 1 < h.size(); ++i) {
			CHECK_FALSE(angle_less(h[i + 1], h[i]));
			CHECK(halfplane_sweep_less{}(h[i], h[i + 1]) == angle_less(h[i], h[i + 1]));
		}
		check_instance(h, rng);
	}
	// presorted input of the radix-sort size takes the skip path
	generator g{std::mt19937_64(5), 1000, 1'000'000};
	auto big = g.instance(5000);
	auto const shuffled = halfplane_intersection(big);
	std::ranges::sort(big, halfplane_sweep_less{});
	auto const sorted = halfplane_intersection(big);
	CHECK(sorted.kind == shuffled.kind);
	CHECK(sorted.vertices.size() == shuffled.vertices.size());
}

TEST_CASE("halfplane_sweep_less: the inscribed-circle recipe", "[geometry][halfplane_intersection]") {
	// the right triangle (0, 0), (4, 0), (0, 3) has inradius (3 + 4 - 5) / 2 = 1
	using fh = halfplane<double>;
	std::vector<fh> h = {fh::at_most(3, 4, 12), fh::at_most(-1, 0, 0), fh::at_most(0, -1, 0)};
	std::ranges::sort(h, atl::geometry::halfplane_sweep_less{}); // once
	double lo = 0, hi = 10;
	std::vector<fh> shifted = h;
	for (int step = 0; step < 100; ++step) {
		double const rho = (lo + hi) / 2;
		for (std::size_t i = 0; i < h.size(); ++i)
			shifted[i] = {h[i].p, h[i].q, h[i].r - rho * std::hypot(h[i].p, h[i].q)};
		(halfplane_intersection(shifted).kind == region_kind::empty ? hi : lo) = rho;
	}
	CHECK(std::abs(lo - 1) < 1e-12);
}

TEST_CASE("halfplane_intersection: measured run dispatch boundaries", "[geometry][halfplane_intersection]") {
	namespace ex = atl::experimental;
	namespace gd = atl::geometry::detail;
	auto const same = [](halfplane_region const &a, halfplane_region const &b) { return same_region(a, b); };
	std::vector<hp> directions = {{1, 0, 10}, {1, 1, 10}, {0, 1, 10}, {-1, 1, 10}, {-1, 0, 10}, {-1, -1, 10}, {0, -1, 10}, {1, -1, 10}};
	std::ranges::sort(directions, atl::geometry::halfplane_sweep_less{});
	for (std::size_t const n : {std::size_t(7), std::size_t(2'047), std::size_t(2'048), std::size_t(200'000)}) {
		for (std::size_t const runs : {1, 2, 3, 7, 8, 9}) {
			if (n != 200'000 && runs != 2) continue;
			CAPTURE(n, runs);
			std::vector<hp> h;
			h.reserve(n);
			for (std::size_t run = 0; run < runs; ++run) {
				std::size_t const count = n / runs + (run < n % runs);
				for (std::size_t i = 0; i < count; ++i) h.push_back(directions[i * directions.size() / count]);
			}
			auto const reference = ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::comparator>(h);
			REQUIRE(same(halfplane_intersection(h), reference));
			REQUIRE(same(ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::key>(h), reference));
			if (runs == 1 || runs == 2) {
				std::ranges::sort(h, atl::geometry::halfplane_sweep_less{});
				std::ranges::rotate(h, h.begin() + std::ptrdiff_t(n / 3));
				auto const rotated_reference = ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::comparator>(h);
				REQUIRE(same(halfplane_intersection(h), rotated_reference));
			}
		}
	}
}

TEST_CASE("halfplane: floating construction bound and scaled exact coefficients", "[geometry][halfplane]") {
	constexpr int64 m = (int64(1) << 26) - 1;
	std::array const points{point<int64>{-m, -m}, point<int64>{m, m - 1}, point<int64>{m - 1, m}};
	for (std::size_t i = 0; i < points.size(); ++i) {
		auto const a = points[i], b = points[(i + 1) % points.size()];
		auto const integral = hp::left_of(a, b);
		auto const floating = halfplane<double>::left_of({double(a.x), double(a.y)}, {double(b.x), double(b.y)});
		double const p = double(integral.p), q = double(integral.q), r = double(integral.r);
		CHECK(floating.p == p);
		CHECK(floating.q == q);
		CHECK(floating.r == r);
	}
	std::vector<hp> const exact_box{{1, 0, 3}, {-1, 0, 2}, {0, 1, 5}, {0, -1, 1}, {1, 1, 6}};
	auto const want = oracle(exact_box);
	for (int const exponent : {-300, 0, 300}) {
		double const scale = std::ldexp(1.0, exponent);
		std::vector<halfplane<double>> scaled;
		for (auto const &h : exact_box) scaled.push_back({double(h.p) * scale, double(h.q) * scale, double(h.r) * scale});
		check_against(halfplane_intersection(scaled), want, exact_box);
		CHECK(contains(scaled.front(), point{3.0, 0.0}) == containment::boundary);
		CHECK(contains(scaled.front(), point{std::nextafter(3.0, 4.0), 0.0}) == containment::outside);
		CHECK(contains(scaled.front(), point{3.0 + 0x1p-30, 0.0}, atl::geometry::approx{scale * 0x1p-29}) == containment::boundary);
	}
}

TEST_CASE("half-plane oracle: exact segment direction despite binary64 collapse", "[geometry][halfplane]") {
	int64 const k = int64(1) << 60;
	std::vector<hp> const h{{1, 1, 2 * k}, {-1, -1, -2 * k}, {1, 0, k}, {-1, 0, 1 - k}};
	auto const want = oracle(h);
	REQUIRE(want.vertices.size() == 2);
	CHECK(want.vertices[0].x == int128(k) * want.vertices[0].d);
	CHECK(want.vertices[0].y == int128(k) * want.vertices[0].d);
	CHECK(want.vertices[1].x == int128(k - 1) * want.vertices[1].d);
	CHECK(want.vertices[1].y == int128(k + 1) * want.vertices[1].d);
	check_against(halfplane_intersection(h), want, h);
}

TEST_CASE("half-plane equivalence rejects nonfinite vertices", "[geometry][halfplane]") {
	double const nan = std::numeric_limits<double>::quiet_NaN();
	double const inf = std::numeric_limits<double>::infinity();
	CHECK_FALSE(same_vertices({{nan, nan}}, {{0, 0}}));
	CHECK_FALSE(same_vertices({{0, 0}}, {{nan, nan}}));
	CHECK_FALSE(same_vertices({{inf, 0}}, {{inf, 0}}));
}

TEST_CASE("half-plane containment asserts finite expansion envelope", "[geometry][halfplane]") {
	using hd = halfplane<double>;
	CHECK(atl_test::aborts([] { (void)hd::at_most(1, 1, 0x1p1023).contains({0x1p1023, 0x1p1023}); }));
	CHECK(atl_test::aborts([] { (void)atl::geometry::contains(hd::at_most(1, 1, 0x1p1023), point<double>{0x1p1023, 0x1p1023}); }));
	CHECK(hd::at_most(1, 1, 0x1p1020).contains({0x1p1020, 0x1p1020}) == containment::outside);
}

namespace halfplane_review_proxies {
	struct halfplane_proxy {
		atl::geometry::halfplane<int> const *value;

		operator atl::geometry::halfplane<int>() && { return *value; } // NOLINT(google-explicit-constructor): proxy conversion
	};

	struct halfplane_member_proxy {
		double p, q, r;

		explicit halfplane_member_proxy(atl::geometry::halfplane<int> const *h) : halfplane_member_proxy(*h) {}

		halfplane_member_proxy(atl::geometry::halfplane<int> const &h) : p(double(h.p)), q(double(h.q)), r(double(h.r)) {} // NOLINT(google-explicit-constructor): common reference of proxy and value
	};

	template<typename Proxy = halfplane_proxy>
	struct halfplane_proxy_iterator {
		using value_type = atl::geometry::halfplane<int>;
		using difference_type = std::ptrdiff_t;
		using iterator_concept = std::random_access_iterator_tag;
		value_type const *position{};

		Proxy operator*() const { return Proxy{position}; }

		Proxy operator[](difference_type i) const { return Proxy{position + i}; }

		halfplane_proxy_iterator &operator++() {
			++position;
			return *this;
		}

		halfplane_proxy_iterator operator++(int) {
			auto copy = *this;
			++*this;
			return copy;
		}

		halfplane_proxy_iterator &operator--() {
			--position;
			return *this;
		}

		halfplane_proxy_iterator operator--(int) {
			auto copy = *this;
			--*this;
			return copy;
		}

		halfplane_proxy_iterator &operator+=(difference_type i) {
			position += i;
			return *this;
		}

		halfplane_proxy_iterator &operator-=(difference_type i) {
			position -= i;
			return *this;
		}

		friend halfplane_proxy_iterator operator+(halfplane_proxy_iterator it, difference_type i) { return it += i; }

		friend halfplane_proxy_iterator operator+(difference_type i, halfplane_proxy_iterator it) { return it += i; }

		friend halfplane_proxy_iterator operator-(halfplane_proxy_iterator it, difference_type i) { return it -= i; }

		friend difference_type operator-(halfplane_proxy_iterator a, halfplane_proxy_iterator b) { return a.position - b.position; }

		friend auto operator<=>(halfplane_proxy_iterator, halfplane_proxy_iterator) = default;
	};
}

TEST_CASE("half-plane ranges preserve rvalue-only proxy conversions", "[geometry][accepted]") {
	using ProxyHalfplane = atl::geometry::halfplane<int>;
	using it = halfplane_review_proxies::halfplane_proxy_iterator<>;
	std::array const h{ProxyHalfplane{1, 0, 2}, ProxyHalfplane{-1, 0, 0}, ProxyHalfplane{0, 1, 2}, ProxyHalfplane{0, -1, 0}};
	auto const proxies = std::ranges::subrange(it{h.data()}, it{h.data() + h.size()});
	STATIC_REQUIRE(atl::geometry::halfplane_range<decltype(proxies)>);
	CHECK(atl::geometry::halfplane_intersection(proxies).kind == atl::geometry::region_kind::polygon);
}

TEST_CASE("half-plane member proxies explicitly convert coefficient types", "[geometry][accepted]") {
	using H = atl::geometry::halfplane<int>;
	std::vector<H> const h{{1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1}};
	using it = halfplane_review_proxies::halfplane_proxy_iterator<halfplane_review_proxies::halfplane_member_proxy>;
	auto const proxies = std::ranges::subrange(it{h.data()}, it{h.data() + h.size()});
	static_assert(atl::geometry::halfplane_range<decltype(proxies)>);
	auto const got = atl::geometry::halfplane_intersection(proxies);
	CHECK(got.kind == atl::geometry::region_kind::polygon);
	CHECK(got.vertices.size() == 4);
}
