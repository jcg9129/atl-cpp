#include <atl/geometry/calipers.hpp>
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/convex_polygon_operations.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <ranges>
#include <set>
#include <vector>

using atl::uint32;
using atl::geometry::anti_parallel;
using atl::geometry::antipodal_view;
using atl::geometry::convex_hull;
using atl::geometry::dual_support_view;
using atl::geometry::farthest_pair_indices;
using atl::geometry::narrowest_strip;
using atl::geometry::point;

namespace {
	using pt = point<std::int64_t>;

	struct exact_ratio {
		atl::int128 numerator = 0;
		atl::int128 denominator = 1;

		[[nodiscard]] bool operator<(exact_ratio const &other) const {
			return numerator * other.denominator < other.numerator * denominator;
		}

		[[nodiscard]] double value() const { return double(numerator) / double(denominator); }
	};

	[[nodiscard]] std::int64_t determinant(pt const a, pt const b, pt const c) {
		return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
	}

	std::int64_t cx(pt const a, pt const b) { return a.x * b.y - a.y * b.x; }

	// independent antipodality brute via normal cones: (u, w) antipodal iff
	// the closed normal cone of u intersects the NEGATED cone of w. Cones
	// span less than pi on a strictly convex polygon, so closed-arc overlap
	// reduces to wedge membership of the boundary rays.
	std::set<std::pair<uint32, uint32>> brute_antipodal(std::vector<pt> const &hull) {
		uint32 const h = uint32(hull.size());
		auto const normal = [&](uint32 e) { // outward normal of CCW edge e
			pt const d = hull[(e + 1) % h] - hull[e];
			return pt{d.y, -d.x};
		};
		auto const in_arc = [&](pt const s, pt const e, pt const v) { // span <= pi, closed
			return cx(s, v) >= 0 && cx(v, e) >= 0;
		};
		std::set<std::pair<uint32, uint32>> pairs;
		for (uint32 u = 0; u < h; ++u)
			for (uint32 w = 0; w < h; ++w) {
				if (u == w) continue;
				pt const a1 = normal((u + h - 1) % h), a2 = normal(u);
				pt const b1 = pt{} - normal((w + h - 1) % h), b2 = pt{} - normal(w);
				if (in_arc(a1, a2, b1) || in_arc(a1, a2, b2) || in_arc(b1, b2, a1)
					|| in_arc(b1, b2, a2))
					pairs.insert({(std::min)(u, w), (std::max)(u, w)});
			}
		return pairs;
	}

	std::vector<pt> random_cloud(std::mt19937_64 &rng, std::size_t n, std::int64_t radius) {
		std::vector<pt> pts(n);
		for (auto &p : pts)
			p = {std::int64_t(rng() % (2 * radius + 1)) - radius,
				 std::int64_t(rng() % (2 * radius + 1)) - radius};
		return pts;
	}

	void verify_antipodal(std::vector<pt> const &hull) {
		if (hull.size() < 3) return;
		std::set<std::pair<uint32, uint32>> got;
		uint32 events = 0, last_edge = 0;
		for (auto const ev : antipodal_view(hull)) {
			REQUIRE(ev.edge_start < hull.size());
			REQUIRE(ev.edge_end == (ev.edge_start + 1) % hull.size());
			REQUIRE(ev.opposite < hull.size());
			REQUIRE(ev.edge_start >= last_edge);
			last_edge = ev.edge_start;
			for (auto const [i, j] : ev.antipodal_pairs()) {
				REQUIRE(i != j);
				got.insert({(std::min)(i, j), (std::max)(i, j)});
			}
			++events;
		}
		REQUIRE(events >= hull.size());
		REQUIRE(events <= 2 * hull.size());
		REQUIRE(got == brute_antipodal(hull));

		std::vector<point<double>> hd;
		for (auto const p : hull) hd.push_back({double(p.x), double(p.y)});
		auto const vi = antipodal_view(hull);
		auto const vd = antipodal_view(hd);
		REQUIRE(std::ranges::equal(vi, vd));
	}
}

TEST_CASE("antipodal_view: pairs match the normal-cone brute", "[geometry]") {
	// handcrafted: triangle, square (all ties), rectangle, hexagon with
	// parallel edge pairs, sliver
	verify_antipodal({{0, 0}, {4, 0}, {2, 6}});
	verify_antipodal({{0, 0}, {4, 0}, {4, 4}, {0, 4}});
	verify_antipodal({{0, 0}, {9, 0}, {9, 2}, {0, 2}});
	verify_antipodal({{2, 0}, {5, 0}, {7, 3}, {5, 6}, {2, 6}, {0, 3}});
	verify_antipodal({{0, 0}, {50, 1}, {100, 3}, {50, 4}});

	auto seed = GENERATE(take(25, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const hull = convex_hull(random_cloud(rng, 5 + std::size_t(rng() % 60), 1 + std::int64_t(rng() % 50)));
	verify_antipodal(hull);
}

TEST_CASE("farthest_pair: exact vs quadratic brute", "[geometry]") {
	auto seed = GENERATE(take(25, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 200);
	auto pts = random_cloud(rng, n, 1 + std::int64_t(rng() % 100));
	if (std::size_t(rng() % 3) == 0) // heavy duplicates
		for (auto &p : pts) p = pts[std::size_t(std::size_t(rng() % pts.size()))];
	atl::int128 best = 0;
	for (std::size_t i = 0; i < n; ++i)
		for (std::size_t j = i + 1; j < n; ++j)
			best = (std::max)(best, atl::int128(squared_distance(pts[i], pts[j])));
	auto const [i, j] = farthest_pair_indices(pts);
	REQUIRE(i < n);
	REQUIRE(j < n);
	// a real pair whenever one exists — self-pair only for n == 1
	REQUIRE((n == 1) == (i == j));
	REQUIRE(atl::int128(squared_distance(pts[i], pts[j])) == best);
	auto const [a, b] = atl::geometry::farthest_pair(pts);
	REQUIRE(atl::int128(squared_distance(a, b)) == best);
	// double parity on the distance
	std::vector<point<double>> pd;
	for (auto const p : pts) pd.push_back({double(p.x), double(p.y)});
	auto const [di, dj] = farthest_pair_indices(pd);
	REQUIRE((n == 1) == (di == dj));
	REQUIRE(atl::int128(squared_distance(pts[di], pts[dj])) == best);
}

TEST_CASE("narrowest_strip: exact strip object, width matches the projection brute",
		  "[geometry]") {
	using atl::geometry::narrowest_strip;
	auto const strip_width = [](atl::geometry::strip<std::int64_t> const &s) {
		if (s.edge_start == s.edge_end) return 0.0;
		return double(cross(s.edge_start, s.edge_end, s.opposite))
			 / atl::geometry::distance(s.edge_start, s.edge_end);
	};
	auto const brute_width = [](std::vector<pt> const &hull) {
		exact_ratio best{atl::int128(1) << 100, 1};
		for (std::size_t e = 0; e < hull.size(); ++e) {
			auto const a = hull[e], b = hull[(e + 1) % hull.size()];
			std::int64_t height = 0;
			for (auto const v : hull) height = (std::max)(height, determinant(a, b, v));
			auto const dx = b.x - a.x, dy = b.y - a.y;
			exact_ratio const candidate{atl::int128(height) * height, atl::int128(dx) * dx + atl::int128(dy) * dy};
			if (candidate < best) best = candidate;
		}
		return std::sqrt(best.value());
	};
	auto const square_strip = narrowest_strip(std::vector<pt>{{0, 0}, {4, 0}, {4, 4}, {0, 4}});
	REQUIRE(strip_width(square_strip) == Catch::Approx(4.0));
	REQUIRE(strip_width(narrowest_strip(std::vector<pt>{{0, 0}, {7, 7}})) == 0.0);
	REQUIRE(narrowest_strip(std::vector<pt>{{1, 1}})
			== atl::geometry::strip<std::int64_t>{{1, 1}, {1, 1}, {1, 1}});
	REQUIRE(strip_width(narrowest_strip(std::vector<pt>{{0, 0}, {3, 3}, {6, 6}, {1, 1}})) == 0.0);

	auto seed = GENERATE(take(25, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const pts = random_cloud(rng, 3 + std::size_t(rng() % 100), 1 + std::int64_t(rng() % 1000));
	auto const hull = convex_hull(pts);
	if (hull.size() >= 3) {
		auto const got = narrowest_strip(pts);
		// the strip's defining points are input points, the flush pair is a
		// hull edge, and EVERY point lies between the lines — exact checks
		REQUIRE(std::ranges::count(pts, got.edge_start) > 0);
		REQUIRE(std::ranges::count(pts, got.edge_end) > 0);
		REQUIRE(std::ranges::count(pts, got.opposite) > 0);
		auto const height = cross(got.edge_start, got.edge_end, got.opposite);
		REQUIRE(height >= 0);
		for (auto const q : pts) {
			auto const c = cross(got.edge_start, got.edge_end, q);
			REQUIRE(c >= 0);
			REQUIRE(c <= height);
		}
		double const expected = brute_width(hull);
		REQUIRE(std::abs(strip_width(got) - expected) <= 1e-9 * (std::max)(1.0, expected));
	}
}

TEST_CASE("dual_support_view parallel: Minkowski sum oracle", "[geometry]") {
	auto seed = GENERATE(take(25, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const make = [&](std::size_t n) {
		auto const cloud = random_cloud(rng, n, 1 + std::int64_t(rng() % 100));
		return convex_hull(cloud);
	};
	auto const P = make(3 + std::size_t(rng() % 30));
	auto const Q = std::size_t(rng() % 5) == 0 ? std::vector<pt>{{7, -3}} : make(3 + std::size_t(rng() % 30));

	std::vector<pt> sum;
	for (auto const ev : dual_support_view(P, Q))
		sum.push_back(P[ev.p_support] + Q[ev.q_support]);
	if (sum.empty()) return; // both degenerate
	std::vector<pt> all;
	for (auto const a : P)
		for (auto const b : Q) all.push_back(a + b);
	REQUIRE(convex_hull(sum) == convex_hull(all));
}

TEST_CASE("dual_support_view anti_parallel: max distance oracle", "[geometry]") {
	auto seed = GENERATE(take(25, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const make = [&](std::size_t n) {
		auto const cloud = random_cloud(rng, n, 1 + std::int64_t(rng() % 100));
		return convex_hull(cloud);
	};
	auto const P = make(3 + std::size_t(rng() % 30));
	auto const Q = std::size_t(rng() % 5) == 0 ? std::vector<pt>{{250, 111}} : make(3 + std::size_t(rng() % 30));

	atl::int128 got = 0;
	for (auto const ev : dual_support_view(P, Q, anti_parallel))
		for (auto const [i, j] : ev.support_pairs())
			got = (std::max)(got, atl::int128(squared_distance(P[i], Q[j])));
	atl::int128 expected = 0;
	for (auto const a : P)
		for (auto const b : Q) expected = (std::max)(expected, atl::int128(squared_distance(a, b)));
	REQUIRE(got == expected);
}

TEST_CASE("calipers: degenerate polygons", "[geometry]") {
	std::vector<pt> const empty{};
	std::vector<pt> const single{{5, 5}};
	std::vector<pt> const segment{{0, 0}, {4, 2}};
	REQUIRE(antipodal_view(empty).begin() == antipodal_view(empty).end());
	REQUIRE(antipodal_view(single).begin() == antipodal_view(single).end());
	REQUIRE(antipodal_view(segment).begin() == antipodal_view(segment).end());
	// an empty side yields an empty dual view — no support to pair
	std::vector<pt> const triangle{{0, 0}, {6, 0}, {2, 5}};
	REQUIRE(dual_support_view(empty, triangle).begin() == dual_support_view(empty, triangle).end());
	REQUIRE(dual_support_view(triangle, empty).begin() == dual_support_view(triangle, empty).end());

	// segments are legal dual inputs: oracle both modes against brutes
	for (auto const &other : {triangle, segment, std::vector<pt>{{9, -3}}}) {
		std::vector<pt> sum;
		for (auto const ev : dual_support_view(segment, other))
			sum.push_back(segment[ev.p_support] + other[ev.q_support]);
		std::vector<pt> all;
		for (auto const a : segment)
			for (auto const b : other) all.push_back(a + b);
		REQUIRE(convex_hull(sum) == convex_hull(all));

		atl::int128 got = 0, expected = 0;
		for (auto const ev : dual_support_view(segment, other, anti_parallel))
			for (auto const [i, j] : ev.support_pairs())
				got = (std::max)(got, atl::int128(squared_distance(segment[i], other[j])));
		for (auto const a : segment)
			for (auto const b : other) expected = (std::max)(expected, atl::int128(squared_distance(a, b)));
		REQUIRE(got == expected);
	}
	auto const [i, j] = farthest_pair_indices(single);
	REQUIRE(i == 0);
	REQUIRE(j == 0);
	// Duplicate-only clouds still have two distinct input indices when n >= 2.
	for (std::size_t count : {std::size_t(2), std::size_t(3), std::size_t(7)}) {
		std::vector<pt> const equal(count, pt{5, 5});
		auto const [di, dj] = farthest_pair_indices(equal);
		REQUIRE(di < count);
		REQUIRE(dj < count);
		REQUIRE(di != dj);
		REQUIRE(squared_distance(equal[di], equal[dj]) == 0);
	}
	auto const [a, b] = farthest_pair_indices(segment);
	REQUIRE(squared_distance(segment[a], segment[b]) == 20);
	REQUIRE(dual_support_view(single, single).begin() == dual_support_view(single, single).end());
}

TEST_CASE("farthest_pair between two sets: vs quadratic brute", "[geometry]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const cloud = [&](std::size_t n, std::int64_t cx, std::int64_t cy) {
		std::vector<pt> pts(n);
		for (auto &p : pts)
			p = {cx + std::int64_t(rng() % 201) - 100, cy + std::int64_t(rng() % 201) - 100};
		return pts;
	};
	auto const a = cloud(1 + std::size_t(rng() % 100), 0, 0);
	auto const b = cloud(1 + std::size_t(rng() % 100), std::int64_t(rng() % 500), std::int64_t(rng() % 500));
	atl::int128 expected = -1;
	for (auto const p : a)
		for (auto const q : b) expected = (std::max)(expected, atl::int128(squared_distance(p, q)));
	auto const [i, j] = atl::geometry::farthest_pair_between_indices(a, b);
	REQUIRE(i < a.size());
	REQUIRE(j < b.size());
	REQUIRE(atl::int128(squared_distance(a[i], b[j])) == expected);
	auto const [pa, pb] = atl::geometry::farthest_pair_between(a, b);
	REQUIRE(atl::int128(squared_distance(pa, pb)) == expected);
}

TEST_CASE("distance_between_convex_hulls: vs feature brute, overlap and touch", "[geometry]") {
	using atl::geometry::distance_between_convex_hulls;
	using atl::geometry::segment;
	auto const brute = [](std::vector<pt> const &a, std::vector<pt> const &b) {
		auto const ha = convex_hull(a), hb = convex_hull(b);
		auto const inside = [](std::vector<pt> const &h, pt const q) {
			if (h.size() < 3) return false;
			for (std::size_t i = 0; i < h.size(); ++i)
				if (determinant(h[i], h[(i + 1) % h.size()], q) < 0) return false;
			return true;
		};
		for (auto const q : ha)
			if (inside(hb, q)) return 0.0;
		for (auto const q : hb)
			if (inside(ha, q)) return 0.0;
		auto const on_segment = [](pt const u, pt const v, pt const q) {
			return determinant(u, v, q) == 0 && (std::min)(u.x, v.x) <= q.x && q.x <= (std::max)(u.x, v.x)
				&& (std::min)(u.y, v.y) <= q.y && q.y <= (std::max)(u.y, v.y);
		};
		auto const point_gap = [](pt const u, pt const v, pt const q) -> exact_ratio {
			auto const dx = v.x - u.x, dy = v.y - u.y, qx = q.x - u.x, qy = q.y - u.y;
			auto const length2 = dx * dx + dy * dy, projection = dx * qx + dy * qy;
			if (projection <= 0) return {qx * qx + qy * qy, 1};
			if (projection >= length2) return {(q.x - v.x) * (q.x - v.x) + (q.y - v.y) * (q.y - v.y), 1};
			auto const area2 = determinant(u, v, q);
			return {atl::int128(area2) * area2, length2};
		};
		exact_ratio best{atl::int128(1) << 100, 1};
		for (std::size_t i = 0; i < ha.size(); ++i)
			for (std::size_t j = 0; j < hb.size(); ++j) {
				auto const u = ha[i], v = ha[(i + 1) % ha.size()], w = hb[j], z = hb[(j + 1) % hb.size()];
				auto const c1 = determinant(u, v, w), c2 = determinant(u, v, z), c3 = determinant(w, z, u), c4 = determinant(w, z, v);
				if (((c1 < 0 && c2 > 0) || (c1 > 0 && c2 < 0)) && ((c3 < 0 && c4 > 0) || (c3 > 0 && c4 < 0))) return 0.0;
				if (on_segment(u, v, w) || on_segment(u, v, z) || on_segment(w, z, u) || on_segment(w, z, v)) return 0.0;
				for (auto const candidate : {point_gap(u, v, w), point_gap(u, v, z), point_gap(w, z, u), point_gap(w, z, v)})
					if (candidate < best) best = candidate;
			}
		return std::sqrt(best.value());
	};
	// handcrafted: separated, touching at a point, edge overlap, containment
	std::vector<pt> const sq{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
	REQUIRE(distance_between_convex_hulls(sq, std::vector<pt>{{6, 0}, {9, 0}, {7, 3}}) == Catch::Approx(2.0));
	REQUIRE(distance_between_convex_hulls(sq, std::vector<pt>{{4, 4}, {8, 4}, {6, 7}}) == 0.0); // touch
	REQUIRE(distance_between_convex_hulls(sq, std::vector<pt>{{1, 1}, {2, 1}, {2, 2}}) == 0.0); // inside
	REQUIRE(distance_between_convex_hulls(sq, std::vector<pt>{{3, 3}, {9, 3}, {9, 9}}) == 0.0); // overlap
	REQUIRE(distance_between_convex_hulls(std::vector<pt>{{0, 0}}, std::vector<pt>{{3, 4}}) == Catch::Approx(5.0));
	REQUIRE(distance_between_convex_hulls(std::vector<pt>{{0, 5}}, std::vector<pt>{{-3, 0}, {3, 0}}) == Catch::Approx(5.0));

	auto seed = GENERATE(take(25, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const cloud = [&](std::int64_t cx, std::int64_t cy, std::int64_t r, std::size_t n) {
		std::vector<pt> pts(n);
		for (auto &p : pts)
			p = {cx + std::int64_t(rng() % (2 * r + 1)) - r, cy + std::int64_t(rng() % (2 * r + 1)) - r};
		return pts;
	};
	auto const a = cloud(0, 0, 40, 1 + std::size_t(rng() % 40));
	auto const b = cloud(std::int64_t(rng() % 160), std::int64_t(rng() % 160), 40, 1 + std::size_t(rng() % 40));
	double const got = distance_between_convex_hulls(a, b);
	double const expected = brute(a, b);
	REQUIRE(std::abs(got - expected) <= 1e-9 * (std::max)(1.0, expected));
}

TEST_CASE("enclosing rectangles: properties and brute values", "[geometry]") {
	using atl::geometry::min_area_enclosing_rectangle;
	using atl::geometry::min_perimeter_enclosing_rectangle;
	using dp = point<double>;
	auto const rect_area = [](std::array<dp, 4> const &r) {
		return atl::geometry::polygon_area(std::vector<dp>(r.begin(), r.end()));
	};
	auto const rect_perimeter = [](std::array<dp, 4> const &r) {
		double s = 0;
		for (int k = 0; k < 4; ++k) s += atl::geometry::distance(r[k], r[(k + 1) % 4]);
		return s;
	};
	// brute: for every hull edge direction, box extents by projection
	auto const brute = [](std::vector<pt> const &hull) {
		exact_ratio best_area{atl::int128(1) << 100, 1}, best_perimeter2 = best_area;
		for (std::size_t e = 0; e < hull.size(); ++e) {
			auto const a = hull[e], b = hull[(e + 1) % hull.size()];
			auto const ux = b.x - a.x, uy = b.y - a.y, length2 = ux * ux + uy * uy;
			if (length2 == 0) continue;
			std::int64_t alo = 0, ahi = 0, clo = 0, chi = 0;
			for (auto const v : hull) {
				auto const along = (v.x - a.x) * ux + (v.y - a.y) * uy;
				auto const across = determinant(a, b, v);
				alo = (std::min)(alo, along);
				ahi = (std::max)(ahi, along);
				clo = (std::min)(clo, across);
				chi = (std::max)(chi, across);
			}
			auto const width = ahi - alo, height = chi - clo;
			exact_ratio const area{atl::int128(width) * height, length2};
			exact_ratio const perimeter2{4 * atl::int128(width + height) * (width + height), length2};
			if (area < best_area) best_area = area;
			if (perimeter2 < best_perimeter2) best_perimeter2 = perimeter2;
		}
		return std::pair{best_area.value(), std::sqrt(best_perimeter2.value())};
	};
	// exact handcrafted: axis-aligned square is its own best rectangle
	std::vector<pt> const sq{{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}, {1, 3}};
	auto const r0 = min_area_enclosing_rectangle(sq);
	std::array<dp, 4> const square_corners{dp{0, 0}, dp{4, 0}, dp{4, 4}, dp{0, 4}};
	for (auto const rectangle : {r0, min_perimeter_enclosing_rectangle(sq)})
		for (std::size_t k = 0; k < 4; ++k) REQUIRE(atl::geometry::distance(rectangle[k], square_corners[k]) <= 1e-12);
	// degenerate: point and collinear inputs
	auto const rp = min_area_enclosing_rectangle(std::vector<pt>{{7, 9}});
	REQUIRE(rp == std::array<dp, 4>{dp{7, 9}, dp{7, 9}, dp{7, 9}, dp{7, 9}});
	auto const rl = min_perimeter_enclosing_rectangle(std::vector<pt>{{0, 0}, {2, 2}, {6, 6}});
	REQUIRE(std::abs(rect_area(rl)) <= 1e-12);
	REQUIRE(std::abs(rect_perimeter(rl) - 2 * std::hypot(6.0, 6.0)) <= 1e-9);

	auto seed = GENERATE(take(25, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<pt> pts(3 + std::size_t(rng() % 60));
	for (auto &p : pts)
		p = {std::int64_t(rng() % 2001) - 1000, std::int64_t(rng() % 2001) - 1000};
	auto const hull = convex_hull(pts);
	if (hull.size() < 3) return;
	auto const [expected_area, expected_perimeter] = brute(hull);
	auto const ra = min_area_enclosing_rectangle(pts);
	auto const rp2 = min_perimeter_enclosing_rectangle(pts);
	REQUIRE(std::abs(rect_area(ra) - expected_area) <= 1e-9 * expected_area);
	REQUIRE(std::abs(rect_perimeter(rp2) - expected_perimeter) <= 1e-9 * expected_perimeter);
	for (auto const r : {ra, rp2}) {
		// CCW right-angled, and every point inside (within construction slack)
		double const scale = 2000.0;
		for (int k = 0; k < 4; ++k) {
			auto const u = r[(k + 1) % 4] - r[k], w = r[(k + 3) % 4] - r[k];
			REQUIRE(cross(u, w) >= 0.0);
			REQUIRE(std::abs(dot(u, w)) <= 1e-6 * scale * scale);
		}
		for (auto const q : pts)
			for (int k = 0; k < 4; ++k)
				REQUIRE(cross(r[k], r[(k + 1) % 4], dp{double(q.x), double(q.y)})
						>= -1e-6 * scale * scale);
	}
}

TEST_CASE("closest_points_between_convex_hulls: witnesses achieve the distance, on both hulls",
		  "[geometry]") {
	using atl::geometry::closest_points_between_convex_hulls;
	using atl::geometry::convex_hull;
	using atl::geometry::distance_between_convex_hulls;
	auto const boundary_gap = [](std::vector<pt> const &hull, point<double> const w) {
		double best = 1e300;
		if (hull.size() == 1)
			return std::hypot(w.x - double(hull[0].x), w.y - double(hull[0].y));
		for (std::size_t i = 0, j = hull.size() - 1; i < hull.size(); j = i++) {
			atl::geometry::segment<double> const e{{double(hull[j].x), double(hull[j].y)},
												   {double(hull[i].x), double(hull[i].y)}};
			best = (std::min)(best, distance(e, w));
		}
		return best;
	};
	auto seed = GENERATE(take(25, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	// two clusters, sometimes touching at a planted shared vertex
	std::vector<pt> a, b;
	std::int64_t const gap = std::int64_t(rng() % 3) * 40; // 0 = likely touching/close
	for (int k = 0; k < 12; ++k) {
		a.push_back({std::int64_t(rng() % 20), std::int64_t(rng() % 20)});
		b.push_back({gap + 20 + std::int64_t(rng() % 20), std::int64_t(rng() % 20)});
	}
	if (gap == 0) { // force a touch: plant the same point in both
		a.push_back({20, 10});
		b.push_back({20, 10});
	}
	double const dist = distance_between_convex_hulls(a, b);
	auto const [wa, wb] = closest_points_between_convex_hulls(a, b);
	REQUIRE(std::abs(std::hypot(wa.x - wb.x, wa.y - wb.y) - dist) <= 1e-9 * (1 + dist));
	REQUIRE(boundary_gap(convex_hull(a), wa) <= 1e-9);
	REQUIRE(boundary_gap(convex_hull(b), wb) <= 1e-9);
	if (gap == 0) REQUIRE(dist == 0.0); // the planted shared vertex touches
}

TEST_CASE("closest_points_between_convex_hulls: degenerate summands and exact spots", "[geometry]") {
	using atl::geometry::closest_points_between_convex_hulls;
	auto const pp = closest_points_between_convex_hulls(std::vector<pt>{{0, 0}}, std::vector<pt>{{3, 4}});
	REQUIRE(pp[0].x == 0.0);
	REQUIRE(pp[1].x == 3.0);
	REQUIRE(std::hypot(pp[0].x - pp[1].x, pp[0].y - pp[1].y) == Catch::Approx(5.0));
	auto const sp = closest_points_between_convex_hulls(std::vector<pt>{{0, 0}, {10, 0}},
														std::vector<pt>{{4, 3}});
	REQUIRE(std::abs(sp[0].x - 4.0) <= 1e-9);
	REQUIRE(std::abs(sp[0].y - 0.0) <= 1e-9);
	REQUIRE(sp[1].x == 4.0);
	std::vector<pt> const left{{0, 0}, {2, 0}, {2, 10}, {0, 10}};
	std::vector<pt> const right{{7, 3}, {9, 3}, {9, 6}, {7, 6}};
	auto const ee = closest_points_between_convex_hulls(left, right);
	REQUIRE(std::abs(ee[0].x - 2.0) <= 1e-9);
	REQUIRE(std::abs(ee[1].x - 7.0) <= 1e-9);
	REQUIRE(std::abs(std::hypot(ee[0].x - ee[1].x, ee[0].y - ee[1].y) - 5.0) <= 1e-9);
	REQUIRE(ee[0].y >= 3.0 - 1e-9);
	REQUIRE(ee[0].y <= 6.0 + 1e-9);
	auto const touch = closest_points_between_convex_hulls(std::vector<pt>{{0, 0}, {4, 0}, {4, 4}, {0, 4}},
														   std::vector<pt>{{4, 1}, {8, 1}, {8, 3}, {4, 3}});
	REQUIRE(std::abs(touch[0].x - touch[1].x) <= 1e-9);
	REQUIRE(std::abs(touch[0].y - touch[1].y) <= 1e-9);
	REQUIRE(std::abs(touch[0].x - 4.0) <= 1e-9);
}

TEST_CASE("closest_points_between_convex_hulls: overlapping hulls give a common point",
		  "[geometry]") {
	using atl::geometry::closest_points_between_convex_hulls;
	using atl::geometry::distance_between_convex_hulls;
	std::vector<pt> const a{{0, 0}, {10, 0}, {10, 10}, {0, 10}};
	std::vector<pt> const b{{5, 5}, {15, 5}, {15, 15}, {5, 15}};
	auto const [wa, wb] = closest_points_between_convex_hulls(a, b);
	REQUIRE(wa.x == wb.x);
	REQUIRE(wa.y == wb.y);
	REQUIRE(distance_between_convex_hulls(a, b) == 0.0);
	// the common point is in (or on) both squares
	REQUIRE(wa.x >= 5.0 - 1e-9);
	REQUIRE(wa.x <= 10.0 + 1e-9);
	REQUIRE(wa.y >= 5.0 - 1e-9);
	REQUIRE(wa.y <= 10.0 + 1e-9);
	std::vector<pt> const wide{{-10, 4}, {10, 4}, {10, 6}, {-10, 6}};
	std::vector<pt> const tall{{4, -10}, {6, -10}, {6, 10}, {4, 10}};
	auto const [la, lb] = closest_points_between_convex_hulls(wide, tall);
	REQUIRE(la.x == lb.x);
	REQUIRE(la.y == lb.y);
	REQUIRE(la.x >= 4.0 - 1e-9);
	REQUIRE(la.x <= 6.0 + 1e-9);
	REQUIRE(la.y >= 4.0 - 1e-9);
	REQUIRE(la.y <= 6.0 + 1e-9);
}

TEST_CASE("enclosing rectangles are translation-invariant translation from the origin", "[geometry]") {
	// Differencing in the widened type preserves short edges at large
	// translations; rounding both endpoints can turn their direction into zero.
	using atl::int64;
	using atl::geometry::min_area_enclosing_rectangle;
	using atl::geometry::min_perimeter_enclosing_rectangle;
	using atl::geometry::point;
	std::vector<point<int64>> const origin_cloud{{0, 0}, {7, 1}, {9, 6}, {2, 8}, {-3, 3}, {3, 4}}; // no two edges tie
	for (int shift = 54; shift <= 61; ++shift) {
		int64 const translation = int64(1) << shift;
		std::vector<point<int64>> moved;
		for (auto const q : origin_cloud) moved.push_back({q.x + translation, q.y - translation});
		// one ulp of a coordinate at this magnitude
		double const resolution = std::ldexp(1.0, shift - 52);
		auto const check = [&](std::array<point<double>, 4> const &at_origin, std::array<point<double>, 4> const &at_distance) {
			for (std::size_t k = 0; k < 4; ++k) {
				REQUIRE(std::isfinite(at_distance[k].x));
				REQUIRE(std::abs(at_distance[k].x - double(translation) - at_origin[k].x) <= 2 * resolution);
				REQUIRE(std::abs(at_distance[k].y + double(translation) - at_origin[k].y) <= 2 * resolution);
			}
		};
		check(min_area_enclosing_rectangle(origin_cloud), min_area_enclosing_rectangle(moved));
		check(min_perimeter_enclosing_rectangle(origin_cloud), min_perimeter_enclosing_rectangle(moved));
	}
}

TEST_CASE("calipers: exact integral limit and floating construction scales", "[geometry]") {
	std::int64_t const limit = (std::int64_t(1) << 62) - 1;
	std::vector<pt> const square{{-limit, -limit}, {limit, -limit}, {limit, limit}, {-limit, limit}};
	auto const pair = atl::geometry::farthest_pair(square);
	atl::int128 const side_length = atl::int128(limit) * 2;
	REQUIRE(squared_distance(pair[0], pair[1]) == 2 * side_length * side_length);
	REQUIRE(std::ranges::distance(antipodal_view(square)) == 8);
	std::int64_t const gap_limit = (std::int64_t(1) << 61) - 1;
	double const expected_gap = double(atl::int128(gap_limit) * 2);
	REQUIRE(std::abs(atl::geometry::distance_between_convex_hulls(std::vector<pt>{{-gap_limit, 0}}, std::vector<pt>{{gap_limit, 0}}) - expected_gap) <= 1e-14 * expected_gap);
	for (int exponent : {-120, 0, 120}) {
		double const scale = std::ldexp(1.0, exponent);
		std::vector<point<double>> const points{{0, 0}, {4 * scale, 0}, {4 * scale, 4 * scale}, {0, 4 * scale}};
		auto const rectangle = atl::geometry::min_area_enclosing_rectangle(points);
		double const expected_area = 16 * scale * scale;
		REQUIRE(std::abs(atl::geometry::polygon_area(rectangle) - expected_area) <= 1e-12 * expected_area);
		auto const band = narrowest_strip(points);
		double const width = cross(band.edge_start, band.edge_end, band.opposite) / distance(band.edge_start, band.edge_end);
		REQUIRE(std::abs(width - 4 * scale) <= 1e-12 * scale);
		REQUIRE(std::ranges::distance(dual_support_view(points, points)) == 8);
	}
}

namespace {
	struct counted_caliper_policy : atl::geometry::exact {
		std::size_t *calls;

		int cross_sign(pt a, pt b, pt c, pt d) const {
			++*calls;
			return atl::geometry::exact::cross_sign(a, b, c, d);
		}
	};
}

TEST_CASE("caliper views: initializer-list ownership and constant-time begin", "[geometry]") {
	auto owned = antipodal_view(std::initializer_list<pt>{{0, 0}, {4, 0}, {0, 4}});
	auto dual = dual_support_view(std::initializer_list<pt>{{0, 0}, {4, 0}, {0, 4}},
								  std::initializer_list<pt>{{0, 0}, {2, 0}, {0, 2}});
	auto opposed = dual_support_view(std::initializer_list<pt>{{0, 0}, {4, 0}, {0, 4}},
									 std::initializer_list<pt>{{0, 0}, {2, 0}, {0, 2}}, anti_parallel);
	CHECK(std::ranges::distance(owned) == 3);
	CHECK(std::ranges::distance(dual) == 6);
	CHECK(std::ranges::distance(opposed) == 6);
	auto copied = owned;
	auto moved = std::move(dual);
	CHECK(*copied.begin() == *owned.begin());
	CHECK(std::ranges::distance(moved) == 6);
	std::vector<pt> parabola;
	for (std::int64_t i = 0; i < 200; ++i) parabola.push_back({i, i * i});
	std::size_t calls = 0;
	antipodal_view cached(parabola, counted_caliper_policy{{}, &calls});
	calls = 0;
	for (int i = 0; i < 200; ++i) CHECK(*cached.begin() == *cached.begin());
	CHECK(calls == 0);
}
