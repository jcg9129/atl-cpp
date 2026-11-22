#include <atl/ds/dsu.hpp>
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/delaunay.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/detail/int256.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <vector>
#include "../expect_abort.hpp"

using atl::int128;
using atl::geometry::convex_hull;
using atl::geometry::delaunay;
using atl::geometry::delaunay_result;
using atl::geometry::keep_collinear;
using atl::geometry::point;
using atl::geometry::polygon_area2;
using atl::geometry::voronoi_from_delaunay;

namespace {
	using pt = point<std::int64_t>;

	int empty_disk_sign(pt const a, pt const b, pt const c, pt const q) {
		using W = atl::int128;
		W const ux = W(b.x) - a.x, uy = W(b.y) - a.y;
		W const vx = W(c.x) - a.x, vy = W(c.y) - a.y;
		W const qx = W(q.x) - a.x, qy = W(q.y) - a.y;
		W const u2 = ux * ux + uy * uy, v2 = vx * vx + vy * vy;
		W const den = ux * vy - uy * vx;
		W const nx = vy * u2 - uy * v2, ny = ux * v2 - vx * u2;
		auto power = atl::detail::int256::product(qx, nx);
		power += atl::detail::int256::product(qy, ny);
		power += atl::detail::int256::product(-(qx * qx + qy * qy), den);
		return power.sign();
	}

	bool topology_certificate(std::vector<pt> const &pts, delaunay_result const &d) {
		if (d.neighbors.size() != d.triangles.size()) return false;
		std::set<std::array<atl::geometry::vertex_index, 3>> unique;
		std::set<pt> used;
		using edge = std::pair<pt, pt>;
		std::map<edge, std::vector<std::pair<std::size_t, std::size_t>>> owners;
		for (std::size_t t = 0; t < d.triangles.size(); ++t) {
			auto sorted = d.triangles[t];
			std::ranges::sort(sorted);
			if (!unique.insert(sorted).second) return false;
			for (auto const v : sorted) {
				if (v >= pts.size()) return false;
				used.insert(pts[v]);
			}
			for (std::size_t e = 0; e < 3; ++e) {
				auto const a = pts[d.triangles[t][e]], b = pts[d.triangles[t][(e + 1) % 3]];
				owners[{(std::min)(a, b), (std::max)(a, b)}].push_back({t, e});
			}
		}
		if (used != std::set<pt>(pts.begin(), pts.end())) return false;
		auto const boundary = convex_hull(pts, keep_collinear);
		std::set<edge> hull_edges;
		for (std::size_t i = 0; i < boundary.size(); ++i) {
			auto const a = boundary[i], b = boundary[(i + 1) % boundary.size()];
			hull_edges.emplace((std::min)(a, b), (std::max)(a, b));
		}
		for (auto const &[key, incident] : owners) {
			if (incident.size() == 1) {
				if (hull_edges.erase(key) != 1) return false;
				auto const [t, e] = incident[0];
				if (d.neighbors[t][e] != delaunay_result::no_neighbor) return false;
			} else if (incident.size() == 2) {
				if (hull_edges.contains(key)) return false;
				auto const [t, e] = incident[0];
				auto const [u, f] = incident[1];
				if (d.neighbors[t][e] != u || d.neighbors[u][f] != t) return false;
				if (pts[d.triangles[t][e]] != pts[d.triangles[u][(f + 1) % 3]]
					|| pts[d.triangles[t][(e + 1) % 3]] != pts[d.triangles[u][f]]) return false;
			} else
				return false;
		}
		return hull_edges.empty();
	}

	void require_center_in_hull(std::vector<pt> const &pts, point<double> q) {
		auto const hull = convex_hull(pts);
		REQUIRE(!hull.empty());
		// Rounded constructions get a local length tolerance, including translations.
		double const ax = double(hull[0].x), ay = double(hull[0].y);
		double const qx = q.x - ax, qy = q.y - ay;
		if (hull.size() == 1) {
			REQUIRE(std::hypot(qx, qy) < 1e-7);
			return;
		}
		if (hull.size() == 2) {
			double const dx = double(hull[1].x) - ax, dy = double(hull[1].y) - ay;
			double const length = std::hypot(dx, dy);
			REQUIRE(std::abs(dx * qy - dy * qx) <= 1e-7 * length);
			REQUIRE(dx * qx + dy * qy >= -1e-7 * length);
			REQUIRE(dx * qx + dy * qy <= length * length + 1e-7 * length);
			return;
		}
		for (std::size_t i = 0; i < hull.size(); ++i) {
			auto const a = hull[i], b = hull[(i + 1) % hull.size()];
			double const dx = double(b.x - a.x), dy = double(b.y - a.y);
			double const side = dx * (q.y - double(a.y)) - dy * (q.x - double(a.x));
			REQUIRE(side >= -1e-7 * std::hypot(dx, dy));
		}
	}

	// full certificate: Euler count, positive CCW triangles, exact area
	// partition of the hull, the EMPTY-CIRCUMCIRCLE property via the exact
	// incircle, and symmetric adjacency
	void verify(std::vector<pt> const &pts) {
		auto const d = delaunay(pts);
		std::vector<pt> distinct(pts);
		std::ranges::sort(distinct);
		auto const [d0, d1] = std::ranges::unique(distinct);
		distinct.erase(d0, d1);
		auto const hull = convex_hull(distinct);
		auto const boundary = convex_hull(distinct, keep_collinear);
		if (distinct.size() < 3 || hull.size() < 3) {
			REQUIRE(d.triangles.empty());
			return;
		}
		REQUIRE(d.triangles.size() == 2 * distinct.size() - boundary.size() - 2);
		REQUIRE(d.neighbors.size() == d.triangles.size());
		REQUIRE(topology_certificate(pts, d));
		int128 area = 0;
		for (auto const &t : d.triangles) {
			auto const o = cross(pts[t[0]], pts[t[1]], pts[t[2]]);
			REQUIRE(o > 0);
			area += o;
		}
		auto const hull_area2 = polygon_area2(hull);
		REQUIRE(area == int128(hull_area2 < 0 ? -hull_area2 : hull_area2));
		for (auto const &t : d.triangles)
			for (auto const &q : distinct) {
				if (q == pts[t[0]] || q == pts[t[1]] || q == pts[t[2]]) continue;
				REQUIRE(empty_disk_sign(pts[t[0]], pts[t[1]], pts[t[2]], q)
						<= 0);
			}
		for (std::size_t t = 0; t < d.triangles.size(); ++t)
			for (int e = 0; e < 3; ++e) {
				auto const o = d.neighbors[t][e];
				if (o == delaunay_result::no_neighbor) continue;
				bool back = false;
				for (int e2 = 0; e2 < 3; ++e2) back |= d.neighbors[o][e2] == t;
				REQUIRE(back);
			}
	}
}

TEST_CASE("delaunay: degenerate zoo and cocircular grids", "[geometry]") {
	verify({{0, 0}, {4, 0}, {0, 4}});
	verify({{0, 0}, {4, 0}, {0, 4}, {4, 4}});		  // cocircular square
	verify({{0, 0}, {1, 1}, {2, 2}, {3, 3}});		  // collinear: empty
	verify({{0, 0}, {0, 0}, {4, 0}, {0, 4}, {4, 0}}); // duplicates
	verify({{5, 5}});
	verify({{1, 2}, {3, 4}});
	verify({});
	std::vector<pt> grid; // heavy cocircularity
	for (std::int64_t x = 0; x < 7; ++x)
		for (std::int64_t y = 0; y < 7; ++y) grid.push_back({x, y});
	verify(grid);
	std::vector<pt> circle; // many cocircular points on one circle
	for (auto const &[x, y] : {std::pair{3LL, 4LL}, {4LL, 3LL}, {5LL, 0LL}, {0LL, 5LL}})
		for (auto const &[sx, sy] : {std::pair{1, 1}, {1, -1}, {-1, 1}, {-1, -1}})
			circle.push_back({x * sx, y * sy});
	verify(circle);
}

TEST_CASE("delaunay: random point sets vs the certificate", "[geometry]") {
	auto seed = GENERATE(take(40, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 3 + std::size_t(rng() % 60);
	std::int64_t const span = 3 + std::int64_t(rng() % 50);
	std::vector<pt> pts;
	for (std::size_t k = 0; k < n; ++k)
		pts.push_back({std::int64_t(rng() % std::uint64_t(span)), std::int64_t(rng() % std::uint64_t(span))});
	verify(pts);
}

TEST_CASE("delaunay: coordinates nearby the 2^32 bound", "[geometry]") {
	std::int64_t const c = (std::int64_t(1) << 32) - 1;
	verify({{-c, -c}, {c, -c}, {c, c}, {-c, c}, {0, 0}, {1, 2}, {-3, 5}});
}

TEST_CASE("voronoi: circumcenter vertices and bisector edges", "[geometry]") {
	auto seed = GENERATE(take(15, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::vector<pt> pts;
	std::size_t const n = 4 + std::size_t(rng() % 30);
	for (std::size_t k = 0; k < n; ++k)
		pts.push_back({std::int64_t(rng() % 1000), std::int64_t(rng() % 1000)});
	auto const d = delaunay(pts);
	if (d.triangles.empty()) return;
	auto const vor = voronoi_from_delaunay(pts, d);
	REQUIRE(vor.vertices.size() == d.triangles.size());
	for (std::size_t t = 0; t < d.triangles.size(); ++t) {
		auto const c = vor.vertices[t];
		double r0 = 0;
		for (int k = 0; k < 3; ++k) {
			auto const s = pts[d.triangles[t][k]];
			double const r = std::hypot(c.x - double(s.x), c.y - double(s.y));
			if (k == 0)
				r0 = r;
			else
				REQUIRE(std::abs(r - r0) <= 1e-6 * (1 + r0));
		}
	}
	for (auto const &e : vor.edges) {
		auto const a = pts[e.site_a], b = pts[e.site_b];
		auto const equidistant = [&](point<double> const q) {
			double const da = std::hypot(q.x - double(a.x), q.y - double(a.y));
			double const db = std::hypot(q.x - double(b.x), q.y - double(b.y));
			return std::abs(da - db) <= 1e-6 * (1 + da);
		};
		REQUIRE(equidistant(e.from));
		point<double> const probe =
			e.is_unbounded ? point<double>{e.from.x + 0.37 * e.direction.x,
										   e.from.y + 0.37 * e.direction.y}
						   : point<double>{(e.from.x + e.to.x) / 2, (e.from.y + e.to.y) / 2};
		REQUIRE(equidistant(probe));
	}
	// edge count: one dual edge per Delaunay edge
	std::size_t interior = 0, hull_edges = 0;
	for (std::size_t t = 0; t < d.triangles.size(); ++t)
		for (int k = 0; k < 3; ++k) {
			if (d.neighbors[t][k] == delaunay_result::no_neighbor)
				++hull_edges;
			else
				++interior;
		}
	REQUIRE(vor.edges.size() == interior / 2 + hull_edges);
}

TEST_CASE("delaunay: double coordinates, parity spot check", "[geometry]") {
	std::vector<point<double>> pts{{0, 0}, {4, 0}, {0, 4}, {4, 4}, {2, 1.5}};
	auto const d = delaunay(pts);
	REQUIRE(d.triangles.size() == 4); // interior point fans the square
	for (auto const &t : d.triangles)
		REQUIRE(atl::geometry::exact{}.orientation(pts[t[0]], pts[t[1]], pts[t[2]]) > 0);
}

TEST_CASE("voronoi: degenerate semantics pinned", "[geometry]") {
	// cocircular square: adjacent cocircular triangles share a circumcenter
	// -> the dual segment between them has zero length (cells touch at one
	// point); the four hull rays are the real cell walls
	std::vector<pt> const sq{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
	auto const v = voronoi_from_delaunay(sq, delaunay(sq));
	std::size_t rays = 0, zero_segments = 0;
	for (auto const &e : v.edges) {
		if (e.is_unbounded)
			++rays;
		else if (atl::geometry::approx{1e-12}.equal(e.from, e.to))
			++zero_segments;
	}
	REQUIRE(rays == 4);
	REQUIRE(zero_segments == 1);
	// no triangles => empty dual (the true Voronoi of collinear sites is a
	// family of bisector LINES — outside this representation, documented)
	std::vector<pt> const line{{0, 0}, {2, 0}, {5, 0}};
	auto const v2 = voronoi_from_delaunay(line, delaunay(line));
	REQUIRE(v2.vertices.empty());
	REQUIRE(v2.edges.empty());
}

TEST_CASE("euclidean_mst: weight multiset matches the all-pairs Kruskal brute",
		  "[geometry]") {
	using atl::geometry::euclidean_mst;
	auto seed = GENERATE(take(25, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 40);
	std::int64_t const span = 2 + std::int64_t(rng() % 40);
	std::vector<pt> pts;
	for (std::size_t k = 0; k < n; ++k)
		pts.push_back({std::int64_t(rng() % std::uint64_t(span)), std::int64_t(rng() % std::uint64_t(span))});
	if (rng() % 3 == 0) // collinear regime
		for (auto &q : pts) q.y = q.x;
	auto const mst = euclidean_mst(pts);
	REQUIRE(mst.size() == n - 1);

	// brute Kruskal over all pairs
	struct pair_edge {
		atl::int128 w;
		std::uint32_t a, b;
	};

	std::vector<pair_edge> all;
	for (std::uint32_t i = 0; i < n; ++i)
		for (std::uint32_t j = i + 1; j < n; ++j)
			all.push_back({atl::int128(squared_distance(pts[i], pts[j])), i, j});
	std::ranges::sort(all, [](pair_edge const &l, pair_edge const &r) { return l.w < r.w; });
	atl::dsu uf(static_cast<atl::uint32>(n));
	std::vector<atl::int128> brute_weights, got_weights;
	for (auto const &e : all)
		if (uf.unite(e.a, e.b)) brute_weights.push_back(e.w);
	for (auto const &e : mst)
		got_weights.push_back(atl::int128(squared_distance(pts[e[0]], pts[e[1]])));
	std::ranges::sort(got_weights);
	REQUIRE(got_weights == brute_weights); // the MST weight multiset is unique
	// spanning: union of the returned edges connects everything
	atl::dsu uf2(static_cast<atl::uint32>(n));
	for (auto const &e : mst) uf2.unite(e[0], e[1]);
	REQUIRE(uf2.count() == 1);
}

TEST_CASE("closest_pair_between: vs the quadratic brute", "[geometry]") {
	using atl::geometry::closest_pair_between;
	using atl::geometry::closest_pair_between_indices;
	auto seed = GENERATE(take(25, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 30), m = 1 + std::size_t(rng() % 30);
	std::int64_t const span = 2 + std::int64_t(rng() % 30);
	std::vector<pt> a, b;
	for (std::size_t k = 0; k < n; ++k)
		a.push_back({std::int64_t(rng() % std::uint64_t(span)), std::int64_t(rng() % std::uint64_t(span))});
	for (std::size_t k = 0; k < m; ++k)
		b.push_back({std::int64_t(rng() % std::uint64_t(span)) + (rng() % 2 ? span : 0),
					 std::int64_t(rng() % std::uint64_t(span))});
	if (rng() % 4 == 0) // collinear union regime
		for (auto *v : {&a, &b})
			for (auto &q : *v) q.y = 0;
	auto const [i, j] = closest_pair_between_indices(a, b);
	REQUIRE(i < n);
	REQUIRE(j < m);
	atl::int128 brute = -1;
	for (auto const &qa : a)
		for (auto const &qb : b) {
			auto const g = atl::int128(squared_distance(qa, qb));
			if (brute < 0 || g < brute) brute = g;
		}
	REQUIRE(atl::int128(squared_distance(a[i], b[j])) == brute);
	auto const [pa, pb] = closest_pair_between(a, b);
	REQUIRE(atl::int128(squared_distance(pa, pb)) == brute);
}

TEST_CASE("closest_pair_between: shared positions and degenerates", "[geometry]") {
	using atl::geometry::closest_pair_between_indices;
	// a shared position across the sets: distance exactly 0
	std::vector<pt> const a{{5, 5}, {0, 0}}, b{{9, 9}, {5, 5}};
	auto const [i, j] = closest_pair_between_indices(a, b);
	REQUIRE(a[i] == b[j]);
	// singletons
	auto const [i2, j2] =
		closest_pair_between_indices(std::vector<pt>{{1, 2}}, std::vector<pt>{{4, 6}});
	REQUIRE(i2 == 0);
	REQUIRE(j2 == 0);
}

TEST_CASE("largest_empty_circle: vs a dense grid brute", "[geometry]") {
	using atl::geometry::largest_empty_circle;
	auto seed = GENERATE(take(10, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 3 + std::size_t(rng() % 12);
	std::vector<pt> pts;
	for (std::size_t k = 0; k < n; ++k)
		pts.push_back({std::int64_t(rng() % 40), std::int64_t(rng() % 40)});
	auto const lec = largest_empty_circle(pts);
	require_center_in_hull(pts, lec.center);
	// the radius really is the nearest-site distance at the center
	double nearest = 1e300;
	for (auto const &q : pts)
		nearest = (std::min)(nearest, std::hypot(lec.center.x - double(q.x),
												 lec.center.y - double(q.y)));
	REQUIRE(std::abs(nearest - lec.radius) <= 1e-6 * (1 + lec.radius));
	// no grid point inside the hull beats it (dense sample, with slack)
	auto const hull = convex_hull(pts);
	if (hull.size() >= 3) {
		for (double x = 0; x <= 40; x += 0.5)
			for (double y = 0; y <= 40; y += 0.5) {
				bool inside = true;
				for (std::size_t i = 0, j = hull.size() - 1; i < hull.size(); j = i++) {
					double const cx =
						(double(hull[i].x) - double(hull[j].x)) * (y - double(hull[j].y))
						- (double(hull[i].y) - double(hull[j].y)) * (x - double(hull[j].x));
					if (cx < 0) inside = false;
				}
				if (!inside) continue;
				double gap = 1e300;
				for (auto const &q : pts)
					gap = (std::min)(gap, std::hypot(x - double(q.x), y - double(q.y)));
				REQUIRE(gap <= lec.radius + 0.71); // grid pitch slack
			}
	}
	// degenerate: collinear -> largest gap midpoint; single point -> 0
	auto const seg = largest_empty_circle(std::vector<pt>{{0, 0}, {2, 0}, {10, 0}});
	REQUIRE(std::abs(seg.radius - 4.0) < 1e-12);
	REQUIRE(std::abs(seg.center.x - 6.0) < 1e-12);
	require_center_in_hull({{0, 0}, {2, 0}, {10, 0}}, seg.center);
	auto const single = largest_empty_circle(std::vector<pt>{{7, 7}});
	REQUIRE(single.radius == 0.0);
	require_center_in_hull({{7, 7}}, single.center);
}

TEST_CASE("largest_empty_circle: a thin hull whose circumcenter is distant away", "[geometry]") {
	// A thin triangle puts the circumcenter outside the hull by ~125000;
	// truncating rays by the hull span alone loses boundary candidates.
	using atl::geometry::largest_empty_circle;
	for (std::int64_t const shift : {std::int64_t(0), std::int64_t(1'000'000), std::int64_t(-7'000'000)}) {
		std::vector<pt> const sliver{{shift, shift}, {shift + 1000, shift}, {shift + 500, shift + 1}};
		auto const lec = largest_empty_circle(sliver);
		require_center_in_hull(sliver, lec.center);
		// the best center is on the base, a quarter of the way in from either end
		REQUIRE(std::abs(lec.radius - 250.0) < 0.01);
		REQUIRE(std::abs(lec.center.y - double(shift)) < 0.01);
		double nearest = 1e300;
		for (auto const &q : sliver)
			nearest = (std::min)(nearest, std::hypot(lec.center.x - double(q.x), lec.center.y - double(q.y)));
		REQUIRE(std::abs(nearest - lec.radius) <= 1e-6 * (1 + lec.radius));
	}
}

TEST_CASE("delaunay family: the 2^61 int64 bound is asserted at every entry", "[geometry]") {
	using p64 = atl::geometry::point<std::int64_t>;
	std::int64_t const bound = std::int64_t(1) << 61;
	std::vector<p64> const distant{{0, 0}, {1, 0}, {0, bound}};
	std::vector<p64> const nearby{{5, 5}};
	CHECK(atl_test::aborts([&] { (void)atl::geometry::delaunay(distant); }));
	CHECK(atl_test::aborts([&] { (void)atl::geometry::delaunay_edges(distant); }));
	CHECK(atl_test::aborts([&] { (void)atl::geometry::voronoi_from_delaunay(distant, delaunay_result{}); }));
	CHECK(atl_test::aborts([&] { (void)atl::geometry::closest_pair_between_indices(distant, distant); }));
	CHECK(atl_test::aborts([&] { (void)atl::geometry::euclidean_mst(distant); }));
	CHECK(atl_test::aborts([&] { (void)atl::geometry::closest_pair_between_indices(distant, nearby); }));
	CHECK(atl_test::aborts([&] { (void)atl::geometry::largest_empty_circle(distant); }));
	CHECK(atl_test::aborts([&] { (void)atl::geometry::closest_pair_between(distant, nearby); }));
	std::vector<p64> const ok{{0, 0}, {1, 0}, {0, bound - 1}};
	REQUIRE(atl::geometry::delaunay_edges(ok).size() == 3);
}

TEST_CASE("Delaunay and enclosing circles use bounded stack on deep inputs", "[geometry]") {
	std::vector<pt> points(100'000);
	for (std::size_t i = 0; i < points.size(); ++i) points[i] = {std::int64_t(i), 0};
	REQUIRE(delaunay(points).triangles.empty());
	REQUIRE(atl::geometry::delaunay_edges(points).size() + 1 == points.size());
	auto const c = atl::geometry::minimum_enclosing_circle(points);
	REQUIRE(std::abs(c.radius - 49999.5) < 1e-6);
	std::ranges::fill(points, pt{7, 7});
	REQUIRE(delaunay(points).triangles.empty());
	REQUIRE(atl::geometry::euclidean_mst(points).size() + 1 == points.size());
}

TEST_CASE("largest empty circle matches rational candidate enumeration", "[geometry]") {
	using W = atl::int128;

	struct equation {
		W a, b, c;
	};

	std::mt19937_64 rng(139);
	for (int trial = 0; trial < 80; ++trial) {
		std::vector<pt> points;
		for (int i = 0; i < 7; ++i) points.push_back({std::int64_t(rng() % 9) - 4, std::int64_t(rng() % 9) - 4});
		if (trial % 5 == 0)
			for (auto &q : points) q.y = 0;
		std::vector<equation> boundaries, bisectors;
		for (std::size_t i = 0; i < points.size(); ++i)
			for (std::size_t j = i + 1; j < points.size(); ++j) {
				auto const a = points[i], b = points[j];
				if (a == b) continue;
				W const dx = W(b.x) - a.x, dy = W(b.y) - a.y;
				equation const edge{-dy, dx, -dy * a.x + dx * a.y};
				bool has_left = false, has_right = false;
				for (auto const q : points) {
					W const side = edge.a * q.x + edge.b * q.y - edge.c;
					has_left |= side > 0;
					has_right |= side < 0;
				}
				if (!has_right) boundaries.push_back(edge);
				if (!has_left) boundaries.push_back({-edge.a, -edge.b, -edge.c});
				bisectors.push_back({2 * dx, 2 * dy, W(b.x) * b.x + W(b.y) * b.y - W(a.x) * a.x - W(a.y) * a.y});
			}
		W best_n = 0, best_d = 1;
		auto const consider = [&](W x, W y, W den) {
			if (den == 0) return;
			if (den < 0) {
				x = -x;
				y = -y;
				den = -den;
			}
			for (auto const e : boundaries)
				if (e.a * x + e.b * y < e.c * den) return;
			// Supporting lines alone do not bound a collinear hull longitudinally.
			auto const [minx, maxx] = std::ranges::minmax(points, {}, &pt::x);
			auto const [miny, maxy] = std::ranges::minmax(points, {}, &pt::y);
			if (x < W(minx.x) * den || x > W(maxx.x) * den || y < W(miny.y) * den || y > W(maxy.y) * den) return;
			W nearest = -1;
			for (auto const q : points) {
				W const dx = W(q.x) * den - x, dy = W(q.y) * den - y;
				W const gap = dx * dx + dy * dy;
				if (nearest < 0 || gap < nearest) nearest = gap;
			}
			W const d2 = den * den;
			if (nearest * best_d > best_n * d2) {
				best_n = nearest;
				best_d = d2;
			}
		};
		for (auto const a : points)
			for (auto const b : points) consider(W(a.x) + b.x, W(a.y) + b.y, 2);
		auto const cross_lines = [&](equation a, equation b) { consider(a.c * b.b - a.b * b.c, a.a * b.c - a.c * b.a, a.a * b.b - a.b * b.a); };
		for (auto const a : bisectors) {
			for (auto const b : boundaries) cross_lines(a, b);
			for (auto const b : bisectors) cross_lines(a, b);
		}
		auto const got = atl::geometry::largest_empty_circle(points);
		require_center_in_hull(points, got.center);
		REQUIRE(std::abs(got.radius * got.radius - double(best_n) / double(best_d)) < 1e-7);
	}
}

TEST_CASE("Delaunay accepts full int32 coordinate limits", "[geometry]") {
	using p32 = point<std::int32_t>;
	std::int32_t const low = (std::numeric_limits<std::int32_t>::min)();
	std::int32_t const high = (std::numeric_limits<std::int32_t>::max)();
	std::array const points{p32{low, low}, p32{high, low}, p32{high, high}, p32{low, high}, p32{0, 0}};
	auto const d = delaunay(points);
	REQUIRE(d.triangles.size() == 4);
	for (auto const t : d.triangles) {
		auto const promote = [&](atl::geometry::vertex_index i) { return pt{points[i].x, points[i].y}; };
		for (auto const q : points) REQUIRE(empty_disk_sign(promote(t[0]), promote(t[1]), promote(t[2]), pt{q.x, q.y}) <= 0);
	}
}

TEST_CASE("Delaunay certificate rejects duplicated coverage and wrong adjacency", "[geometry]") {
	std::vector<pt> const square{{0, 0}, {4, 0}, {0, 4}, {4, 4}};
	delaunay_result invalid;
	invalid.triangles = {{0, 1, 2}, {0, 1, 2}};
	invalid.neighbors.assign(2, {delaunay_result::no_neighbor, delaunay_result::no_neighbor, delaunay_result::no_neighbor});
	REQUIRE(!topology_certificate(square, invalid));
	auto valid = delaunay(square);
	REQUIRE(topology_certificate(square, valid));
	valid.neighbors.assign(2, {delaunay_result::no_neighbor, delaunay_result::no_neighbor, delaunay_result::no_neighbor});
	REQUIRE(!topology_certificate(square, valid));
}

TEST_CASE("Delaunay wide path: cocircular sites close to the int64 bound", "[geometry]") {
	std::int64_t const k = (std::int64_t(1) << 61) - 1;
	verify({{k - 100, k - 100}, {k, k - 100}, {k, k}, {k - 100, k}});
	std::vector<pt> const corners{{-k, -k}, {k, -k}, {k, k}, {-k, k}, {0, 0}};
	auto const d = delaunay(corners);
	REQUIRE(d.triangles.size() == 4);
	REQUIRE(topology_certificate(corners, d));
	for (auto const &t : d.triangles)
		REQUIRE(atl::geometry::exact{}.orientation(corners[t[0]], corners[t[1]], corners[t[2]]) > 0);
	auto const mst = atl::geometry::euclidean_mst(corners);
	REQUIRE(mst.size() == 4);
	for (auto const e : mst) REQUIRE((e[0] == 4 || e[1] == 4));
	auto const vor = voronoi_from_delaunay(corners, d);
	REQUIRE(vor.vertices.size() == 4);
	for (auto const v : vor.vertices) REQUIRE((std::isfinite(v.x) && std::isfinite(v.y)));
	auto const empty = atl::geometry::largest_empty_circle(corners);
	REQUIRE(std::isfinite(empty.radius));
	require_center_in_hull(corners, empty.center);
	REQUIRE(std::abs(empty.radius / double(k) - 1.0) < 1e-12);
	std::vector<pt> const a{{-k, 0}, {k, 0}}, b{{0, 0}};
	auto const indices = atl::geometry::closest_pair_between_indices(a, b);
	REQUIRE(indices[0] < a.size());
	REQUIRE(indices[1] == 0);
}

TEST_CASE("voronoi: ray directions keep close sites far from the origin", "[geometry]") {
	// near 2^60 converting each coordinate to double first erases the unit offsets; the ray must stay
	// perpendicular to its hull edge, checked with the exact integer edge vector
	std::int64_t const base = std::int64_t(1) << 60;
	std::vector<pt> const pts{{base, 0}, {base + 512, 0}, {base + 1, 1}};
	auto const d = delaunay(pts);
	REQUIRE(d.triangles.size() == 1);
	auto const vor = voronoi_from_delaunay(pts, d);
	std::size_t rays = 0;
	for (auto const &e : vor.edges) {
		if (!e.is_unbounded) continue;
		++rays;
		auto const a = pts[e.site_a], b = pts[e.site_b];
		double const ex = double(b.x - a.x), ey = double(b.y - a.y); // small exact differences
		REQUIRE(std::abs(e.direction.x * ex + e.direction.y * ey) <= 1e-9 * std::hypot(ex, ey) * std::hypot(e.direction.x, e.direction.y));
	}
	REQUIRE(rays == 3);
}
