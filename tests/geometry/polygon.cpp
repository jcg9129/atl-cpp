#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/triangulation.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::geometry::containment;
using atl::geometry::contains;
using atl::geometry::lattice_count_of;
using atl::geometry::point;
using atl::geometry::polygon_area;
using atl::geometry::polygon_area2;
using atl::geometry::polygon_centroid;

namespace {
	struct pt {
		std::int64_t x, y;
	};

	// raw wrap-around shoelace, deliberately not the fan the library sums
	atl::int128 brute_area2(std::vector<pt> const &poly) {
		atl::int128 sum = 0;
		for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
			sum += atl::int128(poly[j].x) * poly[i].y - atl::int128(poly[j].y) * poly[i].x;
		return sum;
	}

	bool brute_on_segment(pt const a, pt const b, pt const q) {
		if ((b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x) != 0) return false;
		return (std::min)(a.x, b.x) <= q.x && q.x <= (std::max)(a.x, b.x) && (std::min)(a.y, b.y) <= q.y
			&& q.y <= (std::max)(a.y, b.y);
	}

	// crossing parity via explicit rational ray intersections — a different
	// arithmetic route than the orientation tests under test
	int brute_contains(std::vector<pt> const &poly, pt const q) {
		for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
			if (brute_on_segment(poly[j], poly[i], q)) return 0;
		bool in = false;
		for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
			pt const u = poly[j], v = poly[i];
			if ((u.y > q.y) != (v.y > q.y)) {
				// x of the edge at height q.y exceeds q.x <=> num/den > 0
				std::int64_t const den = v.y - u.y;
				std::int64_t const num = (u.x - q.x) * den + (q.y - u.y) * (v.x - u.x);
				if ((num > 0) == (den > 0)) in = !in;
			}
		}
		return in ? 1 : -1;
	}

	template<atl::geometry::coordinate T>
	std::vector<point<T>> as_points(std::vector<pt> const &poly) {
		std::vector<point<T>> out;
		for (auto const p : poly) out.push_back({T(p.x), T(p.y)});
		return out;
	}

	void check_polygon(std::vector<pt> const &poly, std::int64_t const margin) {
		auto const p32 = as_points<std::int32_t>(poly);
		auto const p64 = as_points<std::int64_t>(poly);
		auto const pd = as_points<double>(poly);
		REQUIRE(atl::int128(polygon_area2(p32)) == brute_area2(poly));
		REQUIRE(atl::int128(polygon_area2(p64)) == brute_area2(poly));
		REQUIRE(polygon_area2(pd) == double(atl::int64(brute_area2(poly))));

		auto const [xlo, xhi] = std::ranges::minmax(poly, {}, &pt::x);
		auto const [ylo, yhi] = std::ranges::minmax(poly, {}, &pt::y);
		for (std::int64_t x = xlo.x - margin; x <= xhi.x + margin; ++x)
			for (std::int64_t y = ylo.y - margin; y <= yhi.y + margin; ++y) {
				int const expected = brute_contains(poly, {x, y});
				REQUIRE(int(contains(p32, point{std::int32_t(x), std::int32_t(y)})) == expected);
				REQUIRE(int(contains(p64, point{std::int64_t(x), std::int64_t(y)})) == expected);
				REQUIRE(int(contains(pd, point{double(x), double(y)})) == expected);
			}
	}
}

TEST_CASE("polygon: areas and containment on handcrafted shapes", "[geometry]") {
	// convex square, concave L, star with spikes, collinear boundary points
	check_polygon({{0, 0}, {4, 0}, {4, 4}, {0, 4}}, 2);
	check_polygon({{0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 5}, {0, 5}}, 2);
	check_polygon({{0, 0}, {8, 1}, {4, 3}, {7, 7}, {3, 5}, {1, 8}, {2, 4}, {-3, 3}}, 2);
	check_polygon({{0, 0}, {2, 0}, {4, 0}, {4, 4}, {2, 4}, {0, 4}, {0, 2}}, 2); // collinear edges
	check_polygon({{0, 0}, {6, 0}, {3, 1}}, 2);									// thin sliver
	check_polygon({{0, 0}, {4, 0}, {2, 6}}, 3);									// plain triangle, cw
	// orientation flips the sign, not the magnitude or containment
	std::vector<point<int>> const ccw{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
	std::vector<point<int>> const cw{{0, 4}, {4, 4}, {4, 0}, {0, 0}};
	REQUIRE(polygon_area2(ccw) == 32);
	REQUIRE(polygon_area2(cw) == -32);
	REQUIRE(polygon_area(ccw) == 16.0);
	REQUIRE(polygon_area(cw) == 16.0);
}

TEST_CASE("polygon: degenerate vertex counts", "[geometry]") {
	std::vector<point<int>> const none{};
	std::vector<point<int>> const one{{2, 3}};
	std::vector<point<int>> const two{{0, 0}, {4, 0}};
	REQUIRE(polygon_area2(none) == 0);
	REQUIRE(polygon_area2(one) == 0);
	REQUIRE(polygon_area2(two) == 0);
	REQUIRE(contains(none, point{0, 0}) == containment::outside);
	REQUIRE(contains(one, point{2, 3}) == containment::boundary);
	REQUIRE(contains(one, point{2, 4}) == containment::outside);
	REQUIRE(contains(two, point{2, 0}) == containment::boundary);
	REQUIRE(contains(two, point{2, 1}) == containment::outside);
}

namespace {
	// star-shaped around a center: strictly increasing angles keep it simple
	std::vector<pt> star_polygon(std::mt19937_64 &rng, pt &center_out) {
		pt const center{std::int64_t(rng() % 21) - 10, std::int64_t(rng() % 21) - 10};
		center_out = center;
		std::vector<pt> raw;
		for (int k = 0; k < 40; ++k) {
			pt const p{center.x + std::int64_t(rng() % 31) - 15,
					   center.y + std::int64_t(rng() % 31) - 15};
			if (p.x != center.x || p.y != center.y) raw.push_back(p);
		}
		// one point per direction from the center (keep the farthest), ordered by an exact integer angle
		// comparison: atan2 keys can compare inconsistently under excess precision (x87), which breaks the
		// strict weak ordering std::ranges::sort requires
		auto const upper = [&](pt const p) { return p.y > center.y || (p.y == center.y && p.x > center.x); };
		std::ranges::sort(raw, [&](pt const a, pt const b) {
			bool const is_upper_a = upper(a), is_upper_b = upper(b);
			if (is_upper_a != is_upper_b) return is_upper_a;
			std::int64_t const turn = (a.x - center.x) * (b.y - center.y) - (a.y - center.y) * (b.x - center.x);
			if (turn != 0) return turn > 0;
			return std::abs(a.x - center.x) + std::abs(a.y - center.y)
				 < std::abs(b.x - center.x) + std::abs(b.y - center.y);
		});
		std::vector<pt> poly;
		for (auto const p : raw) {
			if (!poly.empty()) {
				pt const &q = poly.back();
				std::int64_t const cx =
					(q.x - center.x) * (p.y - center.y) - (q.y - center.y) * (p.x - center.x);
				std::int64_t const dot =
					(q.x - center.x) * (p.x - center.x) + (q.y - center.y) * (p.y - center.y);
				if (cx == 0 && dot > 0) { // same direction: keep the farther
					poly.back() = p;
					continue;
				}
			}
			poly.push_back(p);
		}
		return poly;
	}

	std::vector<pt> star_polygon(std::mt19937_64 &rng) {
		pt center{};
		return star_polygon(rng, center);
	}
}

TEST_CASE("polygon: random star-shaped polygons vs rational oracle", "[geometry]") {
	auto seed = GENERATE(take(15, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const poly = star_polygon(rng);
	if (poly.size() >= 3) check_polygon(poly, 1);
}

TEST_CASE("convexity_of: handcrafted classification", "[geometry]") {
	using atl::geometry::convexity;
	using atl::geometry::convexity_of;
	using atl::geometry::winding;
	using gp = point<std::int64_t>;
	auto const cls = [](std::vector<gp> const &poly) { return convexity_of(poly); };

	// strict convex, both windings
	REQUIRE(cls({{0, 0}, {4, 0}, {2, 6}}) == convexity{true, true, winding::counterclockwise});
	REQUIRE(cls({{2, 6}, {4, 0}, {0, 0}}) == convexity{true, true, winding::clockwise});
	REQUIRE(cls({{0, 0}, {4, 0}, {4, 4}, {0, 4}}) == convexity{true, true, winding::counterclockwise});
	// weakly convex: collinear vertices
	REQUIRE(cls({{0, 0}, {2, 0}, {4, 0}, {4, 4}, {0, 4}}) == convexity{true, false, winding::counterclockwise});
	REQUIRE(cls({{0, 4}, {4, 4}, {4, 0}, {2, 0}, {0, 0}}) == convexity{true, false, winding::clockwise});
	// rejected shapes
	REQUIRE(cls({{0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 5}, {0, 5}}) == convexity{false, false, winding::none}); // concave L
	REQUIRE(cls({{0, 0}, {4, 4}, {4, 0}, {0, 4}}) == convexity{false, false, winding::none});				  // bowtie
	REQUIRE(cls({{0, 0}, {4, 0}}) == convexity{false, false, winding::none});								  // too few
	REQUIRE(cls({{0, 0}, {2, 2}, {5, 5}}) == convexity{false, false, winding::none});						  // all collinear
	REQUIRE(cls({{0, 0}, {4, 0}, {4, 0}, {4, 4}}) == convexity{false, false, winding::none});				  // repeated vertex
	REQUIRE(cls({{0, 0}, {4, 0}, {2, 1}, {4, 4}}) == convexity{false, false, winding::none});				  // reflex
	REQUIRE(cls({{0, 0}, {6, 0}, {3, 0}, {3, 4}}) == convexity{false, false, winding::none});				  // reversal spike
	// the classic gap of a bare turn-sign test: a DOUBLE-WOUND convex
	// sequence has all-left turns and must still be rejected
	REQUIRE(cls({{4, 0}, {2, 3}, {-2, 3}, {-4, 0}, {-2, -3}, {2, -3}, {4, 0}, {2, 3}, {-2, 3}, {-4, 0}, {-2, -3}, {2, -3}})
			== convexity{false, false, winding::none});
}

TEST_CASE("polygon: int64 area at scale, constructive", "[geometry]") {
	std::int64_t const base = std::int64_t(1) << 59;
	std::int64_t const w = 123'456'789, h = 987'654'321;
	std::vector<point<std::int64_t>> const rect{
		{base, base}, {base + w, base}, {base + w, base + h}, {base, base + h}};
	REQUIRE(polygon_area2(rect) == atl::int128(w) * atl::int128(h) * 2);
	REQUIRE(contains(rect, point{base + w / 2, base + h / 2}) == containment::inside);
	REQUIRE(contains(rect, point{base, base + h / 2}) == containment::boundary);
	REQUIRE(contains(rect, point{base - 1, base}) == containment::outside);
}

namespace {
	using gp = point<std::int64_t>;
}

TEST_CASE("normalized_polygon: handcrafted shapes", "[geometry]") {
	using atl::geometry::normalized_polygon;
	using atl::geometry::winding;
	using atl::geometry::winding_of;
	std::vector<gp> const square{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
	// midpoints on every edge, a duplicated corner, CW order
	std::vector<gp> const doctored{{0, 0}, {0, 2}, {0, 4}, {2, 4}, {4, 4}, {4, 4}, {4, 2}, {4, 0}, {2, 0}, {1, 0}};
	REQUIRE(winding_of(square) == winding::counterclockwise);
	REQUIRE(winding_of(doctored) == winding::clockwise);
	auto const got = normalized_polygon(doctored);
	REQUIRE(got == square);						   // first survivor (0,0) first, CCW restored
	REQUIRE(normalized_polygon(square) == square); // already normal: unchanged
	// a collinear RUN spanning the cyclic wrap
	std::vector<gp> const wrapped{{2, 0}, {3, 0}, {3, 3}, {0, 3}, {0, 0}, {1, 0}};
	auto const unwrapped = normalized_polygon(wrapped);
	REQUIRE(unwrapped == std::vector<gp>{{3, 0}, {3, 3}, {0, 3}, {0, 0}});
	// degenerate inputs: the deduplicated sequence, unchanged
	REQUIRE(normalized_polygon(std::vector<gp>{}) == std::vector<gp>{});
	REQUIRE(normalized_polygon(std::vector<gp>{{7, 7}}) == std::vector<gp>{{7, 7}});
	REQUIRE(normalized_polygon(std::vector<gp>{{7, 7}, {7, 7}, {7, 7}})
			== std::vector<gp>{{7, 7}});
	REQUIRE(normalized_polygon(std::vector<gp>{{1, 2}, {5, 6}})
			== std::vector<gp>{{1, 2}, {5, 6}});
	REQUIRE(normalized_polygon(std::vector<gp>{{1, 2}, {5, 6}, {5, 6}, {1, 2}})
			== std::vector<gp>{{1, 2}, {5, 6}});
}

TEST_CASE("normalized_polygon: doctored star polygons, full certificate", "[geometry]") {
	using atl::geometry::normalized_polygon;
	using atl::geometry::orientation;
	using atl::geometry::winding;
	using atl::geometry::winding_of;
	auto seed = GENERATE(take(20, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const raw_star = star_polygon(rng);
	std::vector<gp> base;
	for (auto const v : raw_star) base.push_back({v.x, v.y});
	if (base.size() < 3) return;
	// double the coordinates so edge midpoints are exact lattice points
	std::vector<gp> doctored;
	for (std::size_t i = 0; i < base.size(); ++i) {
		gp const v{base[i].x * 2, base[i].y * 2};
		gp const w{base[(i + 1) % base.size()].x * 2, base[(i + 1) % base.size()].y * 2};
		doctored.push_back(v);
		if (rng() % 2) doctored.push_back(v);								   // duplicate
		if (rng() % 2) doctored.push_back({(v.x + w.x) / 2, (v.y + w.y) / 2}); // midpoint
	}
	if (rng() % 2) std::ranges::reverse(doctored); // either winding
	auto const got = normalized_polygon(doctored);
	std::size_t const m = got.size();
	REQUIRE(m >= 3);
	REQUIRE(winding_of(got) == winding::counterclockwise);
	// no straight vertex survives
	for (std::size_t i = 0; i < m; ++i)
		REQUIRE(orientation(got[(i + m - 1) % m], got[i], got[(i + 1) % m]) != 0);
	// boundary preserved: same absolute area, vertices drawn from the input
	auto const a1 = polygon_area2(doctored), a2 = polygon_area2(got);
	REQUIRE((a1 < 0 ? -a1 : a1) == (a2 < 0 ? -a2 : a2));
	for (auto const v : got)
		REQUIRE(std::ranges::find(doctored, v) != doctored.end());
	// the first surviving input vertex stays first
	for (auto const v : doctored) {
		bool found = false;
		for (auto const g : got)
			if (g == v) found = true;
		if (found) {
			REQUIRE(got.front() == v);
			break;
		}
	}
	// idempotent
	REQUIRE(normalized_polygon(got) == got);
	// double parity
	std::vector<point<double>> dd;
	for (auto const v : doctored) dd.push_back({double(v.x), double(v.y)});
	auto const gd = normalized_polygon(dd);
	REQUIRE(gd.size() == m);
	for (std::size_t i = 0; i < m; ++i) {
		REQUIRE(gd[i].x == double(got[i].x));
		REQUIRE(gd[i].y == double(got[i].y));
	}
}

TEST_CASE("winding_of: corner orientation, exact where a shoelace sign rounds",
		  "[geometry]") {
	using atl::geometry::winding;
	using atl::geometry::winding_of;
	// duplicated extreme corner: nearest distinct neighbors decide
	REQUIRE(winding_of(std::vector<gp>{{5, 0}, {5, 0}, {8, 4}, {0, 4}})
			== winding::counterclockwise);
	REQUIRE(winding_of(std::vector<gp>{{5, 0}, {5, 0}, {0, 4}, {8, 4}}) == winding::clockwise);
	// degenerate: too few / zero-area corner
	REQUIRE(winding_of(std::vector<gp>{{0, 0}, {1, 1}}) == winding::none);
	REQUIRE(winding_of(std::vector<gp>{{0, 0}, {2, 2}, {4, 4}}) == winding::none);
	// double coordinates where the SHOELACE SIGN is lost to rounding but
	// the corner predicate is adaptive-exact: cross products near 1e32
	// with true doubled area -4
	double const big = 1e16;
	std::vector<point<double>> const sliver{{0.0, 0.0}, {big, big + 2}, {big + 2, big + 4}};
	REQUIRE(winding_of(sliver) == winding::clockwise);
	// parity with the exact integer verdict
	std::vector<gp> const sliver_int{{0, 0},
									 {std::int64_t(big), std::int64_t(big) + 2},
									 {std::int64_t(big) + 2, std::int64_t(big) + 4}};
	REQUIRE(winding_of(sliver_int) == winding::clockwise);
	REQUIRE(polygon_area2(sliver_int) == -4);
}

TEST_CASE("polygon_area2: winding-number weighted value on non-simple sequences",
		  "[geometry]") {
	// symmetric bowtie: lobes wound oppositely cancel exactly
	REQUIRE(polygon_area2(std::vector<gp>{{0, 0}, {2, 0}, {0, 2}, {2, 2}}) == 0);
	// doubly-wound triangle: twice the single traversal
	std::vector<gp> const once{{0, 0}, {4, 0}, {0, 4}};
	std::vector<gp> const twice{{0, 0}, {4, 0}, {0, 4}, {0, 0}, {4, 0}, {0, 4}};
	REQUIRE(polygon_area2(once) == 16);
	REQUIRE(polygon_area2(twice) == 32);
	// square with a square hole traversed through a cut: outer minus hole
	std::vector<gp> const holed{{0, 0}, {6, 0}, {6, 6}, {0, 6}, {0, 0}, {2, 2}, {2, 4}, {4, 4}, {4, 2}, {2, 2}};
	REQUIRE(polygon_area2(holed) == 2 * (36 - 4));
}

TEST_CASE("polygon_centroid: handcrafted and against a fan from the star's center", "[geometry]") {
	using lp = point<std::int64_t>; // the LIBRARY point (pt is the brute struct)
	// square: the center, either winding; anchored triangle exactness
	REQUIRE(polygon_centroid(std::vector<lp>{{0, 0}, {4, 0}, {4, 4}, {0, 4}}) == point{2.0, 2.0});
	REQUIRE(polygon_centroid(std::vector<lp>{{0, 4}, {4, 4}, {4, 0}, {0, 0}}) == point{2.0, 2.0});
	// triangle: the vertex average
	REQUIRE(polygon_centroid(std::vector<lp>{{0, 0}, {3, 0}, {0, 3}}) == point{1.0, 1.0});
	// L-shape as two rectangles: (8 * (2,1) + 8 * (1,4)) / 16 = (1.5, 2.5)
	REQUIRE(polygon_centroid(std::vector<lp>{{0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 6}, {0, 6}}) == point{1.5, 2.5});
	// NOT the vertex average: a dense run of collinear vertices on one side
	// must not move the lamina's balance point
	REQUIRE(polygon_centroid(std::vector<lp>{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {4, 4}, {0, 4}}) == point{2.0, 2.0});

	// random stars against an independent decomposition: signed triangles
	// (center, v_i, v_i+1) from the star's center, in exact integers, which
	// tile the polygon with signs for any anchor point
	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 30; ++it) {
		pt center{};
		auto const raw = star_polygon(rng, center);
		std::vector<lp> poly;
		for (auto const &v : raw) poly.push_back({v.x, v.y});
		if (poly.size() < 3 || polygon_area2(poly) == 0) continue;
		auto const got = polygon_centroid(poly);
		std::int64_t mx = 0, my = 0, total = 0; // 3 x moment and doubled area, exact
		for (std::size_t i = 0; i < raw.size(); ++i) {
			pt const u = raw[i], v = raw[(i + 1) % raw.size()];
			std::int64_t const a = (u.x - center.x) * (v.y - center.y) - (u.y - center.y) * (v.x - center.x);
			mx += a * (center.x + u.x + v.x);
			my += a * (center.y + u.y + v.y);
			total += a;
		}
		REQUIRE(total == std::int64_t(polygon_area2(poly)));
		REQUIRE(std::abs(got.x - double(mx) / double(3 * total)) <= 1e-9 * (1 + std::abs(got.x)));
		REQUIRE(std::abs(got.y - double(my) / double(3 * total)) <= 1e-9 * (1 + std::abs(got.y)));
	}

	// double parity
	using pd = point<double>;
	auto const c = polygon_centroid(std::vector<pd>{{0, 0}, {1, 0}, {1, 0.5}, {0, 0.5}});
	REQUIRE(std::abs(c.x - 0.5) <= 1e-12);
	REQUIRE(std::abs(c.y - 0.25) <= 1e-12);
	// at 2^61 converting each coordinate first would collapse the square to
	// zero area; the anchored exact differences keep it
	std::int64_t const distant = std::int64_t(1) << 61;
	auto const shifted = polygon_centroid(std::vector<lp>{{distant, distant}, {distant + 4, distant}, {distant + 4, distant + 4}, {distant, distant + 4}});
	REQUIRE(shifted == point{double(distant) + 2.0, double(distant) + 2.0});
}

TEST_CASE("polygon_centroid: against the triangulation oracle", "[geometry]") {
	using lp = point<std::int64_t>;
	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 30; ++it) {
		pt center{};
		auto const raw = star_polygon(rng, center);
		std::vector<lp> poly;
		for (auto const &v : raw) poly.push_back({v.x, v.y});
		if (poly.size() < 3 || polygon_area2(poly) == 0) continue;
		auto const got = polygon_centroid(poly);
		auto const tri = atl::geometry::triangulation_of(poly);
		double cx = 0, cy = 0, total = 0;
		for (auto const &t : tri.triangles) {
			double const a = double(atl::geometry::cross(poly[t[0]], poly[t[1]], poly[t[2]]));
			cx += a * (double(poly[t[0]].x) + double(poly[t[1]].x) + double(poly[t[2]].x));
			cy += a * (double(poly[t[0]].y) + double(poly[t[1]].y) + double(poly[t[2]].y));
			total += a;
		}
		REQUIRE(std::abs(got.x - cx / (3 * total)) <= 1e-9 * (1 + std::abs(got.x)));
		REQUIRE(std::abs(got.y - cy / (3 * total)) <= 1e-9 * (1 + std::abs(got.y)));
	}
}

TEST_CASE("lattice_count_of: handcrafted and against the containment brute",
		  "[geometry]") {
	using lp = point<std::int64_t>;
	using lc = atl::geometry::lattice_count<atl::int128>;
	// unit square: 4 boundary, 0 interior; 4x4: 16 boundary, 9 interior
	REQUIRE(lattice_count_of(std::vector<lp>{{0, 0}, {1, 0}, {1, 1}, {0, 1}})
			== lc{4, 0});
	REQUIRE(lattice_count_of(std::vector<lp>{{0, 0}, {4, 0}, {4, 4}, {0, 4}})
			== lc{16, 9});
	// right triangle (0,0)(4,0)(0,4): B = 4+4+gcd(4,4) = 12, I = 3
	REQUIRE(lattice_count_of(std::vector<lp>{{0, 0}, {4, 0}, {0, 4}}) == lc{12, 3});
	// winding-independent
	REQUIRE(lattice_count_of(std::vector<lp>{{0, 4}, {4, 0}, {0, 0}}) == lc{12, 3});
	// collinear (straight) vertices split an edge but keep the counts
	REQUIRE(lattice_count_of(std::vector<lp>{{0, 0}, {2, 0}, {4, 0}, {4, 4}, {0, 4}})
			== lc{16, 9});

	// random stars vs an exhaustive containment scan (exact: contains()
	// reports boundary and inside separately)
	auto seed = GENERATE(take(10, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 25; ++it) {
		auto const raw = star_polygon(rng);
		std::vector<lp> poly;
		for (auto const &v : raw) poly.push_back({v.x, v.y});
		if (poly.size() < 3 || polygon_area2(poly) == 0) continue;
		std::int64_t lo_x = poly[0].x, hi_x = poly[0].x, lo_y = poly[0].y,
					 hi_y = poly[0].y;
		for (auto const &v : poly) {
			lo_x = (std::min)(lo_x, v.x);
			hi_x = (std::max)(hi_x, v.x);
			lo_y = (std::min)(lo_y, v.y);
			hi_y = (std::max)(hi_y, v.y);
		}
		atl::int128 boundary = 0, interior = 0;
		for (std::int64_t x = lo_x; x <= hi_x; ++x)
			for (std::int64_t y = lo_y; y <= hi_y; ++y)
				switch (atl::geometry::contains(poly, lp{x, y})) {
				case atl::geometry::containment::boundary:
					++boundary;
					break;
				case atl::geometry::containment::inside:
					++interior;
					break;
				default:
					break;
				}
		auto const got = lattice_count_of(poly);
		REQUIRE(got.boundary == boundary);
		REQUIRE(got.interior == interior);
	}
}

namespace {
	// independent convexity oracle in exact integers: a convex simple polygon
	// has no repeated vertex, edges meeting only at shared endpoints, and every
	// vertex on the same closed side of every edge
	std::int64_t brute_cross(pt const o, pt const a, pt const b) {
		return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
	}

	bool brute_segments_meet(pt const a, pt const b, pt const c, pt const d) {
		std::int64_t const d1 = brute_cross(a, b, c), d2 = brute_cross(a, b, d);
		std::int64_t const d3 = brute_cross(c, d, a), d4 = brute_cross(c, d, b);
		if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
		return brute_on_segment(a, b, c) || brute_on_segment(a, b, d) || brute_on_segment(c, d, a)
			|| brute_on_segment(c, d, b);
	}

	atl::geometry::convexity brute_convexity(std::vector<pt> const &poly) {
		using atl::geometry::winding;
		atl::geometry::convexity const rejected{false, false, winding::none};
		std::size_t const n = poly.size();
		if (n < 3) return rejected;
		for (std::size_t i = 0; i < n; ++i)
			for (std::size_t j = i + 1; j < n; ++j)
				if (poly[i].x == poly[j].x && poly[i].y == poly[j].y) return rejected;
		for (std::size_t i = 0; i < n; ++i)
			for (std::size_t j = i + 1; j < n; ++j) {
				pt const a = poly[i], b = poly[(i + 1) % n], c = poly[j], d = poly[(j + 1) % n];
				bool const is_next = j == i + 1, is_wrap = i == 0 && j == n - 1;
				if (is_next) { // share b == c only: no collinear fold-back
					if (brute_on_segment(a, b, d) || brute_on_segment(c, d, a)) return rejected;
				} else if (is_wrap) { // share a == d only
					if (brute_on_segment(a, b, c) || brute_on_segment(c, d, b)) return rejected;
				} else if (brute_segments_meet(a, b, c, d))
					return rejected;
			}
		bool has_left = false, has_right = false, has_straight = false;
		for (std::size_t i = 0; i < n; ++i)
			for (std::size_t k = 0; k < n; ++k) {
				std::int64_t const side = brute_cross(poly[i], poly[(i + 1) % n], poly[k]);
				has_left = has_left || side > 0;
				has_right = has_right || side < 0;
			}
		if (has_left == has_right) return rejected; // both: not convex; neither: zero area
		for (std::size_t i = 0; i < n; ++i)
			if (brute_cross(poly[(i + n - 1) % n], poly[i], poly[(i + 1) % n]) == 0) has_straight = true;
		return {true, !has_straight, has_left ? winding::counterclockwise : winding::clockwise};
	}

	// a random convex polygon: random edge vectors sorted by angle sum to zero
	// and wind once (opposite vectors cancel, so the sum is closed)
	std::vector<pt> random_convex(std::mt19937_64 &rng) {
		std::vector<pt> edges;
		std::size_t const half = static_cast<std::size_t>(2 + rng() % 6);
		for (std::size_t k = 0; k < half; ++k) {
			pt const e{std::int64_t(rng() % 13) - 6, std::int64_t(rng() % 13) - 6};
			if (e.x == 0 && e.y == 0) continue;
			edges.push_back(e);
			edges.push_back({-e.x, -e.y});
		}

		struct keyed_direction {
			pt value;
			double angle;
		};

		std::vector<keyed_direction> keyed;
		for (auto const value : edges) keyed.push_back({value, std::atan2(double(value.y), double(value.x))});
		std::ranges::sort(keyed, {}, &keyed_direction::angle);
		for (std::size_t i = 0; i < edges.size(); ++i) edges[i] = keyed[i].value;
		// merge parallel same-direction edges so no vertex is straight
		std::vector<pt> merged;
		for (auto const e : edges)
			if (!merged.empty() && brute_cross({0, 0}, merged.back(), e) == 0
				&& merged.back().x * e.x + merged.back().y * e.y > 0)
				merged.back() = {merged.back().x + e.x, merged.back().y + e.y};
			else
				merged.push_back(e);
		std::vector<pt> poly;
		pt at{std::int64_t(rng() % 21) - 10, std::int64_t(rng() % 21) - 10};
		for (auto const e : merged) {
			poly.push_back(at);
			at = {at.x + e.x, at.y + e.y};
		}
		return poly;
	}
}

TEST_CASE("convexity_of: random sequences and convex polygons vs the exact brute", "[geometry]") {
	using atl::geometry::convexity;
	using atl::geometry::convexity_of;
	using atl::geometry::winding;
	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const check = [](std::vector<pt> const &poly) {
		auto const expected = brute_convexity(poly);
		REQUIRE(convexity_of(as_points<std::int64_t>(poly)) == expected);
		REQUIRE(convexity_of(as_points<std::int16_t>(poly)) == expected);
		REQUIRE(convexity_of(as_points<double>(poly)) == expected);
		REQUIRE(convexity_of(as_points<double>(poly), atl::geometry::fast{}) == expected);
	};
	for (int it = 0; it < 200; ++it) {
		// small random sequences on a 4 x 4 grid: duplicates, collinear runs,
		// reversals, self-intersections
		std::vector<pt> raw(static_cast<std::size_t>(3 + rng() % 5));
		for (auto &p : raw) p = {std::int64_t(rng() % 4), std::int64_t(rng() % 4)};
		check(raw);
		auto poly = random_convex(rng);
		if (poly.size() < 3) continue;
		REQUIRE(brute_convexity(poly) == convexity{true, true, winding::counterclockwise});
		check(poly);
		std::ranges::reverse(poly);
		check(poly);
		// an edge midpoint (exact after doubling) is a straight vertex
		std::vector<pt> dented;
		for (std::size_t i = 0; i < poly.size(); ++i) {
			pt const v{2 * poly[i].x, 2 * poly[i].y};
			pt const w{2 * poly[(i + 1) % poly.size()].x, 2 * poly[(i + 1) % poly.size()].y};
			dented.push_back(v);
			if (i == 0) dented.push_back({(v.x + w.x) / 2, (v.y + w.y) / 2});
		}
		REQUIRE(brute_convexity(dented) == convexity{true, false, winding::clockwise});
		check(dented);
		// winding twice around passes every local test but is not simple
		std::vector<pt> twice = poly;
		twice.insert(twice.end(), poly.begin(), poly.end());
		check(twice);
	}
}

TEST_CASE("polygon: exactness at the limits of each coordinate type", "[geometry]") {
	using atl::geometry::convexity;
	using atl::geometry::convexity_of;
	using atl::geometry::normalized_polygon;
	using atl::geometry::winding;
	using atl::geometry::winding_of;
	// The full int16 square needs the int64 difference product type.
	using p8 = point<atl::int16>;
	std::vector<p8> const full8{{-32768, -32768}, {32767, -32768}, {32767, 32767}, {-32768, 32767}};
	REQUIRE(polygon_area2(full8) == 2LL * 65535 * 65535);
	REQUIRE(lattice_count_of(full8) == atl::geometry::lattice_count<atl::int64>{4LL * 65535, 65534LL * 65534});
	REQUIRE(contains(full8, p8{32767, 32767}) == containment::boundary);
	REQUIRE(contains(full8, p8{32766, -32767}) == containment::inside);
	REQUIRE(convexity_of(full8) == convexity{true, true, winding::counterclockwise});
	REQUIRE(polygon_centroid(full8) == point{-0.5, -0.5});

	// int64 just inside |c| < 2^62: a thin sliver whose turns need every bit
	using p64 = point<std::int64_t>;
	std::int64_t const m = (std::int64_t(1) << 62) - 1;
	std::vector<p64> const sliver{{-m, -m}, {m, m - 1}, {m, m}};
	REQUIRE(polygon_area2(sliver) == atl::int128(2) * m);
	REQUIRE(winding_of(sliver) == winding::counterclockwise);
	REQUIRE(convexity_of(sliver) == convexity{true, true, winding::counterclockwise});
	REQUIRE(normalized_polygon(sliver) == sliver);
	REQUIRE(contains(sliver, p64{0, 0}) == containment::boundary); // (0, 0) lies on the edge from (-m, -m) to (m, m)
	REQUIRE(contains(sliver, p64{m, m - 1}) == containment::boundary);
	REQUIRE(contains(sliver, p64{m - 1, m - 1}) == containment::boundary);
	REQUIRE(contains(sliver, p64{0, -1}) == containment::outside);
	std::vector<p64> const box{{-m, -m}, {m, -m}, {m, m}, {-m, m}};
	atl::int128 const side = atl::int128(2) * m;
	REQUIRE(polygon_area2(box) == 2 * side * side); // 2^127 - 2^66 + 8, just below the int128 limit
	REQUIRE(lattice_count_of(box) == atl::geometry::lattice_count<atl::int128>{4 * side, (side - 1) * (side - 1)});
	REQUIRE(contains(box, p64{m - 1, 1 - m}) == containment::inside);

	// float: exact signs on non-dyadic coordinates
	using pf = point<float>;
	std::vector<pf> const tenths{{0.1f, 0.1f}, {0.7f, 0.1f}, {0.7f, 0.3f}, {0.1f, 0.3f}};
	REQUIRE(contains(tenths, pf{0.4f, 0.1f}) == containment::boundary);
	REQUIRE(contains(tenths, pf{0.4f, 0.2f}) == containment::inside);
	REQUIRE(winding_of(tenths) == winding::counterclockwise);
}

TEST_CASE("polygon_area2: partial sums wrap, the result stays exact", "[geometry]") {
	// a triangle wound 40000 times one way, then 40000 times back: the fan's
	// partial sums exceed the area of one traversal; the result is 0
	using p8 = point<atl::int16>;
	std::array<p8, 3> const tri{p8{-128, -128}, p8{127, -128}, p8{-128, 127}};
	std::vector<p8> wound;
	for (int k = 0; k < 40000; ++k) wound.insert(wound.end(), tri.begin(), tri.end());
	for (int k = 0; k < 40000; ++k) wound.insert(wound.end(), tri.rbegin(), tri.rend());
	REQUIRE(polygon_area2(wound) == 0);
	// int64 at the bound: a square wound twice forward and twice back, whose
	// partial sums pass 2^128
	using p64 = point<std::int64_t>;
	std::int64_t const m = (std::int64_t(1) << 62) - 1;
	std::array<p64, 4> const square{p64{-m, -m}, p64{m, -m}, p64{m, m}, p64{-m, m}};
	std::vector<p64> loops;
	for (int k = 0; k < 2; ++k) loops.insert(loops.end(), square.begin(), square.end());
	for (int k = 0; k < 2; ++k) loops.insert(loops.end(), square.rbegin(), square.rend());
	REQUIRE(polygon_area2(loops) == 0);
}

TEST_CASE("polygon: degenerate inputs", "[geometry]") {
	using atl::geometry::convexity;
	using atl::geometry::convexity_of;
	using atl::geometry::normalized_polygon;
	using atl::geometry::winding;
	using atl::geometry::winding_of;
	std::vector<gp> const same{{3, 3}, {3, 3}, {3, 3}, {3, 3}};
	std::vector<gp> const line{{0, 0}, {1, 1}, {2, 2}, {5, 5}};
	REQUIRE(polygon_area2(same) == 0);
	REQUIRE(polygon_area2(line) == 0);
	REQUIRE(winding_of(same) == winding::none);
	REQUIRE(winding_of(line) == winding::none);
	REQUIRE(convexity_of(same) == convexity{false, false, winding::none});
	REQUIRE(convexity_of(line) == convexity{false, false, winding::none});
	REQUIRE(normalized_polygon(same) == std::vector<gp>{{3, 3}});
	REQUIRE(contains(same, gp{3, 3}) == containment::boundary);
	REQUIRE(contains(same, gp{3, 4}) == containment::outside);
	REQUIRE(contains(line, gp{4, 4}) == containment::boundary);
	REQUIRE(contains(line, gp{4, 3}) == containment::outside);
}

TEST_CASE("polygon: a long thin polygon of 1e6 vertices", "[geometry]") {
	using atl::geometry::convexity;
	using atl::geometry::convexity_of;
	using atl::geometry::normalized_polygon;
	using atl::geometry::winding;
	using atl::geometry::winding_of;
	// a 1 by (n - 1) rectangle whose long sides carry every lattice point
	std::int64_t const n = 500'000;
	std::vector<gp> thin;
	for (std::int64_t x = 0; x < n; ++x) thin.push_back({x, 0});
	for (std::int64_t x = n - 1; x >= 0; --x) thin.push_back({x, 1});
	REQUIRE(polygon_area2(thin) == 2 * (n - 1));
	REQUIRE(winding_of(thin) == winding::counterclockwise);
	REQUIRE(convexity_of(thin) == convexity{true, false, winding::counterclockwise});
	REQUIRE(normalized_polygon(thin) == std::vector<gp>{{0, 0}, {n - 1, 0}, {n - 1, 1}, {0, 1}});
	REQUIRE(lattice_count_of(thin) == atl::geometry::lattice_count<atl::int128>{2 * n, 0});
	REQUIRE(contains(thin, gp{n / 2, 0}) == containment::boundary);
	REQUIRE(contains(thin, gp{n / 2, 2}) == containment::outside);
	REQUIRE(polygon_centroid(thin) == point{double(n - 1) / 2, 0.5});
}

TEST_CASE("polygon: asserted preconditions", "[geometry]") {
	using lp = point<std::int64_t>;
	std::int64_t const bound = std::int64_t(1) << 62;
	CHECK(atl_test::aborts([] { (void)lattice_count_of(std::vector<lp>{{0, 0}, {1, 1}, {2, 2}}); })); // zero area
	CHECK(atl_test::aborts([] { (void)lattice_count_of(std::vector<lp>{{0, 0}, {1, 1}}); }));
	CHECK(atl_test::aborts([] { (void)polygon_centroid(std::vector<lp>{{0, 0}, {4, 0}}); }));
	CHECK(atl_test::aborts([] { (void)polygon_centroid(std::vector<lp>{{0, 0}, {4, 0}, {8, 0}}); }));
	CHECK(atl_test::aborts([] { (void)atl::geometry::normalized_polygon(std::vector<lp>{{0, 0}, {4, 0}, {8, 0}}); }));
	CHECK(atl_test::aborts([&] { (void)polygon_area2(std::vector<lp>{{0, 0}, {bound, 0}, {0, 1}}); }));
	CHECK(atl_test::aborts([&] { (void)contains(std::vector<lp>{{0, 0}, {4, 0}, {0, 4}}, lp{bound, 0}); }));
	CHECK(atl_test::aborts([&] { (void)atl::geometry::convexity_of(std::vector<lp>{{0, 0}, {bound, 0}, {0, 1}}); }));
}

TEST_CASE("polygon_centroid: invalid moment magnitude is asserted", "[geometry]") {
	CHECK(atl_test::aborts([] {
		(void)polygon_centroid(std::vector<point<double>>{{0, 0}, {1e150, 0}, {1e150, 1e150}, {0, 1e150}});
	}));
}

TEST_CASE("convexity_of: hull classification in both windings", "[geometry]") {
	using atl::geometry::convexity;
	using atl::geometry::convexity_of;
	using atl::geometry::winding;
	std::mt19937_64 rng(20260823);
	for (int iteration = 0; iteration < 100; ++iteration) {
		std::vector<gp> cloud(5 + static_cast<std::size_t>(rng() % 50));
		for (auto &p : cloud) p = {static_cast<std::int64_t>(rng() % 41) - 20, static_cast<std::int64_t>(rng() % 41) - 20};
		auto const strict_hull = atl::geometry::convex_hull(cloud);
		auto const keep_hull = atl::geometry::convex_hull(cloud, atl::geometry::keep_collinear);
		if (strict_hull.size() < 3) continue;
		REQUIRE(convexity_of(strict_hull) == convexity{true, true, winding::counterclockwise});
		auto reversed_hull = strict_hull;
		std::ranges::reverse(reversed_hull);
		REQUIRE(convexity_of(reversed_hull) == convexity{true, true, winding::clockwise});
		bool const has_straight = keep_hull.size() > strict_hull.size();
		REQUIRE(convexity_of(keep_hull) == convexity{true, !has_straight, winding::counterclockwise});
		auto dented = strict_hull;
		// An exact midpoint makes a straight vertex; rounding one can change its side.
		if (dented.size() >= 4 && ((dented[0].x + dented[2].x) & 1) == 0 && ((dented[0].y + dented[2].y) & 1) == 0) {
			dented[1] = {(dented[0].x + dented[2].x) / 2, (dented[0].y + dented[2].y) / 2};
			REQUIRE(convexity_of(dented) == convexity{true, false, winding::counterclockwise});
		}
		std::vector<point<double>> hd;
		for (auto const p : strict_hull) hd.push_back({static_cast<double>(p.x), static_cast<double>(p.y)});
		REQUIRE(convexity_of(hd) == convexity_of(strict_hull));
	}
}

TEST_CASE("normalized_polygon: keep-collinear hull normalizes to the strict hull", "[geometry]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::vector<gp> pts;
	for (int k = 0; k < 60; ++k) pts.push_back({static_cast<std::int64_t>(rng() % 13) - 6, static_cast<std::int64_t>(rng() % 13) - 6});
	auto const keep = atl::geometry::convex_hull(pts, atl::geometry::keep_collinear);
	if (keep.size() < 3 || polygon_area2(keep) == 0) return;
	REQUIRE(atl::geometry::normalized_polygon(keep) == atl::geometry::convex_hull(pts));
}
