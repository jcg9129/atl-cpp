#include <atl/geometry/angular_sort.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using atl::geometry::fast;
using atl::geometry::point;
using atl::geometry::polar_angle_less;
using atl::geometry::sort_by_polar_angle;

TEST_CASE("polar_angle_less: agrees with atan2 on the exhaustive small grid", "[geometry]") {
	std::vector<point<int>> grid;
	for (int x = -3; x <= 3; ++x)
		for (int y = -3; y <= 3; ++y) grid.push_back({x, y});
	std::int64_t mismatches = 0;
	for (auto const a : grid)
		for (auto const b : grid) {
			// C atan2 encodes the exact convention: (-pi, pi], zero at +x
			// axis (and for the zero vector), pi on the -x axis
			double const aa = std::atan2(double(a.y), double(a.x));
			double const ab = std::atan2(double(b.y), double(b.x));
			bool const less = polar_angle_less{}(a, b);
			if (std::abs(aa - ab) > 1e-9) {
				if (less != (aa < ab)) ++mismatches;
			} else if (less || polar_angle_less{}(b, a)) {
				++mismatches; // same direction: must be equivalent
			}
			point<double> const da{double(a.x), double(a.y)}, db{double(b.x), double(b.y)};
			if (polar_angle_less{}(da, db) != less) ++mismatches;
			if (polar_angle_less{fast{}}(da, db) != less) ++mismatches;
		}
	REQUIRE(mismatches == 0);
}

TEST_CASE("polar_angle_less: strict weak ordering on axes, origin and diagonals", "[geometry]") {
	std::vector<point<int>> const pts{{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {2, 0}, {-3, 0}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1}, {3, 1}, {1, 3}, {-3, 1}, {-1, -3}, {2, 2}, {0, 2}};
	for (auto const a : pts)
		for (auto const b : pts) {
			REQUIRE(!(polar_angle_less{}(a, b) && polar_angle_less{}(b, a))); // antisymmetry
			for (auto const c : pts)										  // transitivity
				if (polar_angle_less{}(a, b) && polar_angle_less{}(b, c)) REQUIRE(polar_angle_less{}(a, c));
		}
	// spot the convention directly
	REQUIRE(polar_angle_less{}(point{1, -1}, point{1, 0}));	  // -pi/4 < 0
	REQUIRE(polar_angle_less{}(point{1, 0}, point{1, 1}));	  // 0 < pi/4
	REQUIRE(polar_angle_less{}(point{1, 1}, point{-1, 0}));	  // pi/4 < pi
	REQUIRE(polar_angle_less{}(point{-1, -1}, point{1, -1})); // -3pi/4 < -pi/4
	REQUIRE(!polar_angle_less{}(point{-2, 0}, point{-5, 0})); // both exactly pi
	REQUIRE(!polar_angle_less{}(point{2, 2}, point{3, 3}));	  // same ray: equivalent
	REQUIRE(!polar_angle_less{}(point{3, 3}, point{2, 2}));
	REQUIRE(polar_angle_less{}(point{0, 0}, point{0, 1}));	// zero vector at angle 0
	REQUIRE(!polar_angle_less{}(point{0, 0}, point{1, 0})); // ... equivalent to +x axis
}

TEST_CASE("polar_angle_less: exact policy certifies ulp-close directions", "[geometry]") {
	// directions one ulp apart at 1e6 scale: the cross product magnitude
	// (~1e-4) sits below the naive rounding noise of 1e12-sized products
	double const x = 1e6, y = 1e6;
	point<double> const a{x, y};
	point<double> const b{x, std::nextafter(y, 2e6)};
	REQUIRE(polar_angle_less{}(a, b));
	REQUIRE(!polar_angle_less{}(b, a));
	point<double> const c{x, std::nextafter(y, 0.0)};
	REQUIRE(polar_angle_less{}(c, a));
	REQUIRE(!polar_angle_less{}(a, c));
}

TEST_CASE("polar_angle_less: int64 magnitudes near the bound", "[geometry]") {
	std::int64_t const big = std::int64_t(1) << 61;
	point<std::int64_t> const a{big, big - 1}, b{big, big}, c{big - 1, big};
	REQUIRE(polar_angle_less{}(a, b)); // slopes (big-1)/big < 1 < big/(big-1)
	REQUIRE(polar_angle_less{}(b, c));
	REQUIRE(polar_angle_less{}(a, c));
	REQUIRE(!polar_angle_less{}(c, b));
}

TEST_CASE("sort_by_polar_angle: random points end up in atan2 order", "[geometry]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<point<std::int64_t>> pts(300);
	for (auto &p : pts)
		p = {std::int64_t(rng() % 2001) - 1000, std::int64_t(rng() % 2001) - 1000};
	auto sorted = pts;
	sort_by_polar_angle(sorted);
	REQUIRE(std::ranges::is_sorted(sorted, [](auto const a, auto const b) {
		return polar_angle_less{}(a, b);
	}));
	// non-decreasing atan2 confirms the convention end to end
	for (std::size_t i = 0; i + 1 < sorted.size(); ++i) {
		double const cur = std::atan2(double(sorted[i].y), double(sorted[i].x));
		double const nxt = std::atan2(double(sorted[i + 1].y), double(sorted[i + 1].x));
		REQUIRE(cur <= nxt + 1e-12);
	}
	// same multiset of points
	auto expected = pts;
	std::ranges::sort(expected);
	auto got = sorted;
	std::ranges::sort(got);
	REQUIRE(got == expected);
}

namespace {
	using pt = point<std::int64_t>;
}

TEST_CASE("polar_angle_order: stable argsort vs the stable_sort oracle", "[geometry]") {
	using atl::geometry::polar_angle_less;
	using atl::geometry::polar_angle_order;
	auto seed = GENERATE(take(20, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = static_cast<std::size_t>(rng() % 300);
	std::vector<pt> pts(n);
	for (auto &q : pts) {
		if (rng() % 3 == 0) { // same-ray tie classes, zero vectors included
			std::int64_t const m = std::int64_t(rng() % 4);
			q = {m * (std::int64_t(rng() % 7) - 3), m * (std::int64_t(rng() % 7) - 3)};
		} else {
			q = {std::int64_t(rng() % 41) - 20, std::int64_t(rng() % 41) - 20};
		}
	}
	auto const order = polar_angle_order(pts);
	REQUIRE(order.size() == n);
	// exactly the stable argsort
	std::vector<atl::geometry::vertex_index> expected(n);
	for (std::size_t i = 0; i < n; ++i) expected[i] = atl::geometry::vertex_index(i);
	std::ranges::stable_sort(expected, [&](auto const i, auto const j) {
		return polar_angle_less{}(pts[i], pts[j]);
	});
	REQUIRE(order == expected);
	// and consistent with the in-place sort's value order
	auto sorted = pts;
	atl::geometry::sort_by_polar_angle(sorted);
	for (std::size_t k = 0; k < n; ++k) {
		REQUIRE(!polar_angle_less{}(pts[order[k]], sorted[k]));
		REQUIRE(!polar_angle_less{}(sorted[k], pts[order[k]]));
	}
	// double parity
	std::vector<point<double>> pd;
	for (auto const q : pts) pd.push_back({double(q.x), double(q.y)});
	REQUIRE(polar_angle_order(pd) == order);
}

TEST_CASE("polar_angle_less: int64 over the whole range, int32 parity", "[geometry]") {
	// integral coordinates take the two-point cross, exact for every int64
	// (no 2^62 bound): compare against the long-double atan2 order on
	// directions far apart, and against the int32 instantiation
	auto seed = GENERATE(take(4, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	using p64 = point<std::int64_t>;
	std::int64_t const top = (std::numeric_limits<std::int64_t>::max)(), low = (std::numeric_limits<std::int64_t>::min)();
	std::vector<p64> const corners{{top, 0}, {top, top}, {0, top}, {low, top}, {low, 0}, {low, low}, {0, low}, {top, low}};
	for (std::size_t i = 0; i < corners.size(); ++i)
		for (std::size_t j = 0; j < corners.size(); ++j) {
			auto const angle = [](p64 const q) { return std::atan2((long double)q.y, (long double)q.x); };
			REQUIRE(polar_angle_less{}(corners[i], corners[j]) == (angle(corners[i]) < angle(corners[j])));
		}
	std::vector<point<std::int32_t>> narrow(300);
	for (auto &q : narrow) q = {std::int32_t(rng()), std::int32_t(rng())};
	std::vector<p64> wide;
	for (auto const q : narrow) wide.push_back({q.x, q.y});
	REQUIRE(atl::geometry::polar_angle_order(narrow) == atl::geometry::polar_angle_order(wide));
}

TEST_CASE("polar angle sort: degenerate inputs", "[geometry]") {
	using atl::geometry::polar_angle_order;
	using order = std::vector<atl::geometry::vertex_index>;
	// empty and one point
	std::vector<pt> none;
	sort_by_polar_angle(none);
	REQUIRE(none.empty());
	REQUIRE(polar_angle_order(none).empty());
	std::vector<pt> one{{5, -3}};
	sort_by_polar_angle(one);
	REQUIRE(one == std::vector<pt>{{5, -3}});
	REQUIRE(polar_angle_order(one) == order{0});
	// all equal: one ray, input order kept by the argsort
	std::vector<pt> const same(4, pt{2, 7});
	REQUIRE(polar_angle_order(same) == order{0, 1, 2, 3});
	// zero vectors sort with the positive x axis, in input order
	std::vector<pt> const zeros{{0, 1}, {0, 0}, {3, 0}, {0, 0}, {0, -1}};
	REQUIRE(polar_angle_order(zeros) == order{4, 1, 2, 3, 0});
	// all collinear through the origin: two rays, the negative one at pi (last)
	std::vector<pt> const collinear{{-2, -4}, {1, 2}, {3, 6}, {-1, -2}, {0, 0}};
	REQUIRE(polar_angle_order(collinear) == order{0, 3, 4, 1, 2});
	std::vector<pt> const axis{{-1, 0}, {4, 0}, {-9, 0}, {0, 0}, {2, 0}};
	REQUIRE(polar_angle_order(axis) == order{1, 3, 4, 0, 2});
	// the same on doubles, with the fast policy too
	std::vector<point<double>> axis_double;
	for (auto const q : axis) axis_double.push_back({double(q.x), double(q.y)});
	REQUIRE(polar_angle_order(axis_double) == order{1, 3, 4, 0, 2});
	REQUIRE(polar_angle_order(axis_double, fast{}) == order{1, 3, 4, 0, 2});
	// Narrow integer coordinates use a widened two-point cross.
	std::vector<point<std::int16_t>> const extremes{{-128, 127}, {127, -128}, {-128, -128}, {127, 127}, {-128, 0}};
	REQUIRE(polar_angle_order(extremes) == order{2, 1, 3, 0, 4});
}
