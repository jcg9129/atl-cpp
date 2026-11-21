#include <atl/geometry/point.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using atl::geometry::approx;
using atl::geometry::contains;
using atl::geometry::direction;
using atl::geometry::fast;
using atl::geometry::intersects;
using atl::geometry::point;
using atl::geometry::segment;
using atl::geometry::squared_length;

namespace {
	// independent oracle: exact rationals via cross-multiplied parametric
	// solve, structured nothing like the orientation casework under test
	struct pt {
		std::int64_t x, y;
	};

	std::int64_t cx(pt const a, pt const b) { return a.x * b.y - a.y * b.x; }

	pt sub(pt const a, pt const b) { return {a.x - b.x, a.y - b.y}; }

	bool brute_on(pt const a, pt const b, pt const q) {
		if (cx(sub(b, a), sub(q, a)) != 0) return false;
		return (std::min)(a.x, b.x) <= q.x && q.x <= (std::max)(a.x, b.x) && (std::min)(a.y, b.y) <= q.y
			&& q.y <= (std::max)(a.y, b.y);
	}

	bool brute_intersects(pt const a, pt const b, pt const c, pt const d) {
		pt const r = sub(b, a), s = sub(d, c), qp = sub(c, a);
		if (r.x == 0 && r.y == 0 && s.x == 0 && s.y == 0) return a.x == c.x && a.y == c.y;
		if (r.x == 0 && r.y == 0) return brute_on(c, d, a);
		if (s.x == 0 && s.y == 0) return brute_on(a, b, c);
		std::int64_t const rxs = cx(r, s);
		if (rxs != 0) {
			// intersection at parameters t = cx(qp,s)/rxs, u = cx(qp,r)/rxs
			auto const in01 = [](std::int64_t num, std::int64_t den) {
				if (den < 0) {
					num = -num;
					den = -den;
				}
				return 0 <= num && num <= den;
			};
			return in01(cx(qp, s), rxs) && in01(cx(qp, r), rxs);
		}
		if (cx(qp, r) != 0) return false; // parallel, distinct lines
		// collinear: 1D overlap of projections onto r
		std::int64_t const hi = r.x * r.x + r.y * r.y;
		std::int64_t tc = qp.x * r.x + qp.y * r.y;
		std::int64_t td = (d.x - a.x) * r.x + (d.y - a.y) * r.y;
		if (tc > td) std::swap(tc, td);
		return (std::max<std::int64_t>)(0, tc) <= (std::min)(hi, td);
	}
}

TEST_CASE("segment: distance to a point", "[geometry]") {
	segment<std::int64_t> const s{{0, 0}, {10, 0}};
	REQUIRE(atl::geometry::distance(s, point<std::int64_t>{5, 0}) == 0.0);	// on it: exact zero
	REQUIRE(atl::geometry::distance(s, point<std::int64_t>{5, 3}) == 3.0);	// interior projection
	REQUIRE(atl::geometry::distance(s, point<std::int64_t>{-3, 4}) == 5.0); // clamped to endpoint
	REQUIRE(atl::geometry::distance(s, point<std::int64_t>{13, -4}) == 5.0);
	segment<std::int64_t> const degenerate{{2, 2}, {2, 2}};
	REQUIRE(atl::geometry::distance(degenerate, point<std::int64_t>{5, 6}) == 5.0);
}

TEST_CASE("segment: basics and accessors", "[geometry]") {
	segment const s{point{1, 2}, point{4, 6}};
	REQUIRE(direction(s) == point{3, 4});
	REQUIRE(squared_length(s) == 25);
	REQUIRE(atl::geometry::length(s) == 5.0);
	REQUIRE(s == segment{point{1, 2}, point{4, 6}});
}

TEST_CASE("segment contains: exhaustive small grid, every type and policy", "[geometry]") {
	std::int64_t mismatches = 0;
	for (int ax = -2; ax <= 2; ++ax)
		for (int ay = -2; ay <= 2; ++ay)
			for (int bx = -2; bx <= 2; ++bx)
				for (int by = -2; by <= 2; ++by)
					for (int qx = -3; qx <= 3; ++qx)
						for (int qy = -3; qy <= 3; ++qy) {
							bool const expected = brute_on({ax, ay}, {bx, by}, {qx, qy});
							auto const check = [&]<typename T>(T /*tag*/) {
								segment<T> const s{{T(ax), T(ay)}, {T(bx), T(by)}};
								point<T> const q{T(qx), T(qy)};
								if (contains(s, q) != expected) ++mismatches;
								if (contains(s, q, fast{}) != expected) ++mismatches;
							};
							check(std::int32_t{});
							check(std::int64_t{});
							check(float{});
							check(double{});
						}
	REQUIRE(mismatches == 0);
}

TEST_CASE("segment intersects: exhaustive pairs vs parametric rational oracle", "[geometry]") {
	std::vector<pt> points;
	for (int x = -2; x <= 2; ++x)
		for (int y = -2; y <= 2; ++y) points.push_back({x, y});
	std::vector<std::pair<pt, pt>> segments;
	for (auto const a : points)
		for (auto const b : points) segments.push_back({a, b}); // degenerate included
	std::int64_t mismatches = 0;
	for (auto const &[a, b] : segments)
		for (auto const &[c, d] : segments) {
			bool const expected = brute_intersects(a, b, c, d);
			segment<std::int32_t> const s{{std::int32_t(a.x), std::int32_t(a.y)},
										  {std::int32_t(b.x), std::int32_t(b.y)}};
			segment<std::int32_t> const t{{std::int32_t(c.x), std::int32_t(c.y)},
										  {std::int32_t(d.x), std::int32_t(d.y)}};
			if (intersects(s, t) != expected) ++mismatches;
			segment<double> const sd{{double(a.x), double(a.y)}, {double(b.x), double(b.y)}};
			segment<double> const td{{double(c.x), double(c.y)}, {double(d.x), double(d.y)}};
			if (intersects(sd, td) != expected) ++mismatches;
			if (intersects(sd, td, fast{}) != expected) ++mismatches;
		}
	REQUIRE(mismatches == 0);
}

TEST_CASE("segment: int64 near the bound, constructive collinear cases", "[geometry]") {
	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const uniform = [&](std::int64_t lo, std::int64_t hi) {
		return std::int64_t(lo + rng() % std::uint64_t(hi - lo + 1));
	};
	for (int iteration = 0; iteration < 200; ++iteration) {
		std::int64_t const bound = std::int64_t(1) << 59;
		point<std::int64_t> const o{uniform(-bound, bound), uniform(-bound, bound)};
		std::int64_t const dx = uniform(1, 1 << 20) * (rng() % 2 ? 1 : -1);
		std::int64_t const dy = uniform(1, 1 << 20) * (rng() % 2 ? 1 : -1);
		std::int64_t const len = uniform(1, std::int64_t(1) << 38);
		segment<std::int64_t> const s{o, o + point<std::int64_t>{dx, dy} * len};
		std::int64_t const u = uniform(-len / 2, len + len / 2); // on-line parameter
		point<std::int64_t> q = o + point<std::int64_t>{dx, dy} * u;
		REQUIRE(contains(s, q) == (0 <= u && u <= len));
		q.y += rng() % 2 ? 1 : -1; // off the line by one
		REQUIRE(!contains(s, q));
	}
}

TEST_CASE("segment: approx policy tolerates a bend inside eps", "[geometry]") {
	segment<double> const s{{0.0, 0.0}, {10.0, 0.0}};
	REQUIRE(!contains(s, point<double>{5.0, 0.01}));			 // exact sees the bend
	REQUIRE(contains(s, point<double>{5.0, 0.01}, approx{1.0})); // |cross| = 0.1 <= 1
	REQUIRE(!contains(s, point<double>{5.0, 0.2}, approx{1.0})); // |cross| = 2 > 1
	// crossing within the dead zone reads as touching
	segment<double> const t{{5.0, 0.01}, {5.0, 5.0}};
	REQUIRE(intersects(s, t, approx{1.0}));
	REQUIRE(!intersects(s, t));
}

TEST_CASE("segment: segment-to-segment distance", "[geometry]") {
	using sg = atl::geometry::segment<std::int64_t>;
	// exact spots
	REQUIRE(distance(sg{{0, 0}, {4, 0}}, sg{{1, 1}, {3, 1}}) == 1.0);  // parallel
	REQUIRE(distance(sg{{0, 0}, {2, 0}}, sg{{5, 0}, {9, 0}}) == 3.0);  // collinear gap
	REQUIRE(distance(sg{{0, 0}, {4, 4}}, sg{{4, 0}, {0, 4}}) == 0.0);  // crossing
	REQUIRE(distance(sg{{0, 0}, {2, 0}}, sg{{2, 0}, {2, 5}}) == 0.0);  // shared endpoint
	REQUIRE(distance(sg{{0, 0}, {0, 0}}, sg{{3, 4}, {3, 4}}) == 5.0);  // two points
	REQUIRE(distance(sg{{0, 0}, {10, 0}}, sg{{5, 3}, {5, 3}}) == 3.0); // point vs interior
	// exhaustive small grid: zero iff intersecting, symmetry, endpoint
	// consistency, and agreement with a dense parameter-sampling bound
	std::vector<atl::geometry::point<std::int64_t>> grid;
	for (int x = -2; x <= 2; ++x)
		for (int y = -2; y <= 2; ++y) grid.push_back({x, y});
	std::int64_t bad = 0;
	for (auto const a : grid)
		for (auto const b : grid)
			for (auto const c : grid)
				for (auto const d : grid) {
					if ((a.x + b.x + c.x + d.x) % 2 != 0) continue; // thin the 390k grid
					sg const s{a, b}, t{c, d};
					double const got = distance(s, t);
					if ((got == 0.0) != intersects(s, t)) ++bad;
					if (got != distance(t, s)) ++bad;
					double const cap = (std::min)((std::min)(distance(s, c), distance(s, d)),
												  (std::min)(distance(t, a), distance(t, b)));
					if (!intersects(s, t) && got != cap) ++bad;
					if (got > cap + 1e-12) ++bad;
					// dense sampling can only overestimate the true minimum
					double sampled = 1e300;
					for (int i = 0; i <= 8; ++i)
						for (int j = 0; j <= 8; ++j) {
							double const px = double(a.x) + double(b.x - a.x) * (i / 8.0);
							double const py = double(a.y) + double(b.y - a.y) * (i / 8.0);
							double const qx = double(c.x) + double(d.x - c.x) * (j / 8.0);
							double const qy = double(c.y) + double(d.y - c.y) * (j / 8.0);
							sampled = (std::min)(sampled, std::hypot(px - qx, py - qy));
						}
					if (got > sampled + 1e-9) ++bad;
				}
	REQUIRE(bad == 0);
}

TEST_CASE("segment: point distance keeps its accuracy far from the origin", "[geometry]") {
	// rounding each coordinate to double before differencing would cancel a
	// short segment near 2^59 to a zero direction and return NaN
	using atl::int128;
	using atl::int64;
	using atl::geometry::distance;
	using atl::geometry::point;
	using atl::geometry::segment;
	int64 const offset_base = int64(1) << 59;
	segment<int64> const s{{offset_base, offset_base}, {offset_base + 1, offset_base + 3}};
	REQUIRE(std::abs(distance(s, point<int64>{offset_base + 5, offset_base}) - 15 / std::sqrt(10.0)) < 1e-12);
	REQUIRE(distance(s, point<int64>{offset_base + 1, offset_base + 3}) == 0.0); // an endpoint
	REQUIRE(distance(s, point<int64>{offset_base - 3, offset_base - 4}) == 5.0); // behind the start

	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const offset = [&](int64 const span) { return int64(rng() % std::uint64_t(2 * span + 1)) - span; };
	for (int round = 0; round < 200; ++round) {
		int64 const base_x = (int64(1) << (54 + rng() % 8)) * (rng() % 2 ? 1 : -1);
		int64 const base_y = (int64(1) << (54 + rng() % 8)) * (rng() % 2 ? 1 : -1);
		// the oracle works in offsets from the base, which are small and exact
		int64 const bx = offset(1000), by = offset(1000), qx = offset(3000), qy = offset(3000);
		segment<int64> const far_segment{{base_x, base_y}, {base_x + bx, base_y + by}};
		point<int64> const q{base_x + qx, base_y + qy};
		long double expected;
		int128 const along = int128(qx) * bx + int128(qy) * by, length2 = int128(bx) * bx + int128(by) * by;
		if (length2 == 0 || along <= 0) {
			expected = std::sqrt(static_cast<long double>(int128(qx) * qx + int128(qy) * qy));
		} else if (along >= length2) {
			expected = std::sqrt(static_cast<long double>(int128(qx - bx) * (qx - bx) + int128(qy - by) * (qy - by)));
		} else {
			int128 const doubled = int128(bx) * qy - int128(by) * qx;
			expected = std::abs(static_cast<long double>(doubled)) / std::sqrt(static_cast<long double>(length2));
		}
		double const got = distance(far_segment, q);
		REQUIRE(std::isfinite(got));
		REQUIRE(std::abs(got - double(expected)) <= 1e-9 * (1 + double(expected)));
	}
}

TEST_CASE("segment: distance is exactly 0 for collinear floating points on the segment", "[geometry]") {
	// all points on y = x / 4: with plain IEEE evaluation (no contraction)
	// the rounded cross of this triple is about -12.6, the exact orientation 0
	point<double> const a{1.0, 0.25}, b{0x1.8p30, 0x1.8p28};
	point<double> const q{0x1.a25bb59a64a56p+29, 0x1.a25bb59a64a56p+27};
	REQUIRE(atl::geometry::orientation(a, b, q) == 0);
	REQUIRE(distance(segment<double>{a, b}, q) == 0.0);
	REQUIRE(distance(segment<double>{b, a}, q) == 0.0);
	REQUIRE(contains(segment<double>{a, b}, q));
	// off the segment along the same line: the endpoint distance
	point<double> const beyond{0x1p31, 0x1p29};
	REQUIRE(distance(segment<double>{a, b}, beyond) == atl::geometry::distance(b, beyond));
}

TEST_CASE("segment: random degenerate grids vs the rational oracle, int16 too", "[geometry]") {
	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::int64_t mismatches = 0;
	for (int iteration = 0; iteration < 5000; ++iteration) {
		// Narrow integer values: differences, duplicates and collinearity.
		// differences, duplicates and collinear sets are all frequent
		static constexpr int values[] = {-128, -127, -1, 0, 1, 2, 126, 127};
		auto const v = [&] { return values[rng() % 8]; };
		pt const a{v(), v()}, b{v(), v()}, c{v(), v()}, d{v(), v()};
		bool const expected = brute_intersects(a, b, c, d);
		auto const check = [&]<typename T>(T /*tag*/) {
			segment<T> const s{{T(a.x), T(a.y)}, {T(b.x), T(b.y)}}, t{{T(c.x), T(c.y)}, {T(d.x), T(d.y)}};
			if (intersects(s, t) != expected) ++mismatches;
			if (contains(s, t.a) != brute_on(a, b, c)) ++mismatches;
		};
		check(std::int16_t{});
		check(double{});
	}
	REQUIRE(mismatches == 0);
}
