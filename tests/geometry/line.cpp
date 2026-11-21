#include <atl/geometry/line.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

using atl::geometry::contains;
using atl::geometry::fast;
using atl::geometry::intersection;
using atl::geometry::intersects;
using atl::geometry::line;
using atl::geometry::parallel;
using atl::geometry::point;
using atl::geometry::projection;
using atl::geometry::same_line;
using atl::geometry::segment;
using atl::geometry::side;
using atl::geometry::side_of;

namespace {
	using pt = point<std::int64_t>;
	using ln = line<std::int64_t>;

	std::int64_t cx(pt const a, pt const b, pt const c) {
		return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
	}
}

TEST_CASE("line predicates: exhaustive small grid, every type and policy", "[geometry]") {
	std::vector<pt> grid;
	for (int x = -2; x <= 2; ++x)
		for (int y = -2; y <= 2; ++y) grid.push_back({x, y});
	std::int64_t mismatches = 0;
	for (auto const la : grid)
		for (auto const lb : grid) {
			if (la == lb) continue;
			for (auto const ma : grid)
				for (auto const mb : grid) {
					if (ma == mb) continue;
					// brute in plain int64 formulas
					std::int64_t const dir_cross = (lb.x - la.x) * (mb.y - ma.y)
												 - (lb.y - la.y) * (mb.x - ma.x);
					bool const expect_parallel = dir_cross == 0;
					bool const expect_same = expect_parallel && cx(la, lb, ma) == 0;
					bool const expect_meet = !expect_parallel || expect_same;
					side const expect_side = cx(la, lb, ma) > 0 ? side::left
										   : cx(la, lb, ma) < 0 ? side::right
																: side::on;
					ln const l{la, lb};
					auto const check = [&]<typename T>(T /*tag*/) {
						line<T> const lt{{T(la.x), T(la.y)}, {T(lb.x), T(lb.y)}};
						line<T> const mt{{T(ma.x), T(ma.y)}, {T(mb.x), T(mb.y)}};
						point<T> const q{T(ma.x), T(ma.y)};
						if (parallel(lt, mt) != expect_parallel) ++mismatches;
						if (same_line(lt, mt) != expect_same) ++mismatches;
						if (intersects(lt, mt) != expect_meet) ++mismatches;
						if (side_of(lt, q) != expect_side) ++mismatches;
						if (contains(lt, q) != (expect_side == side::on)) ++mismatches;
						if (parallel(lt, mt, fast{}) != expect_parallel) ++mismatches;
					};
					check(std::int64_t{});
					check(double{});
					// line-segment: sign product casework
					bool const expect_seg = cx(la, lb, ma) == 0 || cx(la, lb, mb) == 0
										 || (cx(la, lb, ma) > 0) != (cx(la, lb, mb) > 0);
					if (intersects(l, segment<std::int64_t>{ma, mb}) != expect_seg) ++mismatches;
					if (intersects(segment<std::int64_t>{ma, mb}, l) != expect_seg) ++mismatches;
				}
		}
	REQUIRE(mismatches == 0);
}

TEST_CASE("line intersection and projection: within ulps near 2^60", "[geometry]") {
	// the directions are differenced exactly before rounding: small offsets
	// on a base near 2^60 must not cancel (the result ulp there is 256)
	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::int64_t const bx = (std::int64_t(1) << 60) + std::int64_t(rng() % 4096);
	std::int64_t const by = -(std::int64_t(1) << 60) + std::int64_t(rng() % 4096);
	auto const coord = [&] { return std::int64_t(rng() % 2001) - 1000; };
	long double const ulp = 256;
	for (int iteration = 0; iteration < 500; ++iteration) {
		std::int64_t const lax = coord(), lay = coord(), lbx = coord(), lby = coord();
		std::int64_t const max_ = coord(), may = coord(), mbx = coord(), mby = coord();
		ln const l{{bx + lax, by + lay}, {bx + lbx, by + lby}};
		ln const m{{bx + max_, by + may}, {bx + mbx, by + mby}};
		if (l.a == l.b || m.a == m.b || parallel(l, m)) continue;
		atl::int128 const den = atl::int128(lbx - lax) * (mby - may) - atl::int128(lby - lay) * (mbx - max_);
		atl::int128 const num = atl::int128(max_ - lax) * (mby - may) - atl::int128(may - lay) * (mbx - max_);
		long double const t = (long double)num / (long double)den;
		// keep to intersections near the base, where the ulp bound is meaningful
		if (std::abs(t) > 4) continue;
		long double const ex = (long double)bx + lax + t * (lbx - lax), ey = (long double)by + lay + t * (lby - lay);
		auto const got = intersection(l, m);
		REQUIRE(std::isfinite(got.x));
		REQUIRE(std::abs((long double)got.x - ex) <= 4 * ulp);
		REQUIRE(std::abs((long double)got.y - ey) <= 4 * ulp);
		// projection of m.a onto l: t' = dot(m.a - l.a, d) / |d|^2
		atl::int128 const pn = atl::int128(max_ - lax) * (lbx - lax) + atl::int128(may - lay) * (lby - lay);
		atl::int128 const pd = atl::int128(lbx - lax) * (lbx - lax) + atl::int128(lby - lay) * (lby - lay);
		long double const s = (long double)pn / (long double)pd;
		if (std::abs(s) > 4) continue;
		auto const foot = projection(l, m.a);
		REQUIRE(std::abs((long double)foot.x - ((long double)bx + lax + s * (lbx - lax))) <= 4 * ulp);
		REQUIRE(std::abs((long double)foot.y - ((long double)by + lay + s * (lby - lay))) <= 4 * ulp);
	}
}

TEST_CASE("line intersection: matches the exact rational point", "[geometry]") {
	auto seed = GENERATE(take(25, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const coord = [&] { return std::int64_t(rng() % 2001) - 1000; };
	for (int iteration = 0; iteration < 500; ++iteration) {
		ln const l{{coord(), coord()}, {coord(), coord()}};
		ln const m{{coord(), coord()}, {coord(), coord()}};
		if (l.a == l.b || m.a == m.b || parallel(l, m)) continue;
		// exact rational parameter t = num / den in int128
		atl::int128 const den = atl::int128(l.b.x - l.a.x) * (m.b.y - m.a.y)
							  - atl::int128(l.b.y - l.a.y) * (m.b.x - m.a.x);
		atl::int128 const num = atl::int128(m.a.x - l.a.x) * (m.b.y - m.a.y)
							  - atl::int128(m.a.y - l.a.y) * (m.b.x - m.a.x);
		double const t = double(num) / double(den);
		point<double> const expected{double(l.a.x) + t * double(l.b.x - l.a.x),
									 double(l.a.y) + t * double(l.b.y - l.a.y)};
		auto const got = intersection(l, m);
		REQUIRE(std::abs(got.x - expected.x) <= 1e-9 * (std::max)(1.0, std::abs(expected.x)));
		REQUIRE(std::abs(got.y - expected.y) <= 1e-9 * (std::max)(1.0, std::abs(expected.y)));
		// and the point satisfies both line equations to construction slack
		REQUIRE(std::abs(cross(point<double>{double(l.a.x), double(l.a.y)},
							   point<double>{double(l.b.x), double(l.b.y)}, got))
				<= 1e-6 * 4e6);
		REQUIRE(std::abs(cross(point<double>{double(m.a.x), double(m.a.y)},
							   point<double>{double(m.b.x), double(m.b.y)}, got))
				<= 1e-6 * 4e6);
	}
	// axis case: a rounded construction (t = 7/10 is inexact; FMA contraction or x87 may round the last step
	// differently), so within the construction's error bound
	auto const axis = intersection(ln{{0, -5}, {0, 5}}, ln{{-3, 2}, {9, 2}});
	REQUIRE(std::abs(axis.x - 0.0) <= 1e-15);
	REQUIRE(std::abs(axis.y - 2.0) <= 1e-15);
	// segment overload: side relative to the spanned directed line
	REQUIRE(side_of(segment<std::int64_t>{{0, 0}, {4, 0}}, pt{2, 7}) == side::left);
	REQUIRE(side_of(segment<std::int64_t>{{0, 0}, {4, 0}}, pt{-100, 0}) == side::on);
	REQUIRE(side_of(segment<std::int64_t>{{0, 0}, {4, 0}}, pt{2, -7}) == side::right);
}

TEST_CASE("line: distance and projection constructions", "[geometry]") {
	ln const l{{0, 0}, {10, 0}};
	REQUIRE(atl::geometry::distance(l, pt{4, 0}) == 0.0);	 // exact zero on the line
	REQUIRE(atl::geometry::distance(l, pt{-100, 3}) == 3.0); // infinite line: no clamping
	REQUIRE(projection(l, pt{7, -4}) == point<double>{7.0, 0.0});

	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const coord = [&] { return std::int64_t(rng() % 201) - 100; };
	for (int iteration = 0; iteration < 200; ++iteration) {
		ln const m{{coord(), coord()}, {coord(), coord()}};
		if (m.a == m.b) continue;
		pt const q{coord(), coord()};
		auto const foot = projection(m, q);
		point<double> const qd{double(q.x), double(q.y)};
		point<double> const ma{double(m.a.x), double(m.a.y)}, mb{double(m.b.x), double(m.b.y)};
		// foot on the line, and q - foot perpendicular to it
		REQUIRE(std::abs(cross(ma, mb, foot)) <= 1e-6 * 4e4);
		REQUIRE(std::abs(dot(mb - ma, qd - foot)) <= 1e-6 * 4e4);
		// distance agrees with |q - foot|
		REQUIRE(std::abs(atl::geometry::distance(m, q) - atl::geometry::distance(qd, foot))
				<= 1e-9 * 200);
	}
}

TEST_CASE("segment-segment intersection: exhaustive certificate", "[geometry]") {
	using atl::geometry::intersection_kind;
	std::vector<pt> grid;
	for (int x = -2; x <= 2; ++x)
		for (int y = -2; y <= 2; ++y) grid.push_back({x, y});
	std::int64_t failures = 0;
	for (auto const sa : grid)
		for (auto const sb : grid)
			for (auto const ta : grid)
				for (auto const tb : grid) {
					segment<std::int64_t> const s{sa, sb}, t{ta, tb};
					auto const r = intersection(s, t);
					bool const meet = intersects(s, t);
					if ((r.kind == intersection_kind::empty) == meet) ++failures;
					if (r.kind == intersection_kind::at_point) {
						// the point lies on both segments (tolerance for the
						// constructed crossing; grid scale ~2)
						segment<double> const sd{{double(sa.x), double(sa.y)},
												 {double(sb.x), double(sb.y)}};
						segment<double> const td{{double(ta.x), double(ta.y)},
												 {double(tb.x), double(tb.y)}};
						if (atl::geometry::distance(sd, point<double>{r.at.x, r.at.y}) > 1e-9)
							++failures;
						if (atl::geometry::distance(td, point<double>{r.at.x, r.at.y}) > 1e-9)
							++failures;
					}
					if (r.kind == intersection_kind::along_segment) {
						// exact: endpoints inside both, lex ordered, maximal
						if (!(r.along.a < r.along.b)) ++failures;
						for (auto const e : {r.along.a, r.along.b}) {
							if (!contains(s, e) || !contains(t, e)) ++failures;
							bool endpoint = e == sa || e == sb || e == ta || e == tb;
							if (!endpoint) ++failures;
						}
						for (auto const e : {sa, sb, ta, tb})
							if (contains(s, e) && contains(t, e)
								&& !contains(r.along, e))
								++failures;
					}
				}
	REQUIRE(failures == 0);
}

TEST_CASE("line-segment intersection: exhaustive certificate", "[geometry]") {
	using atl::geometry::intersection_kind;
	std::vector<pt> grid;
	for (int x = -2; x <= 2; ++x)
		for (int y = -2; y <= 2; ++y) grid.push_back({x, y});
	std::int64_t failures = 0;
	for (auto const la : grid)
		for (auto const lb : grid) {
			if (la == lb) continue;
			ln const l{la, lb};
			for (auto const sa : grid)
				for (auto const sb : grid) {
					segment<std::int64_t> const s{sa, sb};
					auto const r = intersection(l, s);
					if ((r.kind == intersection_kind::empty) == intersects(l, s)) ++failures;
					if (r.kind == intersection_kind::at_point) {
						line<double> const ld{{double(la.x), double(la.y)},
											  {double(lb.x), double(lb.y)}};
						segment<double> const sd{{double(sa.x), double(sa.y)},
												 {double(sb.x), double(sb.y)}};
						if (atl::geometry::distance(ld, point<double>{r.at.x, r.at.y}) > 1e-9)
							++failures;
						if (atl::geometry::distance(sd, point<double>{r.at.x, r.at.y}) > 1e-9)
							++failures;
					}
					if (r.kind == intersection_kind::along_segment) {
						bool const whole = (r.along == segment<std::int64_t>{sa, sb}
											|| r.along == segment<std::int64_t>{sb, sa});
						if (!whole || !contains(l, sa) || !contains(l, sb)) ++failures;
					}
					if (intersection(s, l) != r) ++failures; // flipped overload
				}
		}
	REQUIRE(failures == 0);
}

TEST_CASE("segment-segment intersection: exact spot cases", "[geometry]") {
	using atl::geometry::intersection_kind;
	using st = segment<std::int64_t>;
	auto const cross_case = intersection(st{{0, 0}, {4, 4}}, st{{0, 4}, {4, 0}});
	REQUIRE(cross_case.kind == intersection_kind::at_point);
	REQUIRE(cross_case.at == point<double>{2.0, 2.0});
	auto const overlap_case = intersection(st{{0, 0}, {6, 0}}, st{{8, 0}, {4, 0}});
	REQUIRE(overlap_case.kind == intersection_kind::along_segment);
	REQUIRE(overlap_case.along == st{{4, 0}, {6, 0}});
	auto const touch = intersection(st{{0, 0}, {4, 0}}, st{{4, 0}, {9, 5}});
	REQUIRE(touch.kind == intersection_kind::at_point);
	REQUIRE(touch.at == point<double>{4.0, 0.0});
	REQUIRE(intersection(st{{0, 0}, {1, 0}}, st{{3, 0}, {5, 0}}).kind
			== intersection_kind::empty);
}

TEST_CASE("line coefficients: exact round trip and side mapping", "[geometry]") {
	using atl::geometry::line_coefficients;
	using atl::geometry::line_from_coefficients;
	std::vector<pt> grid;
	for (int x = -2; x <= 2; ++x)
		for (int y = -2; y <= 2; ++y) grid.push_back({x, y});
	std::int64_t failures = 0;
	for (auto const la : grid)
		for (auto const lb : grid) {
			if (la == lb) continue;
			ln const l{la, lb};
			auto const [a, b, c] = line_coefficients(l);
			for (auto const q : grid) {
				// side_of == sign(A x + B y + C), the documented orientation
				auto const value = a * q.x + b * q.y + c;
				int const expected = int(value > 0) - int(value < 0);
				if (int(side_of(l, q)) != expected) ++failures;
			}
			// round trip through the double factory: geometric coincidence is
			// assessed with the approx policy — the factory's anchor is a
			// CONSTRUCTION, so exact same_line cannot (and should not) hold
			auto const back = line_from_coefficients(line_coefficients(l));
			if (back != line_from_coefficients(double(a), double(b), double(c))) ++failures;
			line<double> const ld{{double(la.x), double(la.y)}, {double(lb.x), double(lb.y)}};
			if (!same_line(back, ld, atl::geometry::approx{1e-6})) ++failures;
			// and the same DIRECTION: a point clearly left of the original
			// must be left of the reconstruction too
			point<double> const left_probe{double(la.x) - double(lb.y - la.y),
										   double(la.y) + double(lb.x - la.x)};
			if (side_of(ld, left_probe) != atl::geometry::side::left) ++failures;
			if (side_of(back, left_probe) != atl::geometry::side::left) ++failures;
		}
	REQUIRE(failures == 0);
	// the no-lattice-point line: contains (0.5, 0) etc.
	auto const odd = line_from_coefficients(2, 2, -1);
	REQUIRE(atl::geometry::distance(odd, point<double>{0.5, 0.0}) <= 1e-12);
	REQUIRE(side_of(odd, point<double>{1.0, 1.0}) == atl::geometry::side::left); // 2+2-1 > 0
}

TEST_CASE("line: distance is exactly 0 for collinear floating points", "[geometry]") {
	// on y = x / 4; without contraction the rounded cross of the triple is not 0
	line<double> const l{{1.0, 0.25}, {0x1.8p30, 0x1.8p28}};
	point<double> const q{0x1.a25bb59a64a56p+29, 0x1.a25bb59a64a56p+27};
	REQUIRE(contains(l, q));
	REQUIRE(atl::geometry::distance(l, q) == 0.0);
	REQUIRE(atl::geometry::distance(l, point<double>{0x1p40, 0x1p38}) == 0.0);
	REQUIRE(atl::geometry::distance(l, point<double>{0.0, 1.0}) > 0.9);
	// Narrow integer differences use the widened type.
	line<std::int16_t> const diagonal{{-128, -128}, {127, 127}};
	REQUIRE(atl::geometry::distance(diagonal, point<std::int16_t>{0, 0}) == 0.0);
	REQUIRE(side_of(diagonal, point<std::int16_t>{-128, 127}) == side::left);
	auto const [a, b, c] = atl::geometry::line_coefficients(diagonal);
	REQUIRE(a == -255);
	REQUIRE(b == 255);
	REQUIRE(c == 0);
}

TEST_CASE("line: the asserted preconditions", "[geometry]") {
	ln const degenerate{{3, 3}, {3, 3}};
	ln const x_axis{{0, 0}, {1, 0}};
	CHECK(atl_test::aborts([&] { (void)side_of(degenerate, pt{0, 0}); }));
	CHECK(atl_test::aborts([&] { (void)parallel(x_axis, degenerate); }));
	CHECK(atl_test::aborts([&] { (void)atl::geometry::line_coefficients(degenerate); }));
	CHECK(atl_test::aborts([&] { (void)intersection(x_axis, ln{{0, 1}, {5, 1}}); })); // parallel
	CHECK(atl_test::aborts([] { (void)atl::geometry::line_from_coefficients(0, 0, 1); }));
	CHECK(atl_test::aborts([&] { (void)side_of(x_axis, pt{std::int64_t(1) << 62, 0}); }));
	// a valid call next to them is accepted
	REQUIRE(side_of(x_axis, pt{0, -1}) == side::right);
}

TEST_CASE("line: integer crossings retain the exact determinant", "[geometry]") {
	std::int32_t const n = 1 << 30;
	segment<std::int32_t> const s{{0, 0}, {n, n - 2}}, t{{1, 1}, {n - 1, n - 3}};
	point<double> const expected{double(n / 2), double(n / 2 - 1)};
	REQUIRE(intersects(s, t));
	REQUIRE(intersection(line<std::int32_t>{s.a, s.b}, line<std::int32_t>{t.a, t.b}) == expected);
	REQUIRE(intersection(s, t).at == expected);
	REQUIRE(intersection(line<std::int32_t>{s.a, s.b}, t).at == expected);
}

TEST_CASE("line: coefficient tangents remain distinct at large anchors", "[geometry]") {
	auto const l = atl::geometry::line_from_coefficients(std::array<std::int64_t, 3>{1, 1, -(std::int64_t(1) << 61)});
	REQUIRE(l.a == point{0x1p60, 0x1p60});
	REQUIRE(l.a != l.b);
	REQUIRE(side_of(l, l.a) == side::on);
	REQUIRE(side_of(l, point{0.0, 0.0}) == side::right);
	auto const tall = atl::geometry::line_from_coefficients(std::array{1e-150, 0.0, 1e150});
	REQUIRE(tall.a != tall.b);
	REQUIRE(std::isfinite(tall.b.x));
	REQUIRE(std::isfinite(tall.b.y));
}

TEST_CASE("line: tolerant disjoint overlap cannot be reversed", "[geometry]") {
	auto const result = intersection(segment<double>{{0, 0}, {2, 2}}, segment<double>{{2.5, 2.5}, {5, 5}}, atl::geometry::approx{0.6});
	REQUIRE(result.kind == atl::geometry::intersection_kind::empty);
}

TEST_CASE("line: coefficient product overflow is asserted", "[geometry]") {
	CHECK(atl_test::aborts([] {
		auto const coefficients = atl::geometry::line_coefficients(line<double>{{1e150, 0}, {0, 1e150}});
		(void)atl::geometry::line_from_coefficients(coefficients);
	}));
}
