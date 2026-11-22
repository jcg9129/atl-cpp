#include <atl/geometry/angular_sort.hpp>
#include <atl/geometry/circle.hpp>
#include <atl/geometry/line.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

using atl::float64;
using atl::geometry::circle;
using atl::geometry::circle_relation;
using atl::geometry::circumcircle;
using atl::geometry::containment;
using atl::geometry::contains;
using atl::geometry::diametral_circle;
using atl::geometry::fast;
using atl::geometry::intersection;
using atl::geometry::line;
using atl::geometry::minimum_enclosing_circle;
using atl::geometry::point;
using atl::geometry::relation;

namespace {
	using pt = point<std::int64_t>;
	using ci = circle<std::int64_t>;
	using ln = line<std::int64_t>;
}

TEST_CASE("circle contains: exhaustive small grid, every type and policy", "[geometry]") {
	std::int64_t mismatches = 0;
	for (int cx = -3; cx <= 3; ++cx)
		for (int cy = -3; cy <= 3; ++cy)
			for (int r = 0; r <= 5; ++r)
				for (int qx = -6; qx <= 6; ++qx)
					for (int qy = -6; qy <= 6; ++qy) {
						std::int64_t const gap = std::int64_t(r) * r
											   - (std::int64_t(qx - cx) * (qx - cx)
												  + std::int64_t(qy - cy) * (qy - cy));
						auto const expected = containment(int(gap > 0) - int(gap < 0));
						auto const check = [&]<atl::geometry::coordinate T>(T /*tag*/) {
							circle<T> const c{{T(cx), T(cy)}, T(r)};
							point<T> const q{T(qx), T(qy)};
							if (contains(c, q) != expected) ++mismatches;
							if (contains(c, q, fast{}) != expected) ++mismatches;
						};
						check(std::int32_t{});
						check(std::int64_t{});
						check(double{});
					}
	REQUIRE(mismatches == 0);
}

TEST_CASE("circle relation: exhaustive small configurations, all six classes", "[geometry]") {
	std::array<std::int64_t, 6> seen{};
	std::int64_t mismatches = 0;
	for (int ax = -2; ax <= 2; ++ax)
		for (int ay = -2; ay <= 2; ++ay)
			for (int ra = 0; ra <= 4; ++ra)
				for (int bx = -2; bx <= 6; ++bx)
					for (int rb = 0; rb <= 4; ++rb) {
						ci const a{{ax, ay}, ra}, b{{bx, 0}, rb};
						// independent plain-integer brute
						std::int64_t const d2 = std::int64_t(bx - ax) * (bx - ax)
											  + std::int64_t(ay) * ay;
						std::int64_t const sum2 = std::int64_t(ra + rb) * (ra + rb);
						std::int64_t const dif2 = std::int64_t(ra - rb) * (ra - rb);
						circle_relation expected;
						if (d2 == 0 && ra == rb)
							expected = circle_relation::coincident;
						else if (d2 > sum2)
							expected = circle_relation::disjoint;
						else if (d2 == sum2)
							expected = circle_relation::externally_tangent;
						else if (d2 < dif2)
							expected = circle_relation::nested;
						else if (d2 == dif2)
							expected = circle_relation::internally_tangent;
						else
							expected = circle_relation::overlapping;
						if (relation(a, b) != expected) ++mismatches;
						if (relation(b, a) != expected) ++mismatches; // symmetric
						circle<double> const ad{{double(ax), double(ay)}, double(ra)};
						circle<double> const bd{{double(bx), 0.0}, double(rb)};
						if (relation(ad, bd) != expected) ++mismatches;
						++seen[std::size_t(expected)];
					}
	REQUIRE(mismatches == 0);
	for (auto const count : seen) REQUIRE(count > 0); // every class exercised
}

TEST_CASE("circumcircle: matches the exact rational center", "[geometry]") {
	auto seed = GENERATE(take(15, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	auto const coord = [&] { return std::int64_t(rng() % 2001) - 1000; };
	for (int iteration = 0; iteration < 300; ++iteration) {
		pt const a{coord(), coord()}, b{coord(), coord()}, c{coord(), coord()};
		if (atl::geometry::orientation(a, b, c) == 0) continue;
		auto const got = circumcircle(a, b, c);
		// exact rational: center = a + (num_x, num_y) / den in int128
		atl::int128 const bx = b.x - a.x, by = b.y - a.y, cx = c.x - a.x, cy = c.y - a.y;
		atl::int128 const b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
		atl::int128 const den = 2 * (bx * cy - by * cx);
		double const ex = double(a.x) + double(cy * b2 - by * c2) / double(den);
		double const ey = double(a.y) + double(bx * c2 - cx * b2) / double(den);
		REQUIRE(std::abs(got.center.x - ex) <= 1e-9 * std::max(1.0, std::abs(ex)));
		REQUIRE(std::abs(got.center.y - ey) <= 1e-9 * std::max(1.0, std::abs(ey)));
		// equidistant from all three, radius matches
		for (auto const v : {a, b, c}) {
			double const d = std::hypot(got.center.x - double(v.x), got.center.y - double(v.y));
			REQUIRE(std::abs(d - got.radius) <= 1e-9 * std::max(1.0, got.radius));
		}
	}
	auto const diameter = diametral_circle(pt{0, 0}, pt{6, 8});
	REQUIRE(atl::geometry::approx{1e-12}.equal(diameter.center, point{3.0, 4.0}));
	REQUIRE(std::abs(diameter.radius - 5.0) < 1e-12);
}

TEST_CASE("circle-line intersection: exact counts on small inputs, residuals", "[geometry]") {
	ci const c{{0, 0}, 5};
	auto const chord = intersection(c, ln{{3, -9}, {3, 4}});
	REQUIRE(chord.size() == 2);
	REQUIRE(atl::geometry::approx{1e-12}.equal(chord[0], point{3.0, -4.0}));
	REQUIRE(atl::geometry::approx{1e-12}.equal(chord[1], point{3.0, 4.0}));
	auto const tangent = intersection(c, ln{{5, -1}, {5, 1}});
	REQUIRE(tangent.size() == 1);
	REQUIRE(atl::geometry::approx{1e-12}.equal(tangent[0], point<double>{5.0, 0.0}));
	REQUIRE(intersection(c, ln{{6, -1}, {6, 1}}).empty());
	REQUIRE(intersection(ci{{2, 2}, 0}, ln{{0, 0}, {1, 1}}).size() == 1); // point circle on line

	auto seed = GENERATE(take(15, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	auto const coord = [&] { return std::int64_t(rng() % 41) - 20; };
	for (int iteration = 0; iteration < 500; ++iteration) {
		ci const cc{{coord(), coord()}, std::int64_t(rng() % 15)};
		ln const l{{coord(), coord()}, {coord(), coord()}};
		if (l.a == l.b) continue;
		// exact expected count: cross^2 vs r^2 * |direction|^2 in int128
		atl::int128 const doubled = cross(l.a, l.b, cc.center);
		atl::int128 const len2 = squared_distance(l.a, l.b);
		atl::int128 const lhs = doubled * doubled;
		atl::int128 const rhs = atl::int128(cc.radius) * cc.radius * len2;
		std::size_t const expected = lhs > rhs ? 0 : lhs == rhs ? 1
																: 2;
		auto const got = intersection(cc, l);
		REQUIRE(got.size() == expected);
		for (auto const q : got) {
			double const from_center =
				std::hypot(q.x - double(cc.center.x), q.y - double(cc.center.y));
			REQUIRE(std::abs(from_center - double(cc.radius)) <= 1e-9 * 40);
			double const off_line = cross(point<double>{double(l.a.x), double(l.a.y)},
										  point<double>{double(l.b.x), double(l.b.y)},
										  q);
			REQUIRE(std::abs(off_line) <= 1e-9 * 1e4);
		}
	}
}

TEST_CASE("circle-circle intersection: count matches relation, residuals", "[geometry]") {
	ci const a{{0, 0}, 5};
	auto const two = intersection(a, ci{{6, 0}, 5});
	REQUIRE(two.size() == 2);
	REQUIRE(atl::geometry::approx{1e-12}.equal(two[0], point{3.0, 4.0}));
	REQUIRE(atl::geometry::approx{1e-12}.equal(two[1], point{3.0, -4.0}));

	auto seed = GENERATE(take(15, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	auto const coord = [&] { return std::int64_t(rng() % 21) - 10; };
	for (int iteration = 0; iteration < 500; ++iteration) {
		ci const p{{coord(), coord()}, std::int64_t(rng() % 12)};
		ci const q{{coord(), coord()}, std::int64_t(rng() % 12)};
		auto const rel = relation(p, q);
		std::size_t const expected = rel == circle_relation::overlapping ? 2
								   : rel == circle_relation::externally_tangent
											 || rel == circle_relation::internally_tangent
									   ? 1
									   : 0; // disjoint, nested, coincident
		auto const got = intersection(p, q);
		REQUIRE(got.size() == expected);
		for (auto const x : got)
			for (auto const cc : {p, q}) {
				double const d =
					std::hypot(x.x - double(cc.center.x), x.y - double(cc.center.y));
				REQUIRE(std::abs(d - double(cc.radius)) <= 1e-9 * 30);
			}
	}
}

TEST_CASE("minimum_enclosing_circle: vs candidate brute, coverage, degenerates", "[geometry]") {
	auto const covered = [](circle<double> const c, std::vector<pt> const &pts, double slack) {
		return std::ranges::all_of(pts, [&](pt const q) {
			return std::hypot(double(q.x) - c.center.x, double(q.y) - c.center.y)
				<= c.radius * (1 + slack) + slack;
		});
	};
	// Rational centers and squared radii certify every support pair/triple;
	// the brute never calls the library's circle constructions or predicates.
	auto const brute_radius = [](std::vector<pt> const &pts) {
		using W = atl::int128;
		W best_n = 0, best_d = 1;
		bool has_best = false;
		auto const consider = [&](W x, W y, W den, pt const support) {
			if (den == 0) return;
			if (den < 0) {
				x = -x;
				y = -y;
				den = -den;
			}
			W const dx = W(support.x) * den - x, dy = W(support.y) * den - y;
			W const r2 = dx * dx + dy * dy;
			for (auto const q : pts) {
				W const qx = W(q.x) * den - x, qy = W(q.y) * den - y;
				if (qx * qx + qy * qy > r2) return;
			}
			W const d2 = den * den;
			if (!has_best || r2 * best_d < best_n * d2) {
				has_best = true;
				best_n = r2;
				best_d = d2;
			}
		};
		for (std::size_t i = 0; i < pts.size(); ++i) {
			consider(pts[i].x, pts[i].y, 1, pts[i]);
			for (std::size_t j = i + 1; j < pts.size(); ++j) {
				consider(W(pts[i].x) + pts[j].x, W(pts[i].y) + pts[j].y, 2, pts[i]);
				for (std::size_t k = j + 1; k < pts.size(); ++k) {
					W const ux = W(pts[j].x) - pts[i].x, uy = W(pts[j].y) - pts[i].y;
					W const vx = W(pts[k].x) - pts[i].x, vy = W(pts[k].y) - pts[i].y;
					W const u2 = ux * ux + uy * uy, v2 = vx * vx + vy * vy;
					W const den = 2 * (ux * vy - uy * vx);
					consider(W(pts[i].x) * den + vy * u2 - uy * v2, W(pts[i].y) * den + ux * v2 - vx * u2, den, pts[i]);
				}
			}
		}
		REQUIRE(has_best);
		return std::sqrt(double(best_n) / double(best_d));
	};
	REQUIRE(covered(minimum_enclosing_circle(std::vector<pt>{{4, 7}}), std::vector<pt>{{4, 7}}, 1e-12));
	REQUIRE(std::abs(minimum_enclosing_circle(std::vector<pt>{{0, 0}, {6, 8}}).radius - 5.0) < 1e-12);
	REQUIRE(minimum_enclosing_circle(std::vector<pt>{{1, 1}, {1, 1}, {1, 1}}).radius == 0.0);

	auto seed = GENERATE(take(20, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + std::size_t(rng() % 9);
	std::vector<pt> pts(n);
	for (auto &q : pts) q = {std::int64_t(rng() % 201) - 100, std::int64_t(rng() % 201) - 100};
	if (rng() % 4 == 0) // collinear sets
		for (std::size_t i = 0; i < n; ++i) pts[i] = {std::int64_t(i) * 3 - 10, std::int64_t(i) * 6 - 20};
	auto const got = minimum_enclosing_circle(pts);
	REQUIRE(covered(got, pts, 1e-7));
	double const expected = brute_radius(pts);
	REQUIRE(got.radius <= expected * (1 + 1e-7) + 1e-7);
	REQUIRE(got.radius >= expected * (1 - 1e-7) - 1e-7);

	// larger cloud: coverage plus at least two nearby-boundary witnesses
	std::vector<pt> big(500);
	for (auto &q : big) q = {std::int64_t(rng() % 20001) - 10000, std::int64_t(rng() % 20001) - 10000};
	auto const mec = minimum_enclosing_circle(big);
	REQUIRE(covered(mec, big, 1e-7));
	int witnesses = 0;
	for (auto const q : big)
		witnesses += std::abs(std::hypot(double(q.x) - mec.center.x, double(q.y) - mec.center.y)
							  - mec.radius)
				  <= 1e-6 * mec.radius;
	REQUIRE(witnesses >= 2);
}

TEST_CASE("circle-segment intersection: exact counts vs quadratic root brute", "[geometry]") {
	using st = atl::geometry::segment<std::int64_t>;
	// exact spot cases on the 3-4-5 circle
	ci const c{{0, 0}, 5};
	REQUIRE(intersection(c, st{{3, -9}, {3, 9}}).size() == 2); // full chord
	REQUIRE(intersection(c, st{{3, 0}, {3, 9}}).size() == 1);  // half chord
	REQUIRE(intersection(c, st{{3, -1}, {3, 1}}).size() == 0); // inside only
	REQUIRE(intersection(c, st{{5, -2}, {5, 3}}).size() == 1); // tangent through
	REQUIRE(intersection(c, st{{5, 1}, {5, 9}}).size() == 0);  // tangent line, off point
	REQUIRE(intersection(c, st{{3, 4}, {3, 4}}).size() == 1);  // point on boundary
	REQUIRE(intersection(c, st{{3, 4}, {-3, 4}}).size() == 2); // endpoints on boundary
	auto const half = intersection(c, st{{0, 0}, {9, 0}});
	REQUIRE(half.size() == 1);
	REQUIRE(atl::geometry::approx{1e-12}.equal(half[0], point<double>{5.0, 0.0}));

	auto seed = GENERATE(take(15, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	auto const coord = [&] { return std::int64_t(rng() % 41) - 20; };
	for (int iteration = 0; iteration < 1000; ++iteration) {
		ci const cc{{coord(), coord()}, std::int64_t(rng() % 15)};
		st const s{{coord(), coord()}, {coord(), coord()}};
		if (s.a == s.b) continue;
		// independent brute: integer quadratic A t^2 + B t + C on [0, 1]
		atl::int128 const dx = s.b.x - s.a.x, dy = s.b.y - s.a.y;
		atl::int128 const wx = s.a.x - cc.center.x, wy = s.a.y - cc.center.y;
		atl::int128 const A = dx * dx + dy * dy;
		atl::int128 const B = 2 * (dx * wx + dy * wy);
		atl::int128 const C = wx * wx + wy * wy - atl::int128(cc.radius) * cc.radius;
		atl::int128 const disc = B * B - 4 * A * C;
		std::size_t expected = 0;
		if (disc == 0) {
			expected = (0 <= -B && -B <= 2 * A) ? 1 : 0;
		} else if (disc > 0) {
			auto const f0 = C, f1 = A + B + C;
			auto const sgn = [](atl::int128 v) { return int(v > 0) - int(v < 0); };
			int const s0 = sgn(f0), s1 = sgn(f1);
			bool const vertex_in = 0 < -B && -B < 2 * A;
			if (s0 < 0 && s1 < 0)
				expected = 0;
			else if (s0 * s1 < 0)
				expected = 1;
			else if (s0 == 0 && s1 == 0)
				expected = 2;
			else if (s0 == 0)
				expected = s1 < 0 ? 1 : 1 + std::size_t(-B > 0);
			else if (s1 == 0)
				expected = s0 < 0 ? 1 : 1 + std::size_t(-B < 2 * A);
			else
				expected = vertex_in ? 2 : 0;
		}
		auto const got = intersection(cc, s);
		REQUIRE(got.size() == expected);
		// residuals: on the circle and on the segment, ordered along it
		double last_t = -1;
		for (auto const q : got) {
			double const d = std::hypot(q.x - double(cc.center.x), q.y - double(cc.center.y));
			REQUIRE(std::abs(d - double(cc.radius)) <= 1e-9 * 60);
			double const t = ((q.x - double(s.a.x)) * double(dx) + (q.y - double(s.a.y)) * double(dy))
						   / double(A);
			REQUIRE(t >= -1e-9);
			REQUIRE(t <= 1 + 1e-9);
			REQUIRE(t >= last_t - 1e-12);
			last_t = t;
		}
	}
}

TEST_CASE("tangent_points: from a point to a circle", "[geometry]") {
	using atl::geometry::tangent_points;
	circle<std::int64_t> const c{{0, 0}, 5};
	// inside: none; on the boundary: the point itself
	REQUIRE(tangent_points(c, point<std::int64_t>{1, 1}).empty());
	auto const on = tangent_points(c, point<std::int64_t>{3, 4});
	REQUIRE(on.size() == 1);
	REQUIRE(on[0] == point{3.0, 4.0});
	// outside: two, LEFT of center->q first; each on the circle and
	// perpendicular to its own radius
	auto const two = tangent_points(c, point<std::int64_t>{10, 0});
	REQUIRE(two.size() == 2);
	REQUIRE(two[0].y > 0); // left of the +x directed line
	REQUIRE(two[1].y < 0);
	for (auto const t : two) {
		REQUIRE(std::abs(t.x * t.x + t.y * t.y - 25.0) <= 1e-9);
		// radius perpendicular to tangent: (t - c) . (t - q) == 0
		REQUIRE(std::abs(t.x * (t.x - 10.0) + t.y * t.y) <= 1e-9);
	}
	REQUIRE(std::abs(two[0].x - 2.5) <= 1e-12); // r^2/d along the axis

	// random: every tangent point lies on the circle, radius perpendicular
	std::mt19937_64 rng(7);
	for (int it = 0; it < 3000; ++it) {
		std::int64_t const r = 1 + std::int64_t(rng() % 50);
		circle<std::int64_t> const cc{{std::int64_t(rng() % 200) - 100,
									   std::int64_t(rng() % 200) - 100},
									  r};
		point<std::int64_t> const q{std::int64_t(rng() % 400) - 200,
									std::int64_t(rng() % 400) - 200};
		auto const pts = tangent_points(cc, q);
		auto const at = atl::geometry::contains(cc, q);
		REQUIRE(pts.size()
				== (at == atl::geometry::containment::inside	 ? 0U
					: at == atl::geometry::containment::boundary ? 1U
																 : 2U));
		for (auto const t : pts) {
			float64 const ox = t.x - float64(cc.center.x), oy = t.y - float64(cc.center.y);
			REQUIRE(std::abs(ox * ox + oy * oy - float64(r) * float64(r))
					<= 1e-7 * float64(r) * float64(r));
			REQUIRE(std::abs(ox * (t.x - float64(q.x)) + oy * (t.y - float64(q.y)))
					<= 1e-6 * float64(r) * std::sqrt(float64(atl::geometry::squared_distance(cc.center, q))));
		}
	}
}

TEST_CASE("common_tangents: counts by relation, every line distance-true",
		  "[geometry]") {
	using atl::geometry::common_tangents;
	using cir = circle<std::int64_t>;
	auto const line_gap = [](atl::geometry::line<float64> const l,
							 point<std::int64_t> const p) {
		float64 const ax = l.b.x - l.a.x, ay = l.b.y - l.a.y;
		float64 const cx = float64(p.x) - l.a.x, cy = float64(p.y) - l.a.y;
		return std::abs(ax * cy - ay * cx) / std::sqrt(ax * ax + ay * ay);
	};
	auto const check = [&](cir const a, cir const b, std::size_t const want) {
		auto const ts = common_tangents(a, b);
		REQUIRE(ts.size() == want);
		for (auto const &l : ts) {
			REQUIRE(std::abs(line_gap(l, a.center) - float64(a.radius)) <= 1e-7);
			REQUIRE(std::abs(line_gap(l, b.center) - float64(b.radius)) <= 1e-7);
		}
	};
	check({{0, 0}, 5}, {{20, 0}, 5}, 4);							  // disjoint, equal radii (parallel outers)
	check({{0, 0}, 5}, {{20, 0}, 3}, 4);							  // disjoint, distinct radii
	check({{0, 0}, 5}, {{10, 0}, 5}, 3);							  // externally tangent
	check({{0, 0}, 5}, {{8, 0}, 5}, 2);								  // overlapping
	check({{0, 0}, 5}, {{2, 0}, 3}, 1);								  // internally tangent
	check({{0, 0}, 5}, {{1, 0}, 3}, 0);								  // nested
	check({{0, 0}, 5}, {{0, 0}, 3}, 0);								  // concentric nested
	REQUIRE(common_tangents(cir{{2, 3}, 4}, cir{{2, 3}, 4}).empty()); // coincident
	// point circles: through-the-point degeneracies
	check({{0, 0}, 0}, {{10, 0}, 0}, 1); // two points: the joining line
	check({{0, 0}, 5}, {{10, 0}, 0}, 2); // external point: two tangents
	check({{0, 0}, 5}, {{3, 4}, 0}, 1);	 // point ON the circle: the tangent
	check({{0, 0}, 5}, {{1, 1}, 0}, 0);	 // point inside

	// random differential: count table by relation + distance truth
	std::mt19937_64 rng(11);
	for (int it = 0; it < 4000; ++it) {
		cir const a{{std::int64_t(rng() % 100) - 50, std::int64_t(rng() % 100) - 50},
					std::int64_t(rng() % 30) + 1};
		cir const b{{std::int64_t(rng() % 100) - 50, std::int64_t(rng() % 100) - 50},
					std::int64_t(rng() % 30) + 1};
		if (a == b) continue;
		std::size_t want = 0;
		switch (atl::geometry::relation(a, b)) {
		case atl::geometry::circle_relation::disjoint:
			want = 4;
			break;
		case atl::geometry::circle_relation::externally_tangent:
			want = 3;
			break;
		case atl::geometry::circle_relation::overlapping:
			want = 2;
			break;
		case atl::geometry::circle_relation::internally_tangent:
			want = 1;
			break;
		default:
			want = 0;
			break;
		}
		check(a, b, want);
	}
}

namespace {
	bool is_simple(std::vector<point<std::int64_t>> const &poly) {
		for (std::size_t i = 0; i < poly.size(); ++i) {
			if (poly[i] == poly[(i + 1) % poly.size()]) return false;
			for (std::size_t j = i + 1; j < poly.size(); ++j) {
				auto const a = atl::geometry::segment{poly[i], poly[(i + 1) % poly.size()]};
				auto const b = atl::geometry::segment{poly[j], poly[(j + 1) % poly.size()]};
				if (j == i + 1 || (i == 0 && j + 1 == poly.size())) {
					if (atl::geometry::intersection(a, b).kind != atl::geometry::intersection_kind::at_point) return false;
				} else if (atl::geometry::intersects(a, b))
					return false;
			}
		}
		return true;
	}
}

TEST_CASE("intersection_area: circle vs polygon", "[geometry]") {
	using atl::geometry::intersection_area;
	using lp = point<std::int64_t>;
	std::vector<lp> const square{{-4, -4}, {4, -4}, {4, 4}, {-4, 4}};
	float64 const pi = std::numbers::pi_v<float64>;

	// disk strictly inside the polygon: the full disk
	REQUIRE(std::abs(intersection_area(circle<std::int64_t>{{0, 0}, 2}, square)
					 - 4 * pi)
			<= 1e-9);
	// polygon strictly inside the disk: the polygon's area
	REQUIRE(std::abs(intersection_area(circle<std::int64_t>{{0, 0}, 20}, square)
					 - 64.0)
			<= 1e-9);
	// disjoint
	REQUIRE(std::abs(intersection_area(circle<std::int64_t>{{100, 100}, 3}, square)) < 1e-12);
	// center ON an edge: half the disk; ON a corner: a quarter
	REQUIRE(std::abs(intersection_area(circle<std::int64_t>{{4, 0}, 2}, square)
					 - 2 * pi)
			<= 1e-9);
	REQUIRE(std::abs(intersection_area(circle<std::int64_t>{{4, 4}, 2}, square)
					 - pi)
			<= 1e-9);
	// winding independence + argument order
	std::vector<lp> const cw(square.rbegin(), square.rend());
	REQUIRE(std::abs(intersection_area(circle<std::int64_t>{{4, 0}, 2}, cw) - 2 * pi)
			<= 1e-9);
	REQUIRE(std::abs(intersection_area(cw, circle<std::int64_t>{{4, 0}, 2}) - 2 * pi)
			<= 1e-9);
	// degenerate inputs
	REQUIRE(intersection_area(circle<std::int64_t>{{0, 0}, 0}, square) == 0.0);
	REQUIRE(intersection_area(circle<std::int64_t>{{0, 0}, 5},
							  std::vector<lp>{{0, 0}, {1, 1}})
			== 0.0);

	// random stars and circles vs a fine-grid integration: sample the disk's
	// bounding box; each cell centre is classified with the DOUBLE polygon
	// (generic offsets — boundary hits are measure-zero, absorbed by the
	// perimeter-scaled tolerance)
	std::mt19937_64 rng(5);
	int checked = 0;
	while (checked < 10) {
		std::vector<lp> poly;
		std::vector<lp> raw;
		for (int k = 0; k < 12; ++k)
			raw.push_back({std::int64_t(rng() % 21) - 10, std::int64_t(rng() % 21) - 10});
		std::erase_if(raw, [](lp const p) { return p.x == 0 && p.y == 0; });
		std::ranges::sort(raw, atl::geometry::polar_angle_less{});
		for (auto const p : raw)
			if (poly.empty() || !(poly.back() == p)) poly.push_back(p);
		if (poly.size() < 3 || atl::geometry::polygon_area2(poly) == 0
			|| !is_simple(poly))
			continue;
		++checked;
		circle<std::int64_t> const c{{std::int64_t(rng() % 11) - 5,
									  std::int64_t(rng() % 11) - 5},
									 std::int64_t(rng() % 8) + 2};
		float64 const got = intersection_area(c, poly);
		std::vector<point<float64>> dpoly;
		for (auto const p : poly) dpoly.push_back({float64(p.x), float64(p.y)});
		int const grid = 600;
		float64 const r = float64(c.radius);
		float64 const pitch = 2 * r / grid;
		float64 covered = 0;
		for (int gx = 0; gx < grid; ++gx)
			for (int gy = 0; gy < grid; ++gy) {
				float64 const x = float64(c.center.x) - r + (gx + 0.5) * pitch;
				float64 const y = float64(c.center.y) - r + (gy + 0.5) * pitch;
				float64 const ox = x - float64(c.center.x), oy = y - float64(c.center.y);
				if (ox * ox + oy * oy > r * r) continue;
				if (atl::geometry::contains(dpoly, point<float64>{x, y})
					== atl::geometry::containment::inside)
					covered += pitch * pitch;
			}
		float64 const boundary_slack =
			3 * pitch * (2 * pi * r + 8 * 30); // circumference + perimeter bound
		REQUIRE(std::abs(got - covered) <= boundary_slack);
	}
}

TEST_CASE("circle constructions keep their accuracy distant from the origin", "[geometry]") {
	// Differences must be widened before conversion: nearby centers at 2^60
	// otherwise collapse to the same double, causing division by zero.
	using atl::int64;
	using atl::geometry::circle;
	using atl::geometry::circumcircle;
	using atl::geometry::intersection;
	using atl::geometry::point;
	using atl::geometry::tangent_points;
	int64 const distant = int64(1) << 60;
	auto const crossings = intersection(circle<int64>{{distant, 0}, 2}, circle<int64>{{distant + 3, 0}, 2});
	REQUIRE(crossings.size() == std::size_t(2));
	for (auto const q : crossings) {
		REQUIRE(std::isfinite(q.x));
		REQUIRE(std::abs(q.x - double(distant)) <= 1024); // x = distant + 1.5, at this magnitude's resolution
		REQUIRE(std::abs(std::abs(q.y) - std::sqrt(1.75)) < 1e-12);
	}
	REQUIRE(std::abs(crossings[0].y + crossings[1].y) < 1e-12);

	// 6-8-10 right triangle: circumradius 5, wherever it sits
	auto const around = circumcircle(point<int64>{distant, distant}, point<int64>{distant + 6, distant}, point<int64>{distant, distant + 8});
	REQUIRE(std::abs(around.radius - 5.0) < 1e-12);

	// tangents from a point 5 away from a radius-3 circle: the chord sits
	// 9/5 along the center line and reaches 12/5 to either side of it
	auto const touch = tangent_points(circle<int64>{{distant, 0}, 3}, point<int64>{distant + 5, 0});
	REQUIRE(touch.size() == std::size_t(2));
	for (auto const q : touch) {
		REQUIRE(std::abs(q.x - double(distant)) <= 1024);
		REQUIRE(std::abs(std::abs(q.y) - 2.4) < 1e-12);
	}
}

TEST_CASE("common_tangents: a point circle on the other's boundary gives one line", "[geometry]") {
	// A zero radius has only one meaningful side sign; deduplication must
	// still suppress the global sign flip at a collapsed tangent.
	using atl::geometry::circle;
	using atl::geometry::common_tangents;
	circle<std::int64_t> const dot{{3, 4}, 0}, disk{{0, 0}, 5};
	REQUIRE(common_tangents(dot, disk).size() == std::size_t(1));
	REQUIRE(common_tangents(disk, dot).size() == std::size_t(1));

	auto seed = GENERATE(take(20, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	for (int round = 0; round < 300; ++round) {
		// small grids hit tangencies and zero radii often
		auto const pick = [&] {
			return circle<std::int64_t>{{std::int64_t(rng() % 9) - 4, std::int64_t(rng() % 9) - 4},
										std::int64_t(rng() % 4)};
		};
		auto const a = pick(), b = pick();
		REQUIRE(common_tangents(a, b).size() == common_tangents(b, a).size());
	}
}

TEST_CASE("circle-line relation: the per-width bound is asserted below 64 bits too", "[geometry]") {
	using atl::geometry::line;
	using atl::geometry::point;
	using atl::geometry::relation;
	// int32 at +-2e9 would overflow the int128 discriminant
	circle<std::int32_t> const c{{0, 0}, 5};
	CHECK(atl_test::aborts([&] { (void)relation(c, line<std::int32_t>{{2'000'000'000, 0}, {2'000'000'000, 1}}); }));
	CHECK(atl_test::aborts([&] { (void)relation(circle<std::int32_t>{{0, 0}, 1 << 30}, line<std::int32_t>{{0, 0}, {1, 1}}); }));
	REQUIRE(relation(circle<std::int32_t>{{0, 0}, (1 << 30) - 1}, line<std::int32_t>{{0, 0}, {1, 1}})
			== atl::geometry::tangency::secant);
	// int16: degree-four products require coordinates below 2^14
	CHECK(atl_test::aborts([&] { (void)relation(circle<std::int16_t>{{0, 0}, 1}, line<std::int16_t>{{16384, 0}, {16384, 1}}); }));
	REQUIRE(relation(circle<std::int16_t>{{0, 0}, 1}, line<std::int16_t>{{16383, 0}, {16383, 1}})
			== atl::geometry::tangency::disjoint);
}

TEST_CASE("common_tangents: outer pair first, then inner", "[geometry]") {
	using atl::geometry::common_tangents;
	using atl::geometry::side_of;
	using cir = circle<std::int64_t>;
	auto const same_side = [](line<float64> const l, cir const a, cir const b) {
		return side_of(l, atl::geometry::to_float64(a.center)) == side_of(l, atl::geometry::to_float64(b.center));
	};
	for (auto const [a, b] : {std::array<cir, 2>{cir{{0, 0}, 2}, cir{{10, 0}, 3}},
							  std::array<cir, 2>{cir{{-3, 7}, 4}, cir{{9, -2}, 1}},
							  std::array<cir, 2>{cir{{0, 0}, 1}, cir{{0, 9}, 6}}}) {
		auto const ts = common_tangents(a, b);
		REQUIRE(ts.size() == std::size_t(4));
		REQUIRE(same_side(ts[0], a, b));
		REQUIRE(same_side(ts[1], a, b));
		REQUIRE(!same_side(ts[2], a, b));
		REQUIRE(!same_side(ts[3], a, b));
	}
	// externally tangent: the collapsed inner tangent comes last
	cir const a{{0, 0}, 2}, b{{5, 0}, 3};
	auto const ts = common_tangents(a, b);
	REQUIRE(ts.size() == std::size_t(3));
	REQUIRE(same_side(ts[0], a, b));
	REQUIRE(same_side(ts[1], a, b));
	REQUIRE(!same_side(ts[2], a, b));
}

TEST_CASE("circle precision tiers at their magnitude bounds", "[geometry]") {
	std::int64_t const quadratic_limit = (std::int64_t(1) << 62) - 1;
	ci const a{{-quadratic_limit, 0}, quadratic_limit}, b{{quadratic_limit, 0}, quadratic_limit};
	REQUIRE(relation(a, b) == circle_relation::externally_tangent);
	REQUIRE(contains(a, pt{0, 0}) == containment::boundary);
	REQUIRE(intersection(a, b).size() == 1);
	REQUIRE(atl::geometry::common_tangents(a, b).size() == 3);
	std::int64_t const quartic_limit = (std::int64_t(1) << 30) - 1;
	ci const c{{0, 0}, quartic_limit};
	ln const tangent{{-quartic_limit, quartic_limit}, {quartic_limit, quartic_limit}};
	REQUIRE(relation(c, tangent) == atl::geometry::tangency::tangent);
	auto const hit = intersection(c, tangent);
	REQUIRE(hit.size() == 1);
	REQUIRE(atl::geometry::approx{1e-6}.equal(hit[0], point{0.0, double(quartic_limit)}));
	circle<std::int32_t> const c32{{0, 0}, std::int32_t(quartic_limit)};
	line<std::int32_t> const l32{{-std::int32_t(quartic_limit), std::int32_t(quartic_limit)}, {std::int32_t(quartic_limit), std::int32_t(quartic_limit)}};
	REQUIRE(relation(c32, l32) == atl::geometry::tangency::tangent);
	for (double const scale : {0x1p-200, 0x1p200}) {
		circle<double> const disk{{0, 0}, scale};
		line<double> const support{{-scale, scale}, {scale, scale}};
		REQUIRE(relation(disk, support) == atl::geometry::tangency::tangent);
		REQUIRE(contains(disk, point{scale, 0.0}) == containment::boundary);
		REQUIRE(contains(disk, point{scale, 0.0}, atl::geometry::approx{0.0}) == containment::boundary);
	}
	CHECK(atl_test::aborts([&] { (void)contains(ci{{0, 0}, std::int64_t(1) << 62}, pt{0, 0}); }));
	ci const invalid{{std::int64_t(1) << 62, 0}, 0};
	CHECK(atl_test::aborts([&] { (void)relation(invalid, invalid); }));
	CHECK(atl_test::aborts([&] { (void)atl::geometry::common_tangents(invalid, invalid); }));
}

TEST_CASE("enclosing-circle absolute slack at tiny floating scales", "[geometry]") {
	std::array const points{point{0.0, 0.0}, point{0x1p-200, 0.0}, point{0.0, 0x1p-200}};
	REQUIRE(minimum_enclosing_circle(points).radius == 0.0);
}

TEST_CASE("disk polygon area: tangent edges contribute sectors", "[geometry]") {
	circle<int> const disk{{0, 0}, 1};
	std::array const square{point{-1, -1}, point{1, -1}, point{1, 1}, point{-1, 1}};
	REQUIRE(std::abs(atl::geometry::intersection_area(disk, square) - std::numbers::pi) < 1e-12);
	auto reversed = square;
	std::ranges::reverse(reversed);
	REQUIRE(std::abs(atl::geometry::intersection_area(disk, reversed) - std::numbers::pi) < 1e-12);
}
