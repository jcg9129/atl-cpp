#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include "../expect_abort.hpp"

using atl::geometry::approx;
using atl::geometry::cross;
using atl::geometry::dot;
using atl::geometry::exact;
using atl::geometry::fast;
using atl::geometry::orientation;
using atl::geometry::perp;
using atl::geometry::point;
using atl::geometry::squared_distance;
using atl::geometry::squared_length;

namespace {
	// exact for small coordinates: the reference all typed paths must match
	int brute(std::int64_t ox, std::int64_t oy, std::int64_t ax, std::int64_t ay, std::int64_t bx,
			  std::int64_t by) {
		auto const d = (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
		return int(d > 0) - int(d < 0);
	}

	int nudged_ulps(double &coord, int const ulps, std::mt19937_64 &rng) {
		int const dir = rng() % 2 ? 1 : -1;
		double const to = dir > 0 ? std::numeric_limits<double>::infinity()
								  : -std::numeric_limits<double>::infinity();
		for (int k = 0; k < ulps; ++k) coord = std::nextafter(coord, to);
		return dir;
	}
}

// the widening scheme, on every target (int128 is software where it is not native)
static_assert(std::is_same_v<atl::geometry::product_t<atl::int16>, atl::int32>);
static_assert(std::is_same_v<atl::geometry::product_t<atl::int32>, atl::int64>);
static_assert(std::is_same_v<atl::geometry::product_t<atl::int64>, atl::int128>);
static_assert(std::is_same_v<atl::geometry::product_t<float>, double>);
static_assert(std::is_same_v<atl::geometry::product_t<double>, double>);
static_assert(std::is_same_v<atl::geometry::difference_product_t<atl::int16>, atl::int64>);
static_assert(std::is_same_v<atl::geometry::difference_product_t<atl::int32>, atl::int128>);
static_assert(std::is_same_v<atl::geometry::difference_product_t<atl::int64>, atl::int128>);
static_assert(std::is_same_v<atl::geometry::difference_product_t<float>, double>);
static_assert(std::is_same_v<atl::geometry::difference_product_t<double>, double>);

TEST_CASE("orientation: exhaustive small coordinates, every type and policy agrees", "[geometry]") {
	std::int64_t mismatches = 0;
	for (int ox = -3; ox <= 3; ++ox)
		for (int oy = -3; oy <= 3; ++oy)
			for (int ax = -3; ax <= 3; ++ax)
				for (int ay = -3; ay <= 3; ++ay)
					for (int bx = -3; bx <= 3; ++bx)
						for (int by = -3; by <= 3; ++by) {
							int const expected = brute(ox, oy, ax, ay, bx, by);
							auto const check = [&]<typename T>(T /*tag*/) {
								point<T> const o{T(ox), T(oy)}, a{T(ax), T(ay)}, b{T(bx), T(by)};
								if (orientation(o, a, b) != expected) ++mismatches;
								if (orientation(o, a, b, fast{}) != expected) ++mismatches;
							};
							check(std::int16_t{});
							check(std::int32_t{});
							check(std::int64_t{});
							check(float{});
							check(double{});
							// naive double on small integers is exact too
							if (orientation(point<double>{double(ox), double(oy)},
											{double(ax), double(ay)}, {double(bx), double(by)},
											approx{0.25})
								!= expected)
								++mismatches;
						}
	REQUIRE(mismatches == 0);
}

TEST_CASE("orientation: int64 near the 2^62 bound, constructive ground truth", "[geometry]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const uniform = [&](std::int64_t lo, std::int64_t hi) {
		return std::int64_t(lo + rng() % std::uint64_t(hi - lo + 1));
	};
	for (int iteration = 0; iteration < 200; ++iteration) {
		std::int64_t const bound = std::int64_t(1) << 60;
		point<std::int64_t> const o{uniform(-bound, bound), uniform(-bound, bound)};
		std::int64_t dx = 0, dy = 0;
		while (dx == 0) dx = uniform(-(1 << 20), 1 << 20);
		while (dy == 0) dy = uniform(-(1 << 20), 1 << 20);
		std::int64_t const s = uniform(1, std::int64_t(1) << 39);
		std::int64_t const t = uniform(1, std::int64_t(1) << 39);
		point<std::int64_t> const a{o.x + s * dx, o.y + s * dy};
		point<std::int64_t> b{o.x + t * dx, o.y + t * dy};
		REQUIRE(orientation(o, a, b) == 0);

		// bump one coordinate of b: the determinant becomes +-1 * (a - o)
		// component, whose sign is known in closed form
		std::int64_t const delta = rng() % 2 ? 1 : -1;
		if (rng() % 2) {
			b.y += delta; // det = delta * s * dx
			REQUIRE(orientation(o, a, b) == (delta > 0 ? 1 : -1) * (dx > 0 ? 1 : -1));
			b.y -= delta;
		} else {
			b.x += delta; // det = -delta * s * dy
			REQUIRE(orientation(o, a, b) == (delta > 0 ? -1 : 1) * (dy > 0 ? 1 : -1));
		}
	}
}

TEST_CASE("orientation double: exact certifies ulp-perturbed collinear triples", "[geometry]") {
	// A fixed seed makes the perturbation cases reproducible.
	std::mt19937_64 rng(20260818);
	auto const uniform = [&](std::int64_t lo, std::int64_t hi) {
		return std::int64_t(lo + rng() % std::uint64_t(hi - lo + 1));
	};
	for (int iteration = 0; iteration < 2000; ++iteration) {
		point<std::int64_t> const oi{uniform(-(1 << 20), 1 << 20), uniform(-(1 << 20), 1 << 20)};
		std::int64_t const dx = uniform(1, 1 << 10) * (rng() % 2 ? 1 : -1);
		std::int64_t const dy = uniform(1, 1 << 10) * (rng() % 2 ? 1 : -1);
		// large multipliers force the naive products past 53 bits
		std::int64_t const s = uniform(std::int64_t(1) << 19, std::int64_t(1) << 20);
		std::int64_t const t = uniform(std::int64_t(1) << 19, std::int64_t(1) << 20);
		point<double> const o{double(oi.x), double(oi.y)};
		point<double> const a{double(oi.x + s * dx), double(oi.y + s * dy)};
		point<double> b{double(oi.x + t * dx), double(oi.y + t * dy)};
		// only exact is REQUIRED to see the collinearity: the naive policy's
		// result here depends on -ffp-contract (a fused a*b - c*d keeps one
		// product exact and leaves a rounding residual of the other)
		REQUIRE(orientation(o, a, b) == 0);

		// nudge ONE coordinate of b by 1..3 ulps: the true determinant is
		// delta * (a.x - o.x) or -delta * (a.y - o.y), sign known exactly
		int const ulps = 1 + int(rng() % 3);
		int expected = 0;
		if (rng() % 2) {
			int const dir = nudged_ulps(b.y, ulps, rng);
			expected = dir * (dx > 0 ? 1 : -1);
		} else {
			int const dir = nudged_ulps(b.x, ulps, rng);
			expected = -dir * (dy > 0 ? 1 : -1);
		}
		REQUIRE(orientation(o, a, b) == expected);
	}
	// the loose policy must actually be loose here, or this test proves nothing
}

TEST_CASE("orientation double: full expansion stage on mixed-exponent collinear points",
		  "[geometry]") {
	// full 53-bit mantissas at spread exponents on a power-of-two-slope line:
	// the coordinates are exact, the coordinate DIFFERENCES are not, and the
	// true determinant is 0 or one ulp-nudge away — driving the predicate
	// past every estimate into the exact expansion (verified to reach it in
	// ~half of these cases by a stage-count probe)
	std::mt19937_64 rng(20260818);
	for (int iteration = 0; iteration < 2000; ++iteration) {
		auto const coord = [&] {
			return std::ldexp(1.0 + double(rng() & ((std::uint64_t(1) << 52) - 1)) * 0x1p-52,
							  int(rng() % 40));
		};
		double const x0 = coord(), x1 = coord(), x2 = coord();
		double const slope = std::ldexp(1.0, -int(rng() % 20)); // exact y = slope * x
		point<double> const o{x0, x0 * slope}, a{x1, x1 * slope};
		point<double> b{x2, x2 * slope};
		if (x0 == x1) continue;
		REQUIRE(orientation(o, a, b) == 0);

		// nudging b.y by delta makes the true determinant delta * (a.x - o.x)
		int const dir = nudged_ulps(b.y, 1 + int(rng() % 2), rng);
		int const expected = dir * (x1 > x0 ? 1 : -1);
		REQUIRE(orientation(o, a, b) == expected);
	}
}

TEST_CASE("orientation double: exact sign identities on adversarial reals", "[geometry]") {
	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<double> unit(-1.0, 1.0);
	std::uniform_real_distribution<double> tiny(-1e-14, 1e-14);
	for (int iteration = 0; iteration < 2000; ++iteration) {
		// near-collinear with mixed magnitudes: b close to the line o + t(a - o)
		double const scale = std::pow(10.0, double(rng() % 13) - 6.0);
		point<double> const o{unit(rng) * scale, unit(rng) * scale};
		point<double> const a{unit(rng) * scale, unit(rng) * scale};
		double const t = unit(rng) * 2;
		point<double> const b{o.x + t * (a.x - o.x) + tiny(rng) * scale,
							  o.y + t * (a.y - o.y) + tiny(rng) * scale};
		int const r = orientation(o, a, b);
		// antisymmetry and cyclic invariance hold EXACTLY for exact signs
		REQUIRE(orientation(o, b, a) == -r);
		REQUIRE(orientation(a, b, o) == r);
		REQUIRE(orientation(b, o, a) == r);
		REQUIRE(orientation(b, a, o) == -r);
	}
}

TEST_CASE("cross_sign: exhaustive small quadruples, every type and policy", "[geometry]") {
	std::int64_t mismatches = 0;
	int coords[8];
	auto const sweep = [&](auto &&self, int const k) -> void {
		if (k == 8) {
			std::int64_t const det =
				std::int64_t(coords[2] - coords[0]) * (coords[7] - coords[5])
				- std::int64_t(coords[3] - coords[1]) * (coords[6] - coords[4]);
			int const expected = int(det > 0) - int(det < 0);
			auto const check = [&]<typename T>(T /*tag*/) {
				point<T> const a{T(coords[0]), T(coords[1])}, b{T(coords[2]), T(coords[3])};
				point<T> const c{T(coords[4]), T(coords[5])}, d{T(coords[6]), T(coords[7])};
				if (exact{}.cross_sign(a, b, c, d) != expected) ++mismatches;
				if (fast{}.cross_sign(a, b, c, d) != expected) ++mismatches;
			};
			check(std::int32_t{});
			check(std::int64_t{});
			check(double{});
			return;
		}
		for (coords[k] = -2; coords[k] <= 2; ++coords[k]) self(self, k + 1);
	};
	sweep(sweep, 0);
	REQUIRE(mismatches == 0);
}

TEST_CASE("cross_sign double: exact certifies ulp-perturbed parallel edges", "[geometry]") {
	std::mt19937_64 rng(20260822);
	auto const uniform = [&](std::int64_t lo, std::int64_t hi) {
		return std::int64_t(lo + rng() % std::uint64_t(hi - lo + 1));
	};
	for (int iteration = 0; iteration < 2000; ++iteration) {
		std::int64_t const vx = uniform(1, 1 << 10) * (rng() % 2 ? 1 : -1);
		std::int64_t const vy = uniform(1, 1 << 10) * (rng() % 2 ? 1 : -1);
		std::int64_t const sm = uniform(std::int64_t(1) << 19, std::int64_t(1) << 20);
		std::int64_t const tm = uniform(std::int64_t(1) << 19, std::int64_t(1) << 20);
		point<double> const a{double(uniform(0, 1 << 20)), double(uniform(0, 1 << 20))};
		point<double> const c{double(uniform(0, 1 << 20)), double(uniform(0, 1 << 20))};
		point<double> const b{a.x + double(sm * vx), a.y + double(sm * vy)};
		point<double> d{c.x + double(tm * vx), c.y + double(tm * vy)};
		REQUIRE(exact{}.cross_sign(a, b, c, d) == 0); // exactly parallel

		// nudging d.y by delta makes the true value delta * (b.x - a.x)
		int const dir = nudged_ulps(d.y, 1 + int(rng() % 3), rng);
		int const expected = dir * (vx > 0 ? 1 : -1);
		REQUIRE(exact{}.cross_sign(a, b, c, d) == expected);
	}
}

TEST_CASE("policies: approx dead zone, equality semantics", "[geometry]") {
	approx const loose{0.5};
	REQUIRE(loose.sign(0.4) == 0);
	REQUIRE(loose.sign(-0.5) == 0);
	REQUIRE(loose.sign(0.6) == 1);
	REQUIRE(loose.sign(-0.6) == -1);
	REQUIRE(loose.equal(1.0, 1.4));
	REQUIRE(!loose.equal(1.0, 1.6));

	// equal is by euclidean distance, rotation-invariant
	point<double> const p{0.0, 0.0};
	REQUIRE(loose.equal(p, point<double>{0.3, 0.3}));  // dist ~0.424 <= 0.5
	REQUIRE(!loose.equal(p, point<double>{0.4, 0.4})); // dist ~0.566 > 0.5

	// a triple bent by less than the dead zone reads collinear
	point<double> const o{0.0, 0.0}, a{10.0, 0.0};
	REQUIRE(orientation(o, a, point<double>{20.0, 0.04}, loose) == 0); // det = 0.4
	REQUIRE(orientation(o, a, point<double>{20.0, 0.06}, loose) == 1); // det = 0.6
	REQUIRE(orientation(o, a, point<double>{20.0, 0.04}) == 1);		   // exact sees the bend

	// exact equality is value identity
	REQUIRE(exact{}.equal(point<double>{0.1, 0.2}, point<double>{0.1, 0.2}));
	REQUIRE(!exact{}.equal(point<double>{0.1, 0.2}, point<double>{0.1, std::nextafter(0.2, 1.0)}));
	REQUIRE(exact{}.sign(atl::geometry::difference_product_t<atl::int64>(-7)) == -1);
}

TEST_CASE("orientation double: the finite, no-underflow precondition is asserted", "[geometry]") {
	point<double> const o{0.0, 0.0};
	double const nan = std::numeric_limits<double>::quiet_NaN();
	// both products underflow to zero: a positive determinant would read collinear
	CHECK(atl_test::aborts([&] { (void)orientation(o, point<double>{1e-200, 0.0}, point<double>{0.0, 1e-200}); }));
	// a product overflows to infinity
	CHECK(atl_test::aborts([&] { (void)orientation(o, point<double>{1e200, 1.0}, point<double>{1.0, 1e200}); }));
	// NaN would read as collinear
	CHECK(atl_test::aborts([&] { (void)orientation(o, point<double>{nan, 0.0}, point<double>{0.0, 1.0}); }));
	CHECK(orientation(o, point<double>{1e-100, 0.0}, point<double>{0.0, 1e-100}) == 1);
}

TEST_CASE("point: operators and widened value functions", "[geometry]") {
	point<int> const a{1, 2}, b{3, -4};
	REQUIRE(a + b == point<int>{4, -2});
	REQUIRE(a - b == point<int>{-2, 6});
	REQUIRE(-a == point<int>{-1, -2});
	REQUIRE(a * 3 == point<int>{3, 6});
	REQUIRE(3 * a == point<int>{3, 6});
	REQUIRE(perp(a) == point<int>{-2, 1});
	REQUIRE(point<int>{1, 2} < point<int>{1, 3});
	REQUIRE(point<int>{1, 3} < point<int>{2, 0});

	REQUIRE(cross(a, b) == 1 * -4 - 2 * 3);
	REQUIRE(dot(a, b) == 1 * 3 + 2 * -4);
	REQUIRE(squared_length(b) == 25);
	REQUIRE(squared_distance(a, b) == 4 + 36);
	REQUIRE(atl::geometry::distance(a, b) == std::sqrt(40.0));

	// three-point cross: widened differences, exact at int32 extremes
	std::int32_t const big = (std::numeric_limits<std::int32_t>::max)();
	point<std::int32_t> const o{-big, -big}, u{big, -big}, v{-big, big};
	REQUIRE(cross(o, u, v) == atl::geometry::difference_product_t<std::int32_t>(2 * std::int64_t(big)) * atl::geometry::difference_product_t<std::int32_t>(2 * std::int64_t(big)));
	REQUIRE(orientation(o, u, v) == 1);

	// float products are exact in double
	point<float> const fa{16777215.0f, 1.0f}, fb{1.0f, 16777215.0f}; // 2^24 - 1
	REQUIRE(cross(fa, fb) == 16777215.0 * 16777215.0 - 1.0);
}

TEST_CASE("cross is total on int64; dot asserts its one overflowing input", "[geometry]") {
	std::int64_t const lo = (std::numeric_limits<std::int64_t>::min)();
	std::int64_t const hi = (std::numeric_limits<std::int64_t>::max)();
	point<std::int64_t> const m{lo, lo};
	REQUIRE(cross(m, m) == 0);
	// the extreme value, 2^127 - 2^63
	atl::int128 const extreme = cross(point<std::int64_t>{lo, lo}, point<std::int64_t>{hi, lo});
	REQUIRE(extreme == (atl::int128(1) << 126) + atl::int128(hi) * (atl::int128(1) << 63));
	CHECK(atl_test::aborts([&] { (void)dot(m, m); }));
	CHECK(atl_test::aborts([&] { (void)squared_length(m); }));
}

TEST_CASE("to_float64: coordinatewise conversion", "[geometry]") {
	using atl::geometry::to_float64;
	REQUIRE(to_float64(point<int>{3, -4}) == point<double>{3.0, -4.0});
	// a float variable, not the literal: with excess precision (x87) the literal 0.1f may carry more than float's
	// precision into the double conversion, while the stored coordinate is rounded to float
	float const tenth = 0.1f;
	REQUIRE(to_float64(point<float>{tenth, 2.5f}) == point<double>{double(tenth), 2.5});
	REQUIRE(to_float64(point<std::int64_t>{std::int64_t(1) << 53, -7}) == point<double>{0x1p53, -7.0});
	static_assert(to_float64(point<int>{1, 2}) == point<double>{1.0, 2.0});
}

TEST_CASE("approx::cross_sign on int64: exact determinant, dead zone on its double", "[geometry]") {
	approx const zero{0.0};
	// above 2^53 the raw double differences collapse; the widened determinant does not
	std::int64_t const big = std::int64_t(1) << 60;
	point<std::int64_t> const a{big, big}, b{big + 1, big}, c{big, big}, d{big, big + 1};
	REQUIRE(zero.cross_sign(a, b, c, d) == 1);
	REQUIRE(zero.cross_sign(a, b, c, d) == zero.orientation(a, b, d));
	REQUIRE(zero.cross_sign(a, d, c, b) == -1);
	// the dead zone applies in cross-product units
	REQUIRE(approx{1.5}.cross_sign(a, b, c, d) == 0);
	REQUIRE(approx{0.5}.cross_sign(a, b, c, d) == 1);
	// the int64 bound is asserted, as on the exact path
	point<std::int64_t> const remote{std::int64_t(1) << 62, 0};
	CHECK(atl_test::aborts([&] { (void)zero.cross_sign(remote, a, c, d); }));
}

namespace {
	// a dyadic double m * 2^e as the integer m * 2^(e + 24): every coordinate
	// of the test below is an integer multiple of 2^-24 below 2^60
	atl::int128 scaled(double const v) { return atl::int128(std::ldexp(v, 24)); }

	int exact_sign(atl::int128 const v) { return int(v > 0) - int(v < 0); }
}

TEST_CASE("orientation and cross_sign double: random dyadic inputs vs an exact int128 oracle", "[geometry]") {
	// mixed exponents make the coordinate differences inexact; small mantissa
	// ranges make exact collinearity, parallel edges and duplicates frequent
	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const coordinate = [&](bool const is_coarse) {
		std::int64_t const mantissa = is_coarse ? std::int64_t(rng() % 9) - 4
												: std::int64_t(rng() % (std::uint64_t(1) << 21)) - (std::int64_t(1) << 20);
		return std::ldexp(double(mantissa), int(rng() % 40) - 24);
	};
	std::int64_t mismatches = 0;
	for (int iteration = 0; iteration < 20000; ++iteration) {
		bool const is_coarse = rng() % 2 == 0;
		point<double> p[4];
		for (auto &q : p) q = {coordinate(is_coarse), coordinate(is_coarse)};
		if (rng() % 8 == 0) p[2] = p[0]; // duplicates
		atl::int128 const ax = scaled(p[0].x), ay = scaled(p[0].y), bx = scaled(p[1].x), by = scaled(p[1].y);
		atl::int128 const cx = scaled(p[2].x), cy = scaled(p[2].y), dx = scaled(p[3].x), dy = scaled(p[3].y);
		int const turn = exact_sign((bx - ax) * (cy - ay) - (by - ay) * (cx - ax));
		int const edges = exact_sign((bx - ax) * (dy - cy) - (by - ay) * (dx - cx));
		if (orientation(p[0], p[1], p[2]) != turn) ++mismatches;
		if (exact{}.cross_sign(p[0], p[1], p[2], p[3]) != edges) ++mismatches;
		point<float> const f0{float(p[0].x), float(p[0].y)}, f1{float(p[1].x), float(p[1].y)};
		point<float> const f2{float(p[2].x), float(p[2].y)};
		if (is_coarse && orientation(f0, f1, f2) != turn) ++mismatches; // coarse values are exact floats
	}
	REQUIRE(mismatches == 0);
}

TEST_CASE("orientation: exact at the documented integral bounds", "[geometry]") {
	// int32 has no bound: the extreme diagonal is collinear, one step off is not
	std::int32_t const lo32 = (std::numeric_limits<std::int32_t>::min)(), hi32 = (std::numeric_limits<std::int32_t>::max)();
	point<std::int32_t> const o32{lo32, lo32}, a32{hi32, hi32};
	REQUIRE(orientation(o32, a32, point<std::int32_t>{0, 0}) == 0);
	REQUIRE(orientation(o32, a32, point<std::int32_t>{0, 1}) == 1);
	REQUIRE(orientation(o32, a32, point<std::int32_t>{1, 0}) == -1);
	REQUIRE(orientation(o32, a32, point<std::int32_t>{hi32, lo32}) == -1);
	REQUIRE(exact{}.cross_sign(o32, a32, point<std::int32_t>{hi32, lo32}, point<std::int32_t>{lo32, hi32}) == 1);
	// Narrow integer coordinates likewise.
	point<std::int16_t> const o8{-128, -128}, a8{127, 127};
	REQUIRE(orientation(o8, a8, point<std::int16_t>{-1, -1}) == 0);
	REQUIRE(orientation(o8, a8, point<std::int16_t>{-1, 0}) == 1);
	point<std::int16_t> const o16{-32768, -32768}, a16{32767, 32767};
	REQUIRE(orientation(o16, a16, point<std::int16_t>{5, 4}) == -1);

	// int64 just inside |c| < 2^62: products of differences near 2^126
	std::int64_t const edge = (std::int64_t(1) << 62) - 1;
	point<std::int64_t> const o64{-edge, -edge}, a64{edge, edge};
	REQUIRE(orientation(o64, a64, point<std::int64_t>{0, 0}) == 0);
	REQUIRE(orientation(o64, a64, point<std::int64_t>{-1, 0}) == 1);
	REQUIRE(orientation(o64, a64, point<std::int64_t>{edge, edge - 1}) == -1);
	REQUIRE(orientation(o64, a64, point<std::int64_t>{edge, -edge}) == -1);
	REQUIRE(exact{}.cross_sign(o64, a64, point<std::int64_t>{edge, -edge}, point<std::int64_t>{-edge, edge}) == 1);
	REQUIRE(fast{}.cross_sign(o64, a64, point<std::int64_t>{-edge, -edge + 1}, point<std::int64_t>{edge, edge}) == -1);
	// and the bound itself is asserted
	point<std::int64_t> const outside{std::int64_t(1) << 62, 0};
	CHECK(atl_test::aborts([&] { (void)orientation(o64, a64, outside); }));
	CHECK(atl_test::aborts([&] { (void)exact{}.cross_sign(o64, a64, o64, outside); }));
	CHECK(atl_test::aborts([&] { (void)squared_distance(o64, outside); }));
}

TEST_CASE("dot and squared_length: the one overflowing input is asserted for every integer width", "[geometry]") {
	// all four coordinates at the minimum: 2^(2w-1) overflows product_t
	point<std::int16_t> const m8{-32768, -32768};
	point<std::int16_t> const m16{-32768, -32768};
	point<std::int32_t> const m32{(std::numeric_limits<std::int32_t>::min)(), (std::numeric_limits<std::int32_t>::min)()};
	CHECK(atl_test::aborts([&] { (void)dot(m8, m8); }));
	CHECK(atl_test::aborts([&] { (void)squared_length(m16); }));
	CHECK(atl_test::aborts([&] { (void)dot(m32, m32); }));
	// one step away every value fits exactly
	REQUIRE(dot(m8, point<std::int16_t>{-32768, -32767}) == 32768 * 32768 + 32768 * 32767);
	REQUIRE(squared_length(point<std::int16_t>{-32768, 32767}) == 32768 * 32768 + 32767 * 32767);
	REQUIRE(dot(m32, point<std::int32_t>{(std::numeric_limits<std::int32_t>::min)(), 0}) == std::int64_t(1) << 62);
	// cross is total: its extreme 2^(2w-1) - 2^(w-1)
	REQUIRE(cross(m8, point<std::int16_t>{32767, -32768}) == 2147483648LL - 32768);
	REQUIRE(cross(point<std::int32_t>{(std::numeric_limits<std::int32_t>::min)(), (std::numeric_limits<std::int32_t>::min)()},
				  point<std::int32_t>{(std::numeric_limits<std::int32_t>::max)(), (std::numeric_limits<std::int32_t>::min)()})
			== (std::int64_t(1) << 62) + std::int64_t(std::numeric_limits<std::int32_t>::max()) * (std::int64_t(1) << 31));
	// squared_distance needs no bound below 64 bits
	REQUIRE(squared_distance(m8, point<std::int16_t>{32767, 32767}) == 2LL * 65535 * 65535);
}

TEST_CASE("orientation: degenerate triples", "[geometry]") {
	// duplicates and all-equal triples are collinear under every policy
	point<double> const p{0.1, 0.7};
	REQUIRE(orientation(p, p, p) == 0);
	REQUIRE(orientation(p, p, point<double>{3.0, -1.0}) == 0);
	REQUIRE(orientation(p, point<double>{3.0, -1.0}, p) == 0);
	REQUIRE(exact{}.cross_sign(p, p, point<double>{1.0, 2.0}, point<double>{-5.0, 0.25}) == 0);
	point<std::int64_t> const q{-7, 9};
	REQUIRE(orientation(q, q, q, fast{}) == 0);
	REQUIRE(orientation(q, q, q, approx{0.0}) == 0);
}

TEST_CASE("approx equal: integer offsets survive large translations", "[geometry]") {
	for (std::int64_t const base : {std::int64_t(1) << 60, std::int64_t(1) << 61}) {
		REQUIRE_FALSE(approx{0}.equal(point{base, std::int64_t(0)}, point{base + 1, std::int64_t(0)}));
		REQUIRE_FALSE(approx{0.5}.equal(point{base, std::int64_t(0)}, point{base + 1, std::int64_t(0)}));
	}
}

TEST_CASE("error-free transforms: runtime binary64 tails", "[geometry]") {
	double volatile one = 1.0, tiny = 0x1p-60;
	double error = 0;
	double volatile sum = atl::geometry::detail::two_sum(one, tiny, error);
	REQUIRE(sum == 1.0);
	REQUIRE(error == tiny);
	double volatile difference = atl::geometry::detail::two_diff(one, tiny, error);
	REQUIRE(difference == 1.0);
	REQUIRE(error == -tiny);
	double volatile factor = 1.0 + 0x1p-27;
	double volatile product = atl::geometry::detail::two_product(factor, factor, error);
	REQUIRE(product == 1.0 + 0x1p-26);
	REQUIRE(error == 0x1p-54);
}
