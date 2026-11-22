#include <atl/geometry/delaunay.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include "../expect_abort.hpp"

using atl::float64;
using atl::int64;
using atl::geometry::point;

namespace {
	int determinant_oracle(std::array<point<int64>, 4> const &p) {
		std::array<int, 4> permutation{0, 1, 2, 3};
		// Independent base-2^32 schoolbook accumulator; no production int256 arithmetic.
		std::array<atl::uint32, 8> positive{}, negative{};
		auto const accumulate = [](std::array<atl::uint32, 8> &sum, atl::uint128 a, atl::uint128 b) {
			std::array<atl::uint32, 8> product{};
			for (std::size_t i = 0; i < 4; ++i) {
				atl::uint64 carry = 0;
				for (std::size_t j = 0; j < 4; ++j) {
					atl::uint64 const v = atl::uint64(atl::uint32(a >> (32 * i))) * atl::uint32(b >> (32 * j)) + product[i + j] + carry;
					product[i + j] = atl::uint32(v);
					carry = v >> 32;
				}
				product[i + 4] = atl::uint32(carry);
			}
			atl::uint64 carry = 0;
			for (std::size_t i = 0; i < 8; ++i) {
				atl::uint64 const v = atl::uint64(sum[i]) + product[i] + carry;
				sum[i] = atl::uint32(v);
				carry = v >> 32;
			}
		};
		do {
			int inversions = 0;
			for (int i = 0; i < 4; ++i)
				for (int j = i + 1; j < 4; ++j) inversions += permutation[i] > permutation[j];
			atl::int128 xy = 1, lift = 1;
			for (int i = 0; i < 4; ++i) {
				auto const q = p[std::size_t(i)];
				if (permutation[i] == 0) xy *= q.x;
				if (permutation[i] == 1) xy *= q.y;
				if (permutation[i] == 2) lift = atl::int128(q.x) * q.x + atl::int128(q.y) * q.y;
			}
			bool const is_negative = (inversions % 2 != 0) != (xy < 0);
			accumulate(is_negative ? negative : positive, atl::uint128(xy < 0 ? -xy : xy), atl::uint128(lift));
		} while (std::ranges::next_permutation(permutation).found);
		for (std::size_t i = 8; i != 0; --i)
			if (positive[i - 1] != negative[i - 1]) return positive[i - 1] > negative[i - 1] ? 1 : -1;
		return 0;
	}

	// The incircle sign is invariant under translation and positive scaling,
	// so doubles that are multiples of 2^-k scale to integers and the int256
	// integer branch (exact within |coordinate| < 2^61) becomes an EXACT
	// oracle for the adaptive floating branch.
	int oracle(point<float64> const a, point<float64> const b, point<float64> const c,
			   point<float64> const q, float64 const scale) {
		auto const lift = [&](point<float64> const p) {
			float64 const x = p.x * scale, y = p.y * scale;
			REQUIRE(x == std::floor(x));
			REQUIRE(y == std::floor(y));
			REQUIRE(std::abs(x) < 0x1p61);
			REQUIRE(std::abs(y) < 0x1p61);
			return point<int64>{int64(x), int64(y)};
		};
		return determinant_oracle({lift(a), lift(b), lift(c), lift(q)});
	}

	int adaptive(point<float64> const a, point<float64> const b, point<float64> const c,
				 point<float64> const q) {
		return atl::geometry::detail::incircle_sign(a, b, c, q);
	}

	// integer lattice points on the circle of radius 25 about the origin
	constexpr point<int64> circle25[]{{25, 0}, {24, 7}, {20, 15}, {15, 20}, {7, 24}, {0, 25}, {-7, 24}, {-15, 20}, {-20, 15}, {-24, 7}, {-25, 0}, {-24, -7}, {-20, -15}, {-15, -20}, {-7, -24}, {0, -25}, {7, -24}, {15, -20}, {20, -15}, {24, -7}};
}

TEST_CASE("incircle: dyadic random inputs match the exact integer oracle",
		  "[geometry]") {
	auto seed = GENERATE(take(6, random(std::uint64_t(0),
										std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::uniform_int_distribution<int64> coord(-(int64(1) << 30), int64(1) << 30);
	for (int it = 0; it < 20000; ++it) {
		point<float64> p[4];
		for (auto &v : p)
			v = {float64(coord(rng)) * 0x1p-20, float64(coord(rng)) * 0x1p-20};
		REQUIRE(adaptive(p[0], p[1], p[2], p[3])
				== oracle(p[0], p[1], p[2], p[3], 0x1p20));
	}
}

TEST_CASE("incircle: exactly cocircular dyadic quadruples report zero",
		  "[geometry]") {
	auto seed = GENERATE(take(4, random(std::uint64_t(0),
										std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::uniform_int_distribution<int64> shift(-(int64(1) << 20), int64(1) << 20);
	std::uniform_int_distribution<std::size_t> pick(0, std::size(circle25) - 1);
	for (int it = 0; it < 20000; ++it) {
		// four DISTINCT points of an integer circle, translated and dyadically
		// scaled: still exactly cocircular as doubles
		std::size_t idx[4];
		do {
			for (auto &i : idx) i = pick(rng);
		} while (idx[0] == idx[1] || idx[0] == idx[2] || idx[0] == idx[3]
				 || idx[1] == idx[2] || idx[1] == idx[3] || idx[2] == idx[3]);
		float64 const dx = float64(shift(rng)), dy = float64(shift(rng));
		float64 const s = std::ldexp(1.0, int(rng() % 21) - 10);
		point<float64> p[4];
		for (int k = 0; k < 4; ++k)
			p[k] = {(float64(circle25[idx[k]].x) + dx) * s,
					(float64(circle25[idx[k]].y) + dy) * s};
		REQUIRE(adaptive(p[0], p[1], p[2], p[3]) == 0);
	}
}

TEST_CASE("incircle: ulp perturbations around cocircular quadruples",
		  "[geometry]") {
	auto seed = GENERATE(take(6, random(std::uint64_t(0),
										std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::uniform_int_distribution<int64> shift(-(int64(1) << 18), int64(1) << 18);
	std::uniform_int_distribution<std::size_t> pick(0, std::size(circle25) - 1);
	std::uniform_int_distribution<int> steps(-3, 3);
	for (int it = 0; it < 20000; ++it) {
		std::size_t idx[4];
		do {
			for (auto &i : idx) i = pick(rng);
		} while (idx[0] == idx[1] || idx[0] == idx[2] || idx[0] == idx[3]
				 || idx[1] == idx[2] || idx[1] == idx[3] || idx[2] == idx[3]);
		float64 const dx = float64(shift(rng)), dy = float64(shift(rng));
		point<float64> p[4];
		for (int k = 0; k < 4; ++k)
			p[k] = {float64(circle25[idx[k]].x) + dx, float64(circle25[idx[k]].y) + dy};
		// nudge every coordinate by a few quanta of 2^-34 — the ulp of the
		// top binade (magnitudes < 2^19), and exact at any magnitude since
		// integer + k * 2^-34 needs at most 53 mantissa bits
		for (auto &v : p) {
			v.x += float64(steps(rng)) * 0x1p-34;
			v.y += float64(steps(rng)) * 0x1p-34;
		}
		REQUIRE(adaptive(p[0], p[1], p[2], p[3])
				== oracle(p[0], p[1], p[2], p[3], 0x1p34));
	}
}

TEST_CASE("incircle: inexact differences drive the generic exact phase",
		  "[geometry]") {
	auto seed = GENERATE(take(6, random(std::uint64_t(0),
										std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::uniform_int_distribution<std::size_t> pick(0, std::size(circle25) - 1);
	std::uniform_int_distribution<int64> noise(-4, 4);
	int deep = 0;
	for (int it = 0; it < 20000; ++it) {
		std::size_t idx[4];
		do {
			for (auto &i : idx) i = pick(rng);
		} while (idx[0] == idx[1] || idx[0] == idx[2] || idx[0] == idx[3]
				 || idx[1] == idx[2] || idx[1] == idx[3] || idx[2] == idx[3]);
		// translate the circle so q's base point sits at the origin, then
		// scale a, b, c to magnitude ~2^16 and give q noise at granularity
		// 2^-40: the noise survives (q is tiny) but the differences a - q
		// need more than 53 bits, so their rounding tails are nonzero and
		// only the generic exact phase can decide the near-zero determinant
		point<float64> p[4];
		for (int k = 0; k < 3; ++k)
			p[k] = {float64(circle25[idx[k]].x - circle25[idx[3]].x) * 0x1p10,
					float64(circle25[idx[k]].y - circle25[idx[3]].y) * 0x1p10};
		p[3] = {float64(noise(rng)) * 0x1p-40, float64(noise(rng)) * 0x1p-40};
		// confirm the deep path is reached: the rounded difference drops the
		// noise entirely whenever it is nonzero
		float64 const adx = p[0].x - p[3].x, ady = p[0].y - p[3].y;
		if (atl::geometry::detail::two_diff_tail(p[0].x, p[3].x, adx) != 0
			|| atl::geometry::detail::two_diff_tail(p[0].y, p[3].y, ady) != 0)
			++deep;
		REQUIRE(adaptive(p[0], p[1], p[2], p[3])
				== oracle(p[0], p[1], p[2], p[3], 0x1p40));
	}
	// the deep path must actually be exercised, not vacuously green
	REQUIRE(deep > 10000);
}

TEST_CASE("incircle: sign convention matches the integer branch", "[geometry]") {
	// CCW triangle, query inside / outside / on the circumcircle
	point<float64> const a{0, 0}, b{4, 0}, c{0, 4};
	REQUIRE(adaptive(a, b, c, {2, 2}) == 1);  // centre of the circumcircle
	REQUIRE(adaptive(a, b, c, {4, 4}) == 0);  // fourth corner of the square
	REQUIRE(adaptive(a, b, c, {5, 5}) == -1); // beyond it
	REQUIRE(atl::geometry::detail::incircle_sign<int64>({0, 0}, {4, 0}, {0, 4}, {2, 2}) == 1);
	REQUIRE(atl::geometry::detail::incircle_sign<int64>({0, 0}, {4, 0}, {0, 4}, {4, 4}) == 0);
	REQUIRE(atl::geometry::detail::incircle_sign<int64>({0, 0}, {4, 0}, {0, 4}, {5, 5}) == -1);
}

TEST_CASE("incircle degree-four bounds and floating scale envelope", "[geometry]") {
	int64 const limit = (int64(1) << 61) - 1;
	std::array<point<int64>, 4> const p{{{-limit, -limit}, {limit, -limit}, {limit, limit}, {-limit, limit}}};
	REQUIRE(atl::geometry::detail::incircle_sign_int256(p[0], p[1], p[2], p[3]) == determinant_oracle(p));
	REQUIRE(atl::geometry::detail::incircle_sign(p[0], p[1], p[2], point<int64>{0, 0}) == 1);
	for (double const scale : {0x1p-200, 0x1p200}) {
		point<double> const a{0, 0}, b{scale, 0}, c{scale, scale};
		REQUIRE(adaptive(a, b, c, {0, scale}) == 0);
		REQUIRE(adaptive(a, b, c, {scale / 2, scale / 2}) == 1);
		REQUIRE(adaptive(a, b, c, {-scale, -scale}) == -1);
	}
}

TEST_CASE("incircle: wide inputs and forced wide agreement with fast path", "[geometry]") {
	using atl::geometry::detail::incircle_sign;
	using atl::geometry::detail::incircle_sign_int256;
	std::mt19937_64 rng(977);
	for (int exponent : {32, 60}) {
		int64 const scale = int64(1) << exponent;
		for (int trial = 0; trial < 1000; ++trial) {
			std::array<point<int64>, 4> p{};
			for (auto &v : p) v = {int64(rng() % atl::uint64(2 * scale)) - scale, int64(rng() % atl::uint64(2 * scale)) - scale};
			int const expected = determinant_oracle(p);
			REQUIRE(incircle_sign(p[0], p[1], p[2], p[3]) == expected);
			REQUIRE(incircle_sign_int256(p[0], p[1], p[2], p[3]) == expected);
		}
	}
	// Exactly represented integer doubles can still have rounded differences.
	double const n = 0x1p53;
	REQUIRE(adaptive({n, 0}, {0, n}, {-n, 0}, {0, -n - 2}) == -1);
	int64 const bound = int64(1) << 61;
	for (int64 const coordinate : {-bound, bound}) {
		CHECK(atl_test::aborts([&] { (void)incircle_sign(point<int64>{coordinate, 0}, point<int64>{0, 1}, point<int64>{1, 0}, point<int64>{0, 0}); }));
	}
}
