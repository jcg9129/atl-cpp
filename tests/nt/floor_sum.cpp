#include <atl/experimental/nt/floor_sum_variants.hpp>
#include <atl/nt/floor_sum.hpp>
#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include "../expect_abort.hpp"

using atl::int128;
using atl::int64;
using atl::uint128;
using atl::uint64;

namespace {
	// (a·x + b) mod m, portable
	uint64 affine_mod(uint64 a, uint64 x, uint64 b, uint64 m) {
		return atl::add_mod_unchecked(atl::mul_mod<uint64>(a, x, m), b % m, m);
	}

	// The staircase word written out letter by letter: 'U' per unit climbed, 'R' per step.
	std::string brute_staircase(uint64 n, uint64 m, uint64 a, uint64 b) {
		std::string word(static_cast<std::size_t>(b / m), 'U');
		for (uint64 x = 1; x <= n; ++x) {
			uint64 const climb = (a * x + b) / m - (a * (x - 1) + b) / m;
			word.append(static_cast<std::size_t>(climb), 'U');
			word.push_back('R');
		}
		return word;
	}

	std::optional<uint64> brute_first_residue(uint64 m, uint64 a, uint64 b, uint64 lo, uint64 hi) {
		for (uint64 x = 0; x < m; ++x) {
			uint64 const r = affine_mod(a, x, b, m);
			if (lo <= r && r <= hi) return x;
		}
		return std::nullopt;
	}

	int128 brute_floor_sum(int64 n, int64 m, int64 a, int64 b) {
		int128 sum = 0;
		for (int64 i = 0; i < n; ++i) {
			int128 const numerator = int128(a) * i + b;
			int128 q = numerator / m;
			if (numerator % m != 0 && numerator < 0) --q;
			sum += q;
		}
		return sum;
	}

	// (rights, ups, Σ over rights of the ups preceding it): the staircase monoid for Σ f(x)
	struct floor_sum_element {
		uint128 rights = 0, ups = 0, sum = 0;
	};

	constexpr auto combine = [](floor_sum_element const &l, floor_sum_element const &r) {
		return floor_sum_element{l.rights + r.rights, l.ups + r.ups, l.sum + r.sum + l.ups * r.rights};
	};

	// Σ_{i<n} f(i) through the staircase product: an independent algorithm
	uint128 staircase_floor_sum(uint64 n, uint64 m, uint64 a, uint64 b) {
		if (n == 0) return 0;
		auto const e = atl::floor_staircase_product(n - 1, m, a, b, floor_sum_element{0, 1, 0},
													floor_sum_element{1, 0, 0}, floor_sum_element{}, combine);
		return e.sum + b / m;
	}
}

TEST_CASE("mul_add_divmod divides x·y + z without a wider type", "[floor_sum]") {
	uint64 const top = (std::numeric_limits<uint64>::max)();
	auto const check = [](uint64 x, uint64 y, uint64 z, uint64 d, uint64 quotient, uint64 remainder) {
		auto const got = atl::detail::mul_add_divmod(x, y, z, d);
		REQUIRE(got.quotient == quotient);
		REQUIRE(got.remainder == remainder);
	};
	check(0, 0, 0, 1, 0, 0);
	check(7, 5, 3, 4, 9, 2);
	check(top, top - 1, top - 1, top, top - 1, top - 1);
	check(top - 1, top - 1, top - 1, top, top - 1, 0);	  // (top - 1)·top
	check(5, uint64(1) << 63, 1, uint64(1) << 62, 10, 1); // x > d
	check(top, 1, 0, top, 1, 0);
	std::mt19937_64 rng(99);
	for (int iter = 0; iter < 20000; ++iter) {
		uint64 const d = (rng() >> (rng() % 64)) | 1;
		uint64 const x = rng() >> (rng() % 64), z = rng() >> (rng() % 64);
		// keep the quotient within 64 bits: y < 2^64·d / (x + 1)
		uint64 y = rng() >> (rng() % 64);
		uint128 const limit = (uint128(top) * d) / (uint128(x) + 1);
		if (y > limit) y = uint64(limit);
		uint128 const value = uint128(x) * y + z;
		if (value / d > top) continue;
		check(x, y, z, d, uint64(value / d), uint64(value % d));
	}
}

TEST_CASE("floor_sum matches the brute force on small signed inputs", "[floor_sum]") {
	for (int64 n = 0; n <= 12; ++n)
		for (int64 m = 1; m <= 9; ++m)
			for (int64 a = -20; a <= 20; ++a)
				for (int64 b = -20; b <= 20; ++b) REQUIRE(atl::floor_sum(n, m, a, b) == brute_floor_sum(n, m, a, b));
}

TEST_CASE("floor_sum matches the brute force on random wide inputs", "[floor_sum]") {
	uint64 const seed = GENERATE(uint64(0), uint64(23), uint64(0xC0FFEE), uint64(29), ~uint64(0));
	std::mt19937_64 rng(seed);
	std::uniform_int_distribution<int64> any((std::numeric_limits<int64>::min)(), (std::numeric_limits<int64>::max)());
	std::uniform_int_distribution<int64> positive(1, (std::numeric_limits<int64>::max)());
	std::uniform_int_distribution<int64> count(0, 300);
	for (int iter = 0; iter < 400; ++iter) {
		int64 const n = count(rng);
		int64 m = positive(rng);
		if (iter % 3 == 0) m = m % 1000 + 1;
		if (iter % 3 == 1) m = m % (int64(1) << 32) + 1;
		int64 const a = any(rng) >> (iter % 64), b = any(rng) >> (iter % 61);
		REQUIRE(atl::floor_sum(n, m, a, b) == brute_floor_sum(n, m, a, b));
	}
}

TEST_CASE("floor_sum agrees with the staircase product on huge n", "[floor_sum]") {
	uint64 const seed = GENERATE(uint64(0), uint64(23), uint64(0xC0FFEE), uint64(29), ~uint64(0));
	std::mt19937_64 rng(seed);
	uint64 const top = uint64((std::numeric_limits<int64>::max)());
	for (int iter = 0; iter < 2000; ++iter) {
		uint64 const shift = rng() % 64;
		uint64 const n = (rng() & top) >> shift;
		uint64 const m = ((rng() & top) >> (rng() % 64)) | 1;
		uint64 const a = (rng() & top) % m, b = (rng() & top) % m;
		REQUIRE(uint128(atl::floor_sum(int64(n), int64(m), int64(a), int64(b))) == staircase_floor_sum(n, m, a, b));
	}
	// both 32-bit dispatch boundaries and the 64-bit extremes
	for (uint64 n : {uint64(1) << 32, (uint64(1) << 32) - 1, top})
		for (uint64 m : {uint64(1) << 32, (uint64(1) << 32) - 1, top, top - 1})
			for (uint64 a : {uint64(0), uint64(1), m / 2, m - 1})
				for (uint64 b : {uint64(0), m - 1}) {
					REQUIRE(uint128(atl::floor_sum(int64(n), int64(m), int64(a), int64(b)))
							== staircase_floor_sum(n, m, a, b));
				}
}

// The experimental reduced-loop variants against the brute force (the family's oracle)
// and against the library function on wide inputs.
TEST_CASE("floor_sum variants match the brute force", "[floor_sum]") {
	std::mt19937_64 rng(2024);
	int64 const top = (std::numeric_limits<int64>::max)();
	for (int iter = 0; iter < 3000; ++iter) {
		int64 const n = int64(rng() % 200);
		int64 const m = int64((rng() & uint64(top)) >> (rng() % 63)) + 1;
		int64 const a = int64((rng() & uint64(top)) % uint64(m)), b = int64((rng() & uint64(top)) % uint64(m));
		int128 const want = brute_floor_sum(n, m, a, b);
		REQUIRE(atl::experimental::floor_sum_by_wide_divisions(n, m, a, b) == want);
		REQUIRE(atl::experimental::floor_sum_by_entry_dispatch(n, m, a, b) == want);
	}
	for (int iter = 0; iter < 3000; ++iter) {
		int64 const n = int64((rng() & uint64(top)) >> (rng() % 63));
		int64 const m = int64((rng() & uint64(top)) >> (rng() % 63)) + 1;
		int64 const a = int64((rng() & uint64(top)) % uint64(m)), b = int64((rng() & uint64(top)) % uint64(m));
		int128 const want = atl::floor_sum(n, m, a, b);
		REQUIRE(atl::experimental::floor_sum_by_wide_divisions(n, m, a, b) == want);
		REQUIRE(atl::experimental::floor_sum_by_entry_dispatch(n, m, a, b) == want);
	}
}

TEST_CASE("floor_sum closed forms", "[floor_sum]") {
	int64 const top = (std::numeric_limits<int64>::max)();
	// a = m: Σ i = n(n-1)/2
	REQUIRE(atl::floor_sum(top, top, top, 0) == int128(top) * (top - 1) / 2);
	// a = 0: n·floor(b/m)
	REQUIRE(atl::floor_sum(top, 7, 0, -15) == int128(top) * -3);
	int64 const low = (std::numeric_limits<int64>::min)(), n31 = int64(1) << 31;
	REQUIRE(atl::floor_sum(n31, 1, low, low) == int128(low) * (int128(n31) * (n31 - 1) / 2) + int128(low) * n31);
	REQUIRE(atl::floor_sum(0, 5, 123, 456) == 0);
	REQUIRE(atl::floor_sum(1, 5, 123, 456) == 91);
}

TEST_CASE("floor_sum preconditions abort", "[floor_sum]") {
	CHECK(atl_test::aborts([] { (void)atl::floor_sum(-1, 5, 1, 1); }));
	CHECK(atl_test::aborts([] { (void)atl::floor_sum(5, 0, 1, 1); }));
	CHECK(atl_test::aborts([] { (void)atl::floor_sum(5, -3, 1, 1); }));
	// a/m·n²/2 ≈ 2^62 · 2^125 does not fit int128
	CHECK(atl_test::aborts([] { (void)atl::floor_sum((std::numeric_limits<int64>::max)(), 1, int64(1) << 62, 0); }));
	// the three parts each fit, their sum does not
	CHECK(atl_test::aborts([] {
		int64 const top = (std::numeric_limits<int64>::max)();
		(void)atl::floor_sum(top, 1, 3, top);
	}));
}

TEST_CASE("floor_staircase_product spells the staircase word", "[floor_sum]") {
	auto const concat = [](std::string const &l, std::string const &r) { return l + r; };
	for (uint64 n = 0; n <= 10; ++n)
		for (uint64 m = 1; m <= 8; ++m)
			for (uint64 a = 0; a <= 20; ++a)
				for (uint64 b = 0; b <= 20; ++b) {
					auto const word = atl::floor_staircase_product(n, m, a, b, std::string("U"), std::string("R"),
																   std::string(), concat);
					REQUIRE(word == brute_staircase(n, m, a, b));
				}
	std::mt19937_64 rng(12345);
	for (int iter = 0; iter < 3000; ++iter) {
		uint64 const n = rng() % 200, m = rng() % 300 + 1, a = rng() % 900, b = rng() % 900;
		auto const word = atl::floor_staircase_product(n, m, a, b, std::string("U"), std::string("R"), std::string(),
													   concat);
		REQUIRE(word == brute_staircase(n, m, a, b));
	}
}

TEST_CASE("floor_staircase_product computes Σ A^x B^f(x) with matrices", "[floor_sum]") {
	// Non-commutative monoid: (A^x, B^y, Σ A^x B^y over the rights) with 2x2 matrices mod p,
	// matrix-valued floor sums, against direct summation.
	constexpr uint64 p = 1'000'000'007;
	using matrix = std::array<uint64, 4>;
	auto const mul = [](matrix const &l, matrix const &r) {
		return matrix{(l[0] * r[0] + l[1] * r[2]) % p, (l[0] * r[1] + l[1] * r[3]) % p,
					  (l[2] * r[0] + l[3] * r[2]) % p, (l[2] * r[1] + l[3] * r[3]) % p};
	};
	auto const add = [](matrix const &l, matrix const &r) {
		return matrix{(l[0] + r[0]) % p, (l[1] + r[1]) % p, (l[2] + r[2]) % p, (l[3] + r[3]) % p};
	};
	matrix const identity{1, 0, 0, 1}, zero{0, 0, 0, 0};
	matrix const A{1, 2, 3, 4}, B{5, 6, 7, 9};

	struct element {
		matrix a_power, b_power, sum;
	};

	// a word w acts on (X, Y): adds X·sum(w)·Y, then X <- X·a_power, Y <- b_power·Y; rights see
	// the B-power of the ups preceding them, ups multiply B on the right of the y-exponent
	auto const combine_matrices = [&](element const &l, element const &r) {
		return element{mul(l.a_power, r.a_power), mul(l.b_power, r.b_power),
					   add(l.sum, mul(mul(l.a_power, r.sum), l.b_power))};
	};
	std::mt19937_64 rng(777);
	for (int iter = 0; iter < 300; ++iter) {
		uint64 const n = rng() % 60, m = rng() % 40 + 1, a = rng() % 100, b = rng() % 100;
		// right: contributes A^x B^y then advances x; up: advances y
		element const up{identity, B, zero}, right{A, identity, identity};
		auto const got = atl::floor_staircase_product(n, m, a, b, up, right, element{identity, identity, zero},
													  combine_matrices);
		matrix expect = zero, ax = identity;
		for (uint64 x = 1; x <= n; ++x) {
			matrix by = identity;
			for (uint64 y = 0; y < (a * x + b) / m; ++y) by = mul(by, B);
			expect = add(expect, mul(ax, by));
			ax = mul(ax, A);
		}
		REQUIRE(got.sum == expect);
	}
}

TEST_CASE("floor_staircase_product precondition aborts", "[floor_sum]") {
	auto const concat = [](std::string const &l, std::string const &r) { return l + r; };
	CHECK(atl_test::aborts([&] {
		(void)atl::floor_staircase_product(3, 0, 1, 1, std::string("U"), std::string("R"), std::string(), concat);
	}));
}

TEST_CASE("first_residue_in_range matches the brute force exhaustively", "[floor_sum]") {
	for (uint64 m = 1; m <= 22; ++m)
		for (uint64 a = 0; a < m; ++a)
			for (uint64 b = 0; b < m; ++b)
				for (uint64 lo = 0; lo < m; ++lo)
					for (uint64 hi = lo; hi < m; ++hi)
						REQUIRE(atl::first_residue_in_range(m, a, b, lo, hi) == brute_first_residue(m, a, b, lo, hi));
}

TEST_CASE("first_residue_in_range on random inputs", "[floor_sum]") {
	uint64 const seed = GENERATE(uint64(0), uint64(23), uint64(0xC0FFEE), uint64(29), ~uint64(0));
	std::mt19937_64 rng(seed);
	for (int iter = 0; iter < 300; ++iter) {
		uint64 const m = rng() % 20000 + 1;
		uint64 a = rng() % m;
		if (iter % 4 == 0) a = a / 6 * 6; // shared factors with m, often unsolvable
		uint64 const b = rng(), lo = rng() % m, width = iter % 2 ? rng() % 8 : rng() % (m - lo);
		uint64 const hi = (std::min)(m - 1, lo + width);
		REQUIRE(atl::first_residue_in_range(m, a, b, lo, hi) == brute_first_residue(m, a, b, lo, hi));
	}
	// 64-bit moduli: the answer lands in the range, and narrow ranges keep the brute force short
	for (int iter = 0; iter < 2000; ++iter) {
		uint64 const m = rng() | (uint64(1) << 63), a = rng(), b = rng();
		uint64 const lo = rng() % m, hi = lo + (std::min)(m - 1 - lo, rng() >> (iter % 64));
		auto const x = atl::first_residue_in_range(m, a, b, lo, hi);
		// reachable residues are exactly those ≡ b (mod gcd(a, m))
		uint64 const g = std::gcd(a % m, m);
		uint128 const first_reachable = uint128(lo) + ((uint128(b % m % g) + g - lo % g) % g);
		REQUIRE(x.has_value() == (first_reachable >= lo && first_reachable <= hi));
		if (!x) continue;
		uint64 const r = affine_mod(a % m, *x, b, m);
		REQUIRE((lo <= r && r <= hi));
		REQUIRE(*x < m);
		if (*x < 4096) {
			for (uint64 y = 0; y < *x; ++y) {
				uint64 const s = affine_mod(a % m, y, b, m);
				REQUIRE(!(lo <= s && s <= hi));
			}
		}
	}
	uint64 const top = (std::numeric_limits<uint64>::max)();
	REQUIRE(atl::first_residue_in_range(top, 1, 0, top - 1, top - 1) == top - 1);
	REQUIRE(atl::first_residue_in_range(top, top - 1, 0, 1, 1) == top - 1);
	REQUIRE(atl::first_residue_in_range(top, 2, 1, 0, 0) == top / 2);
	REQUIRE(atl::first_residue_in_range(top, 0, 5, 0, 4) == std::nullopt);
}

TEST_CASE("first_residue_in_range preconditions abort", "[floor_sum]") {
	CHECK(atl_test::aborts([] { (void)atl::first_residue_in_range(0, 1, 0, 0, 0); }));
	CHECK(atl_test::aborts([] { (void)atl::first_residue_in_range(10, 1, 0, 5, 4); }));
	CHECK(atl_test::aborts([] { (void)atl::first_residue_in_range(10, 1, 0, 5, 10); }));
}

TEST_CASE("floor_sum permits cancellation across the signed result limit", "[floor_sum]") {
	int64 const n = (std::numeric_limits<int64>::max)();
	int128 const pairs = int128(n) * (n - 1) / 2;
	REQUIRE(atl::floor_sum(n, 1, 5, (std::numeric_limits<int64>::min)()) == 3 * pairs - 2 * int128(n));
	REQUIRE(atl::floor_sum(n, 1, -5, n) == -3 * pairs + n);
}
