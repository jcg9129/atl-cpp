#include <atl/nt/crt.hpp>
#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

using atl::int64;
using atl::uint128;
using atl::uint32;
using atl::uint64;

namespace {
	// The solution set of a·x ≡ c (mod m) within [0, m), as a class: least solution and
	// the gap between consecutive solutions (which is constant).
	std::optional<atl::congruence<uint64>> brute_linear(int64 a, int64 c, uint64 m) {
		std::vector<uint64> xs;
		for (uint64 x = 0; x < m; ++x) {
			int64 const lhs = ((a % int64(m)) * int64(x) - c) % int64(m);
			if (lhs == 0) xs.push_back(x);
		}
		if (xs.empty()) return std::nullopt;
		uint64 const step = xs.size() == 1 ? m : xs[1] - xs[0];
		return atl::congruence<uint64>{xs[0], step};
	}

	std::optional<atl::congruence<uint64>> brute_system(std::vector<atl::congruence<uint64>> const &system) {
		uint64 lcm = 1;
		for (auto const &c : system) lcm = std::lcm(lcm, c.modulus);
		for (uint64 x = 0; x < lcm; ++x) {
			bool is_solution = true;
			for (auto const &c : system) is_solution = is_solution && x % c.modulus == c.residue;
			if (is_solution) return atl::congruence<uint64>{x, lcm};
		}
		return std::nullopt;
	}

	constexpr bool is_prime_small(uint64 n) {
		if (n < 2) return false;
		for (uint64 d = 2; d * d <= n; ++d)
			if (n % d == 0) return false;
		return true;
	}
}

TEST_CASE("solve_linear_congruence matches the brute force", "[crt]") {
	for (uint64 m = 1; m <= 40; ++m)
		for (int64 a = -45; a <= 45; ++a)
			for (int64 c = -45; c <= 45; ++c) REQUIRE(atl::solve_linear_congruence(a, c, m) == brute_linear(a, c, m));
}

TEST_CASE("solve_linear_congruence on wide moduli", "[crt]") {
	std::mt19937_64 rng(99);
	for (int iter = 0; iter < 20000; ++iter) {
		uint64 const m = rng() >> (iter % 64) | 1;
		uint64 const a = rng(), c = rng();
		auto const r = atl::solve_linear_congruence(a, c, m);
		uint64 const g = std::gcd(a % m, m);
		REQUIRE(r.has_value() == (c % m % g == 0));
		if (!r) continue;
		REQUIRE(r->modulus == m / g);
		REQUIRE(r->residue < r->modulus);
		REQUIRE(atl::mul_mod<uint64>(a % m, r->residue, m) == c % m);
	}
	uint64 const top = (std::numeric_limits<uint64>::max)();
	REQUIRE(atl::solve_linear_congruence(top - 1, uint64(1), top) == atl::congruence<uint64>{top - 1, top});
	REQUIRE(atl::solve_linear_congruence(0, 0, top) == atl::congruence<uint64>{0, 1});
	REQUIRE(atl::solve_linear_congruence(0, 3, top) == std::nullopt);
	REQUIRE(atl::solve_linear_congruence(5, 7, uint32(1)) == atl::congruence<uint32>{0, 1});
	CHECK(atl_test::aborts([] { (void)atl::solve_linear_congruence(1, 1, uint64(0)); }));
}

TEST_CASE("chinese_remainder of two classes matches the brute force", "[crt]") {
	for (uint64 m1 = 1; m1 <= 24; ++m1)
		for (uint64 m2 = 1; m2 <= 24; ++m2)
			for (uint64 r1 = 0; r1 < m1; ++r1)
				for (uint64 r2 = 0; r2 < m2; ++r2) {
					atl::congruence<uint64> const x{r1, m1}, y{r2, m2};
					REQUIRE(atl::chinese_remainder(x, y) == brute_system({x, y}));
				}
}

TEST_CASE("chinese_remainder of a list matches the brute force", "[crt]") {
	uint64 const seed = GENERATE(uint64(0), uint64(23), uint64(0xC0FFEE), uint64(29), ~uint64(0));
	std::mt19937_64 rng(seed);
	for (int iter = 0; iter < 2000; ++iter) {
		std::vector<atl::congruence<uint64>> system(static_cast<std::size_t>(rng() % 5));
		uint64 const x = rng() % 5000;
		for (auto &c : system) {
			c.modulus = rng() % 30 + 1;
			// mostly consistent (all from x), sometimes perturbed
			c.residue = (rng() % 4 == 0 ? rng() : x) % c.modulus;
		}
		REQUIRE(atl::chinese_remainder(system) == brute_system(system));
	}
	REQUIRE(atl::chinese_remainder(std::vector<atl::congruence<uint32>>{}) == atl::congruence<uint32>{0, 1});
}

TEST_CASE("chinese_remainder past 64 bits with uint128 classes", "[crt]") {
	std::mt19937_64 rng(5);
	std::vector<uint64> primes;
	for (uint64 p = (uint64(1) << 40) + 1; primes.size() < 6; p += 2)
		if (is_prime_small(p)) primes.push_back(p);
	for (int iter = 0; iter < 200; ++iter) {
		uint128 const x = (uint128(rng()) << 64 | rng()) % (uint128(primes[0]) * primes[1] * primes[2]);
		std::vector<atl::congruence<uint128>> system;
		for (int i = 0; i < 3; ++i) {
			// a shared factor makes the moduli non-coprime
			uint128 const m = uint128(primes[i]) * (i == 2 ? 6 : 2);
			system.push_back({x % m, m});
		}
		auto const r = atl::chinese_remainder(system);
		REQUIRE(r.has_value());
		REQUIRE(r->modulus == uint128(primes[0]) * primes[1] * primes[2] * 6);
		for (auto const &c : system) REQUIRE(r->residue % c.modulus == c.residue);
		REQUIRE(r->residue % (uint128(primes[0]) * primes[1] * primes[2]) == x);
	}
	// the same system in uint64 overflows its lcm: asserted
	CHECK(atl_test::aborts([&] {
		(void)atl::chinese_remainder(atl::congruence<uint64>{1, primes[0]}, atl::congruence<uint64>{1, primes[1]});
	}));
	CHECK(atl_test::aborts([] {
		(void)atl::chinese_remainder(atl::congruence<uint64>{3, 3}, atl::congruence<uint64>{0, 5});
	}));
	CHECK(atl_test::aborts([] {
		(void)atl::chinese_remainder(atl::congruence<uint64>{0, 0}, atl::congruence<uint64>{0, 5});
	}));
}

TEST_CASE("chinese_remainder_mod reconstructs modulo a target", "[crt]") {
	uint64 const seed = GENERATE(uint64(0), uint64(23), uint64(0xC0FFEE), uint64(29), ~uint64(0));
	std::mt19937_64 rng(seed);
	// small coprime moduli: the full solution fits, compare modulo random targets
	for (int iter = 0; iter < 2000; ++iter) {
		std::vector<atl::congruence<uint64>> system;
		uint64 product = 1;
		while (system.size() < 4) {
			uint64 const m = rng() % 60 + 1;
			if (std::gcd(m, product) != 1) continue;
			system.push_back({rng() % m, m});
			product *= m;
		}
		auto const full = atl::chinese_remainder(system);
		REQUIRE(full.has_value());
		uint64 const target = rng() % 1000 + 1;
		REQUIRE(atl::chinese_remainder_mod(system, target) == full->residue % target);
	}
	// three ~2^30 primes and a ~2^60 one: the product passes 2^128 for the last case; the
	// uint128 fold is the oracle while it fits
	std::vector<uint64> primes;
	for (uint64 p = (uint64(1) << 30) + 1; primes.size() < 3; p += 2)
		if (is_prime_small(p)) primes.push_back(p);
	for (int iter = 0; iter < 500; ++iter) {
		std::vector<atl::congruence<uint64>> system;
		std::vector<atl::congruence<uint128>> wide;
		for (uint64 p : primes) {
			uint64 const r = rng() % p;
			system.push_back({r, p});
			wide.push_back({r, p});
		}
		uint64 const target = iter % 2 ? 998244353 : rng() | 1;
		auto const full = atl::chinese_remainder(wide);
		REQUIRE(atl::chinese_remainder_mod(system, target) == uint64(full->residue % target));
	}
	uint64 const top = (std::numeric_limits<uint64>::max)();
	REQUIRE(atl::chinese_remainder_mod(std::vector<atl::congruence<uint64>>{{top - 1, top}, {5, top - 1}}, top)
			== top - 1);
	REQUIRE(atl::chinese_remainder_mod(std::vector<atl::congruence<uint64>>{}, uint64(7)) == 0);
	REQUIRE(atl::chinese_remainder_mod(std::vector<atl::congruence<uint64>>{{0, 1}, {2, 3}}, uint64(1)) == 0);
	CHECK(atl_test::aborts([] {
		(void)atl::chinese_remainder_mod(std::vector<atl::congruence<uint64>>{{1, 4}, {1, 6}}, uint64(100));
	}));
	CHECK(atl_test::aborts(
		[] { (void)atl::chinese_remainder_mod(std::vector<atl::congruence<uint64>>{{1, 4}}, uint64(0)); }));
}
