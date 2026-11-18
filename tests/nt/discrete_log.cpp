#include <atl/experimental/nt/discrete_log_variants.hpp>
#include <atl/nt/discrete_log.hpp>
#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstdio>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint64;

namespace {
	// first[v] = least x with a^x ≡ v (mod m); the sequence is periodic within at most log2 m
	// steps with period at most m, so x < m + 64 covers it
	std::vector<std::optional<uint64>> brute_logs(uint64 a, uint64 m) {
		std::vector<std::optional<uint64>> first(static_cast<std::size_t>(m));
		uint64 power = 1 % m;
		for (uint64 x = 0; x < m + 64; ++x) {
			if (!first[static_cast<std::size_t>(power)]) first[static_cast<std::size_t>(power)] = x;
			power = power * a % m;
		}
		return first;
	}

	uint64 brute_order(uint64 a, uint64 m) {
		uint64 power = a % m, k = 1;
		while (power != 1 % m) power = power * a % m, ++k;
		return k;
	}

	uint64 pow_mod(uint64 a, uint64 e, uint64 m) { return atl::pow_mod(a, e, m); }
}

TEST_CASE("discrete_log matches the brute force on every small modulus", "[discrete_log]") {
	std::fprintf(stderr, "PROBE start %s\n", "discrete_log matches the brute force on every small modulus"); std::fflush(stderr);
	for (uint64 m = 1; m <= 130; ++m)
		for (uint64 a = 0; a < m; ++a) {
			auto const first = brute_logs(a, m);
			for (uint64 b = 0; b < m; ++b) REQUIRE(atl::discrete_log(a, b, m) == first[static_cast<std::size_t>(b)]);
		}
	// unreduced arguments
	REQUIRE(atl::discrete_log(12, 15, 7) == atl::discrete_log(5, 1, 7));
	REQUIRE(atl::discrete_log(0, 1, 1) == 0);
	REQUIRE(atl::discrete_log(5, 3, 1) == 0);
}

TEST_CASE("discrete_log on a non-coprime modulus that needs many reduction steps", "[discrete_log]") {
	std::fprintf(stderr, "PROBE start %s\n", "discrete_log on a non-coprime modulus that needs many reduction steps"); std::fflush(stderr);
	// m = 2^40·3^5·7: a = 6 shares 2 and 3; solutions start beyond the direct checks
	uint64 const m = (uint64(1) << 40) * 243 * 7;
	for (uint64 x : {uint64(0), uint64(1), uint64(39), uint64(40), uint64(63), uint64(64), uint64(65), uint64(100)}) {
		uint64 const b = pow_mod(6, x, m);
		auto const r = atl::discrete_log(6, b, m);
		REQUIRE(r.has_value());
		REQUIRE(pow_mod(6, *r, m) == b);
		REQUIRE(*r <= x);
		if (x >= 40) REQUIRE(*r >= 40); // below 40 the power of two is not yet absorbed
	}
	REQUIRE(atl::discrete_log(6, 5, m) == std::nullopt);
}

TEST_CASE("discrete_log on wide prime and composite moduli", "[discrete_log]") {
	std::fprintf(stderr, "PROBE start %s\n", "discrete_log on wide prime and composite moduli"); std::fflush(stderr);
	uint64 const seed = GENERATE(uint64(0), uint64(23), uint64(0xC0FFEE));
	std::mt19937_64 rng(seed);
	// smooth-order and safe primes, an odd composite past 2^32 (Montgomery), an even one (Barrett64)
	std::vector<uint64> const moduli{998244353,
									 1'000'000'007,
									 4179340454199820289ull, // 29·2^57 + 1
									 (uint64(1) << 61) - 1,
									 uint64(1'000'003) * 999'983 * 1009,
									 uint64(4) * 1'000'003 * 999'983 * 15};
	for (uint64 m : moduli) {
		std::fprintf(stderr, "PROBE modulus %llu seed %llu\n", (unsigned long long)m, (unsigned long long)seed); std::fflush(stderr);
		for (int iter = 0; iter < 20; ++iter) {
			std::fprintf(stderr, "PROBE iter %d\n", iter); std::fflush(stderr);
			uint64 a = rng() % m;
			while (std::gcd(a, m) != 1) a = rng() % m;
			uint64 const x = rng() % m, b = pow_mod(a, x, m);
			auto const r = atl::discrete_log(a, b, m);
			REQUIRE(r.has_value());
			// least: the solutions are exactly x + k·ord(a)
			REQUIRE(*r == x % atl::multiplicative_order(a, m));
			REQUIRE(pow_mod(a, *r, m) == b);
		}
		// a non-power of a quadratic residue base: b = a non-residue for a = g^2
		if (atl::is_prime(m)) {
			uint64 const g = atl::primitive_root(m), a = atl::mul_mod<uint64>(g, g, m);
			REQUIRE(atl::discrete_log(a, g, m) == std::nullopt);
		}
	}
}

TEST_CASE("multiplicative_order matches the brute force", "[discrete_log]") {
	std::fprintf(stderr, "PROBE start %s\n", "multiplicative_order matches the brute force"); std::fflush(stderr);
	for (uint64 m = 1; m <= 600; ++m)
		for (uint64 a = 0; a < m; ++a)
			if (std::gcd(a, m) == 1) REQUIRE(atl::multiplicative_order(a, m) == brute_order(a, m));
	REQUIRE(atl::multiplicative_order(3, 998244353) == 998244352);
	REQUIRE(atl::multiplicative_order(uint64(-3), uint64(-2)) == 2); // -1 mod 2^64-2
	CHECK(atl_test::aborts([] { (void)atl::multiplicative_order(4, 6); }));
	CHECK(atl_test::aborts([] { (void)atl::multiplicative_order(1, 0); }));
}

TEST_CASE("primitive_root matches the brute force", "[discrete_log]") {
	std::fprintf(stderr, "PROBE start %s\n", "primitive_root matches the brute force"); std::fflush(stderr);
	for (uint64 p = 2; p <= 3000; ++p) {
		if (!atl::is_prime(p)) continue;
		uint64 g = 1;
		while (brute_order(g, p) != p - 1) ++g;
		REQUIRE(atl::primitive_root(p) == g);
	}
	REQUIRE(atl::primitive_root(998244353) == 3);
	REQUIRE(atl::primitive_root(1'000'000'007) == 5);
	for (uint64 p : {uint64(4179340454199820289ull), (uint64(1) << 61) - 1, uint64(18446744073709551557ull)}) {
		uint64 const g = atl::primitive_root(p);
		REQUIRE(atl::multiplicative_order(g, p) == p - 1);
		for (uint64 h = 2; h < g; ++h) REQUIRE(atl::multiplicative_order(h, p) != p - 1);
	}
	CHECK(atl_test::aborts([] { (void)atl::primitive_root(1); }));
	CHECK(atl_test::aborts([] { (void)atl::primitive_root(91); }));
}

TEST_CASE("discrete_log precondition aborts", "[discrete_log]") {
	std::fprintf(stderr, "PROBE start %s\n", "discrete_log precondition aborts"); std::fflush(stderr);
	CHECK(atl_test::aborts([] { (void)atl::discrete_log(2, 1, 0); }));
}

// The experimental BSGS solver, run behind the library reduction, against the
// brute force (the oracle) on every small modulus and on both 64-bit engines.
TEST_CASE("experimental discrete_log_by_baby_step_giant_step matches the brute force", "[discrete_log]") {
	std::fprintf(stderr, "PROBE start %s\n", "experimental discrete_log_by_baby_step_giant_step matches the brute force"); std::fflush(stderr);
	auto const bsgs = [](uint64 a, uint64 b, uint64 m) {
		return atl::detail::discrete_log_reduced(a, b, m, [](uint64 base, uint64 target, uint64 mod) {
			return atl::detail::with_engine(mod, [&]<atl::detail::modular_multiplier Engine>(Engine const &engine) {
				return atl::experimental::discrete_log_by_baby_step_giant_step(
					engine, engine.to_form(base), engine.to_form(target));
			});
		});
	};
	for (uint64 m = 1; m <= 90; ++m)
		for (uint64 a = 0; a < m; ++a) {
			auto const first = brute_logs(a, m);
			for (uint64 b = 0; b < m; ++b) REQUIRE(bsgs(a, b, m) == first[static_cast<std::size_t>(b)]);
		}
	std::mt19937_64 rng(31);
	for (uint64 m : {uint64(1'000'003) * 1'000'033, uint64(2) * 1'000'003 * 1'000'033 + 0}) {
		for (int iter = 0; iter < 5; ++iter) {
			uint64 a = rng() % m;
			while (std::gcd(a, m) != 1) a = rng() % m;
			uint64 const b = pow_mod(a, rng() % m, m);
			REQUIRE(bsgs(a, b, m) == atl::discrete_log(a, b, m));
		}
	}
}
