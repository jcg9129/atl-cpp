#include <atl/experimental/nt/factorization_trial.hpp>
#include <atl/experimental/nt/factorization_variants.hpp>
#include <atl/nt/basic.hpp>
#include <atl/nt/constants.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;

namespace {
	// Straightforward trial-division reference used to cross-check factorization.
	std::vector<atl::prime_factor> reference_factorization(uint64 n) {
		std::vector<atl::prime_factor> factors;
		for (uint64 p = 2; p * p <= n; ++p) {
			if (n % p == 0) {
				uint64 e = 0;
				while (n % p == 0) {
					++e;
					n /= p;
				}
				factors.push_back({p, e});
			}
		}
		if (n > 1) factors.push_back({n, 1});
		return factors;
	}

	bool reference_is_prime(uint64 n) {
		if (n < 2) return false;
		for (uint64 p = 2; p * p <= n; ++p)
			if (n % p == 0) return false;
		return true;
	}
}

TEST_CASE("is_prime matches trial division on small values", "[basic]") {
	for (uint64 n = 0; n < 10000; ++n) {
		REQUIRE(atl::is_prime(n) == reference_is_prime(n));
	}
}

TEST_CASE("is_prime handles known large primes and composites", "[basic]") {
	REQUIRE(atl::is_prime(atl::largest_prime_64));
	REQUIRE(atl::is_prime(atl::largest_prime_32));
	REQUIRE(atl::is_prime(1000000007ull));
	REQUIRE(atl::is_prime(1000000009ull));
	REQUIRE_FALSE(atl::is_prime(atl::largest_prime_64 - 1));
	// Carmichael numbers must not fool the test.
	REQUIRE_FALSE(atl::is_prime(561ull));
	REQUIRE_FALSE(atl::is_prime(41041ull));
	REQUIRE_FALSE(atl::is_prime(1000000007ull * 1000000009ull));
}

TEST_CASE("factorization edge cases", "[basic]") {
	REQUIRE(atl::factorization(0).empty());
	REQUIRE(atl::factorization(1).empty());
	REQUIRE(atl::factorization(2) == std::vector<atl::prime_factor>({{2, 1}}));
	REQUIRE(atl::factorization(atl::largest_prime_64) == std::vector<atl::prime_factor>({{atl::largest_prime_64, 1}}));
}

TEST_CASE("factorization matches trial division", "[basic]") {
	for (uint64 n = 1; n < 5000; ++n) {
		REQUIRE(atl::factorization(n) == reference_factorization(n));
	}
}

TEST_CASE("factorization on hard 64-bit inputs", "[basic]") {
	uint64 const seed = 0xC0FFEE;
	std::mt19937_64 rng(seed);

	for (int t = 0; t < 200; ++t) {
		uint64 n = rng() | 1; // odd, to hit large odd factors
		auto factors = atl::factorization(n);

		uint64 product = 1;
		uint64 last_prime = 0;
		for (auto [p, e] : factors) {
			REQUIRE(atl::is_prime(p));
			REQUIRE(p > last_prime);
			last_prime = p;
			for (uint64 i = 0; i < e; ++i) product *= p;
		}
		REQUIRE(product == n);
	}
}

TEST_CASE("factorization_by_trial_division: no wraparound past 2^32", "[basic]") {
	// The quotient bound cannot wrap at 2^32; the full sweep certifies the 64-bit domain.
	// Each case needs roughly 2^32 trial divisions.
	using atl::experimental::factorization_by_trial_division;
	REQUIRE(factorization_by_trial_division(atl::largest_prime_64)
			== std::vector<atl::prime_factor>({{atl::largest_prime_64, 1}}));
	uint64 const p = 4294967279, q = 4294967291; // the two largest primes below 2^32
	REQUIRE(factorization_by_trial_division(p * q) == std::vector<atl::prime_factor>({{p, 1}, {q, 1}}));
}

TEST_CASE("nontrivial_factor asserts a composite input", "[basic]") {
	// A prime has no proper divisor, so the splitter must reject it.
	uint64 const d = atl::detail::nontrivial_factor(15);
	REQUIRE((d == 3 || d == 5));
	CHECK(atl_test::aborts([] { (void)atl::detail::nontrivial_factor(atl::largest_prime_64); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::nontrivial_factor(1); }));
}

TEST_CASE("is_prime: Montgomery Miller-Rabin agrees with the sieve and the twelve-base witness set", "[basic]") {
	atl::sieve const s(10'000'000);
	for (std::size_t i = 0; i < s.size(); ++i) REQUIRE(atl::is_prime(i) == (i >= 2 && s.is_prime(i)));
	// strong pseudoprimes to small base sets, on either side of the 3-base cutoff
	for (uint64 n : {uint64(3215031751ull), uint64(4759123141ull), uint64(1122004669633ull),
					 uint64(3825123056546413051ull)}) {
		REQUIRE(atl::is_prime(n) == atl::experimental::is_prime_by_twelve_bases(n));
	}
	REQUIRE_FALSE(atl::is_prime(3215031751ull));
	REQUIRE_FALSE(atl::is_prime(3825123056546413051ull));
	REQUIRE(atl::is_prime(4759123129ull) == atl::experimental::is_prime_by_twelve_bases(4759123129ull));
	std::mt19937_64 rng(23);
	for (int i = 0; i < 200000; ++i) {
		uint64 const n = rng() >> (rng() % 40);
		REQUIRE(atl::is_prime(n) == atl::experimental::is_prime_by_twelve_bases(n));
	}
	REQUIRE(atl::is_prime(atl::largest_prime_64));
	REQUIRE(atl::is_prime(4294967291ull));
}

TEST_CASE("factorization: agrees with the remainder-product method on random and smooth inputs", "[basic]") {
	std::mt19937_64 rng(29);
	for (int i = 0; i < 3000; ++i) {
		uint64 n = rng() >> (rng() % 64);
		if (i % 3 == 0) n = (n % 100000 + 1) * 6720 * (rng() % 251 + 1); // smooth-heavy
		REQUIRE(atl::factorization(n) == atl::experimental::factorization_by_remainder_products(n));
	}
	REQUIRE(atl::factorization(uint64(1) << 63) == std::vector<atl::prime_factor>{{2, 63}});
	REQUIRE(atl::factorization(251ull * 251 * 257) == std::vector<atl::prime_factor>{{251, 2}, {257, 1}});
	REQUIRE(atl::factorization(65521ull * 65521) == std::vector<atl::prime_factor>{{65521, 2}});
}
