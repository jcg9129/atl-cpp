/** @file
 * @brief Throughput of 1e5 two-class intersections with shared-factor moduli
 * below 1e9, three-prime Garner reconstructions, and linear congruences below
 * 2^63. Each row uses the library entry point; there are no contenders.
 * Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, 2026-10-15, -O2 / -O2
 * -march=native: intersections 8.5/8.5 ms, Garner 26.3/26.7 ms, linear
 * congruences 23.0/23.0 ms. Full table: docs/crt.md.
 */
#include <atl/nt/crt.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <numeric>
#include <random>
#include <vector>

using atl::uint64;

TEST_CASE("crt: merges and reconstructions", "[crt]") {
	constexpr int count = 100'000;
	std::mt19937_64 rng(11);
	std::vector<std::array<atl::congruence<uint64>, 2>> pairs(count);
	for (auto &p : pairs) {
		// moduli share a random factor so the lcm stays below 2^64
		uint64 const shared = rng() % 1000 + 1, m1 = shared * (rng() % 1'000'000 + 1),
					 m2 = shared * (rng() % 1'000'000 + 1), x = rng();
		p = {atl::congruence<uint64>{x % m1, m1}, atl::congruence<uint64>{x % m2, m2}};
	}
	for (auto const &pair : pairs) {
		auto const result = atl::chinese_remainder(pair[0], pair[1]);
		REQUIRE(result.has_value());
		for (auto const &c : pair) REQUIRE(result->residue % c.modulus == c.residue);
	}

	BENCHMARK("chinese_remainder pair, moduli<1e9") {
		uint64 acc = 0;
		for (auto const &p : pairs) acc += atl::chinese_remainder(p[0], p[1])->residue;
		return acc;
	};

	constexpr uint64 primes[3] = {167772161, 469762049, 998244353};
	std::vector<std::array<atl::congruence<uint64>, 3>> triples(count);
	for (auto &t : triples)
		for (int i = 0; i < 3; ++i) t[i] = {rng() % primes[i], primes[i]};
	for (auto const &triple : triples) {
		std::array<atl::congruence<atl::uint128>, 3> wide{};
		for (std::size_t i = 0; i < wide.size(); ++i) wide[i] = {triple[i].residue, triple[i].modulus};
		auto const full = atl::chinese_remainder(wide);
		REQUIRE(atl::chinese_remainder_mod(triple, uint64(1'000'000'007)) == uint64(full->residue % 1'000'000'007));
	}

	BENCHMARK("chinese_remainder_mod 3 NTT primes -> 1e9+7") {
		uint64 acc = 0;
		for (auto const &t : triples) acc += atl::chinese_remainder_mod(t, uint64(1'000'000'007));
		return acc;
	};

	std::vector<std::array<uint64, 3>> equations(count);
	for (auto &e : equations) e = {rng(), rng(), rng() >> 1 | 1};
	for (auto const &[a, c, m] : equations) {
		auto const result = atl::solve_linear_congruence(a, c, m);
		uint64 const divisor = std::gcd(a, m);
		REQUIRE(result.has_value() == (c % divisor == 0));
		if (result) REQUIRE(atl::mul_mod(a, result->residue, m) == c % m);
	}

	BENCHMARK("solve_linear_congruence m<2^63") {
		uint64 acc = 0;
		for (auto const &[a, c, m] : equations) {
			auto const r = atl::solve_linear_congruence(a, c, m);
			acc += r ? r->residue : 0;
		}
		return acc;
	};
}
