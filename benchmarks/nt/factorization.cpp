/** @file
 * @brief Batch throughput for rho versus trial division near 1e9/1e12;
 * twelve-base remainder-product primality versus the library witness sets;
 * rho on remainder products versus Montgomery with trial bounds 0, 64, 256.
 * Prime batches, random odds, semiprimes and smooth values exercise different
 * factor shapes. A dependent 5e6-product chain measures per-step latency of
 * remainder, Barrett and Montgomery multiplication.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, 2026-10-15, -O2 / -O2
 * -march=native: 200 semiprimes take 501/501 ms by remainder products versus
 * library 75.7/74.3 ms. Full rows are in docs/factorization.md.
 */
#include <atl/experimental/nt/factorization_trial.hpp>
#include <atl/experimental/nt/factorization_variants.hpp>
#include <atl/nt/basic.hpp>
#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using atl::uint64;
namespace ex = atl::experimental;

namespace {
	std::vector<uint64> random_odds(uint64 lo, uint64 hi, int count, std::mt19937_64 &rng) {
		atl_assert(lo <= hi && count >= 0);
		std::vector<uint64> values(static_cast<std::size_t>(count));
		std::uniform_int_distribution<uint64> dist(lo, hi);
		for (auto &x : values) x = dist(rng) | 1;
		return values;
	}

	/** @brief `count` primes selected by testing odd candidates from the bounded generator. */
	std::vector<uint64> random_primes(uint64 lo, uint64 hi, int count, std::mt19937_64 &rng) {
		atl_assert(lo <= hi && count >= 0);
		std::vector<uint64> values;
		std::uniform_int_distribution<uint64> dist(lo, hi);
		while (int(values.size()) < count)
			if (uint64 const x = dist(rng) | 1; atl::is_prime(x)) values.push_back(x);
		return values;
	}
}

TEST_CASE("factorization: Pollard's rho vs trial division", "[factorization]") {
	std::mt19937_64 rng(0xC0FFEE);

	auto compare = [&](std::string const &label, uint64 lo, uint64 hi, int count) {
		auto inputs = random_odds(lo, hi, count, rng);

		for (uint64 n : inputs)
			REQUIRE(atl::factorization(n) == atl::experimental::factorization_by_trial_division(n));

		auto const run = [&]<typename Factorize>(Factorize factorize)
			requires std::is_invocable_r_v<std::vector<atl::prime_factor>, Factorize &, uint64>
		{
			uint64 acc = 0;
			for (uint64 n : inputs) acc += factorize(n).size();
			return acc;
		};
		BENCHMARK("library " + label) {
			return run([](uint64 n) { return atl::factorization(n); });
		};
		BENCHMARK("by trial division " + label) {
			return run([](uint64 n) { return atl::experimental::factorization_by_trial_division(n); });
		};
	};

	compare("~1e9", 1'000'000'000ull, 1'000'100'000ull, 100);
	compare("~1e12", 1'000'000'000'000ull, 1'000'000'100'000ull, 20);
}

TEST_CASE("factorization: 64-bit contenders", "[factorization]") {
	std::mt19937_64 rng(0xFAC7);
	{
		constexpr std::size_t iters = 5'000'000;
		uint64 volatile modulus = (uint64(1) << 62) + 135; // odd, not prime-specific
		uint64 const check_modulus = modulus;
		atl::barrett64 const br(check_modulus);
		atl::detail::montgomery64 const mg(check_modulus);
		uint64 residue = 3, barrett_form = 3, montgomery_form = mg.to_form(3);
		uint64 const multiplier = 0x123456789abcdefull % check_modulus;
		for (std::size_t i = 0; i < iters; ++i) {
			residue = uint64(atl::uint128(residue) * multiplier % check_modulus);
			barrett_form = br.mul(barrett_form, multiplier);
			montgomery_form = mg.mul(montgomery_form, mg.to_form(multiplier));
		}
		REQUIRE(residue == barrett_form);
		REQUIRE(residue == mg.from_form(montgomery_form));

		BENCHMARK("mul chain 5e6, 64-bit runtime mod | uint128 %") {
			uint64 const m = modulus;
			uint64 a = 3, b = 0x123456789abcdefull % m;
			for (std::size_t i = 0; i < iters; ++i) a = uint64(atl::uint128(a) * b % m);
			return a;
		};
		BENCHMARK("mul chain 5e6, 64-bit runtime mod | barrett64") {
			atl::barrett64 const br(modulus);
			uint64 a = 3, b = 0x123456789abcdefull % br.mod();
			for (std::size_t i = 0; i < iters; ++i) a = br.mul(a, b);
			return a;
		};
		BENCHMARK("mul chain 5e6, 64-bit runtime mod | montgomery64") {
			atl::detail::montgomery64 const mg(modulus);
			uint64 a = mg.to_form(3), b = mg.to_form(0x123456789abcdefull % modulus);
			for (std::size_t i = 0; i < iters; ++i) a = mg.mul(a, b);
			return a;
		};
	}
	auto primality = [&](std::string const &label, std::vector<uint64> const &inputs) {
		for (uint64 n : inputs) REQUIRE(ex::is_prime_by_twelve_bases(n) == atl::is_prime(n));
		auto const run = [&]<typename Test>(Test test)
			requires std::is_invocable_r_v<bool, Test &, uint64>
		{
			uint64 acc = 0;
			for (uint64 n : inputs) acc += test(n);
			return acc;
		};
		BENCHMARK("is_prime " + label + " | by twelve bases") {
			return run([](uint64 n) { return ex::is_prime_by_twelve_bases(n); });
		};
		BENCHMARK("is_prime " + label + " | library") {
			return run([](uint64 n) { return atl::is_prime(n); });
		};
	};
	primality("1e4 primes ~1e9", random_primes(1'000'000'000ull, 2'000'000'000ull, 10000, rng));
	primality("1e4 primes ~1e12", random_primes(1'000'000'000'000ull, 2'000'000'000'000ull, 10000, rng));
	primality("1e4 primes ~2^62", random_primes(uint64(1) << 62, (uint64(1) << 62) + (uint64(1) << 50), 10000, rng));
	primality("1e5 random 64-bit odds", random_odds(uint64(1) << 32, ~uint64(0), 100000, rng));

	auto factoring = [&](std::string const &label, std::vector<uint64> const &inputs) {
		for (uint64 n : inputs) {
			auto const want = atl::factorization(n);
			REQUIRE(ex::factorization_by_remainder_products(n) == want);
			REQUIRE(ex::factorization_by_bounded_trial<0>(n) == want);
			REQUIRE(ex::factorization_by_bounded_trial<64>(n) == want);
		}
		auto const run = [&]<typename Factorize>(Factorize factorize)
			requires std::is_invocable_r_v<std::vector<atl::prime_factor>, Factorize &, uint64>
		{
			uint64 acc = 0;
			for (uint64 n : inputs) acc += factorize(n).size();
			return acc;
		};
		BENCHMARK("factorization " + label + " | by remainder products") {
			return run([](uint64 n) { return ex::factorization_by_remainder_products(n); });
		};
		BENCHMARK("factorization " + label + " | by bounded trial 0") {
			return run([](uint64 n) { return ex::factorization_by_bounded_trial<0>(n); });
		};
		BENCHMARK("factorization " + label + " | by bounded trial 64") {
			return run([](uint64 n) { return ex::factorization_by_bounded_trial<64>(n); });
		};
		BENCHMARK("factorization " + label + " | library") {
			return run([](uint64 n) { return atl::factorization(n); });
		};
	};
	{
		auto const p = random_primes(uint64(1) << 30, uint64(1) << 31, 400, rng);
		std::vector<uint64> semiprimes;
		for (std::size_t i = 0; i + 1 < p.size(); i += 2) semiprimes.push_back(p[i] * p[i + 1]);
		factoring("200 semiprimes of two ~2^31 primes", semiprimes);
	}
	{
		std::vector<uint64> random(2000);
		for (auto &x : random) x = rng();
		factoring("2000 random 64-bit", random);
	}
	{
		std::vector<uint64> smooth(2000); // products of primes below 1000
		auto const trial_primes = random_primes(3, 1000, 50, rng);
		for (auto &x : smooth) {
			x = 1;
			while (x < (uint64(1) << 40)) x *= trial_primes[static_cast<std::size_t>(rng() % trial_primes.size())];
		}
		factoring("2000 smooth ~2^40+", smooth);
	}
}
