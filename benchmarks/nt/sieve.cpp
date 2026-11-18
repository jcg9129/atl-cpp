/** @file
 * @brief Throughput of linear sieve builds (modulo test versus least-prime
 * comparison), factor walks (size_t versus uint32 and a scalar exponent sum),
 * prime enumeration (segmented versus one byte array), and multiplicative
 * tables (two passes versus a fused sieve). Shapes: 1e6/1e7 builds and tables,
 * all factor queries below 1e7, prime lists through 1e8. See docs/sieve.md and
 * docs/multiplicative-functions.md for all rows.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, 2026-10-15, -O2 / -O2
 * -march=native: segmented n=1e8 162/152 ms versus 617/646 ms unsegmented;
 * fused totient n=1e7 427/440 ms versus library 259/258 ms.
 */
#include <atl/ds/static_vector.hpp>
#include <atl/experimental/nt/sieve_variants.hpp>
#include <atl/nt/multiplicative.hpp>
#include <atl/nt/sieve.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

using namespace atl;
namespace ex = atl::experimental;

namespace {
	/** @brief Factor through size_t divisions rather than uint32 divisions. */
	static_vector<prime_factor, 9> factorization_by_size_type_walk(
		std::vector<sieve_item_info> const &info, std::size_t i) {
		static_vector<prime_factor, 9> factors;
		while (i > 1) {
			sieve_item_info const &it = info[i];
			factors.push_back({it.prime, it.exponent});
			i /= it.prime_power;
		}
		return factors;
	}

	/** @brief Accumulate exponents directly, omitting the factor-result object. */
	uint64 exponent_sum_by_uint32_walk(std::vector<sieve_item_info> const &info, uint32 i) {
		uint64 sum = 0;
		while (i > 1) {
			sieve_item_info const &it = info[i];
			sum += it.exponent;
			i /= it.prime_power;
		}
		return sum;
	}
}

TEST_CASE("sieve contenders", "[sieve]") {
	for (std::size_t const n : {std::size_t(1'000'000), std::size_t(10'000'000)}) {
		std::string const shape = "n=" + std::to_string(n);
		REQUIRE(sieve_info(n) == ex::sieve_info_by_least_prime_comparison(n));
		BENCHMARK("sieve build " + shape + " | by least prime comparison") {
			return ex::sieve_info_by_least_prime_comparison(n)[n - 1].prime;
		};
		BENCHMARK("sieve build " + shape + " | library") { return sieve_info(n)[n - 1].prime; };
	}
	{
		std::size_t const n = 10'000'000;
		sieve const s(n);
		auto const info = sieve_info(n);
		for (std::size_t i = 1; i < n; ++i) {
			REQUIRE(factorization_by_size_type_walk(info, i) == s.factorization(i));
			uint64 sum = 0;
			for (auto const &f : s.factorization(i)) sum += f.exponent;
			REQUIRE(exponent_sum_by_uint32_walk(info, uint32(i)) == sum);
		}
		auto const run = [&]<typename ExponentSum>(ExponentSum sum_exponents)
			requires std::is_invocable_r_v<uint64, ExponentSum &, uint32>
		{
			uint64 sum = 0;
			for (uint32 i = 1; i < n; ++i) sum += sum_exponents(i);
			return sum;
		};
		BENCHMARK("factorization(i) all i < 1e7 | by size type walk") {
			return run([&](uint32 i) {
				uint64 sum = 0;
				for (auto const &f : factorization_by_size_type_walk(info, i)) sum += f.exponent;
				return sum;
			});
		};
		BENCHMARK("factorization(i) all i < 1e7 | library") {
			return run([&](uint32 i) {
				uint64 sum = 0;
				for (auto const &f : s.factorization(i)) sum += f.exponent;
				return sum;
			});
		};
		BENCHMARK("factorization(i) all i < 1e7 | by uint32 exponent sum") {
			return run([&](uint32 i) { return exponent_sum_by_uint32_walk(info, i); });
		};
		BENCHMARK("prime_factors(i) all i < 1e7 | library") {
			uint64 sum = 0;
			for (std::size_t i = 1; i < n; ++i) sum += s.prime_factors(i).size();
			return sum;
		};
	}
	for (std::size_t const n : {std::size_t(1'000'000), std::size_t(10'000'000), std::size_t(100'000'000)}) {
		std::string const shape = "n=" + std::to_string(n);
		REQUIRE(prime_range(n) == ex::prime_range_by_unsegmented_sieve(n));
		BENCHMARK("prime_range " + shape + " | by unsegmented sieve") {
			return ex::prime_range_by_unsegmented_sieve(n).size();
		};
		BENCHMARK("prime_range " + shape + " | library") { return prime_range(n).size(); };
	}
	for (std::size_t const n : {std::size_t(1'000'000), std::size_t(10'000'000)}) {
		std::string const shape = "n=" + std::to_string(n);
		auto const phi = [](uint64 p, uint64, uint64 v, std::vector<uint32> const &) {
			return uint32(v / p * (p - 1));
		};
		auto const tau = [](uint64, uint64 e, uint64, std::vector<uint32> const &) { return uint32(e + 1); };
		auto const sigma = [](uint64 p, uint64, uint64 v, std::vector<uint64> const &) {
			return (v * p - 1) / (p - 1);
		};
		REQUIRE(euler_totient_range(n) == ex::multiplicative_fn_range_by_fused_sieve<uint32>(n, phi));
		REQUIRE(divisor_count_range(n) == ex::multiplicative_fn_range_by_fused_sieve<uint32>(n, tau));
		REQUIRE(divisor_sum_range(n) == ex::multiplicative_fn_range_by_fused_sieve<uint64>(n, sigma));
		BENCHMARK("euler_totient_range " + shape + " | library") { return euler_totient_range(n)[n - 1]; };
		BENCHMARK("euler_totient_range " + shape + " | by fused sieve") {
			return ex::multiplicative_fn_range_by_fused_sieve<uint32>(n, phi)[n - 1];
		};
		BENCHMARK("divisor_count_range " + shape + " | library") { return divisor_count_range(n)[n - 1]; };
		BENCHMARK("divisor_count_range " + shape + " | by fused sieve") {
			return ex::multiplicative_fn_range_by_fused_sieve<uint32>(n, tau)[n - 1];
		};
		BENCHMARK("divisor_sum_range " + shape + " | library") { return divisor_sum_range(n)[n - 1]; };
		BENCHMARK("divisor_sum_range " + shape + " | by fused sieve") {
			return ex::multiplicative_fn_range_by_fused_sieve<uint64>(n, sigma)[n - 1];
		};
	}
}
