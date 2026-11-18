/**
 * @file
 * @brief Recurrence-term latency at k=1e18, orders 10, 32, 64, 100, 1000, 100000.
 * Bostan-Mori uses transform-domain rounds, convolution products, or schoolbook
 * products; Kitamasa uses convolution/Newton division or schoolbook reduction;
 * companion matrix power runs at order 10. Quadratic contenders stop at 1000.
 * The same large index prevents short-prefix squarings from dominating Kitamasa.
 * Each sample includes allocation. Separate Berlekamp-Massey samples use 2d terms,
 * d=1000, 5000. Intel Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions
 * disabled, 2026-10-15, -O2 -g -DNDEBUG -std=c++23 with/without -march=native:
 * transform Bostan-Mori d=100000 1204 / 1162 ms versus convolution 2270 / 2243;
 * schoolbook Kitamasa d=10 0.018 / 0.018 ms. Full tables: docs/linear-recurrence.md.
 */
#include <atl/experimental/nt/linear_recurrence_variants.hpp>
#include <atl/nt/linear_recurrence.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using atl::uint64;
namespace ex = atl::experimental;

namespace {
	using mint = atl::mod_int32<998244353>;

	std::vector<mint> random_vector(std::size_t n, std::mt19937_64 &rng) {
		std::vector<mint> v(n);
		for (auto &x : v) x = mint(rng());
		return v;
	}
}

TEST_CASE("linear recurrence k-th term: Bostan-Mori vs Kitamasa", "[linear_recurrence]") {
	std::mt19937_64 rng(1);
	uint64 const k = 1'000'000'000'000'000'000ull;
	for (std::size_t const d : {10, 32, 64, 100, 1000, 100'000}) {
		auto const c = random_vector(d, rng), a = random_vector(d, rng);
		std::string const shape = " d=" + std::to_string(d);
		mint const expected = atl::linear_recurrence_term(c, a, k);
		auto run = [&]<typename Term>(std::string const &label, Term term)
			requires std::invocable<Term &> && std::same_as<std::invoke_result_t<Term &>, mint>
		{
			REQUIRE(term() == expected);
			BENCHMARK(label + shape) { return term(); };
		};
		run("by transform bostan mori", [&] { return ex::linear_recurrence_term_by_transform_bostan_mori(c, a, k); });
		run("by convolution bostan mori", [&] { return ex::linear_recurrence_term_by_convolution_bostan_mori(c, a, k); });
		run("by convolution kitamasa", [&] { return ex::linear_recurrence_term_by_convolution_kitamasa(c, a, k); });
		if (d <= 1000) {
			run("by schoolbook bostan mori", [&] { return ex::linear_recurrence_term_by_schoolbook_bostan_mori(c, a, k); });
			run("by schoolbook kitamasa", [&] { return ex::linear_recurrence_term_by_schoolbook_kitamasa(c, a, k); });
		}
		if (d <= 10)
			run("by matrix power", [&] { return ex::linear_recurrence_term_by_matrix_power(c, a, k); });
	}
}

TEST_CASE("berlekamp_massey", "[linear_recurrence]") {
	std::mt19937_64 rng(2);
	for (std::size_t const d : {1000, 5000}) {
		auto const c = random_vector(d, rng);
		auto s = random_vector(d, rng);
		for (std::size_t i = d; i < 2 * d; ++i) {
			mint next = 0;
			for (std::size_t j = 0; j < d; ++j) next += c[j] * s[i - 1 - j];
			s.push_back(next);
		}
		BENCHMARK("berlekamp_massey 2d terms d=" + std::to_string(d)) { return atl::berlekamp_massey(s); };
	}
}
