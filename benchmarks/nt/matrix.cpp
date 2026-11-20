/**
 * @file
 * @brief Matrix product latency: library lazy i-k-j versus Montgomery i-k-j,
 * Montgomery transpose, lazy transpose, 256-column tiles and per-term subtraction.
 * Each sample allocates and computes one square product at n=200, 300, 500.
 * Additional samples measure power (n=100, exponent=1e18), min-plus (n=200),
 * determinant / Euclidean determinant / inverse (n=200, 500), GF(2) solve (n=2000),
 * and matrix-tree counting (n=500, m=10000; 54.7 / 55.0 ms).
 * Intel Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions disabled,
 * 2026-10-14, -O2 -g -DNDEBUG -std=c++23 with/without -march=native:
 * library n=500 80.4 / 32.1 ms; Montgomery i-k-j 190 / 189 ms.
 * Tiles give 81.5 / 32.1 ms, so the simpler library loop stays. Full tables: docs/matrix.md.
 */
#include <atl/experimental/nt/matrix_variants.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/graph/spanning_tree_count.hpp>
#include <atl/nt/bit_matrix.hpp>
#include <atl/nt/gaussian_elimination.hpp>
#include <atl/nt/matrix.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	using mint = atl::mod_int32<998244353>;

	atl::matrix<mint> random_matrix(uint32 n, std::mt19937_64 &rng) {
		atl::matrix<mint> m(n, n);
		for (auto &x : m.entries()) x = mint(rng());
		return m;
	}
}

TEST_CASE("matrix product mod 998244353: loop orders and reduction", "[matrix]") {
	namespace ex = atl::experimental;
	std::mt19937_64 rng(1);
	for (uint32 const n : {200u, 300u, 500u}) {
		auto const a = random_matrix(n, rng), b = random_matrix(n, rng);
		std::span<mint const> const sa = a.entries(), sb = b.entries();
		std::string const size = " n=" + std::to_string(n);
		auto const expected = ex::matrix_product_by_montgomery_ikj<mint>(sa, sb, n);
		auto run = [&]<typename Product>(std::string const &label, Product product)
			requires std::invocable<Product &>
				  && (std::same_as<std::invoke_result_t<Product &>, atl::matrix<mint>>
					  || std::same_as<std::invoke_result_t<Product &>, std::vector<mint>>)
		{
			auto const check = product();
			if constexpr (std::same_as<std::invoke_result_t<Product &>, atl::matrix<mint>>)
				REQUIRE(std::ranges::equal(check.entries(), expected));
			else
				REQUIRE(check == expected);
			BENCHMARK(label + size) { return product(); };
		};
		run("library", [&] { return a * b; });
		run("by montgomery ikj", [&] { return ex::matrix_product_by_montgomery_ikj<mint>(sa, sb, n); });
		run("by montgomery transpose", [&] { return ex::matrix_product_by_montgomery_transpose<mint>(sa, sb, n); });
		run("by lazy transpose", [&] { return ex::matrix_product_by_lazy_transpose<mint>(sa, sb, n); });
		run("by lazy tiles 256", [&] { return ex::matrix_product_by_lazy_tiles<mint, 256>(sa, sb, n); });
		run("by lazy subtract", [&] { return ex::matrix_product_by_lazy_subtract<mint>(sa, sb, n); });
	}
}

TEST_CASE("matrix: power, semiring and elimination costs", "[matrix]") {
	std::mt19937_64 rng(2);
	{
		auto const a = random_matrix(100, rng);
		BENCHMARK("power n=100 exp=1e18") { return atl::power(a, uint64(1'000'000'000'000'000'000ull)); };
	}
	{
		using shortest = atl::min_plus_semiring<int64>;
		atl::matrix<int64, shortest> w(200, 200);
		for (auto &x : w.entries()) x = int64(rng() % 1'000'000);
		BENCHMARK("min-plus product n=200") { return w * w; };
	}
	for (uint32 const n : {200u, 500u}) {
		auto const a = random_matrix(n, rng);
		std::string const size = " n=" + std::to_string(n);
		REQUIRE(atl::determinant_any_modulus(a) == atl::determinant(a));
		BENCHMARK("determinant" + size) { return atl::determinant(a); };
		BENCHMARK("determinant_any_modulus" + size) { return atl::determinant_any_modulus(a); };
		BENCHMARK("inverse" + size) { return atl::inverse(a); };
	}
	{
		uint32 const n = 2000;
		atl::bit_matrix a(n, n);
		for (uint32 i = 0; i < n; ++i)
			for (uint32 j = 0; j < n; ++j)
				if (rng() & 1) a.set(i, j);
		std::vector<bool> b(n);
		for (uint32 i = 0; i < n; ++i) b[i] = rng() & 1;
		BENCHMARK("bit_matrix solve n=2000") { return atl::solve_linear_system(a, b); };
	}
	{
		uint32 const n = 500;
		std::vector<atl::graph::arc<>> edges;
		for (uint32 i = 0; i < 20 * n; ++i) edges.push_back({uint32(rng() % n), uint32(rng() % n)});
		BENCHMARK("count_spanning_trees n=500 m=10000") { return atl::graph::count_spanning_trees<mint>(n, edges); };
	}
}
