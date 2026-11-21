/**
 * @file
 * @brief 1D/1D race: atl::optimal_segmentation (the monotone decision queue,
 * O(n log n) evaluations) against three experimental contenders that differ
 * in how they find each boundary's best decision: halving divide and conquer
 * across halves with the monotone divide and conquer (O(n log^2 n)) or with
 * SMAWK (O(n log n)), and SMAWK on doubling blocks (O(n)). Throughput of one
 * whole solve at n = 1e5 and 1e6, segment cost of [j, i) over prefix sums S:
 * "cheap" = (S_i - S_j - L)^2 (two loads, a multiply), "pow1.5" = |S_i - S_j
 * - L|^1.5 (a sqrt). Contenders agree exactly (decisions included) before
 * timing. Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-10-15, plain -O2 / -march=native: the library is fastest on three
 * shapes of four by 1.05-1.8x (1e6 cheap: 50.2 / 52.4 ms; SMAWK blocks 81.9 /
 * 74.7); the SMAWK blocks win the 1e6 pow1.5 shape by 5-6% (87.7 / 86.4
 * against 93.7 / 91.0). Full table: docs/dp-optimization.md.
 */
#include <atl/dp/segmentation.hpp>
#include <atl/experimental/dp/segmentation_variants.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace ex = atl::experimental;
using atl::float64;
using atl::int64;
using atl::uint32;

TEST_CASE("segmentation contenders", "[segmentation]") {
	std::mt19937_64 rng(1);

	auto race = [&](std::string const &shape, std::size_t n) {
		std::vector<int64> s(n + 1, 0);
		for (std::size_t i = 1; i <= n; ++i) s[i] = s[i - 1] + 1 + int64(rng() % 100);
		int64 const target = 50 * 40; // segments of about 40 items
		auto const cheap = [&](uint32 j, uint32 i) {
			int64 const d = s[i] - s[j] - target;
			return d * d;
		};
		auto const pow15 = [&](uint32 j, uint32 i) {
			float64 const d = std::abs(float64(s[i] - s[j] - target));
			return d * std::sqrt(d);
		};
		auto const want = atl::optimal_segmentation(n, cheap);
		REQUIRE(ex::optimal_segmentation_by_halving_divide_and_conquer(n, cheap).decision == want.decision);
		REQUIRE(ex::optimal_segmentation_by_halving_smawk(n, cheap).decision == want.decision);
		REQUIRE(ex::optimal_segmentation_by_smawk_blocks(n, cheap).decision == want.decision);
		auto const want15 = atl::optimal_segmentation(n, pow15);
		REQUIRE(ex::optimal_segmentation_by_halving_divide_and_conquer(n, pow15).value == want15.value);
		REQUIRE(ex::optimal_segmentation_by_halving_smawk(n, pow15).value == want15.value);
		REQUIRE(ex::optimal_segmentation_by_smawk_blocks(n, pow15).value == want15.value);

		BENCHMARK(shape + " cheap | library") { return atl::optimal_segmentation(n, cheap).value[n]; };
		BENCHMARK(shape + " cheap | by halving divide and conquer") { return ex::optimal_segmentation_by_halving_divide_and_conquer(n, cheap).value[n]; };
		BENCHMARK(shape + " cheap | by halving SMAWK") { return ex::optimal_segmentation_by_halving_smawk(n, cheap).value[n]; };
		BENCHMARK(shape + " cheap | by SMAWK blocks") { return ex::optimal_segmentation_by_smawk_blocks(n, cheap).value[n]; };
		BENCHMARK(shape + " pow1.5 | library") {
			return atl::optimal_segmentation(n, pow15).value[n];
		};
		BENCHMARK(shape + " pow1.5 | by halving divide and conquer") { return ex::optimal_segmentation_by_halving_divide_and_conquer(n, pow15).value[n]; };
		BENCHMARK(shape + " pow1.5 | by halving SMAWK") { return ex::optimal_segmentation_by_halving_smawk(n, pow15).value[n]; };
		BENCHMARK(shape + " pow1.5 | by SMAWK blocks") { return ex::optimal_segmentation_by_smawk_blocks(n, pow15).value[n]; };
	};

	race("n=1e5", 100'000);
	race("n=1e6", 1'000'000);
}
