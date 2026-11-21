/**
 * @file
 * @brief atl::exact_count_optimum (Lagrangian relaxation) over
 * atl::optimal_segmentation, against atl::optimal_partition as the reference
 * for one count: cutting n = 1e5 prefix-summed items (steps below 1000) into
 * exactly k segments of squared sum. Throughput of one whole solve; each
 * probe is one O(n log n) optimal_segmentation, about 52 probes over the
 * price range [-1, S^2 + 1], against the layered O(k n log n) solver at k =
 * 20. Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-10-15, plain -O2 / -march=native: the relaxation costs 228-251 ms
 * whatever k (k = 20: 250 / 251), the layered solver 55.1 / 50.9 ms at k =
 * 20. Table: docs/exact-count-optimum.md.
 */
#include <atl/dp/exact_count_optimum.hpp>
#include <atl/dp/optimal_partition.hpp>
#include <atl/dp/segmentation.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <compare>
#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

using atl::int64;
using atl::uint32;

namespace {
	/**
	 * @brief A value with a segment count, ordered lexicographically so ties
	 *        go to fewer segments (benchmarks do not see the test headers,
	 *        so the test's copy is repeated here).
	 */
	struct counted {
		int64 value = 0, count = 0;

		friend constexpr counted operator+(counted const &a, counted const &b) {
			return {a.value + b.value, a.count + b.count};
		}

		friend constexpr auto operator<=>(counted const &, counted const &) = default;
	};
}

TEST_CASE("exact_count_optimum", "[exact_count_optimum]") {
	std::mt19937_64 rng(1);
	std::size_t const n = 100'000;
	std::vector<int64> s(n + 1, 0);
	for (std::size_t i = 1; i <= n; ++i) s[i] = s[i - 1] + int64(rng() % 1000);
	auto const cost = [&](uint32 j, uint32 i) {
		int64 const d = s[i] - s[j];
		return d * d;
	};
	auto solve = [&](int64 price) {
		auto const r = atl::optimal_segmentation(n, [&](uint32 j, uint32 i) { return counted{cost(j, i) + price, 1}; });
		return std::pair{r.value[n].value, r.value[n].count};
	};
	REQUIRE(atl::exact_count_optimum(int64(20), int64(-1), s[n] * s[n] + 1, solve).value == atl::optimal_partition(n, 20, cost).back());
	for (std::size_t const k : {20, 1000, 50'000}) {
		BENCHMARK("exact_count_optimum over optimal_segmentation n=1e5 k=" + std::to_string(k)) {
			return atl::exact_count_optimum(int64(k), int64(-1), s[n] * s[n] + 1, solve).value;
		};
	}
	BENCHMARK("optimal_partition n=1e5 k=20") { return atl::optimal_partition(n, 20, cost).back(); };
}
