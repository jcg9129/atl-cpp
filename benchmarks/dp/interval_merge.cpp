/**
 * @file
 * @brief atl::optimal_interval_merge (Knuth's optimization) alone, no race:
 * throughput of one whole table on stone merging, where merging [l, r)
 * costs the total weight of its stones, n = 2000 and 5000 random weights.
 * Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-10-15, plain -O2 / -march=native: 18.1 / 19.5 ms at n = 2000, 214 /
 * 222 ms at n = 5000. Table: docs/dp-optimization.md.
 */
#include <atl/dp/interval_merge.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

using atl::int64;
using atl::uint32;

TEST_CASE("optimal_interval_merge", "[interval_merge]") {
	std::mt19937_64 rng(1);
	for (std::size_t const n : {2000u, 5000u}) {
		std::vector<int64> prefix(n + 1, 0);
		for (std::size_t i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + 1 + int64(rng() % 1'000'000);
		auto const cost = [&](uint32 l, uint32 r) { return prefix[r] - prefix[l]; };
		BENCHMARK("stone merging n=" + std::to_string(n)) { return atl::optimal_interval_merge(n, cost).value(0, n); };
	}
}
