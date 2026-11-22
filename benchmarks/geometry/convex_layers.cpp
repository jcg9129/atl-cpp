/**
 * @file
 * @brief Onion-peeling throughput on uniform squares, disks and nested octagons.
 * One timed call sorts once and repeatedly scans surviving sorted points;
 * there is no alternative contender. Layer count drives the O(n L) scan cost.
 * Intel Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-10-15, -O2 / -O2 -march=native: square 100k 1991 / 2009 ms,
 * disk 100k 1797 / 1812 ms, 2500 rings 641 / 662 ms.
 * Full table: docs/convex-hull.md.
 */
#include <atl/geometry/convex_layers.hpp>
#include <atl/geometry/point.hpp>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using atl::geometry::convex_layers;
using lp = atl::geometry::point<std::int64_t>;

TEST_CASE("convex_layers benchmarks", "[geometry]") {
	std::mt19937_64 rng(1);

	std::vector<lp> uniform;
	for (int k = 0; k < 100'000; ++k)
		uniform.push_back({std::int64_t(rng() % 1'000'000),
						   std::int64_t(rng() % 1'000'000)});

	std::vector<lp> disk;
	while (disk.size() < 100'000) {
		std::int64_t const x = std::int64_t(rng() % 2'000'001) - 1'000'000;
		std::int64_t const y = std::int64_t(rng() % 2'000'001) - 1'000'000;
		if (x * x + y * y <= 1'000'000LL * 1'000'000) disk.push_back({x, y});
	}

	std::vector<lp> rings;
	for (std::int64_t r = 1; r <= 2500; ++r)
		for (auto const &[dx, dy] : std::vector<std::pair<std::int64_t, std::int64_t>>{
				 {2, 0}, {2, 1}, {0, 2}, {-2, 1}, {-2, 0}, {-2, -1}, {0, -2}, {2, -1}})
			rings.push_back({dx * r, dy * r});

	BENCHMARK("uniform square 100k") { return convex_layers(uniform).size(); };
	BENCHMARK("uniform disk 100k") { return convex_layers(disk).size(); };
	BENCHMARK("nested rings 20k (L = 2500)") { return convex_layers(rings).size(); };
}
