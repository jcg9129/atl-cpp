/**
 * @file
 * @brief Throughput of circle containment, circle relations and randomized
 * incremental enclosing circles on uniform int64 clouds. No contender is raced.
 * Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-15:
 * -O2 / -O2 -march=native, ms: enclosing 1e6 12.5 / 11.7 (spread up to 2.59x),
 * relation 1e6 pairs 16.0 / 18.7, containment 1e6 pairs 4.3 / 4.8.
 * Full table: docs/circle.md.
 */
#include <atl/geometry/circle.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <vector>

using atl::geometry::circle;
using atl::geometry::point;

TEST_CASE("circle benchmarks", "[geometry]") {
	std::mt19937_64 rng(1);
	auto const coord = [&] { return atl::int64(rng() % 2'000'000'001) - 1'000'000'000; };
	std::vector<point<atl::int64>> cloud(1'000'000);
	for (auto &p : cloud) p = {coord(), coord()};

	std::vector<circle<atl::int64>> circles(2'000'000);
	for (auto &c : circles) c = {{coord() % 1'000'000, coord() % 1'000'000}, atl::int64(rng() % 1'000'000)};

	BENCHMARK("minimum_enclosing_circle, 1e6 cloud") {
		return atl::geometry::minimum_enclosing_circle(cloud).radius;
	};
	BENCHMARK("relation, 1e6 circle pairs") {
		int acc = 0;
		for (std::size_t i = 0; i < 1'000'000; ++i)
			acc += int(relation(circles[2 * i], circles[2 * i + 1]));
		return acc;
	};
	BENCHMARK("contains, 1e6 circle-point") {
		int acc = 0;
		for (std::size_t i = 0; i < 1'000'000; ++i)
			acc += int(contains(circles[i], cloud[i]));
		return acc;
	};
}
