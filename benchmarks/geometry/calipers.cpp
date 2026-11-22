/**
 * @file
 * @brief Caliper throughput: one complete antipodal or dual-support event stream
 * on large strict hulls, then hull-plus-fold applications on scattered clouds.
 * The mechanisms differ by one polygon, parallel supports or opposed supports;
 * each application folds the events for its own objective. No rival is timed.
 * Intel Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-10-15, -O2 / -O2 -march=native: antipodal 2.1 / 2.3 ms,
 * dual parallel and opposed 4.3 / 4.3 ms; diameter 12.4 / 11.6 ms.
 * Full table: docs/calipers.md.
 */
#include <atl/geometry/angular_sort.hpp>
#include <atl/geometry/calipers.hpp>
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <ranges>
#include <vector>

using atl::geometry::point;

TEST_CASE("calipers benchmarks", "[geometry]") {
	std::mt19937_64 rng(1);
	auto const coord = [&] { return atl::int64(rng() % 2'000'000'001) - 1'000'000'000; };

	// large hull: sorted random edge vectors accumulate to an exactly convex
	// polygon with ~1e5 vertices
	auto const big_hull = [&](std::size_t n) {
		// antipodal vector pairs sum to zero, so the accumulated polygon
		// CLOSES (a plain random-edge accumulation does not)
		std::vector<point<atl::int64>> edges;
		edges.reserve(n);
		while (edges.size() < n) {
			point<atl::int64> e{atl::int64(rng() % 20001) - 10000, atl::int64(rng() % 20001) - 10000};
			if (e == point<atl::int64>{}) continue;
			edges.push_back(e);
			edges.push_back(-e);
		}
		atl::geometry::sort_by_polar_angle(edges);
		std::vector<point<atl::int64>> turning;
		for (auto const e : edges) {
			if (!turning.empty() && cross(turning.back(), e) == 0)
				turning.back() += e;
			else
				turning.push_back(e);
		}
		std::vector<point<atl::int64>> poly;
		point<atl::int64> at{};
		for (auto const e : turning) {
			poly.push_back(at);
			at += e;
		}
		return poly;
	};
	auto const P = big_hull(100'000);
	auto const Q = big_hull(100'000);

	std::vector<point<atl::int64>> cloud(1'000'000);
	for (auto &p : cloud) p = {coord(), coord()};

	BENCHMARK("antipodal sweep, 1e5-vertex hull") {
		atl::int64 acc = 0;
		for (auto const ev : atl::geometry::antipodal_view(P)) acc += ev.opposite;
		return acc;
	};
	BENCHMARK("dual sweep parallel, 1e5 + 1e5") {
		atl::int64 acc = 0;
		for (auto const ev : atl::geometry::dual_support_view(P, Q)) acc += ev.p_support;
		return acc;
	};
	BENCHMARK("dual sweep anti, 1e5 + 1e5") {
		atl::int64 acc = 0;
		for (auto const ev : atl::geometry::dual_support_view(P, Q, atl::geometry::anti_parallel))
			acc += ev.q_support;
		return acc;
	};
	BENCHMARK("farthest_pair end-to-end, 1e6 cloud") {
		return atl::geometry::farthest_pair_indices(cloud)[0];
	};
	BENCHMARK("narrowest_strip end-to-end, 1e6 cloud") {
		return atl::geometry::narrowest_strip(cloud).opposite.x;
	};
	BENCHMARK("farthest_pair between two 5e5 clouds") {
		return atl::geometry::farthest_pair_between_indices(
			std::ranges::subrange(cloud.begin(), cloud.begin() + 500'000),
			std::ranges::subrange(cloud.begin() + 500'000, cloud.end()))[0];
	};
	BENCHMARK("distance_between_convex_hulls, two 5e5 clouds") {
		return atl::geometry::distance_between_convex_hulls(
			std::ranges::subrange(cloud.begin(), cloud.begin() + 500'000),
			std::ranges::subrange(cloud.begin() + 500'000, cloud.end()));
	};
	BENCHMARK("min_area_enclosing_rectangle, 1e6 cloud") {
		return atl::geometry::min_area_enclosing_rectangle(cloud)[2].x;
	};
}
