/**
 * @file
 * @brief Throughput of incircle evaluations (library adaptive filter versus
 * incircle_sign_by_int256, which always evaluates widened integer products),
 * Delaunay on uniform integer/double clouds and a cocircular grid, Voronoi dual
 * extraction and largest empty circles. Each row measures full-call throughput.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-15,
 * -O2 / -O2 -march=native: incircle int64 26.2 / 23.2 versus int256 64.6 / 61.5
 * ms per 1e6; Delaunay uniform int64 346 / 360 ms per 2e5, double 331 / 322;
 * grid 160 / 159; dual 50.8 / 49.5; empty circle 32.4 / 33.3 per 2e4,
 * 432 / 445 per 2e5. Known nearest sites avoid an all-sites candidate scan,
 * which measured 3751 ms at 2e4 on the same CPU/compiler, -O2, 2026-10-14.
 * Full table: docs/delaunay.md.
 */
#include <atl/geometry/delaunay.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <ranges>
#include <type_traits>
#include <vector>

using atl::geometry::point;

namespace {
	/** @brief Evaluates every integer incircle directly in int256, bypassing the adaptive filter. */
	template<atl::geometry::coordinate T>
		requires std::signed_integral<T>
	int incircle_sign_by_int256(point<T> a, point<T> b, point<T> c, point<T> q) {
		return atl::geometry::detail::incircle_sign_int256(a, b, c, q);
	}
}

TEST_CASE("delaunay benchmarks", "[geometry]") {
	std::mt19937_64 rng(1);
	auto const coord = [&] { return atl::int64(rng() % 2'000'000'001) - 1'000'000'000; };
	std::vector<point<atl::int64>> queries(4'000'000);
	for (auto &p : queries) p = {coord(), coord()};
	std::vector<point<atl::int32>> queries32(queries.size());
	for (std::size_t k = 0; k < queries.size(); ++k)
		queries32[k] = {atl::int32(queries[k].x / 2), atl::int32(queries[k].y / 2)};
	for (std::size_t k = 0; k < queries.size(); k += 4) {
		REQUIRE(atl::geometry::detail::incircle_sign(queries[k], queries[k + 1], queries[k + 2], queries[k + 3]) == atl::geometry::detail::incircle_sign_int256(queries[k], queries[k + 1], queries[k + 2], queries[k + 3]));
		REQUIRE(atl::geometry::detail::incircle_sign(queries32[k], queries32[k + 1], queries32[k + 2], queries32[k + 3]) == atl::geometry::detail::incircle_sign_int256(queries32[k], queries32[k + 1], queries32[k + 2], queries32[k + 3]));
	}
	auto const incircle_loop = []<atl::geometry::vertex_range R, typename Sign>(R const &pts, Sign const &sign)
		requires std::is_invocable_r_v<int, Sign const &, std::ranges::range_value_t<R>, std::ranges::range_value_t<R>, std::ranges::range_value_t<R>, std::ranges::range_value_t<R>>
	{
		int total = 0;
		for (std::size_t k = 0; k + 3 < std::ranges::size(pts); k += 4) total += sign(pts[k], pts[k + 1], pts[k + 2], pts[k + 3]);
		return total;
	};
	BENCHMARK("incircle int64 by int256, 1e6 calls") {
		return incircle_loop(queries, [](atl::geometry::point<atl::int64> a, atl::geometry::point<atl::int64> b, atl::geometry::point<atl::int64> c, atl::geometry::point<atl::int64> q) {
			return incircle_sign_by_int256(a, b, c, q);
		});
	};
	BENCHMARK("incircle int64 library, 1e6 calls") {
		return incircle_loop(queries, [](atl::geometry::point<atl::int64> a, atl::geometry::point<atl::int64> b, atl::geometry::point<atl::int64> c, atl::geometry::point<atl::int64> q) {
			return atl::geometry::detail::incircle_sign(a, b, c, q);
		});
	};
	BENCHMARK("incircle int32 by int256, 1e6 calls") {
		return incircle_loop(queries32, [](atl::geometry::point<atl::int32> a, atl::geometry::point<atl::int32> b, atl::geometry::point<atl::int32> c, atl::geometry::point<atl::int32> q) {
			return incircle_sign_by_int256(a, b, c, q);
		});
	};
	BENCHMARK("incircle int32 library, 1e6 calls") {
		return incircle_loop(queries32, [](atl::geometry::point<atl::int32> a, atl::geometry::point<atl::int32> b, atl::geometry::point<atl::int32> c, atl::geometry::point<atl::int32> q) {
			return atl::geometry::detail::incircle_sign(a, b, c, q);
		});
	};

	std::vector<point<atl::int64>> const uniform(queries.begin(), queries.begin() + 200'000);
	std::vector<point<double>> uniformd;
	for (auto const q : uniform) uniformd.push_back({double(q.x), double(q.y)});
	std::vector<point<atl::int64>> grid; // cocircular quadruples everywhere
	for (atl::int64 x = 0; x < 450; ++x)
		for (atl::int64 y = 0; y < 450; ++y) grid.push_back({x * 1000, y * 1000});
	std::ranges::shuffle(grid, rng);

	BENCHMARK("delaunay int64, uniform 2e5") { return atl::geometry::delaunay(uniform).triangles.size(); };
	BENCHMARK("delaunay double, uniform 2e5") { return atl::geometry::delaunay(uniformd).triangles.size(); };
	BENCHMARK("delaunay int64, cocircular grid 450x450") {
		return atl::geometry::delaunay(grid).triangles.size();
	};
	auto const d = atl::geometry::delaunay(uniform);
	BENCHMARK("voronoi_from_delaunay, uniform 2e5") {
		return atl::geometry::voronoi_from_delaunay(uniform, d).edges.size();
	};

	std::vector<point<atl::int64>> const subset(queries.begin(), queries.begin() + 20'000);
	BENCHMARK("largest_empty_circle, uniform 2e4") { return atl::geometry::largest_empty_circle(subset).radius; };
	BENCHMARK("largest_empty_circle, uniform 2e5") { return atl::geometry::largest_empty_circle(uniform).radius; };
}
