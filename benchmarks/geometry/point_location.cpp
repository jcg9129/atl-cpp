/**
 * @file
 * @brief Full build and batch-query throughput on sparse/dense subdivisions:
 * library persistent slabs versus randomized incremental trapezoid-map search,
 * plus a single mutable locations_of sweep (including its frame construction).
 * Intel Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-10-15, -O2 / -O2 -march=native: dense build 375 / 381 ms versus
 * 671 / 707, 1e5 queries 182 / 203 versus 305 / 320; batch 221 / 233 ms.
 */
#include <atl/experimental/geometry/trapezoid_point_location.hpp>
#include <atl/geometry/arrangement.hpp>
#include <atl/geometry/point_location.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

using atl::int64;
using atl::experimental::point_location_by_trapezoid_map;
using atl::geometry::arrangement_of;
using atl::geometry::locations_of;
using atl::geometry::point;
using atl::geometry::point_location;
using sg = atl::geometry::segment<int64>;

TEST_CASE("point_location benchmarks", "[geometry]") {
	std::mt19937_64 rng(1);

	std::vector<sg> sparse;
	for (int k = 0; k < 5000; ++k) {
		int64 const x = int64(rng() % 1'000'000), y = int64(rng() % 1'000'000);
		sparse.push_back({{x, y},
						  {x + 1 + int64(rng() % 300), y + 1 + int64(rng() % 300)}});
	}
	std::vector<sg> dense;
	for (int k = 0; k < 700; ++k) {
		auto const c = [&] { return int64(rng() % 1'000'000); };
		dense.push_back({{c(), c()}, {c(), c()}});
	}
	auto const sparse_arr = arrangement_of(sparse);
	auto const dense_arr = arrangement_of(dense);
	std::vector<point<int64>> queries;
	for (int k = 0; k < 100'000; ++k)
		queries.push_back({int64(rng() % 1'100'000) - 50'000,
						   int64(rng() % 1'100'000) - 50'000});

	auto const build = []<typename Build>(std::string const &label, Build make)
		requires std::invocable<Build &>
	{
		BENCHMARK(label.c_str()) { return make().location_of({0, 0}).index; };
	};
	build("library build -- sparse (V=10k)", [&] { return point_location(sparse_arr, sparse); });
	build("by trapezoid map build -- sparse (V=10k)", [&] { return point_location_by_trapezoid_map(sparse_arr, sparse); });
	build("library build -- dense (V=60k, E=118k)", [&] { return point_location(dense_arr, dense); });
	build("by trapezoid map build -- dense (V=60k, E=118k)", [&] { return point_location_by_trapezoid_map(dense_arr, dense); });

	point_location const slabs_sparse(sparse_arr, sparse);
	point_location_by_trapezoid_map const trap_sparse(sparse_arr, sparse);
	point_location const slabs_dense(dense_arr, dense);
	point_location_by_trapezoid_map const trap_dense(dense_arr, dense);
	// Agreement is checked independently of the timed loop.
	for (auto const q : queries) {
		REQUIRE(slabs_sparse.location_of(q) == trap_sparse.location_of(q));
		REQUIRE(slabs_dense.location_of(q) == trap_dense.location_of(q));
	}

	auto const query = [&]<typename Query>(std::string const &label, Query locate)
		requires std::is_invocable_r_v<atl::geometry::location, Query &, point<int64>>
	{
		BENCHMARK(label.c_str()) {
			atl::uint64 acc = 0;
			for (auto const q : queries) acc += locate(q).index;
			return acc;
		};
	};
	query("library query -- 1e5 points, sparse", [&](point<int64> q) { return slabs_sparse.location_of(q); });
	query("by trapezoid map query -- 1e5 points, sparse", [&](point<int64> q) { return trap_sparse.location_of(q); });
	query("library query -- 1e5 points, dense", [&](point<int64> q) { return slabs_dense.location_of(q); });
	query("by trapezoid map query -- 1e5 points, dense", [&](point<int64> q) { return trap_dense.location_of(q); });

	auto const sparse_batch = locations_of(sparse_arr, sparse, queries);
	auto const dense_batch = locations_of(dense_arr, dense, queries);
	for (std::size_t i = 0; i < queries.size(); ++i) {
		REQUIRE(sparse_batch[i] == slabs_sparse.location_of(queries[i]));
		REQUIRE(dense_batch[i] == slabs_dense.location_of(queries[i]));
	}
	BENCHMARK("locations_of -- 1e5 points, sparse") { return locations_of(sparse_arr, sparse, queries).size(); };
	BENCHMARK("locations_of -- 1e5 points, dense") { return locations_of(dense_arr, dense, queries).size(); };
	std::vector<point<int64>> const no_queries;
	BENCHMARK("locations_of -- no queries, dense sweep") { return locations_of(dense_arr, dense, no_queries).size(); };
}
