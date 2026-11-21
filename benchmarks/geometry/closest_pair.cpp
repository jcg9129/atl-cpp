/**
 * @file
 * @brief atl::geometry::closest_pair_indices (an x-sweep over a window of
 *        active points kept in a set ordered by y) raced against
 *        atl::experimental::closest_pair_indices_by_divide_and_conquer (a
 *        recursion on the x-sorted points with a mergesort by y riding along)
 *        and atl::experimental::closest_pair_indices_by_randomized_grid (a hash
 *        grid at half the running best distance, rebuilt on every
 *        improvement).
 *
 * Each row is one full call on 1e6 int64 points (throughput). Shapes: uniform
 * in a 2e9 square, about one point in the x window; a Gaussian cluster with
 * duplicates, where the distance-0 exit decides; the same cluster without
 * duplicates; a strip x < 1000 and one vertical line, distinct points whose
 * window holds almost every point, the sweep's worst case. The grid runs only
 * where it has a chance (uniform, cluster). Every contender must find the
 * library's distance.
 *
 * Result (Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, asserts off,
 * 2026-10-15; ms, `-O2` / `-march=native`): the library wins every shape but
 * the vertical line, e.g. 162 / 163 against 290 / 275 (divide and conquer) and
 * 1201 / 1126 (grid) on uniform points; on the vertical line divide and
 * conquer takes 261 / 264 against 519 / 516. The full table is on the guide
 * page (docs/closest-pair.md).
 */
#include <atl/experimental/geometry/closest_pair_variants.hpp>
#include <atl/geometry/closest_pair.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

using atl::geometry::point;

namespace {
	using pt = point<atl::int64>;

	/** @brief Checks that @p contender finds the library's distance on @p pts, then times it under @p label. */
	template<typename Contender>
		requires std::is_invocable_r_v<std::array<atl::geometry::vertex_index, 2>, Contender const &, std::vector<pt> const &>
	void measure(std::string const &label, std::vector<pt> const &pts, Contender const &contender) {
		auto const [i, j] = atl::geometry::closest_pair_indices(pts);
		auto const [ci, cj] = contender(pts);
		REQUIRE(ci != cj);
		REQUIRE((squared_distance(pts[ci], pts[cj]) == squared_distance(pts[i], pts[j])));
		BENCHMARK(std::string(label)) { return contender(pts)[0]; };
	}
}

TEST_CASE("closest pair benchmarks", "[geometry]") {
	constexpr std::size_t n = 1'000'000;
	std::mt19937_64 rng(1);
	auto const coord = [&] { return atl::int64(rng() % 2'000'000'001) - 1'000'000'000; };
	std::vector<pt> uniform(n);
	for (auto &p : uniform) p = {coord(), coord()};
	std::vector<pt> cluster(n);
	std::normal_distribution<double> gauss(0.0, 3e4);
	for (auto &p : cluster) p = {atl::int64(gauss(rng)), atl::int64(gauss(rng))};
	auto const distinct = [&](std::vector<pt> v) {
		std::ranges::sort(v);
		v.erase(std::ranges::unique(v).begin(), v.end());
		std::ranges::shuffle(v, rng);
		return v;
	};
	std::vector<pt> const dedup_cluster = distinct(cluster);
	std::vector<pt> strip(n);
	for (auto &p : strip) p = {atl::int64(rng() % 1000), coord()};
	strip = distinct(strip);
	std::vector<pt> line(n);
	for (auto &p : line) p = {0, coord()};
	line = distinct(line);

	auto const library = [](std::vector<pt> const &pts) { return atl::geometry::closest_pair_indices(pts); };
	auto const by_divide_and_conquer = [](std::vector<pt> const &pts) {
		return atl::experimental::closest_pair_indices_by_divide_and_conquer(pts);
	};
	auto const by_randomized_grid = [](std::vector<pt> const &pts) {
		return atl::experimental::closest_pair_indices_by_randomized_grid(pts);
	};
	auto const race = [&](std::string const &shape, std::vector<pt> const &pts, bool const with_grid) {
		measure("library, " + shape, pts, library);
		measure("by divide and conquer, " + shape, pts, by_divide_and_conquer);
		if (with_grid) measure("by randomized grid, " + shape, pts, by_randomized_grid);
	};
	race("uniform 1e6", uniform, true);
	race("cluster 1e6", cluster, true);
	race("dedup cluster", dedup_cluster, false);
	race("strip x < 1000, 1e6", strip, false);
	race("vertical line, 1e6", line, false);
}
