/**
 * @file
 * @brief atl::geometry::manhattan_mst (four octant sweeps whose active points
 *        sit in an ordered map, matched entries erased) raced against
 *        atl::experimental::manhattan_mst_by_fenwick_prefix_minimum, which
 *        collects the same candidate edges through a Fenwick prefix minimum
 *        over ranked x.
 *
 * Each row is one full call (throughput) on int64 points: 1e5 uniform in a 1e6
 * square; 4e5 uniform over a 1e9 span; 1e5 on a 300 by 300 grid, heavy with
 * duplicates and ties; a noisy descending staircase of 1e5 points with every x
 * distinct. Both contenders must give trees of equal weight.
 *
 * Result (Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, asserts off,
 * 2026-10-15; ms, `-O2` / `-march=native`): the library wins the uniform
 * shapes, 174 / 175 against 182 / 178 at 1e5 and 976 / 966 against 1134 /
 * 1109 at 4e5, ties the grid (129 / 129 against 131 / 128) and loses the
 * staircase, 128 / 125 against 117 / 114. The full table is on the guide page
 * (docs/manhattan-mst.md).
 */
#include <atl/experimental/geometry/manhattan_mst_variants.hpp>
#include <atl/geometry/manhattan_mst.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using atl::geometry::manhattan_mst;
using lp = atl::geometry::point<std::int64_t>;

namespace {
	using tree = std::vector<std::array<atl::geometry::vertex_index, 2>>;

	/** @brief The L1 weight of @p edges over @p pts. */
	atl::int128 tree_weight(std::vector<lp> const &pts, tree const &edges) {
		atl::int128 total = 0;
		for (auto const &e : edges) {
			atl::int128 const dx = atl::int128(pts[e[0]].x) - pts[e[1]].x, dy = atl::int128(pts[e[0]].y) - pts[e[1]].y;
			total += (dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy);
		}
		return total;
	}

	/** @brief Checks that @p contender's tree weighs as the library's on @p pts, then times it under @p label. */
	template<typename Contender>
		requires std::is_invocable_r_v<tree, Contender const &, std::vector<lp> const &>
	void measure(std::string const &label, std::vector<lp> const &pts, Contender const &contender) {
		REQUIRE((tree_weight(pts, contender(pts)) == tree_weight(pts, manhattan_mst(pts))));
		BENCHMARK(std::string(label)) { return contender(pts).size(); };
	}
}

TEST_CASE("manhattan_mst benchmarks", "[geometry]") {
	std::mt19937_64 rng(1);
	auto const cloud = [&](int const count, std::int64_t const span) {
		std::vector<lp> v;
		for (int k = 0; k < count; ++k)
			v.push_back({std::int64_t(rng() % std::uint64_t(span)), std::int64_t(rng() % std::uint64_t(span))});
		return v;
	};
	auto const uniform = cloud(100'000, 1'000'000);
	auto const large = cloud(400'000, 1'000'000'000);
	auto const gridded = cloud(100'000, 300);
	std::vector<lp> staircase;
	for (int k = 0; k < 100'000; ++k)
		staircase.push_back({std::int64_t(k) * 7, -std::int64_t(k) * 5 + std::int64_t(rng() % 50)});
	std::ranges::shuffle(staircase, rng);

	auto const library = [](std::vector<lp> const &pts) { return manhattan_mst(pts); };
	auto const by_fenwick_prefix_minimum = [](std::vector<lp> const &pts) {
		return atl::experimental::manhattan_mst_by_fenwick_prefix_minimum(pts);
	};
	auto const race = [&](std::string const &shape, std::vector<lp> const &pts) {
		measure("library -- " + shape, pts, library);
		measure("by Fenwick prefix minimum -- " + shape, pts, by_fenwick_prefix_minimum);
	};
	race("uniform 100k", uniform);
	race("uniform 400k, 1e9 span", large);
	race("gridded 100k (many ties)", gridded);
	race("staircase 100k (distinct x)", staircase);
}
