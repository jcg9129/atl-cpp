/**
 * @file
 * @brief atl::geometry::max_points_in_circle (per anchor point, a coverage arc
 *        for every neighbor within 2r, the neighbors found through a grid of
 *        cell side a little over 2r, then the radial sweep) and
 *        atl::geometry::max_arc_overlap on their own. Nothing is raced: a
 *        variant scanning all pairs instead of the grid took 225 ms on uniform
 *        5000 and 787 ms on dense 2000 (Xeon Platinum 8280, GCC 15.2, -O2,
 *        2026-10-14) and is not kept.
 *
 * Each row is one full call (throughput). Shapes: uniform clouds in a 10k box
 * at r = 300; a dense 2000-point cloud where every pair is within 2r, the worst
 * case (every anchor sorts n - 1 arcs and the grid cannot help); one tight
 * 500-point cluster in a uniform background, where most anchors cannot beat
 * the cluster's count and skip their sweep; 1e5 points in a 1e6 box; and
 * max_arc_overlap on 1e6 random arcs.
 *
 * Result (Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, asserts off,
 * 2026-10-15; ms, `-O2` / `-march=native`): uniform 5000 42.3 / 41.5, dense
 * 2000 283 / 278, 1e5 in a 1e6 box 56.0 / 52.8, 1e6 arcs 187 / 190. The full
 * table is on the guide page (docs/radial-sweep.md).
 */
#include <atl/geometry/point.hpp>
#include <atl/geometry/radial_sweep.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using atl::geometry::max_points_in_circle;
using lp = atl::geometry::point<std::int64_t>;

namespace {
	/** @brief Times @p call under @p label; its count keeps the call alive. */
	template<typename Call>
		requires std::is_invocable_r_v<atl::uint32, Call const &>
	void measure(std::string const &label, Call const &call) {
		BENCHMARK(std::string(label)) { return call(); };
	}
}

TEST_CASE("max_points_in_circle benchmarks", "[geometry]") {
	std::mt19937_64 rng(1);
	auto const cloud = [&](int const count, std::int64_t const span) {
		std::vector<lp> v;
		for (int k = 0; k < count; ++k)
			v.push_back({std::int64_t(rng() % std::uint64_t(span)), std::int64_t(rng() % std::uint64_t(span))});
		return v;
	};
	auto const uniform5k = cloud(5000, 10'000);
	std::vector<lp> const uniform2k(uniform5k.begin(), uniform5k.begin() + 2000);
	auto const dense2k = cloud(2000, 500);
	auto clustered = cloud(4500, 10'000);
	for (int k = 0; k < 500; ++k)
		clustered.push_back({5000 + std::int64_t(rng() % 100), 5000 + std::int64_t(rng() % 100)});
	auto const sparse = cloud(100'000, 1'000'000);

	auto const cover = [](std::vector<lp> const &pts, double const r) {
		return [&pts, r] { return max_points_in_circle(pts, r).count; };
	};
	measure("library, uniform 2000, r = 300", cover(uniform2k, 300.0));
	measure("library, uniform 5000, r = 300", cover(uniform5k, 300.0));
	measure("library, dense 2000, r = 400 (all arcs)", cover(dense2k, 400.0));
	measure("library, cluster 500 + background 4500, r = 300", cover(clustered, 300.0));
	measure("library, uniform 1e5 in a 1e6 box, r = 300", cover(sparse, 300.0));

	std::vector<atl::geometry::arc> arcs(1'000'000);
	std::uniform_real_distribution<double> angle(-10.0, 10.0), span(0.0, 1.0);
	for (auto &a : arcs) {
		double const from = angle(rng);
		a = {from, from + span(rng)};
	}
	measure("max_arc_overlap library, 1e6 random arcs", [&arcs] { return atl::geometry::max_arc_overlap(arcs).count; });
}
