/**
 * @file
 * @brief Hull throughput: Andrew with adaptive octagon pruning, unpruned chain,
 * Graham angular sort, Quickhull splits, Chan mini-hulls and Jarvis wrapping.
 * Each timed call builds a hull of a uniform, circle or Gaussian cloud; the
 * Minkowski row merges finished circle hulls. Jarvis omits the circle because
 * its O(n h) work would require about 1e11 orientations.
 * Intel Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-10-15, -O2 / -O2 -march=native: library uniform 12.1 / 11.3 ms,
 * unpruned 136 / 136; circle 132 / 132 vs unpruned 133 / 132.
 * Full measured rows: docs/convex-hull.md.
 */
#include <atl/experimental/geometry/convex_hull_variants.hpp>
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <ranges>
#include <string>
#include <type_traits>
#include <vector>

namespace {
	template<typename Call>
		requires std::invocable<Call &>
	void measure(std::string const &name, Call call) {
		BENCHMARK(std::string(name)) { return std::invoke(call); };
	}
}

using atl::geometry::point;

TEST_CASE("convex hull benchmarks", "[geometry]") {
	constexpr std::size_t n = 1'000'000;
	std::mt19937_64 rng(1);
	auto const coord = [&] { return atl::int64(rng() % 2'000'000'001) - 1'000'000'000; };

	std::vector<point<atl::int64>> uniform(n);
	for (auto &p : uniform) p = {coord(), coord()};

	auto const circle_cloud = [&](std::size_t const count, atl::int64 const cx) {
		std::vector<point<atl::int64>> v(count);
		for (auto &p : v) {
			double const a = double(rng() % (1 << 30)) / double(1 << 30) * 6.283185307179586;
			p = {cx + atl::int64(std::llround(1e9 * std::cos(a))), atl::int64(std::llround(1e9 * std::sin(a)))};
		}
		return v;
	};
	auto const circle = circle_cloud(n, 0);

	std::vector<point<atl::int64>> cluster(n);
	std::normal_distribution<double> gauss(0.0, 1e5);
	for (auto &p : cluster) p = {atl::int64(gauss(rng)), atl::int64(gauss(rng))};

	std::vector<point<double>> uniformd(n);
	for (std::size_t i = 0; i < n; ++i) uniformd[i] = {double(uniform[i].x), double(uniform[i].y)};

	/** @brief Runs the same monotone chain without the adaptive octagon filter. */
	auto const convex_hull_indices_by_unpruned_chain = []<atl::geometry::vertex_range R>(R const &pts) {
		return atl::geometry::detail::hull_indices<true>(pts, atl::geometry::exact{}, false).size();
	};

	for (auto const *cloud : std::array<std::vector<point<atl::int64>> const *, 3>{&uniform, &circle, &cluster}) {
		auto const expected = atl::geometry::convex_hull_indices(*cloud);
		REQUIRE(atl::geometry::detail::hull_indices<true>(*cloud, atl::geometry::exact{}, false) == expected);
		REQUIRE(atl::experimental::convex_hull_indices_by_graham_scan(*cloud) == expected);
		REQUIRE(atl::experimental::convex_hull_indices_by_quickhull(*cloud) == expected);
		REQUIRE(atl::experimental::convex_hull_indices_by_chan(*cloud) == expected);
	}
	REQUIRE(atl::experimental::convex_hull_indices_by_jarvis_march(uniform) == atl::geometry::convex_hull_indices(uniform));

	measure("library, uniform 1e6", [&] { return atl::geometry::convex_hull_indices(uniform).size(); });
	measure("by unpruned chain, uniform 1e6", [&] { return convex_hull_indices_by_unpruned_chain(uniform); });
	measure("library, circle 1e6", [&] { return atl::geometry::convex_hull_indices(circle).size(); });
	measure("by unpruned chain, circle 1e6", [&] { return convex_hull_indices_by_unpruned_chain(circle); });
	measure("library, cluster 1e6", [&] { return atl::geometry::convex_hull_indices(cluster).size(); });
	measure("by unpruned chain, cluster 1e6", [&] { return convex_hull_indices_by_unpruned_chain(cluster); });
	measure("library, uniform double 1e6", [&] {
		return atl::geometry::convex_hull_indices(uniformd).size();
	});

	measure("by Graham scan, uniform 1e6", [&] { return atl::experimental::convex_hull_indices_by_graham_scan(uniform).size(); });
	measure("by Graham scan, circle 1e6", [&] { return atl::experimental::convex_hull_indices_by_graham_scan(circle).size(); });
	measure("by Quickhull, uniform 1e6", [&] { return atl::experimental::convex_hull_indices_by_quickhull(uniform).size(); });
	measure("by Quickhull, cluster 1e6", [&] { return atl::experimental::convex_hull_indices_by_quickhull(cluster).size(); });
	measure("by Quickhull, circle 1e6", [&] { return atl::experimental::convex_hull_indices_by_quickhull(circle).size(); });
	measure("by Chan, uniform 1e6", [&] { return atl::experimental::convex_hull_indices_by_chan(uniform).size(); });
	measure("by Chan, circle 1e6", [&] { return atl::experimental::convex_hull_indices_by_chan(circle).size(); });
	measure("by Jarvis march, uniform 1e6 (h ~ 30)", [&] {
		return atl::experimental::convex_hull_indices_by_jarvis_march(uniform).size();
	});
	// jarvis on the circle would be n * h ~ 1e11 orientation calls: skipped

	auto const second = circle_cloud(n, 500'000'000);
	auto const hull_a = atl::geometry::convex_hull(circle), hull_b = atl::geometry::convex_hull(second);

	measure("minkowski_sum, two circle hulls", [&] { return atl::geometry::minkowski_sum(hull_a, hull_b).size(); });
}
