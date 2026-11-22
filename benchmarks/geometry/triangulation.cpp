/**
 * @file
 * @brief Full-call triangulation throughput over convex, reflex-heavy star,
 * adversarial comb and collinear-repair shapes; the last row directly runs
 * monotone decomposition, the others use the adaptive library dispatch.
 * Intel Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-10-15, -O2 / -O2 -march=native: convex nominal 1e5 4.4 / 4.9 ms,
 * star 184 / 187, double star 192 / 194, comb 22.7 / 23.7,
 * collinear-top comb 131 / 134. Convex generation merges parallel edges.
 */
#include <atl/geometry/angular_sort.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/triangulation.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using atl::geometry::point;
using atl::geometry::triangulation_of;

// Ear clipping is O(n * reflex-count): near-linear on convex input, O(n^2)
// on adversarial combs. These pin the three regimes.
TEST_CASE("triangulation benchmarks", "[geometry]") {
	std::mt19937_64 rng(1);

	// exactly convex: random edge vectors sorted by polar angle and accumulated
	// (zero reflex vertices — the linear regime; a circle ROUNDED to integers
	// is not convex, its rounding noise makes ~half the vertices reflex)
	std::vector<point<atl::int64>> convex;
	{
		// antipodal vector pairs sum to zero: the polygon closes exactly
		std::vector<point<atl::int64>> edges;
		edges.reserve(100'000);
		while (edges.size() < 100'000) {
			point<atl::int64> e{atl::int64(rng() % 2001) - 1000, atl::int64(rng() % 2001) - 1000};
			if (e == point<atl::int64>{}) continue;
			edges.push_back(e);
			edges.push_back(-e);
		}
		atl::geometry::sort_by_polar_angle(edges);
		// merge parallel edges: straight vertices are blockers too (they
		// must be, to prevent T-vertices), so true reflex = 0 needs strictly
		// turning corners
		std::vector<point<atl::int64>> turning;
		for (auto const e : edges) {
			if (!turning.empty() && cross(turning.back(), e) == 0)
				turning.back() += e;
			else
				turning.push_back(e);
		}
		point<atl::int64> at{};
		for (auto const e : turning) {
			convex.push_back(at);
			at += e;
		}
	}

	// star-shaped: 1e5 spokes around a center (reflex ~ half)
	std::vector<point<atl::int64>> star;
	{
		std::vector<point<atl::int64>> raw(100'000);
		for (auto &p : raw)
			p = {atl::int64(rng() % 2'000'001) - 1'000'000, atl::int64(rng() % 2'000'001) - 1'000'000};

		struct keyed_point {
			point<atl::int64> value;
			double angle;
		};

		std::vector<keyed_point> keyed;
		for (auto const value : raw) keyed.push_back({value, std::atan2(double(value.y), double(value.x))});
		std::ranges::sort(keyed, {}, &keyed_point::angle);
		for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = keyed[i].value;
		for (auto const p : raw) {
			if (!star.empty() && cross(star.back(), p) == 0 && dot(star.back(), p) > 0) {
				star.back() = p; // same ray from the origin: keep one
				continue;
			}
			if (p != point<atl::int64>{}) star.push_back(p);
		}
	}

	// adversarial comb: 1e4 teeth, every tip's ear test scans all reflex roots
	std::vector<point<atl::int64>> comb;
	{
		atl::int64 const teeth = 10'000;
		for (atl::int64 i = 0; i < teeth; ++i) {
			comb.push_back({2 * i, 0});
			comb.push_back({2 * i + 1, 3});
		}
		comb.push_back({2 * teeth, 0});
		comb.push_back({2 * teeth, 6});
		comb.push_back({0, 6});
	}

	std::vector<point<double>> stard;
	for (auto const p : star) stard.push_back({double(p.x), double(p.y)});

	BENCHMARK("exactly convex, 1e5 (reflex = 0)") { return triangulation_of(convex).triangles.size(); };
	BENCHMARK("random star, 1e5 (reflex ~ n/2)") { return triangulation_of(star).triangles.size(); };
	BENCHMARK("random star, 1e5, double coords") { return triangulation_of(stard).triangles.size(); };
	BENCHMARK("adversarial comb, 2e4 (library)") { return triangulation_of(comb).triangles.size(); };

	// zero-area repair load: 2.5e4 teeth under a top edge subdivided into
	// 5e4 collinear vertices, so the sweep's monotone pieces emit long fans
	// of degenerate triangles that the repair flips (one O(1) lookup each)
	std::vector<point<atl::int64>> collinear_comb;
	{
		atl::int64 const teeth = 25'000;
		for (atl::int64 i = 0; i < teeth; ++i) {
			collinear_comb.push_back({2 * i, 0});
			collinear_comb.push_back({2 * i + 1, 5});
		}
		collinear_comb.push_back({2 * teeth, 0});
		for (atl::int64 x = 2 * teeth; x >= 0; --x) collinear_comb.push_back({x, 10});
	}
	BENCHMARK("collinear-top comb, 1e5 (sweep, degenerate repair)") {
		return atl::geometry::detail::monotone_sweep_triangulation(collinear_comb).triangles.size();
	};
}
