/**
 * @file
 * @brief Per-call costs of the segment predicate, the polar-angle sorts and the
 *        polygon queries. One race: atl::geometry::sort_by_polar_angle (an
 *        unstable std::ranges::sort with polar_angle_less) against the same
 *        comparator passed to std::ranges::stable_sort.
 *
 * The sort rows sort a copy of 1e6 points uniform in a 2e9 square (int64, the
 * same values narrowed to int32 or converted to double, under the exact and
 * the fast policy) or of 1e6 multiples of 441 directions (same-ray ties); the
 * polar_angle_order rows compute the stable argsort. The intersects rows call
 * the predicate on 1e6 segment pairs: short segments in a 1000 square, which
 * mostly cross, and short segments scattered over the full range, which mostly
 * miss and leave through the early exit. The polygon rows are 1000
 * point-in-polygon queries against a 1e5-vertex star and one polygon_area2
 * over it. A bounding-box-first polygon contains lost on the int64 star (937
 * against 664 ms, -O2, 2026-10-14) and is not kept.
 *
 * Result (Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, asserts off,
 * 2026-10-15; ms, `-O2` / `-march=native`): sort_by_polar_angle 147 / 148 on
 * int64 against 150 / 149 by stable_sort; intersects 24.1 / 30.9 ns per call
 * on dense int64 hits, 39.0 / 38.6 on doubles (14.9 / 15.2 with fast);
 * contains 6.7 / 7.9 ns per vertex. The full table is on the guide page
 * (docs/polygon.md).
 */
#include <atl/geometry/angular_sort.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <random>
#include <ranges>
#include <string>
#include <vector>

using atl::geometry::fast;
using atl::geometry::point;
using atl::geometry::polar_angle_less;
using atl::geometry::segment;

namespace {
	/** @brief Times @p sorter on a fresh copy of @p points under @p label; the first point keeps the call alive. */
	template<atl::geometry::coordinate T, typename Sorter>
		requires std::invocable<Sorter const &, std::vector<point<T>> &>
	void time_sort(std::string const &label, std::vector<point<T>> const &points, Sorter const &sorter) {
		BENCHMARK(std::string(label)) {
			auto v = points;
			sorter(v);
			return v.front();
		};
	}
}

TEST_CASE("segment, polygon and polar-angle benchmarks", "[geometry]") {
	constexpr std::size_t n = 1'000'000;
	std::mt19937_64 rng(1);
	auto const coord = [&] { return atl::int64(rng() % 2'000'000'001) - 1'000'000'000; };

	std::vector<point<atl::int64>> pts(n);
	for (auto &p : pts) p = {coord(), coord()};
	std::vector<point<atl::int32>> pts32(n);
	for (std::size_t i = 0; i < n; ++i) pts32[i] = {atl::int32(pts[i].x), atl::int32(pts[i].y)};
	std::vector<point<double>> ptsd(n);
	for (std::size_t i = 0; i < n; ++i) ptsd[i] = {double(pts[i].x), double(pts[i].y)};
	// heavy same-ray ties: 1000 directions, many multiples along each
	std::vector<point<atl::int64>> rays(n);
	for (auto &p : rays) {
		atl::int64 const dx = atl::int64(rng() % 21) - 10, dy = atl::int64(rng() % 21) - 10, k = 1 + atl::int64(rng() % 1000);
		p = {dx * k, dy * k};
	}

	auto const library = [](auto &v) { atl::geometry::sort_by_polar_angle(v); };
	auto const library_fast = [](auto &v) { atl::geometry::sort_by_polar_angle(v, fast{}); };
	auto const by_stable_sort = [](auto &v) { std::ranges::stable_sort(v, polar_angle_less{}); };
	time_sort("sort_by_polar_angle library, int64", pts, library);
	time_sort("sort_by_polar_angle library, int32", pts32, library);
	time_sort("sort_by_polar_angle by stable_sort, int64", pts, by_stable_sort);
	time_sort("sort_by_polar_angle library, double, exact", ptsd, library);
	time_sort("sort_by_polar_angle library, double, fast", ptsd, library_fast);

	BENCHMARK("polar_angle_order, int64 1e6") { return atl::geometry::polar_angle_order(pts)[0]; };
	BENCHMARK("polar_angle_order, double 1e6") { return atl::geometry::polar_angle_order(ptsd)[0]; };
	BENCHMARK("polar_angle_order, int64 same-ray ties 1e6") { return atl::geometry::polar_angle_order(rays)[0]; };

	std::vector<segment<atl::int64>> segs(2 * n);
	for (auto &s : segs) s = {{coord() % 1000, coord() % 1000}, {coord() % 1000, coord() % 1000}};
	std::vector<segment<double>> segsd(2 * n);
	for (std::size_t i = 0; i < 2 * n; ++i)
		segsd[i] = {{double(segs[i].a.x), double(segs[i].a.y)}, {double(segs[i].b.x), double(segs[i].b.y)}};
	// sparse hits: short segments scattered over the full range
	std::vector<segment<atl::int64>> sparse(2 * n);
	for (auto &s : sparse) {
		point<atl::int64> const a{coord(), coord()};
		s = {a, {a.x + coord() / 1000, a.y + coord() / 1000}};
	}
	std::vector<segment<double>> sparsed(2 * n);
	for (std::size_t i = 0; i < 2 * n; ++i)
		sparsed[i] = {{double(sparse[i].a.x), double(sparse[i].a.y)}, {double(sparse[i].b.x), double(sparse[i].b.y)}};

	auto const intersects_rows = [&]<typename T>(std::string const &shape, std::vector<segment<T>> const &v) {
		BENCHMARK("intersects " + shape) {
			int acc = 0;
			for (std::size_t i = 0; i < n; ++i) acc += intersects(v[2 * i], v[2 * i + 1]);
			return acc;
		};
		BENCHMARK("intersects fast " + shape) {
			int acc = 0;
			for (std::size_t i = 0; i < n; ++i) acc += intersects(v[2 * i], v[2 * i + 1], fast{});
			return acc;
		};
	};
	intersects_rows("int64, short segments, dense hits", segs);
	intersects_rows("double, short segments, dense hits", segsd);
	intersects_rows("int64, sparse hits", sparse);
	intersects_rows("double, sparse hits", sparsed);

	// point-in-polygon: cost per vertex on a large star polygon
	std::vector<point<atl::int64>> poly;
	{
		std::vector<point<atl::int64>> raw(100'000);
		for (auto &p : raw) p = {coord(), coord()};
		atl::geometry::sort_by_polar_angle(raw);
		for (auto const p : raw)
			if (poly.empty() || atl::geometry::orientation(point<atl::int64>{}, poly.back(), p) != 0) poly.push_back(p);
	}
	std::vector<point<atl::int64>> queries(1000);
	for (auto &q : queries) q = {coord(), coord()};
	BENCHMARK("polygon contains, 1e5-vertex star x 1000 queries") {
		int acc = 0;
		for (auto const q : queries) acc += int(atl::geometry::contains(poly, q));
		return acc;
	};
	BENCHMARK("polygon_area2, 1e5-vertex") { return atl::int64(atl::geometry::polygon_area2(poly) % 1009); };
}
