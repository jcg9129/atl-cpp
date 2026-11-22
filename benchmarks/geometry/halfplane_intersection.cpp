/**
 * @file
 * @brief Complete half-plane intersection throughput: library dispatch versus
 * key/comparator/radix sorting, exact/product-permanent/Dekker/rounded
 * determinants, every-front/envelope/corner-cache passes and std::deque storage.
 * Each loop includes sorting, allocation and region construction on tangent,
 * redundant, empty, sorted and rotated-run inputs (int64 and double); a hull-edge
 * row compares the convex polygon operation; large-coordinate polygon rows
 * exercise automatic wide dispatch near 2^60 (not measured yet). Each contender changes the named
 * mechanism and is checked against the matching public policy outside timing.
 * Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-11-02, -O2 -g -DNDEBUG -std=c++23, optionally -march=native:
 * 500000 constraints, two ascending runs: library 35.8 / 37.1 ms versus
 * by key sort 41.9 / 42.6 ms. Other mechanisms lack supplied publication rows.
 */
#include <atl/experimental/geometry/halfplane_intersection_variants.hpp>
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/convex_polygon_operations.hpp>
#include <atl/geometry/halfplane.hpp>
#include <atl/geometry/halfplane_intersection.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using atl::int64;
using atl::geometry::halfplane;
using atl::geometry::point;

namespace {
	namespace ex = atl::experimental;
	namespace gd = atl::geometry::detail;

	// tangents of the radius-1e9 circle at random angles: nothing redundant
	std::vector<halfplane<int64>> tangents(std::size_t const n, std::mt19937_64 &rng) {
		std::vector<halfplane<int64>> h(n);
		for (auto &c : h) {
			double const a = double(rng() % (uint64_t(1) << 40)) * 0x1p-40 * 6.283185307179586;
			int64 const p = std::llround(0x1p30 * std::cos(a)), q = std::llround(0x1p30 * std::sin(a));
			c = halfplane<int64>::at_most(p, q, std::llround(1e9 * std::hypot(double(p), double(q))));
		}
		return h;
	}

	// random half-planes containing the origin: mostly redundant
	std::vector<halfplane<int64>> around_origin(std::size_t const n, std::mt19937_64 &rng) {
		std::vector<halfplane<int64>> h(n);
		auto const normal = [&] { return int64(rng() % (uint64_t(1) << 31)) - (int64(1) << 30); };
		for (auto &c : h) {
			int64 p = normal(), q = normal();
			if (p == 0 && q == 0) p = 1;
			c = halfplane<int64>::at_most(p, q, 1 + int64(rng() % (uint64_t(1) << 61)));
		}
		return h;
	}

	// the outside of the circle in every direction: empty
	std::vector<halfplane<int64>> outside(std::size_t const n, std::mt19937_64 &rng) {
		auto h = tangents(n, rng);
		for (auto &c : h) c = {-c.p, -c.q, -c.r};
		return h;
	}

	// tangents already in the [0, 2pi) direction sweep
	std::vector<halfplane<int64>> presorted(std::size_t const n, std::mt19937_64 &rng) {
		auto h = tangents(n, rng);
		std::ranges::sort(h, [](halfplane<int64> const &a, halfplane<int64> const &b) {
			return gd::sweeps_before(gd::halfplane_constraint<int64>{a.p, a.q, a.r, 0},
									 gd::halfplane_constraint<int64>{b.p, b.q, b.r, 0});
		});
		return h;
	}

	// presorted tangents rotated by a third: one rotated sorted run, the
	// order a convex polygon's edges come in
	std::vector<halfplane<int64>> rotated_run(std::size_t const n, std::mt19937_64 &rng) {
		auto h = presorted(n, rng);
		std::ranges::rotate(h, h.begin() + std::ptrdiff_t(n / 3));
		return h;
	}

	std::vector<halfplane<double>> as_doubles(std::vector<halfplane<int64>> const &h) {
		std::vector<halfplane<double>> out;
		out.reserve(h.size());
		for (auto const &c : h) out.push_back({double(c.p), double(c.q), double(c.r)});
		return out;
	}

	// the hull of 1e6 points on the radius-5e8 circle centred at (cx, 0)
	std::vector<point<int64>> circle_hull(int64 const cx, std::mt19937_64 &rng) {
		std::vector<point<int64>> v(1'000'000);
		for (auto &p : v) {
			double const a = double(rng() % (1 << 30)) / double(1 << 30) * 6.283185307179586;
			p = {cx + int64(std::llround(1e9 * std::cos(a))) / 2, int64(std::llround(1e9 * std::sin(a))) / 2};
		}
		return atl::geometry::convex_hull(v);
	}

	// the edges of two hulls as left_of half-planes: each hull's edges are
	// one rotated sorted run of directions
	std::vector<halfplane<int64>> edges_of(std::vector<point<int64>> const &a, std::vector<point<int64>> const &b) {
		std::vector<halfplane<int64>> edges;
		for (auto const *hull : {&a, &b})
			for (std::size_t i = 0; i < hull->size(); ++i)
				edges.push_back(halfplane<int64>::left_of((*hull)[i], (*hull)[(i + 1) % hull->size()]));
		return edges;
	}

	std::size_t digest(atl::geometry::halfplane_region const &r) {
		return r.boundary.size() * 8 + std::size_t(r.kind);
	}

	/** @brief Time one complete intersection, checking kind, indices and corner tolerance outside timing. */
	template<atl::geometry::halfplane_range R, typename Compute, atl::geometry::halfplane_intersection_policy Policy = atl::geometry::exact>
		requires std::is_invocable_r_v<atl::geometry::halfplane_region, Compute &>
	void race(std::string const &label, R const &input, Compute compute, Policy const policy = {}) {
		auto const expected = atl::geometry::halfplane_intersection(input, policy);
		auto const got = std::invoke(compute);
		REQUIRE(got.kind == expected.kind);
		REQUIRE(got.boundary == expected.boundary);
		REQUIRE(got.vertices.size() == expected.vertices.size());
		for (std::size_t i = 0; i < got.vertices.size(); ++i) {
			double const scale = 1 + (std::max)(std::abs(expected.vertices[i].x), std::abs(expected.vertices[i].y));
			REQUIRE(std::hypot(got.vertices[i].x - expected.vertices[i].x, got.vertices[i].y - expected.vertices[i].y) <= 0x1p-48 * scale);
		}
		BENCHMARK(std::string(label)) { return digest(std::invoke(compute)); };
	}

	/** @brief The library integral direction key used to count ascending runs before timing. */
	atl::uint32 sweep_key(halfplane<int64> const &h) {
		return gd::direction_key(gd::halfplane_constraint<int64>{h.p, h.q, h.r, 0});
	}

	/** @brief Concatenate circular runs, cutting at the first wrap to expose exactly count ascending key runs. */
	std::vector<halfplane<int64>> rotated_key_runs(std::size_t const n, std::size_t const count, std::mt19937_64 &rng) {
		auto const sorted = presorted(n, rng);
		std::vector<std::vector<halfplane<int64>>> runs(count);
		for (std::size_t i = 0; i < n; ++i) runs[i % count].push_back(sorted[(i + n / 3) % n]);
		std::vector<halfplane<int64>> out;
		out.reserve(n);
		for (auto const &run : runs) out.insert(out.end(), run.begin(), run.end());
		for (std::size_t i = 1; i < n; ++i) {
			if (sweep_key(out[i]) < sweep_key(out[i - 1])) {
				std::ranges::rotate(out, out.begin() + std::ptrdiff_t(i));
				break;
			}
		}
		return out;
	}
}

TEST_CASE("halfplane intersection benchmarks", "[geometry]") {
	std::mt19937_64 rng(1);

	struct shape {
		std::string name;
		std::vector<halfplane<int64>> halfplanes;
		std::vector<halfplane<double>> doubles;
	};

	std::vector<shape> shapes;
	for (std::size_t const n : {std::size_t(200'000), std::size_t(1'000'000)}) {
		std::string const size = n == 200'000 ? " 2e5" : " 1e6";
		for (auto const &[name, make] : {std::pair{"tangents", &tangents}, std::pair{"around origin", &around_origin},
										 std::pair{"empty", &outside}, std::pair{"presorted", &presorted}}) {
			auto h = make(n, rng);
			auto d = as_doubles(h);
			shapes.push_back({name + size, std::move(h), std::move(d)});
		}
	}
	using filtered = gd::filtered_determinant;
	using exact = ex::detail::always_exact_determinant;
	using rounded = ex::detail::rounded_determinant;
	using ex::halfplane_sort;
	for (auto const &s : shapes) {
		race("library, " + s.name + " int64", s.halfplanes, [&] { return atl::geometry::halfplane_intersection(s.halfplanes); });
		race("library, filtered, " + s.name + " double", s.doubles, [&] { return atl::geometry::halfplane_intersection(s.doubles); });
		race("by exact expansion, " + s.name + " double", s.doubles, [&] { return ex::halfplane_intersection_by_composition<exact, gd::deque_pass, halfplane_sort::key>(s.doubles); });
		race("by product permanent, " + s.name + " double", s.doubles, [&] { return ex::halfplane_intersection_by_composition<ex::detail::product_filter_determinant, gd::deque_pass, halfplane_sort::key>(
																				 s.doubles); });
		race("by rounded determinant, " + s.name + " double", s.doubles, [&] { return ex::halfplane_intersection_by_composition<rounded, gd::deque_pass, halfplane_sort::key>(s.doubles); }, atl::geometry::fast{});
		race("by envelope split, " + s.name + " int64", s.halfplanes, [&] { return ex::halfplane_intersection_by_composition<filtered, ex::detail::envelope_pass, halfplane_sort::key>(
																				s.halfplanes); });
		if (s.halfplanes.size() != 1'000'000) continue;
		race("by envelope split, " + s.name + " double", s.doubles, [&] { return ex::halfplane_intersection_by_composition<filtered, ex::detail::envelope_pass, halfplane_sort::key>(
																			  s.doubles); });
		race("by Dekker split, " + s.name + " double", s.doubles, [&] { return ex::halfplane_intersection_by_composition<ex::detail::dekker_determinant, gd::deque_pass, halfplane_sort::key>(
																			s.doubles); });
		race("by every front, " + s.name + " int64", s.halfplanes, [&] { return ex::halfplane_intersection_by_composition<filtered, ex::detail::deque_pass_testing_every_front, halfplane_sort::key>(
																			 s.halfplanes); });
		race("by std deque, " + s.name + " int64", s.halfplanes, [&] { return ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::key, true>(s.halfplanes); });
		race("by comparator sort, " + s.name + " int64", s.halfplanes, [&] { return ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::comparator>(s.halfplanes); });
		race("by comparator sort, " + s.name + " double", s.doubles, [&] { return ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::comparator>(s.doubles); });
		race("by 32-bit comparison sort, " + s.name + " int64", s.halfplanes, [&] { return ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::key_std_sort>(s.halfplanes); });
		race("by 32-bit radix 16, " + s.name + " int64", s.halfplanes, [&] { return ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::key_radix_16>(s.halfplanes); });
		race("by 64-bit comparison sort, " + s.name + " int64", s.halfplanes, [&] { return ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::key_64_std_sort>(s.halfplanes); });
		race("by 64-bit radix 16, " + s.name + " int64", s.halfplanes, [&] { return ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::key_64_radix_16>(s.halfplanes); });
	}

	auto const hull_a = circle_hull(0, rng), hull_b = circle_hull(300'000'000, rng);
	auto const edges = edges_of(hull_a, hull_b);
	BENCHMARK("intersection_of_convex_polygons, pre-hulled 2 x 1e6 circle") {
		return atl::geometry::intersection_of_convex_polygons(hull_a, hull_b).size();
	};
	race("library, hull edges, pre-hulled 2 x 1e6 circle", edges, [&] { return atl::geometry::halfplane_intersection(edges); });
}

TEST_CASE("halfplane intersection benchmarks: corner cache and rotated runs", "[geometry]") {
	std::mt19937_64 rng(1);
	using filtered = gd::filtered_determinant;
	using ex::halfplane_sort;
	std::vector<std::pair<std::string, std::vector<halfplane<int64>>>> shapes;
	for (std::size_t const n : {std::size_t(200'000), std::size_t(1'000'000)}) {
		std::string const size = n == 200'000 ? " 2e5" : " 1e6";
		for (auto const &[name, make] :
			 {std::pair{"tangents", &tangents}, std::pair{"around origin", &around_origin}, std::pair{"empty", &outside},
			  std::pair{"presorted", &presorted}, std::pair{"rotated run", &rotated_run}})
			shapes.emplace_back(name + size, make(n, rng));
	}
	auto const hull_a = circle_hull(0, rng), hull_b = circle_hull(300'000'000, rng);
	shapes.emplace_back("pre-hulled 2 x 1e6 circle edges", edges_of(hull_a, hull_b));
	for (auto const &[name, h] : shapes) {
		auto const expected = atl::geometry::halfplane_intersection(h);
		auto const both = ex::halfplane_intersection_by_composition<filtered, ex::detail::corner_cached_pass,
																	halfplane_sort::rotated_runs>(h);
		REQUIRE((both.kind == expected.kind && both.boundary == expected.boundary));
		race("by key sort, " + name + " int64", h, [&] { return ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::key>(h); });
		race("by corner cache, " + name + " int64", h, [&] { return ex::halfplane_intersection_by_composition<filtered, ex::detail::corner_cached_pass, halfplane_sort::key>(h); });
		race("library, " + name + " int64", h, [&] { return atl::geometry::halfplane_intersection(h); });
		race("by corner cache and rotated runs, " + name + " int64", h, [&] { return ex::halfplane_intersection_by_composition<filtered, ex::detail::corner_cached_pass,
																															   halfplane_sort::rotated_runs>(h); });
	}
}

TEST_CASE("halfplane intersection benchmarks: rotated run dispatch sizes", "[geometry]") {
	std::mt19937_64 rng(31);
	using filtered = gd::filtered_determinant;
	using ex::halfplane_sort;
	for (std::size_t const n : {std::size_t(1000), std::size_t(10'000), std::size_t(50'000), std::size_t(500'000)}) {
		for (std::size_t const runs : {std::size_t(1), std::size_t(8), std::size_t(9)}) {
			auto const h = runs == 1 ? rotated_run(n, rng) : rotated_key_runs(n, runs, rng);
			std::size_t ascending_runs = 1;
			for (std::size_t i = 1; i < h.size(); ++i)
				if (sweep_key(h[i]) < sweep_key(h[i - 1])) ++ascending_runs;
			REQUIRE(ascending_runs == (runs == 1 ? 2 : runs));
			if (runs == 1) REQUIRE(sweep_key(h.back()) <= sweep_key(h.front()));
			auto const expected = atl::geometry::halfplane_intersection(h);
			{
				auto const key_sorted = ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::key>(h);
				REQUIRE(key_sorted.kind == expected.kind);
				REQUIRE(key_sorted.boundary == expected.boundary);
			}
			if (runs <= 8) {
				auto const rotated = ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::rotated_runs>(h);
				REQUIRE(rotated.kind == expected.kind);
				REQUIRE(rotated.boundary == expected.boundary);
			}
			std::string const tag = "n=" + std::to_string(n) + " "
								  + (runs == 1 ? "rotated_run" : std::to_string(runs) + " rotated runs")
								  + " (ascending key runs=" + std::to_string(ascending_runs) + ") int64";
			race("library, " + tag, h, [&] { return atl::geometry::halfplane_intersection(h); });
			race("by key sort, " + tag, h, [&] { return ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::key>(h); });
			if (runs <= 8) {
				race("by rotated runs, " + tag, h, [&] { return ex::halfplane_intersection_by_composition<filtered, gd::deque_pass, halfplane_sort::rotated_runs>(h); });
			}
		}
	}
}

TEST_CASE("convex polygon operations: coordinates near 2^60", "[geometry]") {
	std::mt19937_64 rng(9281);
	int64 const k = int64(1) << 60;
	std::vector<point<int64>> a(100000), b(100000);
	for (auto *cloud : {&a, &b})
		for (auto &v : *cloud) v = {int64(rng() % std::uint64_t(2 * k)) - k,
									int64(rng() % std::uint64_t(2 * k)) - k};
	BENCHMARK("intersection_of_convex_polygons, coordinates near 2^60") {
		return atl::geometry::intersection_of_convex_polygons(a, b).size();
	};
	BENCHMARK("union_of_convex_polygons, coordinates near 2^60") {
		return atl::geometry::union_of_convex_polygons(a, b).size();
	};
	BENCHMARK("closest_points_between_convex_hulls, coordinates near 2^60") {
		return atl::geometry::closest_points_between_convex_hulls(a, b)[0].x;
	};
}
