/**
 * @file
 * @brief Full-call arrangement throughput: library event sweep versus direct
 * pair intersections with shared half-edge assembly, on 5000 sparse segments
 * and 700 dense chords. Intel Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo,
 * assertions off, 2026-10-15, -O2 / -O2 -march=native: sparse library
 * 14.7 / 13.8 ms, by all pairs 1920 / 1705; dense 295 / 303 versus 303 / 292.
 */
#include <atl/experimental/geometry/arrangement_all_pairs.hpp>
#include <atl/geometry/arrangement.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using atl::int64;
using atl::experimental::arrangement_of_by_all_pairs;
using atl::geometry::arrangement_of;
using sg = atl::geometry::segment<int64>;

TEST_CASE("arrangement benchmarks", "[geometry]") {
	std::mt19937_64 rng(1);

	// sparse: many short segments, few crossings (k << n)
	std::vector<sg> sparse;
	for (int k = 0; k < 5000; ++k) {
		int64 const x = int64(rng() % 1'000'000), y = int64(rng() % 1'000'000);
		sparse.push_back({{x, y},
						  {x + 1 + int64(rng() % 300), y + 1 + int64(rng() % 300)}});
	}

	// dense: long random chords of a box — k grows as n^2
	std::vector<sg> dense;
	for (int k = 0; k < 700; ++k) {
		auto const c = [&] { return int64(rng() % 1'000'000); };
		dense.push_back({{c(), c()}, {c(), c()}});
	}

	auto const run = []<typename Build>(std::string const &label, std::vector<sg> const &input, Build build)
		requires std::is_invocable_r_v<atl::geometry::arrangement, Build &, std::vector<sg> const &>
	{
		BENCHMARK(label.c_str()) { return build(input).faces.size(); };
	};
	for (auto const *input : {&sparse, &dense}) {
		auto const library = arrangement_of(*input);
		auto const contender = arrangement_of_by_all_pairs(*input);
		REQUIRE(library.origin == contender.origin);
		REQUIRE(library.next == contender.next);
		REQUIRE(library.face == contender.face);
		REQUIRE(library.supporting == contender.supporting);
		REQUIRE(library.vertices.size() == contender.vertices.size());
		REQUIRE(library.faces.size() == contender.faces.size());
		for (std::size_t i = 0; i < library.vertices.size(); ++i)
			REQUIRE(atl::geometry::approx{1e-8}.equal(library.vertices[i], contender.vertices[i]));
		for (std::size_t i = 0; i < library.faces.size(); ++i) {
			REQUIRE(library.faces[i].outer == contender.faces[i].outer);
			REQUIRE(library.faces[i].holes == contender.faces[i].holes);
			REQUIRE(library.faces[i].isolated == contender.faces[i].isolated);
			REQUIRE(std::abs(library.faces[i].area - contender.faces[i].area) <= 1e-8 * (1 + std::abs(library.faces[i].area)));
		}
	}
	run("library -- sparse 5000 segments", sparse, [](auto const &input)
			requires atl::geometry::segment_range<decltype(input)>
		{ return arrangement_of(input); });
	run("by all pairs -- sparse 5000 segments", sparse, [](auto const &input)
			requires atl::geometry::segment_range<decltype(input)>
		{ return arrangement_of_by_all_pairs(input); });
	run("library -- dense 700 chords", dense, [](auto const &input)
			requires atl::geometry::segment_range<decltype(input)>
		{ return arrangement_of(input); });
	run("by all pairs -- dense 700 chords", dense, [](auto const &input)
			requires atl::geometry::segment_range<decltype(input)>
		{ return arrangement_of_by_all_pairs(input); });
}
