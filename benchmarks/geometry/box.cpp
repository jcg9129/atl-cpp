/**
 * @file
 * @brief atl::geometry::union_area (an x-sweep over a Klee segment tree updated
 *        iteratively bottom-up, with a precomputed length array) raced against
 *        atl::experimental::union_area_by_top_down_tree, which differs only in
 *        the tree: a recursive top-down update on one interleaved node array
 *        that reads node lengths from the y values.
 *
 * Each row is one full `union_area` call (throughput of the whole sweep) on
 * 2e5 or 1e6 random boxes in a 1e9 square: "small" boxes have sides below 1e7
 * and touch few tree nodes per update, "nested" boxes span random intervals of
 * the whole square and stretch every update across the tree; the int32 rows
 * narrow the same boxes.
 *
 * Result (Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, asserts off,
 * 2026-10-15; ms, `-O2` / `-march=native`): the library wins every shape, e.g.
 * 2001 / 1995 against 2587 / 2561 on 1e6 small int64 boxes and 1603 / 1577
 * against 2206 / 2205 on 1e6 small int32 boxes; the contender takes 27-40%
 * longer. The full table is on the guide page (docs/geometry-primitives.md).
 */
#include <atl/experimental/geometry/box_variants.hpp>
#include <atl/geometry/box.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using atl::geometry::box;

namespace {
	/** @brief Times @p contender on @p boxes under @p label; its result's low bit keeps the call alive. */
	template<atl::geometry::box_range Boxes, typename Contender>
		requires std::invocable<Contender const &, Boxes const &>
	void measure(std::string const &label, Boxes const &boxes, Contender const &contender) {
		BENCHMARK(std::string(label)) { return atl::int64(contender(boxes) & 1); };
	}
}

TEST_CASE("union_area benchmarks", "[geometry]") {
	std::mt19937_64 rng(1);
	auto const small_boxes = [&](std::size_t const n, std::int64_t const span) {
		std::vector<box<std::int64_t>> v(n);
		for (auto &b : v) {
			std::int64_t const x = std::int64_t(rng() % std::uint64_t(span)), y = std::int64_t(rng() % std::uint64_t(span));
			b = {{x, y}, {x + 1 + std::int64_t(rng() % std::uint64_t(span / 100)), y + 1 + std::int64_t(rng() % std::uint64_t(span / 100))}};
		}
		return v;
	};
	auto const nested_boxes = [&](std::size_t const n, std::int64_t const span) {
		std::vector<box<std::int64_t>> v(n);
		for (auto &b : v) {
			std::int64_t x1 = std::int64_t(rng() % std::uint64_t(span)), x2 = std::int64_t(rng() % std::uint64_t(span));
			std::int64_t y1 = std::int64_t(rng() % std::uint64_t(span)), y2 = std::int64_t(rng() % std::uint64_t(span));
			b = {{(std::min)(x1, x2), (std::min)(y1, y2)}, {(std::max)(x1, x2) + 1, (std::max)(y1, y2) + 1}};
		}
		return v;
	};
	auto const narrowed = [](std::vector<box<std::int64_t>> const &v) {
		std::vector<box<std::int32_t>> out;
		for (auto const &b : v)
			out.push_back({{std::int32_t(b.lo.x), std::int32_t(b.lo.y)}, {std::int32_t(b.hi.x), std::int32_t(b.hi.y)}});
		return out;
	};
	auto const library = [](atl::geometry::box_range auto const &boxes) { return atl::geometry::union_area(boxes); };
	auto const by_top_down_tree = [](atl::geometry::box_range auto const &boxes) {
		return atl::experimental::union_area_by_top_down_tree(boxes);
	};
	auto const race = [&](std::string const &shape, atl::geometry::box_range auto const &boxes) {
		REQUIRE(library(boxes) == by_top_down_tree(boxes));
		measure("library, " + shape, boxes, library);
		measure("by top-down tree, " + shape, boxes, by_top_down_tree);
	};
	auto const small2 = small_boxes(200'000, 1'000'000'000), small10 = small_boxes(1'000'000, 1'000'000'000);
	auto const nested2 = nested_boxes(200'000, 1'000'000'000), nested10 = nested_boxes(1'000'000, 1'000'000'000);
	race("int64 small boxes 2e5", small2);
	race("int64 small boxes 1e6", small10);
	race("int64 nested boxes 2e5", nested2);
	race("int64 nested boxes 1e6", nested10);
	race("int32 small boxes 1e6", narrowed(small10));
	race("int32 nested boxes 1e6", narrowed(nested10));
}
