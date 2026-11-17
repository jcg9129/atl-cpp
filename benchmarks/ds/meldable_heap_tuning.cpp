/**
 * @file
 * @brief Races branch/select root ordering, free-list reuse/append-only allocation,
 * node/split-array layout, lazy-tag identity detection, leftist recursion/rank width,
 * and binary push/rebuild melding. Complete-script throughput: 2e5 random tree
 * vertices, 1e6 mixed operations over 1000 heaps, and lazy-shift slope patterns.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-15,
 * -O2 / -O2 -march=native (also -g -DNDEBUG -std=c++23): pairing branch/free-list
 * random tree 12.6 / 13.2 ms versus select/free-list 12.9 / 13.8.
 * See docs/meldable-heap-forest.md for the full table.
 */

#include <atl/experimental/ds/meldable_heap_variants.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "meldable_heap_workloads.hpp"

namespace {
	/** @brief Time a complete workload through a direct, constrained callable. */
	template<typename Run>
		requires std::invocable<Run &>
	void measure(std::string const &label, Run run) {
		BENCHMARK(std::string(label)) { return run(); };
	}
}

using atl::int64;
using atl::uint32;
using atl::uint8;
namespace ex = atl::experimental;
namespace mb = meldable_heap_bench;
using add = atl::shift_by_addition;
using compose = atl::compose_by_addition;

namespace {
	/** @brief The two tuning rows of one contender. */
	template<mb::int64_forest Forest>
	void rows(std::string const &label, std::vector<uint32> const &parent, std::vector<int64> const &values,
			  std::vector<mb::operation> const &ops, uint32 heap_count) {
		REQUIRE(mb::tree_melds<Forest>(parent, values, false) == mb::tree_melds<atl::meldable_heap_forest<int64>>(parent, values, false));
		REQUIRE(mb::replay<Forest>(ops, heap_count) == mb::replay<atl::meldable_heap_forest<int64>>(ops, heap_count));
		measure("tree " + label, [&] { return mb::tree_melds<Forest>(parent, values, false); });
		measure("mix " + label, [&] { return mb::replay<Forest>(ops, heap_count); });
	}

	/** @brief The slope-trick row of one contender with a lazy add (min-heaps). */
	template<mb::shifting_int64_forest Forest>
	void slope_row(std::string const &label, std::vector<uint32> const &parent, std::vector<int64> const &values) {
		REQUIRE(mb::slope_trick_pattern<Forest>(parent, values) == mb::slope_trick_pattern<atl::meldable_heap_forest<int64, std::greater<>, int64>>(parent, values));
		measure("slope " + label, [&] { return mb::slope_trick_pattern<Forest>(parent, values); });
	}
}

TEST_CASE("meldable_heap_tuning", "[meldable_heap_tuning]") {
	std::mt19937_64 rng(1);
	uint32 const n = 200'000;
	auto const parent = mb::tree_parents(mb::tree_shape::random, n, rng);
	auto const values = mb::random_values(3 * std::size_t(n), rng);
	uint32 const heap_count = 1000;
	auto const ops = mb::random_mix(1'000'000, heap_count, 50, 30, rng);

	rows<ex::meldable_heap_forest_by_rank_leftist<int64, std::less<>, void, add, compose, uint8, true>>("leftist / recursive, uint8 rank", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_rank_leftist<int64, std::less<>, void, add, compose, uint8, false>>("leftist / iterative, uint8 rank", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_rank_leftist<int64, std::less<>, void, add, compose, uint32, false>>("leftist / iterative, uint32 rank", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_skew<int64, std::less<>, void, add, compose, false>>("skew / branch", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_skew<int64, std::less<>, void, add, compose, true>>("skew / select", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_pairing<int64, std::less<>, void, add, compose, false, true>>("pairing / branch, free list", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_pairing<int64, std::less<>, void, add, compose, true, true>>("pairing / select, free list", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_pairing<int64, std::less<>, void, add, compose, false, false>>("pairing / branch, no free list", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_split_pairing<int64, std::less<>, void>>("pairing / split arrays", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_pairing<int64, std::less<>, int64>>("pairing / lazy-add tags, unused", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_pairing<int64, std::less<>, int64, add, compose, false, true, false>>(
		"pairing / lazy-add tags, unused, no identity skip", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_rank_leftist<int64, std::less<>, int64>>("leftist / lazy-add tags, unused", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_binary<int64, std::less<>, void, add, compose, false>>("binary / push each", parent, values, ops, heap_count);
	rows<ex::meldable_heap_forest_by_binary<int64, std::less<>, void, add, compose, true>>("binary / bulk rebuild", parent, values, ops, heap_count);

	slope_row<ex::meldable_heap_forest_by_pairing<int64, std::greater<>, int64>>("pairing", parent, values);
	slope_row<ex::meldable_heap_forest_by_pairing<int64, std::greater<>, int64, add, compose, false, true, false>>(
		"pairing / no identity skip", parent, values);
	slope_row<ex::meldable_heap_forest_by_skew<int64, std::greater<>, int64, add, compose, true>>("skew / select", parent, values);
	slope_row<ex::meldable_heap_forest_by_rank_leftist<int64, std::greater<>, int64>>("leftist", parent, values);
	slope_row<ex::meldable_heap_forest_by_binary<int64, std::greater<>, int64>>("binary", parent, values);

	measure("tree pairing / branch, free list, reserved", [&] {
		return mb::tree_melds<ex::meldable_heap_forest_by_pairing<int64>>(parent, values, true);
	});
	measure("tree leftist / iterative, uint8 rank, reserved", [&] {
		return mb::tree_melds<ex::meldable_heap_forest_by_rank_leftist<int64>>(parent, values, true);
	});
}
