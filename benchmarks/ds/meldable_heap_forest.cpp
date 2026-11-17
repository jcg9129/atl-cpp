/**
 * @file
 * @brief Races two-pass pairing (library, with optional predecessor handles), rank-leftist,
 * skew and binary heaps on bottom-up path/star/caterpillar/random tree melds at
 * 2e5 and 1e6 vertices, lazy-shift slope patterns and 1e6-step push/pop/meld mixes.
 * Measures complete-script throughput including allocation. Xeon Platinum 8280,
 * GCC 15.2, RelWithDebInfo, assertions off, 2026-10-16, -O2 / -O2 -march=native
 * (also -g -DNDEBUG -std=c++23): random tree 2e5 library 12.7 / 13.1 ms,
 * rank-leftist 19.1 / 19.4, skew 13.7 / 13.4, binary 45.8 / 44.5.
 * See docs/meldable-heap-forest.md for the full table.
 */

#include <atl/ds/meldable_heap_forest.hpp>
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
namespace ex = atl::experimental;
namespace mb = meldable_heap_bench;

namespace {
	template<typename Compare, typename Shift>
	using heap_by_rank_leftist = ex::meldable_heap_forest_by_rank_leftist<int64, Compare, Shift>;
	template<typename Compare, typename Shift>
	using heap_by_skew = ex::meldable_heap_forest_by_skew<int64, Compare, Shift, atl::shift_by_addition, atl::compose_by_addition, true>;
	template<typename Compare, typename Shift>
	using heap_by_pairing = atl::meldable_heap_forest<int64, Compare, Shift>; // the library pairing heap
	template<typename Compare, typename Shift>
	using heap_by_binary = ex::meldable_heap_forest_by_binary<int64, Compare, Shift>;

	/** @brief One row per contender: bottom-up melds over `parent`. */
	void tree_rows(std::string const &label, std::vector<uint32> const &parent, std::vector<int64> const &values) {
		auto const expected = mb::tree_melds<heap_by_pairing<std::less<>, void>>(parent, values, false);
		REQUIRE(mb::tree_melds<heap_by_rank_leftist<std::less<>, void>>(parent, values, false) == expected);
		REQUIRE(mb::tree_melds<heap_by_skew<std::less<>, void>>(parent, values, false) == expected);
		REQUIRE(mb::tree_melds<heap_by_binary<std::less<>, void>>(parent, values, false) == expected);
		REQUIRE(mb::tree_melds<atl::addressable_meldable_heap_forest<int64>>(parent, values, false) == expected);
		measure(label + " / by rank leftist", [&] { return mb::tree_melds<heap_by_rank_leftist<std::less<>, void>>(parent, values, false); });
		measure(label + " / by skew", [&] { return mb::tree_melds<heap_by_skew<std::less<>, void>>(parent, values, false); });
		measure(label + " / library", [&] { return mb::tree_melds<heap_by_pairing<std::less<>, void>>(parent, values, false); });
		measure(label + " / library with handles", [&] {
			return mb::tree_melds<atl::addressable_meldable_heap_forest<int64>>(parent, values, false);
		});
		measure(label + " / by binary", [&] { return mb::tree_melds<heap_by_binary<std::less<>, void>>(parent, values, false); });
	}

	/** @brief One row per contender: the slope-trick pattern (min-heaps, lazy add) over `parent`. */
	void slope_rows(std::string const &label, std::vector<uint32> const &parent, std::vector<int64> const &values) {
		auto const expected = mb::slope_trick_pattern<heap_by_pairing<std::greater<>, int64>>(parent, values);
		REQUIRE(mb::slope_trick_pattern<heap_by_rank_leftist<std::greater<>, int64>>(parent, values) == expected);
		REQUIRE(mb::slope_trick_pattern<heap_by_skew<std::greater<>, int64>>(parent, values) == expected);
		REQUIRE(mb::slope_trick_pattern<heap_by_binary<std::greater<>, int64>>(parent, values) == expected);
		measure(label + " / by rank leftist", [&] { return mb::slope_trick_pattern<heap_by_rank_leftist<std::greater<>, int64>>(parent, values); });
		measure(label + " / by skew", [&] { return mb::slope_trick_pattern<heap_by_skew<std::greater<>, int64>>(parent, values); });
		measure(label + " / library", [&] { return mb::slope_trick_pattern<heap_by_pairing<std::greater<>, int64>>(parent, values); });
		measure(label + " / by binary", [&] { return mb::slope_trick_pattern<heap_by_binary<std::greater<>, int64>>(parent, values); });
	}

	/** @brief One row per contender: a replayed random mix. */
	void mix_rows(std::string const &label, std::vector<mb::operation> const &ops, uint32 heap_count) {
		auto const expected = mb::replay<heap_by_pairing<std::less<>, void>>(ops, heap_count);
		REQUIRE(mb::replay<heap_by_rank_leftist<std::less<>, void>>(ops, heap_count) == expected);
		REQUIRE(mb::replay<heap_by_skew<std::less<>, void>>(ops, heap_count) == expected);
		REQUIRE(mb::replay<heap_by_binary<std::less<>, void>>(ops, heap_count) == expected);
		REQUIRE(mb::replay<atl::addressable_meldable_heap_forest<int64>>(ops, heap_count) == expected);
		measure(label + " / by rank leftist", [&] { return mb::replay<heap_by_rank_leftist<std::less<>, void>>(ops, heap_count); });
		measure(label + " / by skew", [&] { return mb::replay<heap_by_skew<std::less<>, void>>(ops, heap_count); });
		measure(label + " / library", [&] { return mb::replay<heap_by_pairing<std::less<>, void>>(ops, heap_count); });
		measure(label + " / library with handles", [&] {
			return mb::replay<atl::addressable_meldable_heap_forest<int64>>(ops, heap_count);
		});
		measure(label + " / by binary", [&] { return mb::replay<heap_by_binary<std::less<>, void>>(ops, heap_count); });
	}
}

TEST_CASE("meldable_heap_forest", "[meldable_heap_forest]") {
	std::mt19937_64 rng(1);
	for (uint32 const n : {200'000u, 1'000'000u}) {
		auto const values = mb::random_values(3 * std::size_t(n), rng);
		std::string const size = n == 200'000u ? "2e5" : "1e6";
		for (auto const shape :
			 {mb::tree_shape::path, mb::tree_shape::star, mb::tree_shape::caterpillar, mb::tree_shape::random}) {
			auto const parent = mb::tree_parents(shape, n, rng);
			tree_rows("tree " + mb::name(shape) + " " + size, parent, values);
		}
		for (auto const shape : {mb::tree_shape::path, mb::tree_shape::caterpillar, mb::tree_shape::random}) {
			auto const parent = mb::tree_parents(shape, n, rng);
			slope_rows("slope " + mb::name(shape) + " " + size, parent, values);
		}
	}
	mix_rows("mix 1 heap, push 50 / pop 50", mb::random_mix(1'000'000, 1, 50, 50, rng), 1);
	mix_rows("mix 1000 heaps, push 50 / pop 30 / meld 20", mb::random_mix(1'000'000, 1000, 50, 30, rng), 1000);
	mix_rows("mix 1e5 heaps, push 40 / pop 20 / meld 40", mb::random_mix(1'000'000, 100'000, 40, 20, rng), 100'000);
}
