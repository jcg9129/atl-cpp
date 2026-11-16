/**
 * @file
 * Race of `atl::counting_sort` (stable, in place, O(n + span)) against a
 * contender that realises the same permutation differently, and against the
 * comparison sorts.
 *
 * `counting_sort` and `counting_sort_by_cycle_walk` compute the same stable
 * permutation from the same counts and differ only in how they apply it:
 * - `atl::counting_sort` (the library) gathers the elements in order into a
 *   buffer and moves them back;
 * - `counting_sort_by_cycle_walk` inverts the permutation to a scatter and walks
 *   its cycles with swaps, which land at random positions, so it needs no
 *   element buffer.
 *
 * The baselines are `std::ranges::sort` and `std::ranges::stable_sort`. Each
 * row sorts a fresh copy of the same 1e6 elements: plain `int32` keys and 16-byte
 * records sorted by an `int32` field, with key spans 1000 and 1000000. The
 * "copy only" rows copy the input and do not sort; they are the floor shared by
 * every other row. The loop sorts once per sample, so it measures throughput.
 *
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`,
 * 2026-10-15): the library is fastest on all four shapes with both flag sets.
 * `by_cycle_walk` is 1.5 to 5.2 times slower, `std::ranges::sort` 1.8 to 6.0
 * times and `std::ranges::stable_sort` 2.2 to 8.6 times. Numbers:
 * docs/counting-sort.md.
 */
#include <atl/types/alias.hpp>
#include <atl/utils/counting_sort.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace atl;

namespace {
	/** @brief A 16-byte record sorted by one field, the projection use case. */
	struct event {
		int32 key;
		int32 tag;
		int64 payload;
		bool operator==(event const &) const = default;
	};

	/**
	 * @brief Same result as `counting_sort`; applies the permutation in place by
	 *        inverting it to a scatter and walking its cycles with swaps.
	 */
	template<typename R, typename Proj = std::identity>
		requires counting_sortable<R &, Proj>
	void counting_sort_by_cycle_walk(R &r, Proj proj = {}) {
		auto &s = detail::counting_sort_buffers();
		std::size_t const n = detail::counting_permutation(r, proj, s);
		static thread_local std::vector<uint32> scatter; // reused across calls, like the library's scratch
		scatter.resize(n);
		for (std::size_t k = 0; k < n; ++k) scatter[s.perm[k]] = uint32(k);
		for (std::size_t i = 0; i < n; ++i)
			while (scatter[i] != uint32(i)) {
				uint32 const j = scatter[i];
				std::ranges::swap(r[i], r[j]);
				std::swap(scatter[i], scatter[j]);
			}
	}

	template<typename R, typename Proj = std::identity>
	concept cycle_walk_sortable = requires(R &r, Proj proj) { counting_sort_by_cycle_walk(r, proj); };

	struct event_key_ref {
		int32 &operator()(event &e) const { return e.key; }
	};

	static_assert(cycle_walk_sortable<std::vector<int32>>);
	static_assert(cycle_walk_sortable<std::vector<event>, decltype(&event::key)>);
	static_assert(cycle_walk_sortable<std::vector<event>, event_key_ref>);
	static_assert(!cycle_walk_sortable<std::vector<double>>);	   // key not integral
	static_assert(!cycle_walk_sortable<std::vector<event>>);	   // no integral key without a projection
	static_assert(!cycle_walk_sortable<std::vector<int32> const>); // cannot permute in place
}

TEST_CASE("counting_sort vs comparison sorts", "[counting_sort]") {
	std::mt19937 rng(7);
	for (int32 const span : {1000, 1000000}) {
		std::string const shape = "n=1e6 span=" + std::to_string(span);
		std::vector<int32> keys(1000000);
		for (auto &k : keys) k = int32(rng() % uint32(span)) - span / 2;
		{
			auto want = keys;
			std::ranges::stable_sort(want);
			auto got = keys;
			counting_sort(got);
			REQUIRE(got == want);
			got = keys;
			counting_sort_by_cycle_walk(got);
			REQUIRE(got == want);
		}
		BENCHMARK("int " + shape + " | copy only") {
			auto v = keys;
			return v[1];
		};
		BENCHMARK("int " + shape + " | library") {
			auto v = keys;
			counting_sort(v);
			return v[1];
		};
		BENCHMARK("int " + shape + " | by cycle walk") {
			auto v = keys;
			counting_sort_by_cycle_walk(v);
			return v[1];
		};
		BENCHMARK("int " + shape + " | std::ranges::sort") {
			auto v = keys;
			std::ranges::sort(v);
			return v[1];
		};
		BENCHMARK("int " + shape + " | std::ranges::stable_sort") {
			auto v = keys;
			std::ranges::stable_sort(v);
			return v[1];
		};

		std::vector<event> events(keys.size());
		for (std::size_t i = 0; i < keys.size(); ++i) events[i] = {keys[i], int32(i), int64(rng())};
		{
			auto want = events;
			std::ranges::stable_sort(want, {}, &event::key);
			auto got = events;
			counting_sort(got, &event::key);
			REQUIRE(got == want);
			got = events;
			counting_sort_by_cycle_walk(got, &event::key);
			REQUIRE(got == want);
		}
		BENCHMARK("record " + shape + " | copy only") {
			auto v = events;
			return v[1].tag;
		};
		BENCHMARK("record " + shape + " | library") {
			auto v = events;
			counting_sort(v, &event::key);
			return v[1].tag;
		};
		BENCHMARK("record " + shape + " | by cycle walk") {
			auto v = events;
			counting_sort_by_cycle_walk(v, &event::key);
			return v[1].tag;
		};
		BENCHMARK("record " + shape + " | std::ranges::sort") {
			auto v = events;
			std::ranges::sort(v, {}, &event::key);
			return v[1].tag;
		};
		BENCHMARK("record " + shape + " | std::ranges::stable_sort") {
			auto v = events;
			std::ranges::stable_sort(v, {}, &event::key);
			return v[1].tag;
		};
	}
}
