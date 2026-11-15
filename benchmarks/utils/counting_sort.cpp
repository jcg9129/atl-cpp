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
	 * @brief The losing counting_sort contender: the same gather permutation,
	 *        realised in place by inverting it to a scatter and walking its
	 *        cycles with swaps, which land at random positions (the gather measured
	 *        1.5x to 5x faster, docs/counting-sort.md).
	 */
	template<typename R, typename Proj = std::identity>
		requires counting_sortable<R &, Proj>
	void counting_sort_cycles(R &r, Proj proj = {}) {
		auto &s = detail::counting_sort_buffers();
		std::size_t const n = detail::counting_permutation(r, proj, s);
		static thread_local std::vector<uint32> scatter; // reused across calls, like the shipped scratch
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
	concept cycle_sortable = requires(R &r, Proj proj) { counting_sort_cycles(r, proj); };

	struct event_key_ref {
		int32 &operator()(event &e) const { return e.key; }
	};

	static_assert(cycle_sortable<std::vector<int32>>);
	static_assert(cycle_sortable<std::vector<event>, decltype(&event::key)>);
	static_assert(cycle_sortable<std::vector<event>, event_key_ref>);
	static_assert(!cycle_sortable<std::vector<double>>);	  // key not integral
	static_assert(!cycle_sortable<std::vector<event>>);		  // no integral key without a projection
	static_assert(!cycle_sortable<std::vector<int32> const>); // cannot permute in place
}

// counting_sort against the comparison sorts on the shapes docs/counting-sort.md
// claims (n = 1e6, keys in a small band), plain ints and projected records.
// Every row copies the input first; the "copy only" row is that floor.
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
			counting_sort_cycles(got);
			REQUIRE(got == want);
		}
		BENCHMARK("int " + shape + " | copy only") {
			auto v = keys;
			return v[1];
		};
		BENCHMARK("int " + shape + " | counting_sort") {
			auto v = keys;
			counting_sort(v);
			return v[1];
		};
		BENCHMARK("int " + shape + " | cycle walk (before)") {
			auto v = keys;
			counting_sort_cycles(v);
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
			counting_sort_cycles(got, &event::key);
			REQUIRE(got == want);
		}
		BENCHMARK("record " + shape + " | copy only") {
			auto v = events;
			return v[1].tag;
		};
		BENCHMARK("record " + shape + " | counting_sort") {
			auto v = events;
			counting_sort(v, &event::key);
			return v[1].tag;
		};
		BENCHMARK("record " + shape + " | cycle walk (before)") {
			auto v = events;
			counting_sort_cycles(v, &event::key);
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
