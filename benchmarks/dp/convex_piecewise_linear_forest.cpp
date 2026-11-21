/**
 * @file
 * @brief Breakpoint-pool race: the pool of atl::convex_piecewise_linear_forest
 * (`detail::breakpoint_pool`, which empties a discarded heap with
 * `meldable_heap_forest::release`) against two contenders that empty it by
 * repeated pops: by repeated pops (the library's heaps) and by tunable
 * pairing pops (the experimental `meldable_heap_forest_by_pairing` at its
 * defaults).
 *
 * Throughput of whole runs: `clear` or `take_all` (the left heap pops to emit
 * the sorted breakpoints, the right one is discarded) on 1 set of 1e6, 1000
 * sets of 1000 and 1e5 sets of 10 unit breakpoints at positions uniform in
 * [0, 1e9), on 1000 sets of 1000 cleared and refilled three times, and on the
 * root of a random tree of 2e5 sets of 3 melded bottom-up. Contenders agree on
 * every result before timing.
 *
 * Result, Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-11-01, plain -O2 / -march=native: release wins every shape; clear of 1
 * set of 1e6 takes 74.1 / 75.3 ms against 1357 / 1348 by repeated pops and
 * 1332 / 1340 by tunable pairing pops, clear and refill 632 / 631 against
 * 1724 / 1693 and 1666 / 1683. Full table: docs/convex-piecewise-linear-forest.md.
 */
#include <atl/dp/convex_piecewise_linear_forest.hpp>
#include <atl/experimental/dp/convex_piecewise_linear_forest_variants.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

using atl::int64;
using atl::uint64;

namespace {
	using library_pool = atl::detail::breakpoint_pool<int64, int64>;
	using by_repeated_pops = atl::experimental::breakpoint_pool_by_repeated_pops<int64, int64>;
	using by_tunable_pairing_pops = atl::experimental::breakpoint_pool_by_tunable_pairing_pops<int64, int64>;

	/** @brief The operations the workloads drive on every pool. */
	template<typename Pool>
	concept discardable_pool = std::constructible_from<Pool, std::size_t>
							&& requires(Pool pool, std::size_t set, int64 x, std::vector<atl::weighted_breakpoint<int64, int64>> &out) {
								   pool.reserve(set);
								   pool.push(set, x, x);
								   pool.meld(set, set);
								   pool.clear(set);
								   pool.take_all(set, out);
								   { pool.set_count() } -> std::convertible_to<std::size_t>;
							   };

	/** @brief A workload run on one pool type: `run.template operator()<Pool>()` gives its checksum. */
	template<typename Run, typename Pool>
	concept pool_workload = discardable_pool<Pool> && requires(Run const &run) {
		{ run.template operator()<Pool>() } -> std::convertible_to<uint64>;
	};

	/** @brief `sets` sets of `per_set` random breakpoints each, pushed round-robin, then `take_all` or `clear` on every set, `rounds` times over. */
	template<discardable_pool Pool>
	uint64 discard_every_set(std::size_t sets, std::size_t per_set, std::size_t rounds, bool is_taking, std::vector<int64> const &positions) {
		Pool pool(sets);
		pool.reserve(sets * per_set);
		uint64 acc = 0;
		std::vector<atl::weighted_breakpoint<int64, int64>> out;
		std::size_t next = 0;
		for (std::size_t round = 0; round < rounds; ++round) {
			for (std::size_t i = 0; i < per_set; ++i)
				for (std::size_t set = 0; set < sets; ++set) pool.push(set, positions[next++ % positions.size()], 1);
			for (std::size_t set = 0; set < sets; ++set) {
				if (is_taking) {
					out.clear();
					pool.take_all(set, out);
					acc += out.size() + uint64(out.empty() ? 0 : out.back().position);
				} else
					pool.clear(set);
			}
		}
		return acc + pool.set_count();
	}

	/** @brief A tree DP shape: `n` sets each with 3 breakpoints, melded bottom-up along random parents, then the root taken or cleared. */
	template<discardable_pool Pool>
	uint64 discard_melded_tree(std::size_t n, bool is_taking, std::vector<int64> const &positions, std::vector<std::size_t> const &parent) {
		Pool pool(n);
		pool.reserve(3 * n);
		for (std::size_t v = 0; v < n; ++v)
			for (std::size_t i = 0; i < 3; ++i) pool.push(v, positions[(3 * v + i) % positions.size()], 1);
		for (std::size_t v = n; v-- > 1;) pool.meld(parent[v], v);
		std::vector<atl::weighted_breakpoint<int64, int64>> out;
		if (is_taking)
			pool.take_all(0, out);
		else
			pool.clear(0);
		return out.size();
	}

	/** @brief The library pool and both contenders on one workload, checked to agree first. */
	template<typename Run>
		requires pool_workload<Run, library_pool> && pool_workload<Run, by_repeated_pops>
			  && pool_workload<Run, by_tunable_pairing_pops>
	void race(std::string const &shape, Run const &run) {
		uint64 const expected = run.template operator()<library_pool>();
		REQUIRE(run.template operator()<by_repeated_pops>() == expected);
		REQUIRE(run.template operator()<by_tunable_pairing_pops>() == expected);
		BENCHMARK(shape + " -- library (release)") { return run.template operator()<library_pool>(); };
		BENCHMARK(shape + " -- by repeated pops") { return run.template operator()<by_repeated_pops>(); };
		BENCHMARK(shape + " -- by tunable pairing pops") { return run.template operator()<by_tunable_pairing_pops>(); };
	}
}

TEST_CASE("convex_piecewise_linear_forest race: clear and take_all", "[convex_piecewise_linear_forest][race]") {
	std::mt19937_64 rng(2026'10'29);
	std::vector<int64> positions(1 << 20);
	for (int64 &x : positions) x = int64(rng() % 1'000'000'000);
	std::size_t const tree_size = 200'000;
	std::vector<std::size_t> parent(tree_size, 0);
	for (std::size_t v = 1; v < tree_size; ++v) parent[v] = std::size_t(rng() % v);

	race("clear: 1 set x 1e6", [&]<discardable_pool Pool>() { return discard_every_set<Pool>(1, 1'000'000, 1, false, positions); });
	race("clear: 1000 sets x 1000", [&]<discardable_pool Pool>() { return discard_every_set<Pool>(1000, 1000, 1, false, positions); });
	race("clear: 1e5 sets x 10", [&]<discardable_pool Pool>() { return discard_every_set<Pool>(100'000, 10, 1, false, positions); });
	race("clear + refill: 1000 sets x 1000, 3 rounds", [&]<discardable_pool Pool>() { return discard_every_set<Pool>(1000, 1000, 3, false, positions); });
	race("clear: melded random tree n=2e5 (root)", [&]<discardable_pool Pool>() { return discard_melded_tree<Pool>(tree_size, false, positions, parent); });
	race("take_all: 1 set x 1e6", [&]<discardable_pool Pool>() { return discard_every_set<Pool>(1, 1'000'000, 1, true, positions); });
	race("take_all: 1000 sets x 1000", [&]<discardable_pool Pool>() { return discard_every_set<Pool>(1000, 1000, 1, true, positions); });
	race("take_all: 1e5 sets x 10", [&]<discardable_pool Pool>() { return discard_every_set<Pool>(100'000, 10, 1, true, positions); });
	race("take_all: melded random tree n=2e5 (root)", [&]<discardable_pool Pool>() { return discard_melded_tree<Pool>(tree_size, true, positions, parent); });
}
