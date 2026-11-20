/**
 * @file
 * @brief Races root ordering by branch or select and reuse or copying of fresh nodes
 * for randomized, rank-leftist and weight-biased path-copying melds. Measures
 * complete-script throughput on 5e5-step version DAGs, 1e6-step linear push/pop
 * and 1e6 pushes retaining all versions plus 2e5 random pops, plus forest
 * sidetrack searches with n=1e5 m=3e5 K=3e5, including Dijkstra and sidetracks.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-23,
 * -O2 / -O2 -march=native (also -g -DNDEBUG -std=c++23): rank-leftist DAG
 * branch/reuse 143 / 142 ms versus select/reuse 168 / 170.
 * Forest walks by randomized meld branch/reuse take 170 / 168 ms versus
 * rank-leftist 174 / 179; docs/persistent-heap.md has full tables.
 */

#include <atl/experimental/ds/persistent_heap_variants.hpp>
#include <atl/experimental/graph/k_shortest_walks.hpp>
#include <atl/experimental/graph/k_shortest_walks_variants.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "persistent_heap_workloads.hpp"

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
using atl::uint64;
namespace ex = atl::experimental;
using namespace persistent_heap_bench;
using algorithm = ex::persistent_heap_algorithm;

namespace {
	/** @brief A contender's class form on int64 with the given knobs. */
	template<algorithm Algorithm, bool IsBranchLight, bool ReusesFreshNodes>
	using class_heap = ex::persistent_heap_by_meld_policy<int64, std::less<>, Algorithm, void, atl::shift_by_addition,
														  atl::compose_by_addition, IsBranchLight, ReusesFreshNodes>;

	/** @brief A contender's forest of sidetracks with the selected root and copy policies. */
	template<algorithm Algorithm, bool IsBranchLight, bool ReusesFreshNodes>
	using sidetrack_forest = ex::persistent_heap_forest_by_meld_policy<ex::sidetrack, ex::cheaper_sidetrack, Algorithm, void,
																	   atl::shift_by_addition, atl::compose_by_addition, IsBranchLight, ReusesFreshNodes>;

	/** @brief One contender with one knob setting on the four tuning shapes. */
	template<algorithm Algorithm, bool IsBranchLight, bool ReusesFreshNodes>
	void rows(std::string const &knobs, std::vector<dag_step> const &dag, std::vector<int64> const &script,
			  std::vector<int64> const &values, std::vector<uint32> const &probes, std::vector<arc> const &edges, std::size_t n) {
		using heap = class_heap<Algorithm, IsBranchLight, ReusesFreshNodes>;
		using reference = class_heap<algorithm::rank_leftist, false, true>;
		REQUIRE(replay_class<heap>(dag) == replay_class<reference>(dag));
		REQUIRE(linear_class<heap>(script) == linear_class<reference>(script));
		REQUIRE(keep_all_class<heap>(values, probes) == keep_all_class<reference>(values, probes));
		std::string const a = name_of(Algorithm) + " / " + knobs;
		measure("dag 5e5 / " + a, [&] { return replay_class<heap>(dag); });
		measure("linear 1e6 push/pop 2:1 / " + a, [&] { return linear_class<heap>(script); });
		measure("keep-all 1e6 push + 2e5 pop / " + a, [&] { return keep_all_class<heap>(values, probes); });
		auto const tree = ex::shortest_paths_to(n, edges, 1);
		auto const lists = ex::sidetracks_of(n, edges, tree);
		using forest = sidetrack_forest<Algorithm, IsBranchLight, ReusesFreshNodes>;
		REQUIRE(ex::k_shortest_walks_by_forest_walk<forest>(n, tree, lists, 0, 1, 300000) == ex::k_shortest_walks(n, edges, 0, 1, 300000));
		measure("k-shortest random n=1e5 m=3e5 K=3e5, by forest walk / " + a, [&] {
			auto const paths = ex::shortest_paths_to(n, edges, 1);
			auto const tracks = ex::sidetracks_of(n, edges, paths);
			return ex::k_shortest_walks_by_forest_walk<forest>(n, paths, tracks, 0, 1, 300000).back();
		});
	}

	/** @brief One contender with each knob setting. */
	template<algorithm Algorithm>
	void knob_rows(std::vector<dag_step> const &dag, std::vector<int64> const &script, std::vector<int64> const &values,
				   std::vector<uint32> const &probes, std::vector<arc> const &edges, std::size_t n) {
		rows<Algorithm, false, true>("branch, reuse fresh", dag, script, values, probes, edges, n);
		rows<Algorithm, true, true>("select, reuse fresh", dag, script, values, probes, edges, n);
		rows<Algorithm, false, false>("branch, copy fresh", dag, script, values, probes, edges, n);
	}
}

TEST_CASE("persistent heap tuning", "[persistent_heap]") {
	uint64 const seed = 20261023;
	auto const dag = make_dag(seed, 500000, 1 << 16, false);
	std::mt19937_64 rng(seed + 2);
	std::vector<int64> script(1000000);
	for (int64 &x : script) x = rng() % 3 == 0 ? -1 : int64(rng() % 1000000000);
	std::vector<int64> values(1000000);
	for (int64 &x : values) x = int64(rng() % 1000000000);
	std::vector<uint32> probes(200000);
	for (uint32 &p : probes) p = uint32(1 + rng() % values.size());
	std::size_t const n = 100000;
	auto const edges = random_graph(seed, n, 300000);
	knob_rows<algorithm::rank_leftist>(dag, script, values, probes, edges, n);
	knob_rows<algorithm::weight_biased>(dag, script, values, probes, edges, n);
	knob_rows<algorithm::randomized>(dag, script, values, probes, edges, n);
}
