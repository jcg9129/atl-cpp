/**
 * @file
 * @brief Races randomized, rank-leftist and weight-biased path-copying melds in
 * class and forest forms, optional lazy shifts, and in-arena versus vector heapify.
 * Measures complete-script throughput on random version DAGs, linear push/pop,
 * keep-all versions, million-element construction and chained three-element ranges.
 * Also measures Dijkstra plus sidetracks and class/forest walk/pop searches,
 * and isolates heap construction on random n=1e5 m=3e5 and 387x387 grid graphs.
 * Random forest construction library 13.0 / 13.2 ms versus vector heapify 16.0 / 16.5;
 * complete-search contenders without a record are not measured yet. Xeon Platinum 8280, GCC 15.2,
 * RelWithDebInfo, assertions off, 2026-11-01, -O2 / -O2 -march=native
 * (also -g -DNDEBUG -std=c++23): range build library 10.0 / 9.2 ms versus
 * vector heapify 11.3 / 11.2. See docs/persistent-heap.md for the full tables.
 */

#include <atl/ds/persistent_heap.hpp>
#include <atl/experimental/ds/persistent_heap_variants.hpp>
#include <atl/experimental/graph/k_shortest_walks.hpp>
#include <atl/experimental/graph/k_shortest_walks_variants.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
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
	/** @brief A contender's class form on int64 (with `Shift`, its lazy-add variant). */
	template<algorithm Algorithm, typename Shift = void>
	using class_heap = ex::persistent_heap_by_meld_policy<int64, std::less<>, Algorithm, Shift>;
	/** @brief A contender's forest form on int64. */
	template<algorithm Algorithm>
	using forest_heap = ex::persistent_heap_forest_by_meld_policy<int64, std::less<>, Algorithm>;

	/** @brief One contender on the generic shapes: version DAG, linear script, keep-all, build. */
	template<algorithm Algorithm>
	void generic_rows(std::vector<dag_step> const &dag, std::vector<dag_step> const &dag_add,
					  std::vector<int64> const &script, std::vector<int64> const &values, std::vector<uint32> const &probes) {
		std::string const a = name_of(Algorithm);
		REQUIRE(replay_class<class_heap<Algorithm>>(dag) == replay_class<atl::persistent_heap<int64>>(dag));
		REQUIRE(replay_forest<forest_heap<Algorithm>>(dag) == replay_forest<atl::persistent_heap_forest<int64>>(dag));
		REQUIRE(replay_class<class_heap<Algorithm, int64>>(dag) == replay_class<class_heap<Algorithm>>(dag));
		REQUIRE(replay_class<class_heap<Algorithm, int64>>(dag_add) == replay_class<class_heap<algorithm::rank_leftist, int64>>(dag_add));
		REQUIRE(linear_class<class_heap<Algorithm>>(script) == linear_class<atl::persistent_heap<int64>>(script));
		REQUIRE(linear_forest<forest_heap<Algorithm>>(script) == linear_forest<atl::persistent_heap_forest<int64>>(script));
		REQUIRE(linear_class<class_heap<Algorithm, int64>>(script) == linear_class<class_heap<Algorithm>>(script));
		REQUIRE(keep_all_class<class_heap<Algorithm>>(values, probes) == keep_all_class<atl::persistent_heap<int64>>(values, probes));
		REQUIRE(keep_all_forest<forest_heap<Algorithm>>(values, probes) == keep_all_forest<atl::persistent_heap_forest<int64>>(values, probes));
		REQUIRE(class_heap<Algorithm>(values).size() == values.size());
		measure("dag 5e5 / " + a + " / class", [&] { return replay_class<class_heap<Algorithm>>(dag); });
		measure("dag 5e5 / " + a + " / forest", [&] { return replay_forest<forest_heap<Algorithm>>(dag); });
		measure("dag 5e5 / " + a + " / class, lazy-add type unused", [&] {
			return replay_class<class_heap<Algorithm, int64>>(dag);
		});
		measure("dag 5e5 + 10% add_to_all / " + a + " / class, lazy add", [&] {
			return replay_class<class_heap<Algorithm, int64>>(dag_add);
		});
		measure("linear 1e6 push/pop 2:1 / " + a + " / class", [&] { return linear_class<class_heap<Algorithm>>(script); });
		measure("linear 1e6 push/pop 2:1 / " + a + " / forest", [&] { return linear_forest<forest_heap<Algorithm>>(script); });
		measure("linear 1e6 push/pop 2:1 / " + a + " / class, lazy-add type unused", [&] {
			return linear_class<class_heap<Algorithm, int64>>(script);
		});
		measure("keep-all 1e6 push + 2e5 pop / " + a + " / class", [&] {
			return keep_all_class<class_heap<Algorithm>>(values, probes);
		});
		measure("keep-all 1e6 push + 2e5 pop / " + a + " / forest", [&] {
			return keep_all_forest<forest_heap<Algorithm>>(values, probes);
		});
		measure("build 1e6 (range constructor) / " + a + " / class", [&] { return class_heap<Algorithm>(values).size(); });
	}

	using ex::cheaper_sidetrack;
	using ex::sidetrack;

	/** @brief Race class/forest heap-node walks and version pops for one meld policy. */
	template<algorithm Algorithm>
	void walk_rows(std::string const &label, std::size_t n, std::vector<arc> const &edges, uint32 s, uint32 t, std::size_t k,
				   std::vector<int64> const &expected) {
		using heap = ex::persistent_heap_by_meld_policy<sidetrack, cheaper_sidetrack, Algorithm>;
		using forest = ex::persistent_heap_forest_by_meld_policy<sidetrack, cheaper_sidetrack, Algorithm>;
		auto const tree = ex::shortest_paths_to(n, edges, t);
		auto const lists = ex::sidetracks_of(n, edges, tree);
		REQUIRE(ex::k_shortest_walks_by_class_walk<heap>(n, tree, lists, s, t, k) == expected);
		REQUIRE(ex::k_shortest_walks_by_class_pop<heap>(n, tree, lists, s, t, k) == expected);
		REQUIRE(ex::k_shortest_walks_by_forest_walk<forest>(n, tree, lists, s, t, k) == expected);
		REQUIRE(ex::k_shortest_walks_by_forest_pop<forest>(n, tree, lists, s, t, k) == expected);
		std::string const a = label + " / " + name_of(Algorithm);
		measure(a + " / by class walk", [&] {
			auto const paths = ex::shortest_paths_to(n, edges, t);
			auto const tracks = ex::sidetracks_of(n, edges, paths);
			return ex::k_shortest_walks_by_class_walk<heap>(n, paths, tracks, s, t, k).back();
		});
		measure(a + " / by class pop", [&] {
			auto const paths = ex::shortest_paths_to(n, edges, t);
			auto const tracks = ex::sidetracks_of(n, edges, paths);
			return ex::k_shortest_walks_by_class_pop<heap>(n, paths, tracks, s, t, k).back();
		});
		measure(a + " / by forest walk", [&] {
			auto const paths = ex::shortest_paths_to(n, edges, t);
			auto const tracks = ex::sidetracks_of(n, edges, paths);
			return ex::k_shortest_walks_by_forest_walk<forest>(n, paths, tracks, s, t, k).back();
		});
		measure(a + " / by forest pop", [&] {
			auto const paths = ex::shortest_paths_to(n, edges, t);
			auto const tracks = ex::sidetracks_of(n, edges, paths);
			return ex::k_shortest_walks_by_forest_pop<forest>(n, paths, tracks, s, t, k).back();
		});
	}

	/** @brief Include Dijkstra and sidetrack staging in every complete-search sample. */
	void k_shortest_rows(std::string const &label, std::size_t n, std::vector<arc> const &edges, uint32 s, uint32 t, std::size_t k) {
		auto const expected = ex::k_shortest_walks(n, edges, s, t, k);
		REQUIRE(expected.size() == k);
		auto const tree = ex::shortest_paths_to(n, edges, t);
		auto const lists = ex::sidetracks_of(n, edges, tree);
		REQUIRE(ex::k_shortest_walks_by_class_walk<atl::persistent_heap<sidetrack, cheaper_sidetrack>>(n, tree, lists, s, t, k) == expected);
		measure(label + " / stages only (Dijkstra + sidetracks)", [&] {
			auto const paths = ex::shortest_paths_to(n, edges, t);
			return ex::sidetracks_of(n, edges, paths).items.size();
		});
		measure(label + " / library forest walk", [&] { return ex::k_shortest_walks(n, edges, s, t, k).back(); });
		measure(label + " / library heap by class walk", [&] {
			auto const paths = ex::shortest_paths_to(n, edges, t);
			auto const tracks = ex::sidetracks_of(n, edges, paths);
			return ex::k_shortest_walks_by_class_walk<atl::persistent_heap<sidetrack, cheaper_sidetrack>>(n, paths, tracks, s, t, k).back();
		});
		walk_rows<algorithm::rank_leftist>(label, n, edges, s, t, k, expected);
		walk_rows<algorithm::weight_biased>(label, n, edges, s, t, k, expected);
		walk_rows<algorithm::randomized>(label, n, edges, s, t, k, expected);
	}

}

TEST_CASE("persistent heap race: generic shapes", "[persistent_heap]") {
	uint64 const seed = 20261023;
	auto const dag = make_dag(seed, 500000, 1 << 16, false);
	auto const dag_add = make_dag(seed + 1, 500000, 1 << 16, true);
	std::mt19937_64 rng(seed + 2);
	std::vector<int64> script(1000000);
	for (int64 &x : script) x = rng() % 3 == 0 ? -1 : int64(rng() % 1000000000);
	std::vector<int64> values(1000000);
	for (int64 &x : values) x = int64(rng() % 1000000000);
	std::vector<uint32> probes(200000);
	for (uint32 &p : probes) p = uint32(1 + rng() % values.size());

	measure("linear 1e6 push/pop 2:1 / non-persistent leftist forest (reference)", [&] { return linear_ephemeral(script); });
	measure("dag 5e5 / library persistent_heap", [&] { return replay_class<atl::persistent_heap<int64>>(dag); });
	measure("dag 5e5 / library persistent_heap_forest", [&] {
		return replay_forest<atl::persistent_heap_forest<int64>>(dag);
	});
	measure("linear 1e6 push/pop 2:1 / library persistent_heap", [&] {
		return linear_class<atl::persistent_heap<int64>>(script);
	});
	measure("keep-all 1e6 push + 2e5 pop / library persistent_heap", [&] {
		return keep_all_class<atl::persistent_heap<int64>>(values, probes);
	});
	generic_rows<algorithm::rank_leftist>(dag, dag_add, script, values, probes);
	generic_rows<algorithm::weight_biased>(dag, dag_add, script, values, probes);
	generic_rows<algorithm::randomized>(dag, dag_add, script, values, probes);
}

namespace {
	/** @brief The randomized heap with range construction / `push_range` with the by vector heapify (slower variant). */
	template<typename T, typename Compare>
	using persistent_heap_by_vector_heapify = ex::persistent_heap_by_meld_policy<T, Compare, algorithm::randomized, void, atl::shift_by_addition,
																				 atl::compose_by_addition, false, true>;
	/** @brief The forest form of @ref persistent_heap_by_vector_heapify. */
	template<typename T, typename Compare>
	using persistent_heap_forest_by_vector_heapify = ex::persistent_heap_forest_by_meld_policy<T, Compare, algorithm::randomized, void, atl::shift_by_addition,
																							   atl::compose_by_addition, false, true>;

	/** @brief 1e5 chained `push_range` of three values on the newest version. */
	template<int64_class_heap Heap>
		requires requires(Heap h, std::array<int64, 3> values) { h.push_range(values); }
	int64 chained_push_ranges(std::vector<int64> const &values) {
		Heap heap;
		std::size_t const count = values.size() / 3;
		for (std::size_t i = 0; i < count; ++i)
			heap = std::move(heap).push_range(std::array<int64, 3>{values[3 * i], values[3 * i + 1], values[3 * i + 2]});
		return heap.top();
	}

	/** @brief The same chain on the forest form. */
	template<int64_forest_heap Forest>
		requires requires(Forest f, std::array<int64, 3> values) { f.push_range(0, values); }
	int64 chained_push_ranges_forest(std::vector<int64> const &values) {
		Forest forest;
		uint32 heap = Forest::empty_heap;
		std::size_t const count = values.size() / 3;
		for (std::size_t i = 0; i < count; ++i)
			heap = forest.push_range(heap, std::array<int64, 3>{values[3 * i], values[3 * i + 1], values[3 * i + 2]});
		return forest.top(heap);
	}

}

namespace {
	/** @brief Isolate sidetrack-heap construction, comparing in-arena and temporary-vector heapify. */
	void construction_rows(std::string const &label, std::size_t n, std::vector<arc> const &edges, uint32 t) {
		auto const tree = ex::shortest_paths_to(n, edges, t);
		auto const lists = ex::sidetracks_of(n, edges, tree);
		using heap = atl::persistent_heap<sidetrack, cheaper_sidetrack>;
		using vector_heap = persistent_heap_by_vector_heapify<sidetrack, cheaper_sidetrack>;
		using forest = atl::persistent_heap_forest<sidetrack, cheaper_sidetrack>;
		using vector_forest = persistent_heap_forest_by_vector_heapify<sidetrack, cheaper_sidetrack>;
		{
			auto const a = ex::detail::class_sidetrack_heaps<heap>(n, tree, lists, t);
			auto const b = ex::detail::class_sidetrack_heaps<vector_heap>(n, tree, lists, t);
			forest f;
			vector_forest vf;
			auto const roots = ex::detail::forest_sidetrack_heaps(f, n, tree, lists, t);
			auto const vector_roots = ex::detail::forest_sidetrack_heaps(vf, n, tree, lists, t);
			for (uint32 v = 0; v < n; ++v) {
				REQUIRE(a[v].empty() == b[v].empty());
				REQUIRE(a[v].empty() == f.empty(roots[v]));
				REQUIRE(a[v].empty() == vf.empty(vector_roots[v]));
				if (!a[v].empty()) {
					REQUIRE(a[v].top().extra == b[v].top().extra);
					REQUIRE(a[v].top().extra == f.top(roots[v]).extra);
					REQUIRE(a[v].top().extra == vf.top(vector_roots[v]).extra);
				}
			}
		}
		measure(label + " / H-construction / library class in-arena heapify", [&] {
			return ex::detail::class_sidetrack_heaps<heap>(n, tree, lists, t).size();
		});
		measure(label + " / H-construction / class by vector heapify", [&] {
			return ex::detail::class_sidetrack_heaps<vector_heap>(n, tree, lists, t).size();
		});
		measure(label + " / H-construction / library forest in-arena heapify", [&] {
			forest f;
			return ex::detail::forest_sidetrack_heaps(f, n, tree, lists, t).size();
		});
		measure(label + " / H-construction / forest by vector heapify", [&] {
			vector_forest f;
			return ex::detail::forest_sidetrack_heaps(f, n, tree, lists, t).size();
		});
	}
}

TEST_CASE("persistent heap race: in-place heapify", "[persistent_heap][in_place]") {
	std::mt19937_64 rng(20261029);
	std::vector<int64> values(1000000);
	for (int64 &x : values) x = int64(rng() % 1000000000);
	REQUIRE(persistent_heap_by_vector_heapify<int64, std::less<>>(values).size() == atl::persistent_heap<int64>(values).size());
	measure("build 1e6 (range constructor) / persistent_heap (library: in-place)", [&] { return atl::persistent_heap<int64>(values).size(); });
	measure("build 1e6 (range constructor) / persistent_heap by vector heapify", [&] { return persistent_heap_by_vector_heapify<int64, std::less<>>(values).size(); });
	std::vector<int64> short_values(300000);
	for (int64 &x : short_values) x = int64(rng() % 1000000000);
	REQUIRE(chained_push_ranges<atl::persistent_heap<int64>>(short_values) == chained_push_ranges<persistent_heap_by_vector_heapify<int64, std::less<>>>(short_values));
	REQUIRE(chained_push_ranges_forest<atl::persistent_heap_forest<int64>>(short_values) == chained_push_ranges_forest<persistent_heap_forest_by_vector_heapify<int64, std::less<>>>(short_values));
	measure("1e5 x push_range(3) / persistent_heap (library: in-place) / class", [&] { return chained_push_ranges<atl::persistent_heap<int64>>(short_values); });
	measure("1e5 x push_range(3) / persistent_heap by vector heapify / class", [&] { return chained_push_ranges<persistent_heap_by_vector_heapify<int64, std::less<>>>(short_values); });
	measure("1e5 x push_range(3) / persistent_heap_forest (library: in-place) / forest", [&] {
		return chained_push_ranges_forest<atl::persistent_heap_forest<int64>>(short_values);
	});
	measure("1e5 x push_range(3) / persistent_heap_forest by vector heapify / forest", [&] {
		return chained_push_ranges_forest<persistent_heap_forest_by_vector_heapify<int64, std::less<>>>(short_values);
	});
	construction_rows("k-shortest random n=1e5 m=3e5", 100000, random_graph(20261023, 100000, 300000), 1);
	std::size_t const side = 387;
	construction_rows("k-shortest grid 387x387", side * side, grid_graph(side), uint32(side * side - 1));
}

TEST_CASE("persistent heap race: k shortest walks", "[persistent_heap]") {
	std::size_t const k = 300000;
	k_shortest_rows("k-shortest random n=1e5 m=3e5 K=3e5", 100000, random_graph(20261023, 100000, 300000), 0, 1, k);
	std::size_t const side = 387;
	k_shortest_rows("k-shortest grid 387x387 K=3e5", side * side, grid_graph(side), 0, uint32(side * side - 1), k);
}
