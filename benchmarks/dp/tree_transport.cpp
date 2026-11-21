/**
 * @file
 * @brief Tree transport race: the worked example
 * atl::experimental::tree_transport_cost (the dual slope trick on the
 * library's pairing heaps, the segment being eaten held out of its heap, the
 * reverse segments pushed after each re-centering and skipped at the root)
 * against contenders that each change one thing: by skew, by rank leftist,
 * by binary (the same loop on another meldable-heap engine), by pop and push
 * (both tops popped and the rest pushed back on every swap, reverse segments
 * pushed at once) and by length records (that loop with the lengths in a side
 * array).
 *
 * Throughput of whole runs at n = 2.5e5 on a path, a star, a caterpillar and
 * a random recursive tree, costs uniform in [1, 1e6], with two unit layouts:
 * mixed (every vertex a supply and a demand uniform in [0, 7]) and split
 * (supplies on the first half, demands on the second). Contenders agree on
 * every result before timing.
 *
 * Result, Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-10-24, plain -O2 / -march=native: the example beats by pop and push
 * 1.44x / 1.33x on the mixed star; by skew is faster on the paths and random
 * trees and slower on the stars; by binary wins the stars 1.42x to 1.52x and
 * loses the rest 2.14x to 2.90x. Full table: docs/convex-piecewise-linear-forest.md.
 */
#include <atl/experimental/dp/tree_transport.hpp>
#include <atl/experimental/dp/tree_transport_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

using atl::int64;
using atl::uint32;

namespace {
	/** @brief A rooted tree with per-edge costs and per-vertex supplies and demands, vertex 0 the root. */
	struct transport_instance {
		std::vector<uint32> parent;
		std::vector<int64> cost, supply, demand;
	};

	/**
	 * @brief `n` vertices shaped as a path, a star, a caterpillar (a spine of
	 *        n/2 with one leaf on each spine vertex) or a random recursive tree
	 *        (parent uniform among the earlier vertices); costs uniform in [1,
	 *        1e6]. Units: "mixed" gives every vertex a supply and a demand
	 *        uniform in [0, 7]; "split" gives the first half supplies and the
	 *        second half demands, uniform in [0, 7] (so units cross the tree).
	 *        Demands are then lowered until the supply covers them; the total
	 *        supply stays under 1e6.
	 */
	transport_instance make_instance(std::string const &shape, std::string const &units, std::size_t n) {
		std::mt19937_64 rng(1);
		transport_instance instance{std::vector<uint32>(n, 0), std::vector<int64>(n, 0), std::vector<int64>(n, 0),
									std::vector<int64>(n, 0)};
		int64 supply_sum = 0, demand_sum = 0;
		for (std::size_t v = 0; v < n; ++v) {
			if (v > 0) {
				if (shape == "path") instance.parent[v] = uint32(v - 1);
				if (shape == "star") instance.parent[v] = 0;
				if (shape == "caterpillar") instance.parent[v] = uint32(v % 2 == 1 ? (v >= 2 ? v - 2 : 0) : v - 1);
				if (shape == "random") instance.parent[v] = uint32(rng() % v);
				instance.cost[v] = int64(rng() % 1'000'000) + 1;
			}
			bool const is_mixed = units == "mixed";
			instance.supply[v] = is_mixed || v < n / 2 ? int64(rng() % 8) : 0;
			instance.demand[v] = is_mixed || v >= n / 2 ? int64(rng() % 8) : 0;
			supply_sum += instance.supply[v];
			demand_sum += instance.demand[v];
		}
		for (std::size_t v = 0; demand_sum > supply_sum; ++v) {
			int64 const lowered = std::min(instance.demand[v], demand_sum - supply_sum);
			instance.demand[v] -= lowered;
			demand_sum -= lowered;
		}
		return instance;
	}
}

TEST_CASE("tree_transport", "[tree_transport]") {
	namespace ex = atl::experimental;
	std::size_t const n = 250'000;
	for (std::string const units : {"mixed", "split"}) {
		for (std::string const shape : {"path", "star", "caterpillar", "random"}) {
			transport_instance const t = make_instance(shape, units, n);
			std::string const name = units + " " + shape + " n=2.5e5";
			int64 const expected = ex::tree_transport_cost(t.parent, t.cost, t.supply, t.demand);
			REQUIRE(ex::tree_transport_cost_by_pop_and_push(t.parent, t.cost, t.supply, t.demand) == expected);
			REQUIRE(ex::tree_transport_cost_by_length_records(t.parent, t.cost, t.supply, t.demand) == expected);
			REQUIRE(ex::tree_transport_cost_by_skew(t.parent, t.cost, t.supply, t.demand) == expected);
			REQUIRE(ex::tree_transport_cost_by_rank_leftist(t.parent, t.cost, t.supply, t.demand) == expected);
			REQUIRE(ex::tree_transport_cost_by_binary(t.parent, t.cost, t.supply, t.demand) == expected);
			BENCHMARK(name + " / example (pairing)") {
				return ex::tree_transport_cost(t.parent, t.cost, t.supply, t.demand);
			};
			BENCHMARK(name + " / by pop and push") {
				return ex::tree_transport_cost_by_pop_and_push(t.parent, t.cost, t.supply, t.demand);
			};
			BENCHMARK(name + " / by length records") {
				return ex::tree_transport_cost_by_length_records(t.parent, t.cost, t.supply, t.demand);
			};
			BENCHMARK(name + " / by skew") { return ex::tree_transport_cost_by_skew(t.parent, t.cost, t.supply, t.demand); };
			BENCHMARK(name + " / by rank leftist") {
				return ex::tree_transport_cost_by_rank_leftist(t.parent, t.cost, t.supply, t.demand);
			};
			BENCHMARK(name + " / by binary") {
				return ex::tree_transport_cost_by_binary(t.parent, t.cost, t.supply, t.demand);
			};
		}
	}
}
