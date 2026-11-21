/**
 * @file
 * @brief Leaf-path balancing race: the worked example
 * atl::experimental::balance_leaf_paths (two pairing-heap pools of
 * atl::convex_piecewise_linear_forest over shared weight records, children
 * summed by O(1) melds) against by ordered map (each function a `std::map`
 * with a lazy offset, children merged smaller-into-larger) and by fewer
 * entries (the example on the same pool, each bounded edge's segment moving
 * the side with fewer entries to the crossing instead of the side of less
 * weight).
 *
 * Throughput of whole runs at n = 2e5 on a path, a star, a caterpillar (a
 * spine of n/2 with one leaf per spine vertex) and a random recursive tree,
 * costs uniform in [1, 1e6], in two modes: free-cost weighted (weights
 * uniform in [1, 1e6], any new costs) and nonnegative unit-weight (unit
 * weights, new costs >= 0); and the bounded rising-weight path (unit costs,
 * new costs >= 0, weight[1] = 4n, weight[v] = n - v below, n = 2e3), where
 * the weight rule moves about n^2 / 2 pieces. Contenders agree on every
 * total cost and path total before timing (and the map, which keeps the
 * example's weight rule, on every new cost).
 *
 * Result, Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-10-24, plain -O2 / -march=native: the example wins every shape, 1.15x
 * to 2.46x free-cost weighted and 1.03x to 2.43x nonnegative unit-weight (the
 * star: 44.4 / 43.7 against 109 / 106 ms). Full table:
 * docs/convex-piecewise-linear-forest.md. By fewer entries and the
 * rising-weight path: not measured yet.
 */
#include <atl/experimental/dp/leaf_path_balance.hpp>
#include <atl/experimental/dp/leaf_path_balance_variants.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

using atl::int64;
using atl::uint32;

namespace {
	/** @brief A rooted tree with edge data, vertex 0 the root. */
	struct weighted_tree {
		std::vector<uint32> parent;
		std::vector<int64> cost, weight;
	};

	/**
	 * @brief `n` vertices shaped as a path, a star, a caterpillar (a spine of
	 *        n/2 with one leaf on each spine vertex) or a random recursive tree
	 *        (parent uniform among the earlier vertices); costs uniform in [1,
	 *        1e6], weights uniform in [1, max_weight].
	 */
	weighted_tree make_tree(std::string const &shape, std::size_t n, int64 max_weight) {
		std::mt19937_64 rng(1);
		weighted_tree tree{std::vector<uint32>(n, 0), std::vector<int64>(n, 0), std::vector<int64>(n, 0)};
		for (std::size_t v = 1; v < n; ++v) {
			if (shape == "path") tree.parent[v] = uint32(v - 1);
			if (shape == "star") tree.parent[v] = 0;
			if (shape == "caterpillar") tree.parent[v] = uint32(v % 2 == 1 ? (v >= 2 ? v - 2 : 0) : v - 1);
			if (shape == "random") tree.parent[v] = uint32(rng() % v);
			tree.cost[v] = int64(rng() % 1'000'000) + 1;
			tree.weight[v] = int64(rng() % std::uint64_t(max_weight)) + 1;
		}
		return tree;
	}
}

TEST_CASE("leaf_path_balance", "[leaf_path_balance]") {
	std::size_t const n = 200'000;
	for (std::string const shape : {"path", "star", "caterpillar", "random"}) {
		for (bool const is_nonnegative : {false, true}) {
			weighted_tree const tree = make_tree(shape, n, is_nonnegative ? 1 : 1'000'000);
			std::optional<int64> const min_cost = is_nonnegative ? std::optional<int64>(0) : std::nullopt;
			std::string const name =
				std::string(is_nonnegative ? "nonnegative unit-weight " : "free-cost weighted ") + shape + " n=2e5";
			auto const example = atl::experimental::balance_leaf_paths(tree.parent, tree.cost, tree.weight, min_cost);
			auto const by_ordered_map =
				atl::experimental::balance_leaf_paths_by_ordered_map(tree.parent, tree.cost, tree.weight, min_cost);
			REQUIRE(by_ordered_map.total_cost == example.total_cost);
			REQUIRE(by_ordered_map.new_cost == example.new_cost);
			auto const by_fewer_entries =
				atl::experimental::balance_leaf_paths_by_fewer_entries(tree.parent, tree.cost, tree.weight, min_cost);
			REQUIRE(by_fewer_entries.total_cost == example.total_cost);
			REQUIRE(by_fewer_entries.path_sum == example.path_sum);
			BENCHMARK(name + " / example") {
				return atl::experimental::balance_leaf_paths(tree.parent, tree.cost, tree.weight, min_cost).total_cost;
			};
			BENCHMARK(name + " / by ordered map") {
				return atl::experimental::balance_leaf_paths_by_ordered_map(tree.parent, tree.cost, tree.weight, min_cost)
					.total_cost;
			};
			BENCHMARK(name + " / by fewer entries") {
				return atl::experimental::balance_leaf_paths_by_fewer_entries(tree.parent, tree.cost, tree.weight, min_cost)
					.total_cost;
			};
		}
	}
	// the bounded rising-weight path: the light side splinters, the heavy side is one wall
	std::size_t const path_n = 2000;
	weighted_tree path{std::vector<uint32>(path_n, 0), std::vector<int64>(path_n, 1), std::vector<int64>(path_n, 0)};
	path.cost[0] = 0;
	for (std::size_t v = 1; v < path_n; ++v) {
		path.parent[v] = uint32(v - 1);
		path.weight[v] = v == 1 ? int64(4 * path_n) : int64(path_n - v);
	}
	std::optional<int64> const zero = 0;
	auto const example = atl::experimental::balance_leaf_paths(path.parent, path.cost, path.weight, zero);
	auto const by_fewer_entries = atl::experimental::balance_leaf_paths_by_fewer_entries(path.parent, path.cost, path.weight, zero);
	REQUIRE(by_fewer_entries.total_cost == example.total_cost);
	REQUIRE(by_fewer_entries.path_sum == example.path_sum);
	REQUIRE(atl::experimental::balance_leaf_paths_by_ordered_map(path.parent, path.cost, path.weight, zero).total_cost
			== example.total_cost);
	BENCHMARK("bounded rising-weight path n=2e3 / example") {
		return atl::experimental::balance_leaf_paths(path.parent, path.cost, path.weight, zero).total_cost;
	};
	BENCHMARK("bounded rising-weight path n=2e3 / by ordered map") {
		return atl::experimental::balance_leaf_paths_by_ordered_map(path.parent, path.cost, path.weight, zero).total_cost;
	};
	BENCHMARK("bounded rising-weight path n=2e3 / by fewer entries") {
		return atl::experimental::balance_leaf_paths_by_fewer_entries(path.parent, path.cost, path.weight, zero).total_cost;
	};
}
