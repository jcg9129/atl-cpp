#include <atl/experimental/dp/leaf_path_balance.hpp>
#include <atl/experimental/dp/leaf_path_balance_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <vector>
#include "../expect_abort.hpp"

using atl::int64;
using atl::uint32;

namespace {
	/** @brief A rooted tree with edge data, vertex 0 the root. */
	struct weighted_tree {
		std::vector<uint32> parent;
		std::vector<int64> cost, weight;
	};

	/** @brief The optimum of the table DP and the smallest root total attaining it (`-reach` when unbounded below). */
	struct brute_optimum {
		int64 total_cost;
		int64 smallest_path_sum;
	};

	/**
	 * @brief The certifying oracle of the family (the ordered-map contender is
	 *        only a differential partner): the tree DP on an explicit table of `g_v(x)` for
	 *        every integer `x` in `[-reach, reach]` (a leaf is the indicator of
	 *        0, an edge an explicit min over predecessors), O(n reach^2).
	 */
	brute_optimum brute_minimum(weighted_tree const &tree, std::optional<int64> min_cost, int64 reach) {
		std::size_t const n = tree.parent.size();
		std::size_t const width = std::size_t(2 * reach + 1);
		int64 const infinity = std::numeric_limits<int64>::max() / 4;
		std::vector<std::vector<int64>> g(n, std::vector<int64>(width, 0));
		std::vector<bool> has_child(n, false);
		for (std::size_t v = 1; v < n; ++v) has_child[tree.parent[v]] = true;
		for (std::size_t v = 0; v < n; ++v)
			if (!has_child[v])
				for (std::size_t i = 0; i < width; ++i) g[v][i] = int64(i) == reach ? 0 : infinity;
		// children have larger indices in the generated trees
		for (std::size_t v = n; v-- > 1;) {
			std::vector<int64> passed(width, infinity);
			for (std::size_t x = 0; x < width; ++x)
				for (std::size_t y = 0; y < width; ++y) {
					if (g[v][y] >= infinity) continue;
					int64 const t = int64(x) - int64(y);
					if (min_cost && t < *min_cost) continue;
					passed[x] = std::min(passed[x], g[v][y] + tree.weight[v] * std::abs(t - tree.cost[v]));
				}
			for (std::size_t x = 0; x < width; ++x) g[tree.parent[v]][x] = std::min(infinity, g[tree.parent[v]][x] + passed[x]);
		}
		auto const best = std::ranges::min_element(g[0]);
		return {*best, int64(best - g[0].begin()) - reach};
	}

	/**
	 * @brief Check the reconstruction: equal leaf paths at `path_sum`, cost
	 *        recomputed, bound kept (in `int128`: the inputs may sit at the
	 *        ends of `int64`).
	 */
	void check_assignment(weighted_tree const &tree, atl::experimental::leaf_path_balance const &result, std::optional<int64> min_cost) {
		std::size_t const n = tree.parent.size();
		REQUIRE(result.new_cost.size() == n);
		std::vector<atl::int128> depth(n, 0);
		std::vector<bool> has_child(n, false);
		atl::int128 total = 0;
		for (std::size_t v = 1; v < n; ++v) {
			has_child[tree.parent[v]] = true;
			depth[v] = depth[tree.parent[v]] + result.new_cost[v];
			atl::int128 const change = atl::int128(result.new_cost[v]) - tree.cost[v];
			total += atl::int128(tree.weight[v]) * (change < 0 ? -change : change);
			if (min_cost) REQUIRE(result.new_cost[v] >= *min_cost);
		}
		for (std::size_t v = 1; v < n; ++v)
			if (!has_child[v]) REQUIRE(depth[v] == atl::int128(result.path_sum));
		REQUIRE(total == atl::int128(result.total_cost));
	}

	/** @brief The three solvers of the family on one tree: the example and its two contenders. */
	std::vector<atl::experimental::leaf_path_balance> solve_all(weighted_tree const &tree, std::optional<int64> min_cost) {
		return {atl::experimental::balance_leaf_paths(tree.parent, tree.cost, tree.weight, min_cost),
				atl::experimental::balance_leaf_paths_by_ordered_map(tree.parent, tree.cost, tree.weight, min_cost),
				atl::experimental::balance_leaf_paths_by_fewer_entries(tree.parent, tree.cost, tree.weight, min_cost)};
	}

	/** @brief A random tree on `n` vertices, parent[v] < v, shaped by `shape` (0 random, 1 path, 2 star, 3 caterpillar). */
	weighted_tree random_tree(std::mt19937_64 &rng, std::size_t n, int shape, int64 cost_lo, int64 cost_hi, int64 weight_lo,
							  int64 weight_hi) {
		weighted_tree tree{std::vector<uint32>(n, 0), std::vector<int64>(n, 0), std::vector<int64>(n, 0)};
		for (std::size_t v = 1; v < n; ++v) {
			if (shape == 0) tree.parent[v] = uint32(std::uniform_int_distribution<std::size_t>(0, v - 1)(rng));
			if (shape == 1) tree.parent[v] = uint32(v - 1);
			if (shape == 2) tree.parent[v] = 0;
			if (shape == 3) tree.parent[v] = uint32(v % 2 == 1 ? (v >= 2 ? v - 2 : 0) : v - 1);
			tree.cost[v] = std::uniform_int_distribution<int64>(cost_lo, cost_hi)(rng);
			tree.weight[v] = std::uniform_int_distribution<int64>(weight_lo, weight_hi)(rng);
		}
		return tree;
	}

	atl::experimental::leaf_path_balance solve(weighted_tree const &tree, std::optional<int64> min_cost = std::nullopt) {
		return atl::experimental::balance_leaf_paths(tree.parent, tree.cost, tree.weight, min_cost);
	}
}

TEST_CASE("balance_leaf_paths: the free-cost weighted sample", "[dp][leaf_path_balance]") {
	// edges 1-2 (5, 4), 1-3 (15, 15), 2-4 (3, 2), 2-5 (5, 1), vertices renumbered from 0
	weighted_tree const tree{{0, 0, 0, 1, 1}, {0, 5, 15, 3, 5}, {0, 4, 15, 2, 1}};
	auto const result = solve(tree);
	REQUIRE(result.total_cost == 19);
	REQUIRE(result.path_sum == 15);
	REQUIRE(result.new_cost == std::vector<int64>{0, 5, 15, 10, 10});
	auto const by_ordered_map = atl::experimental::balance_leaf_paths_by_ordered_map(tree.parent, tree.cost, tree.weight);
	REQUIRE(by_ordered_map.total_cost == 19);
	REQUIRE(by_ordered_map.new_cost == result.new_cost);
}

TEST_CASE("balance_leaf_paths: unit weights: the nonnegative bound changes the optimum", "[dp][leaf_path_balance]") {
	// root - a (1), a with 10 leaves of 5, root with 20 leaves of 1: shortening the
	// edge to a to -4 costs 5; with lengths >= 0 the best is 41
	weighted_tree tree{{0, 0}, {0, 1}, {0, 1}};
	for (int i = 0; i < 10; ++i) {
		tree.parent.push_back(1);
		tree.cost.push_back(5);
		tree.weight.push_back(1);
	}
	for (int i = 0; i < 20; ++i) {
		tree.parent.push_back(0);
		tree.cost.push_back(1);
		tree.weight.push_back(1);
	}
	auto const free = solve(tree);
	auto const nonnegative_unit = solve(tree, 0);
	REQUIRE(free.total_cost == 5);
	REQUIRE(nonnegative_unit.total_cost == 41);
	check_assignment(tree, free, std::nullopt);
	check_assignment(tree, nonnegative_unit, 0);
}

TEST_CASE("balance_leaf_paths: small trees against the table DP", "[dp][leaf_path_balance]") {
	for (unsigned seed = 0; seed < 1500; ++seed) {
		std::mt19937_64 rng(seed);
		std::size_t const n = std::uniform_int_distribution<std::size_t>(1, 8)(rng);
		int const shape = int(seed % 4);
		int const mode = int(seed / 4 % 4); // 0 free-cost weighted, 1 nonnegative unit-weight, 2 bound -2, 3 bound 2
		std::optional<int64> const min_cost =
			mode == 0 ? std::nullopt : std::optional<int64>(mode == 1 ? 0 : mode == 2 ? -2
																					  : 2);
		int64 const cost_lo = min_cost ? *min_cost : -4;
		bool const is_unit = mode == 1 && seed % 3 == 0;
		weighted_tree const tree = random_tree(rng, n, shape, cost_lo, cost_lo + 6, is_unit ? 1 : 0, is_unit ? 1 : 3);
		int64 const reach = int64(n) * 10 + 5;
		auto const result = solve(tree, min_cost);
		INFO("seed " << seed);
		brute_optimum const brute = brute_minimum(tree, min_cost, reach);
		REQUIRE(result.total_cost == brute.total_cost);
		// ties: the root total is the smallest minimizer, which exists when every weight is positive
		if (n > 1 && std::ranges::min(std::span(tree.weight).subspan(1)) > 0)
			REQUIRE(result.path_sum == brute.smallest_path_sum);
		check_assignment(tree, result, min_cost);
		auto const by_ordered_map = atl::experimental::balance_leaf_paths_by_ordered_map(tree.parent, tree.cost, tree.weight, min_cost);
		REQUIRE(by_ordered_map.total_cost == result.total_cost);
		REQUIRE(by_ordered_map.path_sum == result.path_sum); // the same tie rule
		check_assignment(tree, by_ordered_map, min_cost);
		auto const by_fewer_entries =
			atl::experimental::balance_leaf_paths_by_fewer_entries(tree.parent, tree.cost, tree.weight, min_cost);
		REQUIRE(by_fewer_entries.total_cost == result.total_cost);
		REQUIRE(by_fewer_entries.path_sum == result.path_sum);
		check_assignment(tree, by_fewer_entries, min_cost);
	}
}

TEST_CASE("balance_leaf_paths: large trees against the ordered-map contender", "[dp][leaf_path_balance]") {
	for (unsigned seed = 0; seed < 24; ++seed) {
		std::mt19937_64 rng(seed);
		std::size_t const n = 3000;
		bool const is_nonnegative = seed % 2 == 1;
		weighted_tree const tree =
			random_tree(rng, n, int(seed / 2 % 4), 1, 1'000'000, 1, is_nonnegative ? 1 : 1'000'000);
		std::optional<int64> const min_cost = is_nonnegative ? std::optional<int64>(0) : std::nullopt;
		auto const result = solve(tree, min_cost);
		auto const by_ordered_map = atl::experimental::balance_leaf_paths_by_ordered_map(tree.parent, tree.cost, tree.weight, min_cost);
		INFO("seed " << seed);
		REQUIRE(result.total_cost == by_ordered_map.total_cost);
		REQUIRE(result.path_sum == by_ordered_map.path_sum);
		check_assignment(tree, result, min_cost);
		check_assignment(tree, by_ordered_map, min_cost);
		auto const by_fewer_entries =
			atl::experimental::balance_leaf_paths_by_fewer_entries(tree.parent, tree.cost, tree.weight, min_cost);
		REQUIRE(result.total_cost == by_fewer_entries.total_cost);
		REQUIRE(result.path_sum == by_fewer_entries.path_sum);
		check_assignment(tree, by_fewer_entries, min_cost);
	}
}

TEST_CASE("balance_leaf_paths: inputs at the ends of int64", "[dp][leaf_path_balance]") {
	int64 const top = std::numeric_limits<int64>::max(), bottom = std::numeric_limits<int64>::min();
	SECTION("a zero-weight edge of cost INT64_MIN: x_parent - cost leaves int64") {
		weighted_tree const tree{{0, 0}, {0, bottom}, {0, 0}};
		for (auto const &result : solve_all(tree, std::nullopt)) {
			CHECK(result.total_cost == 0);
			CHECK(result.path_sum == 0);
			CHECK(result.new_cost == std::vector<int64>{0, 0});
			check_assignment(tree, result, std::nullopt);
		}
	}
	SECTION("a merge that rebases stored positions by two int64 ranges") {
		// the map contender swaps vertex 1's map, offset H, into the root and stores
		// the root's breakpoint at 1 - H relative to it: (1 - H) - H
		weighted_tree const tree{{0, 0, 0, 1, 1}, {0, top, 1 - top, -top, 1 - top}, {0, 2, 1, 1, 1}};
		for (auto const &result : solve_all(tree, std::nullopt)) {
			CHECK(result.total_cost == top);
			check_assignment(tree, result, std::nullopt);
		}
	}
}

TEST_CASE("balance_leaf_paths_by_fewer_entries on the bounded path the weight rule makes quadratic", "[dp][leaf_path_balance]") {
	// unit costs, new costs >= 0, weight[1] = 4n and weight[v] = n - v below: the light right
	// side splinters into pieces, which the example moves on every edge (about n^2 / 2 in
	// all), while the left side is the leaf's wall in a record or two
	auto const path = [](std::size_t n) {
		weighted_tree tree{std::vector<uint32>(n, 0), std::vector<int64>(n, 1), std::vector<int64>(n, 0)};
		for (std::size_t v = 1; v < n; ++v) {
			tree.parent[v] = uint32(v - 1);
			tree.weight[v] = v == 1 ? int64(4 * n) : int64(n - v);
		}
		tree.cost[0] = 0;
		return tree;
	};
	weighted_tree const small = path(2000);
	auto const example = solve(small, 0);
	auto const by_fewer_entries = atl::experimental::balance_leaf_paths_by_fewer_entries(small.parent, small.cost, small.weight, 0);
	REQUIRE(example.total_cost == 0); // one leaf: every cost kept
	REQUIRE(by_fewer_entries.total_cost == example.total_cost);
	REQUIRE(by_fewer_entries.path_sum == example.path_sum);
	check_assignment(small, by_fewer_entries, 0);
	// 1e5 vertices: about 5e9 moved pieces for the weight rule, past its record limit
	weighted_tree const large = path(100'000);
	auto const result = atl::experimental::balance_leaf_paths_by_fewer_entries(large.parent, large.cost, large.weight, 0);
	REQUIRE(result.total_cost == 0);
	REQUIRE(result.path_sum == int64(large.parent.size() - 1));
	check_assignment(large, result, 0);
}

TEST_CASE("balance_leaf_paths: edge cases", "[dp][leaf_path_balance]") {
	SECTION("a single vertex") {
		weighted_tree const tree{{0}, {7}, {3}};
		auto const result = solve(tree);
		REQUIRE(result.total_cost == 0);
		REQUIRE(result.path_sum == 0);
		REQUIRE(result.new_cost == std::vector<int64>{0});
	}
	SECTION("one edge keeps its cost") {
		weighted_tree const tree{{0, 0}, {0, -9}, {0, 4}};
		auto const result = solve(tree);
		REQUIRE(result.total_cost == 0);
		REQUIRE(result.path_sum == -9);
	}
	SECTION("all weights zero: any equal paths are free") {
		weighted_tree const tree{{0, 0, 1, 1}, {0, 3, 8, 2}, {0, 0, 0, 0}};
		auto const result = solve(tree);
		REQUIRE(result.total_cost == 0);
		check_assignment(tree, result, std::nullopt);
		auto const bounded = solve(tree, 0);
		REQUIRE(bounded.total_cost == 0);
		check_assignment(tree, bounded, 0);
	}
	SECTION("a path of 1e6 vertices and a star of 1e6 leaves: no recursion") {
		// free costs with weights up to 1e6, and the bounded run with unit weights (its
		// O(n log n) case: a bounded edge moves breakpoints of weight up to 2 weight[v])
		for (int const shape : {1, 2}) {
			for (bool const is_bounded : {false, true}) {
				std::mt19937_64 rng(11);
				weighted_tree const tree =
					random_tree(rng, 1'000'000, shape, 1, 1'000'000, 1, is_bounded ? 1 : 1'000'000);
				std::optional<int64> const min_cost = is_bounded ? std::optional<int64>(0) : std::nullopt;
				auto const result = solve(tree, min_cost);
				check_assignment(tree, result, min_cost);
				if (shape == 1) REQUIRE(result.total_cost == 0); // a single leaf: every cost kept
			}
		}
	}
	SECTION("a deep path with costs near 1e6 and weights 1e6") {
		std::mt19937_64 rng(7);
		weighted_tree tree = random_tree(rng, 200000, 1, 999'000, 1'000'000, 999'000, 1'000'000);
		tree.parent.push_back(0); // a short leaf next to the long path: total near 2e17
		tree.cost.push_back(0);
		tree.weight.push_back(1'000'000);
		auto const result = solve(tree);
		check_assignment(tree, result, std::nullopt);
		REQUIRE(result.total_cost > int64(1) << 50);
	}
}

TEST_CASE("balance_leaf_paths: preconditions", "[dp][leaf_path_balance]") {
	std::vector<uint32> const parent{0, 0, 1};
	std::vector<int64> const cost{0, 1, 2}, weight{0, 1, 1};
	CHECK(atl_test::aborts([&] { (void)atl::experimental::balance_leaf_paths(parent, cost, std::vector<int64>{0, 1}); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::balance_leaf_paths({}, {}, {}); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::balance_leaf_paths(std::vector<uint32>{0, 0, 5}, cost, weight); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::balance_leaf_paths(std::vector<uint32>{0, 2, 1}, cost, weight); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::balance_leaf_paths(parent, cost, std::vector<int64>{0, 1, -1}); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::balance_leaf_paths(parent, cost, weight, 2); }));
	CHECK(atl_test::aborts([&] {
		(void)atl::experimental::balance_leaf_paths(parent, cost, std::vector<int64>{0, int64(1) << 61, int64(1) << 61}, 0);
	}));
}
