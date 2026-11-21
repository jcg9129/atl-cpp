#include <atl/experimental/dp/tree_transport.hpp>
#include <atl/experimental/dp/tree_transport_variants.hpp>
#include <atl/graph/min_cost_flow.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
using atl::int64;
using atl::uint32;

namespace {
	/** @brief A rooted tree with per-edge costs and per-vertex supplies and demands, vertex 0 the root. */
	struct transport_instance {
		std::vector<uint32> parent;
		std::vector<int64> cost, supply, demand;
	};

	/**
	 * @brief The certifying oracle: min-cost max-flow from a source giving
	 *        `supply[v]` to every vertex, to a sink taking `demand[v]` from
	 *        it, over both directions of every tree edge; the flow must meet
	 *        every demand.
	 */
	int64 oracle(transport_instance const &instance) {
		std::size_t const n = instance.parent.size();
		g::vertex_id const source = g::vertex_id(n), sink = g::vertex_id(n + 1);
		int64 supply_sum = 0, demand_sum = 0;
		for (std::size_t v = 0; v < n; ++v) supply_sum += instance.supply[v], demand_sum += instance.demand[v];
		std::vector<g::flow_arc<int64, int64>> arcs;
		for (std::size_t v = 0; v < n; ++v) {
			if (instance.supply[v] > 0) arcs.push_back({source, g::vertex_id(v), instance.supply[v], 0});
			if (instance.demand[v] > 0) arcs.push_back({g::vertex_id(v), sink, instance.demand[v], 0});
			if (v == 0) continue;
			arcs.push_back({g::vertex_id(v), instance.parent[v], supply_sum, instance.cost[v]});
			arcs.push_back({instance.parent[v], g::vertex_id(v), supply_sum, instance.cost[v]});
		}
		auto const flow = g::min_cost_max_flow(n + 2, arcs, source, sink);
		REQUIRE(flow.value == demand_sum);
		return flow.cost;
	}

	/** @brief Every contender's answer, checked equal to the example's; returns it. */
	int64 solve(transport_instance const &instance) {
		auto const &[parent, cost, supply, demand] = instance;
		int64 const expected = atl::experimental::tree_transport_cost(parent, cost, supply, demand);
		REQUIRE(atl::experimental::tree_transport_cost_by_rank_leftist(parent, cost, supply, demand) == expected);
		REQUIRE(atl::experimental::tree_transport_cost_by_skew(parent, cost, supply, demand) == expected);
		REQUIRE(atl::experimental::tree_transport_cost_by_binary(parent, cost, supply, demand) == expected);
		REQUIRE(atl::experimental::tree_transport_cost_by_pop_and_push(parent, cost, supply, demand) == expected);
		REQUIRE(atl::experimental::tree_transport_cost_by_length_records(parent, cost, supply, demand) == expected);
		return expected;
	}

	/**
	 * @brief A random instance on `n` vertices, `parent[v] < v` shaped by
	 *        `shape` (0 random, 1 path, 2 star, 3 caterpillar), costs in
	 *        `[0, max_cost]`, supplies and demands in `[0, max_units]`, the
	 *        demands then lowered until the supply covers them.
	 */
	transport_instance random_instance(std::mt19937_64 &rng, std::size_t n, int shape, int64 max_cost,
									   int64 max_units) {
		transport_instance instance{std::vector<uint32>(n, 0), std::vector<int64>(n, 0), std::vector<int64>(n, 0),
									std::vector<int64>(n, 0)};
		int64 supply_sum = 0, demand_sum = 0;
		for (std::size_t v = 0; v < n; ++v) {
			if (v > 0) {
				if (shape == 0) instance.parent[v] = uint32(std::uniform_int_distribution<std::size_t>(0, v - 1)(rng));
				if (shape == 1) instance.parent[v] = uint32(v - 1);
				if (shape == 2) instance.parent[v] = 0;
				if (shape == 3) instance.parent[v] = uint32(v % 2 == 1 ? (v >= 2 ? v - 2 : 0) : v - 1);
				instance.cost[v] = std::uniform_int_distribution<int64>(0, max_cost)(rng);
			}
			instance.supply[v] = std::uniform_int_distribution<int64>(0, max_units)(rng);
			instance.demand[v] = std::uniform_int_distribution<int64>(0, max_units)(rng);
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

	/** @brief Relabel the vertices by a random permutation keeping 0 the root, so parents need not precede children. */
	transport_instance shuffled(std::mt19937_64 &rng, transport_instance const &instance) {
		std::size_t const n = instance.parent.size();
		std::vector<uint32> label(n);
		for (std::size_t v = 0; v < n; ++v) label[v] = uint32(v);
		std::ranges::shuffle(label.begin() + 1, label.end(), rng);
		transport_instance result{std::vector<uint32>(n, 0), std::vector<int64>(n, 0), std::vector<int64>(n, 0),
								  std::vector<int64>(n, 0)};
		for (std::size_t v = 0; v < n; ++v) {
			result.parent[label[v]] = label[instance.parent[v]];
			result.cost[label[v]] = instance.cost[v];
			result.supply[label[v]] = instance.supply[v];
			result.demand[label[v]] = instance.demand[v];
		}
		return result;
	}
}

TEST_CASE("tree_transport_cost: fixed supply and demand networks", "[dp][tree_transport]") {
	// edges 1-2 (5), 3-1 (5); supplies / demands 2 1, 5 0, 1 3; renumbered from 0
	transport_instance const first{{0, 0, 0}, {0, 5, 5}, {2, 5, 1}, {1, 0, 3}};
	REQUIRE(solve(first) == 15);
	REQUIRE(oracle(first) == 15);
	// edges 1-2 (2), 1-3 (5), 1-4 (1), 2-5 (5), 2-6 (1)
	transport_instance const second{{0, 0, 0, 0, 1, 1}, {0, 2, 5, 1, 5, 1}, {0, 1, 2, 2, 0, 0}, {0, 0, 1, 1, 1, 1}};
	REQUIRE(solve(second) == 9);
	REQUIRE(oracle(second) == 9);
}

TEST_CASE("tree_transport_cost: small trees against min-cost flow", "[dp][tree_transport]") {
	for (unsigned seed = 0; seed < 2000; ++seed) {
		std::mt19937_64 rng(seed);
		std::size_t const n = std::uniform_int_distribution<std::size_t>(1, 9)(rng);
		int64 const max_units = seed % 3 == 0 ? 1 : seed % 3 == 1 ? 4
																  : 30;
		transport_instance instance = random_instance(rng, n, int(seed % 4), seed % 5 == 0 ? 2 : 20, max_units);
		if (seed % 2 == 1) instance = shuffled(rng, instance);
		INFO("seed " << seed);
		REQUIRE(solve(instance) == oracle(instance));
	}
}

TEST_CASE("tree_transport_cost: medium trees against min-cost flow", "[dp][tree_transport]") {
	for (unsigned seed = 0; seed < 24; ++seed) {
		std::mt19937_64 rng(seed);
		transport_instance const instance =
			shuffled(rng, random_instance(rng, 400, int(seed % 4), 1'000'000, seed % 2 == 0 ? 8 : 1000));
		INFO("seed " << seed);
		REQUIRE(solve(instance) == oracle(instance));
	}
}

TEST_CASE("tree_transport_cost: large trees, engines agree", "[dp][tree_transport]") {
	for (unsigned seed = 0; seed < 8; ++seed) {
		std::mt19937_64 rng(seed);
		// the path drives skew heaps' right spines long: the iterative meld must hold
		transport_instance const instance = random_instance(rng, 50'000, int(seed % 4), 1'000'000, 8);
		INFO("seed " << seed);
		REQUIRE(solve(instance) >= 0);
	}
}

TEST_CASE("tree_transport_cost: edge cases", "[dp][tree_transport]") {
	SECTION("a single vertex") {
		transport_instance const instance{{0}, {9}, {5}, {3}};
		REQUIRE(solve(instance) == 0);
	}
	SECTION("no units") {
		transport_instance const instance{{0, 0, 1}, {0, 4, 4}, {0, 0, 0}, {0, 0, 0}};
		REQUIRE(solve(instance) == 0);
	}
	SECTION("free edges") {
		transport_instance const instance{{0, 0, 1, 1}, {0, 0, 0, 0}, {6, 0, 0, 0}, {0, 2, 2, 2}};
		REQUIRE(solve(instance) == 0);
	}
	SECTION("surplus left over stays put") {
		transport_instance const instance{{0, 0}, {0, 7}, {10, 0}, {0, 3}};
		REQUIRE(solve(instance) == 21);
	}
	SECTION("1e6 units across 2.5e5 edges of 1e6") {
		std::size_t const n = 250'000;
		transport_instance instance{std::vector<uint32>(n, 0), std::vector<int64>(n, 1'000'000),
									std::vector<int64>(n, 0), std::vector<int64>(n, 0)};
		for (std::size_t v = 1; v < n; ++v) instance.parent[v] = uint32(v - 1);
		instance.supply[0] = 1'000'000;
		instance.demand[n - 1] = 1'000'000;
		REQUIRE(solve(instance) == int64(1'000'000) * 1'000'000 * int64(n - 1));
	}
	SECTION("a path of 1e6 vertices and a star of 1e6 leaves: no recursion, one stack frame") {
		std::size_t const n = 1'000'000;
		for (bool const is_star : {false, true}) {
			transport_instance instance{std::vector<uint32>(n, 0), std::vector<int64>(n, 1), std::vector<int64>(n, 0),
										std::vector<int64>(n, 0)};
			for (std::size_t v = 1; v < n; ++v) instance.parent[v] = is_star ? 0 : uint32(v - 1);
			instance.supply[n - 1] = 3; // the far end (path) or one leaf (star) ships 3 units to the root
			instance.demand[0] = 3;
			int64 const expected = is_star ? 3 : 3 * int64(n - 1);
			REQUIRE(atl::experimental::tree_transport_cost(instance.parent, instance.cost, instance.supply, instance.demand)
					== expected);
			REQUIRE(atl::experimental::tree_transport_cost_by_skew(instance.parent, instance.cost, instance.supply,
																   instance.demand)
					== expected);
		}
	}
	SECTION("parents need not precede children") {
		// root 0 - 2 - 1, the demand at the far end
		transport_instance const instance{{0, 2, 0}, {0, 3, 4}, {5, 0, 0}, {0, 5, 0}};
		REQUIRE(solve(instance) == 35);
	}
}

TEST_CASE("tree_transport_cost: asserted preconditions", "[dp][tree_transport]") {
	std::vector<uint32> const parent{0, 0, 1};
	std::vector<int64> const cost{0, 1, 1}, supply{1, 1, 1}, demand{1, 1, 1};
	CHECK(atl_test::aborts([&] { (void)atl::experimental::tree_transport_cost({}, {}, {}, {}); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::tree_transport_cost(parent, std::vector<int64>{0, 1}, supply, demand); }));
	CHECK(atl_test::aborts(
		[&] { (void)atl::experimental::tree_transport_cost(std::vector<uint32>{0, 2, 1}, cost, supply, demand); })); // a cycle
	CHECK(atl_test::aborts([&] { (void)atl::experimental::tree_transport_cost(std::vector<uint32>{0, 0, 3}, cost, supply, demand); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::tree_transport_cost(parent, std::vector<int64>{0, -1, 1}, supply, demand); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::tree_transport_cost(parent, cost, std::vector<int64>{1, -1, 1}, demand); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::tree_transport_cost(parent, cost, supply, std::vector<int64>{1, 1, 2}); }));
	// costs summing past int64, and an answer past it
	int64 const half = std::numeric_limits<int64>::max() / 2;
	CHECK(atl_test::aborts(
		[&] { (void)atl::experimental::tree_transport_cost(parent, std::vector<int64>{0, half + 1, half + 1}, supply, demand); }));
	CHECK(atl_test::aborts([&] {
		(void)atl::experimental::tree_transport_cost(std::vector<uint32>{0, 0}, std::vector<int64>{0, half}, std::vector<int64>{3, 0},
													 std::vector<int64>{0, 3});
	}));
	// huge costs alone are fine: only the answer has to fit
	int64 const huge = std::numeric_limits<int64>::max() / 4;
	REQUIRE(atl::experimental::tree_transport_cost(parent, std::vector<int64>{0, huge, huge}, supply, demand) == 0);
}

namespace {
	/**
	 * @brief The library engine behind a `top` that returns a proxy converting
	 *        to `transport_segment const &` and nothing else: what the
	 *        concept admits, so the loop must read the top only through that
	 *        conversion.
	 */
	class proxy_top_forest {
		atl::experimental::transport_heap_forest_default heaps_;

	public:
		/** @brief A top that is not a segment, only convertible to one. */
		struct top_proxy {
			atl::experimental::transport_segment const *segment;

			operator atl::experimental::transport_segment const &() const { return *segment; } // NOLINT(google-explicit-constructor): the converting proxy under test
		};

		proxy_top_forest(std::size_t heaps, atl::experimental::transport_segment_order order, int64 zero)
			: heaps_(heaps, order, zero) {}

		void reserve(uint32 elements) { heaps_.reserve(elements); }

		void push(uint32 heap, atl::experimental::transport_segment segment) { heaps_.push(heap, segment); }

		[[nodiscard]] top_proxy top(uint32 heap) { return {&heaps_.top(heap)}; }

		atl::experimental::transport_segment pop(uint32 heap) { return heaps_.pop(heap); }

		[[nodiscard]] bool empty(uint32 heap) const { return heaps_.empty(heap); }

		void meld(uint32 into, uint32 from) { heaps_.meld(into, from); }

		void add_to_all(uint32 heap, int64 shift) { heaps_.add_to_all(heap, shift); }
	};
}

TEST_CASE("tree_transport_cost: keys and gains past int64", "[dp][tree_transport]") {
	// a unit moves from vertex 2 to its sibling 3 below an edge of cost K: after the match
	// at vertex 1 the root's keys are K and 2K + 1, whose sum leaves int64; the answer is 0
	for (int64 const k : {int64(4'000'000'000'000'000'000), std::numeric_limits<int64>::max() - 1}) {
		transport_instance const instance{{0, 0, 1, 1}, {0, k, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
		INFO("K = " << k);
		REQUIRE(solve(instance) == 0);
	}
	// the same with the unit crossing the expensive edge: 3 units at K / 4 each
	int64 const quarter = std::numeric_limits<int64>::max() / 4;
	transport_instance const across{{0, 0, 1}, {0, quarter, 0}, {3, 0, 0}, {0, 0, 3}};
	REQUIRE(solve(across) == 3 * quarter);
}

TEST_CASE("tree_transport_cost_on reads a proxy top through its conversion", "[dp][tree_transport]") {
	static_assert(atl::experimental::transport_heap_forest<proxy_top_forest>);
	std::mt19937_64 rng(5);
	for (int iter = 0; iter < 50; ++iter) {
		transport_instance const instance = random_instance(rng, 40, iter % 4, 20, 4);
		auto const &[parent, cost, supply, demand] = instance;
		REQUIRE(atl::experimental::tree_transport_cost_on<proxy_top_forest>(parent, cost, supply, demand)
				== atl::experimental::tree_transport_cost(parent, cost, supply, demand));
	}
}
