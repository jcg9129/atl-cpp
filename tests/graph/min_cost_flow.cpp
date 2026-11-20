#include <atl/experimental/graph/min_cost_flow_variants.hpp>
#include <atl/graph/max_flow.hpp>
#include <atl/graph/min_cost_flow.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <ranges>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	/** @brief Min-cost flow is doubly certifying: feasibility + maximum value (the
	 * max_flow cross-check) + the dual (non-negative reduced costs on every
	 * residual arc proves cost-optimality by LP duality). */
	void certify(std::size_t n, std::vector<g::flow_arc<int64, int64>> const &edges, g::vertex_id s,
				 g::vertex_id t, g::min_cost_flow_result<int64, int64> const &r,
				 int64 limit = std::numeric_limits<int64>::max()) {
		REQUIRE(r.edge_flow.size() == edges.size());

		std::vector<int64> net(n, 0);
		int64 cost = 0;
		for (std::size_t i = 0; i < edges.size(); ++i) {
			REQUIRE(r.edge_flow[i] >= 0);
			REQUIRE(r.edge_flow[i] <= edges[i].capacity);
			net[edges[i].from] -= r.edge_flow[i];
			net[edges[i].to] += r.edge_flow[i];
			cost += r.edge_flow[i] * edges[i].cost;
		}
		for (g::vertex_id v = 0; v < n; ++v)
			if (v != s && v != t) REQUIRE(net[v] == 0);
		REQUIRE(net[t] == r.value);
		REQUIRE(cost == r.cost);

		std::vector<g::flow_arc<int64>> plain;
		for (auto const &e : edges) plain.push_back({e.from, e.to, e.capacity});
		REQUIRE(r.value == std::min(limit, g::max_flow_value(n, plain, s, t)));

		auto const potential = g::min_cost_flow_potentials(n, edges, r.edge_flow);
		REQUIRE(potential.size() == n);
		for (std::size_t i = 0; i < edges.size(); ++i) {
			auto const reduced = edges[i].cost + potential[edges[i].from] - potential[edges[i].to];
			if (r.edge_flow[i] < edges[i].capacity) REQUIRE(reduced >= 0);
			if (r.edge_flow[i] > 0) REQUIRE(-reduced >= 0);
		}
	}
}

static_assert(g::cost_capacitated_edge<g::flow_arc<int, int>>);
static_assert(!g::cost_capacitated_edge<g::flow_arc<int>>);
static_assert(g::capacitated_edge<g::flow_arc<int, int>>);
static_assert(!g::bounded_capacitated_edge<g::flow_arc<int, int>>);
static_assert(g::bounded_capacitated_edge<g::bounded_flow_arc<int, int>>);

TEST_CASE("min_cost_flow: random networks are certified optimal", "[graph]") {
	auto seed = GENERATE(take(10, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + std::size_t(rng() % 10);
	std::size_t const m = std::size_t(rng() % 24);
	bool const negatives = rng() % 2 != 0;
	std::vector<g::flow_arc<int64, int64>> edges;
	for (std::size_t i = 0; i < m; ++i) {
		auto a = g::vertex_id(rng() % n), b = g::vertex_id(rng() % n);
		if (negatives && a == b) continue;
		if (negatives && a > b) std::swap(a, b); // index-ordered: no cycles at all
		edges.push_back({a, b, int64(rng() % 9), int64(rng() % 21) - (negatives ? 10 : 0)});
	}
	g::vertex_id const s = g::vertex_id(rng() % n);
	g::vertex_id t = g::vertex_id(rng() % n);
	while (t == s) t = g::vertex_id(rng() % n);

	auto const oracle = ex::min_cost_max_flow_by_spfa(n, edges, s, t);
	certify(n, edges, s, t, oracle);

	auto const library = g::min_cost_max_flow(n, edges, s, t);
	REQUIRE(library.value == oracle.value);
	REQUIRE(library.cost == oracle.cost);
	certify(n, edges, s, t, library);

	int64 const limit = int64(rng() % 6);
	auto const capped = ex::min_cost_max_flow_by_spfa(n, edges, s, t, limit);
	certify(n, edges, s, t, capped, limit);
	auto const library_capped = g::min_cost_max_flow(n, edges, s, t, limit);
	REQUIRE(library_capped.value == capped.value);
	REQUIRE(library_capped.cost == capped.cost);
	certify(n, edges, s, t, library_capped, limit);
}

TEST_CASE("min_cost_flow: the race contenders agree with the library simplex", "[graph]") {
	// Equal pivot rules must preserve exact flows; mixed sizes exercise deep reattachments.
	auto seed = GENERATE(take(40, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + std::size_t(rng() % (rng() % 4 == 0 ? 400 : 12));
	std::size_t const m = std::size_t(rng() % (4 * n + 10));
	std::vector<g::flow_arc<int64, int64>> edges;
	for (std::size_t i = 0; i < m; ++i) {
		auto a = g::vertex_id(rng() % n), b = g::vertex_id(rng() % n);
		if (a == b) continue;
		if (a > b) std::swap(a, b); // acyclic, so cost scaling's precondition holds with negative costs
		edges.push_back({a, b, int64(rng() % 20), int64(rng() % 61) - 20});
	}
	g::vertex_id const s = g::vertex_id(rng() % n);
	g::vertex_id t = g::vertex_id(rng() % n);
	while (t == s) t = g::vertex_id(rng() % n);
	auto const library = g::min_cost_max_flow(n, edges, s, t);

	auto const scaled = ex::min_cost_max_flow_by_cost_scaling_price_update(n, edges, s, t);
	REQUIRE(scaled.value == library.value);
	REQUIRE(scaled.cost == library.cost);
	certify(n, edges, s, t, scaled);

	std::vector<int64> b(n, 0);
	for (std::size_t k = 0; k < n; ++k) { // a feasible b-flow: route random units along edges
		if (edges.empty()) break;
		auto const &e = edges[std::size_t(rng() % edges.size())];
		int64 const units = e.capacity ? int64(rng() % uint64(e.capacity + 1)) : 0;
		b[e.from] += units;
		b[e.to] -= units;
	}
	auto with_reward = edges;
	with_reward.push_back({t, s, 1000, -100000});
	auto const agree = [&](std::vector<g::flow_arc<int64, int64>> const &net, std::vector<int64> const &supply) {
		auto const want = g::detail::network_simplex(n, net, supply);
		auto const got = ex::detail::network_simplex_by_adjacency(n, net, supply);
		REQUIRE(want.has_value() == got.has_value());
		if (want) REQUIRE(*want == *got);
	};
	agree(edges, b);
	agree(with_reward, std::vector<int64>(n, 0));
}

TEST_CASE("min_cost_flow: canonical networks", "[graph]") {
	std::vector<g::flow_arc<int64, int64>> par{{0, 1, 2, 1}, {0, 1, 3, 5}};
	auto const both = ex::min_cost_max_flow_by_spfa(std::size_t(2), par, 0, 1);
	REQUIRE(both.value == 5);
	REQUIRE(both.cost == 2 * 1 + 3 * 5);
	auto const two = ex::min_cost_max_flow_by_spfa(std::size_t(2), par, 0, 1, int64(2));
	REQUIRE(two.value == 2);
	REQUIRE(two.cost == 2);

	std::vector<g::flow_arc<int64, int64>> dia{
		{0, 1, 5, 1}, {0, 2, 5, 10}, {1, 3, 5, 1}, {2, 3, 5, 10}};
	auto const cheap = ex::min_cost_max_flow_by_spfa(std::size_t(4), dia, 0, 3);
	REQUIRE(cheap.value == 10);
	REQUIRE(cheap.cost == 5 * 2 + 5 * 20);
	certify(4, dia, 0, 3, cheap);

	std::vector<g::flow_arc<int64, int64>> neg{{0, 1, 4, -3}, {1, 2, 4, 1}};
	auto const bonus = ex::min_cost_max_flow_by_spfa(std::size_t(3), neg, 0, 2);
	REQUIRE(bonus.value == 4);
	REQUIRE(bonus.cost == 4 * (-3) + 4 * 1);
	certify(3, neg, 0, 2, bonus);

	auto const none =
		ex::min_cost_max_flow_by_spfa(std::size_t(3), std::vector<g::flow_arc<int64, int64>>{{0, 1, 4, 2}}, 0, 2);
	REQUIRE(none.value == 0);
	REQUIRE(none.cost == 0);
}

TEST_CASE("min_cost_flow: negative cycles are saturated", "[graph]") {
	// 1 -> 2 -> 1 is a negative cycle (total -3) off the s-t path; the
	// optimum must run 4 units around it even though s-t flow ignores it
	std::vector<g::flow_arc<int64, int64>> edges{
		{0, 1, 2, 1}, {1, 2, 4, -5}, {2, 1, 4, 2}, {1, 3, 2, 1}};
	auto const r = g::min_cost_max_flow(std::size_t(4), edges, 0, 3);
	REQUIRE(r.value == 2);
	REQUIRE(r.edge_flow == std::vector<int64>{2, 4, 4, 2});
	REQUIRE(r.cost == 2 * 1 + 4 * (-5) + 4 * 2 + 2 * 1);
	auto const potential = g::min_cost_flow_potentials(std::size_t(4), edges, r.edge_flow);
	for (std::size_t i = 0; i < edges.size(); ++i) {
		auto const rc = edges[i].cost + potential[edges[i].from] - potential[edges[i].to];
		if (r.edge_flow[i] < edges[i].capacity) REQUIRE(rc >= 0);
		if (r.edge_flow[i] > 0) REQUIRE(rc <= 0);
	}
}

namespace {
	/** @brief b-flow optimality is certified without an oracle: bounds + exact
	 * divergence prove feasibility, and the dual (rc >= 0 off the upper
	 * bound, rc <= 0 off the lower) proves optimality by LP duality. */
	void certify_b_flow(std::size_t n, std::vector<g::bounded_flow_arc<int64, int64>> const &edges,
						std::vector<int64> const &b,
						g::min_cost_flow_result<int64, int64> const &r) {
		REQUIRE(r.edge_flow.size() == edges.size());
		std::vector<int64> divergence(n, 0);
		int64 cost = 0;
		for (std::size_t i = 0; i < edges.size(); ++i) {
			REQUIRE(r.edge_flow[i] >= edges[i].lower);
			REQUIRE(r.edge_flow[i] <= edges[i].capacity);
			divergence[edges[i].from] += r.edge_flow[i];
			divergence[edges[i].to] -= r.edge_flow[i];
			cost += r.edge_flow[i] * edges[i].cost;
		}
		for (g::vertex_id v = 0; v < n; ++v) REQUIRE(divergence[v] == b[v]);
		REQUIRE(cost == r.cost);
		auto const potential = g::min_cost_flow_potentials(n, edges, r.edge_flow);
		for (std::size_t i = 0; i < edges.size(); ++i) {
			auto const rc = edges[i].cost + potential[edges[i].from] - potential[edges[i].to];
			if (r.edge_flow[i] < edges[i].capacity) REQUIRE(rc >= 0);
			if (r.edge_flow[i] > edges[i].lower) REQUIRE(rc <= 0);
		}
	}
}

static_assert(g::cost_capacitated_edge<g::bounded_flow_arc<int64, int64>>);
static_assert(g::bounded_capacitated_edge<g::bounded_flow_arc<int64, int64>>);

TEST_CASE("min_cost_feasible_flow: witnessed instances are certified optimal", "[graph]") {
	auto seed = GENERATE(take(10, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + std::size_t(rng() % 10);
	std::size_t const m = std::size_t(rng() % 24);
	// build a witness flow first, then derive bounds and supplies from it:
	// feasibility is guaranteed by construction
	std::vector<g::bounded_flow_arc<int64, int64>> edges;
	std::vector<int64> b(n, 0);
	for (std::size_t i = 0; i < m; ++i) {
		auto const from = g::vertex_id(rng() % n), to = g::vertex_id(rng() % n);
		int64 const witness = int64(rng() % 7);
		int64 const lower = witness <= 0 ? 0 : int64(rng() % (witness + 1));
		int64 const capacity = witness + int64(rng() % 7);
		edges.push_back({from, to, lower, capacity, int64(rng() % 21) - 10});
		b[from] += witness;
		b[to] -= witness;
	}
	auto const r = g::min_cost_feasible_flow(n, edges, b);
	REQUIRE(r.has_value());
	certify_b_flow(n, edges, b, *r);
}

TEST_CASE("min_cost_feasible_flow: canonical and infeasible", "[graph]") {
	// forced expensive path: supply 3 at 0, demand 3 at 2, lower bound 2
	// forces the detour
	std::vector<g::bounded_flow_arc<int64, int64>> chain{
		{0, 2, 0, 3, 1}, {0, 1, 2, 5, 3}, {1, 2, 0, 5, 3}};
	auto const forced = g::min_cost_feasible_flow(std::size_t(3), chain, std::vector<int64>{3, 0, -3});
	REQUIRE(forced.has_value());
	REQUIRE(forced->edge_flow == std::vector<int64>{1, 2, 2});
	REQUIRE(forced->cost == 1 + 2 * 3 + 2 * 3);
	REQUIRE(forced->value == 3);
	certify_b_flow(3, chain, {3, 0, -3}, *forced);

	// zero b with a profitable (negative) cycle: the optimum is not the zero flow
	std::vector<g::bounded_flow_arc<int64, int64>> loop{{0, 1, 0, 4, -2}, {1, 0, 0, 4, 1}};
	auto const cycle = g::min_cost_feasible_flow(std::size_t(2), loop, std::vector<int64>{0, 0});
	REQUIRE(cycle.has_value());
	REQUIRE(cycle->edge_flow == std::vector<int64>{4, 4});
	REQUIRE(cycle->cost == -4);
	certify_b_flow(2, loop, {0, 0}, *cycle);

	REQUIRE(!g::min_cost_feasible_flow(std::size_t(2), loop, std::vector<int64>{1, 0}).has_value());
	std::vector<g::bounded_flow_arc<int64, int64>> dry{{0, 1, 0, 5, 1}};
	REQUIRE(!g::min_cost_feasible_flow(std::size_t(3), dry, std::vector<int64>{1, 0, -1}).has_value());
	std::vector<g::bounded_flow_arc<int64, int64>> stuck{{0, 1, 3, 5, 1}};
	REQUIRE(!g::min_cost_feasible_flow(std::size_t(2), stuck, std::vector<int64>{1, -1}).has_value());
	auto const empty = g::min_cost_feasible_flow(
		std::size_t(2), std::vector<g::bounded_flow_arc<int64, int64>>{}, std::vector<int64>{0, 0});
	REQUIRE(empty.has_value());
	REQUIRE(empty->cost == 0);
}

TEST_CASE("min_cost_flow: several unbounded source edges do not overflow", "[graph]") {
	long long const unbounded = 1LL << 62;
	std::vector<g::flow_arc<long long, long long>> const edges{
		{0, 1, unbounded, 1}, {0, 2, unbounded, 2}, {1, 3, 7, 1}, {2, 3, 8, 1}};
	auto const r = g::min_cost_max_flow(4, edges, 0, 3);
	REQUIRE(r.value == 15);
	REQUIRE(r.cost == 7 * 2 + 8 * 3);
	REQUIRE(r.edge_flow == std::vector<long long>{7, 8, 7, 8});
	auto const limited = g::min_cost_max_flow(4, edges, 0, 3, 10LL);
	REQUIRE(limited.value == 10);
	REQUIRE(limited.cost == 7 * 2 + 3 * 3);
}

TEST_CASE("min-cost flow: bad input is asserted at entry", "[graph]") {
	std::vector<g::flow_arc<long long, long long>> const out_of_range{{0, 1, 4, 1}, {1, 9, 4, 1}};
	std::vector<g::flow_arc<long long, long long>> const negative{{0, 1, 4, 1}, {1, 2, -1, 1}};
	REQUIRE(atl_test::aborts([&] { (void)g::min_cost_max_flow(3, out_of_range, 0, 2); }));
	REQUIRE(atl_test::aborts([&] { (void)g::min_cost_max_flow(3, negative, 0, 2); }));
	REQUIRE(atl_test::aborts([&] { (void)g::min_cost_max_flow(3, std::vector<g::flow_arc<long long, long long>>{{0, 2, 4, 1}}, 0, 2, -1LL); }));
	std::vector<long long> const b{1, 0, -1};
	REQUIRE(atl_test::aborts([&] { (void)g::min_cost_feasible_flow(3, out_of_range, b); }));
	REQUIRE(atl_test::aborts([&] { (void)g::min_cost_feasible_flow(3, negative, b); }));
}

struct renamed_cost_arc {
	uint32 tail, head;
	int64 cap, price;
};

template<>
struct atl::graph::graph_traits<renamed_cost_arc> {
	static vertex_id source(renamed_cost_arc const &e) { return e.tail; }

	static vertex_id target(renamed_cost_arc const &e) { return e.head; }

	static int64 capacity(renamed_cost_arc const &e) { return e.cap; }

	static int64 cost(renamed_cost_arc const &e) { return e.price; }
};

static_assert(g::cost_capacitated_edge_list<std::vector<renamed_cost_arc>>);

TEST_CASE("min_cost_flow: the scaling variants read endpoints through graph_traits", "[graph]") {
	std::vector<renamed_cost_arc> const edges{{0, 1, 5, 1}, {0, 2, 4, 3}, {1, 2, 2, -1}, {1, 3, 3, 4}, {2, 3, 6, 2}, {0, 3, 1, 9}};
	std::vector<g::flow_arc<int64, int64>> plain;
	for (auto const &e : edges) plain.push_back({e.tail, e.head, e.cap, e.price});
	auto const want = g::min_cost_max_flow(std::size_t(4), plain, 0, 3);
	auto const scaled = ex::min_cost_max_flow_by_capacity_scaling(std::size_t(4), edges, 0, 3);
	auto const cost_scaled = ex::min_cost_max_flow_by_cost_scaling(std::size_t(4), edges, 0, 3);
	REQUIRE(scaled.value == want.value);
	REQUIRE(scaled.cost == want.cost);
	REQUIRE(cost_scaled.value == want.value);
	REQUIRE(cost_scaled.cost == want.cost);
}

TEST_CASE("min_cost_flow: the SPFA oracle and zkw assert on a negative cycle", "[graph]") {
	std::vector<g::flow_arc<int64, int64>> const cycle{{0, 1, 1, -1}, {1, 0, 1, -1}};
	CHECK(atl_test::aborts([&] { (void)ex::min_cost_max_flow_by_spfa(std::size_t(2), cycle, 0, 1); }));
	CHECK(atl_test::aborts([&] { (void)ex::min_cost_max_flow_by_zkw(std::size_t(2), cycle, 0, 1); }));
	CHECK(atl_test::aborts([&] { (void)ex::min_cost_max_flow_by_dijkstra(std::size_t(2), cycle, 0, 1); }));
}

TEST_CASE("min_cost_flow: costs near the documented n^2 max|cost| bound stay exact", "[graph]") {
	std::mt19937_64 rng(212);
	std::size_t const n = 50, m = 600;
	std::vector<g::flow_arc<int64, int64>> edges;
	for (std::size_t i = 0; i < m; ++i) {
		auto const a = g::vertex_id(rng() % n), b = g::vertex_id(rng() % n);
		edges.push_back({a, b, int64(1 + rng() % 3), int64(90'000'000'000'000) + int64(rng() % 10'000'000'000'000)});
	}
	auto const r = g::min_cost_max_flow(n, edges, 0, 1);
	certify(n, edges, 0, 1, r);
	auto const oracle = ex::min_cost_max_flow_by_spfa(n, edges, 0, 1);
	REQUIRE(r.value == oracle.value);
	REQUIRE(r.cost == oracle.cost);
}

TEST_CASE("min_cost_flow: a cost past the bound is asserted, not wrapped", "[graph]") {
	std::vector<g::flow_arc<int64, int64>> const edges{{0, 1, 1, int64(1) << 61}, {1, 2, 1, 1}};
	CHECK(atl_test::aborts([&] { (void)g::min_cost_max_flow(std::size_t(3), edges, 0, 2); }));
	CHECK(atl_test::aborts(
		[&] { (void)g::min_cost_feasible_flow(std::size_t(3), edges, std::vector<int64>{1, 0, -1}); }));
	std::vector<g::flow_arc<int64, int64>> const fitting{{0, 1, 1, int64(1) << 40}, {1, 2, 1, 1}};
	CHECK(g::min_cost_max_flow(std::size_t(3), fitting, 0, 2).value == 1);
}

namespace {
	template<typename Cost>
	concept min_cost_accepts = requires(std::vector<g::flow_arc<int, int>> const &edges) {
		g::min_cost_max_flow<void, Cost>(std::size_t(2), edges, 0u, 1u);
	};

	template<typename Capacity>
	concept min_cost_accepts_capacity = requires(std::vector<g::flow_arc<Capacity, int>> const &edges) {
		g::min_cost_max_flow(std::size_t(2), edges, 0u, 1u);
	};

	template<typename Flow>
	concept feasible_min_cost_accepts = requires(std::vector<g::flow_arc<int, int>> const &edges) {
		g::min_cost_feasible_flow<Flow>(std::size_t(2), edges, std::vector<Flow>(2));
	};
}

static_assert(min_cost_accepts<int64> && min_cost_accepts<double> && !min_cost_accepts<unsigned>);
static_assert(feasible_min_cost_accepts<int64> && !feasible_min_cost_accepts<uint32>);
static_assert(!min_cost_accepts_capacity<uint32> && min_cost_accepts_capacity<int>);

TEST_CASE("min-cost feasibility accepts supplies through the signed capacity bound", "[graph]") {
	int64 const amount = std::numeric_limits<int64>::max();
	std::vector<g::flow_arc<int64, int64>> const edges{{0, 1, amount, 0}};
	auto const r = g::min_cost_feasible_flow(2, edges, std::vector<int64>{amount, -amount});
	REQUIRE(r.has_value());
	REQUIRE(r->value == amount);
	REQUIRE(r->edge_flow[0] == amount);
	REQUIRE(r->cost == 0);
}

TEST_CASE("zkw uses heap frames on a million-vertex path", "[graph]") {
	uint32 const count = 1'000'000;
	std::vector<g::flow_arc<int64, int64>> edges;
	edges.reserve(count - 1);
	for (uint32 v = 1; v < count; ++v) edges.push_back({v - 1, v, 1, 1});
	auto const r = ex::min_cost_max_flow_by_zkw(count, edges, 0, count - 1);
	REQUIRE(r.value == 1);
	REQUIRE(r.cost == count - 1);
	REQUIRE(std::ranges::all_of(r.edge_flow, [](int64 value) { return value == 1; }));
}

TEST_CASE("min-cost flow rejects objective and supply overflow without wrapping", "[graph]") {
	int64 const cap = std::numeric_limits<int64>::max();
	REQUIRE(atl_test::aborts([&] {
		(void)g::min_cost_max_flow(2, std::vector<g::flow_arc<int64, int64>>{{0, 1, cap, 2}}, 0, 1);
	}));
	REQUIRE(atl_test::aborts([&] {
		(void)g::min_cost_feasible_flow(3, std::vector<g::flow_arc<int64, int64>>{}, std::vector<int64>{cap, 1, -cap});
	}));
	atl::int128 const wide = atl::int128(1) << 80;
	std::vector<g::flow_arc<int, atl::int128>> const edges{{0, 1, 2, wide}, {1, 2, 2, -wide / 2}};
	auto const r = g::min_cost_max_flow(3, edges, 0, 2);
	REQUIRE(r.cost == wide);
	REQUIRE(g::min_cost_flow_potentials(3, edges, r.edge_flow).size() == 3);
}

TEST_CASE("cost certificates honor negative lower bounds with representable residuals", "[graph]") {
	std::vector<g::bounded_flow_arc<int64, int64>> const edges{{0, 1, -2, 3, 1}};
	auto const p = g::min_cost_flow_potentials(2, edges, std::vector<int64>{-2});
	REQUIRE(1 + p[0] - p[1] >= 0);
	std::vector<g::bounded_flow_arc<int64, int64>> const overflowing{{0, 1, -1, std::numeric_limits<int64>::max(), 0}};
	REQUIRE(atl_test::aborts([&] { (void)g::min_cost_flow_potentials(2, overflowing, std::vector<int64>{-1}); }));
	REQUIRE(atl_test::aborts([&] { (void)g::min_cost_feasible_flow(2, edges, std::vector<int64>{-2, 2}); }));
}

TEST_CASE("simplex costs at the checked arithmetic threshold remain exact", "[graph]") {
	int64 const feasible_cost = std::numeric_limits<int64>::max() / 5 / 3 - 1;
	std::vector<g::flow_arc<int, int64>> const feasible{{0, 1, 1, feasible_cost}};
	REQUIRE(g::min_cost_feasible_flow(2, feasible, std::vector<int>{1, -1})->cost == feasible_cost);
	int64 const maximum_cost = (feasible_cost - 1) / 2;
	std::vector<g::flow_arc<int, int64>> const maximum{{0, 1, 1, maximum_cost}};
	REQUIRE(g::min_cost_max_flow(2, maximum, 0, 1).cost == maximum_cost);
	REQUIRE(atl_test::aborts([&] {
		(void)g::min_cost_feasible_flow(2, std::vector<g::flow_arc<int, int64>>{{0, 1, 1, feasible_cost + 1}}, std::vector<int>{1, -1});
	}));
	REQUIRE(atl_test::aborts([&] {
		(void)g::min_cost_max_flow(2, std::vector<g::flow_arc<int, int64>>{{0, 1, 1, maximum_cost + 1}}, 0, 1);
	}));
}

TEST_CASE("min-cost flow represents the finite floating capacity bound exactly", "[graph]") {
	double const capacity = std::numeric_limits<double>::max();
	std::vector<g::flow_arc<double, double>> const edges{{0, 1, capacity, 0}};
	auto const r = g::min_cost_max_flow(2, edges, 0, 1);
	REQUIRE(r.value == capacity);
	REQUIRE(r.edge_flow[0] == capacity);
	REQUIRE(r.cost == 0);
	REQUIRE(g::min_cost_flow_potentials(2, edges, r.edge_flow) == std::vector<double>{0, 0});
}

TEST_CASE("cost scaling checks the absolute scaled cost", "[graph]") {
	std::vector<g::flow_arc<int, int64>> const edges{{0, 1, 1, std::numeric_limits<int64>::lowest() / 4}};
	REQUIRE(atl_test::aborts([&] { (void)ex::min_cost_max_flow_by_cost_scaling(3, edges, 0, 2); }));
}
