#include <atl/experimental/graph/max_flow_variants.hpp>
#include <atl/graph/concepts.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <random>
#include <ranges>
#include <type_traits>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::int64;
using atl::uint64;

namespace {
	/** @brief Feasibility and cut equality certify optimality by weak duality. */
	template<g::signed_flow_value Flow>
	void certify(std::size_t n, std::vector<g::flow_arc<Flow>> const &edges, g::vertex_id s,
				 g::vertex_id t, g::max_flow_result<Flow> const &r) {
		REQUIRE(r.edge_flow.size() == edges.size());
		REQUIRE(r.source_side.size() == n);

		std::vector<Flow> net(n, 0);
		for (std::size_t i = 0; i < edges.size(); ++i) {
			REQUIRE(r.edge_flow[i] >= 0);
			REQUIRE(r.edge_flow[i] <= edges[i].capacity);
			net[edges[i].from] -= r.edge_flow[i];
			net[edges[i].to] += r.edge_flow[i];
		}
		for (g::vertex_id v = 0; v < n; ++v)
			if (v != s && v != t) REQUIRE(net[v] == 0);
		REQUIRE(net[s] == -r.value);
		REQUIRE(net[t] == r.value);

		// a valid cut with every forward edge saturated and every backward
		// edge empty has capacity == value: optimal
		REQUIRE(r.source_side[s] == 1);
		REQUIRE(r.source_side[t] == 0);
		for (std::size_t i = 0; i < edges.size(); ++i) {
			if (r.source_side[edges[i].from] && !r.source_side[edges[i].to])
				REQUIRE(r.edge_flow[i] == edges[i].capacity);
			if (!r.source_side[edges[i].from] && r.source_side[edges[i].to]) REQUIRE(r.edge_flow[i] == 0);
		}
	}
}

static_assert(g::capacitated_edge<g::flow_arc<int>>);
static_assert(!g::bounded_capacitated_edge<g::flow_arc<int>>);

namespace {
	struct bounded_arc {
		g::vertex_id from, to;
		int lower, capacity;
	};
}

static_assert(g::bounded_capacitated_edge<bounded_arc>);

namespace {
	template<typename A>
	concept accepted_by_max_flow = requires(std::vector<A> b) {
		g::max_flow(std::size_t(4), b, g::vertex_id(0), g::vertex_id(1));
	};

	template<typename A>
	concept accepted_by_scaling = requires(std::vector<A> b) {
		ex::max_flow_by_capacity_scaling_dinic(std::size_t(4), b, g::vertex_id(0), g::vertex_id(1));
		ex::max_flow_by_capacity_scaling_edmonds_karp(std::size_t(4), b, g::vertex_id(0), g::vertex_id(1));
	};
}

static_assert(accepted_by_max_flow<g::flow_arc<int>>);
static_assert(!accepted_by_max_flow<bounded_arc>);
static_assert(accepted_by_max_flow<g::flow_arc<double>>);
static_assert(accepted_by_scaling<g::flow_arc<int>>);
static_assert(!accepted_by_scaling<g::flow_arc<double>>);

TEST_CASE("max_flow: random networks are certified optimal", "[graph]") {
	auto seed = GENERATE(take(12, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + std::size_t(rng() % 12);
	std::size_t const m = std::size_t(rng() % 30);
	std::vector<g::flow_arc<int64>> edges;
	for (std::size_t i = 0; i < m; ++i)
		edges.push_back({g::vertex_id(rng() % n), g::vertex_id(rng() % n), int64(rng() % 11)});
	g::vertex_id const s = g::vertex_id(rng() % n);
	g::vertex_id t = g::vertex_id(rng() % n);
	while (t == s) t = g::vertex_id(rng() % n);

	auto const ek = ex::max_flow_by_edmonds_karp(n, edges, s, t);
	certify(n, edges, s, t, ek);

	auto const di = ex::max_flow_by_dinic(n, edges, s, t);
	REQUIRE(di.value == ek.value);
	certify(n, edges, s, t, di);

	auto const is = ex::max_flow_by_isap(n, edges, s, t);
	REQUIRE(is.value == ek.value);
	certify(n, edges, s, t, is);

	auto const hl = g::max_flow(n, edges, s, t);
	REQUIRE(hl.value == ek.value);
	certify(n, edges, s, t, hl);

	auto const sek = ex::max_flow_by_capacity_scaling_edmonds_karp(n, edges, s, t);
	REQUIRE(sek.value == ek.value);
	certify(n, edges, s, t, sek);

	auto const sdi = ex::max_flow_by_capacity_scaling_dinic(n, edges, s, t);
	REQUIRE(sdi.value == ek.value);
	certify(n, edges, s, t, sdi);

	REQUIRE(g::max_flow_value(n, edges, s, t) == ek.value);
}

TEST_CASE("max_flow: mid-size networks with wide capacities are certified for every contender", "[graph]") {
	// Wide capacities exercise up to 51 threshold levels.
	auto seed = GENERATE(take(6, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 200 + std::size_t(rng() % 1800);
	std::size_t const m = 2 * n + std::size_t(rng() % (4 * n));
	// the preflow puts the whole source out-capacity on one excess, so the
	// widest rung stays sufficiently below 2^63 for any source degree here
	int64 const cap = rng() % 2 ? int64(1'000'000'000) : int64(1) << 50;
	std::vector<g::flow_arc<int64>> edges;
	for (std::size_t i = 0; i < m; ++i)
		edges.push_back({g::vertex_id(rng() % n), g::vertex_id(rng() % n), int64(rng() % uint64(cap + 1))});
	g::vertex_id const s = g::vertex_id(rng() % n);
	g::vertex_id t = g::vertex_id(rng() % n);
	while (t == s) t = g::vertex_id(rng() % n);

	auto const hl = g::max_flow(n, edges, s, t);
	certify(n, edges, s, t, hl);
	auto const check = [&](g::max_flow_result<int64> const &r) {
		REQUIRE(r.value == hl.value);
		certify(n, edges, s, t, r);
	};
	check(ex::max_flow_by_edmonds_karp(n, edges, s, t));
	check(ex::max_flow_by_dinic(n, edges, s, t));
	check(ex::max_flow_by_isap(n, edges, s, t));
	check(ex::max_flow_by_capacity_scaling_edmonds_karp(n, edges, s, t));
	check(ex::max_flow_by_capacity_scaling_dinic(n, edges, s, t));
	REQUIRE(g::max_flow_value(n, edges, s, t) == hl.value);
}

TEST_CASE("max_flow: dyadic floating-point capacities are exact", "[graph]") {
	// These small eighth-integer capacities and their accumulated values fit
	// the significand, so the integer twin gives an exact certificate.
	auto seed = GENERATE(take(4, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + std::size_t(rng() % 10);
	std::size_t const m = std::size_t(rng() % 25);
	std::vector<g::flow_arc<double>> edges;
	std::vector<g::flow_arc<int64>> scaled;
	for (std::size_t i = 0; i < m; ++i) {
		auto const a = g::vertex_id(rng() % n), b = g::vertex_id(rng() % n);
		int64 const eighths = int64(rng() % 64);
		edges.push_back({a, b, double(eighths) / 8.0});
		scaled.push_back({a, b, eighths});
	}
	g::vertex_id const s = g::vertex_id(rng() % n);
	g::vertex_id t = g::vertex_id(rng() % n);
	while (t == s) t = g::vertex_id(rng() % n);

	double const want = double(g::max_flow(n, scaled, s, t).value) / 8.0;
	// The integer twin certifies every non-scaling contender exactly.
	auto const check = [&](g::max_flow_result<double> const &r) {
		REQUIRE(r.value == want);
		certify(n, edges, s, t, r);
	};
	check(g::max_flow(n, edges, s, t));
	check(ex::max_flow_by_edmonds_karp(n, edges, s, t));
	check(ex::max_flow_by_dinic(n, edges, s, t));
	check(ex::max_flow_by_isap(n, edges, s, t));
}

TEST_CASE("max_flow: canonical networks", "[graph]") {
	auto const run = []<typename Solve>(Solve const &solve)
		requires std::is_invocable_r_v<g::max_flow_result<int>, Solve const &, std::size_t,
									   std::vector<g::flow_arc<int>> const &, g::vertex_id, g::vertex_id>
	{
		std::vector<g::flow_arc<int>> path{{0, 1, 5}, {1, 2, 3}, {2, 3, 7}};
		auto const chain = solve(std::size_t(4), path, 0, 3);
		REQUIRE(chain.value == 3);
		REQUIRE(chain.edge_flow == std::vector<int>{3, 3, 3});
		certify(4, path, 0, 3, chain);

		std::vector<g::flow_arc<int>> par{{0, 1, 2}, {0, 1, 3}};
		auto const both = solve(std::size_t(2), par, 0, 1);
		REQUIRE(both.value == 5);
		REQUIRE(both.edge_flow == std::vector<int>{2, 3});

		std::vector<g::flow_arc<int>> dia{{0, 1, 10}, {0, 2, 10}, {1, 2, 1}, {1, 3, 10}, {2, 3, 10}};
		auto const cross = solve(std::size_t(4), dia, 0, 3);
		REQUIRE(cross.value == 20);
		certify(4, dia, 0, 3, cross);

		std::vector<g::flow_arc<int>> split{{0, 1, 4}, {2, 3, 4}};
		auto const cut_off = solve(std::size_t(4), split, 0, 3);
		REQUIRE(cut_off.value == 0);
		REQUIRE(cut_off.edge_flow == std::vector<int>{0, 0});
		REQUIRE(cut_off.source_side == std::vector<atl::uint8>{1, 1, 0, 0});

		std::vector<g::flow_arc<int>> loops{{0, 0, 9}, {0, 1, 2}, {1, 0, 5}};
		auto const looped = solve(std::size_t(2), loops, 0, 1);
		REQUIRE(looped.value == 2);
		REQUIRE(looped.edge_flow[0] == 0);
		certify(2, loops, 0, 1, looped);

		std::vector<g::flow_arc<int>> bip{{0, 1, 1}, {0, 2, 1}, {0, 3, 1}, {1, 4, 1}, {1, 5, 1}, {2, 4, 1}, {3, 6, 1}, {4, 7, 1}, {5, 7, 1}, {6, 7, 1}};
		auto const match = solve(std::size_t(8), bip, 0, 7);
		REQUIRE(match.value == 3);
		certify(8, bip, 0, 7, match);
	};
	run([](std::size_t n, std::vector<g::flow_arc<int>> const &e, g::vertex_id s, g::vertex_id t) { return g::max_flow(n, e, s, t); });
	run([](std::size_t n, std::vector<g::flow_arc<int>> const &e, g::vertex_id s, g::vertex_id t) { return ex::max_flow_by_edmonds_karp(n, e, s, t); });
	run([](std::size_t n, std::vector<g::flow_arc<int>> const &e, g::vertex_id s, g::vertex_id t) { return ex::max_flow_by_dinic(n, e, s, t); });
	run([](std::size_t n, std::vector<g::flow_arc<int>> const &e, g::vertex_id s, g::vertex_id t) { return ex::max_flow_by_isap(n, e, s, t); });
	run([](std::size_t n, std::vector<g::flow_arc<int>> const &e, g::vertex_id s, g::vertex_id t) { return ex::max_flow_by_capacity_scaling_edmonds_karp(n, e, s, t); });
	run([](std::size_t n, std::vector<g::flow_arc<int>> const &e, g::vertex_id s, g::vertex_id t) { return ex::max_flow_by_capacity_scaling_dinic(n, e, s, t); });
}

namespace {
	template<typename Capacity>
	concept feasible_flow_accepts = requires(std::vector<g::bounded_flow_arc<Capacity>> const &edges,
											 std::vector<Capacity> const &supply) { g::feasible_flow(2, edges, supply); };
}

TEST_CASE("max_flow: capacity sums that cannot be represented are rejected", "[graph]") {
	static_assert(feasible_flow_accepts<int>);
	static_assert(feasible_flow_accepts<long long>);
	static_assert(!feasible_flow_accepts<unsigned>);
	static_assert(!feasible_flow_accepts<unsigned long long>);

	long long const huge = 5'000'000'000'000'000'000LL;
	std::vector<g::flow_arc<long long>> const overflowing{{0, 1, huge}, {0, 2, huge}, {1, 3, 7}, {2, 3, 8}};
	REQUIRE(atl_test::aborts([&] { (void)g::max_flow(4, overflowing, 0, 3); }));
	REQUIRE(atl_test::aborts([&] { (void)g::max_flow_value(4, overflowing, 0, 3); }));
	long long const large = 4'000'000'000'000'000'000LL; // the sum still fits
	std::vector<g::flow_arc<long long>> const fitting{{0, 1, large}, {0, 2, large}, {1, 3, 7}, {2, 3, 8}};
	REQUIRE(g::max_flow(4, fitting, 0, 3).value == 15);
	REQUIRE(g::max_flow_value(4, fitting, 0, 3) == 15);
}

TEST_CASE("max_flow: million-vertex paths use heap storage", "[graph]") {
	g::vertex_id const n = 1'000'000;
	std::vector<g::flow_arc<int>> edges;
	edges.reserve(n - 1);
	for (g::vertex_id v = 1; v < n; ++v) edges.push_back({v - 1, v, 1});
	auto check = [&](g::max_flow_result<int> const &result) {
		REQUIRE(result.value == 1);
		REQUIRE(result.edge_flow.size() == edges.size());
		REQUIRE(std::ranges::all_of(result.edge_flow, [](int flow) { return flow == 1; }));
		REQUIRE(result.source_side.size() == n);
		REQUIRE(result.source_side.front() == 1);
		REQUIRE(std::ranges::count(result.source_side, atl::uint8(1)) == 1);
	};
	check(g::max_flow(n, edges, 0, n - 1));
	REQUIRE(g::max_flow_value(n, edges, 0, n - 1) == 1);
	check(ex::max_flow_by_edmonds_karp(n, edges, 0, n - 1));
	check(ex::max_flow_by_dinic(n, edges, 0, n - 1));
	check(ex::max_flow_by_isap(n, edges, 0, n - 1));
	check(ex::max_flow_by_capacity_scaling_edmonds_karp(n, edges, 0, n - 1));
	check(ex::max_flow_by_capacity_scaling_dinic(n, edges, 0, n - 1));
}

TEST_CASE("max_flow: exact dyadic arithmetic at the double significand bound", "[graph]") {
	// Every capacity and the source sum, in eighths, is at most 2^53.
	double const half = 0x1p49;
	std::vector<g::flow_arc<double>> const edges{{0, 1, half}, {0, 2, half}, {1, 3, half}, {2, 3, half - 0.125}};
	auto const r = g::max_flow(4, edges, 0, 3);
	REQUIRE(r.value == 0x1p50 - 0.125);
	certify(4, edges, 0, 3, r);
	REQUIRE(g::max_flow_value(4, edges, 0, 3) == r.value);
	REQUIRE(ex::max_flow_by_edmonds_karp(4, edges, 0, 3).value == r.value);
	REQUIRE(ex::max_flow_by_dinic(4, edges, 0, 3).value == r.value);
	REQUIRE(ex::max_flow_by_isap(4, edges, 0, 3).value == r.value);
}

TEST_CASE("max_flow: vertex and residual sizing checks precede allocation", "[graph]") {
	std::array<g::flow_arc<int>, 0> const empty{};
	REQUIRE(atl_test::aborts([&] { (void)g::max_flow(std::numeric_limits<std::size_t>::max(), empty, 0, 1); }));
	auto const excessive = std::views::iota(uint64(0), uint64(1) << 31)
						 | std::views::transform([](uint64) { return g::flow_arc<int>{0, 1, 1}; });
	REQUIRE(atl_test::aborts([&] { (void)g::max_flow(2, excessive, 0, 1); }));
}
