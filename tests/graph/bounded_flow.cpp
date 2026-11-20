#include <atl/graph/max_flow.hpp>
#include <atl/graph/min_cost_flow.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
using atl::int64;
using atl::uint64;
using atl::uint8;

namespace {
	void certify_feasible(std::size_t n, std::vector<g::bounded_flow_arc<int64>> const &edges,
						  std::vector<int64> const &b, std::vector<int64> const &flow) {
		REQUIRE(flow.size() == edges.size());
		std::vector<int64> divergence(n, 0);
		for (std::size_t i = 0; i < edges.size(); ++i) {
			REQUIRE(flow[i] >= edges[i].lower);
			REQUIRE(flow[i] <= edges[i].capacity);
			divergence[edges[i].from] += flow[i];
			divergence[edges[i].to] -= flow[i];
		}
		for (g::vertex_id v = 0; v < n; ++v) REQUIRE(divergence[v] == b[v]);
	}

	/** @brief Cut equality proves both extremal bounds by weak duality. */
	void certify_extremal(std::size_t n, std::vector<g::bounded_flow_arc<int64>> const &edges,
						  g::vertex_id s, g::vertex_id t, g::max_flow_result<int64> const &r,
						  bool is_maximum) {
		std::vector<int64> b(n, 0);
		b[s] = r.value;
		b[t] = -r.value;
		certify_feasible(n, edges, b, r.edge_flow);
		REQUIRE(r.source_side[s] == 1);
		REQUIRE(r.source_side[t] == 0);
		int64 bound = 0;
		for (auto const &e : edges) {
			if (r.source_side[e.from] && !r.source_side[e.to])
				bound += is_maximum ? e.capacity : e.lower;
			if (!r.source_side[e.from] && r.source_side[e.to])
				bound -= is_maximum ? e.lower : e.capacity;
		}
		REQUIRE(bound == r.value);
	}
}

static_assert(requires(std::vector<g::flow_arc<int64>> plain, std::vector<int64> b) {
	g::feasible_flow(std::size_t(2), plain, b);
});

TEST_CASE("feasible_flow: witnessed instances are feasible, perturbed ones are not", "[graph]") {
	auto seed = GENERATE(take(10, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + std::size_t(rng() % 10);
	std::size_t const m = std::size_t(rng() % 24);
	std::vector<g::bounded_flow_arc<int64>> edges;
	std::vector<int64> b(n, 0);
	for (std::size_t i = 0; i < m; ++i) {
		auto const from = g::vertex_id(rng() % n), to = g::vertex_id(rng() % n);
		int64 const witness = int64(rng() % 7);
		int64 const lower = witness <= 0 ? 0 : int64(rng() % (witness + 1));
		edges.push_back({from, to, lower, witness + int64(rng() % 7)});
		b[from] += witness;
		b[to] -= witness;
	}
	auto const r = g::feasible_flow(n, edges, b);
	REQUIRE(r.has_value());
	certify_feasible(n, edges, b, *r);
	if (n > 1) {
		b[0] += 1;
		REQUIRE(!g::feasible_flow(n, edges, b).has_value());
	}
}

TEST_CASE("bounded max/min flow: certified extremal and engine-cross-checked", "[graph]") {
	auto seed = GENERATE(take(10, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + std::size_t(rng() % 8);
	std::size_t const m = std::size_t(rng() % 20);
	std::vector<g::bounded_flow_arc<int64>> edges;
	for (std::size_t i = 0; i < m; ++i) {
		auto const from = g::vertex_id(rng() % n), to = g::vertex_id(rng() % n);
		int64 const lower = int64(rng() % 6);
		edges.push_back({from, to, lower, lower + int64(rng() % 7)});
	}
	g::vertex_id const s = g::vertex_id(rng() % n);
	g::vertex_id t = g::vertex_id(rng() % n);
	while (t == s) t = g::vertex_id(rng() % n);

	auto const mx = g::bounded_max_flow(n, edges, s, t);
	auto const mn = g::bounded_min_flow(n, edges, s, t);
	REQUIRE(mx.has_value() == mn.has_value());

	// independent engine: the same extremal values through network simplex,
	// pricing the value with a +-1 shortcut pair
	int64 wide = 1;
	for (auto const &e : edges) wide += e.capacity;
	std::vector<g::bounded_flow_arc<int64, int64>> priced;
	for (auto const &e : edges) priced.push_back({e.from, e.to, e.lower, e.capacity, 0});
	priced.push_back({t, s, 0, wide, 1});
	priced.push_back({s, t, 0, wide, -1});
	auto const simplex_min = g::min_cost_feasible_flow(n, priced, std::vector<int64>(n, 0));
	REQUIRE(mx.has_value() == simplex_min.has_value());

	if (!mx) return;
	certify_extremal(n, edges, s, t, *mx, true);
	certify_extremal(n, edges, s, t, *mn, false);
	REQUIRE(mn->value <= mx->value);
	REQUIRE(mn->value == simplex_min->cost);
	for (auto &arc : priced) arc.cost = -arc.cost;
	REQUIRE(mx->value == -g::min_cost_feasible_flow(n, priced, std::vector<int64>(n, 0))->cost);
}

TEST_CASE("bounded flow: canonical instances", "[graph]") {
	std::vector<g::bounded_flow_arc<int64>> bounded_network{
		{8, 0, 17, 18}, {8, 1, 12, 13}, {8, 2, 11, 12}, {0, 4, 3, 4}, {0, 5, 6, 7}, {0, 6, 7, 8}, {1, 4, 9, 10}, {1, 5, 2, 3}, {1, 6, 0, 1}, {2, 4, 3, 4}, {2, 5, 1, 2}, {2, 6, 6, 7}, {4, 9, 16, 17}, {5, 9, 10, 11}, {6, 9, 14, 15}};
	auto const mx = g::bounded_max_flow(std::size_t(10), bounded_network, 8, 9);
	REQUIRE(mx.has_value());
	REQUIRE(mx->value == 43);
	certify_extremal(10, bounded_network, 8, 9, *mx, true);

	// bounds forcing net backward flow: the minimum (and even the maximum)
	// s->t value is negative
	std::vector<g::bounded_flow_arc<int64>> backward{{1, 0, 3, 5}};
	auto const forced_mx = g::bounded_max_flow(std::size_t(2), backward, 0, 1);
	auto const forced_mn = g::bounded_min_flow(std::size_t(2), backward, 0, 1);
	REQUIRE(forced_mx->value == -3);
	REQUIRE(forced_mn->value == -5);
	certify_extremal(2, backward, 0, 1, *forced_mx, true);
	certify_extremal(2, backward, 0, 1, *forced_mn, false);

	std::vector<g::bounded_flow_arc<int64>> stuck{{0, 1, 3, 5}, {2, 0, 2, 4}};
	REQUIRE(!g::bounded_max_flow(std::size_t(3), stuck, 0, 1).has_value());
	REQUIRE(!g::feasible_flow(std::size_t(3), stuck, std::vector<int64>(3, 0)).has_value());

	std::vector<g::flow_arc<int64>> plain{{0, 1, 4}, {1, 0, 2}};
	auto const zero = g::feasible_flow(std::size_t(2), plain, std::vector<int64>(2, 0));
	REQUIRE(zero.has_value());
	REQUIRE(*zero == std::vector<int64>{0, 0});
}

static_assert(g::flow_value<int> && g::flow_value<atl::uint32> && g::flow_value<double>);
static_assert(g::signed_flow_value<int64> && g::signed_flow_value<double>);
static_assert(!g::signed_flow_value<atl::uint32> && !g::flow_value<std::vector<int>>);

namespace {
	template<typename Capacity>
	concept bounded_accepts = requires(std::vector<g::bounded_flow_arc<Capacity>> const &edges) {
		g::bounded_max_flow(std::size_t(2), edges, 0u, 1u);
		g::bounded_min_flow(std::size_t(2), edges, 0u, 1u);
		g::feasible_flow(std::size_t(2), edges, std::vector<Capacity>(2));
	};

	template<typename Flow>
	concept max_flow_accepts = requires(std::vector<g::flow_arc<int>> const &edges) {
		g::max_flow<Flow>(std::size_t(2), edges, 0u, 1u);
		g::max_flow_value<Flow>(std::size_t(2), edges, 0u, 1u);
	};

	struct not_a_number {};
}

static_assert(bounded_accepts<int64> && !bounded_accepts<atl::uint32>);
static_assert(max_flow_accepts<void> && max_flow_accepts<atl::uint32> && max_flow_accepts<double>);
static_assert(!max_flow_accepts<not_a_number>);

TEST_CASE("feasible_flow: the supply vector is a sink, moved in", "[graph]") {
	std::vector<g::bounded_flow_arc<int64>> const edges{{0, 1, 1, 3}, {1, 2, 0, 3}};
	std::vector<int64> b{2, 0, -2};
	auto const r = g::feasible_flow(3, edges, std::move(b));
	REQUIRE(r.has_value());
	certify_feasible(3, edges, {2, 0, -2}, *r);
}

TEST_CASE("bounded flow: invalid endpoints and arithmetic bounds abort", "[graph]") {
	std::vector<g::bounded_flow_arc<int>> const invalid{{0, 2, 0, 1}};
	REQUIRE(atl_test::aborts([&] { (void)g::feasible_flow(2, invalid, std::vector<int>(2)); }));
	REQUIRE(atl_test::aborts([&] { (void)g::bounded_max_flow(2, invalid, 0, 1); }));
	REQUIRE(atl_test::aborts([&] { (void)g::bounded_min_flow(2, invalid, 0, 1); }));
	int const limit = std::numeric_limits<int>::max();
	std::vector<g::flow_arc<int>> const empty;
	REQUIRE(atl_test::aborts([&] { (void)g::feasible_flow(3, empty, std::vector<int>{limit, 1, -limit}); }));
	std::vector<g::bounded_flow_arc<int>> const adjust{{1, 0, 1, 1}};
	REQUIRE(atl_test::aborts([&] { (void)g::feasible_flow(2, adjust, std::vector<int>{limit, -limit}); }));
	REQUIRE(atl_test::aborts([&] { (void)g::feasible_flow(3, empty, std::vector<int>{std::numeric_limits<int>::lowest(), limit, 1}); }));
	std::vector<g::bounded_flow_arc<int>> const too_wide{{0, 1, 0, limit}};
	REQUIRE(atl_test::aborts([&] { (void)g::bounded_max_flow(2, too_wide, 0, 1); }));
}

TEST_CASE("bounded flow: representable shortcut and supply limits", "[graph]") {
	int64 const limit = std::numeric_limits<int64>::max() - 1;
	std::vector<g::bounded_flow_arc<int64>> const edges{{0, 1, limit, limit}};
	auto const greatest = g::bounded_max_flow(2, edges, 0, 1);
	auto const least = g::bounded_min_flow(2, edges, 0, 1);
	REQUIRE(greatest);
	REQUIRE(least);
	REQUIRE(greatest->value == limit);
	REQUIRE(least->value == limit);
	certify_extremal(2, edges, 0, 1, *greatest, true);
	certify_extremal(2, edges, 0, 1, *least, false);
	std::vector<g::flow_arc<int64>> const plain{{0, 1, std::numeric_limits<int64>::max()}};
	auto const flow = g::feasible_flow(2, plain, std::vector<int64>{std::numeric_limits<int64>::max(), -std::numeric_limits<int64>::max()});
	REQUIRE(flow);
	REQUIRE((*flow)[0] == std::numeric_limits<int64>::max());
}
