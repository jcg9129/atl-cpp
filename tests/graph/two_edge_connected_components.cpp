#include <atl/ds/dsu.hpp>
#include <atl/experimental/graph/two_edge_connected_components_variants.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/two_edge_connected_components.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::uint64;

namespace {
	/** @brief Bridges by definition — delete each edge in turn and test whether its
endpoints stay connected. O(m^2 alpha), the tiny-n oracle. */
	std::vector<bool> brute_bridges(std::size_t n, std::vector<g::arc<>> const &arcs) {
		std::vector<bool> bridge(arcs.size());
		for (std::size_t i = 0; i < arcs.size(); ++i) {
			atl::dsu d(n);
			for (std::size_t j = 0; j < arcs.size(); ++j)
				if (j != i) d.unite(arcs[j].from, arcs[j].to);
			bridge[i] = !d.same(arcs[i].from, arcs[i].to);
		}
		return bridge;
	}

	/** @brief True iff `p` is exactly connectivity over the non-bridge edges, the
bridges taken from brute_bridges. */
	bool matches_bridge_oracle(std::size_t n, std::vector<g::arc<>> const &arcs, g::vertex_partition const &p) {
		auto const bridge = brute_bridges(n, arcs);
		atl::dsu d(n);
		for (std::size_t i = 0; i < arcs.size(); ++i)
			if (!bridge[i]) d.unite(arcs[i].from, arcs[i].to);
		if (p.component.size() != n || p.count() != d.count()) return false;
		for (g::vertex_id u = 0; u < n; ++u)
			for (g::vertex_id v = 0; v < n; ++v)
				if ((p.component[u] == p.component[v]) != d.same(u, v)) return false;
		return true;
	}
}

TEST_CASE("two_edge_connected_components: components match the delete-an-edge oracle", "[graph]") {
	auto seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = std::size_t(1 + rng() % 40);
	std::size_t const m = std::size_t(rng() % (3 * n));
	std::vector<g::arc<>> arcs;
	for (std::size_t i = 0; i < m; ++i)
		arcs.push_back({g::vertex_id(rng() % n), g::vertex_id(rng() % n)});
	auto const net = g::undirected_csr(n, arcs);

	auto const r = g::two_edge_connected_components(net);
	auto const bridge = brute_bridges(n, arcs);

	REQUIRE(r.component.size() == n);
	for (g::vertex_id v = 0; v < n; ++v) REQUIRE(r.component[v] < r.count());

	atl::dsu d(n);
	for (std::size_t i = 0; i < arcs.size(); ++i)
		if (!bridge[i]) d.unite(arcs[i].from, arcs[i].to);
	REQUIRE(r.count() == d.count());
	for (g::vertex_id u = 0; u < n; ++u)
		for (g::vertex_id v = 0; v < n; ++v) REQUIRE(r.same(u, v) == d.same(u, v));

	for (std::size_t i = 0; i < arcs.size(); ++i)
		REQUIRE(bridge[i] == !r.same(arcs[i].from, arcs[i].to));

	std::vector<std::size_t> smaller_neighbours(r.count(), 0);
	for (std::size_t i = 0; i < arcs.size(); ++i) {
		if (!bridge[i]) continue;
		auto const a = r.component[arcs[i].from], b = r.component[arcs[i].to];
		++smaller_neighbours[(std::max)(a, b)];
	}
	atl::dsu whole(n);
	for (auto const &a : arcs) whole.unite(a.from, a.to);
	std::vector<bool> is_root_component(r.count(), false);
	for (g::vertex_id v = 0; v < n; ++v) {
		bool is_lowest = true;
		for (g::vertex_id u = 0; u < v && is_lowest; ++u) is_lowest = !whole.same(u, v);
		if (is_lowest) is_root_component[r.component[v]] = true;
	}
	for (std::size_t c = 0; c < r.count(); ++c)
		REQUIRE(smaller_neighbours[c] == (is_root_component[c] ? 0u : 1u));

	REQUIRE(matches_bridge_oracle(n, arcs, ex::two_edge_connected_components_by_low_link(net)));
	REQUIRE(matches_bridge_oracle(n, arcs, ex::two_edge_connected_components_by_gabow(net)));
	REQUIRE(matches_bridge_oracle(n, arcs, ex::two_edge_connected_components_by_chain_decomposition(net)));
	std::vector<std::vector<g::vertex_id>> lists(n);
	for (auto const &a : arcs) {
		lists[a.from].push_back(a.to);
		lists[a.to].push_back(a.from);
	}
	REQUIRE(matches_bridge_oracle(n, arcs, g::two_edge_connected_components(lists)));

	auto const grouped = r.groups();
	REQUIRE(grouped.size() == r.count());
	std::size_t total = 0;
	for (std::size_t c = 0; c < grouped.size(); ++c) {
		REQUIRE(!grouped[c].empty());
		for (auto const v : grouped[c]) REQUIRE(r.component[v] == c);
		total += grouped[c].size();
	}
	REQUIRE(total == n);
}

TEST_CASE("two_edge_connected_components: variants match the oracle on canonical shapes", "[graph]") {
	struct shape {
		std::size_t n;
		std::vector<g::arc<>> arcs;
	};

	std::vector<shape> const shapes{
		{1, {}},
		{1, {{0, 0}}},
		{3, {{0, 1}, {0, 1}, {1, 2}}},
		{4, {{0, 0}, {1, 2}}},
		{6, {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {2, 3}}},
		{5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}}},
	};
	for (auto const &[n, arcs] : shapes) {
		auto const net = g::undirected_csr(n, arcs);
		REQUIRE(matches_bridge_oracle(n, arcs, g::two_edge_connected_components(net)));
		REQUIRE(matches_bridge_oracle(n, arcs, ex::two_edge_connected_components_by_low_link(net)));
		REQUIRE(matches_bridge_oracle(n, arcs, ex::two_edge_connected_components_by_gabow(net)));
		REQUIRE(matches_bridge_oracle(n, arcs, ex::two_edge_connected_components_by_chain_decomposition(net)));
	}
}

TEST_CASE("two_edge_connected_components: canonical shapes", "[graph]") {
	std::size_t const n = std::size_t(1000);
	std::vector<g::arc<>> path;
	for (g::vertex_id v = 0; v + 1 < n; ++v) path.push_back({v, g::vertex_id(v + 1)});
	REQUIRE(g::two_edge_connected_components(g::undirected_csr(n, path)).count() == n);

	std::vector<g::arc<>> cyc = path;
	cyc.push_back({g::vertex_id(n - 1), 0});
	REQUIRE(g::two_edge_connected_components(g::undirected_csr(n, cyc)).count() == 1);

	std::vector<g::arc<>> tri{{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {2, 3}};
	auto const pair = g::two_edge_connected_components(g::undirected_csr(std::size_t(6), tri));
	REQUIRE(pair.count() == 2);
	REQUIRE(pair.same(0, 2));
	REQUIRE(pair.same(3, 5));
	REQUIRE(!pair.same(2, 3));

	std::vector<g::arc<>> multi{{0, 1}, {0, 1}, {1, 2}};
	auto const dbl = g::two_edge_connected_components(g::undirected_csr(std::size_t(3), multi));
	REQUIRE(dbl.count() == 2);
	REQUIRE(dbl.same(0, 1));
	REQUIRE(!dbl.same(1, 2));

	std::vector<g::arc<>> loops{{0, 0}, {1, 2}};
	REQUIRE(g::two_edge_connected_components(g::undirected_csr(std::size_t(4), loops)).count() == 4);

	std::size_t const deep = 1'000'000;
	std::vector<std::vector<g::vertex_id>> line(deep);
	for (g::vertex_id v = 0; v + 1 < deep; ++v) {
		line[v].push_back(v + 1);
		line[v + 1].push_back(v);
	}
	REQUIRE(g::two_edge_connected_components(line).count() == deep);

	line[deep - 1].push_back(0);
	line[0].push_back(g::vertex_id(deep - 1));
	REQUIRE(g::two_edge_connected_components(line).count() == 1);
}

TEST_CASE("two_edge_connected_components: an out-of-range edge target is asserted", "[graph]") {
	std::vector<std::vector<int>> const bad{{1}, {7}};
	REQUIRE(atl_test::aborts([&] { (void)g::two_edge_connected_components(bad); }));
}
