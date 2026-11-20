#include <atl/experimental/graph/scc_variants.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/scc.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
using atl::uint64;

namespace {
	/** @brief Mutual reachability by transitive closure — the tiny-n oracle. */
	std::vector<std::vector<bool>> closure(std::size_t n, std::vector<g::arc<>> const &arcs) {
		std::vector reach(n, std::vector<bool>(n, false));
		for (std::size_t v = 0; v < n; ++v) reach[v][v] = true;
		for (auto const &a : arcs) reach[a.from][a.to] = true;
		for (std::size_t k = 0; k < n; ++k)
			for (std::size_t i = 0; i < n; ++i)
				if (reach[i][k])
					for (std::size_t j = 0; j < n; ++j)
						if (reach[k][j]) reach[i][j] = true;
		return reach;
	}
}

TEST_CASE("scc: components are mutual-reachability classes in topological order", "[graph]") {
	auto seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = std::size_t(1 + rng() % 40);
	std::size_t const m = std::size_t(rng() % (3 * n));
	std::vector<g::arc<>> arcs;
	for (std::size_t i = 0; i < m; ++i)
		arcs.push_back({g::vertex_id(rng() % n), g::vertex_id(rng() % n)});
	g::csr const net(n, arcs);

	auto const s = g::scc(net);
	auto const reach = closure(n, arcs);

	REQUIRE(s.component.size() == n);
	for (std::size_t u = 0; u < n; ++u) {
		REQUIRE(s.component[u] < s.count());
		for (std::size_t v = 0; v < n; ++v) {
			bool const is_mutually_reachable = reach[u][v] && reach[v][u];
			REQUIRE(s.same(g::vertex_id(u), g::vertex_id(v)) == is_mutually_reachable);
		}
	}
	for (auto const &a : arcs) REQUIRE(s.component[a.from] <= s.component[a.to]);

	auto const oracle = atl::experimental::scc_by_kosaraju(net);
	REQUIRE(oracle.count() == s.count());
	for (std::size_t u = 0; u < n; ++u)
		for (std::size_t v = 0; v < n; ++v)
			REQUIRE((oracle.component[u] == oracle.component[v])
					== s.same(g::vertex_id(u), g::vertex_id(v)));

	auto const grouped = s.groups();
	REQUIRE(grouped.size() == s.count());
	std::size_t total = 0;
	for (std::size_t c = 0; c < grouped.size(); ++c) {
		REQUIRE(!grouped[c].empty());
		for (auto const v : grouped[c]) REQUIRE(s.component[v] == c);
		total += grouped[c].size();
	}
	REQUIRE(total == n);
}

TEST_CASE("scc: canonical shapes", "[graph]") {
	std::size_t const n = std::size_t(1000);
	std::vector<g::arc<>> cyc;
	for (g::vertex_id v = 0; v < n; ++v) cyc.push_back({v, g::vertex_id((v + 1) % n)});
	auto const one = g::scc(g::csr(n, cyc));
	REQUIRE(one.count() == 1);

	std::vector<g::arc<>> chain;
	for (g::vertex_id v = 0; v + 1 < n; ++v) chain.push_back({v, g::vertex_id(v + 1)});
	auto const dag = g::scc(g::csr(n, chain));
	REQUIRE(dag.count() == n);
	for (g::vertex_id v = 0; v + 1 < n; ++v) REQUIRE(dag.component[v] < dag.component[v + 1]);

	std::vector<g::arc<>> two{{0, 1}, {1, 0}, {2, 3}, {3, 2}, {1, 2}};
	auto const pair = g::scc(g::csr(4, two));
	REQUIRE(pair.count() == 2);
	REQUIRE(pair.same(0, 1));
	REQUIRE(pair.same(2, 3));
	REQUIRE(pair.component[0] < pair.component[2]);

	std::size_t const deep = 1'000'000;
	std::vector<std::vector<g::vertex_id>> path(deep);
	for (g::vertex_id v = 0; v + 1 < deep; ++v) path[v].push_back(v + 1);
	auto const line = g::scc(path);
	REQUIRE(line.count() == deep);
}

TEST_CASE("scc: every experimental variant against the closure oracle", "[graph]") {
	namespace ex = atl::experimental;
	auto seed = GENERATE(take(30, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = std::size_t(rng() % 30);
	std::size_t const m = std::size_t(n == 0 ? 0 : rng() % (4 * n));
	std::vector<g::arc<>> arcs;
	for (std::size_t i = 0; i < m; ++i) {
		g::vertex_id const u = g::vertex_id(rng() % n), v = rng() % 4 ? g::vertex_id(rng() % n) : u;
		arcs.push_back({u, v});
		if (rng() % 4 == 0) arcs.push_back({u, v});
	}
	g::csr const net(n, arcs);
	std::vector<std::vector<g::vertex_id>> lists(n);
	for (auto const &a : arcs) lists[a.from].push_back(a.to);
	auto const reach = closure(n, arcs);

	auto const check = [&](g::vertex_partition const &p) {
		REQUIRE(p.component.size() == n);
		for (std::size_t u = 0; u < n; ++u) {
			REQUIRE(p.component[u] < p.count());
			for (std::size_t v = 0; v < n; ++v)
				REQUIRE(p.same(g::vertex_id(u), g::vertex_id(v)) == (reach[u][v] && reach[v][u]));
		}
		for (auto const &a : arcs) REQUIRE(p.component[a.from] <= p.component[a.to]);
	};
	check(g::scc(net));
	check(g::scc(lists));
	check(ex::scc_by_kosaraju(net));
	check(ex::scc_by_tarjan_events(net));
	check(ex::scc_by_tarjan_events(lists));
	check(ex::scc_by_tarjan(net));
	check(ex::scc_by_gabow(net));
}

TEST_CASE("scc: an out-of-range edge target is asserted", "[graph]") {
	std::vector<std::vector<int>> const bad{{1}, {7}};
	REQUIRE(atl_test::aborts([&] { (void)g::scc(bad); }));
}
