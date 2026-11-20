#include <atl/graph/centroid_decomposition.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/ternarization.hpp>
#include <atl/graph/traversal.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <random>
#include <span>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	template<g::weighted_adjacency_graph Tree>
	std::vector<int64> dists(Tree const &t, uint32 s) {
		std::size_t const n = g::vertex_count(t);
		std::vector<int64> d(n, -1);
		d[s] = 0;
		std::vector<uint32> st{s};
		while (!st.empty()) {
			uint32 const u = st.back();
			st.pop_back();
			for (auto const &e : g::out_edges(t, u))
				if (d[g::target(e)] < 0) {
					d[g::target(e)] = d[u] + int64(g::weight(e));
					st.push_back(g::target(e));
				}
		}
		return d;
	}
}

TEST_CASE("ternarization_of: preserves original distances, bounds degree", "[graph]") {
	auto seed = GENERATE(take(12, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 40;
	bool const weighted = rng() % 2;
	std::vector<std::vector<g::edge<int>>> adj(n);
	std::vector<std::vector<int64>> D(n, std::vector<int64>(n, -1));
	for (uint32 v = 1; v < n; ++v) {
		uint32 const p = uint32(rng() % v);
		adj[p].push_back({v, weighted ? int(rng() % 9) + 1 : 1});
		adj[v].push_back({p, adj[p].back().weight});
	}
	for (uint32 s = 0; s < n; ++s) D[s] = dists(adj, s);

	uint32 const root = uint32(rng() % n);
	auto const t = g::ternarization_of(adj, root);
	std::size_t const N = g::vertex_count(t.tree);
	REQUIRE(N < 2 * n + 1);

	for (uint32 v = 0; v < N; ++v) {
		std::size_t deg = 0;
		for (auto const &e : g::out_edges(t.tree, v)) {
			(void)e;
			++deg;
		}
		REQUIRE(deg <= 3);
	}
	std::size_t root_deg = 0;
	for (auto const &e : g::out_edges(t.tree, t.root)) {
		(void)e;
		++root_deg;
	}
	REQUIRE(root_deg <= 2);

	REQUIRE(t.real_count == n);
	for (uint32 v = 0; v < N; ++v) REQUIRE(t.is_real(v) == (v < n));
	for (uint32 s = 0; s < n; ++s) {
		auto const d = dists(t.tree, s);
		for (uint32 tt = 0; tt < n; ++tt) REQUIRE(d[tt] == D[s][tt]);
	}
}

TEST_CASE("ternarization_of: composes with dfs_fold (<= 2 children per node)", "[graph]") {
	// a star: root with n-1 children -> the fakes must reduce it to <= 2 children
	std::size_t const n = 50;
	std::vector<std::vector<uint32>> star(n);
	for (uint32 v = 1; v < n; ++v) {
		star[0].push_back(v);
		star[v].push_back(0);
	}
	auto const t = g::ternarization_of(star, 0);
	// subtree size over REAL vertices, folded up the binarized tree, is n
	auto const real_count = g::dfs_fold(
		t.tree, t.root,
		[&](g::visit u) { return int64(t.is_real(u.vertex) ? 1 : 0); },
		[](int64 &a, int64 c, g::visit, g::visit, g::edge<int> const &) { a += c; });
	REQUIRE(real_count == int64(n));
}

TEST_CASE("ternarization_of + centroid: weighted path counting over reals", "[graph]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + rng() % 25;
	std::vector<std::vector<g::edge<int>>> adj(n);
	for (uint32 v = 1; v < n; ++v) {
		uint32 const p = uint32(rng() % v);
		adj[p].push_back({v, int(rng() % 5) + 1});
		adj[v].push_back({p, adj[p].back().weight});
	}
	int const W = int(rng() % 15) + 1;

	int64 brute = 0;
	for (uint32 s = 0; s < n; ++s) {
		auto const d = dists(adj, s);
		for (uint32 tt = s + 1; tt < n; ++tt)
			if (d[tt] == W) ++brute;
	}

	auto const t = g::ternarization_of(adj, 0);
	std::size_t const N = g::vertex_count(t.tree);
	std::vector<int64> wdist(N), seen(W + 1, 0);
	int64 ans = 0;
	g::centroid_visit(t.tree, [&](uint32 c, uint32, std::span<g::centroid_vertex const> comp,
								  std::span<g::edge<int> const> edges) {
		uint32 branches = 0;
		for (auto const &r : comp)
			if (r.branch != g::none) branches = (std::max)(branches, r.branch + 1);
		REQUIRE(branches <= 3); // bounded fan-in from ternarization

		wdist[c] = 0;
		for (std::size_t i = 1; i < comp.size(); ++i)
			wdist[comp[i].vertex] = wdist[comp[i].parent] + edges[i].weight;
		std::ranges::fill(seen, 0);
		if (t.is_real(c)) seen[0] = 1;
		std::size_t i = 1;
		while (i < comp.size()) {
			std::size_t const from = i;
			uint32 const br = comp[i].branch;
			for (; i < comp.size() && comp[i].branch == br; ++i)
				if (t.is_real(comp[i].vertex) && wdist[comp[i].vertex] <= W)
					ans += seen[std::size_t(W - wdist[comp[i].vertex])];
			for (std::size_t j = from; j < i; ++j)
				if (t.is_real(comp[j].vertex) && wdist[comp[j].vertex] <= W)
					++seen[std::size_t(wdist[comp[j].vertex])];
		}
	});
	REQUIRE(ans == brute);
}

TEST_CASE("ternarization_of: explicit Weight override and csr input", "[graph]") {
	// a 200-leaf caterpillar given as an undirected csr: the int64 override
	// widens the weights, and distances from the spine's end are kept
	std::size_t const n = 300;
	std::vector<g::arc<int>> arcs;
	for (uint32 v = 1; v < 100; ++v) arcs.push_back({v - 1, v, int(v % 7) + 1});
	for (uint32 v = 100; v < n; ++v) arcs.push_back({v % 100, v, 3});
	auto const tree = g::undirected_csr(n, arcs);
	auto const t = g::ternarization_of<int64>(tree, 0);
	static_assert(std::same_as<decltype(t), g::ternarization<int64> const>);
	REQUIRE(t.real_count == n);
	auto const want = dists(tree, 0), got = dists(t.tree, 0);
	for (uint32 v = 0; v < n; ++v) REQUIRE(got[v] == want[v]);

	struct opaque {}; // not constructible from 1: rejected at the constraint

	static_assert(!g::ternarization_weight<opaque, std::vector<std::vector<uint32>>>);
	static_assert(g::ternarization_weight<void, std::vector<std::vector<uint32>>>);
	static_assert(g::ternarization_weight<double, std::vector<std::vector<g::edge<int>>>>);
}

TEST_CASE("ternarization_of: a cyclic or disconnected input aborts", "[graph]") {
	std::vector<std::vector<uint32>> const triangle{{1, 2}, {0, 2}, {0, 1}};
	CHECK(atl_test::aborts([&] { (void)g::ternarization_of(triangle, 0); }));
	std::vector<std::vector<uint32>> const forest{{1}, {0}, {3}, {2}};
	CHECK(atl_test::aborts([&] { (void)g::ternarization_of(forest, 0); }));
}
