#include <atl/ds/dsu.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/graph/spanning_tree_count.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <bit>
#include <catch2/catch.hpp>
#include <random>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;
using atl::graph::arc;

namespace {
	using mint = atl::mod_int32<998244353>;

	// every (n-1)-subset of edges, kept when it is acyclic
	[[nodiscard]] uint64 brute_spanning_trees(uint32 n, std::vector<arc<>> const &edges) {
		atl_assert(edges.size() < 64);
		uint32 const m = uint32(edges.size());
		uint64 count = 0;
		for (uint64 mask = 0; mask < (uint64(1) << m); ++mask) {
			if (uint32(std::popcount(mask)) != n - 1) continue;
			atl::dsu d(n);
			bool is_forest = true;
			for (uint32 i = 0; i < m && is_forest; ++i)
				if ((mask >> i) & 1) is_forest = edges[i].from != edges[i].to && d.unite(edges[i].from, edges[i].to);
			count += is_forest;
		}
		return count;
	}

	// every choice of one parent arc per non-root vertex, kept when following
	// parents from any vertex reaches the root (an out-arborescence: parent arcs
	// enter each vertex; is_into_root flips to arcs leaving each vertex)
	[[nodiscard]] uint64 brute_arborescences(uint32 n, std::vector<arc<>> const &edges, uint32 root, bool is_into_root) {
		std::vector<std::vector<uint32>> candidates(n);
		for (auto const &e : edges) {
			if (e.from == e.to) continue;
			uint32 const child = is_into_root ? e.from : e.to, parent = is_into_root ? e.to : e.from;
			candidates[child].push_back(parent);
		}
		std::vector<uint32> choice(n, 0), parent(n);
		uint64 count = 0;
		for (uint32 v = 0; v < n; ++v)
			if (v != root && candidates[v].empty()) return 0;
		while (true) {
			for (uint32 v = 0; v < n; ++v)
				if (v != root) parent[v] = candidates[v][choice[v]];
			bool is_tree = true;
			for (uint32 v = 0; v < n && is_tree; ++v) {
				uint32 u = v, steps = 0;
				while (u != root && steps <= n) u = parent[u], ++steps;
				is_tree = u == root;
			}
			count += is_tree;
			uint32 v = 0;
			for (; v < n; ++v) {
				if (v == root) continue;
				if (++choice[v] < candidates[v].size()) break;
				choice[v] = 0;
			}
			if (v == n) break;
		}
		return count;
	}

	[[nodiscard]] std::vector<arc<>> random_edges(uint32 n, uint32 m, std::mt19937_64 &rng) {
		std::vector<arc<>> edges(m);
		for (auto &e : edges) e = {uint32(rng() % n), uint32(rng() % n)};
		return edges;
	}
}

TEST_CASE("spanning_tree_count: known values", "[spanning_tree_count]") {
	// Cayley: K_n has n^(n-2) spanning trees
	for (uint32 n = 1; n <= 9; ++n) {
		std::vector<arc<>> edges;
		for (uint32 u = 0; u < n; ++u)
			for (uint32 v = u + 1; v < n; ++v) edges.push_back({u, v});
		uint64 expected = 1;
		for (uint32 i = 0; i + 2 < n; ++i) expected *= n;
		REQUIRE(atl::graph::count_spanning_trees<mint>(n, edges).value() == expected % mint::mod());
	}
	std::vector<std::pair<uint32, uint32>> const paper{{0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}, {3, 4}};
	std::vector<arc<>> paper_edges;
	for (auto [u, v] : paper) paper_edges.push_back({u, v});
	REQUIRE(atl::graph::count_spanning_trees<mint>(5, paper_edges) == mint(11));
	REQUIRE(atl::graph::count_spanning_trees<mint>(3, std::vector<arc<>>{{0, 1}}) == mint(0));
	REQUIRE(atl::graph::count_spanning_trees<mint>(1, std::vector<arc<>>{{0, 0}}) == mint(1));
	REQUIRE(atl::graph::count_spanning_trees<mint>(2, std::vector<arc<>>{{0, 1}, {1, 0}, {1, 1}}) == mint(2));
}

TEST_CASE("spanning_tree_count: undirected vs subset enumeration", "[spanning_tree_count]") {
	std::mt19937_64 rng(30);
	using z12 = atl::mod_int32<12>;
	for (int round = 0; round < 400; ++round) {
		uint32 const n = 1 + uint32(rng() % 6), m = uint32(rng() % 13);
		auto const edges = random_edges(n, m, rng);
		uint64 const expected = brute_spanning_trees(n, edges);
		REQUIRE(atl::graph::count_spanning_trees<mint>(n, edges).value() == expected % mint::mod());
		REQUIRE(atl::graph::count_spanning_trees<z12>(n, edges).value() == expected % 12);
	}
}

TEST_CASE("spanning_tree_count: arborescences vs parent enumeration", "[spanning_tree_count]") {
	std::mt19937_64 rng(31);
	using dyn = atl::dynamic_mod_int<>;
	dyn::set_mod(1'000'000'000); // composite run-time modulus
	for (int round = 0; round < 500; ++round) {
		uint32 const n = 1 + uint32(rng() % 6), m = uint32(rng() % 14);
		auto const edges = random_edges(n, m, rng);
		uint32 const root = uint32(rng() % n);
		uint64 const out = brute_arborescences(n, edges, root, false);
		uint64 const in = brute_arborescences(n, edges, root, true);
		REQUIRE(atl::graph::count_out_arborescences<mint>(n, edges, root).value() == out % mint::mod());
		REQUIRE(atl::graph::count_in_arborescences<mint>(n, edges, root).value() == in % mint::mod());
		REQUIRE(atl::graph::count_out_arborescences<dyn>(n, edges, root).value() == out);
		REQUIRE(atl::graph::count_in_arborescences<dyn>(n, edges, root).value() == in);
	}
}

TEST_CASE("spanning_tree_count: asserted preconditions", "[spanning_tree_count]") {
	std::vector<arc<>> const edges{{0, 1}};
	CHECK(atl_test::aborts([&] { (void)atl::graph::count_spanning_trees<mint>(0, std::vector<arc<>>{}); }));
	CHECK(atl_test::aborts([&] { (void)atl::graph::count_spanning_trees<mint>(1, edges); }));
	CHECK(atl_test::aborts([&] { (void)atl::graph::count_out_arborescences<mint>(2, edges, 2); }));
	CHECK(atl_test::aborts([&] { (void)atl::graph::count_in_arborescences<mint>(2, edges, 5); }));
}
