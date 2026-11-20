#include <atl/experimental/graph/bipartite_matching_variants.hpp>
#include <atl/graph/bipartite_matching.hpp>
#include <atl/graph/max_flow.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <random>
#include <set>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::uint32;
using atl::uint64;
using atl::uint8;

namespace {
	/** @brief Certify mutual pairs, input-edge membership and cardinality. */
	void certify(std::size_t left, std::size_t right, std::vector<g::arc<>> const &edges,
				 g::bipartite_matching_result const &r) {
		REQUIRE(r.left_pair.size() == left);
		REQUIRE(r.right_pair.size() == right);
		std::set<std::pair<uint32, uint32>> present;
		for (auto const &e : edges) present.insert({e.from, e.to});
		std::size_t matched = 0;
		for (uint32 l = 0; l < left; ++l)
			if (r.left_pair[l] != g::none) {
				REQUIRE(r.right_pair[r.left_pair[l]] == l);
				REQUIRE(present.contains({l, r.left_pair[l]}));
				++matched;
			}
		for (uint32 w = 0; w < right; ++w)
			if (r.right_pair[w] != g::none) REQUIRE(r.left_pair[r.right_pair[w]] == w);
		REQUIRE(matched == r.size);
	}

	/** @brief Independent cardinality oracle via a unit-capacity flow reduction. */
	[[nodiscard]] std::size_t flow_matching_size(std::size_t left, std::size_t right,
												 std::vector<g::arc<>> const &edges) {
		auto const s = uint32(left + right), t = uint32(left + right + 1);
		std::vector<g::flow_arc<int>> arcs;
		for (auto const &e : edges) arcs.push_back({e.from, uint32(left) + e.to, 1});
		for (uint32 l = 0; l < left; ++l) arcs.push_back({s, l, 1});
		for (uint32 w = 0; w < right; ++w) arcs.push_back({uint32(left) + w, t, 1});
		return std::size_t(g::max_flow_value(left + right + 2, arcs, s, t));
	}
}

TEST_CASE("bipartite_matching: maximum and valid on random graphs", "[graph]") {
	auto seed = GENERATE(take(10, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const left = std::size_t(1 + rng() % 12), right = std::size_t(1 + rng() % 12);
	std::size_t const m = std::size_t(rng() % 30);
	std::vector<g::arc<>> edges;
	for (std::size_t i = 0; i < m; ++i)
		edges.push_back({uint32(rng() % left), uint32(rng() % right)}); // parallels legal

	auto const want = flow_matching_size(left, right, edges);
	auto const hk = g::bipartite_matching(left, right, edges);
	REQUIRE(hk.size == want);
	certify(left, right, edges, hk);

	auto const kuhn = ex::bipartite_matching_by_kuhn(left, right, edges);
	REQUIRE(kuhn.size == want);
	certify(left, right, edges, kuhn);

	auto const hopcroft = ex::bipartite_matching_by_hopcroft_karp(left, right, edges);
	REQUIRE(hopcroft.size == want);
	certify(left, right, edges, hopcroft);

	auto const passes = ex::bipartite_matching_by_kuhn_passes(left, right, edges);
	REQUIRE(passes.size == want);
	certify(left, right, edges, passes);

	// König: the cover touches every edge and matches the matching's size —
	// weak duality makes that pair a proof of minimality
	auto const cover = g::minimum_vertex_cover(left, right, edges, hk);
	REQUIRE(cover.size == hk.size);
	std::size_t chosen = 0;
	for (auto const c : cover.left) chosen += c;
	for (auto const c : cover.right) chosen += c;
	REQUIRE(chosen == cover.size);
	for (auto const &e : edges) REQUIRE((cover.left[e.from] || cover.right[e.to]));

	auto const is = g::maximum_independent_set(left, right, edges, hk);
	REQUIRE(is.size == left + right - hk.size);
	for (uint32 l = 0; l < left; ++l) REQUIRE(is.left[l] == !cover.left[l]);
	for (uint32 w = 0; w < right; ++w) REQUIRE(is.right[w] == !cover.right[w]);
	for (auto const &e : edges) REQUIRE(!(is.left[e.from] && is.right[e.to]));
}

TEST_CASE("bipartite_matching: canonical shapes", "[graph]") {
	// perfect matching on a line: l_i - r_i and l_i - r_{i-1}
	std::size_t const n = 1000;
	std::vector<g::arc<>> line;
	for (uint32 i = 0; i < n; ++i) {
		line.push_back({i, i});
		if (i > 0) line.push_back({i, i - 1});
	}
	auto const chain = g::bipartite_matching(n, n, line);
	REQUIRE(chain.size == n);
	certify(n, n, line, chain);

	std::vector<g::arc<>> star;
	for (uint32 w = 0; w < 50; ++w) star.push_back({0, w});
	REQUIRE(g::bipartite_matching(std::size_t(1), std::size_t(50), star).size == 1);

	REQUIRE(g::bipartite_matching(std::size_t(3), std::size_t(4), std::vector<g::arc<>>{}).size == 0);

	std::vector<g::arc<>> par{{0, 0}, {0, 0}, {0, 0}};
	auto const dup = g::bipartite_matching(std::size_t(1), std::size_t(1), par);
	REQUIRE(dup.size == 1);
	certify(1, 1, par, dup);
}

TEST_CASE("bipartite_matching: every contender on canonical and mid-size shapes", "[graph]") {
	auto const all = [](std::size_t left, std::size_t right, std::vector<g::arc<>> const &edges) {
		auto const want = flow_matching_size(left, right, edges);
		auto const library = g::bipartite_matching(left, right, edges);
		REQUIRE(library.size == want);
		certify(left, right, edges, library);
		auto const hk = ex::bipartite_matching_by_hopcroft_karp(left, right, edges);
		REQUIRE(hk.size == want);
		certify(left, right, edges, hk);
		auto const kuhn = ex::bipartite_matching_by_kuhn(left, right, edges);
		REQUIRE(kuhn.size == want);
		certify(left, right, edges, kuhn);
		auto const passes = ex::bipartite_matching_by_kuhn_passes(left, right, edges);
		REQUIRE(passes.size == want);
		certify(left, right, edges, passes);
	};
	std::size_t const n = 2000;
	std::vector<g::arc<>> line;
	for (uint32 i = 0; i < n; ++i) {
		line.push_back({i, i});
		if (i > 0) line.push_back({i, i - 1});
	}
	all(n, n, line);
	std::vector<g::arc<>> star;
	for (uint32 w = 0; w < 50; ++w) star.push_back({0, w});
	all(1, 50, star);
	all(3, 4, {});
	all(1, 1, {{0, 0}, {0, 0}, {0, 0}});
	all(0, 5, {});
	all(5, 0, {});

	std::mt19937_64 rng(7);
	for (int round = 0; round < 3; ++round) {
		std::size_t const left = std::size_t(1500 + rng() % 1000), right = std::size_t(1500 + rng() % 1000);
		std::vector<g::arc<>> random;
		for (std::size_t i = 0; i < 3 * left; ++i) random.push_back({uint32(rng() % left), uint32(rng() % right)});
		all(left, right, random);
	}
	// permuted cycle with window chords: long alternating paths, no locality
	std::vector<uint32> pl(n), pr(n);
	for (uint32 i = 0; i < n; ++i) pl[i] = pr[i] = i;
	std::ranges::shuffle(pl, rng);
	std::ranges::shuffle(pr, rng);
	std::vector<g::arc<>> cycle;
	for (uint32 i = 0; i < n; ++i) {
		cycle.push_back({pl[i], pr[i]});
		cycle.push_back({pl[i], pr[(i + 1) % n]});
		cycle.push_back({pl[i], pr[i < 30 ? uint32(rng() % (i + 1)) : i - uint32(rng() % 30)]});
	}
	std::ranges::shuffle(cycle, rng);
	all(n, n, cycle);
}

TEST_CASE("bipartite_matching: path cover and chain cover with antichain certificate", "[graph]") {
	auto seed = GENERATE(take(10, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = std::size_t(1 + rng() % 14);
	std::vector reach(n, std::vector<bool>(n, false));
	std::vector<g::arc<>> edges;
	for (uint32 a = 0; a < n; ++a)
		for (uint32 b = a + 1; b < n; ++b)
			if (rng() % 3 == 0) { // edges follow index order: a DAG
				reach[a][b] = true;
				edges.push_back({a, b});
			}
	std::set<std::pair<uint32, uint32>> direct;
	for (auto const &e : edges) direct.insert({e.from, e.to});
	for (std::size_t k = 0; k < n; ++k) // reachability for the certificates
		for (std::size_t i = 0; i < n; ++i)
			if (reach[i][k])
				for (std::size_t j = 0; j < n; ++j)
					if (reach[k][j]) reach[i][j] = true;

	// path cover: vertex-disjoint paths along direct edges, minimal by König
	// against the certified matching size
	auto const cover = g::minimum_path_cover(n, edges);
	auto const paths = cover.paths();
	REQUIRE(paths.size() == cover.count);
	REQUIRE(cover.count == n - g::bipartite_matching(n, n, edges).size);
	std::vector<uint8> seen(n, 0);
	for (auto const &path : paths)
		for (std::size_t i = 0; i < path.size(); ++i) {
			REQUIRE(!seen[path[i]]);
			seen[path[i]] = 1;
			if (i + 1 < path.size()) REQUIRE(direct.contains({path[i], path[i + 1]}));
		}
	for (auto const s : seen) REQUIRE(s);

	// chain cover + antichain: mutually certifying by Dilworth equality
	auto const dil = g::minimum_chain_cover(n, edges);
	auto const chains = dil.chains();
	REQUIRE(chains.size() == dil.count);
	std::ranges::fill(seen, uint8(0));
	for (auto const &chain : chains)
		for (std::size_t i = 0; i < chain.size(); ++i) {
			REQUIRE(!seen[chain[i]]);
			seen[chain[i]] = 1;
			if (i + 1 < chain.size()) REQUIRE(reach[chain[i]][chain[i + 1]]);
		}
	for (auto const s : seen) REQUIRE(s);
	std::size_t antichain = 0;
	for (uint32 a = 0; a < n; ++a) {
		antichain += dil.antichain[a];
		for (uint32 b = 0; b < n; ++b)
			if (dil.antichain[a] && dil.antichain[b] && a != b) REQUIRE(!reach[a][b]);
	}
	REQUIRE(antichain == dil.count);
}

TEST_CASE("bipartite_matching: edge classification across all maximum matchings", "[graph]") {
	auto seed = GENERATE(take(10, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const left = std::size_t(1 + rng() % 8), right = std::size_t(1 + rng() % 8);
	std::size_t const m = std::size_t(rng() % 18);
	std::vector<g::arc<>> edges;
	for (std::size_t i = 0; i < m; ++i)
		edges.push_back({uint32(rng() % left), uint32(rng() % right)}); // parallels legal

	auto const matching = g::bipartite_matching(left, right, edges);
	auto const kind = g::matching_edge_roles(left, right, edges, matching);
	REQUIRE(kind.size() == m);

	for (std::size_t i = 0; i < m; ++i) {
		// in SOME maximum matching <=> forcing the edge (removing everything
		// at its endpoints) still reaches the maximum with it included
		std::vector<g::arc<>> forced;
		for (std::size_t j = 0; j < m; ++j)
			if (edges[j].from != edges[i].from && edges[j].to != edges[i].to)
				forced.push_back(edges[j]);
		bool const in_some = 1 + flow_matching_size(left, right, forced) == matching.size;

		// in ALL maximum matchings <=> deleting this copy lowers the maximum
		std::vector<g::arc<>> without;
		for (std::size_t j = 0; j < m; ++j)
			if (j != i) without.push_back(edges[j]);
		bool const in_all = flow_matching_size(left, right, without) < matching.size;

		auto const want = !in_some ? g::matching_edge::forbidden
						: in_all   ? g::matching_edge::mandatory
								   : g::matching_edge::allowed;
		REQUIRE(kind[i] == want);
	}
}

TEST_CASE("bipartite_matching: edge classification canonical shapes", "[graph]") {
	std::vector<g::arc<>> lone{{0, 0}};
	auto const one = g::bipartite_matching(std::size_t(1), std::size_t(1), lone);
	REQUIRE(g::matching_edge_roles(1, 1, lone, one)[0] == g::matching_edge::mandatory);

	std::vector<g::arc<>> fork{{0, 0}, {0, 1}};
	auto const pick = g::bipartite_matching(std::size_t(1), std::size_t(2), fork);
	auto const forked = g::matching_edge_roles(1, 2, fork, pick);
	REQUIRE(forked[0] == g::matching_edge::allowed);
	REQUIRE(forked[1] == g::matching_edge::allowed);

	std::vector<g::arc<>> square{{0, 0}, {1, 1}, {0, 1}};
	auto const two = g::bipartite_matching(std::size_t(2), std::size_t(2), square);
	auto const chords = g::matching_edge_roles(2, 2, square, two);
	REQUIRE(chords[0] == g::matching_edge::mandatory);
	REQUIRE(chords[1] == g::matching_edge::mandatory);
	REQUIRE(chords[2] == g::matching_edge::forbidden);

	std::vector<g::arc<>> par{{0, 0}, {0, 0}};
	auto const dup = g::bipartite_matching(std::size_t(1), std::size_t(1), par);
	auto const copies = g::matching_edge_roles(1, 1, par, dup);
	REQUIRE(copies[0] == g::matching_edge::allowed);
	REQUIRE(copies[1] == g::matching_edge::allowed);
}

TEST_CASE("bipartite cover: a non-maximum or mis-sized matching is asserted", "[graph]") {
	std::vector<g::arc<>> const edges{{0, 0}, {1, 0}, {1, 1}};
	auto shrunk = g::bipartite_matching(2, 2, edges);
	REQUIRE(shrunk.size == 2);
	for (auto &r : shrunk.right_pair)
		if (r == 1) r = g::none; // drop the pair (1, 1)
	shrunk.left_pair[1] = g::none;
	shrunk.size = 1;
	REQUIRE(atl_test::aborts([&] { (void)g::minimum_vertex_cover(2, 2, edges, shrunk); }));
	auto const full = g::bipartite_matching(2, 2, edges);
	REQUIRE(atl_test::aborts([&] { (void)g::minimum_vertex_cover(2, 3, edges, full); }));
	REQUIRE(atl_test::aborts([&] { (void)g::matching_edge_roles(3, 2, edges, full); }));
}

TEST_CASE("bipartite matching uses heap stacks on million-vertex alternating paths", "[graph]") {
	atl::uint32 constexpr n = 1000000;
	std::vector<g::arc<>> edges;
	edges.reserve(2 * std::size_t(n) - 1);
	for (atl::uint32 v = 0; v + 1 < n; ++v) {
		edges.push_back({v, v});
		edges.push_back({v, v + 1});
	}
	edges.push_back({n - 1, 0});
	auto const certify_path = [](g::bipartite_matching_result const &r) {
		REQUIRE(r.size == n);
		for (atl::uint32 v = 0; v < n; ++v) {
			REQUIRE(r.left_pair[v] == (v + 1) % n);
			REQUIRE(r.right_pair[(v + 1) % n] == v);
		}
	};
	certify_path(g::bipartite_matching(n, n, edges));
	certify_path(ex::bipartite_matching_by_hopcroft_karp(n, n, edges));
	certify_path(ex::bipartite_matching_by_kuhn(n, n, edges));
	certify_path(ex::bipartite_matching_by_kuhn_passes(n, n, edges));
}

TEST_CASE("bipartite matching asserts id and DAG bounds", "[graph]") {
	std::vector<g::arc<>> const empty;
	REQUIRE(atl_test::aborts([&] { (void)g::bipartite_matching(g::none, 0, empty); }));
	REQUIRE(atl_test::aborts([&] { (void)g::bipartite_matching(0, g::none, empty); }));
	std::vector<g::arc<>> const invalid{{1, 0}};
	REQUIRE(atl_test::aborts([&] { (void)g::bipartite_matching(1, 1, invalid); }));
	std::vector<g::arc<>> const cycle{{0, 1}, {1, 0}};
	REQUIRE(atl_test::aborts([&] { (void)g::minimum_path_cover(2, cycle); }));
	REQUIRE(atl_test::aborts([&] { (void)g::minimum_chain_cover(2, cycle); }));
}
