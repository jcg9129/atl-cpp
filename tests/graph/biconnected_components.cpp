#include <atl/ds/dsu.hpp>
#include <atl/experimental/graph/biconnected_components_variants.hpp>
#include <atl/graph/biconnected_components.hpp>
#include <atl/graph/csr.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::uint32;
using atl::uint64;
using atl::uint8;

namespace {
	/** @brief Articulation points by definition: removing v must split its component. */
	std::vector<uint8> brute_cuts(std::size_t n, std::vector<g::arc<>> const &arcs) {
		atl::dsu whole(n);
		for (auto const &a : arcs) whole.unite(a.from, a.to);
		std::vector<uint8> cut(n, 0);
		for (g::vertex_id v = 0; v < n; ++v) {
			atl::dsu d(n);
			for (auto const &a : arcs)
				if (a.from != v && a.to != v) d.unite(a.from, a.to);
			cut[v] = d.count() - 1 > whole.count();
		}
		return cut;
	}

	/** @brief Blocks by definition: two edges sharing vertex v are in one block iff
their other endpoints stay connected without v; blocks are the closure.
Self-loops are excluded (they belong to no block). */
	std::vector<uint32> brute_blocks(std::size_t n, std::vector<g::arc<>> const &arcs) {
		atl::dsu classes(arcs.size());
		for (g::vertex_id v = 0; v < n; ++v) {
			atl::dsu d(n);
			for (auto const &a : arcs)
				if (a.from != v && a.to != v) d.unite(a.from, a.to);
			for (std::size_t i = 0; i < arcs.size(); ++i) {
				if (arcs[i].from == arcs[i].to) continue;
				if (arcs[i].from != v && arcs[i].to != v) continue;
				for (std::size_t j = i + 1; j < arcs.size(); ++j) {
					if (arcs[j].from == arcs[j].to) continue;
					if (arcs[j].from != v && arcs[j].to != v) continue;
					auto const other_endpoint = [v](g::arc<> const &a) { return a.from == v ? a.to : a.from; };
					if (d.same(other_endpoint(arcs[i]), other_endpoint(arcs[j]))) classes.unite(i, j);
				}
			}
		}
		std::vector<uint32> label(arcs.size(), g::none);
		for (std::size_t i = 0; i < arcs.size(); ++i)
			if (arcs[i].from != arcs[i].to) label[i] = uint32(classes.find(i));
		return label;
	}

	/** @brief Per-edge block ids recovered from the per-vertex encoding through
`edge_block`; self-loops belong to no block. */
	std::vector<uint32> edge_blocks(std::vector<g::arc<>> const &arcs, g::block_decomposition const &r) {
		std::vector<uint32> eb(arcs.size(), g::none);
		for (std::size_t i = 0; i < arcs.size(); ++i) {
			if (arcs[i].from == arcs[i].to) continue;
			eb[i] = r.edge_block(arcs[i].from, arcs[i].to);
			REQUIRE(eb[i] < r.count());
			REQUIRE(eb[i] == r.edge_block(arcs[i].to, arcs[i].from));
		}
		return eb;
	}

	/** @brief Same partition of {0..k-1} up to relabeling, `none` matching `none`. */
	bool same_labeling(std::vector<uint32> const &a, std::vector<uint32> const &b) {
		if (a.size() != b.size()) return false;
		for (std::size_t i = 0; i < a.size(); ++i)
			if ((a[i] == g::none) != (b[i] == g::none)) return false;
		for (std::size_t i = 0; i < a.size(); ++i)
			for (std::size_t j = i + 1; j < a.size(); ++j)
				if (a[i] != g::none && b[j] != g::none && (a[i] == a[j]) != (b[i] == b[j])) return false;
		return true;
	}

	bool agrees(g::block_decomposition const &a, g::block_decomposition const &b) {
		if (a.count() != b.count()) return false;
		if (!same_labeling(a.block, b.block)) return false;
		std::vector<uint32> map(a.count(), g::none);
		for (std::size_t v = 0; v < a.block.size(); ++v)
			if (a.block[v] != g::none) map[a.block[v]] = b.block[v];
		for (std::size_t c = 0; c < a.count(); ++c)
			if (a.top_vertices[c] != b.top_vertices[map[c]]) return false;
		return true;
	}
}

TEST_CASE("biconnected_components: blocks and cuts match the definitional oracle", "[graph]") {
	auto seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = std::size_t(1 + rng() % 25);
	std::size_t const m = std::size_t(rng() % (3 * n));
	std::vector<g::arc<>> arcs;
	for (std::size_t i = 0; i < m; ++i)
		arcs.push_back({g::vertex_id(rng() % n), g::vertex_id(rng() % n)});
	auto const net = g::undirected_csr(n, arcs);

	auto const r = g::biconnected_components(net);

	REQUIRE(r.block.size() == n);
	for (g::vertex_id v = 0; v < n; ++v)
		if (r.block[v] != g::none) REQUIRE(r.block[v] < r.count());

	REQUIRE(same_labeling(edge_blocks(arcs, r), brute_blocks(n, arcs)));

	auto const grouped = r.groups();
	REQUIRE(grouped.size() == r.count());
	for (std::size_t b = 0; b < grouped.size(); ++b) {
		REQUIRE(grouped[b].size() >= 2);
		REQUIRE(grouped[b].front() == r.top_vertices[b]);
		for (std::size_t i = 1; i < grouped[b].size(); ++i) REQUIRE(r.block[grouped[b][i]] == b);
	}

	auto const flat = r.grouped();
	REQUIRE(flat.count() == grouped.size());
	for (std::size_t b = 0; b < grouped.size(); ++b)
		REQUIRE(std::vector<g::vertex_id>(flat[b].begin(), flat[b].end()) == grouped[b]);

	REQUIRE(r.cut_vertex_mask() == brute_cuts(n, arcs));

	REQUIRE(agrees(r, ex::biconnected_components_by_single_array(net)));
	REQUIRE(agrees(r, ex::biconnected_components_by_low_link(net)));
	REQUIRE(agrees(r, ex::biconnected_components_by_paired_boundaries(net)));
	REQUIRE(ex::cut_vertex_mask_by_chain_decomposition(net) == brute_cuts(n, arcs));
	std::vector<std::vector<g::vertex_id>> lists(n);
	for (auto const &a : arcs) {
		lists[a.from].push_back(a.to);
		lists[a.to].push_back(a.from);
	}
	REQUIRE(same_labeling(edge_blocks(arcs, g::biconnected_components(lists)), brute_blocks(n, arcs)));
}

TEST_CASE("biconnected_components: canonical shapes", "[graph]") {
	std::size_t const n = std::size_t(1000);
	std::vector<g::arc<>> path;
	for (g::vertex_id v = 0; v + 1 < n; ++v) path.push_back({v, g::vertex_id(v + 1)});
	auto const line = g::biconnected_components(g::undirected_csr(n, path));
	REQUIRE(line.count() == n - 1);
	auto const line_cut = line.cut_vertex_mask();
	for (g::vertex_id v = 0; v < n; ++v) REQUIRE(line_cut[v] == (v != 0 && v != n - 1));

	std::vector<g::arc<>> cyc = path;
	cyc.push_back({g::vertex_id(n - 1), 0});
	auto const ring = g::biconnected_components(g::undirected_csr(n, cyc));
	REQUIRE(ring.count() == 1);
	for (auto const c : ring.cut_vertex_mask()) REQUIRE(!c);

	std::vector<g::arc<>> bow{{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 2}};
	auto const tie = g::biconnected_components(g::undirected_csr(std::size_t(5), bow));
	REQUIRE(tie.count() == 2);
	REQUIRE(tie.cut_vertex_mask() == std::vector<uint8>{0, 0, 1, 0, 0});
	REQUIRE(tie.edge_block(0, 2) == tie.edge_block(1, 2));
	REQUIRE(tie.edge_block(2, 3) == tie.edge_block(4, 2));
	REQUIRE(tie.edge_block(0, 1) != tie.edge_block(3, 4));

	std::vector<g::arc<>> bridge_path{{0, 1}, {1, 2}, {2, 3}};
	auto const chain3 = g::biconnected_components(g::undirected_csr(std::size_t(4), bridge_path));
	REQUIRE(chain3.count() == 3);
	REQUIRE(chain3.edge_block(1, 2) != chain3.edge_block(0, 1));
	REQUIRE(chain3.edge_block(2, 1) != chain3.edge_block(2, 3));
	REQUIRE(atl_test::aborts([&] { (void)chain3.edge_block(1, 1); }));

	std::vector<g::arc<>> multi{{0, 1}, {0, 1}, {1, 2}};
	auto const dbl = g::biconnected_components(g::undirected_csr(std::size_t(3), multi));
	REQUIRE(dbl.count() == 2);
	REQUIRE(dbl.block[2] != dbl.block[1]);
	REQUIRE(dbl.cut_vertex_mask() == std::vector<uint8>{0, 1, 0});

	std::vector<g::arc<>> loops{{0, 0}, {1, 2}};
	auto const sparse = g::biconnected_components(g::undirected_csr(std::size_t(4), loops));
	REQUIRE(sparse.count() == 1);
	REQUIRE(sparse.block[0] == g::none);
	REQUIRE(sparse.block[3] == g::none);

	std::size_t const deep = 1'000'000;
	std::vector<std::vector<g::vertex_id>> chain(deep);
	for (g::vertex_id v = 0; v + 1 < deep; ++v) {
		chain[v].push_back(v + 1);
		chain[v + 1].push_back(v);
	}
	REQUIRE(g::biconnected_components(chain).count() == deep - 1);

	chain[deep - 1].push_back(0);
	chain[0].push_back(g::vertex_id(deep - 1));
	REQUIRE(g::biconnected_components(chain).count() == 1);
}

TEST_CASE("biconnected_components: an out-of-range target is asserted", "[graph]") {
	std::vector<std::vector<int>> const bad{{1}, {7}};
	REQUIRE(atl_test::aborts([&] { (void)g::biconnected_components(bad); }));
	REQUIRE(atl_test::aborts([&] { (void)ex::biconnected_components_by_paired_boundaries(bad); }));
}

TEST_CASE("block decomposition: roots, empty graphs and saturated cuts", "[graph]") {
	std::vector<g::arc<>> const arcs{{0, 1}, {0, 2}, {0, 3}};
	auto const star = g::biconnected_components(g::undirected_csr(4, arcs));
	REQUIRE(star.block[0] == g::none);
	REQUIRE(star.cut_vertex_mask() == std::vector<uint8>{1, 0, 0, 0});
	REQUIRE(star.grouped().members.size() == 6);
	std::vector<std::vector<int>> const empty;
	REQUIRE(g::biconnected_components(empty).count() == 0);
}

TEST_CASE("block decomposition: articulation counts do not wrap at 256 blocks", "[graph]") {
	std::vector<g::arc<>> arcs;
	for (g::vertex_id v = 1; v <= 300; ++v) arcs.push_back({0, v});
	auto const star = g::biconnected_components(g::undirected_csr(301, arcs));
	std::vector<uint8> expected(301, 0);
	expected[0] = 1;
	REQUIRE(star.count() == 300);
	REQUIRE(star.cut_vertex_mask() == expected);
	REQUIRE(star.grouped().members.size() == 600);
}
