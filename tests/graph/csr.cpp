#include <atl/experimental/graph/csr_variants.hpp>
#include <atl/graph/csr.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	/** @brief Every row of @p h equals the same row of the list-of-lists oracle. */
	template<g::adjacency_edge Edge>
	void same_rows(g::csr<Edge> const &h, std::vector<std::vector<Edge>> const &want) {
		REQUIRE(h.vertex_count() == want.size());
		std::size_t total = 0;
		for (uint32 u = 0; u < want.size(); ++u) {
			auto const row = h.out_edges(u);
			REQUIRE(h.out_degree(u) == want[u].size());
			REQUIRE(row.size() == want[u].size());
			for (std::size_t i = 0; i < row.size(); ++i) {
				REQUIRE(g::target(row[i]) == g::target(want[u][i]));
				if constexpr (g::weighted_adjacency_edge<Edge>) REQUIRE(g::weight(row[i]) == g::weight(want[u][i]));
			}
			total += row.size();
		}
		REQUIRE(h.edge_count() == total);
	}

	/** @brief The raw arrays of a variant build agree with the library class row by row. */
	template<g::weighted_adjacency_edge Edge>
	void same_arrays(ex::csr_arrays<Edge> const &a, g::csr<Edge> const &h) {
		REQUIRE(a.offset.size() == h.vertex_count() + 1);
		REQUIRE(a.offset.back() == h.edge_count());
		for (uint32 u = 0; u < h.vertex_count(); ++u) {
			auto const row = h.out_edges(u);
			REQUIRE(a.offset[u + 1] - a.offset[u] == row.size());
			for (std::size_t i = 0; i < row.size(); ++i) {
				REQUIRE(g::target(a.edges[a.offset[u] + i]) == g::target(row[i]));
				REQUIRE(g::weight(a.edges[a.offset[u] + i]) == g::weight(row[i]));
			}
		}
	}
}

TEST_CASE("csr: random edge lists against a vector<vector> oracle", "[graph]") {
	auto seed = GENERATE(take(20, random(uint64(0), ~uint64(0))));
	std::mt19937_64 rng(seed);
	std::size_t const n = std::size_t(rng() % 50); // n = 0 included
	std::size_t const m = n == 0 ? 0 : std::size_t(rng() % 200);
	std::vector<g::arc<int64>> arcs(m);
	for (auto &a : arcs) a = {uint32(rng() % n), uint32(rng() % n), int64(rng() % 1000) - 500};

	std::vector<std::vector<g::edge<int64>>> directed(n), both(n);
	std::vector<std::vector<g::vertex_id>> heads(n);
	for (auto const &a : arcs) {
		directed[a.from].push_back({a.to, a.weight});
		heads[a.from].push_back(a.to);
		both[a.from].push_back({a.to, a.weight});
		both[a.to].push_back({a.from, a.weight});
	}

	g::csr const weighted(n, arcs);
	same_rows(weighted, directed);
	auto const undirected = g::undirected_csr(n, arcs);
	same_rows(undirected, both);
	g::csr<g::vertex_id> const projected(n, arcs, [](g::arc<int64> const &a) { return a.to; });
	same_rows(projected, heads);

	same_arrays(ex::csr_by_cursor_copy<g::edge<int64>, false>(n, arcs), weighted);
	same_arrays(ex::csr_by_cursor_copy<g::edge<int64>, true>(n, arcs), undirected);
	same_arrays(ex::csr_by_bucket_gather<g::edge<int64>, false>(n, arcs), weighted);
	same_arrays(ex::csr_by_bucket_gather<g::edge<int64>, true>(n, arcs), undirected);
}

TEST_CASE("csr: empty graphs", "[graph]") {
	std::vector<g::arc<>> const none;
	g::csr const zero(0, none);
	REQUIRE(zero.vertex_count() == 0);
	REQUIRE(zero.edge_count() == 0);
	g::csr const isolated(5, none);
	REQUIRE(isolated.vertex_count() == 5);
	REQUIRE(isolated.edge_count() == 0);
	for (uint32 u = 0; u < 5; ++u) {
		REQUIRE(isolated.out_degree(u) == 0);
		REQUIRE(isolated.out_edges(u).empty());
	}
}

TEST_CASE("csr::adopt: takes ownership of prebuilt arrays", "[graph]") {
	std::mt19937 rng(5);
	for (int iter = 0; iter < 200; ++iter) {
		std::size_t const n = rng() % 20, m = n == 0 ? 0 : rng() % 60;
		std::vector<g::arc<int>> arcs;
		for (std::size_t i = 0; i < m; ++i) arcs.push_back({uint32(rng() % n), uint32(rng() % n), int(rng() % 100)});
		g::csr<g::edge<int>> const built(n, arcs);
		std::vector<uint32> offsets(n + 1, 0);
		for (auto const &a : arcs) ++offsets[a.from + 1];
		for (std::size_t u = 0; u < n; ++u) offsets[u + 1] += offsets[u];
		std::vector<g::edge<int>> edges(m);
		auto cursor = offsets;
		for (auto const &a : arcs) edges[cursor[a.from]++] = {a.to, a.weight};
		auto const adopted = g::csr<g::edge<int>>::adopt(std::move(offsets), std::move(edges));
		REQUIRE(adopted.vertex_count() == n);
		REQUIRE(adopted.edge_count() == m);
		for (uint32 u = 0; u < n; ++u) {
			auto const a = adopted.out_edges(u), b = built.out_edges(u);
			REQUIRE(a.size() == b.size());
			for (std::size_t i = 0; i < a.size(); ++i) REQUIRE((a[i].to == b[i].to && a[i].weight == b[i].weight));
		}
	}
	REQUIRE(g::csr<uint32>::adopt({0}, {}).vertex_count() == 0);
}

TEST_CASE("csr::adopt: broken invariants trip the asserts", "[graph]") {
	using csr_t = g::csr<uint32>;
	CHECK(atl_test::aborts([] { (void)csr_t::adopt({}, {}); }));
	CHECK(atl_test::aborts([] { (void)csr_t::adopt({1, 1}, {0}); }));		   // does not start at 0
	CHECK(atl_test::aborts([] { (void)csr_t::adopt({0, 1}, {0, 0}); }));	   // does not end at edges.size()
	CHECK(atl_test::aborts([] { (void)csr_t::adopt({0, 2, 1, 2}, {0, 1}); })); // decreasing
	CHECK(atl_test::aborts([] { (void)csr_t::adopt({0, 1}, {1}); }));		   // target out of range
}

TEST_CASE("csr: narrow stored targets are checked", "[graph]") {
	std::vector<g::arc<>> const arcs{{0, 40000}};
	REQUIRE(atl_test::aborts([&] { g::csr<short> const rows(40001, arcs); (void)rows; }));
}
