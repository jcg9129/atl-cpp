/** @file
 * @brief Block throughput: library Gabow boundaries, single-array state,
 * low-link arrays, paired vertex/index boundaries and two-pass chains (cuts only).
 * Whole calls are timed on random m=4e6, cycle plus chords, path plus chords,
 * random tree (n=1e6), and path plus chords n=4e6, using CSR, nested-vector
 * and forward-only rows. groups() is separate. Library plain/native n=4e6:
 * 213/216 ms; paired boundaries: 254/259 ms.
 * Intel Xeon Platinum 8280 @ 2.70 GHz, x86-64 Linux, GCC 15.2.0,
 * RelWithDebInfo, -O2 -g -DNDEBUG -std=c++23, also -march=native, 2026-10-15.
 * Full tables: docs/graph-components.md. Contenders are checked outside timing.
 */
#include <atl/experimental/graph/biconnected_components_variants.hpp>
#include <atl/graph/biconnected_components.hpp>
#include <atl/graph/csr.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "forward_rows.hpp"

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::uint32;

namespace {
	using graph_t = g::csr<g::vertex_id>;

	bool agrees(g::block_decomposition const &a, g::block_decomposition const &b) {
		if (a.count() != b.count()) return false;
		std::vector<uint32> map(a.count(), g::none);
		for (std::size_t v = 0; v < a.block.size(); ++v) {
			if ((a.block[v] == g::none) != (b.block[v] == g::none)) return false;
			if (a.block[v] == g::none) continue;
			if (map[a.block[v]] == g::none) map[a.block[v]] = b.block[v];
			if (map[a.block[v]] != b.block[v]) return false;
		}
		for (std::size_t c = 0; c < a.count(); ++c)
			if (b.top_vertices[map[c]] != a.top_vertices[c]) return false;
		return true;
	}
}

TEST_CASE("biconnected_components contenders", "[biconnected_components]") {
	constexpr std::size_t n = 1'000'000;
	std::mt19937 rng(1);

	auto race = [&](char const *name, std::vector<g::arc<>> arcs, std::size_t vertices) {
		auto const net = graph_t::undirected(vertices, arcs);
		auto const lib = g::biconnected_components(net);
		auto measure = [&]<typename Run>(char const *method, Run run)
			requires std::is_invocable_r_v<std::size_t, Run &>
		{
			BENCHMARK(std::string(name) + " | " + method) { return run(); };
		};
		REQUIRE(agrees(lib, ex::biconnected_components_by_single_array(net)));
		REQUIRE(agrees(lib, ex::biconnected_components_by_low_link(net)));
		REQUIRE(lib.cut_vertex_mask() == ex::cut_vertex_mask_by_chain_decomposition(net));

		measure("library", [&] { return g::biconnected_components(net).count(); });
		measure("by single array", [&] { return ex::biconnected_components_by_single_array(net).count(); });
		measure("by low link", [&] { return ex::biconnected_components_by_low_link(net).count(); });
		measure("by chain decomposition (cuts only)", [&] { return ex::cut_vertex_mask_by_chain_decomposition(net).size(); });

		std::vector<std::vector<g::vertex_id>> pushed(vertices);
		for (auto const &a : arcs) {
			pushed[a.from].push_back(a.to);
			pushed[a.to].push_back(a.from);
		}
		// copied so each row is allocated in vertex order, the layout
		// forward_rows gets too: rows grown by push_back scatter over the heap
		std::vector<std::vector<g::vertex_id>> const lists(pushed.begin(), pushed.end());
		auto const rows = atl_bench::forward_rows(lists);
		REQUIRE(agrees(lib, ex::biconnected_components_by_paired_boundaries(net)));
		REQUIRE(agrees(lib, g::biconnected_components(lists)));
		REQUIRE(agrees(lib, g::biconnected_components(rows)));
		REQUIRE(agrees(lib, ex::biconnected_components_by_paired_boundaries(rows)));
		measure("by paired boundaries", [&] { return ex::biconnected_components_by_paired_boundaries(net).count(); });
		measure("library on nested vectors", [&] { return g::biconnected_components(lists).count(); });
		measure("library on forward-only rows", [&] { return g::biconnected_components(rows).count(); });
		measure("by paired boundaries on forward-only rows", [&] { return ex::biconnected_components_by_paired_boundaries(rows).count(); });
		measure("groups()", [&] { return lib.groups().size(); });
	};

	{
		std::vector<g::arc<>> arcs;
		for (std::size_t i = 0; i < 4 * n; ++i)
			arcs.push_back({uint32(rng() % n), uint32(rng() % n)});
		race("random m=4e6", std::move(arcs), n);
	}
	{
		std::vector<g::arc<>> arcs;
		for (uint32 v = 0; v < n; ++v) arcs.push_back({v, uint32((v + 1) % n)});
		for (std::size_t i = 0; i < n; ++i) arcs.push_back({uint32(rng() % n), uint32(rng() % n)});
		race("giant cycle", std::move(arcs), n);
	}
	{
		std::vector<g::arc<>> arcs;
		for (uint32 v = 0; v + 1 < n; ++v) arcs.push_back({v, v + 1});
		for (std::size_t i = 0; i < n; ++i) {
			uint32 const a = uint32(rng() % (n - 1));
			arcs.push_back({a, uint32(a + 1 + rng() % (n - 1 - a))});
		}
		race("deep path + chords", std::move(arcs), n);
	}
	{
		std::vector<g::arc<>> arcs;
		for (uint32 v = 1; v < n; ++v) arcs.push_back({uint32(rng() % v), v});
		race("random tree", std::move(arcs), n);
	}
	{
		std::size_t const big = 4'000'000;
		std::vector<g::arc<>> arcs;
		for (uint32 v = 0; v + 1 < big; ++v) arcs.push_back({v, v + 1});
		for (std::size_t i = 0; i < big; ++i) {
			uint32 const a = uint32(rng() % (big - 1));
			arcs.push_back({a, uint32(a + 1 + rng() % (big - 1 - a))});
		}
		race("deep path + chords n=4e6", std::move(arcs), big);
	}
}
