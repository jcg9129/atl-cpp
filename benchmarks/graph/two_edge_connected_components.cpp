/** @file
 * @brief Bridge-component throughput: library single-array Pearce,
 * low-link arrays, Gabow boundaries and two-pass chain decomposition. Whole
 * calls are timed on random m=4e6, cycle plus chords, deep path plus chords
 * and random tree inputs (n=1e6), with CSR, nested-vector and forward-only rows.
 * Library plain/native random tree: 57.4/60.0 ms; Gabow: 68.4/69.7 ms.
 * Intel Xeon Platinum 8280 @ 2.70 GHz, x86-64 Linux, GCC 15.2.0,
 * RelWithDebInfo, -O2 -g -DNDEBUG -std=c++23, also -march=native, 2026-10-15.
 * Full tables: docs/graph-components.md. Contenders are checked outside timing.
 */
#include <atl/experimental/graph/two_edge_connected_components_variants.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/two_edge_connected_components.hpp>
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

	bool agrees(graph_t const &net, g::vertex_partition const &a, g::vertex_partition const &b) {
		if (a.count() != b.count()) return false;
		std::vector<uint32> map(a.count(), g::none);
		for (uint32 v = 0; v < net.vertex_count(); ++v) {
			if (map[a.component[v]] == g::none) map[a.component[v]] = b.component[v];
			if (map[a.component[v]] != b.component[v]) return false;
		}
		return true;
	}
}

TEST_CASE("two_edge_connected_components contenders", "[two_edge_connected_components]") {
	constexpr std::size_t n = 1'000'000;
	std::mt19937 rng(1);

	auto race = [&](char const *name, std::vector<g::arc<>> arcs) {
		auto const net = graph_t::undirected(n, arcs);
		auto const lib = g::two_edge_connected_components(net);
		auto measure = [&]<typename Run>(char const *method, Run run)
			requires std::is_invocable_r_v<std::size_t, Run &>
		{
			BENCHMARK(std::string(name) + " | " + method) { return run(); };
		};
		REQUIRE(agrees(net, lib, ex::two_edge_connected_components_by_low_link(net)));
		REQUIRE(agrees(net, lib, ex::two_edge_connected_components_by_gabow(net)));
		REQUIRE(agrees(net, lib, ex::two_edge_connected_components_by_chain_decomposition(net)));

		measure("library", [&] { return g::two_edge_connected_components(net).count(); });
		measure("by low link", [&] { return ex::two_edge_connected_components_by_low_link(net).count(); });
		measure("by gabow", [&] { return ex::two_edge_connected_components_by_gabow(net).count(); });
		measure("by chain decomposition", [&] { return ex::two_edge_connected_components_by_chain_decomposition(net).count(); });

		std::vector<std::vector<g::vertex_id>> pushed(n);
		for (auto const &a : arcs) {
			pushed[a.from].push_back(a.to);
			pushed[a.to].push_back(a.from);
		}
		// copied so each row is allocated in vertex order, the layout
		// forward_rows gets too: rows grown by push_back scatter over the heap
		std::vector<std::vector<g::vertex_id>> const lists(pushed.begin(), pushed.end());
		REQUIRE(agrees(net, lib, g::two_edge_connected_components(lists)));
		measure("library on nested vectors", [&] { return g::two_edge_connected_components(lists).count(); });
		auto const rows = atl_bench::forward_rows(lists);
		REQUIRE(agrees(net, lib, g::two_edge_connected_components(rows)));
		measure("library on forward-only rows", [&] { return g::two_edge_connected_components(rows).count(); });
	};

	{
		std::vector<g::arc<>> arcs;
		for (std::size_t i = 0; i < 4 * n; ++i)
			arcs.push_back({uint32(rng() % n), uint32(rng() % n)});
		race("random m=4e6", std::move(arcs));
	}
	{
		std::vector<g::arc<>> arcs;
		for (uint32 v = 0; v < n; ++v) arcs.push_back({v, uint32((v + 1) % n)});
		for (std::size_t i = 0; i < n; ++i) arcs.push_back({uint32(rng() % n), uint32(rng() % n)});
		race("giant cycle", std::move(arcs));
	}
	{
		std::vector<g::arc<>> arcs;
		for (uint32 v = 0; v + 1 < n; ++v) arcs.push_back({v, v + 1});
		for (std::size_t i = 0; i < n; ++i) {
			uint32 const a = uint32(rng() % (n - 1));
			arcs.push_back({a, uint32(a + 1 + rng() % (n - 1 - a))});
		}
		race("deep path + chords", std::move(arcs));
	}
	{
		std::vector<g::arc<>> arcs;
		for (uint32 v = 1; v < n; ++v) arcs.push_back({uint32(rng() % v), v});
		race("random tree", std::move(arcs));
	}
}
