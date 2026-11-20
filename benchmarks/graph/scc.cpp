/** @file
 * @brief SCC throughput: library single-array Pearce, Tarjan low-links,
 * Tarjan via DFS events, Gabow boundaries and Kosaraju transpose. Whole calls
 * are timed on random m=4e6, cycle plus chords and deep DAG inputs (n=1e6),
 * with nested-vector and forward-only row layouts; groups() is timed separately.
 * Library plain/native random: 163/169 ms; Gabow: 181/188 ms.
 * Intel Xeon Platinum 8280 @ 2.70 GHz, x86-64 Linux, GCC 15.2.0,
 * RelWithDebInfo, -O2 -g -DNDEBUG -std=c++23, also -march=native, 2026-10-15.
 * Full tables: docs/graph-components.md. Contenders are checked outside timing.
 */
#include <atl/experimental/graph/scc_variants.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/scc.hpp>
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
		for (uint32 v = 0; v < net.vertex_count(); ++v)
			for (uint32 const w : net.out_edges(v))
				if (b.component[v] > b.component[w]) return false;
		return true;
	}
}

TEST_CASE("scc contenders", "[scc]") {
	constexpr std::size_t n = 1'000'000;
	std::mt19937 rng(1);

	auto race = [&](char const *name, std::vector<g::arc<>> arcs) {
		graph_t net(n, arcs);
		auto const lib = g::scc(net);
		auto measure = [&]<typename Run>(char const *method, Run run)
			requires std::is_invocable_r_v<std::size_t, Run &>
		{
			BENCHMARK(std::string(name) + " | " + method) { return run(); };
		};
		REQUIRE(agrees(net, lib, ex::scc_by_tarjan_events(net)));
		REQUIRE(agrees(net, lib, ex::scc_by_tarjan(net)));
		REQUIRE(agrees(net, lib, ex::scc_by_gabow(net)));
		REQUIRE(agrees(net, lib, ex::scc_by_kosaraju(net)));

		measure("library", [&] { return g::scc(net).count(); });
		measure("by tarjan events", [&] { return ex::scc_by_tarjan_events(net).count(); });
		measure("by tarjan", [&] { return ex::scc_by_tarjan(net).count(); });
		measure("by gabow", [&] { return ex::scc_by_gabow(net).count(); });
		measure("by kosaraju", [&] { return ex::scc_by_kosaraju(net).count(); });

		std::vector<std::vector<g::vertex_id>> pushed(n);
		for (auto const &a : arcs) pushed[a.from].push_back(a.to);
		// copied so each row is allocated in vertex order, the layout
		// forward_rows gets too: rows grown by push_back scatter over the heap
		std::vector<std::vector<g::vertex_id>> const lists(pushed.begin(), pushed.end());
		auto const rows = atl_bench::forward_rows(lists);
		REQUIRE(agrees(net, lib, g::scc(lists)));
		REQUIRE(agrees(net, lib, g::scc(rows)));
		measure("library on nested vectors", [&] { return g::scc(lists).count(); });
		measure("library on forward-only rows", [&] { return g::scc(rows).count(); });
		measure("groups()", [&] { return lib.groups().size(); });
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
		race("deep DAG", std::move(arcs));
	}
}
