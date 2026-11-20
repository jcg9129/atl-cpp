/**
 * @file
 * @brief Complete ternarization build latency for n=1e6 star, path, caterpillar
 * and random-parent trees. Unit-hop and int64 rows differ only in edge payload;
 * there is no competing builder. Construction includes BFS, child bucketing,
 * balanced fan-out and direct CSR emission. The same shape loop drives both rows.
 * Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
 * -O2 / -O2 -march=native, 2026-10-23: unit-hop star 39.4 / 39.8 ms,
 * int64 star 74.9 / 74.7 ms. Explicit-frame fan-out requires remeasurement. Full table: docs/centroid-decomposition.md.
 */
#include <atl/graph/csr.hpp>
#include <atl/graph/ternarization.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace g = atl::graph;
using atl::int64;
using atl::uint32;

TEST_CASE("ternarization build", "[graph]") {
	std::size_t const n = 1'000'000;
	std::mt19937 rng(1);
	auto race = [&](std::string const &name, std::vector<uint32> const &parent) {
		std::vector<g::arc<int64>> arcs;
		arcs.reserve(n - 1);
		for (uint32 v = 1; v < n; ++v) arcs.push_back({parent[v], v, int64(rng() % 1000)});
		auto const weighted = g::undirected_csr(n, arcs);
		std::vector<g::arc<>> hops;
		for (auto const &a : arcs) hops.push_back({a.from, a.to});
		auto const unweighted = g::undirected_csr(n, hops);
		BENCHMARK(name + " | unit hops") { return g::ternarization_of(unweighted, 0).real_count; };
		BENCHMARK(name + " | int64 weights") { return g::ternarization_of(weighted, 0).real_count; };
	};
	race("star", std::vector<uint32>(n, 0));
	{
		std::vector<uint32> parent(n, 0);
		for (uint32 v = 1; v < n; ++v) parent[v] = v - 1;
		race("path", parent);
	}
	{
		std::vector<uint32> parent(n, 0);
		for (uint32 v = 1; v < n; ++v) parent[v] = v % 2 ? v - 1 : v - 2;
		race("caterpillar", parent);
	}
	{
		std::vector<uint32> parent(n, 0);
		for (uint32 v = 1; v < n; ++v) parent[v] = uint32(rng() % v);
		race("random parent", parent);
	}
}
