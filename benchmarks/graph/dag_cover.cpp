/**
 * @file
 * @brief Races reverse-topological bitsets with arc or CSR output, reached-target
 * skipping and target ordering, Floyd-Warshall and per-source DFS; also races
 * scatter versus bucket CSR construction and complete chain covers with shared
 * CSR or materialized arcs. Measures complete-operation latency including
 * allocation on sparse, dense, chain and complete DAGs. Xeon Platinum 8280,
 * GCC 15.2, RelWithDebInfo, assertions off, 2026-10-15,
 * -O2 -g -DNDEBUG -std=c++23 (native adds -march=native): dense n=5000
 * library closure 61.3 / 60.3 ms versus arc output 271 / 253; complete
 * chain cover 24.8 / 24.2 versus arc lists 80.7 / 72.8.
 * Full tables: docs/bipartite-matching.md.
 */
#include <atl/graph/bipartite_matching.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/counting_sort.hpp>
#include <algorithm>
#include <bit>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace g = atl::graph;
using atl::uint32;
using atl::uint64;
using atl::uint8;

namespace {
	template<typename Run>
		requires std::invocable<Run &>
	void measure(std::string const &label, Run run) {
		BENCHMARK(std::string(label)) { return run(); };
	}

	/** @brief Transitive closure by all-pairs bitset row propagation. */
	[[nodiscard]] std::vector<g::arc<>> transitive_closure_by_floyd_warshall(std::size_t n, std::vector<g::arc<>> const &edges) {
		std::size_t const words = (n + 63) / 64;
		std::vector<uint64> reach(n * words, 0);
		for (auto const &e : edges) reach[e.from * words + e.to / 64] |= uint64(1) << (e.to % 64);
		for (uint32 k = 0; k < n; ++k)
			for (uint32 i = 0; i < n; ++i)
				if (reach[i * words + k / 64] >> (k % 64) & 1)
					for (std::size_t w = 0; w < words; ++w) reach[i * words + w] |= reach[k * words + w];
		std::vector<g::arc<>> closed;
		for (uint32 u = 0; u < n; ++u)
			for (std::size_t w = 0; w < words; ++w)
				for (uint64 row = reach[u * words + w]; row; row &= row - 1)
					closed.push_back({u, uint32(w * 64) + uint32(std::countr_zero(row))});
		return closed;
	}

	/** @brief Reverse-topological bitsets with optional reached-target skipping and topological target ordering, materializing arcs. */
	template<bool IsSkip, bool IsNearestFirst>
	[[nodiscard]] std::vector<g::arc<>> transitive_closure_by_topological_bitsets(std::size_t n, std::vector<g::arc<>> const &edges) {
		std::vector<uint32> offset(n + 1, 0), to(edges.size()), indegree(n, 0);
		for (auto const &e : edges) {
			++offset[e.from + 1];
			++indegree[e.to];
		}
		for (std::size_t v = 0; v < n; ++v) offset[v + 1] += offset[v];
		{
			std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
			for (auto const &e : edges) to[cursor[e.from]++] = e.to;
		}
		std::vector<uint32> topo;
		topo.reserve(n);
		for (uint32 v = 0; v < n; ++v)
			if (indegree[v] == 0) topo.push_back(v);
		for (std::size_t at = 0; at < topo.size(); ++at)
			for (uint32 k = offset[topo[at]]; k < offset[topo[at] + 1]; ++k)
				if (--indegree[to[k]] == 0) topo.push_back(to[k]);
		if constexpr (IsNearestFirst) {
			std::vector<uint32> rank(n), cursor(offset.begin(), offset.end() - 1), from_of(edges.size());
			for (uint32 i = 0; i < n; ++i) rank[topo[i]] = i;
			std::vector<uint32> in_offset(n + 1, 0), in_from(edges.size());
			for (auto const &e : edges) ++in_offset[e.to + 1];
			for (std::size_t v = 0; v < n; ++v) in_offset[v + 1] += in_offset[v];
			{
				std::vector<uint32> c(in_offset.begin(), in_offset.end() - 1);
				for (auto const &e : edges) in_from[c[e.to]++] = e.from;
			}
			for (uint32 i = 0; i < n; ++i) {
				uint32 const v = topo[i];
				for (uint32 k = in_offset[v]; k < in_offset[v + 1]; ++k) to[cursor[in_from[k]]++] = v;
			}
		}
		std::size_t const words = (n + 63) / 64;
		std::vector<uint64> reach(n * words, 0);
		for (std::size_t i = topo.size(); i-- > 0;) {
			uint32 const u = topo[i];
			for (uint32 k = offset[u]; k < offset[u + 1]; ++k) {
				uint32 const v = to[k];
				uint64 &word = reach[u * words + v / 64];
				uint64 const bit = uint64(1) << (v % 64);
				if constexpr (IsSkip)
					if (word & bit) continue;
				word |= bit;
				for (std::size_t w = 0; w < words; ++w) reach[u * words + w] |= reach[v * words + w];
			}
		}
		std::size_t bits = 0;
		for (auto const w : reach) bits += std::size_t(std::popcount(w));
		std::vector<g::arc<>> closed;
		closed.reserve(bits);
		for (uint32 u = 0; u < n; ++u)
			for (std::size_t w = 0; w < words; ++w)
				for (uint64 row = reach[u * words + w]; row; row &= row - 1)
					closed.push_back({u, uint32(w * 64) + uint32(std::countr_zero(row))});
		return closed;
	}

	/** @brief Materialize closure arcs separately for matching and independent-set construction. */
	[[nodiscard]] std::size_t minimum_chain_cover_by_arc_lists(std::size_t n, std::vector<g::arc<>> const &edges) {
		auto const closed = transitive_closure_by_topological_bitsets<false, false>(n, edges);
		auto const match = g::bipartite_matching(n, n, closed);
		auto const is = g::maximum_independent_set(n, n, closed, match);
		std::size_t antichain = 0;
		for (uint32 v = 0; v < n; ++v) antichain += is.left[v] && is.right[v];
		return n - match.size + antichain;
	}

	/** @brief Build both matching CSR directions by scatter or by bucket grouping. */
	template<bool IsBucketed>
	[[nodiscard]] std::size_t matching_csr_by_grouping(std::size_t n, std::vector<g::arc<>> const &edges) {
		std::size_t const m = edges.size();
		std::vector<uint32> lo(n + 1, 0), lt(m), ro(n + 1, 0), rt(m);
		if constexpr (IsBucketed) {
			std::vector<uint32> order(m);
			atl::detail::bucket_by_key(
				n, [&](std::size_t i) { return std::size_t(edges[i].from); }, std::span<uint32>(lo),
				std::span<uint32>(order));
			for (std::size_t k = 0; k < m; ++k) lt[k] = edges[order[k]].to;
			atl::detail::bucket_by_key(
				n, [&](std::size_t i) { return std::size_t(edges[i].to); }, std::span<uint32>(ro),
				std::span<uint32>(order));
			for (std::size_t k = 0; k < m; ++k) rt[k] = edges[order[k]].from;
		} else {
			for (auto const &e : edges) {
				++lo[e.from + 1];
				++ro[e.to + 1];
			}
			for (std::size_t v = 0; v < n; ++v) {
				lo[v + 1] += lo[v];
				ro[v + 1] += ro[v];
			}
			std::vector<uint32> cl(lo.begin(), lo.end() - 1), cr(ro.begin(), ro.end() - 1);
			for (auto const &e : edges) {
				lt[cl[e.from]++] = e.to;
				rt[cr[e.to]++] = e.from;
			}
		}
		return lt[m / 2] + rt[m / 3] + lo[n / 2] + ro[n / 3];
	}

	/** @brief Transitive closure by explicit-stack DFS from each source. */
	[[nodiscard]] std::vector<g::arc<>> transitive_closure_by_dfs(std::size_t n, std::vector<g::arc<>> const &edges) {
		std::vector<uint32> offset(n + 1, 0), to(edges.size());
		for (auto const &e : edges) ++offset[e.from + 1];
		for (std::size_t v = 0; v < n; ++v) offset[v + 1] += offset[v];
		{
			std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
			for (auto const &e : edges) to[cursor[e.from]++] = e.to;
		}
		std::vector<g::arc<>> closed;
		std::vector<uint8> seen(n);
		std::vector<uint32> stack;
		for (uint32 s = 0; s < n; ++s) {
			std::ranges::fill(seen, uint8(0));
			stack.assign(1, s);
			seen[s] = 1;
			while (!stack.empty()) {
				uint32 const v = stack.back();
				stack.pop_back();
				for (uint32 k = offset[v]; k < offset[v + 1]; ++k)
					if (!seen[to[k]]) {
						seen[to[k]] = 1;
						closed.push_back({s, to[k]});
						stack.push_back(to[k]);
					}
			}
		}
		return closed;
	}
}

TEST_CASE("dag_cover contenders", "[dag_cover]") {
	std::mt19937 rng(1);

	auto phases = [&](std::string const &name, std::size_t n, std::vector<g::arc<>> const &edges) {
		auto const a = transitive_closure_by_topological_bitsets<false, false>(n, edges);
		{
			auto const [offset, to] = g::detail::closure_csr(n, edges);
			REQUIRE(to.size() == a.size());
			for (uint32 u = 0, k = 0; u < n; ++u)
				for (; k < offset[u + 1]; ++k) REQUIRE((a[k].from == u && a[k].to == to[k]));
		}
		auto const same = [&](std::vector<g::arc<>> const &b) {
			return std::ranges::equal(a, b, [](g::arc<> const &x, g::arc<> const &y) { return x.from == y.from && x.to == y.to; });
		};
		REQUIRE(same(transitive_closure_by_topological_bitsets<true, false>(n, edges)));
		REQUIRE(same(transitive_closure_by_topological_bitsets<true, true>(n, edges)));
		measure(name + " | closure: library", [&] { return g::detail::closure_csr(n, edges).second.size(); });
		measure(name + " | closure: by topological bitsets", [&] { return transitive_closure_by_topological_bitsets<false, false>(n, edges).size(); });
		measure(name + " | closure: by topological bitsets skip", [&] { return transitive_closure_by_topological_bitsets<true, false>(n, edges).size(); });
		measure(name + " | closure: by topological bitsets nearest first", [&] { return transitive_closure_by_topological_bitsets<true, true>(n, edges).size(); });
		{
			auto shuffled = a; // edge-list order as a caller would hand it over
			std::ranges::shuffle(shuffled, rng);
			REQUIRE(matching_csr_by_grouping<false>(n, shuffled) == matching_csr_by_grouping<true>(n, shuffled));
			measure(name + " | matching CSR build: by scatter", [&] { return matching_csr_by_grouping<false>(n, shuffled); });
			measure(name + " | matching CSR build: by bucket grouping", [&] { return matching_csr_by_grouping<true>(n, shuffled); });
		}
		auto const match = g::bipartite_matching(n, n, a);
		measure(name + " | phase: matching on the closure", [&] { return g::bipartite_matching(n, n, a).size; });
		measure(name + " | phase: independent set on the closure", [&] { return g::maximum_independent_set(n, n, a, match).size; });
		{
			auto const cc = g::minimum_chain_cover(n, edges);
			std::size_t antichain = 0;
			for (auto const x : cc.antichain) antichain += x;
			REQUIRE(cc.count + antichain == minimum_chain_cover_by_arc_lists(n, edges));
		}
		measure(name + " | minimum_chain_cover library", [&] { return g::minimum_chain_cover(n, edges).count; });
		measure(name + " | minimum_chain_cover by arc lists", [&] { return minimum_chain_cover_by_arc_lists(n, edges); });
	};

	auto race = [&](char const *name, std::size_t n, std::vector<g::arc<>> edges) {
		auto const a = transitive_closure_by_topological_bitsets<false, false>(n, edges);
		auto const same_closure = [&](std::vector<g::arc<>> b) {
			std::ranges::sort(b, {}, [](g::arc<> const &e) { return std::pair(e.from, e.to); });
			return std::ranges::equal(a, b, [](g::arc<> const &x, g::arc<> const &y) { return x.from == y.from && x.to == y.to; });
		};
		REQUIRE(same_closure(transitive_closure_by_floyd_warshall(n, edges)));
		REQUIRE(same_closure(transitive_closure_by_dfs(n, edges)));
		phases(name, n, edges);

		measure(std::string(name) + " | closure: by floyd warshall", [&] { return transitive_closure_by_floyd_warshall(n, edges).size(); });
		measure(std::string(name) + " | closure: by dfs", [&] { return transitive_closure_by_dfs(n, edges).size(); });
	};

	{
		std::size_t const n = 2'000;
		std::vector<g::arc<>> edges;
		for (std::size_t i = 0; i < 10'000; ++i) {
			uint32 const a = uint32(rng() % (n - 1));
			edges.push_back({a, a + 1 + uint32(rng() % (n - 1 - a))});
		}
		race("sparse n=2e3 m=1e4", n, std::move(edges));
	}
	{
		std::size_t const n = 2'000;
		std::vector<g::arc<>> edges;
		for (std::size_t i = 0; i < 200'000; ++i) {
			uint32 const a = uint32(rng() % (n - 1));
			edges.push_back({a, a + 1 + uint32(rng() % (n - 1 - a))});
		}
		race("dense n=2e3 m=2e5", n, std::move(edges));
	}
	{
		// a long chain with chords: the closure is quadratic
		std::size_t const n = 2'000;
		std::vector<g::arc<>> edges;
		for (uint32 v = 0; v + 1 < n; ++v) edges.push_back({v, v + 1});
		for (std::size_t i = 0; i < 2'000; ++i) {
			uint32 const a = uint32(rng() % (n - 1));
			edges.push_back({a, a + 1 + uint32(rng() % (n - 1 - a))});
		}
		race("chain n=2e3", n, std::move(edges));
	}
	for (std::size_t const n : {std::size_t(2'000), std::size_t(5'000)}) {
		std::string const size = n == 2'000 ? " n=2e3" : " n=5e3";
		{
			std::vector<g::arc<>> edges;
			for (uint32 i = 0; i < n; ++i)
				for (uint32 j = i + 1; j < n; ++j)
					if (rng() % 4 == 0) edges.push_back({i, j});
			phases("dense p=1/4" + size, n, edges);
		}
		{
			std::vector<g::arc<>> edges;
			for (std::size_t i = 0; i < 4 * n; ++i) {
				uint32 const a = uint32(rng() % (n - 1));
				edges.push_back({a, a + 1 + uint32(rng() % (n - 1 - a))});
			}
			phases("sparse m=4n" + size, n, edges);
		}
		{
			// a bare chain: the closure is the complete order
			std::vector<g::arc<>> edges;
			for (uint32 v = 0; v + 1 < n; ++v) edges.push_back({v, v + 1});
			phases("bare chain" + size, n, edges);
		}
	}
	{
		std::size_t const n = 2'000;
		std::vector<g::arc<>> edges;
		for (uint32 i = 0; i < n; ++i)
			for (uint32 j = i + 1; j < n; ++j) edges.push_back({i, j});
		phases("complete n=2e3", n, edges);
	}
}
