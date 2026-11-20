/**
 * @file
 * @brief Races auction with full or level-zero bid scans, Hopcroft-Karp,
 * per-root or pass-structured Kuhn, and a max_flow reduction. Measures one
 * complete matching including adjacency construction on random, band, line,
 * augmented-cycle, skewed, dense and complete shapes. Xeon Platinum 8358,
 * GCC 15.2, RelWithDebInfo, assertions off, 2026-10-15,
 * -O2 -g -DNDEBUG -std=c++23 (native adds -march=native): library augmented
 * cycle 33.6 / 34.9 ms versus Hopcroft-Karp 404 / 401; dense m=3e6
 * 55.7 / 54.3 versus 18.9 / 19.5. Full tables: docs/bipartite-matching.md.
 */
#include <atl/experimental/graph/bipartite_matching_variants.hpp>
#include <atl/graph/bipartite_matching.hpp>
#include <atl/graph/detail/edge_count.hpp>
#include <atl/graph/max_flow.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <random>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::uint32;

namespace {
	template<typename Run>
		requires std::invocable<Run &>
	void measure(std::string const &label, Run run) {
		BENCHMARK(std::string(label)) { return run(); };
	}

	/** @brief Auction with a full scan or a scan ending at the first level-zero bid; ties match the library. */
	template<bool IsEarlyExit, g::edge_list EdgeList>
	[[nodiscard]] g::bipartite_matching_result bipartite_matching_by_bid_scan(std::size_t left, std::size_t right,
																			  EdgeList const &edges) {
		auto const m = g::detail::checked_edge_count(edges, g::none);
		atl_assert(left < g::none && right < g::none && m <= g::none);
		std::vector<uint32> left_offset(left + 1, 0), left_target(m), right_offset(right + 1, 0), right_target(m);
		for (auto const &e : edges) {
			atl_assert(g::source(e) < left && g::target(e) < right);
			++left_offset[g::source(e) + 1];
			++right_offset[g::target(e) + 1];
		}
		for (std::size_t v = 0; v < left; ++v) left_offset[v + 1] += left_offset[v];
		for (std::size_t v = 0; v < right; ++v) right_offset[v + 1] += right_offset[v];
		{
			std::vector<uint32> cl(left_offset.begin(), left_offset.end() - 1), cr(right_offset.begin(), right_offset.end() - 1);
			for (auto const &e : edges) {
				left_target[cl[g::source(e)]++] = g::target(e);
				right_target[cr[g::target(e)]++] = g::source(e);
			}
		}

		g::bipartite_matching_result r{0, std::vector<uint32>(left, g::none), std::vector<uint32>(right, g::none)};
		auto &ml = r.left_pair;
		auto &mr = r.right_pair;
		std::size_t const lim = std::min(left, right);
		std::size_t size = 0;

		std::vector<atl::uint64> lv(left);
		std::vector<uint32> queue;
		auto constexpr infinite_level = std::numeric_limits<atl::uint64>::max();
		queue.reserve(left);
		std::vector<uint32> ring(right + 1); // free rights, each queued at most once
		std::size_t head = 0, tail = 0, live = 0;
		auto const push = [&](uint32 w) {
			ring[tail] = w;
			if (++tail == ring.size()) tail = 0;
			++live;
		};
		auto const pop = [&] {
			uint32 const w = ring[head];
			if (++head == ring.size()) head = 0;
			--live;
			return w;
		};
		for (uint32 w = 0; w < right; ++w) push(w);

		auto const global_relabel = [&] {
			std::ranges::fill(lv, infinite_level);
			queue.clear();
			for (uint32 l = 0; l < left; ++l)
				if (ml[l] == g::none) {
					lv[l] = 0;
					queue.push_back(l);
				}
			for (std::size_t at = 0; at < queue.size(); ++at) {
				uint32 const l = queue[at];
				for (uint32 k = left_offset[l]; k < left_offset[l + 1]; ++k) {
					uint32 const u = mr[left_target[k]];
					if (u != g::none && lv[u] > lv[l] + 2) {
						lv[u] = lv[l] + 2;
						queue.push_back(u);
					}
				}
			}
		};

		atl::uint64 const period = atl::uint64(left) + right;
		atl::uint64 ticks = 0;
		while (live > 0 && size < lim) {
			if (ticks == 0) global_relabel();
			if (++ticks == period) ticks = 0;
			uint32 const w = pop();
			uint32 best = g::none;
			atl::uint64 best_lv = infinite_level;
			for (uint32 k = right_offset[w]; k < right_offset[w + 1]; ++k)
				if (lv[right_target[k]] < best_lv) {
					best_lv = lv[right_target[k]];
					best = right_target[k];
					if constexpr (IsEarlyExit)
						if (best_lv == 0) break;
				}
			if (best_lv == infinite_level) continue; // no left reaches a free one: w rests until HK
			uint32 const old = ml[best];
			if (old != g::none) {
				mr[old] = g::none;
				push(old);
			} else {
				++size;
			}
			ml[best] = w;
			mr[w] = best;
			lv[best] = best_lv + 2;
		}

		// Hopcroft-Karp finish: guarantees maximum cardinality. Phases label
		// everything reachable (no first-free-level cutoff): full labels let
		// the DFS harvest longer label-increasing paths too.
		std::vector<uint32> dist(left), it(left), stack, chosen;
		for (;;) {
			std::ranges::fill(dist, g::none);
			queue.clear();
			for (uint32 l = 0; l < left; ++l)
				if (ml[l] == g::none) {
					dist[l] = 0;
					queue.push_back(l);
				}
			bool is_free_right_reachable = false;
			for (std::size_t at = 0; at < queue.size(); ++at) {
				uint32 const l = queue[at];
				for (uint32 k = left_offset[l]; k < left_offset[l + 1]; ++k) {
					uint32 const l2 = mr[left_target[k]];
					if (l2 == g::none)
						is_free_right_reachable = true;
					else if (dist[l2] == g::none) {
						dist[l2] = dist[l] + 1;
						queue.push_back(l2);
					}
				}
			}
			if (!is_free_right_reachable) break;
			std::ranges::copy(left_offset.begin(), left_offset.end() - 1, it.begin());
			for (uint32 root = 0; root < left; ++root) {
				if (ml[root] != g::none) continue;
				stack.assign(1, root); // chosen[i] is the right vertex stack[i] picked
				chosen.clear();
				while (!stack.empty()) {
					uint32 const l = stack.back();
					bool has_advanced = false;
					while (it[l] < left_offset[l + 1]) {
						uint32 const w = left_target[it[l]++];
						uint32 const l2 = mr[w];
						if (l2 == g::none) {
							chosen.push_back(w); // free right: flip the whole path
							for (std::size_t i = stack.size(); i-- > 0;) {
								ml[stack[i]] = chosen[i];
								mr[chosen[i]] = stack[i];
							}
							stack.clear();
							chosen.clear();
							has_advanced = true;
							break;
						}
						if (dist[l2] == dist[l] + 1) {
							chosen.push_back(w);
							stack.push_back(l2);
							has_advanced = true;
							break;
						}
					}
					if (!has_advanced) {
						dist[l] = g::none;
						stack.pop_back();
						if (!chosen.empty()) chosen.pop_back();
					}
				}
			}
		}
		for (uint32 l = 0; l < left; ++l) r.size += ml[l] != g::none;
		return r;
	}

	/** @brief Unit-capacity max_flow reduction, including network construction. */
	[[nodiscard]] std::size_t bipartite_matching_by_max_flow(std::size_t left, std::size_t right,
															 std::vector<g::arc<>> const &edges) {
		auto const s = uint32(left + right), t = uint32(left + right + 1);
		std::vector<g::flow_arc<int>> arcs;
		arcs.reserve(edges.size() + left + right);
		for (auto const &e : edges) arcs.push_back({e.from, uint32(left) + e.to, 1});
		for (uint32 l = 0; l < left; ++l) arcs.push_back({s, l, 1});
		for (uint32 w = 0; w < right; ++w) arcs.push_back({uint32(left) + w, t, 1});
		return std::size_t(g::max_flow(left + right + 2, arcs, s, t).value);
	}
}

TEST_CASE("bipartite_matching contenders", "[bipartite_matching]") {
	std::mt19937 rng(1);

	auto race = [&](char const *name, std::size_t left, std::size_t right,
					std::vector<g::arc<>> edges) {
		auto const lib = g::bipartite_matching(left, right, edges);
		REQUIRE(lib.size == ex::bipartite_matching_by_kuhn(left, right, edges).size);
		REQUIRE(lib.size == ex::bipartite_matching_by_hopcroft_karp(left, right, edges).size);
		REQUIRE(lib.size == bipartite_matching_by_max_flow(left, right, edges));
		REQUIRE(lib.size == ex::bipartite_matching_by_kuhn_passes(left, right, edges).size);
		REQUIRE(lib.left_pair == bipartite_matching_by_bid_scan<false>(left, right, edges).left_pair);
		REQUIRE(lib.left_pair == bipartite_matching_by_bid_scan<true>(left, right, edges).left_pair);

		measure(std::string(name) + " | library", [&] { return g::bipartite_matching(left, right, edges).size; });
		measure(std::string(name) + " | by bid scan full", [&] { return bipartite_matching_by_bid_scan<false>(left, right, edges).size; });
		measure(std::string(name) + " | by bid scan level zero", [&] { return bipartite_matching_by_bid_scan<true>(left, right, edges).size; });
		measure(std::string(name) + " | by kuhn passes", [&] { return ex::bipartite_matching_by_kuhn_passes(left, right, edges).size; });
		measure(std::string(name) + " | by hopcroft karp", [&] { return ex::bipartite_matching_by_hopcroft_karp(left, right, edges).size; });
		measure(std::string(name) + " | by kuhn", [&] { return ex::bipartite_matching_by_kuhn(left, right, edges).size; });
		measure(std::string(name) + " | by max flow", [&] { return bipartite_matching_by_max_flow(left, right, edges); });
	};

	{
		std::size_t const n = 100'000;
		std::vector<g::arc<>> edges;
		for (std::size_t i = 0; i < 200'000; ++i)
			edges.push_back({uint32(rng() % n), uint32(rng() % n)});
		race("random 1e5+1e5 m=2e5", n, n, std::move(edges));
	}
	{
		// unique-matching band: i connects to [i-2, i] plus a random earlier
		// vertex — near-forced matchings, long augmenting chains
		std::size_t const n = 66'000;
		std::vector<g::arc<>> edges;
		for (uint32 i = 0; i < n; ++i) {
			for (uint32 j = i >= 2 ? i - 2 : 0; j <= i; ++j) edges.push_back({i, j});
			if (i > 0) edges.push_back({i, uint32(rng() % i)});
		}
		race("unique-matching band", n, n, std::move(edges));
	}
	{
		// long line: l_i - r_i and l_i - r_{i-1}; one perfect matching, deep
		// alternating paths
		std::size_t const n = 100'000;
		std::vector<g::arc<>> edges;
		for (uint32 i = 0; i < n; ++i) {
			edges.push_back({i, i});
			if (i > 0) edges.push_back({i, i - 1});
		}
		race("line", n, n, std::move(edges));
	}
	{
		// augmented cycle (adversarial long augmenting paths): a label-permuted cycle
		// with window chords — long augmenting paths, zero locality
		std::size_t const n = 66'000;
		std::vector<g::arc<>> edges;
		std::vector<uint32> pl(n), pr(n);
		for (uint32 i = 0; i < n; ++i) pl[i] = pr[i] = i;
		std::ranges::shuffle(pl, rng);
		std::ranges::shuffle(pr, rng);
		for (uint32 i = 0; i < n; ++i) {
			edges.push_back({pl[i], pr[i]});
			edges.push_back({pl[i], pr[(i + 1) % n]});
			edges.push_back({pl[i], pr[i < 1000 ? uint32(rng() % (i + 1)) : i - uint32(rng() % 1000)]});
		}
		std::ranges::shuffle(edges, rng);
		race("augmented cycle", n, n, std::move(edges));
	}
	{
		std::size_t const left = 100'000, right = 1'000;
		std::vector<g::arc<>> edges;
		for (std::size_t i = 0; i < 200'000; ++i)
			edges.push_back({uint32(rng() % left), uint32(rng() % right)});
		race("skewed 1e5 to 1e3", left, right, std::move(edges));
	}
	{
		// dense random: the first auction sweep is all level-0 lefts
		std::size_t const n = 3'000;
		for (std::size_t const m : {std::size_t(1'000'000), std::size_t(3'000'000)}) {
			std::vector<g::arc<>> edges;
			edges.reserve(m);
			for (std::size_t i = 0; i < m; ++i) edges.push_back({uint32(rng() % n), uint32(rng() % n)});
			race(m == 1'000'000 ? "dense 3e3+3e3 m=1e6" : "dense 3e3+3e3 m=3e6", n, n, std::move(edges));
		}
	}
	{
		std::size_t const n = 2'000;
		std::vector<g::arc<>> edges;
		edges.reserve(n * n);
		for (uint32 l = 0; l < n; ++l)
			for (uint32 w = 0; w < n; ++w) edges.push_back({l, w});
		race("complete 2e3x2e3", n, n, std::move(edges));
	}
}
