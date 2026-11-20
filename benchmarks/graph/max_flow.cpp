/**
 * @file
 * @brief End-to-end maximum-flow latency including residual construction:
 * library two-phase HLPP versus Dinic, capacity-scaling Dinic and ISAP;
 * value-only HLPP versus discharge debit and BFS predicate orders;
 * residual scatter versus bucket grouping. Each sample solves one sparse
 * random, bipartite unit, grid, dense high-capacity or layered unit network.
 * Edmonds-Karp is a correctness oracle excluded from timing at these sizes.
 * Intel Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, assertions disabled,
 * 2026-10-15, -O2 -g -DNDEBUG -std=c++23, with/without -march=native:
 * grid library 162 / 160 ms, ISAP 1103 / 1097, Dinic 3340 / 3320;
 * sparse random favors ISAP (59.5 / 59.1 versus 111 / 110 ms).
 * Direct debit is noisy; height-first BFS loses the native grid (153 to 161 ms);
 * bucket building loses the grid (2.2 to 3.4 ms) with either flag set.
 * Full tables and the remaining shapes: docs/max-flow.md.
 */
#include <atl/experimental/graph/max_flow_variants.hpp>
#include <atl/graph/max_flow.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/counting_sort.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::int64;
using atl::uint32;

namespace {
	template<bool IsDirectDebit, bool IsHeightFirst, g::flow_value F>
	void drain_by_discharge_order(g::detail::residual<F> &net, std::vector<F> &excess, g::vertex_id root, g::vertex_id still_a,
								  g::vertex_id still_b) {
		auto const n = uint32(net.offset.size() - 1);
		std::vector<uint32> height(n, n), it(net.offset.begin(), net.offset.end() - 1);
		std::vector<uint32> bucket_head(n, g::none), bucket_next(n);
		std::vector<uint32> gap_next(std::size_t(2) * n), gap_prev(std::size_t(2) * n);
		uint32 hi = 0, gap_highest = 0;

		auto const enqueue = [&](uint32 v) {
			bucket_next[v] = bucket_head[height[v]];
			bucket_head[height[v]] = v;
			hi = std::max(hi, height[v]);
		};
		auto const gap_insert = [&](uint32 v, uint32 h) {
			gap_next[v] = gap_next[n + h];
			gap_prev[v] = n + h;
			gap_next[n + h] = v;
			gap_prev[gap_next[v]] = v;
			gap_highest = std::max(gap_highest, h);
		};
		auto const gap_erase = [&](uint32 v) {
			gap_next[gap_prev[v]] = gap_next[v];
			gap_prev[gap_next[v]] = gap_prev[v];
		};
		auto const push = [&]([[maybe_unused]] uint32 u, uint32 a, F amount) {
			uint32 const w = net.head[a];
			if (amount > 0 && excess[w] == 0 && w != still_a && w != still_b && height[w] < n)
				enqueue(w);
			net.capacity[a] -= amount;
			net.capacity[net.mate[a]] += amount;
			if constexpr (IsDirectDebit)
				excess[u] -= amount;
			else
				excess[net.head[net.mate[a]]] -= amount;
			excess[w] += amount;
		};

		std::vector<uint32> queue;
		queue.reserve(n);
		auto const global_relabel = [&] {
			std::ranges::fill(height, n); // unreachable to the root: dead
			height[root] = 0;
			queue.assign(1, root);
			for (std::size_t at = 0; at < queue.size(); ++at) {
				uint32 const w = queue[at];
				for (uint32 a = net.offset[w]; a < net.offset[w + 1]; ++a) {
					uint32 const x = net.head[a]; // mate[a] is the arc x -> w
					if (IsHeightFirst ? (height[x] == n && x != root && net.capacity[net.mate[a]] > 0)
									  : (net.capacity[net.mate[a]] > 0 && height[x] == n && x != root)) {
						height[x] = height[w] + 1;
						queue.push_back(x);
					}
				}
			}
			std::ranges::fill(bucket_head, g::none);
			for (uint32 h = 0; h < n; ++h) gap_next[n + h] = gap_prev[n + h] = n + h;
			hi = gap_highest = 0;
			for (uint32 v = 0; v < n; ++v) {
				it[v] = net.offset[v];
				if (height[v] < n) {
					gap_insert(v, height[v]);
					if (excess[v] > 0 && v != still_a && v != still_b) enqueue(v);
				}
			}
		};
		global_relabel();

		uint32 lift = 0;
		auto const discharge_pass = [&](uint32 u) {
			auto const sweep = [&](uint32 from, uint32 to) {
				for (uint32 a = from; a < to; ++a) {
					if (net.capacity[a] <= 0) continue;
					uint32 const hw = height[net.head[a]];
					if (height[u] == hw + 1) {
						push(u, a, std::min(excess[u], net.capacity[a]));
						if (excess[u] == 0) {
							it[u] = a;
							return true;
						}
					} else if (hw + 1 < lift) {
						lift = hw + 1;
					}
				}
				return false;
			};
			lift = n;
			return sweep(it[u], net.offset[u + 1]) || sweep(net.offset[u], it[u]);
		};

		uint32 relabels = 0;
		for (;;) {
			while (hi > 0 && bucket_head[hi] == g::none) --hi;
			uint32 const u = bucket_head[hi];
			if (u == g::none) break; // no live active vertex left
			bucket_head[hi] = bucket_next[u];
			if (height[u] >= n) continue; // lifted to death while queued
			while (excess[u] > 0 && !discharge_pass(u)) {
				uint32 const prior_height = height[u];
				gap_erase(u);
				if (gap_next[n + prior_height] == n + prior_height) {
					// gap: the whole stranded band above it dies at once
					for (uint32 h = prior_height + 1; h <= gap_highest; ++h) {
						for (uint32 v = gap_next[n + h]; v < n; v = gap_next[v]) height[v] = n;
						gap_next[n + h] = gap_prev[n + h] = n + h;
					}
					gap_highest = prior_height - 1;
					height[u] = n;
					break;
				}
				if (lift >= n) {
					height[u] = n; // no live target: dead
					break;
				}
				height[u] = lift;
				gap_insert(u, lift);
				hi = lift;
				if (++relabels >= n) {
					relabels = 0;
					global_relabel();
					break;
				}
			}
		}
	}

	/** @brief Value-only HLPP varying direct/reverse-arc debit and height/capacity-first BFS.
	 * Direct debit is within noise; height-first loses the native grid (153 to 161 ms).
	 */
	template<bool IsDirectDebit, bool IsHeightFirst>
	[[nodiscard]] int64 max_flow_value_by_discharge_order(std::size_t n, std::vector<g::flow_arc<int64>> const &edges, g::vertex_id s,
														  g::vertex_id t) {
		g::detail::residual<int64> net(n, edges);
		std::vector<int64> excess(n, 0);
		for (uint32 a = net.offset[s]; a < net.offset[s + 1]; ++a) {
			uint32 const w = net.head[a];
			if (w == s || net.capacity[a] <= 0) continue;
			excess[w] += net.capacity[a];
			net.capacity[net.mate[a]] += net.capacity[a];
			net.capacity[a] = 0;
		}
		drain_by_discharge_order<IsDirectDebit, IsHeightFirst>(net, excess, t, s, t);
		return excess[t];
	}

	/** @brief Bucket-group half-arcs, gather capacities and invert the order for mates.
	 * Grid construction takes 3.4 ms versus scatter's 2.2 ms with either flag set.
	 */
	struct residual_by_bucket_grouping {
		std::vector<uint32> offset, head, mate, input_arc;
		std::vector<int64> capacity;

		residual_by_bucket_grouping(std::size_t n, std::vector<g::flow_arc<int64>> const &edges) {
			std::size_t const m = edges.size();
			atl_assert(n < (std::size_t(1) << 31) && m <= std::numeric_limits<uint32>::max() / 2);
			offset.resize(n + 1);
			std::vector<uint32> order(2 * m), slot(2 * m);
			atl::detail::bucket_by_key(
				n, [&](std::size_t i) { return std::size_t(i & 1 ? edges[i / 2].to : edges[i / 2].from); },
				std::span<uint32>(offset), std::span<uint32>(order));
			for (uint32 p = 0; p < 2 * m; ++p) slot[order[p]] = p;
			head.resize(2 * m);
			capacity.resize(2 * m);
			mate.resize(2 * m);
			input_arc.resize(m);
			for (uint32 p = 0; p < 2 * m; ++p) {
				uint32 const j = order[p];
				auto const &e = edges[j / 2];
				head[p] = j & 1 ? e.from : e.to;
				capacity[p] = j & 1 ? 0 : e.capacity;
				mate[p] = slot[j ^ 1];
			}
			for (std::size_t i = 0; i < m; ++i) input_arc[i] = slot[2 * i];
		}
	};
}

TEST_CASE("max_flow contenders", "[max_flow]") {
	std::mt19937 rng(1);

	auto race = [&](char const *name, std::size_t n, std::vector<g::flow_arc<int64>> edges,
					g::vertex_id s, g::vertex_id t) {
		auto const expected = g::max_flow(n, edges, s, t).value;
		auto run = [&]<typename Solve>(char const *label, Solve solve)
			requires std::is_invocable_r_v<int64, Solve &>
		{
			REQUIRE(std::invoke(solve) == expected);
			BENCHMARK(std::string(name) + " | " + label) { return std::invoke(solve); };
		};
		run("by dinic", [&] { return ex::max_flow_by_dinic(n, edges, s, t).value; });
		run("by capacity scaling dinic", [&] { return ex::max_flow_by_capacity_scaling_dinic(n, edges, s, t).value; });
		run("by isap", [&] { return ex::max_flow_by_isap(n, edges, s, t).value; });
		run("library", [&] { return g::max_flow(n, edges, s, t).value; });
		run("library value", [&] { return g::max_flow_value(n, edges, s, t); });
		run("by discharge order capacity first", [&] { return max_flow_value_by_discharge_order<false, false>(n, edges, s, t); });
		run("by discharge order direct debit", [&] { return max_flow_value_by_discharge_order<true, false>(n, edges, s, t); });
		run("by discharge order height first", [&] { return max_flow_value_by_discharge_order<false, true>(n, edges, s, t); });
		run("by discharge order direct debit height first", [&] { return max_flow_value_by_discharge_order<true, true>(n, edges, s, t); });
		{
			g::detail::residual<int64> const a(n, edges);
			residual_by_bucket_grouping const b(n, edges);
			REQUIRE((a.offset == b.offset && a.head == b.head && a.mate == b.mate && a.input_arc == b.input_arc
					 && a.capacity == b.capacity));
		}
		auto build = [&]<typename Build>(char const *label, Build construct)
			requires std::is_invocable_r_v<std::size_t, Build &>
		{
			REQUIRE(std::invoke(construct) == 2 * edges.size());
			BENCHMARK(std::string(name) + " | " + label) { return std::invoke(construct); };
		};
		build("residual library", [&] { return g::detail::residual<int64>(n, edges).head.size(); });
		build("residual by bucket grouping", [&] { return residual_by_bucket_grouping(n, edges).head.size(); });
	};

	{
		std::size_t const n = 100'000;
		std::vector<g::flow_arc<int64>> edges;
		for (std::size_t i = 0; i < 5 * n; ++i)
			edges.push_back({uint32(rng() % n), uint32(rng() % n), int64(1 + rng() % 1'000'000)});
		race("random n=1e5 m=5e5", n, std::move(edges), 0, 1);
	}
	{
		uint32 const half = 20'000;
		std::size_t const n = 2 * half + 2;
		uint32 const s = uint32(n - 2), t = uint32(n - 1);
		std::vector<g::flow_arc<int64>> edges;
		for (uint32 l = 0; l < half; ++l) {
			edges.push_back({s, l, 1});
			edges.push_back({half + l, t, 1});
			for (int k = 0; k < 5; ++k) edges.push_back({l, half + uint32(rng() % half), 1});
		}
		race("bipartite unit 2e4+2e4", n, std::move(edges), s, t);
	}
	{
		// grid: source feeds the left column, sink drains the right
		uint32 const rows = 300, cols = 300;
		auto const id = [](uint32 r, uint32 c) { return r * cols + c; };
		std::size_t const n = std::size_t(rows) * cols + 2;
		uint32 const s = uint32(n - 2), t = uint32(n - 1);
		std::vector<g::flow_arc<int64>> edges;
		for (uint32 r = 0; r < rows; ++r) {
			edges.push_back({s, id(r, 0), int64(1 + rng() % 1000)});
			edges.push_back({id(r, cols - 1), t, int64(1 + rng() % 1000)});
			for (uint32 c = 0; c + 1 < cols; ++c) {
				edges.push_back({id(r, c), id(r, c + 1), int64(1 + rng() % 1000)});
				if (r + 1 < rows) {
					edges.push_back({id(r, c), id(r + 1, c), int64(1 + rng() % 1000)});
					edges.push_back({id(r + 1, c), id(r, c), int64(1 + rng() % 1000)});
				}
			}
		}
		race("grid 300x300", n, std::move(edges), s, t);
	}
	{
		std::size_t const n = 2'000;
		std::vector<g::flow_arc<int64>> edges;
		for (std::size_t i = 0; i < 200 * n; ++i)
			edges.push_back({uint32(rng() % n), uint32(rng() % n), int64(1 + rng() % 1'000'000'000)});
		race("dense high-cap n=2e3 m=4e5", n, std::move(edges), 0, 1);
	}
	{
		uint32 const layers = 200, width = 50;
		std::size_t const n = std::size_t(layers) * width + 2;
		uint32 const s = uint32(n - 2), t = uint32(n - 1);
		auto const id = [](uint32 l, uint32 i) { return l * width + i; };
		std::vector<g::flow_arc<int64>> edges;
		for (uint32 i = 0; i < width; ++i) {
			edges.push_back({s, id(0, i), 1});
			edges.push_back({id(layers - 1, i), t, 1});
		}
		for (uint32 l = 0; l + 1 < layers; ++l)
			for (uint32 i = 0; i < width; ++i)
				for (uint32 k = 0; k < 8; ++k)
					edges.push_back({id(l, i), id(l + 1, uint32(rng() % width)), 1});
		race("layered unit 200x50", n, std::move(edges), s, t);
	}
}
