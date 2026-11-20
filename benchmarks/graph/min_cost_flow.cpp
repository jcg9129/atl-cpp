/**
 * @file
 * @brief Complete-operation latency for min-cost flow: library threaded network
 * simplex versus SPFA, Johnson-Dijkstra, zkw admissible DFS, capacity-scaling SSP,
 * cost-scaling push-relabel and cost scaling with global price updates.
 * Core loops isolate struct/flat arc pricing and adjacency/thread tree storage;
 * one loop isolates residual dual certification. Shapes are random, assignment,
 * grid, parallel paths and a negative DAG, plus a larger random simplex race.
 * Intel Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, assertions disabled,
 * 2026-10-15: -O2 -g -DNDEBUG -std=c++23, also with -march=native.
 * Supplied complete-workload record: library 65.3/10.9/86.1/2.6/53.3 ms,
 * faster than every contender on those five shapes. Core and certificate are
 * timed separately here; solve-only timings need a fresh record.
 * Full tables: docs/min-cost-flow.md. Minima of three per-run means, ten samples.
 */
#include <atl/experimental/graph/min_cost_flow_variants.hpp>
#include <atl/graph/min_cost_flow.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::int64;
using atl::int8;
using atl::uint32;
using atl::uint8;

namespace {
	/** @brief Times one complete operation, including its local allocation. */
	template<typename Solver>
		requires std::invocable<Solver &>
	void measure(std::string const &name, Solver solve) {
		BENCHMARK(std::string(name)) { return std::invoke(solve); };
	}

	/** @brief Threaded simplex with struct fields or materialized flat pricing arrays. */
	template<bool is_soa, g::signed_flow_value Flow, g::detail::cost_arithmetic Cost>
		requires(std::semiregular<Flow> && std::convertible_to<int, Flow>)
	std::optional<std::vector<Flow>> network_simplex_by_arc_layout(std::size_t n, std::vector<g::flow_arc<Flow, Cost>> const &arcs,
																   std::vector<Flow> const &b) {
		using F = Flow;
		using C = Cost;
		std::size_t const m = arcs.size();
		uint32 const root = uint32(n);
		C largest = 0;
		for (auto const &e : arcs) {
			largest = std::max(largest, e.cost < C(0) ? C(-e.cost) : e.cost);
		}
		C const artificial_cost = g::detail::simplex_artificial_cost(n, largest);
		F const infinite_capacity = std::numeric_limits<F>::max();

		std::vector<F> x(m, 0);
		std::vector<int8> state(m + n, 1); // 1 at lower bound, -1 at upper, 0 in tree
		std::vector<C> potentials(n + 1, 0);
		std::vector<uint32> parent(n + 1, root), parent_arc(n + 1), thread(n + 1), rev_thread(n + 1),
			size(n + 1, 1), last(n + 1);
		parent[root] = g::none;
		size[root] = uint32(n + 1);

		std::vector<uint8> is_outward(n);
		std::vector<F> artificial_flow(n);
		std::vector<uint32> tail_of, head_of;
		std::vector<C> cost_of;
		if constexpr (is_soa) {
			tail_of.resize(m + n);
			head_of.resize(m + n);
			cost_of.resize(m + n);
			for (std::size_t a = 0; a < m; ++a) {
				tail_of[a] = arcs[a].from;
				head_of[a] = arcs[a].to;
				cost_of[a] = arcs[a].cost;
			}
		}
		auto const from = [&](uint32 a) {
			if constexpr (is_soa) {
				return tail_of[a];
			} else
				return a < m ? uint32(arcs[a].from) : is_outward[a - m] ? a - uint32(m)
																		: root;
		};
		auto const to = [&](uint32 a) {
			if constexpr (is_soa) {
				return head_of[a];
			} else
				return a < m ? uint32(arcs[a].to) : is_outward[a - m] ? root
																	  : a - uint32(m);
		};
		auto const acost = [&](uint32 a) {
			if constexpr (is_soa) {
				return cost_of[a];
			} else
				return a < m ? arcs[a].cost : artificial_cost;
		};
		auto const flow = [&](uint32 a) -> F & { return a < m ? x[a] : artificial_flow[a - m]; };
		auto const capacity = [&](uint32 a) { return a < m ? arcs[a].capacity : infinite_capacity; };
		auto const residual = [&](uint32 a, bool is_along) { return is_along ? capacity(a) - flow(a) : flow(a); };
		auto const link = [&](uint32 u, uint32 v) {
			thread[u] = v;
			rev_thread[v] = u;
		};

		for (uint32 v = 0; v < n; ++v) {
			is_outward[v] = b[v] >= 0;
			artificial_flow[v] = b[v] >= 0 ? b[v] : -b[v];
			potentials[v] = is_outward[v] ? -artificial_cost : artificial_cost;
			state[m + v] = 0;
			parent_arc[v] = uint32(m + v);
			if constexpr (is_soa) {
				tail_of[m + v] = is_outward[v] ? v : root;
				head_of[m + v] = is_outward[v] ? root : v;
				cost_of[m + v] = artificial_cost;
			}
			last[v] = v;
		}
		for (uint32 v = 0; v < n; ++v) link(v == 0 ? root : v - 1, v);
		link(n == 0 ? root : root - 1, root);
		last[root] = n == 0 ? root : root - 1;

		std::vector<uint32> stem, stem_last, stem_prev, stem_after, stem_size;
		std::size_t const block = std::max<std::size_t>(64, std::size_t(std::sqrt(double(m + n))));
		std::size_t probe = 0;
		while (true) {
			uint32 enter = uint32(m + n);
			C best = 0;
			for (std::size_t seen = 0; seen < m + n; seen += std::min(block, m + n - seen)) {
				for (std::size_t k = 0; k < block; ++k, ++probe) {
					if (probe == m + n) probe = 0;
					C const rc = acost(uint32(probe)) + potentials[from(uint32(probe))] - potentials[to(uint32(probe))];
					C const score = C(state[probe]) * rc;
					if (score < best) {
						best = score;
						enter = uint32(probe);
					}
				}
				if (enter != m + n) break;
			}
			if (enter == uint32(m + n)) break;
			bool const is_forward = state[enter] > 0;
			uint32 const a = is_forward ? from(enter) : to(enter);
			uint32 const bb = is_forward ? to(enter) : from(enter);
			F delta = residual(enter, is_forward);
			uint32 leave = enter;
			bool leaves_on_a = false;
			auto const step = [&](uint32 &v, bool is_a_side) {
				uint32 const pa = parent_arc[v];
				F const res = residual(pa, is_a_side ? to(pa) == v : from(pa) == v);
				if (is_a_side ? res < delta : res <= delta) {
					delta = res;
					leave = pa;
					leaves_on_a = is_a_side;
				}
				v = parent[v];
			};
			// an ancestor's subtree is strictly larger, so stepping the
			// smaller side never passes the apex
			uint32 va = a, vb = bb;
			while (va != vb)
				if (size[va] <= size[vb])
					step(va, true);
				else
					step(vb, false);
			uint32 const apex = va;
			flow(enter) += is_forward ? delta : -delta;
			for (uint32 v = a; v != apex; v = parent[v]) {
				uint32 const pa = parent_arc[v];
				flow(pa) += to(pa) == v ? delta : -delta;
			}
			for (uint32 v = bb; v != apex; v = parent[v]) {
				uint32 const pa = parent_arc[v];
				flow(pa) += from(pa) == v ? delta : -delta;
			}
			if (leave == enter) {
				state[enter] = int8(-state[enter]);
				continue;
			}
			state[leave] = flow(leave) <= 0 ? int8(1) : int8(-1);
			state[enter] = 0;
			uint32 const cut = leaves_on_a ? a : bb;
			uint32 const hook = leaves_on_a ? bb : a;
			uint32 const out = parent[from(leave)] == to(leave) ? from(leave) : to(leave);
			uint32 const out_parent = parent[out];
			uint32 const moved = size[out];

			// the stem cut = s_0, ..., s_k = out, with all detached-segment links saved for the splice
			stem.clear();
			for (uint32 v = cut;; v = parent[v]) {
				stem.push_back(v);
				if (v == out) break;
			}
			stem_last.resize(stem.size());
			stem_prev.resize(stem.size());
			stem_after.resize(stem.size());
			stem_size.resize(stem.size());
			for (std::size_t i = 0; i < stem.size(); ++i) {
				stem_last[i] = last[stem[i]];
				stem_prev[i] = rev_thread[stem[i]];
				stem_after[i] = thread[last[stem[i]]];
				stem_size[i] = size[stem[i]];
			}
			std::size_t const k = stem.size() - 1;
			uint32 const segment_last = stem_last[k], predecessor = stem_prev[k];
			link(predecessor, stem_after[k]);

			// S re-rooted at cut: seg(s_0), then seg(s_i) minus seg(s_{i-1})
			uint32 tail = stem_last[0];
			for (std::size_t i = 1; i <= k; ++i) {
				link(tail, stem[i]);
				tail = stem_prev[i - 1];
				if (stem_last[i] != stem_last[i - 1]) {
					link(tail, stem_after[i - 1]);
					tail = stem_last[i];
				}
			}
			link(tail, thread[hook]);
			link(hook, cut);

			// last: the chain above S whose segment ended in S, then the
			// stem, then the chain whose segment ended at the hook
			for (uint32 v = out_parent; v != g::none && last[v] == segment_last; v = parent[v]) last[v] = predecessor;
			for (auto const v : stem) last[v] = tail;
			for (uint32 v = hook; v != g::none && last[v] == hook; v = parent[v]) last[v] = tail;
			// size: only the two paths below the apex change
			for (uint32 v = out_parent; v != apex; v = parent[v]) size[v] -= moved;
			for (uint32 v = hook; v != apex; v = parent[v]) size[v] += moved;
			for (std::size_t i = k; i >= 1; --i) {
				size[stem[i]] = moved - stem_size[i - 1];
				parent[stem[i]] = stem[i - 1];
				parent_arc[stem[i]] = parent_arc[stem[i - 1]];
			}
			size[cut] = moved;
			parent[cut] = hook;
			parent_arc[cut] = enter;

			C const rc = acost(enter) + potentials[from(enter)] - potentials[to(enter)];
			C const shift = cut == from(enter) ? -rc : rc;
			uint32 v = cut;
			for (uint32 i = 0; i < moved; ++i, v = thread[v]) potentials[v] += shift;
		}
		for (uint32 v = 0; v < n; ++v)
			if (artificial_flow[v] != 0) return std::nullopt;
		return x;
	}

	std::vector<g::flow_arc<int64, int64>> circulation(std::size_t n, std::vector<g::flow_arc<int64, int64>> arcs,
													   uint32 s, uint32 t) {
		int64 largest = 0, out = 0;
		for (auto const &e : arcs) {
			largest = std::max(largest, e.cost < 0 ? -e.cost : e.cost);
			if (e.from == s) out += e.capacity;
		}
		arcs.push_back({t, s, out, -(int64(n) * largest + 1)});
		return arcs;
	}

	std::size_t certificate(std::size_t n, std::vector<g::flow_arc<int64, int64>> const &edges,
							std::vector<int64> const &flow) {
		return g::min_cost_flow_potentials(n, edges, flow).size();
	}
}

TEST_CASE("min_cost_flow contenders", "[min_cost_flow]") {
	std::mt19937 rng(1);

	auto race = [&](char const *name, std::size_t n, std::vector<g::flow_arc<int64, int64>> edges,
					g::vertex_id s, g::vertex_id t, bool spfa_feasible = true) {
		auto const want = g::min_cost_max_flow(n, edges, s, t);
		auto const zkw = ex::min_cost_max_flow_by_zkw(n, edges, s, t);
		REQUIRE(want.value == zkw.value);
		REQUIRE(want.cost == zkw.cost);
		auto const sc = ex::min_cost_max_flow_by_capacity_scaling(n, edges, s, t);
		REQUIRE(want.value == sc.value);
		REQUIRE(want.cost == sc.cost);
		auto const dij = ex::min_cost_max_flow_by_dijkstra(n, edges, s, t);
		REQUIRE(want.value == dij.value);
		REQUIRE(want.cost == dij.cost);
		auto const cs = ex::min_cost_max_flow_by_cost_scaling(n, edges, s, t);
		REQUIRE(want.value == cs.value);
		REQUIRE(want.cost == cs.cost);
		if (spfa_feasible) {
			auto const spfa = ex::min_cost_max_flow_by_spfa(n, edges, s, t);
			REQUIRE(want.value == spfa.value);
			REQUIRE(want.cost == spfa.cost);
			measure(std::string(name) + " | by spfa", [&] {
				return ex::min_cost_max_flow_by_spfa(n, edges, s, t).cost;
			});
		}
		measure(std::string(name) + " | by dijkstra", [&] {
			return ex::min_cost_max_flow_by_dijkstra(n, edges, s, t).cost;
		});
		measure(std::string(name) + " | by zkw", [&] {
			return ex::min_cost_max_flow_by_zkw(n, edges, s, t).cost;
		});
		measure(std::string(name) + " | by capacity scaling", [&] {
			return ex::min_cost_max_flow_by_capacity_scaling(n, edges, s, t).cost;
		});
		measure(std::string(name) + " | library", [&] {
			return g::min_cost_max_flow(n, edges, s, t).cost;
		});
		measure(std::string(name) + " | by cost scaling", [&] {
			return ex::min_cost_max_flow_by_cost_scaling(n, edges, s, t).cost;
		});
		auto const csp = ex::min_cost_max_flow_by_cost_scaling_price_update(n, edges, s, t);
		REQUIRE(want.value == csp.value);
		REQUIRE(want.cost == csp.cost);
		measure(std::string(name) + " | by cost scaling price update", [&] {
			return ex::min_cost_max_flow_by_cost_scaling_price_update(n, edges, s, t).cost;
		});
		auto const circ = circulation(n, edges, s, t);
		std::vector<int64> const zero(n, 0);
		REQUIRE(*g::detail::network_simplex(n, circ, zero) == *network_simplex_by_arc_layout<true>(n, circ, zero));
		REQUIRE(*g::detail::network_simplex(n, circ, zero) == *network_simplex_by_arc_layout<false>(n, circ, zero));
		measure(std::string(name) + " | simplex core library", [&] {
			return g::detail::network_simplex(n, circ, zero)->back();
		});
		measure(std::string(name) + " | simplex core by struct arcs", [&] {
			return network_simplex_by_arc_layout<false>(n, circ, zero)->back();
		});
		measure(std::string(name) + " | simplex core by flat arcs", [&] {
			return network_simplex_by_arc_layout<true>(n, circ, zero)->back();
		});
		REQUIRE(*g::detail::network_simplex(n, circ, zero) == *ex::detail::network_simplex_by_adjacency(n, circ, zero));
		measure(std::string(name) + " | simplex core by adjacency", [&] {
			return ex::detail::network_simplex_by_adjacency(n, circ, zero)->back();
		});
		measure(std::string(name) + " | certificate post-pass alone", [&] {
			return certificate(n, edges, want.edge_flow);
		});
	};

	{
		std::size_t const n = 20'000;
		std::vector<g::flow_arc<int64, int64>> edges;
		for (std::size_t i = 0; i < 6 * n; ++i)
			edges.push_back({uint32(rng() % n), uint32(rng() % n), int64(1 + rng() % 100),
							 int64(1 + rng() % 1000)});
		race("random n=2e4 m=1.2e5", n, std::move(edges), 0, 1);
	}
	{
		// assignment: complete bipartite unit matching with random costs — the
		// shape zkw's many-paths-per-round is built for
		uint32 const half = 300;
		std::size_t const n = 2 * half + 2;
		uint32 const s = uint32(n - 2), t = uint32(n - 1);
		std::vector<g::flow_arc<int64, int64>> edges;
		for (uint32 l = 0; l < half; ++l) {
			edges.push_back({s, l, 1, 0});
			edges.push_back({half + l, t, 1, 0});
			for (uint32 r = 0; r < half; ++r)
				edges.push_back({l, half + r, 1, int64(1 + rng() % 10'000)});
		}
		race("assignment 300x300", n, std::move(edges), s, t);
	}
	{
		uint32 const rows = 80, cols = 80;
		auto const id = [&](uint32 r, uint32 c) { return r * cols + c; };
		std::size_t const n = std::size_t(rows) * cols + 2;
		uint32 const s = uint32(n - 2), t = uint32(n - 1);
		std::vector<g::flow_arc<int64, int64>> edges;
		for (uint32 r = 0; r < rows; ++r) {
			edges.push_back({s, id(r, 0), int64(1 + rng() % 30), int64(1 + rng() % 100)});
			edges.push_back({id(r, cols - 1), t, int64(1 + rng() % 30), int64(1 + rng() % 100)});
			for (uint32 c = 0; c + 1 < cols; ++c) {
				edges.push_back({id(r, c), id(r, c + 1), int64(1 + rng() % 30), int64(1 + rng() % 100)});
				if (r + 1 < rows) {
					edges.push_back({id(r, c), id(r + 1, c), int64(1 + rng() % 30), int64(1 + rng() % 100)});
					edges.push_back({id(r + 1, c), id(r, c), int64(1 + rng() % 30), int64(1 + rng() % 100)});
				}
			}
		}
		race("grid 80x80", n, std::move(edges), s, t);
	}
	{
		// many-rounds adversary: parallel two-hop paths with distinct costs, so
		// every SSP round moves exactly one path — F-dependence laid bare
		uint32 const paths = 4'000;
		std::size_t const n = paths + 2;
		uint32 const s = uint32(n - 2), t = uint32(n - 1);
		std::vector<g::flow_arc<int64, int64>> edges;
		for (uint32 i = 0; i < paths; ++i) {
			edges.push_back({s, i, int64(1 + rng() % 1000), int64(i)});
			edges.push_back({i, t, int64(1 + rng() % 1000), int64(i)});
		}
		race("parallel paths 4e3", n, std::move(edges), s, t);
	}
	{
		uint32 const layers = 60, width = 40;
		std::size_t const n = std::size_t(layers) * width + 2;
		uint32 const s = uint32(n - 2), t = uint32(n - 1);
		auto const id = [&](uint32 l, uint32 i) { return l * width + i; };
		std::vector<g::flow_arc<int64, int64>> edges;
		for (uint32 i = 0; i < width; ++i) {
			edges.push_back({s, id(0, i), int64(1 + rng() % 20), 0});
			edges.push_back({id(layers - 1, i), t, int64(1 + rng() % 20), 0});
		}
		for (uint32 l = 0; l + 1 < layers; ++l)
			for (uint32 i = 0; i < width; ++i)
				for (uint32 k = 0; k < 6; ++k)
					edges.push_back({id(l, i), id(l + 1, uint32(rng() % width)), int64(1 + rng() % 20),
									 int64(rng() % 200) - 100});
		race("negative DAG 60x40", n, std::move(edges), s, t);
	}
	{
		// the regime where the subtree walk dominates: n = 5e4, the
		// simplex cores alone (the SSP family takes seconds here)
		std::size_t const n = 50'000;
		std::vector<g::flow_arc<int64, int64>> edges;
		for (std::size_t i = 0; i < 6 * n; ++i)
			edges.push_back({uint32(rng() % n), uint32(rng() % n), int64(1 + rng() % 100), int64(1 + rng() % 1000)});
		auto const circ = circulation(n, edges, 0, 1);
		std::vector<int64> const zero(n, 0);
		REQUIRE(*g::detail::network_simplex(n, circ, zero) == *ex::detail::network_simplex_by_adjacency(n, circ, zero));
		measure("random n=5e4 m=3e5 | library", [&] { return g::min_cost_max_flow(n, edges, 0, 1).cost; });
		measure("random n=5e4 m=3e5 | simplex core library", [&] {
			return g::detail::network_simplex(n, circ, zero)->back();
		});
		measure("random n=5e4 m=3e5 | simplex core by struct arcs", [&] {
			return network_simplex_by_arc_layout<false>(n, circ, zero)->back();
		});
		measure("random n=5e4 m=3e5 | simplex core by flat arcs", [&] {
			return network_simplex_by_arc_layout<true>(n, circ, zero)->back();
		});
		measure("random n=5e4 m=3e5 | simplex core by adjacency", [&] {
			return ex::detail::network_simplex_by_adjacency(n, circ, zero)->back();
		});
	}
}
