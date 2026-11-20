#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/max_flow.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Reference max-flow implementations benchmarked against
 *        HLPP (atl::graph::max_flow), kept for study, as test oracles, and
 *        as benchmark baselines.
 *
 * Same output contract: a @ref atl::graph::max_flow_result whose per-edge flows and minimum cut are
 * verified definitionally (weak duality) by every test.
 *
 * Measured (benchmarks/graph/max_flow.cpp, record 2026-10-15, x86-64 Xeon
 * Platinum 8358, GCC 15.2, RelWithDebInfo, assertions disabled,
 * -O2 -g -DNDEBUG -std=c++23; HLPP / ISAP / Dinic / scaling Dinic, ms): random
 * n=1e5 m=5e5 111 / 59.5 / 177 / 417; bipartite unit 25.1 / 30.2 / 36.9 /
 * 38.3; grid 300x300 162 / 1103 / 3340 / 7952; dense high-cap 34.2 / 24.1 /
 * 31.3 / 270; layered unit 2.6 / 1.6 / 1.7 / 1.8. HLPP wins the hard shapes
 * (grid, unit bipartite matching) while ISAP takes the instances where every
 * contender finishes fast. Dinic wins no shape despite its unit-capacity
 * pedigree.
 *
 * Edmonds–Karp and scaling Edmonds–Karp are excluded from timing: both are test oracles, O(n m^2 + n + m) and O((n+m) m log(U+1) + n + m), too
 * slow to finish the race sizes, so they have no benchmark row.
 *
 * Capacity scaling (timed as scaling Dinic) is a measured *negative* result: it never won a shape and
 * takes 7952 versus 3340 ms for plain Dinic on grids and 270 versus
 * 31.3 ms on dense high-capacity networks. Scaling repairs
 * algorithms whose running time depends on the flow value (the
 * Ford–Fulkerson family); Dinic's phase bound is value-independent, so the
 * Delta ladder only replays phase work log U times. Kept for study. Thresholding ISAP would require
 * distance labels for each threshold residual; that contender is not timed.
 * Push-relabel's excess scaling thresholds excess rather than capacities.
 */
/** @brief Alternative implementations for comparison and independent validation. */
namespace atl::experimental {
	/**
	 * @brief Edmonds–Karp: repeatedly push along a BFS-shortest augmenting
	 *        path — the readable reference the faster contenders are raced
	 *        against, not a candidate itself.
	 *
	 * The final (failing) BFS's visited set is exactly the source side of a
	 * minimum cut, so the certificate costs nothing extra.
	 * @tparam Flow Computation type, void for `capacity_t<EdgeList>`; semiregular flow_value
	 * accepting implicit integer zero and constructible from scanned capacities
	 * (integer-only for capacity scaling).
	 * @tparam EdgeList Const forward capacity range without lower bounds.
	 * @param n Number of vertices.
	 * @param edges Directed edges; loops and parallel edges are supported.
	 * @param source Source vertex.
	 * @param sink Distinct sink vertex.
	 * @pre source, sink < n < 2^31, source != sink, at most UINT32_MAX/2 edges;
	 * endpoints < n and capacities >= 0 (asserted). Conversions must be exact;
	 * the maximum value must fit Flow (accumulation asserted). Floating arithmetic
	 * obeys atl::graph::max_flow's finite/exact intermediate requirements.
	 * @return Owned feasible per-input-edge flow and minimum-cut certificate.
	 * @note Uses O(n+m) heap storage and O(1) call stack; no recursion per vertex.
	 * Allocation failures propagate.
	 * @complexity O(n m^2 + n + m)
	 */
	template<typename Flow = void, graph::capacitated_edge_list EdgeList>
		requires(graph::detail::flow_network<graph::detail::flow_t<Flow, EdgeList>, EdgeList>
				 && !graph::bounded_capacitated_edge<std::ranges::range_value_t<EdgeList>>
				 && !graph::bounded_capacitated_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>)
	[[nodiscard]] constexpr auto max_flow_by_edmonds_karp(std::size_t n, EdgeList const &edges, graph::vertex_id source, graph::vertex_id sink) {
		using F = graph::detail::flow_t<Flow, EdgeList>;
		atl_assert(source < n && sink < n && source != sink);
		graph::detail::residual<F> net(n, edges);
		graph::max_flow_result<F> r;

		std::vector<uint32> in_arc(n);
		std::vector<uint8> seen(n);
		std::vector<uint32> queue;
		queue.reserve(n);
		for (;;) {
			std::ranges::fill(seen, uint8(0));
			queue.assign(1, source);
			seen[source] = 1;
			for (std::size_t at = 0; at < queue.size() && !seen[sink]; ++at) {
				uint32 const v = queue[at];
				for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
					uint32 const w = net.head[a];
					if (net.capacity[a] > 0 && !seen[w]) {
						seen[w] = 1;
						in_arc[w] = a;
						queue.push_back(w);
					}
				}
			}
			if (!seen[sink]) break;
			F push = std::numeric_limits<F>::max();
			for (uint32 v = sink; v != source; v = net.head[net.mate[in_arc[v]]])
				push = std::min(push, net.capacity[in_arc[v]]);
			for (uint32 v = sink; v != source; v = net.head[net.mate[in_arc[v]]]) {
				net.capacity[in_arc[v]] -= push;
				net.capacity[net.mate[in_arc[v]]] += push;
			}
			atl_assert(push <= std::numeric_limits<F>::max() - r.value);
			r.value += push;
		}

		r.source_side = std::move(seen);
		r.edge_flow = graph::detail::edge_flows(net);
		return r;
	}

	/**
	 * @brief Dinic's algorithm: BFS level phases (stopped the moment the sink
	 *        is labeled — deeper vertices cannot lie on an admissible path) +
	 *        blocking flow found by an iterative DFS with current-arc
	 *        pointers; each augmentation remembers its shallowest bottleneck
	 *        arc, so the retreat is free.
	 *
	 * The final (failing, complete) level BFS marks exactly the source side
	 * of a minimum cut. On the recorded grid it takes 3340 ms versus 162 ms for HLPP.
	 * @tparam Flow Computation type, void for `capacity_t<EdgeList>`; semiregular flow_value
	 * accepting implicit integer zero and constructible from scanned capacities
	 * (integer-only for capacity scaling).
	 * @tparam EdgeList Const forward capacity range without lower bounds.
	 * @param n Number of vertices.
	 * @param edges Directed edges; loops and parallel edges are supported.
	 * @param source Source vertex.
	 * @param sink Distinct sink vertex.
	 * @pre source, sink < n < 2^31, source != sink, at most UINT32_MAX/2 edges;
	 * endpoints < n and capacities >= 0 (asserted). Conversions must be exact;
	 * the maximum value must fit Flow (accumulation asserted). Floating arithmetic
	 * obeys atl::graph::max_flow's finite/exact intermediate requirements.
	 * @return Owned feasible per-input-edge flow and minimum-cut certificate.
	 * @note Uses O(n+m) heap storage and O(1) call stack; no recursion per vertex.
	 * Allocation failures propagate.
	 * @complexity O(n^2 m + n + m) in general, but the bounds for unit-capacity networks are
	 *             the specialized ones: O(m sqrt(n)) on unit-capacity
	 *             bipartite graphs (plus O(n) initialization, where it *is* Hopcroft–Karp) and
	 *             O(n + m min(n^(2/3), sqrt(m))) on unit-capacity graphs
	 */
	template<typename Flow = void, graph::capacitated_edge_list EdgeList>
		requires(graph::detail::flow_network<graph::detail::flow_t<Flow, EdgeList>, EdgeList>
				 && !graph::bounded_capacitated_edge<std::ranges::range_value_t<EdgeList>>
				 && !graph::bounded_capacitated_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>)
	[[nodiscard]] constexpr auto max_flow_by_dinic(std::size_t n, EdgeList const &edges, graph::vertex_id source, graph::vertex_id sink) {
		using F = graph::detail::flow_t<Flow, EdgeList>;
		atl_assert(source < n && sink < n && source != sink);
		graph::detail::residual<F> net(n, edges);
		graph::max_flow_result<F> r;

		std::vector<uint32> level(n), it(n), queue, path;
		queue.reserve(n);
		auto const leveled = [&] {
			std::ranges::fill(level, graph::none);
			queue.assign(1, source);
			level[source] = 0;
			for (std::size_t at = 0; at < queue.size(); ++at) {
				uint32 const v = queue[at];
				for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
					uint32 const w = net.head[a];
					if (net.capacity[a] > 0 && level[w] == graph::none) {
						level[w] = level[v] + 1;
						if (w == sink) return true;
						queue.push_back(w);
					}
				}
			}
			return false;
		};

		while (leveled()) {
			for (uint32 v = 0; v < n; ++v) it[v] = net.offset[v];
			path.clear();
			uint32 v = source;
			for (;;) {
				if (v == sink) {
					F push = std::numeric_limits<F>::max();
					std::size_t cut = 0; // shallowest bottleneck: the retreat point
					for (std::size_t j = 0; j < path.size(); ++j)
						if (net.capacity[path[j]] < push) {
							push = net.capacity[path[j]];
							cut = j;
						}
					for (uint32 const a : path) {
						net.capacity[a] -= push;
						net.capacity[net.mate[a]] += push;
					}
					atl_assert(push <= std::numeric_limits<F>::max() - r.value);
					r.value += push;
					v = net.head[net.mate[path[cut]]];
					path.resize(cut);
				} else if (it[v] < net.offset[v + 1]) {
					uint32 const a = it[v];
					if (net.capacity[a] > 0 && level[net.head[a]] == level[v] + 1) {
						path.push_back(a);
						v = net.head[a];
					} else {
						++it[v];
					}
				} else {
					if (v == source) break;
					uint32 const a = path.back(); // dead end: retire the arc that led here
					path.pop_back();
					v = net.head[net.mate[a]];
					++it[v];
				}
			}
		}

		r.source_side.assign(n, 0);
		for (uint32 v = 0; v < n; ++v) r.source_side[v] = level[v] != graph::none;
		r.edge_flow = graph::detail::edge_flows(net);
		return r;
	}

	/**
	 * @brief ISAP (improved shortest augmenting path): one DFS walk guided by
	 *        distance-to-sink lower bounds initialized by reverse BFS and repaired
	 *        incrementally on dead ends — no per-phase BFS like Dinic. The gap heuristic
	 *        terminates the moment a distance level empties (the sink is then
	 *        unreachable below it).
	 *
	 * Sparse random networks take 59.5 ms versus HLPP's 111 ms (record above).
	 * @tparam Flow Computation type, void for `capacity_t<EdgeList>`; semiregular flow_value
	 * accepting implicit integer zero and constructible from scanned capacities
	 * (integer-only for capacity scaling).
	 * @tparam EdgeList Const forward capacity range without lower bounds.
	 * @param n Number of vertices.
	 * @param edges Directed edges; loops and parallel edges are supported.
	 * @param source Source vertex.
	 * @param sink Distinct sink vertex.
	 * @pre source, sink < n < 2^31, source != sink, at most UINT32_MAX/2 edges;
	 * endpoints < n and capacities >= 0 (asserted). Conversions must be exact;
	 * the maximum value must fit Flow (accumulation asserted). Floating arithmetic
	 * obeys atl::graph::max_flow's finite/exact intermediate requirements.
	 * @return Owned feasible per-input-edge flow and minimum-cut certificate.
	 * @note Uses O(n+m) heap storage and O(1) call stack; no recursion per vertex.
	 * Allocation failures propagate.
	 * @complexity O(n^2 m + n + m) worst case
	 */
	template<typename Flow = void, graph::capacitated_edge_list EdgeList>
		requires(graph::detail::flow_network<graph::detail::flow_t<Flow, EdgeList>, EdgeList>
				 && !graph::bounded_capacitated_edge<std::ranges::range_value_t<EdgeList>>
				 && !graph::bounded_capacitated_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>)
	[[nodiscard]] constexpr auto max_flow_by_isap(std::size_t n, EdgeList const &edges, graph::vertex_id source, graph::vertex_id sink) {
		using F = graph::detail::flow_t<Flow, EdgeList>;
		atl_assert(source < n && sink < n && source != sink);
		graph::detail::residual<F> net(n, edges);
		graph::max_flow_result<F> r;

		std::vector<uint32> dist(n, uint32(n)), gap(n + 1, 0), it(net.offset.begin(), net.offset.end() - 1);
		{
			std::vector<uint32> queue{sink};
			dist[sink] = 0;
			for (std::size_t at = 0; at < queue.size(); ++at) {
				uint32 const w = queue[at];
				for (uint32 a = net.offset[w]; a < net.offset[w + 1]; ++a) {
					uint32 const x = net.head[a]; // mate[a] is the arc x -> w
					if (net.capacity[net.mate[a]] > 0 && dist[x] == n && x != sink) {
						dist[x] = dist[w] + 1;
						queue.push_back(x);
					}
				}
			}
		}
		for (uint32 v = 0; v < n; ++v)
			if (dist[v] < n) ++gap[dist[v]];

		std::vector<uint32> path;
		uint32 v = source;
		while (dist[source] < n) {
			if (v == sink) {
				F push = std::numeric_limits<F>::max();
				std::size_t cut = 0; // shallowest bottleneck: the retreat point
				for (std::size_t j = 0; j < path.size(); ++j)
					if (net.capacity[path[j]] < push) {
						push = net.capacity[path[j]];
						cut = j;
					}
				for (uint32 const a : path) {
					net.capacity[a] -= push;
					net.capacity[net.mate[a]] += push;
				}
				atl_assert(push <= std::numeric_limits<F>::max() - r.value);
				r.value += push;
				v = net.head[net.mate[path[cut]]];
				path.resize(cut);
				continue;
			}
			bool has_advanced = false;
			for (; it[v] < net.offset[v + 1]; ++it[v]) {
				uint32 const a = it[v];
				if (net.capacity[a] > 0 && dist[v] == dist[net.head[a]] + 1) {
					path.push_back(a);
					v = net.head[a];
					has_advanced = true;
					break;
				}
			}
			if (has_advanced) continue;
			uint32 lifted = uint32(n); // relabel to one above the closest residual neighbor
			for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a)
				if (net.capacity[a] > 0) lifted = std::min(lifted, dist[net.head[a]] + 1);
			if (--gap[dist[v]] == 0) break; // level empty: the sink is cut off
			dist[v] = std::min<uint32>(lifted, uint32(n));
			if (dist[v] < n) ++gap[dist[v]];
			it[v] = net.offset[v];
			if (v != source) {
				v = net.head[net.mate[path.back()]];
				path.pop_back();
			}
		}

		r.source_side = graph::detail::source_side(net, source);
		r.edge_flow = graph::detail::edge_flows(net);
		return r;
	}

	/**
	 * @brief Capacity-scaling Edmonds–Karp: augment only along paths of
	 *        bottleneck >= Delta, halving Delta from the largest power of two
	 *        under the maximum capacity down to 1, replacing value-dependent
	 *        behavior.
	 *
	 * Integer capacities only (rejected at compile time otherwise): the Delta
	 * ladder and its termination argument have no floating-point meaning.
	 * Excluded from timing: a test oracle, too slow for the race
	 * sizes; capacity scaling is timed through scaling Dinic instead.
	 * @tparam Flow Computation type, void for `capacity_t<EdgeList>`; semiregular flow_value
	 * accepting implicit integer zero and constructible from scanned capacities
	 * (integer-only for capacity scaling).
	 * @tparam EdgeList Const forward capacity range without lower bounds.
	 * @param n Number of vertices.
	 * @param edges Directed edges; loops and parallel edges are supported.
	 * @param source Source vertex.
	 * @param sink Distinct sink vertex.
	 * @pre source, sink < n < 2^31, source != sink, at most UINT32_MAX/2 edges;
	 * endpoints < n and capacities >= 0 (asserted). Conversions must be exact;
	 * the maximum value must fit Flow (accumulation asserted). Floating arithmetic
	 * obeys atl::graph::max_flow's finite/exact intermediate requirements.
	 * @return Owned feasible per-input-edge flow and minimum-cut certificate.
	 * @note Uses O(n+m) heap storage and O(1) call stack; no recursion per vertex.
	 * Allocation failures propagate.
	 * @complexity O((n+m) m log(U+1) + n + m)
	 */
	template<typename Flow = void, graph::capacitated_edge_list EdgeList>
		requires(graph::detail::flow_network<graph::detail::flow_t<Flow, EdgeList>, EdgeList>
				 && !graph::bounded_capacitated_edge<std::ranges::range_value_t<EdgeList>>
				 && !graph::bounded_capacitated_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>
				 && (atl::integral<graph::detail::flow_t<Flow, EdgeList>>
					 || (std::integral<graph::detail::flow_t<Flow, EdgeList>>
						 && !std::same_as<graph::detail::flow_t<Flow, EdgeList>, bool>)))
	[[nodiscard]] constexpr auto max_flow_by_capacity_scaling_edmonds_karp(std::size_t n, EdgeList const &edges, graph::vertex_id source,
																		   graph::vertex_id sink) {
		using F = graph::detail::flow_t<Flow, EdgeList>;
		atl_assert(source < n && sink < n && source != sink);
		graph::detail::residual<F> net(n, edges);
		graph::max_flow_result<F> r;

		F top = 0;
		for (auto const c : net.capacity) top = std::max(top, c);
		std::vector<uint32> in_arc(n), queue;
		std::vector<uint8> seen(n);
		queue.reserve(n);
		F first_delta = top > 0 ? F(1) : F(0);
		while (first_delta > 0 && first_delta <= top / 2) first_delta = F(first_delta * 2);
		for (F delta = first_delta; delta >= 1; delta = F(delta >> 1)) {
			for (;;) {
				std::ranges::fill(seen, uint8(0));
				queue.assign(1, source);
				seen[source] = 1;
				for (std::size_t at = 0; at < queue.size() && !seen[sink]; ++at) {
					uint32 const v = queue[at];
					for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
						uint32 const w = net.head[a];
						if (net.capacity[a] >= delta && !seen[w]) {
							seen[w] = 1;
							in_arc[w] = a;
							queue.push_back(w);
						}
					}
				}
				if (!seen[sink]) break;
				F push = std::numeric_limits<F>::max();
				for (uint32 v = sink; v != source; v = net.head[net.mate[in_arc[v]]])
					push = std::min(push, net.capacity[in_arc[v]]);
				for (uint32 v = sink; v != source; v = net.head[net.mate[in_arc[v]]]) {
					net.capacity[in_arc[v]] -= push;
					net.capacity[net.mate[in_arc[v]]] += push;
				}
				atl_assert(push <= std::numeric_limits<F>::max() - r.value);
				r.value += push;
			}
		}

		r.source_side = graph::detail::source_side(net, source);
		r.edge_flow = graph::detail::edge_flows(net);
		return r;
	}

	/**
	 * @brief Capacity-scaling Dinic: the level graph and the blocking-flow
	 *        DFS admit only arcs with residual >= Delta, so every
	 *        augmentation moves at least Delta and each scale performs O(m)
	 *        of them.
	 *
	 * Integer capacities only (rejected at compile time otherwise).
	 * Dense high-capacity networks take 270 ms versus 31.3 ms for plain Dinic. The retreat from an augmentation must scan
	 * for the first arc under Delta (not the bottleneck: an earlier arc can
	 * fall below the threshold without being the minimum).
	 * @tparam Flow Computation type, void for `capacity_t<EdgeList>`; semiregular flow_value
	 * accepting implicit integer zero and constructible from scanned capacities
	 * (integer-only for capacity scaling).
	 * @tparam EdgeList Const forward capacity range without lower bounds.
	 * @param n Number of vertices.
	 * @param edges Directed edges; loops and parallel edges are supported.
	 * @param source Source vertex.
	 * @param sink Distinct sink vertex.
	 * @pre source, sink < n < 2^31, source != sink, at most UINT32_MAX/2 edges;
	 * endpoints < n and capacities >= 0 (asserted). Conversions must be exact;
	 * the maximum value must fit Flow (accumulation asserted). Floating arithmetic
	 * obeys atl::graph::max_flow's finite/exact intermediate requirements.
	 * @return Owned feasible per-input-edge flow and minimum-cut certificate.
	 * @note Uses O(n+m) heap storage and O(1) call stack; no recursion per vertex.
	 * Allocation failures propagate.
	 * @complexity O(n m log(U+1) + n + m)
	 */
	template<typename Flow = void, graph::capacitated_edge_list EdgeList>
		requires(graph::detail::flow_network<graph::detail::flow_t<Flow, EdgeList>, EdgeList>
				 && !graph::bounded_capacitated_edge<std::ranges::range_value_t<EdgeList>>
				 && !graph::bounded_capacitated_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>
				 && (atl::integral<graph::detail::flow_t<Flow, EdgeList>>
					 || (std::integral<graph::detail::flow_t<Flow, EdgeList>>
						 && !std::same_as<graph::detail::flow_t<Flow, EdgeList>, bool>)))
	[[nodiscard]] constexpr auto max_flow_by_capacity_scaling_dinic(std::size_t n, EdgeList const &edges, graph::vertex_id source, graph::vertex_id sink) {
		using F = graph::detail::flow_t<Flow, EdgeList>;
		atl_assert(source < n && sink < n && source != sink);
		graph::detail::residual<F> net(n, edges);
		graph::max_flow_result<F> r;

		F top = 0;
		for (auto const c : net.capacity) top = std::max(top, c);
		std::vector<uint32> level(n), it(n), queue, path;
		queue.reserve(n);
		F first_delta = top > 0 ? F(1) : F(0);
		while (first_delta > 0 && first_delta <= top / 2) first_delta = F(first_delta * 2);
		for (F delta = first_delta; delta >= 1; delta = F(delta >> 1)) {
			auto const leveled = [&] {
				std::ranges::fill(level, graph::none);
				queue.assign(1, source);
				level[source] = 0;
				for (std::size_t at = 0; at < queue.size(); ++at) {
					uint32 const v = queue[at];
					for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
						uint32 const w = net.head[a];
						if (net.capacity[a] >= delta && level[w] == graph::none) {
							level[w] = level[v] + 1;
							if (w == sink) return true;
							queue.push_back(w);
						}
					}
				}
				return false;
			};

			while (leveled()) {
				for (uint32 v = 0; v < n; ++v) it[v] = net.offset[v];
				path.clear();
				uint32 v = source;
				for (;;) {
					if (v == sink) {
						F push = std::numeric_limits<F>::max();
						for (uint32 const a : path) push = std::min(push, net.capacity[a]);
						for (uint32 const a : path) {
							net.capacity[a] -= push;
							net.capacity[net.mate[a]] += push;
						}
						atl_assert(push <= std::numeric_limits<F>::max() - r.value);
						r.value += push;
						std::size_t cut = 0;
						while (net.capacity[path[cut]] >= delta) ++cut;
						v = net.head[net.mate[path[cut]]];
						path.resize(cut);
					} else if (it[v] < net.offset[v + 1]) {
						uint32 const a = it[v];
						if (net.capacity[a] >= delta && level[net.head[a]] == level[v] + 1) {
							path.push_back(a);
							v = net.head[a];
						} else {
							++it[v];
						}
					} else {
						if (v == source) break;
						uint32 const a = path.back();
						path.pop_back();
						v = net.head[net.mate[a]];
						++it[v];
					}
				}
			}
		}

		r.source_side = graph::detail::source_side(net, source);
		r.edge_flow = graph::detail::edge_flows(net);
		return r;
	}
}
