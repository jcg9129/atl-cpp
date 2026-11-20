#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/max_flow.hpp>
#include <atl/graph/min_cost_flow.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Min-cost max-flow contenders for the algorithm race.
 *
 * All entries use the template defaults, arguments and owned witness of
 * graph::min_cost_max_flow. Endpoints, nonnegative capacities and limit are
 * asserted. Conversions must be exact. Every intermediate is finite and fits
 * its selected type: Flow covers excesses (sum of incident capacities plus
 * injected supply is sufficient); Cost covers path sums, reduced costs,
 * potentials, cost-times-flow products and partial objective sums. No rounding
 * tolerance is applied. SSP costs need headroom for infinity=max(Cost).
 * Capacity scaling also requires 2*(1+sum|cost|) and its potential differences
 * to fit Cost; cost scaling requires abs((n+1)*cost) and eps*distance to fit Cost.
 * Cost scaling inherits max_flow_value's n < 2^31 and source-capacity sum
 * bounds; residuals allow at most UINT32_MAX/2 edges (closure arc included).
 * Integer-valued costs are required for exact epsilon scaling. All traversals
 * have O(1) call stack and O(n+m) heap storage, apart from Dijkstra's heap
 * (O(m) entries). Allocation failures propagate.
 * @note Not constexpr (uses runtime residual construction and priority queues).
 *
 * Shared output contract: an @ref atl::graph::min_cost_flow_result.
 * The tests verify `min_cost_max_flow_by_spfa`'s per-edge flows, value and dual
 * certificate definitionally (feasibility + no negative residual cycle +
 * non-negative reduced costs); `min_cost_max_flow_by_cost_scaling_price_update`
 * is certified the same way and `detail::network_simplex_by_adjacency` must
 * reproduce the library simplex's flows exactly (tests/graph/min_cost_flow.cpp);
 * the others are checked by flow value and cost in the benchmark binary only.
 */
namespace atl::experimental {
	/**
	 * @brief Successive shortest paths with SPFA (Bellman–Ford queue): the
	 *        folklore "EK-MCMF" — augment along the cheapest residual path,
	 *        found without potentials so negative costs need no
	 *        preprocessing. O(flow · n m) worst case, adversarially slow:
	 *        the readable oracle, not a candidate.
	 * @pre no negative-cost cycle in the input (debug-asserted).
	 * @tparam Flow Flow arithmetic, or void for the edge capacity type.
	 * @tparam Cost Cost arithmetic, or void for the edge cost type.
	 * @tparam EdgeList Const forward costed edge range without lower bounds.
	 * @param n Vertex count; ids are below n and n < none-1.
	 * @param edges Stable input edges in witness order.
	 * @param source Source vertex, distinct from sink.
	 * @param sink Sink vertex.
	 * @param limit Nonnegative maximum value.
	 * @return Owned value, cost and per-edge flow, without a dual.
	 * @pre Numeric intermediates obey the bounds in this file's contract.
	 * @note Not constexpr (runtime residual construction and queue operations).
	 */
	template<typename Flow = void, typename Cost = void, graph::cost_capacitated_edge_list EdgeList>
		requires(graph::detail::cost_network<graph::detail::flow_t<Flow, EdgeList>, graph::detail::cost_type<Cost, EdgeList>, EdgeList>
				 && graph::detail::unbounded_cost_edges<EdgeList>)
	[[nodiscard]] auto min_cost_max_flow_by_spfa(
		std::size_t n, EdgeList const &edges, graph::vertex_id source, graph::vertex_id sink,
		graph::detail::flow_t<Flow, EdgeList> limit = (std::numeric_limits<graph::detail::flow_t<Flow, EdgeList>>::max)()) {
		using F = graph::detail::flow_t<Flow, EdgeList>;
		using C = graph::detail::cost_type<Cost, EdgeList>;
		atl_assert(source < n && sink < n && source != sink && limit >= 0);
		atl_assert(n < graph::none);
		graph::detail::cost_residual<F, C> net(n, edges);
		graph::min_cost_flow_result<F, C> r;

		C constexpr unreached = (std::numeric_limits<C>::max)();
		std::vector<C> dist(n);
		std::vector<uint32> in_arc(n), ring(n + 1);
		std::vector<uint8> queued(n);
		while (r.value < limit) {
			std::ranges::fill(dist, unreached);
			std::ranges::fill(queued, uint8(0));
			dist[source] = 0;
			std::size_t hd = 0, tl = 0;
			[[maybe_unused]] uint64 pops = 0;
			ring[tl++] = source;
			queued[source] = 1;
			while (hd != tl) {
				uint32 const v = ring[hd];
				if (++hd == ring.size()) hd = 0;
				queued[v] = 0;
				atl_assert(++pops <= uint64(n) * uint64(n)); // trips on a negative cycle
				for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
					if (net.capacity[a] <= 0) continue;
					uint32 const w = net.head[a];
					if (dist[v] + net.cost[a] < dist[w]) {
						dist[w] = dist[v] + net.cost[a];
						in_arc[w] = a;
						if (!queued[w]) {
							queued[w] = 1;
							ring[tl] = w;
							if (++tl == ring.size()) tl = 0;
						}
					}
				}
			}
			if (dist[sink] == unreached) break;
			F push = limit - r.value;
			for (uint32 v = sink; v != source; v = net.head[net.mate[in_arc[v]]])
				push = (std::min)(push, net.capacity[in_arc[v]]);
			for (uint32 v = sink; v != source; v = net.head[net.mate[in_arc[v]]]) {
				net.capacity[in_arc[v]] -= push;
				net.capacity[net.mate[in_arc[v]]] += push;
			}
			r.value += push;
			graph::detail::add_flow(r.cost, graph::detail::cost_product(C(push), dist[sink]));
		}

		r.edge_flow = graph::detail::edge_flows(net);
		return r;
	}

	/**
	 * @brief zkw / primal–dual: one Dijkstra round over reduced costs, then a
	 *        DFS saturates the whole admissible (zero-reduced-cost) subgraph
	 *        within each round — many augmenting paths per shortest-path
	 *        computation. Wins when rounds are few and wide (assignment-like,
	 *        dense unit networks); degenerates to plain SSP otherwise.
	 * @pre no negative-cost cycle in the input (debug-asserted during the
	 *      warm-up).
	 * @tparam Flow Flow arithmetic, or void for the edge capacity type.
	 * @tparam Cost Cost arithmetic, or void for the edge cost type.
	 * @tparam EdgeList Const forward costed edge range without lower bounds.
	 * @param n Vertex count; ids are below n and n < none-1.
	 * @param edges Stable input edges in witness order.
	 * @param source Source vertex, distinct from sink.
	 * @param sink Sink vertex.
	 * @param limit Nonnegative maximum value.
	 * @return Owned value, cost and per-edge flow, without a dual.
	 * @pre Numeric intermediates obey the bounds in this file's contract.
	 * @note Not constexpr (runtime residual construction and queue operations).
	 */
	template<typename Flow = void, typename Cost = void, graph::cost_capacitated_edge_list EdgeList>
		requires(graph::detail::cost_network<graph::detail::flow_t<Flow, EdgeList>, graph::detail::cost_type<Cost, EdgeList>, EdgeList>
				 && graph::detail::unbounded_cost_edges<EdgeList>)
	[[nodiscard]] auto min_cost_max_flow_by_zkw(
		std::size_t n, EdgeList const &edges, graph::vertex_id source, graph::vertex_id sink,
		graph::detail::flow_t<Flow, EdgeList> limit = (std::numeric_limits<graph::detail::flow_t<Flow, EdgeList>>::max)()) {
		using F = graph::detail::flow_t<Flow, EdgeList>;
		using C = graph::detail::cost_type<Cost, EdgeList>;
		atl_assert(source < n && sink < n && source != sink && limit >= 0);
		atl_assert(n < graph::none);
		graph::detail::cost_residual<F, C> net(n, edges);
		graph::min_cost_flow_result<F, C> r;
		C constexpr unreached = (std::numeric_limits<C>::max)();

		std::vector<C> pi(n, 0), dist(n);
		bool is_negative = false;
		for (auto const c : net.cost) is_negative = is_negative || c < C(0);
		if (is_negative) {
			std::vector<uint8> queued(n, 0);
			std::vector<uint32> ring(n + 1);
			std::vector<C> d(n, unreached);
			std::size_t hd = 0, tl = 0;
			[[maybe_unused]] uint64 pops = 0;
			d[source] = 0;
			ring[tl++] = source;
			queued[source] = 1;
			while (hd != tl) {
				uint32 const v = ring[hd];
				if (++hd == ring.size()) hd = 0;
				queued[v] = 0;
				atl_assert(++pops <= uint64(n) * uint64(n)); // trips on a negative cycle
				for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
					if (net.capacity[a] <= 0) continue;
					uint32 const w = net.head[a];
					if (d[v] + net.cost[a] < d[w]) {
						d[w] = d[v] + net.cost[a];
						if (!queued[w]) {
							queued[w] = 1;
							ring[tl] = w;
							if (++tl == ring.size()) tl = 0;
						}
					}
				}
			}
			for (uint32 v = 0; v < n; ++v) pi[v] = d[v] == unreached ? C(0) : d[v];
		}

		std::vector<uint32> cur(n);
		std::vector<uint8> vis(n);
		std::priority_queue<std::pair<C, uint32>, std::vector<std::pair<C, uint32>>, std::greater<>>
			heap;

		// Heap frames preserve the recursive DFS's arc order and partial pushes.
		struct frame {
			uint32 vertex;
			F requested, pushed;
		};

		std::vector<frame> stack;
		auto const augment = [&](F requested) {
			stack.clear();
			stack.push_back({source, requested, F(0)});
			vis[source] = 1;
			F result = 0;
			while (!stack.empty()) {
				auto &top_frame = stack.back();
				uint32 const v = top_frame.vertex;
				if (v == sink || top_frame.pushed == top_frame.requested || cur[v] == net.offset[v + 1]) {
					result = v == sink ? top_frame.requested : top_frame.pushed;
					stack.pop_back();
					if (stack.empty()) break;
					auto &caller = stack.back();
					uint32 &a = cur[caller.vertex];
					net.capacity[a] -= result;
					net.capacity[net.mate[a]] += result;
					caller.pushed += result;
					// A full return retains the arc for a later push.
					if (caller.pushed != caller.requested) ++a;
					continue;
				}
				uint32 &a = cur[v];
				uint32 const w = net.head[a];
				if (vis[w] || net.capacity[a] <= 0 || net.cost[a] + pi[v] - pi[w] != C(0)) {
					++a;
					continue;
				}
				F const amount = (std::min)(F(top_frame.requested - top_frame.pushed), net.capacity[a]);
				if (w != sink) vis[w] = 1;
				stack.push_back({w, amount, F(0)});
			}
			return result;
		};
		while (r.value < limit) {
			std::ranges::fill(dist, unreached);
			dist[source] = 0;
			heap.push({0, source});
			C reach = unreached;
			while (!heap.empty()) {
				auto const [d, v] = heap.top();
				heap.pop();
				if (d != dist[v]) continue;
				if (v == sink) {
					reach = d;
					break;
				}
				for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
					if (net.capacity[a] <= 0) continue;
					uint32 const w = net.head[a];
					C const step = C(net.cost[a] + pi[v] - pi[w]);
					if (dist[w] == unreached || d + step < dist[w]) {
						dist[w] = d + step;
						heap.push({dist[w], w});
					}
				}
			}
			while (!heap.empty()) heap.pop();
			if (reach == unreached) break;
			for (uint32 v = 0; v < n; ++v) pi[v] += dist[v] == unreached ? reach : (std::min)(dist[v], reach);
			for (uint32 v = 0; v < n; ++v) cur[v] = net.offset[v];
			while (r.value < limit) {
				std::ranges::fill(vis, uint8(0));
				F const f = augment(limit - r.value);
				if (f <= 0) break;
				r.value += f;
				graph::detail::add_flow(r.cost, graph::detail::cost_product(C(f), C(pi[sink] - pi[source])));
			}
		}

		r.edge_flow = graph::detail::edge_flows(net);
		return r;
	}

	/**
	 * @brief Capacity-scaling SSP (Edmonds–Karp scaling), in its natural
	 *        excess/deficit circulation form: phases with threshold delta
	 *        halving each round; at phase start every residual arc with
	 *        capacity >= delta and negative reduced cost is saturated
	 *        outright, then Dijkstra routes delta-sized excesses to deficits.
	 *        Removes SSP's dependence on the flow value: O(m log U · m log n).
	 *        The s-t problem is reduced to a min-cost circulation via a t->s
	 *        arc of cost -2 · (1 + sum |cost|).
	 * @pre no negative-cost cycle in the input.
	 * @tparam Flow Flow arithmetic, or void for the edge capacity type.
	 * @tparam Cost Cost arithmetic, or void for the edge cost type.
	 * @tparam EdgeList Const forward costed edge range without lower bounds.
	 * @param n Vertex count; ids are below n and n < none-1.
	 * @param edges Stable input edges in witness order.
	 * @param source Source vertex, distinct from sink.
	 * @param sink Sink vertex.
	 * @param limit Nonnegative maximum value.
	 * @return Owned value, cost and per-edge flow, without a dual.
	 * @pre Numeric intermediates obey the bounds in this file's contract.
	 * @note Not constexpr (runtime residual construction and queue operations).
	 */
	template<typename Flow = void, typename Cost = void, graph::cost_capacitated_edge_list EdgeList>
		requires(graph::detail::cost_network<graph::detail::flow_t<Flow, EdgeList>, graph::detail::cost_type<Cost, EdgeList>, EdgeList>
				 && graph::detail::unbounded_cost_edges<EdgeList>)
	[[nodiscard]] auto min_cost_max_flow_by_capacity_scaling(
		std::size_t n, EdgeList const &edges, graph::vertex_id source, graph::vertex_id sink,
		graph::detail::flow_t<Flow, EdgeList> limit = (std::numeric_limits<graph::detail::flow_t<Flow, EdgeList>>::max)()) {
		using F = graph::detail::flow_t<Flow, EdgeList>;
		using C = graph::detail::cost_type<Cost, EdgeList>;
		atl_assert(source < n && sink < n && source != sink && limit >= 0);
		atl_assert(n < graph::none);
		graph::min_cost_flow_result<F, C> r;

		C big = 1;
		F out = 0;
		std::vector<graph::flow_arc<F, C>> arcs;
		for (auto const &e : edges) {
			arcs.push_back({graph::vertex_id(graph::source(e)), graph::vertex_id(graph::target(e)), F(graph::capacity(e)),
							C(graph::cost(e))});
			C const c = C(graph::cost(e));
			atl_assert(c != std::numeric_limits<C>::lowest());
			graph::detail::add_flow(big, c < C(0) ? C(-c) : c);
			if (graph::vertex_id(graph::source(e)) == source)
				out = F(graph::capacity(e)) > limit - out ? limit : F(out + F(graph::capacity(e)));
		}
		big = graph::detail::cost_product(big, C(2)); // any simple-path cost is strictly beaten by one t->s unit
		arcs.push_back({sink, source, (std::min)(limit, out), C(-big)});
		graph::detail::cost_residual<F, C> net(n, arcs);

		C constexpr unreached = (std::numeric_limits<C>::max)();
		std::vector<C> pi(n, 0), dist(n);
		std::vector<F> excess(n, 0);
		std::vector<uint32> in_arc(n);
		std::priority_queue<std::pair<C, uint32>, std::vector<std::pair<C, uint32>>, std::greater<>>
			heap;

		F top = 0;
		for (auto const c : net.capacity) top = (std::max)(top, c);
		F delta = 1;
		while (delta <= top / 2) delta *= 2;
		for (; delta > 0; delta /= 2) {
			for (uint32 v = 0; v < n; ++v)
				for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
					if (net.capacity[a] < delta) continue;
					uint32 const w = net.head[a];
					if (net.cost[a] + pi[v] - pi[w] >= C(0)) continue;
					graph::detail::add_flow(excess[v], F(-net.capacity[a]));
					graph::detail::add_flow(excess[w], net.capacity[a]);
					net.capacity[net.mate[a]] += net.capacity[a];
					net.capacity[a] = 0;
				}
			for (uint32 src = 0; src < n; ++src)
				while (excess[src] >= delta) {
					// Dijkstra over reduced costs in the delta-residual, to
					// the nearest delta-deficit
					std::ranges::fill(dist, unreached);
					dist[src] = 0;
					heap.push({0, src});
					uint32 dst = src;
					C reach = unreached;
					while (!heap.empty()) {
						auto const [d, v] = heap.top();
						heap.pop();
						if (d != dist[v]) continue;
						if (excess[v] <= -delta) {
							dst = v;
							reach = d;
							break;
						}
						for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
							if (net.capacity[a] < delta) continue;
							uint32 const w = net.head[a];
							C const step = C(net.cost[a] + pi[v] - pi[w]);
							if (dist[w] == unreached || d + step < dist[w]) {
								dist[w] = d + step;
								in_arc[w] = a;
								heap.push({dist[w], w});
							}
						}
					}
					while (!heap.empty()) heap.pop();
					if (dst == src) break; // no deficit in reach: next phase
					for (uint32 v = 0; v < n; ++v)
						if (dist[v] != unreached)
							pi[v] += (std::min)(dist[v], reach);
						else
							pi[v] += reach;
					// push the whole bottleneck, not just delta: the threshold
					// only gates which arcs and excesses participate
					F push = (std::min)(excess[src], F(-excess[dst]));
					for (uint32 v = dst; v != src; v = net.head[net.mate[in_arc[v]]])
						push = (std::min)(push, net.capacity[in_arc[v]]);
					for (uint32 v = dst; v != src; v = net.head[net.mate[in_arc[v]]]) {
						net.capacity[in_arc[v]] -= push;
						net.capacity[net.mate[in_arc[v]]] += push;
					}
					excess[src] -= push;
					excess[dst] += push;
				}
		}

		r.edge_flow = graph::detail::edge_flows(net);
		r.value = r.edge_flow.back();
		r.edge_flow.pop_back();
		for (std::size_t i = 0; i < r.edge_flow.size(); ++i) graph::detail::add_flow(r.cost, graph::detail::cost_product(C(r.edge_flow[i]), arcs[i].cost));
		return r;
	}

	namespace detail {
		/**
		 * @brief The network simplex with adjacency lists: the same
		 *        pricing and leaving-arc rule as `graph::detail::network_simplex`
		 *        (hence the same pivots and flows), but the spanning tree kept
		 *        as per-vertex adjacency lists with O(1) slot bookkeeping and
		 *        the re-hung subtree walked with an explicit stack, rewriting
		 *        `parent`/`parent_arc`/`depth` of every vertex in it.
		 *
		 * Lost to the thread-index tree on every shape (simplex core
		 * alone, newest record 2026-10-15, x86-64 Xeon Platinum 8358, plain
		 * -O2, library core with flat arc arrays): random n=2e4 m=1.2e5 64.1
		 * vs 42.9 ms, assignment 300x300 10.9 vs 7.1, grid 80x80 123 vs
		 * 53.8, parallel paths 4e3 4.0 vs 2.4, negative DAG 118 vs 50.5,
		 * random n=5e4 m=3e5 279 vs 177. The stack
		 * walk over heap-allocated adjacency rewrote parent pointers the
		 * re-hang leaves unchanged; the thread walk writes potentials only.
		 * @pre as `graph::detail::network_simplex`
		 */
		template<graph::signed_flow_value Flow, graph::detail::cost_arithmetic Cost>
			requires(std::semiregular<Flow> && std::convertible_to<int, Flow>)
		[[nodiscard]] std::optional<std::vector<Flow>> network_simplex_by_adjacency(
			std::size_t n, std::vector<graph::flow_arc<Flow, Cost>> const &arcs, std::vector<Flow> const &b) {
			using F = Flow;
			using C = Cost;
			std::size_t const m = arcs.size();
			atl_assert(n < graph::none - 1 && m <= std::size_t(graph::none) - n && b.size() == n);
			uint32 const root = uint32(n);
			C largest = 0;
			for (auto const &e : arcs) {
				atl_assert(e.cost != std::numeric_limits<C>::lowest());
				largest = (std::max)(largest, e.cost < C(0) ? C(-e.cost) : e.cost);
			}
			C const artificial_cost = graph::detail::simplex_artificial_cost(n, largest);
			F const infinite_capacity = (std::numeric_limits<F>::max)();

			std::vector<F> x(m, 0);
			std::vector<int8> state(m + n, 1); // 1 at lower bound, -1 at upper, 0 in tree
			std::vector<C> potentials(n + 1);
			potentials[root] = 0;
			std::vector<uint32> parent(n + 1, root), parent_arc(n + 1), depth(n + 1, 1);
			parent[root] = root;
			depth[root] = 0;

			/**
			 * @brief An artificial arc (index m + v) between vertex v and the
			 *        root; together they start as the tree, oriented along the
			 *        supply so the initial tree is strongly feasible.
			 */
			struct artificial_arc {
				bool is_outward; ///< oriented vertex -> root (else root -> vertex)
				F flow;
			};

			std::vector<artificial_arc> artificial_arcs(n);
			std::vector<std::vector<std::pair<uint32, uint32>>> adj(n + 1); // (neighbor, arc)

			auto const from = [&](uint32 a) {
				return a < m ? uint32(arcs[a].from) : artificial_arcs[a - m].is_outward ? a - uint32(m)
																						: root;
			};
			auto const to = [&](uint32 a) {
				return a < m ? uint32(arcs[a].to) : artificial_arcs[a - m].is_outward ? root
																					  : a - uint32(m);
			};
			auto const acost = [&](uint32 a) { return a < m ? arcs[a].cost : artificial_cost; };
			auto const flow = [&](uint32 a) -> F & { return a < m ? x[a] : artificial_arcs[a - m].flow; };
			auto const capacity = [&](uint32 a) { return a < m ? arcs[a].capacity : infinite_capacity; };
			auto const residual = [&](uint32 a, bool is_along) {
				return is_along ? F(capacity(a) - flow(a)) : F(flow(a));
			};
			// tree adjacency with per-arc positions so a pivot's attach/detach
			// is O(1) — a linear scan of adj[root] would go quadratic when the
			// supplies make all n artificial arcs pivot out through the root
			std::vector<uint32> slot_at_from(m + n), slot_at_to(m + n);
			auto const attach = [&](uint32 a) {
				uint32 const u = from(a), v = to(a);
				slot_at_from[a] = uint32(adj[u].size());
				adj[u].emplace_back(v, a);
				slot_at_to[a] = uint32(adj[v].size());
				adj[v].emplace_back(u, a);
			};
			auto const detach_at = [&](uint32 vertex, uint32 pos) {
				auto &l = adj[vertex];
				l[pos] = l.back();
				l.pop_back();
				if (pos < l.size()) {
					uint32 const moved = l[pos].second;
					(from(moved) == vertex ? slot_at_from[moved] : slot_at_to[moved]) = pos;
				}
			};

			for (uint32 v = 0; v < n; ++v) {
				atl_assert(b[v] != std::numeric_limits<F>::lowest());
				artificial_arcs[v] = {b[v] >= 0, b[v] >= 0 ? b[v] : F(-b[v])};
				potentials[v] = artificial_arcs[v].is_outward ? -artificial_cost : artificial_cost;
				uint32 const a = uint32(m + v);
				state[a] = 0;
				parent_arc[v] = a;
				attach(a);
			}

			std::vector<uint32> stack;
			std::size_t const block = (std::max<std::size_t>)(64, std::size_t(std::sqrt(double(m + n))));
			std::size_t probe = 0;
			while (true) {
				uint32 enter = uint32(m + n);
				C best = 0;
				for (std::size_t seen = 0; seen < m + n; seen += (std::min)(block, m + n - seen)) {
					for (std::size_t k = 0; k < block; ++k, ++probe) {
						if (probe == m + n) probe = 0;
						C const rc = C(acost(uint32(probe)) + potentials[from(uint32(probe))] - potentials[to(uint32(probe))]);
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
				// a-side arcs push parent->v (the cycle runs join->a there),
				// b-side arcs push v->parent; "last blocking from the apex"
				// makes ties strict on the a-side and inclusive on the b-side
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
				uint32 va = a, vb = bb;
				while (va != vb)
					if (depth[va] >= depth[vb])
						step(va, true);
					else
						step(vb, false);
				flow(enter) = F(flow(enter) + (is_forward ? delta : F(-delta)));
				for (uint32 v = a; v != va; v = parent[v]) {
					uint32 const pa = parent_arc[v];
					flow(pa) = F(flow(pa) + (to(pa) == v ? delta : F(-delta)));
				}
				for (uint32 v = bb; v != va; v = parent[v]) {
					uint32 const pa = parent_arc[v];
					flow(pa) = F(flow(pa) + (from(pa) == v ? delta : F(-delta)));
				}
				if (leave == enter) {
					state[enter] = int8(-state[enter]);
					continue;
				}
				// the leaving arc's detached side re-hangs off the entering arc
				state[leave] = flow(leave) <= 0 ? int8(1) : int8(-1);
				detach_at(from(leave), slot_at_from[leave]);
				detach_at(to(leave), slot_at_to[leave]);
				state[enter] = 0;
				attach(enter);
				uint32 const cut = leaves_on_a ? a : bb; // entering endpoint inside the cut subtree
				uint32 const hook = leaves_on_a ? bb : a;
				C const rc = C(acost(enter) + potentials[from(enter)] - potentials[to(enter)]);
				C const shift = cut == from(enter) ? -rc : rc;
				parent[cut] = hook;
				parent_arc[cut] = enter;
				depth[cut] = depth[hook] + 1;
				potentials[cut] += shift;
				stack.clear();
				stack.push_back(cut);
				while (!stack.empty()) {
					uint32 const v = stack.back();
					stack.pop_back();
					for (auto const &[w, ta] : adj[v]) {
						if (w == parent[v] && parent_arc[v] == ta) continue;
						parent[w] = v;
						parent_arc[w] = ta;
						depth[w] = depth[v] + 1;
						potentials[w] += shift;
						stack.push_back(w);
					}
				}
			}
			for (uint32 v = 0; v < n; ++v)
				if (artificial_arcs[v].flow != 0) return std::nullopt;
			return x;
		}
	}

	namespace detail {
		/**
		 * @brief The cost-scaling engine behind both cost-scaling contenders;
		 *        @p has_price_update adds Goldberg's global price update.
		 */
		template<bool has_price_update, graph::signed_flow_value F, graph::signed_flow_value C, graph::cost_capacitated_edge_list EdgeList>
			requires(graph::detail::cost_network<F, C, EdgeList> && std::numeric_limits<C>::is_integer)
		[[nodiscard]] graph::min_cost_flow_result<F, C> cost_scaling_engine(std::size_t n, EdgeList const &edges,
																			graph::vertex_id source, graph::vertex_id sink, F limit) {
			atl_assert(source < n && sink < n && source != sink && limit >= 0);
			atl_assert(n < graph::none);
			graph::min_cost_flow_result<F, C> r;

			std::vector<graph::flow_arc<F>> plain;
			for (auto const &e : edges)
				plain.push_back({graph::vertex_id(graph::source(e)), graph::vertex_id(graph::target(e)), F(graph::capacity(e))});
			r.value = (std::min)(limit, graph::max_flow_value(n, plain, source, sink));

			graph::detail::cost_residual<F, C> net(n, edges);
			C eps = 1;
			for (auto &c : net.cost) {
				c = graph::detail::cost_product(c, C(n + 1));
				atl_assert(c != std::numeric_limits<C>::lowest());
				eps = (std::max)(eps, c < C(0) ? C(-c) : c);
			}
			std::vector<C> pi(n, 0);
			std::vector<F> excess(n, 0);
			std::vector<uint32> cur(n), ring(n + 1);
			std::vector<uint8> queued(n);
			std::vector<C> label;
			std::vector<uint8> settled;
			using entry = std::pair<C, uint32>;
			std::priority_queue<entry, std::vector<entry>, std::greater<>> heap;
			// global price update (Goldberg 1997): d(v) = shortest distance from v
			// to a deficit vertex under arc lengths floor(c_p / eps) + 1 (>= 0 on
			// an eps-optimal residual), then p(v) -= eps * d(v). It keeps
			// eps-optimality (d(v) <= len + d(w) on every residual arc) and makes
			// a downhill path to a deficit admissible from every active vertex.
			// Dijkstra stops once every active vertex is settled; the rest take
			// the last settled distance, which no residual arc can violate.
			auto const price_update = [&](C phase_eps) {
				C const unset = (std::numeric_limits<C>::max)();
				label.assign(n, unset);
				settled.assign(n, 0);
				std::size_t active = 0;
				for (uint32 v = 0; v < n; ++v) {
					if (excess[v] > 0) ++active;
					if (excess[v] < 0) {
						label[v] = 0;
						heap.push({0, v});
					}
				}
				C reached = 0;
				while (!heap.empty() && active > 0) {
					auto const [d, w] = heap.top();
					heap.pop();
					if (settled[w] || d != label[w]) continue;
					settled[w] = 1;
					reached = d;
					if (excess[w] > 0) --active;
					for (uint32 a = net.offset[w]; a < net.offset[w + 1]; ++a) {
						uint32 const b = net.mate[a], x = net.head[a]; // b is the residual arc x -> w
						if (settled[x] || net.capacity[b] <= 0) continue;
						C const cp = C(net.cost[b] + pi[x] - pi[w]);
						C const nd = C(d + (cp < C(0) ? C(0) : C(cp / phase_eps + 1)));
						if (nd < label[x]) {
							label[x] = nd;
							heap.push({nd, x});
						}
					}
				}
				heap = {};
				for (uint32 v = 0; v < n; ++v) graph::detail::add_flow(pi[v], C(-graph::detail::cost_product(phase_eps, settled[v] ? label[v] : reached)));
			};
			// the supply is injected once: the first refine routes it, every
			// later refine starts from a feasible flow and only repairs the
			// excesses its own saturation step creates
			excess[source] += r.value;
			excess[sink] -= r.value;
			while (true) {
				for (uint32 v = 0; v < n; ++v)
					for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
						if (net.capacity[a] <= 0 || net.cost[a] + pi[v] - pi[net.head[a]] >= C(0)) continue;
						graph::detail::add_flow(excess[v], F(-net.capacity[a]));
						graph::detail::add_flow(excess[net.head[a]], net.capacity[a]);
						net.capacity[net.mate[a]] += net.capacity[a];
						net.capacity[a] = 0;
					}
				if constexpr (has_price_update) price_update(eps);
				[[maybe_unused]] std::size_t relabels = 0;
				std::size_t hd = 0, tl = 0;
				std::ranges::fill(queued, uint8(0));
				for (uint32 v = 0; v < n; ++v) {
					cur[v] = net.offset[v];
					if (excess[v] > 0) {
						ring[tl++] = v;
						queued[v] = 1;
					}
				}
				while (hd != tl) {
					uint32 const v = ring[hd];
					if (++hd == ring.size()) hd = 0;
					queued[v] = 0;
					while (excess[v] > 0) {
						if (cur[v] == net.offset[v + 1]) {
							C best = std::numeric_limits<C>::lowest();
							for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a)
								if (net.capacity[a] > 0) best = (std::max)(best, C(pi[net.head[a]] - net.cost[a]));
							pi[v] = best - eps;
							cur[v] = net.offset[v];
							if constexpr (has_price_update)
								if (++relabels == n) {
									relabels = 0;
									price_update(eps);
									for (uint32 x = 0; x < n; ++x) cur[x] = net.offset[x];
								}
						}
						uint32 const a = cur[v];
						uint32 const w = net.head[a];
						if (net.capacity[a] <= 0 || net.cost[a] + pi[v] - pi[w] >= C(0)) {
							++cur[v];
							continue;
						}
						F const amount = (std::min)(excess[v], net.capacity[a]);
						net.capacity[a] -= amount;
						net.capacity[net.mate[a]] += amount;
						excess[v] -= amount;
						graph::detail::add_flow(excess[w], amount);
						if (excess[w] > 0 && !queued[w]) {
							queued[w] = 1;
							ring[tl] = w;
							if (++tl == ring.size()) tl = 0;
						}
					}
				}
				// feasible again: excess is back to zero everywhere (the source
				// and sink included — their imbalance was consumed as supply)
				if (eps == C(1)) break;
				eps = (std::max)(C(1), C(eps / C(8)));
			}

			r.edge_flow = graph::detail::edge_flows(net);
			for (std::size_t i = 0; auto const &e : edges) graph::detail::add_flow(r.cost, graph::detail::cost_product(C(r.edge_flow[i++]), C(graph::cost(e))));
			return r;
		}
	}

	/**
	 * @brief Cost-scaling push-relabel (Goldberg–Tarjan): costs are scaled by
	 *        n+1 and an eps-optimal pseudoflow is refined through
	 *        geometrically shrinking eps — saturate every negative-reduced-
	 *        cost arc, then FIFO push/relabel the excesses back until
	 *        feasible; eps < 1 on the scaled costs means optimal. The flow
	 *        value is fixed up front by @ref atl::graph::max_flow_value, so no
	 *        circulation gadget (a huge synthetic cost would stretch the
	 *        scaling range) is needed. Bare refine: local relabels are the
	 *        only price change. O(n^2 m log(n·maxcost)) worst case,
	 *        typically far less. Loses to the library network simplex on
	 *        every shape: by 3.6-9.2x on four, 2.9 vs 2.6 ms on parallel
	 *        paths (numbers at @ref atl::experimental::min_cost_max_flow_by_cost_scaling_price_update).
	 * @pre Integer Cost type (constrained); no negative-cost cycle in the input.
	 * @tparam Flow Flow arithmetic, or void for the edge capacity type.
	 * @tparam Cost Cost arithmetic, or void for the edge cost type.
	 * @tparam EdgeList Const forward costed edge range without lower bounds.
	 * @param n Vertex count; ids are below n and n < none-1.
	 * @param edges Stable input edges in witness order.
	 * @param source Source vertex, distinct from sink.
	 * @param sink Sink vertex.
	 * @param limit Nonnegative maximum value.
	 * @return Owned value, cost and per-edge flow, without a dual.
	 * @pre Numeric intermediates obey the bounds in this file's contract.
	 * @note Not constexpr (runtime residual construction and queue operations).
	 */
	template<typename Flow = void, typename Cost = void, graph::cost_capacitated_edge_list EdgeList>
		requires(graph::detail::cost_network<graph::detail::flow_t<Flow, EdgeList>, graph::detail::cost_type<Cost, EdgeList>, EdgeList>
				 && graph::detail::unbounded_cost_edges<EdgeList>
				 && std::numeric_limits<graph::detail::cost_type<Cost, EdgeList>>::is_integer)
	[[nodiscard]] auto min_cost_max_flow_by_cost_scaling(
		std::size_t n, EdgeList const &edges, graph::vertex_id source, graph::vertex_id sink,
		graph::detail::flow_t<Flow, EdgeList> limit = (std::numeric_limits<graph::detail::flow_t<Flow, EdgeList>>::max)()) {
		return detail::cost_scaling_engine<false, graph::detail::flow_t<Flow, EdgeList>,
										   graph::detail::cost_type<Cost, EdgeList>>(n, edges, source, sink, limit);
	}

	/**
	 * @brief @ref atl::experimental::min_cost_max_flow_by_cost_scaling plus Goldberg's global
	 *        price update (a Dijkstra in units of eps from the deficits,
	 *        run at the start of every refine and once per n relabels),
	 *        the heuristic practical cost-scaling codes rely on.
	 *
	 * The update is worth 1.45-5.63x over the bare refine except on parallel
	 * paths, but the verdict stands (newest record 2026-10-15, x86-64 Xeon
	 * Platinum 8358, plain -O2; bare / with update / library network
	 * simplex): random n=2e4 602 / 107 / 65.3 ms, assignment 300x300 48.7 /
	 * 33.7 / 10.9, grid 80x80 706 / 206 / 86.1, parallel paths 4e3 2.9 / 5.7
	 * / 2.6, negative DAG 191 / 71.5 / 53.3.
	 * @pre Integer Cost type (constrained); no negative-cost cycle in the input.
	 * @tparam Flow Flow arithmetic, or void for the edge capacity type.
	 * @tparam Cost Cost arithmetic, or void for the edge cost type.
	 * @tparam EdgeList Const forward costed edge range without lower bounds.
	 * @param n Vertex count; ids are below n and n < none-1.
	 * @param edges Stable input edges in witness order.
	 * @param source Source vertex, distinct from sink.
	 * @param sink Sink vertex.
	 * @param limit Nonnegative maximum value.
	 * @return Owned value, cost and per-edge flow, without a dual.
	 * @pre Numeric intermediates obey the bounds in this file's contract.
	 * @note Not constexpr (runtime residual construction and queue operations).
	 */
	template<typename Flow = void, typename Cost = void, graph::cost_capacitated_edge_list EdgeList>
		requires(graph::detail::cost_network<graph::detail::flow_t<Flow, EdgeList>, graph::detail::cost_type<Cost, EdgeList>, EdgeList>
				 && graph::detail::unbounded_cost_edges<EdgeList>
				 && std::numeric_limits<graph::detail::cost_type<Cost, EdgeList>>::is_integer)
	[[nodiscard]] auto min_cost_max_flow_by_cost_scaling_price_update(
		std::size_t n, EdgeList const &edges, graph::vertex_id source, graph::vertex_id sink,
		graph::detail::flow_t<Flow, EdgeList> limit = (std::numeric_limits<graph::detail::flow_t<Flow, EdgeList>>::max)()) {
		return detail::cost_scaling_engine<true, graph::detail::flow_t<Flow, EdgeList>,
										   graph::detail::cost_type<Cost, EdgeList>>(n, edges, source, sink, limit);
	}

	/**
	 * @brief Successive shortest paths with Johnson
	 *        potentials: augment along the cheapest residual s->t path,
	 *        found by Dijkstra over reduced costs (`cost + pi[u] - pi[v]`,
	 *        non-negative by induction); potentials absorb each round's
	 *        distances. Negative edge costs are handled by one Bellman-Ford
	 *        warm-up that seeds the potentials.
	 *
	 * Maximizes the flow value first (up to @p limit, if given), and among
	 * those flows minimizes the total cost. Certify with
	 * `graph::min_cost_flow_potentials`.
	 *
	 * @pre no negative-cost *cycle* in the input (debug-asserted during the
	 * warm-up); costs may be negative otherwise.
	 * @complexity O(flow · m log n) — pseudo-polynomial in the value, the
	 * classic SSP trade; suitable for moderate network sizes
	 * @tparam Flow Flow arithmetic, or void for the edge capacity type.
	 * @tparam Cost Cost arithmetic, or void for the edge cost type.
	 * @tparam EdgeList Const forward costed edge range without lower bounds.
	 * @param n Vertex count; ids are below n and n < none-1.
	 * @param edges Stable input edges in witness order.
	 * @param source Source vertex, distinct from sink.
	 * @param sink Sink vertex.
	 * @param limit Nonnegative maximum value.
	 * @return Owned value, cost and per-edge flow, without a dual.
	 * @pre Numeric intermediates obey the bounds in this file's contract.
	 * @note Not constexpr (runtime residual construction and queue operations).
	 */
	template<typename Flow = void, typename Cost = void, graph::cost_capacitated_edge_list EdgeList>
		requires(graph::detail::cost_network<graph::detail::flow_t<Flow, EdgeList>, graph::detail::cost_type<Cost, EdgeList>, EdgeList>
				 && graph::detail::unbounded_cost_edges<EdgeList>)
	[[nodiscard]] auto min_cost_max_flow_by_dijkstra(
		std::size_t n, EdgeList const &edges, graph::vertex_id source, graph::vertex_id sink,
		graph::detail::flow_t<Flow, EdgeList> limit = (std::numeric_limits<graph::detail::flow_t<Flow, EdgeList>>::max)()) {
		using F = graph::detail::flow_t<Flow, EdgeList>;
		using C = graph::detail::cost_type<Cost, EdgeList>;
		atl_assert(source < n && sink < n && source != sink && limit >= 0);
		atl_assert(n < graph::none);
		graph::detail::cost_residual<F, C> net(n, edges);
		graph::min_cost_flow_result<F, C> r;
		C constexpr unreached = (std::numeric_limits<C>::max)();

		std::vector<C> pi(n, 0), dist(n);
		bool is_negative = false;
		for (auto const c : net.cost) is_negative = is_negative || c < C(0);
		if (is_negative) {
			// Bellman-Ford queue from the source seeds valid potentials;
			// vertices unreachable from the source can never join an
			// augmenting path, so their zero stays harmless
			std::vector<uint8> queued(n, 0);
			std::vector<uint32> ring(n + 1);
			std::vector<C> d(n, unreached);
			std::size_t hd = 0, tl = 0;
			[[maybe_unused]] uint64 pops = 0;
			d[source] = 0;
			ring[tl++] = source;
			queued[source] = 1;
			while (hd != tl) {
				uint32 const v = ring[hd];
				if (++hd == ring.size()) hd = 0;
				queued[v] = 0;
				atl_assert(++pops <= uint64(n) * uint64(n)); // trips on a negative cycle
				for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
					if (net.capacity[a] <= 0) continue;
					uint32 const w = net.head[a];
					if (d[v] + net.cost[a] < d[w]) {
						d[w] = d[v] + net.cost[a];
						if (!queued[w]) {
							queued[w] = 1;
							ring[tl] = w;
							if (++tl == ring.size()) tl = 0;
						}
					}
				}
			}
			for (uint32 v = 0; v < n; ++v) pi[v] = d[v] == unreached ? C(0) : d[v];
		}

		std::vector<uint32> in_arc(n);
		std::priority_queue<std::pair<C, uint32>, std::vector<std::pair<C, uint32>>, std::greater<>>
			heap;
		while (r.value < limit) {
			std::ranges::fill(dist, unreached);
			dist[source] = 0;
			heap.push({0, source});
			C reach = unreached;
			while (!heap.empty()) {
				auto const [d, v] = heap.top();
				heap.pop();
				if (d != dist[v]) continue;
				if (v == sink) { // stop at the sink: labels past it are dead work
					reach = d;
					break;
				}
				for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
					if (net.capacity[a] <= 0) continue;
					uint32 const w = net.head[a];
					C const step = C(net.cost[a] + pi[v] - pi[w]);
					if (dist[w] == unreached || d + step < dist[w]) {
						dist[w] = d + step;
						in_arc[w] = a;
						heap.push({dist[w], w});
					}
				}
			}
			while (!heap.empty()) heap.pop();
			if (reach == unreached) break;
			// unfinished labels clamp to the sink's distance: still a valid
			// dual, and only finalized vertices lie on the augmenting path
			for (uint32 v = 0; v < n; ++v) pi[v] += dist[v] == unreached ? reach : (std::min)(dist[v], reach);
			F push = limit - r.value;
			C path_cost = 0;
			for (uint32 v = sink; v != source; v = net.head[net.mate[in_arc[v]]]) {
				push = (std::min)(push, net.capacity[in_arc[v]]);
				path_cost += net.cost[in_arc[v]];
			}
			for (uint32 v = sink; v != source; v = net.head[net.mate[in_arc[v]]]) {
				net.capacity[in_arc[v]] -= push;
				net.capacity[net.mate[in_arc[v]]] += push;
			}
			r.value += push;
			graph::detail::add_flow(r.cost, graph::detail::cost_product(C(push), path_cost));
		}

		r.edge_flow = graph::detail::edge_flows(net);
		return r;
	}
}
