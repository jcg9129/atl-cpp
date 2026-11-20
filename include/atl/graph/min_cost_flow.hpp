#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/detail/edge_count.hpp>
#include <atl/graph/max_flow.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::graph {
	/**
	 * @brief An owned optimum flow witness: the flow value, its
	 *        cost and a witness (per-edge flows). The dual
	 *        certificate is a separate call: @ref atl::graph::min_cost_flow_potentials.
	 * @tparam Flow Signed capacity and flow arithmetic.
	 * @tparam Cost Signed objective and potential arithmetic.
	 */
	template<signed_flow_value Flow, signed_flow_value Cost>
	struct min_cost_flow_result {
		Flow value{};				 /**< the flow value (min_cost_max_flow) or the supplied amount (min_cost_feasible_flow) */
		Cost cost{};				 ///< Sum of cost times flow, accumulated in input order.
		std::vector<Flow> edge_flow; ///< flow on each input edge, parallel to the input
	};

	namespace detail {
		template<typename Cost>
		concept cost_arithmetic = signed_flow_value<Cost> && std::semiregular<Cost>
							   && std::convertible_to<int, Cost> && std::constructible_from<Cost, std::size_t>
							   && requires(Cost a, Cost b) {
									  { a * b } -> std::convertible_to<Cost>;
									  { a / b } -> std::convertible_to<Cost>;
									  a *= b;
								  };

		template<typename Flow, typename Cost, typename EdgeList>
		concept cost_network = bounded_flow_network<Flow, EdgeList> && cost_capacitated_edge_list<EdgeList>
							&& cost_arithmetic<Cost>
							&& requires(std::ranges::range_reference_t<EdgeList const> e, Flow flow) {
								   Cost(graph::cost(e));
								   Cost(flow);
							   };

		template<typename EdgeList>
		concept unbounded_cost_edges = !bounded_capacitated_edge<std::ranges::range_value_t<EdgeList>>
									&& !bounded_capacitated_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>;

		// Division checks the product without first forming an overflowing bound.
		template<cost_arithmetic Value>
		[[nodiscard]] constexpr Value cost_product(Value a, Value b) {
			if (a > 0) {
				atl_assert(b > 0 ? a <= (std::numeric_limits<Value>::max)() / b
								 : b >= std::numeric_limits<Value>::lowest() / a);
			} else if (a < 0) {
				atl_assert(b > 0 ? a >= std::numeric_limits<Value>::lowest() / b
								 : b == 0 || a >= (std::numeric_limits<Value>::max)() / b);
			}
			return Value(a * b);
		}

		template<signed_flow_value Value>
		[[nodiscard]] constexpr Value flow_difference(Value value, Value subtract) {
			atl_assert(subtract >= 0 ? value >= std::numeric_limits<Value>::lowest() + subtract
									 : value <= (std::numeric_limits<Value>::max)() + subtract);
			return Value(value - subtract);
		}

		template<cost_arithmetic Cost>
		[[nodiscard]] Cost simplex_artificial_cost(std::size_t n, Cost largest) {
			atl_assert(n < none - 1);
			if constexpr (std::numeric_limits<Cost>::is_integer
						  && std::numeric_limits<Cost>::digits < std::numeric_limits<std::size_t>::digits)
				atl_assert(n + 1 <= std::size_t((std::numeric_limits<Cost>::max)()));
			atl_assert(largest < (std::numeric_limits<Cost>::max)() / Cost(5) / Cost(n + 1));
			return Cost(Cost(n + 1) * (largest + 1));
		}

		template<typename Cost, typename EdgeList>
		using cost_type = std::conditional_t<std::is_void_v<Cost>, cost_t<EdgeList>, Cost>;

		/**
		 * @brief The costed residual: max_flow's scan-friendly layout plus a
		 *        per-arc cost array (`-cost` on the reverse arcs).
		 */
		template<signed_flow_value Flow, cost_arithmetic Cost>
			requires(std::semiregular<Flow> && std::convertible_to<int, Flow>)
		struct cost_residual {
			std::vector<uint32> offset, head, mate, input_arc;
			std::vector<Flow> capacity;
			std::vector<Cost> cost;

			template<cost_capacitated_edge_list EdgeList>
				requires cost_network<Flow, Cost, EdgeList>
			cost_residual(std::size_t n, EdgeList const &edges) {
				auto const m = detail::checked_edge_count(edges, (std::numeric_limits<uint32>::max)() / 2);
				atl_assert(n < none - 1 && m <= (std::numeric_limits<uint32>::max)() / 2);
				offset.assign(n + 1, 0);
				for (auto const &e : edges) {
					atl_assert(graph::source(e) < n && graph::target(e) < n
							   && graph::capacity(e) >= 0);
					++offset[graph::source(e) + 1];
					++offset[graph::target(e) + 1];
				}
				for (std::size_t v = 0; v < n; ++v) offset[v + 1] += offset[v];
				head.resize(2 * m);
				capacity.resize(2 * m);
				cost.resize(2 * m);
				mate.resize(2 * m);
				input_arc.resize(m);
				std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
				std::size_t i = 0;
				for (auto const &e : edges) {
					uint32 const a = cursor[graph::source(e)]++;
					uint32 const b = cursor[graph::target(e)]++;
					head[a] = graph::target(e);
					capacity[a] = Flow(graph::capacity(e));
					cost[a] = Cost(graph::cost(e));
					head[b] = graph::source(e);
					capacity[b] = 0;
					atl_assert(cost[a] != std::numeric_limits<Cost>::lowest());
					cost[b] = -cost[a];
					mate[a] = b;
					mate[b] = a;
					input_arc[i++] = a;
				}
			}
		};

		template<signed_flow_value Flow, cost_arithmetic Cost>
			requires(std::semiregular<Flow> && std::convertible_to<int, Flow>)
		[[nodiscard]] std::vector<Flow> edge_flows(cost_residual<Flow, Cost> const &net) {
			std::vector<Flow> flow(net.input_arc.size());
			for (std::size_t i = 0; i < flow.size(); ++i) flow[i] = net.capacity[net.mate[net.input_arc[i]]];
			return flow;
		}

		/**
		 * @brief A valid dual for an *optimal* residual: shortest distances
		 *        from a virtual all-vertex source (Bellman-Ford queue) —
		 *        well-defined exactly because the optimal residual has no
		 *        negative cycle. Every contender reports its certificate
		 *        through this one computation.
		 * @pre @p net is the residual of a cost-optimal flow (no negative cycle;
		 * asserted: a queue-based pass pops at most n·(n + 1) times otherwise)
		 */
		template<signed_flow_value Flow, cost_arithmetic Cost>
			requires(std::semiregular<Flow> && std::convertible_to<int, Flow>)
		[[nodiscard]] std::vector<Cost> dual_potentials(cost_residual<Flow, Cost> const &net) {
			auto const n = net.offset.size() - 1;
			std::vector<Cost> potentials(n, 0);
			std::vector<uint8> queued(n, 1); // a vertex sits in the ring at most once
			std::vector<uint32> ring(n + 1);
			std::size_t hd = 0, tl = 0;
			[[maybe_unused]] uint64 pops = 0;
			for (uint32 v = 0; v < n; ++v) ring[tl++] = v;
			while (hd != tl) {
				++pops;
				atl_assert(pops <= uint64(n) * (uint64(n) + 1)); // FIFO rounds: n pops each, under n rounds without a negative cycle
				uint32 const v = ring[hd];
				if (++hd == ring.size()) hd = 0;
				queued[v] = 0;
				for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
					if (net.capacity[a] <= 0) continue;
					uint32 const w = net.head[a];
					Cost candidate = potentials[v];
					add_flow(candidate, net.cost[a]);
					if (candidate < potentials[w]) {
						potentials[w] = candidate;
						if (!queued[w]) {
							queued[w] = 1;
							ring[tl] = w;
							if (++tl == ring.size()) tl = 0;
						}
					}
				}
			}
			return potentials;
		}
	}

	namespace detail {
		/**
		 * @brief One network-simplex solve: primal simplex on the
		 *        spanning-tree polytope (Ahuja–Magnanti–Orlin ch. 11).
		 *
		 * Maintains a strongly feasible spanning tree rooted at an artificial
		 * vertex, pivots in a most-violating arc found by wraparound block
		 * search, pushes around the induced cycle and kicks out the last
		 * blocking arc from the apex (this keeps the tree strongly feasible,
		 * which is what rules out degenerate cycling). Potentials of the
		 * re-hung subtree shift by a constant, so a pivot costs one cycle
		 * walk plus one subtree walk.
		 *
		 * The tree is a preorder thread (`thread`/`rev_thread`, cyclic
		 * through the root) with subtree sizes and each subtree's last
		 * vertex: a subtree is the segment `[v .. last[v]]`, so re-hanging
		 * the cut-off subtree S is unlink the segment, re-root it at the
		 * entering endpoint by splicing the stem pieces (the endpoint's detached
		 * segment, then for each stem vertex up to S's detached root its segment
		 * minus the preceding stem vertex's), and link it in after the hook;
		 * `last` and `size` change only along the stem and the two paths to
		 * the apex, and the apex is found by subtree sizes (an ancestor's is
		 * strictly larger). The potential shift is then one walk of |S|
		 * thread links. Raced against per-vertex adjacency lists walked with
		 * an explicit stack (experimental `network_simplex_by_adjacency`; see
		 * its docstring for the numbers).
		 *
		 * The artificial-root arcs cost `(n + 1) · (max|cost| + 1)`, more than
		 * any simple path of real arcs, so a b-flow that exists never keeps
		 * artificial flow. Tree potentials stay within one artificial arc plus
		 * a simple path, and reduced costs within twice that, so the cost type
		 * must hold `5 (n + 1) (max|cost| + 1)` (asserted, overflow-checked).
		 * @param arcs The network; endpoints in `[0, n)`, capacities `>= 0`.
		 * @param b The vertex supplies (positive = surplus to ship out).
		 * @return The per-arc flows, or nullopt if no b-flow exists.
		 */
		template<signed_flow_value Flow, cost_arithmetic Cost>
			requires(std::semiregular<Flow> && std::convertible_to<int, Flow>)
		[[nodiscard]] std::optional<std::vector<Flow>> network_simplex(std::size_t n, std::vector<flow_arc<Flow, Cost>> const &arcs,
																	   std::vector<Flow> const &b) {
			using F = Flow;
			using C = Cost;
			std::size_t const m = arcs.size();
			atl_assert(n < none - 1 && m <= std::size_t(none) - n && b.size() == n);
			uint32 const root = uint32(n);
			C largest = 0;
			for (auto const &e : arcs) {
				atl_assert(e.cost != std::numeric_limits<C>::lowest());
				largest = (std::max)(largest, e.cost < C(0) ? C(-e.cost) : e.cost);
			}
			C const artificial_cost = simplex_artificial_cost(n, largest);
			F const infinite_capacity = (std::numeric_limits<F>::max)();

			std::vector<F> x(m, 0);
			std::vector<int8> state(m + n, 1); // 1 at lower bound, -1 at upper, 0 in tree
			std::vector<C> potentials(n + 1, 0);
			std::vector<uint32> parent(n + 1, root), parent_arc(n + 1), thread(n + 1), rev_thread(n + 1),
				size(n + 1, 1), last(n + 1);
			parent[root] = none;
			size[root] = uint32(n + 1);

			std::vector<uint8> is_outward(n);
			std::vector<F> artificial_flow(n);
			// every arc's tail, head and cost materialized once (artificial arcs
			// oriented at setup), so a pricing probe is three flat loads with
			// no branch: 4-15% faster on every benchmark shape (record
			// 2026-10-15, Xeon Platinum 8358, both flag sets)
			std::vector<uint32> tail(m + n), head(m + n);
			std::vector<C> arc_cost(m + n);
			for (std::size_t a = 0; a < m; ++a) {
				tail[a] = arcs[a].from;
				head[a] = arcs[a].to;
				arc_cost[a] = arcs[a].cost;
			}
			auto const from = [&](uint32 a) { return tail[a]; };
			auto const to = [&](uint32 a) { return head[a]; };
			auto const acost = [&](uint32 a) { return arc_cost[a]; };
			auto const flow = [&](uint32 a) -> F & { return a < m ? x[a] : artificial_flow[a - m]; };
			auto const capacity = [&](uint32 a) { return a < m ? arcs[a].capacity : infinite_capacity; };
			auto const residual = [&](uint32 a, bool is_along) { return is_along ? F(capacity(a) - flow(a)) : F(flow(a)); };
			auto const link = [&](uint32 u, uint32 v) {
				thread[u] = v;
				rev_thread[v] = u;
			};

			for (uint32 v = 0; v < n; ++v) {
				atl_assert(b[v] != std::numeric_limits<F>::lowest());
				is_outward[v] = b[v] >= 0;
				artificial_flow[v] = b[v] >= 0 ? b[v] : -b[v];
				potentials[v] = is_outward[v] ? -artificial_cost : artificial_cost;
				state[m + v] = 0;
				parent_arc[v] = uint32(m + v);
				tail[m + v] = is_outward[v] ? v : root;
				head[m + v] = is_outward[v] ? root : v;
				arc_cost[m + v] = artificial_cost;
				last[v] = v;
			}
			for (uint32 v = 0; v < n; ++v) link(v == 0 ? root : v - 1, v);
			link(n == 0 ? root : root - 1, root);
			last[root] = n == 0 ? root : root - 1;

			std::vector<uint32> stem, stem_last, stem_prev, stem_after, stem_size;
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
				flow(enter) = F(flow(enter) + (is_forward ? delta : F(-delta)));
				for (uint32 v = a; v != apex; v = parent[v]) {
					uint32 const pa = parent_arc[v];
					flow(pa) = F(flow(pa) + (to(pa) == v ? delta : F(-delta)));
				}
				for (uint32 v = bb; v != apex; v = parent[v]) {
					uint32 const pa = parent_arc[v];
					flow(pa) = F(flow(pa) + (from(pa) == v ? delta : F(-delta)));
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
				uint32 segment_tail = stem_last[0];
				for (std::size_t i = 1; i <= k; ++i) {
					link(segment_tail, stem[i]);
					segment_tail = stem_prev[i - 1];
					if (stem_last[i] != stem_last[i - 1]) {
						link(segment_tail, stem_after[i - 1]);
						segment_tail = stem_last[i];
					}
				}
				link(segment_tail, thread[hook]);
				link(hook, cut);

				// last: the chain above S whose segment ended in S, then the
				// stem, then the chain whose segment ended at the hook
				for (uint32 v = out_parent; v != none && last[v] == segment_last; v = parent[v]) last[v] = predecessor;
				for (auto const v : stem) last[v] = segment_tail;
				for (uint32 v = hook; v != none && last[v] == hook; v = parent[v]) last[v] = segment_tail;
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

				C const rc = C(acost(enter) + potentials[from(enter)] - potentials[to(enter)]);
				C const shift = cut == from(enter) ? -rc : rc;
				uint32 v = cut;
				for (uint32 i = 0; i < moved; ++i, v = thread[v]) potentials[v] += shift;
			}
			for (uint32 v = 0; v < n; ++v)
				if (artificial_flow[v] != 0) return std::nullopt;
			return x;
		}
	}

	/**
	 * @brief Min-cost max-flow: maximize the s->t flow value (up to @p limit,
	 *        if given), and among those flows minimize sum of
	 *        `flow(e) · cost(e)`. Costs may be negative; capacities must not.
	 *
	 * Solved as a min-cost circulation (a t->s arc whose reward beats every
	 * simple path) by the primal network simplex — the winner of the
	 * benchmark race against SSP (SPFA and Dijkstra+potentials), zkw,
	 * capacity scaling and cost-scaling push-relabel on all five measured
	 * shapes (see
	 * docs/min-cost-flow.md; the contenders live in
	 * experimental/atl/experimental/graph/min_cost_flow_variants.hpp).
	 *
	 * Optimality can be certified separately by @ref atl::graph::min_cost_flow_potentials
	 * (a dual, by LP duality); the solver itself skips that post-pass, which
	 * costs about a third of the run.
	 *
	 * Negative-cost *cycles* are fine too — the optimum saturates them, as a
	 * min-cost flow must (the experimental SSP-family contenders cannot).
	 *
	 * @pre the cost type must hold n^2 · max|cost| (internal simplex
	 * potentials reach that order; the exact `5 (n + 1) (n · max|cost| + 2)`
	 * is asserted, overflow-checked); endpoints in `[0, n)`, capacities and
	 * @p limit nonnegative (asserted)
	 * @complexity no useful worst-case bound for this pivot rule; in practice
	 * the fastest contender on the recorded five shapes. Pivots walk
	 * the re-hung subtree, which starts to bite around n ~ 5e4 — costless
	 * bounded-flow problems at that scale belong to @ref atl::graph::max_flow via the
	 * double-max-flow reduction instead
	 * @tparam Flow Signed arithmetic type, or void for the capacity type.
	 * @tparam Cost Signed arithmetic type, or void for the edge cost type.
	 * @tparam EdgeList Stable const forward range of costed capacitated edges.
	 * @param n Vertex count, below none-1; none is never a vertex.
	 * @param edges Edges in output order; storage is not modified or retained.
	 * @pre Conversions to Flow and Cost are exact. Every capacity, supply sum,
	 * lower-bound supply adjustment, negated demand and flow fits Flow.
	 * Every cost-times-flow product and partial cost sum fits Cost.
	 * Floating arithmetic is finite and exactly represents all intermediates;
	 * no tolerance is applied. There are at most UINT32_MAX-n simplex arcs,
	 * and all vector sizes fit max_size (index bounds asserted).
	 * @note Not constexpr (network-simplex block sizing uses std::sqrt).
	 * All traversal uses heap storage with O(1) call stack; allocation failures propagate.
	 * @param source Source vertex, distinct from sink.
	 * @param sink Sink vertex.
	 * @param limit Nonnegative value cap.
	 * @return Owned optimum value, cost and per-edge flow witness.
	 * @complexity O(n+m) heap storage; each pivot costs O(n+sqrt(n+m)) apart from pricing,
	 * whose full unsuccessful search costs O(n+m). The pivot count has no polynomial bound.
	 */
	template<typename Flow = void, typename Cost = void, cost_capacitated_edge_list EdgeList>
		requires(detail::cost_network<detail::flow_t<Flow, EdgeList>, detail::cost_type<Cost, EdgeList>, EdgeList>
				 && detail::unbounded_cost_edges<EdgeList>)
	[[nodiscard]] auto min_cost_max_flow(
		std::size_t n, EdgeList const &edges, vertex_id source, vertex_id sink,
		detail::flow_t<Flow, EdgeList> limit = (std::numeric_limits<detail::flow_t<Flow, EdgeList>>::max)()) {
		using F = detail::flow_t<Flow, EdgeList>;
		using C = detail::cost_type<Cost, EdgeList>;
		atl_assert(n < none - 1 && source < n && sink < n && source != sink && limit >= 0);
		min_cost_flow_result<F, C> r;

		C largest = 0;
		F out = 0;
		std::vector<flow_arc<F, C>> arcs;
		if constexpr (std::ranges::sized_range<EdgeList const>) {
			atl_assert(std::ranges::size(edges) < arcs.max_size());
			arcs.reserve(std::ranges::size(edges) + 1);
		}
		for (auto const &e : edges) {
			atl_assert(graph::source(e) < n && graph::target(e) < n && capacity(e) >= 0);
			arcs.push_back({vertex_id(graph::source(e)), vertex_id(graph::target(e)), F(graph::capacity(e)),
							C(graph::cost(e))});
			atl_assert(arcs.size() < std::size_t(none) - n);
			C const c = arcs.back().cost;
			atl_assert(c != std::numeric_limits<C>::lowest());
			largest = (std::max)(largest, c < C(0) ? C(-c) : c);
			// saturating: several "infinite" source edges must not overflow
			if (vertex_id(graph::source(e)) == source)
				out = F(capacity(e)) > limit - out ? limit : F(out + F(capacity(e)));
		}
		// a t->s reward beyond any simple s-t path cost makes every
		// augmenting path a negative cycle, so the optimum is a maximum flow
		if constexpr (std::numeric_limits<C>::is_integer
					  && std::numeric_limits<C>::digits < std::numeric_limits<std::size_t>::digits)
			atl_assert(n <= std::size_t((std::numeric_limits<C>::max)()));
		atl_assert(largest < ((std::numeric_limits<C>::max)() - 1) / C(n));
		arcs.push_back({sink, source, (std::min)(limit, out), C(-(C(n) * largest + 1))});
		auto x = std::move(*detail::network_simplex(n, arcs, std::vector<F>(n, 0)));
		r.value = x.back();
		x.pop_back();
		r.edge_flow = std::move(x);
		arcs.pop_back();
		for (std::size_t i = 0; i < r.edge_flow.size(); ++i) detail::add_flow(r.cost, detail::cost_product(C(r.edge_flow[i]), arcs[i].cost));
		return r;
	}

	/**
	 * @brief Min-cost b-flow: find flows `lower(e) <= f(e) <= capacity(e)`
	 *        whose divergence at every vertex v equals `b[v]` (positive =
	 *        supply to ship out, negative = demand; `sum b = 0`), minimizing
	 *        the sum of `f(e) · cost(e)` — or `std::nullopt` when no such
	 *        flow exists. Costs may be negative, and negative-cost cycles are
	 *        saturated as the optimum requires.
	 *
	 * The general form of min-cost flow: `min_cost_max_flow` is the special
	 * case b = 0 plus a sink-to-source arc. Edge types with lower bounds
	 * (`bounded_capacitated_edge`) are honored by shifting them into the
	 * supplies; plain capacitated edges mean `lower = 0`.
	 *
	 * In the result, `value` is the total positive supply (sum of positive
	 * `b[v]`). Optimality is certified on demand by
	 * @ref atl::graph::min_cost_flow_potentials.
	 *
	 * @p b is a sink: it is moved in and reused as the working supply vector.
	 *
	 * @pre `lower(e) >= 0`, `capacity(e) >= lower(e)` and endpoints in
	 * `[0, n)` (asserted); the cost type must hold n · max|cost| (the exact
	 * `5 (n + 1) (max|cost| + 1)` is asserted, overflow-checked)
	 * @tparam Flow Signed arithmetic type, or void for the capacity type.
	 * @tparam Cost Signed arithmetic type, or void for the edge cost type.
	 * @tparam EdgeList Stable const forward range of costed capacitated edges.
	 * @param n Vertex count, below none-1; none is never a vertex.
	 * @param edges Edges in output order; storage is not modified or retained.
	 * @pre Conversions to Flow and Cost are exact. Every capacity, supply sum,
	 * lower-bound supply adjustment, negated demand and flow fits Flow.
	 * Every cost-times-flow product and partial cost sum fits Cost.
	 * Floating arithmetic is finite and exactly represents all intermediates;
	 * no tolerance is applied. There are at most UINT32_MAX-n simplex arcs,
	 * and all vector sizes fit max_size (index bounds asserted).
	 * @note Not constexpr (network-simplex block sizing uses std::sqrt).
	 * All traversal uses heap storage with O(1) call stack; allocation failures propagate.
	 * @param b Supplies, taken by value and reused; b.size()==n (asserted).
	 * @return Owned optimum witness, or nullopt for infeasible or unbalanced supplies.
	 * @complexity Same network-simplex pivot and O(n+m) storage bounds as min_cost_max_flow.
	 */
	template<typename Flow = void, typename Cost = void, cost_capacitated_edge_list EdgeList>
		requires detail::cost_network<detail::flow_t<Flow, EdgeList>, detail::cost_type<Cost, EdgeList>, EdgeList>
	[[nodiscard]] auto min_cost_feasible_flow(std::size_t n, EdgeList const &edges,
											  std::vector<detail::flow_t<Flow, EdgeList>> b)
		-> std::optional<min_cost_flow_result<detail::flow_t<Flow, EdgeList>, detail::cost_type<Cost, EdgeList>>> {
		using F = detail::flow_t<Flow, EdgeList>;
		using C = detail::cost_type<Cost, EdgeList>;
		using edge = std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>;
		atl_assert(n < none - 1 && b.size() == n);
		auto const lower_of = []([[maybe_unused]] edge const &e) {
			if constexpr (bounded_capacitated_edge<edge>)
				return F(lower(e));
			else
				return F(0);
		};

		std::vector<F> supply = std::move(b);
		F total = 0, supplied = 0;
		for (auto const s : supply) {
			detail::add_flow(total, s);
			if (s > 0) detail::add_flow(supplied, s);
		}
		if (total != 0) return std::nullopt;
		std::vector<flow_arc<F, C>> arcs;
		if constexpr (std::ranges::sized_range<EdgeList const>) arcs.reserve(std::ranges::size(edges));
		for (auto const &e : edges) {
			F const l = lower_of(e);
			atl_assert(graph::source(e) < n && graph::target(e) < n && l >= 0 && capacity(e) >= l);
			arcs.push_back({vertex_id(graph::source(e)), vertex_id(graph::target(e)),
							F(F(capacity(e)) - l), C(cost(e))});
			detail::add_flow(supply[graph::source(e)], F(-l));
			detail::add_flow(supply[graph::target(e)], l);
		}
		auto x = detail::network_simplex(n, arcs, supply);
		if (!x) return std::nullopt;

		min_cost_flow_result<F, C> r;
		r.edge_flow = std::move(*x);
		for (std::size_t i = 0; auto const &e : edges) {
			r.edge_flow[i] += lower_of(e);
			detail::add_flow(r.cost, detail::cost_product(C(r.edge_flow[i]), C(cost(e))));
			++i;
		}
		r.value = supplied;
		return r;
	}

	/**
	 * @brief The dual certificate of a min-cost flow: potentials `p` with
	 *        `cost(a) + p[tail(a)] - p[head(a)] >= 0` on every residual arc
	 *        of @p edge_flow, which holds iff @p edge_flow is cost-optimal
	 *        among the flows with the same supplies (for `min_cost_max_flow`:
	 *        the same value, as the solver's t->s arc is not part of the
	 *        residual).
	 *
	 * Shortest distances from a virtual all-vertex source (Bellman-Ford,
	 * queue-based). Edge types with lower bounds are honored: the reverse arc
	 * of an edge carries `flow - lower`, the forward arc `capacity - flow`.
	 * Opt-in because it costs about a third of a solve.
	 *
	 * @pre `lower(e) <= edge_flow[i] <= capacity(e)` and `edge_flow.size()` equals
	 * the edge count (asserted), and the residual has no negative cycle, i.e.
	 * the flow is cost-optimal for its supplies (asserted by a pop-count bound;
	 * otherwise the distances are unbounded below); the cost type must hold
	 * (n+1) · max|cost|, including relaxation candidates; checking a
	 * reduced cost additionally needs (2n+1) · max|cost|
	 * @complexity O(n m + n + m) worst-case time and O(n+m) heap storage.
	 * @tparam Flow Signed arithmetic type, or void for the capacity type.
	 * @tparam Cost Signed arithmetic type, or void for the edge cost type.
	 * @tparam EdgeList Stable const forward range of costed capacitated edges.
	 * @param n Vertex count, below none-1; none is never a vertex.
	 * @param edges Edges in output order; storage is not modified or retained.
	 * @pre Conversions to Flow and Cost are exact and capacities are nonnegative.
	 * Lower bounds may be negative; both residual differences must fit Flow.
	 * At most UINT32_MAX/2 input edges; all vector lengths fit max_size.
	 * Floating intermediates are finite and exact; no tolerance is applied.
	 * @note Not constexpr (shared residual construction is runtime-only).
	 * All traversal uses heap storage with O(1) call stack; allocation failures propagate.
	 * @param edge_flow Input-order feasible flow to certify.
	 * @return Owned potentials; no solver tree potentials are exposed.
	 * @pre At most UINT32_MAX/2 edges; all residual differences fit Flow.
	 */
	template<typename Flow = void, typename Cost = void, cost_capacitated_edge_list EdgeList>
		requires detail::cost_network<detail::flow_t<Flow, EdgeList>, detail::cost_type<Cost, EdgeList>, EdgeList>
	[[nodiscard]] std::vector<detail::cost_type<Cost, EdgeList>>
	min_cost_flow_potentials(std::size_t n, EdgeList const &edges,
							 std::vector<detail::flow_t<Flow, EdgeList>> const &edge_flow) {
		using F = detail::flow_t<Flow, EdgeList>;
		using C = detail::cost_type<Cost, EdgeList>;
		using edge = std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>;
		atl_assert(n < none - 1);
		detail::cost_residual<F, C> net(n, edges);
		atl_assert(edge_flow.size() == net.input_arc.size());
		for (std::size_t i = 0; auto const &e : edges) {
			F const l = [&] {
				if constexpr (bounded_capacitated_edge<edge>)
					return F(lower(e));
				else
					return F(0);
			}();
			atl_assert(l <= edge_flow[i] && edge_flow[i] <= F(graph::capacity(e)));
			uint32 const a = net.input_arc[i];
			net.capacity[a] = detail::flow_difference(F(graph::capacity(e)), edge_flow[i]);
			net.capacity[net.mate[a]] = detail::flow_difference(edge_flow[i], l);
			++i;
		}
		return detail::dual_potentials(net);
	}
}
