#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/detail/edge_count.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
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
	 * @brief An owned flow witness and cut certificate, indexed by input edge and vertex.
	 * @tparam Flow Ordered additive value type used for capacities and accumulated flow.
	 */
	template<flow_value Flow>
	struct max_flow_result {
		Flow value{};					///< Net source-to-sink flow; bounded flows may be negative.
		std::vector<Flow> edge_flow;	///< Flow on each edge in input iteration order.
		std::vector<uint8> source_side; ///< One for vertices in the certificate's source set, zero otherwise.
	};

	namespace detail {
		template<typename Flow, typename EdgeList>
		concept flow_network = capacitated_edge_list<EdgeList> && flow_value<Flow> && std::semiregular<Flow>
							&& std::convertible_to<int, Flow> && (!std::same_as<Flow, bool>)
							&& requires(std::ranges::range_reference_t<EdgeList const> e) {
								   Flow(graph::capacity(e));
								   bool(graph::capacity(e) >= 0);
							   };

		template<typename Flow, typename EdgeList>
		concept bounded_flow_network = flow_network<Flow, EdgeList> && signed_flow_value<Flow>
									&& ((!bounded_capacitated_edge<std::ranges::range_value_t<EdgeList>>
										 && !bounded_capacitated_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>)
										|| requires(std::ranges::range_reference_t<EdgeList const> e) { Flow(graph::lower(e)); });

		template<signed_flow_value Flow>
		constexpr void add_flow(Flow &value, Flow amount) {
			atl_assert(amount >= 0 ? value <= (std::numeric_limits<Flow>::max)() - amount
								   : value >= std::numeric_limits<Flow>::lowest() - amount);
			value += amount;
		}

		/**
		 * @brief Residual network in scan-friendly CSR order: `head`/`capacity`
		 *        of vertex v's arcs are contiguous in `[offset[v], offset[v + 1])`
		 *        — the hot loops of every algorithm read them sequentially,
		 *        with `mate[a]` (the paired reverse arc) the only scattered
		 *        access: on pushes, and in the global-relabel BFS, which
		 *        tests the reverse arc's capacity. The reverse arc's residual
		 *        *is* the flow pushed through an edge; `input_arc[i]` locates
		 *        input edge i for readout.
		 */
		template<flow_value Flow>
		struct residual {
			std::vector<uint32> offset, head, mate, input_arc;
			std::vector<Flow> capacity;

			/**
			 * @brief The residual network of @p edges on @p n vertices: one
			 *        forward arc at full capacity and one empty reverse arc per
			 *        input edge — the shared entry chokepoint of every flow run.
			 * @pre `n < 2^31` (the gap lists index `n + height`), at most
			 *      `2^32 - 1` arcs, endpoints in `[0, n)` and capacities
			 *      nonnegative (all asserted).
			 */
			template<capacitated_edge_list EdgeList>
				requires flow_network<Flow, EdgeList>
			constexpr residual(std::size_t n, EdgeList const &edges) {
				auto const m = detail::checked_edge_count(edges, (std::numeric_limits<uint32>::max)() / 2);
				atl_assert(n < (std::size_t(1) << 31) && m <= (std::numeric_limits<uint32>::max)() / 2);
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
				mate.resize(2 * m);
				input_arc.resize(m);
				std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
				std::size_t i = 0;
				for (auto const &e : edges) {
					uint32 const a = cursor[graph::source(e)]++;
					uint32 const b = cursor[graph::target(e)]++;
					head[a] = graph::target(e);
					capacity[a] = Flow(graph::capacity(e));
					head[b] = graph::source(e);
					capacity[b] = 0;
					mate[a] = b;
					mate[b] = a;
					input_arc[i++] = a;
				}
			}

			// (Sorting each vertex's arcs by target — a locality trick some
			// tuned HLPP implementations pay an O(m log) build tax for — was
			// measured a net loss here: the sort dominates on fast instances
			// and bought no scan speedup on the hard shapes.)
		};

		/**
		 * @brief Residual reachability from @p source — the source side of a
		 *        minimum cut once no augmenting path is left.
		 */
		template<flow_value Flow>
		[[nodiscard]] constexpr std::vector<uint8> source_side(residual<Flow> const &net, vertex_id source) {
			std::vector<uint8> seen(net.offset.size() - 1, 0);
			std::vector<uint32> queue{source};
			seen[source] = 1;
			for (std::size_t at = 0; at < queue.size(); ++at) {
				uint32 const v = queue[at];
				for (uint32 a = net.offset[v]; a < net.offset[v + 1]; ++a) {
					if (net.capacity[a] > 0 && !seen[net.head[a]]) {
						seen[net.head[a]] = 1;
						queue.push_back(net.head[a]);
					}
				}
			}
			return seen;
		}

		template<flow_value Flow>
		[[nodiscard]] constexpr std::vector<Flow> edge_flows(residual<Flow> const &net) {
			std::vector<Flow> flow(net.input_arc.size());
			for (std::size_t i = 0; i < flow.size(); ++i) flow[i] = net.capacity[net.mate[net.input_arc[i]]];
			return flow;
		}

		template<typename Flow, typename EdgeList>
		using flow_t = std::conditional_t<std::is_void_v<Flow>, capacity_t<EdgeList>, Flow>;
	}

	namespace detail {
		/**
		 * @brief The HLPP engine both public entries share: push all excess
		 *        downhill toward @p root — heights bound residual distances from below; BFS rebuilds exact distances
		 *        every n local relabels, the highest active
		 *        vertex discharges first through one circular arc pass that
		 *        pushes and collects the relabel target simultaneously, and a
		 *        vertex lifted to height n is dead (it cannot reach the root;
		 *        its excess stays put). Excluded vertices never activate.
		 *
		 * Phase 1 roots at the sink: on termination the preflow delivers the
		 * maximum value. Phase 2 roots at the source: every vertex still
		 * holding excess can reach it through the reverse arcs of the flow
		 * that filled it, so no live vertex dies and the drain-back leaves a
		 * valid maximum flow.
		 * @param net Residual network, updated in place.
		 * @param excess Nonnegative vertex excesses, updated in place.
		 * @param root Vertex toward which excess is drained.
		 * @param still_a First vertex that never activates.
		 * @param still_b Second vertex that never activates; excess reaching either stays there.
		 */
		template<flow_value F>
		constexpr void push_relabel_drain(residual<F> &net, std::vector<F> &excess, vertex_id root, vertex_id still_a,
										  vertex_id still_b) {
			auto const n = uint32(net.offset.size() - 1);
			std::vector<uint32> height(n, n), it(net.offset.begin(), net.offset.end() - 1);
			std::vector<uint32> bucket_head(n, none), bucket_next(n);
			std::vector<uint32> gap_next(std::size_t(2) * n), gap_prev(std::size_t(2) * n);
			uint32 hi = 0, gap_highest = 0;

			auto const enqueue = [&](uint32 v) {
				bucket_next[v] = bucket_head[height[v]];
				bucket_head[height[v]] = v;
				hi = (std::max)(hi, height[v]);
			};
			auto const gap_insert = [&](uint32 v, uint32 h) {
				gap_next[v] = gap_next[n + h];
				gap_prev[v] = n + h;
				gap_next[n + h] = v;
				gap_prev[gap_next[v]] = v;
				gap_highest = (std::max)(gap_highest, h);
			};
			auto const gap_erase = [&](uint32 v) {
				gap_next[gap_prev[v]] = gap_next[v];
				gap_prev[gap_next[v]] = gap_prev[v];
			};
			auto const push = [&](uint32 a, F amount) {
				uint32 const w = net.head[a];
				if (amount > 0 && excess[w] == 0 && w != still_a && w != still_b && height[w] < n)
					enqueue(w);
				net.capacity[a] -= amount;
				net.capacity[net.mate[a]] += amount;
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
						if (net.capacity[net.mate[a]] > 0 && height[x] == n && x != root) {
							height[x] = height[w] + 1;
							queue.push_back(x);
						}
					}
				}
				std::ranges::fill(bucket_head, none);
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
							push(a, (std::min)(excess[u], net.capacity[a]));
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
				while (hi > 0 && bucket_head[hi] == none) --hi;
				uint32 const u = bucket_head[hi];
				if (u == none) break; // no live active vertex left
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
	}

	/**
	 * @brief Maximum source-to-sink flow with per-edge flows and a minimum cut.
	 * @tparam Flow Computation type; void selects the edge capacity type. It must
	 * model flow_value, be semiregular, accept implicit integer zero, and be constructible from scanned capacities.
	 * bool is excluded because packed vector storage does not provide Flow references.
	 * @tparam EdgeList Const forward range of capacitated edges without lower bounds.
	 * @param n Number of vertices; valid ids are [0, n), excluding none.
	 * @param edges Directed edges; parallel edges and self-loops are supported.
	 * @param source Source vertex.
	 * @param sink Distinct sink vertex.
	 * @pre source, sink < n < 2^31; at most UINT32_MAX/2 input edges;
	 * endpoints < n and capacities nonnegative (asserted).
	 * @pre Capacities convert exactly to Flow. Their sum over non-loop edges
	 * leaving source fits `numeric_limits<Flow>::max()` (sum asserted).
	 * Flow arithmetic and numeric_limits must model bounded real arithmetic.
	 * Floating values and intermediate results must be finite; exact results
	 * require every intermediate addition/subtraction to be representable exactly.
	 * Dyadic inputs alone do not guarantee this at arbitrarily large magnitudes.
	 * @return Owned result in input iteration order. edge_flow is feasible;
	 * source_side is residual reachability from source, certifying a cut of value.
	 * Self-loops carry zero. The input is neither modified nor retained.
	 * @complexity O(n^2 sqrt(m) + n + m) time, O(n + m) heap storage and O(1) call stack.
	 * Uses two highest-label push-relabel phases with gaps and global relabeling;
	 * no recursion per vertex, including on million-vertex paths. Allocation failures propagate.
	 */
	template<typename Flow = void, capacitated_edge_list EdgeList>
		requires(detail::flow_network<detail::flow_t<Flow, EdgeList>, EdgeList>
				 && !bounded_capacitated_edge<std::ranges::range_value_t<EdgeList>>
				 && !bounded_capacitated_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>)
	[[nodiscard]] constexpr auto max_flow(std::size_t n, EdgeList const &edges, vertex_id source, vertex_id sink) {
		using F = detail::flow_t<Flow, EdgeList>;
		atl_assert(source < n && sink < n && source != sink);
		detail::residual<F> net(n, edges);
		max_flow_result<F> r;

		std::vector<F> excess(n, 0);
		// the preflow saturates every source arc at once, and that much
		// excess can then gather at a single vertex
		[[maybe_unused]] F saturated = 0;
		for (uint32 a = net.offset[source]; a < net.offset[source + 1]; ++a) {
			uint32 const w = net.head[a];
			if (w == source || net.capacity[a] <= 0) continue;
			atl_assert(net.capacity[a] <= (std::numeric_limits<F>::max)() - saturated
					   && ((saturated += net.capacity[a]), true));
			excess[w] += net.capacity[a];
			net.capacity[net.mate[a]] += net.capacity[a];
			net.capacity[a] = 0;
		}
		detail::push_relabel_drain(net, excess, sink, source, sink);
		detail::push_relabel_drain(net, excess, source, source, sink);

		r.value = excess[sink];
		r.source_side = detail::source_side(net, source);
		r.edge_flow = detail::edge_flows(net);
		return r;
	}

	/**
	 * @brief Maximum flow value using only the sink-directed preflow phase.
	 * @tparam Flow Computation type, void for the capacity type; requirements as in atl::graph::max_flow.
	 * @tparam EdgeList Const forward range of capacitated edges without lower bounds.
	 * @param n Number of vertices.
	 * @param edges Directed capacity edges in stable iteration order.
	 * @param source Source vertex.
	 * @param sink Distinct sink vertex.
	 * @pre All size, endpoint, conversion and arithmetic bounds of atl::graph::max_flow apply.
	 * @return Maximum value, with no edge witness or cut certificate.
	 * @complexity O(n^2 sqrt(m) + n + m) time, O(n + m) heap storage and O(1) call stack.
	 */
	template<typename Flow = void, capacitated_edge_list EdgeList>
		requires(detail::flow_network<detail::flow_t<Flow, EdgeList>, EdgeList>
				 && !bounded_capacitated_edge<std::ranges::range_value_t<EdgeList>>
				 && !bounded_capacitated_edge<std::remove_cvref_t<std::ranges::range_reference_t<EdgeList const>>>)
	[[nodiscard]] constexpr auto max_flow_value(std::size_t n, EdgeList const &edges, vertex_id source, vertex_id sink) {
		using F = detail::flow_t<Flow, EdgeList>;
		atl_assert(source < n && sink < n && source != sink);
		detail::residual<F> net(n, edges);

		std::vector<F> excess(n, 0);
		// the preflow saturates every source arc at once, and that much
		// excess can then gather at a single vertex
		[[maybe_unused]] F saturated = 0;
		for (uint32 a = net.offset[source]; a < net.offset[source + 1]; ++a) {
			uint32 const w = net.head[a];
			if (w == source || net.capacity[a] <= 0) continue;
			atl_assert(net.capacity[a] <= (std::numeric_limits<F>::max)() - saturated
					   && ((saturated += net.capacity[a]), true));
			excess[w] += net.capacity[a];
			net.capacity[net.mate[a]] += net.capacity[a];
			net.capacity[a] = 0;
		}
		detail::push_relabel_drain(net, excess, sink, source, sink);
		return excess[sink];
	}

	/**
	 * @brief Find flows within edge bounds whose outgoing minus incoming flow is b[v].
	 * @tparam Flow Signed computation type, void for the capacity type; semiregular,
	 * constructible from capacities and any lower bounds, accepting implicit integer zero
	 * and modeling signed_flow_value.
	 * @tparam EdgeList Const forward range of capacity edges, optionally with lower bounds.
	 * @param n Number of vertices.
	 * @param edges Directed edges; absent lower bounds mean zero.
	 * @param b Supplies (positive) and demands (negative), taken by value as working storage.
	 * @pre b.size() == n, n < 2^31 - 2, endpoints < n, 0 <= lower <= capacity (asserted).
	 * The augmented edge count is at most UINT32_MAX/2 (asserted by the residual builder).
	 * @pre Capacity/lower conversions are exact. Every partial supply sum, each
	 * supply adjusted by lower bounds, each negated demand and total positive
	 * adjusted supply fits Flow (asserted). Floating arithmetic obeys atl::graph::max_flow's bounds.
	 * @return Per-input-edge feasible flows, or nullopt for unequal total supply/demand
	 * or infeasible bounds. Inputs are not retained; allocation failures propagate.
	 * @complexity O((n+2)^2 sqrt(m+n) + n + m) time, O(n+m) heap storage, O(1) call stack.
	 */
	template<typename Flow = void, capacitated_edge_list EdgeList>
		requires detail::bounded_flow_network<detail::flow_t<Flow, EdgeList>, EdgeList>
	[[nodiscard]] constexpr auto feasible_flow(std::size_t n, EdgeList const &edges, std::vector<detail::flow_t<Flow, EdgeList>> b)
		-> std::optional<std::vector<detail::flow_t<Flow, EdgeList>>> {
		using F = detail::flow_t<Flow, EdgeList>;
		atl_assert(b.size() == n);
		atl_assert(n < (std::size_t(1) << 31) - 2);
		auto const lower_of = []<capacitated_edge Edge>(Edge const &e)
			requires(!bounded_capacitated_edge<Edge> || requires { F(graph::lower(e)); })
		{
			if constexpr (bounded_capacitated_edge<Edge>)
				return F(lower(e));
			else
				return F(0);
		};

		std::vector<F> supply = std::move(b);
		F total = 0;
		for (auto const s : supply) detail::add_flow(total, s);
		if (total != 0) return std::nullopt;
		std::vector<flow_arc<F>> arcs;
		if constexpr (std::ranges::sized_range<EdgeList const>) {
			atl_assert(n <= arcs.max_size() && std::ranges::size(edges) <= arcs.max_size() - n);
			arcs.reserve(std::ranges::size(edges) + n);
		}
		for (auto const &e : edges) {
			F const l = lower_of(e);
			atl_assert(graph::source(e) < n && graph::target(e) < n);
			F const c = F(capacity(e));
			atl_assert(l >= 0 && c >= l);
			arcs.push_back({vertex_id(graph::source(e)), vertex_id(graph::target(e)),
							F(c - l)});
			detail::add_flow(supply[graph::source(e)], F(-l));
			detail::add_flow(supply[graph::target(e)], l);
		}
		std::size_t const m = arcs.size();
		uint32 const super_s = uint32(n), super_t = uint32(n) + 1;
		F need = 0;
		for (uint32 v = 0; v < n; ++v) {
			if (supply[v] > 0) {
				arcs.push_back({super_s, v, supply[v]});
				detail::add_flow(need, supply[v]);
			} else if (supply[v] < 0) {
				atl_assert(supply[v] >= -(std::numeric_limits<F>::max)());
				arcs.push_back({v, super_t, F(-supply[v])});
			}
		}

		auto feasible = max_flow(n + 2, arcs, super_s, super_t);
		if (feasible.value != need) return std::nullopt;
		feasible.edge_flow.resize(m);
		std::size_t i = 0;
		for (auto const &e : edges) feasible.edge_flow[i++] += lower_of(e);
		return std::move(feasible.edge_flow);
	}

	namespace detail {
		/**
		 * @brief The two bounded s-t entries differ only in the direction of
		 *        the second HLPP run: close the network with value-shortcut
		 *        arcs in *both* directions (so feasibility is the true b-flow
		 *        condition, not the folklore value >= 0 one), take any feasible
		 *        flow, then push @p is_forward (maximize) or backward
		 *        (minimize) in the shortcut-free residual. The final
		 *        residual's reachability cut is the Gale-Hoffman certificate
		 *        either way.
		 */
		template<signed_flow_value F, capacitated_edge_list EdgeList>
			requires bounded_flow_network<F, EdgeList>
		[[nodiscard]] constexpr std::optional<max_flow_result<F>> bounded_extremal_flow(
			std::size_t n, EdgeList const &edges, vertex_id source, vertex_id sink, bool is_forward) {
			atl_assert(source < n && sink < n && source != sink && n < (std::size_t(1) << 31) - 2);
			std::vector<bounded_flow_arc<F>> closed;
			if constexpr (std::ranges::sized_range<EdgeList const>) {
				atl_assert(std::ranges::size(edges) <= closed.max_size() - 2);
				closed.reserve(std::ranges::size(edges) + 2);
			}
			F wide = 1;
			for (auto const &e : edges) {
				F l = 0;
				if constexpr (bounded_capacitated_edge<std::remove_cvref_t<decltype(e)>>)
					l = F(lower(e));
				closed.push_back({vertex_id(graph::source(e)), vertex_id(graph::target(e)), l,
								  F(capacity(e))});
				atl_assert(graph::source(e) < n && graph::target(e) < n && l >= 0 && F(capacity(e)) >= l);
				add_flow(wide, F(capacity(e)));
			}
			std::size_t const m = closed.size();
			closed.push_back({sink, source, 0, wide});
			closed.push_back({source, sink, 0, wide});
			auto const base = feasible_flow(n, closed, std::vector<F>(n, 0));
			if (!base) return std::nullopt;

			std::vector<flow_arc<F>> residual;
			atl_assert(m <= residual.max_size() / 2);
			residual.reserve(2 * m);
			for (std::size_t i = 0; i < m; ++i) {
				residual.push_back({closed[i].from, closed[i].to,
									F(closed[i].capacity - (*base)[i])});
				residual.push_back({closed[i].to, closed[i].from, F((*base)[i] - closed[i].lower)});
			}
			auto more = is_forward ? max_flow(n, residual, source, sink)
								   : max_flow(n, residual, sink, source);

			max_flow_result<F> r;
			F const shortcut = F((*base)[m] - (*base)[m + 1]);
			r.value = F(is_forward ? shortcut + more.value : shortcut - more.value);
			r.edge_flow.resize(m);
			for (std::size_t i = 0; i < m; ++i)
				r.edge_flow[i] = F((*base)[i] + more.edge_flow[2 * i] - more.edge_flow[2 * i + 1]);
			r.source_side = std::move(more.source_side);
			if (!is_forward) // reorient: the certificate cut contains the source
				for (auto &side : r.source_side) side = uint8(1 - side);
			return r;
		}
	}

	/**
	 * @brief Maximum bounded source-to-sink flow, allowing a negative optimal value.
	 * @tparam Flow Signed computation type, void for the capacity type; as in atl::graph::feasible_flow.
	 * @tparam EdgeList Const forward capacity range with optional lower bounds.
	 * @param n Number of vertices.
	 * @param edges Directed edges, in output order.
	 * @param source Source vertex.
	 * @param sink Distinct sink vertex.
	 * @pre source, sink < n < 2^31 - 2, source != sink; endpoints < n;
	 * 0 <= lower <= capacity; 1 + sum of all capacities fits Flow (asserted).
	 * @pre Conversions and floating arithmetic obey atl::graph::max_flow's bounds.
	 * The first reduction has m+2+n edges at most and the second has 2m;
	 * each actual augmented count must be at most UINT32_MAX/2 (asserted).
	 * @return nullopt when infeasible, otherwise an owned flow witness and cut S
	 * with value = sum upper(S->T) - sum lower(T->S). No input is retained.
	 * Allocation failures propagate.
	 * @complexity O((n+2)^2 sqrt(m+n) + n+m) time, O(n+m) heap storage, O(1) call stack.
	 */
	template<typename Flow = void, capacitated_edge_list EdgeList>
		requires detail::bounded_flow_network<detail::flow_t<Flow, EdgeList>, EdgeList>
	[[nodiscard]] constexpr auto bounded_max_flow(std::size_t n, EdgeList const &edges, vertex_id source, vertex_id sink) {
		return detail::bounded_extremal_flow<detail::flow_t<Flow, EdgeList>>(n, edges, source, sink,
																			 true);
	}

	/**
	 * @brief Minimum bounded source-to-sink flow, allowing negative values.
	 * @tparam Flow Signed computation type, void for the capacity type.
	 * @tparam EdgeList Const forward capacity range with optional lower bounds.
	 * @param n Number of vertices.
	 * @param edges Directed edges, in output order.
	 * @param source Source vertex.
	 * @param sink Distinct sink vertex.
	 * @pre All size, endpoint and arithmetic bounds of atl::graph::bounded_max_flow apply.
	 * @return nullopt when infeasible, otherwise an owned witness and source set S
	 * with value = sum lower(S->T) - sum upper(T->S). Allocation failures propagate.
	 * @complexity O((n+2)^2 sqrt(m+n) + n+m) time, O(n+m) heap storage, O(1) call stack.
	 */
	template<typename Flow = void, capacitated_edge_list EdgeList>
		requires detail::bounded_flow_network<detail::flow_t<Flow, EdgeList>, EdgeList>
	[[nodiscard]] constexpr auto bounded_min_flow(std::size_t n, EdgeList const &edges, vertex_id source, vertex_id sink) {
		return detail::bounded_extremal_flow<detail::flow_t<Flow, EdgeList>>(n, edges, source, sink,
																			 false);
	}
}
