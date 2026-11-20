#pragma once
#include <atl/graph/csr.hpp>
#include <atl/graph/functional_graph.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

/** @brief Experimental algorithms and data structures. */
namespace atl::experimental {
	/** @brief Iterative marking/path-stack decomposition; experimental contender; not measured yet.
	 * Same canonical output and asserted n/id preconditions as graph::functional_graph_decomposition_of.
	 * @param successors Sized input range with one outgoing target per vertex.
	 * @return Owned canonical metadata, in the same order as the public decomposition.
	 * @complexity O(n) time and space, without recursion.
	 */
	template<graph::successor_range Successors>
	[[nodiscard]] constexpr graph::functional_graph_decomposition functional_graph_decomposition_by_marking(Successors &&successors) {
		auto const owned = graph::detail::functional_graph_input(std::forward<Successors>(successors));
		auto const n = owned.size();
		std::vector<uint32> state(n, 0), is_cycle(n, 0);
		std::vector<graph::vertex_id> path, order;
		path.reserve(n);
		order.reserve(n);
		for (graph::vertex_id start = 0; start < n; ++start) {
			if (state[start]) continue;
			graph::vertex_id v = start;
			while (!state[v]) {
				state[v] = 1;
				path.push_back(v);
				v = owned[v];
			}
			if (state[v] == 1) {
				graph::vertex_id cycle = v;
				do {
					is_cycle[cycle] = 1;
					cycle = owned[cycle];
				} while (cycle != v);
			}
			while (!path.empty()) {
				graph::vertex_id const tail = path.back();
				path.pop_back();
				state[tail] = 2;
				if (!is_cycle[tail]) order.push_back(tail);
			}
		}
		return graph::detail::functional_graph_metadata(owned, is_cycle, order);
	}

	namespace detail {
		/** @brief Full level-major doubling table to the fixed 10^18 bound.
		 * Validates and owns input, using O(nL) build/storage, L=bit_width(10^18).
		 */
		struct functional_graph_lifting {
			// jumps is declared first: it takes the validated successors as its first row, and n and the
			// decomposition read that row. MSVC's constant evaluation rejects reading a by-value vector
			// parameter of a delegated-to constructor here ("unevaluable pointer value").
			std::vector<graph::vertex_id> jumps;				 ///< Level-major doubling endpoints.
			std::size_t n;										 ///< Vertex count.
			graph::functional_graph_decomposition decomposition; ///< Canonical metadata.
			/** @brief Number of retained doubling rows. */
			static constexpr std::size_t levels = std::bit_width(graph::maximum_functional_graph_steps);

			/** @brief Build all endpoint rows, asserting the table product before allocation. */
			template<graph::successor_range Successors>
			constexpr explicit functional_graph_lifting(Successors &&successors)
				: jumps(graph::detail::functional_graph_input(std::forward<Successors>(successors))), n(jumps.size()),
				  decomposition(graph::detail::functional_graph_peeling(jumps)) {
				atl_assert(n <= jumps.max_size() / levels);
				// filled by index, one element read per statement (MSVC constant evaluation)
				jumps.resize(n * levels, 0);
				for (std::size_t level = 1; level < levels; ++level)
					for (std::size_t v = 0; v < n; ++v) {
						std::size_t const half = jumps[(level - 1) * n + v];
						jumps[level * n + v] = jumps[(level - 1) * n + half];
					}
			}

			/** @brief Endpoint using only jump rows; asserted start/step bounds. O(L) time. */
			[[nodiscard]] constexpr graph::vertex_id successor(graph::vertex_id start, uint64 steps) const {
				graph::detail::functional_graph_check(n, start, steps);
				for (std::size_t level = 0; steps; ++level, steps >>= 1)
					if (steps & 1) start = jumps[level * n + start];
				return start;
			}
		};

		/** @brief Own validated single-pass queries and size_t links bucketed by start.
		 * size_t linkage preserves the unrestricted query-count contract (bucket_by_key stores uint32).
		 * @complexity O(n+q) time/space. Starts/steps asserted before use.
		 */
		struct functional_graph_batch {
			/** @brief Sentinel for the end of a query chain. */
			static constexpr std::size_t end = (std::numeric_limits<std::size_t>::max)();
			std::vector<graph::functional_graph_successor_query> queries; ///< Input order, including duplicates.
			std::vector<std::size_t> head, next;						  ///< Per-start linked lists of original query indices.

			/** @brief Read queries exactly once and retain their original indices. */
			template<graph::functional_graph_query_range Queries>
			constexpr functional_graph_batch(Queries &&input, std::size_t n) {
				auto const raw_size = std::ranges::size(input);
				atl_assert(raw_size <= (std::numeric_limits<std::size_t>::max)());
				auto const q = std::size_t(raw_size);
				queries.reserve(q);
				next.reserve(q);
				head.assign(n, end);
				for (graph::functional_graph_successor_query const query : input) {
					graph::detail::functional_graph_check(n, query.start, query.steps);
					atl_assert(queries.size() < q);
					next.push_back(head[query.start]);
					head[query.start] = queries.size();
					queries.push_back(query);
				}
				atl_assert(queries.size() == q);
			}
		};

		/** @brief Callback on entering a reverse-tree vertex with its root-to-vertex active stack. */
		template<typename Enter>
		concept functional_graph_dfs_event = std::invocable<Enter &, std::vector<graph::vertex_id> const &>;

		/** @brief Iterative reverse-tree DFS, excluding all outgoing cycle edges.
		 * @pre Validated successors and matching decomposition. @complexity O(n) time/space plus callbacks.
		 */
		template<functional_graph_dfs_event Enter>
		constexpr void functional_graph_reverse_dfs(std::vector<graph::vertex_id> const &successors,
													graph::functional_graph_decomposition const &decomposition, Enter enter) {
			auto const n = successors.size();
			std::vector<graph::arc<>> arcs;
			arcs.reserve(n);
			for (graph::vertex_id v = 0; v < n; ++v)
				if (decomposition.depth_to_cycle[v]) arcs.push_back({successors[v], v});
			graph::csr<graph::vertex_id> const children(n, arcs);
			std::vector<graph::vertex_id> active;
			std::vector<std::size_t> cursor;
			active.reserve(n);
			cursor.reserve(n);
			for (graph::vertex_id root : decomposition.cycles.members) {
				active.push_back(root);
				cursor.push_back(0);
				std::invoke(enter, std::as_const(active));
				while (!active.empty()) {
					auto const edges = children.out_edges(active.back());
					if (cursor.back() == edges.size()) {
						active.pop_back();
						cursor.pop_back();
					} else {
						graph::vertex_id const child = edges[cursor.back()++];
						active.push_back(child);
						cursor.push_back(0);
						std::invoke(enter, std::as_const(active));
					}
				}
			}
		}
	}

	/** @brief Endpoint-only full binary lifting; experimental contender; not measured yet.
	 * Same owning/const semantics and asserted n/id/query bounds as graph::functional_graph_successor_index.
	 * @complexity O(nL) build/space and O(L) query, L=bit_width(10^18).
	 */
	class functional_graph_successor_index_by_lifting {
		detail::functional_graph_lifting engine_;

	public:
		/** @brief Consume successors once; assert graph bounds at input and the product prior to table allocation. */
		template<graph::successor_range Successors>
		constexpr explicit functional_graph_successor_index_by_lifting(Successors &&successors) : engine_(std::forward<Successors>(successors)) {}

		/** @brief Number of vertices. */
		constexpr std::size_t size() const noexcept { return engine_.n; }

		/** @brief Canonical metadata; valid until destruction, move or assignment. */
		[[nodiscard]] constexpr graph::functional_graph_decomposition const &decomposition() const noexcept { return engine_.decomposition; }

		/** @brief Endpoint; asserted start < size() and steps <= 10^18. O(L). */
		[[nodiscard]] constexpr graph::vertex_id successor(graph::vertex_id start, uint64 steps) const { return engine_.successor(start, steps); }
	};

	/** @brief Full binary lifting with ordered fold rows; experimental contender; not measured yet.
	 * Same semantics and asserted preconditions as graph::functional_graph_walk_index, including owning-range forwarding.
	 * @tparam Value Copyable outgoing-edge aggregate.
	 * @tparam Operation Copyable const-invocable ordered binary operation.
	 * @pre Associativity, two-sided identity, purity and representable intermediate blocks (documented).
	 * @complexity O(nL) build/space; O(L) walk or value-independent successor, L=bit_width(10^18).
	 */
	template<std::copyable Value, binary_operation<Value> Operation>
		requires std::copyable<Operation>
	class functional_graph_walk_index_by_lifting {
		detail::functional_graph_lifting endpoints_;
		Value identity_;
		Operation operation_;
		std::vector<Value> folds_;

	public:
		/** @brief Consume inputs once; assert graph bounds, matching value count and the product prior to table allocation. */
		template<graph::successor_range Successors, typename Values>
			requires graph::functional_graph_value_range<Values, Value>
		constexpr functional_graph_walk_index_by_lifting(Successors &&successors, Values &&values, Value identity, Operation operation)
			: endpoints_(std::forward<Successors>(successors)), identity_(std::move(identity)), operation_(std::move(operation)) {
			folds_ = graph::detail::functional_graph_values<Value>(std::forward<Values>(values), size());
			atl_assert(size() <= folds_.max_size() / detail::functional_graph_lifting::levels);
			// sized up front and filled by index, as in functional_graph_lifting (MSVC constant evaluation)
			folds_.resize(size() * detail::functional_graph_lifting::levels, identity_);
			for (std::size_t level = 1; level < detail::functional_graph_lifting::levels; ++level)
				for (std::size_t v = 0; v < size(); ++v) {
					auto const row_offset = (level - 1) * size();
					std::size_t const half = endpoints_.jumps[row_offset + v];
					folds_[level * size() + v] = std::invoke(std::as_const(operation_), folds_[row_offset + v], folds_[row_offset + half]);
				}
		}

		/** @brief Number of vertices. */
		constexpr std::size_t size() const noexcept { return endpoints_.n; }

		/** @brief Canonical metadata; reference lasts until destruction, move or assignment. */
		[[nodiscard]] constexpr graph::functional_graph_decomposition const &decomposition() const noexcept { return endpoints_.decomposition; }

		/** @brief Value-independent endpoint; asserted start/step bounds. O(L). */
		[[nodiscard]] constexpr graph::vertex_id successor(graph::vertex_id start, uint64 steps) const { return endpoints_.successor(start, steps); }

		/** @brief Ordered k-edge fold and endpoint; asserted start/step bounds. O(L). */
		[[nodiscard]] constexpr graph::functional_graph_walk_result<Value> walk(graph::vertex_id start, uint64 steps) const {
			graph::detail::functional_graph_check(size(), start, steps);
			Value fold = identity_;
			for (std::size_t level = 0; steps; ++level, steps >>= 1)
				if (steps & 1) {
					fold = std::invoke(std::as_const(operation_), fold, folds_[level * size() + start]);
					start = endpoints_.jumps[level * size() + start];
				}
			return {start, std::move(fold)};
		}
	};

	/** @brief Heavy-chain upward-prefix folds on the cut lca forest; experimental contender; not measured yet.
	 * Same semantics, ownership and asserted bounds as graph::functional_graph_walk_index.
	 * @tparam Value Copyable outgoing-edge aggregate.
	 * @tparam Operation Copyable const-invocable ordered binary operation.
	 * @pre Associativity, two-sided identity, purity and intermediate-fold representability (documented).
	 * @complexity O(n) build/space, O(log n + log(steps+1)) walk, O(log n) value-independent successor.
	 */
	template<std::copyable Value, binary_operation<Value> Operation>
		requires std::copyable<Operation>
	class functional_graph_walk_index_by_chain_prefix {
		graph::detail::functional_graph_walk_engine<Value, Operation, true> engine_;

	public:
		/** @brief Consume each input once, asserting graph/id/value-count/allocation bounds before use. */
		template<graph::successor_range Successors, typename Values>
			requires graph::functional_graph_value_range<Values, Value>
		constexpr functional_graph_walk_index_by_chain_prefix(Successors &&successors, Values &&values, Value identity, Operation operation)
			: engine_(std::forward<Successors>(successors), std::forward<Values>(values), std::move(identity), std::move(operation)) {}

		/** @brief Number of vertices. */
		constexpr std::size_t size() const noexcept { return engine_.size(); }

		/** @brief Canonical metadata; reference lasts until destruction, move or assignment. */
		[[nodiscard]] constexpr graph::functional_graph_decomposition const &decomposition() const noexcept { return engine_.decomposition(); }

		/** @brief Value-independent endpoint; asserted start/step bounds. O(log n). */
		[[nodiscard]] constexpr graph::vertex_id successor(graph::vertex_id start, uint64 steps) const { return engine_.successor(start, steps); }

		/** @brief Ordered edge fold; asserted start/step bounds. O(log n + log(steps+1)). */
		[[nodiscard]] constexpr graph::functional_graph_walk_result<Value> walk(graph::vertex_id start, uint64 steps) const { return engine_.walk(start, steps); }
	};

	/** @brief Offline reverse-tree DFS endpoint batch; experimental contender; not measured yet.
	 * Owns each input in one pass, preserving query order and duplicates. Empty graphs/batches are valid.
	 * @pre n < 2^31; raw successors nonnegative, representable and < n; query starts < n,
	 * steps <= 10^18 (asserted).
	 * @param successors Sized input range with one target per vertex.
	 * @param queries Sized input range of start/step pairs; references may be convertible proxies.
	 * @return Endpoints in input query order, including duplicate queries.
	 * @complexity O(n+q) time and peak storage, including output. No retained index.
	 */
	template<graph::successor_range Successors, graph::functional_graph_query_range Queries>
	[[nodiscard]] constexpr std::vector<graph::vertex_id> functional_graph_successors_by_reverse_dfs(Successors &&successors, Queries &&queries) {
		auto const owned = graph::detail::functional_graph_input(std::forward<Successors>(successors));
		auto const decomposition = graph::detail::functional_graph_peeling(owned);
		detail::functional_graph_batch const batch(std::forward<Queries>(queries), owned.size());
		std::vector<graph::vertex_id> result(batch.queries.size());
		detail::functional_graph_reverse_dfs(owned, decomposition, [&](std::vector<graph::vertex_id> const &active) {
			auto const v = active.back();
			auto const depth = active.size() - 1;
			for (std::size_t id = batch.head[v]; id != detail::functional_graph_batch::end; id = batch.next[id]) {
				auto const steps = batch.queries[id].steps;
				if (steps < depth) {
					atl_assert(steps <= (std::numeric_limits<std::size_t>::max)());
					result[id] = active[depth - std::size_t(steps)];
				} else
					result[id] = graph::detail::functional_graph_cycle_vertex(decomposition, active.front(), steps - depth);
			}
		});
		return result;
	}

	/** @brief Offline reverse-tree DFS folds with a reversed active-depth segment tree; experimental contender; not measured yet.
	 * Same ordered folds/owning-range semantics and monoid laws as graph::functional_graph_walk_index.
	 * Returns plain vector in input query order, retaining duplicates; empty graphs/batches are valid.
	 * @param successors Sized input range with one target per vertex.
	 * @param values Sized input range of outgoing-edge values in vertex order.
	 * @param queries Sized input range of start/step pairs; references may be convertible proxies.
	 * @param identity Two-sided neutral element. @param operation Pure associative ordered composition.
	 * @return Endpoints and ordered edge folds in input query order, including duplicates.
	 * @pre Graph/id/value-count/query/allocation bounds of the public indices are asserted.
	 * Associativity, two-sided identity, purity and all intermediate folds representable are documented only.
	 * @complexity O((n+q)log(n+1) + sum log(steps+1)) time, O(n+q) peak space including output.
	 */
	template<graph::successor_range Successors, typename Values, graph::functional_graph_query_range Queries,
			 std::copyable Value, binary_operation<Value> Operation>
		requires std::copyable<Operation> && graph::functional_graph_value_range<Values, Value>
	[[nodiscard]] constexpr std::vector<graph::functional_graph_walk_result<Value>> functional_graph_walks_by_reverse_dfs(
		Successors &&successors, Values &&values, Queries &&queries, Value identity, Operation operation) {
		atl_assert(std::ranges::size(successors) <= graph::detail::functional_graph_maximum_fold_vertices);
		auto const owned = graph::detail::functional_graph_input(std::forward<Successors>(successors));
		auto const n = owned.size();
		auto const owned_values = graph::detail::functional_graph_values<Value>(std::forward<Values>(values), n);
		auto const decomposition = graph::detail::functional_graph_peeling(owned);
		detail::functional_graph_batch const batch(std::forward<Queries>(queries), n);
		std::vector<graph::functional_graph_walk_result<Value>> result(batch.queries.size(), {graph::none, identity});
		graph::detail::functional_graph_cycle_folds<Value, Operation> const cycles(decomposition, owned_values, identity, operation);
		segment_tree<Value, graph::detail::functional_graph_operation<Value, Operation>> active_values(n, identity, {operation});
		detail::functional_graph_reverse_dfs(owned, decomposition, [&](std::vector<graph::vertex_id> const &active) {
			auto const v = active.back();
			auto const depth = active.size() - 1;
			active_values.set(n - 1 - depth, owned_values[v]);
			for (std::size_t id = batch.head[v]; id != detail::functional_graph_batch::end; id = batch.next[id]) {
				auto const steps = batch.queries[id].steps;
				uint64 const tail = (std::ranges::min)(steps, uint64(depth));
				atl_assert(tail <= (std::numeric_limits<std::size_t>::max)());
				Value fold = active_values.fold(n - 1 - depth, n - 1 - depth + std::size_t(tail));
				graph::vertex_id endpoint = graph::none;
				if (steps < depth)
					endpoint = active[depth - std::size_t(tail)];
				else {
					endpoint = graph::detail::functional_graph_cycle_vertex(decomposition, active.front(), steps - depth);
					fold = std::invoke(std::as_const(operation), fold, cycles.fold(decomposition, active.front(), steps - depth));
				}
				result[id] = {endpoint, std::move(fold)};
			}
		});
		return result;
	}
}
