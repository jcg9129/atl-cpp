#pragma once

/**
 * @file
 * @brief @ref atl::experimental::deletion_time_connectivity: connectivity of a general
 *        multigraph under edge insertions and deletions, online, when every
 *        edge's deletion time is known as it is inserted (any offline log), on the edge tier of
 *        @ref atl::link_cut_tree. One pass, expected amortized O(n + T log(n + 1)) for T
 *        events; @ref atl::experimental::component_counts
 *        (`atl/experimental/graph/offline_dynamic_connectivity.hpp`) answers the same logs offline by
 *        @ref atl::time_divide_and_conquer in O(n + T log(T + 1) log(n + 1)).
 *
 * The structure keeps a MAXIMUM spanning forest of the live edges, keyed by
 * deletion time. Inserting u - v links it when u and v are apart; otherwise
 * it closes a cycle, and replaces the path's earliest-dying edge if it
 * outlives it (the cycle rule keeps the forest maximum). Deleting an edge
 * needs no replacement search: edges die in deletion-time order, so the one
 * dying is the minimum of the whole graph, and a non-tree edge f across its
 * cut would close a cycle through it with f lighter than every tree edge on
 * that cycle, yet f is alive, so heavier than the minimum: there is none.
 * Hence the forest always spans exactly the live graph's components.
 */

#include <atl/ds/link_cut_tree.hpp>
#include <atl/ds/time_divide_and_conquer.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace atl::experimental {
	using atl::uint32;
	using atl::uint8;
	using atl::graph::vertex_id;

	/**
	 * @brief Online connectivity of a multigraph on `n` vertices whose edges
	 *        each declare their deletion time on insertion.
	 *
	 * Edges are named by handles 0, 1, ... in insertion order. Parallel edges
	 * and self-loops are allowed (a self-loop never joins the forest).
	 * Storage is O(n + I) for I insertions, including erased handles. Handles remain
	 * stable; `graph::none` is reserved. Forest walks are iterative, with no recursion
	 * proportional to the number of vertices.
	 * @note Not constexpr (the experimental interface has runtime-only members).
	 */
	class deletion_time_connectivity {
		/** @brief An edge's forest key: (deletion time, handle), the handle breaking ties. */
		using dying = std::pair<uint32, uint32>;

		/** @brief The earlier-dying of two keys: the path fold finds the edge to replace. */
		struct earlier {
			[[nodiscard]] constexpr dying operator()(dying const &a, dying const &b) const { return b < a ? b : a; }
		};

		static constexpr uint32 none = atl::graph::none;
		/** @brief The fold's identity, later than every key: an empty path has nothing to replace. */
		static constexpr dying never = {(std::numeric_limits<uint32>::max)(), (std::numeric_limits<uint32>::max)()};

		atl::link_cut_tree<dying, earlier, void, void, void, atl::commutative_fold_t, atl::edge_values_t> forest_;
		std::vector<uint32> forest_edge_; ///< per handle: its forest edge id, `none` while not a tree edge
		uint32 components_;

	public:
		/**
		 * @brief Construct `n` isolated vertices.
		 * @param n Number of vertices, with ids in `[0, n)`.
		 * @complexity O(n) time and memory.
		 * @note Not constexpr (the experimental constructor is runtime-only).
		 * @pre `n < 2^31` (asserted by `forest_`, constructed first, so the count's narrowing is safe)
		 */
		explicit deletion_time_connectivity(std::size_t n)
			: forest_(atl::edge_values, atl::commutative_fold, n, never, earlier{}), components_(uint32(n)) {}

		/**
		 * @brief Insert the edge u - v, to be erased at @p deletion_time, and
		 *        return its handle.
		 * @param u First endpoint.
		 * @param v Second endpoint.
		 * @param deletion_time Future deletion key, excluding `2^32 - 1` (asserted).
		 * @return Stable handle in insertion order, excluding `graph::none`.
		 * @pre At most `2^32 - 1` insertions, including this one (asserted).
		 * @pre `u, v < n` (asserted); @p deletion_time is later than (or, with a
		 *      later handle, equal to) the deletion time of every edge erased
		 *      so far, and below `2^32 - 1`
		 * @complexity Expected amortized O(log(n + 1)), including hash-table operations.
		 * @note Not constexpr (the experimental member is runtime-only).
		 */
		uint32 insert(vertex_id u, vertex_id v, uint32 deletion_time) {
			atl_assert(forest_edge_.size() < none && deletion_time != never.first);
			auto const handle = uint32(forest_edge_.size());
			forest_edge_.push_back(none);
			dying const key{deletion_time, handle};
			// one pair of exposures for both the connectivity test and the path fold
			std::optional<dying> const earliest = forest_.try_fold(u, v);
			if (!earliest) {
				forest_edge_[handle] = forest_.link(u, v, key);
				--components_;
				return handle;
			}
			// a cycle (a self-loop folds to `never`): keep the later-dying of key and the path minimum
			if (*earliest < key) {
				forest_.cut(forest_edge_[earliest->second]);
				forest_edge_[earliest->second] = none;
				forest_edge_[handle] = forest_.link(u, v, key);
			}
			return handle;
		}

		/**
		 * @brief Erase the edge @p handle.
		 * @param handle Stable insertion handle.
		 * @pre `handle` was inserted and not erased; edges are erased in
		 *      increasing deletion time, ties in handle order (asserted: only
		 *      that the handle exists)
		 * @complexity Expected amortized O(log(n + 1)), including hash-table operations.
		 * @note Not constexpr (the experimental member is runtime-only).
		 */
		void erase(uint32 handle) {
			atl_assert(handle < forest_edge_.size());
			if (uint32 const e = forest_edge_[handle]; e != none) {
				forest_.cut(e);
				forest_edge_[handle] = none;
				++components_;
			}
		}

		/**
		 * @brief Whether live edges connect two vertices; a vertex is connected to itself.
		 * @param u First endpoint.
		 * @param v Second endpoint.
		 * @pre `u, v < n` (asserted).
		 * @return True exactly when both vertices belong to one component.
		 * @complexity Amortized O(log(n + 1)).
		 * @note Not constexpr (the experimental member is runtime-only).
		 */
		[[nodiscard]] bool connected(vertex_id u, vertex_id v) { return forest_.connected(u, v); }

		/**
		 * @brief Count components, including isolated vertices.
		 * @return A value in `[0, n]`; zero exactly when `n == 0`.
		 * @complexity O(1).
		 * @note Not constexpr (the experimental member is runtime-only).
		 */
		[[nodiscard]] uint32 component_count() const { return components_; }
	};

	/** @brief An add / remove / query log, the format of @ref atl::experimental::component_counts. */
	using deletion_log = std::span<atl::item_event<std::pair<vertex_id, vertex_id>> const>;

	/**
	 * @brief Replay an add / remove / query log through
	 *        @ref atl::experimental::deletion_time_connectivity, calling `on_query(structure, t)` at
	 *        every query event `t`. The deletion times come from
	 *        @ref atl::lifetimes_of (a remove closes the oldest live copy of its
	 *        key; a copy never removed dies at `T`).
	 * @tparam OnQuery Callable on `deletion_time_connectivity &` and `uint32`, including reference wrappers.
	 * @param n Number of vertices, below `2^31` (asserted).
	 * @param events Log with fewer than `2^32 - 1` events (asserted by `lifetimes_of`).
	 * @param on_query Invoked at query times in increasing order; must preserve the represented graph.
	 * @pre Every non-query endpoint is below `n`; every remove matches a live copy (asserted).
	 * @complexity Expected amortized O(n + T log(n + 1)) time and O(n + T) memory.
	 * @note Not constexpr (event pairing uses a runtime-seeded hash map). Exceptions propagate;
	 *       callbacks receive a borrowed structure valid only during the call.
	 */
	template<typename OnQuery>
		requires std::invocable<OnQuery &, deletion_time_connectivity &, uint32>
	void replay_by_deletion_time(std::size_t n, deletion_log events, OnQuery &&on_query) {
		std::vector<atl::item_event<std::pair<vertex_id, vertex_id>>> normalized(events.begin(), events.end());
		for (auto &[kind, key] : normalized) {
			if (kind == atl::event_kind::query) continue;
			atl_assert(key.first < n && key.second < n);
			if (key.first > key.second) std::swap(key.first, key.second);
		}
		auto const lifetimes = atl::lifetimes_of(normalized);
		uint32 const times = uint32(normalized.size()); // lifetimes_of asserted it fits
		std::vector<uint32> closed_by(times, atl::graph::none);
		for (uint32 item = 0; item < lifetimes.size(); ++item)
			if (lifetimes[item].end < times) closed_by[lifetimes[item].end] = item;

		deletion_time_connectivity structure(n);
		uint32 next_item = 0;
		for (uint32 t = 0; t < times; ++t) {
			auto const &[kind, key] = normalized[t];
			if (kind == atl::event_kind::add) {
				// handles follow add order, so the handle is the lifetimes_of item
				structure.insert(key.first, key.second, lifetimes[next_item++].end);
			} else if (kind == atl::event_kind::remove) {
				structure.erase(closed_by[t]);
			} else {
				std::invoke(on_query, structure, t);
			}
		}
	}

	/**
	 * @brief Per query event, the number of connected components (the
	 *        @ref atl::experimental::component_counts contract), in one online pass.
	 * @param n Number of vertices, below `2^31` (asserted).
	 * @param events Add / remove / query log; the input is neither modified nor retained.
	 * @return One answer per query event, in log order.
	 * @pre Fewer than `2^32 - 1` events; every remove matches a live copy;
	 *      every non-query endpoint is below `n` (asserted).
	 * @complexity Expected amortized O(n + T log(n + 1)) time and O(n + T) memory.
	 * @note Not constexpr (event pairing uses a runtime-seeded hash map).
	 */
	[[nodiscard]] inline std::vector<uint32> component_counts_by_deletion_time(std::size_t n, deletion_log events) {
		std::vector<uint32> counts;
		replay_by_deletion_time(n, events, [&](deletion_time_connectivity &s, uint32) { counts.push_back(s.component_count()); });
		return counts;
	}

	/**
	 * @brief Per query event, whether the pair in its key is connected.
	 * @pre every query key's endpoints `< n`
	 * @param n Number of vertices, below `2^31` (asserted).
	 * @param events Add / remove / query log; the input is neither modified nor retained.
	 * @return One answer per query event, in log order.
	 * @pre Fewer than `2^32 - 1` events; every remove matches a live copy;
	 *      every non-query endpoint is below `n` (asserted).
	 * @complexity Expected amortized O(n + T log(n + 1)) time and O(n + T) memory.
	 * @note Not constexpr (event pairing uses a runtime-seeded hash map).
	 */
	[[nodiscard]] inline std::vector<uint8> connected_by_deletion_time(std::size_t n, deletion_log events) {
		std::vector<uint8> answers;
		replay_by_deletion_time(n, events, [&](deletion_time_connectivity &s, uint32 t) {
			auto const [u, v] = events[t].key;
			answers.push_back(s.connected(u, v));
		});
		return answers;
	}
}
