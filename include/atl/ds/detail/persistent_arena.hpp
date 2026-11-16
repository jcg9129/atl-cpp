#pragma once
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace atl::detail {
	/**
	 * @brief Append-only node store backing the persistent structures: nodes are
	 *        addressed by stable `uint32` ids and retained until the arena dies.
	 *        Reallocation moves the objects but preserves their ids, allowing subtree sharing.
	 *
	 * The value constructor reserves id 0 as a supplied null sentinel. The default
	 * constructor reserves no sentinel; node payloads need not be default-constructible.
	 *
	 * **Reallocation contract:** @ref make appends to a `std::vector`, so it may
	 * reallocate — never hold a reference or pointer into the arena across a
	 * `make` call. Copy the node's fields out first, or fully evaluate every
	 * argument before calling (function arguments are sequenced before the call,
	 * so `make(op(arena[a].value, arena[b].value), ...)` is safe; binding
	 * `arena[a].value` to a reference parameter of a function that calls `make` is
	 * not).
	 */
	template<std::move_constructible Node>
	class persistent_arena {
		std::vector<Node> nodes_;

	public:
		/** @brief An empty arena: ids start at 0 with no sentinel reserved. */
		constexpr persistent_arena() = default;

		/** @brief An arena holding only the null sentinel @p null_node (id 0) —
		 *         for structures whose links use id 0 as "none". */
		constexpr explicit persistent_arena(Node null_node) { nodes_.push_back(std::move(null_node)); }

		/** @brief Append @p nd and return its id. May reallocate — see the class contract. */
		constexpr uint32 make(Node nd) {
			atl_assert(nodes_.size() <= (std::numeric_limits<uint32>::max)());
			nodes_.push_back(std::move(nd));
			return uint32(nodes_.size() - 1);
		}

		constexpr Node &operator[](uint32 id) { return nodes_[id]; }

		constexpr Node const &operator[](uint32 id) const { return nodes_[id]; }

		/** @brief Number of nodes, including the sentinel. */
		constexpr std::size_t size() const noexcept { return nodes_.size(); }

		/**
		 * @brief Reserve room for @p more further nodes, so a burst of @ref make
		 *        calls cannot reallocate mid-run.
		 *
		 * @pre `more <= max_size() - size()` (asserted before addition).
		 * A request past capacity grows geometrically, capped at `max_size()`, so reserving per
		 * operation (`reserve(1)` before each update) stays amortised O(1) per node
		 * instead of relocating the whole arena on every call.
		 */
		constexpr void reserve_extra(std::size_t more) {
			atl_assert(more <= nodes_.max_size() - nodes_.size());
			std::size_t const need = nodes_.size() + more;
			if (need > nodes_.capacity()) {
				std::size_t const capacity = nodes_.capacity();
				std::size_t const grown = capacity > nodes_.max_size() / 2 ? nodes_.max_size() : 2 * capacity;
				nodes_.reserve((std::max)(need, grown));
			}
		}
	};
}
