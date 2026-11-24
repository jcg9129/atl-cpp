#pragma once
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash_core.hpp> // Preserve process seed initialization for vertex-only forest users.
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Tag selecting the edge tier of @ref atl::link_cut_tree and
	 *        @ref atl::top_tree : values live on edges, given to `link(u, v, w)`, and
	 *        path folds and updates cover the edges of the path. As the
	 *        `ValuePlacement` template argument, or as the first constructor
	 *        argument (`atl::edge_values`), which deduces it.
	 */
	struct edge_values_t {
		/** @brief Construct the edge-value selector. */
		explicit edge_values_t() = default;
	};

	/** @brief The `atl::edge_values_t` tag value. */
	inline constexpr edge_values_t edge_values{};

	/**
	 * @brief Where a dynamic forest keeps its values: `void` (on vertices;
	 *        the default) or `atl::edge_values_t` (on edges).
	 */
	template<typename ValuePlacement>
	concept value_placement = std::is_void_v<ValuePlacement> || std::same_as<ValuePlacement, edge_values_t>;
}

namespace atl::detail {
	/** @brief The key of the unordered pair {u, v}: (min, max) packed into 64 bits. */
	[[nodiscard]] constexpr uint64 forest_edge_key(uint32 u, uint32 v) {
		return u < v ? uint64(u) << 32 | v : uint64(v) << 32 | u;
	}

	struct forest_edge_registry;

	/**
	 * @brief O(n) acyclicity check of a parent array of @p n vertices
	 *        (assert-only): roots are `graph::none` or self-parents, every
	 *        vertex reaches one, each vertex walked once.
	 */
	[[nodiscard]] constexpr bool is_parent_forest(std::vector<uint32> const &p, std::size_t n) {
		if (p.size() != n) return false;	   // also keeps the uint32 walk indices exact
		std::vector<uint8> state(p.size(), 0); // 0 unseen, 1 on the current walk, 2 reaches a root
		for (std::size_t start = 0; start < p.size(); ++start) {
			uint32 v = uint32(start);
			while (state[v] == 0) {
				state[v] = 1;
				if (p[v] == graph::none || p[v] == v) break;
				if (p[v] >= p.size()) return false;
				v = p[v];
			}
			if (state[v] == 1 && p[v] != graph::none && p[v] != v) return false; // walked into its own walk
			for (uint32 w = uint32(start); state[w] == 1;) {
				state[w] = 2;
				if (p[w] == graph::none || p[w] == w) break;
				w = p[w];
			}
		}
		return true;
	}
}
