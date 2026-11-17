#pragma once
#include <atl/ds/flat_hash_map.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
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
	 *        argument (@ref atl::edge_values), which deduces it.
	 */
	struct edge_values_t {
		/** @brief Construct the edge-value selector. */
		explicit edge_values_t() = default;
	};

	/** @brief The @ref atl::edge_values_t tag value. */
	inline constexpr edge_values_t edge_values{};

	/**
	 * @brief Where a dynamic forest keeps its values: `void` (on vertices;
	 *        the default) or @ref atl::edge_values_t (on edges).
	 */
	template<typename ValuePlacement>
	concept value_placement = std::is_void_v<ValuePlacement> || std::same_as<ValuePlacement, edge_values_t>;
}

namespace atl::detail {
	/**
	 * @brief The (u, v) -> edge id hash of the dynamic forests (edge tier of @ref atl::link_cut_tree,
	 *        @ref atl::euler_tour_tree): the seeded @ref atl::hash at run
	 *        time; an unseeded mix in constant evaluation, which has no seed to
	 *        read and no adversary.
	 */
	struct forest_edge_hash {
		/** @brief The hash of a packed endpoint pair. */
		constexpr std::size_t operator()(uint64 key) const noexcept {
			if consteval {
				return std::size_t(splitmix64(key));
			} else {
				return hash<uint64>{}(key);
			}
		}
	};

	/** @brief The key of the unordered pair {u, v}: (min, max) packed into 64 bits. */
	[[nodiscard]] constexpr uint64 forest_edge_key(uint32 u, uint32 v) {
		return u < v ? uint64(u) << 32 | v : uint64(v) << 32 | u;
	}

	/**
	 * @brief The edge-tier bookkeeping of the edge-as-vertex forests
	 *        (@ref atl::link_cut_tree, @ref atl::top_tree): (min, max) endpoint pair ->
	 *        edge id, the endpoints per id (`graph::none` when free), and the
	 *        free ids, handed out lowest first and reused most recently freed
	 *        first (dense LIFO ids).
	 */
	struct forest_edge_registry {
		flat_hash_map<uint64, uint32, forest_edge_hash> ids;
		std::vector<std::pair<uint32, uint32>> endpoints;
		std::vector<uint32> free_ids;

		/** @brief @p slots free ids, no edges. @complexity O(slots) */
		constexpr void reset(std::size_t slots) {
			atl_assert(slots < graph::none); // ids are 32-bit with none reserved
			ids.clear();
			ids.reserve(slots);
			endpoints.assign(slots, {graph::none, graph::none});
			free_ids.resize(slots);
			// popped from the back: the lowest id first
			for (std::size_t i = 0; i < slots; ++i) free_ids[i] = uint32(slots - 1 - i);
		}

		/** @brief Register the edge u - v under a free id and return it. @pre a free id exists */
		constexpr uint32 acquire(uint32 u, uint32 v) {
			uint32 const e = free_ids.back();
			free_ids.pop_back();
			ids.insert(forest_edge_key(u, v), e);
			endpoints[e] = {u, v};
			return e;
		}

		/** @brief Forget the live edge @p e; its id becomes free. */
		constexpr void release(uint32 e) {
			auto const [a, b] = endpoints[e];
			ids.erase(forest_edge_key(a, b));
			endpoints[e] = {graph::none, graph::none};
			free_ids.push_back(e);
		}

		/** @brief The id of edge u - v, `graph::none` when there is none. @complexity expected O(1) */
		[[nodiscard]] constexpr uint32 find(uint32 u, uint32 v) const { return ids.at_or(forest_edge_key(u, v), graph::none); }

		/** @brief Whether @p e is a live edge id. */
		[[nodiscard]] constexpr bool is_live(uint32 e) const { return e < endpoints.size() && endpoints[e].first != graph::none; }
	};

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
