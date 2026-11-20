#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/counting_sort.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace atl::graph {
	/**
	 * @brief Disjoint vertex lists ("groups") in one flat array: group `g` is
	 *        `members[offsets[g] .. offsets[g + 1])`. The same layout as a
	 *        @ref atl::graph::csr row table, minus the graph interface — at most two allocations
	 *        independent of the number of groups.
	 * @pre Public arrays preserve the stated offsets invariant; group slices
	 * are invalidated by changes to members storage or destruction of its owner.
	 */
	struct vertex_groups {
		std::vector<uint32> offsets{0}; ///< `count() + 1` non-decreasing entries, starting at 0
		std::vector<vertex_id> members; ///< all groups back to back; `offsets.back() == members.size()`

		/** @brief Number of groups. */
		[[nodiscard]] constexpr std::size_t count() const noexcept { return offsets.size() - 1; }

		/** @brief The vertices of group @p g. @pre `g < count()` */
		[[nodiscard]] constexpr std::span<vertex_id const> operator[](std::size_t g) const {
			atl_assert(g < count());
			return std::span<vertex_id const>(members).subspan(offsets[g], offsets[g + 1] - offsets[g]);
		}
	};

	/**
	 * @brief A partition of the vertices into bounded component ids plus
	 *        read-only conveniences over it — the result shape shared by the
	 *        connectivity algorithms (@ref atl::graph::scc,
	 *        @ref atl::graph::two_edge_connected_components).
	 *
	 * Whether ids are dense or carry an order is each algorithm's promise:
	 * see the returning function.
	 */
	struct vertex_partition {
		std::vector<uint32> component; ///< component id per vertex, in `[0, count())`

		/**
		 * @brief Wraps @p component_ids, whose ids lie in `[0, component_count)`; unused ids are allowed.
		 * @pre Vertex and component counts fit `uint32`; `component_count + 1` fits `size_t`.
		 * Every id is `< component_count` (an O(n) scan, run only under `ATL_ASSERT`)
		 */
		constexpr vertex_partition(std::vector<uint32> component_ids, std::size_t component_count)
			: component(std::move(component_ids)), count_(component_count) {
			atl_assert(component.size() <= none && component_count <= none
					   && component_count < (std::numeric_limits<std::size_t>::max)());
			atl_assert(std::ranges::all_of(component, [component_count](uint32 c) { return c < component_count; }));
		}

		/** @brief Number of components; invariant `component[v] < count()` for every v. */
		[[nodiscard]] constexpr std::size_t count() const noexcept { return count_; }

		/** @brief Whether two vertices share a component.
		 * @pre Both ids are below `component.size()` (asserted). @complexity O(1)
		 */
		[[nodiscard]] constexpr bool same(vertex_id u, vertex_id v) const {
			atl_assert(u < component.size() && v < component.size());
			return component[u] == component[v];
		}

		/**
		 * @brief The vertices of every component, bucketed by id and ascending within a
		 *        component, as one flat array: the cheap form (no per-component
		 *        allocation). @complexity O(n + count())
		 */
		[[nodiscard]] constexpr vertex_groups grouped() const {
			vertex_groups out{std::vector<uint32>(count() + 1), std::vector<vertex_id>(component.size())};
			atl::detail::bucket_by_key(
				count(), [&](std::size_t v) { return std::size_t(component[v]); }, out.offsets, out.members);
			return out;
		}

		/**
		 * @brief The vertices of every component, bucketed by id, as nested vectors: the
		 *        convenient form (prints as-is, indexes like any `vector<vector>`);
		 *        prefer @ref atl::graph::vertex_partition::grouped when there are many components.
		 * @complexity O(n + count()) time and storage; vertices ascend within each group.
		 */
		[[nodiscard]] constexpr std::vector<std::vector<vertex_id>> groups() const {
			std::vector<std::vector<vertex_id>> out(count());
			std::vector<uint32> sizes(count(), 0);
			for (auto const c : component) ++sizes[c];
			for (std::size_t c = 0; c < count(); ++c) out[c].reserve(sizes[c]);
			for (vertex_id v = 0; v < component.size(); ++v) out[component[v]].push_back(v);
			return out;
		}

	private:
		// Storing the count avoids a scan of component on every query.
		std::size_t count_;
	};
}
