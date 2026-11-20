/** @file
 * @brief Forward-only row adapters for graph traversal comparisons: stable
 * iterators replace random-access positions. Adjacency is copied outside timing;
 * separate row allocations mean this comparison does not isolate scan positions.
 * Standalone adapter performance: not measured yet.
 */
#pragma once
#include <atl/types/alias.hpp>
#include <cstddef>
#include <iterator>
#include <vector>

namespace atl_bench {
	/**
	 * @brief One adjacency row over a `std::vector<uint32>` exposing forward
	 *        iterators only. A `std::vector<forward_row>` graph makes the
	 *        traversal engine save `{iterator, edge index}` scan positions,
	 *        retaining stable iterators for reference-returning adjacency. Caution: these rows are separate allocations from the
	 *        `vector<vector>` they copy, so comparisons against them also include allocation-layout effects.
	 */
	struct forward_row {
		std::vector<atl::uint32> targets;

		/** @brief A `vector` iterator narrowed to the forward-iterator interface. */
		struct iterator {
			using iterator_concept = std::forward_iterator_tag;
			using iterator_category = std::forward_iterator_tag;
			using value_type = atl::uint32;
			using difference_type = std::ptrdiff_t;

			std::vector<atl::uint32>::const_iterator at{};

			atl::uint32 const &operator*() const { return *at; }

			iterator &operator++() {
				++at;
				return *this;
			}

			iterator operator++(int) {
				iterator const saved = *this;
				++at;
				return saved;
			}

			bool operator==(iterator const &) const = default;
		};

		iterator begin() const { return {targets.begin()}; }

		iterator end() const { return {targets.end()}; }
	};

	/** @brief The forward-only copy of a `vector<vector<uint32>>` adjacency. */
	inline std::vector<forward_row> forward_rows(std::vector<std::vector<atl::uint32>> const &lists) {
		std::vector<forward_row> rows(lists.size());
		for (std::size_t v = 0; v < lists.size(); ++v) rows[v].targets = lists[v];
		return rows;
	}
}
