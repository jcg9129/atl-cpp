#pragma once
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Disjoint-set union (union-find): union by size + full path compression.
	 *
	 * The plain, mutable DSU — amortized near-constant O(α(n)) per operation, when a problem needs neither persistence nor
	 * rollback. `find` compresses the path it walks, so it **mutates** and is not
	 * const; @ref persistent_dsu and @ref rollback_dsu deliberately skip compression to
	 * keep reads pure (which is what lets them branch off / undo a version), paying a
	 * log factor for it. Pick this one unless you need those.
	 *
	 * A single array encodes both fields: a non-negative value is the parent
	 * index; a negative value marks a root whose set size is its magnitude. Vertices are
	 * numbered `[0, n)`; `n` must fit `int32` (≤ 2³¹−1).
	 */
	class dsu {
		// parent_or_size_[x] >= 0: parent index; < 0: x is a root with size -value.
		std::vector<int32> parent_or_size_;
		std::size_t count_ = 0;

	public:
		/** @brief An empty DSU (0 elements). */
		dsu() = default;

		/**
		 * @brief A DSU of `n` singleton sets {0}, {1}, …, {n-1}.
		 * @param n Number of elements.
		 * @pre `n <= 2³¹−1` (the int32 element storage)
		 * @complexity O(n)
		 */
		constexpr explicit dsu(std::size_t n) : count_(n) {
			atl_assert(n <= std::size_t((std::numeric_limits<int32>::max)()));
			parent_or_size_.assign(n, int32(-1));
		}

		/** @brief Number of elements. */
		constexpr std::size_t elements() const noexcept { return parent_or_size_.size(); }

		/** @brief Number of disjoint sets. */
		constexpr std::size_t count() const noexcept { return count_; }

		/**
		 * @brief The representative (root) of `x`'s set, pointing every node on the
		 *        path it walks straight at the root (two passes: find, then relink).
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @return Representative of the component containing `x`.
		 * @complexity O(α(n)) amortized
		 */
		constexpr std::size_t find(std::size_t x) {
			atl_assert(x < parent_or_size_.size());
			// Full compression beat halving on deepest trees: 23.1 vs 25.4 ms,
			// with random unions within 3% (benchmark metadata in docs/dsu.md).
			std::size_t root = x;
			while (parent_or_size_[root] >= 0) root = std::size_t(parent_or_size_[root]);
			while (parent_or_size_[x] >= 0) {
				std::size_t const next = std::size_t(parent_or_size_[x]);
				parent_or_size_[x] = int32(root);
				x = next;
			}
			return root;
		}

		/**
		 * @brief Whether `a` and `b` belong to the same set.
		 * @param a First element index.
		 * @param b Second element index.
		 * @pre `a < elements()` and `b < elements()` (asserted).
		 * @complexity O(α(n)) amortized
		 */
		[[nodiscard]] constexpr bool same(std::size_t a, std::size_t b) { return find(a) == find(b); }

		/**
		 * @brief The number of elements in the set containing `x`.
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @complexity O(α(n)) amortized
		 */
		[[nodiscard]] constexpr std::size_t size(std::size_t x) { return std::size_t(-parent_or_size_[find(x)]); }

		/**
		 * @brief Merge the sets of `a` and `b` (union by size).
		 * @param a First element index.
		 * @param b Second element index.
		 * @pre Both indices are less than `elements()` (asserted).
		 * @return `true` if a merge happened, `false` if they were already together.
		 * @complexity O(α(n)) amortized
		 */
		constexpr bool unite(std::size_t a, std::size_t b) {
			std::size_t ra = find(a), rb = find(b);
			if (ra == rb) return false;
			if (-parent_or_size_[ra] < -parent_or_size_[rb]) std::swap(ra, rb);
			parent_or_size_[ra] += parent_or_size_[rb];
			parent_or_size_[rb] = int32(ra);
			--count_;
			return true;
		}
	};
}
