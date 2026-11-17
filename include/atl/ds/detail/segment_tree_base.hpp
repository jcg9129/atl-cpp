#pragma once
#include <atl/types/algebraic.hpp>
#include <atl/utils/assert.hpp>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace atl::detail {
	/**
	 * @brief Shared iterative segment-tree machinery — geometry, the canonical-range
	 *        decomposition, the boundary-ancestor walks, and bottom-up rebuilds — reused
	 *        by @ref atl::segment_tree, @ref atl::lazy_segment_tree and
	 *        @ref atl::dual_segment_tree. Knows nothing about the algebra.
	 *
	 * The node payload `T` is whatever the user stores per node: the subtree
	 * aggregate in @ref atl::segment_tree and @ref atl::lazy_segment_tree (the lazy
	 * tree keeps its pending tags in a separate array), the pending tag itself in
	 * @ref atl::dual_segment_tree.
	 *
	 * Layout: a flat array of `2 * size_` nodes where `size_` is the smallest power of
	 * two `>= n`, element `i` is the leaf at index `size_ + i`, and node `k` has
	 * children `2k` and `2k + 1`.
	 */
	template<std::copyable T>
	class segment_tree_base {
	protected:
		std::size_t n_ = 0;	   ///< number of elements
		std::size_t size_ = 1; ///< internal capacity: a power of two >= max(n_, 1)
		std::size_t log_ = 0;  ///< log2(size_): the number of internal levels.
							   ///< Unsigned — loops counting down over levels must
							   ///< use the `i >= 1` (or `i-- > 0`) form, never `i >= 0`.
		std::vector<T> nodes_; ///< 2*size_ node payloads; leaves at [size_, size_ + n_)

		segment_tree_base() = default;

		/** @brief Set the geometry for @p n elements and fill every node with @p fill. */
		constexpr void reset(std::size_t n, T const &fill) {
			atl_assert(n <= (std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 2)));
			n_ = n;
			size_ = std::bit_ceil(n ? n : std::size_t{1});
			log_ = std::size_t(std::countr_zero(size_));
			nodes_.assign(2 * size_, fill);
		}

		constexpr std::size_t leaf(std::size_t i) const noexcept { return size_ + i; }

		/**
		 * @brief Visit the O(log n) canonical nodes tiling `[l, r)`: @p on_left on the
		 * left-boundary nodes (ascending position), @p on_right on the right-boundary
		 * nodes (descending). Splitting the two keeps folds correct even when the
		 * operation is not commutative.
		 */
		template<std::invocable<std::size_t> OnLeft, std::invocable<std::size_t> OnRight>
		constexpr void decompose(std::size_t l, std::size_t r, OnLeft on_left, OnRight on_right) const {
			for (l += size_, r += size_; l < r; l >>= 1, r >>= 1) {
				if (l & 1) std::invoke(on_left, l++);
				if (r & 1) std::invoke(on_right, --r);
			}
		}

		/**
		 * @brief Visit, root-first, the ancestors of the boundary leaves of `[l, r)` whose
		 * subtree only *partially* overlaps the range — exactly the nodes a lazy
		 * tree must push before touching the range's canonical nodes (a fully
		 * covered ancestor just composes the new tag instead).
		 */
		template<std::invocable<std::size_t> Visit>
		constexpr void descend_boundaries(std::size_t l, std::size_t r, Visit visit) const {
			l += size_;
			r += size_;
			for (std::size_t i = log_; i >= 1; --i) {
				if ((l >> i) << i != l) std::invoke(visit, l >> i);
				if ((r >> i) << i != r) std::invoke(visit, (r - 1) >> i);
			}
		}

		/**
		 * @brief Mirror of `descend_boundaries`, leaf-first — the recompute pass after
		 * a range mutation.
		 */
		template<std::invocable<std::size_t> Visit>
		constexpr void ascend_boundaries(std::size_t l, std::size_t r, Visit visit) const {
			l += size_;
			r += size_;
			for (std::size_t i = 1; i <= log_; ++i) {
				if ((l >> i) << i != l) std::invoke(visit, l >> i);
				if ((r >> i) << i != r) std::invoke(visit, (r - 1) >> i);
			}
		}

		template<binary_operation<T> Combine>
		constexpr void recompute_up(std::size_t k, Combine const &combine) {
			while (k >>= 1) nodes_[k] = std::invoke(combine, nodes_[2 * k], nodes_[2 * k + 1]);
		}

		/**
		 * @brief Build the internal nodes bottom-up, once the leaves are filled.
		 *
		 * Only ancestors of the first `n_` leaves are combined: the padding subtrees past
		 * them keep the fill value of `reset` (the identity), which is already their
		 * fold, so the work tracks `n_` rather than `size_` (up to 2x at `n_ = 2^k + 1`).
		 */
		template<binary_operation<T> Combine>
		constexpr void build(Combine const &combine) {
			if (n_ == 0) return;
			for (std::size_t lo = size_ >> 1, hi = (size_ + n_ - 1) >> 1; lo >= 1; lo >>= 1, hi >>= 1)
				for (std::size_t k = lo; k <= hi; ++k) nodes_[k] = std::invoke(combine, nodes_[2 * k], nodes_[2 * k + 1]);
		}
	};
}
