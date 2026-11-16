#pragma once
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief One reversible union recorded by @ref rollback_dsu.
		 *
		 * Storing @ref child's prior set size is enough to restore both slots the
		 * merge touched: @ref child goes back to a root of that size, and @ref root
		 * shrinks by it. Kept as 32-bit ids rather than std::size_t: the log holds
		 * one entry per live union, and both ids and sizes are bounded by the int32
		 * element storage, so uint32 halves its footprint.
		 */
		struct dsu_undo {
			uint32 child;	   ///< root that was attached under @ref root
			uint32 root;	   ///< surviving root
			uint32 child_size; ///< @ref child's set size just before the merge
		};
	}

	/**
	 * @brief Disjoint-set union with rollback: unions can be undone in LIFO order.
	 *
	 * Union by size with **no path compression** (compression rewrites nodes on a
	 * read, and logging those writes would add rollback work), so the structure stays a plain pair
	 * of arrays and every operation is cheap: `find`/`same`/`size` and `unite` are
	 * O(log n), while `undo` is O(1). This is the mutable, single-history
	 * counterpart to @ref persistent_dsu — far smaller constants when a problem only
	 * needs to walk one timeline (e.g. an offline DFS over a version tree, undoing
	 * on the way back up), at the cost of not being able to branch off an old
	 * version or keep two versions alive at once.
	 *
	 * A single array encodes both fields: a non-negative value is the
	 * parent index; a negative value marks a root whose set size is its magnitude.
	 * Vertices are numbered `[0, n)`; `n` must fit `int32` (≤ 2³¹−1).
	 *
	 *     rollback_dsu d(n);
	 *     auto t = d.time();
	 *     d.unite(a, b);
	 *     d.rollback(t);   // back to before the unite
	 */
	class rollback_dsu {
		// parent_or_size_[x] >= 0: parent index; < 0: x is a root with size -value.
		std::vector<int32> parent_or_size_;
		std::vector<detail::dsu_undo> history_;
		std::size_t count_ = 0;

	public:
		/** @brief An empty DSU (0 elements). */
		rollback_dsu() = default;

		/**
		 * @brief A DSU of `n` singleton sets {0}, {1}, …, {n-1}.
		 * @param n Number of elements.
		 * @pre `n <= 2³¹−1` (the int32 element storage, which also bounds the uint32 undo log)
		 * @complexity O(n)
		 */
		constexpr explicit rollback_dsu(std::size_t n) : count_(n) {
			atl_assert(n <= std::size_t((std::numeric_limits<int32>::max)()));
			parent_or_size_.assign(n, int32(-1));
			history_.reserve(n); // a spanning forest merges at most n-1 times
		}

		/** @brief Number of elements. */
		constexpr std::size_t elements() const noexcept { return parent_or_size_.size(); }

		/** @brief Number of disjoint sets. */
		constexpr std::size_t count() const noexcept { return count_; }

		/**
		 * @brief The representative (root) of `x`'s set.
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr std::size_t find(std::size_t x) const {
			atl_assert(x < parent_or_size_.size());
			while (parent_or_size_[x] >= 0) x = std::size_t(parent_or_size_[x]);
			return x;
		}

		/**
		 * @brief Whether `a` and `b` belong to the same set.
		 * @param a First element index.
		 * @param b Second element index.
		 * @pre `a < elements()` and `b < elements()` (asserted).
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr bool same(std::size_t a, std::size_t b) const { return find(a) == find(b); }

		/**
		 * @brief The number of elements in the set containing `x`.
		 * @param x Element index.
		 * @pre `x < elements()` (asserted).
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr std::size_t size(std::size_t x) const { return std::size_t(-parent_or_size_[find(x)]); }

		/**
		 * @brief Merge the sets of `a` and `b` (union by size).
		 * @param a First element index.
		 * @param b Second element index.
		 * @pre Both indices are less than `elements()` (asserted).
		 * @return `true` if a merge happened, `false` if they were already together.
		 *         Only true merges are recorded, so only they are undone by @ref undo.
		 * @complexity O(log n)
		 */
		constexpr bool unite(std::size_t a, std::size_t b) {
			std::size_t ra = find(a), rb = find(b);
			if (ra == rb) return false;
			int32 sa = -parent_or_size_[ra], sb = -parent_or_size_[rb];
			if (sa < sb) std::swap(ra, rb), std::swap(sa, sb);
			history_.push_back({uint32(rb), uint32(ra), uint32(sb)});
			parent_or_size_[rb] = int32(ra);
			parent_or_size_[ra] = -(sa + sb);
			--count_;
			return true;
		}

		/**
		 * @brief An opaque token for the current history length; pass it to
		 *        @ref rollback to return here later. Equals the number of merges
		 *        performed and not yet undone.
		 */
		constexpr std::size_t time() const noexcept { return history_.size(); }

		/**
		 * @brief Undo the most recent merge.
		 * @pre `time() > 0` (asserted).
		 * @complexity O(1)
		 */
		constexpr void undo() {
			atl_assert(!history_.empty());
			auto const [child, root, child_size] = history_.back();
			history_.pop_back();
			parent_or_size_[root] += int32(child_size);
			parent_or_size_[child] = -int32(child_size);
			++count_;
		}

		/**
		 * @brief Undo merges until `time() == t`, where `t` came from an earlier
		 *        @ref time call.
		 * @param t Token from this object's still-live history prefix.
		 * @pre `t <= time()` (asserted); provenance and a still-live prefix are documented-only.
		 * Tokens from an abandoned branch do not identify its state after new merges.
		 * @complexity O(time() - t)
		 */
		constexpr void rollback(std::size_t t) {
			atl_assert(t <= history_.size());
			while (history_.size() > t) undo();
		}
	};
}
