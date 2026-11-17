#pragma once
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Dynamic (implicit) segment tree: point assignment and range fold
	 *        over a huge `[0, n)` — nodes materialized only where updates land.
	 *
	 * The online counterpart of coordinate compression: `n` may be 10⁹ or
	 * 10¹⁸. Every position an update never visited holds `identity`, and an
	 * unmaterialized subtree folds to the identity by construction; this
	 * "untouched = identity" contract is what makes sparse folds exact without
	 * per-node width bookkeeping (zeros for sums, identity maps for
	 * composites, +inf for minima).
	 *
	 * A subtree containing a **single** touched position is kept as one
	 * *pending* node storing `(pos, value)` — the path below materializes only
	 * when a second position lands in the subtree, i.e. down to where the two
	 * diverge. Scattered updates avoid materializing most unary paths. A mutation
	 * still takes O(log n) time and new nodes in the worst case; clustered
	 * positions can require long chains of unary internal nodes.
	 *
	 * Same monoid vocabulary as @ref atl::segment_tree. Reads (`fold`,
	 * `at`) never allocate.
	 *
	 * @tparam T  The element type.
	 * @tparam Op Element fold: any `op(T, T)` convertible to `T`.
	 */
	template<std::copyable T, binary_operation<T> Op>
	class dynamic_segment_tree {
		/**
		 * @brief `pending`: `aggregate` is the value of the single touched position `pos`
		 * in this subtree, children unused. Internal: `aggregate` folds the children.
		 */
		struct node {
			T aggregate;
			std::size_t pos;
			uint32 left = 0, right = 0;
			bool is_pending = false;
		};

		[[no_unique_address]] Op op_;
		T element_identity_;
		std::size_t n_ = 0;
		std::vector<node> nodes_; // id 0: the null subtree, aggregate = identity
		uint32 root_ = 0;

		constexpr uint32 make(T aggregate, std::size_t pos, bool is_pending) {
			atl_assert(nodes_.size() < 0xFFFFFFFFull);
			nodes_.push_back({std::move(aggregate), pos, 0, 0, is_pending});
			return uint32(nodes_.size() - 1);
		}

		/**
		 * @brief Replace element @p i with `leaf(current)` in subtree @p x spanning
		 * `[lo, hi)`; returns the subtree's (possibly new) id.
		 */
		template<update_function<T> Leaf>
		constexpr uint32 point_update(uint32 x, std::size_t lo, std::size_t hi, std::size_t i, Leaf &&leaf) {
			if (x == 0) return make(std::invoke(leaf, std::as_const(element_identity_)), i, true);
			if (nodes_[x].is_pending) {
				if (nodes_[x].pos == i) {
					node &nd = nodes_[x];
					nd.aggregate = std::invoke(leaf, std::as_const(nd.aggregate));
					return x;
				}
				// a second position arrived: push the pending element one level
				// down (as pending again) and fall through as an internal node
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 const c = make(std::move(nodes_[x].aggregate), nodes_[x].pos, true);
				node &nd = nodes_[x];
				nd.is_pending = false;
				(nodes_[c].pos < mid ? nd.left : nd.right) = c;
			}
			std::size_t const mid = lo + (hi - lo) / 2;
			if (i < mid) {
				uint32 const c = point_update(nodes_[x].left, lo, mid, i, leaf);
				nodes_[x].left = c; // re-indexed: the recursion may reallocate
			} else {
				uint32 const c = point_update(nodes_[x].right, mid, hi, i, leaf);
				nodes_[x].right = c;
			}
			node &nd = nodes_[x];
			nd.aggregate = std::invoke(std::as_const(op_), nodes_[nd.left].aggregate, nodes_[nd.right].aggregate);
			return x;
		}

		[[nodiscard]] constexpr T fold(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r) const {
			if (x == 0 || r <= lo || hi <= l) return element_identity_;
			node const &nd = nodes_[x];
			if (nd.is_pending) return l <= nd.pos && nd.pos < r ? nd.aggregate : element_identity_;
			if (l <= lo && hi <= r) return nd.aggregate;
			std::size_t const mid = lo + (hi - lo) / 2;
			// recurse only into children that exist and overlap: a one-sided level
			// then costs no call, no identity copy and no op against the identity
			bool const goes_left = l < mid && nd.left != 0, goes_right = mid < r && nd.right != 0;
			if (!goes_right) return goes_left ? fold(nd.left, lo, mid, l, r) : element_identity_;
			if (!goes_left) return fold(nd.right, mid, hi, l, r);
			return std::invoke(std::as_const(op_), fold(nd.left, lo, mid, l, r), fold(nd.right, mid, hi, l, r));
		}

		/** @brief Sentinel for "the whole subtree satisfied the predicate". */
		static constexpr std::size_t npos = std::size_t(-1);

		/**
		 * @brief max_right recursion over subtree @p x spanning `[lo, hi)`: fold
		 * the part at or past @p l into the running prefix @p sm while @p pred
		 * holds. Returns the first index that breaks it, or `npos` if the
		 * whole overlap fits. Untouched positions are the identity and never
		 * break a predicate that already holds.
		 */
		template<std::predicate<T const &> Pred>
		constexpr std::size_t max_right_descend(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, T &sm, Pred &pred) const {
			if (x == 0 || hi <= l) return npos;
			node const &nd = nodes_[x];
			if (nd.is_pending) {
				if (nd.pos < l) return npos;
				T t = std::invoke(std::as_const(op_), sm, nd.aggregate);
				if (!std::invoke(pred, t)) return nd.pos;
				sm = std::move(t);
				return npos;
			}
			if (l <= lo) {
				T t = std::invoke(std::as_const(op_), sm, nd.aggregate);
				if (std::invoke(pred, t)) {
					sm = std::move(t);
					return npos;
				}
				if (hi - lo == 1) return lo;
			}
			std::size_t const mid = lo + (hi - lo) / 2;
			if (std::size_t const b = max_right_descend(nd.left, lo, mid, l, sm, pred); b != npos) return b;
			return max_right_descend(nodes_[x].right, mid, hi, l, sm, pred);
		}

		/**
		 * @brief min_left mirror of `max_right_descend`: folds the part of
		 * `[lo, hi)` before @p r into the running suffix @p sm, right to left.
		 * Returns one past the first (rightmost) index that breaks @p pred, or
		 * zero if the whole overlap fits. A failing endpoint is at least one,
		 * so zero stays unambiguous even when the domain reaches `size_t` max.
		 */
		template<std::predicate<T const &> Pred>
		constexpr std::size_t min_left_descend(uint32 x, std::size_t lo, std::size_t hi, std::size_t r, T &sm, Pred &pred) const {
			if (x == 0 || r <= lo) return 0;
			node const &nd = nodes_[x];
			if (nd.is_pending) {
				if (nd.pos >= r) return 0;
				T t = std::invoke(std::as_const(op_), nd.aggregate, sm);
				if (!std::invoke(pred, t)) return nd.pos + 1;
				sm = std::move(t);
				return 0;
			}
			if (hi <= r) {
				T t = std::invoke(std::as_const(op_), nd.aggregate, sm);
				if (std::invoke(pred, t)) {
					sm = std::move(t);
					return 0;
				}
				if (hi - lo == 1) return hi;
			}
			std::size_t const mid = lo + (hi - lo) / 2;
			if (std::size_t const b = min_left_descend(nd.right, mid, hi, r, sm, pred); b != 0) return b;
			return min_left_descend(nodes_[x].left, lo, mid, r, sm, pred);
		}

	public:
		/** @brief Stored element or aggregate type. */
		using value_type = T;
		/** @brief Type of positions and element counts. */
		using size_type = std::size_t;

		/**
		 * @brief An empty tree over `[0, 0)` with a value-initialized identity — a
		 * placeholder to assign a real tree to later. @complexity O(1)
		 */
		constexpr dynamic_segment_tree()
			requires std::default_initializable<T> && std::default_initializable<Op>
			: dynamic_segment_tree(0, T{}, Op{}) {}

		/**
		 * @brief `[0, n)`, every element `identity`; only the null sentinel is allocated.
		 * @complexity O(1)
		 */
		constexpr dynamic_segment_tree(std::size_t n, T identity, Op op = Op{})
			: op_(std::move(op)), element_identity_(std::move(identity)), n_(n), nodes_(1, node{element_identity_, 0, 0, 0, false}) {}

		/** @brief Number of elements. @complexity O(1) */
		constexpr std::size_t size() const noexcept { return n_; }

		/** @brief Whether the domain contains no elements. @complexity O(1) */
		constexpr bool empty() const noexcept { return n_ == 0; }

		/** @brief Node-pool entries, including the null sentinel. @complexity O(1) */
		constexpr std::size_t node_count() const noexcept { return nodes_.size(); }

		/**
		 * @brief Assign @p x to element @p i. @pre `i < size()`
		 * @complexity O(log n) time and new nodes in the worst case
		 */
		constexpr void set(std::size_t i, T x) {
			atl_assert(i < n_);
			root_ = point_update(root_, 0, n_, i, [&x](T const &) { return std::move(x); });
		}

		/**
		 * @brief Replace element @p i with `f(current)` — one descent; an untouched
		 * position reads as `identity`. The callable runs exactly once. @pre `i < size()` @complexity O(log n)
		 */
		template<update_function<T> Update>
		constexpr void update(std::size_t i, Update &&f) {
			atl_assert(i < n_);
			root_ = point_update(root_, 0, n_, i, f);
		}

		/**
		 * @brief Fold of `[l, r)`, or `identity` if empty — never allocates.
		 * @pre `l <= r <= size()` @complexity O(log n)
		 */
		[[nodiscard]] constexpr T fold(std::size_t l, std::size_t r) const {
			atl_assert(l <= r && r <= n_);
			if (l == r) return element_identity_;
			return fold(root_, 0, n_, l, r);
		}

		/** @brief Fold of the whole range. @complexity O(1) */
		[[nodiscard]] constexpr T const &fold() const { return nodes_[root_].aggregate; }

		/** @brief The value at @p i. @pre `i < size()` @complexity O(log n) */
		[[nodiscard]] constexpr T const &at(std::size_t i) const {
			atl_assert(i < n_);
			uint32 x = root_;
			std::size_t lo = 0, hi = n_;
			while (x != 0) {
				node const &nd = nodes_[x];
				if (nd.is_pending) return nd.pos == i ? nd.aggregate : element_identity_;
				if (hi - lo == 1) return nd.aggregate;
				std::size_t const mid = lo + (hi - lo) / 2;
				if (i < mid) {
					x = nd.left;
					hi = mid;
				} else {
					x = nd.right;
					lo = mid;
				}
			}
			return element_identity_; // untouched subtree
		}

		/**
		 * @brief Largest `r` in `[l, size()]` with `pred(fold(l, r))` true. Requires
		 * `pred` monotone in `r` and `pred(identity)`; untouched positions are
		 * skipped without being visited. Const and allocation-free.
		 * @pre `l <= size()` @complexity O(log n)
		 */
		template<std::predicate<T const &> Pred>
		[[nodiscard]] constexpr std::size_t max_right(std::size_t l, Pred pred) const {
			atl_assert(l <= n_ && std::invoke(pred, element_identity_));
			T sm = element_identity_;
			std::size_t const b = max_right_descend(root_, 0, n_, l, sm, pred);
			return b == npos ? n_ : b;
		}

		/**
		 * @brief Smallest `l` in `[0, r]` with `pred(fold(l, r))` true. Requires `pred`
		 * monotone in `l` and `pred(identity)`. Const and allocation-free.
		 * @pre `r <= size()` @complexity O(log n)
		 */
		template<std::predicate<T const &> Pred>
		[[nodiscard]] constexpr std::size_t min_left(std::size_t r, Pred pred) const {
			atl_assert(r <= n_ && std::invoke(pred, element_identity_));
			T sm = element_identity_;
			return min_left_descend(root_, 0, n_, r, sm, pred);
		}

		/**
		 * @brief Reserve node capacity for @p ops further point mutations (
		 * three entries per operation plus 64 entries of headroom;
		 * this is a capacity hint, not a bound on allocations).
		 * @pre `node_count() + 3 * ops + 64` fits `size_t`.
		 */
		constexpr void reserve(std::size_t ops) {
			constexpr auto limit = (std::numeric_limits<std::size_t>::max)();
			atl_assert(nodes_.size() <= limit - 64);
			atl_assert(ops <= (limit - nodes_.size() - 64) / 3);
			nodes_.reserve(nodes_.size() + ops * 3 + 64);
		}
	};

	/** @brief Deduce stored element and callable types from the constructor arguments. */
	template<typename T, binary_operation<T> Op>
	dynamic_segment_tree(std::size_t, T, Op) -> dynamic_segment_tree<T, Op>;
}
