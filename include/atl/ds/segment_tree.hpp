#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/segment_tree_base.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/utils/assert.hpp>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>

namespace atl {
	/**
	 * @brief Segment tree over a monoid: point assignment and range fold in O(log n).
	 *
	 * Holds `n` values and supports assigning one element and folding any half-open
	 * range `[l, r)` with an associative operation @p op whose neutral element is
	 * @p identity — `(T, op, identity)` must form a monoid. @p op need not be
	 * commutative (matrix product, string concatenation, monoid operations all work); `T` must be copyable.
	 *
	 * @tparam T The element type.
	 * @tparam Op A binary operation on `T` — any `op(T, T)` convertible to `T` (functor,
	 *            function pointer, lambda), enforced by @ref atl::binary_operation.
	 */
	template<std::copyable T, binary_operation<T> Op>
	class segment_tree : detail::segment_tree_base<T> {
		using base = detail::segment_tree_base<T>;
		using base::build;
		using base::decompose;
		using base::leaf;
		using base::n_;
		using base::nodes_;
		using base::recompute_up;
		using base::reset;
		using base::size_;

		[[no_unique_address]] Op op_;
		T identity_;

		/**
		 * @brief Whether the max_right/min_left ascent keeps the fold it tested and
		 *        moves it into the accumulator, instead of recomputing it.
		 *
		 * Pays off for aggregate `T`: sum + 2x2 matrix 592 → 371 ms plain, 594 → 365 ms
		 * -march=native. For a scalar the kept fold measured slower (gcd 209 → 226 ms
		 * -march=native), so scalars keep the recomputing form.
		 * x86-64 Xeon 8280, GCC 15.2, RelWithDebInfo, benchmarks/ds/segment_tree.cpp, 2026-10-15.
		 */
		constexpr static bool reuse_ascent_fold = !std::is_scalar_v<T>;

	public:
		/** @brief Stored element or aggregate type. */
		using value_type = T;
		/** @brief Type of positions and element counts. */
		using size_type = std::size_t;

		/**
		 * @brief An empty tree with a value-initialized identity and operation — a
		 *        placeholder for a member or global assigned a real tree later.
		 * @complexity O(1)
		 */
		constexpr segment_tree()
			requires std::default_initializable<T> && std::default_initializable<Op>
			: segment_tree(0, T{}) {}

		/** @brief @p n elements, all equal to @p identity. @complexity O(n) */
		constexpr segment_tree(std::size_t n, T identity, Op op = Op{}) : op_(std::move(op)), identity_(std::move(identity)) {
			reset(n, identity_);
		}

		/**
		 * @brief From initial values. Elements are moved out only of an rvalue range that
		 * owns them (not a borrowed range or a view), copied otherwise — see
		 * `detail::forward_element`. `T` is non-deduced here so class deduction
		 * goes through the guide below, which rejects narrowing. @complexity O(n)
		 */
		template<std::ranges::sized_range R>
			requires requires { static_cast<T>(std::declval<detail::forwarded_element_t<R>>()); }
		constexpr segment_tree(R &&values, std::type_identity_t<T> identity, Op op = Op{})
			: op_(std::move(op)), identity_(std::move(identity)) {
			reset(std::ranges::size(values), identity_);
			std::size_t i = 0;
			for (auto &&v : values) nodes_[leaf(i++)] = static_cast<T>(detail::forward_element<R>(v));
			build(op_);
		}

		/** @brief Assign @p x to position @p i. @pre `i < size()` @complexity O(log n) */
		constexpr void set(std::size_t i, T x) {
			atl_assert(i < n_);
			nodes_[leaf(i)] = std::move(x);
			recompute_up(leaf(i), op_);
		}

		/**
		 * @brief Replace element @p i with `f(element)`, invoked once on the stored value
		 * (no copy out as `set(i, f(at(i)))` would make). @pre `i < size()`
		 * @complexity O(log n)
		 */
		template<update_function<T> Update>
		constexpr void update(std::size_t i, Update &&f) {
			atl_assert(i < n_);
			T &x = nodes_[leaf(i)];
			x = static_cast<T>(std::invoke(f, std::as_const(x)));
			recompute_up(leaf(i), op_);
		}

		/**
		 * @brief The value at position @p i (a reference into the tree, reflecting subsequent updates to `i`). @pre `i < size()` @complexity O(1)
		 */
		[[nodiscard]] constexpr T const &at(std::size_t i) const {
			atl_assert(i < n_);
			return nodes_[leaf(i)];
		}

		/**
		 * @brief Fold of `[l, r)`, or @p identity if empty. @pre `l <= r <= size()`
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr T fold(std::size_t l, std::size_t r) const {
			atl_assert(l <= r && r <= n_);
			T sl = identity_, sr = identity_;
			decompose(
				l, r, [&](std::size_t k) { sl = std::invoke(std::as_const(op_), sl, nodes_[k]); },
				[&](std::size_t k) { sr = std::invoke(std::as_const(op_), nodes_[k], sr); });
			return std::invoke(std::as_const(op_), sl, sr);
		}

		/** @brief Fold of the whole range `[0, size())`. @complexity O(1) */
		[[nodiscard]] constexpr T const &fold() const { return nodes_[1]; }

		/**
		 * @brief Invoke `f(node_value)` on each of the O(log n) canonical nodes tiling
		 * `[l, r)`, left to right — the decompose-then-aggregate building block.
		 * @pre `l <= r <= size()` @complexity O(log n) calls
		 */
		template<std::invocable<T const &> F>
		constexpr void for_each_segment(std::size_t l, std::size_t r, F f) const {
			atl_assert(l <= r && r <= n_);
			std::array<std::size_t, std::numeric_limits<std::size_t>::digits> right;
			auto *end = right.data();
			decompose(
				l, r, [&](std::size_t k) { std::invoke(f, nodes_[k]); }, [&](std::size_t k) { *end++ = k; });
			while (end != right.data()) std::invoke(f, nodes_[*--end]);
		}

		/**
		 * @brief Largest `r` in `[l, size()]` with `pred(fold(l, r))` true. Requires `pred`
		 * monotone in `r` and `pred(identity)`. @pre `l <= size()` @complexity O(log n)
		 */
		template<std::predicate<T const &> Pred>
		[[nodiscard]] constexpr std::size_t max_right(std::size_t l, Pred pred) const {
			atl_assert(l <= n_ && std::invoke(pred, identity_));
			if (l == n_) return n_;
			l += size_;
			T sm = identity_;
			do {
				while (l % 2 == 0) l >>= 1;
				if constexpr (reuse_ascent_fold) {
					T t = std::invoke(std::as_const(op_), sm, nodes_[l]);
					if (!std::invoke(pred, t)) {
						while (l < size_) {
							l *= 2;
							if (t = std::invoke(std::as_const(op_), sm, nodes_[l]); std::invoke(pred, t)) sm = std::move(t), ++l;
						}
						return l - size_;
					}
					sm = std::move(t);
				} else {
					if (!std::invoke(pred, std::invoke(std::as_const(op_), sm, nodes_[l]))) {
						while (l < size_) {
							l *= 2;
							if (T t = std::invoke(std::as_const(op_), sm, nodes_[l]); std::invoke(pred, t)) sm = std::move(t), ++l;
						}
						return l - size_;
					}
					sm = std::invoke(std::as_const(op_), sm, nodes_[l]);
				}
				++l;
			} while (!std::has_single_bit(l));
			return n_;
		}

		/**
		 * @brief Smallest `l` in `[0, r]` with `pred(fold(l, r))` true. Requires `pred`
		 * monotone in `l` and `pred(identity)`. @pre `r <= size()` @complexity O(log n)
		 */
		template<std::predicate<T const &> Pred>
		[[nodiscard]] constexpr std::size_t min_left(std::size_t r, Pred pred) const {
			atl_assert(r <= n_ && std::invoke(pred, identity_));
			if (r == 0) return 0;
			r += size_;
			T sm = identity_;
			do {
				--r;
				while (r > 1 && (r & 1)) r >>= 1;
				if constexpr (reuse_ascent_fold) {
					T t = std::invoke(std::as_const(op_), nodes_[r], sm);
					if (!std::invoke(pred, t)) {
						while (r < size_) {
							r = 2 * r + 1;
							if (t = std::invoke(std::as_const(op_), nodes_[r], sm); std::invoke(pred, t)) sm = std::move(t), --r;
						}
						return r + 1 - size_;
					}
					sm = std::move(t);
				} else {
					if (!std::invoke(pred, std::invoke(std::as_const(op_), nodes_[r], sm))) {
						while (r < size_) {
							r = 2 * r + 1;
							if (T t = std::invoke(std::as_const(op_), nodes_[r], sm); std::invoke(pred, t)) sm = std::move(t), --r;
						}
						return r + 1 - size_;
					}
					sm = std::invoke(std::as_const(op_), nodes_[r], sm);
				}
			} while (!std::has_single_bit(r));
			return 0;
		}

		/** @brief Number of elements. @complexity O(1) */
		constexpr std::size_t size() const noexcept { return n_; }

		/** @brief Whether the domain contains no elements. @complexity O(1) */
		constexpr bool empty() const noexcept { return n_ == 0; }
	};

	/** @brief Deduce stored element and callable types from the constructor arguments. */
	template<typename T, binary_operation<T> Op>
	segment_tree(std::size_t, T, Op) -> segment_tree<T, Op>;
	/**
	 * @brief Deduces `T` from the identity; rejected if that would narrow the range's
	 * values (e.g. `long long` elements with an `int` identity) — spell
	 * `segment_tree<T, Op>` explicitly if the conversion is intended.
	 */
	template<std::ranges::sized_range R, typename T, binary_operation<T> Op>
		requires std::convertible_to<std::ranges::range_value_t<R>, T> && requires(std::ranges::range_value_t<R> v) { T{std::move(v)}; }
	segment_tree(R &&, T, Op) -> segment_tree<T, Op>;
}
