#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/segment_tree_base.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/utils/assert.hpp>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Lazy segment tree: range apply and range fold in O(log n).
	 *
	 * Holds `n` values of `Value` and supports applying a `Tag` to every element of a
	 * half-open range and folding any range with @p op — both O(log n). The algebra
	 * is passed as plain callables and identities; `Value` and `Tag` are copyable:
	 *  - `(Value, op, identity)` is the element monoid (as in @ref atl::segment_tree);
	 *  - `(Tag, compose, tag_identity)` is the tag monoid, `compose(f, g)` meaning
	 *    "`g` first, then `f`" (order matters for non-commutative tags like assign);
	 *  - `mapping(f, s)` applies a tag to an element aggregate, and must satisfy
	 *    `mapping(f, op(a, b)) == op(mapping(f, a), mapping(f, b))` — tags act on
	 *    folds the same as on elements. When a tag's effect depends on the segment
	 *    width (e.g. add-to-sum), carry the width inside `Value` (see the docs).
	 *    Mapping satisfies
	 *    `mapping(compose(f, g), x) == mapping(f, mapping(g, x))`.
	 *    It need not preserve the element identity: valid updates never tag padding.
	 *
	 * Iterative implementation over the shared `detail::segment_tree_base`.
	 *
	 * @tparam Value   The element type.
	 * @tparam Tag     The tag type applied to ranges.
	 * @tparam Op      Element fold: any `op(Value, Value)` convertible to `Value`.
	 * @tparam Mapping Tag application: any `mapping(Tag, Value)` convertible to `Value`.
	 * @tparam Compose Tag composition: any `compose(Tag, Tag)` convertible to `Tag`.
	 */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping, binary_operation<Tag> Compose>
	class lazy_segment_tree : detail::segment_tree_base<Value> {
		using base = detail::segment_tree_base<Value>;
		using base::ascend_boundaries;
		using base::build;
		using base::decompose;
		using base::descend_boundaries;
		using base::leaf;
		using base::log_;
		using base::n_;
		using base::nodes_;
		using base::recompute_up;
		using base::reset;
		using base::size_;

		[[no_unique_address]] Op op_;
		[[no_unique_address]] Mapping mapping_;
		[[no_unique_address]] Compose compose_;
		Value element_identity_;
		Tag tag_identity_;
		std::vector<Tag> pending_tags_; /**< pending tag per internal node: applied to the node
											 itself already, still owed to its subtree */

		/** @brief Fold @p f into node @p k: update its aggregate now, owe it to the subtree. */
		constexpr void apply_tag(std::size_t k, Tag const &f) {
			nodes_[k] = std::invoke(std::as_const(mapping_), f, nodes_[k]);
			if (k < size_) pending_tags_[k] = std::invoke(std::as_const(compose_), f, pending_tags_[k]);
		}

		/**
		 * @brief Settle node @p k's pending tag onto its two children.
		 *
		 * An identity tag is skipped when `Tag` is equality-comparable: read paths push
		 * the shared upper ancestors on every call, mostly with nothing pending, and the
		 * skip saves the stores (affine mod p, 10% apply / 90% fold, n = 1e6: 890 → 804 ms
		 * plain -O2, 938 → 840 ms -march=native; add/sum ties; x86-64 Xeon 8358,
		 * GCC 15.2, RelWithDebInfo, benchmarks/ds/lazy_segment_tree.cpp, 2026-10-15).
		 */
		constexpr void push_down(std::size_t k) {
			if constexpr (std::equality_comparable<Tag>)
				if (pending_tags_[k] == tag_identity_) return;
			apply_tag(2 * k, pending_tags_[k]);
			apply_tag(2 * k + 1, pending_tags_[k]);
			pending_tags_[k] = tag_identity_;
		}

		/** @brief Settle every pending tag above leaf node @p k, so its value is current. */
		constexpr void push_to_leaf(std::size_t k) {
			for (std::size_t i = log_; i >= 1; --i) push_down(k >> i);
		}

	public:
		/** @brief Stored element or aggregate type. */
		using value_type = Value;
		/** @brief Range action type. */
		using tag_type = Tag;
		/** @brief Type of positions and element counts. */
		using size_type = std::size_t;

		/**
		 * @brief An empty tree with value-initialized identities and callables — a
		 *        placeholder for a member or global assigned a real tree later.
		 * @complexity O(1)
		 */
		constexpr lazy_segment_tree()
			requires std::default_initializable<Value> && std::default_initializable<Tag> && std::default_initializable<Op>
				  && std::default_initializable<Mapping> && std::default_initializable<Compose>
			: lazy_segment_tree(0, Value{}, Tag{}) {}

		/** @brief @p n elements, all equal to @p identity. @complexity O(n) */
		constexpr lazy_segment_tree(std::size_t n, Value identity, Tag tag_identity, Op op = Op{}, Mapping mapping = Mapping{},
									Compose compose = Compose{})
			: op_(std::move(op)),
			  mapping_(std::move(mapping)),
			  compose_(std::move(compose)),
			  element_identity_(std::move(identity)),
			  tag_identity_(std::move(tag_identity)) {
			reset(n, element_identity_);
			pending_tags_.assign(size_, tag_identity_);
		}

		/**
		 * @brief From initial values. Elements are moved out only of an rvalue range that
		 * owns them (not a borrowed range or a view), copied otherwise — see
		 * `detail::forward_element`. `Value` is non-deduced here so class deduction
		 * goes through the guide below, which rejects narrowing. @complexity O(n)
		 */
		template<std::ranges::sized_range R>
			requires requires { static_cast<Value>(std::declval<detail::forwarded_element_t<R>>()); }
		constexpr lazy_segment_tree(R &&values, std::type_identity_t<Value> identity, Tag tag_identity, Op op = Op{},
									Mapping mapping = Mapping{}, Compose compose = Compose{})
			: op_(std::move(op)),
			  mapping_(std::move(mapping)),
			  compose_(std::move(compose)),
			  element_identity_(std::move(identity)),
			  tag_identity_(std::move(tag_identity)) {
			reset(std::ranges::size(values), element_identity_);
			pending_tags_.assign(size_, tag_identity_);
			std::size_t i = 0;
			for (auto &&v : values) nodes_[leaf(i++)] = static_cast<Value>(detail::forward_element<R>(v));
			build(op_);
		}

		/** @brief Assign @p x to position @p i. @pre `i < size()` @complexity O(log n) */
		constexpr void set(std::size_t i, Value x) {
			atl_assert(i < n_);
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			nodes_[k] = std::move(x);
			recompute_up(k, op_);
		}

		/**
		 * @brief The value at position @p i (a reference into the tree, valid until the
		 * next mutation). Settles the tags above `i`, hence not const.
		 * @pre `i < size()` @complexity O(log n)
		 */
		[[nodiscard]] constexpr Value const &at(std::size_t i) {
			atl_assert(i < n_);
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			return nodes_[k];
		}

		/**
		 * @brief Apply tag @p f to the single element at @p i. @pre `i < size()`
		 * @complexity O(log n)
		 */
		constexpr void apply(std::size_t i, Tag const &f) {
			atl_assert(i < n_);
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			nodes_[k] = std::invoke(std::as_const(mapping_), f, nodes_[k]);
			recompute_up(k, op_);
		}

		/**
		 * @brief Apply tag @p f to every element of `[l, r)`. @pre `l <= r <= size()`
		 * @complexity O(log n)
		 */
		constexpr void apply(std::size_t l, std::size_t r, Tag const &f) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return;
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(
				l, r, [&](std::size_t k) { apply_tag(k, f); }, [&](std::size_t k) { apply_tag(k, f); });
			ascend_boundaries(l, r, [&](std::size_t k) { nodes_[k] = std::invoke(std::as_const(op_), nodes_[2 * k], nodes_[2 * k + 1]); });
		}

		/**
		 * @brief Fold of `[l, r)`, or @p identity if empty. Settles boundary tags,
		 * hence not const. @pre `l <= r <= size()` @complexity O(log n)
		 */
		[[nodiscard]] constexpr Value fold(std::size_t l, std::size_t r) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return element_identity_;
			descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			Value sl = element_identity_, sr = element_identity_;
			decompose(
				l, r, [&](std::size_t k) { sl = std::invoke(std::as_const(op_), sl, nodes_[k]); },
				[&](std::size_t k) { sr = std::invoke(std::as_const(op_), nodes_[k], sr); });
			return std::invoke(std::as_const(op_), sl, sr);
		}

		/**
		 * @brief Fold of the whole range `[0, size())`. The root aggregate is always
		 * current (a node's own value already includes its pending tag), so this
		 * is const. @complexity O(1)
		 */
		[[nodiscard]] constexpr Value const &fold() const { return nodes_[1]; }

		/**
		 * @brief Largest `r` in `[l, size()]` with `pred(fold(l, r))` true. Requires
		 * `pred` monotone in `r` and `pred(identity)`. @pre `l <= size()`
		 * @complexity O(log n)
		 */
		template<std::predicate<Value const &> Pred>
		[[nodiscard]] constexpr std::size_t max_right(std::size_t l, Pred pred) {
			atl_assert(l <= n_ && std::invoke(pred, element_identity_));
			if (l == n_) return n_;
			l += size_;
			for (std::size_t i = log_; i >= 1; --i) push_down(l >> i);
			Value sm = element_identity_;
			do {
				while (l % 2 == 0) l >>= 1;
				if (!std::invoke(pred, std::invoke(std::as_const(op_), sm, nodes_[l]))) {
					while (l < size_) {
						push_down(l);
						l *= 2;
						if (Value t = std::invoke(std::as_const(op_), sm, nodes_[l]); std::invoke(pred, t)) sm = std::move(t), ++l;
					}
					return l - size_;
				}
				sm = std::invoke(std::as_const(op_), sm, nodes_[l]);
				++l;
			} while (!std::has_single_bit(l));
			return n_;
		}

		/**
		 * @brief Smallest `l` in `[0, r]` with `pred(fold(l, r))` true. Requires `pred`
		 * monotone in `l` and `pred(identity)`. @pre `r <= size()`
		 * @complexity O(log n)
		 */
		template<std::predicate<Value const &> Pred>
		[[nodiscard]] constexpr std::size_t min_left(std::size_t r, Pred pred) {
			atl_assert(r <= n_ && std::invoke(pred, element_identity_));
			if (r == 0) return 0;
			r += size_;
			for (std::size_t i = log_; i >= 1; --i) push_down((r - 1) >> i);
			Value sm = element_identity_;
			do {
				--r;
				while (r > 1 && (r & 1)) r >>= 1;
				if (!std::invoke(pred, std::invoke(std::as_const(op_), nodes_[r], sm))) {
					while (r < size_) {
						push_down(r);
						r = 2 * r + 1;
						if (Value t = std::invoke(std::as_const(op_), nodes_[r], sm); std::invoke(pred, t)) sm = std::move(t), --r;
					}
					return r + 1 - size_;
				}
				sm = std::invoke(std::as_const(op_), nodes_[r], sm);
			} while (!std::has_single_bit(r));
			return 0;
		}

		/** @brief Number of elements. @complexity O(1) */
		constexpr std::size_t size() const noexcept { return n_; }

		/** @brief Whether the domain contains no elements. @complexity O(1) */
		constexpr bool empty() const noexcept { return n_ == 0; }
	};

	/** @brief Deduce stored element and callable types from the constructor arguments. */
	template<typename Value, typename Tag, binary_operation<Value> Op, action<Tag, Value> Mapping, binary_operation<Tag> Compose>
	lazy_segment_tree(std::size_t, Value, Tag, Op, Mapping, Compose) -> lazy_segment_tree<Value, Tag, Op, Mapping, Compose>;
	/**
	 * @brief Deduces `Value` from the element identity; rejected if that would narrow the
	 * range's values — spell the class arguments explicitly if the conversion is
	 * intended.
	 */
	template<std::ranges::sized_range R, typename Value, typename Tag, binary_operation<Value> Op, action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose>
		requires std::convertible_to<std::ranges::range_value_t<R>, Value> && requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	lazy_segment_tree(R &&, Value, Tag, Op, Mapping, Compose) -> lazy_segment_tree<Value, Tag, Op, Mapping, Compose>;
}
