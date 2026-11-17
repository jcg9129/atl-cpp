#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/segment_tree_base.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Dual segment tree: range apply and point get in O(log n).
	 *
	 * The tags-only member of the segment-tree family — the tool for
	 * update-range / query-point. Holds `n` values of `Value` untouched in a flat
	 * array plus one pending `Tag` per tree node; applying a tag to a range
	 * composes it onto the O(log n) canonical nodes, and reading a position folds
	 * the tags on its root-to-leaf path into the stored value. There are no
	 * aggregates: no element operation, no pull-up pass, and `at` is `const`
	 * (tags are folded on the fly, never settled).
	 *
	 * The algebra is two callables and one identity:
	 *  - `(Tag, compose, tag_identity)` is the tag monoid, `compose(f, g)` meaning
	 *    "`g` first, then `f`" (order matters for non-commutative tags);
	 *  - `mapping(f, s)` applies a tag to a **single element**. Unlike
	 *    @ref atl::lazy_segment_tree, no distributivity law is required — tags never
	 *    act on aggregates here. It must satisfy `mapping(tag_identity, x) == x`
	 *    and `mapping(compose(f, g), x) == mapping(f, mapping(g, x))`.
	 *
	 * Invariant (default mode): along any root-to-leaf path, deeper tags are older —
	 * maintained by pushing the partially-covered boundary ancestors before each
	 * range apply. With `TagOrder = commutative_tags_t` (the caller promises
	 * commuting tags, unchecked) the invariant does not hold and is not needed: the
	 * order on a path is irrelevant, so both applies skip those pushes and only
	 * compose onto the canonical nodes (or the leaf).
	 *
	 * @tparam Value   The element type.
	 * @tparam Tag     The tag type applied to ranges.
	 * @tparam Mapping Tag application: any `mapping(Tag, Value)` convertible to `Value`.
	 * @tparam Compose Tag composition: any `compose(Tag, Tag)` convertible to `Tag`.
	 * @tparam TagOrder `void`, or @ref atl::commutative_tags_t for the commutative mode.
	 */
	template<std::copyable Value, std::copyable Tag, action<Tag, Value> Mapping, binary_operation<Tag> Compose,
			 tag_order TagOrder = void>
	class dual_segment_tree : detail::segment_tree_base<Tag> {
		using base = detail::segment_tree_base<Tag>;
		using base::decompose;
		using base::descend_boundaries;
		using base::leaf;
		using base::log_;
		using base::n_;
		using base::nodes_; ///< nodes_[k] = pending tag of tree node k
		using base::reset;
		using base::size_;

		[[no_unique_address]] Mapping mapping_;
		[[no_unique_address]] Compose compose_;
		Tag tag_identity_;
		std::vector<Value> values_; /**< the underlying elements, untouched by tags */

		static constexpr bool is_commutative = std::same_as<TagOrder, commutative_tags_t>;

		/** @brief Settle node @p k's tag onto its two children (tag-onto-tag only). */
		constexpr void push_down(std::size_t k) {
			nodes_[2 * k] = std::invoke(std::as_const(compose_), nodes_[k], nodes_[2 * k]);
			nodes_[2 * k + 1] = std::invoke(std::as_const(compose_), nodes_[k], nodes_[2 * k + 1]);
			nodes_[k] = tag_identity_;
		}

		/** @brief Settle every pending tag strictly above leaf node @p k. */
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
		 * @brief An empty tree with a value-initialized tag identity and callables —
		 *        a placeholder for a member or global assigned a real tree later.
		 * @complexity O(1)
		 */
		constexpr dual_segment_tree()
			requires std::default_initializable<Value> && std::default_initializable<Tag>
				  && std::default_initializable<Mapping> && std::default_initializable<Compose>
			: dual_segment_tree(0, Value{}, Tag{}) {}

		/** @brief @p n elements, all equal to @p fill. @complexity O(n) */
		constexpr dual_segment_tree(std::size_t n, Value fill, Tag tag_identity, Mapping mapping = Mapping{},
									Compose compose = Compose{})
			: mapping_(std::move(mapping)), compose_(std::move(compose)), tag_identity_(std::move(tag_identity)) {
			reset(n, tag_identity_);
			values_.assign(n, std::move(fill));
		}

		/**
		 * @brief From initial values. Elements are moved out only of an rvalue range that
		 * owns them (not a borrowed range or a view), copied otherwise — see
		 * `detail::forward_element`. @complexity O(n)
		 */
		template<std::ranges::sized_range R>
			requires requires { static_cast<Value>(std::declval<detail::forwarded_element_t<R>>()); }
		constexpr dual_segment_tree(R &&values, Tag tag_identity, Mapping mapping = Mapping{}, Compose compose = Compose{})
			: mapping_(std::move(mapping)), compose_(std::move(compose)), tag_identity_(std::move(tag_identity)) {
			reset(std::ranges::size(values), tag_identity_);
			values_.reserve(n_);
			for (auto &&v : values) values_.push_back(static_cast<Value>(detail::forward_element<R>(v)));
		}

		/** @brief @p n elements, all equal to @p fill, in the commutative mode. @complexity O(n) */
		constexpr dual_segment_tree(commutative_tags_t, std::size_t n, Value fill, Tag tag_identity,
									Mapping mapping = Mapping{}, Compose compose = Compose{})
			requires is_commutative
			: dual_segment_tree(n, std::move(fill), std::move(tag_identity), std::move(mapping), std::move(compose)) {}

		/** @brief From initial values, in the commutative mode. @complexity O(n) */
		template<std::ranges::sized_range R>
			requires is_commutative && requires { static_cast<Value>(std::declval<detail::forwarded_element_t<R>>()); }
		constexpr dual_segment_tree(commutative_tags_t, R &&values, Tag tag_identity, Mapping mapping = Mapping{},
									Compose compose = Compose{})
			: dual_segment_tree(std::forward<R>(values), std::move(tag_identity), std::move(mapping), std::move(compose)) {}

		/**
		 * @brief Apply tag @p f to every element of `[l, r)`.
		 * @pre `l <= r <= size()`
		 * @complexity O(log n)
		 */
		constexpr void apply(std::size_t l, std::size_t r, Tag const &f) {
			atl_assert(l <= r && r <= n_);
			if (l == r) return;
			if constexpr (!is_commutative) descend_boundaries(l, r, [&](std::size_t k) { push_down(k); });
			decompose(
				l, r, [&](std::size_t k) { nodes_[k] = std::invoke(std::as_const(compose_), f, nodes_[k]); },
				[&](std::size_t k) { nodes_[k] = std::invoke(std::as_const(compose_), f, nodes_[k]); });
		}

		/**
		 * @brief Apply tag @p f to the single element at @p i.
		 * @pre `i < size()`
		 * @complexity O(log n)
		 */
		constexpr void apply(std::size_t i, Tag const &f) {
			atl_assert(i < n_);
			std::size_t const k = leaf(i);
			if constexpr (!is_commutative) push_to_leaf(k);
			nodes_[k] = std::invoke(std::as_const(compose_), f, nodes_[k]);
		}

		/**
		 * @brief The value at position @p i, with every pending tag on its path folded
		 *        in on the fly — read-only.
		 * @pre `i < size()`
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr Value at(std::size_t i) const {
			atl_assert(i < n_);
			Value v = values_[i];
			for (std::size_t k = leaf(i); k >= 1; k >>= 1) v = std::invoke(std::as_const(mapping_), nodes_[k], v); // deepest first: oldest first in the default mode, any order when commutative
			return v;
		}

		/**
		 * @brief Assign @p x to position @p i, settling the path tags first.
		 * @pre `i < size()`
		 * @complexity O(log n)
		 */
		constexpr void set(std::size_t i, Value x) {
			atl_assert(i < n_);
			std::size_t const k = leaf(i);
			push_to_leaf(k);
			nodes_[k] = tag_identity_;
			values_[i] = std::move(x);
		}

		/**
		 * @brief Every element, with all pending tags folded in — one top-down pass
		 *        that carries each node's composed tag to its children, without
		 *        settling (so const, unlike n calls of `at` it walks each tag once).
		 * @complexity O(n) time, O(size) scratch tags
		 */
		[[nodiscard]] constexpr std::vector<Value> values() const {
			std::vector<Value> out = values_;
			if (n_ == 0) return out;
			// acc[k] = composition of the tags on the path root..k (deeper applied first)
			std::vector<Tag> acc(nodes_);
			for (std::size_t k = 2; k < 2 * size_; ++k) acc[k] = std::invoke(std::as_const(compose_), acc[k >> 1], acc[k]);
			for (std::size_t i = 0; i < n_; ++i) out[i] = std::invoke(std::as_const(mapping_), acc[leaf(i)], std::move(out[i]));
			return out;
		}

		/** @brief Number of elements. @complexity O(1) */
		constexpr std::size_t size() const noexcept { return n_; }

		/** @brief Whether the domain contains no elements. @complexity O(1) */
		constexpr bool empty() const noexcept { return n_ == 0; }
	};

	/** @brief Deduce stored element and callable types from the constructor arguments. */
	template<std::copyable Value, std::copyable Tag, action<Tag, Value> Mapping, binary_operation<Tag> Compose>
	dual_segment_tree(std::size_t, Value, Tag, Mapping, Compose) -> dual_segment_tree<Value, Tag, Mapping, Compose>;
	/**
	 * @brief `Value` is deduced from the range's elements (there is no separate element
	 * identity to conflict with, so no narrowing is possible).
	 */
	template<std::ranges::sized_range R, std::copyable Tag, action<Tag, std::ranges::range_value_t<R>> Mapping,
			 binary_operation<Tag> Compose>
	dual_segment_tree(R &&, Tag, Mapping, Compose) -> dual_segment_tree<std::ranges::range_value_t<R>, Tag, Mapping, Compose>;
	/** @brief The commutative mode, deduced from the leading @ref atl::commutative_tags argument. */
	template<std::copyable Value, std::copyable Tag, action<Tag, Value> Mapping, binary_operation<Tag> Compose>
	dual_segment_tree(commutative_tags_t, std::size_t, Value, Tag, Mapping, Compose)
		-> dual_segment_tree<Value, Tag, Mapping, Compose, commutative_tags_t>;
	/** @brief The commutative mode from a range; `Value` deduced as in the untagged guide. */
	template<std::ranges::sized_range R, std::copyable Tag, action<Tag, std::ranges::range_value_t<R>> Mapping,
			 binary_operation<Tag> Compose>
	dual_segment_tree(commutative_tags_t, R &&, Tag, Mapping, Compose)
		-> dual_segment_tree<std::ranges::range_value_t<R>, Tag, Mapping, Compose, commutative_tags_t>;
}
