#pragma once

/**
 * @file
 * @brief Contender for @ref atl::persistent_lazy_segment_tree — copy only
 *        the links and the tag of a partially covered node.
 */

#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/persistent_arena.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <bit>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental {
	/**
	 * @brief The library @ref atl::persistent_lazy_segment_tree trimmed to `apply`,
	 *        `set`, `fold`, `at` and the constructors, with the node copy of the
	 *        partially covered levels as a knob.
	 *
	 * The library `apply`, `apply_permanent` and `rebuild` copy the whole node out of
	 * the arena (`Value` aggregate included) before they know whether the node is
	 * covered, a leaf or partial. At a partial level the aggregate is never read
	 * (it is recomputed from the new children), so for a `Value` of more than a word
	 * the copy is wasted traffic on every level of every mutation. With
	 * `LeanPartialCopy` a partial level copies only the two links and the tag, and the
	 * tag is moved into the new node; the aggregate is read only where it is used
	 * (a covered node, a leaf). `LeanPartialCopy = false` is the library behaviour.
	 *
	 * @note Not constexpr (versions share a std::shared_ptr pool). Partial copies
	 * measured 471 / 474 ms versus whole-node 472 / 492 on branching add n=1e5;
	 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-11-01,
	 * -O2 / -O2 -march=native; benchmarks/ds/persistent_lazy_segment_tree.cpp.
	 * The narrow and wide workloads show mixed results, so this remains experimental.
	 * @tparam LeanPartialCopy Copy only links + tag at partial levels (the contender).
	 *
	 * Other template parameters and the algebra as in @ref atl::persistent_lazy_segment_tree.
	 */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose, tag_order TagOrder = void, bool LeanPartialCopy = true>
	class persistent_lazy_segment_tree_by_partial_copy {
		static constexpr bool is_commutative = std::same_as<TagOrder, commutative_tags_t>;

		struct node {
			Value aggregate;
			Tag tag;
			uint32 left, right;
		};

		/** @brief What a partial level reads of a node when it is lean: no aggregate. */
		struct links {
			Tag tag;
			uint32 left, right;
		};

		using partial_copy = std::conditional_t<LeanPartialCopy, links, node>;

		struct pool {
			[[no_unique_address]] Op op;
			[[no_unique_address]] Mapping mapping;
			[[no_unique_address]] Compose compose;
			Value element_identity;
			Tag tag_identity;
			std::size_t n = 0;
			detail::persistent_arena<node> nodes;

			pool(Op o, Mapping m, Compose c, Value eid, Tag tid)
				: op(std::move(o)), mapping(std::move(m)), compose(std::move(c)), element_identity(std::move(eid)),
				  tag_identity(std::move(tid)), nodes(node{element_identity, tag_identity, 0, 0}) {}

			uint32 make(Value aggregate, Tag tag, uint32 left, uint32 right) {
				return nodes.make({std::move(aggregate), std::move(tag), left, right});
			}

			/** @brief The part of node @p x a partial level needs (copied out per the arena's reallocation contract). */
			partial_copy read_partial(uint32 x) const {
				if constexpr (LeanPartialCopy)
					return {nodes[x].tag, nodes[x].left, nodes[x].right};
				else
					return nodes[x];
			}

			uint32 with_tag(uint32 x, Tag const &f) {
				node nd = nodes[x];
				return make(mapping(f, nd.aggregate), compose(f, nd.tag), nd.left, nd.right);
			}

			uint32 build(std::vector<Value> &buf, std::size_t lo, std::size_t hi) {
				if (hi - lo == 1) return make(std::move(buf[lo]), tag_identity, 0, 0);
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 const l = build(buf, lo, mid), r = build(buf, mid, hi);
				return make(op(nodes[l].aggregate, nodes[r].aggregate), tag_identity, l, r);
			}

			uint32 apply(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, Tag const &f) {
				if (r <= lo || hi <= l) return x;
				if (l <= lo && hi <= r) {
					node const nd = nodes[x];
					return make(mapping(f, nd.aggregate), compose(f, nd.tag), nd.left, nd.right);
				}
				partial_copy const nd = read_partial(x);
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 lc = with_tag(nd.left, nd.tag), rc = with_tag(nd.right, nd.tag);
				lc = apply(lc, lo, mid, l, r, f);
				rc = apply(rc, mid, hi, l, r, f);
				return make(op(nodes[lc].aggregate, nodes[rc].aggregate), tag_identity, lc, rc);
			}

			uint32 apply_permanent(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, Tag const &f) {
				if (l <= lo && hi <= r) {
					node const nd = nodes[x];
					return make(mapping(f, nd.aggregate), compose(f, nd.tag), nd.left, nd.right);
				}
				partial_copy nd = read_partial(x);
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 const lc = l < mid ? apply_permanent(nd.left, lo, mid, l, r, f) : nd.left;
				uint32 const rc = mid < r ? apply_permanent(nd.right, mid, hi, l, r, f) : nd.right;
				// the mapping reads nd.tag, so evaluate it before the tag is moved into the node
				Value aggregate = mapping(nd.tag, op(nodes[lc].aggregate, nodes[rc].aggregate));
				return make(std::move(aggregate), std::move(nd.tag), lc, rc);
			}

			uint32 set(uint32 x, std::size_t lo, std::size_t hi, std::size_t i, Value &value) {
				if (hi - lo == 1) return make(std::move(value), tag_identity, 0, 0);
				partial_copy const nd = read_partial(x);
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 lc = with_tag(nd.left, nd.tag), rc = with_tag(nd.right, nd.tag);
				if (i < mid)
					lc = set(lc, lo, mid, i, value);
				else
					rc = set(rc, mid, hi, i, value);
				return make(op(nodes[lc].aggregate, nodes[rc].aggregate), tag_identity, lc, rc);
			}

			Value fold(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, Tag const &acc) const {
				if (r <= lo || hi <= l) return element_identity;
				node const &nd = nodes[x];
				if (l <= lo && hi <= r) return mapping(acc, nd.aggregate);
				Tag const acc2 = compose(acc, nd.tag);
				std::size_t const mid = lo + (hi - lo) / 2;
				return op(fold(nd.left, lo, mid, l, r, acc2), fold(nd.right, mid, hi, l, r, acc2));
			}
		};

		std::shared_ptr<pool> pool_;
		uint32 root_ = 0;

		persistent_lazy_segment_tree_by_partial_copy(std::shared_ptr<pool> shared, uint32 root)
			: pool_(std::move(shared)), root_(root) {}

		persistent_lazy_segment_tree_by_partial_copy apply_in(std::shared_ptr<pool> shared, std::size_t l, std::size_t r,
															  Tag const &f) const {
			atl_assert(l <= r && r <= shared->n);
			if (l == r) return {std::move(shared), root_};
			uint32 next;
			if constexpr (is_commutative)
				next = shared->apply_permanent(root_, 0, shared->n, l, r, f);
			else
				next = shared->apply(root_, 0, shared->n, l, r, f);
			return {std::move(shared), next};
		}

		persistent_lazy_segment_tree_by_partial_copy set_in(std::shared_ptr<pool> shared, std::size_t i, Value x) const {
			atl_assert(i < shared->n);
			auto const next = shared->set(root_, 0, shared->n, i, x);
			return {std::move(shared), next};
		}

	public:
		using value_type = Value;
		using tag_type = Tag;
		using size_type = std::size_t;

		/** @brief From initial values, copied. @complexity O(n) */
		template<std::ranges::sized_range R>
			requires std::convertible_to<std::ranges::range_value_t<R>, Value>
		persistent_lazy_segment_tree_by_partial_copy(R &&values, std::type_identity_t<Value> identity, Tag tag_identity,
													 Op op = Op{}, Mapping mapping = Mapping{}, Compose compose = Compose{})
			: pool_(std::make_shared<pool>(std::move(op), std::move(mapping), std::move(compose), std::move(identity),
										   std::move(tag_identity))) {
			pool_->n = std::ranges::size(values);
			if (pool_->n > 0) {
				std::vector<Value> buf;
				buf.reserve(pool_->n);
				for (auto &&v : values) buf.push_back(static_cast<Value>(detail::forward_element<R>(v)));
				root_ = pool_->build(buf, 0, pool_->n);
			}
		}

		std::size_t size() const noexcept { return pool_->n; }

		/** @brief The version with tag @p f applied to every element of `[l, r)`. @pre `l <= r <= size()` */
		[[nodiscard]] persistent_lazy_segment_tree_by_partial_copy apply(std::size_t l, std::size_t r, Tag const &f) const & {
			return apply_in(pool_, l, r, f);
		}

		/** @brief `apply` consuming this version (the pool is moved, not copied). */
		[[nodiscard]] persistent_lazy_segment_tree_by_partial_copy apply(std::size_t l, std::size_t r, Tag const &f) && {
			return apply_in(std::move(pool_), l, r, f);
		}

		/** @brief The version with element @p i replaced by @p x. @pre `i < size()` */
		[[nodiscard]] persistent_lazy_segment_tree_by_partial_copy set(std::size_t i, Value x) const & {
			return set_in(pool_, i, std::move(x));
		}

		/** @brief `set` consuming this version (the pool is moved, not copied). */
		[[nodiscard]] persistent_lazy_segment_tree_by_partial_copy set(std::size_t i, Value x) && {
			return set_in(std::move(pool_), i, std::move(x));
		}

		/** @brief Fold of `[l, r)`, or the identity if empty. @pre `l <= r <= size()` */
		[[nodiscard]] Value fold(std::size_t l, std::size_t r) const {
			atl_assert(l <= r && r <= size());
			if (l == r) return pool_->element_identity;
			return pool_->fold(root_, 0, size(), l, r, pool_->tag_identity);
		}

		/** @brief Number of nodes in the shared arena. */
		std::size_t arena_size() const noexcept { return pool_->nodes.size(); }

		/** @brief Reserve arena room for @p ops further mutations (the library class's per-level bound). */
		void reserve(std::size_t ops) {
			std::size_t const levels = size() > 1 ? std::size_t(std::bit_width(size() - 1)) : 1;
			std::size_t const per_level = is_commutative ? 4 : 8;
			atl_assert(ops <= (std::numeric_limits<std::size_t>::max)() / per_level / (levels + 1));
			pool_->nodes.reserve_extra(ops * per_level * (levels + 1));
		}
	};
}
