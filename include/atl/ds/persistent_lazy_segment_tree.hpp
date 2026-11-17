#pragma once
#include <atl/ds/detail/algebra.hpp>
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

namespace atl {
	/**
	 * @brief Fully persistent lazy segment tree: range apply, range fold, and
	 *        cross-version range copy — every version kept.
	 *
	 * Mutators return a **new** version sharing all untouched structure with the
	 * old one (path copying); any past version stays queryable and mutable, so
	 * histories may branch freely. Versions derived from one origin share a single
	 * arena (a pool of nodes addressed by uint32 ids); a version is a cheap
	 * `shared_ptr` + id, freed when the last version referencing the arena dies.
	 *
	 * The algebra is the @ref atl::lazy_segment_tree contract, verbatim: element monoid
	 * `(Value, op, identity)`, tag monoid `(Tag, compose, tag_identity)` with
	 * `compose(f, g)` = "`g` first, then `f`", and an action `mapping(f, s)`
	 * satisfying `mapping(f, op(a, b)) == op(mapping(f, a), mapping(f, b))` (carry
	 * the width inside `Value` when a tag's effect depends on it).
	 *
	 * Reads never write: @ref fold / @ref at descend accumulating the pending
	 * path tags instead of settling them, so they are const and allocation-free.
	 * @ref copy_range adopts whole subtrees of the source version by id, making
	 * "paste `[l, r)` from another version" O(log n) — the operation that makes
	 * two live versions interact.
	 *
	 * With `TagOrder = commutative_tags_t` (the caller promises commuting tags,
	 * unchecked) @ref apply keeps tags permanent: a partially covered node is
	 * copied once, keeping its tag, and the descent copies only the nodes it
	 * visits — no settled child copies and no `compose` on the way down. Reads
	 * are unchanged (they already fold the path tags); @ref set, @ref update and
	 * @ref copy_range still settle the tags on their path.
	 *
	 * @note Not constexpr (versions own a std::shared_ptr pool). This applies to all members.
	 * @pre User arithmetic must be representable; at most 2^31 initial elements.
	 * @tparam Value   The element type.
	 * @tparam Tag     The tag type applied to ranges.
	 * @tparam Op      Element fold: any `op(Value, Value)` convertible to `Value`.
	 * @tparam Mapping Tag application: any `mapping(Tag, Value)` convertible to `Value`.
	 * @tparam Compose Tag composition: any `compose(Tag, Tag)` convertible to `Tag`.
	 * @tparam TagOrder `void`, or @ref atl::commutative_tags_t for the commutative mode.
	 */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping, binary_operation<Tag> Compose,
			 tag_order TagOrder = void>
	class persistent_lazy_segment_tree {
		static constexpr bool is_commutative = std::same_as<TagOrder, commutative_tags_t>;

		/**
		 * @brief A node's `aggregate` already includes its own tag; `tag` is pending for the
		 * subtree below only (in the commutative mode it may stay there for good:
		 * the children's aggregates then exclude it). Exact midpoint intervals over
		 * [0, n) — no padding.
		 */
		struct node {
			Value aggregate;
			Tag tag;
			uint32 left, right;
		};

		struct pool {
			[[no_unique_address]] Op op;
			[[no_unique_address]] Mapping mapping;
			[[no_unique_address]] Compose compose;
			Value element_identity;
			Tag tag_identity;
			std::size_t n = 0;
			detail::persistent_arena<node> nodes;

			pool(Op o, Mapping m, Compose c, Value eid, Tag tid)
				: op(detail::move_or_copy(o)), mapping(detail::move_or_copy(m)), compose(detail::move_or_copy(c)), element_identity(std::move(eid)),
				  tag_identity(std::move(tid)), nodes(node{element_identity, tag_identity, 0, 0}) {}

			uint32 make(Value aggregate, Tag tag, uint32 left, uint32 right) {
				return nodes.make({std::move(aggregate), std::move(tag), left, right});
			}

			/**
			 * @brief A copy of @p x with @p f folded in: applied to its aggregate now,
			 * composed onto its pending tag for the subtree below.
			 */
			uint32 with_tag(uint32 x, Tag const &f) {
				node nd = nodes[x]; // copied out per the arena's reallocation contract
				return make(mapping(f, nd.aggregate), compose(f, nd.tag), nd.left, nd.right);
			}

			uint32 build(std::vector<Value> &buf, std::size_t lo, std::size_t hi) {
				if (hi - lo == 1) return make(std::move(buf[lo]), tag_identity, 0, 0);
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 const l = build(buf, lo, mid), r = build(buf, mid, hi);
				return make(op(nodes[l].aggregate, nodes[r].aggregate), tag_identity, l, r);
			}

			uint32 apply(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, Tag const &f) {
				if (r <= lo || hi <= l) return x; // untouched: shared as-is
				node const nd = nodes[x];
				if (l <= lo && hi <= r) return make(mapping(f, nd.aggregate), compose(f, nd.tag), nd.left, nd.right);
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 lc = with_tag(nd.left, nd.tag), rc = with_tag(nd.right, nd.tag);
				lc = apply(lc, lo, mid, l, r, f);
				rc = apply(rc, mid, hi, l, r, f);
				return make(op(nodes[lc].aggregate, nodes[rc].aggregate), tag_identity, lc, rc);
			}

			/**
			 * @brief `apply` for commuting tags: a partial node keeps its tag, which still
			 * covers the whole subtree, so it is folded back over the new children's
			 * aggregates; untouched children are shared, never copied.
			 */
			uint32 apply_permanent(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, Tag const &f) {
				node const nd = nodes[x];
				if (l <= lo && hi <= r) return make(mapping(f, nd.aggregate), compose(f, nd.tag), nd.left, nd.right);
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 const lc = l < mid ? apply_permanent(nd.left, lo, mid, l, r, f) : nd.left;
				uint32 const rc = mid < r ? apply_permanent(nd.right, mid, hi, l, r, f) : nd.right;
				return make(mapping(nd.tag, op(nodes[lc].aggregate, nodes[rc].aggregate)), nd.tag, lc, rc);
			}

			/**
			 * @brief Path copy settling the pending tags on the way down, with the leaf's
			 * value produced by @p leaf from its current (fully tagged) value.
			 */
			template<update_function<Value> Leaf>
			uint32 rebuild(uint32 x, std::size_t lo, std::size_t hi, std::size_t i, Leaf &&leaf) {
				node const nd = nodes[x]; // copied out per the arena's reallocation contract
				if (hi - lo == 1) return make(leaf(nd.aggregate), tag_identity, 0, 0);
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 lc = with_tag(nd.left, nd.tag), rc = with_tag(nd.right, nd.tag);
				if (i < mid)
					lc = rebuild(lc, lo, mid, i, leaf);
				else
					rc = rebuild(rc, mid, hi, i, leaf);
				return make(op(nodes[lc].aggregate, nodes[rc].aggregate), tag_identity, lc, rc);
			}

			/**
			 * @brief Fold of `[l, r)` under @p acc, the composition of the pending tags
			 * on the path above @p x (newest outermost). Pure — no allocation.
			 */
			Value fold(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r, Tag const &acc) const {
				if (r <= lo || hi <= l) return element_identity;
				node const &nd = nodes[x];
				if (l <= lo && hi <= r) return mapping(acc, nd.aggregate);
				Tag const acc2 = compose(acc, nd.tag);
				std::size_t const mid = lo + (hi - lo) / 2;
				return op(fold(nd.left, lo, mid, l, r, acc2), fold(nd.right, mid, hi, l, r, acc2));
			}

			/**
			 * @brief A tree equal to @p x outside `[l, r)` and to @p y inside — subtrees
			 * fully on one side are adopted whole, with that side's accumulated
			 * path tag folded into the adopted root (children shared untouched).
			 */
			uint32 copy_range(uint32 x, Tag const &accx, uint32 y, Tag const &accy, std::size_t lo, std::size_t hi,
							  std::size_t l, std::size_t r) {
				if (r <= lo || hi <= l) return with_tag(x, accx);
				if (l <= lo && hi <= r) return with_tag(y, accy);
				Tag const accx2 = compose(accx, nodes[x].tag), accy2 = compose(accy, nodes[y].tag);
				uint32 const xl = nodes[x].left, xr = nodes[x].right;
				uint32 const yl = nodes[y].left, yr = nodes[y].right;
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 const lc = copy_range(xl, accx2, yl, accy2, lo, mid, l, r);
				uint32 const rc = copy_range(xr, accx2, yr, accy2, mid, hi, l, r);
				return make(op(nodes[lc].aggregate, nodes[rc].aggregate), tag_identity, lc, rc);
			}
		};

		std::shared_ptr<pool> pool_;
		uint32 root_ = 0;

		persistent_lazy_segment_tree(std::shared_ptr<pool> shared, uint32 root)
			: pool_(std::move(shared)), root_(root) {}

		/** @brief `apply` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_lazy_segment_tree apply_in(std::shared_ptr<pool> shared, std::size_t l, std::size_t r, Tag const &f) const {
			atl_assert(l <= r && r <= shared->n);
			if (l == r) return {std::move(shared), root_};
			uint32 next;
			if constexpr (is_commutative)
				next = shared->apply_permanent(root_, 0, shared->n, l, r, f);
			else
				next = shared->apply(root_, 0, shared->n, l, r, f);
			return {std::move(shared), next};
		}

		/** @brief `set` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_lazy_segment_tree set_in(std::shared_ptr<pool> shared, std::size_t i, Value x) const {
			atl_assert(i < shared->n);
			auto const next = shared->rebuild(root_, 0, shared->n, i, [&x](Value const &) { return std::move(x); });
			return {std::move(shared), next};
		}

		/** @brief `update` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		template<update_function<Value> Update>
		persistent_lazy_segment_tree update_in(std::shared_ptr<pool> shared, std::size_t i, Update &&f) const {
			atl_assert(i < shared->n);
			auto const next = shared->rebuild(root_, 0, shared->n, i, f);
			return {std::move(shared), next};
		}

		/** @brief `copy_range` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_lazy_segment_tree copy_range_in(std::shared_ptr<pool> shared, persistent_lazy_segment_tree const &source, std::size_t l, std::size_t r) const {
			atl_assert(shared == source.pool_ && l <= r && r <= shared->n);
			if (l == r) return {std::move(shared), root_};
			auto const next = shared->copy_range(root_, shared->tag_identity, source.root_, shared->tag_identity, 0, shared->n, l, r);
			return {std::move(shared), next};
		}

	public:
		/** @brief Element type. */
		using value_type = Value;
		/** @brief Lazy tag type. */
		using tag_type = Tag;
		/** @brief Type of element counts and positions. */
		using size_type = std::size_t;

		/**
		 * @brief An empty placeholder tree (`size() == 0`, identities `Value{}` and
		 *        `Tag{}`), for a member or global assigned a real tree later.
		 */
		persistent_lazy_segment_tree()
			requires std::default_initializable<Value> && std::default_initializable<Tag> && std::default_initializable<Op>
				  && std::default_initializable<Mapping> && std::default_initializable<Compose>
			: persistent_lazy_segment_tree(0, Value{}, Tag{}) {}

		/** @brief @p n elements, all equal to @p identity. @complexity O(n) */
		persistent_lazy_segment_tree(std::size_t n, Value identity, Tag tag_identity, Op op = Op{},
									 Mapping mapping = Mapping{}, Compose compose = Compose{})
			: pool_(std::make_shared<pool>(detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose),
										   std::move(identity), std::move(tag_identity))) {
			atl_assert(n <= (std::numeric_limits<uint32>::max)() / 2 + 1);
			pool_->n = n;
			if (n > 0) {
				std::vector<Value> buf(n, pool_->element_identity);
				root_ = pool_->build(buf, 0, n);
			}
		}

		/**
		 * @brief From initial values. Elements are moved out only of an owning rvalue
		 * range (not a `borrowed_range`, not a view), copied otherwise — see
		 * the element-forwarding rule (owning rvalues and yielded prvalues may move). `Value` is non-deduced here so class
		 * deduction goes through the guide below, which rejects narrowing.
		 * @complexity O(n)
		 */
		template<std::ranges::sized_range R>
			requires std::convertible_to<std::ranges::range_value_t<R>, Value>
				  && requires { static_cast<Value>(std::declval<detail::forwarded_element_t<R>>()); }
		persistent_lazy_segment_tree(R &&values, std::type_identity_t<Value> identity, Tag tag_identity, Op op = Op{},
									 Mapping mapping = Mapping{}, Compose compose = Compose{})
			: pool_(std::make_shared<pool>(detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose),
										   std::move(identity), std::move(tag_identity))) {
			atl_assert(std::ranges::size(values) <= (std::numeric_limits<uint32>::max)() / 2 + 1);
			pool_->n = std::ranges::size(values);
			if (pool_->n > 0) {
				std::vector<Value> buf;
				buf.reserve(pool_->n);
				for (auto &&v : values) buf.push_back(static_cast<Value>(detail::forward_element<R>(v)));
				root_ = pool_->build(buf, 0, pool_->n);
			}
		}

		/** @brief @p n elements, all equal to @p identity, in the commutative mode. @complexity O(n) */
		persistent_lazy_segment_tree(commutative_tags_t, std::size_t n, Value identity, Tag tag_identity, Op op = Op{},
									 Mapping mapping = Mapping{}, Compose compose = Compose{})
			requires is_commutative
			: persistent_lazy_segment_tree(n, std::move(identity), std::move(tag_identity), detail::move_or_copy(op),
										   detail::move_or_copy(mapping), detail::move_or_copy(compose)) {}

		/** @brief From initial values, in the commutative mode. @complexity O(n) */
		template<std::ranges::sized_range R>
			requires is_commutative && std::convertible_to<std::ranges::range_value_t<R>, Value>
				  && requires { static_cast<Value>(std::declval<detail::forwarded_element_t<R>>()); }
		persistent_lazy_segment_tree(commutative_tags_t, R &&values, std::type_identity_t<Value> identity,
									 Tag tag_identity, Op op = Op{}, Mapping mapping = Mapping{},
									 Compose compose = Compose{})
			: persistent_lazy_segment_tree(std::forward<R>(values), std::move(identity), std::move(tag_identity),
										   detail::move_or_copy(op), detail::move_or_copy(mapping), detail::move_or_copy(compose)) {}

		/** @brief Number of elements in this version. @complexity O(1) */
		std::size_t size() const noexcept { return pool_->n; }

		/** @brief Whether this version contains no elements. @complexity O(1) */
		bool empty() const noexcept { return size() == 0; }

		/**
		 * @brief The version with tag @p f applied to every element of `[l, r)`.
		 * @pre `l <= r <= size()` @complexity O(log n) time and new nodes
		 */
		[[nodiscard]] persistent_lazy_segment_tree apply(std::size_t l, std::size_t r, Tag const &f) const & { return apply_in(pool_, l, r, f); }

		/** @brief `apply` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `t = std::move(t).apply(...)` is the cheap linear idiom. The moved-from tree may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		[[nodiscard]] persistent_lazy_segment_tree apply(std::size_t l, std::size_t r, Tag const &f) && {
			return apply_in(std::move(pool_), l, r, f);
		}

		/**
		 * @brief The version with element `i` replaced by @p x.
		 * @pre `i < size()` @complexity O(log n) time and new nodes
		 */
		[[nodiscard]] persistent_lazy_segment_tree set(std::size_t i, Value x) const & { return set_in(pool_, i, std::move(x)); }

		/** @brief `set` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `t = std::move(t).set(...)` is the cheap linear idiom. The moved-from tree may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		[[nodiscard]] persistent_lazy_segment_tree set(std::size_t i, Value x) && {
			return set_in(std::move(pool_), i, std::move(x));
		}

		/**
		 * @brief The version with element @p i replaced by `f(current)` — one
		 *        descent where `set(i, f(at(i)))` would take two.
		 * @pre `i < size()` @complexity O(log n) time and new nodes
		 */
		template<update_function<Value> Update>
		[[nodiscard]] persistent_lazy_segment_tree update(std::size_t i, Update &&f) const & { return update_in(pool_, i, std::forward<Update>(f)); }

		/** @brief `update` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `t = std::move(t).update(...)` is the cheap linear idiom. The moved-from tree may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		template<update_function<Value> Update>
		[[nodiscard]] persistent_lazy_segment_tree update(std::size_t i, Update &&f) && {
			return update_in(std::move(pool_), i, std::forward<Update>(f));
		}

		/**
		 * @brief The version equal to this one outside `[l, r)` and to @p source
		 *        inside — "paste `[l, r)` from another version". Subtrees fully
		 *        inside (or outside) are adopted whole, so this is O(log n).
		 * @pre Both versions share one arena (derived from the same origin) and
		 *      `l <= r <= size()`.
		 */
		[[nodiscard]] persistent_lazy_segment_tree copy_range(persistent_lazy_segment_tree const &source, std::size_t l, std::size_t r) const & { return copy_range_in(pool_, source, l, r); }

		/** @brief `copy_range` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `t = std::move(t).copy_range(...)` is the cheap linear idiom. The moved-from tree may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		[[nodiscard]] persistent_lazy_segment_tree copy_range(persistent_lazy_segment_tree const &source, std::size_t l, std::size_t r) && {
			if (&source == this) return copy_range_in(pool_, source, l, r);
			return copy_range_in(std::move(pool_), source, l, r);
		}

		/**
		 * @brief Fold of `[l, r)` in this version, or @p identity if empty.
		 *        Const and allocation-free (pending tags are accumulated, never
		 *        settled). @pre `l <= r <= size()` @complexity O(log n)
		 */
		[[nodiscard]] Value fold(std::size_t l, std::size_t r) const {
			atl_assert(l <= r && r <= size());
			if (l == r) return pool_->element_identity;
			return pool_->fold(root_, 0, size(), l, r, pool_->tag_identity);
		}

		/** @brief Fold of the whole range `[0, size())`. @complexity O(1) */
		[[nodiscard]] Value fold() const { return empty() ? pool_->element_identity : pool_->nodes[root_].aggregate; }

		/** @brief The value at position @p i. @pre `i < size()` @complexity O(log n) */
		[[nodiscard]] Value at(std::size_t i) const {
			atl_assert(i < size());
			// one root-to-leaf walk composing the path tags, one mapping at the leaf:
			// no op calls and no identity copies, unlike fold(i, i + 1)
			auto const &nodes = pool_->nodes;
			Tag acc = pool_->tag_identity;
			uint32 x = root_;
			for (std::size_t lo = 0, hi = size(); hi - lo > 1;) {
				node const &nd = nodes[x];
				acc = pool_->compose(acc, nd.tag);
				std::size_t const mid = lo + (hi - lo) / 2;
				if (i < mid)
					x = nd.left, hi = mid;
				else
					x = nd.right, lo = mid;
			}
			return pool_->mapping(acc, nodes[x].aggregate);
		}

		/**
		 * @brief Number of nodes in the shared arena — all versions' total structural
		 * footprint (diagnostics / capacity planning).
		 */
		std::size_t arena_size() const noexcept { return pool_ ? pool_->nodes.size() : 0; }

		/**
		 * @brief Reserve arena capacity for @p ops further mutations, avoiding
		 *        reallocation spikes mid-run (each mutation creates O(log n) nodes).
		 *
		 * Sized by the mode's per-level bound for any one mutation: 8 nodes per level
		 * in the default mode (an `apply` settles both boundary nodes into four child
		 * copies, then rebuilds or retags up to four of them), 4 in the commutative
		 * mode (`apply` allocates one node per visited node). Either bound covers
		 * every mutation: `copy_range` allocates at most 4 per level, `set` and
		 * `update` at most 3 (the rebuilt node and its two settled children). Past
		 * the reserve the arena grows geometrically, as always.
		 */
		void reserve(std::size_t ops) {
			std::size_t const levels = size() > 1 ? std::size_t(std::bit_width(size() - 1)) : 1;
			// default apply: <= 2 partial nodes per level settle into 4 child copies,
			// and up to 4 of those children are then rebuilt (partial) or retagged
			// (covered), 8 per level; copy_range <= 4, set/update <= 3 in both modes;
			// commutative apply: <= 4 visited nodes per level, one new node each
			std::size_t const per_level = is_commutative ? 4 : 8;
			atl_assert(ops <= (std::numeric_limits<std::size_t>::max)() / per_level / (levels + 1));
			pool_->nodes.reserve_extra(ops * per_level * (levels + 1));
		}
	};

	/** @brief Deduce the container types from constructor arguments and their algebra. */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping, binary_operation<Tag> Compose>
	persistent_lazy_segment_tree(std::size_t, Value, Tag, Op, Mapping, Compose)
		-> persistent_lazy_segment_tree<Value, Tag, Op, Mapping, Compose>;
	/**
	 * @brief Deduces `Value` from the element identity; rejected if that would narrow the
	 * range's values — spell the class arguments explicitly if the conversion is
	 * intended.
	 */
	template<std::ranges::sized_range R, std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose>
		requires requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	persistent_lazy_segment_tree(R &&, Value, Tag, Op, Mapping, Compose)
		-> persistent_lazy_segment_tree<Value, Tag, Op, Mapping, Compose>;
	/** @brief The commutative mode, deduced from the leading @ref atl::commutative_tags argument. */
	/** @brief Deduce the container types from constructor arguments and their algebra. */
	template<std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping, binary_operation<Tag> Compose>
	persistent_lazy_segment_tree(commutative_tags_t, std::size_t, Value, Tag, Op, Mapping, Compose)
		-> persistent_lazy_segment_tree<Value, Tag, Op, Mapping, Compose, commutative_tags_t>;
	/** @brief The commutative mode from a range, with the untagged range guide's narrowing check. */
	template<std::ranges::sized_range R, std::copyable Value, std::copyable Tag, binary_operation<Value> Op, action<Tag, Value> Mapping,
			 binary_operation<Tag> Compose>
		requires requires(std::ranges::range_value_t<R> v) { Value{std::move(v)}; }
	persistent_lazy_segment_tree(commutative_tags_t, R &&, Value, Tag, Op, Mapping, Compose)
		-> persistent_lazy_segment_tree<Value, Tag, Op, Mapping, Compose, commutative_tags_t>;
}
