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
	 * @brief Fully persistent segment tree: point assignment and range fold,
	 *        every version kept.
	 *
	 * The plain (tag-free) sibling of @ref atl::persistent_lazy_segment_tree — a node
	 * is just an aggregate and two child ids, so point mutations copy exactly
	 * one root-to-leaf path — the "Chairman tree" of competitive programming.
	 * Mutators return a **new** version sharing all untouched structure; any
	 * version stays queryable forever; a version is a cheap `shared_ptr` + id.
	 *
	 * All versions of one arena share a single fixed shape (exact midpoint
	 * intervals over `[0, n)`), which is what makes @ref descend possible:
	 * a lockstep root-to-leaf walk over *two* versions whose per-node folds
	 * correspond position-for-position — the primitive behind order statistics
	 * over a version difference (k-th smallest in `a[l..r)` from prefix
	 * versions, in a single O(log n) descent).
	 *
	 * @note Not constexpr (versions own a std::shared_ptr pool). This applies to all members.
	 * @pre The fold and identity form a monoid; user arithmetic must be representable.
	 * @tparam T  The element type.
	 * @tparam Op Element fold: any `op(T, T)` convertible to `T`.
	 */
	template<std::copyable T, binary_operation<T> Op>
	class persistent_segment_tree {
		struct node {
			T aggregate;
			uint32 left, right;
		};

		struct pool {
			[[no_unique_address]] Op op;
			T identity;
			std::size_t n = 0;
			detail::persistent_arena<node> nodes;

			pool(Op o, T id) : op(detail::move_or_copy(o)), identity(std::move(id)), nodes(node{identity, 0, 0}) {}

			uint32 make(T aggregate, uint32 left, uint32 right) { return nodes.make({std::move(aggregate), left, right}); }

			uint32 build(std::vector<T> &buf, std::size_t lo, std::size_t hi) {
				if (hi - lo == 1) return make(std::move(buf[lo]), 0, 0);
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 const l = build(buf, lo, mid), r = build(buf, mid, hi);
				return make(op(nodes[l].aggregate, nodes[r].aggregate), l, r);
			}

			/**
			 * @brief Path copy with the leaf's value produced by @p leaf from the old
			 * one — one new node per level, everything else shared.
			 */
			template<update_function<T> Leaf>
			uint32 rebuild(uint32 x, std::size_t lo, std::size_t hi, std::size_t i, Leaf &&leaf) {
				// the leaf's value is consumed before make runs (arguments are sequenced
				// before the call), and internal levels need only the child ids, so no
				// level copies an aggregate out (the arena's reallocation contract)
				if (hi - lo == 1) return make(leaf(std::as_const(nodes[x].aggregate)), 0, 0);
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 lc = nodes[x].left, rc = nodes[x].right;
				if (i < mid)
					lc = rebuild(lc, lo, mid, i, leaf);
				else
					rc = rebuild(rc, mid, hi, i, leaf);
				return make(op(nodes[lc].aggregate, nodes[rc].aggregate), lc, rc);
			}

			/**
			 * @brief A tree equal to @p x outside `[l, r)` and to @p y inside: subtrees
			 * fully on one side are adopted whole by id (there are no tags to
			 * carry), only the O(log n) straddling nodes are rebuilt.
			 */
			uint32 copy_range(uint32 x, uint32 y, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r) {
				if (r <= lo || hi <= l) return x;
				if (l <= lo && hi <= r) return y;
				uint32 const xl = nodes[x].left, xr = nodes[x].right;
				uint32 const yl = nodes[y].left, yr = nodes[y].right;
				std::size_t const mid = lo + (hi - lo) / 2;
				uint32 const lc = copy_range(xl, yl, lo, mid, l, r);
				uint32 const rc = copy_range(xr, yr, mid, hi, l, r);
				return make(op(nodes[lc].aggregate, nodes[rc].aggregate), lc, rc);
			}

			T fold(uint32 x, std::size_t lo, std::size_t hi, std::size_t l, std::size_t r) const {
				if (r <= lo || hi <= l) return identity;
				node const &nd = nodes[x];
				if (l <= lo && hi <= r) return nd.aggregate;
				std::size_t const mid = lo + (hi - lo) / 2;
				return op(fold(nd.left, lo, mid, l, r), fold(nd.right, mid, hi, l, r));
			}
		};

		std::shared_ptr<pool> pool_;
		uint32 root_ = 0;

		persistent_segment_tree(std::shared_ptr<pool> shared, uint32 root)
			: pool_(std::move(shared)), root_(root) {}

		/** @brief `set` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_segment_tree set_in(std::shared_ptr<pool> shared, std::size_t i, T x) const {
			atl_assert(i < shared->n);
			auto const next = shared->rebuild(root_, 0, shared->n, i, [&x](T const &) { return std::move(x); });
			return {std::move(shared), next};
		}

		/** @brief `update` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		template<update_function<T> Update>
		persistent_segment_tree update_in(std::shared_ptr<pool> shared, std::size_t i, Update &&f) const {
			atl_assert(i < shared->n);
			auto const next = shared->rebuild(root_, 0, shared->n, i, f);
			return {std::move(shared), next};
		}

		/** @brief `copy_range` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_segment_tree copy_range_in(std::shared_ptr<pool> shared, persistent_segment_tree const &source, std::size_t l, std::size_t r) const {
			atl_assert(shared == source.pool_ && l <= r && r <= shared->n);
			if (l == r) return {std::move(shared), root_};
			auto const next = shared->copy_range(root_, source.root_, 0, shared->n, l, r);
			return {std::move(shared), next};
		}

	public:
		/** @brief Element type. */
		using value_type = T;
		/** @brief Type of element counts and positions. */
		using size_type = std::size_t;

		/**
		 * @brief An empty placeholder tree (`size() == 0`, identity `T{}`), for a
		 *        member or global assigned a real tree later.
		 */
		persistent_segment_tree()
			requires std::default_initializable<T> && std::default_initializable<Op>
			: persistent_segment_tree(0, T{}) {}

		/** @brief @p n elements, all equal to @p identity. @complexity O(n) */
		persistent_segment_tree(std::size_t n, T identity, Op op = Op{})
			: pool_(std::make_shared<pool>(detail::move_or_copy(op), std::move(identity))) {
			atl_assert(n <= (std::numeric_limits<uint32>::max)() / 2 + 1);
			pool_->n = n;
			if (n > 0) {
				std::vector<T> buf(n, pool_->identity);
				root_ = pool_->build(buf, 0, n);
			}
		}

		/**
		 * @brief From initial values. Elements are moved out only of an owning rvalue
		 * range (not a `borrowed_range`, not a view), copied otherwise — see
		 * the element-forwarding rule (owning rvalues and yielded prvalues may move). `T` is non-deduced here so class
		 * deduction goes through the guide below, which rejects narrowing.
		 * @complexity O(n)
		 */
		template<std::ranges::sized_range R>
			requires std::convertible_to<std::ranges::range_value_t<R>, T>
				  && requires { static_cast<T>(std::declval<detail::forwarded_element_t<R>>()); }
		persistent_segment_tree(R &&values, std::type_identity_t<T> identity, Op op = Op{})
			: pool_(std::make_shared<pool>(detail::move_or_copy(op), std::move(identity))) {
			atl_assert(std::ranges::size(values) <= (std::numeric_limits<uint32>::max)() / 2 + 1);
			pool_->n = std::ranges::size(values);
			if (pool_->n > 0) {
				std::vector<T> buf;
				buf.reserve(pool_->n);
				for (auto &&v : values) buf.push_back(static_cast<T>(detail::forward_element<R>(v)));
				root_ = pool_->build(buf, 0, pool_->n);
			}
		}

		/** @brief Number of elements in this version. @complexity O(1) */
		std::size_t size() const noexcept { return pool_->n; }

		/** @brief Whether this version contains no elements. @complexity O(1) */
		bool empty() const noexcept { return size() == 0; }

		/**
		 * @brief The version with element @p i replaced by @p x.
		 * @pre `i < size()` @complexity O(log n) time and new nodes
		 */
		[[nodiscard]] persistent_segment_tree set(std::size_t i, T x) const & { return set_in(pool_, i, std::move(x)); }

		/** @brief `set` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `t = std::move(t).set(...)` is the cheap linear idiom. The moved-from tree may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		[[nodiscard]] persistent_segment_tree set(std::size_t i, T x) && {
			return set_in(std::move(pool_), i, std::move(x));
		}

		/**
		 * @brief The version with element @p i replaced by `f(current)` — one
		 *        descent where `set(i, f(at(i)))` would take two.
		 * @pre `i < size()` @complexity O(log n) time and new nodes
		 */
		template<update_function<T> Update>
		[[nodiscard]] persistent_segment_tree update(std::size_t i, Update &&f) const & { return update_in(pool_, i, std::forward<Update>(f)); }

		/** @brief `update` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `t = std::move(t).update(...)` is the cheap linear idiom. The moved-from tree may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		template<update_function<T> Update>
		[[nodiscard]] persistent_segment_tree update(std::size_t i, Update &&f) && {
			return update_in(std::move(pool_), i, std::forward<Update>(f));
		}

		/**
		 * @brief The version equal to this one outside `[l, r)` and to @p source
		 *        inside — "paste `[l, r)` from another version". Subtrees fully
		 *        inside (or outside) are adopted whole, so this is O(log n).
		 * @pre Both versions share one arena (derived from the same origin) and
		 *      `l <= r <= size()`.
		 * @complexity O(log n) time and new nodes
		 */
		[[nodiscard]] persistent_segment_tree copy_range(persistent_segment_tree const &source, std::size_t l, std::size_t r) const & { return copy_range_in(pool_, source, l, r); }

		/** @brief `copy_range` consuming this version: the pool is moved, not copied (no atomic refcount traffic), so `t = std::move(t).copy_range(...)` is the cheap linear idiom. The moved-from tree may only be destroyed or assigned to. Otherwise as the lvalue overload. */
		[[nodiscard]] persistent_segment_tree copy_range(persistent_segment_tree const &source, std::size_t l, std::size_t r) && {
			if (&source == this) return copy_range_in(pool_, source, l, r);
			return copy_range_in(std::move(pool_), source, l, r);
		}

		/**
		 * @brief Fold of `[l, r)` in this version, or `identity` if empty —
		 *        const and allocation-free. @pre `l <= r <= size()`
		 * @complexity O(log n)
		 */
		[[nodiscard]] T fold(std::size_t l, std::size_t r) const {
			atl_assert(l <= r && r <= size());
			if (l == r) return pool_->identity;
			return pool_->fold(root_, 0, size(), l, r);
		}

		/** @brief Fold of the whole range `[0, size())`. @complexity O(1) */
		[[nodiscard]] T fold() const { return empty() ? pool_->identity : pool_->nodes[root_].aggregate; }

		/**
		 * @brief The value at position @p i, by value: the arena is shared by every
		 *       version from the same origin, so a reference into it could dangle
		 *       after any later mutation of any of them.
		 * @pre `i < size()` @complexity O(log n)
		 */
		[[nodiscard]] T at(std::size_t i) const {
			atl_assert(i < size());
			uint32 x = root_;
			std::size_t lo = 0, hi = size();
			while (hi - lo > 1) {
				std::size_t const mid = lo + (hi - lo) / 2;
				if (node const &nd = pool_->nodes[x]; i < mid) {
					x = nd.left;
					hi = mid;
				} else {
					x = nd.right;
					lo = mid;
				}
			}
			return pool_->nodes[x].aggregate;
		}

		/**
		 * @brief Lockstep root-to-leaf descent over two versions: at each
		 *        internal node calls `choose(a_left, b_left)` with the two
		 *        versions' left-child folds; `true` descends left. Returns the
		 *        index of the leaf reached.
		 *
		 * Because all versions share one shape, the two folds describe the same
		 * index interval — so a stateful @p choose can binary-search a property
		 * of the version *difference* without any inverse operation. The classic
		 * instance is the Chairman tree's k-th smallest: with prefix count
		 * versions `v_0..v_n`, `descend(v_l, v_r, ...)` counts the slice's
		 * elements per side via `b_left - a_left` and steers toward the k-th.
		 * @pre Both versions share one arena and `!a.empty()`.
		 * @complexity O(log n), allocation-free
		 */
		template<typename Choose>
			requires std::convertible_to<std::invoke_result_t<Choose &, T const &, T const &>, bool>
		[[nodiscard]] static std::size_t descend(persistent_segment_tree const &a, persistent_segment_tree const &b, Choose &&choose) {
			atl_assert(a.pool_ && a.pool_ == b.pool_ && !a.empty());
			pool const &p = *a.pool_;
			uint32 x = a.root_, y = b.root_;
			std::size_t lo = 0, hi = p.n;
			while (hi - lo > 1) {
				node const &nx = p.nodes[x];
				node const &ny = p.nodes[y];
				std::size_t const mid = lo + (hi - lo) / 2;
				if (choose(p.nodes[nx.left].aggregate, p.nodes[ny.left].aggregate)) {
					x = nx.left;
					y = ny.left;
					hi = mid;
				} else {
					x = nx.right;
					y = ny.right;
					lo = mid;
				}
			}
			return lo;
		}

		/**
		 * @brief Number of nodes in the shared arena — all versions' total structural
		 * footprint (diagnostics / capacity planning).
		 */
		std::size_t arena_size() const noexcept { return pool_ ? pool_->nodes.size() : 0; }

		/**
		 * @brief Reserve arena capacity for @p ops further mutations, avoiding
		 *        reallocation spikes mid-run (each mutation creates one node per
		 *        level).
		 */
		void reserve(std::size_t ops) {
			std::size_t const levels = size() > 1 ? std::size_t(std::bit_width(size() - 1)) : 1;
			atl_assert(ops <= (std::numeric_limits<std::size_t>::max)() / (levels + 1));
			pool_->nodes.reserve_extra(ops * (levels + 1));
		}
	};

	/** @brief Deduce the container types from constructor arguments and their algebra. */
	template<std::copyable T, binary_operation<T> Op>
	persistent_segment_tree(std::size_t, T, Op) -> persistent_segment_tree<T, Op>;
	/**
	 * @brief Deduces `T` from the identity; rejected if that would narrow the range's
	 * values — spell the class arguments explicitly if the conversion is
	 * intended.
	 */
	template<std::ranges::sized_range R, std::copyable T, binary_operation<T> Op>
		requires requires(std::ranges::range_value_t<R> v) { T{std::move(v)}; }
	persistent_segment_tree(R &&, T, Op) -> persistent_segment_tree<T, Op>;
}
