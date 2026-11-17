#pragma once

/**
 * @file
 * @brief @ref atl::persistent_heap and @ref atl::persistent_heap_forest — fully
 *        persistent meldable priority queues (push, pop, meld and push_range
 *        return a new version and keep every old one), by path copying on
 *        randomized meldable heaps.
 */

#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/persistent_arena.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <ranges>
#include <utility>

namespace atl {
	namespace detail {
		/**
		 * @brief The node pool behind @ref atl::persistent_heap and @ref
		 *        persistent_heap_forest: heap-ordered binary nodes in a @ref
		 *        persistent_arena (id 0 the null node) and the order, with the
		 *        path-copying meld every operation reduces to.
		 *
		 * The meld is Gambin and Malinowski's randomized one: keep the
		 * comp-larger root and meld the other heap into a child picked by a fair
		 * coin. Its walk down any binary tree of n nodes has expected length at
		 * most `log2(n + 1)` whatever the tree's shape, so the bound holds for
		 * every version, old ones included: each operation flips fresh coins.
		 *
		 * Each operation passes `fresh`, the arena size at its start: a node at
		 * or past it was made by this operation, belongs to no version yet, and
		 * is changed in place instead of copied (the pushed leaf, or the
		 * array-built block of a `push_range`).
		 */
		template<std::semiregular T, typename Compare>
			requires std::strict_weak_order<Compare &, T const &, T const &>
		struct persistent_heap_pool {
			/** @brief A heap node: `child[0]` / `child[1]` are its children (0 for none). */
			struct node {
				T value;
				uint32 child[2];
			};

			/** @brief The nodes; id 0 is the null node. */
			persistent_arena<node> nodes{node{T{}, {0, 0}}};
			/** @brief The order: `comp(a, b)` holds when `a` is below `b`. */
			[[no_unique_address]] Compare comp;
			/** @brief The splitmix64 state the coins are drawn from. */
			uint64 random_state;
			/** @brief The unspent bits of the last coin word. */
			uint64 random_bits = 0;
			/** @brief How many bits of `random_bits` are unspent. */
			uint32 random_left = 0;

			/**
			 * @brief An empty pool; the coins are seeded per process (as @ref
			 *        treap_map's priorities), by a fixed seed during constant evaluation.
			 */
			constexpr explicit persistent_heap_pool(Compare order) : comp(std::move(order)), random_state(0) {
				if consteval {
					random_state = 0x2545f4914f6cdd1dULL;
				} else {
					random_state = hash_seed() ^ 0x9e3779b97f4a7c15ULL;
				}
			}

			/** @brief A fair coin, 0 or 1: one splitmix64 output spent a bit at a time. */
			constexpr uint32 coin() {
				if (random_left == 0) {
					random_bits = splitmix64(random_state += 0x9e3779b97f4a7c15ULL);
					random_left = 64;
				}
				--random_left;
				uint32 const bit = uint32(random_bits & 1);
				random_bits >>= 1;
				return bit;
			}

			/** @brief A fresh leaf holding `value`. */
			constexpr uint32 leaf(T &&value) { return nodes.make(node{std::move(value), {0, 0}}); }

			/**
			 * @brief The meld of the heaps rooted at `a` and `b` (0 = empty),
			 *        sharing every subtree off the walked path.
			 *
			 * Top-down: each step keeps the comp-larger root (copied unless
			 * fresh), hangs it where the preceding step left a hole, and continues
			 * into the child a coin picks, against the other heap.
			 */
			constexpr uint32 meld(uint32 a, uint32 b, uint32 fresh) {
				if (a == 0) return b;
				if (b == 0) return a;
				uint32 root = 0, parent = 0, side = 0;
				while (a != 0 && b != 0) {
					if (comp(nodes[a].value, nodes[b].value)) std::swap(a, b);
					uint32 const c = a < fresh ? nodes.make(nodes[a]) : a;
					if (parent == 0)
						root = c;
					else
						nodes[parent].child[side] = c;
					side = coin();
					a = nodes[c].child[side];
					parent = c;
				}
				nodes[parent].child[side] = a | b;
				return root;
			}

			/** @brief The heap rooted at `root` without its top. @pre `root != 0` */
			constexpr uint32 pop(uint32 root) {
				uint32 const fresh = node_count();
				return meld(nodes[root].child[0], nodes[root].child[1], fresh);
			}

			/** @brief The arena size as a node id. @pre fewer than 2^32 nodes (asserted) */
			constexpr uint32 node_count() const {
				atl_assert(nodes.size() < (std::numeric_limits<uint32>::max)());
				return uint32(nodes.size());
			}

			/**
			 * @brief Move a range into fresh consecutive nodes, then link its complete
			 *        binary tree and sift values down by holes, in O(n); 0 if empty.
			 */
			template<std::ranges::input_range R>
				requires std::convertible_to<forwarded_element_t<R>, T>
			constexpr uint32 heapify(R &&values) {
				uint32 const base = node_count();
				if constexpr (std::ranges::sized_range<R>) {
					atl_assert(std::ranges::size(values) < (std::numeric_limits<uint32>::max)() - base);
					nodes.reserve_extra(std::ranges::size(values));
				}
				for (auto &&v : values) {
					atl_assert(nodes.size() + 1 < (std::numeric_limits<uint32>::max)());
					nodes.make(node{static_cast<T>(forward_element<R>(std::forward<decltype(v)>(v))), {0, 0}});
				}
				uint32 const count = node_count() - base;
				if (count == 0) return 0;
				for (uint32 i = count; i > 0; --i) {
					uint32 const at = i - 1;
					uint32 const left = 2 * uint64(at) + 1 < count ? base + 2 * at + 1 : 0;
					uint32 const right = 2 * uint64(at) + 2 < count ? base + 2 * at + 2 : 0;
					nodes[base + at].child[0] = left;
					nodes[base + at].child[1] = right;
					if (left == 0) continue;
					T held = std::move(nodes[base + at].value);
					uint32 hole = base + at;
					for (;;) {
						uint32 child = nodes[hole].child[0];
						if (child == 0) break;
						uint32 const other = nodes[hole].child[1];
						if (other != 0 && comp(nodes[child].value, nodes[other].value)) child = other;
						if (!comp(held, nodes[child].value)) break;
						nodes[hole].value = std::move(nodes[child].value);
						hole = child;
					}
					nodes[hole].value = std::move(held);
				}
				return base;
			}

			/** @brief `heapify(values)` melded into the heap rooted at `root`. */
			template<std::ranges::input_range R>
				requires std::convertible_to<forwarded_element_t<R>, T>
			constexpr uint32 meld_values(uint32 root, R &&values) {
				uint32 const fresh = node_count();
				uint32 const built = heapify(std::forward<R>(values));
				return meld(root, built, fresh);
			}
		};
	}

	/**
	 * @brief A fully persistent meldable priority queue: `push`, `pop`, `meld`
	 *        and `push_range` return a new version and leave every old one
	 *        intact, at O(log n) expected new nodes per operation.
	 *
	 * A randomized meldable heap (a heap-ordered binary tree whose meld walks
	 * down a coin-picked path), so path copying changes the O(log n) expected
	 * nodes of that path and shares the rest. Every version derived from one
	 * heap shares its node pool (append-only: nodes are never freed while a
	 * version lives), so any two of them meld; a default-constructed heap
	 * starts a new pool. For many versions held as plain ids on one pool, see
	 * @ref atl::persistent_heap_forest.
	 *
	 * The order follows `std::priority_queue`: `top` is the comp-largest
	 * element, so `std::less<>` (the default) gives a max-heap and
	 * `std::greater<>` a min-heap.
	 *
	 * @tparam T The element type (`std::semiregular`: the pool's null node
	 *         holds a default-constructed `T`, and path copying copies values).
	 * @tparam Compare A strict weak order on `T`.
	 *
	 * @note Not constexpr (the shared pool is a `std::shared_ptr`; the forest
	 *       form is). `top` and `node_view::value`
	 *       return references into the pool, invalidated by a later operation
	 *       on any version sharing it.
	 */
	template<std::semiregular T, typename Compare = std::less<>>
		requires std::strict_weak_order<Compare &, T const &, T const &>
	class persistent_heap {
		using pool = detail::persistent_heap_pool<T, Compare>;
		std::shared_ptr<pool> pool_;
		uint32 root_ = 0;
		uint32 size_ = 0;

		persistent_heap(std::shared_ptr<pool> shared, uint32 root, uint32 size)
			: pool_(std::move(shared)), root_(root), size_(size) {}

		/** @brief `push` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_heap push_in(std::shared_ptr<pool> shared, T value) const {
			atl_assert(size_ + 1 < (std::numeric_limits<uint32>::max)());
			uint32 const fresh = shared->node_count();
			uint32 const leaf = shared->leaf(std::move(value));
			uint32 const root = shared->meld(root_, leaf, fresh);
			return {std::move(shared), root, size_ + 1};
		}

		/** @brief `pop` with the pool passed by value. */
		persistent_heap pop_in(std::shared_ptr<pool> shared) const {
			atl_assert(size_ > 0);
			uint32 const root = shared->pop(root_);
			return {std::move(shared), root, size_ - 1};
		}

		/** @brief `meld` with the pool passed by value (the other version's root and size given). */
		persistent_heap meld_in(std::shared_ptr<pool> shared, uint32 other_root, uint32 other_size) const {
			atl_assert(uint64(size_) + other_size < (std::numeric_limits<uint32>::max)());
			uint32 const root = shared->meld(root_, other_root, shared->node_count());
			return {std::move(shared), root, size_ + other_size};
		}

		/** @brief `push_range` with the pool passed by value. */
		template<std::ranges::input_range R>
			requires std::convertible_to<detail::forwarded_element_t<R>, T>
		persistent_heap push_range_in(std::shared_ptr<pool> shared, R &&values) const {
			uint32 const fresh = shared->node_count();
			uint32 const built = shared->heapify(std::forward<R>(values));
			uint32 const count = shared->node_count() - fresh;
			atl_assert(uint64(size_) + count < (std::numeric_limits<uint32>::max)());
			uint32 const size = size_ + count;
			uint32 const root = shared->meld(root_, built, fresh);
			return {std::move(shared), root, size};
		}

	public:
		/** @brief Element type. */
		using value_type = T;

		/**
		 * @brief A read-only position in a version's heap tree: the version's
		 *        root, or a node below it (`empty()` past a leaf). Every subtree
		 *        is heap-ordered, so `value()` is never comp-larger than its
		 *        parent's. Searches that treat heap nodes as graph nodes (k
		 *        shortest walks) walk these instead of popping, and make no node.
		 * @note Holds a pointer to the pool, not a share of it: valid while some
		 *       version sharing the pool lives; `value()` is a reference into the
		 *       pool (see the class note).
		 */
		class node_view {
			pool const *pool_ = nullptr;
			uint32 id_ = 0;

			friend class persistent_heap;

			node_view(pool const *shared, uint32 id) : pool_(shared), id_(id) {}

		public:
			/** @brief An empty view. */
			node_view() = default;

			/** @brief Whether there is no node here. */
			bool empty() const noexcept { return id_ == 0; }

			/** @brief The node's element. @pre `!empty()` (asserted) */
			[[nodiscard]] T const &value() const {
				atl_assert(id_ != 0);
				return pool_->nodes[id_].value;
			}

			/** @brief The left child (empty when none). @pre `!empty()` (asserted) */
			[[nodiscard]] node_view left() const {
				atl_assert(id_ != 0);
				return {pool_, pool_->nodes[id_].child[0]};
			}

			/** @brief The right child (empty when none). @pre `!empty()` (asserted) */
			[[nodiscard]] node_view right() const {
				atl_assert(id_ != 0);
				return {pool_, pool_->nodes[id_].child[1]};
			}
		};

		/** @brief An empty heap with a new pool. @pre Compare is default-initializable (constrained). */
		persistent_heap()
			requires std::default_initializable<Compare>
			: pool_(std::make_shared<pool>(Compare{})) {}

		/** @brief An empty heap ordered by `comp`, with a new pool. */
		explicit persistent_heap(Compare comp) : pool_(std::make_shared<pool>(std::move(comp))) {}

		/**
		 * @brief A heap of `values`, with a new pool (elements moved out of an
		 *        owning rvalue range only, as everywhere in the library).
		 * @complexity O(n)
		 */
		template<std::ranges::input_range R>
			requires std::convertible_to<detail::forwarded_element_t<R>, T>
		explicit persistent_heap(R &&values, Compare comp = {}) : pool_(std::make_shared<pool>(std::move(comp))) {
			root_ = pool_->heapify(std::forward<R>(values));
			size_ = pool_->node_count() - 1;
		}

		/** @brief The number of elements. */
		std::size_t size() const noexcept { return size_; }

		/** @brief Whether the heap holds no element. */
		bool empty() const noexcept { return size_ == 0; }

		/**
		 * @brief The comp-largest element, by reference into the pool.
		 * @pre `!empty()` (asserted)
		 * @complexity O(1)
		 */
		[[nodiscard]] T const &top() const {
			atl_assert(size_ > 0);
			return pool_->nodes[root_].value;
		}

		/** @brief The root of the heap tree (an empty view for an empty heap). @complexity O(1) */
		[[nodiscard]] node_view root() const { return {pool_.get(), root_}; }

		/** @brief The number of nodes in the shared pool: every version's total footprint, the null node included. */
		std::size_t arena_size() const noexcept { return pool_->nodes.size(); }

		/**
		 * @brief Reserve pool room for `operations` further operations (the unit
		 *        is operations, not nodes as in @ref atl::persistent_heap_forest::reserve) on
		 *        versions of about this one's size (`2 log2(n + 1) + 1` nodes
		 *        each, twice the expected meld length), so they rarely reallocate.
		 */
		void reserve(std::size_t operations) {
			std::size_t const per_operation = 2 * std::size_t(std::bit_width(size_ + uint64(1))) + 1;
			atl_assert(operations <= (std::numeric_limits<uint32>::max)() / per_operation);
			pool_->nodes.reserve_extra(operations * per_operation);
		}

		/**
		 * @brief The heap with `value` added.
		 * @complexity O(log n) expected time and new nodes
		 */
		[[nodiscard]] persistent_heap push(T value) const & { return push_in(pool_, std::move(value)); }

		/**
		 * @brief `push` consuming this version: the pool is moved, not copied (no
		 *        atomic refcount traffic), so `h = std::move(h).push(x)` is the cheap
		 *        linear idiom. The moved-from heap may only be destroyed or assigned to.
		 * @complexity O(log n) expected
		 */
		[[nodiscard]] persistent_heap push(T value) && { return push_in(std::move(pool_), std::move(value)); }

		/**
		 * @brief `push` of an element constructed in place from `args`.
		 * @complexity O(log n) expected
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_heap emplace(Args &&...args) const & {
			return push_in(pool_, T(std::forward<Args>(args)...));
		}

		/**
		 * @brief `emplace` consuming this version (see the rvalue `push`).
		 * @complexity O(log n) expected
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_heap emplace(Args &&...args) && {
			return push_in(std::move(pool_), T(std::forward<Args>(args)...));
		}

		/**
		 * @brief The heap without its top.
		 * @pre `!empty()` (asserted)
		 * @complexity O(log n) expected time and new nodes
		 */
		[[nodiscard]] persistent_heap pop() const & { return pop_in(pool_); }

		/**
		 * @brief `pop` consuming this version (see the rvalue `push`).
		 * @pre `!empty()` (asserted)
		 * @complexity O(log n) expected
		 */
		[[nodiscard]] persistent_heap pop() && { return pop_in(std::move(pool_)); }

		/**
		 * @brief The heap holding the elements of both versions (`other` may be
		 *        this version itself).
		 * @pre both share one pool, i.e. derive from one heap (asserted)
		 * @complexity O(log n + log m) expected time and new nodes
		 */
		[[nodiscard]] persistent_heap meld(persistent_heap const &other) const & {
			atl_assert(pool_ == other.pool_);
			return meld_in(pool_, other.root_, other.size_);
		}

		/**
		 * @brief `meld` consuming this version (see the rvalue `push`).
		 * @pre both share one pool (asserted)
		 * @complexity O(log n + log m) expected
		 */
		[[nodiscard]] persistent_heap meld(persistent_heap const &other) && {
			atl_assert(pool_ == other.pool_);
			return meld_in(std::move(pool_), other.root_, other.size_);
		}

		/**
		 * @brief The heap with every element of `values` added: one O(k)
		 *        array-built heap, then one meld.
		 * @complexity O(k + log n) expected time and new nodes
		 */
		template<std::ranges::input_range R>
			requires std::convertible_to<detail::forwarded_element_t<R>, T>
		[[nodiscard]] persistent_heap push_range(R &&values) const & {
			return push_range_in(pool_, std::forward<R>(values));
		}

		/**
		 * @brief `push_range` consuming this version (see the rvalue `push`).
		 * @complexity O(k + log n) expected
		 */
		template<std::ranges::input_range R>
			requires std::convertible_to<detail::forwarded_element_t<R>, T>
		[[nodiscard]] persistent_heap push_range(R &&values) && {
			return push_range_in(std::move(pool_), std::forward<R>(values));
		}
	};

	/** @brief A heap of a range holds the range's value type, ordered by `std::less<>`. */
	template<std::ranges::input_range R>
	persistent_heap(R &&) -> persistent_heap<std::ranges::range_value_t<R>>;

	/** @brief A heap of a range holds the range's value type, ordered by the given `Compare`. */
	template<std::ranges::input_range R, typename Compare>
		requires std::strict_weak_order<Compare &, std::ranges::range_value_t<R> const &,
										std::ranges::range_value_t<R> const &>
	persistent_heap(R &&, Compare) -> persistent_heap<std::ranges::range_value_t<R>, Compare>;

	/**
	 * @brief The forest form of @ref atl::persistent_heap — one object owns one node
	 *        pool, and a heap version is a @ref heap_id (its root node; @ref
	 *        empty_heap = 0 is the empty heap). Operations take ids and return
	 *        new ids; every id stays valid while the forest lives.
	 *
	 * The same randomized meld and path copying as the class form, for code
	 * that stores many versions (one heap per vertex, a search queue of
	 * heaps): a version is 4 bytes with no pool handle to copy. Every subtree
	 * of a version is itself a version, so @ref left and @ref right walk the
	 * heap tree by ids (what @ref atl::persistent_heap::node_view does for the
	 * class). No element count is kept per version.
	 *
	 * @tparam T, Compare As for @ref atl::persistent_heap.
	 * @note `top` returns a reference into the pool, invalidated by a later
	 *       operation on the forest.
	 */
	template<std::semiregular T, typename Compare = std::less<>>
		requires std::strict_weak_order<Compare &, T const &, T const &>
	class persistent_heap_forest {
	public:
		/** @brief Element type. */
		using value_type = T;
		/** @brief A heap version: the id of its root node. */
		using heap_id = uint32;

	private:
		detail::persistent_heap_pool<T, Compare> pool_;

		/** @brief Assert that `heap` names a version of this forest. */
		constexpr void check(heap_id heap) const { atl_assert(heap < pool_.nodes.size()); }

	public:
		/** @brief The id of the empty heap. */
		static constexpr heap_id empty_heap = 0;

		/** @brief An empty forest ordered by `comp`. */
		constexpr explicit persistent_heap_forest(Compare comp = {}) : pool_(std::move(comp)) {}

		/** @brief Whether version `heap` holds no element. */
		constexpr bool empty(heap_id heap) const noexcept { return heap == empty_heap; }

		/**
		 * @brief The comp-largest element of version `heap`, by reference into the pool.
		 * @pre `heap` is a nonempty version of this forest (asserted)
		 */
		[[nodiscard]] constexpr T const &top(heap_id heap) const {
			check(heap);
			atl_assert(heap != empty_heap);
			return pool_.nodes[heap].value;
		}

		/** @brief The version formed by the left subtree of `heap`'s root. @pre `heap` nonempty (asserted) */
		[[nodiscard]] constexpr heap_id left(heap_id heap) const {
			check(heap);
			atl_assert(heap != empty_heap);
			return pool_.nodes[heap].child[0];
		}

		/** @brief The version formed by the right subtree of `heap`'s root. @pre `heap` nonempty (asserted) */
		[[nodiscard]] constexpr heap_id right(heap_id heap) const {
			check(heap);
			atl_assert(heap != empty_heap);
			return pool_.nodes[heap].child[1];
		}

		/** @brief The number of nodes in the pool, the null node included. */
		constexpr std::size_t arena_size() const noexcept { return pool_.nodes.size(); }

		/** @brief Reserve pool room for `nodes` further nodes (a node count, unlike @ref atl::persistent_heap::reserve). */
		constexpr void reserve(std::size_t nodes) { pool_.nodes.reserve_extra(nodes); }

		/**
		 * @brief Version `heap` with `value` added; returns the new version.
		 * @complexity O(log n) expected time and new nodes
		 */
		[[nodiscard]] constexpr heap_id push(heap_id heap, T value) {
			check(heap);
			uint32 const fresh = pool_.node_count();
			uint32 const leaf = pool_.leaf(std::move(value));
			return pool_.meld(heap, leaf, fresh);
		}

		/** @brief `push(heap, T(args...))`. @complexity O(log n) expected */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] constexpr heap_id emplace(heap_id heap, Args &&...args) {
			return push(heap, T(std::forward<Args>(args)...));
		}

		/**
		 * @brief Version `heap` without its top.
		 * @pre `heap` nonempty (asserted)
		 * @complexity O(log n) expected time and new nodes
		 */
		[[nodiscard]] constexpr heap_id pop(heap_id heap) {
			check(heap);
			atl_assert(heap != empty_heap);
			return pool_.pop(heap);
		}

		/**
		 * @brief The version holding the elements of both `a` and `b` (they may be equal).
		 * @complexity O(log n + log m) expected time and new nodes
		 */
		[[nodiscard]] constexpr heap_id meld(heap_id a, heap_id b) {
			check(a);
			check(b);
			return pool_.meld(a, b, pool_.node_count());
		}

		/**
		 * @brief Version `heap` with every element of `values` added (one O(k)
		 *        array-built heap, then one meld); `push_range(empty_heap,
		 *        values)` builds a heap of a range.
		 * @complexity O(k + log n) expected time and new nodes
		 */
		template<std::ranges::input_range R>
			requires std::convertible_to<detail::forwarded_element_t<R>, T>
		[[nodiscard]] constexpr heap_id push_range(heap_id heap, R &&values) {
			check(heap);
			return pool_.meld_values(heap, std::forward<R>(values));
		}
	};
}
