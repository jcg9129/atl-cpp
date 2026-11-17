#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/ds/detail/persistent_arena.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief Shared arena backing @ref atl::persistent_list (Okasaki skew-binary RA-list).
		 *
		 * The list is a spine of complete binary trees whose sizes are 2^k - 1,
		 * laid out per the skew-binary representation of the length. `push_front`
		 * and `front` are O(1); indexed access and set are O(log n). Nodes are
		 * immutable (never overwritten, only appended), so the structure is fully
		 * persistent with O(1) structural sharing across versions. All nodes live
		 * in two `persistent_arena` "arenas" and are referenced by id; id 0 is
		 * the null sentinel.
		 *
		 * @tparam T The element type.
		 */
		template<std::semiregular T>
		struct skew_list_pool {
			/**
			 * @brief A complete-binary-tree node: its `value` plus indices of its
			 *        `left` and `right` subtrees (0 for a leaf).
			 */
			struct tree {
				T value;
				uint32 left, right;
			};

			/**
			 * @brief A spine element: the index of its `tree`, that tree's `size`
			 *        (2^k - 1 elements), and the index of the `next` spine cell.
			 */
			struct cell {
				uint32 tree, size, next;
			};

			persistent_arena<tree> trees{tree{}}; // id 0 = null
			persistent_arena<cell> spine{cell{}};

			/** @brief Room for @p pushes further `push_front`s: each makes one tree node and one spine cell. */
			void reserve(std::size_t pushes) {
				trees.reserve_extra(pushes);
				spine.reserve_extra(pushes);
			}

			uint32 make_tree(T const &v, uint32 l, uint32 r) { return trees.make({v, l, r}); }

			uint32 make_tree(T &&v, uint32 l, uint32 r) { return trees.make({std::move(v), l, r}); }

			uint32 make_cell(uint32 t, uint32 size, uint32 next) { return spine.make({t, size, next}); }

			/**
			 * @brief Prepend @p x, returning the new head cell (skew-binary cons).
			 *
			 * If the two leading spine trees have equal size they are combined
			 * under @p x into a single tree of size 2w+1, keeping the spine a valid
			 * skew-binary layout; otherwise @p x is prepended as a size-1 singleton.
			 * Exactly one of the two cases fires, so this is O(1).
			 * @complexity O(1)
			 */
			uint32 push_front(uint32 head, T &&x) {
				if (head && spine[head].next && spine[head].size == spine[spine[head].next].size) {
					uint32 w = spine[head].size;
					uint32 t = make_tree(std::move(x), spine[head].tree, spine[spine[head].next].tree);
					return make_cell(t, 2 * w + 1, spine[spine[head].next].next);
				}
				return make_cell(make_tree(std::move(x), 0, 0), 1, head);
			}

			/**
			 * @brief Drop the front element, returning the new head cell.
			 *
			 * Inverse of `push_front` — a size-1 head is simply unlinked, while a
			 * larger head tree is split at its root into its two equal-sized
			 * subtrees, which become the two leading spine cells. O(1).
			 * @complexity O(1)
			 */
			uint32 pop_front(uint32 head) {
				uint32 w = (spine[head].size - 1) / 2;
				if (w == 0) return spine[head].next;
				uint32 t = spine[head].tree;
				return make_cell(trees[t].left, w, make_cell(trees[t].right, w, spine[head].next));
			}

			T const &front(uint32 head) const { return trees[spine[head].tree].value; }

			/**
			 * @brief The element at index @p i (0 is the front).
			 *
			 * Walks the spine, skipping whole trees until @p i falls inside one,
			 * then descends that tree by comparing @p i against each subtree's size.
			 * @complexity O(log n)
			 */
			T const &at(uint32 head, uint32 i) const {
				while (i >= spine[head].size) i -= spine[head].size, head = spine[head].next;
				uint32 t = spine[head].tree, size = spine[head].size;
				for (;;) {
					if (i == 0) return trees[t].value;
					uint32 half = (size - 1) / 2;
					--i;
					if (i < half)
						t = trees[t].left, size = half;
					else
						i -= half, t = trees[t].right, size = half;
				}
			}

			/**
			 * @brief Rebuild tree @p t (of @p size) with element @p i set to @p x.
			 *
			 * Copies only the nodes on the root-to-target path, sharing the
			 * untouched subtree at each step, and returns the new tree's index.
			 * @complexity O(log n)
			 */
			uint32 set_tree(uint32 t, uint32 size, uint32 i, T &&x) {
				// Node fields are copied out before the recursion: the recursive
				// set_tree appends to `trees` and may reallocate it, which would
				// dangle any reference into the vector held across the call.
				uint32 const left = trees[t].left, right = trees[t].right;
				if (i == 0) return make_tree(std::move(x), left, right);
				uint32 half = (size - 1) / 2;
				--i;
				T value = trees[t].value;
				if (i < half) {
					uint32 const child = set_tree(left, half, i, std::move(x));
					return make_tree(std::move(value), child, right);
				}
				uint32 const child = set_tree(right, half, i - half, std::move(x));
				return make_tree(std::move(value), left, child);
			}

			/**
			 * @brief The head of a list equal to this one but with index @p i set to @p x.
			 *
			 * Locates the spine cell containing @p i, delegates to `set_tree` to
			 * copy that tree's affected path, and copies the spine cells ahead of it
			 * while sharing everything below.
			 * @complexity O(log n)
			 */
			uint32 set(uint32 head, uint32 i, T &&x) {
				// The cell is copied out first: the recursive set/set_tree appends
				// to `spine`/`trees` and may reallocate them mid-call.
				cell const c = spine[head];
				if (i < c.size) {
					uint32 const t = set_tree(c.tree, c.size, i, std::move(x));
					return make_cell(t, c.size, c.next);
				}
				uint32 const rest = set(c.next, i - c.size, std::move(x));
				return make_cell(c.tree, c.size, rest);
			}
		};
	}

	/**
	 * @brief Fully persistent random-access list (Okasaki skew-binary RA-list).
	 *
	 * A front-oriented immutable sequence: `push_front`/`pop_front`/`front` are
	 * O(1), while `operator[]`/`set` are O(log n) (index 0 is the front, i.e. the
	 * most recently pushed element). Fully persistent with O(1) structural sharing
	 * per push, so it is far lighter and faster than a trie for growth while
	 * keeping every version — an efficient persistent stack / deque-front, at the
	 * cost of slower random indexing.
	 *
	 * @tparam T The element type.
	 *
	 * @note Not constexpr (versions own a std::shared_ptr pool); this applies to all members. `operator[]`/`front` return a reference
	 *       into the arena, invalidated by a later mutation on a shared version.
	 *
	 */
	template<std::semiregular T>
	class persistent_list {
		using pool = detail::skew_list_pool<T>;
		std::shared_ptr<pool> pool_;
		uint32 head_ = 0;
		std::size_t size_ = 0;

		/** @brief Wraps head cell @p head of a list of @p size elements living in @p shared. */
		persistent_list(std::shared_ptr<pool> shared, uint32 head, std::size_t size)
			: pool_(std::move(shared)), head_(head), size_(size) {}

		/** @brief `push_front` with the pool passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_list push_front_in(std::shared_ptr<pool> shared, T value) const {
			atl_assert(size_ < (std::numeric_limits<uint32>::max)());
			uint32 head = shared->push_front(head_, std::move(value));
			return {std::move(shared), head, size_ + 1};
		}

		/** @brief `pop_front` with the pool passed by value. */
		persistent_list pop_front_in(std::shared_ptr<pool> shared) const {
			atl_assert(size_ > 0);
			uint32 head = shared->pop_front(head_);
			return {std::move(shared), head, size_ - 1};
		}

		/** @brief `set` with the pool passed by value. */
		persistent_list set_in(std::shared_ptr<pool> shared, std::size_t i, T value) const {
			atl_assert(i < size_);
			uint32 head = shared->set(head_, uint32(i), std::move(value));
			return {std::move(shared), head, size_};
		}

	public:
		/** @brief An empty list. */
		persistent_list() : pool_(std::make_shared<pool>()) {}

		/**
		 * @brief Build from `values`, preserving order (index 0 is `values[0]`).
		 * @complexity O(n)
		 */
		template<std::ranges::sized_range Range>
			requires std::convertible_to<detail::forwarded_element_t<Range>, T>
		explicit persistent_list(Range &&values) : pool_(std::make_shared<pool>()) {
			atl_assert(std::ranges::size(values) <= (std::numeric_limits<uint32>::max)());
			std::vector<T> buffer;
			buffer.reserve(std::ranges::size(values));
			pool_->reserve(buffer.capacity());
			for (auto &&v : values)
				buffer.push_back(static_cast<T>(detail::forward_element<Range>(std::forward<decltype(v)>(v))));
			for (std::size_t i = buffer.size(); i-- > 0;) head_ = pool_->push_front(head_, std::move(buffer[i]));
			size_ = buffer.size();
		}

		/**
		 * @brief A list of `n` copies of `value`.
		 * @complexity O(n)
		 */
		explicit persistent_list(std::size_t n, T value = T{}) : persistent_list(std::vector<T>(n, std::move(value))) {}

		/** @brief Number of elements in this version. @complexity O(1) */
		std::size_t size() const noexcept { return size_; }

		/** @brief Whether this version contains no elements. @complexity O(1) */
		bool empty() const noexcept { return size_ == 0; }

		/**
		 * @brief Allocation-free forward iterator, front to back: walks the
		 *        spine and each tree in preorder (the index order) with a
		 *        fixed explicit stack, O(1) amortised per step.
		 * @note Reads the arena like `operator[]`, so it is invalidated by a
		 *       mutation on any version sharing it.
		 */
		class iterator {
			pool const *pool_ = nullptr;
			uint32 cell_ = 0;
			std::size_t remaining_ = 0;
			// a spine tree holds 2^k - 1 < 2^32 elements: depth k <= 32
			std::array<uint32, 32> stack_{};
			uint32 depth_ = 0;

			friend class persistent_list;

			iterator(pool const *p, uint32 head, std::size_t size) : pool_(p), cell_(head), remaining_(size) {
				if (remaining_ > 0) stack_[depth_++] = pool_->spine[cell_].tree;
			}

		public:
			/** @brief Forward traversal category. */
			using iterator_category = std::forward_iterator_tag;
			/** @brief Element type. */
			using value_type = T;
			/** @brief Signed iterator distance type. */
			using difference_type = std::ptrdiff_t;
			/** @brief Read-only element reference. */
			using reference = T const &;
			/** @brief Read-only element pointer. */
			using pointer = T const *;

			/** @brief Singular iterator; assign a version iterator before dereferencing. */
			iterator() = default;

			/** @brief Refer to the current element. @pre Not past the end. */
			[[nodiscard]] reference operator*() const { return pool_->trees[stack_[depth_ - 1]].value; }

			/** @brief Point to the current element. @pre Not past the end. */
			[[nodiscard]] pointer operator->() const { return std::addressof(**this); }

			/** @brief Advance to the next element. @pre Not past the end. @return This iterator. */
			iterator &operator++() {
				uint32 const t = stack_[--depth_];
				if (auto const &nd = pool_->trees[t]; nd.left != 0) { // complete trees: both children or none
					stack_[depth_++] = nd.right;
					stack_[depth_++] = nd.left;
				}
				if (--remaining_ > 0 && depth_ == 0) {
					cell_ = pool_->spine[cell_].next;
					stack_[depth_++] = pool_->spine[cell_].tree;
				}
				return *this;
			}

			/** @brief Advance and return the unadvanced iterator. @pre Not past the end. */
			iterator operator++(int) {
				iterator old = *this;
				++*this;
				return old;
			}

			/** @brief Iterators of one version compare by position (elements left to visit). */
			[[nodiscard]] bool operator==(iterator const &o) const { return remaining_ == o.remaining_; }
		};

		/** @brief The front. @complexity O(1) */
		[[nodiscard]] iterator begin() const { return iterator(pool_.get(), head_, size_); }

		/** @brief Past the back. @complexity O(1) */
		[[nodiscard]] iterator end() const { return iterator(pool_.get(), 0, 0); }

		/**
		 * @brief Reserve arena room for @p pushes further `push_front`s (on this
		 *        or any version sharing the arena), so they do not reallocate.
		 */
		void reserve(std::size_t pushes) { pool_->reserve(pushes); }

		/**
		 * @brief The element at `index` (0 is the front / most recently pushed).
		 * @complexity O(log n)
		 */
		[[nodiscard]] T const &operator[](std::size_t i) const {
			atl_assert(i < size_);
			return pool_->at(head_, uint32(i));
		}

		/**
		 * @brief The front element (index 0).
		 * @pre The list is non-empty.
		 * @complexity O(1)
		 */
		[[nodiscard]] T const &front() const {
			atl_assert(size_ > 0);
			return pool_->front(head_);
		}

		/**
		 * @brief The list with `value` prepended at the front.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_list push_front(T value) const & { return push_front_in(pool_, std::move(value)); }

		/**
		 * @brief `push_front` consuming this version: the pool is moved, not copied
		 *        (no atomic refcount traffic), so `v = std::move(v).push_front(x)` is the
		 *        cheap linear idiom. The moved-from list may only be destroyed or assigned to.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_list push_front(T value) && { return push_front_in(std::move(pool_), std::move(value)); }

		/**
		 * @brief `push_front` of an element constructed in place from `args`.
		 * @complexity O(1)
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_list emplace_front(Args &&...args) const & {
			return push_front_in(pool_, T(std::forward<Args>(args)...));
		}

		/**
		 * @brief `emplace_front` consuming this version (see the rvalue `push_front`).
		 * @complexity O(1)
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_list emplace_front(Args &&...args) && {
			return push_front_in(std::move(pool_), T(std::forward<Args>(args)...));
		}

		/**
		 * @brief The list with its front element removed.
		 * @pre The list is non-empty.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_list pop_front() const & { return pop_front_in(pool_); }

		/**
		 * @brief `pop_front` consuming this version (see the rvalue `push_front`).
		 * @pre The list is non-empty.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_list pop_front() && { return pop_front_in(std::move(pool_)); }

		/**
		 * @brief The list with element `index` replaced by `value`.
		 * @complexity O(log n)
		 */
		[[nodiscard]] persistent_list set(std::size_t i, T value) const & { return set_in(pool_, i, std::move(value)); }

		/**
		 * @brief `set` consuming this version (see the rvalue `push_front`).
		 * @complexity O(log n)
		 */
		[[nodiscard]] persistent_list set(std::size_t i, T value) && { return set_in(std::move(pool_), i, std::move(value)); }
	};

	/** @brief Deduce the element type from a sized range. */
	template<std::ranges::sized_range Range>
	persistent_list(Range &&) -> persistent_list<std::ranges::range_value_t<Range>>;
}
