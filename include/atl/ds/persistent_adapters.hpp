#pragma once
#include <atl/ds/persistent_array.hpp>
#include <atl/ds/persistent_list.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <utility>

/**
 * @file
 * @brief Persistent FIFO / deque adapters.
 *
 * @ref atl::persistent_queue is an *unbounded* FIFO built on @ref atl::persistent_list;
 * @ref atl::persistent_deque is a *fixed-capacity* ring over a persistent array, because
 * a double-ended queue must cheaply mutate BOTH ends and the one-ended list trick the
 * queue uses does not extend to it (a two-list banker's deque is only amortised and
 * degrades under full-persistence branching; a finger tree is not implemented). There
 * is deliberately no stack adapter: a persistent stack is exactly
 * @ref atl::persistent_list (unbounded, O(1) push_front/pop_front/front).
 */
namespace atl {
	namespace detail {
		/**
		 * @brief A persistent array of @p T usable as `persistent_deque` storage:
		 *        copyable (a version is returned by value), constructible with a
		 *        capacity, sized, `set` returning the new version, and `b[i]`
		 *        yielding a `T const &` into the backing itself — `front`/`back`
		 *        return that reference, so a by-value `operator[]` would dangle.
		 */
		template<typename Backing, typename T>
		concept persistent_ring_backing =
			std::copyable<Backing> && std::constructible_from<Backing, std::size_t>
			&& requires(Backing const &b, std::size_t i, T v) {
				   { b[i] } -> std::same_as<T const &>;
				   { b.size() } -> std::convertible_to<std::size_t>;
				   { b.set(i, std::move(v)) } -> std::convertible_to<Backing>;
			   };
	}

	/**
	 * @brief Persistent FIFO queue: fully persistent and unbounded.
	 *
	 * Every mutator returns a new version and all past versions stay queryable, so you
	 * may branch off any of them. It keeps a @ref atl::persistent_list of every value ever
	 * pushed on this version's path (newest at index 0) plus a count of how many have
	 * been dequeued; the live queue is the un-popped tail. So `push`, `pop` and `back`
	 * are O(1) and `front` is O(min(size, log pushes)), where `pushes` counts every
	 * value pushed on this version's path: a skew-binary lookup at index `size() - 1`
	 * of the push list. Unlike a ring over a persistent array there is no wide-node
	 * trie and no capacity to fix up front.
	 *
	 * @note Popped values are not released: they stay in the push list (and its arena)
	 *       for as long as any version referencing them lives.
	 *
	 * Member names mirror `std::queue` (`push`/`pop`/`front`/`back`): read the oldest
	 * element with `front()`, then drop it with `pop()`.
	 *
	 * @note Not constexpr (the backing owns a std::shared_ptr pool); this applies to all members.
	 * @tparam T The element type.
	 */
	template<std::semiregular T>
	class persistent_queue {
		persistent_list<T> pushes_; // every value pushed on this version's path, newest at index 0
		std::size_t popped_ = 0;	// how many have been dequeued from the front

		/** @brief Wraps a push list and the count of its oldest values already dequeued. */
		persistent_queue(persistent_list<T> pushes, std::size_t popped) : pushes_(std::move(pushes)), popped_(popped) {}

	public:
		/** @brief An empty queue. */
		persistent_queue() = default;

		/** @brief Number of elements currently in the queue. */
		std::size_t size() const noexcept { return pushes_.size() - popped_; }

		/** @brief Whether this version contains no elements. @complexity O(1) */
		bool empty() const noexcept { return size() == 0; }

		/**
		 * @brief The oldest (front) element.
		 * @pre The queue is non-empty.
		 * @complexity O(min(size, log pushes)), `pushes` being the values ever pushed
		 *             on this version's path
		 */
		[[nodiscard]] T const &front() const {
			atl_assert(!empty());
			return pushes_[size() - 1]; // the oldest value still queued
		}

		/**
		 * @brief The most recently pushed (back) element.
		 * @pre The queue is non-empty.
		 * @complexity O(1)
		 */
		[[nodiscard]] T const &back() const {
			atl_assert(!empty());
			return pushes_[0]; // the newest push is popped last, so it is live whenever size() > 0
		}

		/**
		 * @brief The queue with `value` pushed at the back.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_queue push(T value) const & { return {pushes_.push_front(std::move(value)), popped_}; }

		/**
		 * @brief `push` consuming this version: the underlying list's pool is moved, not
		 *        copied (`q = std::move(q).push(x)`). The moved-from object may only be destroyed or assigned to.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_queue push(T value) && { return {std::move(pushes_).push_front(std::move(value)), popped_}; }

		/**
		 * @brief `push` of an element constructed in place from `args`.
		 * @complexity O(1)
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_queue emplace(Args &&...args) const & {
			return push(T(std::forward<Args>(args)...));
		}

		/**
		 * @brief `emplace` consuming this version (see the rvalue `push`).
		 * @complexity O(1)
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_queue emplace(Args &&...args) && {
			return std::move(*this).push(T(std::forward<Args>(args)...));
		}

		/**
		 * @brief The queue with its front element removed.
		 * @pre The queue is non-empty.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_queue pop() const & {
			atl_assert(!empty());
			return {pushes_, popped_ + 1};
		}

		/**
		 * @brief `pop` consuming this version (see the rvalue `push`).
		 * @pre The queue is non-empty.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_queue pop() && {
			atl_assert(!empty());
			return {std::move(pushes_), popped_ + 1};
		}
	};

	/**
	 * @brief Persistent double-ended queue over a fixed-capacity circular array.
	 *
	 * `front`/`back`/`push_back`/`push_front` are O(log capacity); the pops are
	 * O(1). Physical slot = `index % capacity`, so slots freed at one end are
	 * reused at the other and the deque holds up to `capacity` elements
	 * concurrently regardless of total operations. Persistence is inherited from
	 * `Backing`.
	 *
	 * @note Not constexpr (the backing owns a std::shared_ptr pool); this applies to all members.
	 * @tparam T       The element type (default-constructible).
	 * @tparam Backing The persistent array storing the elements.
	 */
	template<std::semiregular T, detail::persistent_ring_backing<T> Backing = persistent_array<T>>
	class persistent_deque {
		Backing data_;
		std::size_t head_ = 0;	// physical index of the front
		std::size_t count_ = 0; // number of elements

		/** @brief Wraps a ring buffer @p data whose front is at physical index @p head, holding @p count elements. */
		persistent_deque(Backing data, std::size_t head, std::size_t count)
			: data_(std::move(data)), head_(head), count_(count) {}

		/** @brief `push_back` with the backing passed by value, so the lvalue and rvalue overloads share one body. */
		persistent_deque push_back_in(Backing data, T value) const {
			atl_assert(count_ < data.size());
			std::size_t const cap = data.size();
			std::size_t const slot = count_ >= cap - head_ ? count_ - (cap - head_) : head_ + count_;
			return {std::move(data).set(slot, std::move(value)), head_, count_ + 1};
		}

		/** @brief `push_front` with the backing passed by value. */
		persistent_deque push_front_in(Backing data, T value) const {
			atl_assert(count_ < data.size());
			std::size_t new_head = head_ == 0 ? data.size() - 1 : head_ - 1;
			return {std::move(data).set(new_head, std::move(value)), new_head, count_ + 1};
		}

		/** @brief `pop_front` with the backing passed by value. */
		persistent_deque pop_front_in(Backing data) const {
			atl_assert(count_ > 0);
			std::size_t next = head_ + 1;
			if (next == data.size()) next = 0;
			return {std::move(data), next, count_ - 1};
		}

	public:
		/**
		 * @brief An empty deque of capacity 0 — a placeholder (e.g. a member)
		 *        to be assigned a sized deque later; nothing can be pushed.
		 */
		persistent_deque() : data_(std::size_t{0}) {}

		/**
		 * @brief An empty deque holding up to `capacity` elements at once.
		 */
		explicit persistent_deque(std::size_t capacity) : data_(capacity) {}

		/** @brief Number of elements in this version. @complexity O(1) */
		std::size_t size() const noexcept { return count_; }

		/** @brief Whether this version contains no elements. @complexity O(1) */
		bool empty() const noexcept { return count_ == 0; }

		/** @brief Maximum simultaneous element count. @complexity O(1) */
		std::size_t capacity() const noexcept { return data_.size(); }

		/**
		 * @brief The element at position @p i (0 is the front).
		 * @pre `i < size()`
		 * @complexity O(log capacity)
		 */
		[[nodiscard]] T const &operator[](std::size_t i) const {
			atl_assert(i < count_);
			std::size_t const gap = data_.size() - head_;
			std::size_t const idx = i >= gap ? i - gap : head_ + i;
			return data_[idx];
		}

		/**
		 * @brief Make room in the backing's arena for @p pushes further pushes
		 *        (each is one `set` on the backing), when the backing offers a
		 *        `reserve`.
		 */
		void reserve(std::size_t pushes)
			requires requires(Backing &b) { b.reserve(pushes); }
		{
			data_.reserve(pushes);
		}

		/**
		 * @brief The front element.
		 * @pre The deque is non-empty.
		 * @complexity O(log capacity)
		 */
		[[nodiscard]] T const &front() const {
			atl_assert(count_ > 0);
			return data_[head_];
		}

		/**
		 * @brief The back element.
		 * @pre The deque is non-empty.
		 * @complexity O(log capacity)
		 */
		[[nodiscard]] T const &back() const {
			atl_assert(count_ > 0);
			std::size_t cap = data_.size(), idx = head_ + count_ - 1;
			if (idx >= cap) idx -= cap;
			return data_[idx];
		}

		/**
		 * @brief The deque with `value` added at the back.
		 * @pre size() < capacity().
		 * @complexity O(log capacity)
		 */
		[[nodiscard]] persistent_deque push_back(T value) const & { return push_back_in(data_, std::move(value)); }

		/**
		 * @brief `push_back` consuming this version: the backing is moved, not copied. The moved-from object may only be destroyed or assigned to.
		 * @pre size() < capacity().
		 * @complexity O(log capacity)
		 */
		[[nodiscard]] persistent_deque push_back(T value) && { return push_back_in(std::move(data_), std::move(value)); }

		/**
		 * @brief The deque with `value` added at the front.
		 * @pre size() < capacity().
		 * @complexity O(log capacity)
		 */
		[[nodiscard]] persistent_deque push_front(T value) const & { return push_front_in(data_, std::move(value)); }

		/**
		 * @brief `push_front` consuming this version: the backing is moved, not copied. The moved-from object may only be destroyed or assigned to.
		 * @pre size() < capacity().
		 * @complexity O(log capacity)
		 */
		[[nodiscard]] persistent_deque push_front(T value) && { return push_front_in(std::move(data_), std::move(value)); }

		/**
		 * @brief `push_back` of an element constructed in place from `args`.
		 * @pre size() < capacity().
		 * @complexity O(log capacity)
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_deque emplace_back(Args &&...args) const {
			return push_back(T(std::forward<Args>(args)...));
		}

		/**
		 * @brief `push_front` of an element constructed in place from `args`.
		 * @pre size() < capacity().
		 * @complexity O(log capacity)
		 */
		template<typename... Args>
			requires std::constructible_from<T, Args...>
		[[nodiscard]] persistent_deque emplace_front(Args &&...args) const {
			return push_front(T(std::forward<Args>(args)...));
		}

		/**
		 * @brief The deque with its back element removed.
		 * @pre The deque is non-empty.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_deque pop_back() const & {
			atl_assert(count_ > 0);
			return {data_, head_, count_ - 1};
		}

		/**
		 * @brief `pop_back` consuming this version (see the rvalue `push_back`).
		 * @pre The deque is non-empty.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_deque pop_back() && {
			atl_assert(count_ > 0);
			return {std::move(data_), head_, count_ - 1};
		}

		/**
		 * @brief The deque with its front element removed.
		 * @pre The deque is non-empty.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_deque pop_front() const & { return pop_front_in(data_); }

		/**
		 * @brief `pop_front` consuming this version (see the rvalue `push_back`).
		 * @pre The deque is non-empty.
		 * @complexity O(1)
		 */
		[[nodiscard]] persistent_deque pop_front() && { return pop_front_in(std::move(data_)); }
	};
}
