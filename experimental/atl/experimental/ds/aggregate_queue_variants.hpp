#pragma once

#include <atl/ds/detail/aggregate_stack.hpp>
#include <atl/experimental/ds/detail/aggregate_chunks.hpp>
#include <atl/experimental/ds/detail/aggregate_ring.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace atl::experimental {
	/**
	 * @brief aggregate_queue contender by DABA Lite on a growable ring: the same interface and monoid fold, with a
	 * constant number of operation calls per update instead of the queue's occasional O(n) transfer. Not measured
	 * yet.
	 * @tparam Value Copyable element type; neither default construction nor equality is needed.
	 * @tparam Operation Binary operation on `Value`, called on `Value const &` operands through a const callable.
	 * @pre The monoid laws, arithmetic validity and nonthrowing mutations of atl::aggregate_queue.
	 * @complexity At most three operation calls per push_back, two per pop_front and one per fold(), whatever the
	 * occupancy. Ring growth relocates O(n) records (amortized O(1)); after reserve(elements) no update allocates or
	 * relocates while size() stays at most elements. Storage: n raw values, n working folds, two cached folds and the
	 * identity. Copies are deep, O(n); moves are O(1) container work and leave the source only destructible or
	 * assignable.
	 * @note A failed allocation throws and leaves the contents unchanged. References from front() and back() are
	 * valid until the next mutation, reserve, move or assignment.
	 * Implemented independently from the invariants and schedule in Tangwongsan, Hirzel and Schneider, "In-order
	 * sliding-window aggregation in worst-case constant time" (VLDB Journal, 2021), section 6.
	 */
	template<std::copyable Value, binary_operation<Value> Operation>
	class aggregate_queue_by_daba_lite_ring {
		detail::aggregate_ring<atl::detail::aggregate_entry<Value>> entries_;
		Value identity_;
		Value bridge_;
		Value back_aggregate_;
		[[no_unique_address]] Operation operation_;
		std::size_t left_ = 0, right_ = 0, accumulator_ = 0, back_ = 0;

		/**
		 * @brief One scheduled repair step; boundaries are offsets from the live front, so they never outgrow size().
		 * @note Regions: [0, left_) holds folds through back_; [left_, right_) local suffix folds; [right_,
		 * accumulator_) raw values; [accumulator_, back_) suffix folds through back_; [back_, size()) is folded only
		 * into back_aggregate_. bridge_ is the fold of [right_, back_) while the front regions shrink.
		 */
		constexpr void repair() {
			if (back_ == 0) {
				left_ = right_ = accumulator_ = back_ = size();
				bridge_ = back_aggregate_ = identity_;
				return;
			}
			if (left_ == back_) {
				left_ = 0;
				accumulator_ = back_ = size();
				bridge_ = back_aggregate_;
				back_aggregate_ = identity_;
			}
			if (left_ == right_) {
				++left_;
				++right_;
				++accumulator_;
			} else {
				entries_[left_].aggregate = atl::detail::aggregate_combine(operation_, entries_[left_].aggregate, bridge_);
				++left_;
				auto const &suffix = accumulator_ == back_ ? identity_ : entries_[accumulator_].aggregate;
				entries_[accumulator_ - 1].aggregate = atl::detail::aggregate_combine(operation_, entries_[accumulator_ - 1].aggregate, suffix);
				--accumulator_;
			}
		}

	public:
		using value_type = Value;	   ///< Stored element type.
		using size_type = std::size_t; ///< Element count type.

		/**
		 * @brief An empty queue.
		 * @param identity The neutral element; fold() returns it while the queue is empty.
		 * @param operation The monoid operation, stored by value.
		 */
		constexpr explicit aggregate_queue_by_daba_lite_ring(Value identity, Operation operation = Operation{}) : identity_(std::move(identity)), bridge_(identity_), back_aggregate_(identity_), operation_(std::move(operation)) {}

		constexpr aggregate_queue_by_daba_lite_ring(aggregate_queue_by_daba_lite_ring const &) = default;

		constexpr aggregate_queue_by_daba_lite_ring(aggregate_queue_by_daba_lite_ring &&) = default;

		/** @brief Copy into a temporary, then commit by move, so a failed allocation leaves this object unchanged. */
		constexpr aggregate_queue_by_daba_lite_ring &operator=(aggregate_queue_by_daba_lite_ring const &other)
			requires std::is_move_assignable_v<Operation>
		{
			if (this != &other) {
				aggregate_queue_by_daba_lite_ring copy(other);
				*this = std::move(copy);
			}
			return *this;
		}

		constexpr aggregate_queue_by_daba_lite_ring &operator=(aggregate_queue_by_daba_lite_ring &&) = default;

		/** @brief Append @p value; the ring grows first, so a failed allocation changes nothing. @pre size() is below max_size of the ring (asserted). */
		constexpr void push_back(Value value) {
			atl_assert(size() < entries_.max_size());
			entries_.prepare(size() + 1, entries_.max_size());
			Value aggregate = value;
			back_aggregate_ = atl::detail::aggregate_combine(operation_, back_aggregate_, value);
			entries_.push_back({std::move(value), std::move(aggregate)});
			repair();
		}

		/**
		 * @brief Append `Value(arguments...)`, built before the ring grows, so the arguments may refer to front() or back().
		 * @tparam Arguments Constructor arguments of `Value`.
		 */
		template<typename... Arguments>
			requires std::constructible_from<Value, Arguments...>
		constexpr void emplace_back(Arguments &&...arguments) { push_back(Value(std::forward<Arguments>(arguments)...)); }

		/** @brief Remove the first value. @pre The queue is nonempty (asserted). */
		constexpr void pop_front() {
			atl_assert(!empty());
			entries_.pop_front();
			--left_;
			--right_;
			--accumulator_;
			--back_;
			repair();
		}

		/** @brief The first value. @pre The queue is nonempty (asserted). */
		[[nodiscard]] constexpr Value const &front() const {
			atl_assert(!empty());
			return entries_[0].value;
		}

		/** @brief The last value. @pre The queue is nonempty (asserted). */
		[[nodiscard]] constexpr Value const &back() const {
			atl_assert(!empty());
			return entries_[size() - 1].value;
		}

		/** @brief The fold from front to back, or the identity when empty; one operation call; allocates no container storage. */
		[[nodiscard]] constexpr Value fold() const { return atl::detail::aggregate_combine(operation_, back_ == 0 ? identity_ : entries_[0].aggregate, back_aggregate_); }

		/** @brief Number of live values. */
		constexpr size_type size() const noexcept { return entries_.size(); }

		/** @brief Whether no values are live. */
		constexpr bool empty() const noexcept { return entries_.empty(); }

		/** @brief Destroy the live values in O(n), keeping the allocation, identity and operation. */
		constexpr void clear() {
			entries_.clear();
			left_ = right_ = accumulator_ = back_ = 0;
			bridge_ = back_aggregate_ = identity_;
		}

		/** @brief Allocate for @p elements simultaneously live values in O(n + elements). @pre @p elements fits the ring (asserted). */
		constexpr void reserve(size_type elements) { entries_.reserve(elements); }
	};

	/**
	 * @brief aggregate_queue contender by DABA Lite on recycled fixed-size chunks: the schedule of
	 * aggregate_queue_by_daba_lite_ring, but live records never relocate, so an update does constant container work
	 * apart from allocating a chunk. Not measured yet.
	 * @tparam Value Copyable element type; neither default construction nor equality is needed.
	 * @tparam Operation Binary operation on `Value`, called on `Value const &` operands through a const callable.
	 * @pre The monoid laws, arithmetic validity and nonthrowing mutations of atl::aggregate_queue.
	 * @complexity At most three operation calls per push_back, two per pop_front and one per fold(). Constant
	 * container work per update, plus at most one chunk allocation per 32 pushes (none after reserve(elements) while
	 * size() stays at most elements); allocator latency is not bounded. Chunks emptied at the front are reused at the
	 * back. Live storage O(n); retained storage O(H + reserved elements) for the largest occupancy H so far. clear()
	 * O(n), copies O(n), moves O(1) container work, leaving the source only destructible or assignable.
	 * @note A failed allocation throws and leaves the contents unchanged. References from front() and back() are
	 * valid until the next mutation, reserve, move or assignment. Same reference as
	 * aggregate_queue_by_daba_lite_ring.
	 */
	template<std::copyable Value, binary_operation<Value> Operation>
	class aggregate_queue_by_daba_lite_chunks {
		using storage = detail::daba_chunk_storage<atl::detail::aggregate_entry<Value>>;
		storage entries_;
		Value identity_;
		Value bridge_;
		Value back_aggregate_;
		[[no_unique_address]] Operation operation_;

		/** @brief The repair step of aggregate_queue_by_daba_lite_ring, on the four stable cursors of entries_. */
		constexpr void repair() {
			if (entries_.back == entries_.begin()) {
				entries_.left = entries_.right = entries_.accumulator = entries_.back = entries_.end();
				bridge_ = back_aggregate_ = identity_;
				return;
			}
			if (entries_.left == entries_.back) {
				entries_.left = entries_.begin();
				entries_.accumulator = entries_.back = entries_.end();
				bridge_ = back_aggregate_;
				back_aggregate_ = identity_;
			}
			if (entries_.left == entries_.right) {
				entries_.left = storage::next(entries_.left);
				entries_.right = storage::next(entries_.right);
				entries_.accumulator = storage::next(entries_.accumulator);
			} else {
				auto &left = entries_.at(entries_.left).aggregate;
				left = atl::detail::aggregate_combine(operation_, left, bridge_);
				entries_.left = storage::next(entries_.left);
				auto const &suffix = entries_.accumulator == entries_.back ? identity_ : entries_.at(entries_.accumulator).aggregate;
				auto &extended = entries_.at(storage::previous(entries_.accumulator)).aggregate;
				extended = atl::detail::aggregate_combine(operation_, extended, suffix);
				entries_.accumulator = storage::previous(entries_.accumulator);
			}
		}

	public:
		using value_type = Value;	   ///< Stored element type.
		using size_type = std::size_t; ///< Element count type.

		/**
		 * @brief An empty queue; allocates one chunk.
		 * @param identity The neutral element; fold() returns it while the queue is empty.
		 * @param operation The monoid operation, stored by value.
		 */
		constexpr explicit aggregate_queue_by_daba_lite_chunks(Value identity, Operation operation = Operation{}) : identity_(std::move(identity)), bridge_(identity_), back_aggregate_(identity_), operation_(std::move(operation)) {}

		constexpr aggregate_queue_by_daba_lite_chunks(aggregate_queue_by_daba_lite_chunks const &) = default;

		constexpr aggregate_queue_by_daba_lite_chunks(aggregate_queue_by_daba_lite_chunks &&) = default;

		/** @brief Copy into a temporary, then commit by move, so a failed allocation leaves this object unchanged. */
		constexpr aggregate_queue_by_daba_lite_chunks &operator=(aggregate_queue_by_daba_lite_chunks const &other)
			requires std::is_move_assignable_v<Operation>
		{
			if (this != &other) {
				aggregate_queue_by_daba_lite_chunks copy(other);
				*this = std::move(copy);
			}
			return *this;
		}

		constexpr aggregate_queue_by_daba_lite_chunks &operator=(aggregate_queue_by_daba_lite_chunks &&) = default;

		/** @brief Append @p value; a needed chunk is allocated first, so a failed allocation changes nothing. @pre size() is below the chunk limit (asserted). */
		constexpr void push_back(Value value) {
			entries_.prepare_push();
			Value aggregate = value;
			back_aggregate_ = atl::detail::aggregate_combine(operation_, back_aggregate_, value);
			entries_.push_back({std::move(value), std::move(aggregate)});
			repair();
		}

		/**
		 * @brief Append `Value(arguments...)`, built before any allocation, so the arguments may refer to front() or back().
		 * @tparam Arguments Constructor arguments of `Value`.
		 */
		template<typename... Arguments>
			requires std::constructible_from<Value, Arguments...>
		constexpr void emplace_back(Arguments &&...arguments) { push_back(Value(std::forward<Arguments>(arguments)...)); }

		/** @brief Remove the first value. @pre The queue is nonempty (asserted). */
		constexpr void pop_front() {
			atl_assert(!empty());
			entries_.pop_front();
			repair();
		}

		/** @brief The first value. @pre The queue is nonempty (asserted). */
		[[nodiscard]] constexpr Value const &front() const {
			atl_assert(!empty());
			return entries_.at(entries_.begin()).value;
		}

		/** @brief The last value. @pre The queue is nonempty (asserted). */
		[[nodiscard]] constexpr Value const &back() const {
			atl_assert(!empty());
			return entries_.at(storage::previous(entries_.end())).value;
		}

		/** @brief The fold from front to back, or the identity when empty; one operation call; allocates no container storage. */
		[[nodiscard]] constexpr Value fold() const {
			return atl::detail::aggregate_combine(operation_, entries_.back == entries_.begin() ? identity_ : entries_.at(entries_.begin()).aggregate, back_aggregate_);
		}

		/** @brief Number of live values. */
		constexpr size_type size() const noexcept { return entries_.size(); }

		/** @brief Whether no values are live. */
		constexpr bool empty() const noexcept { return entries_.empty(); }

		/** @brief Destroy the live values in O(n), keeping every chunk, the identity and the operation. */
		constexpr void clear() {
			entries_.clear();
			entries_.left = entries_.right = entries_.accumulator = entries_.back = entries_.end();
			bridge_ = back_aggregate_ = identity_;
		}

		/** @brief Allocate chunks for @p elements simultaneously live values in O(elements). @pre @p elements fits the chunk limit (asserted). */
		constexpr void reserve(size_type elements) { entries_.reserve(elements); }
	};

	/**
	 * @brief aggregate_queue contender for minimum or maximum only, by monotonic candidates: instead of a fold per
	 * value it keeps the values that can still become the extremum, each with the number of live values that expire
	 * through it. Not measured yet.
	 * @tparam Value Copyable value type.
	 * @tparam Compare Strict ordering called on `Value const &` through a const callable; std::less selects the
	 *         minimum, std::greater the maximum.
	 * @pre Compare is a strict weak ordering on the stored values, and the identity is neutral for the selected
	 * extremum (it does not precede any stored value). Moves and copies of `Value` and comparisons do not throw during a
	 * mutation.
	 * @complexity push_back O(1) amortized and O(n) worst case (comparisons against pruned candidates); pop_front,
	 * front(), back() and fold() O(1) without comparisons. Storage: the n raw values (for front() and back()) and at
	 * most n candidates; retained storage O(H + reserved elements). Copies O(n); moves O(1) container work, leaving
	 * the source only destructible or assignable.
	 * @note Counts instead of arrival indices keep duplicates apart without equality and cannot overflow. Only FIFO
	 * eviction: it has no pop_back. Not a general idempotent fold (gcd or bitwise and do not fit a strict ordering).
	 * A failed allocation throws and leaves the contents unchanged. The second template argument is an ordering, not
	 * a monoid operation.
	 */
	template<std::copyable Value, typename Compare = std::less<>>
		requires std::strict_weak_order<Compare const &, Value const &, Value const &>
	class aggregate_queue_by_monotonic_candidates {
		struct candidate {
			Value value;	  ///< The extremum of the live values from the preceding candidate through this one.
			std::size_t span; ///< Number of those live values, including this candidate's own; positive.
		};

		detail::aggregate_ring<Value> values_;
		detail::aggregate_ring<candidate> candidates_;
		Value identity_;
		[[no_unique_address]] Compare compare_;

		/** @brief The smaller element limit of the two rings. */
		[[nodiscard]] constexpr std::size_t max_size() const noexcept {
			return values_.max_size() < candidates_.max_size() ? values_.max_size() : candidates_.max_size();
		}

	public:
		using value_type = Value;	   ///< Stored element type.
		using size_type = std::size_t; ///< Element count type.

		/**
		 * @brief An empty queue.
		 * @param identity The neutral element of the selected extremum; fold() returns it while the queue is empty.
		 * @param compare The ordering, stored by value.
		 */
		constexpr explicit aggregate_queue_by_monotonic_candidates(Value identity, Compare compare = Compare{}) : identity_(std::move(identity)), compare_(std::move(compare)) {}

		constexpr aggregate_queue_by_monotonic_candidates(aggregate_queue_by_monotonic_candidates const &) = default;

		constexpr aggregate_queue_by_monotonic_candidates(aggregate_queue_by_monotonic_candidates &&) = default;

		/** @brief Copy into a temporary, then commit by move, so a failed allocation leaves this object unchanged. */
		constexpr aggregate_queue_by_monotonic_candidates &operator=(aggregate_queue_by_monotonic_candidates const &other)
			requires std::is_move_assignable_v<Compare>
		{
			if (this != &other) {
				aggregate_queue_by_monotonic_candidates copy(other);
				*this = std::move(copy);
			}
			return *this;
		}

		constexpr aggregate_queue_by_monotonic_candidates &operator=(aggregate_queue_by_monotonic_candidates &&) = default;

		/**
		 * @brief Append @p value, dropping the candidates it strictly precedes; both rings grow first, so a failed
		 * allocation changes nothing.
		 * @pre size() is below the ring limit (asserted).
		 */
		constexpr void push_back(Value value) {
			atl_assert(size() < max_size());
			values_.prepare(size() + 1, max_size());
			candidates_.prepare(size() + 1, max_size());
			std::size_t span = 1;
			while (!candidates_.empty() && std::invoke(std::as_const(compare_), std::as_const(value), std::as_const(candidates_[candidates_.size() - 1].value))) {
				atl_assert(candidates_[candidates_.size() - 1].span <= max_size() - span);
				span += candidates_[candidates_.size() - 1].span;
				candidates_.pop_back();
			}
			Value selected = value;
			values_.push_back(std::move(value));
			candidates_.push_back({std::move(selected), span});
		}

		/**
		 * @brief Append `Value(arguments...)`, built before any growth, so the arguments may refer to front() or back().
		 * @tparam Arguments Constructor arguments of `Value`.
		 */
		template<typename... Arguments>
			requires std::constructible_from<Value, Arguments...>
		constexpr void emplace_back(Arguments &&...arguments) { push_back(Value(std::forward<Arguments>(arguments)...)); }

		/** @brief Remove the first value. @pre The queue is nonempty (asserted). */
		constexpr void pop_front() {
			atl_assert(!empty());
			values_.pop_front();
			if (--candidates_[0].span == 0) candidates_.pop_front();
		}

		/** @brief The first value. @pre The queue is nonempty (asserted). */
		[[nodiscard]] constexpr Value const &front() const {
			atl_assert(!empty());
			return values_[0];
		}

		/** @brief The last value. @pre The queue is nonempty (asserted). */
		[[nodiscard]] constexpr Value const &back() const {
			atl_assert(!empty());
			return values_[size() - 1];
		}

		/** @brief The selected extremum of the live values, or the identity when empty; no comparison. */
		[[nodiscard]] constexpr Value fold() const { return empty() ? identity_ : candidates_[0].value; }

		/** @brief Number of live values. */
		constexpr size_type size() const noexcept { return values_.size(); }

		/** @brief Whether no values are live. */
		constexpr bool empty() const noexcept { return values_.empty(); }

		/** @brief Destroy the live values in O(n), keeping both rings, the identity and the ordering. */
		constexpr void clear() {
			values_.clear();
			candidates_.clear();
		}

		/** @brief Allocate both rings for @p elements simultaneously live values. @pre @p elements fits both rings (asserted). */
		constexpr void reserve(size_type elements) {
			atl_assert(elements <= max_size());
			values_.reserve(elements);
			candidates_.reserve(elements);
		}
	};
}
