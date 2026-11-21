#pragma once

#include <atl/ds/detail/aggregate_stack.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Mutable deque with an ordered monoid fold: push and pop at both ends, and read the fold of the live
	 * values from front to back in O(1).
	 *
	 * Two stacks of (value, fold) entries meeting in the middle. A pop from an empty side splits the other side in
	 * half (the requested side receives ceil(n/2) values), so alternating drains stay linear in total.
	 *
	 * @tparam Value Copyable element type; neither default construction nor equality is needed.
	 * @tparam Operation Binary operation on `Value`, called on `Value const &` operands through a const callable
	 *         (@ref atl::binary_operation); its result is converted to `Value` explicitly.
	 * @pre `(Value, Operation, identity)` is a monoid: the operation is associative and the identity is neutral on
	 * both sides. Commutativity is not required. Every intermediate fold is valid user arithmetic (no overflow).
	 * Moves and copies of `Value`, calls of the operation and the conversion of its result to `Value` do not
	 * throw during a mutation.
	 * @complexity Pushes O(1) amortized (vector growth), O(n) worst case; pops O(1) amortized, O(n) worst case (the split); readers
	 * O(1) with at most one operation call. Live storage O(n) plus a split buffer that is empty between operations;
	 * retained storage O(H + reserved elements) for the largest occupancy H so far. Copies are deep, O(n); moves
	 * are O(1) container work and leave the source only destructible or assignable.
	 * @note A failed allocation throws and leaves the contents unchanged. References from front() and back() are
	 * valid until the next mutation, reserve, move or assignment of this object.
	 */
	template<std::copyable Value, binary_operation<Value> Operation>
	class aggregate_deque {
		detail::aggregate_stack<Value, true> left_;
		detail::aggregate_stack<Value, false> right_;
		std::vector<Value> scratch_; ///< Empty between operations; keeps its capacity for the next split.
		Value identity_;
		[[no_unique_address]] Operation operation_;

		/**
		 * @brief Split the only occupied side in half, the side @p needs_front names receiving ceil(n/2) values.
		 * Every destination is reserved before the first value moves, so a failed allocation changes nothing.
		 */
		constexpr void rebalance(bool needs_front) {
			auto const n = size();
			auto const left_size = needs_front ? n - n / 2 : n / 2;
			scratch_.reserve(n);
			left_.reserve(left_size);
			right_.reserve(n - left_size);
			// Popping the occupied side fills scratch_ in reverse logical order from the right side, in logical
			// order from the left side; each stack is then rebuilt bottom first.
			if (left_.empty()) {
				while (!right_.empty()) {
					scratch_.push_back(std::move(right_.top()));
					right_.pop();
				}
				for (std::size_t i = n - left_size; i < n; ++i) left_.push(std::move(scratch_[i]), operation_);
				for (std::size_t i = n - left_size; i > 0; --i) right_.push(std::move(scratch_[i - 1]), operation_);
			} else {
				while (!left_.empty()) {
					scratch_.push_back(std::move(left_.top()));
					left_.pop();
				}
				for (std::size_t i = left_size; i > 0; --i) left_.push(std::move(scratch_[i - 1]), operation_);
				for (std::size_t i = left_size; i < n; ++i) right_.push(std::move(scratch_[i]), operation_);
			}
			scratch_.clear();
		}

	public:
		using value_type = Value;	   ///< Stored element type.
		using size_type = std::size_t; ///< Element count type.

		/**
		 * @brief An empty deque.
		 * @param identity The neutral element; fold() returns it while the deque is empty.
		 * @param operation The monoid operation, stored by value (a capturing lambda needs no default constructor).
		 */
		constexpr explicit aggregate_deque(Value identity, Operation operation = Operation{}) : identity_(std::move(identity)), operation_(std::move(operation)) {}

		/** @brief Deep copy, O(n). */
		constexpr aggregate_deque(aggregate_deque const &) = default;

		/** @brief O(1) container work; the source is left only destructible or assignable. */
		constexpr aggregate_deque(aggregate_deque &&) = default;

		/** @brief Copy into a temporary, then commit by move, so a failed allocation leaves this object unchanged. */
		constexpr aggregate_deque &operator=(aggregate_deque const &other)
			requires std::is_move_assignable_v<Operation>
		{
			if (this != &other) {
				aggregate_deque copy(other);
				*this = std::move(copy);
			}
			return *this;
		}

		/** @brief O(1) container work plus releasing this object's values; the source is left only destructible or assignable. */
		constexpr aggregate_deque &operator=(aggregate_deque &&) = default;

		/**
		 * @brief Prepend @p value.
		 * @pre size() is below max_size() (asserted).
		 * @complexity O(1) amortized, O(n) worst case (vector growth); one operation call.
		 */
		constexpr void push_front(Value value) {
			atl_assert(size() < max_size());
			left_.push(std::move(value), operation_);
		}

		/**
		 * @brief Append @p value.
		 * @pre size() is below max_size() (asserted).
		 * @complexity O(1) amortized, O(n) worst case (vector growth); one operation call.
		 */
		constexpr void push_back(Value value) {
			atl_assert(size() < max_size());
			right_.push(std::move(value), operation_);
		}

		/**
		 * @brief Prepend `Value(arguments...)`. The value is built before any storage changes, so the arguments may
		 * refer to front() or back().
		 * @tparam Arguments Constructor arguments of `Value`.
		 */
		template<typename... Arguments>
			requires std::constructible_from<Value, Arguments...>
		constexpr void emplace_front(Arguments &&...arguments) { push_front(Value(std::forward<Arguments>(arguments)...)); }

		/**
		 * @brief Append `Value(arguments...)`. The value is built before any storage changes, so the arguments may
		 * refer to front() or back().
		 * @tparam Arguments Constructor arguments of `Value`.
		 */
		template<typename... Arguments>
			requires std::constructible_from<Value, Arguments...>
		constexpr void emplace_back(Arguments &&...arguments) { push_back(Value(std::forward<Arguments>(arguments)...)); }

		/**
		 * @brief Remove the first value.
		 * @pre The deque is nonempty (asserted).
		 * @complexity O(1) amortized; O(n) and n operation calls when the front side is empty and the back side
		 * (of two or more values) is split. The split allocates first, so a failed allocation leaves the deque
		 * unchanged.
		 */
		constexpr void pop_front() {
			atl_assert(!empty());
			if (left_.empty() && right_.size() == 1) {
				right_.pop();
				return;
			}
			if (left_.empty()) rebalance(true);
			left_.pop();
		}

		/**
		 * @brief Remove the last value.
		 * @pre The deque is nonempty (asserted).
		 * @complexity O(1) amortized; O(n) and n operation calls when the back side is empty and the front side
		 * (of two or more values) is split. The split allocates first, so a failed allocation leaves the deque
		 * unchanged.
		 */
		constexpr void pop_back() {
			atl_assert(!empty());
			if (right_.empty() && left_.size() == 1) {
				left_.pop();
				return;
			}
			if (right_.empty()) rebalance(false);
			right_.pop();
		}

		/** @brief The first value. @pre The deque is nonempty (asserted). @complexity O(1) */
		[[nodiscard]] constexpr Value const &front() const {
			atl_assert(!empty());
			return left_.empty() ? right_.bottom() : left_.top();
		}

		/** @brief The last value. @pre The deque is nonempty (asserted). @complexity O(1) */
		[[nodiscard]] constexpr Value const &back() const {
			atl_assert(!empty());
			return right_.empty() ? left_.bottom() : right_.top();
		}

		/**
		 * @brief The fold of the live values from front to back, or the identity when empty.
		 * @complexity O(1), one operation call; allocates no container storage and does no rebalancing. The
		 * operation and the result's construction may allocate.
		 */
		[[nodiscard]] constexpr Value fold() const { return detail::aggregate_combine(operation_, left_.fold(identity_), right_.fold(identity_)); }

		/** @brief Number of live values. */
		constexpr size_type size() const noexcept { return left_.size() + right_.size(); }

		/** @brief Whether no values are live. */
		constexpr bool empty() const noexcept { return left_.empty() && right_.empty(); }

		/** @brief The largest size() any push or reserve may reach: the smallest element limit of the three backing vectors. */
		[[nodiscard]] constexpr size_type max_size() const noexcept {
			auto const stack_limit = left_.max_size() < right_.max_size() ? left_.max_size() : right_.max_size();
			return stack_limit < scratch_.max_size() ? stack_limit : scratch_.max_size();
		}

		/** @brief Destroy the live values in O(n), keeping the allocation, identity and operation. */
		constexpr void clear() {
			left_.clear();
			right_.clear();
		}

		/**
		 * @brief Allocate for @p elements simultaneously live values, including the split buffer, so no later
		 * operation allocates while size() stays at most @p elements.
		 * @pre @p elements is at most max_size() (asserted).
		 * @complexity O(n + elements); contents do not change, and a failed allocation leaves them intact.
		 */
		constexpr void reserve(size_type elements) {
			atl_assert(elements <= max_size());
			left_.reserve(elements);
			right_.reserve(elements);
			scratch_.reserve(elements);
		}
	};
}
