#pragma once

#include <atl/ds/detail/aggregate_stack.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace atl {
	/**
	 * @brief Mutable FIFO with an ordered monoid fold: push at the back, pop at the front, and read the fold of the
	 * live values from front to back in O(1).
	 *
	 * Two stacks of (value, fold) entries. A pop from an empty front stack first moves the whole back stack onto it,
	 * so every value crosses once.
	 *
	 * @tparam Value Copyable element type; neither default construction nor equality is needed.
	 * @tparam Operation Binary operation on `Value`, called on `Value const &` operands through a const callable
	 *         (@ref atl::binary_operation); its result is converted to `Value` explicitly.
	 * @pre `(Value, Operation, identity)` is a monoid: the operation is associative and the identity is neutral on
	 * both sides. Commutativity is not required. Every intermediate fold is valid user arithmetic (no overflow).
	 * Moves and copies of `Value`, calls of the operation and the conversion of its result to `Value` do not
	 * throw during a mutation.
	 * @complexity push_back O(1) amortized (vector growth), O(n) worst case; pop_front O(1) amortized, O(n) worst case (the transfer); readers O(1) with at
	 * most one operation call. Live storage O(n); retained storage O(H + reserved elements) for the largest
	 * occupancy H so far. Copies are deep, O(n); moves are O(1) container work and leave the source only
	 * destructible or assignable.
	 * @note A failed allocation throws and leaves the contents unchanged. References from front() and back() are
	 * valid until the next mutation, reserve, move or assignment of this object.
	 */
	template<std::copyable Value, binary_operation<Value> Operation>
	class aggregate_queue {
		detail::aggregate_stack<Value, true> left_;
		detail::aggregate_stack<Value, false> right_;
		Value identity_;
		[[no_unique_address]] Operation operation_;

	public:
		using value_type = Value;	   ///< Stored element type.
		using size_type = std::size_t; ///< Element count type.

		/**
		 * @brief An empty queue.
		 * @param identity The neutral element; fold() returns it while the queue is empty.
		 * @param operation The monoid operation, stored by value (a capturing lambda needs no default constructor).
		 */
		constexpr explicit aggregate_queue(Value identity, Operation operation = Operation{}) : identity_(std::move(identity)), operation_(std::move(operation)) {}

		/** @brief Deep copy, O(n). */
		constexpr aggregate_queue(aggregate_queue const &) = default;

		/** @brief O(1) container work; the source is left only destructible or assignable. */
		constexpr aggregate_queue(aggregate_queue &&) = default;

		/** @brief Copy into a temporary, then commit by move, so a failed allocation leaves this object unchanged. */
		constexpr aggregate_queue &operator=(aggregate_queue const &other)
			requires std::is_move_assignable_v<Operation>
		{
			if (this != &other) {
				aggregate_queue copy(other);
				*this = std::move(copy);
			}
			return *this;
		}

		/** @brief O(1) container work plus releasing this object's values; the source is left only destructible or assignable. */
		constexpr aggregate_queue &operator=(aggregate_queue &&) = default;

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
		 * @brief Append `Value(arguments...)`. The value is built before any storage changes, so the arguments may
		 * refer to front() or back().
		 * @tparam Arguments Constructor arguments of `Value`.
		 */
		template<typename... Arguments>
			requires std::constructible_from<Value, Arguments...>
		constexpr void emplace_back(Arguments &&...arguments) { push_back(Value(std::forward<Arguments>(arguments)...)); }

		/**
		 * @brief Remove the first value.
		 * @pre The queue is nonempty (asserted).
		 * @complexity O(1) amortized; O(n) and n operation calls when the front stack is empty and the back stack
		 * is transferred. The transfer allocates first, so a failed allocation leaves the queue unchanged.
		 */
		constexpr void pop_front() {
			atl_assert(!empty());
			if (left_.empty()) {
				left_.reserve(right_.size());
				while (!right_.empty()) {
					left_.push(std::move(right_.top()), operation_);
					right_.pop();
				}
			}
			left_.pop();
		}

		/** @brief The first value. @pre The queue is nonempty (asserted). @complexity O(1) */
		[[nodiscard]] constexpr Value const &front() const {
			atl_assert(!empty());
			return left_.empty() ? right_.bottom() : left_.top();
		}

		/** @brief The last value. @pre The queue is nonempty (asserted). @complexity O(1) */
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

		/** @brief The largest size() any push or reserve may reach: the smaller element limit of the two backing vectors. */
		[[nodiscard]] constexpr size_type max_size() const noexcept { return left_.max_size() < right_.max_size() ? left_.max_size() : right_.max_size(); }

		/** @brief Destroy the live values in O(n), keeping the allocation, identity and operation. */
		constexpr void clear() {
			left_.clear();
			right_.clear();
		}

		/**
		 * @brief Allocate for @p elements simultaneously live values, including the transfer storage, so no later
		 * operation allocates while size() stays at most @p elements.
		 * @pre @p elements is at most max_size() (asserted).
		 * @complexity O(n + elements); contents do not change, and a failed allocation leaves them intact.
		 */
		constexpr void reserve(size_type elements) {
			atl_assert(elements <= max_size());
			left_.reserve(elements);
			right_.reserve(elements);
		}
	};
}
