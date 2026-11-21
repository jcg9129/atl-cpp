#pragma once

#include <atl/types/algebraic.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace atl::detail {
	/**
	 * @brief `operation(left, right)` on const operands, converted to `Value` explicitly.
	 *
	 * binary_operation checks the call on `Value const &` operands, so the call passes exactly those (an overload
	 * for mutable operands is never selected). The explicit conversion keeps a narrow `Value` (`int16_t` under
	 * std::plus, which yields `int`) free of implicit-narrowing warnings.
	 */
	template<std::copyable Value, binary_operation<Value> Operation>
	[[nodiscard]] constexpr Value aggregate_combine(Operation const &operation, Value const &left, Value const &right) {
		return static_cast<Value>(std::invoke(operation, left, right));
	}

	/** @brief A raw endpoint value and its directional stack aggregate. */
	template<std::copyable Value>
	struct aggregate_entry {
		Value value;	 ///< Unmodified element, read by front/back and moved on redistribution.
		Value aggregate; ///< Ordered fold of this entry and every entry below it, in logical order.
	};

	/**
	 * @brief Contiguous stack with a fold per entry. With IsFront the top is the logical first element and an entry
	 * folds itself, then the entries below; otherwise the top is the logical last element and an entry folds the
	 * entries below, then itself.
	 * @pre Value transfers, operation evaluation and the conversion of its result to `Value` during mutation do not
	 * throw.
	 */
	template<std::copyable Value, bool IsFront>
	class aggregate_stack {
		std::vector<aggregate_entry<Value>> entries_; ///< Bottom to top.

	public:
		constexpr std::size_t size() const noexcept { return entries_.size(); }

		constexpr bool empty() const noexcept { return entries_.empty(); }

		[[nodiscard]] constexpr std::size_t max_size() const noexcept { return entries_.max_size(); }

		/** @brief Allocate without changing contents; allocation failure leaves the stack unchanged. */
		constexpr void reserve(std::size_t elements) {
			atl_assert(elements <= max_size());
			entries_.reserve(elements);
		}

		/** @brief Push a materialized value; at most one operation call. */
		template<binary_operation<Value> Operation>
		constexpr void push(Value value, Operation const &operation) {
			atl_assert(size() < max_size());
			Value aggregate = [&]() -> Value {
				if (entries_.empty()) return value;
				if constexpr (IsFront)
					return aggregate_combine(operation, value, entries_.back().aggregate);
				else
					return aggregate_combine(operation, entries_.back().aggregate, value);
			}();
			entries_.push_back({std::move(value), std::move(aggregate)});
		}

		constexpr void pop() {
			atl_assert(!empty());
			entries_.pop_back();
		}

		/** @brief Mutable top value, so redistribution can move it out before pop(). */
		[[nodiscard]] constexpr Value &top() {
			atl_assert(!empty());
			return entries_.back().value;
		}

		[[nodiscard]] constexpr Value const &top() const {
			atl_assert(!empty());
			return entries_.back().value;
		}

		[[nodiscard]] constexpr Value const &bottom() const {
			atl_assert(!empty());
			return entries_.front().value;
		}

		/** @brief Fold of the whole stack in logical order, or @p identity when empty. */
		[[nodiscard]] constexpr Value const &fold(Value const &identity) const {
			// NOLINTNEXTLINE(bugprone-return-const-ref-from-parameter): callers pass a member identity that outlives the result.
			return empty() ? identity : entries_.back().aggregate;
		}

		/** @brief Destroy entries, retaining allocation. */
		constexpr void clear() { entries_.clear(); }
	};
}
