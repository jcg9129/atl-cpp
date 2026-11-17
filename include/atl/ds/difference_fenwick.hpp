#pragma once

/**
 * @file
 * @brief Range updates and point queries over a commutative group.
 */

#include <atl/ds/binary_indexed_tree.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Range apply / point get for tags that form a commutative group acting by
	 *        the group operation itself (add, xor): a @ref atl::binary_indexed_tree
	 *        over the difference array.
	 *
	 * Applying `f` to `[l, r)` combines `f` into cell `l` and `inverse(f)` into
	 * cell `r`; the value at `i` is the prefix fold over `[0, i + 1)`. Against the
	 * commutative @ref atl::dual_segment_tree, it uses one Fenwick array with no
	 * separate base-value array. It reuses @ref atl::binary_indexed_tree.
	 *
	 * The concepts check callable signatures and closure only. Associativity,
	 * commutativity, identity and inverse laws remain caller obligations.
	 * Signed arithmetic, including differences and inverses, must fit Value.
	 * This is not a general tag action: tags and values are the same group.
	 *
	 * @tparam Value   The element type, which is also the tag type.
	 * @tparam Op      The group operation, commutative (`std::plus<>`, `std::bit_xor<>`).
	 * @tparam Inverse The group inverse (`std::negate<>`; `Value`-identity for xor).
	 *
	 * @pre `(Value, op, identity)` is a commutative group with the given inverse; unchecked.
	 */
	template<std::copyable Value, binary_operation<Value> Op = std::plus<>, inverse_operation<Value> Inverse = std::negate<>>
		requires std::copy_constructible<Op> && std::move_constructible<Inverse>
	class difference_fenwick {
		binary_indexed_tree<Value, 1, Op> diff_;
		[[no_unique_address]] Inverse inverse_;

		/** @brief Convert an input element, asserting representability for built-in integer conversions. */
		template<std::convertible_to<Value> Element>
		static constexpr Value checked_value(Element &&value) {
			if constexpr (std::integral<std::remove_cvref_t<Element>> && std::integral<Value>) {
				atl_assert(!std::cmp_less(+value, +std::numeric_limits<Value>::lowest()));
				atl_assert(!std::cmp_greater(+value, +(std::numeric_limits<Value>::max)()));
			}
			return static_cast<Value>(std::forward<Element>(value));
		}

		/** @brief The differences `a[i] op inverse(a[i - 1])` of @p values. */
		template<typename R>
			requires std::ranges::sized_range<R const> && std::ranges::input_range<R const>
				  && std::convertible_to<std::ranges::range_reference_t<R const>, Value>
		static constexpr std::vector<Value> differences(R const &values, Op const &op, Inverse const &inverse) {
			std::vector<Value> out;
			out.reserve(std::ranges::size(values));
			auto it = std::ranges::begin(values);
			auto const last = std::ranges::end(values);
			if (it == last) return out;
			Value prev = checked_value(*it);
			out.push_back(prev);
			for (++it; it != last; ++it) {
				Value cur = checked_value(*it);
				out.push_back(checked_value(op(cur, checked_value(inverse(prev)))));
				prev = std::move(cur);
			}
			return out;
		}

	public:
		/** @brief The element and update type. */
		using value_type = Value;
		/** @brief The index and size type. */
		using size_type = std::size_t;

		/** @brief @p n elements, all equal to @p fill. @complexity O(n) */
		constexpr difference_fenwick(std::size_t n, Value const &fill, Value identity, Op op = Op{}, Inverse inverse = Inverse{})
			: diff_(n, std::move(identity), std::move(op)), inverse_(std::move(inverse)) {
			if (n > 0) diff_.update(fill, 0);
		}

		/** @brief From initial values. @complexity O(n) */
		template<typename R>
			requires std::ranges::sized_range<R const> && std::ranges::input_range<R const>
					  && std::convertible_to<std::ranges::range_reference_t<R const>, Value>
		constexpr difference_fenwick(R const &values, Value identity, Op op = Op{}, Inverse inverse = Inverse{})
			: diff_(differences(values, op, inverse), std::move(identity), op), inverse_(std::move(inverse)) {}

		/**
		 * @brief Combine @p f into every element of `[l, r)`.
		 * @pre `l <= r <= size()`
		 * @complexity O(log n)
		 */
		constexpr void apply(std::size_t l, std::size_t r, Value const &f) {
			atl_assert(l <= r && r <= size());
			if (l == r) return;
			diff_.update(f, l);
			if (r < size()) diff_.update(checked_value(inverse_(f)), r);
		}

		/**
		 * @brief Combine @p f into the single element at @p i.
		 * @pre `i < size()`
		 * @complexity O(log n)
		 */
		constexpr void apply(std::size_t i, Value const &f) {
			atl_assert(i < size());
			apply(i, i + 1, f);
		}

		/**
		 * @brief The value at position @p i.
		 * @pre `i < size()`
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr Value at(std::size_t i) const {
			atl_assert(i < size());
			return diff_.prefix_fold(i + 1);
		}

		/** @brief Number of elements. */
		constexpr std::size_t size() const noexcept { return diff_.size(); }

		/** @brief Whether there are no elements. */
		constexpr bool empty() const noexcept { return size() == 0; }
	};
}
