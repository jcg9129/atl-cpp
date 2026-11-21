#pragma once
#include <atl/types/alias.hpp>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>

namespace atl::detail {
	/**
	 * @brief Whether the integral symbol @p sym lies in `[0, bound)`, decided at the symbol's
	 *        own width: a negative value or one wider than 64 bits is out of range and never aliases
	 *        a smaller symbol through narrowing. Callers narrow to `uint32` only after it returns true.
	 * @tparam Sym A built-in integral type (`bool`, the character types and 128-bit integers included).
	 * @param sym The symbol to test.
	 * @param bound The exclusive upper bound.
	 */
	template<std::integral Sym>
	[[nodiscard]] constexpr bool symbol_below(Sym const sym, uint64 const bound) noexcept {
		if constexpr (std::is_same_v<Sym, bool>) {
			return uint64(sym) < bound;
		} else {
			if constexpr (std::is_signed_v<Sym>) {
				if (sym < Sym(0)) return false;
			}
			using unsigned_symbol = std::make_unsigned_t<Sym>;
			using wide = std::common_type_t<unsigned_symbol, uint64>;
			return wide(unsigned_symbol(sym)) < wide(bound);
		}
	}

	/**
	 * @brief Whether `a < b` is decided by value when both operands are integral:
	 *        true for two integral types other than `bool` (the character types and 128-bit
	 *        integers included) that differ in signedness, where the built-in comparison would
	 *        convert the signed operand to unsigned.
	 */
	template<typename A, typename B>
	inline constexpr bool is_mixed_sign_integral_pair =
		std::integral<A> && std::integral<B> && !std::is_same_v<A, bool> && !std::is_same_v<B, bool>
		&& std::is_signed_v<A> != std::is_signed_v<B>;

	/**
	 * @brief `a < b` compared by mathematical value for integral operands of mixed signedness
	 *        (a negative value is below every unsigned one); the built-in operator otherwise.
	 * @tparam A Left symbol type, `std::totally_ordered_with` `B`.
	 * @tparam B Right symbol type.
	 */
	template<typename A, typename B>
	[[nodiscard]] constexpr bool symbol_less(A const &a, B const &b) {
		if constexpr (is_mixed_sign_integral_pair<A, B>) {
			if constexpr (std::is_signed_v<A>) {
				return a < A(0) || std::make_unsigned_t<A>(a) < b;
			} else {
				return !(b < B(0)) && a < std::make_unsigned_t<B>(b);
			}
		} else {
			return a < b;
		}
	}

	/**
	 * @brief `a == b` compared by mathematical value for integral operands of mixed signedness
	 *        (a negative value equals no unsigned one); the built-in operator otherwise.
	 * @tparam A Left symbol type, `std::equality_comparable_with` `B`.
	 * @tparam B Right symbol type.
	 */
	template<typename A, typename B>
	[[nodiscard]] constexpr bool symbol_equal(A const &a, B const &b) {
		if constexpr (is_mixed_sign_integral_pair<A, B>) {
			if constexpr (std::is_signed_v<A>) {
				return !(a < A(0)) && std::make_unsigned_t<A>(a) == b;
			} else {
				return !(b < B(0)) && a == std::make_unsigned_t<B>(b);
			}
		} else {
			return a == b;
		}
	}

	/**
	 * @brief The elements of the input range @p range as a range-for target, found through
	 *        `std::ranges::begin` and `std::ranges::end`, so a range whose `end` is found only by
	 *        argument-dependent lookup iterates like any other. The elements are the range's own
	 *        references, in order; an input-only range is traversed once.
	 * @tparam R An input range.
	 * @param range The range to iterate; it must outlive the returned view.
	 */
	template<std::ranges::input_range R>
	[[nodiscard]] constexpr auto input_elements(R &range) {
		return std::ranges::subrange(std::ranges::begin(range), std::ranges::end(range));
	}
}
