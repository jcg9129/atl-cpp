#pragma once

#include <atl/config.hpp>
#include <atl/utils/assert.hpp>
#include <bit>
#include <climits>
#include <cstdint>
#if (ATL_HAS_INT128 || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64)))) && !defined(ATL_PORTABLE_WIDE_ARITHMETIC) && !defined(ATL_FORCE_FALLBACKS)
#define ATL_DETAIL_WIDE_RUNTIME_BACKEND 1
#else
#define ATL_DETAIL_WIDE_RUNTIME_BACKEND 0
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
#include <intrin.h>
#endif

namespace atl::detail {
	// This flag shares the dispatch guard so forced builds can verify their backend.
	inline constexpr bool uses_portable_wide_arithmetic = !ATL_DETAIL_WIDE_RUNTIME_BACKEND;
	static_assert(CHAR_BIT == 8 && sizeof(std::uint32_t) == 4 && sizeof(std::uint64_t) == 8);

	/** @brief Little-endian limbs of an unsigned 128-bit value. */
	struct word_pair {
		std::uint64_t low;	///< Low word.
		std::uint64_t high; ///< High word.
	};

	/** @brief Sum word and outgoing carry bit. */
	struct carry_result {
		std::uint64_t value; ///< Low word of the sum.
		std::uint64_t carry; ///< Outgoing carry, zero or one.
	};

	/** @brief Difference word and outgoing borrow bit. */
	struct borrow_result {
		std::uint64_t value;  ///< Low word of the difference.
		std::uint64_t borrow; ///< Outgoing borrow, zero or one.
	};

	// Each primitive pairs a portable body (constant evaluation, targets without a fast path) with the run-time
	// fast path as `if consteval { portable } else { fast }`: code after an always-returning `if !consteval` block
	// would be unreachable in the run-time instantiation, which MSVC reports (C4702) and /WX rejects.
#if ATL_DETAIL_WIDE_RUNTIME_BACKEND && (ATL_HAS_INT128 || (defined(_MSC_VER) && defined(_M_X64)))
#define ATL_DETAIL_WIDE_CARRY_DIVIDE_FAST 1
#else
#define ATL_DETAIL_WIDE_CARRY_DIVIDE_FAST 0
#endif

	/** @brief Portable body of add_with_carry. */
	[[nodiscard]] constexpr carry_result add_with_carry_portable(std::uint64_t left, std::uint64_t right, std::uint64_t carry) {
		auto const sum = left + right;
		auto const value = sum + carry;
		return {value, std::uint64_t(sum < left || value < sum)};
	}

	/** @brief Add words with a carry bit. @pre `carry <= 1`. */
	[[nodiscard]] constexpr carry_result add_with_carry(std::uint64_t left, std::uint64_t right, std::uint64_t carry = 0) {
		atl_assert(carry <= 1);
#if ATL_DETAIL_WIDE_CARRY_DIVIDE_FAST
		if consteval {
			return add_with_carry_portable(left, right, carry);
		} else {
#if ATL_HAS_INT128
			auto const sum = static_cast<__uint128_t>(left) + right + carry;
			return {std::uint64_t(sum), std::uint64_t(sum >> 64)};
#else
			unsigned __int64 value;
			auto const next = _addcarry_u64(static_cast<unsigned char>(carry), left, right, &value);
			return {value, next};
#endif
		}
#else
		return add_with_carry_portable(left, right, carry);
#endif
	}

	/** @brief Portable body of subtract_with_borrow. */
	[[nodiscard]] constexpr borrow_result subtract_with_borrow_portable(std::uint64_t left, std::uint64_t right, std::uint64_t borrow) {
		auto const difference = left - right;
		return {difference - borrow, std::uint64_t(left < right || difference < borrow)};
	}

	/** @brief Subtract words with a borrow bit. @pre `borrow <= 1`. */
	[[nodiscard]] constexpr borrow_result subtract_with_borrow(std::uint64_t left, std::uint64_t right, std::uint64_t borrow = 0) {
		atl_assert(borrow <= 1);
#if ATL_DETAIL_WIDE_CARRY_DIVIDE_FAST
		if consteval {
			return subtract_with_borrow_portable(left, right, borrow);
		} else {
#if ATL_HAS_INT128
			auto const difference = static_cast<__uint128_t>(left) - right - borrow;
			return {std::uint64_t(difference), std::uint64_t(difference >> 64) & 1};
#else
			unsigned __int64 value;
			auto const next = _subborrow_u64(static_cast<unsigned char>(borrow), left, right, &value);
			return {value, next};
#endif
		}
#else
		return subtract_with_borrow_portable(left, right, borrow);
#endif
	}

	/** @brief Portable body of multiply_wide: four base-2^32 products with explicit carries. */
	[[nodiscard]] constexpr word_pair multiply_wide_portable(std::uint64_t left, std::uint64_t right) {
		std::uint64_t const left_low = std::uint32_t(left), left_high = left >> 32;
		std::uint64_t const right_low = std::uint32_t(right), right_high = right >> 32;
		auto const bottom = left_low * right_low;
		auto const cross_left = left_low * right_high;
		auto const cross_right = left_high * right_low;
		auto const middle = (bottom >> 32) + std::uint32_t(cross_left) + std::uint64_t(std::uint32_t(cross_right));
		return {(middle << 32) | std::uint32_t(bottom), left_high * right_high + (cross_left >> 32) + (cross_right >> 32) + (middle >> 32)};
	}

	/** @brief Exact 64 by 64 bit product. */
	[[nodiscard]] constexpr word_pair multiply_wide(std::uint64_t left, std::uint64_t right) {
#if ATL_DETAIL_WIDE_RUNTIME_BACKEND
		if consteval {
			return multiply_wide_portable(left, right);
		} else {
#if ATL_HAS_INT128
			auto const product = static_cast<__uint128_t>(left) * right;
			return {std::uint64_t(product), std::uint64_t(product >> 64)};
#elif defined(_MSC_VER) && defined(_M_X64)
			unsigned __int64 high;
			auto const low = _umul128(left, right, &high);
			return {low, high};
#else // MSVC ARM64, the remaining run-time backend
			return {left * right, __umulh(left, right)};
#endif
		}
#else
		return multiply_wide_portable(left, right);
#endif
	}

	/** @brief High word of the exact 64 by 64 bit product. */
	[[nodiscard]] constexpr std::uint64_t multiply_high(std::uint64_t left, std::uint64_t right) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64)) && ATL_DETAIL_WIDE_RUNTIME_BACKEND && !ATL_HAS_INT128
		if consteval {
			return multiply_wide_portable(left, right).high;
		} else {
			return __umulh(left, right);
		}
#else
		return multiply_wide(left, right).high;
#endif
	}

	/** @brief Division with a word-sized quotient and remainder. */
	struct word_division_result {
		std::uint64_t quotient;	 ///< Quotient.
		std::uint64_t remainder; ///< Remainder, less than the divisor.
	};

	/**
	 * @brief Append one base-2^32 digit to a remainder and divide by a normalized divisor.
	 * @pre `remainder < divisor`, `divisor >= 2^63`, `digit < 2^32`.
	 */
	[[nodiscard]] constexpr word_division_result divide_normalized_digit(std::uint64_t remainder, std::uint64_t digit, std::uint64_t divisor) {
		atl_assert(remainder < divisor && divisor >> 63 && digit <= UINT32_MAX);
		constexpr std::uint64_t base = std::uint64_t(1) << 32;
		auto const divisor_high = divisor >> 32;
		auto const divisor_low = std::uint32_t(divisor);
		auto quotient = remainder / divisor_high;
		auto estimate_remainder = remainder % divisor_high;
		// The leading divisor digit gives an upper estimate. Testing the next
		// digit corrects it before subtraction; at most two decrements suffice.
		while (quotient >= base || quotient * divisor_low > (estimate_remainder << 32) + digit) {
			--quotient;
			estimate_remainder += divisor_high;
			if (estimate_remainder >= base) break;
		}
		// The exact remainder fits one word; modular word arithmetic cancels
		// the omitted top digit of both the dividend and quotient product.
		return {quotient, (remainder << 32) + digit - quotient * divisor};
	}

	/** @brief Portable body of divide_wide_bounded: normalized base-2^32 long division. */
	[[nodiscard]] constexpr word_division_result divide_wide_bounded_portable(std::uint64_t low, std::uint64_t high, std::uint64_t divisor) {
		int const shift = std::countl_zero(divisor);
		auto const normalized_divisor = divisor << shift;
		auto const normalized_high = shift == 0 ? high : (high << shift) | (low >> (64 - shift));
		auto const normalized_low = low << shift;
		auto const upper = divide_normalized_digit(normalized_high, normalized_low >> 32, normalized_divisor);
		auto const lower = divide_normalized_digit(upper.remainder, std::uint32_t(normalized_low), normalized_divisor);
		return {(upper.quotient << 32) | lower.quotient, lower.remainder >> shift};
	}

	/**
	 * @brief Divide a two-word dividend with a word-sized quotient.
	 * @pre `divisor != 0` and `high < divisor`.
	 */
	[[nodiscard]] constexpr word_division_result divide_wide_bounded(std::uint64_t low, std::uint64_t high, std::uint64_t divisor) {
		atl_assert(divisor != 0 && high < divisor);
#if ATL_DETAIL_WIDE_CARRY_DIVIDE_FAST
		if consteval {
			return divide_wide_bounded_portable(low, high, divisor);
		} else {
#if ATL_HAS_INT128
			auto const dividend = (static_cast<__uint128_t>(high) << 64) | low;
			return {std::uint64_t(dividend / divisor), std::uint64_t(dividend % divisor)};
#else
			unsigned __int64 remainder;
			auto const quotient = _udiv128(high, low, divisor, &remainder);
			return {quotient, remainder};
#endif
		}
#else
		return divide_wide_bounded_portable(low, high, divisor);
#endif
	}

	/** @brief Division with a two-word quotient and word-sized remainder. */
	struct wide_division_result {
		word_pair quotient;		 ///< Full quotient.
		std::uint64_t remainder; ///< Remainder, less than the divisor.
	};

	/** @brief Divide an arbitrary two-word dividend by a word. @pre `divisor != 0`. */
	[[nodiscard]] constexpr wide_division_result divide_wide(std::uint64_t low, std::uint64_t high, std::uint64_t divisor) {
		atl_assert(divisor != 0);
		auto const lower = divide_wide_bounded(low, high % divisor, divisor);
		return {{lower.quotient, high / divisor}, lower.remainder};
	}

	/** @brief High half of an exact 128 by 128 bit product, including the 0/1/2 middle carry. */
	[[nodiscard]] constexpr word_pair multiply_high_wide(word_pair left, word_pair right) {
		auto const bottom = multiply_wide(left.low, right.low);
		auto const cross_left = multiply_wide(left.low, right.high);
		auto const cross_right = multiply_wide(left.high, right.low);
		auto const top = multiply_wide(left.high, right.high);
		auto const middle_first = add_with_carry(bottom.high, cross_left.low);
		auto const middle_second = add_with_carry(middle_first.value, cross_right.low);
		auto const first = add_with_carry(top.low, cross_left.high);
		auto const second = add_with_carry(first.value, cross_right.high);
		auto const third = add_with_carry(second.value, middle_first.carry + middle_second.carry);
		return {third.value, top.high + first.carry + second.carry + third.carry};
	}
}

#undef ATL_DETAIL_WIDE_RUNTIME_BACKEND
