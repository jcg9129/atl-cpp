#pragma once

#include <atl/config.hpp>
#include <atl/types/detail/wide_arithmetic.hpp>
#include <atl/types/traits/is_one_of.hpp>
#include <atl/utils/assert.hpp>
#include <bit>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

namespace atl::detail {
	// Every operand concept here is one atomic constraint over a type-list trait, never a disjunction of concepts.
	// C++20 gives `==`, `!=`, `<`, `<=>` ... reversed and rewritten candidates from the same operator templates,
	// and choosing between them orders the templates by constraint subsumption, which normalizes the constraints to
	// disjunctive normal form. With `std::same_as` disjunctions over ~20 types per operand that form had millions
	// of clauses: Clang 22 needed over 180 s for one `software_int128 <=> software_int128` (-fsyntax-only) and
	// never finished the test, and MSVC ran out of heap. Atomic constraints normalize to a single clause.

	/** @brief Whether a type is a built-in integer operand, including native extended integers when enabled. */
	template<typename Integer>
	inline constexpr bool is_builtin_integer = atl::traits::is_one_of_v<Integer, bool, char, signed char, unsigned char, wchar_t, char8_t, char16_t, char32_t,
																		short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long
#if defined(__SIZEOF_INT128__)
																		,
																		__int128_t, __uint128_t
#endif
																		>;

	/** @brief Built-in integer operands, including native extended integers when enabled. */
	template<typename Integer>
	concept builtin_integer = is_builtin_integer<Integer>;

	/** @brief Unsigned ordering of two limb patterns. */
	[[nodiscard]] constexpr std::strong_ordering compare_words(word_pair left, word_pair right) {
		if (auto const order = left.high <=> right.high; order != 0) return order;
		return left.low <=> right.low;
	}

	/** @brief Add limb patterns modulo 2^128. */
	[[nodiscard]] constexpr word_pair add_words(word_pair left, word_pair right) {
		auto const bottom = add_with_carry(left.low, right.low);
		return {bottom.value, left.high + right.high + bottom.carry};
	}

	/** @brief Subtract limb patterns modulo 2^128. */
	[[nodiscard]] constexpr word_pair subtract_words(word_pair left, word_pair right) {
		auto const bottom = subtract_with_borrow(left.low, right.low);
		return {bottom.value, left.high - right.high - bottom.borrow};
	}

	/** @brief Negate a limb pattern modulo 2^128. */
	[[nodiscard]] constexpr word_pair negate_words(word_pair value) {
		return subtract_words({0, 0}, value);
	}

	/** @brief Left shift a limb pattern. @pre `count < 128`. */
	[[nodiscard]] constexpr word_pair shift_left_words(word_pair value, unsigned count) {
		atl_assert(count < 128);
		if (count == 0) return value;
		if (count >= 64) return {0, value.low << (count - 64)};
		return {value.low << count, (value.high << count) | (value.low >> (64 - count))};
	}

	/** @brief Logical right shift of a limb pattern. @pre `count < 128`. */
	[[nodiscard]] constexpr word_pair shift_right_words(word_pair value, unsigned count) {
		atl_assert(count < 128);
		if (count == 0) return value;
		if (count >= 64) return {value.high >> (count - 64), 0};
		return {(value.low >> count) | (value.high << (64 - count)), value.high >> count};
	}

	/** @brief Arithmetic right shift of a two's-complement limb pattern. @pre `count < 128`. */
	[[nodiscard]] constexpr word_pair shift_right_signed_words(word_pair value, unsigned count) {
		auto result = shift_right_words(value, count);
		if (count && value.high >> 63) {
			auto const mask = shift_left_words({UINT64_MAX, UINT64_MAX}, 128 - count);
			result.low |= mask.low;
			result.high |= mask.high;
		}
		return result;
	}

	/** @brief Low half of a two-word product. */
	[[nodiscard]] constexpr word_pair multiply_words(word_pair left, word_pair right) {
		auto const bottom = multiply_wide(left.low, right.low);
		return {bottom.low, bottom.high + left.low * right.high + left.high * right.low};
	}

	/** @brief Quotient and remainder of two unsigned limb patterns. */
	struct integer_division_result {
		word_pair quotient;	 ///< Quotient.
		word_pair remainder; ///< Remainder.
	};

	/**
	 * @brief Unsigned division of two limb patterns.
	 * A one-word divisor uses the word division directly. Otherwise the quotient fits one word: dividing the halved
	 * dividend by the normalised top word of the divisor estimates it to within one below the true value, so a
	 * decrement and at most one correcting step give the exact quotient.
	 * @pre Divisor is nonzero.
	 */
	[[nodiscard]] constexpr integer_division_result divide_words(word_pair dividend, word_pair divisor) {
		atl_assert((divisor.low | divisor.high) != 0);
		if (divisor.high == 0) {
			auto const result = divide_wide(dividend.low, dividend.high, divisor.low);
			return {result.quotient, {result.remainder, 0}};
		}
		if (compare_words(dividend, divisor) < 0) return {{0, 0}, dividend};
		auto const shift = unsigned(std::countl_zero(divisor.high));
		auto const normalized_top = shift_left_words(divisor, shift).high;
		// Halving keeps the high word below the normalised top word (< 2^63 <= normalized_top), as the bounded
		// division requires.
		auto const halved = shift_right_words(dividend, 1);
		auto const estimate = divide_wide_bounded(halved.low, halved.high, normalized_top).quotient;
		std::uint64_t quotient = shift_right_words(shift_left_words({estimate, 0}, shift), 63).low;
		if (quotient != 0) --quotient;
		// quotient * divisor <= dividend < 2^128, so the high word of the product is exact modulo 2^64.
		auto product = multiply_wide(quotient, divisor.low);
		product.high += quotient * divisor.high;
		auto remainder = subtract_words(dividend, product);
		if (compare_words(remainder, divisor) >= 0) {
			++quotient;
			remainder = subtract_words(remainder, divisor);
		}
		return {{quotient, 0}, remainder};
	}

	/** @brief Convert an integer to its sign-extended two's-complement pattern. */
	template<builtin_integer Integer>
	[[nodiscard]] constexpr word_pair integer_words(Integer value) {
#if defined(__SIZEOF_INT128__)
		if constexpr (sizeof(Integer) > 8)
			return {std::uint64_t(value), std::uint64_t(static_cast<__uint128_t>(value) >> 64)};
		else
#endif
			if constexpr (std::is_signed_v<Integer>)
			return {std::uint64_t(value), value < 0 ? UINT64_MAX : 0};
		else
			return {std::uint64_t(value), 0};
	}

	/**
	 * @brief Truncate a floating value into an unsigned magnitude, validating its range first.
	 * @pre Finite value; truncated magnitude fits the specified signedness.
	 */
	template<bool IsSigned, std::floating_point Floating>
	[[nodiscard]] constexpr word_pair floating_words(Floating value) {
		bool const is_negative = value < 0;
		using Working = std::conditional_t<(std::numeric_limits<Floating>::max_exponent <= 128), long double, Floating>;
		Working magnitude = is_negative ? -Working(value) : Working(value);
		Working limit = 1;
		for (int bit = 0; bit < (IsSigned ? 127 : 128); ++bit) limit *= 2;
		atl_assert(value == value && magnitude <= std::numeric_limits<Floating>::max());
		if constexpr (IsSigned) {
			atl_assert(is_negative ? (magnitude <= limit || magnitude - limit < 1) : magnitude < limit);
		} else {
			atl_assert(magnitude < limit && (!is_negative || magnitude < 1));
		}
		word_pair result{};
		Working power = limit;
		// Subtractions of powers of two recover the represented integer bits
		// exactly, including long double formats wider than either limb.
		for (int bit = IsSigned ? 127 : 128; bit >= 0; --bit) {
			if (bit < 128 && magnitude >= power) {
				if (bit >= 64)
					result.high |= std::uint64_t(1) << (bit - 64);
				else
					result.low |= std::uint64_t(1) << bit;
				magnitude -= power;
			}
			power /= 2;
		}
		return is_negative ? negate_words(result) : result;
	}

	/** @brief Round an unsigned bit pattern once to a contiguous binary significand, to nearest with ties to even. */
	template<std::floating_point Floating>
	[[nodiscard]] constexpr Floating words_binary_floating(word_pair value) {
		int const width = value.high ? 128 - std::countl_zero(value.high) : 64 - std::countl_zero(value.low);
		constexpr int precision = std::numeric_limits<Floating>::digits;
		int const discarded = width > precision ? width - precision : 0;
		auto retained = shift_right_words(value, unsigned(discarded));
		bool should_round = false;
		if (discarded) {
			auto const half = shift_left_words({1, 0}, unsigned(discarded - 1));
			auto const tail = subtract_words(value, shift_left_words(retained, unsigned(discarded)));
			auto const order = compare_words(tail, half);
			should_round = order > 0 || (order == 0 && (retained.low & 1));
		}
		// Build the retained significand before scaling; every intermediate is
		// exact. Adding converted halves would permit a second rounding.
		Floating result = 0;
		for (int bit = width - discarded - 1; bit >= 0; --bit) {
			result *= 2;
			result += Floating(bit >= 64 ? (retained.high >> (bit - 64)) & 1 : (retained.low >> bit) & 1);
		}
		if (should_round) result += 1;
		for (int bit = 0; bit < discarded; ++bit) {
			if (result > std::numeric_limits<Floating>::max() / 2) return std::numeric_limits<Floating>::infinity();
			result *= 2;
		}
		return result;
	}

	/** @brief A floating expansion whose low component approximates the exact residual after rounding the high component. */
	template<std::floating_point Component>
	struct floating_components {
		/** @brief Nearest component value, ties to even. */
		Component high;
		/** @brief Signed residual rounded to the component format. */
		Component low;
	};

	/** @brief Split an unsigned magnitude into a rounded high component and a rounded exact limb residual. */
	template<std::floating_point Component>
	[[nodiscard]] constexpr floating_components<Component> words_double_double(word_pair value) {
		static_assert(std::numeric_limits<Component>::radix == 2 && std::numeric_limits<Component>::digits < 128 && std::numeric_limits<Component>::max_exponent > 128);
		Component const high = words_binary_floating<Component>(value);
		// Rounding can produce 2^128, which cannot be converted back into the unsigned limb range.
		if (high == Component(0x1p128)) return {high, -words_binary_floating<Component>(negate_words(value))};
		auto const rounded = floating_words<false>(high);
		bool const is_negative = compare_words(rounded, value) > 0;
		auto const residual = is_negative ? subtract_words(rounded, value) : subtract_words(value, rounded);
		Component const low = words_binary_floating<Component>(residual);
		return {high, is_negative ? -low : low};
	}

	/** @brief Convert an unsigned magnitude using the target's binary or double-double representation. */
	template<std::floating_point Floating>
	[[nodiscard]] constexpr Floating words_floating(word_pair value) {
		if constexpr (std::same_as<Floating, long double> && std::numeric_limits<Floating>::digits == 106) {
			static_assert(std::numeric_limits<Floating>::radix == 2 && std::numeric_limits<Floating>::max_exponent == 1024 && sizeof(Floating) == 2 * sizeof(double)
							  && std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53 && std::numeric_limits<double>::max_exponent == 1024,
						  "106-bit long double must use two IEEE binary64 components");
			auto const components = words_double_double<double>(value);
			return Floating(components.high) + Floating(components.low);
		} else
			return words_binary_floating<Floating>(value);
	}

	/** @brief Structural unsigned 128-bit integer with arithmetic modulo 2^128. */
	struct software_uint128;
	/** @brief Structural signed two's-complement 128-bit integer with wrapping arithmetic. */
	struct software_int128;

	/** @brief Whether a type is either concrete software integer class. */
	template<typename Integer>
	inline constexpr bool is_software_integer = atl::traits::is_one_of_v<Integer, software_uint128, software_int128>;

	/** @brief Either concrete software integer class. */
	template<typename Integer>
	concept software_integer = is_software_integer<Integer>;

	/** @brief Whether a type is an operand accepted by software integer operators. */
	template<typename Integer>
	inline constexpr bool is_integer_operand = is_builtin_integer<Integer> || is_software_integer<Integer>;

	/** @brief An operand accepted by software integer operators. */
	template<typename Integer>
	concept integer_operand = is_integer_operand<Integer>;

	/** @brief Whether software integer operators participate for a pair. */
	template<typename Left, typename Right>
	inline constexpr bool is_software_integer_pair = is_integer_operand<Left> && is_integer_operand<Right> && (is_software_integer<Left> || is_software_integer<Right>);

	/** @brief A pair for which software integer operators participate. */
	template<typename Left, typename Right>
	concept software_integer_pair = is_software_integer_pair<Left, Right>;

	/** @brief Unsigned structural 128-bit integer; default initialization leaves the limbs uninitialized. */
	struct software_uint128 {
	private:
		/** @brief Construct from one already computed limb pattern. */
		explicit constexpr software_uint128(word_pair value) : low(value.low), high(value.high) {}

	public:
		std::uint64_t low;	///< Low two's-complement word.
		std::uint64_t high; ///< High two's-complement word.
		/** @brief Default initialization matches built-in integers; value initialization produces zero. */
		constexpr software_uint128() = default;

		/** @brief Construct directly from a low and high word. */
		constexpr software_uint128(std::uint64_t low_word, std::uint64_t high_word) : low(low_word), high(high_word) {}

		/** @brief Implicitly sign-extend any built-in integer operand. */
		template<builtin_integer Integer>
		constexpr software_uint128(Integer value) : software_uint128(integer_words(value)) {}

		/** @brief Preserve the bit pattern when changing signedness. */
		constexpr software_uint128(software_int128 value);

		/** @brief Truncate a finite floating value. @pre Truncated value is representable. */
		template<std::floating_point Floating>
		explicit constexpr software_uint128(Floating value) : software_uint128(floating_words<false>(value)) {}

		/** @brief Test whether any bit is set. */
		[[nodiscard]] explicit constexpr operator bool() const { return (low | high) != 0; }

		/** @brief Convert the low bits modulo the destination width (bool tests the complete value). */
		template<builtin_integer Integer>
		[[nodiscard]] explicit constexpr operator Integer() const {
			if constexpr (std::same_as<Integer, bool>) return (low | high) != 0;
#if defined(__SIZEOF_INT128__)
			else if constexpr (sizeof(Integer) > 8)
				return Integer((static_cast<__uint128_t>(high) << 64) | low);
#endif
			else
				return Integer(low);
		}

		/** @brief Correctly round the magnitude to the target floating format. */
		template<std::floating_point Floating>
		[[nodiscard]] explicit constexpr operator Floating() const {
			word_pair const bits{low, high};
			return words_floating<Floating>(bits);
		}

		/** @brief Unary identity. */
		[[nodiscard]] constexpr software_uint128 operator+() const { return *this; }

		/** @brief Two's-complement negation modulo 2^128. */
		[[nodiscard]] constexpr software_uint128 operator-() const {
			auto const bits = negate_words({low, high});
			return {bits.low, bits.high};
		}

		/** @brief Complement every bit. */
		[[nodiscard]] constexpr software_uint128 operator~() const { return {~low, ~high}; }

		/** @brief Test for zero. */
		[[nodiscard]] constexpr bool operator!() const { return (low | high) == 0; }

		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_uint128 &operator+=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_uint128 &operator-=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_uint128 &operator*=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. @pre Nonzero divisor; no common signed minimum / -1. */
		template<integer_operand Right>
		constexpr software_uint128 &operator/=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. @pre Nonzero divisor; no common signed minimum / -1. */
		template<integer_operand Right>
		constexpr software_uint128 &operator%=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_uint128 &operator&=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_uint128 &operator|=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_uint128 &operator^=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. @pre Count is valid for the promoted left type. */
		template<integer_operand Right>
		constexpr software_uint128 &operator<<=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. @pre Count is valid for the promoted left type. */
		template<integer_operand Right>
		constexpr software_uint128 &operator>>=(Right right);
		/** @brief Increment and return this object. */
		constexpr software_uint128 &operator++();
		/** @brief Increment and return the previous value. */
		constexpr software_uint128 operator++(int);
		/** @brief Decrement and return this object. */
		constexpr software_uint128 &operator--();
		/** @brief Decrement and return the previous value. */
		constexpr software_uint128 operator--(int);
	};

	/** @brief Signed wrapping structural 128-bit integer; default initialization leaves the limbs uninitialized. */
	struct software_int128 {
	private:
		/** @brief Construct from one already computed limb pattern. */
		explicit constexpr software_int128(word_pair value) : low(value.low), high(value.high) {}

	public:
		std::uint64_t low;	///< Low two's-complement word.
		std::uint64_t high; ///< High two's-complement word.
		/** @brief Default initialization matches built-in integers; value initialization produces zero. */
		constexpr software_int128() = default;

		/** @brief Construct directly from a low and high word. */
		constexpr software_int128(std::uint64_t low_word, std::uint64_t high_word) : low(low_word), high(high_word) {}

		/** @brief Implicitly sign-extend any built-in integer operand. */
		template<builtin_integer Integer>
		constexpr software_int128(Integer value) : software_int128(integer_words(value)) {}

		/** @brief Preserve the bit pattern when changing signedness. */
		explicit constexpr software_int128(software_uint128 value);

		/** @brief Truncate a finite floating value. @pre Truncated value is representable. */
		template<std::floating_point Floating>
		explicit constexpr software_int128(Floating value) : software_int128(floating_words<true>(value)) {}

		/** @brief Test whether any bit is set. */
		[[nodiscard]] explicit constexpr operator bool() const { return (low | high) != 0; }

		/** @brief Convert the low bits modulo the destination width (bool tests the complete value). */
		template<builtin_integer Integer>
		[[nodiscard]] explicit constexpr operator Integer() const {
			if constexpr (std::same_as<Integer, bool>) return (low | high) != 0;
#if defined(__SIZEOF_INT128__)
			else if constexpr (sizeof(Integer) > 8)
				return Integer((static_cast<__uint128_t>(high) << 64) | low);
#endif
			else
				return Integer(low);
		}

		/** @brief Correctly round the magnitude to the target floating format. */
		template<std::floating_point Floating>
		[[nodiscard]] explicit constexpr operator Floating() const {
			word_pair const bits{low, high};
			return high >> 63 ? -words_floating<Floating>(negate_words(bits)) : words_floating<Floating>(bits);
		}

		/** @brief Unary identity. */
		[[nodiscard]] constexpr software_int128 operator+() const { return *this; }

		/** @brief Two's-complement negation modulo 2^128. */
		[[nodiscard]] constexpr software_int128 operator-() const {
			auto const bits = negate_words({low, high});
			return {bits.low, bits.high};
		}

		/** @brief Complement every bit. */
		[[nodiscard]] constexpr software_int128 operator~() const { return {~low, ~high}; }

		/** @brief Test for zero. */
		[[nodiscard]] constexpr bool operator!() const { return (low | high) == 0; }

		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_int128 &operator+=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_int128 &operator-=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_int128 &operator*=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. @pre Nonzero divisor; no common signed minimum / -1. */
		template<integer_operand Right>
		constexpr software_int128 &operator/=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. @pre Nonzero divisor; no common signed minimum / -1. */
		template<integer_operand Right>
		constexpr software_int128 &operator%=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_int128 &operator&=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_int128 &operator|=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. */
		template<integer_operand Right>
		constexpr software_int128 &operator^=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. @pre Count is valid for the promoted left type. */
		template<integer_operand Right>
		constexpr software_int128 &operator<<=(Right right);
		/** @brief Compute in the operator's result type, convert back and return this object. @pre Count is valid for the promoted left type. */
		template<integer_operand Right>
		constexpr software_int128 &operator>>=(Right right);
		/** @brief Increment and return this object. */
		constexpr software_int128 &operator++();
		/** @brief Increment and return the previous value. */
		constexpr software_int128 operator++(int);
		/** @brief Decrement and return this object. */
		constexpr software_int128 &operator--();
		/** @brief Decrement and return the previous value. */
		constexpr software_int128 operator--(int);
	};

	constexpr software_uint128::software_uint128(software_int128 value) : low(value.low), high(value.high) {}

	constexpr software_int128::software_int128(software_uint128 value) : low(value.low), high(value.high) {}

	/** @brief Whether an operand contributes unsigned rank 128 to the usual arithmetic conversions. */
	template<typename Integer>
	inline constexpr bool is_unsigned_128 = std::same_as<Integer, software_uint128>
#if defined(__SIZEOF_INT128__)
										 || std::same_as<Integer, __uint128_t>
#endif
		;

	/** @brief Common software integer type for arithmetic operands. */
	template<integer_operand Left, integer_operand Right>
	using common_integer = std::conditional_t<is_unsigned_128<Left> || is_unsigned_128<Right>, software_uint128, software_int128>;

	/** @brief Extract the limb pattern of any supported operand. */
	template<integer_operand Integer>
	[[nodiscard]] constexpr word_pair operand_words(Integer value) {
		if constexpr (software_integer<Integer>)
			return {value.low, value.high};
		else
			return integer_words(value);
	}

	/** @brief Apply + after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr common_integer<Left, Right> operator+(Left left, Right right) {
		auto const a = operand_words(left), b = operand_words(right);
		auto const result = add_words(a, b);
		return {result.low, result.high};
	}

	/** @brief Apply - after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr common_integer<Left, Right> operator-(Left left, Right right) {
		auto const a = operand_words(left), b = operand_words(right);
		auto const result = subtract_words(a, b);
		return {result.low, result.high};
	}

	/** @brief Apply * after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr common_integer<Left, Right> operator*(Left left, Right right) {
		auto const a = operand_words(left), b = operand_words(right);
		auto const result = multiply_words(a, b);
		return {result.low, result.high};
	}

	/** @brief Apply & after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr common_integer<Left, Right> operator&(Left left, Right right) {
		auto const a = operand_words(left), b = operand_words(right);
		auto const result = word_pair{a.low & b.low, a.high & b.high};
		return {result.low, result.high};
	}

	/** @brief Apply | after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr common_integer<Left, Right> operator|(Left left, Right right) {
		auto const a = operand_words(left), b = operand_words(right);
		auto const result = word_pair{a.low | b.low, a.high | b.high};
		return {result.low, result.high};
	}

	/** @brief Apply ^ after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr common_integer<Left, Right> operator^(Left left, Right right) {
		auto const a = operand_words(left), b = operand_words(right);
		auto const result = word_pair{a.low ^ b.low, a.high ^ b.high};
		return {result.low, result.high};
	}

	/**
	 * @brief Divide in the common type, using unsigned magnitudes for signed operands.
	 * @pre Nonzero divisor and no signed minimum divided by minus one.
	 */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr integer_division_result divide_operands(Left left, Right right) {
		auto a = operand_words(left), b = operand_words(right);
		if constexpr (std::same_as<common_integer<Left, Right>, software_int128>) {
			bool const is_left_negative = (a.high >> 63) != 0, is_right_negative = (b.high >> 63) != 0;
			atl_assert(!(a.high == (std::uint64_t(1) << 63) && a.low == 0 && b.high == UINT64_MAX && b.low == UINT64_MAX));
			if (is_left_negative) a = negate_words(a);
			if (is_right_negative) b = negate_words(b);
			auto result = divide_words(a, b);
			if (is_left_negative != is_right_negative) result.quotient = negate_words(result.quotient);
			if (is_left_negative) result.remainder = negate_words(result.remainder);
			return result;
		} else
			return divide_words(a, b);
	}

	/** @brief Quotient truncating toward zero. @pre Nonzero divisor; no signed minimum / -1. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr common_integer<Left, Right> operator/(Left left, Right right) {
		auto const result = divide_operands(left, right).quotient;
		return {result.low, result.high};
	}

	/** @brief Remainder with the dividend sign. @pre Nonzero divisor; no signed minimum / -1. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr common_integer<Left, Right> operator%(Left left, Right right) {
		auto const result = divide_operands(left, right).remainder;
		return {result.low, result.high};
	}

	/** @brief Comparison in the usual arithmetic common type. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr std::strong_ordering compare_operands(Left left, Right right) {
		auto a = operand_words(left), b = operand_words(right);
		if constexpr (std::same_as<common_integer<Left, Right>, software_int128>) {
			a.high ^= std::uint64_t(1) << 63;
			b.high ^= std::uint64_t(1) << 63;
		}
		return compare_words(a, b);
	}

	/** @brief Compare after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr bool operator==(Left left, Right right) { return compare_operands(left, right) == 0; }

	/** @brief Compare after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr bool operator!=(Left left, Right right) { return compare_operands(left, right) != 0; }

	/** @brief Compare after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr bool operator<(Left left, Right right) { return compare_operands(left, right) < 0; }

	/** @brief Compare after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr bool operator<=(Left left, Right right) { return compare_operands(left, right) <= 0; }

	/** @brief Compare after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr bool operator>(Left left, Right right) { return compare_operands(left, right) > 0; }

	/** @brief Compare after the usual arithmetic conversions. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr bool operator>=(Left left, Right right) { return compare_operands(left, right) >= 0; }

	/** @brief Whether a pair has a three-way comparison: no bool and no narrowing signed-to-unsigned conversion. */
	template<typename Left, typename Right>
	inline constexpr bool is_three_way_integer_pair = is_software_integer_pair<Left, Right> && !std::same_as<Left, bool> && !std::same_as<Right, bool>
												   && (!(is_unsigned_128<Left> || is_unsigned_128<Right>)
													   || ((is_unsigned_128<Left> || (is_builtin_integer<Left> && !std::numeric_limits<Left>::is_signed))
														   && (is_unsigned_128<Right> || (is_builtin_integer<Right> && !std::numeric_limits<Right>::is_signed))));

	/** @brief A three-way comparison without bool or narrowing signed-to-unsigned conversions. */
	template<typename Left, typename Right>
	concept three_way_integer_pair = is_three_way_integer_pair<Left, Right>;

	/** @brief Three-way comparison where the built-in equivalent is well formed. */
	template<integer_operand Left, integer_operand Right>
		requires three_way_integer_pair<Left, Right>
	[[nodiscard]] constexpr std::strong_ordering operator<=>(Left left, Right right) { return compare_operands(left, right); }

	/** @brief Validate a shift count before narrowing it. @pre Count is in [0, limit). */
	template<integer_operand Count>
	[[nodiscard]] constexpr unsigned shift_count(Count count, unsigned limit = 128) {
		auto const words = operand_words(count);
		atl_assert(words.high == 0 && words.low < limit);
		return unsigned(words.low);
	}

	/** @brief Type of the promoted left operand of a shift. */
	template<integer_operand Left>
	using promoted_integer = decltype(+Left{});

	/** @brief Left shift with the promoted left operand type. @pre Count is valid for the promoted left type. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr promoted_integer<Left> operator<<(Left left, Right right) {
		using Result = promoted_integer<Left>;
		auto const count = shift_count(right, unsigned(sizeof(Result) * 8));
		auto const result = shift_left_words(operand_words(left), count);
		if constexpr (software_integer<Left>)
			return {result.low, result.high};
		else
			return Result(software_uint128{result.low, result.high});
	}

	/** @brief Right shift with the promoted left operand type. @pre Count is valid for the promoted left type. */
	template<integer_operand Left, integer_operand Right>
		requires software_integer_pair<Left, Right>
	[[nodiscard]] constexpr promoted_integer<Left> operator>>(Left left, Right right) {
		using Result = promoted_integer<Left>;
		auto const count = shift_count(right, unsigned(sizeof(Result) * 8));
		auto const bits = operand_words(left);
		constexpr bool is_signed = std::same_as<Left, software_int128> || (builtin_integer<Left> && std::numeric_limits<Left>::is_signed);
		auto const result = is_signed ? shift_right_signed_words(bits, count) : shift_right_words(bits, count);
		if constexpr (software_integer<Left>)
			return {result.low, result.high};
		else
			return Result(software_uint128{result.low, result.high});
	}

	template<integer_operand Right>
	constexpr software_uint128 &software_uint128::operator+=(Right right) { return *this = software_uint128(*this + right); }

	template<integer_operand Right>
	constexpr software_uint128 &software_uint128::operator-=(Right right) { return *this = software_uint128(*this - right); }

	template<integer_operand Right>
	constexpr software_uint128 &software_uint128::operator*=(Right right) { return *this = software_uint128(*this * right); }

	template<integer_operand Right>
	constexpr software_uint128 &software_uint128::operator/=(Right right) { return *this = software_uint128(*this / right); }

	template<integer_operand Right>
	constexpr software_uint128 &software_uint128::operator%=(Right right) { return *this = software_uint128(*this % right); }

	template<integer_operand Right>
	constexpr software_uint128 &software_uint128::operator&=(Right right) { return *this = software_uint128(*this & right); }

	template<integer_operand Right>
	constexpr software_uint128 &software_uint128::operator|=(Right right) { return *this = software_uint128(*this | right); }

	template<integer_operand Right>
	constexpr software_uint128 &software_uint128::operator^=(Right right) { return *this = software_uint128(*this ^ right); }

	template<integer_operand Right>
	constexpr software_uint128 &software_uint128::operator<<=(Right right) { return *this = software_uint128(*this << right); }

	template<integer_operand Right>
	constexpr software_uint128 &software_uint128::operator>>=(Right right) { return *this = software_uint128(*this >> right); }

	constexpr software_uint128 &software_uint128::operator++() { return *this += 1; }

	constexpr software_uint128 software_uint128::operator++(int) {
		auto const previous = *this;
		++*this;
		return previous;
	}

	constexpr software_uint128 &software_uint128::operator--() { return *this -= 1; }

	constexpr software_uint128 software_uint128::operator--(int) {
		auto const previous = *this;
		--*this;
		return previous;
	}

	template<integer_operand Right>
	constexpr software_int128 &software_int128::operator+=(Right right) { return *this = software_int128(*this + right); }

	template<integer_operand Right>
	constexpr software_int128 &software_int128::operator-=(Right right) { return *this = software_int128(*this - right); }

	template<integer_operand Right>
	constexpr software_int128 &software_int128::operator*=(Right right) { return *this = software_int128(*this * right); }

	template<integer_operand Right>
	constexpr software_int128 &software_int128::operator/=(Right right) { return *this = software_int128(*this / right); }

	template<integer_operand Right>
	constexpr software_int128 &software_int128::operator%=(Right right) { return *this = software_int128(*this % right); }

	template<integer_operand Right>
	constexpr software_int128 &software_int128::operator&=(Right right) { return *this = software_int128(*this & right); }

	template<integer_operand Right>
	constexpr software_int128 &software_int128::operator|=(Right right) { return *this = software_int128(*this | right); }

	template<integer_operand Right>
	constexpr software_int128 &software_int128::operator^=(Right right) { return *this = software_int128(*this ^ right); }

	template<integer_operand Right>
	constexpr software_int128 &software_int128::operator<<=(Right right) { return *this = software_int128(*this << right); }

	template<integer_operand Right>
	constexpr software_int128 &software_int128::operator>>=(Right right) { return *this = software_int128(*this >> right); }

	constexpr software_int128 &software_int128::operator++() { return *this += 1; }

	constexpr software_int128 software_int128::operator++(int) {
		auto const previous = *this;
		++*this;
		return previous;
	}

	constexpr software_int128 &software_int128::operator--() { return *this -= 1; }

	constexpr software_int128 software_int128::operator--(int) {
		auto const previous = *this;
		--*this;
		return previous;
	}

	/** @brief Compute with a software right operand, convert back to the built-in left type and return its reference. */
	template<builtin_integer Left, software_integer Right>
	constexpr Left &operator+=(Left &left, Right right) { return left = Left(left + right); }

	/** @brief Compute with a software right operand, convert back to the built-in left type and return its reference. */
	template<builtin_integer Left, software_integer Right>
	constexpr Left &operator-=(Left &left, Right right) { return left = Left(left - right); }

	/** @brief Compute with a software right operand, convert back to the built-in left type and return its reference. */
	template<builtin_integer Left, software_integer Right>
	constexpr Left &operator*=(Left &left, Right right) { return left = Left(left * right); }

	/** @brief Compute with a software right operand, convert back to the built-in left type and return its reference. @pre Nonzero divisor; no common signed minimum / -1. */
	template<builtin_integer Left, software_integer Right>
	constexpr Left &operator/=(Left &left, Right right) { return left = Left(left / right); }

	/** @brief Compute with a software right operand, convert back to the built-in left type and return its reference. @pre Nonzero divisor; no common signed minimum / -1. */
	template<builtin_integer Left, software_integer Right>
	constexpr Left &operator%=(Left &left, Right right) { return left = Left(left % right); }

	/** @brief Compute with a software right operand, convert back to the built-in left type and return its reference. */
	template<builtin_integer Left, software_integer Right>
	constexpr Left &operator&=(Left &left, Right right) { return left = Left(left & right); }

	/** @brief Compute with a software right operand, convert back to the built-in left type and return its reference. */
	template<builtin_integer Left, software_integer Right>
	constexpr Left &operator|=(Left &left, Right right) { return left = Left(left | right); }

	/** @brief Compute with a software right operand, convert back to the built-in left type and return its reference. */
	template<builtin_integer Left, software_integer Right>
	constexpr Left &operator^=(Left &left, Right right) { return left = Left(left ^ right); }

	/** @brief Compute with a software right operand, convert back to the built-in left type and return its reference. @pre Count is valid for the promoted left type. */
	template<builtin_integer Left, software_integer Right>
	constexpr Left &operator<<=(Left &left, Right right) { return left = Left(left << right); }

	/** @brief Compute with a software right operand, convert back to the built-in left type and return its reference. @pre Count is valid for the promoted left type. */
	template<builtin_integer Left, software_integer Right>
	constexpr Left &operator>>=(Left &left, Right right) { return left = Left(left >> right); }

}

namespace atl::detail {
	/** @brief Complete numeric limits shared by the two concrete software specializations. */
	template<software_integer Integer>
	struct software_numeric_limits {
		/** @brief This specialization describes an integer type. */
		static constexpr bool is_specialized = true;
		/** @brief Whether the type has a sign bit. */
		static constexpr bool is_signed = std::same_as<Integer, software_int128>;
		/** @brief Number of value bits. */
		static constexpr int digits = is_signed ? 127 : 128;
		/** @brief Number of decimal digits preserved exactly. */
		static constexpr int digits10 = 38;
		/** @brief Not applicable to integers. */
		static constexpr int max_digits10 = 0;
		/** @brief Integer values have no fractional part. */
		static constexpr bool is_integer = true;
		/** @brief Every representable integer is exact. */
		static constexpr bool is_exact = true;
		/** @brief Binary representation. */
		static constexpr int radix = 2;
		/** @brief Integer exponent limits are not applicable. */
		static constexpr int min_exponent = 0, min_exponent10 = 0, max_exponent = 0, max_exponent10 = 0;
		/** @brief No infinity representation. */
		static constexpr bool has_infinity = false;
		/** @brief No quiet NaN representation. */
		static constexpr bool has_quiet_NaN = false; // NOLINT(readability-identifier-naming) -- required numeric_limits member spelling.
		/** @brief No signaling NaN representation. */
		static constexpr bool has_signaling_NaN = false; // NOLINT(readability-identifier-naming) -- required numeric_limits member spelling.

		// C++23 deprecates the denormal members (MSVC STL4042 as C4996, libc++ as -Wdeprecated-declarations), but
		// generic code still reads them, so the specialization keeps them and silences only these declarations.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4996)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
		/** @brief No denormal representation. */
		static constexpr std::float_denorm_style has_denorm = std::denorm_absent;
		/** @brief Denormal loss is not applicable. */
		static constexpr bool has_denorm_loss = false;
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
		/** @brief This is not an IEC 559 floating type. */
		static constexpr bool is_iec559 = false;
		/** @brief The representable interval is bounded. */
		static constexpr bool is_bounded = true;
		/** @brief Both signed and unsigned arithmetic wrap modulo 2^128. */
		static constexpr bool is_modulo = true;
		/** @brief Invalid operations can abort with ATL_ASSERT enabled. */
		static constexpr bool traps = true;
		/** @brief Floating underflow properties do not apply. */
		static constexpr bool tinyness_before = false;
		/** @brief Integer division truncates toward zero. */
		static constexpr std::float_round_style round_style = std::round_toward_zero;

		/** @brief Least representable value. */
		[[nodiscard]] static constexpr Integer min() noexcept { return is_signed ? Integer{0, std::uint64_t(1) << 63} : Integer{}; }

		/** @brief Least representable value. */
		[[nodiscard]] static constexpr Integer lowest() noexcept { return min(); }

		/** @brief Greatest representable value. */
		[[nodiscard]] static constexpr Integer max() noexcept { return {UINT64_MAX, is_signed ? UINT64_MAX >> 1 : UINT64_MAX}; }

		/** @brief Zero; integers have no fractional epsilon. */
		[[nodiscard]] static constexpr Integer epsilon() noexcept { return {}; }

		/** @brief Zero; integers have no floating rounding error. */
		[[nodiscard]] static constexpr Integer round_error() noexcept { return {}; }

		/** @brief Zero; infinity is unavailable. */
		[[nodiscard]] static constexpr Integer infinity() noexcept { return {}; }

		/** @brief Zero; NaNs are unavailable. */
		[[nodiscard]] static constexpr Integer quiet_NaN() noexcept { return {}; } // NOLINT(readability-identifier-naming) -- required numeric_limits member spelling.

		/** @brief Zero; NaNs are unavailable. */
		[[nodiscard]] static constexpr Integer signaling_NaN() noexcept { return {}; } // NOLINT(readability-identifier-naming) -- required numeric_limits member spelling.

		/** @brief Zero; denormals are unavailable. */
		[[nodiscard]] static constexpr Integer denorm_min() noexcept { return {}; }
	};

	/** @brief Hash both words byte by byte, also on targets with a narrow size_t. */
	template<software_integer Integer>
	struct software_hash {
		/** @brief Combine all sixteen bytes into a size_t. */
		[[nodiscard]] constexpr std::size_t operator()(Integer value) const noexcept {
			/** @brief Unsigned accumulator that avoids promotion to a narrower signed int. */
			using Accumulator = std::conditional_t<(sizeof(std::size_t) < sizeof(std::uint64_t)), std::uint64_t, std::size_t>;
			Accumulator result = 0;
			std::uint64_t const words[] = {value.low, value.high};
			for (auto word : words) {
				for (int byte = 0; byte < 8; ++byte) {
					result = result * Accumulator(131) + Accumulator(word & 255);
					word >>= 8;
				}
			}
			return std::size_t(result);
		}
	};
}

namespace std {
	/** @brief Numeric limits of the concrete unsigned software integer. */
	template<>
	struct numeric_limits<atl::detail::software_uint128> : atl::detail::software_numeric_limits<atl::detail::software_uint128> {};

	/** @brief Numeric limits of the concrete signed software integer. */
	template<>
	struct numeric_limits<atl::detail::software_int128> : atl::detail::software_numeric_limits<atl::detail::software_int128> {};

	/** @brief Hash support for the concrete unsigned software integer. */
	template<>
	struct hash<atl::detail::software_uint128> : atl::detail::software_hash<atl::detail::software_uint128> {};

	/** @brief Hash support for the concrete signed software integer. */
	template<>
	struct hash<atl::detail::software_int128> : atl::detail::software_hash<atl::detail::software_int128> {};

	/** @brief Common type for any pair involving a concrete software integer. */
	template<atl::detail::integer_operand Left, atl::detail::integer_operand Right>
		requires atl::detail::software_integer_pair<Left, Right>
	struct common_type<Left, Right> { // NOLINT(bugprone-std-namespace-modification) -- constraint requires a program-defined software type.
		/** @brief Result of the usual arithmetic conversions. */
		using type = atl::detail::common_integer<Left, Right>;
	};
}
