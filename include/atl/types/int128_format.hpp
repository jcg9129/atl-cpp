#pragma once

#include <atl/types/int128.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <type_traits>

#if defined(__cpp_lib_format)
namespace atl::detail {
	/** @brief Character types accepted by the standard format contexts. */
	template<typename Character>
	concept format_character = std::same_as<Character, char> || std::same_as<Character, wchar_t>;

	/** @brief Integer presentation parser and renderer shared by the concrete specializations. */
	template<software_integer Integer, format_character Character>
	struct software_integer_formatter {
		/** @brief Maximum code units in one Unicode scalar for this character type. */
		static constexpr int fill_capacity = std::same_as<Character, char> ? 4 : (sizeof(Character) == 2 ? 2 : 1);

		/** @brief Code units of the padding scalar. */
		Character fill[fill_capacity]{Character(' ')};
		/** @brief Number of code units per padding position. */
		int fill_size = 1;
		char alignment = '>';			 ///< Left, right or centered alignment.
		char sign = '-';				 ///< Sign policy.
		char presentation = 'd';		 ///< Integer base and letter case.
		bool has_alternate_form = false; ///< Whether to emit a radix prefix.
		bool has_zero_padding = false;	 ///< Whether to pad with zeros after the prefix.
		bool has_alignment = false;		 ///< Whether alignment was explicitly selected.
		int width = 0;					 ///< Literal minimum field width.
		int width_argument = -1;		 ///< Dynamic-width argument index, or minus one.

		/** @brief Recognize a standard alignment character. */
		[[nodiscard]] static constexpr bool is_alignment(Character character) {
			return character == Character('<') || character == Character('>') || character == Character('^');
		}

		/** @brief Count and validate the code units of the first Unicode scalar. @pre The range is nonempty. */
		[[nodiscard]] static constexpr int fill_length(typename std::basic_format_parse_context<Character>::iterator iterator, typename std::basic_format_parse_context<Character>::iterator end) {
			atl_assert(iterator != end);
			std::uint32_t scalar = std::make_unsigned_t<Character>(*iterator);
			int length = 1;
			if constexpr (std::same_as<Character, char>) {
				if (scalar >= 0x80) {
					if (scalar >= 0xc2 && scalar <= 0xdf)
						length = 2;
					else if (scalar >= 0xe0 && scalar <= 0xef)
						length = 3;
					else if (scalar >= 0xf0 && scalar <= 0xf4)
						length = 4;
					else
						throw std::format_error("invalid integer fill");
					scalar &= (1u << (7 - length)) - 1;
					for (int index = 1; index < length; ++index) {
						if (++iterator == end) throw std::format_error("incomplete integer fill");
						auto const unit = static_cast<unsigned char>(*iterator);
						if ((unit & 0xc0) != 0x80) throw std::format_error("invalid integer fill");
						scalar = (scalar << 6) | (unit & 0x3f);
					}
					if (scalar < (length == 2 ? 0x80u : length == 3 ? 0x800u
																	: 0x10000u)) throw std::format_error("overlong integer fill");
				}
			} else if constexpr (sizeof(Character) == 2) {
				if (scalar >= 0xd800 && scalar <= 0xdbff) {
					if (++iterator == end) throw std::format_error("incomplete integer fill");
					std::uint32_t const unit = std::make_unsigned_t<Character>(*iterator);
					if (unit < 0xdc00 || unit > 0xdfff) throw std::format_error("invalid integer fill");
					scalar = 0x10000 + ((scalar - 0xd800) << 10) + (unit - 0xdc00);
					length = 2;
				}
			}
			if (scalar > 0x10ffff || (scalar >= 0xd800 && scalar <= 0xdfff)) throw std::format_error("invalid integer fill");
			return length;
		}

		/** @brief Parse integer presentation, including static and dynamic width. */
		constexpr auto parse(std::basic_format_parse_context<Character> &context) {
			auto iterator = context.begin(), end = context.end();
			if (iterator == end || *iterator == Character('}')) return iterator;
			int const length = fill_length(iterator, end);
			auto next = iterator;
			for (int index = 0; index < length; ++index) ++next;
			if (next != end && is_alignment(*next)) {
				if (*iterator == Character('{') || *iterator == Character('}')) throw std::format_error("invalid integer fill");
				fill_size = length;
				for (int index = 0; index < length; ++index) fill[index] = *iterator++;
				alignment = char(*next);
				has_alignment = true;
				iterator = ++next;
			} else if (is_alignment(*iterator)) {
				alignment = char(*iterator++);
				has_alignment = true;
			}
			if (iterator != end && (*iterator == Character('+') || *iterator == Character('-') || *iterator == Character(' '))) sign = char(*iterator++);
			if (iterator != end && *iterator == Character('#')) {
				has_alternate_form = true;
				++iterator;
			}
			if (iterator != end && *iterator == Character('0')) {
				has_zero_padding = true;
				++iterator;
			}
			while (iterator != end && *iterator >= Character('0') && *iterator <= Character('9')) {
				int const digit = int(*iterator++ - Character('0'));
				if (width > (std::numeric_limits<int>::max() - digit) / 10) throw std::format_error("integer width overflow");
				width = width * 10 + digit;
			}
			if (iterator != end && *iterator == Character('{')) {
				if (width != 0) throw std::format_error("duplicate integer width");
				++iterator;
				if (iterator != end && *iterator == Character('}')) {
					auto const argument = context.next_arg_id();
					if (argument > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::format_error("width argument overflow");
					width_argument = static_cast<int>(argument);
				} else {
					width_argument = 0;
					bool has_digit = false;
					while (iterator != end && *iterator >= Character('0') && *iterator <= Character('9')) {
						has_digit = true;
						int const digit = int(*iterator++ - Character('0'));
						if (width_argument > (std::numeric_limits<int>::max() - digit) / 10) throw std::format_error("width argument overflow");
						width_argument = width_argument * 10 + digit;
					}
					if (!has_digit) throw std::format_error("invalid width argument");
					context.check_arg_id(width_argument);
				}
				if (iterator == end || *iterator++ != Character('}')) throw std::format_error("invalid dynamic width");
			}
			if (iterator != end && *iterator != Character('}')) {
				// Validate before narrowing: a wide non-ASCII code point can have an ASCII low byte.
				auto const character = *iterator++;
				if (character != Character('b') && character != Character('B') && character != Character('o') && character != Character('d') && character != Character('x') && character != Character('X')) throw std::format_error("invalid integer presentation");
				presentation = char(character);
			}
			if (iterator != end && *iterator != Character('}')) throw std::format_error("invalid integer format specification");
			return iterator;
		}

		/** @brief Render a magnitude and sign, then place padding according to the parsed specification.
		 * @note Not constexpr because standard format contexts are runtime facilities.
		 */
		template<typename Context>
			requires std::same_as<typename Context::char_type, Character>
		auto format(Integer value, Context &context) const {
			int field_width = width;
			if (width_argument >= 0) {
				field_width = std::visit_format_arg([](auto argument) -> int {
					using Argument = decltype(argument);
					if constexpr (std::is_integral_v<Argument> && !std::same_as<Argument, bool> && !std::same_as<Argument, Character>) {
						if constexpr (std::is_signed_v<Argument>) {
							if (argument < 0) throw std::format_error("invalid integer width");
						}
						if (static_cast<unsigned long long>(argument) > static_cast<unsigned long long>(std::numeric_limits<int>::max())) throw std::format_error("invalid integer width");
						return int(argument);
					} else
						throw std::format_error("noninteger width");
				},
													context.arg(width_argument));
			}
			bool const is_negative = std::same_as<Integer, software_int128> && (value.high >> 63);
			auto magnitude = software_uint128(value);
			if (is_negative) magnitude = -magnitude;
			unsigned const base = presentation == 'b' || presentation == 'B' ? 2 : presentation == 'o'					  ? 8
																			 : presentation == 'x' || presentation == 'X' ? 16
																														  : 10;
			char const *alphabet = presentation == 'B' || presentation == 'X' ? "0123456789ABCDEF" : "0123456789abcdef";
			Character digits[128];
			int digit_count = 0;
			do {
				digits[digit_count++] = Character(alphabet[unsigned(magnitude % base)]);
				magnitude /= base;
			} while (magnitude);
			Character prefix[3];
			int prefix_count = 0;
			if (is_negative)
				prefix[prefix_count++] = Character('-');
			else if (sign != '-')
				prefix[prefix_count++] = Character(sign);
			if (has_alternate_form) {
				if (base == 2 || base == 16) {
					prefix[prefix_count++] = Character('0');
					prefix[prefix_count++] = Character(presentation);
				} else if (base == 8 && (digit_count != 1 || digits[0] != Character('0')))
					prefix[prefix_count++] = Character('0');
			}
			int const padding = field_width > digit_count + prefix_count ? field_width - digit_count - prefix_count : 0;
			bool const should_pad_zero = has_zero_padding && !has_alignment;
			int const before = should_pad_zero || alignment == '<' ? 0 : alignment == '^' ? padding / 2
																						  : padding;
			int const after = should_pad_zero ? 0 : padding - before;
			auto output = context.out();
			for (int index = 0; index < before; ++index)
				for (int unit = 0; unit < fill_size; ++unit) *output++ = fill[unit];
			for (int index = 0; index < prefix_count; ++index) *output++ = prefix[index];
			if (should_pad_zero)
				for (int index = 0; index < padding; ++index) *output++ = Character('0');
			while (digit_count) *output++ = digits[--digit_count];
			for (int index = 0; index < after; ++index)
				for (int unit = 0; unit < fill_size; ++unit) *output++ = fill[unit];
			return output;
		}
	};
}

namespace std {
	/** @brief Integer presentation for the concrete unsigned software integer. */
	template<atl::detail::format_character Character>
	struct formatter<atl::detail::software_uint128, Character> : atl::detail::software_integer_formatter<atl::detail::software_uint128, Character> {};

	/** @brief Integer presentation for the concrete signed software integer. */
	template<atl::detail::format_character Character>
	struct formatter<atl::detail::software_int128, Character> : atl::detail::software_integer_formatter<atl::detail::software_int128, Character> {};
}
#endif
