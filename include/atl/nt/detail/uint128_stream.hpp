#pragma once

#include <atl/types/alias.hpp>
#include <climits>
#include <concepts>
#include <cstddef>
#include <ios>
#include <istream>
#include <limits>
#include <locale>
#include <ostream>
#include <streambuf>
#include <string>

namespace atl::detail {
	/** @brief Write an unsigned 128-bit integer with integer stream formatting and bounded working storage. */
	inline std::ostream &write_uint128(std::ostream &os, uint128 value) {
		std::ostream::sentry const sentry(os);
		if (!sentry) return os;
		auto state = std::ios_base::goodbit;
		try {
			auto const flags = os.flags();
			auto const basefield = flags & std::ios_base::basefield;
			unsigned const base = basefield == std::ios_base::hex ? 16u : basefield == std::ios_base::oct ? 8u
																										  : 10u;
			bool const is_uppercase = (flags & std::ios_base::uppercase) != 0;
			char const *alphabet = is_uppercase ? "0123456789ABCDEF" : "0123456789abcdef";
			auto const &punct = std::use_facet<std::numpunct<char>>(os.getloc());
			std::string const grouping = punct.grouping();
			std::size_t group_index = 0;
			int group_size = grouping.empty() ? 0 : grouping[0];
			int group_digits = 0;
			// At most 43 octal digits and 42 separators; prefixes are kept separately.
			char digits[85];
			char *const end = digits + sizeof(digits);
			char *begin = end;
			bool const is_nonzero = value != 0;
			do {
				*--begin = alphabet[static_cast<unsigned>(value % base)];
				value /= base;
				++group_digits;
				if (value != 0 && group_size > 0 && group_size < CHAR_MAX && group_digits == group_size) {
					*--begin = punct.thousands_sep();
					group_digits = 0;
					if (group_index + 1 < grouping.size()) {
						int const next = grouping[++group_index];
						if (next == 0)
							group_index = grouping.size() - 1;
						else
							group_size = next;
					}
				}
			} while (value != 0);

			char prefix[2] = {'0', is_uppercase ? 'X' : 'x'};
			std::streamsize const prefix_size = (flags & std::ios_base::showbase) != 0 && is_nonzero && base != 10 ? (base == 16 ? 2 : 1) : 0;
			std::streamsize const digit_size = end - begin;
			auto const width = os.width(0);
			auto const padding = width > digit_size + prefix_size ? width - digit_size - prefix_size : 0;
			auto *buffer = os.rdbuf();
			auto const write = [&](char const *data, std::streamsize size) {
				if (state == std::ios_base::goodbit && size != 0 && buffer->sputn(data, size) != size) state |= std::ios_base::badbit;
			};
			auto const pad = [&] {
				if (padding == 0) return;
				char fill[64];
				char const fill_char = os.fill();
				for (char &ch : fill) ch = fill_char;
				auto remaining = padding;
				while (remaining > 0 && state == std::ios_base::goodbit) {
					auto const count = remaining < std::streamsize(sizeof(fill)) ? remaining : std::streamsize(sizeof(fill));
					write(fill, count);
					remaining -= count;
				}
			};
			auto const adjustment = flags & std::ios_base::adjustfield;
			if (adjustment != std::ios_base::left && adjustment != std::ios_base::internal) pad();
			write(prefix, prefix_size);
			if (adjustment == std::ios_base::internal) pad();
			write(begin, digit_size);
			if (adjustment == std::ios_base::left) pad();
		} catch (...) {
			os.setstate(std::ios_base::badbit);
			return os;
		}
		os.setstate(state);
		return os;
	}

	template<typename Accumulator>
		requires(std::same_as<Accumulator, uint64> || std::same_as<Accumulator, uint128>)
	struct stream_magnitude {
		Accumulator value = 0;
		bool is_negative = false;
		std::ios_base::iostate state = std::ios_base::goodbit;
	};

	/**
	 * @brief Parse one optional sign and a checked magnitude, without grouping or base autodetection.
	 * The caller commits only if state has neither failbit nor badbit, then applies state to the stream;
	 * this permits committing a valid value before an eofbit exception is raised.
	 */
	template<typename Accumulator>
		requires(std::same_as<Accumulator, uint64> || std::same_as<Accumulator, uint128>)
	[[nodiscard]] stream_magnitude<Accumulator> read_stream_magnitude(std::istream &is) {
		stream_magnitude<Accumulator> result;
		std::istream::sentry const sentry(is);
		if (!sentry) {
			result.state = std::ios_base::failbit;
			return result;
		}
		try {
			using Traits = std::char_traits<char>;
			auto *buffer = is.rdbuf();
			auto ch = buffer->sgetc();
			if (ch == Traits::to_int_type('-') || ch == Traits::to_int_type('+')) {
				result.is_negative = ch == Traits::to_int_type('-');
				ch = buffer->snextc();
			}
			auto const basefield = is.flags() & std::ios_base::basefield;
			unsigned const base = basefield == std::ios_base::hex ? 16u : basefield == std::ios_base::oct ? 8u
																										  : 10u;
			bool has_digits = false;
			// A hexadecimal prefix is optional only with an explicitly hexadecimal base.
			if (base == 16 && ch == Traits::to_int_type('0')) {
				ch = buffer->snextc();
				has_digits = true;
				if (ch == Traits::to_int_type('x') || ch == Traits::to_int_type('X')) {
					ch = buffer->snextc();
					has_digits = false;
				}
			}
			constexpr auto maximum = (std::numeric_limits<Accumulator>::max)();
			Accumulator const quotient = maximum / base;
			unsigned const remainder = static_cast<unsigned>(maximum % base);
			bool has_overflow = false;
			while (!Traits::eq_int_type(ch, Traits::eof())) {
				unsigned digit = base;
				if (ch >= Traits::to_int_type('0') && ch <= Traits::to_int_type('9'))
					digit = static_cast<unsigned>(ch - Traits::to_int_type('0'));
				else if (ch >= Traits::to_int_type('a') && ch <= Traits::to_int_type('f'))
					digit = static_cast<unsigned>(ch - Traits::to_int_type('a')) + 10u;
				else if (ch >= Traits::to_int_type('A') && ch <= Traits::to_int_type('F'))
					digit = static_cast<unsigned>(ch - Traits::to_int_type('A')) + 10u;
				if (digit >= base) break;
				has_digits = true;
				if (!has_overflow) {
					has_overflow = result.value > quotient || (result.value == quotient && digit > remainder);
					if (!has_overflow) result.value = result.value * base + digit;
				}
				ch = buffer->snextc();
			}
			if (Traits::eq_int_type(ch, Traits::eof())) result.state |= std::ios_base::eofbit;
			if (!has_digits || has_overflow) result.state |= std::ios_base::failbit;
		} catch (...) {
			is.setstate(std::ios_base::badbit);
			result.state |= std::ios_base::badbit;
		}
		return result;
	}
}
