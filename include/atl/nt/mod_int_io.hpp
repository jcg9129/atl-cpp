#pragma once

/** @file
 * @brief Stream insertion and extraction for static modular integers, including 128-bit residues.
 */

#include <atl/nt/detail/uint128_stream.hpp>
#include <atl/nt/mod_int_core.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <concepts>
#include <ios>
#include <istream>
#include <ostream>
#include <type_traits>

namespace atl {
	/**
	 * @brief Write the plain representative with integer formatting, including 128-bit residues.
	 * @pre T is uint128 or its insertion returns an ostream reference.
	 * @return os, with badbit if writing uint128 fails.
	 * @note Honors base, showbase, uppercase, locale grouping, width, fill and adjustment; resets width.
	 * @note Not constexpr (uses stream I/O).
	 */
	template<unsigned_integral T, T Mod>
		requires(Mod > 0)
	std::ostream &operator<<(std::ostream &os, mod_int<T, Mod> const &mi)
		requires(std::same_as<T, atl::uint128> || requires(std::ostream &s, T r) { { s << r } -> std::convertible_to<std::ostream &>; })
	{
		if constexpr (std::same_as<T, atl::uint128>)
			return detail::write_uint128(os, mi.value());
		else
			return os << mi.value();
	}

	/**
	 * @brief Read an optional sign and a decimal, octal or hexadecimal magnitude, then reduce it.
	 * @pre The magnitude fits uint128 for 128-bit residues, otherwise uint64 for successful extraction.
	 * @return is, with failbit on malformed or overflowing input; mi is unchanged on failure.
	 * @note Honors skipws; a sign must immediately precede the magnitude. Input grouping is unsupported.
	 *        Valid digits ending at EOF succeed with eofbit only. An unset basefield means decimal.
	 * @note Not constexpr (uses stream I/O).
	 */
	template<unsigned_integral T, T Mod>
		requires(Mod > 0)
	std::istream &operator>>(std::istream &is, mod_int<T, Mod> &mi) {
		auto const parsed = detail::read_stream_magnitude<std::conditional_t<(sizeof(T) > sizeof(uint64)), uint128, uint64>>(is);
		if ((parsed.state & (std::ios_base::failbit | std::ios_base::badbit)) == 0) {
			mod_int<T, Mod> const reduced(parsed.value);
			mi = parsed.is_negative ? -reduced : reduced;
		}
		is.setstate(parsed.state);
		return is;
	}

}
