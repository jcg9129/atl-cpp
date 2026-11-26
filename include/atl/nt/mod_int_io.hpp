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

namespace atl::detail {
	template<typename Mint>
	struct mod_int_stream<Mint, void> {
		using T = decltype(Mint::mod());

		static std::ostream &write(std::ostream &os, Mint const &mi) {
			if constexpr (std::same_as<T, atl::uint128>)
				return write_uint128(os, mi.value());
			else
				return os << mi.value();
		}

		static std::istream &read(std::istream &is, Mint &mi) {
			auto const parsed = read_stream_magnitude<std::conditional_t<(sizeof(T) > sizeof(uint64)), uint128, uint64>>(is);
			if ((parsed.state & (std::ios_base::failbit | std::ios_base::badbit)) == 0) {
				Mint const reduced(parsed.value);
				mi = parsed.is_negative ? -reduced : reduced;
			}
			is.setstate(parsed.state);
			return is;
		}
	};
}
