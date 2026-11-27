#pragma once

#include <atl/nt/detail/prime_mod_int_base.hpp>
#include <atl/nt/detail/tonelli_shanks.hpp>
#include <atl/nt/montgomery_static.hpp>
#include <atl/utils/assert.hpp>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace atl {

	/**
	 * @brief Square root in a static prime field.
	 * @tparam Coefficient Static prime-modulus integer type.
	 * @param input The residue.
	 * @return The smaller root as a residue (`root.value() <= mod / 2`), zero for zero, or nullopt for a non-residue.
	 * @complexity O(log(mod) + s^2) modular multiplications after the non-residue search, where s is the exponent of 2 in mod-1; O(1) storage.
	 */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::optional<Coefficient> square_root_mod(Coefficient input) {
		constexpr std::uint64_t prime = std::uint64_t(Coefficient::mod());
		using U = std::conditional_t<(prime <= (std::numeric_limits<uint32>::max)()), uint32, uint64>;
		atl_assert(input.value() < Coefficient::mod());
		U const value = U(input.value());
		if (value == 0) return Coefficient(0);
		if constexpr (prime == 2) {
			return input;
		} else {
			auto const root = detail::tonelli_shanks(value, prime, montgomery<U(prime)>{});
			if (!root) return std::nullopt;
			return Coefficient(*root);
		}
	}

}
