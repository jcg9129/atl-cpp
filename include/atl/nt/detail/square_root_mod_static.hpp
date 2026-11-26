#pragma once

#include <atl/nt/detail/prime_mod_int_base.hpp>
#include <atl/nt/montgomery_static.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
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
			using arithmetic = montgomery<U(prime)>;
			auto const one = arithmetic::one();
			auto const value_form = arithmetic::to_form(value);
			if (arithmetic::power(value_form, (prime - 1) / 2) != one) return std::nullopt;
			if (prime % 4 == 3) {
				auto const root = arithmetic::from_form(arithmetic::power(value_form, prime / 4 + 1));
				return Coefficient((std::min)(uint64(root), prime - root));
			}
			int order = std::countr_zero(prime - 1);
			auto const odd_part = (prime - 1) >> order;
			U non_residue = 2;
			while (arithmetic::power(arithmetic::to_form(non_residue), (prime - 1) / 2) == one) {
				atl_assert(non_residue < prime - 1);
				++non_residue;
			}
			auto generator = arithmetic::power(arithmetic::to_form(non_residue), odd_part);
			auto remainder = arithmetic::power(value_form, odd_part);
			auto root = arithmetic::power(value_form, odd_part / 2 + 1);
			while (remainder != one) {
				int exponent = 0;
				for (auto power = remainder; power != one; power = arithmetic::mul(power, power)) ++exponent;
				// The remainder lies in the subgroup of squares of the current generator.
				atl_assert(exponent > 0 && exponent < order);
				auto correction = generator;
				for (int i = 0; i < order - exponent - 1; ++i) correction = arithmetic::mul(correction, correction);
				root = arithmetic::mul(root, correction);
				generator = arithmetic::mul(correction, correction);
				remainder = arithmetic::mul(remainder, generator);
				order = exponent;
			}
			auto const canonical = arithmetic::from_form(root);
			return Coefficient((std::min)(uint64(canonical), prime - canonical));
		}
	}

}
