#pragma once

#include <atl/nt/montgomery.hpp>
#include <atl/nt/primality.hpp>
#include <atl/nt/prime_mod_int_core.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>

namespace atl {
	/**
	 * @brief Square root of a residue modulo a prime.
	 * @param value The residue, reduced into `[0, prime)`.
	 * @param prime A prime, including 2 and primes above 2^63.
	 * @return The smaller of the two roots (`root <= prime / 2`), zero for zero, or nullopt when `value` is a
	 *         quadratic non-residue.
	 * @pre `prime` is prime and `value < prime` (both asserted, also for `value == 0`).
	 * @complexity For a prime that is 3 mod 4, two exponentiations (the residue check and the root). Otherwise O(log(prime) + s^2) modular
	 *             multiplications after the non-residue search, where s is the exponent of 2 in `prime - 1`;
	 *             the search tries 2, 3, ... and costs one exponentiation per candidate. O(1) memory.
	 * @note Tonelli-Shanks is the provisional default; the race against Cipolla
	 *       (`atl::experimental::square_root_mod_by_cipolla`) is pending.
	 */
	[[nodiscard]] constexpr std::optional<std::uint64_t> square_root_mod(std::uint64_t value, std::uint64_t prime) {
		atl_assert(is_prime(prime));
		atl_assert(value < prime);
		if (value == 0 || prime == 2) return value;
		detail::montgomery64 const arithmetic(prime);
		auto const one = arithmetic.one();
		auto const value_form = arithmetic.to_form(value);
		if (arithmetic.power(value_form, (prime - 1) / 2) != one) return std::nullopt;
		if (prime % 4 == 3) {
			auto const root = arithmetic.from_form(arithmetic.power(value_form, prime / 4 + 1));
			return (std::ranges::min)(root, prime - root);
		}
		int order = std::countr_zero(prime - 1);
		auto const odd_part = (prime - 1) >> order;
		std::uint64_t non_residue = 2;
		while (arithmetic.power(arithmetic.to_form(non_residue), (prime - 1) / 2) == one) {
			atl_assert(non_residue < prime - 1);
			++non_residue;
		}
		auto generator = arithmetic.power(arithmetic.to_form(non_residue), odd_part);
		auto remainder = arithmetic.power(value_form, odd_part);
		auto root = arithmetic.power(value_form, odd_part / 2 + 1);
		while (remainder != one) {
			int exponent = 0;
			for (auto power = remainder; power != one; power = arithmetic.mul(power, power)) ++exponent;
			// The remainder lies in the subgroup of squares of the current generator.
			atl_assert(exponent > 0 && exponent < order);
			auto correction = generator;
			for (int i = 0; i < order - exponent - 1; ++i) correction = arithmetic.mul(correction, correction);
			root = arithmetic.mul(root, correction);
			generator = arithmetic.mul(correction, correction);
			remainder = arithmetic.mul(remainder, generator);
			order = exponent;
		}
		auto const canonical = arithmetic.from_form(root);
		return (std::ranges::min)(canonical, prime - canonical);
	}

	/**
	 * @brief Square root in a static prime field.
	 * @tparam Coefficient Static prime-modulus integer type.
	 * @param value The residue.
	 * @return The smaller root as a residue (`root.value() <= mod / 2`), zero for zero, or nullopt for a non-residue.
	 * @complexity As for the integer overload.
	 */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::optional<Coefficient> square_root_mod(Coefficient value) {
		auto const root = square_root_mod(std::uint64_t(value.value()), std::uint64_t(Coefficient::mod()));
		if (!root) return std::nullopt;
		return Coefficient(*root);
	}
}
