#pragma once

#include <atl/nt/basic.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstdint>
#include <optional>

namespace atl::experimental {
	/** @brief An element `constant + extension * u` of the quadratic extension `u^2 = difference` used by Cipolla, in Montgomery form. */
	struct cipolla_extension_element {
		/** @brief Coefficient of 1. */
		std::uint64_t constant;
		/** @brief Coefficient of u. */
		std::uint64_t extension;
	};

	/**
	 * @brief Contender for `atl::square_root_mod`: Cipolla's method instead of Tonelli-Shanks. Race pending.
	 *
	 * Finds a parameter `a` with `a^2 - value` a non-residue and raises `a + u` to `(prime + 1) / 2` in the
	 * quadratic extension `u^2 = a^2 - value`; the constant part is the root. The work does not depend on the
	 * 2-adic valuation of `prime - 1`, unlike Tonelli-Shanks.
	 * @param value The residue, reduced into `[0, prime)`.
	 * @param prime A prime, including 2 and primes above 2^63.
	 * @return The smaller root, zero for zero, or nullopt for a non-residue; the same as `atl::square_root_mod`.
	 * @pre `prime` is prime and `value < prime` (both asserted).
	 * @complexity O(log(prime)) modular multiplications after the extension-parameter search, which tries
	 *             0, 1, ... and costs one exponentiation per candidate. O(1) memory.
	 */
	[[nodiscard]] constexpr std::optional<std::uint64_t> square_root_mod_by_cipolla(std::uint64_t value, std::uint64_t prime) {
		atl_assert(is_prime(prime));
		atl_assert(value < prime);
		if (value == 0 || prime == 2) return value;
		atl::detail::montgomery64 const arithmetic(prime);
		auto const one = arithmetic.one();
		auto const value_form = arithmetic.to_form(value);
		if (arithmetic.power(value_form, (prime - 1) / 2) != one) return std::nullopt;
		std::uint64_t parameter = 0, difference = 0;
		for (;;) {
			auto const parameter_form = arithmetic.to_form(parameter);
			difference = sub_mod_unchecked(arithmetic.mul(parameter_form, parameter_form), value_form, prime);
			if (difference != 0 && arithmetic.power(difference, (prime - 1) / 2) != one) break;
			atl_assert(parameter < prime - 1);
			++parameter;
		}
		auto const multiply = [&](cipolla_extension_element left, cipolla_extension_element right) {
			return cipolla_extension_element{
				arithmetic.add(arithmetic.mul(left.constant, right.constant), arithmetic.mul(arithmetic.mul(left.extension, right.extension), difference)),
				arithmetic.add(arithmetic.mul(left.constant, right.extension), arithmetic.mul(left.extension, right.constant))};
		};
		cipolla_extension_element result{one, 0}, base{arithmetic.to_form(parameter), one};
		// prime / 2 + 1 equals (prime + 1) / 2 and cannot overflow at 64 bits.
		auto exponent = prime / 2 + 1;
		while (exponent != 0) {
			if ((exponent & 1) != 0) result = multiply(result, base);
			exponent >>= 1;
			if (exponent != 0) base = multiply(base, base);
		}
		auto const root = arithmetic.from_form(result.constant);
		return (std::ranges::min)(root, prime - root);
	}

	/**
	 * @brief Static-prime-field overload of `square_root_mod_by_cipolla`. Race pending.
	 * @tparam Coefficient Static prime-modulus integer type.
	 * @param value The residue.
	 * @return The smaller root as a residue, zero for zero, or nullopt for a non-residue.
	 * @complexity As for the integer overload.
	 */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::optional<Coefficient> square_root_mod_by_cipolla(Coefficient value) {
		auto const root = square_root_mod_by_cipolla(std::uint64_t(value.value()), std::uint64_t(Coefficient::mod()));
		if (!root) return std::nullopt;
		return Coefficient(*root);
	}
}
