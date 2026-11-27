#pragma once

#include <atl/nt/detail/tonelli_shanks.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/primality.hpp>
#include <atl/nt/prime_mod_int_core.hpp>
#include <atl/utils/assert.hpp>
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
		return detail::tonelli_shanks(value, prime, detail::montgomery64(prime));
	}

}

#include <atl/nt/detail/square_root_mod_static.hpp>
