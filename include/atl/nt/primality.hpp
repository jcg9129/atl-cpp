#pragma once

/** @file
 * @brief Deterministic 64-bit primality without sieve or factorization storage.
 */

#include <atl/nt/detail/primality_base.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/types/alias.hpp>
#include <span>

namespace atl {
	namespace detail {
		/**
		 * @brief Miller-Rabin over Montgomery arithmetic for odd `n > 37` with no
		 *        prime factor up to 37: true iff no base in @p bases witnesses
		 *        compositeness (a base ≡ 0 mod n is skipped, as it proves nothing).
		 */
		constexpr bool miller_rabin(uint64 n, std::span<uint64 const> bases) {
			return miller_rabin_impl(n, bases, montgomery64(n));
		}
	}

	/**
	 * @brief Deterministic primality test for any 64-bit unsigned integer.
	 *
	 * Trial division by the primes up to 37, then Miller-Rabin on Montgomery
	 * arithmetic with a witness set proven deterministic for its range: {2, 7,
	 * 61} below 4,759,123,141 and the 7 bases {2, 325, 9375, 28178, 450775,
	 * 9780504, 1795265022} for every larger 64-bit n (deterministic witness sets).
	 *
	 * @param n The number to test.
	 * @return `true` if `n` is prime, `false` otherwise.
	 *
	 * @complexity O(log n) modular multiplications (deterministic Miller-Rabin)
	 */
	[[nodiscard]] constexpr bool is_prime(uint64 n) {
		return detail::is_prime_impl(n, [](uint64 modulus) { return detail::montgomery64(modulus); });
	}

}
