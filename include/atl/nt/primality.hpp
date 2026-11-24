#pragma once

/** @file
 * @brief Deterministic 64-bit primality without sieve or factorization storage.
 */

#include <atl/nt/montgomery.hpp>
#include <atl/types/alias.hpp>
#include <bit>
#include <span>

namespace atl {
	namespace detail {
		/**
		 * @brief Miller-Rabin over Montgomery arithmetic for odd `n > 37` with no
		 *        prime factor up to 37: true iff no base in @p bases witnesses
		 *        compositeness (a base ≡ 0 mod n is skipped, as it proves nothing).
		 */
		constexpr bool miller_rabin(uint64 n, std::span<uint64 const> bases) {
			montgomery64 const mg(n);
			int const s = std::countr_zero(n - 1);
			uint64 const d = (n - 1) >> s;
			uint64 const one = mg.one(), minus_one = n - one; // Montgomery forms of 1 and n - 1
			for (uint64 a : bases) {
				uint64 const base = a % n;
				if (base == 0) continue;
				uint64 x = mg.power(mg.to_form(base), d);
				if (x == one || x == minus_one) continue;
				bool is_composite = true;
				for (int i = 1; i < s && is_composite; ++i) {
					x = mg.mul(x, x);
					is_composite = x != minus_one;
				}
				if (is_composite) return false;
			}
			return true;
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
		if (n < 2) return false;
		constexpr uint64 small_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
		for (uint64 p : small_primes)
			if (n % p == 0) return n == p;
		if (n < 4'759'123'141ull) {
			constexpr uint64 bases[] = {2, 7, 61};
			return detail::miller_rabin(n, bases);
		}
		constexpr uint64 bases[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
		return detail::miller_rabin(n, bases);
	}

}
