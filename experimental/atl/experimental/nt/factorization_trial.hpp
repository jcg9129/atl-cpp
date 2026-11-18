#pragma once
#include <atl/nt/sieve.hpp>
#include <atl/types/alias.hpp>
#include <vector>

/** @file
 * @brief Trial division oracle and benchmark contender for atl::factorization.
 */
namespace atl::experimental {
	/**
	 * @brief Factor by increasing trial divisors, using a quotient bound to avoid overflow.
	 *
	 * Kept as a reference/benchmark counterpart to atl::factorization, which
	 * uses Pollard's rho + Miller-Rabin; same result contract.
	 * benchmarks/nt/factorization.cpp, Xeon Platinum 8280, GCC 15.2, RelWithDebInfo,
	 * 2026-10-15, -O2: 43.4 ms versus library 0.076 ms for 20 inputs near 1e12;
	 * trial division scales with sqrt(n).
	 * @param n Unsigned 64-bit value.
	 * @return The distinct prime factors of `n`, in ascending order, each
	 *         paired with its exponent. Empty when `n <= 1`.
	 * @complexity O(sqrt(n)) divisions, for every n < 2^64
	 */
	constexpr std::vector<prime_factor> factorization_by_trial_division(uint64 n) {
		std::vector<prime_factor> factors;
		// p <= n / p, not p * p <= n: the square wraps once p reaches 2^32
		for (uint64 p = 2; p <= n / p; ++p) {
			if (n % p == 0) {
				uint64 e = 0;
				while (n % p == 0) {
					++e;
					n /= p;
				}
				factors.push_back({p, e});
			}
		}
		if (n > 1) factors.push_back({n, 1});
		return factors;
	}
}
