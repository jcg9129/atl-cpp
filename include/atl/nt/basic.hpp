#pragma once

#include <atl/config.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/primality.hpp>
#include <atl/nt/sieve.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief Find a non-trivial factor of the composite `n` using Pollard's rho algorithm.
		 *
		 * Retries with successive values of the polynomial constant `c` until the
		 * cycle-detection walk yields a proper divisor; the running time has
		 * an expected bound, rather than a fixed retry limit. The walk runs in Montgomery form:
		 * `x_M - y_M = (x - y)·R` and `R` is coprime to `n`, so every gcd is the
		 * same as on plain residues.
		 *
		 * @param n The composite number to split.
		 * @return A divisor `d` of `n` with `1 < d < n`; not necessarily prime.
		 *
		 * @pre `n` is composite and greater than 1.
		 *
		 * @complexity expected O(n^(1/4))
		 */
		constexpr uint64 nontrivial_factor(uint64 n) {
			atl_assert(n > 1 && !is_prime(n));
			if (n % 2 == 0) return 2;
			// Brent's cycle detection with batched gcds: |x - y| differences are
			// accumulated as a running product mod n so only one gcd is taken per
			// `batch` steps (a gcd costs ~10 multiplies). When the batch trips (the
			// product shares a factor with n, or hit 0 on a self-collision), the
			// last batch is replayed one gcd per step from `ys` to recover the
			// factor, since the product may have absorbed *all* prime factors of n.
			montgomery64 const mg(n);
			constexpr uint64 batch = 128;
			for (uint64 c = 1;; ++c) {
				uint64 const cm = mg.to_form(c % n);
				auto next = [&](uint64 v) { return mg.add(mg.mul(v, v), cm); };
				uint64 y = mg.to_form(2), q = mg.one(), g = 1, x = 0, ys = 0;
				for (uint64 r = 1; g == 1; r <<= 1) {
					x = y;
					for (uint64 i = 0; i < r; ++i) y = next(y);
					for (uint64 k = 0; k < r && g == 1; k += batch) {
						ys = y;
						for (uint64 i = 0, m = (std::min)(batch, r - k); i < m; ++i) {
							y = next(y);
							q = mg.mul(q, x > y ? x - y : y - x);
						}
						g = std::gcd(q, n);
					}
				}
				if (g == n) {
					do {
						ys = next(ys);
						g = std::gcd(x > ys ? x - ys : ys - x, n);
					} while (g == 1);
				}
				if (g != n) return g;
			}
		}
	}

	/**
	 * @brief Returns the prime factorization of n.
	 * @param n The number to factorize.
	 * @return The distinct prime factors of `n`, in ascending order, each paired
	 *         with its exponent. Empty when `n <= 1`.
	 *
	 * Runs in roughly O(n^{1/4}) time by combining Pollard's rho for factor
	 * extraction with a deterministic Miller-Rabin primality test. The walk
	 * uses successive uint64 polynomial constants, independent of hashing and size_t.
	 *
	 * @complexity expected O(n^(1/4) polylog n)
	 */
	[[nodiscard]] constexpr std::vector<prime_factor> factorization(uint64 n) {
		std::vector<prime_factor> factors;
		if (n <= 1) return factors;
		if (int const twos = std::countr_zero(n); twos > 0) {
			factors.push_back({2, uint64(twos)});
			n >>= twos;
		}
		// Trial bound 256 reduces rho work on 2000 smooth inputs: 11.1 vs 11.9 ms for bound 0
		// (benchmarks/nt/factorization.cpp, Xeon Platinum 8280, GCC 15.2, RelWithDebInfo,
		// -O2 -g -DNDEBUG -std=c++23, assertions disabled, 2026-10-15; docs/factorization.md).
		for (uint64 p = 3; p < 256 && p * p <= n; p += 2) {
			if (n % p != 0) continue;
			uint64 e = 0;
			do {
				n /= p;
				++e;
			} while (n % p == 0);
			factors.push_back({p, e});
		}
		std::vector<uint64> primes, stack;
		if (n > 1) stack.push_back(n);
		while (!stack.empty()) {
			uint64 const m = stack.back();
			stack.pop_back();
			if (is_prime(m)) {
				primes.push_back(m);
			} else {
				uint64 const d = detail::nontrivial_factor(m);
				stack.push_back(d);
				stack.push_back(m / d);
			}
		}
		std::ranges::sort(primes);
		for (std::size_t i = 0; i < primes.size();) {
			std::size_t j = i;
			while (j < primes.size() && primes[j] == primes[i]) ++j;
			factors.push_back({primes[i], uint64(j - i)});
			i = j;
		}
		// already ascending: what reached the stack has no prime factor below
		// the last trial divisor, and every prime divided out lies below it
		return factors;
	}
}
