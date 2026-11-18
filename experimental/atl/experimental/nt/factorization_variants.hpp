#pragma once
#include <atl/nt/basic.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/sieve.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

/**
 * @file
 * @brief Race losers and contenders for `is_prime` and `factorization` on
 *        64-bit inputs, measured in benchmarks/nt/factorization.cpp
 *        (2026-10-15).
 */
namespace atl::experimental {
	/**
	 * @brief Test primality by Miller-Rabin over the 12
	 *        prime bases up to 37 on `mul_mod_unchecked` (`uint128 % uint64`, a
	 *        library division per product).
	 *
	 * Division per product loses to the library Montgomery form with 3 or 7 bases on x86-64 (Xeon Platinum
	 * 8280, GCC 15.2, RelWithDebInfo, 2026-10-15, -O2): 1e4 primes near 2^62 266 ms vs 22.2 ms, near 1e9
	 * 51.5 ms vs 6.1 ms, 1e5 random 64-bit odds 200 ms vs 25.5 ms.
	 * @param n Unsigned 64-bit input.
	 * @return Whether n is prime.
	 * @complexity O(log n) modular products, O(1) storage.
	 */
	constexpr bool is_prime_by_twelve_bases(uint64 n) {
		if (n < 2) return false;
		constexpr uint64 witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
		for (uint64 p : witnesses)
			if (n % p == 0) return n == p;
		int const s = std::countr_zero(n - 1);
		uint64 const d = (n - 1) >> s;
		for (uint64 a : witnesses) {
			uint64 x = pow_mod(a, d, n);
			if (x == 1 || x == n - 1) continue;
			bool is_composite = true;
			for (int i = 1; i < s && is_composite; ++i) {
				x = mul_mod_unchecked(x, x, n);
				is_composite = x != n - 1;
			}
			if (is_composite) return false;
		}
		return true;
	}

	/**
	 * @brief Pollard's rho (Brent, batched gcds) on plain residues with
	 *        `mul_mod_unchecked` instead of Montgomery forms.
	 * @param n Odd composite to split.
	 * @return A proper divisor, not necessarily prime.
	 * @pre `n` composite and odd.
	 * @complexity Expected O(n^(1/4)) word steps and O(1) storage.
	 */
	constexpr uint64 nontrivial_factor_by_remainder_products(uint64 n) {
		constexpr uint64 batch = 128;
		for (uint64 c = 1;; ++c) {
			auto next = [&](uint64 v) { return add_mod_unchecked(mul_mod_unchecked(v, v, n), c % n, n); };
			uint64 y = 2, q = 1, g = 1, x = 0, ys = 0;
			for (uint64 r = 1; g == 1; r <<= 1) {
				x = y;
				for (uint64 i = 0; i < r; ++i) y = next(y);
				for (uint64 k = 0; k < r && g == 1; k += batch) {
					ys = y;
					for (uint64 i = 0, m = (std::min)(batch, r - k); i < m; ++i) {
						y = next(y);
						q = mul_mod_unchecked(q, x > y ? x - y : y - x, n);
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

	/**
	 * @brief The rho-and-test stack loop shared by the factorization variants:
	 *        appends the prime factors of `n` (no prime factor below those already in factors)
	 *        to `factors`, ascending, with exponents.
	 * @tparam IsPrime Lvalue-callable primality test returning a contextually bool-convertible result.
	 * @tparam Split Lvalue-callable composite splitter returning a uint64-convertible result.
	 * @param n Cofactor to factor.
	 * @param factors Ascending trial factors to append to; all remaining primes are larger.
	 * @param is_prime_fn Primality test.
	 * @param split Proper-divisor finder for odd composites.
	 */
	template<typename IsPrime, typename Split>
		requires requires(IsPrime &fn, uint64 m) { static_cast<bool>(std::invoke(fn, m)); }
			  && std::is_invocable_r_v<uint64, Split &, uint64>
	constexpr void factor_cofactor(uint64 n, std::vector<prime_factor> &factors, IsPrime is_prime_fn,
								   Split split) {
		std::vector<uint64> primes, stack;
		if (n > 1) stack.push_back(n);
		while (!stack.empty()) {
			uint64 const m = stack.back();
			stack.pop_back();
			if (std::invoke(is_prime_fn, m)) {
				primes.push_back(m);
			} else {
				uint64 const d = m % 2 == 0 ? 2 : std::invoke_r<uint64>(split, m);
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
	}

	/**
	 * @brief Factor every composite
	 *        straight into rho on `uint128 %` products, no trial division.
	 *
	 * Lost on x86-64 (Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, 2026-10-15, -O2): 200 semiprimes of two ~2^31
	 * primes 501 ms vs 75.7 ms for the library, 2000 random 64-bit values 228 vs
	 * 35.9 ms.
	 * @param n Unsigned 64-bit input.
	 * @return Ascending prime powers, empty for n<=1.
	 * @complexity Expected O(n^(1/4) polylog n), O(log n) storage.
	 */
	constexpr std::vector<prime_factor> factorization_by_remainder_products(uint64 n) {
		std::vector<prime_factor> factors;
		if (n <= 1) return factors;
		factor_cofactor(
			n, factors,
			[](uint64 value) { return is_prime_by_twelve_bases(value); },
			[](uint64 value) { return nontrivial_factor_by_remainder_products(value); });
		return factors;
	}

	/**
	 * @brief Factorization with the trial-division bound as a
	 *        parameter (`TrialBound <= 3`: twos stripped only), for the factorization
	 *        race (benchmarks/nt/factorization.cpp, Xeon Platinum 8280, GCC 15.2,
	 *        RelWithDebInfo, 2026-10-15, -O2). 256 reduces rho work on smooth inputs: 11.1 versus 11.9 ms for bound 0;
	 *        random values take 35.9 versus 36.8 ms. Semiprimes tie at 75.7 ms.
	 * @tparam TrialBound Exclusive bound for odd trial divisors.
	 * @param n Unsigned 64-bit input.
	 * @return Ascending prime powers, empty for n<=1.
	 * @complexity O(min(TrialBound,sqrt(n))) trial steps plus expected rho time; O(log n) storage.
	 */
	template<uint64 TrialBound>
	constexpr std::vector<prime_factor> factorization_by_bounded_trial(uint64 n) {
		std::vector<prime_factor> factors;
		if (n <= 1) return factors;
		if (int const twos = std::countr_zero(n); twos > 0) {
			factors.push_back({2, uint64(twos)});
			n >>= twos;
		}
		auto const square_within_cofactor = [&](uint64 p) {
			if constexpr (TrialBound <= (uint64(1) << 32))
				return p * p <= n;
			else
				return p <= n / p;
		};
		for (uint64 p = 3; p < TrialBound && square_within_cofactor(p); p += 2) {
			if (n % p != 0) continue;
			uint64 e = 0;
			do {
				n /= p;
				++e;
			} while (n % p == 0);
			factors.push_back({p, e});
		}
		factor_cofactor(
			n, factors,
			[](uint64 m) { return is_prime(m); },
			[](uint64 m) { return atl::detail::nontrivial_factor(m); });
		return factors;
	}
}
