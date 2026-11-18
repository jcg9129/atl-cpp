#pragma once
#include <atl/nt/multiplicative.hpp>
#include <atl/nt/sieve.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

/**
 * @file
 * @brief Race losers for nt/sieve.hpp and nt/multiplicative.hpp,
 *        measured in benchmarks/nt/sieve.cpp.
 */
namespace atl::experimental {
	/**
	 * @brief Test the least-prime record instead of computing i % p.
	 * @param n Exclusive bound, below 2^32 (asserted).
	 * @return The same records as atl::sieve_info.
	 * @complexity O(n) time and storage.
	 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, 2026-10-15,
	 * benchmarks/nt/sieve.cpp: by comparison / library, n=1e6: 5.7/5.7 ms
	 * (-O2), 5.8/6.5 ms (-O2 -march=native); n=1e7: 169/163 and 166/164 ms.
	 * The comparison gives no consistent gain across shapes and flags.
	 */
	constexpr std::vector<sieve_item_info> sieve_info_by_least_prime_comparison(std::size_t n) {
		atl_assert(uint64(n) <= (std::numeric_limits<uint32>::max)());
		std::vector<sieve_item_info> info(n);
		std::vector<uint32> primes;
		if (n > 1) info[1] = {1, 0, 1};
		for (uint32 i = 2; i < n; ++i) {
			if (info[i].exponent == 0) {
				info[i] = {i, 1, i};
				primes.push_back(i);
			}
			for (uint32 p : primes) {
				uint64 const target = uint64(i) * p;
				if (target >= n) break;
				if (p == info[i].prime) {
					info[static_cast<std::size_t>(target)] =
						{p, info[i].exponent + 1, uint32(uint64(info[i].prime_power) * p)};
					break;
				}
				info[static_cast<std::size_t>(target)] = {p, 1, p};
			}
		}
		return info;
	}

	/**
	 * @brief Strike one odd-only byte array instead of cache-sized segments.
	 * @param n Exclusive bound, below 2^32 (asserted).
	 * @return Ascending primes below n.
	 * @complexity O(n log log n) time and O(n) storage.
	 * benchmarks/nt/sieve.cpp, Xeon Platinum 8280, GCC 15.2, RelWithDebInfo,
	 * 2026-10-15, -O2: 22.3/14.9 ms at n=1e7 and 617/162 ms at n=1e8
	 * (unsegmented/library). Segmentation keeps strikes in cache.
	 */
	constexpr std::vector<uint32> prime_range_by_unsegmented_sieve(std::size_t n) {
		atl_assert(uint64(n) <= (std::numeric_limits<uint32>::max)());
		std::vector<uint32> primes;
		if (n < 3) return primes;
		primes.push_back(2);
		std::size_t const m = n / 2;
		std::vector<uint8> composite(m, 0);
		for (std::size_t i = 1; i < m; ++i)
			if (!composite[i]) {
				uint64 const p = 2 * uint64(i) + 1;
				primes.push_back(uint32(p));
				for (uint64 j = (p * p) / 2; j < m; j += p) composite[j] = 1;
			}
		return primes;
	}

	/**
	 * @brief Fuse prime-power callbacks and coprime multiplication into one sieve.
	 * @tparam R Same value requirements as atl::multiplicative_fn_range.
	 * @tparam Fn Same callable signature and result requirements as atl::multiplicative_fn_range.
	 * @param n Exclusive bound, below 2^32 (asserted).
	 * @param fn Prime-power rule invoked as `fn(p, e, v, pre)` for `v = p^e`.
	 *           For `e == 1`, entries below `p` are complete and may be read.
	 *           For `e > 1`, only entries through `v / p` (inclusive) are guaranteed
	 *           complete and may be read; the rest of the ascending prefix need not exist yet.
	 * @return n values with f(0) value-initialized and f(1)=R(1), when present.
	 * @pre The rule is multiplicative and every evaluated arithmetic operation is valid.
	 * @complexity O(n) products plus callback costs, O(n) storage.
	 * Four scattered tables cost more than the separate pass's divide:
	 * benchmarks/nt/sieve.cpp, Xeon Platinum 8280, GCC 15.2, RelWithDebInfo,
	 * 2026-10-15, -O2, n=1e7: totient 427/259 ms and divisor sum 500/299 ms
	 * (fused/library).
	 */
	template<atl::detail::multiplicative_value R, partial_multiplicative_function<R> Fn>
	constexpr std::vector<R> multiplicative_fn_range_by_fused_sieve(std::size_t n, Fn &&fn) {
		atl_assert(uint64(n) <= (std::numeric_limits<uint32>::max)());
		std::vector<R> f(n);
		std::vector<uint32> least_prime(n, 0), prime_power(n), cofactor(n), primes;
		std::vector<uint8> exponent(n);
		if (n > 1) f[1] = R(1);
		for (uint32 i = 2; i < n; ++i) {
			if (least_prime[i] == 0) {
				least_prime[i] = prime_power[i] = i;
				cofactor[i] = 1;
				exponent[i] = 1;
				primes.push_back(i);
				f[i] = std::invoke_r<R>(fn, uint64(i), uint64(1), uint64(i), f);
			}
			for (uint32 p : primes) {
				uint64 const target = uint64(i) * p;
				if (target >= n) break;
				least_prime[static_cast<std::size_t>(target)] = p;
				if (p == least_prime[i]) {
					uint32 const pp = prime_power[i] * p;
					prime_power[static_cast<std::size_t>(target)] = pp;
					cofactor[static_cast<std::size_t>(target)] = cofactor[i];
					exponent[static_cast<std::size_t>(target)] = uint8(exponent[i] + 1);
					f[static_cast<std::size_t>(target)] =
						cofactor[i] == 1
							? std::invoke_r<R>(fn, uint64(p),
											   uint64(exponent[static_cast<std::size_t>(target)]), target, f)
							: f[pp] * f[cofactor[i]];
					break;
				}
				prime_power[static_cast<std::size_t>(target)] = p;
				cofactor[static_cast<std::size_t>(target)] = i;
				exponent[static_cast<std::size_t>(target)] = 1;
				f[static_cast<std::size_t>(target)] = f[i] * f[p];
			}
		}
		return f;
	}
}
