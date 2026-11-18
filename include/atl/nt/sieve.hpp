#pragma once

#include <atl/ds/static_vector.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief What a linear sieve records about one value `i`: its smallest prime
	 *        factor, that prime's exponent in `i`, and the prime raised to it.
	 */
	struct sieve_item_info {
		/** @brief Smallest prime factor (1 for `i == 1`, 0 for `i == 0`). */
		uint32 prime;
		/** @brief Exponent of `prime` in `i`. */
		uint32 exponent;
		/** @brief `prime^exponent`. */
		uint32 prime_power;

		/** @brief Compare all recorded fields. */
		[[nodiscard]] constexpr bool operator==(sieve_item_info const &) const = default;
	};

	/**
	 * @brief A prime together with its exponent in a factorization.
	 */
	struct prime_factor {
		uint64 prime;	 ///< Prime base.
		uint64 exponent; ///< Positive exponent in the factorization.

		/** @brief Compare all recorded fields. */
		[[nodiscard]] constexpr bool operator==(prime_factor const &) const = default;
	};

	namespace detail {
		/**
		 * @brief The two products of a linear sieve pass, kept together so a single
		 * pass can serve both per-value factorization queries and prime enumeration.
		 */
		struct sieve_data {
			std::vector<sieve_item_info> info;
			std::vector<uint32> primes;
		};

		/**
		 * @brief Run the linear sieve over [0, n), producing the per-value info and
		 * the ascending prime list in one O(n) pass.
		 * @return info[i] describes `i` (see sieve_item_info); primes lists every
		 * prime below n in ascending order.
		 * @pre `n < 2^32` (asserted): values are stored as uint32.
		 * @complexity O(n)
		 */
		constexpr sieve_data linear_sieve(std::size_t n) {
			atl_assert(uint64(n) <= (std::numeric_limits<uint32>::max)());
			sieve_data d;
			d.info.resize(n);
			if (n > 1) d.info[1] = {1, 0, 1};

			for (uint32 i = 2; i < n; ++i) {
				if (d.info[i].exponent == 0) {
					d.info[i] = {i, 1, i};
					d.primes.push_back(i);
				}

				for (std::size_t j = 0; j < d.primes.size(); ++j) {
					uint64 const target = uint64(i) * d.primes[j];
					if (target >= n) break;

					if (i % d.primes[j] == 0) {
						d.info[static_cast<std::size_t>(target)] =
							{d.primes[j], d.info[i].exponent + 1, uint32(uint64(d.info[i].prime_power) * d.primes[j])};
						break;
					}
					d.info[static_cast<std::size_t>(target)] = {d.primes[j], 1, d.primes[j]};
				}
			}
			return d;
		}
	}

	/**
	 * @brief Tabulate least prime factors, their exponents and powers on [0, n).
	 * @param n Exclusive upper bound.
	 * @return n records; index 0 is {0,0,0}, index 1 is {1,0,1} when present.
	 * @pre `n < 2^32` (asserted).
	 * @complexity O(n) time and storage.
	 */
	[[nodiscard]] constexpr std::vector<sieve_item_info> sieve_info(std::size_t n) {
		return std::move(detail::linear_sieve(n).info);
	}

	/**
	 * @brief Compute all primes in the range [2, n)
	 * @param n The upper bound of the sieve. All values in the range [2, n) will be computed.
	 *
	 * @return A vector of all primes in the range [2, n)
	 *
	 * Uses a segmented odd-only byte sieve of Eratosthenes: O(sqrt(n)) memory
	 * besides the result plus one 64 KB block, versus the O(n) sieve_item_info
	 * structs a linear sieve would need to answer the same query.
	 *
	 * @pre `n < 2^32` (asserted): primes are returned as uint32.
	 * @complexity O(n log log n)
	 */
	[[nodiscard]] constexpr std::vector<uint32> prime_range(std::size_t n) {
		atl_assert(uint64(n) <= (std::numeric_limits<uint32>::max)());
		std::vector<uint32> primes;
		if (n < 3) return primes;
		// Segmented odd-only sieve: byte i of the whole range stands for the odd
		// number 2i+1, and the base primes up to sqrt(n) strike one cache-sized
		// block of bytes at a time, each resuming from its saved next index.

		std::size_t const m = n / 2; // odd values below n are 2i+1 for i in [0, m)
		uint32 root = 1;
		while (uint64(root + 1) * (root + 1) < n) ++root;
		std::vector<uint32> stride, next; // odd base primes, next byte each strikes
		{
			std::vector<uint8> composite(root / 2 + 1, 0);
			for (uint32 i = 1; 2 * i + 1 <= root; ++i)
				if (!composite[i]) {
					uint32 const p = 2 * i + 1;
					for (uint32 j = (p * p) / 2; 2 * j + 1 <= root; j += p) composite[j] = 1;
					stride.push_back(p);
					next.push_back(uint32((uint64(p) * p) / 2));
				}
		}
		primes.push_back(2);
		constexpr std::size_t block = std::size_t(1) << 16; // 64 KB: an L2-resident block
		std::vector<uint8> composite((std::min)(block, m));
		for (std::size_t lo = 0; lo < m; lo += block) {
			std::size_t const hi = (std::min)(m, lo + block);
			std::ranges::fill(composite, uint8(0));
			for (std::size_t k = 0; k < stride.size(); ++k) {
				std::size_t j = next[k];
				for (uint32 const p = stride[k]; j < hi; j += p) composite[j - lo] = 1;
				next[k] = uint32(j);
			}
			for (std::size_t i = (std::max)(lo, std::size_t(1)); i < hi; ++i)
				if (!composite[i - lo]) primes.push_back(uint32(2 * i + 1));
		}
		return primes;
	}

	/**
	 * @brief Linear-time sieve supporting multiple number-theoretic queries.
	 *
	 * Builds the sieve information for every value in the range [0, n) once, in
	 * O(n) time and memory, and then answers primality queries in O(1), factorizations in O(log i),
	 * and divisor enumeration with an O(d(i) log d(i)) sort.
	 *
	 * Primality accepts [0, size()); factorization and divisors require [1, size()),
	 * and minimum-prime-factor queries require [2, size()). All are asserted.
	 *
	 * @see atl::sieve_info
	 */
	class sieve {
		std::vector<sieve_item_info> info_;
		std::vector<uint32> primes_;

		/**
		 * @brief A value below 2^32 has at most 9 distinct prime factors
		 *        (2*3*5*7*11*13*17*19*23 < 2^32 <= that * 29).
		 */
		static constexpr std::size_t max_distinct_primes = 9;

	public:
		/**
		 * @brief Build the sieve for every value in the range [0, n).
		 * @param n Exclusive upper bound of the sieved range.
		 * @pre `n < 2^32` (asserted).
		 * @complexity O(n)
		 */
		constexpr explicit sieve(std::size_t n) {
			detail::sieve_data d = detail::linear_sieve(n);
			info_ = std::move(d.info);
			primes_ = std::move(d.primes);
		}

		/**
		 * @brief Number of values the sieve was built for.
		 * @return The construction bound n.
		 * @complexity O(1).
		 */
		constexpr std::size_t size() const { return info_.size(); }

		/**
		 * @brief Whether `i` is prime.
		 * @param i Value in [0, size()).
		 * @pre `i < size()` (asserted).
		 * @return True exactly for primes.
		 * @complexity O(1)
		 */
		[[nodiscard]] constexpr bool is_prime(std::size_t i) const {
			atl_assert(i < size());
			return i >= 2 && info_[i].prime == i;
		}

		/**
		 * @brief The smallest prime factor of `i`.
		 * @param i A value in the range [2, size()).
		 * @pre `2 <= i && i < size()` (asserted).
		 * @return The smallest prime divisor.
		 * @complexity O(1)
		 */
		[[nodiscard]] constexpr uint32 min_prime_factor(std::size_t i) const {
			atl_assert(2 <= i && i < size());
			return info_[i].prime;
		}

		/**
		 * @brief All primes in the range [2, size()), in ascending order.
		 * @return A reference valid until this sieve is assigned to, moved from or destroyed.
		 * @complexity O(1)
		 */
		constexpr std::vector<uint32> const &primes() const { return primes_; }

		/**
		 * @brief The prime factorization of `i`, in ascending order of prime.
		 * @param i A value in the range [1, size()).
		 * @pre `1 <= i && i < size()` (asserted).
		 * @return A vector of {prime, exponent}; empty when i == 1.
		 * @complexity O(omega(i)) time and O(1) bounded result storage; omega(i) <= 9.
		 */
		[[nodiscard]] constexpr static_vector<prime_factor, max_distinct_primes> factorization(std::size_t i) const {
			atl_assert(1 <= i && i < size());
			static_vector<prime_factor, max_distinct_primes> factors;
			// a 32-bit walk (size() fits uint32): the divide per prime is a 32-bit one
			for (uint32 v = uint32(i); v > 1;) {
				sieve_item_info const &it = info_[v];
				factors.push_back({it.prime, it.exponent});
				v /= it.prime_power;
			}
			return factors;
		}

		/**
		 * @brief The distinct prime factors of `i`, in ascending order.
		 * @return The prime bases; empty for 1.
		 * @param i A value in the range [1, size()).
		 * @pre `1 <= i && i < size()` (asserted).
		 * @complexity O(omega(i)) time and O(1) bounded result storage; omega(i) <= 9.
		 */
		[[nodiscard]] constexpr static_vector<uint32, max_distinct_primes> prime_factors(std::size_t i) const {
			atl_assert(1 <= i && i < size());
			static_vector<uint32, max_distinct_primes> primes;
			for (uint32 v = uint32(i); v > 1;) {
				sieve_item_info const &it = info_[v];
				primes.push_back(it.prime);
				v /= it.prime_power;
			}
			return primes;
		}

		/**
		 * @brief All divisors of `i`, in ascending order.
		 * @return The sorted positive divisors; {1} for 1.
		 * @param i A value in the range [1, size()).
		 * @pre `1 <= i && i < size()` (asserted).
		 * @complexity O(d(i) log d(i)) where d(i) is the divisor count
		 */
		[[nodiscard]] constexpr std::vector<uint32> divisors(std::size_t i) const {
			atl_assert(1 <= i && i < size());
			std::vector<uint32> divs = {1};
			for (prime_factor const &pf : factorization(i)) {
				std::size_t base = divs.size();
				uint32 power = 1;
				for (uint32 e = 1; e <= pf.exponent; ++e) {
					power *= uint32(pf.prime);
					for (std::size_t k = 0; k < base; ++k)
						divs.push_back(divs[k] * power);
				}
			}
			std::ranges::sort(divs);
			return divs;
		}
	};
}
