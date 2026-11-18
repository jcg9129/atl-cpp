#pragma once

#include <atl/nt/power.hpp>
#include <atl/nt/sieve.hpp>
#include <atl/types/alias.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief A value type a multiplicative function can take: default-constructible
		 *        (the result table is sized up front), copyable, built from `1`, and
		 *        closed under multiplication.
		 */
		template<typename R>
		concept multiplicative_value = std::default_initializable<R> && std::copyable<R>
									&& requires(R const &a, R const &b) {
										   R(1);
										   { a * b } -> std::convertible_to<R>;
									   };
	}

	/**
	 * @brief Partial multiplicative function that knows how to compute the values at
	 *        prime powers: invoked as `fn(p, e, v, pre)` for `v = p^e`, where `pre`
	 *        holds the values already computed below `v`.
	 * @tparam Fn Callable invoked as an lvalue.
	 * @tparam R Result-table element type.
	 */
	template<typename Fn, typename R>
	concept partial_multiplicative_function =
		std::invocable<Fn &, uint64, uint64, uint64, std::vector<R> const &>
		&& std::convertible_to<std::invoke_result_t<Fn &, uint64, uint64, uint64, std::vector<R> const &>, R>;

	/**
	 * @brief Compute multiplicative function f at every value in the range [1, n)
	 *
	 * @tparam R Default-initializable, copyable value, directly constructible from 1,
	 *           with multiplication convertible to R.
	 * @tparam Fn Callable on an lvalue with (uint64, uint64, uint64, vector<R> const&).
	 * @param n The upper bound of the range.
	 * @param fn The partial multiplicative function.
	 *
	 * @return A vector of size n where f[i] is the value of the multiplicative function at i.
	 * @pre `n < 2^32` (asserted by the sieve).
	 * @pre fn returns the intended value at each prime power, and the function is
	 *      multiplicative with f(1) = R(1). All evaluated arithmetic is valid.
	 *      Index zero is value-initialized; callbacks may read only entries below v.
	 *
	 * @complexity O(n) multiplications plus callback costs; O(n) storage.
	 */
	template<detail::multiplicative_value R, partial_multiplicative_function<R> Fn>
	[[nodiscard]] constexpr std::vector<R> multiplicative_fn_range(std::size_t n, Fn &&fn) {
		auto info = sieve_info(n);
		std::vector<R> f(n);
		if (n > 1) f[1] = R(1);

		for (uint32 i = 2; i < n; ++i) {
			if (i == info[i].prime_power)
				f[i] = std::invoke_r<R>(fn, info[i].prime, info[i].exponent, i, f);
			else
				f[i] = f[info[i].prime_power] * f[i / info[i].prime_power];
		}

		return f;
	}

	/**
	 * @brief Compute Euler's totient function for every integer in [1, n).
	 *
	 * @param n The exclusive upper bound of the range.
	 * @return A vector of size n whose entry i is phi(i), for i in [1, n).
	 *
	 * @pre `n < 2^32` (asserted by the sieve).
	 * @complexity O(n) time and storage.
	 */
	[[nodiscard]] constexpr std::vector<uint32> euler_totient_range(std::size_t n) {
		return multiplicative_fn_range<uint32>(n, [](uint64 p, uint64, uint64 v, std::vector<uint32> const &) {
			return uint32(v / p * (p - 1));
		});
	}

	/**
	 * @brief Compute the divisor function sigma_k for every integer in [1, n).
	 *
	 * @param n The exclusive upper bound of the range.
	 * @param k The power of the divisor function.
	 * @return A vector of size n whose entry i is sigma_k(i), for i in [1, n).
	 *
	 * The divisor function is defined as the sum of the k-th powers of the divisors of n.
	 * `sigma_k(n) = sum_{d | n} d^k`.
	 *
	 * @pre `n < 2^32` (asserted); all evaluated arithmetic is valid in R.
	 * @tparam R Multiplicative value with direct uint64 construction, powers,
	 *           subtraction from 1 and a*step+1 convertible to R for a materialized step=a-1.
	 * @complexity O(n log(k + 2)) arithmetic operations and O(n) storage.
	 */
	template<detail::multiplicative_value R>
		requires requires(R const &a, uint64 v) {
			R(v);
			{ a + 1 } -> std::convertible_to<R>;
			{ a - 1 } -> std::convertible_to<R>;
			power(R(v), uint64(0));
			requires requires(decltype(a - 1) &prime_power_step) {
				{ a * prime_power_step + 1 } -> std::convertible_to<R>;
			};
		}
	[[nodiscard]] constexpr std::vector<R> divisor_power_sum_range(std::size_t n, uint64 k) {
		return multiplicative_fn_range<R>(n, [k](uint64 p, uint64 e, uint64 v, std::vector<R> const &f) -> R {
			if (e == 1) {
				return power(R(p), k) + 1;
			} else {
				auto pk = f[static_cast<std::size_t>(p)] - 1;
				return f[static_cast<std::size_t>(v / p)] * pk + 1;
			}
		});
	}

	/**
	 * @brief Compute the number of divisors of every integer in [1, n).
	 *
	 * @param n The exclusive upper bound of the range.
	 * @return A vector of size n whose entry i is d(i), for i in [1, n).
	 *
	 * @pre `n < 2^32` (asserted by the sieve).
	 * @complexity O(n) time and storage.
	 */
	[[nodiscard]] constexpr std::vector<uint32> divisor_count_range(std::size_t n) {
		return multiplicative_fn_range<uint32>(n, [](uint64, uint64 e, uint64, std::vector<uint32> const &) {
			return uint32(e + 1);
		});
	}

	/**
	 * @brief Compute the sum of divisors of every integer in [1, n).
	 *
	 * @param n The exclusive upper bound of the range.
	 * @return A vector of size n whose entry i is sigma_1(i), for i in [1, n).
	 *
	 * @pre `n < 2^32` (asserted by the sieve).
	 * @complexity O(n) time and storage.
	 */
	[[nodiscard]] constexpr std::vector<uint64> divisor_sum_range(std::size_t n) {
		return multiplicative_fn_range<uint64>(n, [](uint64 p, uint64, uint64 v, std::vector<uint64> const &) {
			return (v * p - 1) / (p - 1);
		});
	}
}
