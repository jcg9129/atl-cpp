#pragma once

/** @file
 * @brief Exact nonnegative integer convolution with 64-bit results.
 */
#include <atl/nt/convolution_mod.hpp>
#include <atl/nt/mod_int_core.hpp>
#include <atl/nt/ntt_engine_core.hpp>
#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <type_traits>
#include <vector>

namespace atl {
	/**
	 * @brief Exact nonnegative integer polynomial product as uint64 coefficients.
	 * @tparam Engine Raw NTT engine; constant evaluation uses atl::scalar_ntt.
	 * @param a First sized random-access integral range.
	 * @param b Second sized random-access integral range.
	 * @return Empty if either input is empty; otherwise n+m-1 coefficients.
	 * @pre Inputs are nonnegative and fit uint64, input lengths are at most
	 *      `2^(size_t digits - 2)`, and the sufficient coefficient bound
	 *      `B = min(n,m) max(a) max(b) <= UINT64_MAX` holds (asserted).
	 *      Above the naive cutoff the planned
	 *      transform size is at most 2^23 (asserted).
	 * @complexity O((n+m) log(n+m)) time and O(n+m) auxiliary storage.
	 * @note Uses two-prime CRT when `B < 469762049*998244353`, otherwise
	 *       three-prime Garner with 167772161 as the third prime. Selection
	 *       uses an exact bound in every build, including without ATL_ASSERT.
	 *       Reconstruction uses portable atl::uint128.
	 */
	template<ntt_engine Engine = default_ntt_engine, integral_range A, integral_range B>
		requires integral_range<A const> && integral_range<B const>
	constexpr std::vector<uint64> convolution_uint64(A const &a, B const &b) {
		std::size_t const n = std::ranges::size(a), m = std::ranges::size(b);
		auto const first_a = std::ranges::begin(a);
		auto const first_b = std::ranges::begin(b);
		if (n == 0 || m == 0) return {};
		constexpr std::size_t limit = std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 2);
		atl_assert(n <= limit && m <= limit);
		std::size_t const need = n + m - 1;
		constexpr uint64 p1 = detail::crt_primes[1], p2 = detail::crt_primes[2];
		bool has_two_prime_bound = false;
		{
			uint128 top_a = 0, top_b = 0;
			for (std::size_t i = 0; i < n; ++i) {
				using Value = std::ranges::range_value_t<A>;
				Value const value = Value(first_a[i]);
				if constexpr (signed_integral<Value>) atl_assert(value >= 0);
				top_a = (std::max)(top_a, uint128(value));
			}
			for (std::size_t i = 0; i < m; ++i) {
				using Value = std::ranges::range_value_t<B>;
				Value const value = Value(first_b[i]);
				if constexpr (signed_integral<Value>) atl_assert(value >= 0);
				top_b = (std::max)(top_b, uint128(value));
			}
			atl_assert(top_a <= (std::numeric_limits<uint64>::max)() && top_b <= (std::numeric_limits<uint64>::max)());
			uint128 const product = top_a * top_b;
			uint128 const lower = uint128(uint64(product)) * (std::min)(n, m);
			// B = upper * 2^64 + uint64(lower). Each partial product fits uint128;
			// upper < 2^128 even for 64-bit size_t and maximal uint64 inputs.
			uint128 const upper = (product >> 64) * (std::min)(n, m) + (lower >> 64);
			has_two_prime_bound = upper == 0 && uint64(lower) < p1 * p2;
			atl_assert(upper == 0);
		}
		if ((std::min)(n, m) <= 48) {
			std::vector<uint64> c(need, 0);
			for (std::size_t i = 0; i < n; ++i)
				for (std::size_t j = 0; j < m; ++j) c[i + j] += uint64(first_a[i]) * uint64(first_b[j]);
			return c;
		}
		bool const is_square = detail::is_same_sequence(a, b);
		auto const run = [&]<uint32 P>(std::integral_constant<uint32, P>) {
			auto const at_a = [&](std::size_t i) { return uint32(uint64(first_a[i]) % P); };
			auto const at_b = [&](std::size_t i) { return uint32(uint64(first_b[i]) % P); };
			if consteval {
				return detail::raw_convolution<scalar_ntt, mod_int<uint32, P>>(n, at_a, m, at_b, is_square);
			} else {
				return detail::raw_convolution<Engine, mod_int<uint32, P>>(n, at_a, m, at_b, is_square);
			}
		};
		auto const r1 = run(std::integral_constant<uint32, uint32(p1)>{});
		auto const r2 = run(std::integral_constant<uint32, uint32(p2)>{});
		constexpr uint64 inv_p1_p2 =
			power(mod_int<uint32, detail::crt_primes[2]>(uint32(p1 % p2)), p2 - 2).value();
		std::vector<uint64> c(need);
		if (has_two_prime_bound) {
			for (std::size_t i = 0; i < need; ++i) {
				uint64 const v1 = r1[i];
				uint64 const v2 = (r2[i] + p2 - v1 % p2) * inv_p1_p2 % p2;
				c[i] = v1 + p1 * v2;
			}
		} else {
			constexpr uint64 p3 = detail::crt_primes[0];
			auto const r3 = run(std::integral_constant<uint32, uint32(p3)>{});
			constexpr uint64 inv_p12_p3 =
				power(mod_int<uint32, detail::crt_primes[0]>(uint32(p1 * p2 % p3)), p3 - 2).value();
			for (std::size_t i = 0; i < need; ++i) {
				uint64 const v1 = r1[i];
				uint64 const v2 = (r2[i] + p2 - v1 % p2) * inv_p1_p2 % p2;
				uint64 const t12 = v1 + p1 * v2;
				uint64 const v3 = (r3[i] + p3 - t12 % p3) * inv_p12_p3 % p3;
				// t12 < p1*p2 < 2^59; v3 < p3 < 2^28. Widening first
				// keeps the reconstruction below p1*p2*p3 < 2^87.
				c[i] = uint64(uint128(t12) + uint128(p1 * p2) * v3);
			}
		}
		return c;
	}
}
