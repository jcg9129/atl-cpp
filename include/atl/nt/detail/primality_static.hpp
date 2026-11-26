#pragma once

#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <bit>
#include <limits>
#include <type_traits>

namespace atl::detail {
	/** @brief Deterministic static-modulus primality with the same witnesses as is_prime.
	 * The 32-bit case uses 64-bit products, so static field constraints need no runtime Montgomery reducer.
	 */
	template<auto Mod>
	constexpr bool static_is_prime() {
		constexpr uint64 n = uint64(Mod);
		if constexpr (n < 2) {
			return false;
		} else {
			constexpr uint64 small_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
			for (uint64 p : small_primes)
				if (n % p == 0) return n == p;
			using U = std::conditional_t<(n <= (std::numeric_limits<uint32>::max)()), uint32, uint64>;
			auto const multiply = [](uint64 a, uint64 b) { return uint64(mul_mod_unchecked(U(a), U(b), U(n))); };
			auto const exponentiate = [&](uint64 a, uint64 exponent) {
				uint64 result = 1;
				for (; exponent; exponent >>= 1, a = multiply(a, a))
					if (exponent & 1) result = multiply(result, a);
				return result;
			};
			int const order = std::countr_zero(n - 1);
			uint64 const odd_part = (n - 1) >> order;
			uint64 const bases[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
			uint64 const small_bases[] = {2, 7, 61};
			auto const witness = [&](uint64 a) {
				if (a % n == 0) return true;
				uint64 x = exponentiate(a % n, odd_part);
				if (x == 1 || x == n - 1) return true;
				for (int i = 1; i < order; ++i) {
					x = multiply(x, x);
					if (x == n - 1) return true;
				}
				return false;
			};
			if constexpr (n < 4'759'123'141ull) {
				for (uint64 a : small_bases)
					if (!witness(a)) return false;
			} else {
				for (uint64 a : bases)
					if (!witness(a)) return false;
			}
			return true;
		}
	}
}
