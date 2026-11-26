#pragma once

#include <atl/types/alias.hpp>
#include <bit>
#include <span>

namespace atl::detail {
	/** @brief Strong probable-prime test on reduced Montgomery or ordinary residues, for odd n > 37. */
	template<typename Arithmetic>
	constexpr bool miller_rabin_impl(uint64 n, std::span<uint64 const> bases, Arithmetic const &mg) {
		int const s = std::countr_zero(n - 1);
		uint64 const d = (n - 1) >> s;
		uint64 const one = mg.one(), minus_one = n - one; // Montgomery forms of 1 and n - 1
		for (uint64 a : bases) {
			uint64 const base = a % n;
			if (base == 0) continue;
			auto x = mg.power(mg.to_form(base), d);
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

	/** @brief Deterministic 64-bit primality; construct arithmetic only after trial division. */
	template<typename MakeArithmetic>
	constexpr bool is_prime_impl(uint64 n, MakeArithmetic make_arithmetic) {
		if (n < 2) return false;
		constexpr uint64 small_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
		for (uint64 p : small_primes)
			if (n % p == 0) return n == p;
		auto const arithmetic = make_arithmetic(n);
		if (n < 4'759'123'141ull) {
			constexpr uint64 bases[] = {2, 7, 61};
			return miller_rabin_impl(n, bases, arithmetic);
		}
		constexpr uint64 bases[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
		return miller_rabin_impl(n, bases, arithmetic);
	}
}
