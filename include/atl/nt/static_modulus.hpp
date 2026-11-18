#pragma once
#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <limits>

namespace atl {
	namespace detail {
		/** @brief Shared immutable reciprocal for the positive modulus M. */
		template<uint64 M>
			requires(M > 0)
		inline constexpr barrett64 static_barrett64{M};
	}

	/**
	 * @brief Modular arithmetic with a **compile-time** modulus.
	 *
	 * All arithmetic operands are residues in `[0, Mod)`. Products use
	 * a Mersenne fold when a double-width type is available, a reciprocal
	 * for other 64-bit moduli, and binary multiplication for 128-bit moduli.
	 * @complexity O(1) arithmetic through 64 bits; O(log Mod) multiplication at
	 * 128 bits. Power takes O(log exp) multiplications; inverse takes O(log Mod).
	 * Storage is O(1).
	 * @tparam Mod The modulus — a positive unsigned integer constant; its type is the
	 *             type the operations work on.
	 */
	template<auto Mod>
		requires unsigned_integral<decltype(Mod)> && (Mod > 0)
	struct static_modulus {
		/** @brief Unsigned residue type selected by the modulus. */
		using value_type = decltype(Mod);

		/** @brief Positive modulus, with the full value-type range supported. */
		static constexpr value_type mod = Mod;

		/**
		 * @brief Whether @p Mod has the form `2ᵏ − 1` with `k` below the type's width
		 *        (which enables the fold; primality is not required).
		 *
		 * The all-ones modulus of the full width is excluded: its fold `lo + hi`
		 * needs one more bit than `value_type` has, so it reduces like any other
		 * modulus.
		 */
		static constexpr bool is_mersenne =
			Mod > 1 && value_type(Mod + 1) != 0 && (value_type(Mod + 1) & Mod) == 0;

		/**
		 * @brief `(a + b) mod Mod`.
		 * @pre `a < Mod` and `b < Mod`.
		 */
		static constexpr value_type add(value_type a, value_type b) {
			if constexpr (mod <= (std::numeric_limits<value_type>::max)() / 2) {
				value_type const r = value_type(a + b);
				return value_type(r >= mod ? r - mod : r);
			} else {
				// Subtract the modulus gap to avoid forming a sum outside the value type.
				return value_type(b != 0 && a >= mod - b ? a - (mod - b) : a + b);
			}
		}

		/**
		 * @brief `(a - b) mod Mod`.
		 * @pre `a < Mod` and `b < Mod`.
		 */
		static constexpr value_type sub(value_type a, value_type b) {
			return value_type(a >= b ? a - b : a + (mod - b));
		}

		/**
		 * @brief `-a mod Mod`.
		 * @pre `a < Mod`.
		 */
		static constexpr value_type neg(value_type a) { return a == 0 ? value_type(0) : value_type(mod - a); }

		/**
		 * @brief `(a + 1) mod Mod`.
		 * @pre `a < Mod`.
		 */
		static constexpr value_type incr(value_type a) {
			value_type const r = value_type(a + 1);
			return r == mod ? value_type(0) : r;
		}

		/**
		 * @brief `(a - 1) mod Mod`.
		 * @pre `a < Mod`.
		 */
		static constexpr value_type decr(value_type a) { return value_type(a == 0 ? mod - 1 : a - 1); }

		/**
		 * @brief `(a · b) mod Mod`.
		 * @pre `a < Mod` and `b < Mod`.
		 */
		static constexpr value_type mul(value_type a, value_type b) {
			if constexpr (traits::has_widen_mul_v<value_type> && sizeof(value_type) == 8 && !is_mersenne) {
				// the compiler strength-reduces a constant `%` only up to 64 bits;
				// `uint128 % constant` is a library division, which a compile-time
				// Barrett reciprocal beat 4.4x (plain -O2) and 5.0x (-march=native)
				// on Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, 2026-10-15
				// (-g -DNDEBUG -std=c++23; 1e18 rows, benchmarks/nt/static_modulus.cpp)
				return detail::static_barrett64<uint64(Mod)>.mul(a, b);
			} else if constexpr (traits::has_widen_mul_v<value_type>) {
				using wide = traits::widen_mul_type_t<value_type>;
				return reduce(wide(wide(a) * wide(b)));
			} else {
				// No double-width type (e.g. uint128): O(log) double-and-add.
				return mul_mod_binary<value_type>(a, b, mod);
			}
		}

		/**
		 * @brief `base^exp mod Mod`, by binary exponentiation.
		 * @pre `base < Mod`.
		 */
		static constexpr value_type power(value_type base, uint64 exp) {
			value_type result = value_type(1 % mod), b = base;
			for (; exp; exp >>= 1) {
				if (exp & 1) result = mul(result, b);
				b = mul(b, b);
			}
			return result;
		}

		/**
		 * @brief Multiplicative inverse, via extended Euclid.
		 * @pre `a < Mod` and `gcd(a, Mod) == 1`.
		 */
		static constexpr value_type inverse(value_type a) { return inverse_mod(a, mod); }

		/**
		 * @brief Identity: static_modulus stores plain residues. The form conversions
		 *        let mod_int drive any engine uniformly; see @ref atl::modulus_engine.
		 */
		static constexpr value_type to_form(value_type a) { return a; }

		/** @brief Identity; see @ref atl::static_modulus::to_form. */
		static constexpr value_type from_form(value_type a) { return a; }

		/** @brief The multiplicative identity, `1 mod Mod`. */
		static constexpr value_type one() { return value_type(1 % mod); }

	private:
		/** @brief The `k` of a Mersenne modulus `2ᵏ − 1`, else 0. */
		static constexpr int mersenne_bits = [] {
			int width = 0;
			for (value_type value = Mod; value != 0; value >>= 1) ++width;
			return is_mersenne ? width : 0;
		}();

		/**
		 * @brief Reduce a double-width product. Mersenne: fold with
		 *        `2^mersenne_bits ≡ 1`; otherwise a constant modulo (the compiler emits
		 *        a reciprocal multiply).
		 * @pre `c < Mod²`.
		 */
		template<unsigned_integral Wide>
		static constexpr value_type reduce(Wide c) {
			if constexpr (is_mersenne) {
				value_type const lo = value_type(c) & mod, hi = value_type(c >> mersenne_bits);
				value_type const r = value_type(lo + hi);
				return value_type(r >= mod ? r - mod : r);
			} else {
				return value_type(c % mod);
			}
		}
	};
}
