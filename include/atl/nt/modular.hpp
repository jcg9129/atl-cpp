#pragma once

#include <atl/nt/gcd.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/detail/wide_arithmetic.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <atl/utils/assert.hpp>

namespace atl {
	/**
	 * @brief Add two already-reduced values modulo m, avoiding overflow.
	 * @param a First reduced operand.
	 * @param b Second reduced operand.
	 * @param m Positive modulus.
	 * @return The reduced result in `[0, m)`.
	 * @pre `a < m` and `b < m`.
	 * @complexity O(1)
	 * @tparam T Unsigned integral residue and modulus type.
	 */
	template<unsigned_integral T>
	constexpr inline T add_mod_unchecked(T a, T b, T m) {
		atl_assert(a < m);
		atl_assert(b < m);
		b = T(m - b);
		return T((a >= b) ? a - b : (m - b) + a);
	}

	/**
	 * @brief Subtract two already-reduced values modulo m, avoiding underflow.
	 * @param a First reduced operand.
	 * @param b Second reduced operand.
	 * @param m Positive modulus.
	 * @return The reduced result in `[0, m)`.
	 * @pre `a < m` and `b < m`.
	 * @complexity O(1)
	 * @tparam T Unsigned integral residue and modulus type.
	 */
	template<unsigned_integral T>
	constexpr inline T sub_mod_unchecked(T a, T b, T m) {
		atl_assert(a < m);
		atl_assert(b < m);
		return T((a >= b) ? a - b : (m - b) + a);
	}

	/**
	 * @brief Multiply two already-reduced values modulo m without a wider type,
	 *        using binary (double-and-add) multiplication.
	 *
	 * The portable fallback used when no integer type twice as wide as T exists
	 * (e.g. T == uint128, which has no 256-bit integer counterpart).
	 *
	 * @param a First reduced operand.
	 * @param b Second reduced operand.
	 * @param m Positive modulus.
	 * @return The reduced result in `[0, m)`.
	 * @pre `a < m` and `b < m`.
	 * @complexity O(log b)
	 * @tparam T Unsigned integral residue and modulus type.
	 */
	template<unsigned_integral T>
	constexpr inline T mul_mod_binary(T a, T b, T m) {
		atl_assert(a < m);
		atl_assert(b < m);
		T result = 0;
		while (b > 0) {
			if (b & 1) result = add_mod_unchecked(result, a, m);
			a = add_mod_unchecked(a, a, m);
			b >>= 1;
		}
		return result;
	}

	/**
	 * @brief Multiply two already-reduced values modulo m.
	 *
	 * Uses a double-width intermediate (see traits::widen_mul_type_t) when one
	 * exists; otherwise falls back to mul_mod_binary.
	 *
	 * @param a First reduced operand.
	 * @param b Second reduced operand.
	 * @param m Positive modulus.
	 * @return The reduced result in `[0, m)`.
	 * @pre `a < m` and `b < m`.
	 *
	 * @complexity O(1) with a wider type (native or software), otherwise O(log b)
	 * @tparam T Unsigned integral residue and modulus type.
	 */
	template<unsigned_integral T>
	constexpr inline T mul_mod_unchecked(T a, T b, T m) {
		atl_assert(a < m);
		atl_assert(b < m);
		if constexpr (traits::has_widen_mul_v<T>) {
			using W = traits::widen_mul_type_t<T>;
			return T(W(a) * W(b) % W(m));
		} else {
			return mul_mod_binary(a, b, m);
		}
	}

	/**
	 * @brief Barrett reducer for a fixed 32-bit modulus.
	 *
	 * Precomputes a reciprocal for a positive modulus through `2^32 - 1`.
	 * Multiplication uses the portable high-word product primitive to avoid
	 * division for already-reduced operands.
	 * @complexity O(1) setup and multiplication; O(1) storage.
	 */
	class barrett {
		uint32 m_ = 1;
		/** @brief `ceil(2^64 / m)`; 0 for `m == 1`, where every product is 0 anyway. */
		uint64 im_ = 0;

	public:
		/** @brief Construct the reducer for modulus 1. */
		constexpr barrett() = default;

		/**
		 * @brief Reducer for modulus @p m.
		 * @pre `m > 0` (asserted).
		 */
		constexpr explicit barrett(uint32 m) : m_(m), im_(m <= 1 ? 0 : (~uint64(0)) / m + 1) { atl_assert(m > 0); }

		/** @brief The modulus. */
		constexpr uint32 mod() const { return m_; }

		/**
		 * @brief `(a * b) mod m`.
		 * @param a First reduced operand.
		 * @param b Second reduced operand.
		 * @return The reduced product in `[0, mod())`.
		 * @pre `a < mod()` and `b < mod()` (asserted).
		 */
		constexpr uint32 mul(uint32 a, uint32 b) const {
			atl_assert(a < m_ && b < m_);
			uint64 z = uint64(a) * b;
			uint64 x = detail::multiply_high(z, im_);
			uint64 y = x * m_;
			return uint32(z - y + (z < y ? m_ : 0));
		}
	};

	namespace detail {
		/**
		 * @brief High 128 bits of the full 256-bit product `a * b`.
		 *
		 * Uses the shared limb-product primitive, including both middle carries. It
		 * provides the wide reciprocal multiply that `barrett64`
		 * needs, since no native 256-bit type is available to hold `a * b` directly.
		 *
		 * @complexity O(1)
		 */
		constexpr uint128 mulhi_u128(uint128 a, uint128 b) {
			auto const product = multiply_high_wide({uint64(a), uint64(a >> 64)}, {uint64(b), uint64(b >> 64)});
			return (uint128(product.high) << 64) | product.low;
		}
	}

	/**
	 * @brief Barrett reducer for a fixed 64-bit modulus.
	 *
	 * The 64-bit analogue of `barrett`: since `a * b` is now 128 bits, the
	 * reciprocal step multiplies two 128-bit values and takes the high 128 bits
	 * (`detail::mulhi_u128`). Correct for any 64-bit modulus with a single
	 * correction, using native or software uint128.
	 * @complexity O(1) setup and multiplication; O(1) storage.
	 */
	class barrett64 {
		uint64 m_ = 1;
		/** @brief `ceil(2^128 / m)`; 0 for `m == 1`, where every product is 0 anyway. */
		uint128 im_ = 0;

	public:
		/** @brief Construct the reducer for modulus 1. */
		constexpr barrett64() = default;

		/**
		 * @brief Reducer for modulus @p m.
		 * @pre `m > 0` (asserted).
		 */
		constexpr explicit barrett64(uint64 m) : m_(m), im_(m <= 1 ? uint128(0) : (~uint128(0)) / m + 1) {
			atl_assert(m > 0);
		}

		/** @brief The modulus. */
		constexpr uint64 mod() const { return m_; }

		/**
		 * @brief `(a * b) mod m`.
		 * @param a First reduced operand.
		 * @param b Second reduced operand.
		 * @return The reduced product in `[0, mod())`.
		 * @pre `a < mod()` and `b < mod()` (asserted).
		 */
		constexpr uint64 mul(uint64 a, uint64 b) const {
			atl_assert(a < m_ && b < m_);
			uint128 z = uint128(a) * b;
			uint128 q = detail::mulhi_u128(z, im_);
			uint128 y = q * m_;
			return uint64(z - y + (z < y ? uint128(m_) : uint128(0)));
		}
	};

	/**
	 * @brief Reduce a number modulo m.
	 *
	 * @return Returns a number in the range [0, m) that is congruent to `a` modulo m.
	 * @pre `m > 0` (asserted).
	 *
	 * @complexity O(1)
	 * @tparam I Integral source type.
	 * @tparam T Unsigned integral modulus and result type.
	 * @param a Integral value to reduce.
	 * @param m Positive modulus.
	 */
	template<integral I, unsigned_integral T>
	constexpr inline T reduce_mod(I a, T m) {
		atl_assert(m > 0);
		using U = traits::make_unsigned_t<I>;
		if constexpr (signed_integral<I>) {
			if (a < 0) {
				// Unsigned magnitude also represents the signed minimum exactly.
				U const magnitude = U(U(0) - U(a));
				T const remainder = T(magnitude % m);
				return remainder == 0 ? T(0) : T(m - remainder);
			}
		}
		return U(a) < m ? T(a) : T(U(a) % m);
	}

	/**
	 * @brief Add two values modulo m, reducing the operands first.
	 * @pre `m > 0` (asserted).
	 * @complexity O(1)
	 * @tparam T Unsigned integral residue and modulus type.
	 * @param a First operand, reduced internally.
	 * @param b Second operand, reduced internally.
	 * @param m Positive modulus.
	 * @return The reduced sum in [0, m).
	 */
	template<unsigned_integral T>
	constexpr inline T add_mod(T a, T b, T m) {
		a = reduce_mod(a, m);
		b = reduce_mod(b, m);
		return add_mod_unchecked(a, b, m);
	}

	/**
	 * @brief Subtract two values modulo m, reducing the operands first.
	 * @pre `m > 0` (asserted).
	 * @complexity O(1)
	 * @tparam T Unsigned integral residue and modulus type.
	 * @param a First operand, reduced internally.
	 * @param b Second operand, reduced internally.
	 * @param m Positive modulus.
	 * @return The reduced difference in [0, m).
	 */
	template<unsigned_integral T>
	constexpr inline T sub_mod(T a, T b, T m) {
		a = reduce_mod(a, m);
		b = reduce_mod(b, m);
		return sub_mod_unchecked(a, b, m);
	}

	/**
	 * @brief Multiply two values modulo m, reducing the operands first.
	 * @pre `m > 0` (asserted).
	 * @complexity O(1) with a wider type, otherwise O(log b)
	 * @tparam T Unsigned integral residue and modulus type.
	 * @param a First operand, reduced internally.
	 * @param b Second operand, reduced internally.
	 * @param m Positive modulus.
	 * @return The reduced product in [0, m).
	 */
	template<unsigned_integral T>
	constexpr inline T mul_mod(T a, T b, T m) {
		a = reduce_mod(a, m);
		b = reduce_mod(b, m);
		return mul_mod_unchecked(a, b, m);
	}

	/**
	 * @brief Raise `base` to `exp` modulo m via binary exponentiation.
	 *
	 * `base` is reduced first, so any value is accepted.
	 *
	 * @pre `m > 0` (asserted).
	 * @complexity O(log exp) modular multiplications; each is O(1) with a wider type, otherwise O(log m).
	 * @tparam T Unsigned integral residue and modulus type.
	 * @param base Value to exponentiate, reduced internally.
	 * @param exp Nonnegative exponent.
	 * @param m Positive modulus.
	 * @return The reduced power; 1 % m for exponent zero.
	 */
	template<unsigned_integral T>
	constexpr inline T pow_mod(T base, uint64 exp, T m) {
		atl_assert(m > 0);
		T result = T(1 % m);
		base %= m;
		while (exp > 0) {
			if (exp & 1) result = mul_mod_unchecked(result, base, m);
			base = mul_mod_unchecked(base, base, m);
			exp >>= 1;
		}
		return result;
	}

	/**
	 * @brief Increment a value modulo m, wrapping `m - 1` back to `0`.
	 * @pre `a < m`.
	 * @complexity O(1)
	 * @tparam T Unsigned integral residue and modulus type.
	 * @param a Reduced input.
	 * @param m Positive modulus.
	 * @return The reduced successor.
	 */
	template<unsigned_integral T>
	constexpr inline T incr_mod(T a, T m) {
		atl_assert(a < m);
		if (++a == m) return 0;
		return a;
	}

	/**
	 * @brief Decrement a value modulo m, wrapping `0` back to `m - 1`.
	 * @pre `a < m`.
	 * @complexity O(1)
	 * @tparam T Unsigned integral residue and modulus type.
	 * @param a Reduced input.
	 * @param m Positive modulus.
	 * @return The reduced predecessor.
	 */
	template<unsigned_integral T>
	constexpr inline T decr_mod(T a, T m) {
		atl_assert(a < m);
		if (a == 0) return T(m - 1);
		return --a;
	}

	/**
	 * @brief Negate a value modulo m, i.e. compute `(m - a) mod m`.
	 * @pre `a < m`.
	 * @complexity O(1)
	 * @tparam T Unsigned integral residue and modulus type.
	 * @param a Reduced input.
	 * @param m Positive modulus.
	 * @return The additive inverse in [0, m).
	 */
	template<unsigned_integral T>
	constexpr inline T neg_mod(T a, T m) {
		atl_assert(a < m);
		if (a == 0) return 0;
		return T(m - a);
	}

	/**
	 * @brief Modular multiplicative inverse of `v` modulo `m`, via the extended GCD.
	 * @pre `m > 0` and `gcd(v, m) == 1` (asserted).
	 * @return The unique value in [0, m) whose product with `v` is `1` modulo m.
	 * @complexity O(log m)
	 * @tparam T Unsigned integral residue and modulus type.
	 * @param v Value to invert; need not be reduced.
	 * @param m Positive modulus.
	 */
	template<unsigned_integral T>
	constexpr inline T inverse_mod(T v, T m) {
		atl_assert(m > 0);
		auto [g, x, _] = extended_gcd(v, m);
		atl_assert(g == 1);
		return reduce_mod(x, m);
	}
}
