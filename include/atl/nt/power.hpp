#pragma once

#include <atl/types/concepts.hpp>
#include <concepts>
#include <utility>

namespace atl {
	/**
	 * @brief Raise `base` to a non-negative power via binary exponentiation,
	 *        seeding the accumulator with a caller-supplied `identity`.
	 *
	 * The form for monoids whose identity is not `Base(1)` because it depends
	 * on run-time state: a matrix's identity needs its dimension.
	 *
	 * @tparam Base Move-constructible monoid element with convertible, assignable products.
	 * @tparam Exponent An unsigned integer type.
	 * @param base The value to exponentiate.
	 * @param exp The non-negative exponent.
	 * @param identity The identity of `base`'s monoid, returned for `exp == 0`.
	 * @return The product of exp copies of base, or identity for exponent zero.
	 * @pre Multiplication is associative and identity is a two-sided identity.
	 * @pre Every evaluated multiplication is valid; signed integral products fit Base.
	 *
	 * @complexity O(log exp) multiplications and O(1) auxiliary Base objects.
	 */
	template<typename Base, unsigned_integral Exponent>
		requires requires(Base b) {
			{ b * b } -> std::convertible_to<Base>;
			b = b * b;
			Base(std::move(b));
		}
	constexpr Base power(Base base, Exponent exp, Base identity) {
		auto res = std::move(identity);
		while (exp > 0) {
			if (exp & 1)
				res = res * base;
			exp >>= 1;
			// An unused final square can overflow even when the requested power fits.
			if (exp > 0) base = base * base;
		}
		return res;
	}

	/**
	 * @brief Raise `base` to a non-negative power via binary exponentiation.
	 *
	 * @tparam Base Move-constructible element with convertible, assignable products
	 *              and a direct construction from `1`.
	 * @tparam Exponent An unsigned integer type.
	 * @param base The value to exponentiate.
	 * @param exp The non-negative exponent.
	 * @return `base` raised to the power `exp` (with `base^0 == Base(1)`).
	 * @pre Multiplication is associative and Base(1) is a two-sided identity.
	 * @pre Every evaluated signed integral product fits Base. Unsigned arithmetic wraps.
	 *
	 * @complexity O(log exp) multiplications and O(1) auxiliary Base objects.
	 */
	template<typename Base, unsigned_integral Exponent>
		requires requires(Base b) {
			{ b * b } -> std::convertible_to<Base>;
			b = b * b;
			Base(std::move(b));
			Base(1);
		}
	constexpr Base power(Base base, Exponent exp) {
		return power(std::move(base), exp, Base(1));
	}
}
