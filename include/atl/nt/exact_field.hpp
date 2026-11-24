#pragma once

/** @file
 * @brief The exact-field contract, independent of matrices and elimination.
 */

#include <atl/types/concepts.hpp>
#include <concepts>

namespace atl {
	/**
	 * @brief A field whose arithmetic is exact: `+ - * /`, `T(0)`, `T(1)`,
	 *        equality — `mod_int` / `dynamic_mod_int` with a PRIME modulus, or
	 *        a rational type. Dynamic-modulus arithmetic is runtime-only. Integers (whose `/` truncates) and floating
	 *        point (which needs pivoting and a tolerance, and has its own
	 *        overloads) are excluded.
	 *
	 * That every nonzero element is invertible is a precondition: for a
	 * modular type, a prime modulus. A composite one trips `inverse()`'s
	 * assert on the first non-unit pivot; use @ref atl::determinant_any_modulus.
	 */
	template<typename T>
	concept exact_field = std::regular<T> && !integral<T> && !std::floating_point<T>
					   && requires(T const a) {
							  { a + a } -> std::convertible_to<T>;
							  { a - a } -> std::convertible_to<T>;
							  { a * a } -> std::convertible_to<T>;
							  { a / a } -> std::convertible_to<T>;
							  T(0);
							  T(1);
						  };

}
