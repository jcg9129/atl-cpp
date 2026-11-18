#pragma once
#include <atl/types/concepts.hpp>
#include <atl/types/traits/integral.hpp>

namespace atl {
	/**
	 * @brief Result of the extended Euclidean algorithm.
	 *
	 * Holds `gcd(a, b)` together with the Bézout coefficients `x` and `y`
	 * satisfying `a*x + b*y == gcd`. The coefficients are stored in the signed
	 * counterpart of `T` since they may be negative.
	 *
	 * @tparam T The unsigned type of the gcd (both @ref atl::extended_gcd overloads return
	 *           an unsigned gcd).
	 */
	template<unsigned_integral T>
	struct gcd_result {
		T gcd;						///< Nonnegative greatest common divisor; zero when both inputs are zero.
		traits::make_signed_t<T> x; ///< Signed coefficient of the first operand.
		traits::make_signed_t<T> y; ///< Signed coefficient of the second operand.
	};

	/**
	 * @brief Extended GCD algorithm for unsigned operands.
	 *
	 * Finds `x` and `y` such that `a*x + b*y == gcd(a, b)`. With `g = gcd(a, b) > 0`, the
	 * coefficients satisfy `|x| <= max(1, b / (2g))` and `|y| <= max(1, a / (2g))`, so
	 * they fit the signed counterpart of `U` over the full unsigned range: the update
	 * is lagged one step, so the final cofactor pair `(b/g, a/g)` is never formed, and
	 * a quotient above the signed maximum only arises when the remainder is 1, on the
	 * iteration that ends the loop.
	 *
	 * @tparam U Unsigned integral operand type.
	 * @param a First operand.
	 * @param b Second operand.
	 * @return The gcd and the Bézout coefficients (see gcd_result).
	 *
	 * @complexity O(log min(a, b)) time and O(1) storage.
	 */
	template<unsigned_integral U>
	constexpr gcd_result<U> extended_gcd(U const &a, U const &b) {
		using S = traits::make_signed_t<U>;

		S x1 = 1, y1 = 0;
		U a1 = a;
		S x0 = 0, y0 = 1;
		U a2 = b, q = 0;

		while (a2 != 0) {
			S x2 = S(x0 - static_cast<S>(q) * x1);
			S y2 = S(y0 - static_cast<S>(q) * y1);
			x0 = x1;
			y0 = y1;

			U a0 = a1;
			x1 = x2;
			y1 = y2;
			a1 = a2;

			q = U(a0 / a1);
			a2 = U(a0 - q * a1);
		}

		return {a1, x1, y1};
	}

	/**
	 * @brief Extended GCD algorithm for signed operands.
	 *
	 * Finds `x` and `y` such that `a*x + b*y == gcd(a, b)`. Handles the full
	 * signed range, including the signed minimum, without overflow.
	 *
	 * @tparam S Signed integral operand type.
	 * @param a First operand.
	 * @param b Second operand.
	 * @return The (non-negative, unsigned) gcd and the signed Bézout coefficients.
	 *
	 * @complexity O(log min(|a|, |b|)) time and O(1) storage.
	 */
	template<signed_integral S>
	constexpr gcd_result<traits::make_unsigned_t<S>> extended_gcd(S const &a, S const &b) {
		using U = traits::make_unsigned_t<S>;
		auto safe_abs = [](S value) -> U {
			// Unsigned subtraction represents the magnitude of the signed minimum.
			return value < 0 ? U(U(0) - U(value)) : U(value);
		};
		auto [gcd, x, y] = extended_gcd(safe_abs(a), safe_abs(b));
		if (a < 0) x = S(-x);
		if (b < 0) y = S(-y);
		return {gcd, x, y};
	}
}
