#pragma once
#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <cstddef>

namespace atl::detail {
	/**
	 * @brief Minimal signed 256-bit accumulator for EXACT sign decisions on
	 *        expressions of 128-bit products — the arithmetic core of the
	 *        sweep-line event ordering (comparing rational intersection
	 *        points cross-multiplies to ~2^(5B+7) for B-bit coordinates,
	 *        past int128 even at 2^30). Two's-complement over four 64-bit
	 *        limbs, little-endian; only what the sign decisions need:
	 *        exact 128x128 products, add/sub, sign.
	 */
	struct int256 {
		/** @brief Little-endian two's-complement limbs. */
		uint64 limb[4]{};

		/**
		 * @brief The exact product of two signed 128-bit values: the unsigned
		 *        product of the two's-complement bits, then the signed correction
		 *        on the high half (`a < 0` subtracts `b·2^128`, `b < 0` subtracts
		 *        `a·2^128`), by masks. No negation and no branch on the signs: the
		 *        sign-magnitude form is 9.5-22% slower in compare_products
		 *        (benchmarks/types/int256.cpp; numbers in docs/int128.md).
		 */
		static constexpr int256 product(int128 a, int128 b) {
			uint128 const ua = uint128(a), ub = uint128(b);
			uint64 const a0 = uint64(ua), a1 = uint64(ua >> 64);
			uint64 const b0 = uint64(ub), b1 = uint64(ub >> 64);
			// schoolbook: four 64x64 -> 128 partials
			uint128 const p00 = uint128(a0) * b0;
			uint128 const p01 = uint128(a0) * b1;
			uint128 const p10 = uint128(a1) * b0;
			uint128 const p11 = uint128(a1) * b1;
			int256 r;
			r.limb[0] = uint64(p00);
			uint128 const mid = uint128(uint64(p01)) + uint64(p10) + uint64(p00 >> 64);
			r.limb[1] = uint64(mid);
			// the high half modulo 2^128, where the correction lives
			uint128 high = (p01 >> 64) + (p10 >> 64) + p11 + uint64(mid >> 64);
			high -= (ub & (uint128(0) - uint128(a < 0))) + (ua & (uint128(0) - uint128(b < 0)));
			r.limb[2] = uint64(high);
			r.limb[3] = uint64(high >> 64);
			return r;
		}

		/** @brief Two's-complement negation; `-(-2^255)` wraps to itself. */
		constexpr int256 operator-() const {
			int256 r;
			uint64 carry = 1;
			for (std::size_t i = 0; i < 4; ++i) {
				uint128 const s = uint128(~limb[i]) + carry;
				r.limb[i] = uint64(s);
				carry = uint64(s >> 64);
			}
			return r;
		}

		/**
		 * @brief Add @p other in place.
		 * @pre The exact sum lies in [-2^255, 2^255); overflow wraps silently.
		 */
		constexpr int256 &operator+=(int256 const &other) {
			uint64 carry = 0;
			for (std::size_t i = 0; i < 4; ++i) {
				uint128 const s = uint128(limb[i]) + other.limb[i] + carry;
				limb[i] = uint64(s);
				carry = uint64(s >> 64);
			}
			return *this;
		}

		/**
		 * @brief Subtract @p other in place.
		 * @pre The exact difference lies in [-2^255, 2^255).
		 */
		constexpr int256 &operator-=(int256 const &other) {
			uint64 borrow = 0;
			for (std::size_t i = 0; i < 4; ++i) {
				uint128 const d = uint128(limb[i]) - other.limb[i] - borrow;
				limb[i] = uint64(d);
				borrow = uint64(d >> 64) & 1; // the wrap sets every high bit
			}
			return *this;
		}

		friend constexpr int256 operator+(int256 a, int256 const &b) { return a += b; }

		friend constexpr int256 operator-(int256 a, int256 const &b) { return a -= b; }

		/** @brief -1 / 0 / +1 by the two's-complement top bit. */
		constexpr int sign() const {
			if (limb[3] >> 63) return -1;
			return (limb[0] | limb[1] | limb[2] | limb[3]) != 0 ? 1 : 0;
		}

		friend constexpr bool operator==(int256 const &, int256 const &) = default;
	};

	/**
	 * @brief Exact sign of `a*b - c*d` for signed 128-bit operands: the
	 *        cross-multiplied rational comparison in one call.
	 *
	 * Exact for every input: each product lies in `[-2^254 + 2^127, 2^254]`,
	 * so the difference stays inside `(-2^255, 2^255)` and never wraps.
	 */
	constexpr int compare_products(int128 const a, int128 const b, int128 const c,
								   int128 const d) {
		return (int256::product(a, b) - int256::product(c, d)).sign();
	}

	/**
	 * @brief Sign of three signed 128x128 products summed exactly.
	 * @pre Both the first two products' sum and the total fit signed int256.
	 */
	[[nodiscard]] constexpr int sum_three_products_sign(int128 const a, int128 const b, int128 const c,
														int128 const d, int128 const e, int128 const f) {
		return (int256::product(a, b) + int256::product(c, d) + int256::product(e, f)).sign();
	}

}
