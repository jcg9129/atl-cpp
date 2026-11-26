#pragma once

#include <atl/types/alias.hpp>
#include <atl/types/detail/wide_arithmetic.hpp>
#include <atl/utils/assert.hpp>

namespace atl {
	namespace detail {
		/**
		 * @brief Montgomery arithmetic for one odd run-time 64-bit modulus `n`
		 *        (`R = 2^64`): a product is three 64x64 multiplies and no
		 *        division, against the library division `uint128 % uint64`
		 *        costs. This runtime counterpart of @ref atl::montgomery is the modular core of
		 *        `is_prime`, Pollard's rho and the discrete logarithm.
		 *        Values in Montgomery form lie in `[0, n)`.
		 */
		class montgomery64 {
			uint64 n_ = 1, n_inv_ = 1, r2_ = 0;

		public:
			/** @brief The modulus. */
			constexpr uint64 mod() const { return n_; }

			/**
			 * @brief Reducer for the odd modulus @p n (one 128-bit division, for
			 *        `R² mod n`).
			 * @pre `n` odd and `n > 1` (asserted).
			 */
			constexpr explicit montgomery64(uint64 n) : n_(n) {
				atl_assert(n > 1 && n % 2 == 1);
				// Newton: each step doubles the correct low bits of n⁻¹ mod 2^64
				uint64 inv = n;
				for (int i = 0; i < 5; ++i) inv *= 2 - n * inv;
				n_inv_ = inv;
				uint64 const r = (uint64(0) - n) % n; // 2^64 mod n
				r2_ = uint64(uint128(r) * r % n);
			}

			/**
			 * @brief `x·R⁻¹ mod n`.
			 * @pre `x < n·R`.
			 */
			constexpr uint64 reduce(uint128 x) const {
				uint64 const q = uint64(x) * n_inv_;
				uint64 const t = multiply_high(q, n_);
				uint64 const hi = uint64(x >> 64);
				return hi >= t ? hi - t : hi + (n_ - t);
			}

			/**
			 * @brief Product of two Montgomery-form values.
			 * @pre both `< n`.
			 */
			constexpr uint64 mul(uint64 a, uint64 b) const { return reduce(uint128(a) * b); }

			/** @brief Lift `a < n` into Montgomery form. */
			constexpr uint64 to_form(uint64 a) const {
				return mul(a, r2_);
			}

			/** @brief The residue of the Montgomery-form value @p x (in `[0, n)`). */
			constexpr uint64 from_form(uint64 x) const {
				return reduce(x);
			}

			/** @brief Montgomery form of 1. */
			constexpr uint64 one() const { return to_form(1); }

			/** @brief Sum of two Montgomery-form values. */
			constexpr uint64 add(uint64 a, uint64 b) const { return a >= n_ - b ? a - (n_ - b) : a + b; }

			/** @brief `base^exp` for a Montgomery-form base. */
			constexpr uint64 power(uint64 base, uint64 exp) const {
				uint64 r = one();
				for (; exp; exp >>= 1) {
					if (exp & 1) r = mul(r, base);
					base = mul(base, base);
				}
				return r;
			}
		};
	}
}
