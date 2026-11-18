#pragma once
#include <atl/nt/modular.hpp>
#include <atl/nt/static_modulus.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/detail/wide_arithmetic.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <atl/utils/assert.hpp>
#include <limits>

namespace atl {
	/**
	 * @brief Modular arithmetic with a compile-time **odd** modulus, via Montgomery
	 *        multiplication.
	 *
	 * Values are held in *Montgomery form*: the residue `a` is stored as `a·R mod Mod`,
	 * where `R = 2^bits` and bits is the width of value_type. Multiplication is then
	 * division-free (REDC: two multiplies, a shift, one conditional correction).
	 * Addition and subtraction are
	 * unchanged, since Montgomery form is additive-linear (`a·R + b·R = (a+b)·R`).
	 * Lift residues in and out with @ref atl::montgomery::to_form / @ref atl::montgomery::from_form.
	 *
	 * Like @ref atl::static_modulus, every arithmetic input is assumed already reduced —
	 * here, a reduced value *in Montgomery form*, as `mod_int` maintains.
	 *
	 * @complexity O(1) arithmetic and storage; O(log exp) power; O(log Mod) inverse.
	 * @tparam Mod An odd unsigned integer constant that has a double-width type
	 *             (i.e. up to 64 bits); its type is the type the operations work on.
	 */
	template<auto Mod>
		requires unsigned_integral<decltype(Mod)> && traits::has_widen_mul_v<decltype(Mod)> && ((Mod & 1) != 0)
	struct montgomery {
		/** @brief Unsigned word represented in Montgomery form. */
		using value_type = decltype(Mod);

		/** @brief Odd modulus; the full word range is supported. */
		static constexpr value_type mod = Mod;

	private:
		using wide = traits::widen_mul_type_t<value_type>;
		static_assert(traits::has_widen_mul_v<value_type>,
					  "montgomery needs a double-width type (Mod up to 64 bits)");
		static_assert((Mod & 1) != 0, "montgomery requires an odd modulus");

		/** @brief Width of value_type in bits; `R = 2^bits`. */
		static constexpr int bits = int(sizeof(value_type)) * 8;

		/**
		 * @brief `Mod⁻¹ mod R`, by Newton's iteration (`Mod` is odd, hence invertible mod
		 *        a power of two); each step doubles the number of correct low bits, so
		 *        six steps suffice through 64 bits.
		 */
		static constexpr value_type mod_inv = [] {
			value_type inv = 1;
			for (int i = 0; i < 6; ++i) inv = value_type(uint64(inv) * (uint64(2) - uint64(mod) * uint64(inv)));
			return inv;
		}();

		/** @brief `R² mod Mod`, used to lift a residue into Montgomery form. */
		static constexpr value_type r_squared = [] {
			value_type const r = value_type((wide(1) << bits) % mod); // R mod Mod
			return value_type((wide(r) * r) % mod);
		}();

		/**
		 * @brief Montgomery reduction: returns `x·R⁻¹ mod Mod`, division-free. Computed
		 *        from high halves so the intermediate never exceeds wide.
		 * @pre `x < Mod·R`.
		 */
		static constexpr value_type reduce(wide x) {
			value_type const q = value_type(uint64(value_type(x)) * uint64(mod_inv));
			value_type const hi = value_type((wide(q) * mod) >> bits);
			value_type const xhi = value_type(x >> bits);
			return xhi >= hi ? value_type(xhi - hi) : value_type(mod - (hi - xhi));
		}

	public:
		/**
		 * @brief Lift a plain residue into Montgomery form (`a·R mod Mod`).
		 * @pre `a < Mod`.
		 */
		static constexpr value_type to_form(value_type a) { return mul(a, r_squared); }

		/** @brief Recover the plain residue from a Montgomery-form value. */
		static constexpr value_type from_form(value_type a) { return reduce(wide(a)); }

		/** @brief Montgomery form of 1 (`R mod Mod`), the multiplicative identity. */
		static constexpr value_type one() { return to_form(value_type(1 % mod)); }

		/**
		 * @brief Sum of two Montgomery-form values.
		 * @pre `a < Mod` and `b < Mod`.
		 */
		static constexpr value_type add(value_type a, value_type b) {
			if constexpr (mod <= (std::numeric_limits<value_type>::max)() / 2) {
				value_type const r = value_type(a + b);
				return value_type(r >= mod ? r - mod : r);
			} else {
				// Subtract the modulus gap to keep the sum within the word range.
				return value_type(b != 0 && a >= mod - b ? a - (mod - b) : a + b);
			}
		}

		/**
		 * @brief Difference of two Montgomery-form values.
		 * @pre `a < Mod` and `b < Mod`.
		 */
		static constexpr value_type sub(value_type a, value_type b) {
			return value_type(a >= b ? a - b : mod - (b - a));
		}

		/**
		 * @brief Negation of a Montgomery-form value.
		 * @pre `a < Mod`.
		 */
		static constexpr value_type neg(value_type a) { return a == 0 ? value_type(0) : value_type(mod - a); }

		/**
		 * @brief `a + 1` in Montgomery form.
		 * @pre `a < Mod`.
		 */
		static constexpr value_type incr(value_type a) { return add(a, one()); }

		/**
		 * @brief `a - 1` in Montgomery form.
		 * @pre `a < Mod`.
		 */
		static constexpr value_type decr(value_type a) { return sub(a, one()); }

		/**
		 * @brief Product of two Montgomery-form values, itself in Montgomery form.
		 * @pre `a < Mod` and `b < Mod`.
		 */
		static constexpr value_type mul(value_type a, value_type b) { return reduce(wide(wide(a) * wide(b))); }

		/**
		 * @brief Montgomery reduction without the final tightening: result in
		 *        [0, 2·Mod). With `Mod <= (R-1)/4` (the requires-clause) one factor may be
		 *        lazy (< 2·Mod) against a tight one, or both lazy: the product stays
		 *        below `Mod·R`. Callers can postpone tightening.
		 * @pre `x < Mod·R`.
		 */
		static constexpr value_type reduce_lazy(wide x)
			requires(Mod <= (std::numeric_limits<value_type>::max)() / 4)
		{
			value_type const q = value_type(uint64(value_type(x)) * uint64(mod_inv));
			value_type const hi = value_type((wide(q) * mod) >> bits);
			return value_type(value_type(x >> bits) + mod - hi);
		}

		/**
		 * @brief Lazy product, output in [0, 2·Mod).
		 * @pre `a < 2·Mod` and `b < 2·Mod` (tight or lazy, per @ref atl::montgomery::reduce_lazy).
		 */
		static constexpr value_type mul_lazy(value_type a, value_type b)
			requires(Mod <= (std::numeric_limits<value_type>::max)() / 4)
		{
			return reduce_lazy(wide(wide(a) * wide(b)));
		}

		/**
		 * @brief Companion of a fixed multiplier for @ref atl::montgomery::mul_lazy_fixed,
		 *        `w·Mod⁻¹ mod R`, one low multiply.
		 * @pre `w < Mod`.
		 */
		static constexpr value_type companion(value_type w) { return value_type(uint64(w) * uint64(mod_inv)); }

		/**
		 * @brief @ref atl::montgomery::mul_lazy with a fixed multiplier whose companion is precomputed
		 *        (Shoup-style hoisting), output in [0, 2·Mod).
		 *
		 * The quotient estimate `q = x·wq` no longer depends on the wide product, so
		 * the two widening multiplies can run in parallel, shortening the dependency chain.
		 * @pre `x < 2·Mod`, `w < Mod` and `wq == companion(w)`.
		 */
		static constexpr value_type mul_lazy_fixed(value_type x, value_type w, value_type wq)
			requires(Mod <= (std::numeric_limits<value_type>::max)() / 4)
		{
			value_type const q = value_type(uint64(x) * uint64(wq));
			value_type const hi = value_type((wide(q) * mod) >> bits);
			value_type const xw = value_type((wide(x) * w) >> bits);
			return value_type(xw + mod - hi);
		}

		/**
		 * @brief `base^exp` in Montgomery form, by binary exponentiation.
		 * @pre `base < Mod`.
		 */
		static constexpr value_type power(value_type base, uint64 exp) {
			value_type result = one(), b = base;
			for (; exp; exp >>= 1) {
				if (exp & 1) result = mul(result, b);
				b = mul(b, b);
			}
			return result;
		}

		/**
		 * @brief Multiplicative inverse, in Montgomery form.
		 * @pre `a < Mod` and `gcd(from_form(a), Mod) == 1`.
		 */
		static constexpr value_type inverse(value_type a) { return to_form(inverse_mod(from_form(a), mod)); }
	};

	namespace detail {
		/**
		 * @brief Picks @ref atl::montgomery only for the moduli where REDC beats a
		 *        reciprocal: 64-bit, non-Mersenne, odd.
		 *
		 * The primary template is the fallback engine and never names @ref atl::montgomery,
		 * so an ineligible modulus (e.g. 128-bit, which has no double-width type)
		 * never instantiates it.
		 */
		template<auto Mod, bool UsesMontgomery>
			requires unsigned_integral<decltype(Mod)> && (Mod > 0)
		struct modulus_engine {
			using type = static_modulus<Mod>;
		};

		template<auto Mod>
			requires unsigned_integral<decltype(Mod)> && (Mod > 0)
		struct modulus_engine<Mod, true> {
			using type = montgomery<Mod>;
		};
	}

	/**
	 * @brief The modular-arithmetic engine @ref atl::mod_int uses for a given modulus:
	 *        @ref atl::montgomery for a 64-bit non-Mersenne odd modulus (where REDC wins)
	 *        when a 128-bit product type exists, @ref atl::static_modulus otherwise —
	 *        which is correct at every width and folds Mersenne moduli.
	 */
	template<auto Mod>
		requires unsigned_integral<decltype(Mod)> && (Mod > 0)
	using modulus_engine =
		typename detail::modulus_engine<Mod, (sizeof(decltype(Mod)) == 8) && traits::has_widen_mul_v<decltype(Mod)>
												 && !static_modulus<Mod>::is_mersenne && (Mod & 1)>::type;

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
