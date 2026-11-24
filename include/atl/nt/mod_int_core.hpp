#pragma once

/** @file
 * @brief Static modular arithmetic without stream I/O; mod_int.hpp includes the stream extension.
 */

#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>

namespace atl {
	/**
	 * @brief Modular integer. All arithmetic operations are performed modulo Mod.
	 *
	 * @complexity O(1) storage; O(1) arithmetic through 64 bits, O(log Mod)
	 * multiplication at 128 bits. Inverse/division use O(log Mod) Euclidean steps,
	 * and power uses O(log exp) modular multiplications.
	 * @tparam T Unsigned integral residue type, including uint128.
	 * @tparam Mod The modulus of the modular integer (positive).
	 */
	template<unsigned_integral T, T Mod>
		requires(Mod > 0)
	class mod_int {
		T value_;
		using mod_ops = modulus_engine<Mod>;

		/**
		 * @brief Wrap a value already in the engine's stored form; unlike
		 *        @ref atl::mod_int::unchecked it does not lift a plain residue through `to_form`.
		 */
		static constexpr mod_int from_stored(T x) {
			mod_int a;
			a.value_ = x;
			return a;
		}

	public:
		/** @brief Return the positive compile-time modulus. */
		static constexpr inline T mod() { return Mod; }

		/** @brief Construct the zero residue. */
		constexpr mod_int() : value_(T(0)) {}

		/**
		 * @brief Reduce the integral argument v into [0, Mod).
		 * @tparam I Integral source type.
		 * @param v Value to reduce.
		 */
		template<integral I>
		constexpr mod_int(I v) { // NOLINT(google-explicit-constructor): integers convert, as for built-in arithmetic
			value_ = mod_ops::to_form(reduce_mod<I, T>(v, mod()));
		}

		/**
		 * @brief Wrap a plain residue, skipping the constructor's range reduction (it
		 *        is still lifted into the engine's stored form).
		 * @tparam I Integral source type.
		 * @param x Plain residue to wrap.
		 * @return A value representing x modulo the current modulus.
		 * @pre `0 <= x < mod()` (asserted).
		 */
		template<integral I>
		static constexpr mod_int unchecked(I x) {
			atl_assert(x >= I(0) && traits::make_unsigned_t<I>(x) < Mod);
			return from_stored(mod_ops::to_form(T(x)));
		}

		/** @brief Add rhs modulo the modulus and return this object. */
		constexpr mod_int &operator+=(mod_int const &rhs) {
			value_ = mod_ops::add(value_, rhs.value_);
			return *this;
		}

		/** @brief Subtract rhs modulo the modulus and return this object. */
		constexpr mod_int &operator-=(mod_int const &rhs) {
			value_ = mod_ops::sub(value_, rhs.value_);
			return *this;
		}

		/** @brief Multiply by rhs modulo the modulus and return this object. */
		constexpr mod_int &operator*=(mod_int const &rhs) {
			value_ = mod_ops::mul(value_, rhs.value_);
			return *this;
		}

		/**
		 * @brief Divide by rhs and return this object.
		 * @pre rhs is invertible (asserted).
		 */
		constexpr mod_int &operator/=(mod_int const &rhs) {
			return *this *= rhs.inverse();
		}

		/**
		 * @brief Multiplicative inverse, via the extended GCD.
		 * @pre `gcd(value(), mod()) == 1` (asserted).
		 * @return The multiplicative inverse in [0, Mod).
		 */
		constexpr mod_int inverse() const {
			return from_stored(mod_ops::inverse(value_));
		}

		/**
		 * @brief `*this` raised to `exp`, via the engine's exponentiation (stays in
		 *        the engine's stored form throughout).
		 */
		constexpr mod_int power(uint64 exp) const {
			return from_stored(mod_ops::power(value_, exp));
		}

		/** @brief Return the modular sum of lhs and rhs. */
		friend constexpr mod_int operator+(mod_int lhs, mod_int const &rhs) {
			return lhs += rhs;
		}

		/** @brief Return the modular difference of lhs and rhs. */
		friend constexpr mod_int operator-(mod_int lhs, mod_int const &rhs) {
			return lhs -= rhs;
		}

		/** @brief Return the modular product of lhs and rhs. */
		friend constexpr mod_int operator*(mod_int lhs, mod_int const &rhs) {
			return lhs *= rhs;
		}

		/**
		 * @brief Return lhs divided by rhs.
		 * @pre rhs is invertible (asserted).
		 */
		friend constexpr mod_int operator/(mod_int lhs, mod_int const &rhs) {
			return lhs /= rhs;
		}

		/** @brief Return this residue unchanged. */
		constexpr mod_int operator+() const {
			return *this;
		}

		/** @brief Return the additive inverse. */
		constexpr mod_int operator-() const {
			return from_stored(mod_ops::neg(value_));
		}

		/** @brief Return whether the stored representatives agree. */
		constexpr bool operator==(mod_int const &rhs) const {
			return value_ == rhs.value_;
		}

		/** @brief Increment modulo the modulus and return this object. */
		constexpr mod_int &operator++() {
			value_ = mod_ops::incr(value_);
			return *this;
		}

		/** @brief Increment modulo the modulus and return the saved value. */
		constexpr mod_int operator++(int) {
			mod_int saved = *this;
			operator++();
			return saved;
		}

		/** @brief Decrement modulo the modulus and return this object. */
		constexpr mod_int &operator--() {
			value_ = mod_ops::decr(value_);
			return *this;
		}

		/** @brief Decrement modulo the modulus and return the saved value. */
		constexpr mod_int operator--(int) {
			mod_int saved = *this;
			operator--();
			return saved;
		}

		/** @brief Return the plain representative in [0, mod()). */
		constexpr T value() const { return mod_ops::from_form(value_); }
	};

	/**
	 * @brief 32-bit modular integer alias.
	 * @tparam Mod Positive compile-time modulus.
	 */
	template<uint32 Mod>
	using mod_int32 = mod_int<uint32, Mod>;

	/**
	 * @brief 64-bit modular integer alias.
	 * @tparam Mod Positive compile-time modulus.
	 */
	template<uint64 Mod>
	using mod_int64 = mod_int<uint64, Mod>;
}
