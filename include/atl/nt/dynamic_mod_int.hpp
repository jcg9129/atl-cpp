#pragma once

#include <atl/nt/detail/uint128_stream.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/types/traits/is_one_of.hpp>
#include <atl/utils/assert.hpp>
#include <ios>
#include <istream>
#include <ostream>
#include <type_traits>

namespace atl {
	/**
	 * @brief Modular integer whose modulus is fixed at runtime (Barrett-reduced).
	 *
	 * Unlike mod_int<T, Mod> (compile-time modulus), the modulus is chosen at
	 * runtime via set_mod(). It is stored once as a static per-type Barrett
	 * reducer shared by every instance of a given Id, so the object stays one
	 * word and arithmetic reads like plain ints. The trade is a global, set-once
	 * modulus: choose it with set_mod() when no values are live. Use distinct Id values
	 * for two moduli at once (e.g. double hashing).
	 *
	 *     using Z = atl::dynamic_mod_int<>; // 32-bit modulus
	 *     Z::set_mod(m); // m read at runtime
	 *     Z a = 5, b = 7;
	 *     Z c = a * b + 1; // multiply uses Barrett reduction
	 *
	 * @complexity O(1) storage and arithmetic except O(log mod()) inverse/division
	 * and O(log exp) power. Modulus changes invalidate all values and require external
	 * synchronization with reads and writes of the shared reducer.
	 * @note Stream extraction leaves the destination unchanged on malformed or overflowing input.
	 * @tparam T Modulus/value type: uint32 (default) or uint64.
	 * @tparam Id Distinguishes independent moduli (default 0).
	 */
	template<typename T = uint32, int32 Id = 0>
		requires traits::is_one_of_v<T, uint32, uint64>
	class dynamic_mod_int {
		using reducer = std::conditional_t<sizeof(T) == 8, barrett64, barrett>;

		static inline reducer reducer_;
		/**
		 * @brief `ceil(2^64 / mod)` for the 32-bit type (0 while the modulus is
		 *        1): reduces any 64-bit unsigned input with one multiply.
		 */
		static inline uint64 reciprocal_ = 0;
		T value_ = 0;

		/**
		 * @brief `z mod m` by Barrett: the quotient estimate
		 *        `floor(z·reciprocal_ / 2^64)` is the true quotient or one more,
		 *        so one conditional add corrects it. Constructing from 10^7 random
		 *        64-bit values: uint64 75.1 -> 19.7 ms, int64 118 -> 58.7 ms against
		 *        `% m` (Xeon Platinum 8280, GCC 15.2, RelWithDebInfo,
		 *        -O2 -g -DNDEBUG -std=c++23, 2026-10-15,
		 *        benchmarks/nt/dynamic_mod_int.cpp).
		 */
		static T reduce_wide(uint64 z)
			requires(sizeof(T) == 4)
		{
			if (reciprocal_ == 0) return 0; // modulus 1
			uint64 const x = detail::multiply_high(z, reciprocal_);
			uint64 const y = x * mod();
			return T(z - y + (z < y ? mod() : 0));
		}

	public:
		/**
		 * @brief Set the modulus shared by every value of this type (and @p Id).
		 *        Changing the modulus invalidates all live values; their stored
		 *        representatives are not re-reduced.
		 * @pre `m > 0` (asserted).
		 * @note Not constexpr (updates the shared runtime reducer).
		 */
		static void set_mod(T m) {
			atl_assert(m > 0);
			reducer_ = reducer(m);
			if constexpr (sizeof(T) == 4) reciprocal_ = m <= 1 ? 0 : (~uint64(0)) / m + 1;
		}

		/**
		 * @brief The current modulus, initially 1.
		 * @note Not constexpr (reads shared runtime state).
		 */
		static T mod() { return reducer_.mod(); }

		/** @brief Construct zero without accessing the shared modulus; defaulting permits constant evaluation. */
		dynamic_mod_int() = default;

		/**
		 * @brief Reduce the integral argument v into [0, mod()).
		 * @tparam I Integral source type.
		 * @param v Value to reduce.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		template<integral I>
		dynamic_mod_int(I v) {
			if constexpr (sizeof(T) == 4 && sizeof(I) == 8) {
				if constexpr (std::is_unsigned_v<I>) {
					value_ = reduce_wide(uint64(v));
				} else { // the magnitude through the reciprocal, then the sign
					T const r = reduce_wide(v < 0 ? uint64(0) - uint64(v) : uint64(v));
					value_ = v < 0 && r != 0 ? T(mod() - r) : r;
				}
			} else {
				value_ = reduce_mod<I, T>(v, mod());
			}
		}

		/**
		 * @brief Wrap a plain residue, skipping the constructor's range reduction.
		 * @tparam I Integral source type.
		 * @param x Plain residue to wrap.
		 * @return A value representing x modulo the current modulus.
		 * @pre `0 <= x < mod()` (asserted).
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		template<integral I>
		static dynamic_mod_int unchecked(I x) {
			atl_assert(x >= I(0) && traits::make_unsigned_t<I>(x) < mod());
			dynamic_mod_int a;
			a.value_ = T(x);
			return a;
		}

		/**
		 * @brief Add rhs modulo the modulus and return this object.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		dynamic_mod_int &operator+=(dynamic_mod_int const &rhs) {
			value_ = add_mod_unchecked<T>(value_, rhs.value_, mod());
			return *this;
		}

		/**
		 * @brief Subtract rhs modulo the modulus and return this object.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		dynamic_mod_int &operator-=(dynamic_mod_int const &rhs) {
			value_ = sub_mod_unchecked<T>(value_, rhs.value_, mod());
			return *this;
		}

		/**
		 * @brief Multiply by rhs modulo the modulus and return this object.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		dynamic_mod_int &operator*=(dynamic_mod_int const &rhs) {
			value_ = reducer_.mul(value_, rhs.value_);
			return *this;
		}

		/**
		 * @brief Divide by rhs and return this object.
		 * @pre rhs is invertible (asserted).
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		dynamic_mod_int &operator/=(dynamic_mod_int const &rhs) {
			return *this *= rhs.inverse();
		}

		/**
		 * @brief Multiplicative inverse, via the extended GCD.
		 * @pre `gcd(value(), mod()) == 1` (asserted).
		 * @complexity O(log mod()) time and O(1) storage.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		dynamic_mod_int inverse() const {
			return dynamic_mod_int::unchecked(inverse_mod(value_, mod()));
		}

		/**
		 * @brief Return this residue raised to exp.
		 * @complexity O(log exp) multiplications.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		dynamic_mod_int power(uint64 exp) const { return atl::power(*this, exp); }

		/**
		 * @brief Return the modular sum of lhs and rhs.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		friend dynamic_mod_int operator+(dynamic_mod_int lhs, dynamic_mod_int const &rhs) {
			return lhs += rhs;
		}

		/**
		 * @brief Return the modular difference of lhs and rhs.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		friend dynamic_mod_int operator-(dynamic_mod_int lhs, dynamic_mod_int const &rhs) {
			return lhs -= rhs;
		}

		/**
		 * @brief Return the modular product of lhs and rhs.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		friend dynamic_mod_int operator*(dynamic_mod_int lhs, dynamic_mod_int const &rhs) {
			return lhs *= rhs;
		}

		/**
		 * @brief Return lhs divided by rhs.
		 * @pre rhs is invertible (asserted).
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		friend dynamic_mod_int operator/(dynamic_mod_int lhs, dynamic_mod_int const &rhs) {
			return lhs /= rhs;
		}

		/**
		 * @brief Write the representative using the stream formatting.
		 * @pre Residue type is at most 64 bits.
		 * @return os.
		 * @note Not constexpr (uses stream I/O).
		 */
		friend std::ostream &operator<<(std::ostream &os, dynamic_mod_int const &mi) {
			return os << mi.value_;
		}

		/**
		 * @brief Read an optional sign and a decimal, octal or hexadecimal magnitude, then reduce it.
		 * @pre The magnitude fits uint64 for successful extraction.
		 * @return is, with failbit on malformed or overflowing input; mi is unchanged on failure.
		 * @note Honors skipws; a sign must immediately precede the magnitude. Input grouping is unsupported.
		 *        Valid digits ending at EOF succeed with eofbit only. An unset basefield means decimal.
		 * @note Not constexpr (uses stream I/O).
		 */
		friend std::istream &operator>>(std::istream &is, dynamic_mod_int &mi) {
			auto const parsed = detail::read_stream_magnitude<uint64>(is);
			if ((parsed.state & (std::ios_base::failbit | std::ios_base::badbit)) == 0) {
				dynamic_mod_int const reduced(parsed.value);
				mi = parsed.is_negative ? -reduced : reduced;
			}
			is.setstate(parsed.state);
			return is;
		}

		/**
		 * @brief Return this residue unchanged.
		 */
		constexpr dynamic_mod_int operator+() const { return *this; }

		/**
		 * @brief Return the additive inverse.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		dynamic_mod_int operator-() const {
			return dynamic_mod_int::unchecked(neg_mod(value_, mod()));
		}

		/**
		 * @brief Return whether the stored representatives agree.
		 */
		constexpr bool operator==(dynamic_mod_int const &rhs) const { return value_ == rhs.value_; }

		/**
		 * @brief Increment modulo the modulus and return this object.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		dynamic_mod_int &operator++() {
			value_ = incr_mod<T>(value_, mod());
			return *this;
		}

		/**
		 * @brief Increment modulo the modulus and return the saved value.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		dynamic_mod_int operator++(int) {
			dynamic_mod_int saved = *this;
			++*this;
			return saved;
		}

		/**
		 * @brief Decrement modulo the modulus and return this object.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		dynamic_mod_int &operator--() {
			value_ = decr_mod<T>(value_, mod());
			return *this;
		}

		/**
		 * @brief Decrement modulo the modulus and return the saved value.
		 * @note Not constexpr (reads the shared runtime modulus).
		 */
		dynamic_mod_int operator--(int) {
			dynamic_mod_int saved = *this;
			--*this;
			return saved;
		}

		/**
		 * @brief Return the plain representative in [0, mod()).
		 */
		constexpr T value() const { return value_; }
	};

	/**
	 * @brief 32-bit modular integer alias.
	 * @tparam Id Independent runtime modulus tag.
	 */
	template<int32 Id = 0>
	using dynamic_mod_int32 = dynamic_mod_int<uint32, Id>;

	/**
	 * @brief 64-bit modular integer alias.
	 * @tparam Id Independent runtime modulus tag.
	 */
	template<int32 Id = 0>
	using dynamic_mod_int64 = dynamic_mod_int<uint64, Id>;
}
