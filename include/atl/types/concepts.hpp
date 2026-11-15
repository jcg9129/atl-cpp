#pragma once

#include <atl/types/alias.hpp>
#include <atl/types/traits/integral.hpp>
#include <concepts>
#include <type_traits>

namespace atl {
	/**
	 * @brief Satisfied by any built-in integer type, including 128-bit integers.
	 *
	 * Unlike `std::integral`, this recognizes `atl::int128` / `atl::uint128`, and it
	 * deliberately excludes `bool` and the character types (`char`, `wchar_t`,
	 * `char8_t`, `char16_t`, `char32_t`); `signed char` and `unsigned char` count
	 * as integers.
	 */
	template<typename T>
	concept integral = traits::is_integral_v<T>;

	/**
	 * @brief Satisfied by any signed integer type, including `atl::int128`; like
	 *        @ref atl::integral, excludes `bool` and the character types.
	 */
	template<typename T>
	concept signed_integral = traits::is_signed_integral_v<T>;

	/**
	 * @brief Satisfied by any unsigned integer type, including `atl::uint128`; like
	 *        @ref atl::integral, excludes `bool` and the character types.
	 */
	template<typename T>
	concept unsigned_integral = traits::is_unsigned_integral_v<T>;

	/**
	 * @brief A modular integer: a value type whose arithmetic runs modulo
	 *        `Z::mod()`, an unsigned integral that `value()` reads residues
	 *        back in. It is constructible from `int64` and `uint64` (reducing
	 *        the value), a field-style ring (`+ - * /`, compound forms,
	 *        negation, `==`), and has `inverse()` and `power(uint64)`.
	 *        Division and `inverse()` are meaningful only for an invertible
	 *        operand (what they do otherwise is the model's contract, not this
	 *        concept's). `mod_int` and
	 *        `dynamic_mod_int` both model it; generic code such as the fast
	 *        reader/writer keys on this concept.
	 */
	template<typename Z>
	concept modular_integer =
		std::regular<Z> && std::constructible_from<Z, int64> && std::constructible_from<Z, uint64>
		&& requires(Z a, Z const b, uint64 e) {
			   { Z::mod() } -> unsigned_integral;
			   { b.value() } -> std::same_as<decltype(Z::mod())>;
			   { b + b } -> std::same_as<Z>;
			   { b - b } -> std::same_as<Z>;
			   { b * b } -> std::same_as<Z>;
			   { b / b } -> std::same_as<Z>;
			   { -b } -> std::same_as<Z>;
			   { a += b } -> std::same_as<Z &>;
			   { a -= b } -> std::same_as<Z &>;
			   { a *= b } -> std::same_as<Z &>;
			   { a /= b } -> std::same_as<Z &>;
			   { b.inverse() } -> std::same_as<Z>;
			   { b.power(e) } -> std::same_as<Z>;
		   };

	/**
	 * @brief A @ref atl::modular_integer whose modulus is a constant expression, as
	 *        the transform engines need it (they instantiate `montgomery<Z::mod()>`).
	 *        `mod_int` models it; `dynamic_mod_int` does not, as its modulus is
	 *        set at run time.
	 */
	template<typename Z>
	concept static_mod_int =
		modular_integer<Z> && requires { typename std::integral_constant<decltype(Z::mod()), Z::mod()>; };
}
