#pragma once

#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/traits/detail/extended_integer.hpp>
#include <atl/types/traits/is_one_of.hpp>
#include <type_traits>

namespace atl::traits {
	/**
	 * @brief Whether @p T is a (possibly cv-qualified) signed integer type:
	 *        `signed char` .. `long long`, `int128` and registered class-type integers.
	 *        Plain `char`, `bool` and references are not.
	 */
	template<typename T>
	struct is_signed_integral
		: public std::bool_constant<is_one_of_v<std::remove_cv_t<T>, signed char, signed short, signed int, signed long,
												signed long long, int128>
									|| detail::extended_integer<std::remove_cv_t<T>>::is_signed> {
	};

	/** @brief Shorthand for `is_signed_integral<T>::value`. */
	template<typename T>
	constexpr bool is_signed_integral_v = is_signed_integral<T>::value;

	/**
	 * @brief Whether @p T is a (possibly cv-qualified) unsigned integer type:
	 *        `unsigned char` .. `unsigned long long`, and `uint128` and registered class-type integers. `bool`, the character types and references are not.
	 */
	template<typename T>
	struct is_unsigned_integral
		: public std::bool_constant<is_one_of_v<std::remove_cv_t<T>, unsigned char, unsigned short, unsigned int,
												unsigned long, unsigned long long, uint128>
									|| detail::extended_integer<std::remove_cv_t<T>>::is_unsigned> {
	};

	/** @brief Shorthand for `is_unsigned_integral<T>::value`. */
	template<typename T>
	constexpr bool is_unsigned_integral_v = is_unsigned_integral<T>::value;

	/**
	 * @brief Whether @p T is a (possibly cv-qualified) integer type: signed or
	 *        unsigned integral as above.
	 *
	 * Unlike std::is_integral this deliberately excludes bool and the character
	 * types (char, wchar_t, char8_t/char16_t/char32_t), which do not model
	 * numbers here, and it includes the 128-bit types
	 * in every language mode.
	 */
	template<typename T>
	struct is_integral : public std::bool_constant<is_signed_integral_v<T> || is_unsigned_integral_v<T>> {};

	/** @brief Shorthand for `is_integral<T>::value`. */
	template<typename T>
	constexpr bool is_integral_v = is_integral<T>::value;

	namespace detail {
		/** @brief `To` with the cv-qualifiers of `From` (any already on `To` are kept). */
		template<typename From, typename To>
		struct copy_cv {
			using type = To;
		};

		template<typename From, typename To>
		struct copy_cv<From const, To> {
			using type = To const;
		};

		template<typename From, typename To>
		struct copy_cv<From volatile, To> {
			using type = To volatile;
		};

		template<typename From, typename To>
		struct copy_cv<From const volatile, To> {
			using type = To const volatile;
		};

		template<typename From, typename To>
		using copy_cv_t = typename copy_cv<From, To>::type;
	}

	/**
	 * @brief The signed integer type of the same width as @p T, cv-qualifiers
	 *        kept: std::make_signed extended to `int128`/`uint128`.
	 *
	 * Like std::make_signed it accepts the character types (`char` maps to
	 * `signed char`) and has no `type` for `bool`, enums and non-integral types, so detection
	 * in a requires-expression is safe. Enums require an explicit integer conversion.
	 */
	template<typename T>
	struct make_signed {};

	/**
	 * @brief Map a standard integral type other than `bool` to its signed counterpart.
	 * @tparam T Source type; cv-qualifiers are preserved in the inherited `type`.
	 */
	template<typename T>
		requires(std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool> && sizeof(T) <= sizeof(unsigned long long))
	struct make_signed<T> : std::make_signed<T> {};

#if ATL_HAS_INT128
	/**
	 * @brief Map the native `uint128` and `int128` to `int128`, preserving cv-qualifiers.
	 * @tparam T A possibly cv-qualified native 128-bit integer.
	 */
	template<typename T>
		requires(std::is_same_v<std::remove_cv_t<T>, uint128> || std::is_same_v<std::remove_cv_t<T>, int128>)
	struct make_signed<T> {
		/** @brief The signed counterpart with the source cv-qualifiers. */
		using type = detail::copy_cv_t<T, int128>;
	};
#endif

	/**
	 * @brief Map a registered class-type integer to its signed counterpart, preserving cv-qualifiers.
	 * @tparam T A possibly cv-qualified class type registered through `detail::extended_integer`.
	 */
	template<typename T>
		requires(detail::extended_integer<std::remove_cv_t<T>>::is_signed
				 || detail::extended_integer<std::remove_cv_t<T>>::is_unsigned)
	struct make_signed<T> {
		/** @brief The signed counterpart with the source cv-qualifiers. */
		using type = detail::copy_cv_t<T, typename detail::extended_integer<std::remove_cv_t<T>>::signed_type>;
	};

	/** @brief Shorthand for `make_signed<T>::type`. */
	template<typename T>
	using make_signed_t = typename make_signed<T>::type;

	/**
	 * @brief The unsigned integer type of the same width as @p T, cv-qualifiers
	 *        kept: std::make_unsigned extended to `int128`/`uint128`, with
	 *        enums rejected (as for @ref make_signed).
	 */
	template<typename T>
	struct make_unsigned {};

	/**
	 * @brief Map a standard integral type other than `bool` to its unsigned counterpart.
	 * @tparam T Source type; cv-qualifiers are preserved in the inherited `type`.
	 */
	template<typename T>
		requires(std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool> && sizeof(T) <= sizeof(unsigned long long))
	struct make_unsigned<T> : std::make_unsigned<T> {};

#if ATL_HAS_INT128
	/**
	 * @brief Map the native `int128` and `uint128` to `uint128`, preserving cv-qualifiers.
	 * @tparam T A possibly cv-qualified native 128-bit integer.
	 */
	template<typename T>
		requires(std::is_same_v<std::remove_cv_t<T>, int128> || std::is_same_v<std::remove_cv_t<T>, uint128>)
	struct make_unsigned<T> {
		/** @brief The unsigned counterpart with the source cv-qualifiers. */
		using type = detail::copy_cv_t<T, uint128>;
	};
#endif

	/**
	 * @brief Map a registered class-type integer to its unsigned counterpart, preserving cv-qualifiers.
	 * @tparam T A possibly cv-qualified class type registered through `detail::extended_integer`.
	 */
	template<typename T>
		requires(detail::extended_integer<std::remove_cv_t<T>>::is_signed
				 || detail::extended_integer<std::remove_cv_t<T>>::is_unsigned)
	struct make_unsigned<T> {
		/** @brief The unsigned counterpart with the source cv-qualifiers. */
		using type = detail::copy_cv_t<T, typename detail::extended_integer<std::remove_cv_t<T>>::unsigned_type>;
	};

	/** @brief Shorthand for `make_unsigned<T>::type`. */
	template<typename T>
	using make_unsigned_t = typename make_unsigned<T>::type;
}
