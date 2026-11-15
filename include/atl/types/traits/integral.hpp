#pragma once

#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/traits/is_one_of.hpp>
#include <type_traits>

namespace atl::traits {
	/**
	 * @brief Whether @p T is a (possibly cv-qualified) signed integer type:
	 *        `signed char` .. `long long`, and `int128` and the concrete software type.
	 *        Plain `char`, `bool` and references are not.
	 */
	template<typename T>
	struct is_signed_integral
		: public is_one_of<std::remove_cv_t<T>, signed char, signed short, signed int, signed long, signed long long,
						   int128, atl::detail::software_int128> {
	};

	/** @brief Shorthand for `is_signed_integral<T>::value`. */
	template<typename T>
	constexpr bool is_signed_integral_v = is_signed_integral<T>::value;

	/**
	 * @brief Whether @p T is a (possibly cv-qualified) unsigned integer type:
	 *        `unsigned char` .. `unsigned long long`, and `uint128` (native or software). `bool`, the character types and references are not.
	 */
	template<typename T>
	struct is_unsigned_integral
		: public is_one_of<std::remove_cv_t<T>, unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long,
						   uint128, atl::detail::software_uint128> {
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

	/**
	 * @brief Map `uint128` to `int128`, preserving cv-qualifiers.
	 * @tparam T A possibly cv-qualified `uint128`.
	 */
	template<typename T>
		requires(std::is_same_v<std::remove_cv_t<T>, uint128> || std::is_same_v<std::remove_cv_t<T>, atl::detail::software_uint128>)
	struct make_signed<T> {
		/** @brief The signed counterpart with the source cv-qualifiers. */
		using type = detail::copy_cv_t<T, std::conditional_t<std::is_same_v<std::remove_cv_t<T>, atl::detail::software_uint128>, atl::detail::software_int128, int128>>;
	};

	/**
	 * @brief Map `int128` to `int128`, preserving cv-qualifiers.
	 * @tparam T A possibly cv-qualified `int128`.
	 */
	template<typename T>
		requires(std::is_same_v<std::remove_cv_t<T>, int128> || std::is_same_v<std::remove_cv_t<T>, atl::detail::software_int128>)
	struct make_signed<T> {
		/** @brief The signed counterpart with the source cv-qualifiers. */
		using type = T;
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

	/**
	 * @brief Map `int128` to `uint128`, preserving cv-qualifiers.
	 * @tparam T A possibly cv-qualified `int128`.
	 */
	template<typename T>
		requires(std::is_same_v<std::remove_cv_t<T>, int128> || std::is_same_v<std::remove_cv_t<T>, atl::detail::software_int128>)
	struct make_unsigned<T> {
		/** @brief The unsigned counterpart with the source cv-qualifiers. */
		using type = detail::copy_cv_t<T, std::conditional_t<std::is_same_v<std::remove_cv_t<T>, atl::detail::software_int128>, atl::detail::software_uint128, uint128>>;
	};

	/**
	 * @brief Map `uint128` to `uint128`, preserving cv-qualifiers.
	 * @tparam T A possibly cv-qualified `uint128`.
	 */
	template<typename T>
		requires(std::is_same_v<std::remove_cv_t<T>, uint128> || std::is_same_v<std::remove_cv_t<T>, atl::detail::software_uint128>)
	struct make_unsigned<T> {
		/** @brief The unsigned counterpart with the source cv-qualifiers. */
		using type = T;
	};

	/** @brief Shorthand for `make_unsigned<T>::type`. */
	template<typename T>
	using make_unsigned_t = typename make_unsigned<T>::type;
}
