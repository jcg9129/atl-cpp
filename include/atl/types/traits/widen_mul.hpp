#pragma once
#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <cstddef>
#include <type_traits>

namespace atl::traits {
	namespace detail {
		/**
		 * @brief The built-in integer of exactly @p Bytes bytes and the given
		 *        signedness, as `type`; no member when the target has none
		 *        (16 bytes without `__int128`).
		 */
		template<std::size_t Bytes, bool IsSigned>
		struct integer_of_size {};

		template<>
		struct integer_of_size<2, true> {
			using type = int16;
		};

		template<>
		struct integer_of_size<4, true> {
			using type = int32;
		};

		template<>
		struct integer_of_size<8, true> {
			using type = int64;
		};

		template<>
		struct integer_of_size<2, false> {
			using type = uint16;
		};

		template<>
		struct integer_of_size<4, false> {
			using type = uint32;
		};

		template<>
		struct integer_of_size<8, false> {
			using type = uint64;
		};
#if ATL_HAS_INT128
		template<>
		struct integer_of_size<16, true> {
			using type = int128;
		};

		template<>
		struct integer_of_size<16, false> {
			using type = uint128;
		};
#endif

		/** @brief Whether a built-in integer twice as wide as @p T exists. */
		template<typename T>
		concept has_double_width = std::is_integral_v<T> && !std::is_same_v<T, bool>
								&& requires { typename integer_of_size<2 * sizeof(T), std::is_signed_v<T>>::type; };
	}

	/**
	 * @brief The type to multiply two values of type @p T in: the built-in
	 *        integer of twice the width and the same signedness, which holds
	 *        every product exactly.
	 *
	 * Chosen by SIZE and signedness, not by spelling, so every 8-byte integer
	 * widens to the 128-bit type — `long` and `long long` alike, on LP64 and
	 * LLP64. When no wider built-in exists (`int128` itself, a 64-bit type on
	 * a target without `__int128`, or any non-integer type) `type` is @p T
	 * itself, so the product is NOT guaranteed to fit: callers test
	 * `atl::traits::has_widen_mul_v<T>` and take a fallback path (see mul_mod_unchecked /
	 * static_modulus::mul). cv-qualifiers are ignored.
	 */
	template<typename T>
	struct widen_mul_type {
		/** @brief Unchanged source type when no wider built-in integer is available. */
		using type = T;
	};

	/**
	 * @brief Select the integer of twice the source width and matching signedness.
	 * @tparam T An integer with an available double-width built-in type.
	 */
	template<detail::has_double_width T>
	struct widen_mul_type<T> {
		/** @brief Integer type that holds every product of two source values exactly. */
		using type = typename detail::integer_of_size<2 * sizeof(T), std::is_signed_v<T>>::type;
	};

	/**
	 * @brief Ignore `const` qualification when selecting a multiplication type.
	 * @tparam T Unqualified source type.
	 */
	template<typename T>
	struct widen_mul_type<T const> : widen_mul_type<T> {};

	/**
	 * @brief Ignore `volatile` qualification when selecting a multiplication type.
	 * @tparam T Unqualified source type.
	 */
	template<typename T>
	struct widen_mul_type<T volatile> : widen_mul_type<T> {};

	/**
	 * @brief Ignore `const volatile` qualification when selecting a multiplication type.
	 * @tparam T Unqualified source type.
	 */
	template<typename T>
	struct widen_mul_type<T const volatile> : widen_mul_type<T> {};

	/** @brief Shorthand for `widen_mul_type<T>::type`. */
	template<typename T>
	using widen_mul_type_t = typename widen_mul_type<T>::type;

	/**
	 * @brief Whether @p T has a strictly wider multiplication type, i.e. whether
	 *        `widen_mul_type_t<T>` holds every product of two `T`s exactly.
	 */
	template<typename T>
	inline constexpr bool has_widen_mul_v = sizeof(widen_mul_type_t<T>) > sizeof(T);
}
