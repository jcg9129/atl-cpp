#pragma once

/**
 * @file
 * @brief Fixed-width names for the built-in arithmetic types (`int8` .. `uint64`,
 *        `int128`/`uint128` when `ATL_HAS_INT128`, and `float32`/`float64`).
 *        There is no `float128`: `long double` is 80-bit x87, binary128 or plain
 *        `double` depending on the target, so no alias could promise the width.
 */

#include <atl/config.hpp>
#include <cstdint>

namespace atl {
	/** @brief Exactly 8-bit signed integer type. */
	using int8 = std::int8_t;
	/** @brief Exactly 16-bit signed integer type. */
	using int16 = std::int16_t;
	/** @brief Exactly 32-bit signed integer type. */
	using int32 = std::int32_t;
	/** @brief Exactly 64-bit signed integer type. */
	using int64 = std::int64_t;

	/** @brief Exactly 8-bit unsigned integer type. */
	using uint8 = std::uint8_t;
	/** @brief Exactly 16-bit unsigned integer type. */
	using uint16 = std::uint16_t;
	/** @brief Exactly 32-bit unsigned integer type. */
	using uint32 = std::uint32_t;
	/** @brief Exactly 64-bit unsigned integer type. */
	using uint64 = std::uint64_t;

#if ATL_HAS_INT128
	/** @brief Exactly 128-bit signed integer type. Available only when `ATL_HAS_INT128` is 1. */
	using int128 = __int128_t;
	/** @brief Exactly 128-bit unsigned integer type. Available only when `ATL_HAS_INT128` is 1. */
	using uint128 = __uint128_t;
#endif

	/** @brief Native `float`; precision and representation follow the target ABI. */
	using float32 = float;
	/** @brief Native `double`; precision and representation follow the target ABI. */
	using float64 = double;
}
