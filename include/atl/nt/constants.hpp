#pragma once

#include <atl/types/alias.hpp>

namespace atl {
	/** @brief The largest prime below 2^64: `2^64 - 59`. */
	inline constexpr uint64 largest_prime_64 = 18'446'744'073'709'551'557ull;
	/** @brief The largest prime below 2^32: `2^32 - 5`. */
	inline constexpr uint32 largest_prime_32 = 4'294'967'291u;
}
