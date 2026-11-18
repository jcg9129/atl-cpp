#pragma once

/**
 * @file
 * @brief The race losers for the reduced floor-sum loop of atl::floor_sum.
 *        Experimental, not part of the public atl API.
 *
 * Race (benchmarks/nt/floor_sum.cpp, measured 2026-10-15, x86-64 Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, 1e5
 * reduced queries per
 * shape). The library loop, `atl::detail::floor_sum_narrowing`, starts in 128-bit
 * divisions and drops to 64-bit ones once n and m shrink below 2^32. Plain -O2 /
 * -march=native, ms:
 *
 * | shape | library (narrowing) | wide | entry dispatch |
 * |------------------|---------------------|-------------|----------------|
 * | n, m < 1e9 | 28.4 / 28.5 | 45.4 / 45.8 | 26.6 / 26.8 |
 * | n, m < 1e18 | 64.2 / 64.2 | 92.3 / 93.1 | 107 / 108 |
 * | n < 1e18, m < 1e9| 36.6 / 36.8 | 46.2 / 46.6 | 53.7 / 53.9 |
 *
 * The contenders keep
 * the public function's argument shape, reduced: `int64` in and the exact
 * `int128` out, so they slot in behind atl::floor_sum's signed normalization.
 */

#include <atl/config.hpp>
#include <atl/nt/floor_sum.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>

namespace atl::experimental {
	/**
	 * @brief Every round in 128-bit divisions, whatever the magnitudes.
	 *
	 * Lost: a 128-bit division costs a library call, so the narrowing loop is 1.60x
	 * faster at n, m < 1e9 (45.4 vs 28.4 ms) and 1.44x at n, m < 1e18 (92.3 vs 64.2 ms,
	 * plain -O2).
	 *
	 * @param n Number of terms.
	 * @param m Positive denominator.
	 * @param a Reduced step.
	 * @param b Reduced offset.
	 * @return Exact floor sum, which fits int128 throughout this reduced domain.
	 * @pre `0 <= n`, `1 <= m`, `0 <= a, b < m` (asserted).
	 * @complexity O(log m) 128-bit divisions
	 */
	constexpr int128 floor_sum_by_wide_divisions(int64 n, int64 m, int64 a, int64 b) {
		atl_assert(n >= 0 && m >= 1 && 0 <= a && a < m && 0 <= b && b < m);
		return int128(atl::detail::floor_sum_reduced<uint128>(uint128(n), uint128(m), uint128(a), uint128(b)));
	}

	/**
	 * @brief Chooses 64-bit divisions once, at entry, when `n, m < 2^32`, and 128-bit
	 *        ones for the whole run otherwise.
	 *
	 * Lost: it is 6.34% ahead of the narrowing loop on small inputs (26.6 vs 28.4 ms
	 * at n, m < 1e9) but never benefits from the values shrinking, so it pays
	 * 128-bit divisions to the end on wide inputs (107 vs 64.2 ms at n, m < 1e18,
	 * plain -O2): the larger loss outweighs the small win.
	 *
	 * @param n Number of terms.
	 * @param m Positive denominator.
	 * @param a Reduced step.
	 * @param b Reduced offset.
	 * @return Exact floor sum, which fits int128 throughout this reduced domain.
	 * @pre `0 <= n`, `1 <= m`, `0 <= a, b < m` (asserted).
	 * @complexity O(log m) divisions
	 */
	constexpr int128 floor_sum_by_entry_dispatch(int64 n, int64 m, int64 a, int64 b) {
		atl_assert(n >= 0 && m >= 1 && 0 <= a && a < m && 0 <= b && b < m);
		if (n < (int64(1) << 32) && m < (int64(1) << 32))
			return int128(atl::detail::floor_sum_reduced<uint64>(uint64(n), uint64(m), uint64(a), uint64(b)));
		return int128(atl::detail::floor_sum_reduced<uint128>(uint128(n), uint128(m), uint128(a), uint128(b)));
	}
}
