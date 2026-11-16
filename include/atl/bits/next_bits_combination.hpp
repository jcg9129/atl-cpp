#pragma once
#include <atl/bits/bitlow.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <bit>

namespace atl {
	/**
	 * @brief The next larger value of type T with the same number of set
	 *        bits as @p x, or 0 when there is none.
	 *
	 * Iterating from `(T(1) << k) - 1` enumerates the k-subsets when
	 * `0 < k < sizeof(T) * 8`; start with all bits set for the full subset.
	 * @note The count-on-input variant beat the serial count by 24% to 31% on
	 *       loop-carried x86-64 enumeration (benchmarks/bits/next_bits_combination.cpp,
	 *       2026-10-15). CPU and flags were not recorded; this historical conclusion
	 *       needs a fully attributed rerun before publication as measured performance.
	 * @tparam T An unsigned integer recognized by `atl::unsigned_integral`.
	 * @param x The current combination; every value is valid.
	 * @return 0 when x == 0 or when x is the largest combination of its
	 *         popcount that fits the width of T.
	 * @complexity O(1) time and space.
	 */
	template<unsigned_integral T>
	constexpr T next_bits_combination(T x) noexcept {
		T const r = lowest_one(x);
		if (x == 0 || x > T(~T(0)) - r) return 0; // x + r would carry out of T
		int shift;
		if constexpr (sizeof(T) <= sizeof(uint64))
			shift = std::countr_zero(uint64(x));
		else {
			// Counting native limbs also supports the software 128-bit type.
			uint64 const low = uint64(x);
			shift = low != 0 ? std::countr_zero(low) : 64 + std::countr_zero(uint64(x >> 64));
		}
		// Counting x overlaps the addition rather than extending its dependency chain.
		// Splitting the shift avoids a count equal to the word width.
		x += r;
		return T(x | (T(lowest_one(x) - r) >> shift >> 1));
	}
}
