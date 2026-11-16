/**
 * @file
 * Race between three formulations of `next_bits_combination` (Gosper's hack: the
 * next larger 64-bit word with the same number of set bits).
 *
 * All three add the lowest set bit `r` to `x`, which carries the lowest run of
 * ones one place up, and then OR the rest of that run back in, shifted down to
 * bit 0. They differ only in how they find that shift:
 * - `library_next` (the shipped form) counts the trailing zeros of `x` before the
 *   addition, so the count executes in parallel with it;
 * - `next_ctz_after_add` isolates the moved run `z = lowest_one(x + r) - r` and
 *   counts its trailing zeros, so the count waits for the addition;
 * - `next_shift_loop_after_add` normalises the same `z` with a one-bit shift loop
 *   instead of a count instruction.
 *
 * Each benchmark enumerates every k-subset of n bits, each step feeding the next
 * (`sum_combinations`), so it measures the latency of one step.
 *
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`,
 * 2026-10-15): `library_next` is fastest on both shapes. `next_ctz_after_add` is
 * 30-44% slower. `next_shift_loop_after_add` ties on C(28,14) but is 52-70% slower
 * on C(40,6). Numbers: docs/bit-utilities.md.
 */
#include <atl/bits/bitlow.hpp>
#include <atl/bits/next_bits_combination.hpp>
#include <atl/types/alias.hpp>
#include <bit>
#include <catch2/catch.hpp>
#include <string>
#include <utility>

using namespace atl;

namespace {
	/** @brief The shipped `next_bits_combination`: trailing zeros counted on `x`, before the addition. */
	uint64 library_next(uint64 x) noexcept { return next_bits_combination(x); }

	/** @brief Contender: trailing zeros counted on the moved run `z`, after the addition. */
	constexpr uint64 next_ctz_after_add(uint64 x) noexcept {
		uint64 const r = lowest_one(x);
		if (x == 0 || x > ~uint64(0) - r) return 0;
		x += r;
		uint64 z = lowest_one(x) - r;
		z >>= std::countr_zero(z);
		return x | (z >> 1);
	}

	/** @brief Contender: the moved run `z` normalised by a one-bit shift loop instead of a count instruction. */
	constexpr uint64 next_shift_loop_after_add(uint64 x) noexcept {
		uint64 const r = lowest_one(x);
		if (x == 0 || x > ~uint64(0) - r) return 0;
		x += r;
		uint64 z = lowest_one(x) - r;
		while ((z & 1) == 0) z >>= 1;
		return x | (z >> 1);
	}

	/**
	 * @brief Sum of all k-subsets of n bits, enumerated with @p Next; each step depends
	 *        on the previous one, so the time is the per-step latency. The sum (modulo
	 *        2^64) keeps the loop from being optimised away and checks the contenders agree.
	 * @pre `0 < k <= n < 64`; Next returns the correct combination successor.
	 */
	template<uint64 (*Next)(uint64) noexcept>
		requires(Next != nullptr)
	uint64 sum_combinations(int n, int k) {
		uint64 x = (uint64(1) << k) - 1, sum = 0;
		uint64 const limit = uint64(1) << n;
		while (x < limit) {
			sum += x;
			x = Next(x);
		}
		return sum;
	}
}

TEST_CASE("next_bits_combination contenders", "[bits]") {
	for (uint64 x = 1; x < (uint64(1) << 20); ++x) {
		REQUIRE(next_ctz_after_add(x) == next_bits_combination(x));
		REQUIRE(next_shift_loop_after_add(x) == next_bits_combination(x));
	}
	REQUIRE(next_bits_combination(~uint64(0) << 3) == 0);
	for (auto [n, k] : {std::pair{28, 14}, std::pair{40, 6}}) {
		std::string const shape = "C(" + std::to_string(n) + "," + std::to_string(k) + ")";
		REQUIRE(sum_combinations<library_next>(n, k) == sum_combinations<next_ctz_after_add>(n, k));
		REQUIRE(sum_combinations<library_next>(n, k) == sum_combinations<next_shift_loop_after_add>(n, k));
		BENCHMARK(shape + " | library (count before add)") { return sum_combinations<library_next>(n, k); };
		BENCHMARK(shape + " | count after add") { return sum_combinations<next_ctz_after_add>(n, k); };
		BENCHMARK(shape + " | shift loop after add") { return sum_combinations<next_shift_loop_after_add>(n, k); };
	}
}
