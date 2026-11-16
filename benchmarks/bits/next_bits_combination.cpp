/**
 * @file
 * Race between three implementations of `next_bits_combination` (Gosper's hack:
 * the next larger 64-bit word with the same number of set bits).
 *
 * All three add the lowest set bit `r` to `x`, which carries the lowest run of
 * ones one place up, and then OR the rest of that run back in, shifted down to
 * bit 0. They differ only in how they find that shift:
 * - `atl::next_bits_combination` (the library) counts the trailing zeros of `x`
 *   before the addition, so the count executes in parallel with it;
 * - `next_bits_combination_by_run_ctz` counts the trailing zeros of the carried
 *   run `z = lowest_one(x + r) - r`, so the count waits for the addition;
 * - `next_bits_combination_by_shift_loop` shifts the same run down one bit at a
 *   time instead of counting.
 *
 * Each benchmark enumerates every k-subset of n bits, each step feeding the next
 * (`sum_all_k_subsets`), so it measures the latency of one step.
 *
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`,
 * 2026-10-15): the library is fastest on both shapes. `by_run_ctz` is 30-44%
 * slower. `by_shift_loop` ties on C(28,14) but is 52-70% slower on C(40,6).
 * Numbers: docs/bit-utilities.md.
 */
#include <atl/bits/bitlow.hpp>
#include <atl/bits/next_bits_combination.hpp>
#include <atl/types/alias.hpp>
#include <bit>
#include <catch2/catch.hpp>
#include <string>
#include <type_traits>
#include <utility>

using namespace atl;

namespace {
	/** @brief Same result as `next_bits_combination`; the shift is the trailing-zero count of the carried run. */
	constexpr uint64 next_bits_combination_by_run_ctz(uint64 x) noexcept {
		uint64 const r = lowest_one(x);
		if (x == 0 || x > ~uint64(0) - r) return 0;
		x += r;
		uint64 z = lowest_one(x) - r;
		z >>= std::countr_zero(z);
		return x | (z >> 1);
	}

	/** @brief Same result as `next_bits_combination`; the carried run is shifted down one bit at a time. */
	constexpr uint64 next_bits_combination_by_shift_loop(uint64 x) noexcept {
		uint64 const r = lowest_one(x);
		if (x == 0 || x > ~uint64(0) - r) return 0;
		x += r;
		uint64 z = lowest_one(x) - r;
		while ((z & 1) == 0) z >>= 1;
		return x | (z >> 1);
	}

	/**
	 * @brief Sum of all k-subsets of n bits, enumerated with @p next; each step depends
	 *        on the previous one, so the time is the per-step latency. The sum (modulo
	 *        2^64) keeps the loop from being optimised away and checks the contenders agree.
	 * @pre `0 < k <= n < 64`; next returns the correct combination successor.
	 */
	template<typename Successor>
		requires std::is_invocable_r_v<uint64, Successor const &, uint64>
	uint64 sum_all_k_subsets(int n, int k, Successor const &next) {
		uint64 x = (uint64(1) << k) - 1, sum = 0;
		uint64 const limit = uint64(1) << n;
		while (x < limit) {
			sum += x;
			x = next(x);
		}
		return sum;
	}
}

TEST_CASE("next_bits_combination contenders", "[bits]") {
	for (uint64 x = 1; x < (uint64(1) << 20); ++x) {
		REQUIRE(next_bits_combination_by_run_ctz(x) == next_bits_combination(x));
		REQUIRE(next_bits_combination_by_shift_loop(x) == next_bits_combination(x));
	}
	REQUIRE(next_bits_combination(~uint64(0) << 3) == 0);
	// Lambdas give each contender its own type, so every call below is a direct, inlinable call.
	auto const library = [](uint64 x) noexcept { return next_bits_combination(x); };
	auto const by_run_ctz = [](uint64 x) noexcept { return next_bits_combination_by_run_ctz(x); };
	auto const by_shift_loop = [](uint64 x) noexcept { return next_bits_combination_by_shift_loop(x); };
	for (auto [n, k] : {std::pair{28, 14}, std::pair{40, 6}}) {
		std::string const shape = "C(" + std::to_string(n) + "," + std::to_string(k) + ")";
		REQUIRE(sum_all_k_subsets(n, k, library) == sum_all_k_subsets(n, k, by_run_ctz));
		REQUIRE(sum_all_k_subsets(n, k, library) == sum_all_k_subsets(n, k, by_shift_loop));
		BENCHMARK(shape + " | library") { return sum_all_k_subsets(n, k, library); };
		BENCHMARK(shape + " | by run ctz") { return sum_all_k_subsets(n, k, by_run_ctz); };
		BENCHMARK(shape + " | by shift loop") { return sum_all_k_subsets(n, k, by_shift_loop); };
	}
}
