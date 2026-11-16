#include <atl/bits/bitlow.hpp>
#include <atl/bits/next_bits_combination.hpp>
#include <atl/types/alias.hpp>
#include <bit>
#include <catch2/catch.hpp>
#include <string>
#include <utility>

using namespace atl;

namespace {
	/**
	 * @brief The serial-count contender: the
	 *        trailing-zero count of `z` taken at the end of the chain.
	 */
	constexpr uint64 next_ctz_of_z(uint64 x) noexcept {
		uint64 const r = lowest_one(x);
		if (x == 0 || x > ~uint64(0) - r) return 0;
		x += r;
		uint64 z = lowest_one(x) - r;
		z >>= std::countr_zero(z);
		return x | (z >> 1);
	}

	/** @brief The classic form: a shift loop instead of countr_zero. */
	constexpr uint64 next_shift_loop(uint64 x) noexcept {
		uint64 const r = lowest_one(x);
		if (x == 0 || x > ~uint64(0) - r) return 0;
		x += r;
		uint64 z = lowest_one(x) - r;
		while ((z & 1) == 0) z >>= 1;
		return x | (z >> 1);
	}

	/**
	 * @brief Loop-carried enumeration with a checksum modulo 2^64.
	 * @pre `0 < k <= n < 64`; Next returns the correct combination successor.
	 */
	template<uint64 (*Next)(uint64) noexcept>
		requires(Next != nullptr)
	uint64 walk(int n, int k) {
		uint64 x = (uint64(1) << k) - 1, sum = 0;
		uint64 const limit = uint64(1) << n;
		while (x < limit) {
			sum += x;
			x = Next(x);
		}
		return sum;
	}

	uint64 shipped(uint64 x) noexcept { return next_bits_combination(x); }
}

TEST_CASE("next_bits_combination contenders", "[bits]") {
	for (uint64 x = 1; x < (uint64(1) << 20); ++x) {
		REQUIRE(next_ctz_of_z(x) == next_bits_combination(x));
		REQUIRE(next_shift_loop(x) == next_bits_combination(x));
	}
	REQUIRE(next_bits_combination(~uint64(0) << 3) == 0);
	for (auto [n, k] : {std::pair{28, 14}, std::pair{40, 6}}) {
		std::string const shape = "C(" + std::to_string(n) + "," + std::to_string(k) + ")";
		REQUIRE(walk<shipped>(n, k) == walk<next_ctz_of_z>(n, k));
		BENCHMARK(shape + " | ctz of z (serial)") { return walk<next_ctz_of_z>(n, k); };
		BENCHMARK(shape + " | shipped (ctz of x, parallel)") { return walk<shipped>(n, k); };
		BENCHMARK(shape + " | shift loop") { return walk<next_shift_loop>(n, k); };
	}
}
