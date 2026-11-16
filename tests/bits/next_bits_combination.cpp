#include <atl/bits/next_bits_combination.hpp>
#include <atl/types/alias.hpp>
#include <bit>
#include <catch2/catch.hpp>
#include <cstddef>

using atl::uint16;
using atl::uint32;
using atl::uint64;
using atl::uint8;

namespace {
	/** @brief The definitional answer: the next larger 16-bit value with the
	 *         same popcount, by linear scan, or 0 when there is none. */
	uint16 naive_next(uint16 x) {
		if (x == 0) return 0;
		for (uint32 y = uint32(x) + 1; y <= 0xFFFF; ++y)
			if (std::popcount(y) == std::popcount(x)) return uint16(y);
		return 0;
	}
}

TEST_CASE("next_bits_combination exhaustive on 16 bits", "[bits]") {
	for (uint32 v = 0; v <= 0xFFFF; ++v) {
		uint16 const x = uint16(v);
		REQUIRE(atl::next_bits_combination(x) == naive_next(x));
	}
}

TEST_CASE("next_bits_combination enumerates every k-subset of 8 bits once, in order", "[bits]") {
	std::size_t const binomial[9] = {1, 8, 28, 56, 70, 56, 28, 8, 1};
	for (uint32 k = 1; k <= 8; ++k) {
		uint8 x = uint8((1u << k) - 1);
		std::size_t count = 0;
		for (uint8 previous = 0; x != 0; x = atl::next_bits_combination(x)) {
			REQUIRE(x > previous);
			REQUIRE(std::popcount(x) == int(k));
			previous = x;
			++count;
		}
		REQUIRE(count == binomial[k]);
	}
}

TEST_CASE("next_bits_combination at the top of wide types", "[bits]") {
	REQUIRE(atl::next_bits_combination(~uint64(0)) == 0);
	REQUIRE(atl::next_bits_combination(uint64(1) << 63) == 0);
	REQUIRE(atl::next_bits_combination(uint64(0b11) << 62) == 0);
	REQUIRE(atl::next_bits_combination(uint64(1) << 62) == uint64(1) << 63);
	REQUIRE(atl::next_bits_combination(uint32(0b0111)) == 0b1011);
	REQUIRE(atl::next_bits_combination(atl::uint128(1) << 64) == atl::uint128(1) << 65);
	REQUIRE(atl::next_bits_combination(atl::uint128(0b11) << 126) == 0);
}
