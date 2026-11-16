#include <atl/bits/bitlow.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <catch2/catch.hpp>

using atl::uint16;
using atl::uint32;
using atl::uint64;

namespace {
	template<atl::unsigned_integral T>
	T ref_lowest_one(T x) {
		if (x == 0) return 0;
		T bit = 1;
		while ((x & bit) == 0) bit = T(bit << 1);
		return bit;
	}
}

TEST_CASE("lowest_one exhaustive on 16 bits", "[bitlow]") {
	for (uint32 v = 0; v <= 0xFFFF; ++v) {
		uint16 x = uint16(v);
		REQUIRE(atl::lowest_one(x) == ref_lowest_one(x));
	}
}

TEST_CASE("lowest_one boundary values", "[bitlow]") {
	REQUIRE(atl::lowest_one(uint32(0)) == 0);
	REQUIRE(atl::lowest_one(uint32(0b101000)) == 0b1000);
	REQUIRE(atl::lowest_one(~uint64(0)) == 1);
}
