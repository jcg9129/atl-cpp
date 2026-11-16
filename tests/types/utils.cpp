#include <atl/types/alias.hpp>
#include <atl/types/utils.hpp>
#include <catch2/catch.hpp>

namespace {
	template<typename T>
	concept has_num_bits = requires { atl::num_bits<T>; };
}

TEST_CASE("num_bits reports the bit width of a type", "[utils]") {
	STATIC_REQUIRE(atl::num_bits<atl::uint8> == 8);
	STATIC_REQUIRE(atl::num_bits<atl::int8> == 8);
	STATIC_REQUIRE(atl::num_bits<atl::uint16> == 16);
	STATIC_REQUIRE(atl::num_bits<atl::uint32> == 32);
	STATIC_REQUIRE(atl::num_bits<atl::int32> == 32);
	STATIC_REQUIRE(atl::num_bits<atl::uint64> == 64);
	STATIC_REQUIRE(atl::num_bits<atl::int64> == 64);
	STATIC_REQUIRE(atl::num_bits<atl::uint128> == 128);
	STATIC_REQUIRE(!has_num_bits<bool>); // integers only
	STATIC_REQUIRE(!has_num_bits<double>);
}
