#include <atl/ds/detail/radix.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/int128.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include "../expect_abort.hpp"

namespace {
	template<atl::unsigned_integral Index>
		requires(std::numeric_limits<Index>::digits == 128)
	constexpr bool check_radix() {
		using radix = atl::detail::radix_index<16>;
		Index const high = Index(1) << 100;
		return radix::height_for(Index(0)) == 0 && radix::height_for(Index(16)) == 0
			&& radix::height_for(Index(17)) == 1 && radix::height_for(high) == 24
			&& radix::height_for(Index(high + 1)) == 25 && radix::digit(high, 25) == 1
			&& radix::digit((std::numeric_limits<Index>::max)(), 31) == 15;
	}

	static_assert(check_radix<atl::uint128>());
	// Narrow indices take the size_t overload: a digit above their own width is simply zero.
	static_assert(atl::detail::radix_index<16>::digit(atl::uint8{128}, 2) == 0);
	static_assert(check_radix<atl::detail::software_uint128>());
	static_assert(atl::detail::radix_index<65536>::height_for(atl::uint8(255)) == 0);
	static_assert(atl::detail::radix_index<16>::height_for(std::size_t(257)) == 2);
	static_assert(!atl::detail::valid_radix<3>);
}

TEST_CASE("radix_index: wide keys and digit shift bounds", "[ds][radix]") {
	REQUIRE(check_radix<atl::uint128>());
	REQUIRE(check_radix<atl::detail::software_uint128>());
	REQUIRE(atl_test::aborts([] { (void)atl::detail::radix_index<16>::digit(std::size_t(0), std::numeric_limits<std::size_t>::digits / 4); }));
}
