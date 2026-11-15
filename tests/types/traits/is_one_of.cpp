#include <atl/types/alias.hpp>
#include <atl/types/traits/is_one_of.hpp>
#include <catch2/catch.hpp>

TEST_CASE("Test trait atl::traits::is_one_of", "[types/traits]") {
	STATIC_CHECK(!atl::traits::is_one_of_v<bool>);
	STATIC_CHECK(!atl::traits::is_one_of_v<bool, void>);
	STATIC_CHECK(!atl::traits::is_one_of_v<int, int const, int const volatile, int *, int const *, int const volatile *>);
	STATIC_CHECK(atl::traits::is_one_of_v<int, int const, int const volatile, int *, int const *, int const volatile *, int>);
	STATIC_CHECK(!atl::traits::is_one_of_v<atl::int128, atl::int128 const, atl::int128 const volatile, atl::int128 *, atl::int128 const *, atl::int128 const volatile *>);
	STATIC_CHECK(atl::traits::is_one_of_v<atl::int128, atl::int128 const, atl::int128 const volatile, atl::int128 *, atl::int128 const *, atl::int128 const volatile *, atl::int128>);
	STATIC_CHECK(!atl::traits::is_one_of_v<atl::uint128, atl::uint128 const, atl::uint128 const volatile, atl::uint128 *, atl::uint128 const *, atl::uint128 const volatile *>);
	STATIC_CHECK(atl::traits::is_one_of_v<atl::uint128, atl::uint128 const, atl::uint128 const volatile, atl::uint128 *, atl::uint128 const *, atl::uint128 const volatile *, atl::uint128>);
}
