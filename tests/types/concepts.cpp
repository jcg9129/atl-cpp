#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <catch2/catch.hpp>
#include <concepts>

TEST_CASE("Test concept atl::integral", "[types/concepts]") {
	STATIC_CHECK(std::integral<bool>);
	STATIC_CHECK(!atl::integral<bool>);
	STATIC_CHECK(std::integral<char>);
	STATIC_CHECK(!atl::integral<char>);
	STATIC_CHECK(std::integral<int>);
	STATIC_CHECK(atl::integral<int>);
	STATIC_CHECK(std::integral<long long const>);
	STATIC_CHECK(atl::integral<long long const>);
	STATIC_CHECK(std::integral<unsigned const volatile>);
	STATIC_CHECK(atl::integral<unsigned const volatile>);
	STATIC_CHECK(!std::integral<double>);
	STATIC_CHECK(!atl::integral<double>);
	STATIC_CHECK(!std::integral<decltype("")>);
	STATIC_CHECK(!atl::integral<decltype("")>);

	// int128 — std's classification of __int128 is implementation-defined, so we
	// assert only atl's normalized behavior: it always counts as integral.
#if ATL_HAS_INT128
	STATIC_CHECK(atl::integral<atl::int128>);
	STATIC_CHECK(atl::integral<atl::int128 const>);
	STATIC_CHECK(atl::integral<atl::int128 const volatile>);
	STATIC_CHECK(!std::integral<atl::int128 *>);
	STATIC_CHECK(!atl::integral<atl::int128 *>);
	STATIC_CHECK(!std::integral<atl::int128 const *>);
	STATIC_CHECK(!atl::integral<atl::int128 const *>);
	STATIC_CHECK(!std::integral<atl::int128 const volatile *>);
	STATIC_CHECK(!atl::integral<atl::int128 const volatile *>);
#endif

	// uint128 — see the int128 note above.
#if ATL_HAS_INT128
	STATIC_CHECK(atl::integral<atl::uint128>);
	STATIC_CHECK(atl::integral<atl::uint128 const>);
	STATIC_CHECK(atl::integral<atl::uint128 const volatile>);
	STATIC_CHECK(!std::integral<atl::uint128 *>);
	STATIC_CHECK(!atl::integral<atl::uint128 *>);
	STATIC_CHECK(!std::integral<atl::uint128 const *>);
	STATIC_CHECK(!atl::integral<atl::uint128 const *>);
	STATIC_CHECK(!std::integral<atl::uint128 const volatile *>);
	STATIC_CHECK(!atl::integral<atl::uint128 const volatile *>);
#endif
}

TEST_CASE("Test concept atl::signed_integral", "[types/concepts]") {
	STATIC_CHECK(!std::signed_integral<bool>);
	STATIC_CHECK(!atl::signed_integral<bool>);
	// `char` has implementation-defined signedness; atl treats plain `char` as
	// neither signed nor unsigned integral regardless of platform.
	STATIC_CHECK(!atl::signed_integral<char>);
	STATIC_CHECK(std::signed_integral<int>);
	STATIC_CHECK(atl::signed_integral<int>);
	STATIC_CHECK(std::signed_integral<long long const>);
	STATIC_CHECK(atl::signed_integral<long long const>);
	STATIC_CHECK(!std::signed_integral<unsigned const volatile>);
	STATIC_CHECK(!atl::signed_integral<unsigned const volatile>);
	STATIC_CHECK(!std::signed_integral<double>);
	STATIC_CHECK(!atl::signed_integral<double>);
	STATIC_CHECK(!std::signed_integral<decltype("")>);
	STATIC_CHECK(!atl::signed_integral<decltype("")>);

	// int128
#if ATL_HAS_INT128
	STATIC_CHECK(atl::signed_integral<atl::int128>);
	STATIC_CHECK(atl::signed_integral<atl::int128 const>);
	STATIC_CHECK(atl::signed_integral<atl::int128 const volatile>);
	STATIC_CHECK(!std::signed_integral<atl::int128 *>);
	STATIC_CHECK(!atl::signed_integral<atl::int128 *>);
	STATIC_CHECK(!std::signed_integral<atl::int128 const *>);
	STATIC_CHECK(!atl::signed_integral<atl::int128 const *>);
	STATIC_CHECK(!std::signed_integral<atl::int128 const volatile *>);
	STATIC_CHECK(!atl::signed_integral<atl::int128 const volatile *>);
#endif

	// uint128
#if ATL_HAS_INT128
	STATIC_CHECK(!std::signed_integral<atl::uint128>);
	STATIC_CHECK(!atl::signed_integral<atl::uint128>);
	STATIC_CHECK(!std::signed_integral<atl::uint128 const>);
	STATIC_CHECK(!atl::signed_integral<atl::uint128 const>);
	STATIC_CHECK(!std::signed_integral<atl::uint128 const volatile>);
	STATIC_CHECK(!atl::signed_integral<atl::uint128 const volatile>);
	STATIC_CHECK(!std::signed_integral<atl::uint128 *>);
	STATIC_CHECK(!atl::signed_integral<atl::uint128 *>);
	STATIC_CHECK(!std::signed_integral<atl::uint128 const *>);
	STATIC_CHECK(!atl::signed_integral<atl::uint128 const *>);
	STATIC_CHECK(!std::signed_integral<atl::uint128 const volatile *>);
	STATIC_CHECK(!atl::signed_integral<atl::uint128 const volatile *>);
#endif
}

TEST_CASE("Test concept atl::unsigned_integral", "[types/concepts]") {
	STATIC_CHECK(std::unsigned_integral<bool>);
	STATIC_CHECK(!atl::unsigned_integral<bool>);
	// See the `char` note in the signed_integral test above.
	STATIC_CHECK(!atl::unsigned_integral<char>);
	STATIC_CHECK(!std::unsigned_integral<int>);
	STATIC_CHECK(!atl::unsigned_integral<int>);
	STATIC_CHECK(!std::unsigned_integral<long long const>);
	STATIC_CHECK(!atl::unsigned_integral<long long const>);
	STATIC_CHECK(std::unsigned_integral<unsigned const volatile>);
	STATIC_CHECK(atl::unsigned_integral<unsigned const volatile>);
	STATIC_CHECK(!std::unsigned_integral<double>);
	STATIC_CHECK(!atl::unsigned_integral<double>);
	STATIC_CHECK(!std::unsigned_integral<decltype("")>);
	STATIC_CHECK(!atl::unsigned_integral<decltype("")>);

	// int128
#if ATL_HAS_INT128
	STATIC_CHECK(!std::unsigned_integral<atl::int128>);
	STATIC_CHECK(!atl::unsigned_integral<atl::int128>);
	STATIC_CHECK(!std::unsigned_integral<atl::int128 const>);
	STATIC_CHECK(!atl::unsigned_integral<atl::int128 const>);
	STATIC_CHECK(!std::unsigned_integral<atl::int128 const volatile>);
	STATIC_CHECK(!atl::unsigned_integral<atl::int128 const volatile>);
	STATIC_CHECK(!std::unsigned_integral<atl::int128 *>);
	STATIC_CHECK(!atl::unsigned_integral<atl::int128 *>);
	STATIC_CHECK(!std::unsigned_integral<atl::int128 const *>);
	STATIC_CHECK(!atl::unsigned_integral<atl::int128 const *>);
	STATIC_CHECK(!std::unsigned_integral<atl::int128 const volatile *>);
	STATIC_CHECK(!atl::unsigned_integral<atl::int128 const volatile *>);
#endif

	// uint128
#if ATL_HAS_INT128
	STATIC_CHECK(atl::unsigned_integral<atl::uint128>);
	STATIC_CHECK(atl::unsigned_integral<atl::uint128 const>);
	STATIC_CHECK(atl::unsigned_integral<atl::uint128 const volatile>);
	STATIC_CHECK(!std::unsigned_integral<atl::uint128 *>);
	STATIC_CHECK(!atl::unsigned_integral<atl::uint128 *>);
	STATIC_CHECK(!std::unsigned_integral<atl::uint128 const *>);
	STATIC_CHECK(!atl::unsigned_integral<atl::uint128 const *>);
	STATIC_CHECK(!std::unsigned_integral<atl::uint128 const volatile *>);
	STATIC_CHECK(!atl::unsigned_integral<atl::uint128 const volatile *>);
#endif
}
