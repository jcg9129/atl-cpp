#include <atl/types/alias.hpp>
#include <atl/types/traits/integral.hpp>
#include <catch2/catch.hpp>
#include <type_traits>

namespace {
	template<typename T>
	concept has_make_signed = requires { typename atl::traits::make_signed_t<T>; };

	template<typename T>
	concept has_make_unsigned = requires { typename atl::traits::make_unsigned_t<T>; };

	struct unsupported {};

	static_assert(has_make_signed<int> && has_make_unsigned<int>);
	static_assert(has_make_signed<char> && has_make_unsigned<char>);
	static_assert(!has_make_signed<bool> && !has_make_unsigned<bool>);
	static_assert(!has_make_signed<bool const> && !has_make_unsigned<bool const>);
	static_assert(!has_make_signed<float> && !has_make_unsigned<float>);
	static_assert(!has_make_signed<unsupported> && !has_make_unsigned<unsupported>);
	static_assert(!has_make_signed<int &> && !has_make_unsigned<int &>);
	static_assert(std::is_same_v<atl::traits::make_signed_t<unsigned const volatile>, int const volatile>);
	static_assert(std::is_same_v<atl::traits::make_unsigned_t<int const volatile>, unsigned const volatile>);
	static_assert(has_make_signed<atl::uint128> && has_make_unsigned<atl::int128>);

}

TEST_CASE("Test trait atl::traits::is_integral", "[types/traits]") {
	STATIC_CHECK(std::is_integral_v<bool>);
	STATIC_CHECK(!atl::traits::is_integral_v<bool>);
	STATIC_CHECK(std::is_integral_v<char>);
	STATIC_CHECK(!atl::traits::is_integral_v<char>);
	STATIC_CHECK(std::is_integral_v<int>);
	STATIC_CHECK(atl::traits::is_integral_v<int>);
	STATIC_CHECK(std::is_integral_v<long long const>);
	STATIC_CHECK(atl::traits::is_integral_v<long long const>);
	STATIC_CHECK(std::is_integral_v<unsigned const volatile>);
	STATIC_CHECK(atl::traits::is_integral_v<unsigned const volatile>);
	STATIC_CHECK(!std::is_integral_v<double>);
	STATIC_CHECK(!atl::traits::is_integral_v<double>);
	STATIC_CHECK(!std::is_integral_v<decltype("")>);
	STATIC_CHECK(!atl::traits::is_integral_v<decltype("")>);

	// int128 — std::is_integral for __int128 is implementation-defined (false
	// under a strict -std=c++NN on some toolchains, true on others), so we assert
	// only atl's normalized behavior: it always counts as integral.
	STATIC_CHECK(atl::traits::is_integral_v<atl::int128>);
	STATIC_CHECK(atl::traits::is_integral_v<atl::int128 const>);
	STATIC_CHECK(atl::traits::is_integral_v<atl::int128 const volatile>);
	STATIC_CHECK(!std::is_integral_v<atl::int128 *>);
	STATIC_CHECK(!atl::traits::is_integral_v<atl::int128 *>);
	STATIC_CHECK(!std::is_integral_v<atl::int128 const *>);
	STATIC_CHECK(!atl::traits::is_integral_v<atl::int128 const *>);
	STATIC_CHECK(!std::is_integral_v<atl::int128 const volatile *>);
	STATIC_CHECK(!atl::traits::is_integral_v<atl::int128 const volatile *>);

	// uint128 — see the int128 note above.
	STATIC_CHECK(atl::traits::is_integral_v<atl::uint128>);
	STATIC_CHECK(atl::traits::is_integral_v<atl::uint128 const>);
	STATIC_CHECK(atl::traits::is_integral_v<atl::uint128 const volatile>);
	STATIC_CHECK(!std::is_integral_v<atl::uint128 *>);
	STATIC_CHECK(!atl::traits::is_integral_v<atl::uint128 *>);
	STATIC_CHECK(!std::is_integral_v<atl::uint128 const *>);
	STATIC_CHECK(!atl::traits::is_integral_v<atl::uint128 const *>);
	STATIC_CHECK(!std::is_integral_v<atl::uint128 const volatile *>);
	STATIC_CHECK(!atl::traits::is_integral_v<atl::uint128 const volatile *>);
}

TEST_CASE("Test trait atl::traits::is_signed_integral", "[types/traits]") {
	STATIC_CHECK(!atl::traits::is_signed_integral_v<bool>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<char>);
	STATIC_CHECK(atl::traits::is_signed_integral_v<int>);
	STATIC_CHECK(atl::traits::is_signed_integral_v<long long const>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<unsigned const volatile>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<double>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<decltype("")>);

	static_assert(atl::traits::is_signed_integral_v<int>);
	static_assert(!atl::traits::is_signed_integral_v<int &>);
	static_assert(!atl::traits::is_signed_integral_v<int *>);
	static_assert(!atl::traits::is_signed_integral_v<void>);
	static_assert(!atl::traits::is_signed_integral_v<void *>);
	static_assert(!atl::traits::is_signed_integral_v<float>);
	static_assert(!atl::traits::is_signed_integral_v<float &>);
	static_assert(!atl::traits::is_signed_integral_v<float *>);
	static_assert(!atl::traits::is_signed_integral_v<std::nullptr_t>);
	static_assert(!atl::traits::is_signed_integral_v<std::is_fundamental<int>>);

	// int128
	STATIC_CHECK(atl::traits::is_signed_integral_v<atl::int128>);
	STATIC_CHECK(atl::traits::is_signed_integral_v<atl::int128 const>);
	STATIC_CHECK(atl::traits::is_signed_integral_v<atl::int128 const volatile>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<atl::int128 *>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<atl::int128 const *>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<atl::int128 const volatile *>);

	// uint128
	STATIC_CHECK(!atl::traits::is_signed_integral_v<atl::uint128>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<atl::uint128 const>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<atl::uint128 const volatile>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<atl::uint128 *>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<atl::uint128 const *>);
	STATIC_CHECK(!atl::traits::is_signed_integral_v<atl::uint128 const volatile *>);
}

TEST_CASE("Test trait atl::traits::is_unsigned_integral", "[types/traits]") {
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<bool>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<char>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<int>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<long long const>);
	STATIC_CHECK(atl::traits::is_unsigned_integral_v<unsigned const volatile>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<double>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<decltype("")>);

	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<int>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<int &>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<int *>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<void>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<void *>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<float>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<float &>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<float *>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<std::nullptr_t>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<std::is_fundamental<int>>);

	// int128
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<atl::int128>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<atl::int128 const>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<atl::int128 const volatile>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<atl::int128 *>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<atl::int128 const *>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<atl::int128 const volatile *>);

	// uint128
	STATIC_CHECK(atl::traits::is_unsigned_integral_v<atl::uint128>);
	STATIC_CHECK(atl::traits::is_unsigned_integral_v<atl::uint128 const>);
	STATIC_CHECK(atl::traits::is_unsigned_integral_v<atl::uint128 const volatile>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<atl::uint128 *>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<atl::uint128 const *>);
	STATIC_CHECK(!atl::traits::is_unsigned_integral_v<atl::uint128 const volatile *>);
}

TEST_CASE("Test trait atl::traits::make_unsigned", "[types/traits]") {
	enum struct E : signed short {};

	STATIC_CHECK(std::is_same_v<std::make_unsigned_t<char>, unsigned char>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_unsigned_t<char>, unsigned char>);
	STATIC_CHECK(std::is_same_v<std::make_unsigned_t<signed int>, unsigned int>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_unsigned_t<signed int>, unsigned int>);
	STATIC_CHECK(std::is_same_v<std::make_unsigned_t<signed long volatile>, unsigned long volatile>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_unsigned_t<signed long volatile>, unsigned long volatile>);
	STATIC_CHECK(std::is_same_v<std::make_unsigned_t<E>, unsigned short>);
	STATIC_CHECK(has_make_unsigned<signed short>);
	STATIC_CHECK(!has_make_unsigned<E>); // enums are rejected on purpose

	// int128: std::make_unsigned need not accept it in a strict -std=c++NN mode
	STATIC_CHECK(std::is_same_v<atl::traits::make_unsigned_t<atl::int128>, atl::uint128>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_unsigned_t<atl::int128 const>, atl::uint128 const>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_unsigned_t<atl::int128 const volatile>, atl::uint128 const volatile>);

	// uint128
	STATIC_CHECK(std::is_same_v<atl::traits::make_unsigned_t<atl::uint128>, atl::uint128>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_unsigned_t<atl::uint128 const>, atl::uint128 const>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_unsigned_t<atl::uint128 const volatile>, atl::uint128 const volatile>);
}

TEST_CASE("Test trait atl::traits::make_signed", "[types/traits]") {
	enum struct E : unsigned short {};

	STATIC_CHECK(std::is_same_v<std::make_signed_t<unsigned char>, signed char>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_signed_t<unsigned char>, signed char>);
	STATIC_CHECK(std::is_same_v<std::make_signed_t<unsigned int>, signed int>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_signed_t<unsigned int>, signed int>);
	STATIC_CHECK(std::is_same_v<std::make_signed_t<unsigned long volatile>, signed long volatile>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_signed_t<unsigned long volatile>, signed long volatile>);
	STATIC_CHECK(std::is_same_v<std::make_signed_t<E>, signed short>);
	STATIC_CHECK(has_make_signed<unsigned short>);
	STATIC_CHECK(!has_make_signed<E>); // enums are rejected on purpose

	// uint128
	STATIC_CHECK(std::is_same_v<atl::traits::make_signed_t<atl::uint128>, atl::int128>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_signed_t<atl::uint128 const>, atl::int128 const>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_signed_t<atl::uint128 const volatile>, atl::int128 const volatile>);

	// int128
	STATIC_CHECK(std::is_same_v<atl::traits::make_signed_t<atl::int128>, atl::int128>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_signed_t<atl::int128 const>, atl::int128 const>);
	STATIC_CHECK(std::is_same_v<atl::traits::make_signed_t<atl::int128 const volatile>, atl::int128 const volatile>);
}
