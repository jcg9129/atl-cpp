#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/concepts.hpp>
#include <catch2/catch.hpp>
#include <functional>

TEST_CASE("Test concepts atl::modular_integer and atl::static_mod_int", "[types/concepts]") {
	STATIC_CHECK(atl::modular_integer<atl::mod_int32<998244353>>);
	STATIC_CHECK(atl::modular_integer<atl::mod_int64<(atl::uint64(1) << 61) - 1>>);
	STATIC_CHECK(atl::modular_integer<atl::dynamic_mod_int32<>>);
	STATIC_CHECK(atl::modular_integer<atl::dynamic_mod_int64<1>>);
	STATIC_CHECK(!atl::modular_integer<atl::uint32>);
	STATIC_CHECK(!atl::modular_integer<double>);

	STATIC_CHECK(atl::static_mod_int<atl::mod_int32<998244353>>);
	STATIC_CHECK(atl::static_mod_int<atl::mod_int64<(atl::uint64(1) << 61) - 1>>);

	STATIC_CHECK(!atl::static_mod_int<atl::dynamic_mod_int32<>>);
	STATIC_CHECK(!atl::static_mod_int<atl::uint32>);
}

TEST_CASE("modular types satisfy the algebraic operation concepts", "[nt][concepts]") {
	auto check = []<atl::modular_integer Z>() {
		STATIC_CHECK(atl::binary_operation<std::plus<>, Z>);
		STATIC_CHECK(atl::binary_operation<std::multiplies<>, Z>);
		STATIC_CHECK(atl::inverse_operation<std::negate<>, Z>);
	};
	check.operator()<atl::mod_int32<998244353>>();
	check.operator()<atl::mod_int64<18446744073709551557ull>>();
	check.operator()<atl::mod_int<atl::uint128, (atl::uint128(1) << 89) - 1>>();
	check.operator()<atl::dynamic_mod_int32<>>();
	check.operator()<atl::dynamic_mod_int64<>>();
	STATIC_CHECK(atl::static_mod_int<atl::mod_int<atl::uint128, (atl::uint128(1) << 89) - 1>>);
}
