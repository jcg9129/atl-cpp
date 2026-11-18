#include <atl/nt/modular.hpp>
#include <atl/types/concepts.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>

TEST_CASE("widen_mul_type: every 64-bit spelling multiplies exactly", "[types]") {
	auto check = []<atl::unsigned_integral Integer>() {
		if constexpr (sizeof(Integer) == 8) {
			Integer const a = static_cast<Integer>(0xfedcba9876543210ull);
			Integer const b = static_cast<Integer>(0x123456789abcdef1ull);
			Integer const modulus = static_cast<Integer>(0xffffffffffffffc5ull);
			REQUIRE(atl::mul_mod_unchecked(a % modulus, b % modulus, modulus)
					== atl::mul_mod_binary(a % modulus, b % modulus, modulus));
			REQUIRE(atl::mul_mod_unchecked(a % modulus, b % modulus, modulus)
					== atl::mul_mod_unchecked(
						atl::uint64(a % modulus), atl::uint64(b % modulus), atl::uint64(modulus)));
		}
	};
	check.operator()<unsigned long>();
	check.operator()<unsigned long long>();
	check.operator()<atl::uint64>();
}

TEST_CASE("widening products retain both words and middle carries", "[types][widen_mul]") {
	using atl::uint128;
	using atl::uint64;
	STATIC_CHECK(atl::traits::has_widen_mul_v<uint64>);
	STATIC_CHECK(atl::traits::has_widen_mul_v<unsigned long long>);
	STATIC_CHECK(!atl::traits::has_widen_mul_v<uint128>);
	STATIC_CHECK(std::same_as<atl::traits::widen_mul_type_t<uint64>, uint128>);
	uint128 const all_bits = ~uint128(0);
	REQUIRE(atl::detail::mulhi_u128(all_bits, all_bits) == all_bits - 1);
	// Both additions of the middle limb carry for (2^65 - 1)^2.
	uint128 const two_word = (uint128(1) << 65) - 1;
	REQUIRE(atl::detail::mulhi_u128(two_word, two_word) == 3);
	REQUIRE(atl::detail::mulhi_u128(uint128(1) << 127, uint128(2)) == 1);
	constexpr auto product = atl::detail::multiply_wide(~uint64(0), ~uint64(0));
	STATIC_CHECK(product.low == 1);
	STATIC_CHECK(product.high == ~uint64(0) - 1);
	REQUIRE(uint128(~uint64(0)) * ~uint64(0) == (uint128(product.high) << 64) + product.low);
	static_assert(sizeof(std::size_t) == 4 || sizeof(std::size_t) == 8);
}
