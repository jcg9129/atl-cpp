#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/detail/checked_arithmetic.hpp>
#include <atl/types/detail/int256.hpp>
#include <atl/types/int128.hpp>
#include <catch2/catch.hpp>
#include <limits>

using atl::uint64;

static_assert([] {
	using atl::int128;
	using atl::detail::int256;
	using atl::detail::compare_products;
	int128 const max = int128((~atl::uint128(0)) >> 1), min = -max - 1;
	// (-2^127)^2 = 2^254: the top limb is 2^62
	int256 const sq = int256::product(min, min);
	bool const is_square_right = sq.limb[3] == uint64(1) << 62 && sq.limb[2] == 0 && sq.limb[1] == 0 && sq.limb[0] == 0;
	return is_square_right && int256::product(min, min).sign() == 1 && int256::product(min, max).sign() == -1
		&& int256::product(max, min).sign() == -1 && int256::product(max, max).sign() == 1
		&& int256::product(min, 0).sign() == 0 && int256::product(min, -1).sign() == 1
		&& int256::product(min, 1) == -int256::product(max, 1) - int256::product(1, 1)
		&& compare_products(min, min, max, max) == 1 && compare_products(max, max, min, min) == -1
		&& compare_products(min, max, max, min) == 0 && compare_products(min, 1, max, 1) == -1
		&& compare_products(-1, min, max, 1) == 1;
}());

TEST_CASE("int256 products are constant-evaluable", "[types][constexpr]") {
	STATIC_REQUIRE(atl::detail::int256::product(-2, 3).sign() == -1);
}

namespace {
	template<atl::integral Integer>
	constexpr bool checked_arithmetic_boundaries() {
		using atl::detail::add_overflows;
		using atl::detail::mul_overflows;
		using atl::detail::sub_overflows;
		constexpr Integer low = (std::numeric_limits<Integer>::lowest)(), high = (std::numeric_limits<Integer>::max)();
		Integer result{};
		if (add_overflows(Integer(2), Integer(3), result) || result != 5) return false;
		if (sub_overflows(Integer(5), Integer(3), result) || result != 2) return false;
		if (mul_overflows(Integer(2), Integer(3), result) || result != 6) return false;
		if (mul_overflows(high, Integer(0), result) || result != 0) return false;
		if (mul_overflows(high, Integer(1), result) || result != high) return false;
		if (!add_overflows(high, Integer(1), result) || result != low) return false;
		if (!sub_overflows(low, Integer(1), result) || result != high) return false;
		if (!mul_overflows(high, Integer(2), result)) return false;
		if constexpr (atl::signed_integral<Integer>) {
			if (result != -2) return false;
			if (!mul_overflows(low, Integer(-1), result) || result != low) return false;
			if (!mul_overflows(Integer(-1), low, result) || result != low) return false;
			if (!add_overflows(low, Integer(-1), result) || result != high) return false;
			if (!sub_overflows(high, Integer(-1), result) || result != low) return false;
			if (mul_overflows(low, Integer(1), result) || result != low) return false;
			if (mul_overflows(Integer(-2), Integer(-3), result) || result != 6) return false;
			if (mul_overflows(Integer(-2), Integer(3), result) || result != -6) return false;
			if (mul_overflows(Integer(2), Integer(-3), result) || result != -6) return false;
			if (add_overflows(low, high, result) || result != -1) return false;
			if (sub_overflows(low, low, result) || result != 0) return false;
		} else {
			if (result != high - 1) return false;
			if (!mul_overflows(high, high, result) || result != 1) return false;
			if (!add_overflows(high, high, result) || result != high - 1) return false;
			if (!sub_overflows(Integer(0), high, result) || result != 1) return false;
		}
		return true;
	}

	static_assert(checked_arithmetic_boundaries<signed char>() && checked_arithmetic_boundaries<unsigned char>());
	static_assert(checked_arithmetic_boundaries<short>() && checked_arithmetic_boundaries<unsigned short>());
	static_assert(checked_arithmetic_boundaries<int>() && checked_arithmetic_boundaries<unsigned>());
	static_assert(checked_arithmetic_boundaries<long>() && checked_arithmetic_boundaries<unsigned long>());
	static_assert(checked_arithmetic_boundaries<long long>() && checked_arithmetic_boundaries<unsigned long long>());
	static_assert(checked_arithmetic_boundaries<atl::int128>() && checked_arithmetic_boundaries<atl::uint128>());
	static_assert(checked_arithmetic_boundaries<atl::detail::software_int128>() && checked_arithmetic_boundaries<atl::detail::software_uint128>());
}

TEMPLATE_TEST_CASE("checked arithmetic stores wrapped results and detects overflow", "[types][checked_arithmetic]",
				   signed char, unsigned char, short, unsigned short, int, unsigned, long, unsigned long,
				   long long, unsigned long long, atl::int128, atl::uint128, atl::detail::software_int128, atl::detail::software_uint128) {
	REQUIRE(checked_arithmetic_boundaries<TestType>());
}
