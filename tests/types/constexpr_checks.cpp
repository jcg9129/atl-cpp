#include <atl/types/alias.hpp>
#include <atl/types/detail/int256.hpp>
#include <catch2/catch.hpp>

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
