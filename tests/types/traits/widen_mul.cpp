#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/int128.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <limits>

using atl::traits::has_widen_mul_v;
using atl::traits::widen_mul_type_t;

// the trait picks by SIZE and signedness, so every spelling of a width widens
static_assert(std::same_as<widen_mul_type_t<atl::int8>, atl::int16>);
static_assert(std::same_as<widen_mul_type_t<atl::uint8>, atl::uint16>);
static_assert(std::same_as<widen_mul_type_t<atl::int16>, atl::int32>);
static_assert(std::same_as<widen_mul_type_t<atl::uint16>, atl::uint32>);
static_assert(std::same_as<widen_mul_type_t<atl::int32>, atl::int64>);
static_assert(std::same_as<widen_mul_type_t<atl::uint32>, atl::uint64>);
static_assert(std::same_as<widen_mul_type_t<atl::uint32 const>, atl::uint64>);
static_assert(sizeof(widen_mul_type_t<int>) == 2 * sizeof(int));
static_assert(sizeof(widen_mul_type_t<unsigned>) == 2 * sizeof(unsigned));
// every 8-byte spelling widens: a trait keyed on the fixed-width aliases alone
// would miss the other one (long long where int64 is long), whose modular
// multiply would then silently take the bit-by-bit fallback
static_assert(sizeof(widen_mul_type_t<long>) == 2 * sizeof(long));
static_assert(sizeof(widen_mul_type_t<long long>) == 16);
static_assert(sizeof(widen_mul_type_t<unsigned long>) == 2 * sizeof(unsigned long));
static_assert(sizeof(widen_mul_type_t<unsigned long long>) == 16);
static_assert(sizeof(widen_mul_type_t<unsigned long long const>) == 16);
// no wider built-in: the type stays as it is
static_assert(std::same_as<widen_mul_type_t<atl::int128>, atl::int128>);
static_assert(std::same_as<widen_mul_type_t<atl::uint128>, atl::uint128>);
static_assert(std::same_as<widen_mul_type_t<double>, double>);

static_assert(has_widen_mul_v<atl::int8> && has_widen_mul_v<atl::uint8>);
static_assert(has_widen_mul_v<atl::int16> && has_widen_mul_v<atl::uint16>);
static_assert(has_widen_mul_v<atl::int32> && has_widen_mul_v<atl::uint32 const>);
static_assert(has_widen_mul_v<atl::uint64>);
static_assert(has_widen_mul_v<long long>);
static_assert(!has_widen_mul_v<atl::int128> && !has_widen_mul_v<atl::uint128>);
static_assert(!has_widen_mul_v<double>);

TEST_CASE("widen_mul_type: integral widths are checked at compile time", "[types]") {
	STATIC_CHECK(has_widen_mul_v<atl::uint32>);
}

namespace {
	template<atl::detail::builtin_integer Integer>
	void check_product() {
		if constexpr (sizeof(Integer) == 8) {
			using Wide = widen_mul_type_t<Integer>;
			if constexpr (std::numeric_limits<Integer>::is_signed) {
				auto const minimum = std::numeric_limits<Integer>::min();
				REQUIRE(Wide(minimum) * minimum == (atl::int128{1} << 126));
				REQUIRE(Wide(minimum) * Integer(-1) == (atl::int128{1} << 63));
			} else {
				auto const maximum = std::numeric_limits<Integer>::max();
				Wide const product = Wide(maximum) * maximum;
				REQUIRE(atl::uint64(product) == 1);
				REQUIRE(atl::uint64(product >> 64) == std::numeric_limits<atl::uint64>::max() - 1);
			}
		}
	}
}

TEST_CASE("widen_mul_type: every 64-bit spelling multiplies exactly", "[types]") {
	check_product<long>();
	check_product<unsigned long>();
	check_product<long long>();
	check_product<unsigned long long>();
	check_product<atl::int64>();
	check_product<atl::uint64>();
}
