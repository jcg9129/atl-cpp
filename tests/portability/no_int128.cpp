#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/detail/int256.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <atl/utils/floor_midpoint.hpp>
#include <atl/utils/hash.hpp>
#include <atl/utils/radix_sort.hpp>
#include <concepts>
#include <vector>

static_assert(ATL_HAS_INT128 == 0);
#ifdef ATL_FORCE_FALLBACKS
static_assert(atl::detail::uses_portable_wide_arithmetic);
#endif
static_assert(std::same_as<atl::int128, atl::detail::software_int128>);
static_assert(std::same_as<atl::uint128, atl::detail::software_uint128>);
static_assert(atl::traits::has_widen_mul_v<atl::uint64>);
static_assert(atl::detail::floor_midpoint(atl::uint64(0), atl::uint64(-1)) == atl::uint64(-1) / 2);
static_assert((atl::uint128{1} << 100) / 16 == (atl::uint128{1} << 96));
static_assert(atl::detail::compare_products(atl::int128{1} << 100, 7, atl::int128{1} << 100, 6) == 1);

int main() {
	atl::uint128 const high = atl::uint128{1} << 100;
	std::vector<atl::detail::keyed_position<atl::uint128>> keys{{high + 1, 0}, {1, 1}, {high, 2}};
	atl::detail::radix_sort<11>(keys);
	if (keys[0].position != 1 || keys[1].position != 2 || keys[2].position != 0) return 1;
	atl::set_hash_seed(42);
	return atl::hash<atl::uint128>{}(high) != atl::hash<atl::uint128>{}(1) ? 0 : 1;
}
