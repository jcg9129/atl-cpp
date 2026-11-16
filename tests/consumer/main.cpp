#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/int128_format.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/types/traits/widen_mul.hpp>

static_assert(atl::integral<atl::uint32>);
static_assert(atl::integral<atl::int128> && atl::unsigned_integral<atl::uint128>);
static_assert(atl::traits::has_widen_mul_v<atl::uint64>);
static_assert((atl::uint128{1} << 100) / 16 == (atl::uint128{1} << 96));
static_assert(sizeof(atl::traits::make_unsigned_t<atl::int32>) == 4);

int main() {
	return atl::uint32(10) + atl::uint32(20) == 30 ? 0 : 1;
}
