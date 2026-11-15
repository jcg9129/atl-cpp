#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/integral.hpp>

static_assert(atl::integral<atl::uint32>);
static_assert(sizeof(atl::traits::make_unsigned_t<atl::int32>) == 4);

int main() {
	return atl::uint32(10) + atl::uint32(20) == 30 ? 0 : 1;
}
