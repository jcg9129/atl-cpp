#include <atl/config.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/floor_midpoint.hpp>
// These runtime checks join the skeleton smoke check when number theory arrives.
#if __has_include(<atl/nt/binomial.hpp>) && __has_include(<atl/nt/basic.hpp>)
#include <atl/nt/basic.hpp>
#include <atl/nt/binomial.hpp>
#define ATL_CHECK_NT 1
#endif

static_assert(ATL_HAS_INT128 == 0);
static_assert(atl::detail::floor_midpoint(atl::uint64(0), atl::uint64(-1)) == atl::uint64(-1) / 2);

int main() {
#ifdef ATL_CHECK_NT
	if (atl::binomial(64, 32) != 1832624140942590534ull) return 1;
	if (!atl::is_prime(1000000007ull)) return 1;
	if (atl::is_prime(1000000007ull * 1000000009ull)) return 1;
#endif
	return 0;
}
