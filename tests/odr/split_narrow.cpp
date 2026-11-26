#include <concepts>
#include "split_probe.hpp"

static_assert(atl::detail::hasher_for<atl::hash<std::string>, std::string>);
template int split_route<std::string>();

bool split_narrow() {
	return split_route<std::string>() == 1 && split_geometry() && split_scalar_map() && split_series<atl::mod_int32<998244353>>();
}

static_assert(atl::static_prime_mod_int<atl::mod_int32<998244353>>);
static_assert(!atl::static_prime_mod_int<atl::mod_int32<998244351>>);
static_assert(atl::static_prime_mod_int<atl::mod_int64<18446744073709551557ull>>);
static_assert(std::same_as<atl::modulus_engine<atl::uint64(5)>, atl::montgomery<atl::uint64(5)>>);
