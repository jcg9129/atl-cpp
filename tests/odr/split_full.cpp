#include <atl/ds/flat_hash_map.hpp>
#include <atl/nt/online_convolution.hpp>
#include <atl/nt/power_series.hpp>
#include <concepts>
#include "split_probe.hpp"

template int split_route<std::string>();

bool split_full() {
	return split_route<std::string>() == 1 && split_scalar_map() && split_series<atl::mod_int32<998244353>>();
}

static_assert(atl::static_prime_mod_int<atl::mod_int32<998244353>>);
static_assert(!atl::static_prime_mod_int<atl::mod_int32<998244351>>);
static_assert(atl::static_prime_mod_int<atl::mod_int64<18446744073709551557ull>>);
static_assert(std::same_as<atl::modulus_engine<atl::uint64(5)>, atl::montgomery<atl::uint64(5)>>);
