#include "split_probe.hpp"

static_assert(atl::detail::hasher_for<atl::hash<std::string>, std::string>);
template int split_route<std::string>();

bool split_narrow() {
	return split_route<std::string>() == 1 && split_scalar_map() && split_series<atl::mod_int32<998244353>>();
}
