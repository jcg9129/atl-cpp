#include <atl/ds/flat_hash_map.hpp>
#include <atl/nt/power_series.hpp>
#include "split_probe.hpp"

template int split_route<std::string>();

bool split_full() {
	return split_route<std::string>() == 1 && split_scalar_map() && split_series<atl::mod_int32<998244353>>();
}
