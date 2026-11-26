#pragma once
#include <atl/ds/flat_hash_map_core.hpp>
#include <atl/nt/power_series_native.hpp>
#include <string>
#include <vector>

template<typename Key>
inline int split_route() {
	if constexpr (atl::detail::hasher_for<atl::hash<Key>, Key>)
		return 1;
	else
		return 0;
}

template<typename C>
inline bool split_series() {
	return atl::series_inverse(std::vector<C>{1, -1}, 129) == std::vector<C>(129, 1);
}

inline bool split_scalar_map() {
	atl::flat_hash_map<int, int> map;
	map[2] = 7;
	return map.at(2) == 7;
}
