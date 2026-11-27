#pragma once
#include <atl/ds/flat_hash_map_core.hpp>
#include <atl/geometry/triangulation_integral.hpp>
#include <atl/nt/mod_int_static.hpp>
#include <atl/nt/online_convolution_ntt.hpp>
#include <atl/nt/power_series_ntt.hpp>
#include <atl/nt/prime_field_helpers.hpp>
#include <array>
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
	atl::online_convolution<C> product;
	atl::online_fixed_convolution<C> fixed(std::vector<C>{1, 2});
	atl::factorial_table<C> const table(10);
	return product.push(C(2), C(3)) == C(6) && fixed.push(C(3)) == C(3)
		&& table.binomial(10, 3) == C(120) && atl::series_inverse(std::vector<C>{1, -1}, 129) == std::vector<C>(129, 1)
		&& atl::square_root_mod(C(4)) == C(2);
}

inline bool split_scalar_map() {
	atl::flat_hash_map<int, int> map;
	map[2] = 7;
	return map.at(2) == 7;
}

inline bool split_geometry() {
	std::array const points{atl::geometry::point{0, 0}, atl::geometry::point{4, 0}, atl::geometry::point{0, 4}};
	return atl::geometry::triangulation_of(points).triangles.size() == 1;
}
