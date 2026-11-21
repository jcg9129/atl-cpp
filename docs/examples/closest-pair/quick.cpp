#include <atl/geometry/closest_pair.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <iostream>
#include <vector>

int main() {
	namespace geo = atl::geometry;
	// int64 coordinates near 2^61: the squared distances are exact int128 values
	atl::int64 const far = atl::int64(1) << 61;
	std::vector<geo::point<atl::int64>> const points{{0, 0}, {10, 0}, {far, far}, {far + 3, far + 4}, {-far, 5}};
	auto const [i, j] = geo::closest_pair_indices(points);
	std::cout << "closest: " << (i < j ? i : j) << ' ' << (i < j ? j : i) << ", squared distance "
			  << atl::int64(geo::squared_distance(points[i], points[j])) << '\n';
	auto const [p, q] = geo::closest_pair(std::vector<geo::point<double>>{{0.5, 0.5}, {2.0, 2.0}, {0.75, 0.25}});
	std::cout << "distance " << geo::distance(p, q) << '\n';
}
