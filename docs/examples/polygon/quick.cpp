#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon_core.hpp>
#include <atl/types/alias.hpp>
#include <iostream>
#include <vector>

int main() {
	namespace geo = atl::geometry;
	// an L-shaped polygon, clockwise, with a straight vertex at (2, 0)
	std::vector<geo::point<int>> const l_shape{{0, 0}, {0, 6}, {2, 6}, {2, 2}, {4, 2}, {4, 0}, {2, 0}};
	std::cout << "doubled signed area " << atl::int64(geo::polygon_area2(l_shape)) << ", area " << geo::polygon_area(l_shape) << '\n';
	std::cout << "winding " << int(geo::winding_of(l_shape)) << ", convex " << geo::convexity_of(l_shape).is_convex << '\n';
	// containment is exact, boundary included
	for (geo::point<int> const q : {geo::point{1, 1}, geo::point{2, 4}, geo::point{3, 3}})
		std::cout << '(' << q.x << ", " << q.y << ") -> " << int(geo::contains(l_shape, q)) << '\n';
	auto const centroid = geo::polygon_centroid(l_shape);
	std::cout << "centroid " << centroid.x << ' ' << centroid.y << '\n';
	auto const lattice = geo::lattice_count_of(l_shape);
	std::cout << "lattice points: " << atl::int64(lattice.boundary) << " on the boundary, " << atl::int64(lattice.interior) << " inside\n";
	std::cout << "normalized:";
	for (auto const v : geo::normalized_polygon(l_shape)) std::cout << " (" << v.x << ", " << v.y << ')';
	std::cout << '\n';
}
