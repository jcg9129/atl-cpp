#include <atl/geometry/point.hpp>
#include <atl/geometry/radial_sweep.hpp>
#include <iostream>
#include <vector>

int main() {
	namespace geo = atl::geometry;
	// closed arcs of directions in radians; the second wraps through angle 0
	std::vector<geo::arc> const arcs{{0.0, 1.0}, {-0.5, 0.25}, {1.0, 2.0}, {3.0, 4.0}};
	auto const overlap = geo::max_arc_overlap(arcs);
	std::cout << overlap.count << " arcs share the direction " << overlap.angle << '\n';

	// the most points one disk of radius 2 can cover
	std::vector<geo::point<int>> const points{{0, 0}, {3, 0}, {0, 3}, {3, 3}, {10, 10}, {11, 10}};
	auto const cover = geo::max_points_in_circle(points, 2.0);
	auto const radius3 = geo::max_points_in_circle(points, 2.2);
	std::cout << "radius 2: " << cover.count << " points, radius 2.2: " << radius3.count << " points\n";
}
