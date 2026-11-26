#include <atl/geometry/convex_polygon_operations_core.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon_core.hpp>
#include <array>
#include <iostream>

int main() {
	using atl::geometry::point;
	std::array const a{point{0, 0}, point{4, 0}, point{4, 4}, point{0, 4}};
	std::array const b{point{2, 0}, point{6, 0}, point{6, 4}, point{2, 4}};
	auto const common = atl::geometry::intersection_of_convex_polygons(a, b);
	auto const parts = atl::geometry::union_of_convex_polygons(a, b);
	std::cout << "intersection area: " << atl::geometry::polygon_area(common) << '\n';
	std::cout << "union components: " << parts.size() << '\n';
}
