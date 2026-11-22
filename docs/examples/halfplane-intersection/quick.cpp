#include <atl/geometry/halfplane.hpp>
#include <atl/geometry/halfplane_intersection.hpp>
#include <array>
#include <iostream>

int main() {
	using hp = atl::geometry::halfplane<int>;
	std::array const constraints{hp::at_most(1, 0, 2), hp::at_most(-1, 0, 0),
								 hp::at_most(0, 1, 3), hp::at_most(0, -1, 0)};
	auto const region = atl::geometry::halfplane_intersection(constraints);
	std::cout << "polygon: " << (region.kind == atl::geometry::region_kind::polygon) << '\n';
	std::cout << "corners: " << region.vertices.size() << '\n';
	for (auto const v : region.vertices) std::cout << v.x << ' ' << v.y << '\n';
}
