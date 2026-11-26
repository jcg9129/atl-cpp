#include <atl/geometry/arrangement_core.hpp>
#include <array>
#include <iostream>

int main() {
	namespace geo = atl::geometry;
	std::array const edges{geo::segment<int>{{0, 0}, {4, 0}}, geo::segment<int>{{4, 0}, {4, 4}}, geo::segment<int>{{4, 4}, {0, 4}}, geo::segment<int>{{0, 4}, {0, 0}}};
	auto const arr = geo::arrangement_of(edges);
	std::cout << arr.vertices.size() << ' ' << arr.edge_count() << ' ' << arr.faces.size() << '\n';
}
