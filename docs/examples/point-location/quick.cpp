#include <atl/geometry/point_location_integral.hpp>
#include <array>
#include <iostream>

int main() {
	namespace geo = atl::geometry;
	std::array const edges{geo::segment<int>{{0, 0}, {4, 0}}, geo::segment<int>{{4, 0}, {4, 4}}, geo::segment<int>{{4, 4}, {0, 4}}, geo::segment<int>{{0, 4}, {0, 0}}};
	auto const arr = geo::arrangement_of(edges);
	geo::point_location const index(arr, edges);
	std::cout << (index.location_of({0, 0}).kind == geo::location_kind::vertex) << ' '
			  << (index.location_of({2, 0}).kind == geo::location_kind::edge) << ' '
			  << (index.location_of({2, 2}).index != geo::arrangement::unbounded_face) << '\n';
}
