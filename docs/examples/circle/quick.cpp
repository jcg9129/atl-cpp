#include <atl/geometry/circle.hpp>
#include <atl/geometry/ray.hpp>
#include <iostream>

int main() {
	namespace geo = atl::geometry;
	geo::circle<int> const disk{{0, 0}, 5};
	geo::ray<int> const beam{{0, 0}, {1, 0}};
	std::cout << (geo::contains(disk, geo::point{3, 4}) == geo::containment::boundary) << '\n';
	std::cout << geo::intersection(disk, beam).size() << '\n';
	std::cout << (geo::intersection(beam, geo::segment<int>{{-2, 0}, {4, 0}}).kind == geo::intersection_kind::along_segment) << '\n';
}
