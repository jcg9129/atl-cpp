#include <atl/geometry/segment_intersections.hpp>
#include <array>
#include <iostream>

int main() {
	namespace geo = atl::geometry;
	std::array const segments{geo::segment<int>{{0, 0}, {4, 4}}, geo::segment<int>{{0, 4}, {4, 0}}};
	auto const events = geo::segment_intersections(segments);
	std::cout << events.size() << ' ' << events[0].at.x << ' ' << events[0].at.y << '\n';
}
