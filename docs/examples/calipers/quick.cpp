#include <atl/geometry/calipers.hpp>
#include <array>
#include <iostream>

int main() {
	namespace geo = atl::geometry;
	std::array const square{geo::point{0, 0}, geo::point{4, 0}, geo::point{4, 4}, geo::point{0, 4}};
	auto const pair = geo::farthest_pair(square);
	std::cout << "diameter squared: " << static_cast<long long>(geo::squared_distance(pair[0], pair[1])) << '\n';
	unsigned events = 0;
	for ([[maybe_unused]] auto const event : geo::antipodal_view(square)) ++events;
	std::cout << "antipodal events: " << events << '\n';
	std::array const other{geo::point{7, 1}, geo::point{9, 1}, geo::point{9, 3}, geo::point{7, 3}};
	std::cout << "hull distance: " << geo::distance_between_convex_hulls(square, other) << '\n';
}
