#include <atl/geometry/polygon_overlay_integral.hpp>
#include <array>
#include <iostream>

int main() {
	namespace geo = atl::geometry;
	std::array const a{geo::point{0, 0}, geo::point{4, 0}, geo::point{4, 4}, geo::point{0, 4}};
	std::array const b{geo::point{2, 2}, geo::point{6, 2}, geo::point{6, 6}, geo::point{2, 6}};
	auto const overlay = geo::overlay_of(a, b);
	std::cout << geo::intersection_area(overlay) << ' ' << geo::union_area(overlay) << ' '
			  << geo::difference_area(overlay) << ' ' << geo::symmetric_difference_area(overlay) << '\n';
}
