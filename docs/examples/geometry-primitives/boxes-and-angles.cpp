#include <atl/geometry/angular_sort_core.hpp>
#include <atl/geometry/box.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <iostream>
#include <vector>

int main() {
	namespace geo = atl::geometry;
	// the area covered by at least one box, exact in the widened integer type
	std::vector<geo::box<int>> const boxes{{{0, 0}, {4, 4}}, {{2, 2}, {6, 6}}, {{10, 0}, {11, 1}}};
	std::cout << atl::int64(geo::union_area(boxes)) << '\n';

	// atan2 order from signs only; same-ray points keep their input order
	std::vector<geo::point<int>> const directions{{0, 1}, {-1, 0}, {2, 2}, {1, 0}, {0, -1}, {1, 1}, {0, 0}};
	std::cout << "order:";
	for (auto const i : geo::polar_angle_order(directions)) std::cout << ' ' << i;
	std::cout << '\n';
}
