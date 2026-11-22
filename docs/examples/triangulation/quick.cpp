#include <atl/geometry/triangulation.hpp>
#include <array>
#include <iostream>

int main() {
	namespace geo = atl::geometry;
	std::array const polygon{geo::point{0, 0}, geo::point{4, 0}, geo::point{4, 4}, geo::point{0, 4}};
	auto const result = geo::triangulation_of(polygon);
	std::cout << result.triangles.size() << ' ' << result.dual_edges.size() << '\n';
	constexpr auto ear_count = [] {
		std::array const triangle{geo::point{0, 0}, geo::point{4, 0}, geo::point{0, 4}};
		return geo::triangulation_of_by_ear_clipping(triangle).triangles.size();
	}();
	static_assert(ear_count == 1);
}
