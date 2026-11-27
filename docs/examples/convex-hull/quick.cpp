#include <atl/geometry/convex_hull_core.hpp>
#include <atl/geometry/convex_layers_core.hpp>
#include <array>
#include <iostream>

int main() {
	namespace geo = atl::geometry;
	std::array const points{geo::point{0, 0}, geo::point{4, 0}, geo::point{4, 4}, geo::point{0, 4}, geo::point{2, 2}, geo::point{2, 0}};
	auto const hull = geo::convex_hull(points);
	for (auto const p : hull) std::cout << p.x << ' ' << p.y << '\n';
	std::cout << "boundary vertices: " << geo::convex_hull(points, geo::keep_collinear).size() << '\n';
	std::cout << "layers: " << geo::convex_layers(points).size() << '\n';
	std::cout << "centre inside: " << (geo::convex_contains(hull, {2, 2}) == geo::containment::inside) << '\n';
}
