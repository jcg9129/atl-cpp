#include <atl/geometry/delaunay.hpp>
#include <array>
#include <iostream>

int main() {
	namespace geo = atl::geometry;
	std::array const points{geo::point{0, 0}, geo::point{4, 0}, geo::point{0, 4}};
	auto const triangles = geo::delaunay(points);
	std::cout << triangles.triangles.size() << '\n';
	std::cout << geo::voronoi_from_delaunay(points, triangles).edges.size() << '\n';
	std::cout << geo::euclidean_mst(points).size() << '\n';
}
