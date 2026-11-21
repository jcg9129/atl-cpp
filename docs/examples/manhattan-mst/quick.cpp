#include <atl/geometry/manhattan_mst.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
	namespace geo = atl::geometry;
	std::vector<geo::point<int>> const points{{0, 0}, {2, 1}, {5, 0}, {2, 4}, {2, 1}};
	atl::int64 total = 0;
	for (auto const [a, b] : geo::manhattan_mst(points)) {
		atl::int64 const length = std::abs(points[a].x - points[b].x) + std::abs(points[a].y - points[b].y);
		std::cout << "edge length " << length << '\n';
		total += length;
	}
	std::cout << "total " << total << '\n';
}
