#include <atl/geometry/line_core.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/segment_core.hpp>
#include <atl/types/alias.hpp>
#include <iostream>

int main() {
	namespace geo = atl::geometry;
	using segment = geo::segment<int>;
	// closed segments: a collinear overlap comes back exactly, as input points
	auto const overlap = geo::intersection(segment{{0, 0}, {6, 0}}, segment{{8, 0}, {4, 0}});
	std::cout << (overlap.kind == geo::intersection_kind::along_segment) << ' ' << overlap.along.a.x << ".."
			  << overlap.along.b.x << '\n';
	// a proper crossing is a double construction
	auto const crossing = geo::intersection(segment{{0, 0}, {4, 4}}, segment{{0, 3}, {3, 0}});
	std::cout << crossing.at.x << ' ' << crossing.at.y << '\n';
	std::cout << geo::intersects(segment{{0, 0}, {2, 0}}, segment{{2, 0}, {2, 5}}) << ' '
			  << geo::distance(segment{{0, 0}, {2, 0}}, segment{{5, 4}, {9, 9}}) << '\n';

	// a directed line: left is the positive side of A x + B y + C
	geo::line<int> const l{{0, 0}, {4, 2}};
	auto const [a, b, c] = geo::line_coefficients(l);
	std::cout << atl::int64(a) << ' ' << atl::int64(b) << ' ' << atl::int64(c) << ' '
			  << (geo::side_of(l, geo::point{1, 3}) == geo::side::left) << '\n';
	geo::point<double> const foot = geo::projection(l, geo::point{0, 5});
	std::cout << foot.x << ' ' << foot.y << ' ' << geo::distance(l, geo::point{0, 5}) << '\n';
}
