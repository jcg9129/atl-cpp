#include <atl/geometry/point_location_integral.hpp>
#include <atl/geometry/polygon_overlay_integral.hpp>
#include <array>
#include <catch2/catch.hpp>

TEST_CASE("integral overlay with only narrow geometry headers", "[geometry]") {
	namespace geo = atl::geometry;
	std::array const a{geo::point{0, 0}, geo::point{4, 0}, geo::point{4, 4}, geo::point{0, 4}};
	std::array const b{geo::point{2, 0}, geo::point{6, 0}, geo::point{6, 4}, geo::point{2, 4}};
	auto const overlay = geo::overlay_of(a, b);
	REQUIRE(geo::intersection_area(overlay) == 8);
	REQUIRE(geo::union_area(overlay) == 24);
}
