#include <atl/geometry/halfplane_intersection_core.hpp>
#include <atl/geometry/point_location_integral.hpp>
#include <atl/geometry/polygon_overlay_integral.hpp>

// Instantiate the integral policy before installing the floating extension.
static_assert(atl::geometry::orientation(atl::geometry::point{0, 0}, atl::geometry::point{2, 0}, atl::geometry::point{0, 3}) == 1);

static_assert(atl::geometry::halfplane<int>::at_most(1, 0, 3).contains({2, 0}) == atl::geometry::containment::inside);

#include <atl/geometry/halfplane_intersection.hpp>
#include <atl/geometry/point_location.hpp>
#include <atl/geometry/polygon_overlay.hpp>
#include <atl/geometry/predicates_floating.hpp>
#include <array>
#include <catch2/catch.hpp>

TEST_CASE("geometry headers: integral surfaces and later floating extension", "[geometry][predicates]") {
	namespace geo = atl::geometry;
	std::array const square{geo::point{0, 0}, geo::point{4, 0}, geo::point{4, 4}, geo::point{0, 4}};
	auto const overlay = geo::overlay_of(square, square);
	REQUIRE(geo::intersection_area(overlay) == 16);
	REQUIRE(geo::orientation(geo::point{0.0, 0.0}, geo::point{2.0, 0.0}, geo::point{0.0, 3.0}) == 1);
}
