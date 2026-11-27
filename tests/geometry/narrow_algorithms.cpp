#include <atl/geometry/calipers_core.hpp>
#include <atl/geometry/convex_layers_core.hpp>
#include <atl/geometry/convex_polygon_operations_core.hpp>
#include <atl/geometry/halfplane_intersection_core.hpp>
#include <atl/geometry/ray_core.hpp>
#include <atl/geometry/triangulation_core.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <random>
#include <vector>

TEST_CASE("narrow integer algorithms preserve degenerate and constructed results", "[geometry]") {
	namespace geo = atl::geometry;
	std::array const square{geo::point{0, 0}, geo::point{4, 0}, geo::point{4, 4}, geo::point{0, 4}};
	std::array const adjacent{geo::point{4, 0}, geo::point{8, 0}, geo::point{8, 4}, geo::point{4, 4}};
	auto const contact = geo::intersection_of_convex_polygons(square, adjacent);
	REQUIRE(contact.size() == 2);
	REQUIRE(geo::polygon_area(contact) == 0);
	REQUIRE(geo::union_of_convex_polygons(square, adjacent).size() == 1);
	REQUIRE(geo::squared_distance(geo::farthest_pair(square)[0], geo::farthest_pair(square)[1]) == 32);
	REQUIRE(geo::triangulation_of(square).triangles.size() == 2);
	REQUIRE(geo::convex_layers(square).size() == 1);
	REQUIRE(geo::intersection(geo::circle<int>{{0, 0}, 5}, geo::ray<int>{{0, 0}, {1, 0}}).size() == 1);
	using hp = geo::halfplane<int>;
	std::array const constraints{hp::at_most(1, 0, 4), hp::at_most(-1, 0, 0), hp::at_most(0, 1, 4), hp::at_most(0, -1, 0)};
	REQUIRE(geo::halfplane_intersection(constraints).kind == geo::region_kind::polygon);
}

TEST_CASE("Minkowski difference keeps the support stream including parallel ties", "[geometry]") {
	namespace geo = atl::geometry;
	std::mt19937 rng(173);
	for (int trial = 0; trial < 100; ++trial) {
		std::vector<geo::point<int>> a, b;
		for (int i = 0; i < 1 + trial % 13; ++i) a.push_back({int(rng() % 21) - 10, int(rng() % 21) - 10});
		for (int i = 0; i < 1 + trial % 7; ++i) b.push_back({int(rng() % 21) - 10, int(rng() % 21) - 10});
		auto const d = geo::detail::difference_of(a, b, geo::exact{});
		std::size_t i = 0;
		for (auto const event : geo::dual_support_view(d.a_hull, d.negated_b_hull)) {
			REQUIRE(i < d.boundary.size());
			REQUIRE(d.supports[i] == std::array{event.p_support, event.q_support});
			REQUIRE(d.boundary[i] == d.a_hull[event.p_support] + d.negated_b_hull[event.q_support]);
			++i;
		}
		REQUIRE(d.boundary.size() == (i == 0 ? 1 : i));
	}
}
