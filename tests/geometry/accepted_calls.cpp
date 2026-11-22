#include <atl/experimental/geometry/arrangement_all_pairs.hpp>
#include <atl/experimental/geometry/box_variants.hpp>
#include <atl/experimental/geometry/closest_pair_variants.hpp>
#include <atl/experimental/geometry/convex_hull_variants.hpp>
#include <atl/experimental/geometry/halfplane_intersection_variants.hpp>
#include <atl/experimental/geometry/manhattan_mst_variants.hpp>
#include <atl/experimental/geometry/trapezoid_point_location.hpp>
#include <atl/geometry/angular_sort.hpp>
#include <atl/geometry/box.hpp>
#include <atl/geometry/calipers.hpp>
#include <atl/geometry/circle.hpp>
#include <atl/geometry/closest_pair.hpp>
#include <atl/geometry/convex_layers.hpp>
#include <atl/geometry/convex_polygon_operations.hpp>
#include <atl/geometry/delaunay.hpp>
#include <atl/geometry/halfplane.hpp>
#include <atl/geometry/halfplane_intersection.hpp>
#include <atl/geometry/line.hpp>
#include <atl/geometry/manhattan_mst.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/point_location.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/polygon_overlay.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/radial_sweep.hpp>
#include <atl/geometry/ray.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/geometry/triangulation.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// a named namespace: unused members of an unnamed one draw unneeded-declaration warnings
namespace geometry_accepted_values {
	/**
	 * @brief A user sign policy: exact signs through the library policy, its
	 *        scalar `sign` a template over any ordered value, as the concept
	 *        requires.
	 */
	struct counting_policy {
		int offset = 0;

		template<atl::geometry::coordinate T>
		[[nodiscard]] constexpr int orientation(atl::geometry::point<T> const o, atl::geometry::point<T> const a,
												atl::geometry::point<T> const b) const {
			return atl::geometry::exact{}.orientation(o, a, b) + offset;
		}

		template<atl::geometry::coordinate T>
		[[nodiscard]] constexpr int cross_sign(atl::geometry::point<T> const a, atl::geometry::point<T> const b,
											   atl::geometry::point<T> const c, atl::geometry::point<T> const d) const {
			return atl::geometry::exact{}.cross_sign(a, b, c, d) + offset;
		}

		template<typename V>
		[[nodiscard]] constexpr int sign(V const v) const {
			return atl::geometry::exact{}.sign(v) + offset;
		}

		template<atl::geometry::coordinate T>
		[[nodiscard]] constexpr bool equal(atl::geometry::point<T> const a, atl::geometry::point<T> const b) const {
			return a == b;
		}
	};

	/** @brief A policy deriving from the library one, overriding nothing. */
	struct derived_policy : atl::geometry::exact {};
}

namespace {
	using atl::geometry::point;
	using geometry_accepted_values::counting_policy;
	using geometry_accepted_values::derived_policy;

	/** @brief Every predicate and construction of the primitives headers on coordinate type `T`. */
	template<atl::geometry::coordinate T, atl::geometry::sign_policy<T> Policy>
	void exercise_primitives(Policy const policy) {
		point<T> const o{T(0), T(0)}, a{T(4), T(0)}, b{T(0), T(4)}, c{T(1), T(1)};
		REQUIRE(atl::geometry::orientation(o, a, b, policy) == 1);
		REQUIRE(policy.cross_sign(o, a, o, b) == 1);
		REQUIRE(atl::geometry::cross(a, b) == atl::geometry::product_t<T>(16));
		REQUIRE(atl::geometry::cross(o, a, b) == atl::geometry::difference_product_t<T>(16));
		REQUIRE(atl::geometry::dot(a, b) == atl::geometry::product_t<T>(0));
		REQUIRE(atl::geometry::squared_length(a) == atl::geometry::product_t<T>(16));
		REQUIRE(atl::geometry::squared_distance(a, b) == atl::geometry::difference_product_t<T>(32));
		REQUIRE(atl::geometry::distance(a, b) > 5.6);
		REQUIRE(atl::geometry::length(a) == 4.0);
		REQUIRE(atl::geometry::to_float64(c) == point<double>{1.0, 1.0});
		REQUIRE(atl::geometry::perp(a) == point<T>{T(0), T(4)});
		REQUIRE(a - c + c == a);

		atl::geometry::segment<T> const s{a, b};
		REQUIRE(atl::geometry::contains(s, point<T>{T(2), T(2)}, policy));
		REQUIRE(atl::geometry::intersects(s, atl::geometry::segment<T>{o, c}, policy) == false);
		REQUIRE(atl::geometry::intersects(s, atl::geometry::segment<T>{o, point<T>{T(3), T(3)}}, policy));
		REQUIRE(atl::geometry::direction(s) == point<T>{T(-4), T(4)});
		REQUIRE(atl::geometry::squared_length(s) == atl::geometry::difference_product_t<T>(32));
		REQUIRE(atl::geometry::length(s) > 5.6);
		REQUIRE(atl::geometry::distance(s, o) > 2.8);
		REQUIRE(atl::geometry::distance(s, atl::geometry::segment<T>{o, c}) > 1.4);

		atl::geometry::line<T> const l{o, a};
		REQUIRE(atl::geometry::side_of(l, b, policy) == atl::geometry::side::left);
		REQUIRE(atl::geometry::side_of(s, o, policy) == atl::geometry::side::left);
		REQUIRE(atl::geometry::contains(l, point<T>{T(-3), T(0)}, policy));
		REQUIRE(atl::geometry::parallel(l, atl::geometry::line<T>{b, point<T>{T(1), T(4)}}, policy));
		REQUIRE(atl::geometry::same_line(l, atl::geometry::line<T>{a, o}, policy));
		REQUIRE(atl::geometry::intersects(l, atl::geometry::line<T>{o, b}, policy));
		REQUIRE(atl::geometry::intersects(l, s, policy));
		REQUIRE(atl::geometry::intersects(s, l, policy));
		REQUIRE(atl::geometry::intersection(s, atl::geometry::segment<T>{o, point<T>{T(3), T(3)}}, policy).kind
				== atl::geometry::intersection_kind::at_point);
		REQUIRE(atl::geometry::intersection(l, s, policy).at == point<double>{4.0, 0.0});
		REQUIRE(atl::geometry::intersection(s, l, policy).at == point<double>{4.0, 0.0});
		REQUIRE(atl::geometry::intersection(l, atl::geometry::line<T>{c, point<T>{T(1), T(3)}}) == point<double>{1.0, 0.0});
		REQUIRE(atl::geometry::projection(l, c) == point<double>{1.0, 0.0});
		REQUIRE(atl::geometry::distance(l, c) == 1.0);
		REQUIRE(atl::geometry::distance(l, atl::geometry::line<T>{b, point<T>{T(1), T(4)}}) == 4.0);
		REQUIRE(atl::geometry::distance(l, s) == 0.0);
		REQUIRE(atl::geometry::distance(s, l) == 0.0);
		auto const coefficients = atl::geometry::line_coefficients(l);
		REQUIRE(atl::geometry::line_from_coefficients(coefficients) == atl::geometry::line<double>{{0.0, 0.0}, {4.0, 0.0}});

		std::vector<point<T>> points{b, a, c, o};
		atl::geometry::sort_by_polar_angle(points, policy);
		REQUIRE(atl::geometry::polar_angle_order(points, policy).size() == 4);
		REQUIRE(atl::geometry::polar_angle_less<Policy>{policy}(a, b));
	}
}

TEST_CASE("geometry primitives: every coordinate type and policy form", "[geometry][accepted]") {
	// narrow integers promote in arithmetic; the headers convert back
	exercise_primitives<short>(atl::geometry::exact{});
	exercise_primitives<atl::int16>(atl::geometry::fast{});
	exercise_primitives<atl::int32>(atl::geometry::approx{0.5});
	exercise_primitives<atl::int64>(counting_policy{});
	exercise_primitives<long>(atl::geometry::exact{});
	exercise_primitives<long long>(derived_policy{});
	exercise_primitives<float>(atl::geometry::exact{});
	exercise_primitives<double>(atl::geometry::fast{});
	exercise_primitives<double>(counting_policy{});
}

TEST_CASE("angular sort: accepted ranges and comparator uses", "[geometry][accepted]") {
	std::vector<point<int>> pts{{0, 1}, {1, 0}, {-1, 0}, {0, -1}};
	std::vector<atl::geometry::vertex_index> const expected{3, 1, 0, 2};
	// const and non-const lvalues, rvalues, arrays, spans and views
	std::vector<point<int>> const fixed = pts;
	REQUIRE(atl::geometry::polar_angle_order(fixed) == expected);
	REQUIRE(atl::geometry::polar_angle_order(std::vector<point<int>>(pts)) == expected);
	std::array<point<int>, 4> arr{pts[0], pts[1], pts[2], pts[3]};
	REQUIRE(atl::geometry::polar_angle_order(arr) == expected);
	REQUIRE(atl::geometry::polar_angle_order(std::span<point<int> const>(pts)) == expected);
	auto const negated = pts | std::views::transform([](point<int> const q) { return -q; });
	REQUIRE(atl::geometry::polar_angle_order(negated) == std::vector<atl::geometry::vertex_index>{0, 2, 3, 1});
	atl::geometry::sort_by_polar_angle(std::span<point<int>>(arr));
	REQUIRE(arr[0] == point{0, -1});
	atl::geometry::sort_by_polar_angle(pts | std::views::take(2));
	REQUIRE(pts[0] == point{1, 0});
	atl::geometry::sort_by_polar_angle(pts, atl::geometry::approx{0.0});
	REQUIRE(pts[0] == point{0, -1});
	// the comparator in standard algorithms, with a deduced policy
	std::ranges::stable_sort(pts, atl::geometry::polar_angle_less{atl::geometry::fast{}});
	REQUIRE(pts[3] == point{-1, 0});
}

TEST_CASE("union_area: accepted box ranges", "[geometry][accepted]") {
	using atl::geometry::box;
	std::vector<box<int>> boxes{{{0, 0}, {2, 2}}, {{1, 1}, {3, 3}}};
	std::vector<box<int>> const fixed = boxes;
	REQUIRE(atl::geometry::union_area(boxes) == atl::int128(7));
	REQUIRE(atl::geometry::union_area(fixed) == atl::int128(7));
	REQUIRE(atl::geometry::union_area(std::span<box<int> const>(fixed)) == atl::int128(7));
	REQUIRE(atl::geometry::union_area(std::array{box<atl::int16>{{0, 0}, {2, 2}}}) == atl::int64(4));
	REQUIRE(atl::geometry::union_area(boxes | std::views::take(1)) == atl::int128(4));
	REQUIRE(atl::experimental::union_area_by_top_down_tree(fixed) == atl::int128(7));
	REQUIRE(atl::geometry::union_area(std::vector<box<float>>{{{0.0f, 0.0f}, {0.5f, 2.0f}}}) == 1.0);
}

TEST_CASE("polygons and point sets: accepted ranges", "[geometry][accepted]") {
	std::vector<point<int>> pts{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
	std::vector<point<int>> const fixed = pts;
	std::array<point<int>, 4> const arr{pts[0], pts[1], pts[2], pts[3]};
	auto const shifted = pts | std::views::transform([](point<int> const q) { return q + point{1, 1}; });
	REQUIRE(atl::geometry::polygon_area2(fixed) == atl::int128(32));
	REQUIRE(atl::geometry::polygon_area2(std::vector<point<int>>(pts)) == atl::int128(32));
	REQUIRE(atl::geometry::polygon_area2(arr) == atl::int128(32));
	REQUIRE(atl::geometry::polygon_area2(std::span<point<int> const>(fixed)) == atl::int128(32));
	REQUIRE(atl::geometry::polygon_area2(shifted) == atl::int128(32));
	REQUIRE(atl::geometry::contains(shifted, point{1, 1}) == atl::geometry::containment::boundary);
	REQUIRE(atl::geometry::contains(pts, point{2, 2}, atl::geometry::exact{}) == atl::geometry::containment::inside);
	REQUIRE(atl::geometry::winding_of(shifted) == atl::geometry::winding::counterclockwise);
	REQUIRE(atl::geometry::convexity_of(arr).is_strict);
	REQUIRE(atl::geometry::normalized_polygon(shifted).size() == 4);
	REQUIRE(atl::geometry::lattice_count_of(std::span<point<int> const>(fixed)).boundary == 16);
	REQUIRE(atl::geometry::polygon_centroid(arr) == point{2.0, 2.0});
	REQUIRE(atl::geometry::closest_pair_indices(shifted)[0] != atl::geometry::closest_pair_indices(shifted)[1]);
	REQUIRE(atl::geometry::closest_pair(arr)[0] != atl::geometry::closest_pair(arr)[1]);
	REQUIRE(atl::geometry::manhattan_mst(std::span<point<int> const>(fixed)).size() == 3);
	REQUIRE(atl::geometry::manhattan_mst(shifted).size() == 3);
	REQUIRE(atl::geometry::max_points_in_circle(shifted, 2.0).count == 2);
	REQUIRE(atl::geometry::max_points_in_circle(std::vector<point<int>>(pts), 3.0).count == 4);
	std::vector<atl::geometry::arc> arcs{{0.0, 1.0}, {0.5, 2.0}};
	std::vector<atl::geometry::arc> const fixed_arcs = arcs;
	REQUIRE(atl::geometry::max_arc_overlap(arcs).count == 2);
	REQUIRE(atl::geometry::max_arc_overlap(fixed_arcs).count == 2);
	REQUIRE(atl::geometry::max_arc_overlap(std::span<atl::geometry::arc const>(fixed_arcs)).count == 2);
	REQUIRE(atl::geometry::max_arc_overlap(std::vector<atl::geometry::arc>{{1.0, 1.0}}).count == 1);
	auto const turned = arcs | std::views::transform([](atl::geometry::arc const a) { return atl::geometry::arc{a.from + 1, a.to + 1}; });
	REQUIRE(atl::geometry::max_arc_overlap(turned).count == 2);
}

namespace geometry_accepted_values {
	struct vertex_proxy {
		point<int> value;

		operator point<int>() const { return value; } // NOLINT(google-explicit-constructor): range reference conversion
	};

	struct proxy_iterator {
		using value_type = point<int>;
		using difference_type = std::ptrdiff_t;
		using iterator_concept = std::random_access_iterator_tag;
		point<int> const *position{};

		vertex_proxy operator*() const { return {*position}; }

		vertex_proxy operator[](difference_type offset) const { return {position[offset]}; }

		proxy_iterator &operator++() {
			++position;
			return *this;
		}

		proxy_iterator operator++(int) {
			auto previous = *this;
			++*this;
			return previous;
		}

		proxy_iterator &operator--() {
			--position;
			return *this;
		}

		proxy_iterator operator--(int) {
			auto previous = *this;
			--*this;
			return previous;
		}

		proxy_iterator &operator+=(difference_type offset) {
			position += offset;
			return *this;
		}

		proxy_iterator &operator-=(difference_type offset) {
			position -= offset;
			return *this;
		}

		friend proxy_iterator operator+(proxy_iterator it, difference_type offset) { return it += offset; }

		friend proxy_iterator operator+(difference_type offset, proxy_iterator it) { return it += offset; }

		friend proxy_iterator operator-(proxy_iterator it, difference_type offset) { return it -= offset; }

		friend difference_type operator-(proxy_iterator a, proxy_iterator b) { return a.position - b.position; }

		friend auto operator<=>(proxy_iterator, proxy_iterator) = default;
	};
}

TEST_CASE("polygons: proxy references materialize before point calls", "[geometry][accepted]") {
	std::array const square{point{0, 0}, point{4, 0}, point{4, 4}, point{0, 4}};
	using geometry_accepted_values::proxy_iterator;
	auto const poly = std::ranges::subrange(proxy_iterator{square.data()}, proxy_iterator{square.data() + square.size()});
	STATIC_REQUIRE(atl::geometry::vertex_range<decltype(poly)>);
	REQUIRE(atl::geometry::polygon_area2(poly) == 32);
	REQUIRE(atl::geometry::polygon_area(poly) == 16);
	REQUIRE(atl::geometry::polygon_centroid(poly) == point{2.0, 2.0});
	REQUIRE(atl::geometry::lattice_count_of(poly).boundary == 16);
	REQUIRE(atl::geometry::winding_of(poly) == atl::geometry::winding::counterclockwise);
	REQUIRE(atl::geometry::convexity_of(poly).is_strict);
	REQUIRE(atl::geometry::normalized_polygon(poly) == std::vector(square.begin(), square.end()));
	REQUIRE(atl::geometry::contains(poly, {1, 2}) == atl::geometry::containment::inside);
	std::array const poly64{point<atl::int64>{0, 0}, point<atl::int64>{4, 0}, point<atl::int64>{0, 4}};
	REQUIRE(atl::geometry::contains(poly64, point{1, 2}) == atl::geometry::containment::inside);
	REQUIRE(atl::geometry::contains(poly64, {1, 2}) == atl::geometry::containment::inside);
}

namespace geometry_hull_accepted {
	template<atl::geometry::vertex_range R, atl::geometry::sign_policy<atl::geometry::vertex_coordinate_t<R>> Policy>
	void exercise(R &&points, Policy const policy) {
		using namespace atl::geometry;
		using T = vertex_coordinate_t<R>;
		auto const ids = convex_hull_indices(points, policy);
		auto const hull = convex_hull(points, policy);
		REQUIRE(ids.size() == 4);
		REQUIRE(convex_hull_indices(points, keep_collinear, policy) == ids);
		REQUIRE(convex_hull(points, keep_collinear, policy) == hull);
		REQUIRE(convex_contains(points, {T(1), T(2)}, policy) == containment::inside);
		REQUIRE(merge_convex_hulls(points, hull, policy) == hull);
		REQUIRE(minkowski_sum(points, hull, policy).size() == 4);
		REQUIRE(convex_layers(points, policy).size() == 1);
		REQUIRE(std::ranges::distance(antipodal_view(points, policy)) == 8);
		REQUIRE(std::ranges::distance(dual_support_view(points, hull, policy)) == 8);
		REQUIRE(std::ranges::distance(dual_support_view(points, hull, anti_parallel, policy)) == 8);
		auto const pair = farthest_pair_indices(points, policy);
		REQUIRE(pair[0] != pair[1]);
		REQUIRE(squared_distance(farthest_pair(points, policy)[0], farthest_pair(points, policy)[1]) == 32);
		REQUIRE(farthest_pair_between_indices(points, hull, policy)[0] < 4);
		REQUIRE(squared_distance(farthest_pair_between(points, hull, policy)[0], farthest_pair_between(points, hull, policy)[1]) == 32);
		REQUIRE(narrowest_strip(points, policy).edge_start != narrowest_strip(points, policy).edge_end);
		REQUIRE(distance_between_convex_hulls(points, hull, policy) == 0);
		REQUIRE(min_area_enclosing_rectangle(points, policy).size() == 4);
		REQUIRE(min_perimeter_enclosing_rectangle(points, policy).size() == 4);
	}

	template<atl::geometry::vertex_range R, atl::geometry::sign_policy<atl::geometry::vertex_coordinate_t<R>> Policy>
	void exercise_input(R &&points, Policy const policy) {
		REQUIRE(atl::geometry::convex_hull_indices(points, policy).size() == 4);
		REQUIRE(atl::geometry::convex_hull(points, atl::geometry::keep_collinear, policy).size() == 4);
		REQUIRE(atl::geometry::convex_layers(points, policy).size() == 1);
	}

	template<atl::geometry::vertex_range R>
	void exercise_contenders(R &&points) {
		using namespace atl::geometry;
		exact const policy{};
		auto const ids = convex_hull_indices(points);
		auto const hull = convex_hull(points);
		REQUIRE(atl::experimental::convex_hull_indices_by_graham_scan(points, policy) == ids);
		REQUIRE(atl::experimental::convex_hull_indices_by_jarvis_march(points, policy) == ids);
		REQUIRE(atl::experimental::convex_hull_indices_by_quickhull(points, policy) == ids);
		REQUIRE(atl::experimental::convex_hull_indices_by_chan(points, policy) == ids);
		REQUIRE(atl::experimental::merge_convex_hulls_by_streamed_chains(points, hull, policy) == hull);
		REQUIRE(atl::experimental::merge_convex_hulls_by_materialized_chains(points, hull, policy) == hull);
		REQUIRE(atl::experimental::merge_convex_hulls_by_copodal_walk(points, hull, policy) == hull);
		REQUIRE(atl::experimental::convex_hull_union_by_materialized_chains(points, hull, policy) == hull);
		REQUIRE(atl::experimental::convex_hull_union_by_copodal_walk(points, hull, policy) == hull);
	}

}

TEST_CASE("hulls and calipers: coordinate types, policies and range storage", "[geometry][accepted]") {
	using atl::geometry::point;
	std::array const square{point{0, 0}, point{4, 0}, point{4, 4}, point{0, 4}};
	geometry_hull_accepted::exercise_input(square, geometry_accepted_values::counting_policy{});
	geometry_hull_accepted::exercise_input(std::span(square), geometry_accepted_values::derived_policy{});
	geometry_hull_accepted::exercise_input(std::vector(square.begin(), square.end()), atl::geometry::exact{});
	geometry_hull_accepted::exercise(std::array{point<atl::int16>{0, 0}, point<atl::int16>{4, 0}, point<atl::int16>{4, 4}, point<atl::int16>{0, 4}}, atl::geometry::exact{});
	geometry_hull_accepted::exercise_input(std::array{point<float>{0, 0}, point<float>{4, 0}, point<float>{4, 4}, point<float>{0, 4}}, atl::geometry::fast{});
	geometry_hull_accepted::exercise_input(std::array{point<double>{0, 0}, point<double>{4, 0}, point<double>{4, 4}, point<double>{0, 4}}, atl::geometry::approx{1e-12});
	using geometry_accepted_values::proxy_iterator;
	auto const proxies = std::ranges::subrange(proxy_iterator{square.data()}, proxy_iterator{square.data() + square.size()});
	geometry_hull_accepted::exercise(proxies, atl::geometry::exact{});
	geometry_hull_accepted::exercise_contenders(proxies);
}

namespace geometry_operations_accepted {
	template<atl::geometry::vertex_range R, atl::geometry::sign_policy<atl::geometry::vertex_coordinate_t<R>> Policy>
	void exercise(R &&points, Policy const p) {
		using namespace atl::geometry;
		REQUIRE(intersection_of_convex_polygons(points, points, p).size() == 4);
		REQUIRE(union_of_convex_polygons(points, points, p).size() == 1);
		auto const closest = closest_points_between_convex_hulls(points, points, p);
		REQUIRE(std::hypot(closest[0].x - closest[1].x, closest[0].y - closest[1].y) < 1e-12);
	}

	template<atl::geometry::coordinate T>
	void exercise_halfplanes() {
		using namespace atl::geometry;
		using hp = halfplane<T>;
		std::array const hs{hp::at_most(1, 0, 4), hp::at_most(-1, 0, 0), hp::at_most(0, 1, 4), hp::left_of({0, 0}, {4, 0})};
		REQUIRE(hs[0].contains({T(1), T(1)}) == containment::inside);
		REQUIRE(contains(hs[0], {T(1), T(1)}, fast{}) == containment::inside);
		REQUIRE(contains(hs[0], {T(1), T(1)}, approx{1e-12}) == containment::inside);
		REQUIRE(hs[0].boundary().a != hs[0].boundary().b);
		REQUIRE(halfplane_sweep_less{}(hs[3], hs[0]));
		REQUIRE(halfplane_intersection(hs).kind == region_kind::polygon);
		REQUIRE(halfplane_intersection(std::span(hs), fast{}).kind == region_kind::polygon);
		REQUIRE(halfplane_intersection(std::vector(hs.begin(), hs.end())).kind == region_kind::polygon);
		auto values = hs | std::views::transform([](hp const &h) { return h; });
		REQUIRE(halfplane_intersection(values).kind == region_kind::polygon);
		REQUIRE((atl::experimental::halfplane_intersection_by_composition<detail::filtered_determinant, detail::deque_pass, atl::experimental::halfplane_sort::key>(values).kind == region_kind::polygon));
	}
}

TEST_CASE("convex operations and half-planes accept policies, narrow types and proxy ranges", "[geometry][accepted]") {
	using atl::geometry::point;
	std::array const square{point{0, 0}, point{4, 0}, point{4, 4}, point{0, 4}};
	geometry_operations_accepted::exercise(square, geometry_accepted_values::counting_policy{});
	geometry_operations_accepted::exercise(std::span(square), geometry_accepted_values::derived_policy{});
	geometry_operations_accepted::exercise(std::vector(square.begin(), square.end()), atl::geometry::exact{});
	geometry_operations_accepted::exercise(std::array{point<atl::int16>{0, 0}, point<atl::int16>{4, 0}, point<atl::int16>{4, 4}, point<atl::int16>{0, 4}}, atl::geometry::exact{});
	geometry_operations_accepted::exercise(std::array{point<float>{0, 0}, point<float>{4, 0}, point<float>{4, 4}, point<float>{0, 4}}, atl::geometry::fast{});
	geometry_operations_accepted::exercise(std::array{point<double>{0, 0}, point<double>{4, 0}, point<double>{4, 4}, point<double>{0, 4}}, atl::geometry::approx{1e-12});
	using geometry_accepted_values::proxy_iterator;
	auto const proxies = std::ranges::subrange(proxy_iterator{square.data()}, proxy_iterator{square.data() + square.size()});
	geometry_operations_accepted::exercise(proxies, atl::geometry::exact{});
	geometry_operations_accepted::exercise_halfplanes<atl::int16>();
	geometry_operations_accepted::exercise_halfplanes<int>();
	geometry_operations_accepted::exercise_halfplanes<atl::int64>();
	geometry_operations_accepted::exercise_halfplanes<float>();
	geometry_operations_accepted::exercise_halfplanes<double>();
}

namespace geometry_operations_accepted {
	struct halfplane_proxy {
		atl::geometry::halfplane<int> const *value;

		operator atl::geometry::halfplane<int>() const { return *value; } // NOLINT(google-explicit-constructor): proxy conversion
	};

	struct halfplane_proxy_iterator {
		using value_type = atl::geometry::halfplane<int>;
		using difference_type = std::ptrdiff_t;
		using iterator_concept = std::random_access_iterator_tag;
		value_type const *position{};

		halfplane_proxy operator*() const { return {position}; }

		halfplane_proxy operator[](difference_type i) const { return {position + i}; }

		halfplane_proxy_iterator &operator++() {
			++position;
			return *this;
		}

		halfplane_proxy_iterator operator++(int) {
			auto copy = *this;
			++*this;
			return copy;
		}

		halfplane_proxy_iterator &operator--() {
			--position;
			return *this;
		}

		halfplane_proxy_iterator operator--(int) {
			auto copy = *this;
			--*this;
			return copy;
		}

		halfplane_proxy_iterator &operator+=(difference_type i) {
			position += i;
			return *this;
		}

		halfplane_proxy_iterator &operator-=(difference_type i) {
			position -= i;
			return *this;
		}

		friend halfplane_proxy_iterator operator+(halfplane_proxy_iterator it, difference_type i) { return it += i; }

		friend halfplane_proxy_iterator operator+(difference_type i, halfplane_proxy_iterator it) { return it += i; }

		friend halfplane_proxy_iterator operator-(halfplane_proxy_iterator it, difference_type i) { return it -= i; }

		friend difference_type operator-(halfplane_proxy_iterator a, halfplane_proxy_iterator b) { return a.position - b.position; }

		friend auto operator<=>(halfplane_proxy_iterator, halfplane_proxy_iterator) = default;
	};
}

TEST_CASE("half-plane ranges materialize prvalue proxy references", "[geometry][accepted]") {
	using hp = atl::geometry::halfplane<int>;
	using it = geometry_operations_accepted::halfplane_proxy_iterator;
	std::array const h{hp{1, 0, 2}, hp{-1, 0, 0}, hp{0, 1, 2}, hp{0, -1, 0}};
	auto const proxies = std::ranges::subrange(it{h.data()}, it{h.data() + h.size()});
	STATIC_REQUIRE(atl::geometry::halfplane_range<decltype(proxies)>);
	CHECK(atl::geometry::halfplane_intersection(proxies).kind == atl::geometry::region_kind::polygon);
}

namespace circle_ray_delaunay_accepted {
	using namespace atl::geometry;

	struct widened_policy : exact {
		template<typename V>
			requires std::same_as<V, atl::int64> || std::same_as<V, atl::int128>
		constexpr int sign(V const value) const { return exact{}.sign(value); }
	};

	static_assert(sign_policy<widened_policy, atl::int32>);

	template<coordinate T>
	void check() {
		point<T> const q{T(0), T(0)}, a{T(4), T(0)}, b{T(0), T(4)};
		circle<T> const c{q, T(2)}, other{a, T(2)};
		line<T> const l{q, a};
		segment<T> const s{q, a};
		ray<T> const r{q, a}, second{b, q};
		geometry_accepted_values::counting_policy const policy{};
		(void)contains(c, q, policy);
		(void)relation(c, other, policy);
		(void)relation(c, l, policy);
		(void)relation(l, c, policy);
		(void)diametral_circle(q, a);
		(void)circumcircle(q, a, b);
		(void)intersection(c, l);
		(void)intersection(l, c);
		(void)intersection(c, s);
		(void)intersection(s, c);
		(void)intersection(c, other);
		(void)tangent_points(c, a, policy);
		(void)common_tangents(c, other);
		(void)direction(r);
		(void)side_of(r, b, policy);
		(void)contains(r, a, policy);
		(void)intersects(r, l, policy);
		(void)intersects(l, r, policy);
		(void)intersects(r, s, policy);
		(void)intersects(s, r, policy);
		(void)intersects(r, second, policy);
		(void)intersection(r, l, policy);
		(void)intersection(l, r, policy);
		(void)intersection(r, s, policy);
		(void)intersection(s, r, policy);
		(void)intersection(r, second, policy);
		(void)intersection(c, r);
		(void)intersection(r, c);
		(void)distance(r, b);
		(void)distance(r, l);
		(void)distance(l, r);
		(void)distance(r, s);
		(void)distance(s, r);
		(void)distance(r, second);
		std::array const points{q, a, b};
		std::span<point<T> const> const view(points);
		(void)intersection_area(c, view);
		(void)intersection_area(view, c);
		(void)minimum_enclosing_circle(view);
		auto const d = delaunay(view);
		(void)delaunay_edges(view);
		(void)voronoi_from_delaunay(view, d);
		(void)euclidean_mst(view);
		(void)closest_pair_between_indices(view, points);
		(void)closest_pair_between(points, view);
		(void)largest_empty_circle(view);
	}
}

TEST_CASE("circle ray and Delaunay accepted coordinate and range calls", "[geometry]") {
	REQUIRE(atl::geometry::relation(atl::geometry::circle<atl::int32>{{0, 0}, 2}, atl::geometry::circle<atl::int32>{{0, 0}, 2}, circle_ray_delaunay_accepted::widened_policy{}) == atl::geometry::circle_relation::coincident);
	circle_ray_delaunay_accepted::check<atl::int16>();
	circle_ray_delaunay_accepted::check<atl::int32>();
	circle_ray_delaunay_accepted::check<atl::int64>();
	circle_ray_delaunay_accepted::check<float>();
	circle_ray_delaunay_accepted::check<double>();
	auto values = std::views::iota(0, 3) | std::views::transform([](int i) { return atl::geometry::point{i, i}; });
	REQUIRE(atl::geometry::delaunay_edges(values).size() == 2);
}

TEST_CASE("Delaunay accepts non-const and non-common point views", "[geometry]") {
	auto values = std::views::iota(0, 4) | std::views::transform([offset = 0](int i) mutable { return atl::geometry::point<atl::int64>{i + offset, 0}; });
	REQUIRE(atl::geometry::delaunay_edges(values).size() == 3);
	std::array const storage{atl::geometry::point{0, 0}, atl::geometry::point{4, 0}, atl::geometry::point{2, 0}};
	auto non_common = std::ranges::subrange(std::counted_iterator(storage.begin(), 3), std::default_sentinel);
	REQUIRE(std::abs(atl::geometry::largest_empty_circle(non_common).radius - 1.0) < 1e-12);
}

namespace geometry_subdivision_accepted {
	template<atl::geometry::coordinate T>
	void check() {
		using namespace atl::geometry;
		std::array const polygon{point<T>{0, 0}, point<T>{4, 0}, point<T>{0, 4}};
		std::array const segments{segment<T>{{0, 0}, {4, 0}}, segment<T>{{4, 0}, {0, 4}}, segment<T>{{0, 4}, {0, 0}}};
		std::span<segment<T> const> const view(segments);
		segment_intersections_view lazy(view);
		for (auto const event : lazy) (void)event;
		(void)segment_intersections(view);
		REQUIRE(simplicity_of(polygon | std::views::reverse).is_simple);
		auto const arr = arrangement_of(view);
		point_location const index(arr, view);
		(void)index.location_of({1, 1});
		(void)locations_of(arr, view, polygon);
		auto const overlay = overlay_of(polygon, polygon | std::views::reverse);
		(void)intersection_area(overlay);
		(void)union_area(overlay);
		(void)difference_area(overlay);
		(void)symmetric_difference_area(overlay);
		(void)intersection_area(polygon, polygon);
		(void)union_area(polygon, polygon);
		(void)difference_area(polygon, polygon);
		(void)symmetric_difference_area(polygon, polygon);
		(void)triangulation_of(polygon, geometry_accepted_values::counting_policy{});
		if constexpr (std::signed_integral<T>) {
			(void)atl::experimental::arrangement_of_by_all_pairs(view);
			atl::experimental::point_location_by_trapezoid_map const map(arr, view);
			(void)map.location_of({1, 1});
		}
	}
}

TEST_CASE("subdivision APIs accept coordinate tiers and borrowed ranges", "[geometry]") {
	geometry_subdivision_accepted::check<atl::int16>();
}
