#include <atl/experimental/geometry/box_variants.hpp>
#include <atl/experimental/geometry/closest_pair_variants.hpp>
#include <atl/experimental/geometry/manhattan_mst_variants.hpp>
#include <atl/geometry/angular_sort.hpp>
#include <atl/geometry/box.hpp>
#include <atl/geometry/closest_pair.hpp>
#include <atl/geometry/line.hpp>
#include <atl/geometry/manhattan_mst.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/radial_sweep.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <span>
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
