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
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstdint>
#include <list>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

// a named namespace: unused members of an unnamed one draw unneeded-declaration warnings
namespace geometry_boundary_values {
	using atl::geometry::point;

	/** @brief A policy whose scalar `sign` takes only the difference product type, not `product_t`. */
	struct difference_sign_only {
		template<atl::geometry::coordinate T>
		[[nodiscard]] constexpr int orientation(point<T>, point<T>, point<T>) const { return 0; }

		template<atl::geometry::coordinate T>
		[[nodiscard]] constexpr int cross_sign(point<T>, point<T>, point<T>, point<T>) const { return 0; }

		template<typename V>
			requires std::same_as<V, atl::int128>
		[[nodiscard]] constexpr int sign(V) const { return 0; }

		template<atl::geometry::coordinate T>
		[[nodiscard]] constexpr bool equal(point<T>, point<T>) const { return true; }
	};

	/** @brief A policy whose members are not const. */
	struct mutable_policy {
		template<atl::geometry::coordinate T>
		int orientation(point<T>, point<T>, point<T>) { return 0; }

		template<atl::geometry::coordinate T>
		int cross_sign(point<T>, point<T>, point<T>, point<T>) { return 0; }

		template<typename V>
		int sign(V) { return 0; }

		template<atl::geometry::coordinate T>
		bool equal(point<T>, point<T>) { return true; }
	};

	/** @brief A policy that cannot be copied. */
	struct pinned_policy : atl::geometry::exact {
		pinned_policy() = default;
		pinned_policy(pinned_policy const &) = delete;
		pinned_policy &operator=(pinned_policy const &) = delete;
	};

	/** @brief A policy whose orientation reports a `bool`, not an `int`. */
	struct boolean_orientation : atl::geometry::exact {
		template<atl::geometry::coordinate T>
		[[nodiscard]] constexpr bool orientation(point<T>, point<T>, point<T>) const { return true; }
	};

	template<typename R>
	concept polar_sortable = requires(R &&r) { atl::geometry::sort_by_polar_angle(std::forward<R>(r)); };

	template<typename R>
	concept polar_orderable = requires(R &&r) { atl::geometry::polar_angle_order(std::forward<R>(r)); };

	template<typename R, typename Policy>
	concept polar_orderable_with = requires(R &&r, Policy p) { atl::geometry::polar_angle_order(std::forward<R>(r), p); };

	template<typename R>
	concept box_measurable = requires(R &&r) { atl::geometry::union_area(std::forward<R>(r)); };

	template<typename R>
	concept box_measurable_by_contender = requires(R &&r) {
		atl::experimental::union_area_by_top_down_tree(std::forward<R>(r));
	};

	template<typename Policy, typename T>
	concept polar_comparable = requires(atl::geometry::polar_angle_less<Policy> const less, point<T> const p) { less(p, p); };

	template<typename Coefficient>
	concept line_buildable = requires(std::array<Coefficient, 3> const c) { atl::geometry::line_from_coefficients(c); };

	template<typename T, typename Policy>
	concept orientable = requires(point<T> const p, Policy const policy) { atl::geometry::orientation(p, p, p, policy); };

	template<typename R>
	concept area_measurable = requires(R &&r) { atl::geometry::polygon_area2(std::forward<R>(r)); };

	template<typename R>
	concept lattice_countable = requires(R &&r) { atl::geometry::lattice_count_of(std::forward<R>(r)); };

	template<typename R, typename Policy>
	concept convexity_testable = requires(R &&r, Policy p) { atl::geometry::convexity_of(std::forward<R>(r), p); };

	template<typename R, typename Q>
	concept locatable = requires(R &&r, Q q) { atl::geometry::contains(std::forward<R>(r), q); };

	template<typename R>
	concept arc_sweepable = requires(R &&r) { atl::geometry::max_arc_overlap(std::forward<R>(r)); };

	template<typename R>
	concept circle_coverable = requires(R &&r) { atl::geometry::max_points_in_circle(std::forward<R>(r), 1.0); };

	template<typename R>
	concept closest_pairable = requires(R &&r) { atl::geometry::closest_pair_indices(std::forward<R>(r)); };

	template<typename R>
	concept closest_pairable_by_contenders = requires(R &&r) {
		atl::experimental::closest_pair_indices_by_divide_and_conquer(std::forward<R>(r));
		atl::experimental::closest_pair_indices_by_randomized_grid(std::forward<R>(r));
	};

	template<typename R>
	concept l1_spannable = requires(R &&r) { atl::geometry::manhattan_mst(std::forward<R>(r)); };

	template<typename R>
	concept l1_spannable_by_contender = requires(R &&r) {
		atl::experimental::manhattan_mst_by_fenwick_prefix_minimum(std::forward<R>(r));
	};

	template<typename Policy>
	concept polar_comparator_type = requires { typename atl::geometry::polar_angle_less<Policy>; };

	// Each expression checks the corresponding coordinate boundary directly.
	template<typename T>
	constexpr bool rejects_character_calls() {
		using namespace atl::geometry;
		return !requires { typename point<T>; }
			&& !requires { cross(point<T>{}, point<T>{}); }
			&& !requires { cross(point<T>{}, point<T>{}, point<T>{}); }
			&& !requires { dot(point<T>{}, point<T>{}); }
			&& !requires { squared_length(point<T>{}); }
			&& !requires { squared_distance(point<T>{}, point<T>{}); }
			&& !requires { length(point<T>{}); }
			&& !requires { distance(point<T>{}, point<T>{}); }
			&& !requires { perp(point<T>{}); }
			&& !requires { to_float64(point<T>{}); }
			&& !requires { orientation(point<T>{}, point<T>{}, point<T>{}); }
			&& !requires { exact{}.equal(point<T>{}, point<T>{}); }
			&& !requires { fast{}.cross_sign(point<T>{}, point<T>{}, point<T>{}, point<T>{}); }
			&& !requires { approx{0}.equal(point<T>{}, point<T>{}); }
			&& !requires { typename segment<T>; }
			&& !requires { direction(segment<T>{}); }
			&& !requires { squared_length(segment<T>{}); }
			&& !requires { length(segment<T>{}); }
			&& !requires { contains(segment<T>{}, point<T>{}); }
			&& !requires { intersects(segment<T>{}, segment<T>{}); }
			&& !requires { distance(segment<T>{}, point<T>{}); }
			&& !requires { distance(segment<T>{}, segment<T>{}); }
			&& !requires { typename line<T>; }
			&& !requires { direction(line<T>{}); }
			&& !requires { side_of(line<T>{}, point<T>{}); }
			&& !requires { side_of(segment<T>{}, point<T>{}); }
			&& !requires { contains(line<T>{}, point<T>{}); }
			&& !requires { parallel(line<T>{}, line<T>{}); }
			&& !requires { same_line(line<T>{}, line<T>{}); }
			&& !requires { intersects(line<T>{}, line<T>{}); }
			&& !requires { intersects(line<T>{}, segment<T>{}); }
			&& !requires { intersects(segment<T>{}, line<T>{}); }
			&& !requires { line_coefficients(line<T>{}); }
			&& !requires { distance(line<T>{}, line<T>{}); }
			&& !requires { distance(line<T>{}, segment<T>{}); }
			&& !requires { distance(segment<T>{}, line<T>{}); }
			&& !requires { intersection(line<T>{}, line<T>{}); }
			&& !requires { intersection(line<T>{}, segment<T>{}); }
			&& !requires { intersection(segment<T>{}, line<T>{}); }
			&& !requires { intersection(segment<T>{}, segment<T>{}); }
			&& !requires { distance(line<T>{}, point<T>{}); }
			&& !requires { projection(line<T>{}, point<T>{}); }
			&& !requires { typename box<T>; }
			&& !requires(T value) { line_from_coefficients(value, value, value); }
			&& !requires { polar_angle_less{}(point<T>{}, point<T>{}); }
			&& !requires(std::array<point<T>, 3> values) { polygon_area2(values); }
			&& !requires(std::array<point<T>, 3> values) { polygon_area(values); }
			&& !requires(std::array<point<T>, 3> values) { polygon_centroid(values); }
			&& !requires(std::array<point<T>, 3> values) { contains(values, point<T>{}); }
			&& !requires(std::array<point<T>, 3> values) { convexity_of(values); }
			&& !requires(std::array<point<T>, 3> values) { winding_of(values); }
			&& !requires(std::array<point<T>, 3> values) { normalized_polygon(values); }
			&& !requires(std::array<point<T>, 3> values) { lattice_count_of(values); }
			&& !requires(std::array<point<T>, 3> values) { sort_by_polar_angle(values); }
			&& !requires(std::array<point<T>, 3> values) { polar_angle_order(values); }
			&& !requires(std::array<point<T>, 3> values) { closest_pair_indices(values); }
			&& !requires(std::array<point<T>, 3> values) { closest_pair(values); }
			&& !requires(std::array<point<T>, 3> values) { manhattan_mst(values); }
			&& !requires(std::array<point<T>, 3> values) { max_points_in_circle(values, 1.0); }
			&& !requires(std::array<box<T>, 1> values) { union_area(values); }
			&& !requires(std::array<box<T>, 1> values) { atl::experimental::union_area_by_top_down_tree(values); }
			&& !requires(std::array<point<T>, 3> values) { atl::experimental::closest_pair_indices_by_divide_and_conquer(values); }
			&& !requires(std::array<point<T>, 3> values) { atl::experimental::closest_pair_indices_by_randomized_grid(values); }
			&& !requires(std::array<point<T>, 3> values) { atl::experimental::manhattan_mst_by_fenwick_prefix_minimum(values); };
	}

	static_assert(rejects_character_calls<char>());
	static_assert(rejects_character_calls<char const>());
	static_assert(!atl::geometry::coordinate<wchar_t const volatile>);
	static_assert(rejects_character_calls<signed char>());
	static_assert(rejects_character_calls<unsigned char>());
	static_assert(rejects_character_calls<wchar_t>());
	static_assert(rejects_character_calls<char8_t>());
	static_assert(rejects_character_calls<char16_t>());
	static_assert(rejects_character_calls<char32_t>());
	static_assert(!polar_comparator_type<pinned_policy>);

	template<typename Count>
	concept lattice_count_type = requires { typename atl::geometry::lattice_count<Count>; };
}

namespace {
	using namespace geometry_boundary_values;
	using atl::geometry::box;
	using atl::geometry::coordinate;
	using atl::geometry::exact;
	using atl::geometry::sign_policy;

	// coordinates: signed integers up to 64 bits, float and double only
	static_assert(coordinate<atl::int16> && coordinate<atl::int64> && coordinate<float> && coordinate<double>);
	static_assert(!coordinate<atl::uint32>);
	static_assert(!coordinate<atl::uint64>);
	static_assert(!coordinate<bool>);
	static_assert(!coordinate<long double>);
	static_assert(!coordinate<atl::int128>);
	static_assert(!coordinate<point<int>>);

	// the policy concept: every member, const, int results, copyable, both product types
	static_assert(sign_policy<exact, int> && sign_policy<atl::geometry::fast, double> && sign_policy<atl::geometry::approx, atl::int64>);
	static_assert(sign_policy<difference_sign_only, atl::int64>); // int128 is both product types there
	static_assert(!sign_policy<difference_sign_only, atl::int32>);
	static_assert(!sign_policy<mutable_policy, int>);
	static_assert(!sign_policy<pinned_policy, int>);
	static_assert(!sign_policy<boolean_orientation, int>);
	static_assert(!sign_policy<exact, unsigned>);
	static_assert(!orientable<int, difference_sign_only>);
	static_assert(!orientable<int, mutable_policy>);
	static_assert(orientable<int, exact>);
	static_assert(!polar_comparable<difference_sign_only, int>);
	static_assert(polar_comparable<exact, int>);

	// the angular sorts: random-access sized ranges of points, sortable for the in-place form
	static_assert(polar_sortable<std::vector<point<int>> &>);
	static_assert(!polar_sortable<std::vector<point<int>> const &>);
	static_assert(!polar_sortable<std::list<point<int>> &>);
	static_assert(!polar_sortable<std::vector<std::pair<int, int>> &>);
	static_assert(!polar_sortable<decltype(std::declval<std::vector<point<int>> &>()
										   | std::views::transform([](point<int> p) { return p; }))>);
	static_assert(polar_orderable<std::vector<point<int>> const &>);
	static_assert(!polar_orderable<std::list<point<int>> const &>);
	static_assert(!polar_orderable<std::vector<int> const &>);
	static_assert(!polar_orderable_with<std::vector<point<int>> const &, difference_sign_only>);
	static_assert(polar_orderable_with<std::vector<point<double>> const &, atl::geometry::fast>);

	// union_area: random-access sized ranges whose value type is a box
	static_assert(box_measurable<std::vector<box<int>> const &>);
	static_assert(!box_measurable<std::vector<point<int>> const &>);
	static_assert(!box_measurable<std::list<box<int>> const &>);
	static_assert(!box_measurable<std::vector<std::pair<point<int>, point<int>>> const &>);
	static_assert(box_measurable_by_contender<std::vector<box<int>> const &>);
	static_assert(!box_measurable_by_contender<std::list<box<int>> const &>);

	// line_from_coefficients: signed integral (int128 included) or floating coefficients
	static_assert(line_buildable<int> && line_buildable<atl::int128> && line_buildable<double>);
	static_assert(!line_buildable<unsigned>);
	static_assert(!line_buildable<bool>);
	static_assert(!line_buildable<point<int>>);
}

namespace {
	using namespace geometry_boundary_values;
	using atl::geometry::exact;

	// polygons: random-access sized ranges of points; integral coordinates
	// for lattice counts; a query of the vertices' own coordinate type
	static_assert(area_measurable<std::vector<point<int>> const &>);
	static_assert(!area_measurable<std::list<point<int>> const &>);
	static_assert(!area_measurable<std::vector<std::pair<int, int>> const &>);
	static_assert(lattice_countable<std::vector<point<atl::int16>> const &>);
	static_assert(!lattice_countable<std::vector<point<double>> const &>);
	static_assert(convexity_testable<std::vector<point<double>> const &, exact>);
	static_assert(!convexity_testable<std::vector<point<int>> const &, difference_sign_only>);
	static_assert(!convexity_testable<std::vector<point<int>> const &, mutable_policy>);
	static_assert(locatable<std::vector<point<int>> const &, point<int>>);
	static_assert(!locatable<std::vector<point<int>> const &, point<long long>>);
	static_assert(!locatable<std::list<point<int>> const &, point<int>>);
	static_assert(lattice_count_type<atl::int128> && lattice_count_type<atl::int32>);
	static_assert(!lattice_count_type<double>);
	static_assert(!lattice_count_type<unsigned>);

	// radial sweep: ranges of exactly arc; point ranges for the circle cover
	static_assert(arc_sweepable<std::vector<atl::geometry::arc> const &>);
	static_assert(!arc_sweepable<std::vector<double> const &>);
	static_assert(!arc_sweepable<std::list<atl::geometry::arc> const &>);
	static_assert(!arc_sweepable<std::vector<std::pair<double, double>> const &>);
	static_assert(circle_coverable<std::vector<point<atl::int64>> const &>);
	static_assert(!circle_coverable<std::list<point<int>> const &>);
	static_assert(!circle_coverable<std::vector<atl::geometry::arc> const &>);

	// closest pair and Manhattan MST, library and contenders
	static_assert(closest_pairable<std::vector<point<double>> const &>);
	static_assert(!closest_pairable<std::list<point<int>> const &>);
	static_assert(!closest_pairable<std::vector<int> const &>);
	static_assert(closest_pairable_by_contenders<std::vector<point<int>> const &>);
	static_assert(!closest_pairable_by_contenders<std::list<point<int>> const &>);
	static_assert(l1_spannable<std::vector<point<atl::int16>> const &>);
	static_assert(!l1_spannable<std::list<point<int>> const &>);
	static_assert(!l1_spannable<std::vector<std::pair<int, int>> const &>);
	static_assert(l1_spannable_by_contender<std::vector<point<int>> const &>);
	static_assert(!l1_spannable_by_contender<std::list<point<int>> const &>);
}

TEST_CASE("geometry primitives reject calls outside their constraints", "[geometry][boundary]") {
	// the static_asserts above are the proof; this case keeps the unit a test
	CHECK(polar_orderable<std::vector<point<int>> const &>);
	CHECK(!box_measurable<std::vector<point<int>> const &>);
	CHECK(!closest_pairable<std::list<point<int>> const &>);
}

namespace geometry_hull_boundary {
	using namespace atl::geometry;
	template<typename R, typename Policy = exact>
	concept hull_calls = requires(R &&r, Policy p) {
		convex_hull(r, p);
		convex_hull(r, keep_collinear, p);
		convex_hull_indices(r, p);
		convex_hull_indices(r, keep_collinear, p);
		convex_layers(r, p);
		convex_contains(r, {}, p);
		merge_convex_hulls(r, r, p);
		minkowski_sum(r, r, p);
		farthest_pair(r, p);
		farthest_pair_indices(r, p);
		farthest_pair_between(r, r, p);
		farthest_pair_between_indices(r, r, p);
		narrowest_strip(r, p);
		distance_between_convex_hulls(r, r, p);
		min_area_enclosing_rectangle(r, p);
		min_perimeter_enclosing_rectangle(r, p);
		antipodal_view(r, p);
		dual_support_view(r, r, p);
		dual_support_view(r, r, anti_parallel, p);
		atl::experimental::convex_hull_indices_by_graham_scan(r, p);
		atl::experimental::convex_hull_indices_by_jarvis_march(r, p);
		atl::experimental::convex_hull_indices_by_quickhull(r, p);
		atl::experimental::convex_hull_indices_by_chan(r, p);
		atl::experimental::merge_convex_hulls_by_streamed_chains(r, r, p);
		atl::experimental::merge_convex_hulls_by_materialized_chains(r, r, p);
		atl::experimental::merge_convex_hulls_by_copodal_walk(r, r, p);
		atl::experimental::convex_hull_union_by_materialized_chains(r, r, p);
		atl::experimental::convex_hull_union_by_copodal_walk(r, r, p);
	};

	template<typename R, typename Policy = exact>
	constexpr bool rejects_every_call() {
		return !requires(R &&r, Policy p) { convex_hull(r, p); }
			&& !requires(R &&r, Policy p) { convex_hull(r, keep_collinear, p); }
			&& !requires(R &&r, Policy p) { convex_hull_indices(r, p); }
			&& !requires(R &&r, Policy p) { convex_hull_indices(r, keep_collinear, p); }
			&& !requires(R &&r, Policy p) { convex_layers(r, p); }
			&& !requires(R &&r, Policy p) { convex_contains(r, {}, p); }
			&& !requires(R &&r, Policy p) { merge_convex_hulls(r, r, p); }
			&& !requires(R &&r, Policy p) { minkowski_sum(r, r, p); }
			&& !requires(R &&r, Policy p) { farthest_pair(r, p); }
			&& !requires(R &&r, Policy p) { farthest_pair_indices(r, p); }
			&& !requires(R &&r, Policy p) { farthest_pair_between(r, r, p); }
			&& !requires(R &&r, Policy p) { farthest_pair_between_indices(r, r, p); }
			&& !requires(R &&r, Policy p) { narrowest_strip(r, p); }
			&& !requires(R &&r, Policy p) { distance_between_convex_hulls(r, r, p); }
			&& !requires(R &&r, Policy p) { min_area_enclosing_rectangle(r, p); }
			&& !requires(R &&r, Policy p) { min_perimeter_enclosing_rectangle(r, p); }
			&& !requires(R &&r, Policy p) { antipodal_view(r, p); }
			&& !requires(R &&r, Policy p) { dual_support_view(r, r, p); }
			&& !requires(R &&r, Policy p) { dual_support_view(r, r, anti_parallel, p); }
			&& !requires(R &&r, Policy p) { atl::experimental::convex_hull_indices_by_graham_scan(r, p); }
			&& !requires(R &&r, Policy p) { atl::experimental::convex_hull_indices_by_jarvis_march(r, p); }
			&& !requires(R &&r, Policy p) { atl::experimental::convex_hull_indices_by_quickhull(r, p); }
			&& !requires(R &&r, Policy p) { atl::experimental::convex_hull_indices_by_chan(r, p); }
			&& !requires(R &&r, Policy p) { atl::experimental::merge_convex_hulls_by_streamed_chains(r, r, p); }
			&& !requires(R &&r, Policy p) { atl::experimental::merge_convex_hulls_by_materialized_chains(r, r, p); }
			&& !requires(R &&r, Policy p) { atl::experimental::merge_convex_hulls_by_copodal_walk(r, r, p); }
			&& !requires(R &&r, Policy p) { atl::experimental::convex_hull_union_by_materialized_chains(r, r, p); }
			&& !requires(R &&r, Policy p) { atl::experimental::convex_hull_union_by_copodal_walk(r, r, p); };
	}

	static_assert(rejects_every_call<std::list<point<int>> &>());
	static_assert(rejects_every_call<std::vector<int> &>());
	static_assert(rejects_every_call<std::vector<point<int>> &, geometry_boundary_values::mutable_policy>());

	static_assert(hull_calls<std::vector<point<int>> const &>);
	static_assert(!hull_calls<std::list<point<int>> &>);
	static_assert(!hull_calls<std::vector<int> &>);
	static_assert(!hull_calls<std::vector<point<int>> &, geometry_boundary_values::mutable_policy>);
	static_assert(!hull_calls<std::vector<point<int>> &, geometry_boundary_values::difference_sign_only>);
	static_assert(!std::constructible_from<antipodal_view<std::vector<point<int>>>, std::string>);
	static_assert(!std::constructible_from<dual_support_view<std::vector<point<int>>, std::vector<point<int>>>, std::string, std::string>);
	template<typename A, typename B>
	concept mergeable = requires(A a, B b) { merge_convex_hulls(a, b); };
	static_assert(!mergeable<std::array<point<int>, 3>, std::array<point<double>, 3>>);
}

namespace geometry_operations_boundary {
	using namespace atl::geometry;

	template<typename A, typename B = A, typename Policy = exact>
	constexpr bool rejects_operations() {
		return !requires(A a, B b, Policy p) { intersection_of_convex_polygons(a, b, p); }
			&& !requires(A a, B b, Policy p) { union_of_convex_polygons(a, b, p); }
			&& !requires(A a, B b, Policy p) { closest_points_between_convex_hulls(a, b, p); };
	}

	static_assert(rejects_operations<std::vector<int>>());
	static_assert(rejects_operations<std::list<point<int>>>());
	static_assert(rejects_operations<std::vector<point<int>>, std::vector<point<double>>>());
	static_assert(rejects_operations<std::vector<point<int>>, std::vector<point<int>>, geometry_boundary_values::mutable_policy>());
	template<typename T>
	concept halfplane_type = requires { typename halfplane<T>; };
	static_assert(halfplane_type<atl::int16> && halfplane_type<double>);
	static_assert(!halfplane_type<char> && !halfplane_type<signed char> && !halfplane_type<unsigned> && !halfplane_type<long double>);
	template<typename R, typename Policy = exact>
	concept intersectable = requires(R const &r, Policy p) { halfplane_intersection(r, p); };
	static_assert(intersectable<std::array<halfplane<int>, 4>>);
	static_assert(!intersectable<std::list<halfplane<int>>> && !intersectable<std::vector<point<int>>>);
	static_assert(!intersectable<std::vector<halfplane<int>>, approx>);
	template<typename T, typename Policy>
	concept halfplane_contains = requires(halfplane<T> h, point<T> v, Policy p) { contains(h, v, p); };
	static_assert(!halfplane_contains<int, geometry_boundary_values::mutable_policy>);
	static_assert(!halfplane_contains<int, geometry_boundary_values::difference_sign_only>);
	template<typename T>
	concept sweep_comparable = requires(T const &a) { halfplane_sweep_less{}(a, a); };
	static_assert(!sweep_comparable<point<int>>);
	template<typename R, typename Determinant = detail::filtered_determinant, typename Pass = detail::deque_pass>
	concept composable = requires(R const &r) {
		atl::experimental::halfplane_intersection_by_composition<Determinant, Pass, atl::experimental::halfplane_sort::key>(r);
	};
	static_assert(composable<std::vector<halfplane<int>>>);
	static_assert(!composable<std::list<halfplane<int>>>);
	static_assert(!composable<std::vector<halfplane<int>>, int>);
	static_assert(!composable<std::vector<halfplane<int>>, detail::filtered_determinant, int>);
}

namespace geometry_operations_boundary {
	struct narrow_sign_policy : exact {
		int sign(atl::int32 const v) const { return (v > 0) - (v < 0); }

		int sign(atl::int64 const v) const { return (v > 0) - (v < 0); }
	};

	static_assert(sign_policy<narrow_sign_policy, atl::int16>);
	static_assert(!halfplane_contains<atl::int16, narrow_sign_policy>);

	struct mutable_halfplanes {
		halfplane<int> *begin();
		halfplane<int> *end();
		std::size_t size() const;
	};

	static_assert(std::ranges::random_access_range<mutable_halfplanes>);
	static_assert(!intersectable<mutable_halfplanes>);
}

namespace circle_ray_delaunay_boundaries {
	using namespace atl::geometry;
	template<typename T>
	constexpr bool rejects_coordinate = !requires { typename circle<T>; } && !requires { typename ray<T>; } && !requires { typename ray_intersection<T>; };
	static_assert(rejects_coordinate<char> && rejects_coordinate<unsigned> && rejects_coordinate<long double>);

	template<typename R>
	constexpr bool rejects_range =
		!requires(R &&r) { delaunay(r); } && !requires(R &&r) { delaunay_edges(r); } && !requires(R &&r) { euclidean_mst(r); } && !requires(R &&r) { largest_empty_circle(r); } && !requires(R &&r) { voronoi_from_delaunay(r, delaunay_result{}); } && !requires(R &&r) { minimum_enclosing_circle(r); };
	static_assert(rejects_range<std::vector<int>>);
	static_assert(rejects_range<std::list<point<int>>>);

	template<typename A, typename B>
	constexpr bool rejects_pair = !requires(A &&a, B &&b) { closest_pair_between_indices(a, b); } && !requires(A &&a, B &&b) { closest_pair_between(a, b); };
	static_assert(rejects_pair<std::vector<point<int>>, std::vector<point<double>>>);

	template<typename P>
	constexpr bool rejects_policy =
		!requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { contains(c, q, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { relation(c, c, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { relation(c, l, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { relation(l, c, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { tangent_points(c, q, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { side_of(r, q, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { contains(r, q, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { intersects(r, l, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { intersects(l, r, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { intersects(r, s, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { intersects(s, r, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { intersects(r, r, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { intersection(r, l, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { intersection(l, r, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { intersection(r, s, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { intersection(s, r, p); } && !requires(P p, circle<int> c, ray<int> r, point<int> q, line<int> l, segment<int> s) { intersection(r, r, p); };
	static_assert(rejects_policy<geometry_boundary_values::mutable_policy>);
	static_assert(rejects_policy<int>);

	template<typename R>
	constexpr bool rejects_disk_polygon = !requires(R &&r, circle<int> c) { intersection_area(c, r); } && !requires(R &&r, circle<int> c) { intersection_area(r, c); };
	static_assert(rejects_disk_polygon<std::vector<point<double>>>);
	static_assert(rejects_disk_polygon<std::list<point<int>>>);
}

namespace geometry_subdivision_boundaries {
	using namespace atl::geometry;
	template<typename R>
	constexpr bool rejects_segments =
		!requires(R &&r) { segment_intersections(r); } && !requires(R &&r) { arrangement_of(r); }
		&& !requires(R &&r) { segment_intersections_view(r); };
	static_assert(rejects_segments<std::vector<int>> && rejects_segments<std::list<segment<int>>>);
	template<typename R>
	constexpr bool rejects_polygon =
		!requires(R &&r) { simplicity_of(r); } && !requires(R &&r) { triangulation_of(r); } && !requires(R &&r) { triangulation_of_by_ear_clipping(r); };
	static_assert(rejects_polygon<std::vector<int>> && rejects_polygon<std::list<point<int>>>);
	template<typename P>
	concept triangulation_policy = requires(std::vector<point<int>> v, P p) { triangulation_of(v, p); };
	template<typename P>
	concept ear_policy = requires(std::vector<point<int>> v, P p) { triangulation_of_by_ear_clipping(v, p); };
	static_assert(!ear_policy<int> && !ear_policy<geometry_boundary_values::mutable_policy>);
	template<typename O>
	constexpr bool rejects_overlay_areas = !requires(O const &o) { intersection_area(o); }
										&& !requires(O const &o) { union_area(o); } && !requires(O const &o) { difference_area(o); }
										&& !requires(O const &o) { symmetric_difference_area(o); };
	static_assert(rejects_overlay_areas<int> && rejects_overlay_areas<std::vector<point<int>>>);
	static_assert(!triangulation_policy<int> && !triangulation_policy<geometry_boundary_values::mutable_policy>);
	template<typename A, typename B>
	constexpr bool rejects_overlay =
		!requires(A a, B b) { overlay_of(a, b); } && !requires(A a, B b) { intersection_area(a, b); }
		&& !requires(A a, B b) { union_area(a, b); } && !requires(A a, B b) { difference_area(a, b); }
		&& !requires(A a, B b) { symmetric_difference_area(a, b); };
	static_assert(rejects_overlay<std::vector<point<int>>, std::vector<point<double>>>);
	using segments = std::vector<segment<int>>;
	static_assert(!std::constructible_from<point_location<int>, arrangement, segments const &>);
	static_assert(!std::constructible_from<point_location<int>, arrangement const &, std::vector<segment<double>> const &>);
	static_assert(!std::constructible_from<atl::experimental::point_location_by_trapezoid_map<int>, arrangement, segments const &>);
	template<typename R, typename Q>
	concept batch = requires(arrangement const &a, R const &r, Q const &q) { locations_of(a, r, q); };
	static_assert(!batch<segments, std::vector<point<double>>> && !batch<segments, std::list<point<int>>>);

	struct mutable_segments {
		segment<int> *begin();
		segment<int> *end();
		std::size_t size() const;
	};

	static_assert(segment_range<mutable_segments> && !segment_range<mutable_segments const>);
	static_assert(!std::constructible_from<point_location<int>, arrangement const &, mutable_segments const &>);
	static_assert(!std::constructible_from<atl::experimental::point_location_by_trapezoid_map<int>, arrangement const &, mutable_segments const &>);
	static_assert(!batch<mutable_segments, std::vector<point<int>>>);
	static_assert(!std::is_copy_constructible_v<segment_intersections_view<segments &>>);
	static_assert(!std::is_move_constructible_v<segment_intersections_view<segments &>>);

	template<typename R>
	concept all_pairs = requires(R &&r) { atl::experimental::arrangement_of_by_all_pairs(r); };
	static_assert(!all_pairs<std::vector<segment<double>>> && !all_pairs<std::list<segment<int>>>);
	template<typename T>
	concept trapezoid_coordinate = requires { typename atl::experimental::point_location_by_trapezoid_map<T>; };
	static_assert(!trapezoid_coordinate<double> && !trapezoid_coordinate<char> && !trapezoid_coordinate<unsigned>);
}
