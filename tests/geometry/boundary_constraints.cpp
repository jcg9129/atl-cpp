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
#include <concepts>
#include <cstdint>
#include <list>
#include <ranges>
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
