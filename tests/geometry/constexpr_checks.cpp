#include <atl/experimental/geometry/box_variants.hpp>
#include <atl/experimental/geometry/closest_pair_variants.hpp>
#include <atl/experimental/geometry/convex_hull_variants.hpp>
#include <atl/experimental/geometry/halfplane_intersection_variants.hpp>
#include <atl/geometry/angular_sort.hpp>
#include <atl/geometry/arrangement.hpp>
#include <atl/geometry/box.hpp>
#include <atl/geometry/calipers.hpp>
#include <atl/geometry/circle.hpp>
#include <atl/geometry/convex_layers.hpp>
#include <atl/geometry/convex_polygon_operations.hpp>
#include <atl/geometry/delaunay.hpp>
#include <atl/geometry/halfplane.hpp>
#include <atl/geometry/halfplane_intersection.hpp>
#include <atl/geometry/line.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/point_location.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/polygon_overlay.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/ray.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/geometry/triangulation.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/radix_sort.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <compare>
#include <cstdint>
#include <initializer_list>
#include <ranges>
#include <string>
#include <vector>

namespace geometry_primitive_checks {
	using atl::geometry::point;

	// point arithmetic and the widened value functions
	static_assert(atl::geometry::dot(point{3, 4}, point{5, -6}) == 15 - 24);
	static_assert(atl::geometry::dot(point{3LL, 4LL}, point{5LL, 6LL}) == atl::int128(39));
	static_assert(atl::geometry::dot(point{1.5f, 2.0f}, point{2.0f, 0.25f}) == 3.5);
	static_assert(atl::geometry::squared_length(point{3, 4}) == 25);
	static_assert(atl::geometry::cross(point{2, 0}, point{0, 3}) == 6);
	static_assert(atl::geometry::cross(point{0, 0}, point{2, 0}, point{0, 3}) == 6);
	static_assert(atl::geometry::cross(point{1LL, 1LL}, point{3LL, 1LL}, point{1LL, 4LL}) == atl::int128(6));
	static_assert(atl::geometry::cross(point{0.0f, 0.0f}, point{2.0f, 0.0f}, point{0.0f, 3.0f}) == 6.0);
	static_assert(atl::geometry::squared_distance(point<atl::int16>{-128, -128}, point<atl::int16>{127, 127}) == 2 * 255 * 255);
	static_assert(atl::geometry::cross(point<atl::int16>{-128, -128}, point<atl::int16>{127, -128}) == 32768 - 128);
	static_assert(-point{1, -2} == point{-1, 2});
	static_assert(point{1, 2} * 3 == point{3, 6});
	static_assert(3 * point{1, 2} == point{3, 6});
	static_assert(point{1, 2} - point{3, 5} == point{-2, -3});
	static_assert(point{1, 2} + point{3, 5} == point{4, 7});
	static_assert(atl::geometry::perp(point{1, 2}) == point{-2, 1});
	static_assert([] {
		point p{4, 6};
		p -= point{1, 1};
		p *= 2;
		p += point{0, 1};
		return p;
	}() == point{6, 11});
	static_assert((point{1, 5} <=> point{2, 0}) == std::strong_ordering::less);
	static_assert(point{1, 5} < point{1, 6});
	static_assert(atl::geometry::to_float64(point{3, -4}) == point{3.0, -4.0});

	// segment
	static_assert(atl::geometry::direction(atl::geometry::segment{point{1, 2}, point{4, 6}}) == point{3, 4});
	static_assert(atl::geometry::squared_length(atl::geometry::segment{point{1, 2}, point{4, 6}}) == 25);
	static_assert(atl::geometry::contains(atl::geometry::segment{point{0, 0}, point{4, 2}}, point{2, 1}));
	static_assert(!atl::geometry::contains(atl::geometry::segment{point{0, 0}, point{4, 2}}, point{6, 3}));
	static_assert(atl::geometry::intersects(atl::geometry::segment{point{0, 0}, point{4, 4}},
											atl::geometry::segment{point{0, 4}, point{4, 0}}));
	static_assert(!atl::geometry::intersects(atl::geometry::segment{point{0.0, 0.0}, point{1.0, 0.0}},
											 atl::geometry::segment{point{2.0, 0.0}, point{3.0, 0.0}}));

	// policies, the exact floating path and the approx::sign constraint
	static_assert(atl::geometry::fast{}.orientation(point{0, 0}, point{1, 0}, point{0, 1}) == 1);
	static_assert(atl::geometry::fast{}.orientation(point{0.0, 0.0}, point{1.0, 0.0}, point{0.0, 1.0}) == 1);
	static_assert(atl::geometry::fast{}.cross_sign(point{0, 0}, point{1, 0}, point{0, 0}, point{0, -1}) == -1);
	static_assert(atl::geometry::fast{}.cross_sign(point{0.0, 0.0}, point{1.0, 0.0}, point{0.0, 0.0}, point{0.0, 1.0}) == 1);
	static_assert(atl::geometry::approx{0.5}.orientation(point{0.0, 0.0}, point{10.0, 0.0}, point{20.0, 0.04}) == 0);
	static_assert(atl::geometry::approx{0.5}.orientation(point{0.0, 0.0}, point{10.0, 0.0}, point{20.0, 0.06}) == 1);
	static_assert(atl::geometry::approx{0.5}.cross_sign(point{0, 0}, point{10, 0}, point{0, 0}, point{20, 1}) == 1);
	static_assert(atl::geometry::approx{0.5}.cross_sign(point{0.0, 0.0}, point{10.0, 0.0}, point{0.0, 0.0}, point{20.0, 0.04}) == 0);
	static_assert(atl::geometry::approx{0.5}.cross_sign(point<atl::int64>{0, 0}, point<atl::int64>{1, 0},
														point<atl::int64>{0, 0}, point<atl::int64>{0, 1})
				  == 1);
	static_assert(atl::geometry::approx{0.5}.equal(point{0.0, 0.0}, point{0.3, 0.3}));
	static_assert(!atl::geometry::approx{0.5}.equal(point{0.0, 0.0}, point{0.4, 0.4}));
	static_assert(atl::geometry::approx{0.5}.equal(1.0, 1.4));
	static_assert(atl::geometry::exact{}.equal(point{1, 2}, point{1, 2}));
	static_assert(!atl::geometry::exact{}.equal(point{0.1, 0.2}, point{0.1, 0.25}));
	static_assert(atl::geometry::exact{}.sign(atl::int128(-7)) == -1);
	// the exact double predicate in constant evaluation (Dekker products instead
	// of fma): a mixed-exponent collinear triple nudged by one ulp, whose
	// coordinate differences are inexact; plain IEEE evaluation reads it as 0
	static_assert(atl::geometry::orientation(point{0x1.567547a34c162p+38, 0x1.567547a34c162p+23},
											 point{0x1.46c04d9ff7cf6p+21, 0x1.46c04d9ff7cf6p+6},
											 point{0x1.a95d201fdd96cp+9, 0x1.a95d201fdd96dp-6})
				  == -1);
	static_assert(atl::geometry::orientation(point{0.1, 0.1}, point{0.3, 0.3}, point{0.5, 0.5000000000000001}) == 1);
	static_assert(atl::geometry::exact{}.cross_sign(point{0.1, 0.1}, point{0.3, 0.3}, point{0.1, 0.1}, point{0.5, 0.5}) == 0);

	// expansion arithmetic: (1 + 2^-60)(1 - 2^-60) = 1 - 2^-120 exactly, the
	// cancelled middle components dropped
	constexpr bool expansion_product_proof() {
		atl::float64 const e[2] = {0x1p-60, 1.0}, f[2] = {-0x1p-60, 1.0};
		atl::float64 h[8] = {}, scratch[8] = {}, scaled[4] = {};
		int const n = atl::geometry::detail::expansion_product(2, e, 2, f, h, scratch, scaled);
		return n == 2 && h[0] == -0x1p-120 && h[1] == 1.0;
	}

	static_assert(expansion_product_proof());

	template<typename V>
	concept approx_signable = requires(atl::geometry::approx const a, V const v) { a.sign(v); };
	static_assert(approx_signable<atl::int128>);
	static_assert(!approx_signable<point<int>>);

	// line
	constexpr atl::geometry::line<int> horizontal{{0, 0}, {4, 0}};
	static_assert(atl::geometry::direction(horizontal) == point{4, 0});
	static_assert(atl::geometry::contains(horizontal, point{-7, 0}));
	static_assert(!atl::geometry::contains(horizontal, point{1, 1}));
	static_assert(atl::geometry::same_line(horizontal, atl::geometry::line<int>{{9, 0}, {-3, 0}}));
	static_assert(atl::geometry::parallel(horizontal, atl::geometry::line<int>{{9, 1}, {-3, 1}}));
	static_assert(atl::geometry::intersects(horizontal, atl::geometry::line<int>{{0, 1}, {1, 2}}));
	static_assert(!atl::geometry::intersects(horizontal, atl::geometry::line<int>{{0, 1}, {1, 1}}));
	static_assert(atl::geometry::intersects(horizontal, atl::geometry::segment<int>{{2, -1}, {3, 1}}));
	static_assert(!atl::geometry::intersects(horizontal, atl::geometry::segment<int>{{2, 1}, {3, 1}}));
	static_assert(atl::geometry::side_of(atl::geometry::segment<int>{{0, 0}, {4, 0}}, point{2, -3})
				  == atl::geometry::side::right);
	static_assert(atl::geometry::side_of(horizontal, point{2, 3}) == atl::geometry::side::left);
	static_assert(atl::geometry::line_coefficients(atl::geometry::line<int>{{0, 0}, {4, 2}})
				  == std::array<atl::int128, 3>{-2, 4, 0});
	static_assert(atl::geometry::line_coefficients(atl::geometry::line<atl::int16>{{-128, -128}, {127, 127}})
				  == std::array<atl::int64, 3>{-255, 255, 0});
	static_assert(atl::geometry::line_from_coefficients(0.0, 1.0, -2.0)
				  == atl::geometry::line<atl::float64>{{0.0, 2.0}, {1.0, 2.0}});
	static_assert(atl::geometry::line_from_coefficients(atl::geometry::line_coefficients(atl::geometry::line<int>{{0, 0}, {4, 0}}))
				  == atl::geometry::line<atl::float64>{{0.0, 0.0}, {4.0, 0.0}});
	static_assert(atl::geometry::intersection(horizontal, atl::geometry::segment<int>{{2, -2}, {2, 2}}).at == point{2.0, 0.0});
	static_assert(atl::geometry::intersection(horizontal, atl::geometry::line<int>{{1, -1}, {3, 1}}) == point{2.0, 0.0});
	static_assert(atl::geometry::projection(horizontal, point{3, 5}) == point{3.0, 0.0});
	static_assert([] {
		using atl::geometry::segment;
		auto const overlap = atl::geometry::intersection(segment<int>{{0, 0}, {6, 0}}, segment<int>{{8, 0}, {4, 0}});
		auto const touch = atl::geometry::intersection(segment<int>{{0, 0}, {4, 0}}, segment<int>{{4, 0}, {9, 5}});
		auto const none = atl::geometry::intersection(segment<int>{{0, 0}, {1, 0}}, segment<int>{{3, 0}, {5, 0}});
		return overlap.kind == atl::geometry::intersection_kind::along_segment
			&& overlap.along == segment<int>{{4, 0}, {6, 0}} && touch.kind == atl::geometry::intersection_kind::at_point
			&& touch.at == point{4.0, 0.0} && none.kind == atl::geometry::intersection_kind::empty;
	}());

	// polar-angle sort, both forms
	static_assert([] {
		std::array pts{point{0, 1}, point{-1, 0}, point{1, 0}, point{0, -1}};
		atl::geometry::sort_by_polar_angle(pts);
		return pts;
	}() == std::array{point{0, -1}, point{1, 0}, point{0, 1}, point{-1, 0}});
	static_assert(atl::geometry::polar_angle_order(std::array{point{0, 1}, point{2, 0}, point{0, 0}, point{-1, -1}})
				  == std::vector<atl::geometry::vertex_index>{3, 1, 2, 0});
	static_assert(atl::geometry::polar_angle_less{}(point{1.0, -1.0}, point{1.0, 0.0}));
	static_assert(!atl::geometry::polar_angle_less{atl::geometry::fast{}}(point{2, 2}, point{3, 3}));

	// boxes: the library tree and its raced contender
	static_assert(atl::geometry::union_area(std::array{atl::geometry::box<int>{{0, 0}, {4, 4}},
													   atl::geometry::box<int>{{2, 2}, {6, 6}}})
				  == atl::int128(28));
	static_assert(atl::experimental::union_area_by_top_down_tree(std::array{atl::geometry::box<int>{{0, 0}, {4, 4}},
																			atl::geometry::box<int>{{2, 2}, {6, 6}}})
				  == atl::int128(28));
	static_assert(atl::geometry::union_area(std::array{atl::geometry::box<atl::int16>{{-128, -128}, {127, 127}}}) == 255 * 255);
	static_assert(atl::geometry::union_area(std::array{atl::geometry::box<double>{{0.0, 0.0}, {1.5, 1.0}},
													   atl::geometry::box<double>{{0.5, 0.0}, {2.0, 1.0}}})
				  == 2.0);
	static_assert(atl::geometry::union_area(std::vector<atl::geometry::box<atl::int64>>{}) == atl::int128(0));
}

namespace geometry_polygon_checks {
	using atl::geometry::point;
	using atl::geometry::winding;

	static_assert(atl::geometry::polygon_area(std::array{point{0, 0}, point{0, 4}, point{4, 4}, point{4, 0}}) == 16.0);
	static_assert(atl::geometry::polygon_area2(std::array{point{0, 0}, point{4, 0}, point{0, 4}}) == atl::int128(16));
	// int64 at the bound, wound twice forward and twice back: the partial sums
	// pass 2^128 and wrap (a signed overflow would not be a constant expression)
	constexpr atl::int64 m = (atl::int64(1) << 62) - 1;
	static_assert(atl::geometry::polygon_area2(std::array{
					  point{-m, -m}, point{m, -m}, point{m, m}, point{-m, m}, point{-m, -m}, point{m, -m}, point{m, m}, point{-m, m},
					  point{-m, m}, point{m, m}, point{m, -m}, point{-m, -m}, point{-m, m}, point{m, m}, point{m, -m}, point{-m, -m}})
				  == atl::int128(0));
	static_assert(atl::geometry::contains(std::array{point{0, 0}, point{4, 0}, point{4, 4}, point{0, 4}}, point{2, 2})
				  == atl::geometry::containment::inside);
	static_assert(atl::geometry::contains(std::array{point{0.0, 0.0}, point{4.0, 0.0}, point{0.0, 4.0}}, point{2.0, 2.0})
				  == atl::geometry::containment::boundary);
	static_assert(atl::geometry::convexity_of(std::array{point{0, 0}, point{2, 0}, point{4, 0}, point{4, 4}, point{0, 4}})
				  == atl::geometry::convexity{true, false, winding::counterclockwise});
	static_assert(atl::geometry::winding_of(std::array{point{0, 4}, point{4, 4}, point{4, 0}, point{0, 0}}) == winding::clockwise);
	static_assert(atl::geometry::normalized_polygon(std::array{point{0, 4}, point{4, 4}, point{4, 0}, point{4, 0}, point{2, 0}, point{0, 0}})
				  == std::vector{point{0, 4}, point{0, 0}, point{4, 0}, point{4, 4}});
	static_assert(atl::geometry::lattice_count_of(std::array{point{0, 0}, point{4, 0}, point{0, 4}})
				  == atl::geometry::lattice_count<atl::int128>{12, 3});
	static_assert(atl::geometry::polygon_centroid(std::array{point{0, 0}, point{3, 0}, point{0, 3}}) == point{1.0, 1.0});

	/** @brief 40 points on a 5-wide lattice with one planted pair, deep enough for the strip and merge levels. */
	constexpr std::array<point<int>, 40> closest_pair_cloud() {
		std::array<point<int>, 40> pts{};
		for (int k = 0; k < 40; ++k) pts[std::size_t(k)] = {(k * 7) % 40 * 5, (k * 13) % 40 * 5};
		pts[17] = {pts[3].x + 2, pts[3].y + 1};
		return pts;
	}

	/** @brief Whether the divide and conquer finds the planted pair (indices 3 and 17). */
	constexpr bool divide_and_conquer_finds_planted_pair() {
		auto const pair = atl::experimental::closest_pair_indices_by_divide_and_conquer(closest_pair_cloud());
		return (pair[0] == 3 && pair[1] == 17) || (pair[0] == 17 && pair[1] == 3);
	}

#if !defined(_MSC_VER) || defined(__clang__)
	// MSVC rejects the recursion's caller-local references in constant
	// evaluation; the test case below runs the same check
	static_assert(divide_and_conquer_finds_planted_pair());
#endif
}

TEST_CASE("geometry primitives constant-evaluate", "[geometry][constexpr]") {
	// the static_asserts above are the proof; this case keeps the unit a test
	using atl::geometry::point;
	CHECK(atl::geometry::orientation(point{0, 0}, point{1, 0}, point{0, 1}) == 1);
	CHECK(atl::geometry::union_area(std::array{atl::geometry::box<int>{{0, 0}, {2, 3}}}) == atl::int128(6));
	CHECK(geometry_primitive_checks::expansion_product_proof()); // the fma path at run time
	CHECK(geometry_polygon_checks::divide_and_conquer_finds_planted_pair());
}

static_assert([] {
	using atl::geometry::point;
	constexpr int n = 1 << 30;
	return atl::geometry::intersection(atl::geometry::line<int>{{0, 0}, {n, n - 2}}, atl::geometry::line<int>{{1, 1}, {n - 1, n - 3}})
		== point{double(n / 2), double(n / 2 - 1)};
}());
static_assert([] {
	auto const l = atl::geometry::line_from_coefficients(std::array<atl::int64, 3>{1, 1, -(atl::int64(1) << 61)});
	return l.a != l.b;
}());
static_assert([] {
	atl::geometry::point<atl::int64> const a{atl::int64(1) << 61, 0}, b{(atl::int64(1) << 61) + 1, 0};
	return !atl::geometry::approx{0.5}.equal(a, b);
}());

static_assert(atl::geometry::contains(std::array{atl::geometry::point<atl::int64>{0, 0},
												 atl::geometry::point<atl::int64>{4, 0},
												 atl::geometry::point<atl::int64>{0, 4}},
									  atl::geometry::point{1, 2})
			  == atl::geometry::containment::inside);
static_assert(atl::geometry::contains(std::array{atl::geometry::point<atl::int64>{0, 0},
												 atl::geometry::point<atl::int64>{4, 0},
												 atl::geometry::point<atl::int64>{0, 4}},
									  {1, 2})
			  == atl::geometry::containment::inside);
static_assert(atl::geometry::union_area(std::array{atl::geometry::box<float>{{0, -0x1p30f}, {1, -1}},
												   atl::geometry::box<float>{{0, 0}, {1, 0x1p30f}}})
			  == 0x1p31 - 1);

namespace geometry_hull_checks {
	using namespace atl::geometry;
	constexpr std::array triangle{point{0, 0}, point{4, 0}, point{0, 4}};
	static_assert(convex_hull(std::array{point{0, 0}, point{4, 0}, point{2, 1}, point{4, 4}, point{0, 4}})
				  == std::vector{point{0, 0}, point{4, 0}, point{4, 4}, point{0, 4}});
	static_assert(convex_hull_indices(std::array{point{0, 0}, point{2, 0}, point{4, 0}, point{2, 3}}, keep_collinear)
				  == std::vector<vertex_index>{0, 1, 2, 3});
	static_assert(convex_hull(triangle, keep_collinear) == convex_hull(triangle));
	static_assert([] {
		std::array<point<int>, 256> grid{};
		for (int i = 0; i < 256; ++i) grid[std::size_t(i)] = {i % 16, i / 16};
		return convex_hull(grid) == std::vector{point{0, 0}, point{15, 0}, point{15, 15}, point{0, 15}};
	}());
	static_assert(convex_contains(triangle, {1, 1}) == containment::inside);
	static_assert(merge_convex_hulls(std::array{point{0, 0}, point{2, 0}, point{0, 2}},
									 std::array{point{1, 3}, point{3, 1}, point{3, 3}})
				  == std::vector{point{0, 0}, point{2, 0}, point{3, 1}, point{3, 3}, point{1, 3}, point{0, 2}});
	static_assert(minkowski_sum(triangle, std::array{point{1, 2}}) == std::vector{point{1, 2}, point{5, 2}, point{1, 6}});
	static_assert([] {
		auto const layers = convex_layers(std::array{point{0, 0}, point{4, 0}, point{4, 4}, point{0, 4}, point{2, 2}});
		return layers.size() == 2 && layers[0] == std::vector<vertex_index>{0, 1, 2, 3} && layers[1] == std::vector<vertex_index>{4};
	}());
	static_assert(convex_layers(std::array{point{0, 0}, point{2, 0}, point{4, 0}, point{2, 0}}).size() == 1);
	static_assert(!std::constructible_from<antipodal_view<std::vector<point<int>>>, std::string>);
	static_assert([] {
		std::array const square{point{0, 0}, point{2, 0}, point{2, 2}, point{0, 2}};
		int events = 0;
		for ([[maybe_unused]] auto const event : dual_support_view(square, square)) ++events;
		for ([[maybe_unused]] auto const event : dual_support_view(square, square, anti_parallel)) ++events;
		return events == 16;
	}());
	static_assert([] {
		auto const view = antipodal_view(std::array{point{0, 0}, point{2, 0}, point{2, 2}, point{0, 2}});
		auto it = view.begin();
		auto const first = it++;
		return (*first).antipodal_pairs()[0] == std::array<vertex_index, 2>{0, 2}
			&& *it == antipodal_event{0, 1, 3} && it != view.end();
	}());
	static_assert(support_event{0, 1, false, 0, 3}.support_pairs() == std::array<std::array<vertex_index, 2>, 2>{{{0, 3}, {1, 3}}});
	static_assert(support_event{2, 3, true, 1, 2}.support_pairs() == std::array<std::array<vertex_index, 2>, 2>{{{1, 2}, {1, 3}}});
	static_assert(strip<int>{{0, 0}, {1, 0}, {0, 1}} == strip<int>{{0, 0}, {1, 0}, {0, 1}});
	static_assert(farthest_pair(std::array{point{0, 0}, point{4, 0}, point{1, 1}, point{4, 4}}) == std::array{point{0, 0}, point{4, 4}});
	static_assert(farthest_pair_between_indices(std::array{point{0, 0}, point{1, 0}, point{0, 1}},
												std::array{point{5, 5}, point{9, 9}, point{6, 5}})
				  == std::array<vertex_index, 2>{0, 1});
	static_assert(farthest_pair_between(std::array{point{0, 0}, point{1, 0}, point{0, 1}},
										std::array{point{5, 5}, point{9, 9}, point{6, 5}})
				  == std::array{point{0, 0}, point{9, 9}});
	static_assert(atl::experimental::convex_hull_indices_by_graham_scan(triangle) == convex_hull_indices(triangle));
	static_assert(atl::experimental::convex_hull_indices_by_jarvis_march(triangle) == convex_hull_indices(triangle));
	static_assert(atl::experimental::convex_hull_indices_by_quickhull(triangle) == convex_hull_indices(triangle));
	static_assert(atl::experimental::convex_hull_indices_by_chan(triangle) == convex_hull_indices(triangle));
	static_assert(atl::experimental::merge_convex_hulls_by_streamed_chains(
					  std::array{point{0, 0}, point{1, 0}, point{0, 1}}, std::array{point{3, 0}, point{4, 1}, point{3, 2}})
				  == std::vector{point{0, 0}, point{3, 0}, point{4, 1}, point{3, 2}, point{0, 1}});
	static_assert(atl::experimental::merge_convex_hulls_by_materialized_chains(triangle, triangle) == convex_hull(triangle));
	static_assert(atl::experimental::merge_convex_hulls_by_copodal_walk(triangle, triangle) == convex_hull(triangle));
	static_assert(atl::experimental::convex_hull_union_by_materialized_chains(triangle, triangle) == convex_hull(triangle));
	static_assert(atl::experimental::convex_hull_union_by_copodal_walk(triangle, triangle) == convex_hull(triangle));
}

namespace geometry_halfplane_checks {
	using namespace atl::geometry;
	using hp = halfplane<long long>;

	constexpr bool members() {
		auto const h = hp::left_of({0, 0}, {2, 1});
		auto const third = halfplane<double>::at_most(3, 0, 1);
		double const below_third = 1.0 / 3.0;
		return h.contains({0, 1}) == containment::inside && h.contains({4, 2}) == containment::boundary
			&& third.contains({below_third, 0}) == containment::inside
			&& contains(h, point<long long>{0, 1}) == containment::inside
			&& contains(h, point<long long>{4, 2}, fast{}) == containment::boundary
			&& contains(third, point<double>{below_third, 0}, exact{}) == containment::inside
			&& h == hp::at_most(1, -2, 0) && approx{0x1p-48}.equal(direction(h.boundary()), point{2.0, 1.0});
	}

	static_assert(members());
	static_assert(halfplane<double>::left_of({0.0, 0.0}, {2.0, 1.0}) == halfplane<double>{1.0, -2.0, 0.0});
	static_assert(contains(halfplane<int>::at_most(1, 0, 1), point{2, 0}, approx{2.0}) == containment::boundary);

	constexpr bool square_with_diagonal() {
		std::vector<hp> const h{{1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1}, {1, 1, 2}};
		auto const region = halfplane_intersection(h);
		return region.kind == region_kind::polygon && region.boundary == std::vector<atl::uint32>{3, 0, 2, 1}
			&& approx{0x1p-48}.equal(region.vertices[2], point{1.0, 1.0});
	}

	static_assert(square_with_diagonal());

	constexpr bool degenerate_kinds() {
		using hd = halfplane<double>;
		auto const kind = [](std::vector<hd> const &h) { return halfplane_intersection(h).kind; };
		return kind({{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}}) == region_kind::point
			&& kind({{0, 1, 1}, {0, -1, -1}, {-1, 0, 0}, {1, 0, 2}}) == region_kind::segment
			&& kind({{1, 0, -1}, {-1, 0, -1}, {0, 1, 1}, {0, -1, 1}}) == region_kind::empty
			&& kind({{0, 1, 0}, {0, -1, 0}, {-1, 0, 0}}) == region_kind::unbounded
			&& kind({}) == region_kind::unbounded;
	}

	static_assert(degenerate_kinds());

	constexpr bool policies_and_sweep_order() {
		using hd = halfplane<double>;
		std::vector<hd> h{{1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1}, {1, 1, 2}};
		std::ranges::sort(h, halfplane_sweep_less{});
		auto const want = halfplane_intersection(h, exact{});
		auto const got = halfplane_intersection(h, fast{});
		return std::ranges::is_sorted(h, halfplane_sweep_less{}) && h[0] == hd{0, -1, 1}
			&& want.kind == region_kind::polygon && got.kind == want.kind && got.boundary == want.boundary;
	}

	static_assert(policies_and_sweep_order());

	constexpr bool shared_radix_sort() {
		using keyed = atl::detail::keyed_position<atl::uint32>;
		std::vector<keyed> keys{{0x30000u, 0}, {5u, 1}, {0x30000u, 2}, {0u, 3}, {0xffffffffu, 4}};
		atl::detail::radix_sort<11>(keys);
		std::vector<atl::uint32> order;
		for (auto const &k : keys) order.push_back(k.position);
		return order == std::vector<atl::uint32>{3, 1, 0, 2, 4};
	}

	static_assert(shared_radix_sort());

	constexpr bool compositions() {
		namespace ex = atl::experimental;
		namespace gd = atl::geometry::detail;
		std::vector<hp> const h{{1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1}, {1, 1, 2}};
		std::vector<halfplane<double>> const hd{{1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1}, {1, 1, 2}};
		auto const want = halfplane_intersection(h).boundary;
		// 16-bit radix sorts exceed default constexpr step limits; shared_radix_sort proves the radix core.
		return ex::halfplane_intersection_by_composition<gd::filtered_determinant, gd::deque_pass, ex::halfplane_sort::key>(h).boundary == want
			&& ex::halfplane_intersection_by_composition<gd::filtered_determinant, ex::detail::corner_cached_pass, ex::halfplane_sort::rotated_runs>(h).boundary == want
			&& ex::halfplane_intersection_by_composition<gd::filtered_determinant, ex::detail::envelope_pass, ex::halfplane_sort::comparator>(h).boundary == want
			&& ex::halfplane_intersection_by_composition<gd::filtered_determinant, ex::detail::deque_pass_testing_every_front, ex::halfplane_sort::key_std_sort>(h).boundary == want
			&& ex::halfplane_intersection_by_composition<ex::detail::always_exact_determinant, gd::deque_pass, ex::halfplane_sort::comparator>(hd).boundary == want
			&& ex::halfplane_intersection_by_composition<ex::detail::product_filter_determinant, gd::deque_pass, ex::halfplane_sort::key_std_sort>(hd).boundary == want
			&& ex::halfplane_intersection_by_composition<ex::detail::dekker_determinant, gd::deque_pass, ex::halfplane_sort::key_64_std_sort>(hd).boundary == want
			&& ex::halfplane_intersection_by_composition<ex::detail::rounded_determinant, gd::deque_pass, ex::halfplane_sort::key>(hd).boundary == want;
	}

	static_assert(compositions());
}

TEST_CASE("half-plane constexpr proofs also run with runtime product transforms", "[geometry][constexpr]") {
	CHECK(geometry_halfplane_checks::members());
	CHECK(geometry_halfplane_checks::square_with_diagonal());
	CHECK(geometry_halfplane_checks::degenerate_kinds());
	CHECK(geometry_halfplane_checks::policies_and_sweep_order());
	CHECK(geometry_halfplane_checks::shared_radix_sort());
	CHECK(geometry_halfplane_checks::compositions());
}

namespace geometry_publication_checks {
	constexpr bool caliper_initializer_lists() {
		using namespace atl::geometry;
		using P = point<int>;
		auto a = antipodal_view(std::initializer_list<P>{{0, 0}, {4, 0}, {0, 4}});
		auto d = dual_support_view(std::initializer_list<P>{{0, 0}, {4, 0}, {0, 4}},
								   std::initializer_list<P>{{0, 0}, {2, 0}, {0, 2}});
		return (*a.begin()).opposite == 2 && std::ranges::distance(d) == 6;
	}

	constexpr bool wide_predicates() {
		namespace gd = atl::geometry::detail;
		using P = atl::geometry::point<atl::int64>;
		using L = atl::geometry::line<atl::int64>;
		atl::int64 const k = atl::int64(1) << 60;
		std::array<P, 2> const inputs{{{-k, k}, {k, -k}}};
		L const diagonal{{-k, -k}, {k, k}}, opposite{{-k, k}, {k, -k}}, horizontal{{-k, 0}, {k, 0}};
		atl::int128 const factor = atl::int128(1) << 124;
		return gd::check_operation_bounds(inputs)
			&& gd::compare_wide_products(factor, factor, factor, factor - 1) > 0
			&& gd::compare_operation_products(factor, factor, factor, factor, true) == 0
			&& gd::crossing_side(diagonal, opposite, horizontal, true) == 0
			&& !gd::param_before(horizontal, diagonal, opposite, true);
	}

	static_assert(caliper_initializer_lists());
	static_assert(wide_predicates());
}

TEST_CASE("publication changes evaluate at compile time and runtime", "[geometry][constexpr]") {
	CHECK(geometry_publication_checks::caliper_initializer_lists());
	CHECK(geometry_publication_checks::wide_predicates());
}

namespace circle_ray_delaunay_checks {
	using namespace atl::geometry;
	constexpr ray<int> proof_beam{{0, 0}, {2, 1}};
	static_assert(direction(proof_beam) == point{2, 1});
	static_assert(contains(proof_beam, point{4, 2}));
	static_assert(!contains(proof_beam, point{-2, -1}));
	static_assert(side_of(proof_beam, point{0, 5}) == side::left);
	static_assert(intersects(proof_beam, segment<int>{{3, 0}, {3, 5}}));
	static_assert(!intersects(proof_beam, segment<int>{{-1, 0}, {-1, 5}}));
	static_assert(intersects(proof_beam, ray<int>{{4, 0}, {4, 5}}));
	static_assert(intersects(proof_beam, line<int>{{4, -5}, {4, 5}}));
	static_assert(!intersects(line<int>{{-4, -5}, {-4, 5}}, proof_beam));
	static_assert(intersects(segment<int>{{3, 0}, {3, 5}}, proof_beam));
	static_assert(intersection(proof_beam, line<int>{{0, 0}, {4, 2}}).kind == intersection_kind::along_ray);
	static_assert(intersection(proof_beam, ray<int>{{6, 3}, {8, 4}}).beam == ray<int>{{6, 3}, {8, 4}});
	static_assert(intersection(proof_beam, ray<int>{{6, 3}, {0, 0}}).kind == intersection_kind::along_segment);
	static_assert(intersection(proof_beam, segment<int>{{-2, -1}, {4, 2}}).along == segment<int>{{0, 0}, {4, 2}});
	static_assert(approx{1e-12}.equal(intersection(line<int>{{4, -5}, {4, 5}}, proof_beam).at, point{4.0, 2.0}));
	static_assert(approx{1e-12}.equal(intersection(segment<int>{{4, -5}, {4, 5}}, proof_beam).at, point{4.0, 2.0}));
	static_assert(ray_intersection<int>{} == ray_intersection<int>{});
	static_assert(circle{point{1, 2}, 3} == circle{point{1, 2}, 3});
	static_assert(circle{point{1, 2}, 3} != circle{point{1, 2}, 4});
	static_assert(relation(line{point{5, -3}, point{5, 8}}, circle{point{0, 0}, 5}) == tangency::tangent);
	static_assert(contains(circle{point{0.0, 0.0}, 5.0}, point{3.0, 4.0}) == containment::boundary);
	static_assert(contains(circle{point{0, 0}, 5}, point{3, 4}, fast{}) == containment::boundary);
	static_assert(contains(circle{point{0.0, 0.0}, 5.0}, point{3.0, 4.1}, approx{1.0}) == containment::boundary);
	static_assert(relation(circle{point{0.0, 0.0}, 5.0}, circle{point{10.0, 0.0}, 5.0}) == circle_relation::externally_tangent);
	static_assert(relation(circle{point{0, 0}, 5}, circle{point{3, 0}, 1}, approx{0.5}) == circle_relation::nested);
	static_assert(relation(circle{point{0.0, 0.0}, 5.0}, line{point{5.0, -3.0}, point{5.0, 8.0}}) == tangency::tangent);
	static_assert(relation(circle{point{0, 0}, 5}, line{point{6, -3}, point{6, 8}}, fast{}) == tangency::disjoint);
	static_assert(detail::incircle_sign(point{0, 0}, point{2, 0}, point{2, 2}, point{0, 2}) == 0);
	static_assert(detail::incircle_sign(point{0, 0}, point{2, 0}, point{2, 2}, point{1, 1}) == 1);
	static_assert(detail::incircle_sign(point{0, 0}, point{2, 0}, point{2, 2}, point{5, 5}) == -1);
	static_assert([] { atl::float64 const expansion[3]{0.0, 1e-30, -2.0}; return detail::expansion_sign(3, expansion); }() == -1);
	static_assert(relation(circle<atl::int16>{{0, 0}, 16383}, line<atl::int16>{{-16383, 16383}, {16383, 16383}}) == tangency::tangent);
}

TEST_CASE("circle ray and incircle constexpr proofs", "[geometry][constexpr]") {
	REQUIRE(atl::geometry::contains(circle_ray_delaunay_checks::proof_beam, atl::geometry::point{4, 2}));
}

namespace geometry_subdivision_checks {
	using namespace atl::geometry;
	static_assert(arrangement::twin(4) == 5 && arrangement::twin(5) == 4);
	static_assert(arrangement::edge_of(5) == 2);

	constexpr bool accessors() {
		arrangement a;
		a.origin = {0, 1};
		a.supporting = {{0}};
		return a.to(0) == 1 && a.to(1) == 0 && a.edge_count() == 1;
	}

	static_assert(accessors());
	static_assert(location{location_kind::edge, 3} == location{location_kind::edge, 3});
	static_assert(location{location_kind::edge, 3} != location{location_kind::face, 3});

	constexpr bool ears() {
		std::array const poly{point<atl::int16>{0, 0}, point<atl::int16>{4, 0}, point<atl::int16>{4, 4}, point<atl::int16>{0, 4}};
		auto const tri = triangulation_of_by_ear_clipping(poly);
		return tri.triangles.size() == 2 && tri.dual_edges.size() == 1
			&& detail::in_closed_triangle(point{0, 0}, point{4, 0}, point{0, 4}, point{1, 1}, exact{});
	}

	static_assert(ears());
}

TEST_CASE("arrangement accessors and ear clipping constexpr proofs", "[geometry][constexpr]") {
	CHECK(geometry_subdivision_checks::accessors());
	CHECK(geometry_subdivision_checks::ears());
}

namespace geometry_gp_checks {
	using namespace atl::geometry;

	constexpr bool overlay_sums() {
		// A face-area/membership fixture is constexpr; the arrangement sweep is runtime-only.
		polygon_overlay<int> overlay{};
		overlay.arr.faces = {{arrangement::none, {}, {}, -12.0}, {0, {}, {}, 3.0}, {0, {}, {}, 4.0}, {0, {}, {}, 5.0}};
		overlay.in_first = {0, 1, 1, 0};
		overlay.in_second = {0, 0, 1, 1};
		return intersection_area(overlay) == 4.0 && union_area(overlay) == 12.0
			&& difference_area(overlay) == 3.0 && symmetric_difference_area(overlay) == 8.0;
	}

	constexpr bool concave_ears() {
		std::array const poly{point{0, 0}, point{4, 0}, point{4, 1}, point{1, 1}, point{1, 4}, point{0, 4}};
		for (bool const is_reversed : {false, true}) {
			auto vertices = poly;
			if (is_reversed) std::ranges::reverse(vertices);
			auto const result = triangulation_of_by_ear_clipping(vertices);
			if (result.triangles.size() != 4 || result.dual_edges.size() != 3) return false;
			int total = 0;
			for (auto const t : result.triangles) {
				auto const area = cross(vertices[t[0]], vertices[t[1]], vertices[t[2]]);
				if (area <= 0) return false;
				total += int(area);
			}
			if (total != 14) return false;
		}
		return triangulation_of_by_ear_clipping(std::array<point<int>, 0>{}).triangles.empty();
	}

	constexpr bool wide_incircle() {
		using P = point<atl::int64>;
		atl::int64 const k = (atl::int64(1) << 61) - 1;
		return detail::incircle_sign(P{-k, -k}, P{k, -k}, P{k, k}, P{-k, k}) == 0
			&& detail::incircle_sign(P{-k, -k}, P{k, -k}, P{k, k}, P{0, 0}) == 1
			&& detail::sum_three_wide_products_sign(atl::int128(1) << 124, atl::int128(1) << 124,
													-(atl::int128(1) << 124), atl::int128(1) << 124, 3, -2)
				   == -1;
	}

	static_assert(overlay_sums());
	static_assert(concave_ears());
	static_assert(wide_incircle());
}

TEST_CASE("GP constexpr APIs also evaluate at runtime", "[geometry][constexpr]") {
	CHECK(geometry_gp_checks::overlay_sums());
	CHECK(geometry_gp_checks::concave_ears());
	CHECK(geometry_gp_checks::wide_incircle());
}
