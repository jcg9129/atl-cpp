#include <atl/experimental/geometry/box_variants.hpp>
#include <atl/experimental/geometry/closest_pair_variants.hpp>
#include <atl/geometry/angular_sort.hpp>
#include <atl/geometry/box.hpp>
#include <atl/geometry/line.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <compare>
#include <cstdint>
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
