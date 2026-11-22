#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/convex_polygon_operations.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::geometry::closest_points_between_convex_hulls;
using atl::geometry::containment;
using atl::geometry::contains;
using atl::geometry::convex_hull;
using atl::geometry::intersection_of_convex_polygons;
using atl::geometry::point;
using atl::geometry::polygon_area;
using atl::geometry::union_of_convex_polygons;

namespace {
	using pt = point<std::int64_t>;
	using fpt = point<double>;

	// distance to a (possibly NONCONVEX) simple polygon of CONSTRUCTED
	// vertices: parity along an OBLIQUE ray (axis-parallel rays can run
	// along nearly horizontal constructed edges), else boundary distance
	double distance_to_polygon(std::vector<fpt> const &poly, fpt const q) {
		if (poly.empty()) return 1e300;
		if (poly.size() == 1) return std::hypot(q.x - poly[0].x, q.y - poly[0].y);
		double best = 1e300;
		for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
			fpt const a = poly[j], b = poly[i];
			double const dx = b.x - a.x, dy = b.y - a.y;
			double const len2 = dx * dx + dy * dy;
			double t = len2 == 0 ? 0 : ((q.x - a.x) * dx + (q.y - a.y) * dy) / len2;
			t = t < 0 ? 0 : t > 1 ? 1
								  : t;
			best = (std::min)(best, std::hypot(q.x - (a.x + t * dx), q.y - (a.y + t * dy)));
		}
		if (poly.size() < 3) return best;
		double const rx = 1.0, ry = 0.5372112416; // oblique ray direction
		int crossings = 0;
		for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
			fpt const a = poly[j], b = poly[i];
			double const denom = (b.x - a.x) * ry - (b.y - a.y) * rx;
			if (denom == 0) continue;
			double const u = ((q.x - a.x) * ry - (q.y - a.y) * rx) / denom;
			double const s = ((a.x - q.x) * (b.y - a.y) - (a.y - q.y) * (b.x - a.x)) / -denom;
			if (u >= 0 && u < 1 && s > 0) ++crossings;
		}
		return crossings % 2 == 1 ? 0.0 : best;
	}

	double union_area(std::vector<std::vector<fpt>> const &parts) {
		double total = 0;
		for (auto const &part : parts) total += polygon_area(part);
		return total;
	}

	std::vector<pt> random_hull_cloud(std::mt19937_64 &rng, std::int64_t cx, std::int64_t cy,
									  std::int64_t r, std::size_t n) {
		std::vector<pt> pts;
		for (std::size_t k = 0; k < n; ++k)
			pts.push_back({cx + std::int64_t(rng() % (2 * r + 1)) - r,
						   cy + std::int64_t(rng() % (2 * r + 1)) - r});
		return pts;
	}

	// full certificate for one (A, B) input pair
	void verify_pair(std::vector<pt> const &a, std::vector<pt> const &b, std::int64_t lo,
					 std::int64_t hi) {
		auto const inter = intersection_of_convex_polygons(a, b);
		auto const parts = union_of_convex_polygons(a, b);
		REQUIRE(parts.size() >= 1);
		REQUIRE(parts.size() <= 2);
		auto const ha = convex_hull(a), hb = convex_hull(b);
		// membership grid oracle: q in A and in B <=> q in the intersection
		// (mismatches counted only with a clear 1e-6 margin)
		for (std::int64_t x = lo; x <= hi; ++x)
			for (std::int64_t y = lo; y <= hi; ++y) {
				pt const q{x, y};
				bool const in_both = ha.size() >= 3 && hb.size() >= 3
								  && contains(ha, q) != containment::outside
								  && contains(hb, q) != containment::outside;
				if (ha.size() < 3 || hb.size() < 3) continue; // degenerate: skip grid
				double const gap = distance_to_polygon(inter, {double(x), double(y)});
				if (in_both) REQUIRE(gap <= 1e-6);
				if (!in_both) REQUIRE(gap >= -1e-6); // no strict-interior claim
				// the reverse direction: q strictly inside the intersection
				// must be in both (strictness via the 1e-3 margin)
				if (!in_both && inter.size() >= 3) {
					// q must not be DEEP inside the reported intersection
					bool deep = true;
					for (std::size_t i = 0, j = inter.size() - 1; i < inter.size(); j = i++) {
						double const cx = (inter[i].x - inter[j].x) * (double(y) - inter[j].y)
										- (inter[i].y - inter[j].y) * (double(x) - inter[j].x);
						if (cx < 1e-3) deep = false;
					}
					REQUIRE(!deep);
				}
			}
		// area certificate: |A| + |B| = |A u B| + |A n B|
		double const area_a = ha.size() >= 3 ? polygon_area(ha) : 0.0;
		double const area_b = hb.size() >= 3 ? polygon_area(hb) : 0.0;
		double const area_i = inter.size() >= 3 ? polygon_area(inter) : 0.0;
		double const area_u = union_area(parts);
		REQUIRE(std::abs(area_a + area_b - area_i - area_u)
				<= 1e-6 * (1 + area_a + area_b));
		// union certificates: every input point covered; every component
		// vertex on one of the hull boundaries
		auto const fa = [&](std::vector<pt> const &v) {
			std::vector<fpt> f;
			for (auto const q : v) f.push_back({double(q.x), double(q.y)});
			return f;
		};
		auto const fha = fa(ha), fhb = fa(hb);
		for (auto const &part : parts)
			for (auto const v : part) {
				double const on_a = [&] {
					double best = 1e300;
					for (std::size_t i = 0, j = fha.size() - 1; i < fha.size(); j = i++) {
						double const dx = fha[i].x - fha[j].x, dy = fha[i].y - fha[j].y;
						double const l2 = dx * dx + dy * dy;
						double t = l2 == 0 ? 0
										   : ((v.x - fha[j].x) * dx + (v.y - fha[j].y) * dy) / l2;
						t = t < 0 ? 0 : t > 1 ? 1
											  : t;
						best = (std::min)(best, std::hypot(v.x - (fha[j].x + t * dx),
														   v.y - (fha[j].y + t * dy)));
					}
					return best;
				}();
				double const on_b = [&] {
					double best = 1e300;
					for (std::size_t i = 0, j = fhb.size() - 1; i < fhb.size(); j = i++) {
						double const dx = fhb[i].x - fhb[j].x, dy = fhb[i].y - fhb[j].y;
						double const l2 = dx * dx + dy * dy;
						double t = l2 == 0 ? 0
										   : ((v.x - fhb[j].x) * dx + (v.y - fhb[j].y) * dy) / l2;
						t = t < 0 ? 0 : t > 1 ? 1
											  : t;
						best = (std::min)(best, std::hypot(v.x - (fhb[j].x + t * dx),
														   v.y - (fhb[j].y + t * dy)));
					}
					return best;
				}();
				REQUIRE((std::min)(on_a, on_b) <= 1e-6);
			}
		for (auto const q : ha) {
			double best = 1e300;
			for (auto const &part : parts)
				best = (std::min)(best, distance_to_polygon(part, {double(q.x), double(q.y)}));
			REQUIRE(best <= 1e-6);
		}
		for (auto const q : hb) {
			double best = 1e300;
			for (auto const &part : parts)
				best = (std::min)(best, distance_to_polygon(part, {double(q.x), double(q.y)}));
			REQUIRE(best <= 1e-6);
		}
	}
}

TEST_CASE("convex intersection/union: random pairs, membership + area certificates",
		  "[geometry]") {
	auto seed = GENERATE(take(40, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	// regimes: heavy overlap, partial, almost disjoint, nested
	std::int64_t const shift = std::int64_t(rng() % 4) * 5;
	auto const a = random_hull_cloud(rng, 0, 0, 6, 3 + std::size_t(rng() % 25));
	auto const b = rng() % 4 == 0
					 ? random_hull_cloud(rng, 1, 1, 2, 3 + std::size_t(rng() % 8)) // likely nested
					 : random_hull_cloud(rng, shift, shift / 2, 6, 3 + std::size_t(rng() % 25));
	verify_pair(a, b, -10, 18);
}

TEST_CASE("convex intersection/union: handcrafted exact cases", "[geometry]") {
	using atl::geometry::intersection_of_convex_polygons;
	std::vector<pt> const square{{0, 0}, {10, 0}, {10, 10}, {0, 10}};
	SECTION("disjoint") {
		std::vector<pt> const distant{{20, 0}, {30, 0}, {30, 10}, {20, 10}};
		REQUIRE(intersection_of_convex_polygons(square, distant).empty());
		auto const parts = union_of_convex_polygons(square, distant);
		REQUIRE(parts.size() == 2);
	}
	SECTION("identical") {
		auto const inter = intersection_of_convex_polygons(square, square);
		REQUIRE(inter.size() == 4);
		REQUIRE(std::abs(polygon_area(inter) - 100.0) <= 1e-9);
		auto const parts = union_of_convex_polygons(square, square);
		REQUIRE(parts.size() == 1);
		REQUIRE(std::abs(polygon_area(parts[0]) - 100.0) <= 1e-9);
	}
	SECTION("vertex touch: intersection is the exact point, union two parts") {
		std::vector<pt> const kissing{{10, 10}, {20, 10}, {20, 20}, {10, 20}};
		auto const inter = intersection_of_convex_polygons(square, kissing);
		REQUIRE(inter.size() == 1);
		REQUIRE(inter[0].x == 10.0);
		REQUIRE(inter[0].y == 10.0);
		REQUIRE(union_of_convex_polygons(square, kissing).size() == 2);
	}
	SECTION("edge touch: exact shared segment, union stitches to ONE polygon") {
		std::vector<pt> const beside{{10, 2}, {17, 2}, {17, 7}, {10, 7}};
		auto const inter = intersection_of_convex_polygons(square, beside);
		REQUIRE(inter.size() == 2);
		REQUIRE(inter[0].x == 10.0);
		REQUIRE(inter[1].x == 10.0);
		auto const parts = union_of_convex_polygons(square, beside);
		REQUIRE(parts.size() == 1);
		REQUIRE(std::abs(polygon_area(parts[0]) - (100.0 + 35.0)) <= 1e-9);
	}
	SECTION("proper overlap: two offset squares") {
		std::vector<pt> const offset{{5, 5}, {15, 5}, {15, 15}, {5, 15}};
		auto const inter = intersection_of_convex_polygons(square, offset);
		REQUIRE(inter.size() == 4);
		REQUIRE(std::abs(polygon_area(inter) - 25.0) <= 1e-9);
		auto const parts = union_of_convex_polygons(square, offset);
		REQUIRE(parts.size() == 1);
		REQUIRE(std::abs(polygon_area(parts[0]) - 175.0) <= 1e-9);
		REQUIRE(parts[0].size() == 8); // the L-union has 8 corners
	}
	SECTION("rotated-square star: 8 crossings") {
		std::vector<pt> const diamond{{5, -4}, {14, 5}, {5, 14}, {-4, 5}};
		auto const inter = intersection_of_convex_polygons(square, diamond);
		REQUIRE(inter.size() == 8);
		auto const parts = union_of_convex_polygons(square, diamond);
		REQUIRE(parts.size() == 1);
		REQUIRE(parts[0].size() == 16); // 8 crossings + 4 + 4 corner tips
		double const total = polygon_area(square) + polygon_area(diamond);
		REQUIRE(std::abs(polygon_area(parts[0]) + polygon_area(inter) - total) <= 1e-6);
	}
	SECTION("containment, strict and with shared boundary") {
		std::vector<pt> const inner{{2, 2}, {8, 2}, {8, 8}};
		REQUIRE(std::abs(polygon_area(intersection_of_convex_polygons(square, inner)) - 18.0)
				<= 1e-9);
		auto const parts = union_of_convex_polygons(square, inner);
		REQUIRE(parts.size() == 1);
		REQUIRE(std::abs(polygon_area(parts[0]) - 100.0) <= 1e-9);
		std::vector<pt> const flush{{0, 0}, {10, 0}, {5, 5}}; // shares an edge piece
		REQUIRE(std::abs(polygon_area(intersection_of_convex_polygons(square, flush)) - 25.0)
				<= 1e-9);
		auto const parts2 = union_of_convex_polygons(square, flush);
		REQUIRE(parts2.size() == 1);
		REQUIRE(std::abs(polygon_area(parts2[0]) - 100.0) <= 1e-9);
	}
	SECTION("degenerate summands") {
		// point inside / outside
		REQUIRE(intersection_of_convex_polygons(square, std::vector<pt>{{4, 4}}).size() == 1);
		REQUIRE(intersection_of_convex_polygons(square, std::vector<pt>{{40, 4}}).empty());
		REQUIRE(union_of_convex_polygons(square, std::vector<pt>{{4, 4}}).size() == 1);
		REQUIRE(union_of_convex_polygons(square, std::vector<pt>{{40, 4}}).size() == 2);
		// segment crossing through: clipped exactly to the chord
		auto const chord = intersection_of_convex_polygons(
			square, std::vector<pt>{{-5, 5}, {15, 5}});
		REQUIRE(chord.size() == 2);
		REQUIRE(std::abs(chord[0].x - 0.0) <= 1e-9);
		REQUIRE(std::abs(chord[1].x - 10.0) <= 1e-9);
		// protruding segment: union cannot be a polygon -> two components
		REQUIRE(union_of_convex_polygons(square, std::vector<pt>{{-5, 5}, {15, 5}}).size() == 2);
		// fully inside segment: swallowed
		REQUIRE(union_of_convex_polygons(square, std::vector<pt>{{2, 5}, {8, 5}}).size() == 1);
		// segment x segment crossing
		auto const x = intersection_of_convex_polygons(std::vector<pt>{{0, 0}, {10, 10}},
													   std::vector<pt>{{0, 10}, {10, 0}});
		REQUIRE(x.size() == 1);
		REQUIRE(std::abs(x[0].x - 5.0) <= 1e-9);
	}
	SECTION("at the 2^30 bound") {
		std::int64_t const c = (std::int64_t(1) << 30) - 1;
		std::vector<pt> const big1{{-c, -c}, {c, -c}, {c, c}, {-c, c}};
		std::vector<pt> const big2{{0, -c}, {c, 0}, {0, c}, {-c, 0}};
		auto const inter = intersection_of_convex_polygons(big1, big2);
		REQUIRE(inter.size() == 4); // the diamond is inside the square
		double const want = 2.0 * double(c) * double(c);
		REQUIRE(std::abs(polygon_area(inter) - want) <= 1e-3 * want);
		REQUIRE(union_of_convex_polygons(big1, big2).size() == 1);
	}
}

TEST_CASE("convex intersection/union: double parity spot checks", "[geometry]") {
	std::vector<fpt> const sq{{0, 0}, {10, 0}, {10, 10}, {0, 10}};
	std::vector<fpt> const off{{5.5, 5.5}, {15.5, 5.5}, {15.5, 15.5}, {5.5, 15.5}};
	auto const inter = intersection_of_convex_polygons(sq, off);
	REQUIRE(inter.size() == 4);
	REQUIRE(std::abs(polygon_area(inter) - 4.5 * 4.5) <= 1e-9);
	auto const parts = union_of_convex_polygons(sq, off);
	REQUIRE(parts.size() == 1);
	REQUIRE(std::abs(polygon_area(parts[0]) - (200.0 - 4.5 * 4.5)) <= 1e-9);
}

TEST_CASE("convex intersection/union: the degenerate-summand matrix", "[geometry]") {
	// empty inputs are an asserted precondition; every 1- and 2-point
	// combination is pinned here (intersections exact; union component
	// counts per the contract, collinear-touching degenerate pairs MERGE)
	auto const check = [](std::vector<pt> const &a, std::vector<pt> const &b,
						  std::size_t inter_size, std::size_t union_parts) {
		auto const i = intersection_of_convex_polygons(a, b);
		auto const u = union_of_convex_polygons(a, b);
		REQUIRE(i.size() == inter_size);
		REQUIRE(u.size() == union_parts);
		// symmetry
		REQUIRE(intersection_of_convex_polygons(b, a).size() == inter_size);
		REQUIRE(union_of_convex_polygons(b, a).size() == union_parts);
	};
	std::vector<pt> const square{{0, 0}, {10, 0}, {10, 10}, {0, 10}};
	check({{3, 3}}, {{3, 3}}, 1, 1);				 // point == point
	check({{3, 3}}, {{5, 5}}, 0, 2);				 // distinct points
	check({{2, 2}}, {{0, 0}, {6, 6}}, 1, 1);		 // point on segment interior
	check({{0, 0}}, {{0, 0}, {6, 6}}, 1, 1);		 // point = segment endpoint
	check({{2, 3}}, {{0, 0}, {6, 6}}, 0, 2);		 // point off segment
	check({{0, 0}, {6, 6}}, {{0, 0}, {6, 6}}, 2, 1); // identical segments
	check({{2, 2}, {4, 4}}, {{0, 0}, {6, 6}}, 2, 1); // nested segments
	check({{0, 0}, {4, 4}}, {{2, 2}, {8, 8}}, 2, 1); // collinear partial overlap
	check({{0, 0}, {3, 3}}, {{3, 3}, {8, 8}}, 1, 1); // collinear endpoint touch
	check({{0, 0}, {2, 2}}, {{4, 4}, {8, 8}}, 0, 2); // collinear disjoint
	check({{0, 0}, {4, 4}}, {{0, 4}, {4, 0}}, 1, 2); // X-crossing
	check({{0, 0}, {3, 3}}, {{3, 3}, {6, 0}}, 1, 2); // bent endpoint touch
	check({{5, 0}}, square, 1, 1);					 // point on polygon edge
	check({{0, 0}}, square, 1, 1);					 // point on polygon vertex
	check({{5, 0}, {5, -7}}, square, 1, 2);			 // segment endpoint on edge
	check({{2, 0}, {7, 0}}, square, 2, 1);			 // segment collinear ON edge
	check({{-4, 0}, {6, 0}}, square, 2, 2);			 // collinear, half outside
	check({{-3, 3}, {3, -3}}, square, 1, 2);		 // touches a vertex only
	// merged values, not just counts: collinear partial overlap
	auto const merged = union_of_convex_polygons(std::vector<pt>{{0, 0}, {4, 4}},
												 std::vector<pt>{{2, 2}, {8, 8}});
	REQUIRE(merged[0] == std::vector<fpt>{{0.0, 0.0}, {8.0, 8.0}});
	auto const chopped = intersection_of_convex_polygons(std::vector<pt>{{0, 0}, {4, 4}},
														 std::vector<pt>{{2, 2}, {8, 8}});
	REQUIRE(chopped == std::vector<fpt>{{2.0, 2.0}, {4.0, 4.0}});
}

TEST_CASE("convex intersection: tiny and huge results are tidied at their own scale", "[geometry]") {
	// An absolute 1e-6 cross-product dead zone flattens a 1e-3 square;
	// cleanup must scale with the constructed coordinates.
	using atl::geometry::intersection_of_convex_polygons;
	using atl::geometry::point;
	for (double const side : {1e-3, 1e-4, 1e-7, 1.0, 1e6, 1e9}) {
		std::vector<point<double>> const a{{0, 0}, {side, 0}, {side, side}, {0, side}};
		std::vector<point<double>> b;
		for (auto const q : a) b.push_back({q.x + side / 2, q.y + side / 2});
		auto const common = intersection_of_convex_polygons(a, b);
		REQUIRE(common.size() == std::size_t(4));
		double doubled = 0;
		for (std::size_t i = 0, j = common.size() - 1; i < common.size(); j = i++)
			doubled += common[j].x * common[i].y - common[j].y * common[i].x;
		REQUIRE(doubled > 0); // counterclockwise
		REQUIRE(std::abs(doubled / 2 - side * side / 4) <= 1e-9 * side * side);
	}
}

TEST_CASE("convex polygon operations: raw coordinate bounds are asserted before Minkowski sums", "[geometry]") {
	// int32 at +-2e9: the Minkowski difference would overflow int32 before any hull check
	std::vector<point<int>> const distant{{2'000'000'000, 0}, {2'000'000'001, 0}, {2'000'000'000, 1}};
	std::vector<point<int>> const close_input{{-2'000'000'000, 0}, {-1'999'999'999, 0}, {-2'000'000'000, 1}};
	CHECK(atl_test::aborts([&] { (void)intersection_of_convex_polygons(distant, close_input); }));
	CHECK(atl_test::aborts([&] { (void)union_of_convex_polygons(distant, close_input); }));
	CHECK(atl_test::aborts([&] { (void)closest_points_between_convex_hulls(distant, close_input); }));
	// int64 rejects the 2^61 boundary even on the non-overlap path
	std::vector<point<std::int64_t>> const wide{{std::int64_t(1) << 61, 0}};
	std::vector<point<std::int64_t>> const origin{{0, 0}};
	CHECK(atl_test::aborts([&] { (void)closest_points_between_convex_hulls(wide, origin); }));
	CHECK(atl_test::aborts([&] { (void)intersection_of_convex_polygons(origin, wide); }));
	CHECK(atl_test::aborts([&] { (void)union_of_convex_polygons(wide, origin); }));
	std::vector<pt> const negative{{0, -(std::int64_t(1) << 61)}};
	CHECK(atl_test::aborts([&] { (void)intersection_of_convex_polygons(negative, origin); }));
	CHECK(atl_test::aborts([&] { (void)union_of_convex_polygons(origin, negative); }));
	CHECK(atl_test::aborts([&] { (void)closest_points_between_convex_hulls(origin, negative); }));
}

namespace {
	// Independent arbitrary-precision signed integers in radix 2^16. The oracle
	// shares no arithmetic or predicates with the implementation under test.
	class oracle_int {
		std::vector<std::uint32_t> digits_;
		bool is_negative_ = false;

		void normalize() {
			while (!digits_.empty() && digits_.back() == 0) digits_.pop_back();
			if (digits_.empty()) is_negative_ = false;
		}

		static int magnitude_order(oracle_int const &a, oracle_int const &b) {
			if (a.digits_.size() != b.digits_.size()) return a.digits_.size() < b.digits_.size() ? -1 : 1;
			for (std::size_t i = a.digits_.size(); i != 0; --i)
				if (a.digits_[i - 1] != b.digits_[i - 1]) return a.digits_[i - 1] < b.digits_[i - 1] ? -1 : 1;
			return 0;
		}

	public:
		oracle_int() = default;

		template<std::integral Integer>
		oracle_int(Integer value) { // NOLINT(google-explicit-constructor): integer value conversion
			if constexpr (std::signed_integral<Integer>) is_negative_ = value < 0;
			std::uint64_t magnitude = is_negative_ ? std::uint64_t(0) - std::uint64_t(value) : std::uint64_t(value);
			while (magnitude != 0) {
				digits_.push_back(std::uint32_t(magnitude & 65535));
				magnitude >>= 16;
			}
		}

		explicit operator double() const {
			double result = 0;
			for (std::size_t i = digits_.size(); i != 0; --i) result = result * 65536 + digits_[i - 1];
			return is_negative_ ? -result : result;
		}

		oracle_int operator-() const {
			auto result = *this;
			if (!digits_.empty()) result.is_negative_ = !is_negative_;
			return result;
		}

		friend oracle_int operator+(oracle_int a, oracle_int const &b) {
			if (a.is_negative_ == b.is_negative_) {
				a.digits_.resize((std::max)(a.digits_.size(), b.digits_.size()) + 1);
				std::uint32_t carry = 0;
				for (std::size_t i = 0; i < a.digits_.size(); ++i) {
					std::uint32_t const sum = a.digits_[i] + (i < b.digits_.size() ? b.digits_[i] : 0) + carry;
					a.digits_[i] = sum & 65535;
					carry = sum >> 16;
				}
			} else {
				if (magnitude_order(a, b) < 0) return b + a;
				std::uint32_t borrow = 0;
				for (std::size_t i = 0; i < a.digits_.size(); ++i) {
					std::uint32_t const sub = (i < b.digits_.size() ? b.digits_[i] : 0) + borrow;
					borrow = a.digits_[i] < sub ? 1 : 0;
					a.digits_[i] = (a.digits_[i] + 65536 - sub) & 65535;
				}
			}
			a.normalize();
			return a;
		}

		friend oracle_int operator-(oracle_int const &a, oracle_int const &b) { return a + -b; }

		friend oracle_int operator*(oracle_int const &a, oracle_int const &b) {
			oracle_int result;
			result.is_negative_ = a.is_negative_ != b.is_negative_;
			result.digits_.resize(a.digits_.size() + b.digits_.size());
			for (std::size_t i = 0; i < a.digits_.size(); ++i) {
				std::uint64_t carry = 0;
				for (std::size_t j = 0; j < b.digits_.size(); ++j) {
					std::uint64_t const product = std::uint64_t(a.digits_[i]) * b.digits_[j] + result.digits_[i + j] + carry;
					result.digits_[i + j] = std::uint32_t(product & 65535);
					carry = product >> 16;
				}
				if (!b.digits_.empty()) result.digits_[i + b.digits_.size()] = std::uint32_t(carry);
			}
			result.normalize();
			return result;
		}

		friend oracle_int operator<<(oracle_int value, unsigned bits) {
			for (unsigned i = 0; i < bits; ++i) value = value + value;
			return value;
		}

		friend bool operator==(oracle_int const &, oracle_int const &) = default;

		friend std::strong_ordering operator<=>(oracle_int const &a, oracle_int const &b) {
			int const order = a.is_negative_ != b.is_negative_ ? (a.is_negative_ ? -1 : 1)
															   : magnitude_order(a, b) * (a.is_negative_ ? -1 : 1);
			return order <=> 0;
		}
	};

	struct convex_rational {
		oracle_int x, y, d;
	};

	convex_rational rational(pt const p) { return {p.x, p.y, 1}; }

	oracle_int orient(pt const a, pt const b, convex_rational const q) {
		return oracle_int(b.x - a.x) * (q.y - oracle_int(a.y) * q.d)
			 - oracle_int(b.y - a.y) * (q.x - oracle_int(a.x) * q.d);
	}

	// Caratheodory's theorem gives an independent oracle without building either hull.
	bool in_cloud(std::vector<pt> const &cloud, convex_rational const q) {
		for (auto const a : cloud)
			if (q.x == oracle_int(a.x) * q.d && q.y == oracle_int(a.y) * q.d) return true;
		for (std::size_t i = 0; i < cloud.size(); ++i)
			for (std::size_t j = i + 1; j < cloud.size(); ++j) {
				auto const a = cloud[i], b = cloud[j];
				if (orient(a, b, q) == 0
					&& oracle_int((std::min)(a.x, b.x)) * q.d <= q.x && q.x <= oracle_int((std::max)(a.x, b.x)) * q.d
					&& oracle_int((std::min)(a.y, b.y)) * q.d <= q.y && q.y <= oracle_int((std::max)(a.y, b.y)) * q.d) return true;
				for (std::size_t k = j + 1; k < cloud.size(); ++k) {
					auto const c = cloud[k];
					if (orient(a, b, rational(c)) == 0) continue;
					auto const u = orient(a, b, q), v = orient(b, c, q), w = orient(c, a, q);
					if ((u >= 0 && v >= 0 && w >= 0) || (u <= 0 && v <= 0 && w <= 0)) return true;
				}
			}
		return false;
	}

	oracle_int rational_turn(convex_rational const a, convex_rational const b, convex_rational const c) {
		return (b.x * a.d - a.x * b.d) * (c.y * a.d - a.y * c.d)
			 - (b.y * a.d - a.y * b.d) * (c.x * a.d - a.x * c.d);
	}

	std::vector<convex_rational> intersection_oracle(std::vector<pt> const &a, std::vector<pt> const &b) {
		std::vector<convex_rational> candidates;
		auto const add = [&](convex_rational const q) {
			if (in_cloud(a, q) && in_cloud(b, q)) candidates.push_back(q);
		};
		for (auto const p : a) add(rational(p));
		for (auto const p : b) add(rational(p));
		for (std::size_t i = 0; i < a.size(); ++i)
			for (std::size_t j = i + 1; j < a.size(); ++j)
				for (std::size_t k = 0; k < b.size(); ++k)
					for (std::size_t l = k + 1; l < b.size(); ++l) {
						oracle_int const dx = a[j].x - a[i].x, dy = a[j].y - a[i].y;
						oracle_int const ex = b[l].x - b[k].x, ey = b[l].y - b[k].y;
						oracle_int d = dx * ey - dy * ex;
						if (d == 0) continue;
						oracle_int n = oracle_int(b[k].x - a[i].x) * ey - oracle_int(b[k].y - a[i].y) * ex;
						if (d < 0) {
							d = -d;
							n = -n;
						}
						add({oracle_int(a[i].x) * d + dx * n, oracle_int(a[i].y) * d + dy * n, d});
					}
		auto const less = [](convex_rational const u, convex_rational const v) {
			return u.x * v.d != v.x * u.d ? u.x * v.d < v.x * u.d : u.y * v.d < v.y * u.d;
		};
		std::ranges::sort(candidates, less);
		auto const same = [](convex_rational const u, convex_rational const v) { return u.x * v.d == v.x * u.d && u.y * v.d == v.y * u.d; };
		auto const tail = std::ranges::unique(candidates, same);
		candidates.erase(tail.begin(), tail.end());
		if (candidates.size() < 2) return candidates;
		std::vector<convex_rational> boundary;
		std::size_t at = 0;
		do {
			boundary.push_back(candidates[at]);
			std::size_t next = at == 0 ? 1 : 0;
			for (std::size_t i = 0; i < candidates.size(); ++i) {
				auto const u = candidates[at], v = candidates[next], w = candidates[i];
				auto const turn = rational_turn(u, v, w);
				oracle_int const vx = v.x * u.d - u.x * v.d, vy = v.y * u.d - u.y * v.d;
				oracle_int const wx = w.x * u.d - u.x * w.d, wy = w.y * u.d - u.y * w.d;
				bool const farther = (wx * wx + wy * wy) * v.d * v.d > (vx * vx + vy * vy) * w.d * w.d;
				if (turn < 0 || (turn == 0 && farther)) next = i;
			}
			at = next;
		} while (at != 0 && boundary.size() <= candidates.size());
		return boundary;
	}

	struct squared_fraction {
		oracle_int n = 0, d = 1;
	};

	squared_fraction closest_squared_oracle(std::vector<pt> const &a, std::vector<pt> const &b, bool const overlaps) {
		if (overlaps) return {};
		squared_fraction best{oracle_int(1) << 200, 1};
		auto const visit = [&](std::vector<pt> const &sites, std::vector<pt> const &edges) {
			for (auto const p : sites)
				for (auto const u : edges)
					for (auto const v : edges) {
						oracle_int const dx = v.x - u.x, dy = v.y - u.y, px = p.x - u.x, py = p.y - u.y;
						oracle_int const length = dx * dx + dy * dy, along = px * dx + py * dy;
						squared_fraction value{px * px + py * py, 1};
						if (length != 0 && along > 0) {
							if (along >= length)
								value = {(px - dx) * (px - dx) + (py - dy) * (py - dy), 1};
							else {
								oracle_int const cross = dx * py - dy * px;
								value = {cross * cross, length};
							}
						}
						if (value.n * best.d < best.n * value.d) best = value;
					}
		};
		visit(a, b);
		visit(b, a);
		return best;
	}
}

TEST_CASE("convex operations: rational all-pairs and triangle oracle on degenerate clouds", "[geometry]") {
	std::mt19937_64 rng(83);
	for (int trial = 0; trial < 500; ++trial) {
		std::vector<pt> a(1 + std::size_t(rng() % 5)), b(1 + std::size_t(rng() % 5));
		for (auto *cloud : {&a, &b})
			for (auto &p : *cloud) p = {std::int64_t(rng() % 7) - 3, trial % 5 == 0 ? 0 : std::int64_t(rng() % 7) - 3};
		if (trial % 11 == 0) {
			std::ranges::fill(a, a.front());
			std::ranges::fill(b, b.front());
		}
		auto const want = intersection_oracle(a, b);
		auto const got = intersection_of_convex_polygons(a, b);
		REQUIRE(got.size() == want.size());
		for (auto const &q : want) {
			fpt const exact_value{double(q.x) / double(q.d), double(q.y) / double(q.d)};
			REQUIRE(std::ranges::any_of(got, [&](fpt const v) { return std::hypot(v.x - exact_value.x, v.y - exact_value.y) <= 1e-10; }));
		}
		auto const closest = closest_points_between_convex_hulls(a, b);
		auto const squared = closest_squared_oracle(a, b, !want.empty());
		double const dx = closest[1].x - closest[0].x, dy = closest[1].y - closest[0].y;
		REQUIRE(std::abs(dx * dx + dy * dy - double(squared.n) / double(squared.d)) <= 1e-10);
		auto const parts = union_of_convex_polygons(a, b);
		for (int x = -3; x <= 3; ++x)
			for (int y = -3; y <= 3; ++y) {
				bool const inside = in_cloud(a, {x, y, 1}) || in_cloud(b, {x, y, 1});
				double gap = 1e300;
				for (auto const &part : parts) gap = (std::min)(gap, distance_to_polygon(part, {double(x), double(y)}));
				if (inside)
					REQUIRE(gap <= 1e-10);
				else
					REQUIRE(gap > 1e-10);
			}
	}
}

TEST_CASE("convex operations: empty input preconditions and bound-distance witnesses", "[geometry]") {
	std::vector<pt> const empty, origin{{0, 0}};
	CHECK(atl_test::aborts([&] { (void)intersection_of_convex_polygons(empty, origin); }));
	CHECK(atl_test::aborts([&] { (void)union_of_convex_polygons(origin, empty); }));
	CHECK(atl_test::aborts([&] { (void)closest_points_between_convex_hulls(empty, empty); }));
	constexpr std::int64_t m = (std::int64_t(1) << 30) - 1;
	std::vector<pt> const left{{-m, -m}, {0, -m}, {0, m}, {-m, m}};
	std::vector<pt> const right{{1, -m}, {m, -m}, {m, m}, {1, m}};
	auto const pair = closest_points_between_convex_hulls(left, right);
	CHECK(std::abs(std::hypot(pair[0].x - pair[1].x, pair[0].y - pair[1].y) - 1) < 1e-6);
	CHECK(intersection_of_convex_polygons(left, right).empty());
	CHECK(union_of_convex_polygons(left, right).size() == 2);
}

namespace {
	template<atl::geometry::coordinate T>
		requires std::signed_integral<T>
	void verify_wide_contact() {
		constexpr T m = T((atl::int64(1) << (sizeof(T) == 8 ? 61 : sizeof(T) * 8 - 2)) - 1);
		std::vector<point<T>> const a{{T(-m), T(-m)}, {0, T(-m)}, {0, m}, {T(-m), m}};
		std::vector<point<T>> const b{{0, T(-m)}, {m, T(-m)}, {m, m}, {0, m}};
		auto const common = intersection_of_convex_polygons(a, b);
		REQUIRE(common.size() == 2);
		CHECK(std::abs(common[0].x) < 1e-10);
		CHECK(std::abs(common[1].x) < 1e-10);
		CHECK(std::abs(common[0].y + double(m)) <= 1e-14 * double(m));
		CHECK(std::abs(common[1].y - double(m)) <= 1e-14 * double(m));
		auto const parts = union_of_convex_polygons(a, b);
		REQUIRE(parts.size() == 1);
		CHECK(std::abs(polygon_area(parts[0]) / (4 * double(m) * double(m)) - 1) < 1e-12);
	}
}

TEST_CASE("convex contact: difference edges wider than the coordinate type", "[geometry]") {
	verify_wide_contact<atl::int16>();
	verify_wide_contact<atl::int32>();
	verify_wide_contact<atl::int64>();
}

namespace {
	oracle_int oracle_from_128(atl::int128 value) {
		bool const is_negative = value < 0;
		atl::uint128 magnitude = is_negative ? atl::uint128(0) - atl::uint128(value) : atl::uint128(value);
		oracle_int result;
		for (unsigned i = 0; i < 8; ++i) {
			result = result + (oracle_int(std::uint32_t(magnitude & 65535)) << (16 * i));
			magnitude >>= 16;
		}
		return is_negative ? -result : result;
	}

	int oracle_sign(oracle_int const &value) { return value > 0 ? 1 : value < 0 ? -1
																				: 0; }

	using polygon_line = atl::geometry::line<std::int64_t>;

	oracle_int oracle_cross(pt a, pt b, pt c, pt d) {
		return (oracle_int(b.x) - a.x) * (oracle_int(d.y) - c.y)
			 - (oracle_int(b.y) - a.y) * (oracle_int(d.x) - c.x);
	}

	// Cramer's rule in unbounded integers, independent of the factored kernel.
	int crossing_oracle(polygon_line a, polygon_line b, polygon_line c) {
		auto const coefficients = [](polygon_line l) {
			return std::array<oracle_int, 3>{oracle_int(l.a.y) - l.b.y, oracle_int(l.b.x) - l.a.x,
											 oracle_int(l.a.x) * l.b.y - oracle_int(l.a.y) * l.b.x};
		};
		auto const u = coefficients(a), v = coefficients(b), w = coefficients(c);
		auto const denominator = u[0] * v[1] - v[0] * u[1];
		auto const numerator = w[0] * (u[1] * v[2] - v[1] * u[2])
							 + w[1] * (u[2] * v[0] - v[2] * u[0]) + w[2] * denominator;
		return oracle_sign(numerator) * oracle_sign(denominator);
	}

	bool parameter_oracle(polygon_line host, polygon_line a, polygon_line b) {
		auto const parameter = [&](polygon_line cut) {
			auto n = oracle_cross(cut.a, cut.b, cut.a, host.a);
			auto d = n - oracle_cross(cut.a, cut.b, cut.a, host.b);
			if (d < 0) {
				n = -n;
				d = -d;
			}
			return std::pair{n, d};
		};
		auto const u = parameter(a), v = parameter(b);
		return u.first * v.second < v.first * u.second;
	}

	void verify_large_pair(std::vector<pt> const &a, std::vector<pt> const &b) {
		auto const want = intersection_oracle(a, b);
		auto const got = intersection_of_convex_polygons(a, b);
		REQUIRE(got.empty() == want.empty());
		// Cleanup may coalesce vertices at large translations. Compare the
		// constructed boundaries geometrically, never using floats as sort keys.
		double scale = 1;
		for (auto const *cloud : {&a, &b})
			for (auto const v : *cloud) scale = (std::max)({scale, std::abs(double(v.x)), std::abs(double(v.y))});
		double const tolerance = 0x1p-40 * scale;
		for (auto const &q : want) {
			fpt const ref{double(q.x) / double(q.d), double(q.y) / double(q.d)};
			CHECK(distance_to_polygon(got, ref) <= tolerance);
		}
		for (auto const q : got) {
			REQUIRE(std::isfinite(q.x));
			REQUIRE(std::isfinite(q.y));
			CHECK(std::ranges::any_of(want, [&](convex_rational const &v) {
				return std::hypot(q.x - double(v.x) / double(v.d), q.y - double(v.y) / double(v.d)) <= tolerance;
			}));
		}
		auto const closest = closest_points_between_convex_hulls(a, b);
		auto const squared = closest_squared_oracle(a, b, !want.empty());
		double const expected_distance = std::sqrt(double(squared.n) / double(squared.d));
		CHECK(std::abs(std::hypot(closest[1].x - closest[0].x, closest[1].y - closest[0].y) - expected_distance) <= tolerance);
		auto const parts = union_of_convex_polygons(a, b);
		double total_area = 0;
		for (auto const &part : parts) {
			for (auto const v : part) {
				REQUIRE(std::isfinite(v.x));
				REQUIRE(std::isfinite(v.y));
			}
			total_area += polygon_area(part);
		}
		// Independent exact rational intersection area, accumulated as separate
		// fractions only after exact boundary ordering has been established.
		double reference_area = 0;
		for (std::size_t i = 0; i < want.size(); ++i) {
			auto const &u = want[i], &v = want[(i + 1) % want.size()];
			reference_area += double(u.x * v.y - u.y * v.x) / double(u.d * v.d) / 2;
		}
		double const area_a = polygon_area(convex_hull(a)), area_b = polygon_area(convex_hull(b));
		CHECK(std::abs(total_area + reference_area - area_a - area_b) <= 0x1p-36 * scale * scale);
	}
}

TEST_CASE("convex operations: wide products and quartic predicates vs unbounded integers", "[geometry]") {
	namespace gd = atl::geometry::detail;
	std::mt19937_64 rng(92821);
	for (int trial = 0; trial < 600; ++trial) {
		auto const operand = [&] {
			atl::int128 const magnitude = atl::int128(rng() >> 1) * (atl::int128(1) << 64) + rng();
			return rng() % 2 == 0 ? magnitude : -magnitude;
		};
		auto const a = operand(), b = operand(), c = trial % 3 == 0 ? a : operand(), d = trial % 3 == 0 ? b : operand();
		CHECK(gd::compare_wide_products(a, b, c, d) == oracle_sign(oracle_from_128(a) * oracle_from_128(b) - oracle_from_128(c) * oracle_from_128(d)));
		std::int64_t const radius = trial % 2 == 0 ? std::int64_t(1) << 60 : std::int64_t(1) << 28;
		auto const random_point = [&] { return pt{std::int64_t(rng() % std::uint64_t(2 * radius)) - radius, std::int64_t(rng() % std::uint64_t(2 * radius)) - radius}; };
		polygon_line const u{random_point(), random_point()}, v{random_point(), random_point()}, w{random_point(), random_point()};
		if (oracle_cross(u.a, u.b, v.a, v.b) == 0 || oracle_cross(u.a, u.b, w.a, w.b) == 0) continue;
		CHECK(gd::crossing_side(u, v, w, true) == crossing_oracle(u, v, w));
		CHECK(gd::param_before(u, v, w, true) == parameter_oracle(u, v, w));
		if (trial % 2 != 0) {
			CHECK(gd::crossing_side(u, v, w, false) == gd::crossing_side(u, v, w, true));
			CHECK(gd::param_before(u, v, w, false) == gd::param_before(u, v, w, true));
		}
	}
	atl::int128 const lowest = -(atl::int128(1) << 126) - (atl::int128(1) << 126);
	CHECK(gd::compare_wide_products(lowest, lowest, lowest, lowest + 1) > 0);
	CHECK(gd::compare_wide_products(lowest, 1, 0, lowest) < 0);
	CHECK(gd::compare_wide_products(lowest, 0, 0, lowest) == 0);
	std::int64_t const k = std::int64_t(1) << 60;
	polygon_line const u{{0, 0}, {k, k - 1}}, v{{0, 1}, {k - 1, k - 1}}, w{{-k, k}, {k, -k}};
	CHECK(gd::crossing_side(u, v, w, true) == crossing_oracle(u, v, w)); // determinant -1
	polygon_line const diagonal{{-k, -k}, {k, k}}, horizontal{{-k, 0}, {k, 0}};
	CHECK(gd::crossing_side(diagonal, w, horizontal, true) == 0); // concurrent
	CHECK_FALSE(gd::param_before(horizontal, diagonal, w, true)); // equal parameters
	CHECK_FALSE(gd::param_before(horizontal, w, diagonal, true));
}

TEST_CASE("convex operations: coordinates near 2^60 vs rational cloud oracle", "[geometry]") {
	std::mt19937_64 rng(83938);
	std::int64_t const k = std::int64_t(1) << 60;
	for (int trial = 0; trial < 70; ++trial) {
		std::vector<pt> a(1 + std::size_t(rng() % 5)), b(1 + std::size_t(rng() % 5));
		for (auto *cloud : {&a, &b})
			for (auto &v : *cloud) v = {std::int64_t(rng() % std::uint64_t(2 * k)) - k,
										std::int64_t(rng() % std::uint64_t(2 * k)) - k};
		verify_large_pair(a, b);
	}
	std::vector<pt> const square{{-k, -k}, {k, -k}, {k, k}, {-k, k}};
	verify_large_pair(square, {{-k, 0}, {k, 0}});
	verify_large_pair(square, {{-k, -k}, {k, k}});
	verify_large_pair(square, {{-k, k}, {k, -k}});
	std::int64_t const m = (std::int64_t(1) << 61) - 1;
	verify_large_pair(square, {{-m, -m}, {m, m}}); // equal entering and leaving parameters at corners
	verify_large_pair(square, {{m, 1}, {1, m}});   // point contact
	using atl::geometry::segment;
	namespace gd = atl::geometry::detail;
	auto const tangent = gd::segment_clipped_to_hull(segment<std::int64_t>{{m, 1}, {1, m}}, square, atl::geometry::exact{}, true);
	REQUIRE(tangent.size() == 1); // exactly equal lower/upper parameters
	CHECK(std::hypot(tangent[0].x - double(k), tangent[0].y - double(k)) <= 1024);
	CHECK(gd::segment_clipped_to_hull(segment<std::int64_t>{{m, 3}, {3, m}}, square, atl::geometry::exact{}, true).empty());
	verify_large_pair({{k, k}, {k + (k / 4), k}, {k, k + (k / 4)}},
					  {{k, k}, {k + (k / 4), k}, {k + (k / 4), k + (k / 4)}});
	verify_large_pair({{-k, -k}, {k, k - 1}}, {{0, 1}, {k - 1, k - 1}}); // near-parallel segments
	verify_large_pair({{-k, 0}, {k, 0}}, {{0, 0}, {k, 0}});
	verify_large_pair({{0, 0}}, {{0, 0}});
}

TEST_CASE("convex operations: forced wide clipping and deque agree with fast path", "[geometry]") {
	namespace gd = atl::geometry::detail;
	using atl::geometry::exact;
	std::mt19937_64 rng(72813);
	for (int trial = 0; trial < 100; ++trial) {
		auto a = convex_hull(random_hull_cloud(rng, 0, 0, 100000000, 10));
		auto b = convex_hull(random_hull_cloud(rng, 50000000, 0, 100000000, 10));
		gd::rotate_hull_to_low(a);
		gd::rotate_hull_to_low(b);
		auto const constraints = gd::merged_halfplanes(a, b, exact{});
		auto const fast_result = gd::halfplane_intersection(constraints, exact{}, false);
		auto const wide_result = gd::halfplane_intersection(constraints, exact{}, true);
		REQUIRE(fast_result.size() == wide_result.size());
		for (std::size_t i = 0; i < fast_result.size(); ++i) {
			CHECK(fast_result[i].boundary == wide_result[i].boundary);
			CHECK(fast_result[i].is_from_second == wide_result[i].is_from_second);
		}
		atl::geometry::segment<std::int64_t> const cut{{-200000000, 0}, {200000000, 0}};
		auto const clipped_fast = gd::segment_clipped_to_hull(cut, a, exact{}, false);
		auto const clipped_wide = gd::segment_clipped_to_hull(cut, a, exact{}, true);
		REQUIRE(clipped_fast.size() == clipped_wide.size());
		for (std::size_t i = 0; i < clipped_fast.size(); ++i)
			CHECK(std::hypot(clipped_fast[i].x - clipped_wide[i].x, clipped_fast[i].y - clipped_wide[i].y) <= 1e-7);
	}
}

TEST_CASE("convex contact: combined difference direction exceeds 2^62", "[geometry]") {
	std::int64_t const m = (std::int64_t(1) << 61) - 1;
	std::vector<pt> const a{{-m, -m}, {0, -m}, {0, m}, {-m, m}};
	std::vector<pt> const b{{0, -m}, {m, -m}, {m, m}, {0, m}};
	// A Minkowski boundary with parallel input edges combined: each support
	// advances in both summands. All vertices satisfy the predicate bound,
	// but the vertical edge through the origin has direction 4m > 2^62.
	atl::geometry::detail::minkowski_difference<std::int64_t> d;
	d.a_hull = a;
	d.negated_b_hull = a;
	d.boundary = {{-2 * m, -2 * m}, {0, -2 * m}, {0, 2 * m}, {-2 * m, 2 * m}};
	d.supports = {{{0, 0}}, {{1, 1}}, {{2, 2}}, {{3, 3}}};
	auto const got = atl::geometry::detail::boundary_contact(d, atl::geometry::exact{});
	REQUIRE(got.kind == atl::geometry::intersection_kind::along_segment);
	auto const want = intersection_oracle(a, b);
	REQUIRE(want.size() == 2);
	for (auto const q : {got.along.a, got.along.b})
		CHECK(std::ranges::any_of(want, [&](convex_rational const &v) {
			return v.x == oracle_int(q.x) * v.d && v.y == oracle_int(q.y) * v.d;
		}));
}
