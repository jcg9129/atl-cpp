#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/convex_polygon_operations.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/point_location.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/polygon_overlay.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <random>
#include <ranges>
#include <vector>

using atl::float64;
using atl::geometry::arrangement;
using atl::geometry::difference_area;
using atl::geometry::intersection_area;
using atl::geometry::overlay_of;
using atl::geometry::point;
using atl::geometry::polygon_area;
using atl::geometry::symmetric_difference_area;
using atl::geometry::union_area;

namespace {
	using lp = point<std::int64_t>;

	std::vector<lp> star(std::mt19937_64 &rng, std::int64_t const ox,
						 std::int64_t const oy) {
		std::vector<lp> raw;
		for (int k = 0; k < 14; ++k)
			raw.push_back({std::int64_t(rng() % 25) - 12, std::int64_t(rng() % 25) - 12});
		std::erase_if(raw, [](lp const p) { return p.x == 0 && p.y == 0; });
		std::ranges::sort(raw, atl::geometry::polar_angle_less{});
		std::vector<lp> poly;
		for (auto const p : raw)
			if (poly.empty() || !(poly.back() == p)) poly.push_back(p);
		for (auto &p : poly) {
			p.x += ox;
			p.y += oy;
		}
		return poly;
	}

	bool usable(std::vector<lp> const &poly) {
		return poly.size() >= 3 && atl::geometry::polygon_area2(poly) != 0
			&& atl::geometry::simplicity_of(poly).is_simple;
	}
}

TEST_CASE("polygon booleans: handcrafted", "[geometry]") {
	std::vector<lp> const big{{0, 0}, {8, 0}, {8, 8}, {0, 8}};
	std::vector<lp> const shifted{{4, 4}, {12, 4}, {12, 12}, {4, 12}};
	std::vector<lp> const inner{{2, 2}, {5, 2}, {5, 5}, {2, 5}};
	std::vector<lp> const offset{{20, 20}, {24, 20}, {24, 24}, {20, 24}};
	std::vector<lp> const touching{{8, 0}, {12, 0}, {12, 8}, {8, 8}};

	// classic partial overlap of squares
	REQUIRE(std::abs(intersection_area(big, shifted) - 16.0) <= 1e-9);
	REQUIRE(std::abs(union_area(big, shifted) - 112.0) <= 1e-9);
	REQUIRE(std::abs(difference_area(big, shifted) - 48.0) <= 1e-9);
	REQUIRE(std::abs(symmetric_difference_area(big, shifted) - 96.0) <= 1e-9);
	// nested: the difference is a RING (a face with a hole)
	REQUIRE(std::abs(intersection_area(big, inner) - 9.0) <= 1e-9);
	REQUIRE(std::abs(union_area(big, inner) - 64.0) <= 1e-9);
	REQUIRE(std::abs(difference_area(big, inner) - 55.0) <= 1e-9);
	// disjoint and edge-touching: empty intersection, additive union
	REQUIRE(intersection_area(big, offset) == 0.0);
	REQUIRE(std::abs(union_area(big, offset) - 80.0) <= 1e-9);
	REQUIRE(intersection_area(big, touching) == 0.0);
	REQUIRE(std::abs(union_area(big, touching) - 96.0) <= 1e-9);
	// identical polygons (every edge shared)
	REQUIRE(std::abs(intersection_area(big, big) - 64.0) <= 1e-9);
	REQUIRE(std::abs(union_area(big, big) - 64.0) <= 1e-9);
	REQUIRE(difference_area(big, big) <= 1e-9);
	// windings must not matter
	std::vector<lp> const big_cw(big.rbegin(), big.rend());
	REQUIRE(std::abs(intersection_area(big_cw, shifted) - 16.0) <= 1e-9);

	// the overlay itself: unbounded face outside both, flags coherent
	auto const ov = overlay_of(big, inner);
	REQUIRE(ov.in_first[arrangement::unbounded_face] == 0);
	REQUIRE(ov.in_second[arrangement::unbounded_face] == 0);
	bool ring_seen = false;
	for (std::size_t f = 0; f < ov.arr.faces.size(); ++f)
		if (ov.in_first[f] && !ov.in_second[f])
			ring_seen = ring_seen || !ov.arr.faces[f].holes.empty();
	REQUIRE(ring_seen); // the nested difference face carries its hole
}

TEST_CASE("polygon booleans: convex differential and identities", "[geometry]") {
	auto seed = GENERATE(take(15, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 40; ++it) {
		// two random convex polygons via hulls of point clouds
		std::vector<lp> cloud_a, cloud_b;
		for (int k = 0; k < 20; ++k) {
			cloud_a.push_back({std::int64_t(rng() % 30), std::int64_t(rng() % 30)});
			cloud_b.push_back(
				{std::int64_t(rng() % 30) + 10, std::int64_t(rng() % 30) + 10});
		}
		auto const a = atl::geometry::convex_hull(cloud_a);
		auto const b = atl::geometry::convex_hull(cloud_b);
		if (a.size() < 3 || b.size() < 3) continue;
		// the overlay's intersection area must match the convex clipper's
		auto const common =
			atl::geometry::intersection_of_convex_polygons(a, b);
		float64 const convex_area =
			common.size() < 3 ? 0.0 : polygon_area(common);
		REQUIRE(std::abs(intersection_area(a, b) - convex_area)
				<= 1e-6 * (1 + convex_area));
	}
	// inclusion-exclusion identities on random star pairs
	for (int it = 0; it < 40; ++it) {
		auto const a = star(rng, 0, 0);
		auto const b = star(rng, std::int64_t(rng() % 15) - 7,
							std::int64_t(rng() % 15) - 7);
		if (!usable(a) || !usable(b)) continue;
		float64 const both = intersection_area(a, b);
		float64 const either = union_area(a, b);
		float64 const only_a = difference_area(a, b);
		float64 const sym = symmetric_difference_area(a, b);
		float64 const area_a = polygon_area(a), area_b = polygon_area(b);
		float64 const scale = 1 + area_a + area_b;
		REQUIRE(std::abs(both + either - area_a - area_b) <= 1e-6 * scale);
		REQUIRE(std::abs(only_a - (area_a - both)) <= 1e-6 * scale);
		REQUIRE(std::abs(sym - (either - both)) <= 1e-6 * scale);
		REQUIRE(both <= std::min(area_a, area_b) + 1e-6 * scale);
		REQUIRE(either + 1e-6 * scale >= std::max(area_a, area_b));
	}
}

TEST_CASE("polygon booleans: double coordinates parity spot check", "[geometry]") {
	using pd = point<double>;
	std::vector<pd> const a{{0, 0}, {2, 0}, {2, 2}, {0, 2}};
	std::vector<pd> const b{{1, 1}, {3, 1}, {3, 3}, {1, 3}};
	REQUIRE(std::abs(intersection_area(a, b) - 1.0) <= 1e-9);
	REQUIRE(std::abs(union_area(a, b) - 7.0) <= 1e-9);
}

TEST_CASE("polygon booleans: areas do not depend on the distance from the origin", "[geometry]") {
	// Absolute-position shoelace products at 3e9 have magnitude about 1e19;
	// local anchors preserve a face of area 2 against that cancellation.
	using atl::int64;
	using atl::geometry::intersection_area;
	using atl::geometry::union_area;
	std::vector<point<int64>> const a{{0, 0}, {4, 0}, {0, 4}}, b{{1, 1}, {5, 1}, {1, 5}};
	REQUIRE(std::abs(intersection_area(a, b) - 2.0) < 1e-12);
	for (int64 const offset : {int64(3'000'000'001), int64(-4'000'000'000), int64(1) << 31}) {
		auto moved_a = a, moved_b = b;
		for (auto &q : moved_a) q = {q.x + offset, q.y - offset};
		for (auto &q : moved_b) q = {q.x + offset, q.y - offset};
		REQUIRE(std::abs(intersection_area(moved_a, moved_b) - 2.0) < 1e-6);
		REQUIRE(std::abs(union_area(moved_a, moved_b) - 14.0) < 1e-6);
	}
}

namespace {
	// mixed coordinate types are rejected by overload resolution, not inside the body
	template<typename A, typename B>
	concept has_intersection_area = requires(A const &a, B const &b) { atl::geometry::intersection_area(a, b); };
	template<typename A, typename B>
	concept has_symmetric_difference_area =
		requires(A const &a, B const &b) { atl::geometry::symmetric_difference_area(a, b); };
	using int_polygon = std::vector<point<std::int64_t>>;
	using double_polygon = std::vector<point<double>>;
	static_assert(!has_intersection_area<int_polygon, double_polygon>);
	static_assert(!has_symmetric_difference_area<int_polygon, double_polygon>);
	static_assert(has_intersection_area<int_polygon, int_polygon>);
}

TEST_CASE("polygon booleans: the overlay carries its segments for point location", "[geometry]") {
	using atl::geometry::containment;
	using atl::geometry::contains;
	std::mt19937_64 rng(2026);
	int located = 0;
	for (int it = 0; it < 30; ++it) {
		auto const a = star(rng, 0, 0);
		auto const b = star(rng, std::int64_t(rng() % 9) - 4, std::int64_t(rng() % 9) - 4);
		if (!usable(a) || !usable(b)) continue;
		auto const ov = overlay_of(a, b);
		REQUIRE(ov.segments.size() == a.size() + b.size());
		REQUIRE(ov.first_edge_count == atl::uint32(a.size()));
		REQUIRE(ov.segments[0].a == a[0]);
		REQUIRE(ov.segments[ov.first_edge_count].a == b[0]);
		atl::geometry::point_location const where(ov.arr, ov.segments);
		for (std::int64_t x = -16; x <= 16; ++x)
			for (std::int64_t y = -16; y <= 16; ++y) {
				lp const q{x, y};
				auto const at = where.location_of(q);
				if (at.kind != atl::geometry::location_kind::face) continue;
				++located;
				REQUIRE(bool(ov.in_first[at.index]) == (contains(a, q) == containment::inside));
				REQUIRE(bool(ov.in_second[at.index]) == (contains(b, q) == containment::inside));
			}
	}
	REQUIRE(located > 0);
}

TEST_CASE("polygon booleans: area overloads read one overlay", "[geometry]") {
	std::mt19937_64 rng(77);
	for (int it = 0; it < 30; ++it) {
		auto const a = star(rng, 0, 0);
		auto const b = star(rng, std::int64_t(rng() % 15) - 7, std::int64_t(rng() % 15) - 7);
		if (!usable(a) || !usable(b)) continue;
		auto const ov = overlay_of(a, b);
		REQUIRE(std::abs(intersection_area(ov) - intersection_area(a, b)) <= 1e-10 * (1 + std::abs(intersection_area(ov))));
		REQUIRE(std::abs(union_area(ov) - union_area(a, b)) <= 1e-10 * (1 + std::abs(union_area(ov))));
		REQUIRE(std::abs(difference_area(ov) - difference_area(a, b)) <= 1e-10 * (1 + std::abs(difference_area(ov))));
		REQUIRE(std::abs(symmetric_difference_area(ov) - symmetric_difference_area(a, b)) <= 1e-10 * (1 + std::abs(symmetric_difference_area(ov))));
	}
	// R && admits rvalues and views alike
	std::vector<lp> const sq{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
	REQUIRE(std::abs(intersection_area(std::vector<lp>(sq), sq | std::views::reverse) - 16.0) <= 1e-9);
}

namespace {
	// Reduced rational arithmetic keeps this oracle independent of floating clipping and the segment sweep.
	struct oracle_fraction {
		atl::int128 n = 0, d = 1;

		explicit oracle_fraction(atl::int128 numerator = 0, atl::int128 denominator = 1) : n(numerator), d(denominator) {
			if (d < 0) {
				n = -n;
				d = -d;
			}
			atl::int128 divisor = n < 0 ? -n : n, remainder = d;
			while (remainder != 0) {
				auto const next = divisor % remainder;
				divisor = remainder;
				remainder = next;
			}
			n /= divisor;
			d /= divisor;
		}

		friend oracle_fraction operator+(oracle_fraction a, oracle_fraction b) { return oracle_fraction(a.n * b.d + b.n * a.d, a.d * b.d); }

		friend oracle_fraction operator-(oracle_fraction a, oracle_fraction b) { return oracle_fraction(a.n * b.d - b.n * a.d, a.d * b.d); }

		friend oracle_fraction operator*(oracle_fraction a, oracle_fraction b) { return oracle_fraction(a.n * b.n, a.d * b.d); }

		friend oracle_fraction operator/(oracle_fraction a, oracle_fraction b) { return oracle_fraction(a.n * b.d, a.d * b.n); }
	};

	struct oracle_vertex {
		oracle_fraction x, y;
	};

	oracle_fraction triangle_overlap(std::array<lp, 3> const &a, std::array<lp, 3> const &b) {
		std::vector<oracle_vertex> polygon;
		for (auto const p : a) polygon.push_back({oracle_fraction(p.x), oracle_fraction(p.y)});
		for (std::size_t e = 0; e < 3; ++e) {
			auto const u = b[e], v = b[(e + 1) % 3];
			auto const side = [&](oracle_vertex const &p) {
				return oracle_fraction(v.x - u.x) * (p.y - oracle_fraction(u.y))
					 - oracle_fraction(v.y - u.y) * (p.x - oracle_fraction(u.x));
			};
			std::vector<oracle_vertex> clipped;
			for (std::size_t i = 0; i < polygon.size(); ++i) {
				auto const p = polygon[i], q = polygon[(i + 1) % polygon.size()];
				auto const sp = side(p), sq = side(q);
				if (sp.n >= 0) clipped.push_back(p);
				if ((sp.n < 0 && sq.n > 0) || (sp.n > 0 && sq.n < 0)) {
					auto const t = sp / (sp - sq);
					clipped.push_back({p.x + t * (q.x - p.x), p.y + t * (q.y - p.y)});
				}
			}
			polygon = std::move(clipped);
		}
		oracle_fraction twice;
		for (std::size_t i = 0; i < polygon.size(); ++i) {
			auto const p = polygon[i], q = polygon[(i + 1) % polygon.size()];
			twice = twice + p.x * q.y - p.y * q.x;
		}
		return twice / oracle_fraction(2);
	}
}

TEST_CASE("polygon overlay: random triangles against reduced-rational exhaustive clipping", "[geometry]") {
	std::mt19937_64 rng(887);
	for (int iteration = 0; iteration < 250; ++iteration) {
		std::array<lp, 3> a{}, b{};
		for (auto *triangle : {&a, &b}) {
			for (auto &p : *triangle) p = {std::int64_t(rng() % 7) - 3, std::int64_t(rng() % 7) - 3};
			auto const &t = *triangle;
			std::int64_t const determinant = (t[1].x - t[0].x) * (t[2].y - t[0].y) - (t[1].y - t[0].y) * (t[2].x - t[0].x);
			if (determinant < 0) std::swap((*triangle)[1], (*triangle)[2]);
		}
		auto const area2 = [](std::array<lp, 3> const &t) {
			return (t[1].x - t[0].x) * (t[2].y - t[0].y) - (t[1].y - t[0].y) * (t[2].x - t[0].x);
		};
		if (area2(a) == 0 || area2(b) == 0) continue;
		auto const fraction = triangle_overlap(a, b);
		double const common = double(fraction.n) / double(fraction.d);
		double const first = double(area2(a)) / 2, second = double(area2(b)) / 2;
		auto const overlay = overlay_of(a, b);
		REQUIRE(std::abs(intersection_area(overlay) - common) < 1e-9);
		auto const floating_parity = [&]<atl::geometry::coordinate T>()
			requires std::floating_point<T>
		{
			std::array<point<T>, 3> first_points{}, second_points{};
			for (std::size_t i = 0; i < 3; ++i) {
				first_points[i] = {T(a[i].x), T(a[i].y)};
				second_points[i] = {T(b[i].x), T(b[i].y)};
			}
			REQUIRE(std::abs(intersection_area(first_points, second_points) - common) < 1e-9);
		};
		floating_parity.template operator()<float>();
		floating_parity.template operator()<double>();
		REQUIRE(std::abs(union_area(overlay) - (first + second - common)) < 1e-9);
		REQUIRE(std::abs(difference_area(overlay) - (first - common)) < 1e-9);
		REQUIRE(std::abs(symmetric_difference_area(overlay) - (first + second - 2 * common)) < 1e-9);
	}
}

TEST_CASE("polygon overlay: int64 bound and floating tier", "[geometry]") {
	std::int64_t const bound = (std::int64_t(1) << 32) - 1;
	std::array const polygon{lp{-bound, -bound}, lp{bound, -bound}, lp{bound, bound}, lp{-bound, bound}};
	double const side = 2 * double(bound), expected = side * side;
	REQUIRE(std::abs(intersection_area(polygon, polygon) - expected) <= 1e-12 * expected);
	using p32 = point<std::int32_t>;
	std::int32_t const lo = (std::numeric_limits<std::int32_t>::min)();
	std::int32_t const hi = (std::numeric_limits<std::int32_t>::max)();
	std::array const full{p32{lo, lo}, p32{hi, lo}, p32{hi, hi}, p32{lo, hi}};
	double const width = double(hi) - double(lo);
	REQUIRE(std::abs(intersection_area(full, full) - width * width) <= 1e-12 * width * width);
	using pf = point<float>;
	std::array const floating{pf{0, 0}, pf{4, 0}, pf{4, 4}, pf{0, 4}};
	REQUIRE(std::abs(intersection_area(floating, floating) - 16) <= 1e-9);
}
