#include <atl/experimental/geometry/box_variants.hpp>
#include <atl/geometry/box.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::int128;
using atl::geometry::box;
using atl::geometry::union_area;

namespace {
	using bx = box<std::int64_t>;
}

TEST_CASE("union_area: handcrafted", "[geometry]") {
	// single, disjoint, overlapping, nested, touching, duplicates
	REQUIRE(union_area(std::vector<bx>{}) == 0);
	REQUIRE(union_area(std::vector<bx>{{{0, 0}, {4, 3}}}) == 12);
	REQUIRE(union_area(std::vector<bx>{{{0, 0}, {2, 2}}, {{5, 5}, {7, 7}}}) == 8);
	REQUIRE(union_area(std::vector<bx>{{{0, 0}, {4, 4}}, {{2, 2}, {6, 6}}}) == 28);
	REQUIRE(union_area(std::vector<bx>{{{0, 0}, {10, 10}}, {{2, 2}, {5, 5}}}) == 100);
	// edge-touching boxes: no seam, no double count
	REQUIRE(union_area(std::vector<bx>{{{0, 0}, {2, 2}}, {{2, 0}, {4, 2}}}) == 8);
	REQUIRE(union_area(std::vector<bx>{{{0, 0}, {3, 3}}, {{0, 0}, {3, 3}}}) == 9);
	// degenerate lines and points contribute nothing
	REQUIRE(union_area(std::vector<bx>{{{0, 0}, {0, 5}}, {{1, 1}, {4, 1}}, {{2, 2}, {2, 2}}})
			== 0);
	// exactness at the coordinate bound: two giant overlapping halves
	std::int64_t const big = (std::int64_t(1) << 61);
	int128 const width = int128(2) * big;
	REQUIRE(union_area(std::vector<bx>{{{-big, -big}, {big, 0}},
									   {{-big, -1}, {big, big}}})
			== width * width);
	REQUIRE(atl::experimental::union_area_by_top_down_tree(std::vector<bx>{{{-big, -big}, {big, 0}}, {{-big, -1}, {big, big}}})
			== width * width);
}

TEST_CASE("union_area: random grids vs the cell brute", "[geometry]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 200; ++it) {
		std::size_t const n = static_cast<std::size_t>(1 + rng() % 10);
		std::int64_t const span = 2 + std::int64_t(rng() % 20);
		std::vector<bx> boxes;
		for (std::size_t k = 0; k < n; ++k) {
			std::int64_t const x1 = std::int64_t(rng() % std::uint64_t(span));
			std::int64_t const y1 = std::int64_t(rng() % std::uint64_t(span));
			std::int64_t const x2 = x1 + std::int64_t(rng() % 6);
			std::int64_t const y2 = y1 + std::int64_t(rng() % 6);
			boxes.push_back({{x1, y1}, {x2, y2}});
		}
		int128 brute = 0;
		for (std::int64_t x = 0; x < span + 6; ++x)
			for (std::int64_t y = 0; y < span + 6; ++y)
				for (auto const &b : boxes)
					if (b.lo.x <= x && x + 1 <= b.hi.x && b.lo.y <= y
						&& y + 1 <= b.hi.y) {
						++brute;
						break;
					}
		REQUIRE(union_area(boxes) == brute);
		REQUIRE(atl::experimental::union_area_by_top_down_tree(boxes) == brute);
		// the narrower and floating instantiations on the same small boxes
		std::vector<box<std::int16_t>> narrow;
		std::vector<box<double>> floating;
		for (auto const &b : boxes) {
			narrow.push_back({{std::int16_t(b.lo.x), std::int16_t(b.lo.y)}, {std::int16_t(b.hi.x), std::int16_t(b.hi.y)}});
			floating.push_back({{double(b.lo.x), double(b.lo.y)}, {double(b.hi.x), double(b.hi.y)}});
		}
		REQUIRE(union_area(narrow) == brute);
		REQUIRE(union_area(floating) == double(brute)); // small integers: exact in double
	}
}

TEST_CASE("union_area: the raced contenders agree at size and at the bound", "[geometry]") {
	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	// thousands of boxes: deep trees, non-power-of-two strip counts; offsets
	// up to 2^61 so covered lengths need the full unsigned range
	std::size_t const n = static_cast<std::size_t>(1 + rng() % 5000);
	std::int64_t const scale = std::int64_t(1) << (rng() % 2 ? 61 : 10);
	std::vector<bx> boxes;
	for (std::size_t k = 0; k < n; ++k) {
		auto const at = [&] { return std::int64_t(rng() % 4001) - 2000; };
		std::int64_t x1 = at(), x2 = at(), y1 = at(), y2 = at();
		if (x1 > x2) std::swap(x1, x2);
		if (y1 > y2) std::swap(y1, y2);
		boxes.push_back({{x1 * (scale / 2048), y1 * (scale / 2048)}, {x2 * (scale / 2048), y2 * (scale / 2048)}});
	}
	auto const expected = union_area(boxes);
	REQUIRE(atl::experimental::union_area_by_top_down_tree(boxes) == expected);
	std::vector<box<std::int32_t>> narrow;
	for (auto const &b : boxes)
		narrow.push_back({{std::int32_t(b.lo.x % 1000), std::int32_t(b.lo.y % 1000)},
						  {std::int32_t(b.lo.x % 1000) + 7, std::int32_t(b.lo.y % 1000) + 3}});
	REQUIRE(atl::experimental::union_area_by_top_down_tree(narrow) == union_area(narrow));
}

TEST_CASE("union_area: double coordinates parity spot check", "[geometry]") {
	using bd = box<double>;
	REQUIRE(union_area(std::vector<bd>{{{0, 0}, {1.5, 1}}, {{0.5, 0}, {2, 1}}})
			== 2.0);
}

TEST_CASE("union_area: int16 over the whole coordinate range", "[geometry]") {
	// Lengths up to 65535 in uint16 and the area in int64.
	using b8 = box<std::int16_t>;
	REQUIRE(union_area(std::vector<b8>{{{-32768, -32768}, {32767, 32767}}}) == 65535LL * 65535);
	REQUIRE(union_area(std::vector<b8>{{{-32768, -32768}, {32767, 0}}, {{-32768, -1}, {32767, 32767}}}) == 65535LL * 65535);
	REQUIRE(union_area(std::vector<b8>{{{-32768, -32768}, {0, 0}}, {{0, 0}, {32767, 32767}}}) == 32768LL * 32768 + 32767LL * 32767); // corner touch
	REQUIRE(atl::experimental::union_area_by_top_down_tree(std::vector<b8>{{{-32768, -32768}, {32767, 32767}}, {{-5, -5}, {5, 5}}}) == 65535LL * 65535);
}

TEST_CASE("union_area: degenerate inputs", "[geometry]") {
	// all boxes of zero area, all equal, one box
	REQUIRE(union_area(std::vector<bx>{{{3, 3}, {3, 3}}, {{3, 3}, {3, 3}}}) == 0);
	REQUIRE(union_area(std::vector<bx>(5, bx{{-2, -1}, {2, 1}})) == 8);
	REQUIRE(union_area(std::vector<bx>{{{-2, -1}, {2, 1}}}) == 8);
	REQUIRE(union_area(std::vector<box<double>>{}) == 0.0);
	// an unordered box is a precondition break
	CHECK(atl_test::aborts([] { (void)union_area(std::vector<bx>{{{1, 0}, {0, 1}}}); }));
	// the int64 bound is asserted
	CHECK(atl_test::aborts([] { (void)union_area(std::vector<bx>{{{0, 0}, {std::int64_t(1) << 62, 1}}}); }));
}

TEST_CASE("union_area: float lengths accumulate in double", "[geometry]") {
	std::vector<box<float>> const boxes{{{0, -0x1p30f}, {1, -1}}, {{0, 0}, {1, 0x1p30f}}};
	REQUIRE(union_area(boxes) == 0x1p31 - 1);
	REQUIRE(atl::experimental::union_area_by_top_down_tree(boxes) == 0x1p31 - 1);
}
