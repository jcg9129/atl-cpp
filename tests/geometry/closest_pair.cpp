#include <atl/experimental/geometry/closest_pair_variants.hpp>
#include <atl/geometry/closest_pair.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/hash.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

using atl::geometry::closest_pair;
using atl::geometry::closest_pair_indices;
using atl::geometry::point;

namespace {
	using pt = point<std::int64_t>;

	atl::int128 brute(std::vector<pt> const &pts) {
		atl::int128 best = (std::numeric_limits<atl::int128>::max)();
		for (std::size_t i = 0; i < pts.size(); ++i)
			for (std::size_t j = i + 1; j < pts.size(); ++j) {
				atl::int128 const dx = atl::int128(pts[i].x) - atl::int128(pts[j].x);
				atl::int128 const dy = atl::int128(pts[i].y) - atl::int128(pts[j].y);
				best = (std::min)(best, dx * dx + dy * dy);
			}
		return best;
	}

	void verify(std::vector<pt> const &pts, bool const is_double_exact = true) {
		auto const expected = brute(pts);
		auto const [i, j] = closest_pair_indices(pts);
		REQUIRE(i < pts.size());
		REQUIRE(j < pts.size());
		REQUIRE(i != j);
		REQUIRE(atl::int128(squared_distance(pts[i], pts[j])) == expected);
		auto const [a, b] = closest_pair(pts);
		REQUIRE(atl::int128(squared_distance(a, b)) == expected);
		// the raced contenders (experimental) must find the same distance
		auto const [si, sj] = atl::experimental::closest_pair_indices_by_divide_and_conquer(pts);
		REQUIRE(atl::int128(squared_distance(pts[si], pts[sj])) == expected);
		// double parity, and the grid contender, whose float64 cells need
		// coordinates that convert exactly (|x|, |y| <= 2^53)
		if (!is_double_exact) return;
		auto const [gi, gj] = atl::experimental::closest_pair_indices_by_randomized_grid(pts);
		REQUIRE(atl::int128(squared_distance(pts[gi], pts[gj])) == expected);
		std::vector<point<double>> pd;
		for (auto const q : pts) pd.push_back({double(q.x), double(q.y)});
		auto const [di, dj] = closest_pair_indices(pd);
		REQUIRE(atl::int128(squared_distance(pts[di], pts[dj])) == expected);
	}
}

TEST_CASE("closest_pair: random shapes vs quadratic brute", "[geometry]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	atl::set_hash_seed(seed); // the grid's shuffle replays from the Catch seed
	std::mt19937_64 rng(seed);
	std::size_t const n = static_cast<std::size_t>(2 + rng() % 300);
	std::vector<pt> pts(n);
	switch (rng() % 5) {
	case 0: // uniform
		for (auto &p : pts)
			p = {std::int64_t(rng() % 2001) - 1000, std::int64_t(rng() % 2001) - 1000};
		break;
	case 1: // heavy duplicates (distance 0)
		for (auto &p : pts) p = {std::int64_t(rng() % 15), std::int64_t(rng() % 15)};
		break;
	case 2: // collinear — horizontally, or stacked on one shared x (duplicates
			// coexisting in the window set, evictions must hit one tuple only)
		for (std::size_t k = 0; k < n; ++k)
			pts[k] = rng() % 2 ? pt{std::int64_t(rng() % 5000), 0}
							   : pt{7, std::int64_t(rng() % 40)};
		break;
	case 3: // two tight clusters distant apart
		for (auto &p : pts) {
			std::int64_t const cx = rng() % 2 ? 0 : 1'000'000;
			p = {cx + std::int64_t(rng() % 30), std::int64_t(rng() % 30)};
		}
		break;
	default: // grid points (many equal distances: tie stress)
		for (auto &p : pts) p = {std::int64_t(rng() % 20) * 10, std::int64_t(rng() % 20) * 10};
	}
	verify(pts);
}

TEST_CASE("closest_pair: degenerates and exact spots", "[geometry]") {
	verify({{0, 0}, {0, 0}});						  // the minimal duplicate pair
	verify({{5, 5}, {9, 9}});						  // n == 2
	verify({{0, 0}, {1'000'000, 0}, {0, 1'000'000}}); // sparse triangle
	auto const [i, j] = closest_pair_indices(std::vector<pt>{{0, 0}, {100, 100}, {3, 4}});
	REQUIRE(atl::int64(squared_distance(pt{0, 0}, pt{3, 4})) == 25);
	REQUIRE(((i == 0 && j == 2) || (i == 2 && j == 0)));
}

TEST_CASE("closest_pair: int64 near the coordinate bound, constructive", "[geometry]") {
	std::mt19937_64 rng(20260827);
	std::int64_t const base = std::int64_t(1) << 61;
	std::vector<pt> pts;
	// far-flung points at huge coordinates, plus one planted close pair
	for (int k = 0; k < 200; ++k)
		pts.push_back({std::int64_t(rng() % (std::uint64_t(1) << 61)) - base / 2,
					   std::int64_t(rng() % (std::uint64_t(1) << 61)) - base / 2});
	pts.push_back({base, base});
	pts.push_back({base + 3, base + 4});
	auto const [i, j] = closest_pair_indices(pts);
	REQUIRE(atl::int128(squared_distance(pts[i], pts[j])) <= 25);
	verify(pts, false); // the contenders too, against the brute
}

TEST_CASE("closest_pair: contenders at mid size and near the bound vs brute", "[geometry]") {
	auto seed = GENERATE(take(6, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	atl::set_hash_seed(seed);
	std::mt19937_64 rng(seed);
	// 2000 points: deep enough for the divide and conquer's strip and merge
	// levels above its 16-point leaves; offsets up to 2^61 in a thin strip
	std::int64_t const base = (std::int64_t(1) << 61) * (rng() % 2 ? 1 : -1);
	std::uint64_t const spread = std::uint64_t(1) << (10 + rng() % 50);
	std::vector<pt> pts(2000);
	for (auto &p : pts)
		p = {base / 2 + std::int64_t(rng() % 1000), base / 2 + std::int64_t(rng() % spread)};
	verify(pts, false);
}

TEST_CASE("closest_pair: the grid's cell index saturates for double coordinates", "[geometry]") {
	// cell side ~ 5e-4 against coordinates up to 1e150: the cell quotient is
	// far beyond int64, so the saturating fold must clamp it, and distant
	// points share clamped cells (harmless hash collisions)
	atl::set_hash_seed(7);
	std::vector<point<double>> pts{{1e150, -1e150}, {-1e150, 1e150}, {1e150, 1e150}, {0.25, 0.5}, {0.25, 0.501}};
	for (int k = 0; k < 200; ++k) pts.push_back({1e140 * k, -1e140 * k});
	auto const [i, j] = atl::experimental::closest_pair_indices_by_randomized_grid(pts);
	REQUIRE(((std::min)(i, j) == 3 && (std::max)(i, j) == 4));
	auto const [u, v] = closest_pair_indices(pts);
	REQUIRE(((std::min)(u, v) == 3 && (std::max)(u, v) == 4));
}

TEST_CASE("closest_pair: the window never drops a candidate above 2^53", "[geometry]") {
	// a double y-window would round by up to 512 at |y| ~ 2^61 against a
	// slack of 1, so its lower bound would skip the true partner and return
	// the far-apart first pair
	using atl::int64;
	using atl::geometry::squared_distance;
	int64 const distant = int64(1) << 61;
	std::vector<point<int64>> const four{{0, 0}, {10, 0}, {distant, distant + 254}, {distant + 4, distant + 257}};
	auto const [i, j] = closest_pair_indices(four);
	REQUIRE((squared_distance(four[i], four[j]) == 25)); // int128: compared outside Catch

	auto seed = GENERATE(take(40, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<point<int64>> pts;
	// a loose pair near the origin fixes a modest best distance early; then
	// clusters far away whose members sit a few hundred apart in y
	pts.push_back({0, 0});
	pts.push_back({int64(5 + rng() % 2000), 0});
	std::size_t const clusters = static_cast<std::size_t>(1 + rng() % 4);
	for (std::size_t c = 0; c < clusters; ++c) {
		int64 const base_x = int64(1) << (54 + rng() % 8), base_y = (int64(1) << (54 + rng() % 8)) * (rng() % 2 ? 1 : -1);
		std::size_t const members = static_cast<std::size_t>(2 + rng() % 6);
		for (std::size_t k = 0; k < members; ++k)
			pts.push_back({base_x + int64(rng() % 40), base_y + int64(rng() % 1200)});
	}
	auto expected = squared_distance(pts[0], pts[1]);
	for (std::size_t a = 0; a < pts.size(); ++a)
		for (std::size_t b = a + 1; b < pts.size(); ++b) expected = (std::min)(expected, squared_distance(pts[a], pts[b]));
	auto const [u, v] = closest_pair_indices(pts);
	REQUIRE(u != v);
	REQUIRE((squared_distance(pts[u], pts[v]) == expected));
}

TEST_CASE("closest_pair: an isolated point beyond the int64 bound is asserted", "[geometry]") {
	using p64 = point<std::int64_t>;
	std::int64_t const bound = std::int64_t(1) << 62;
	// the far point never becomes a window candidate, yet its x gap is squared
	std::vector<p64> const pts{{0, 0}, {1, 0}, {bound, 0}};
	CHECK(atl_test::aborts([&] { (void)closest_pair_indices(pts); }));
}

TEST_CASE("closest_pair: 1e6 points, duplicates and collinear", "[geometry]") {
	constexpr std::size_t n = 1'000'000;
	std::vector<pt> same(n, pt{-7, 9});
	auto const [a, b] = closest_pair_indices(same);
	REQUIRE(a != b);
	// one vertical line, distinct y spaced 3 apart except one pair 2 apart:
	// the sweep's window and the divide and conquer's recursion at full size
	std::vector<pt> line(n);
	for (std::size_t k = 0; k < n; ++k) line[k] = {5, 3 * std::int64_t(k)};
	line[n / 2].y -= 1;
	std::mt19937_64 rng(3);
	std::ranges::shuffle(line, rng);
	auto const [i, j] = closest_pair_indices(line);
	REQUIRE(atl::int128(squared_distance(line[i], line[j])) == 4);
	auto const [di, dj] = atl::experimental::closest_pair_indices_by_divide_and_conquer(line);
	REQUIRE(atl::int128(squared_distance(line[di], line[dj])) == 4);
	// a sorted horizontal line
	std::vector<pt> row(n);
	for (std::size_t k = 0; k < n; ++k) row[k] = {2 * std::int64_t(k), 0};
	row.back().x -= 1;
	auto const [u, v] = closest_pair_indices(row);
	REQUIRE(atl::int128(squared_distance(row[u], row[v])) == 1);
}

TEST_CASE("closest_pair: narrow and floating coordinate types", "[geometry]") {
	using p8 = point<atl::int16>;
	std::vector<p8> const corners{{-128, -128}, {127, 127}, {-128, 127}, {127, -128}, {120, -128}};
	auto const [i, j] = closest_pair_indices(corners);
	REQUIRE(squared_distance(corners[i], corners[j]) == 49);
	auto const [di, dj] = atl::experimental::closest_pair_indices_by_divide_and_conquer(corners);
	REQUIRE(squared_distance(corners[di], corners[dj]) == 49);
	auto const [gi, gj] = atl::experimental::closest_pair_indices_by_randomized_grid(corners);
	REQUIRE(squared_distance(corners[gi], corners[gj]) == 49);
	// tiny floating scales: the window is the rounded-up root, not root + 1
	std::mt19937_64 rng(11);
	std::uniform_real_distribution<double> unit(0.0, 1e-6);
	std::vector<point<double>> tiny(3000);
	for (auto &p : tiny) p = {unit(rng), unit(rng)};
	double best = squared_distance(tiny[0], tiny[1]);
	for (std::size_t a = 0; a < tiny.size(); ++a)
		for (std::size_t b = a + 1; b < tiny.size(); ++b) best = (std::min)(best, squared_distance(tiny[a], tiny[b]));
	auto const [u, v] = closest_pair_indices(tiny);
	REQUIRE(squared_distance(tiny[u], tiny[v]) == best);
	std::vector<point<float>> const floats{{0.1f, 0.2f}, {0.3f, 0.2f}, {0.1f, 0.25f}};
	auto const [fu, fv] = closest_pair(floats);
	REQUIRE(squared_distance(fu, fv) == squared_distance(floats[0], floats[2]));
}

TEST_CASE("closest_pair: fewer than two points is asserted", "[geometry]") {
	CHECK(atl_test::aborts([] { (void)closest_pair_indices(std::vector<pt>{{1, 1}}); }));
	CHECK(atl_test::aborts([] { (void)atl::experimental::closest_pair_indices_by_divide_and_conquer(std::vector<pt>{}); }));
}

TEST_CASE("closest_pair: oracle covers the largest valid squared distance", "[geometry]") {
	std::int64_t const m = (std::int64_t(1) << 62) - 1;
	verify({{-m, -m}, {m, m}}, false);
}

TEST_CASE("closest_pair: grid division preserves neighbors near 2^53", "[geometry]") {
	std::vector<pt> const points{{0, 0}, {1, 3}, {7120816245988177LL, 0}, {7120816245988180LL, 0}};
	for (std::uint64_t seed = 0; seed < 200; ++seed) {
		atl::set_hash_seed(seed);
		auto const pair = atl::experimental::closest_pair_indices_by_randomized_grid(points);
		REQUIRE(squared_distance(points[pair[0]], points[pair[1]]) == 9);
	}
}
