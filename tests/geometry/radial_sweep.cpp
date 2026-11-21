#include <atl/geometry/point.hpp>
#include <atl/geometry/radial_sweep.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

using atl::float64;
using atl::uint32;
using atl::geometry::arc;
using atl::geometry::max_arc_overlap;
using atl::geometry::max_points_in_circle;
using atl::geometry::point;

namespace {
	using lp = point<std::int64_t>;
	constexpr float64 test_turn = 2 * std::numbers::pi;

	float64 wrap(float64 const a) {
		float64 const m = std::fmod(a, test_turn);
		return m < 0 ? m + test_turn : m;
	}

	// Closed membership compares separately normalized endpoints.
	bool member(arc const &a, float64 const angle) {
		float64 const from = wrap(a.from), to = wrap(a.to), probe = wrap(angle);
		return from <= to ? from <= probe && probe <= to : probe >= from || probe <= to;
	}

	// the maximum is attained at some arc's `from`: try them all
	uint32 brute_overlap(std::vector<arc> const &arcs) {
		uint32 best = 0;
		for (auto const &probe : arcs) {
			uint32 open = 0;
			for (auto const &a : arcs) open += member(a, probe.from);
			best = (std::max)(best, open);
		}
		return best;
	}

	uint32 count_within(std::vector<lp> const &ps, point<float64> const c,
						float64 const limit) {
		uint32 open = 0;
		for (auto const &q : ps) {
			float64 const dx = float64(q.x) - c.x, dy = float64(q.y) - c.y;
			open += dx * dx + dy * dy <= limit * limit;
		}
		return open;
	}

	// margin brute: best count achievable with slack, over all candidate
	// centers (every point; both circles through every close-enough pair)
	uint32 brute_cover_with(std::vector<lp> const &ps, float64 const r,
							float64 const margin) {
		uint32 best = 0;
		for (auto const &q : ps)
			best = (std::max)(best, count_within(ps, {float64(q.x), float64(q.y)}, r + margin));
		for (std::size_t i = 0; i < ps.size(); ++i)
			for (std::size_t j = i + 1; j < ps.size(); ++j) {
				float64 const dx = float64(ps[j].x) - float64(ps[i].x);
				float64 const dy = float64(ps[j].y) - float64(ps[i].y);
				float64 const d2 = dx * dx + dy * dy;
				if (d2 == 0 || d2 > 4 * r * r) continue;
				float64 const h = std::sqrt((std::max)(r * r - d2 / 4, 0.0));
				float64 const d = std::sqrt(d2);
				point<float64> const mid{float64(ps[i].x) + dx / 2,
										 float64(ps[i].y) + dy / 2};
				for (float64 const s : {h / d, -h / d})
					best = (std::max)(best,
									  count_within(ps, {mid.x - dy * s, mid.y + dx * s},
												   r + margin));
			}
		return best;
	}
}

TEST_CASE("max_arc_overlap: handcrafted", "[geometry]") {
	REQUIRE(max_arc_overlap(std::vector<arc>{}).count == 0);

	auto const lone = max_arc_overlap(std::vector<arc>{{1.0, 2.0}});
	REQUIRE(lone.count == 1);
	REQUIRE((lone.angle >= 1.0 && lone.angle <= 2.0));

	// disjoint arcs never stack
	REQUIRE(max_arc_overlap(std::vector<arc>{{0.0, 1.0}, {2.0, 3.0}}).count == 1);
	// CLOSED arcs: touching endpoints overlap
	REQUIRE(max_arc_overlap(std::vector<arc>{{0.0, 1.0}, {1.0, 2.0}}).count == 2);
	// zero-length arc stabs a covering arc
	REQUIRE(max_arc_overlap(std::vector<arc>{{0.5, 0.5}, {0.0, 1.0}}).count == 2);
	// wrapping arc (through 2π) meets one across the seam
	REQUIRE(max_arc_overlap(std::vector<arc>{{-0.5, 0.5}, {0.25, 1.0}}).count == 2);
	// negative and offset radians normalize consistently
	REQUIRE(max_arc_overlap(std::vector<arc>{{-0.5 - test_turn, 0.5 - test_turn},
											 {0.25 + test_turn, 1.0 + test_turn}})
				.count
			== 2);

	auto const triple = max_arc_overlap(
		std::vector<arc>{{0.0, 2.0}, {1.0, 3.0}, {1.5, 2.5}, {5.0, 5.5}});
	REQUIRE(triple.count == 3);
	REQUIRE((triple.angle >= 1.5 && triple.angle <= 2.0));
}

TEST_CASE("max_arc_overlap: random vs the stab brute", "[geometry]") {
	auto seed = GENERATE(take(12, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float64> angle(-10.0, 10.0);
	for (int it = 0; it < 300; ++it) {
		std::size_t const n = static_cast<std::size_t>(1 + rng() % 20);
		std::vector<arc> arcs;
		for (std::size_t k = 0; k < n; ++k) arcs.push_back({angle(rng), angle(rng)});
		auto const got = max_arc_overlap(arcs);
		REQUIRE(got.count == brute_overlap(arcs));
		// the witness stabs what it claims
		uint32 open = 0;
		for (auto const &a : arcs) open += member(a, got.angle);
		REQUIRE(open == got.count);
	}
}

TEST_CASE("max_points_in_circle: handcrafted", "[geometry]") {
	REQUIRE(max_points_in_circle(std::vector<lp>{{7, -3}}, 10.0).count == 1);

	// radius 0: the largest coincident cluster
	std::vector<lp> const clusters{{1, 1}, {2, 2}, {1, 1}, {1, 1}, {2, 2}};
	REQUIRE(max_points_in_circle(clusters, 0.0).count == 3);

	// two points at distance EXACTLY 2r: one circle grazes both
	REQUIRE(max_points_in_circle(std::vector<lp>{{0, 0}, {4, 0}}, 2.0).count == 2);
	// pull them one apart and it no longer can
	REQUIRE(max_points_in_circle(std::vector<lp>{{0, 0}, {5, 0}}, 2.0).count == 1);

	// four corners of a square inside a circumscribing radius
	std::vector<lp> const square{{0, 0}, {2, 0}, {2, 2}, {0, 2}, {100, 100}};
	auto const sq = max_points_in_circle(square, 1.5);
	REQUIRE(sq.count == 4);
	REQUIRE(count_within(square, sq.center, 1.5 + 1e-9) == 4);

	// three separated pairs: any best answer is a pair, witness must work
	std::vector<lp> const pairs{{0, 0}, {1, 0}, {50, 50}, {50, 51}, {-40, 7}, {-41, 7}};
	auto const pr = max_points_in_circle(pairs, 1.0);
	REQUIRE(pr.count == 2);
	REQUIRE(count_within(pairs, pr.center, 1.0 + 1e-9) == 2);
}

TEST_CASE("max_points_in_circle: sandwich oracle on random grids", "[geometry]") {
	auto seed = GENERATE(take(8, random(std::uint64_t(0),
										(std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 40; ++it) {
		std::size_t const n = static_cast<std::size_t>(1 + rng() % 24);
		std::vector<lp> ps;
		for (std::size_t k = 0; k < n; ++k)
			ps.push_back({std::int64_t(rng() % 16), std::int64_t(rng() % 16)});
		for (float64 const r : {0.6, 1.3, 2.5, 4.0, 11.0}) {
			auto const got = max_points_in_circle(ps, r);
			// upper: the witness center must deliver the claimed count
			REQUIRE(count_within(ps, got.center, r * (1 + 1e-9) + 1e-9) >= got.count);
			// lower: anything achievable with slack must be found
			REQUIRE(got.count >= brute_cover_with(ps, r, -1e-7));
			// and the claim never beats the tolerant optimum
			REQUIRE(got.count <= brute_cover_with(ps, r, 1e-7));
		}
		// monotone in the radius
		uint32 last = 0;
		for (float64 const r : {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0}) {
			uint32 const c = max_points_in_circle(ps, r).count;
			REQUIRE(c >= last);
			last = c;
		}
		REQUIRE(last == ps.size()); // a huge circle covers everything
	}
}

TEST_CASE("max_points_in_circle: translation to 2^60 keeps the counts", "[geometry]") {
	// the directions are differenced exactly before rounding, so a grid
	// translated near the int64 bound sees the same arcs; the witness
	// center is a rounded construction and is checked only for finiteness
	auto seed = GENERATE(take(6, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::int64_t const base_x = (std::int64_t(1) << 60) + std::int64_t(rng() % 1000);
	std::int64_t const base_y = -(std::int64_t(1) << 60) + std::int64_t(rng() % 1000);
	for (int it = 0; it < 20; ++it) {
		std::size_t const n = static_cast<std::size_t>(1 + rng() % 40);
		std::vector<lp> ps, distant;
		for (std::size_t k = 0; k < n; ++k) {
			ps.push_back({std::int64_t(rng() % 16), std::int64_t(rng() % 16)});
			distant.push_back({base_x + ps.back().x, base_y + ps.back().y});
		}
		for (float64 const r : {0.0, 0.6, 1.3, 2.5, 4.0, 11.0}) {
			auto const near_answer = max_points_in_circle(ps, r);
			auto const far_answer = max_points_in_circle(distant, r);
			REQUIRE(far_answer.count == near_answer.count);
			REQUIRE(std::isfinite(far_answer.center.x));
			REQUIRE(std::isfinite(far_answer.center.y));
		}
	}
}

TEST_CASE("max_points_in_circle: grid path vs all pairs on larger clouds", "[geometry]") {
	// a few hundred points over a span of many cells: the 3x3 block scan
	// must find every neighbor the sandwich brute counts
	auto seed = GENERATE(take(4, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<lp> ps;
	for (int k = 0; k < 300; ++k) ps.push_back({std::int64_t(rng() % 200), std::int64_t(rng() % 200)});
	for (float64 const r : {3.0, 7.5, 20.0}) {
		auto const got = max_points_in_circle(ps, r);
		REQUIRE(count_within(ps, got.center, r * (1 + 1e-9) + 1e-9) >= got.count);
		REQUIRE(got.count >= brute_cover_with(ps, r, -1e-7));
		REQUIRE(got.count <= brute_cover_with(ps, r, 1e-7));
	}
}

TEST_CASE("max_points_in_circle: double coordinates parity spot check", "[geometry]") {
	using pd = point<double>;
	std::vector<pd> const ps{{0, 0}, {1.5, 0}, {0.75, 0.5}, {10, 10}};
	auto const got = max_points_in_circle(ps, 1.0);
	REQUIRE(got.count == 3);
}

TEST_CASE("max_arc_overlap: an arc whose span rounds up to a full turn counts once", "[geometry]") {
	// to one ulp below from: the span is a nearly full turn and rounds up to
	// 2 pi; the wrapped exit must not reach the arc's own entry
	for (float64 from = 0.01; from < 6.2; from += 0.01) {
		auto const got = max_arc_overlap(std::vector<arc>{{from, std::nextafter(from, 0.0)}});
		REQUIRE(got.count == 1);
		REQUIRE((got.angle >= 0.0 && got.angle < test_turn));
	}
	// a start just below 0 reduces to 0, inside [0, 2 pi)
	auto const low = max_arc_overlap(std::vector<arc>{{-1e-17, 1.0}, {0.5, 2.0}});
	REQUIRE(low.count == 2);
	REQUIRE((low.angle >= 0.0 && low.angle < test_turn));
}

TEST_CASE("max_points_in_circle: degenerate clouds", "[geometry]") {
	std::vector<lp> const same(50, lp{4, -4});
	REQUIRE(max_points_in_circle(same, 0.0).count == 50);
	REQUIRE(max_points_in_circle(same, 3.0).count == 50);
	std::vector<lp> line;
	for (std::int64_t k = 0; k < 30; ++k) line.push_back({3 * k, 0});
	// a disk of radius 4.5 covers a diameter of 9: four points 3 apart
	REQUIRE(max_points_in_circle(line, 4.5).count == 4);
	REQUIRE(max_points_in_circle(line, 4.4).count == 3);
}

TEST_CASE("max_points_in_circle: no coordinate bound, clouds at the int64 limits", "[geometry]") {
	auto seed = GENERATE(take(4, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::int64_t const top = (std::numeric_limits<std::int64_t>::max)() - 20;
	std::int64_t const bottom = (std::numeric_limits<std::int64_t>::min)();
	for (int it = 0; it < 10; ++it) {
		std::size_t const n = static_cast<std::size_t>(1 + rng() % 30);
		std::vector<lp> ps, distant;
		for (std::size_t k = 0; k < n; ++k) {
			ps.push_back({std::int64_t(rng() % 16), std::int64_t(rng() % 16)});
			distant.push_back({top + ps.back().x, bottom + ps.back().y});
		}
		for (float64 const r : {0.0, 0.6, 1.3, 2.5, 11.0}) REQUIRE(max_points_in_circle(distant, r).count == max_points_in_circle(ps, r).count);
	}
	// a cloud spanning the whole type: the grid is skipped, pairs are scanned
	std::vector<lp> const corners{{bottom, bottom}, {bottom + 1, bottom}, {top, top}, {bottom, top}};
	REQUIRE(max_points_in_circle(corners, 1.0).count == 2);
}

TEST_CASE("max_points_in_circle: asserted preconditions", "[geometry]") {
	CHECK(atl_test::aborts([] { (void)max_points_in_circle(std::vector<lp>{}, 1.0); }));
	CHECK(atl_test::aborts([] { (void)max_points_in_circle(std::vector<lp>{{0, 0}}, -1.0); }));
}

TEST_CASE("max_arc_overlap: unreduced angles and closed endpoint oracle", "[geometry]") {
	REQUIRE(max_arc_overlap(std::vector<arc>{{1e20, 1}, {0.5, 0.5}}).count == 2);
	REQUIRE(max_arc_overlap(std::vector<arc>{{-1e-17, -2e-17}}).count == 1);
	REQUIRE(max_arc_overlap(std::vector<arc>{{-1e-17, -2e-17}, {0.5, 0.5}}).count == 2);
	double const endpoint = wrap(-7.7);
	std::vector<arc> const arcs{{-10, -7.7}, {endpoint, endpoint}};
	REQUIRE(brute_overlap(arcs) == 2);
	REQUIRE(max_arc_overlap(arcs).count == 2);
	double const largest = (std::numeric_limits<double>::max)();
	REQUIRE(max_arc_overlap(std::vector<arc>{{-largest, largest}}).count == 1);
}

TEST_CASE("max_arc_overlap: integer endpoints give an exact independent oracle", "[geometry]") {
	std::mt19937_64 rng(59);
	for (int iteration = 0; iteration < 300; ++iteration) {
		std::vector<arc> arcs;
		for (int j = 0; j < 20; ++j) arcs.push_back({double(rng() % 7), double(rng() % 7)});
		REQUIRE(max_arc_overlap(arcs).count == brute_overlap(arcs));
	}
}

TEST_CASE("max_points_in_circle: radius zero uses exact equality at tiny scales", "[geometry]") {
	using pd = point<double>;
	REQUIRE(max_points_in_circle(std::vector<pd>{{0, 0}, {1e-190, 0}, {0, 1e-190}}, 0).count == 1);
	std::vector<pd> points(20000, pd{1e-190, 0});
	points.push_back({0, 0});
	REQUIRE(max_points_in_circle(points, 0).count == 20000);
}

TEST_CASE("max_points_in_circle: invalid floating magnitudes assert", "[geometry]") {
	using pd = point<double>;
	CHECK(atl_test::aborts([] { (void)max_points_in_circle(std::vector<pd>{{0, 0}, {1e200, 0}, {0, 1e200}}, 1e200); }));
	CHECK(atl_test::aborts([] { (void)max_points_in_circle(std::vector<pd>{{0, 0}, {1e-190, 0}, {0, 1e-190}}, 1e-200); }));
	CHECK(atl_test::aborts([] { (void)max_arc_overlap(std::vector<arc>{{std::numeric_limits<double>::quiet_NaN(), 0}}); }));
}
