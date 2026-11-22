#include <atl/experimental/geometry/convex_hull_variants.hpp>
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <ranges>
#include <set>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::geometry::containment;
using atl::geometry::contains;
using atl::geometry::convex_hull;
using atl::geometry::convex_hull_indices;
using atl::geometry::keep_collinear;
using atl::geometry::orientation;
using atl::geometry::point;

namespace {
	using pt = point<std::int64_t>;

	// q in conv(S) iff q lies in some closed triangle/segment/point of S
	// (Caratheodory in 2D) — the independent membership brute
	bool in_convex_closure(std::vector<pt> const &s, pt const q) {
		for (std::size_t i = 0; i < s.size(); ++i) {
			if (s[i] == q) return true;
			for (std::size_t j = i + 1; j < s.size(); ++j) {
				if (contains(atl::geometry::segment<std::int64_t>{s[i], s[j]}, q)) return true;
				for (std::size_t k = j + 1; k < s.size(); ++k) {
					// degenerate triples would read "all zero orientations"
					// for any on-line query; the segment checks cover them
					if (orientation(s[i], s[j], s[k]) == 0) continue;
					int const o1 = orientation(s[i], s[j], q), o2 = orientation(s[j], s[k], q),
							  o3 = orientation(s[k], s[i], q);
					if ((o1 >= 0 && o2 >= 0 && o3 >= 0) || (o1 <= 0 && o2 <= 0 && o3 <= 0))
						return true;
				}
			}
		}
		return false;
	}

	std::set<pt> as_set(std::vector<pt> const &pts) { return {pts.begin(), pts.end()}; }

	// structural checks + ground-truth vertex sets from the brute
	void verify(std::vector<pt> const &pts) {
		auto const strict_ids = convex_hull_indices(pts);
		auto const strict = convex_hull(pts);
		auto const keep_ids = convex_hull_indices(pts, keep_collinear);
		auto const keep = convex_hull(pts, keep_collinear);
		REQUIRE(strict.size() == strict_ids.size());
		REQUIRE(keep.size() == keep_ids.size());
		for (std::size_t k = 0; k < strict_ids.size(); ++k) {
			REQUIRE(strict_ids[k] < pts.size());
			REQUIRE(pts[strict_ids[k]] == strict[k]);
		}
		for (std::size_t k = 0; k < keep_ids.size(); ++k) REQUIRE(pts[keep_ids[k]] == keep[k]);

		// ground truth: strict vertices are exactly the points outside the
		// closure of the others (computed over distinct points)
		std::vector<pt> distinct;
		for (auto const p : pts)
			if (!std::ranges::count(distinct, p)) distinct.push_back(p);
		std::set<pt> expected_strict;
		for (std::size_t i = 0; i < distinct.size(); ++i) {
			auto others = distinct;
			others.erase(others.begin() + std::ptrdiff_t(i));
			if (!in_convex_closure(others, distinct[i])) expected_strict.insert(distinct[i]);
		}
		if (distinct.size() <= 2) {
			REQUIRE(as_set(strict) == as_set(distinct));
			REQUIRE(as_set(keep) == as_set(distinct));
		} else {
			bool all_collinear = true;
			for (std::size_t i = 2; all_collinear && i < distinct.size(); ++i)
				all_collinear = orientation(distinct[0], distinct[1], distinct[i]) == 0;
			if (all_collinear) {
				REQUIRE(strict.size() == 2); // the two endpoints
				REQUIRE(as_set(keep) == as_set(distinct));
				auto const [lo, hi] = std::ranges::minmax(distinct);
				REQUIRE(as_set(strict) == std::set<pt>{lo, hi});
			} else {
				REQUIRE(as_set(strict) == expected_strict);
				// keep set = every distinct point on the strict hull boundary
				std::set<pt> boundary;
				for (auto const p : distinct)
					if (contains(strict, p) == containment::boundary) boundary.insert(p);
				REQUIRE(as_set(keep) == boundary);

				// order: CCW from the lex-min, strict turns strictly left,
				// keep turns non-negative; every input point inside-or-on
				auto const check_order = [&](std::vector<pt> const &h, bool strict_turns) {
					REQUIRE(*std::ranges::min_element(h) == h[0]);
					for (std::size_t k = 0; k < h.size(); ++k) {
						int const o = orientation(h[k], h[(k + 1) % h.size()],
												  h[(k + 2) % h.size()]);
						if (strict_turns)
							REQUIRE(o > 0);
						else
							REQUIRE(o >= 0);
					}
				};
				check_order(strict, true);
				check_order(keep, false);
				for (auto const p : pts) REQUIRE(contains(strict, p) != containment::outside);
			}
		}

		// the four experimental contenders must produce the IDENTICAL strict
		// sequence (same conventions, unique hull)
		REQUIRE(atl::experimental::convex_hull_indices_by_graham_scan(pts) == strict_ids);
		REQUIRE(atl::experimental::convex_hull_indices_by_jarvis_march(pts) == strict_ids);
		REQUIRE(atl::experimental::convex_hull_indices_by_quickhull(pts) == strict_ids);
		REQUIRE(atl::experimental::convex_hull_indices_by_chan(pts) == strict_ids);
	}
}

TEST_CASE("convex_hull: exhaustive subsets of a 4x3 grid", "[geometry]") {
	std::vector<pt> const grid = [] {
		std::vector<pt> g;
		for (std::int64_t x = 0; x < 4; ++x)
			for (std::int64_t y = 0; y < 3; ++y) g.push_back({x, y});
		return g;
	}();
	for (unsigned mask = 0; mask < (1u << grid.size()); ++mask) {
		std::vector<pt> pts;
		for (std::size_t b = 0; b < grid.size(); ++b)
			if (mask >> b & 1) pts.push_back(grid[b]);
		verify(pts);
	}
}

TEST_CASE("convex_hull: type and policy parity on the grid subsets", "[geometry]") {
	std::mt19937_64 rng(20260821);
	for (int iteration = 0; iteration < 300; ++iteration) {
		std::vector<pt> pts(3 + std::size_t(rng() % 20));
		for (auto &p : pts) p = {std::int64_t(rng() % 15) - 7, std::int64_t(rng() % 15) - 7};
		auto const expected = convex_hull_indices(pts);
		std::vector<point<std::int32_t>> p32;
		std::vector<point<double>> pd;
		for (auto const p : pts) {
			p32.push_back({std::int32_t(p.x), std::int32_t(p.y)});
			pd.push_back({double(p.x), double(p.y)});
		}
		REQUIRE(convex_hull_indices(p32) == expected);
		REQUIRE(convex_hull_indices(pd) == expected);
		REQUIRE(atl::experimental::convex_hull_indices_by_graham_scan(pd) == expected);
		REQUIRE(atl::experimental::convex_hull_indices_by_jarvis_march(pd) == expected);
		REQUIRE(atl::experimental::convex_hull_indices_by_quickhull(pd) == expected);
		REQUIRE(atl::experimental::convex_hull_indices_by_chan(pd) == expected);
		REQUIRE(convex_hull_indices(pd, atl::geometry::fast{}) == expected);
		REQUIRE(convex_hull_indices(p32, keep_collinear) == convex_hull_indices(pts, keep_collinear));
		REQUIRE(convex_hull_indices(pd, keep_collinear) == convex_hull_indices(pts, keep_collinear));
	}
}

TEST_CASE("convex_hull: duplicates, degenerates, conventions", "[geometry]") {
	REQUIRE(convex_hull_indices(std::vector<pt>{}).empty());
	REQUIRE(convex_hull_indices(std::vector<pt>{{5, 5}}) == std::vector<uint32>{0});
	REQUIRE(convex_hull_indices(std::vector<pt>{{5, 5}, {5, 5}, {5, 5}}) == std::vector<uint32>{0});
	REQUIRE(convex_hull_indices(std::vector<pt>{{3, 1}, {0, 0}, {3, 1}})
			== std::vector<uint32>{1, 0}); // smallest index per duplicate point
	// all-collinear conventions
	std::vector<pt> const line{{4, 4}, {0, 0}, {2, 2}, {3, 3}, {2, 2}};
	REQUIRE(convex_hull_indices(line) == std::vector<uint32>{1, 0});
	REQUIRE(convex_hull_indices(line, keep_collinear) == std::vector<uint32>{1, 2, 3, 0});
	verify(line);
}

TEST_CASE("convex_hull: random clouds with structure, prune parity", "[geometry]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<pt> pts;
	std::size_t const n = 300 + std::size_t(rng() % 700); // above the pruning threshold
	for (std::size_t i = 0; i < n; ++i) {
		switch (std::size_t(rng() % 4)) {
		case 0:
			pts.push_back({std::int64_t(rng() % 2001) - 1000, std::int64_t(rng() % 2001) - 1000});
			break;
		case 1: { // grid lines: collinear runs
			std::int64_t const x = std::int64_t(rng() % 21) * 100 - 1000;
			pts.push_back({x, std::int64_t(rng() % 2001) - 1000});
			break;
		}
		case 2: { // co-circular integer points (3-4-5 flavors)
			pt const base[]{{600, 800}, {800, 600}, {1000, 0}, {0, 1000}, {-600, 800}, {-800, -600}, {280, 960}, {-960, 280}};
			pts.push_back(base[std::size_t(rng() % 8)]);
			break;
		}
		default:
			pt const duplicate = pts.empty() ? pt{0, 0} : pts[std::size_t(rng() % pts.size())];
			pts.push_back(duplicate);
		}
	}
	auto const library = convex_hull_indices(pts);
	auto const unpruned = atl::geometry::detail::hull_indices<true>(pts, atl::geometry::exact{}, false);
	REQUIRE(library == unpruned);
	REQUIRE(atl::geometry::detail::hull_indices<false>(pts, atl::geometry::exact{}, false)
			== convex_hull_indices(pts, keep_collinear));
	// cross-algorithm agreement (five independent implementations)
	REQUIRE(atl::experimental::convex_hull_indices_by_graham_scan(pts) == library);
	REQUIRE(atl::experimental::convex_hull_indices_by_jarvis_march(pts) == library);
	REQUIRE(atl::experimental::convex_hull_indices_by_quickhull(pts) == library);
	REQUIRE(atl::experimental::convex_hull_indices_by_chan(pts) == library);
	// invariants at this size (brute is too slow): CCW strict turns, start,
	// containment of every input point
	auto const hull = convex_hull(pts);
	REQUIRE(*std::ranges::min_element(hull) == hull[0]);
	for (std::size_t k = 0; k < hull.size(); ++k)
		REQUIRE(orientation(hull[k], hull[(k + 1) % hull.size()], hull[(k + 2) % hull.size()]) > 0);
	for (auto const p : pts) REQUIRE(contains(hull, p) != containment::outside);
}

TEST_CASE("convex_hull: prune parity above the sampled-gate size", "[geometry]") {
	// n > 2048 so the gate probes a strided sample with the sample's own
	// octagon and the rectangle pre-accept; each shape vs the unpruned path
	auto seed = GENERATE(take(12, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 3000 + std::size_t(rng() % 20000);
	std::vector<pt> pts(n);
	switch (std::size_t(rng() % 4)) {
	case 0: // uniform square: the rectangle accepts almost everything
		for (auto &p : pts) p = {std::int64_t(rng() % 2001) - 1000, std::int64_t(rng() % 2001) - 1000};
		break;
	case 1: { // gaussian-ish cluster with duplicates
		std::normal_distribution<double> gauss(0.0, 50.0);
		for (auto &p : pts) p = {std::int64_t(gauss(rng)), std::int64_t(gauss(rng))};
		break;
	}
	case 2: // rotated thin diamond: skewed diagonal extremes
		for (auto &p : pts) {
			std::int64_t const u = std::int64_t(rng() % 20001) - 10000, v = std::int64_t(rng() % 41) - 20;
			p = {u + v, u - v};
		}
		break;
	default: // integer points near a circle: the gate bails
		for (auto &p : pts) {
			double const a = double(std::size_t(rng() % 100000)) / 100000.0 * 6.283185307179586;
			p = {std::int64_t(std::llround(1e6 * std::cos(a))), std::int64_t(std::llround(1e6 * std::sin(a)))};
		}
	}
	auto const library = convex_hull_indices(pts);
	REQUIRE(library == atl::geometry::detail::hull_indices<true>(pts, atl::geometry::exact{}, false));
	REQUIRE(convex_hull_indices(pts, keep_collinear)
			== atl::geometry::detail::hull_indices<false>(pts, atl::geometry::exact{}, false));
	REQUIRE(atl::experimental::convex_hull_indices_by_graham_scan(pts) == library);
	REQUIRE(atl::experimental::convex_hull_indices_by_quickhull(pts) == library);
	std::vector<point<double>> pd;
	for (auto const q : pts) pd.push_back({double(q.x), double(q.y)});
	REQUIRE(convex_hull_indices(pd) == library);
}

TEST_CASE("convex_hull: int64 at scale, constructive", "[geometry]") {
	std::mt19937_64 rng(7);
	std::int64_t const bound = std::int64_t(1) << 61;
	// known hull: 4 extreme corners + interior points strictly inside
	std::vector<pt> pts{{-bound, -bound}, {bound, -bound}, {bound, bound}, {-bound, bound}};
	for (int i = 0; i < 500; ++i)
		pts.push_back({std::int64_t(rng() % (2 * (std::uint64_t(bound) / 2))) - bound / 2,
					   std::int64_t(rng() % (2 * (std::uint64_t(bound) / 2))) - bound / 2});
	auto const hull = convex_hull_indices(pts);
	REQUIRE(hull == std::vector<uint32>{0, 1, 2, 3});
}

TEST_CASE("convex_contains: log-n locate vs the O(n) polygon oracle", "[geometry]") {
	using atl::geometry::convex_contains;
	auto seed = GENERATE(take(20, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<pt> cloud;
	std::size_t const n = 1 + std::size_t(rng() % 40);
	std::int64_t const spread = 1 + std::int64_t(rng() % 12);
	for (std::size_t k = 0; k < n; ++k)
		cloud.push_back({std::int64_t(rng() % (2 * spread + 1)) - spread,
						 std::int64_t(rng() % (2 * spread + 1)) - spread});
	auto const hull = convex_hull(cloud); // strict CCW: the documented input
	for (std::int64_t x = -spread - 2; x <= spread + 2; ++x)
		for (std::int64_t y = -spread - 2; y <= spread + 2; ++y) {
			pt const q{x, y};
			containment expected;
			if (hull.size() >= 3)
				expected = contains(hull, q); // the O(n) simple-polygon test
			else if (hull.size() == 2)
				expected = atl::geometry::contains(
							   atl::geometry::segment<std::int64_t>{hull[0], hull[1]}, q)
							 ? containment::boundary
							 : containment::outside;
			else
				expected = hull[0] == q ? containment::boundary : containment::outside;
			REQUIRE(convex_contains(hull, q) == expected);
		}
	std::vector<point<double>> hd;
	for (auto const v : hull) hd.push_back({double(v.x), double(v.y)});
	for (std::int64_t x = -spread - 2; x <= spread + 2; ++x)
		for (std::int64_t y = -spread - 2; y <= spread + 2; ++y)
			REQUIRE(convex_contains(hd, point<double>{double(x), double(y)})
					== convex_contains(hull, pt{x, y}));
}

TEST_CASE("convex_contains: exactness at huge coordinates", "[geometry]") {
	using atl::geometry::convex_contains;
	std::int64_t const big = std::int64_t(1) << 60;
	std::vector<pt> const hull{{0, 0}, {big, 1}, {big - 1, big}};
	REQUIRE(convex_contains(hull, pt{big / 2, big / 4}) == containment::inside);
	REQUIRE(convex_contains(hull, pt{big, 1}) == containment::boundary);
	// above the near-flat edge (0,0)-(big,1) is INSIDE the sliver
	// (cross = big - 2 > 0); below it is out (cross = -2)
	REQUIRE(convex_contains(hull, pt{2, 1}) == containment::inside);
	REQUIRE(convex_contains(hull, pt{2, 0}) == containment::outside);
}

TEST_CASE("minkowski_sum: vs the hull-of-pairwise-sums oracle", "[geometry]") {
	using atl::geometry::convexity;
	using atl::geometry::convexity_of;
	using atl::geometry::minkowski_sum;
	using atl::geometry::winding;
	auto const to_lexmin = [](std::vector<pt> v) {
		auto const it = std::ranges::min_element(v);
		std::ranges::rotate(v, it);
		return v;
	};
	auto seed = GENERATE(take(25, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const make_summand = [&](std::size_t const kind) -> std::vector<pt> {
		std::int64_t const cx = std::int64_t(rng() % 21) - 10;
		std::int64_t const cy = std::int64_t(rng() % 21) - 10;
		if (kind == 0) return {{cx, cy}};					// point
		if (kind == 1) return {{cx, cy}, {cx + 3, cy + 1}}; // segment
		std::vector<pt> cloud;								// polygon
		for (int k = 0; k < 12; ++k)
			cloud.push_back({cx + std::int64_t(rng() % 15) - 7,
							 cy + std::int64_t(rng() % 15) - 7});
		auto hull = convex_hull(cloud);
		if (std::size_t(rng() % 2)) std::ranges::reverse(hull); // either winding is legal input
		return hull;
	};
	auto const a = make_summand(std::size_t(rng() % 3)), b = make_summand(std::size_t(rng() % 3));
	auto const got = minkowski_sum(a, b);
	// oracle: strict hull of all pairwise sums
	std::vector<pt> sums;
	for (auto const u : a)
		for (auto const v : b) sums.push_back(u + v);
	auto const expected = convex_hull(sums);
	if (expected.size() <= 2) { // degenerate: compare as sets
		REQUIRE(got.size() == expected.size());
		for (auto const v : expected)
			REQUIRE(std::ranges::find(got, v) != got.end());
	} else {
		REQUIRE(to_lexmin(got) == expected); // hull output IS lex-min-first CCW
		auto const shape = convexity_of(got);
		REQUIRE(shape.is_convex);
		REQUIRE(shape.is_strict);
		REQUIRE(shape.direction == winding::counterclockwise);
	}
	// vertex-count bound and the documented start
	REQUIRE(got.size() <= a.size() + b.size());
	for (auto const v : got) {
		REQUIRE((v.y > got[0].y || (v.y == got[0].y && v.x >= got[0].x)));
	}
}

TEST_CASE("minkowski_sum: handcrafted shapes", "[geometry]") {
	using atl::geometry::minkowski_sum;
	// square + square with PARALLEL edges: still strict (edges merged)
	std::vector<pt> const sq{{0, 0}, {2, 0}, {2, 2}, {0, 2}};
	auto const doubled = minkowski_sum(sq, sq);
	REQUIRE(doubled == std::vector<pt>{{0, 0}, {4, 0}, {4, 4}, {0, 4}});
	// point + polygon = translation
	REQUIRE(minkowski_sum(std::vector<pt>{{10, 20}}, sq)
			== std::vector<pt>{{10, 20}, {12, 20}, {12, 22}, {10, 22}});
	// point + point / collinear parallel segments
	REQUIRE(minkowski_sum(std::vector<pt>{{1, 2}}, std::vector<pt>{{3, 4}})
			== std::vector<pt>{{4, 6}});
	REQUIRE(minkowski_sum(std::vector<pt>{{0, 0}, {2, 0}}, std::vector<pt>{{0, 0}, {5, 0}})
			== std::vector<pt>{{0, 0}, {7, 0}});
	// perpendicular segments = a rectangle
	REQUIRE(minkowski_sum(std::vector<pt>{{0, 0}, {3, 0}}, std::vector<pt>{{0, 0}, {0, 2}})
			== std::vector<pt>{{0, 0}, {3, 0}, {3, 2}, {0, 2}});
	// CW input with collinear vertices: normalized internally
	std::vector<pt> const doctored{{0, 0}, {0, 1}, {0, 2}, {2, 2}, {2, 0}, {1, 0}};
	REQUIRE(minkowski_sum(doctored, std::vector<pt>{{0, 0}})
			== std::vector<pt>{{0, 0}, {2, 0}, {2, 2}, {0, 2}});
}

TEST_CASE("minkowski_sum: the int64 2^61 bound is asserted", "[geometry]") {
	using atl::geometry::minkowski_sum;
	using p64 = point<std::int64_t>;
	std::int64_t const bound = std::int64_t(1) << 61;
	REQUIRE(minkowski_sum(std::vector<p64>{{bound - 1, 0}}, std::vector<p64>{{bound - 1, 0}})
			== std::vector<p64>{{2 * bound - 2, 0}});
	CHECK(atl_test::aborts([&] { (void)minkowski_sum(std::vector<p64>{{bound, 0}}, std::vector<p64>{{0, 0}}); }));
	CHECK(atl_test::aborts([&] { (void)minkowski_sum(std::vector<p64>{{0, 0}}, std::vector<p64>{{0, -bound}}); }));
}

namespace {
	/**
	 * @brief The merge oracle: merge_convex_hulls of the two hulls must equal
	 *        the strict hull of the concatenation exactly, in both orders.
	 */
	template<atl::geometry::coordinate Coordinate>
	void check_merge(std::vector<point<Coordinate>> const &a, std::vector<point<Coordinate>> const &b) {
		std::vector<point<Coordinate>> all(a);
		all.insert(all.end(), b.begin(), b.end());
		auto const expected = convex_hull(all);
		auto const hull_a = convex_hull(a), hull_b = convex_hull(b);
		REQUIRE(atl::geometry::merge_convex_hulls(hull_a, hull_b) == expected);
		REQUIRE(atl::geometry::merge_convex_hulls(hull_b, hull_a) == expected);
		REQUIRE(atl::experimental::merge_convex_hulls_by_streamed_chains(hull_a, hull_b) == expected);
		REQUIRE(atl::experimental::merge_convex_hulls_by_streamed_chains(hull_b, hull_a) == expected);
	}
}

TEST_CASE("merge_convex_hulls: exhaustive pairs of 3x3 grid subsets", "[geometry]") {
	// every pair of point subsets of a 3x3 grid: empty, single points,
	// segments, collinear vertices on the other hull's edges, shared
	// vertices, nesting, identical hulls
	std::vector<pt> grid;
	for (std::int64_t x = 0; x < 3; ++x)
		for (std::int64_t y = 0; y < 3; ++y) grid.push_back({x, y});
	std::vector<std::vector<pt>> hulls;
	for (unsigned mask = 0; mask < (1u << grid.size()); ++mask) {
		std::vector<pt> pts;
		for (std::size_t k = 0; k < grid.size(); ++k)
			if (mask >> k & 1) pts.push_back(grid[k]);
		hulls.push_back(convex_hull(pts));
	}
	std::ranges::sort(hulls);
	auto const [dead0, dead1] = std::ranges::unique(hulls);
	hulls.erase(dead0, dead1);
	for (auto const &a : hulls)
		for (auto const &b : hulls) {
			std::vector<pt> all(a);
			all.insert(all.end(), b.begin(), b.end());
			REQUIRE(atl::geometry::merge_convex_hulls(a, b) == convex_hull(all));
			REQUIRE(atl::experimental::merge_convex_hulls_by_streamed_chains(a, b) == convex_hull(all));
		}
}

TEST_CASE("merge_convex_hulls: lexicographically separated hulls", "[geometry]") {
	auto seed = GENERATE(take(40, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (int round = 0; round < 200; ++round) {
		// small boxes put many points on shared lines (collinear bridges,
		// vertical splits at one x); the split is lexicographic, so a
		// column can be cut between two y values
		std::int64_t const w = 1 + std::int64_t(rng() % 8), h = 1 + std::int64_t(rng() % 8);
		std::vector<pt> pts(1 + std::size_t(rng() % 40));
		for (auto &q : pts) q = {std::int64_t(rng() % std::uint64_t(w)), std::int64_t(rng() % std::uint64_t(h))};
		if (std::size_t(rng() % 4) == 0)
			for (auto &q : pts) q.y = q.x * 2 - 3; // all collinear
		std::ranges::sort(pts);
		std::size_t const cut = std::size_t(rng() % (pts.size() + 1));
		std::vector<pt> const left(pts.begin(), pts.begin() + cut), right(pts.begin() + cut, pts.end());
		if (!left.empty() && !right.empty() && left.back() == right.front()) continue;
		auto const expected = convex_hull(pts);
		auto const hull_left = convex_hull(left), hull_right = convex_hull(right);
		REQUIRE(atl::experimental::merge_convex_hulls_by_streamed_chains(hull_left, hull_right) == expected);
		REQUIRE(atl::experimental::merge_convex_hulls_by_streamed_chains(hull_right, hull_left) == expected);
		REQUIRE(atl::geometry::merge_convex_hulls(hull_left, hull_right) == expected);
		if (!hull_left.empty() && !hull_right.empty()) {
			namespace gd = atl::geometry::detail;
			REQUIRE(gd::merge_separated_hulls(hull_left.begin(), hull_left.size(), gd::lexicographic_top(hull_left.begin(), hull_left.size()),
											  hull_right.begin(), hull_right.size(), gd::lexicographic_top(hull_right.begin(), hull_right.size()), atl::geometry::exact{})
					== expected);
		}
	}
	// big circle halves: long chains on both sides of the bridges
	std::vector<pt> circle(2000);
	for (auto &q : circle) {
		double const t = double(rng() % (1u << 30)) / double(1u << 30) * 6.283185307179586;
		q = {std::int64_t(std::llround(1e6 * std::cos(t))), std::int64_t(std::llround(1e6 * std::sin(t)))};
	}
	std::ranges::sort(circle);
	auto const middle = circle.begin() + std::ptrdiff_t(rng() % circle.size());
	std::vector<pt> const left(circle.begin(), middle), right(middle, circle.end());
	if (left.empty() || left.back() != right.front())
		REQUIRE(atl::experimental::merge_convex_hulls_by_streamed_chains(convex_hull(left), convex_hull(right))
				== convex_hull(circle));
}

TEST_CASE("merge_convex_hulls: random hull pairs vs the hull of the concatenation", "[geometry]") {
	auto seed = GENERATE(take(40, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const cloud = [&](std::size_t const n, std::int64_t const cx, std::int64_t const cy, std::int64_t const r) {
		std::vector<pt> pts(n);
		for (auto &q : pts)
			q = {cx + std::int64_t(rng() % std::uint64_t(2 * r + 1)) - r,
				 cy + std::int64_t(rng() % std::uint64_t(2 * r + 1)) - r};
		return pts;
	};
	for (int round = 0; round < 50; ++round) {
		// small radii put many vertices on the other hull's edges and
		// share vertices; the shift spans overlapping, touching, disjoint
		std::int64_t const r = 1 + std::int64_t(rng() % 12);
		std::int64_t const shift = std::int64_t(rng() % 5) * r / 2;
		auto const a = cloud(std::size_t(rng() % 60), 0, 0, r);
		auto const b = cloud(std::size_t(rng() % 60), shift, std::int64_t(rng() % 3) * shift / 2,
							 1 + std::int64_t(rng() % std::uint64_t(r)));
		check_merge(a, b);
		check_merge(a, a); // identical hulls
		// one hull inside the other, and a hull made of a subset of the
		// other's vertices
		check_merge(a, cloud(std::size_t(rng() % 10), 0, 0, std::max<std::int64_t>(1, r / 3)));
		std::vector<pt> subset;
		for (auto const q : convex_hull(a))
			if (std::size_t(rng() % 2)) subset.push_back(q);
		check_merge(a, subset);
	}
}

TEST_CASE("merge_convex_hulls: large circle hulls and other coordinate types", "[geometry]") {
	std::mt19937_64 rng(20261024);
	auto const circle = [&](std::size_t const n, double const cx, double const radius) {
		std::vector<pt> pts(n);
		for (auto &q : pts) {
			double const t = double(rng() % (1u << 30)) / double(1u << 30) * 6.283185307179586;
			q = {std::int64_t(std::llround(cx + radius * std::cos(t))),
				 std::int64_t(std::llround(radius * std::sin(t)))};
		}
		return pts;
	};
	auto const a = circle(20000, 0, 1e9);
	check_merge(a, circle(20000, 5e8, 1e9));					  // overlapping
	check_merge(a, circle(20000, 0, 5e8));						  // nested
	check_merge(a, circle(20000, 4e9, 1e9));					  // disjoint
	check_merge(a, circle(20000, 0, 1e9));						  // one circle, other vertices
	check_merge(circle(5000, 0, 2e18), circle(5000, 1e17, 2e18)); // near the int64 bound
	for (int round = 0; round < 200; ++round) {
		std::vector<point<std::int32_t>> a32(std::size_t(rng() % 30)), b32(std::size_t(rng() % 30));
		std::vector<point<double>> ad, bd;
		for (auto &q : a32) q = {std::int32_t(rng() % 15) - 7, std::int32_t(rng() % 15) - 7};
		for (auto &q : b32) q = {std::int32_t(rng() % 15) - 4, std::int32_t(rng() % 15) - 7};
		for (auto const q : a32) ad.push_back({double(q.x), double(q.y)});
		for (auto const q : b32) bd.push_back({double(q.x), double(q.y)});
		check_merge(a32, b32);
		check_merge(ad, bd);
		std::vector<point<double>> all(ad);
		all.insert(all.end(), bd.begin(), bd.end());
		REQUIRE(atl::geometry::merge_convex_hulls(convex_hull(ad), convex_hull(bd), atl::geometry::fast{})
				== convex_hull(all));
	}
}

TEST_CASE("merge_convex_hulls: approx / fast hull output merges without aborting", "[geometry]") {
	std::mt19937_64 rng(20261029);
	auto const jittered = [&](std::size_t const n) {
		std::vector<point<double>> pts(n);
		for (auto &q : pts) {
			double const t = double(rng() % (1u << 30)) / double(1u << 30) * 6.283185307179586;
			q = {std::cos(t) + 1e-9 * double(std::size_t(rng() % 100)), std::sin(t) + 1e-9 * double(std::size_t(rng() % 100))};
		}
		return pts;
	};
	{
		// convex_hull joins its chains at (100, 0) untested: that turn's
		// cross is 0.5, inside the dead zone
		atl::geometry::approx const policy{0.6};
		std::vector<point<double>> const pts{{0, 0}, {50, 50}, {100, -0.01}, {100, 0}};
		auto const hull = convex_hull(pts, policy);
		REQUIRE(hull.size() == 4);
		REQUIRE(!atl::geometry::merge_convex_hulls(hull, hull, policy).empty());
	}
	for (int round = 0; round < 50; ++round) {
		auto const a = jittered(200), b = jittered(200);
		atl::geometry::approx const approx_policy{1e-7};
		REQUIRE(!atl::geometry::merge_convex_hulls(convex_hull(a, approx_policy), convex_hull(b, approx_policy), approx_policy).empty());
		atl::geometry::fast const fast_policy{};
		REQUIRE(!atl::geometry::merge_convex_hulls(convex_hull(a, fast_policy), convex_hull(b, fast_policy), fast_policy).empty());
	}
}

TEST_CASE("merge_convex_hulls: degenerate inputs and the asserted precondition", "[geometry]") {
	using atl::geometry::merge_convex_hulls;
	using hull = std::vector<pt>;
	REQUIRE(merge_convex_hulls(hull{}, hull{}).empty());
	REQUIRE(merge_convex_hulls(hull{}, hull{{1, 2}}) == hull{{1, 2}});
	REQUIRE(merge_convex_hulls(hull{{5, 5}}, hull{{5, 5}}) == hull{{5, 5}});
	REQUIRE(merge_convex_hulls(hull{{3, 4}}, hull{{1, 2}}) == hull{{1, 2}, {3, 4}});
	REQUIRE(merge_convex_hulls(hull{{0, 0}, {2, 0}}, hull{{5, 0}, {9, 0}}) == hull{{0, 0}, {9, 0}});
	REQUIRE(merge_convex_hulls(hull{{0, 0}, {4, 0}}, hull{{2, 0}, {2, 3}}) == hull{{0, 0}, {4, 0}, {2, 3}});
	// a vertex strictly on the other hull's edge is dropped
	REQUIRE(merge_convex_hulls(hull{{0, 0}, {4, 0}, {0, 4}}, hull{{1, 1}, {2, 2}}) == hull{{0, 0}, {4, 0}, {0, 4}});
	std::int64_t const big = (std::int64_t(1) << 62) - 1;
	REQUIRE(merge_convex_hulls(hull{{-big, -big}, {big, -big}}, hull{{-big, big}, {big, big}})
			== hull{{-big, -big}, {big, -big}, {big, big}, {-big, big}});
#ifdef ATL_ASSERT
	hull const square{{0, 0}, {2, 0}, {2, 2}, {0, 2}};
	REQUIRE(merge_convex_hulls(square, square) == square);
	// not starting at the lexicographic minimum; clockwise; collinear
	// vertex; duplicate; a segment listed backwards
	CHECK(atl_test::aborts([&] { (void)merge_convex_hulls(square, hull{{2, 0}, {2, 2}, {0, 2}, {0, 0}}); }));
	CHECK(atl_test::aborts([&] { (void)merge_convex_hulls(hull{{0, 0}, {0, 2}, {2, 2}, {2, 0}}, square); }));
	CHECK(atl_test::aborts([&] { (void)merge_convex_hulls(square, hull{{0, 0}, {1, 0}, {2, 0}, {2, 2}}); }));
	CHECK(atl_test::aborts([&] { (void)merge_convex_hulls(square, hull{{0, 0}, {0, 0}}); }));
	CHECK(atl_test::aborts([&] { (void)merge_convex_hulls(square, hull{{1, 1}, {0, 0}}); }));
	// a pentagram: every turn left, but it winds twice
	CHECK(atl_test::aborts(
		[&] { (void)merge_convex_hulls(square, hull{{-10, 3}, {6, -8}, {0, 10}, {-6, -8}, {10, 3}}); }));
	CHECK(atl_test::aborts([&] { (void)merge_convex_hulls(square, hull{{0, 0}, {big + 1, 0}}); }));
#endif
}

TEST_CASE("experimental hull merges: equal the hull of the concatenation",
		  "[geometry]") {
	using atl::experimental::convex_hull_union_by_copodal_walk;
	auto seed = GENERATE(take(30, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto const cloud = [&](std::size_t n, std::int64_t cx, std::int64_t cy, std::int64_t r) {
		std::vector<pt> pts;
		for (std::size_t k = 0; k < n; ++k)
			pts.push_back({cx + std::int64_t(rng() % (2 * r + 1)) - r,
						   cy + std::int64_t(rng() % (2 * r + 1)) - r});
		return pts;
	};
	// shapes: overlapping, disjoint, nested, degenerate sizes
	std::int64_t const shift = std::int64_t(rng() % 4) * 15;
	auto const a = cloud(1 + std::size_t(rng() % 40), 0, 0, 1 + std::int64_t(rng() % 10));
	auto const b = cloud(1 + std::size_t(rng() % 40), shift, shift / 2,
						 1 + std::int64_t(rng() % (shift ? 4 : 20)));
	// oracle: hull of the concatenation — must match EXACTLY
	std::vector<pt> all(a);
	all.insert(all.end(), b.begin(), b.end());
	// the experimental linear merges must match convex_hull of the
	// concatenation, as merge_convex_hulls does
	auto const expected = convex_hull(all);
	REQUIRE(atl::experimental::convex_hull_union_by_materialized_chains(a, b) == expected);
	REQUIRE(atl::experimental::convex_hull_union_by_materialized_chains(b, a) == expected);
	REQUIRE(convex_hull_union_by_copodal_walk(a, b) == expected);
	REQUIRE(convex_hull_union_by_copodal_walk(b, a) == expected);
	auto const hull_a = convex_hull(a), hull_b = convex_hull(b);
	REQUIRE(atl::experimental::merge_convex_hulls_by_materialized_chains(hull_a, hull_b) == expected);
	REQUIRE(atl::experimental::merge_convex_hulls_by_copodal_walk(hull_b, hull_a) == expected);
}

TEST_CASE("experimental hull merges: handcrafted degenerates", "[geometry]") {
	// Sub-three-vertex inputs exercise copodal fallback; flush edges exercise ties.
	auto const check = []<typename Merge>(Merge merge)
		requires std::invocable<Merge &, std::vector<pt> const &, std::vector<pt> const &>
	{
		// point + point, shared points, collinear unions
		REQUIRE(merge(std::vector<pt>{{1, 2}}, std::vector<pt>{{3, 4}}) == std::vector<pt>{{1, 2}, {3, 4}});
		REQUIRE(merge(std::vector<pt>{{5, 5}}, std::vector<pt>{{5, 5}}) == std::vector<pt>{{5, 5}});
		REQUIRE(merge(std::vector<pt>{{0, 0}, {2, 0}}, std::vector<pt>{{5, 0}, {9, 0}})
				== std::vector<pt>{{0, 0}, {9, 0}}); // collinear: the two endpoints
		// identical polygons, one containing the other, a flush shared edge
		std::vector<pt> const sq{{0, 0}, {8, 0}, {8, 8}, {0, 8}};
		REQUIRE(merge(sq, sq) == sq);
		REQUIRE(merge(sq, std::vector<pt>{{3, 3}, {4, 3}, {4, 4}}) == sq);
		std::vector<pt> const right{{8, 0}, {16, 0}, {16, 8}, {8, 8}};
		std::vector<pt> const both{{0, 0}, {16, 0}, {16, 8}, {0, 8}};
		REQUIRE(merge(sq, right) == both);
		REQUIRE(merge(right, sq) == both);
		// int64 at scale: planted extreme corners survive exactly
		std::int64_t const big = std::int64_t(1) << 60;
		auto const wide = merge(std::vector<pt>{{-big, -big}, {big, -big}}, std::vector<pt>{{big, big}, {-big, big}});
		REQUIRE(wide == std::vector<pt>{{-big, -big}, {big, -big}, {big, big}, {-big, big}});
	};
	check([](std::vector<pt> const &a, std::vector<pt> const &b) { return atl::experimental::convex_hull_union_by_materialized_chains(a, b); });
	check([](std::vector<pt> const &a, std::vector<pt> const &b) { return atl::experimental::convex_hull_union_by_copodal_walk(a, b); });
	check([](std::vector<pt> const &a, std::vector<pt> const &b) {
		return atl::experimental::merge_convex_hulls_by_materialized_chains(convex_hull(a), convex_hull(b));
	});
	check([](std::vector<pt> const &a, std::vector<pt> const &b) {
		return atl::experimental::merge_convex_hulls_by_copodal_walk(convex_hull(a), convex_hull(b));
	});
}

TEST_CASE("convex_hull under approx: exact dedup, near-duplicates split in lex order", "[geometry]") {
	using dp = point<double>;
	// a grid of integer-valued doubles, each point with a near-duplicate
	// 1e-12 to its right: the column above a point sorts between the two
	std::vector<dp> ps;
	for (int x = 0; x < 6; ++x)
		for (int y = 0; y < 6; ++y) {
			ps.push_back({double(x), double(y)});
			ps.push_back({double(x) + 1e-12, double(y)});
		}
	ps.push_back(ps[0]); // an exact duplicate collapses to the smaller index
	atl::geometry::approx const loose{1e-9};
	auto const hull = convex_hull_indices(ps, loose);
	REQUIRE(hull.size() >= std::size_t(3));
	REQUIRE(std::ranges::find(hull, uint32(ps.size() - 1)) == hull.end());
	std::set<std::pair<double, double>> positions;
	for (uint32 const i : hull) positions.insert({ps[i].x, ps[i].y});
	REQUIRE(positions.size() == hull.size());
	// strict: every turn is a certified left turn
	for (std::size_t k = 0; k < hull.size(); ++k)
		REQUIRE(orientation(ps[hull[k]], ps[hull[(k + 1) % hull.size()]], ps[hull[(k + 2) % hull.size()]])
				== 1);
}

TEST_CASE("merge_convex_hulls: measured bridge dispatch boundaries", "[geometry]") {
	for (atl::int64 const n : {511, 512, 513, 154'912}) {
		CAPTURE(n);
		std::vector<pt> a, b, expected;
		for (atl::int64 x = -n; x < 0; ++x) a.push_back({x, x * x});
		for (atl::int64 x = 1; x <= n; ++x) b.push_back({x, x * x});
		expected = a;
		expected.insert(expected.end(), b.begin(), b.end());
		REQUIRE(atl::geometry::merge_convex_hulls(a, b) == expected);
		REQUIRE(atl::geometry::merge_convex_hulls(b, a) == expected);
		REQUIRE(atl::experimental::merge_convex_hulls_by_streamed_chains(a, b) == expected);
	}
}

TEST_CASE("merge_convex_hulls: measured touching, overlap and trimmed bridges", "[geometry]") {
	atl::int64 constexpr n = 154'912;
	std::vector<pt> a;
	for (atl::int64 x = -n; x < 0; ++x) a.push_back({x, x * x});
	for (atl::int64 const shift : {n - 1, n / 2, atl::int64(0), 2 * n}) {
		CAPTURE(shift);
		auto b = a;
		for (auto &q : b) q.x += shift;
		auto both = a;
		both.insert(both.end(), b.begin(), b.end());
		auto const expected = convex_hull(both);
		REQUIRE(atl::geometry::merge_convex_hulls(a, b) == expected);
		REQUIRE(atl::geometry::merge_convex_hulls(b, a) == expected);
		REQUIRE(atl::experimental::merge_convex_hulls_by_streamed_chains(a, b) == expected);
	}
}

TEST_CASE("convex_hull: full coordinate bounds and dyadic floating input", "[geometry]") {
	std::int64_t const limit = (std::int64_t(1) << 62) - 1;
	std::vector<pt> const square{{-limit, -limit}, {limit, -limit}, {limit, limit}, {-limit, limit}};
	verify(square);
	std::int32_t const lo = (std::numeric_limits<std::int32_t>::lowest)(), hi = (std::numeric_limits<std::int32_t>::max)();
	REQUIRE(convex_hull_indices(std::vector<point<std::int32_t>>{{lo, lo}, {hi, lo}, {hi, hi}, {lo, hi}, {0, 0}}) == std::vector<uint32>{0, 1, 2, 3});
	for (int exponent : {-500, 0, 500}) {
		double const scale = std::ldexp(1.0, exponent);
		std::vector<point<double>> const cloud{{0, 0}, {4 * scale, 0}, {4 * scale, 4 * scale}, {0, 4 * scale}, {2 * scale, 2 * scale}, {2 * scale, 0}};
		REQUIRE(convex_hull_indices(cloud) == std::vector<uint32>{0, 1, 2, 3});
		REQUIRE(convex_hull_indices(cloud, keep_collinear) == std::vector<uint32>{0, 5, 1, 2, 3});
		REQUIRE(atl::geometry::convex_contains(convex_hull(cloud), {scale, scale}) == containment::inside);
	}
	CHECK(atl_test::aborts([&] { (void)convex_hull(std::vector<pt>{{limit + 1, 0}}); }));
}

TEST_CASE("convex hull implementations: heap storage on deep degenerate inputs", "[geometry]") {
	std::vector<point<int>> cloud(1'000'000);
	for (std::size_t i = 0; i < cloud.size(); ++i) cloud[i] = {int(i), int(i)};
	auto const expected = std::vector<uint32>{0, uint32(cloud.size() - 1)};
	REQUIRE(convex_hull_indices(cloud) == expected);
	REQUIRE(atl::experimental::convex_hull_indices_by_quickhull(cloud) == expected);
	REQUIRE(atl::experimental::convex_hull_indices_by_chan(cloud) == expected);
	std::ranges::fill(cloud, point<int>{1, 1});
	REQUIRE(convex_hull_indices(cloud) == std::vector<uint32>{0});
	REQUIRE(atl::experimental::convex_hull_indices_by_quickhull(cloud) == std::vector<uint32>{0});
}

TEST_CASE("experimental hulls: floating support and same-ray order use exact signs", "[geometry]") {
	using dp = point<double>;
	double const translation = 0x1p60;
	std::vector<dp> const cloud{{-translation, 0}, {1, 0}, {0, 0}, {0, 1}};
	std::vector<uint32> const expected{0, 1, 3};
	REQUIRE(convex_hull_indices(cloud) == expected);
	REQUIRE(atl::experimental::convex_hull_indices_by_graham_scan(cloud) == expected);
	REQUIRE(atl::experimental::convex_hull_indices_by_jarvis_march(cloud) == expected);
	REQUIRE(atl::experimental::convex_hull_indices_by_chan(cloud) == expected);
	REQUIRE(atl::experimental::convex_hull_indices_by_quickhull(cloud) == expected);
}

TEST_CASE("hull merges: mutable transforms and non-common empty branches", "[geometry]") {
	std::vector<pt> a{{0, 0}, {2, 0}, {0, 2}}, b{{3, 0}, {5, 0}, {3, 2}};
	auto mutable_a = a | std::views::transform([offset = pt{}](pt v) mutable { return v + offset; });
	auto mutable_b = b | std::views::transform([offset = pt{}](pt v) mutable { return v + offset; });
	auto all = a;
	all.insert(all.end(), b.begin(), b.end());
	auto const want = convex_hull(all);
	CHECK(atl::geometry::merge_convex_hulls(mutable_a, mutable_b) == want);
	CHECK(atl::experimental::merge_convex_hulls_by_materialized_chains(mutable_a, mutable_b) == want);
	std::vector<pt> const empty;
	auto const segment_range = std::ranges::subrange(std::counted_iterator(a.begin(), 2), std::default_sentinel);
	std::vector<pt> const segment_points(a.begin(), a.begin() + 2);
	CHECK(atl::experimental::merge_convex_hulls_by_materialized_chains(empty, segment_range) == segment_points);
	CHECK(atl::experimental::merge_convex_hulls_by_materialized_chains(segment_range, empty) == segment_points);
}
