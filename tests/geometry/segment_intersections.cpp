#include <atl/geometry/line.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/geometry/segment_intersections.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/detail/int256.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <ranges>
#include <vector>
#include "../expect_abort.hpp"

using atl::int128;
using atl::geometry::intersection_kind;
using atl::geometry::point;
using atl::geometry::segment;
using atl::geometry::segment_index;
using atl::geometry::segment_intersections;
using atl::geometry::segment_intersections_view;

namespace {
	using sg = segment<std::int64_t>;

	bool brute_intersects(sg const s, sg const t) {
		auto const orient = [](auto a, auto b, auto c)
			requires atl::geometry::coordinate<typename decltype(a)::coordinate_type>
		{
			atl::int128 const dx = atl::int128(b.x) - a.x, dy = atl::int128(b.y) - a.y;
			atl::int128 const value = dx * (atl::int128(c.y) - a.y) - dy * (atl::int128(c.x) - a.x);
			return int(value > 0) - int(value < 0);
		};
		auto const on = [&](auto a, auto b, auto q)
			requires atl::geometry::coordinate<typename decltype(a)::coordinate_type>
		{
			return orient(a, b, q) == 0 && (std::min)(a.x, b.x) <= q.x && q.x <= (std::max)(a.x, b.x)
				&& (std::min)(a.y, b.y) <= q.y && q.y <= (std::max)(a.y, b.y);
		};
		int const a = orient(s.a, s.b, t.a), b = orient(s.a, s.b, t.b);
		int const c = orient(t.a, t.b, s.a), d = orient(t.a, t.b, s.b);
		return (a * b < 0 && c * d < 0) || on(s.a, s.b, t.a) || on(s.a, s.b, t.b)
			|| on(t.a, t.b, s.a) || on(t.a, t.b, s.b);
	}

	// exact rational point for the oracle: (nx/d, ny/d), d > 0
	struct rat_point {
		int128 nx, ny, d;

		static rat_point of(std::int64_t x, std::int64_t y) { return {x, y, 1}; }

		friend bool operator<(rat_point const &l, rat_point const &r) {
			// int256 compares: crossing numerators overflow int128 cross-
			// multiplication already at 2^31 coordinates
			int const by_x = atl::detail::compare_products(l.nx, r.d, r.nx, l.d);
			if (by_x != 0) return by_x < 0;
			return atl::detail::compare_products(l.ny, r.d, r.ny, l.d) < 0;
		}

		friend bool operator==(rat_point const &l, rat_point const &r) {
			return atl::detail::compare_products(l.nx, r.d, r.nx, l.d) == 0
				&& atl::detail::compare_products(l.ny, r.d, r.ny, l.d) == 0;
		}
	};

	bool on_segment(sg const s, rat_point const &p) {
		int128 const a = s.b.y - s.a.y, b = s.a.x - s.b.x;
		int128 const c = -(a * s.a.x + b * s.a.y);
		auto acc = atl::detail::int256::product(a, p.nx); // a*nx overflows
		acc += atl::detail::int256::product(b, p.ny);	  // int128 at 2^31
		acc += atl::detail::int256::product(c, p.d);
		if (acc.sign() != 0) return false;
		auto const [lo_x, hi_x] = std::minmax(s.a.x, s.b.x);
		auto const [lo_y, hi_y] = std::minmax(s.a.y, s.b.y);
		return int128(lo_x) * p.d <= p.nx && p.nx <= int128(hi_x) * p.d
			&& int128(lo_y) * p.d <= p.ny && p.ny <= int128(hi_y) * p.d;
	}

	rat_point sweep_first(sg const s) {
		bool const forward = s.a.x != s.b.x ? s.a.x < s.b.x : s.a.y <= s.b.y;
		auto const e = forward ? s.a : s.b;
		return rat_point::of(e.x, e.y);
	}

	rat_point sweep_last(sg const s) {
		bool const forward = s.a.x != s.b.x ? s.a.x < s.b.x : s.a.y <= s.b.y;
		auto const e = forward ? s.b : s.a;
		return rat_point::of(e.x, e.y);
	}

	struct oracle_event {
		std::vector<segment_index> starting, ending, passing;
	};

	// O(n^2 * points) reference: candidate points = pairwise intersections
	// (crossings + touch points); at each, classify every segment exactly.
	std::map<rat_point, oracle_event> oracle(std::vector<sg> const &segs) {
		std::vector<rat_point> candidates;
		for (std::size_t i = 0; i < segs.size(); ++i)
			for (std::size_t j = i + 1; j < segs.size(); ++j) {
				sg const s = segs[i], t = segs[j];
				if (!brute_intersects(s, t)) continue;
				int128 const as = s.b.y - s.a.y, bs = s.a.x - s.b.x;
				int128 const cs = -(as * s.a.x + bs * s.a.y);
				int128 const at2 = t.b.y - t.a.y, bt2 = t.a.x - t.b.x;
				int128 const ct2 = -(at2 * t.a.x + bt2 * t.a.y);
				int128 d = as * bt2 - at2 * bs;
				if (d == 0) { // touching parallels/degenerates: endpoints on
							  // the other segment are the touch points
					for (auto const e : {s.a, s.b})
						if (on_segment(t, rat_point::of(e.x, e.y)))
							candidates.push_back(rat_point::of(e.x, e.y));
					for (auto const e : {t.a, t.b})
						if (on_segment(s, rat_point::of(e.x, e.y)))
							candidates.push_back(rat_point::of(e.x, e.y));
					continue;
				}
				int128 nx = bs * ct2 - bt2 * cs, ny = cs * at2 - ct2 * as;
				if (d < 0) {
					d = -d;
					nx = -nx;
					ny = -ny;
				}
				candidates.push_back({nx, ny, d});
			}
		std::map<rat_point, oracle_event> events;
		for (auto const &p : candidates) {
			if (events.contains(p)) continue;
			oracle_event ev;
			for (segment_index k = 0; k < segment_index(segs.size()); ++k) {
				if (!on_segment(segs[k], p)) continue;
				bool const at_first = sweep_first(segs[k]) == p;
				bool const at_last = sweep_last(segs[k]) == p;
				if (at_first) ev.starting.push_back(k);
				if (at_last) ev.ending.push_back(k);
				if (!at_first && !at_last) ev.passing.push_back(k);
			}
			if (ev.starting.size() + ev.passing.size()
					+ (ev.ending.size() - [&] { // degenerates are in both
						  std::size_t both = 0;
						  for (auto const k : ev.starting)
							  if (segs[k].a == segs[k].b) ++both;
						  return both;
					  }())
				>= 2)
				events.emplace(p, std::move(ev));
		}
		return events;
	}

	// The brute below-neighbor: among non-vertical segments alive at p and
	// not through it, the highest strictly under p (any of the exact-tie
	// set is a valid answer). Exact rational compares — int128 suffices
	// only away from the coordinate bound, so callers gate on magnitude.
	std::vector<segment_index> below_candidates(std::vector<sg> const &segs,
												rat_point const &p) {
		struct rat_y {
			int128 n, d; // n/d, d > 0
		};

		auto const y_at = [&](sg const s2) {
			int128 const a = s2.b.y - s2.a.y, b = s2.a.x - s2.b.x;
			int128 const c = -(a * s2.a.x + b * s2.a.y);
			int128 n = -(a * p.nx + c * p.d), dd = b * p.d;
			if (dd < 0) {
				n = -n;
				dd = -dd;
			}
			return rat_y{n, dd};
		};
		auto const cmp = [](rat_y const l, rat_y const r) {
			return atl::detail::compare_products(l.n, r.d, r.n, l.d);
		};
		std::vector<segment_index> best_set;
		rat_y best{};
		for (segment_index k = 0; k < segment_index(segs.size()); ++k) {
			if (segs[k].a.x == segs[k].b.x) continue; // never below (see engine)
			if (on_segment(segs[k], p)) continue;	  // through p: in the block
			if (!(sweep_first(segs[k]) < p) || !(p < sweep_last(segs[k])))
				continue; // not alive at p
			auto const y = y_at(segs[k]);
			if (atl::detail::compare_products(y.n, p.d, p.ny, y.d) >= 0)
				continue; // not strictly below
			if (best_set.empty() || cmp(best, y) < 0) {
				best = y;
				best_set = {k};
			} else if (cmp(y, best) == 0)
				best_set.push_back(k);
		}
		return best_set;
	}

	void verify(std::vector<sg> const &segs) {
		bool can_check_below = true;
		for (auto const &s2 : segs)
			for (auto const c :
				 {s2.a.x, s2.a.y, s2.b.x, s2.b.y})
				can_check_below = can_check_below && std::abs(c) <= (std::int64_t(1) << 20);
		auto const expected = oracle(segs);
		auto const got = segment_intersections(segs);
		REQUIRE(got.size() == expected.size());
		std::size_t at = 0;
		for (auto const &[p, ev] : expected) { // map iterates in sweep order
			auto const &g = got[at++];
			// the reported double point must match the exact rational
			double const expected_x = double(p.nx) / double(p.d);
			REQUIRE(std::abs(g.at.x - expected_x) <= 1e-12 * (1 + std::abs(expected_x)));
			double const expected_y = double(p.ny) / double(p.d);
			REQUIRE(std::abs(g.at.y - expected_y) <= 1e-12 * (1 + std::abs(expected_y)));
			auto sorted = [](std::vector<segment_index> v) {
				std::ranges::sort(v);
				return v;
			};
			REQUIRE(sorted(g.starting) == sorted(ev.starting));
			REQUIRE(sorted(g.ending) == sorted(ev.ending));
			REQUIRE(sorted(g.passing) == sorted(ev.passing));
			if (can_check_below) { // int128 brute overflows near the coordinate bound
				auto const cands = below_candidates(segs, p);
				if (cands.empty())
					REQUIRE(g.below == atl::geometry::intersection_event::no_below);
				else
					REQUIRE(std::ranges::contains(cands, g.below));
			}
		}
	}
}

TEST_CASE("segment_intersections: random degenerate-dense grids vs the oracle",
		  "[geometry]") {
	auto seed = GENERATE(take(60, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = std::size_t(2 + rng() % 11);
	std::int64_t const span = 2 + std::int64_t(rng() % 5);
	std::vector<sg> segs;
	for (std::size_t k = 0; k < n; ++k) {
		auto const coord = [&] { return std::int64_t(rng() % (span + 1)); };
		sg s{{coord(), coord()}, {coord(), coord()}};
		if (rng() % 7 == 0) s.b = s.a; // degenerate point segments
		segs.push_back(s);
	}
	// overlap-rich mode: fold some segments onto a shared collinear family
	if (rng() % 2) {
		std::int64_t const ox = std::int64_t(rng() % 3) - 1, oy = std::int64_t(rng() % 3) - 1;
		std::int64_t const dx = std::int64_t(rng() % 3) - 1, dy = std::int64_t(rng() % 3);
		if (dx != 0 || dy != 0)
			for (auto &s2 : segs)
				if (rng() % 2) {
					std::int64_t const t0 = std::int64_t(rng() % 5) - 2;
					std::int64_t const t1 = t0 + std::int64_t(rng() % 4);
					s2 = {{ox + t0 * dx, oy + t0 * dy}, {ox + t1 * dx, oy + t1 * dy}};
				}
	}
	verify(segs);
}

TEST_CASE("segment_intersections: handcrafted degeneracies", "[geometry]") {
	SECTION("k-star: one event listing all segments") {
		std::vector<sg> star;
		for (std::int64_t d = 1; d <= 5; ++d) {
			star.push_back({{-d, -d * d}, {d, d * d}}); // distinct slopes through 0
		}
		star.push_back({{0, -7}, {0, 9}}); // a vertical through 0 too
		auto const events = segment_intersections(star);
		REQUIRE(events.size() == 1);
		REQUIRE(events[0].passing.size() == 6);
		REQUIRE(std::abs(events[0].at.x - (0.0)) <= 1e-12);
		REQUIRE(std::abs(events[0].at.y - (0.0)) <= 1e-12);
	}
	SECTION("grid: verticals x horizontals, all crossings") {
		std::vector<sg> grid;
		for (std::int64_t x = 0; x < 4; ++x) grid.push_back({{x, -1}, {x, 4}});
		for (std::int64_t y = 0; y < 3; ++y) grid.push_back({{-1, y}, {4, y}});
		auto const events = segment_intersections(grid);
		REQUIRE(events.size() == 12);
		for (auto const &e : events) REQUIRE(e.passing.size() == 2);
		verify(grid);
	}
	SECTION("shared endpoint fan + T junction + point on interior") {
		std::vector<sg> zoo{{{0, 0}, {4, 0}}, {{0, 0}, {4, 3}}, {{0, 0}, {0, 5}}, {{2, -2}, {2, 0}}, {{3, 3}, {3, 3}}, {{1, 0}, {1, 0}}};
		// events: (0,0) fan of 3 starters; (2,0) T-junction (segment 3 ends
		// on segment 0's interior); (1,0) point-segment on interior
		verify(zoo);
		auto const events = segment_intersections(zoo);
		REQUIRE(events.size() == 3);
		REQUIRE(events[0].starting.size() == 3);			// (0,0)
		REQUIRE(std::abs(events[1].at.x - (1.0)) <= 1e-12); // point-segment event
		REQUIRE(std::abs(events[2].at.x - (2.0)) <= 1e-12); // T junction
	}
	SECTION("no intersections / singletons / empty") {
		REQUIRE(segment_intersections(std::vector<sg>{}).empty());
		REQUIRE(segment_intersections(std::vector<sg>{{{0, 0}, {5, 5}}}).empty());
		REQUIRE(segment_intersections(
					std::vector<sg>{{{0, 0}, {1, 0}}, {{3, 3}, {4, 3}}, {{9, 9}, {9, 9}}})
					.empty());
	}
	SECTION("collinear touching at one point is legal (not an overlap)") {
		std::vector<sg> chain{{{0, 0}, {2, 2}}, {{2, 2}, {5, 5}}};
		auto const events = segment_intersections(chain);
		REQUIRE(events.size() == 1);
		REQUIRE(events[0].starting.size() == 1);
		REQUIRE(events[0].ending.size() == 1);
		verify(chain);
	}
}

TEST_CASE("segment_intersections: early termination pays only for the prefix",
		  "[geometry]") {
	std::mt19937_64 rng(7);
	std::vector<sg> segs;
	for (int k = 0; k < 300; ++k) {
		std::int64_t const x = std::int64_t(rng() % 1000), y = std::int64_t(rng() % 1000);
		segs.push_back({{x, y}, {x + 1 + std::int64_t(rng() % 40), y + 1 + std::int64_t(rng() % 40)}});
	}
	auto const all = segment_intersections(segs);
	segment_intersections_view view(segs);
	auto it = view.begin();
	if (all.empty()) {
		REQUIRE(it == std::default_sentinel);
	} else {
		REQUIRE(it != std::default_sentinel);
		auto const first = *it;
		REQUIRE(std::abs(first.at.x - (all.front().at.x)) <= 1e-12);
		REQUIRE(std::abs(first.at.y - (all.front().at.y)) <= 1e-12);
	}
	// the sweep's status comparator points at the engine, so neither the
	// engine nor the view may be copied or moved
	static_assert(!std::movable<segment_intersections_view<std::vector<sg> &>>);
	static_assert(!std::copy_constructible<atl::geometry::detail::segment_sweep<std::int64_t>>);
	// an rvalue vector moves into the sweep; the view owns its copy
	segment_intersections_view moved{std::vector<sg>(segs)};
	REQUIRE((moved.begin() == std::default_sentinel) == all.empty());
}

TEST_CASE("segment_intersections: coordinates near the 2^32 bound", "[geometry]") {
	std::int64_t const c = (std::int64_t(1) << 31);
	std::vector<sg> segs{{{-c, -c}, {c, c}},
						 {{-c, c}, {c, -c}},
						 {{-c, 1 - c}, {c, 1 - c}}}; // near-bottom horizontal
	auto const events = segment_intersections(segs);
	REQUIRE(events.size() == 3); // the X at (0,0) + two horizontal crossings
	REQUIRE(std::abs(events[1].at.x - (0.0)) <= 1e-12);
	REQUIRE(std::abs(events[1].at.y - (0.0)) <= 1e-12);
	// below at the bound (the in-verify brute would overflow): only the
	// origin crossing has anything under it — the near-bottom horizontal
	REQUIRE(events[0].below == atl::geometry::intersection_event::no_below);
	REQUIRE(events[1].below == 2);
	REQUIRE(events[2].below == atl::geometry::intersection_event::no_below);
	verify(segs);
}

TEST_CASE("segment_intersections: double coordinates, parity spot check", "[geometry]") {
	std::vector<segment<double>> segs{{{0, 0}, {10, 10}},
									  {{0, 10}, {10, 0}},
									  {{2.5, 0}, {2.5, 10}}};
	auto const events = segment_intersections(segs);
	REQUIRE(events.size() == 3);
	REQUIRE(std::abs(events[0].at.x - (2.5)) <= 1e-12);
	REQUIRE(std::abs(events[2].at.x - (5.0)) <= 1e-12);
	REQUIRE(std::abs(events[2].at.y - (5.0)) <= 1e-12);
	// below parity: (2.5, 7.5) sits over the main diagonal; the other two
	// events have nothing under them
	REQUIRE(events[0].below == atl::geometry::intersection_event::no_below);
	REQUIRE(events[1].below == 0);
	REQUIRE(events[2].below == atl::geometry::intersection_event::no_below);
}

TEST_CASE("segment_intersections: collinear overlaps, set-change semantics",
		  "[geometry]") {
	SECTION("partial overlap: both overlap endpoints reported") {
		std::vector<sg> pair{{{0, 0}, {6, 6}}, {{2, 2}, {9, 9}}};
		auto const events = segment_intersections(pair);
		REQUIRE(events.size() == 2);
		REQUIRE(std::abs(events[0].at.x - (2.0)) <= 1e-12); // second starts on the first
		REQUIRE(events[0].starting.size() == 1);
		REQUIRE(events[0].passing.size() == 1);
		REQUIRE(std::abs(events[1].at.x - (6.0)) <= 1e-12); // first ends on the second
		REQUIRE(events[1].ending.size() == 1);
		REQUIRE(events[1].passing.size() == 1);
		verify(pair);
	}
	SECTION("identical duplicates and nested overlaps") {
		std::vector<sg> dup{{{0, 0}, {5, 1}}, {{0, 0}, {5, 1}}};
		auto const events = segment_intersections(dup);
		REQUIRE(events.size() == 2); // both shared endpoints
		REQUIRE(events[0].starting.size() == 2);
		REQUIRE(events[1].ending.size() == 2);
		std::vector<sg> nested{{{0, 0}, {9, 0}}, {{3, 0}, {5, 0}}};
		auto const inner = segment_intersections(nested);
		REQUIRE(inner.size() == 2); // the inner segment's endpoints
		REQUIRE(std::abs(inner[0].at.x - (3.0)) <= 1e-12);
		REQUIRE(std::abs(inner[1].at.x - (5.0)) <= 1e-12);
		verify(dup);
		verify(nested);
	}
	SECTION("vertical overlaps") {
		std::vector<sg> vert{{{4, 0}, {4, 6}}, {{4, 3}, {4, 9}}, {{4, 9}, {4, 9}}};
		auto const events = segment_intersections(vert);
		// (4,3): second starts on first; (4,6): first ends on second;
		// (4,9): the point segment at the second's endpoint
		REQUIRE(events.size() == 3);
		verify(vert);
	}
	SECTION("crossing through an overlap interior lists all three") {
		std::vector<sg> mix{{{0, 0}, {8, 0}}, {{2, 0}, {6, 0}}, {{4, -3}, {4, 5}}};
		auto const events = segment_intersections(mix);
		// (2,0), (4,0) with all three, (6,0)
		REQUIRE(events.size() == 3);
		REQUIRE(std::abs(events[1].at.x - (4.0)) <= 1e-12);
		REQUIRE(events[1].passing.size() == 3);
		verify(mix);
	}
	SECTION("three-way collinear overlap ladder") {
		std::vector<sg> ladder{{{0, 0}, {4, 2}}, {{2, 1}, {8, 4}}, {{4, 2}, {12, 6}}};
		verify(ladder);
		auto const events = segment_intersections(ladder);
		REQUIRE(events.size() == 3); // set changes at (2,1), (4,2), (8,4)
	}
}

TEST_CASE("segment intersections: full-range int32 and all precision coordinate types", "[geometry]") {
	using p32 = point<std::int32_t>;
	std::int32_t const lo = (std::numeric_limits<std::int32_t>::min)(), hi = (std::numeric_limits<std::int32_t>::max)();
	std::vector<segment<std::int32_t>> const segments{{p32{lo, lo}, p32{hi, hi}}, {p32{lo, hi}, p32{hi, lo}}};
	auto const events = segment_intersections(segments);
	REQUIRE(events.size() == 1);
	REQUIRE(atl::geometry::approx{1e-12}.equal(events[0].at, point{-0.5, -0.5}));
	verify({{{lo, lo}, {hi, hi}}, {{lo, hi}, {hi, lo}}});
	std::vector<segment<float>> const floating{{{0, 0}, {4, 4}}, {{0, 4}, {4, 0}}};
	auto const floating_events = segment_intersections(floating);
	REQUIRE(floating_events.size() == 1);
	REQUIRE(atl::geometry::approx{1e-12}.equal(floating_events[0].at, point{2.0, 2.0}));
}

TEST_CASE("segment sweep: non-common and mutable segment views", "[geometry]") {
	std::array const segments{segment<int>{{0, 0}, {4, 4}}, segment<int>{{0, 4}, {4, 0}}};
	auto non_common = std::ranges::subrange(std::counted_iterator(segments.begin(), 2), std::default_sentinel);
	REQUIRE(segment_intersections(non_common).size() == 1);
	auto mutable_view = std::views::iota(0, 2) | std::views::transform([segments](int i) mutable { return segments[std::size_t(i)]; });
	REQUIRE(segment_intersections(mutable_view).size() == 1);
}

TEST_CASE("segment sweep: coordinate bounds are asserted", "[geometry]") {
	REQUIRE(atl_test::aborts([] {
		std::int64_t const bound = std::int64_t(1) << 32;
		(void)segment_intersections(std::vector<sg>{{{bound, 0}, {0, 0}}});
	}));
}

TEST_CASE("segment_intersections: simultaneous x crossings keep resident order", "[geometry]") {
	verify({{{0, 0}, {2, 2}}, {{0, 2}, {2, 0}}, {{0, 3}, {2, 5}}, {{0, 5}, {2, 3}}});
}
