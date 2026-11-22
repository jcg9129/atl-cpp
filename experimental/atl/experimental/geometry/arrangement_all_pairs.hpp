#pragma once
#include <atl/geometry/arrangement.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/geometry/segment_intersections.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/detail/int256.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <initializer_list>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file
 * @brief The quadratic all-pairs arrangement builder that lost the race to
 *        the library sweep-based atl::geometry::arrangement_of.
 */
namespace atl::experimental {
	/**
	 * @brief The quadratic arrangement builder raced against the library
	 *        sweep-based @ref atl::geometry::arrangement_of . ALL pairwise
	 *        intersections computed directly (no event queue, no status
	 *        tree), vertices sweep-sorted once, then the SHARED assembly
	 *        (identical topology on identical input, full degeneracy
	 *        contract). Direct pair enumeration removes queue/status costs;
	 *        sorting and indexing the cuts retain a logarithmic factor,
	 *        including in the dense regime k = Theta(n^2).
	 *
	 * The below-anchors the assembly needs for hole nesting are computed on
	 * demand by an O(n) scan (once per inner cycle / isolated vertex, so
	 * only degenerate hole-heavy inputs pay repeatedly).
	 *
	 * @pre Integral coordinates with `|x|, |y| < 2^26` (asserted): the
	 *      on-demand anchor scan compares rational y-values by cross
	 *      multiplication whose operands must fit int128 for the int256
	 *      product comparator. (The library sweep builder allows 2^32.)
	 *
	 * RACE RESULT (the reason this lives in experimental; x86-64 Xeon
	 * 8280, record 2026-10-15, -O2 / -march=native): dense 700 chords
	 * (58.7k crossings) 303 / 292 ms vs the sweep's 295 / 303 — a tie —
	 * and sparse 5000 segments 1920 / 1705 ms vs the sweep's 14.7 / 13.8,
	 * over 120 times slower on sparse inputs.
	 * @note Not constexpr (runtime shared assembly). All scans are iterative.
	 * @complexity Expected O(n^2 + (n + k) log(n + k)), plus O(n) per hole anchor
	 */
	template<geometry::segment_range R>
		requires std::signed_integral<geometry::segment_coordinate_t<R>>
	[[nodiscard]] geometry::arrangement arrangement_of_by_all_pairs(R &&segments) {
		using T = geometry::segment_coordinate_t<R>;
		using geometry::arrangement;
		using geometry::point;
		atl_assert(std::ranges::size(segments) < (uint64(1) << 32));
		uint32 const segment_count = uint32(std::ranges::size(segments));
		auto const seg = [&](uint32 const s) { return std::ranges::begin(segments)[s]; };
		if constexpr (std::signed_integral<T>) {
			[[maybe_unused]] constexpr int64 bound = int64(1) << 26;
			for (uint32 s = 0; s < segment_count; ++s)
				for ([[maybe_unused]] auto const c :
					 {seg(s).a.x, seg(s).a.y, seg(s).b.x, seg(s).b.y})
					atl_assert(-bound < int64(c) && int64(c) < bound);
		}

		// exact rational point (nx/d, ny/d), d > 0, compared in sweep order
		struct rat {
			int128 nx, ny, d;
		};

		auto const rat_less = [](rat const &l, rat const &r) {
			int const by_x = atl::detail::compare_products(l.nx, r.d, r.nx, l.d);
			if (by_x != 0) return by_x < 0;
			return atl::detail::compare_products(l.ny, r.d, r.ny, l.d) < 0;
		};
		auto const rat_eq = [&](rat const &l, rat const &r) {
			return !rat_less(l, r) && !rat_less(r, l);
		};

		struct line {
			int128 a, b, c; // a x + b y + c = 0
		};

		auto const line_of = [&](uint32 const s) {
			int128 const a = int128(seg(s).b.y) - int128(seg(s).a.y), b = int128(seg(s).a.x) - int128(seg(s).b.x);
			return line{a, b, -(a * seg(s).a.x + b * seg(s).a.y)};
		};
		auto const on_segment = [&](uint32 const s, rat const &p) {
			auto const [a, b, c] = line_of(s);
			auto acc = atl::detail::int256::product(a, p.nx);
			acc += atl::detail::int256::product(b, p.ny);
			acc += atl::detail::int256::product(c, p.d);
			if (acc.sign() != 0) return false;
			auto const s2 = seg(s);
			auto const [lo_x, hi_x] = std::minmax(s2.a.x, s2.b.x);
			auto const [lo_y, hi_y] = std::minmax(s2.a.y, s2.b.y);
			return int128(lo_x) * p.d <= p.nx && p.nx <= int128(hi_x) * p.d
				&& int128(lo_y) * p.d <= p.ny && p.ny <= int128(hi_y) * p.d;
		};

		std::vector<std::vector<rat>> cuts(segment_count);
		for (uint32 s = 0; s < segment_count; ++s) {
			cuts[s].push_back({seg(s).a.x, seg(s).a.y, 1});
			cuts[s].push_back({seg(s).b.x, seg(s).b.y, 1});
		}
		for (uint32 i = 0; i < segment_count; ++i)
			for (uint32 j = i + 1; j < segment_count; ++j) {
				if (!intersects(seg(i), seg(j))) continue;
				auto const [ai, bi, ci] = line_of(i);
				auto const [aj, bj, cj] = line_of(j);
				int128 d = ai * bj - aj * bi;
				if (d == 0) { // parallel touch, overlap or degenerate: the
							  // endpoints landing on the other are the cuts
					for (auto const e : {seg(i).a, seg(i).b})
						if (rat const p{e.x, e.y, 1}; on_segment(j, p)) {
							cuts[j].push_back(p);
							cuts[i].push_back(p);
						}
					for (auto const e : {seg(j).a, seg(j).b})
						if (rat const p{e.x, e.y, 1}; on_segment(i, p)) {
							cuts[i].push_back(p);
							cuts[j].push_back(p);
						}
					continue;
				}
				int128 nx = bi * cj - bj * ci, ny = ci * aj - cj * ai;
				if (d < 0) {
					d = -d;
					nx = -nx;
					ny = -ny;
				}
				cuts[i].push_back({nx, ny, d});
				cuts[j].push_back({nx, ny, d});
			}

		std::vector<rat> points;
		for (auto const &list : cuts) points.insert(points.end(), list.begin(), list.end());
		std::ranges::sort(points, rat_less);
		auto const dup = std::ranges::unique(points, rat_eq);
		points.erase(dup.begin(), dup.end());
		atl_assert(points.size() < arrangement::none); // vertex ids are uint32, none reserved
		auto const id_of = [&](rat const &p) {
			return uint32(std::ranges::lower_bound(points, p, rat_less) - points.begin());
		};
		std::vector<point<float64>> vertices;
		vertices.reserve(points.size());
		for (auto const &p : points)
			vertices.push_back({float64(p.nx) / float64(p.d), float64(p.ny) / float64(p.d)});

		std::vector<std::vector<uint32>> along(segment_count);
		for (uint32 s = 0; s < segment_count; ++s) {
			std::ranges::sort(cuts[s], rat_less);
			for (auto const &p : cuts[s]) {
				uint32 const v = id_of(p);
				if (along[s].empty() || along[s].back() != v) along[s].push_back(v);
			}
		}

		// integral directions difference in int64, as the library arrangement_of
		// does: an int16 span overflows its own type
		using Direction = std::conditional_t<std::floating_point<T>, T, int64>;
		auto const forward_direction = [&](uint32 const s) {
			auto const sg2 = seg(s);
			point<Direction> const a{Direction(sg2.a.x), Direction(sg2.a.y)};
			point<Direction> const b{Direction(sg2.b.x), Direction(sg2.b.y)};
			bool const is_forward = a.x != b.x ? a.x < b.x : a.y <= b.y;
			return is_forward ? b - a : a - b;
		};

		// The below-anchor is the highest active segment not through the vertex,
		// max by y at the vertex's x, ties by SLOPE then index, mirroring
		// the sweep's post-event status order. The slope tie is load-bearing:
		// when the vertical ray down from the vertex hits an arrangement
		// VERTEX, the tied edges bound different regions and only the
		// steepest is the wall of the region containing the query.
		auto const slope_less = [&](uint32 const si, uint32 const ti) {
			auto const sd = forward_direction(si), td = forward_direction(ti);
			// widened: direction products reach 2^54, past int32 inside the bound
			return int128(sd.y) * td.x - int128(td.y) * sd.x < 0; // both dx > 0 here
		};
		auto const anchor_of = [&, points = std::move(points)](uint32 const v) {
			rat const &p = points[v];
			uint32 best = arrangement::none;
			int128 best_n = 0, best_d = 1;
			for (uint32 s = 0; s < segment_count; ++s) {
				if (seg(s).a.x == seg(s).b.x) continue; // verticals: never below
				// alive at p in sweep terms: strictly between the segment's
				// sweep-first and sweep-last points (matches the engine's
				// status: a segment starting at p's x below it counts)
				bool const is_forward = seg(s).a.x != seg(s).b.x ? seg(s).a.x < seg(s).b.x
																 : seg(s).a.y <= seg(s).b.y;
				auto const fp = is_forward ? seg(s).a : seg(s).b;
				auto const lp = is_forward ? seg(s).b : seg(s).a;
				if (!rat_less({fp.x, fp.y, 1}, p) || !rat_less(p, {lp.x, lp.y, 1}))
					continue;
				if (on_segment(s, p)) continue; // through p: not below
				auto const [a, b, c] = line_of(s);
				int128 n = -(a * p.nx + c * p.d), dd = b * p.d;
				if (dd < 0) {
					n = -n;
					dd = -dd;
				}
				// strictly below p, higher than the best so far
				if (atl::detail::compare_products(n, p.d, p.ny, dd) >= 0) continue;
				int const versus_best =
					best == arrangement::none
						? 1
						: atl::detail::compare_products(n, best_d, best_n, dd);
				if (versus_best > 0
					|| (versus_best == 0
						&& (slope_less(best, s) || (!slope_less(s, best) && s > best)))) {
					best = s;
					best_n = n;
					best_d = dd;
				}
			}
			if (best == arrangement::none)
				return std::pair{arrangement::none, arrangement::none};
			// the piece of `best` spanning p's x: ids are sweep-ordered, so
			// binary search for the last cut before v
			auto const &list = along[best];
			uint32 const j =
				uint32(std::ranges::upper_bound(list, v) - list.begin()) - 1;
			return std::pair{best, j};
		};

		return geometry::detail::assemble_arrangement<Direction>(std::move(vertices), along,
																 forward_direction, anchor_of);
	}
}
