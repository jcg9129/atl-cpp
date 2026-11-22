#pragma once
#include <atl/geometry/calipers.hpp>
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Benchmark contenders for the STRICT convex hull, raced against the
 *        library Andrew monotone chain (atl/geometry/convex_hull.hpp), plus
 *        the hull-of-two-hulls contenders that lost to the library
 *        atl::geometry::merge_convex_hulls.
 *
 * The four hull contenders return original indices in CCW order starting at
 * the lexicographically smallest point, so their outputs must be IDENTICAL
 * to convex_hull_indices under exact — locked to an independent brute-force
 * oracle in the tests. Duplicates collapse to their smallest index. Non-exact
 * policies need geometrically coherent signs; the angular scan additionally
 * requires a strict weak ordering from its policy comparisons.
 */
namespace atl::experimental {
	namespace detail {
		/** @brief Exact distance order on one ray without rounded norms.
		 * @pre a and b lie on the same ray from origin.
		 */
		template<geometry::coordinate T>
		[[nodiscard]] constexpr bool is_farther_on_ray(geometry::point<T> const origin, geometry::point<T> const a, geometry::point<T> const b) {
			if (a.y != b.y) return a.y >= origin.y ? b.y > a.y : b.y < a.y;
			return a.x >= origin.x ? b.x > a.x : b.x < a.x;
		}

		/** @brief Lexicographically sorted distinct positions, each with its smallest input index. */
		template<geometry::vertex_range R, geometry::sign_policy<geometry::vertex_coordinate_t<R>> Policy>
		[[nodiscard]] constexpr std::vector<geometry::detail::hull_entry<geometry::vertex_coordinate_t<R>>>
		distinct_entries(R &&points, [[maybe_unused]] Policy const p) {
			using T = geometry::vertex_coordinate_t<R>;
			atl_assert(std::ranges::size(points) < (uint64(1) << 32));
			auto const first = std::ranges::begin(points);
			std::vector<geometry::detail::hull_entry<T>> e;
			e.reserve(std::ranges::size(points));
			for (uint32 i = 0; i < uint32(std::ranges::size(points)); ++i)
				e.push_back({first[i], i});
			std::ranges::sort(e, [](geometry::detail::hull_entry<T> const &a, geometry::detail::hull_entry<T> const &b) {
				return a.position != b.position ? a.position < b.position : a.index < b.index;
			});
			auto const [d0, d1] =
				std::ranges::unique(e, [](geometry::detail::hull_entry<T> const &a, geometry::detail::hull_entry<T> const &b) { return a.position == b.position; });
			e.erase(d0, d1);
			return e;
		}
	}

	/**
	 * @brief Graham scan: ONE angular sort around the (y, x)-smallest pivot
	 *        (same-ray ties by raw y, x, then index, so duplicates sit together and
	 *        keep their smallest index), one stack pass; the comparator
	 *        costs one orientation per compare. Intel Xeon Platinum 8280,
	 *        GCC 15.2.0, RelWithDebInfo, assertions off, 2026-10-15,
	 *        -O2 / -O2 -march=native: uniform 256 / 287 ms and circle
	 *        257 / 287 ms, against library 12.1 / 11.3 and 132 / 132 ms.
	 *        Angular comparisons lose to raw lexicographic ordering.
	 * @param points Input cloud, not modified.
	 * @param p Predicate policy.
	 * @return Strict hull indices CCW from lex-min, smallest index per duplicate.
	 * @pre Fewer than 2^32 points; coordinate bounds as atl::geometry::convex_hull.
	 * @complexity O(n log n)
	 */
	template<geometry::vertex_range R,
			 geometry::sign_policy<geometry::vertex_coordinate_t<R>> Policy = geometry::exact>
	[[nodiscard]] constexpr std::vector<uint32> convex_hull_indices_by_graham_scan(R &&points, Policy const p = {}) {
		using T = geometry::vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(points) < (uint64(1) << 32));
		auto const first = std::ranges::begin(points);
		uint32 const n = uint32(std::ranges::size(points));
		std::vector<geometry::detail::hull_entry<T>> e;
		e.reserve(n);
		for (uint32 i = 0; i < n; ++i) e.push_back({first[i], i});
		std::vector<uint32> hull;
		if (n == 0) return hull;
		auto const pivot_it = std::ranges::min_element(e, [](geometry::detail::hull_entry<T> const &a, geometry::detail::hull_entry<T> const &b) {
			if (a.position.y != b.position.y) return a.position.y < b.position.y;
			return a.position.x != b.position.x ? a.position.x < b.position.x : a.index < b.index;
		});
		std::swap(*e.begin(), *pivot_it);
		auto const pivot = e[0].position;
		std::ranges::sort(e.begin() + 1, e.end(), [&](geometry::detail::hull_entry<T> const &a, geometry::detail::hull_entry<T> const &b) {
			int const o = p.orientation(pivot, a.position, b.position);
			if (o != 0) return o > 0;
			// The (y, x)-minimum pivot puts every ray in the upper half-plane;
			// increasing y, then x, orders a ray exactly without a rounded norm.
			if (a.position.y != b.position.y) return a.position.y < b.position.y;
			return a.position.x != b.position.x ? a.position.x < b.position.x : a.index < b.index;
		});
		auto const [d0, d1] =
			std::ranges::unique(e, [](geometry::detail::hull_entry<T> const &a, geometry::detail::hull_entry<T> const &b) { return a.position == b.position; });
		e.erase(d0, d1);
		if (e.size() <= 2) {
			for (auto const &x : e) hull.push_back(x.index);
			if (e.size() == 2 && e[1].position < e[0].position) std::swap(hull[0], hull[1]);
			return hull;
		}
		// no last-ray reversal: for the STRICT hull the o == 0 pops already
		// keep only the farthest point of every ray, the closing one included
		std::vector<uint32> stack;
		stack.reserve(e.size());
		auto const at = [&](uint32 const k) { return e[k].position; };
		for (uint32 k = 0; k < uint32(e.size()); ++k) {
			while (stack.size() >= 2 && p.orientation(at(stack[stack.size() - 2]), at(stack.back()), at(k)) <= 0)
				stack.pop_back();
			stack.push_back(k);
		}
		std::size_t start = 0;
		for (std::size_t k = 1; k < stack.size(); ++k)
			if (at(stack[k]) < at(stack[start])) start = k;
		hull.reserve(stack.size());
		for (std::size_t k = 0; k < stack.size(); ++k) hull.push_back(e[stack[(start + k) % stack.size()]].index);
		return hull;
	}

	/**
	 * @brief Gift wrapping with a full point scan per extreme vertex.
	 *        Intel Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo,
	 *        assertions off, 2026-10-15, -O2 / -O2 -march=native:
	 *        uniform 1e6 int64 (h about 30) 333 / 406 ms against library
	 *        12.1 / 11.3 ms. The O(n h) circle workload is excluded.
	 * @param points Input cloud, not modified.
	 * @param p Predicate policy.
	 * @return Strict hull indices CCW from lex-min, smallest index per duplicate.
	 * @pre Fewer than 2^32 points; coordinate bounds as atl::geometry::convex_hull.
	 * @complexity O(n log n) dedupe + O(n h) orientation scans
	 */
	template<geometry::vertex_range R,
			 geometry::sign_policy<geometry::vertex_coordinate_t<R>> Policy = geometry::exact>
	[[nodiscard]] constexpr std::vector<uint32> convex_hull_indices_by_jarvis_march(R &&points, Policy const p = {}) {
		auto const e = detail::distinct_entries(points, p);
		if (e.size() <= 2) {
			std::vector<uint32> out;
			for (auto const &x : e) out.push_back(x.index);
			return out;
		}
		std::vector<uint32> hull;
		uint32 const start = 0;
		uint32 cur = start;
		do {
			hull.push_back(e[cur].index);
			uint32 best = cur == 0 ? 1 : 0;
			for (uint32 c = 0; c < uint32(e.size()); ++c) {
				if (c == cur || c == best) continue;
				int const o = p.orientation(e[cur].position, e[best].position, e[c].position);
				if (o < 0
					|| (o == 0
						&& detail::is_farther_on_ray(e[cur].position, e[best].position, e[c].position)))
					best = c;
			}
			cur = best;
		} while (cur != start);
		return hull;
	}

	/**
	 * @brief Quickhull: farthest-point splitting with a heap task stack, sort-free: the
	 *        lexicographic extremes come from one linear scan and each
	 *        split partitions its candidate range in place. Call-stack depth is O(1). Ties between
	 *        equally distant candidates go to the lexicographically smallest
	 *        (then the smallest index), which is an end of the tied run and
	 *        the duplicate the library hull keeps. Intel Xeon Platinum 8280,
	 *        GCC 15.2.0, RelWithDebInfo, assertions off, 2026-10-15,
	 *        -O2 / -O2 -march=native: uniform 37.5 / 42.2 ms, cluster
	 *        32.3 / 36.3 and circle 289 / 342, against library 12.1 / 11.3,
	 *        12.5 / 11.5 and 132 / 132. Heap-task costs need measurement.
	 * @param points Input cloud, not modified.
	 * @param p Predicate policy.
	 * @return Strict hull indices CCW from lex-min, smallest index per duplicate.
	 * @pre Fewer than 2^32 points; coordinate bounds as atl::geometry::convex_hull.
	 * @complexity O(n log n) average, O(n^2) worst with adversarial split balance, O(n) heap memory and O(1) call-stack depth
	 */
	template<geometry::vertex_range R,
			 geometry::sign_policy<geometry::vertex_coordinate_t<R>> Policy = geometry::exact>
	[[nodiscard]] constexpr std::vector<uint32> convex_hull_indices_by_quickhull(R &&points, Policy const p = {}) {
		using T = geometry::vertex_coordinate_t<R>;
		using entry = geometry::detail::hull_entry<T>;
		atl_assert(std::ranges::size(points) < (uint64(1) << 32));
		auto const first = std::ranges::begin(points);
		uint32 const n = uint32(std::ranges::size(points));
		std::vector<uint32> hull;
		if (n == 0) return hull;
		std::vector<entry> e;
		e.reserve(n);
		for (uint32 i = 0; i < n; ++i) e.push_back({first[i], i});
		auto const lex_less = [](entry const &l, entry const &r) {
			return l.position != r.position ? l.position < r.position : l.index < r.index;
		};
		entry const a = *std::ranges::min_element(e, lex_less);
		entry const b = *std::ranges::max_element(e, [](entry const &l, entry const &r) {
			return l.position != r.position ? l.position < r.position : l.index > r.index;
		});
		if (a.position == b.position) return {a.index};

		struct split_task {
			entry u, v;
			std::size_t lo, hi;
			bool is_vertex;
		};

		auto const side = [&](entry const first_vertex, entry const last_vertex, std::size_t const begin, std::size_t const finish) {
			std::vector<split_task> pending{{first_vertex, last_vertex, begin, finish, false}};
			while (!pending.empty()) {
				auto const task = pending.back();
				pending.pop_back();
				auto const u = task.u, v = task.v;
				auto const lo = task.lo, hi = task.hi;
				if (task.is_vertex) {
					hull.push_back(u.index);
					continue;
				}
				if (lo == hi) continue;
				std::size_t farthest = lo;
				for (std::size_t c = lo + 1; c < hi; ++c) {
					int const order = p.cross_sign(u.position, v.position, e[farthest].position, e[c].position);
					if (order < 0 || (order == 0 && lex_less(e[c], e[farthest]))) farthest = c;
				}
				entry const f = e[farthest];
				auto const first_end =
					std::ranges::partition(e.begin() + std::ptrdiff_t(lo), e.begin() + std::ptrdiff_t(hi),
										   [&](entry const &c) { return p.orientation(u.position, f.position, c.position) < 0; })
						.begin();
				auto const second_end = std::ranges::partition(first_end, e.begin() + std::ptrdiff_t(hi), [&](entry const &c) {
											return p.orientation(f.position, v.position, c.position) < 0;
										}).begin();
				std::size_t const mid = std::size_t(first_end - e.begin()), end = std::size_t(second_end - e.begin());
				pending.push_back({f, v, mid, end, false});
				pending.push_back({f, f, 0, 0, true});
				pending.push_back({u, f, lo, mid, false});
			}
		};
		// [0, below) strictly below a->b, [below, above) strictly above
		auto const below_end =
			std::ranges::partition(e, [&](entry const &c) { return p.orientation(a.position, b.position, c.position) < 0; })
				.begin();
		auto const above_end = std::ranges::partition(below_end, e.end(), [&](entry const &c) {
								   return p.orientation(a.position, b.position, c.position) > 0;
							   }).begin();
		std::size_t const below = std::size_t(below_end - e.begin()), above = std::size_t(above_end - e.begin());
		hull.push_back(a.index);
		side(a, b, 0, below);
		hull.push_back(b.index);
		// walking back: candidates right of b->a are the ones above a->b
		side(b, a, below, above);
		return hull;
	}

	/**
	 * @brief Chan's output-sensitive algorithm: mini-hulls over groups of m,
	 *        then a gift wrap whose per-group tangents advance monotonically
	 *        (the rotating-pointer variant); m squares until the wrap closes.
	 *        Intel Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo,
	 *        assertions off, 2026-10-15, -O2 / -O2 -march=native:
	 *        uniform 278 / 279 ms, circle 462 / 479, against library
	 *        12.1 / 11.3 and 132 / 132. Mini-hull constants dominate.
	 * @param points Input cloud, not modified.
	 * @param p Predicate policy.
	 * @return Strict hull indices CCW from lex-min, smallest index per duplicate.
	 * @pre Fewer than 2^32 points; coordinate bounds as atl::geometry::convex_hull.
	 * @complexity O(n log n) dedupe + O(n log h) wrap
	 */
	template<geometry::vertex_range R,
			 geometry::sign_policy<geometry::vertex_coordinate_t<R>> Policy = geometry::exact>
	[[nodiscard]] constexpr std::vector<uint32> convex_hull_indices_by_chan(R &&points, Policy const p = {}) {
		using T = geometry::vertex_coordinate_t<R>;
		auto const e = detail::distinct_entries(points, p);
		uint32 const m = uint32(e.size());
		std::vector<uint32> out;
		if (m <= 2) {
			for (auto const &x : e) out.push_back(x.index);
			return out;
		}
		std::vector<geometry::point<T>> pts(m);
		for (uint32 i = 0; i < m; ++i) pts[i] = e[i].position;

		auto const beats = [&](uint32 const cur, uint32 const c1, uint32 const c2) {
			if (c2 == cur || pts[c2] == pts[cur]) return false;
			if (c1 == cur) return true;
			int const o = p.orientation(pts[cur], pts[c1], pts[c2]);
			if (o != 0) return o < 0;
			return detail::is_farther_on_ray(pts[cur], pts[c1], pts[c2]);
		};

		for (uint32 group_size = 8; group_size != 0; group_size = group_size > 65535 ? m : group_size * group_size) {
			group_size = (std::min)(group_size, m);
			std::vector<std::vector<uint32>> mini;
			for (uint32 lo = 0; lo < m;) {
				uint32 const hi = lo + (std::min)(group_size, m - lo);
				auto const sub = std::ranges::subrange(pts.begin() + lo, pts.begin() + hi);
				auto ids = geometry::convex_hull_indices(sub, p);
				for (auto &k : ids) k += lo;
				mini.push_back(std::move(ids));
				lo = hi;
			}
			std::vector<uint32> ptr(mini.size(), 0);
			uint32 const start = 0;
			for (uint32 g = 0; g < mini.size(); ++g)
				for (uint32 k = 1; k < mini[g].size(); ++k)
					if (beats(start, mini[g][ptr[g]], mini[g][k])) ptr[g] = k;

			std::vector<uint32> hull;
			uint32 cur = start;
			bool is_closed = false;
			for (uint64 step = 0; step <= group_size && !is_closed; ++step) {
				hull.push_back(cur);
				uint32 best = cur;
				for (uint32 g = 0; g < mini.size(); ++g) {
					auto const &ring = mini[g];
					// tangent advances monotonically around the wrap; allow
					// a full lap as the safety budget
					for (uint32 laps = 0; laps < uint32(ring.size()); ++laps) {
						uint32 const nxt = (ptr[g] + 1) % uint32(ring.size());
						if (beats(cur, ring[ptr[g]], ring[nxt]))
							ptr[g] = nxt;
						else
							break;
					}
					if (beats(cur, best, ring[ptr[g]])) best = ring[ptr[g]];
				}
				cur = best;
				is_closed = cur == start;
			}
			if (is_closed) {
				for (auto &k : hull) k = e[k].index;
				return hull;
			}
			if (group_size == m) return hull; // unreachable safety
		}
		return {};
	}

	namespace detail {
		/**
		 * @brief The hull's vertices in lexicographic order: a strict CCW hull
		 *        from the lex-min vertex is an increasing lower chain up to
		 *        the lex-max vertex followed by a decreasing upper chain —
		 *        merge the two runs.
		 * @pre @p hull is a strict CCW hull starting at its lex-min vertex
		 * @complexity O(k)
		 */
		template<geometry::vertex_range R>
		[[nodiscard]] constexpr std::vector<geometry::point<geometry::vertex_coordinate_t<R>>> lexicographic_hull_sequence(R &&hull) {
			auto const first = std::ranges::begin(hull);
			std::size_t const size = std::ranges::size(hull);
			std::size_t const lexmax = std::size_t(std::ranges::max_element(hull, {}, [](geometry::point<geometry::vertex_coordinate_t<R>> const q) { return q; }) - first);
			std::vector<geometry::point<geometry::vertex_coordinate_t<R>>> sorted;
			sorted.reserve(size);
			std::size_t low = 0, high = size;
			while (low <= lexmax || high > lexmax + 1) {
				bool const takes_low = low <= lexmax && (high == lexmax + 1 || geometry::point<geometry::vertex_coordinate_t<R>>(first[low]) < geometry::point<geometry::vertex_coordinate_t<R>>(first[high - 1]));
				sorted.push_back(takes_low ? first[low++] : first[--high]);
			}
			return sorted;
		}
	}

	/**
	 * @brief Materializes and merges full lexicographic hull sequences, then
	 * runs a strict monotone chain. Intel Xeon Platinum 8280, GCC 15.2.0,
	 * RelWithDebInfo, assertions off, 2026-10-15, -O2 / -O2 -march=native:
	 * finished circle hulls 26.4 / 26.1 ms vs concatenation 48.2 / 47.8.
	 * @pre Inputs are in atl::geometry::merge_convex_hulls form (asserted).
	 * @complexity O(h_a + h_b), O(h_a + h_b) auxiliary memory.
	 */
	template<geometry::vertex_range RA, geometry::vertex_range RB,
			 geometry::sign_policy<geometry::vertex_coordinate_t<RA>> Policy = geometry::exact>
		requires std::same_as<geometry::vertex_coordinate_t<RA>, geometry::vertex_coordinate_t<RB>>
	[[nodiscard]] constexpr std::vector<geometry::point<geometry::vertex_coordinate_t<RA>>>
	merge_convex_hulls_by_materialized_chains(RA &&a, RB &&b, Policy const p = {}) {
		using T = geometry::vertex_coordinate_t<RA>;
		atl_assert(geometry::detail::is_lexicographic_strict_hull(a, p));
		atl_assert(geometry::detail::is_lexicographic_strict_hull(b, p));
		if (std::ranges::empty(a) || std::ranges::empty(b)) {
			if (std::ranges::empty(a)) return {std::ranges::begin(b), std::ranges::begin(b) + std::ranges::ssize(b)};
			return {std::ranges::begin(a), std::ranges::begin(a) + std::ranges::ssize(a)};
		}
		auto const a_sorted = detail::lexicographic_hull_sequence(a);
		auto const b_sorted = detail::lexicographic_hull_sequence(b);
		std::vector<geometry::detail::hull_entry<T>> merged;
		merged.reserve(a_sorted.size() + b_sorted.size());
		std::size_t i = 0, j = 0; // merge, deduplicating equal points
		while (i < a_sorted.size() || j < b_sorted.size()) {
			geometry::point<T> next;
			if (j == b_sorted.size() || (i < a_sorted.size() && a_sorted[i] < b_sorted[j]))
				next = a_sorted[i++];
			else {
				next = b_sorted[j++];
				if (i < a_sorted.size() && next == a_sorted[i]) ++i;
			}
			if (merged.empty() || merged.back().position != next)
				merged.push_back({next, uint32(merged.size())});
		}
		auto const ids = geometry::detail::chain_hull<true>(merged, p);
		std::vector<geometry::point<T>> out;
		out.reserve(ids.size());
		for (geometry::vertex_index const at : ids) out.push_back(merged[at].position);
		return out;
	}

	/**
	 * @brief Hulls both point clouds, then merges materialized lexicographic chains.
	 * Intel Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
	 * 2026-10-15, -O2 / -O2 -march=native: raw circle clouds 303 / 301 ms
	 * vs concatenation 320 / 317; finished hulls 87.4 / 87.2 vs 48.2 / 47.8.
	 * @pre Nonempty inputs; coordinate bounds as atl::geometry::convex_hull.
	 * @complexity O(n log n + m log m) time, O(n + m) memory.
	 */
	template<geometry::vertex_range RA, geometry::vertex_range RB,
			 geometry::sign_policy<geometry::vertex_coordinate_t<RA>> Policy = geometry::exact>
		requires std::same_as<geometry::vertex_coordinate_t<RA>, geometry::vertex_coordinate_t<RB>>
	[[nodiscard]] constexpr std::vector<geometry::point<geometry::vertex_coordinate_t<RA>>>
	convex_hull_union_by_materialized_chains(RA &&a_points, RB &&b_points, Policy const p = {}) {
		atl_assert(std::ranges::size(a_points) >= 1 && std::ranges::size(b_points) >= 1);
		return merge_convex_hulls_by_materialized_chains(geometry::convex_hull(a_points, p), geometry::convex_hull(b_points, p),
														 p);
	}

	/**
	 * @brief Walks copodal support events and switches outward extreme vertices,
	 * then removes duplicate and straight vertices and rotates to lex-min.
	 * Intel Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
	 * 2026-10-15, -O2 / -O2 -march=native: finished circles 56.2 / 56.6 ms
	 * vs materialized chains 26.4 / 26.1; event storage and cleanup lose.
	 * @pre Inputs are in atl::geometry::merge_convex_hulls form (asserted).
	 * @complexity O(h_a + h_b), O(h_a + h_b) auxiliary memory.
	 */
	template<geometry::vertex_range RA, geometry::vertex_range RB,
			 geometry::sign_policy<geometry::vertex_coordinate_t<RA>> Policy = geometry::exact>
		requires std::same_as<geometry::vertex_coordinate_t<RA>, geometry::vertex_coordinate_t<RB>>
	[[nodiscard]] constexpr std::vector<geometry::point<geometry::vertex_coordinate_t<RA>>>
	merge_convex_hulls_by_copodal_walk(RA &&a, RB &&b, Policy const p = {}) {
		using geometry::perp;
		using T = geometry::vertex_coordinate_t<RA>;
		using point = geometry::point<T>;
		atl_assert(geometry::detail::is_lexicographic_strict_hull(a, p));
		atl_assert(geometry::detail::is_lexicographic_strict_hull(b, p));
		if (std::ranges::size(a) < 3 || std::ranges::size(b) < 3) return merge_convex_hulls_by_materialized_chains(a, b, p);
		auto const first_a = std::ranges::begin(a);
		auto const first_b = std::ranges::begin(b);

		// one co-podal event per edge of either polygon; supports are
		// CONSTANT between consecutive events, so the winner (the support
		// farther along the RIGHT normal — the view's convention, pinned by
		// an event dump) is evaluated at BOTH ends of each interval: a
		// disagreement means the support functions crossed mid-interval,
		// and the two vertices are exactly the bridge pair. Everything
		// emitted is extreme for some direction in the closed interval, so
		// nothing interior ever enters the walk.
		struct copodal_event {
			point from, to, a_support, b_support;
		};

		std::vector<copodal_event> events;
		for (auto const event : geometry::dual_support_view(a, b, p)) {
			point const from = event.is_on_second ? first_b[event.edge_start] : first_a[event.edge_start];
			point const to = event.is_on_second ? first_b[event.edge_end] : first_a[event.edge_end];
			events.push_back({from, to, first_a[event.p_support], first_b[event.q_support]});
		}
		std::vector<point> walk;
		for (std::size_t k = 0; k < events.size(); ++k) {
			auto const &at = events[k];
			auto const &next = events[k + 1 == events.size() ? 0 : k + 1];
			// an event reports the supports valid on the interval ENDING at
			// it (the owner's support is its edge START — pre-advance), so
			// the interval (at, next) carries NEXT's supports; b wins on
			// NEGATIVE signs (right-normal extremes)
			int const s0 = p.cross_sign(at.from, at.to, next.a_support, next.b_support);
			int const s1 = p.cross_sign(next.from, next.to, next.a_support, next.b_support);
			if (s0 == 0) { // exact tie: both on the flush line, which the
						   // merged CCW boundary traverses ALONG the edge
						   // direction (right-normal extremes)
				int const ahead = -p.cross_sign(perp(at.from), perp(at.to), next.a_support,
												next.b_support);
				walk.push_back(ahead > 0 ? next.a_support : next.b_support);
				walk.push_back(ahead > 0 ? next.b_support : next.a_support);
			} else if (s0 > 0) {
				walk.push_back(next.a_support);
				if (s1 <= 0) walk.push_back(next.b_support); // bridge a -> b
			} else {
				walk.push_back(next.b_support);
				if (s1 >= 0) walk.push_back(next.a_support); // bridge b -> a
			}
		}
		// cyclic cleanup: consecutive duplicates, then straight vertices
		std::vector<point> distinct;
		for (auto const v : walk)
			if (distinct.empty() || v != distinct.back()) distinct.push_back(v);
		while (distinct.size() > 1 && distinct.front() == distinct.back()) distinct.pop_back();
		if (distinct.size() < 3) return merge_convex_hulls_by_materialized_chains(a, b, p);
		std::vector<point> out;
		std::size_t const m = distinct.size();
		point predecessor = distinct[m - 1];
		for (std::size_t i = 0; i < m; ++i) {
			point const successor = distinct[i + 1 == m ? 0 : i + 1];
			if (p.orientation(predecessor, distinct[i], successor) != 0) out.push_back(distinct[i]);
			predecessor = distinct[i];
		}
		std::ranges::rotate(out, std::ranges::min_element(out));
		return out;
	}

	/**
	 * @brief Hulls both clouds then walks their copodal support events.
	 * Intel Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
	 * 2026-10-15, -O2 / -O2 -march=native: raw circles 334 / 332 ms
	 * vs concatenation 320 / 317; rehulling finished circles 117 / 117
	 * vs concatenation 48.2 / 47.8. The support walk adds substantial cost.
	 * @pre Nonempty inputs; coordinate bounds as atl::geometry::convex_hull.
	 * @complexity O(n log n + m log m) time, O(n + m) memory.
	 */
	template<geometry::vertex_range RA, geometry::vertex_range RB,
			 geometry::sign_policy<geometry::vertex_coordinate_t<RA>> Policy = geometry::exact>
		requires std::same_as<geometry::vertex_coordinate_t<RA>, geometry::vertex_coordinate_t<RB>>
	[[nodiscard]] constexpr std::vector<geometry::point<geometry::vertex_coordinate_t<RA>>>
	convex_hull_union_by_copodal_walk(RA &&a_points, RB &&b_points, Policy const p = {}) {
		atl_assert(std::ranges::size(a_points) >= 1 && std::ranges::size(b_points) >= 1);
		return merge_convex_hulls_by_copodal_walk(geometry::convex_hull(a_points, p), geometry::convex_hull(b_points, p), p);
	}

	/**
	 * @brief Streams merged lower and upper chains without bridge dispatch.
	 * Intel Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, assertions off,
	 * 2026-11-02, -O2 / -O2 -march=native: disjoint circles, total 300000,
	 * 2.2 / 2.6 ms vs library 0.186 / 0.168. Retains the linear scan cost
	 * that binary bridges avoid on separated inputs.
	 * @pre Inputs are in atl::geometry::merge_convex_hulls form (asserted).
	 * @complexity O(h_a + h_b), O(h_a + h_b) output storage.
	 */
	template<geometry::vertex_range RA, geometry::vertex_range RB, geometry::sign_policy<geometry::vertex_coordinate_t<RA>> Policy = geometry::exact>
		requires std::same_as<geometry::vertex_coordinate_t<RA>, geometry::vertex_coordinate_t<RB>>
	[[nodiscard]] constexpr std::vector<geometry::point<geometry::vertex_coordinate_t<RA>>> merge_convex_hulls_by_streamed_chains(RA &&a, RB &&b, Policy const p = {}) {
		using T = geometry::vertex_coordinate_t<RA>;
		atl_assert(geometry::detail::is_lexicographic_strict_hull(a, p));
		atl_assert(geometry::detail::is_lexicographic_strict_hull(b, p));
		std::size_t const na = std::ranges::size(a), nb = std::ranges::size(b);
		auto const first_a = std::ranges::begin(a);
		auto const first_b = std::ranges::begin(b);
		if (na == 0 || nb == 0) return na == 0 ? std::vector<geometry::point<T>>(first_b, first_b + nb)
											   : std::vector<geometry::point<T>>(first_a, first_a + na);
		std::size_t const top_a = geometry::detail::lexicographic_top(first_a, na);
		std::size_t const top_b = geometry::detail::lexicographic_top(first_b, nb);
		std::vector<geometry::point<T>> out;
		if (na > out.max_size() || nb > out.max_size() - na) throw std::length_error("convex hull is too large");
		out.reserve(na + nb);
		geometry::detail::merge_hull_chains<true>(geometry::detail::hull_chain{first_a, 0, top_a + 1, false},
												  geometry::detail::hull_chain{first_b, 0, top_b + 1, false}, 2, out, p);
		std::size_t const lower = out.size();
		if (lower == 1) return out;
		// the upper pass restarts at the union's maximum, rewriting it in
		// place, and never pops into the lower chain
		out.pop_back();
		geometry::detail::merge_hull_chains<false>(geometry::detail::hull_chain{first_a, top_a, na, na > 1},
												   geometry::detail::hull_chain{first_b, top_b, nb, nb > 1}, lower + 1, out, p);
		out.pop_back();
		return out;
	}

}
