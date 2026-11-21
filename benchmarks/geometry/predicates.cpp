/**
 * @file
 * @brief Cost of atl::geometry::orientation under each sign policy and input
 *        shape, raced against a branchless static filter.
 *
 * Library rows time `orientation` with the `exact`, `fast` and `approx{1e-9}`
 * policies. The contender, `orientation_by_branchless_filter`, differs from
 * the exact `double` path in one thing: its static filter bounds the rounding
 * error by `|left| + |right|` whatever the signs of the two products, instead
 * of first returning on opposite signs and bounding same-sign products by
 * their sum; both fall through to the same adaptive stages.
 *
 * Each row runs 1e6 orientations of independent triples (throughput; the
 * loops can vectorize), or, in the "serialized" rows, a chain in which each
 * result picks the next triple (per-call latency, closer to a hull loop).
 * Shapes: uniform int32 within 1e9 and int64 within 2^61; uniform doubles
 * within 1e6 (generic position, the filter certifies at once); exactly
 * collinear grid triples (the exit after the exact product of the rounded
 * differences); the same triples nudged by one ulp; and exactly collinear
 * mixed-exponent triples with full mantissas, about half of which reach the
 * full exact expansion.
 *
 * Result (Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, asserts off,
 * 2026-10-15; ms per 1e6 calls, `-O2` / `-march=native`): int32 exact 2.7 /
 * 2.6, int64 exact 7.7 / 7.7; double generic position exact 16.6 / 16.5 vs
 * fast 5.8 / 5.7; collinear grid exact 24.1 / 18.1; one-ulp nudge exact
 * 24.2 / 18.0; mixed exponents exact 85.7 / 68.6; serialized exact 25.9 /
 * 24.8 vs fast 16.3 / 15.7. The branchless filter is faster in generic
 * position (6.6 / 6.1 throughput, 17.4 / 16.4 serialized) and slower one ulp
 * from degeneracy (28.1 / 22.8); whether to adopt it is open. The full table
 * is on the guide page (docs/geometry-primitives.md).
 */
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

using atl::geometry::approx;
using atl::geometry::exact;
using atl::geometry::fast;
using atl::geometry::orientation;
using atl::geometry::point;

namespace {
	/**
	 * @brief The exact `double` orientation with a branchless static filter:
	 * `|left| + |right|` bounds the rounding error whatever the signs (when
	 * the signs differ, `|det|` equals that sum and certifies itself), then
	 * the library's adaptive stages.
	 */
	int orientation_by_branchless_filter(point<double> const o, point<double> const a, point<double> const b) {
		double const left = (a.x - o.x) * (b.y - o.y);
		double const right = (a.y - o.y) * (b.x - o.x);
		double const det = left - right;
		double const magnitude_sum = std::fabs(left) + std::fabs(right);
		if (std::fabs(det) >= atl::geometry::detail::orient_bound_a * magnitude_sum)
			return atl::geometry::detail::sign_of(det);
		return atl::geometry::detail::adaptive_cross_tail(o, a, o, b, magnitude_sum);
	}

	/** @brief Sum of the orientations of the independent triples `pts[3i..3i+2]`. */
	template<atl::geometry::coordinate T, typename Orient>
		requires std::is_invocable_r_v<int, Orient const &, point<T>, point<T>, point<T>>
	atl::int64 independent_triples(std::vector<point<T>> const &pts, Orient const &orient) {
		atl::int64 acc = 0;
		for (std::size_t i = 0; i + 2 < pts.size(); i += 3) acc += orient(pts[i], pts[i + 1], pts[i + 2]);
		return acc;
	}

	/** @brief One orientation per triple, each result choosing the next triple. */
	template<atl::geometry::coordinate T, typename Orient>
		requires std::is_invocable_r_v<int, Orient const &, point<T>, point<T>, point<T>>
	atl::int64 serialized_triples(std::vector<point<T>> const &pts, Orient const &orient) {
		std::size_t const triples = pts.size() / 3;
		atl::int64 acc = 0;
		std::size_t i = 0;
		for (std::size_t step = 0; step < triples; ++step) {
			int const r = orient(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
			acc += r;
			i = (i + 1 + std::size_t(r > 0)) % triples;
		}
		return acc;
	}

	/** @brief `orientation` under @p policy, as a contender callable. */
	template<typename Policy>
	auto library_orientation(Policy const policy) {
		return [policy]<atl::geometry::coordinate T>(point<T> const o, point<T> const a, point<T> const b)
			requires atl::geometry::sign_policy<Policy, T>
		{ return orientation(o, a, b, policy); };
	}
}

TEST_CASE("orientation benchmarks", "[geometry]") {
	constexpr std::size_t n = 1'000'000;
	std::mt19937_64 rng(1);

	std::vector<point<atl::int32>> p32(3 * n);
	for (auto &p : p32) p = {atl::int32(rng() % 2'000'000'001) - 1'000'000'000,
							 atl::int32(rng() % 2'000'000'001) - 1'000'000'000};

	std::vector<point<atl::int64>> p64(3 * n);
	for (auto &p : p64) p = {atl::int64(rng() % (atl::uint64(1) << 62)) - (atl::int64(1) << 61),
							 atl::int64(rng() % (atl::uint64(1) << 62)) - (atl::int64(1) << 61)};

	std::uniform_real_distribution<double> real(-1e6, 1e6);
	std::vector<point<double>> preal(3 * n);
	for (auto &p : preal) p = {real(rng), real(rng)};

	// exactly collinear grid triples: the common geometric degeneracy
	std::vector<point<double>> pgrid(3 * n);
	for (std::size_t i = 0; i < n; ++i) {
		double const ox = double(rng() % (1 << 20)), oy = double(rng() % (1 << 20));
		double const dx = double(1 + rng() % 1024), dy = double(1 + rng() % 1024);
		double const s = double(1 + rng() % (1 << 20)), t = double(1 + rng() % (1 << 20));
		pgrid[3 * i] = {ox, oy};
		pgrid[3 * i + 1] = {ox + s * dx, oy + s * dy};
		pgrid[3 * i + 2] = {ox + t * dx, oy + t * dy};
	}

	// the same triples nudged by one ulp
	std::vector<point<double>> pnear = pgrid;
	for (std::size_t i = 0; i < n; ++i) {
		double &c = rng() % 2 ? pnear[3 * i + 2].y : pnear[3 * i + 2].x;
		c = std::nextafter(c, rng() % 2 ? std::numeric_limits<double>::infinity()
										: -std::numeric_limits<double>::infinity());
	}

	// exactly collinear with mixed exponents and full mantissas: inexact
	// differences push past every estimate, about half of these into the full
	// exact expansion
	std::vector<point<double>> pdeep(3 * n);
	for (std::size_t i = 0; i < n; ++i) {
		auto const coord = [&] {
			return std::ldexp(1.0 + double(rng() & ((atl::uint64(1) << 52) - 1)) * 0x1p-52,
							  int(rng() % 40));
		};
		double const slope = std::ldexp(1.0, -int(rng() % 20));
		for (int k = 0; k < 3; ++k) {
			double const x = coord();
			pdeep[3 * i + k] = {x, x * slope};
		}
		if (rng() % 2)
			pdeep[3 * i + 2].y = std::nextafter(pdeep[3 * i + 2].y,
												rng() % 2 ? std::numeric_limits<double>::infinity()
														  : -std::numeric_limits<double>::infinity());
	}

	auto const by_exact = library_orientation(exact{});
	auto const by_fast = library_orientation(fast{});
	auto const by_approx = library_orientation(approx{1e-9});
	auto const by_branchless_filter = [](point<double> const o, point<double> const a, point<double> const b) {
		return orientation_by_branchless_filter(o, a, b);
	};

	// the contender must agree with the exact sign everywhere
	for (auto const *pts : {&preal, &pgrid, &pnear, &pdeep})
		for (std::size_t i = 0; i < n; ++i)
			REQUIRE(by_branchless_filter((*pts)[3 * i], (*pts)[3 * i + 1], (*pts)[3 * i + 2])
					== by_exact((*pts)[3 * i], (*pts)[3 * i + 1], (*pts)[3 * i + 2]));

	BENCHMARK("int32 exact") { return independent_triples(p32, by_exact); };
	BENCHMARK("int64 exact (int128 products)") { return independent_triples(p64, by_exact); };
	BENCHMARK("double exact, generic position") { return independent_triples(preal, by_exact); };
	BENCHMARK("double fast, generic position") { return independent_triples(preal, by_fast); };
	BENCHMARK("double approx{1e-9}, generic position") { return independent_triples(preal, by_approx); };
	BENCHMARK("double exact, collinear grid") { return independent_triples(pgrid, by_exact); };
	BENCHMARK("double fast, collinear grid") { return independent_triples(pgrid, by_fast); };
	BENCHMARK("double exact, ulp near-degenerate") { return independent_triples(pnear, by_exact); };
	BENCHMARK("double fast, ulp near-degenerate") { return independent_triples(pnear, by_fast); };
	BENCHMARK("double exact, mixed-exponent deep fallback") { return independent_triples(pdeep, by_exact); };
	BENCHMARK("double fast, mixed-exponent deep fallback") { return independent_triples(pdeep, by_fast); };
	BENCHMARK("double exact, generic serialized") { return serialized_triples(preal, by_exact); };
	BENCHMARK("double by branchless filter, generic position") { return independent_triples(preal, by_branchless_filter); };
	BENCHMARK("double by branchless filter, collinear grid") { return independent_triples(pgrid, by_branchless_filter); };
	BENCHMARK("double by branchless filter, ulp near-degenerate") { return independent_triples(pnear, by_branchless_filter); };
	BENCHMARK("double by branchless filter, generic serialized") { return serialized_triples(preal, by_branchless_filter); };
	BENCHMARK("double fast, generic serialized") { return serialized_triples(preal, by_fast); };
	BENCHMARK("int32 exact, serialized") { return serialized_triples(p32, by_exact); };
}
