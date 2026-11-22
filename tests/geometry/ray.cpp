#include <atl/geometry/circle.hpp>
#include <atl/geometry/line.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/ray.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>
#include "../expect_abort.hpp"

using atl::int128;
using atl::geometry::circle;
using atl::geometry::contains;
using atl::geometry::intersection;
using atl::geometry::intersection_kind;
using atl::geometry::intersects;
using atl::geometry::line;
using atl::geometry::point;
using atl::geometry::ray;
using atl::geometry::segment;

namespace {
	using pt = point<std::int64_t>;
	using ry = ray<std::int64_t>;

	int128 cross_of(pt const o, pt const a, pt const b) {
		return int128(a.x - o.x) * (b.y - o.y) - int128(a.y - o.y) * (b.x - o.x);
	}

	int128 dot_of(pt const o, pt const a, pt const b) {
		return int128(a.x - o.x) * (b.x - o.x) + int128(a.y - o.y) * (b.y - o.y);
	}

	int sign_of(int128 const v) { return int(v > 0) - int(v < 0); }

	// independent rational brute: exists t >= 0 with O + tD on the object?
	bool brute_contains(ry const r, pt const q) {
		return cross_of(r.origin, r.through, q) == 0 && dot_of(r.origin, r.through, q) >= 0;
	}

	// exists t >= 0, u in [0, 1]: O + tD = A + u(B - A) — signs only, exact
	bool brute_ray_segment(ry const r, pt const a, pt const b) {
		pt const o = r.origin, h = r.through;
		int128 const den = int128(h.x - o.x) * (b.y - a.y) - int128(h.y - o.y) * (b.x - a.x);
		if (den == 0) { // parallel: either collinear 1D overlap or nothing
			if (cross_of(o, h, a) != 0) return false;
			if (a == b) return brute_contains(r, a);
			return brute_contains(r, a) || brute_contains(r, b)
				|| (cross_of(a, b, o) == 0 && dot_of(o, a, b) <= 0); // origin inside [a, b]
		}
		int128 const t_num = int128(a.x - o.x) * (b.y - a.y) - int128(a.y - o.y) * (b.x - a.x);
		int128 const u_num = int128(a.x - o.x) * (h.y - o.y) - int128(a.y - o.y) * (h.x - o.x);
		int const ds = sign_of(den);
		// t >= 0 and 0 <= u <= 1, comparing rationals by the denominator sign
		return sign_of(t_num) * ds >= 0 && sign_of(u_num) * ds >= 0
			&& sign_of(u_num - den) * ds <= 0;
	}

	bool brute_ray_line(ry const r, pt const la, pt const lb) {
		pt const o = r.origin, h = r.through;
		int128 const den = int128(h.x - o.x) * (lb.y - la.y) - int128(h.y - o.y) * (lb.x - la.x);
		if (den == 0) return cross_of(la, lb, o) == 0; // parallel: on it or miss
		int128 const t_num =
			int128(la.x - o.x) * (lb.y - la.y) - int128(la.y - o.y) * (lb.x - la.x);
		return sign_of(t_num) * sign_of(den) >= 0;
	}

	bool brute_ray_ray(ry const r1, ry const r2) {
		pt const o = r1.origin, h = r1.through, q = r2.origin, g = r2.through;
		int128 const den = int128(h.x - o.x) * (g.y - q.y) - int128(h.y - o.y) * (g.x - q.x);
		if (den == 0) {
			if (cross_of(o, h, q) != 0) return false;
			return brute_contains(r1, q) || brute_contains(r2, o);
		}
		int128 const t_num = int128(q.x - o.x) * (g.y - q.y) - int128(q.y - o.y) * (g.x - q.x);
		int128 const u_num = int128(q.x - o.x) * (h.y - o.y) - int128(q.y - o.y) * (h.x - o.x);
		int const ds = sign_of(den);
		return sign_of(t_num) * ds >= 0 && sign_of(u_num) * ds >= 0;
	}

	std::vector<pt> small_grid(int const radius) {
		std::vector<pt> g;
		for (int x = -radius; x <= radius; ++x)
			for (int y = -radius; y <= radius; ++y) g.push_back({x, y});
		return g;
	}

	std::vector<ry> grid_rays(int const radius) {
		std::vector<ry> rays;
		for (auto const o : small_grid(radius))
			for (auto const h : small_grid(radius))
				if (o != h) rays.push_back({o, h});
		return rays;
	}

	bool on_ray_approx(ry const r, point<double> const q) {
		point<double> const o{double(r.origin.x), double(r.origin.y)};
		point<double> const h{double(r.through.x), double(r.through.y)};
		double const cx = (h.x - o.x) * (q.y - o.y) - (h.y - o.y) * (q.x - o.x);
		double const dt = (h.x - o.x) * (q.x - o.x) + (h.y - o.y) * (q.y - o.y);
		return std::abs(cx) <= 1e-6 && dt >= -1e-6;
	}
}

TEST_CASE("ray: contains and intersects vs rational brutes, exhaustive", "[geometry]") {
	auto const rays = grid_rays(2);
	auto const grid = small_grid(2);
	std::int64_t contains_bad = 0, seg_bad = 0, line_bad = 0, ray_bad = 0;
	for (auto const r : rays) {
		for (auto const q : grid)
			if (contains(r, q) != brute_contains(r, q)) ++contains_bad;
		for (auto const a : grid)
			for (auto const b : grid) {
				if (a != b) {
					if (intersects(r, line<std::int64_t>{a, b}) != brute_ray_line(r, a, b))
						++line_bad;
					if (intersects(r, ry{a, b}) != brute_ray_ray(r, ry{a, b})) ++ray_bad;
				}
				if (intersects(r, segment<std::int64_t>{a, b}) != brute_ray_segment(r, a, b))
					++seg_bad;
			}
	}
	REQUIRE(contains_bad == 0);
	REQUIRE(seg_bad == 0);
	REQUIRE(line_bad == 0);
	REQUIRE(ray_bad == 0);
}

TEST_CASE("ray: intersects vs brutes, random larger coordinates", "[geometry]") {
	auto seed = GENERATE(take(12, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	auto const coord = [&] { return std::int64_t(rng() % 2'000'001) - 1'000'000; };
	auto const some_point = [&] { return pt{coord(), coord()}; };
	for (int it = 0; it < 400; ++it) {
		pt const o = some_point();
		pt h = some_point();
		if (h == o) h.x += 1;
		ry const r{o, h};
		pt const a = some_point();
		pt b = rng() % 4 ? some_point() : pt{a.x + coord() % 3, a.y + coord() % 3};
		REQUIRE(intersects(r, segment<std::int64_t>{a, b}) == brute_ray_segment(r, a, b));
		if (b == a) b.x += 1;
		REQUIRE(intersects(r, line<std::int64_t>{a, b}) == brute_ray_line(r, a, b));
		REQUIRE(intersects(r, ry{a, b}) == brute_ray_ray(r, ry{a, b}));
	}
}

TEST_CASE("ray: intersection certificates, exhaustive small grid", "[geometry]") {
	auto const rays = grid_rays(2);
	auto const grid = small_grid(2);
	std::int64_t bad = 0;
	auto const check_common_point = [&](ry const r, segment<std::int64_t> const other, point<double> const at) {
		if (!on_ray_approx(r, at)) ++bad;
		using O = decltype(other);
		if constexpr (std::is_same_v<O, segment<std::int64_t> const>) {
			// residual against the segment's line, plus the box
			point<double> const a{double(other.a.x), double(other.a.y)};
			point<double> const b{double(other.b.x), double(other.b.y)};
			double const cx = (b.x - a.x) * (at.y - a.y) - (b.y - a.y) * (at.x - a.x);
			if (std::abs(cx) > 1e-6) ++bad;
			if (at.x < std::min(a.x, b.x) - 1e-6 || at.x > std::max(a.x, b.x) + 1e-6) ++bad;
			if (at.y < std::min(a.y, b.y) - 1e-6 || at.y > std::max(a.y, b.y) + 1e-6) ++bad;
		}
	};
	for (auto const r : rays) {
		for (auto const a : grid)
			for (auto const b : grid) {
				{
					segment<std::int64_t> const s{a, b};
					auto const got = intersection(r, s);
					if ((got.kind != intersection_kind::empty) != intersects(r, s)) ++bad;
					if (got.kind == intersection_kind::at_point) check_common_point(r, s, got.at);
					if (got.kind == intersection_kind::along_segment) {
						// exact endpoints, on both, lexicographic, non-degenerate
						if (!(got.along.a < got.along.b)) ++bad;
						for (auto const e : {got.along.a, got.along.b})
							if (!contains(r, e) || !contains(s, e)) ++bad;
					}
				}
				if (a == b) continue;

				{
					line<std::int64_t> const l{a, b};
					auto const got = intersection(r, l);
					if ((got.kind != intersection_kind::empty) != intersects(r, l)) ++bad;
					if (got.kind == intersection_kind::at_point) {
						if (!on_ray_approx(r, got.at)) ++bad;
						point<double> const la{double(a.x), double(a.y)};
						point<double> const lb{double(b.x), double(b.y)};
						double const cx = (lb.x - la.x) * (got.at.y - la.y)
										- (lb.y - la.y) * (got.at.x - la.x);
						if (std::abs(cx) > 1e-6) ++bad;
					}
					if (got.kind == intersection_kind::along_ray) {
						if (got.beam != r) ++bad; // the whole ray, exactly
						if (!contains(l, r.origin) || !contains(l, r.through)) ++bad;
					}
				}

				{
					ry const r2{a, b};
					auto const got = intersection(r, r2);
					if ((got.kind != intersection_kind::empty) != intersects(r, r2)) ++bad;
					if (got.kind == intersection_kind::at_point) {
						if (!on_ray_approx(r, got.at) || !on_ray_approx(r2, got.at)) ++bad;
					}
					if (got.kind == intersection_kind::along_segment) {
						// collinear facing rays: exactly the origins' gap
						if (!(got.along.a < got.along.b)) ++bad;
						std::array<pt, 2> const origins =
							r.origin < r2.origin ? std::array{r.origin, r2.origin}
												 : std::array{r2.origin, r.origin};
						if (got.along.a != origins[0] || got.along.b != origins[1]) ++bad;
						for (auto const e : {got.along.a, got.along.b})
							if (!contains(r, e) || !contains(r2, e)) ++bad;
					}
					if (got.kind == intersection_kind::along_ray) {
						// the later origin's ray: contained in both
						if (got.beam != r && got.beam != r2) ++bad;
						if (!contains(r, got.beam.origin) || !contains(r2, got.beam.origin))
							++bad;
					}
				}
			}
	}
	REQUIRE(bad == 0);
}

TEST_CASE("ray: circle intersection count vs quadratic brute, exhaustive", "[geometry]") {
	// |O + tD - C|^2 = r^2: count roots with t >= 0, exactly in int128
	auto const brute_count = [](circle<std::int64_t> const c, ry const r) {
		pt const o = r.origin;
		pt const d{r.through.x - r.origin.x, r.through.y - r.origin.y};
		pt const w{o.x - c.center.x, o.y - c.center.y};
		int128 const qa = int128(d.x) * d.x + int128(d.y) * d.y;
		int128 const qb = 2 * (int128(d.x) * w.x + int128(d.y) * w.y);
		int128 const qc = int128(w.x) * w.x + int128(w.y) * w.y - int128(c.radius) * c.radius;
		int128 const disc = qb * qb - 4 * qa * qc;
		if (disc < 0) return std::size_t(0);
		std::size_t count = 0;
		// root signs of (-qb ± sqrt(disc)) / 2qa, qa > 0: compare qb^2 vs disc
		for (int const s : {-1, +1}) {
			int128 const numerator_sign_probe = -qb; // + s*sqrt(disc)
			if (s < 0 && disc == 0) continue;		 // double root: count once
			// numerator >= 0 ⟺ -qb >= -s*sqrt(disc): case on signs
			bool nonneg = false;
			if (numerator_sign_probe >= 0)
				nonneg = s > 0 || numerator_sign_probe * numerator_sign_probe >= disc;
			else
				nonneg = s > 0 && numerator_sign_probe * numerator_sign_probe <= disc;
			count += std::size_t(nonneg);
		}
		return count;
	};
	std::int64_t bad = 0;
	for (auto const r : grid_rays(2))
		for (auto const center : small_grid(2))
			for (std::int64_t radius = 0; radius <= 3; ++radius) {
				circle<std::int64_t> const c{center, radius};
				auto const got = intersection(c, r);
				if (got.size() != brute_count(c, r)) ++bad;
				double last = -1e300;
				for (auto const q : got) {
					if (!on_ray_approx(r, q)) ++bad;
					double const gap =
						std::hypot(q.x - double(center.x), q.y - double(center.y))
						- double(radius);
					if (std::abs(gap) > 1e-6) ++bad;
					// ordered along the ray
					double const t = (q.x - double(r.origin.x))
									   * double(r.through.x - r.origin.x)
								   + (q.y - double(r.origin.y))
										 * double(r.through.y - r.origin.y);
					if (t < last - 1e-9) ++bad;
					last = t;
				}
			}
	REQUIRE(bad == 0);
}

TEST_CASE("ray: distance vs brute, exhaustive small grid", "[geometry]") {
	std::int64_t bad = 0;
	for (auto const r : grid_rays(2))
		for (auto const q : small_grid(3)) {
			double const got = distance(r, q);
			double expected = 0;
			if (brute_contains(r, q))
				expected = 0;
			else if (dot_of(r.origin, r.through, q) < 0)
				expected = distance(r.origin, q);
			else
				expected = distance(line<std::int64_t>{r.origin, r.through}, q);
			if (brute_contains(r, q) && got != 0) ++bad; // exact zero on the ray
			if (std::abs(got - expected) > 1e-9) ++bad;
		}
	REQUIRE(bad == 0);
}

TEST_CASE("ray: exactness at huge coordinates", "[geometry]") {
	std::int64_t const big = (std::int64_t(1) << 62) - 2;
	ry const r{{0, 0}, {2, 1}};
	REQUIRE(contains(r, pt{big, big / 2}));
	REQUIRE_FALSE(contains(r, pt{big, big / 2 + 1})); // one off the exact line
	REQUIRE_FALSE(contains(r, pt{-big, -big / 2}));	  // exactly collinear, behind
	// a ray from afar hitting / just missing a tiny distant segment
	ry const distant{{-big, -big / 2 + 1}, {-big + 2, -big / 2 + 2}}; // slope 1/2 offset
	REQUIRE(intersects(ry{{0, 0}, {2, 1}}, segment<std::int64_t>{{big, big / 2},
																 {big, big / 2 + 5}}));
	REQUIRE_FALSE(intersects(ry{{0, 0}, {2, 1}},
							 segment<std::int64_t>{{big, big / 2 + 1}, {big, big / 2 + 5}}));
	(void)distant;
}

TEST_CASE("ray/line/segment: pairwise distances", "[geometry]") {
	using sg = segment<std::int64_t>;
	using ln = line<std::int64_t>;
	using atl::geometry::distance;
	// exact spots per pair
	REQUIRE(std::abs(distance(ln{{0, 0}, {4, 0}}, ln{{0, 3}, {4, 3}}) - 3.0) < 1e-12);	  // parallel lines
	REQUIRE(distance(ln{{0, 0}, {4, 0}}, ln{{2, -1}, {2, 7}}) == 0.0);					  // crossing lines
	REQUIRE(std::abs(distance(ln{{0, 0}, {4, 0}}, sg{{1, 2}, {3, 5}}) - 2.0) < 1e-12);	  // line vs segment
	REQUIRE(distance(ry{{0, 0}, {1, 0}}, ln{{5, -1}, {5, 1}}) == 0.0);					  // ray hits line
	REQUIRE(std::abs(distance(ry{{0, 0}, {-1, 0}}, ln{{5, -1}, {5, 1}}) - 5.0) < 1e-12);  // heading away
	REQUIRE(std::abs(distance(ry{{0, 0}, {1, 0}}, sg{{3, 4}, {9, 4}}) - 4.0) < 1e-12);	  // above the ray
	REQUIRE(std::abs(distance(ry{{0, 0}, {1, 0}}, sg{{-3, 4}, {-3, -4}}) - 3.0) < 1e-12); // behind: origin
	REQUIRE(std::abs(distance(ry{{0, 0}, {1, 0}}, ry{{4, 3}, {9, 3}}) - 3.0) < 1e-12);	  // parallel rays
	REQUIRE(std::abs(distance(ry{{0, 0}, {1, 0}}, ry{{4, 3}, {4, 9}}) - 3.0) < 1e-12);	  // away: origins
	REQUIRE(distance(ry{{0, 0}, {1, 1}}, ry{{4, 0}, {0, 4}}) == 0.0);					  // crossing rays
	// exhaustive thinned grid: certificates for every pair kind
	auto const grid = small_grid(2);
	std::int64_t bad = 0;
	auto const sample = []<typename First, typename Second>(First const &at_p, Second const &at_q)
		requires std::is_invocable_r_v<point<double>, First const &, double> && std::is_invocable_r_v<point<double>, Second const &, double>
	{
		double best = 1e300;
		for (int i = 0; i <= 12; ++i)
			for (int j = 0; j <= 12; ++j) {
				auto const p = at_p(i / 4.0), q = at_q(j / 4.0); // t in [0, 3]
				best = std::min(best, std::hypot(p.x - q.x, p.y - q.y));
			}
		return best;
	};
	auto const along = [](pt const o, pt const h) {
		return [=](double const t) {
			return point<double>{o.x + (h.x - o.x) * t, o.y + (h.y - o.y) * t};
		};
	};
	using atl::geometry::distance;
	for (auto const a : grid)
		for (auto const b : grid) {
			if (a == b) continue;
			ry const r{a, b};
			for (auto const c : grid)
				for (auto const d : grid) {
					if ((a.x + b.x + c.x + d.x + a.y) % 2 != 0) continue; // thin
					// ray x segment (sampling covers u in [0,1] via clamp below)
					{
						sg const s{c, d};
						double const got = distance(r, s);
						if ((got == 0.0) != intersects(r, s)) ++bad;
						if (std::abs(got - distance(s, r)) > 1e-12) ++bad;
						double const cap = std::min(std::min(distance(r, c), distance(r, d)),
													distance(s, r.origin));
						if (!intersects(r, s) && std::abs(got - cap) > 1e-12) ++bad;
						auto const seg_at = [&](double t) {
							t = std::min(t, 1.0);
							return point<double>{c.x + (d.x - c.x) * t, c.y + (d.y - c.y) * t};
						};
						if (got > sample(along(a, b), seg_at) + 1e-9) ++bad;
					}
					if (c == d) continue;
					// line x line, line x segment
					{
						ln const l{a, b}, m{c, d};
						double const got = distance(l, m);
						if ((got == 0.0) != intersects(l, m)) ++bad;
						if (got > sample(along(a, b), along(c, d)) + 1e-9) ++bad;
						sg const s{c, d};
						double const gls = distance(l, s);
						if ((gls == 0.0) != intersects(l, s)) ++bad;
						if (std::abs(gls - distance(s, l)) > 1e-12) ++bad;
						if (!intersects(l, s)
							&& std::abs(gls - std::min(distance(l, s.a), distance(l, s.b))) > 1e-12)
							++bad;
					}
					// ray x line, ray x ray
					{
						ln const m{c, d};
						double const got = distance(r, m);
						if ((got == 0.0) != intersects(r, m)) ++bad;
						if (std::abs(got - distance(m, r)) > 1e-12) ++bad;
						if (!intersects(r, m) && std::abs(got - distance(m, r.origin)) > 1e-12) ++bad;
						ry const r2{c, d};
						double const grr = distance(r, r2);
						if ((grr == 0.0) != intersects(r, r2)) ++bad;
						if (std::abs(grr - distance(r2, r)) > 1e-12) ++bad;
						if (!intersects(r, r2)
							&& std::abs(grr - std::min(distance(r, r2.origin), distance(r2, r.origin))) > 1e-12)
							++bad;
						if (grr > sample(along(a, b), along(c, d)) + 1e-9) ++bad;
					}
				}
		}
	REQUIRE(bad == 0);
}

TEST_CASE("ray: intersection asserts a non-degenerate ray", "[geometry]") {
	ray<int> const degenerate{{1, 1}, {1, 1}}, good{{0, 0}, {1, 0}};
	CHECK(atl_test::aborts([&] { (void)intersection(degenerate, line<int>{{0, 0}, {2, 2}}); }));
	CHECK(atl_test::aborts([&] { (void)intersection(degenerate, segment<int>{{0, 0}, {2, 2}}); }));
	CHECK(atl_test::aborts([&] { (void)intersection(good, degenerate); }));
	CHECK(atl_test::aborts([&] { (void)intersection(degenerate, good); }));
}

TEST_CASE("ray floating input and constructed precision tiers", "[geometry]") {
	for (double const scale : {0x1p-200, 0x1p200}) {
		ray<double> const r{{0, 0}, {scale, scale}};
		REQUIRE(contains(r, point{scale, scale}));
		REQUIRE_FALSE(contains(r, point{-scale, -scale}));
		REQUIRE(intersects(r, segment<double>{{0, scale}, {scale, 0}}));
	}
	ray<double> const r{{0, 0}, {3, 1}};
	auto const hit = intersection(r, line<double>{{1, -1}, {1, 2}});
	REQUIRE(hit.kind == intersection_kind::at_point);
	REQUIRE(atl::geometry::approx{1e-12}.equal(hit.at, point{1.0, 1.0 / 3}));
}
