#include <atl/geometry/line.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/geometry/segment_intersections.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using atl::float64;
using atl::geometry::intersection_kind;
using atl::geometry::point;
using atl::geometry::segment;
using atl::geometry::simplicity_of;

namespace {
	using pt = point<std::int64_t>;

	// Quadratic oracle straight from the definition: n >= 3, no zero-length
	// edge, adjacent edges meet EXACTLY at their shared vertex, non-adjacent
	// edges not at all.
	bool brute_simple(std::vector<pt> const &poly) {
		std::size_t const n = poly.size();
		if (n < 3) return false;
		auto const orient = [](pt a, pt b, pt c) {
			atl::int128 const value = (atl::int128(b.x) - a.x) * (atl::int128(c.y) - a.y)
									- (atl::int128(b.y) - a.y) * (atl::int128(c.x) - a.x);
			return int(value > 0) - int(value < 0);
		};
		auto const on = [&](pt a, pt b, pt q) {
			return orient(a, b, q) == 0 && (std::min)(a.x, b.x) <= q.x && q.x <= (std::max)(a.x, b.x)
				&& (std::min)(a.y, b.y) <= q.y && q.y <= (std::max)(a.y, b.y);
		};
		for (std::size_t i = 0; i < n; ++i) {
			pt const a = poly[i], b = poly[(i + 1) % n];
			if (a == b) return false;
			for (std::size_t j = i + 1; j < n; ++j) {
				pt const c = poly[j], d = poly[(j + 1) % n];
				if (j == i + 1 || (i == 0 && j == n - 1)) {
					pt const shared = j == i + 1 ? b : a;
					pt const u = shared == a ? b : a, v = shared == c ? d : c;
					if (u == v || on(shared, u, v) || on(shared, v, u)) return false;
				} else if ((orient(a, b, c) * orient(a, b, d) < 0 && orient(c, d, a) * orient(c, d, b) < 0)
						   || on(a, b, c) || on(a, b, d) || on(c, d, a) || on(c, d, b))
					return false;
			}
		}
		return true;
	}

	std::vector<pt> star_polygon(std::mt19937_64 &rng) {
		pt const center{std::int64_t(rng() % 21) - 10, std::int64_t(rng() % 21) - 10};
		std::vector<pt> raw;
		for (int k = 0; k < 40; ++k) {
			pt const p{center.x + std::int64_t(rng() % 31) - 15,
					   center.y + std::int64_t(rng() % 31) - 15};
			if (p.x != center.x || p.y != center.y) raw.push_back(p);
		}

		struct keyed_point {
			pt value;
			double angle;
		};

		std::vector<keyed_point> keyed;
		for (auto const value : raw)
			keyed.push_back({value, std::atan2(double(value.y - center.y), double(value.x - center.x))});
		std::ranges::sort(keyed, [&](keyed_point const &a, keyed_point const &b) {
			if (a.angle != b.angle) return a.angle < b.angle;
			return std::abs(a.value.x - center.x) + std::abs(a.value.y - center.y)
				 < std::abs(b.value.x - center.x) + std::abs(b.value.y - center.y);
		});
		for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = keyed[i].value;
		std::vector<pt> poly;
		for (auto const p : raw) {
			if (!poly.empty()) {
				pt const &q = poly.back();
				std::int64_t const cx =
					(q.x - center.x) * (p.y - center.y) - (q.y - center.y) * (p.x - center.x);
				std::int64_t const dot =
					(q.x - center.x) * (p.x - center.x) + (q.y - center.y) * (p.y - center.y);
				if (cx == 0 && dot > 0) { // same direction: keep the farther
					poly.back() = p;
					continue;
				}
			}
			poly.push_back(p);
		}
		return poly;
	}
}

TEST_CASE("simplicity_of: handcrafted zoo", "[geometry]") {
	// simple: square either winding, triangle, straight (collinear) vertex
	REQUIRE(simplicity_of(std::vector<pt>{{0, 0}, {4, 0}, {4, 4}, {0, 4}}).is_simple);
	REQUIRE(simplicity_of(std::vector<pt>{{0, 4}, {4, 4}, {4, 0}, {0, 0}}).is_simple);
	REQUIRE(simplicity_of(std::vector<pt>{{0, 0}, {4, 0}, {2, 3}}).is_simple);
	REQUIRE(simplicity_of(std::vector<pt>{{0, 0}, {2, 0}, {4, 0}, {4, 4}}).is_simple);

	// too few vertices
	REQUIRE_FALSE(simplicity_of(std::vector<pt>{}).is_simple);
	REQUIRE_FALSE(simplicity_of(std::vector<pt>{{1, 1}}).is_simple);
	REQUIRE_FALSE(simplicity_of(std::vector<pt>{{0, 0}, {4, 4}}).is_simple);

	// bowtie: proper crossing, witness at the crossing point
	auto const bow = simplicity_of(std::vector<pt>{{0, 0}, {4, 4}, {4, 0}, {0, 4}});
	REQUIRE_FALSE(bow.is_simple);
	REQUIRE(bow.conflict == point{2.0, 2.0});

	// touch: a vertex lands on a non-adjacent edge's interior
	auto const touch =
		simplicity_of(std::vector<pt>{{0, 0}, {8, 0}, {8, 4}, {4, 0}, {0, 4}});
	REQUIRE_FALSE(touch.is_simple);
	REQUIRE(touch.conflict == point{4.0, 0.0});

	// pinch: one vertex used twice (figure eight)
	auto const pinch = simplicity_of(
		std::vector<pt>{{0, 0}, {2, 2}, {4, 0}, {4, 4}, {2, 2}, {0, 4}});
	REQUIRE_FALSE(pinch.is_simple);
	REQUIRE(pinch.conflict == point{2.0, 2.0});

	// zero-length edge (repeated consecutive vertex)
	REQUIRE_FALSE(simplicity_of(std::vector<pt>{{0, 0}, {4, 0}, {4, 0}, {2, 3}}).is_simple);

	// partial backtrack spike: boundary folds onto the adjacent edge
	REQUIRE_FALSE(simplicity_of(std::vector<pt>{{0, 0}, {6, 0}, {3, 0}, {3, 3}}).is_simple);

	// full doubled-back edge pair [a,b], [b,a]
	REQUIRE_FALSE(simplicity_of(std::vector<pt>{{0, 0}, {4, 0}, {0, 0}, {0, 4}}).is_simple);

	// entirely collinear "triangle": the closing edge covers the others
	REQUIRE_FALSE(simplicity_of(std::vector<pt>{{0, 0}, {4, 0}, {8, 0}}).is_simple);

	// two non-adjacent edges share a full segment (overlap, no crossing)
	REQUIRE_FALSE(simplicity_of(
					  std::vector<pt>{{0, 0}, {4, 0}, {4, 2}, {0, 2}, {0, 0}, {-4, 0}, {-4, -2}, {0, -2}})
					  .is_simple);
}

TEST_CASE("simplicity_of: matches the quadratic brute on dense small grids",
		  "[geometry]") {
	auto seed = GENERATE(take(10, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 4000; ++it) {
		std::size_t const n = std::size_t(3 + rng() % 6);
		std::int64_t const span = 3 + std::int64_t(rng() % 6);
		std::vector<pt> poly(n);
		for (auto &v : poly)
			v = {std::int64_t(rng() % std::uint64_t(span)),
				 std::int64_t(rng() % std::uint64_t(span))};
		auto const got = simplicity_of(poly);
		REQUIRE(got.is_simple == brute_simple(poly));
	}
}

TEST_CASE("simplicity_of: star polygons are simple, doctored ones checked "
		  "against the brute",
		  "[geometry]") {
	auto seed = GENERATE(take(10, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 300; ++it) {
		auto poly = star_polygon(rng);
		if (poly.size() < 3) continue;
		REQUIRE(simplicity_of(poly).is_simple);
		REQUIRE(brute_simple(poly));
		// swapping two vertices usually breaks it — whatever happens, the
		// sweep must agree with the brute
		std::size_t const a = std::size_t(rng() % poly.size());
		std::size_t const b = std::size_t(rng() % poly.size());
		std::swap(poly[a], poly[b]);
		REQUIRE(simplicity_of(poly).is_simple == brute_simple(poly));
	}
}

TEST_CASE("simplicity_of: coordinates near the sweep bound", "[geometry]") {
	std::int64_t const big = (std::int64_t(1) << 32) - 1;
	REQUIRE(simplicity_of(
				std::vector<pt>{{-big, -big}, {big, -big}, {big, big}, {-big, big}})
				.is_simple);
	auto const cross = simplicity_of(
		std::vector<pt>{{-big, -big}, {big, big}, {big, -big}, {-big, big}});
	REQUIRE_FALSE(cross.is_simple);
	REQUIRE(cross.conflict == point{0.0, 0.0});
}

TEST_CASE("simplicity_of: double coordinates parity spot check", "[geometry]") {
	using pd = point<double>;
	REQUIRE(simplicity_of(std::vector<pd>{{0, 0}, {4, 0}, {4, 4}, {0, 4}}).is_simple);
	auto const bow = simplicity_of(std::vector<pd>{{0, 0}, {4, 4}, {4, 0}, {0, 4}});
	REQUIRE_FALSE(bow.is_simple);
	REQUIRE(bow.conflict == point{2.0, 2.0});
	REQUIRE_FALSE(
		simplicity_of(std::vector<pd>{{0, 0}, {4, 0}, {2, 1.5}, {2, -1.5}}).is_simple);
}

TEST_CASE("simplicity_of: random-access ranges need no subscript member", "[geometry]") {
	std::vector<pt> const polygon{{0, 0}, {4, 0}, {4, 4}, {0, 4}};

	struct vertices_view {
		pt const *first;
		pt const *last;

		pt const *begin() const { return first; }

		pt const *end() const { return last; }

		std::size_t size() const { return std::size_t(last - first); }
	};

	vertices_view const view{polygon.data(), polygon.data() + polygon.size()};
	static_assert(atl::geometry::vertex_range<vertices_view>);
	REQUIRE(simplicity_of(view).is_simple);
}
