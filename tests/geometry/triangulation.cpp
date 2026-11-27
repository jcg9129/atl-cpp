#include <atl/ds/dsu.hpp>
#include <atl/geometry/angular_sort.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/geometry/triangulation.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/hash_seed.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

using atl::uint32;
using atl::geometry::containment;
using atl::geometry::contains;
using atl::geometry::orientation;
using atl::geometry::point;
using atl::geometry::polygon_area2;
using atl::geometry::triangulation_of;

namespace {
	atl::int128 oracle_cross(point<std::int64_t> a, point<std::int64_t> b, point<std::int64_t> c) {
		return (atl::int128(b.x) - a.x) * (atl::int128(c.y) - a.y)
			 - (atl::int128(b.y) - a.y) * (atl::int128(c.x) - a.x);
	}

	bool oracle_contains(std::vector<point<std::int64_t>> const &poly, point<std::int64_t> q) {
		bool inside = false;
		for (std::size_t i = 0; i < poly.size(); ++i) {
			auto const a = poly[i], b = poly[(i + 1) % poly.size()];
			auto const det = oracle_cross(a, b, q);
			if (det == 0 && (std::min)(a.x, b.x) <= q.x && q.x <= (std::max)(a.x, b.x)
				&& (std::min)(a.y, b.y) <= q.y && q.y <= (std::max)(a.y, b.y)) return true;
			if ((a.y > q.y) != (b.y > q.y) && (b.y > a.y ? det > 0 : det < 0)) inside = !inside;
		}
		return inside;
	}

	template<atl::geometry::coordinate T>
	bool in_closed_triangle(point<T> const a, point<T> const b, point<T> const c, point<T> const q) {
		if constexpr (std::same_as<T, std::int64_t>)
			return oracle_cross(a, b, q) >= 0 && oracle_cross(b, c, q) >= 0 && oracle_cross(c, a, q) >= 0;
		else
			return orientation(a, b, q) >= 0 && orientation(b, c, q) >= 0 && orientation(c, a, q) >= 0;
	}

	// full validity certificate: counts, positive CCW triangles, exact area
	// partition, edge-manifold structure, dual tree, no T-vertices, and
	// behavioral agreement with contains()
	template<atl::geometry::coordinate T>
	void verify_one(std::vector<point<T>> const &poly,
					atl::geometry::triangulation const &tri) {
		std::size_t const n = poly.size();
		auto const area2 = polygon_area2(poly);
		if (n < 3 || area2 == 0) {
			REQUIRE(tri.triangles.empty());
			REQUIRE(tri.dual_edges.empty());
			return;
		}
		REQUIRE(tri.triangles.size() == n - 2);
		REQUIRE(tri.dual_edges.size() == n - 3);

		atl::geometry::difference_product_t<T> total = 0;
		std::map<std::pair<uint32, uint32>, int> directed;
		for (auto const &[a, b, c] : tri.triangles) {
			REQUIRE(a < n);
			REQUIRE(b < n);
			REQUIRE(c < n);
			REQUIRE(a != b);
			REQUIRE(b != c);
			REQUIRE(a != c);
			if constexpr (std::same_as<T, std::int64_t>) {
				REQUIRE(oracle_cross(poly[a], poly[b], poly[c]) > 0);
			} else {
				REQUIRE(orientation(poly[a], poly[b], poly[c]) == 1);
			} // strictly positive, CCW
			if constexpr (std::same_as<T, std::int64_t>)
				total += oracle_cross(poly[a], poly[b], poly[c]);
			else
				total += cross(poly[a], poly[b], poly[c]);
			for (auto const &[u, v] : std::array{std::pair{a, b}, std::pair{b, c}, std::pair{c, a}}) {
				REQUIRE(directed[{u, v}] == 0); // each directed edge at most once
				directed[{u, v}] = 1;
			}
		}
		if constexpr (std::floating_point<T>) {
			REQUIRE(std::abs(total - std::abs(area2)) <= 1e-10 * (1 + std::abs(area2)));
		} else {
			REQUIRE(total == (area2 < 0 ? -area2 : area2));
		} // exact area partition

		// every polygon edge (in CCW direction) is used exactly once; every
		// other used edge is a diagonal, present in both directions
		std::size_t diagonals = 0;
		for (auto const &[e, cnt] : directed) {
			auto const [u, v] = e;
			bool const boundary = area2 > 0 ? (u + 1) % n == v : (v + 1) % n == u;
			if (boundary) continue;
			REQUIRE(directed.contains({v, u}));
			++diagonals;
		}
		for (std::size_t i = 0; i < n; ++i) {
			auto const u = uint32(area2 > 0 ? i : (i + 1) % n);
			auto const v = uint32(area2 > 0 ? (i + 1) % n : i);
			REQUIRE(directed.contains({u, v}));
		}
		REQUIRE(diagonals == 2 * (n - 3));

		// dual edges: valid distinct triangle pairs sharing exactly one edge,
		// forming a tree (n-3 successful unions over n-2 nodes)
		atl_assert(n - 2 < (atl::uint64(1) << 32));
		atl::dsu components(uint32(n - 2));
		for (auto const &[s, t] : tri.dual_edges) {
			REQUIRE(s < tri.triangles.size());
			REQUIRE(t < tri.triangles.size());
			REQUIRE(s != t);
			int shared = 0;
			for (auto const x : tri.triangles[s])
				shared += std::ranges::count(tri.triangles[t], x) > 0;
			REQUIRE(shared == 2);
			REQUIRE(components.unite(s, t)); // acyclic; count makes it spanning
		}

		if (n <= 60) {
			std::vector<bool> used(n, false);
			for (auto const &t : tri.triangles)
				for (auto const index : t) used[index] = true;
			REQUIRE(std::ranges::all_of(used, [](bool value) { return value; }));
			for (std::size_t i = 0; i < tri.triangles.size(); ++i)
				for (std::size_t j = i + 1; j < tri.triangles.size(); ++j) {
					bool has_separating_edge = false;
					for (int side = 0; side < 2; ++side) {
						auto const &a = tri.triangles[side == 0 ? i : j];
						auto const &b = tri.triangles[side == 0 ? j : i];
						for (std::size_t e = 0; e < 3; ++e) {
							bool is_outside = true;
							for (auto const index : b) {
								if constexpr (std::same_as<T, std::int64_t>)
									is_outside &= oracle_cross(poly[a[e]], poly[a[(e + 1) % 3]], poly[index]) <= 0;
								else
									is_outside &= orientation(poly[a[e]], poly[a[(e + 1) % 3]], poly[index]) <= 0;
							}
							has_separating_edge |= is_outside;
						}
					}
					REQUIRE(has_separating_edge);
				}
			// no T-vertices: a vertex may touch a triangle only as a corner
			for (std::size_t q = 0; q < n; ++q)
				for (auto const &[a, b, c] : tri.triangles) {
					if (q == a || q == b || q == c) continue;
					REQUIRE(!in_closed_triangle(poly[a], poly[b], poly[c], poly[q]));
				}
			// triangle cover agrees with point-in-polygon on a grid sweep
			// (small bounding boxes only — this is a full scan)
			auto const [xlo, xhi] = std::ranges::minmax(poly, {}, &point<T>::x);
			auto const [ylo, yhi] = std::ranges::minmax(poly, {}, &point<T>::y);
			if (double(xhi.x) - double(xlo.x) > 100 || double(yhi.y) - double(ylo.y) > 100) return;
			for (double x = double(xlo.x) - 1; x <= double(xhi.x) + 1; ++x)
				for (double y = double(ylo.y) - 1; y <= double(yhi.y) + 1; ++y) {
					point<T> const q{T(x), T(y)};
					bool covered = false;
					for (auto const &[a, b, c] : tri.triangles)
						covered |= in_closed_triangle(poly[a], poly[b], poly[c], q);
					if constexpr (std::same_as<T, std::int64_t>)
						REQUIRE(covered == oracle_contains(poly, q));
					else
						REQUIRE(covered == (contains(poly, q) != containment::outside));
				}
		}
	}

	template<atl::geometry::coordinate T>
	std::vector<point<T>> reversed(std::vector<point<T>> poly) {
		std::ranges::reverse(poly);
		return poly;
	}

	std::vector<point<std::int64_t>> random_star(std::mt19937_64 &rng, std::int64_t radius) {
		point<std::int64_t> const center{std::int64_t(rng() % 21) - 10, std::int64_t(rng() % 21) - 10};
		std::vector<point<std::int64_t>> raw;
		for (int k = 0; k < 40; ++k) {
			point<std::int64_t> const p{center.x + std::int64_t(rng() % (2 * radius + 1)) - radius,
										center.y + std::int64_t(rng() % (2 * radius + 1)) - radius};
			if (p != center) raw.push_back(p);
		}

		struct keyed_point {
			point<std::int64_t> value;
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
		std::vector<point<std::int64_t>> poly;
		for (auto const p : raw) {
			if (!poly.empty()) {
				auto const &q = poly.back();
				auto const cx = cross(center, q, p);
				auto const dt = dot(q - center, p - center);
				if (cx == 0 && dt > 0) { // same ray: keep the farther
					poly.back() = p;
					continue;
				}
			}
			poly.push_back(p);
		}
		return poly;
	}

	// every certificate runs against both triangulation engines and the
	// adaptive public gate
	template<atl::geometry::coordinate T>
	void verify(std::vector<point<T>> const &poly) {
		verify_one(poly, triangulation_of(poly));
		verify_one(poly, atl::geometry::triangulation_of_by_ear_clipping(poly));
		verify_one(poly, atl::geometry::detail::monotone_sweep_triangulation(poly));
	}
}

TEST_CASE("triangulation_of: handcrafted zoo, both orientations", "[geometry]") {
	std::vector<std::vector<point<std::int64_t>>> const zoo{
		{{0, 0}, {4, 0}, {2, 6}},																			  // triangle
		{{0, 0}, {4, 0}, {4, 4}, {0, 4}},																	  // square
		{{0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 5}, {0, 5}},													  // concave L
		{{0, 0}, {8, 1}, {4, 3}, {7, 7}, {3, 5}, {1, 8}, {2, 4}, {-3, 3}},									  // star
		{{0, 0}, {1, 0}, {2, 0}, {1, 2}},																	  // straight vertex on the base
		{{0, 0}, {2, 0}, {4, 0}, {4, 4}, {2, 4}, {0, 4}, {0, 2}},											  // collinear edges
		{{0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}},									  // all-midpoint square
		{{0, 0}, {6, 0}, {3, 1}},																			  // sliver
		{{0, 0}, {1, 3}, {2, 0}, {3, 3}, {4, 0}, {5, 3}, {6, 0}, {6, 5}, {0, 5}},							  // comb
		{{0, 0}, {10, 0}, {10, 10}, {2, 10}, {2, 4}, {6, 4}, {6, 6}, {4, 6}, {4, 8}, {8, 8}, {8, 2}, {0, 2}}, // spiral
	};
	for (auto const &poly : zoo) {
		verify(poly);
		verify(reversed(poly));
		// exact-policy parity across coordinate types: identical decisions,
		// identical triples
		std::vector<point<double>> polyd;
		for (auto const p : poly) polyd.push_back({double(p.x), double(p.y)});
		REQUIRE(triangulation_of(polyd).triangles == triangulation_of(poly).triangles);
		verify(polyd);
	}
}

TEST_CASE("triangulation_of: degenerate inputs are empty", "[geometry]") {
	verify(std::vector<point<std::int64_t>>{});
	verify(std::vector<point<std::int64_t>>{{1, 2}});
	verify(std::vector<point<std::int64_t>>{{1, 2}, {3, 4}});
	verify(std::vector<point<std::int64_t>>{{0, 0}, {1, 0}, {2, 0}}); // zero area
}

TEST_CASE("triangulation_of: random star polygons", "[geometry]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	auto const poly = random_star(rng, 3 + std::int64_t(rng() % 20));
	if (poly.size() >= 3) {
		verify(poly);
		verify(reversed(poly));
	}
}

TEST_CASE("triangulation_of: int64 coordinates at scale", "[geometry]") {
	std::int64_t const base = std::int64_t(1) << 59;
	std::int64_t const w = 1'000'000'007, h = 998'244'353;
	// rectangle with a notch: one reflex vertex at scale
	std::vector<point<std::int64_t>> const poly{{base, base}, {base + w, base}, {base + w, base + h}, {base + w / 2, base + h / 3}, {base + w / 3, base + h}, {base, base + h}};
	verify(poly);
	verify(reversed(poly));
}

TEST_CASE("triangulation_of: the gate takes the sweep at scale, both windings", "[geometry]") {
	// r ~ n/2 reflex vertices with n in the hundreds to thousands: above the
	// 8 log2 n threshold, so the public entry must dispatch to the sweep
	auto const reflex_count = [](std::vector<point<std::int64_t>> const &poly) {
		bool const is_ccw = polygon_area2(poly) > 0;
		std::size_t count = 0;
		for (std::size_t v = 0; v < poly.size(); ++v) {
			int const turn = orientation(poly[v], poly[(v + 1) % poly.size()], poly[(v + 2) % poly.size()]);
			count += (is_ccw ? turn : -turn) <= 0;
		}
		return count;
	};
	auto const check = [&](std::vector<point<std::int64_t>> const &poly) {
		REQUIRE(reflex_count(poly) > 8 * std::size_t(std::bit_width(poly.size())));
		auto const tri = triangulation_of(poly);
		REQUIRE(tri.triangles == atl::geometry::detail::monotone_sweep_triangulation(poly).triangles);
		verify_one(poly, tri);
	};
	// comb: k teeth, each tooth's tip reflex-flanked, plus a subdivided
	// (collinear) base
	for (std::int64_t const k : {150, 700}) {
		std::vector<point<std::int64_t>> comb;
		for (std::int64_t t = 0; t < k; ++t) {
			comb.push_back({2 * t, 0});
			comb.push_back({2 * t + 1, 5});
		}
		comb.push_back({2 * k, 0});
		comb.push_back({2 * k, 10});
		for (std::int64_t x = 2 * k - 1; x >= 0; --x) comb.push_back({x, 10}); // long collinear run
		check(comb);
		check(reversed(comb));
	}
	auto seed = GENERATE(take(4, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	// a star around the origin: random radii on distinct rays, ~half reflex
	std::vector<point<std::int64_t>> raw;
	for (int k = 0; k < 1500; ++k) raw.push_back({std::int64_t(rng() % 20001) - 10000, std::int64_t(rng() % 20001) - 10000});
	atl::geometry::sort_by_polar_angle(raw);
	std::vector<point<std::int64_t>> star;
	for (auto const q : raw)
		if (q != point<std::int64_t>{} && (star.empty() || orientation(point<std::int64_t>{}, star.back(), q) != 0))
			star.push_back(q);
	if (!star.empty() && orientation(point<std::int64_t>{}, star.back(), star.front()) == 0) star.pop_back();
	check(star);
	check(reversed(star));
}

TEST_CASE("triangulation: full-range int32 and narrow angular decisions", "[geometry]") {
	using p32 = point<std::int32_t>;
	auto const low = (std::numeric_limits<std::int32_t>::min)();
	auto const high = (std::numeric_limits<std::int32_t>::max)();
	std::vector<p32> const polygon{{low, low}, {high, low}, {high, high}, {0, 0}, {low, high}};
	verify(polygon);
	std::vector<point<atl::int16>> const narrow{{-32768, -32768}, {32767, -32768}, {32767, 32767}, {0, 0}, {-32768, 32767}};
	verify(narrow);
	atl::int64 const bound = (atl::int64(1) << 62) - 1;
	std::vector<point<atl::int64>> const wide{{-bound, -bound}, {bound, -bound}, {bound, bound}, {0, 0}, {-bound, bound}};
	verify(wide);
}

TEST_CASE("triangulation: large collinear and comb inputs use allocated work storage", "[geometry]") {
	std::vector<point<atl::int64>> collinear(1'000'000);
	for (std::size_t i = 0; i < collinear.size(); ++i) collinear[i] = {atl::int64(i), 0};
	REQUIRE(triangulation_of(collinear).triangles.empty());
	collinear.clear();
	std::vector<point<atl::int64>> comb;
	for (atl::int64 i = 0; i < 25'000; ++i) {
		comb.push_back({2 * i, 0});
		comb.push_back({2 * i + 1, 5});
	}
	comb.push_back({50'000, 0});
	for (atl::int64 x = 50'000; x >= 0; --x) comb.push_back({x, 10});
	auto const result = triangulation_of(comb);
	REQUIRE(result.triangles.size() == comb.size() - 2);
	REQUIRE(result.dual_edges.size() == comb.size() - 3);
	atl::int128 total = 0;
	for (auto const &t : result.triangles) {
		auto const area = oracle_cross(comb[t[0]], comb[t[1]], comb[t[2]]);
		REQUIRE(area > 0);
		total += area;
	}
	REQUIRE(total == polygon_area2(comb));
}

TEST_CASE("triangulation: finite floating product envelopes", "[geometry]") {
	auto const check = []<atl::geometry::coordinate T>(T const scale)
		requires std::floating_point<T>
	{
		std::array const polygon{point<T>{-scale, -scale}, point<T>{scale, -scale}, point<T>{scale, scale}, point<T>{-scale, scale}};
		auto const result = triangulation_of(polygon);
		REQUIRE(result.triangles.size() == 2);
		for (auto const &t : result.triangles) REQUIRE(orientation(polygon[t[0]], polygon[t[1]], polygon[t[2]]) > 0);
	};
	float const float_scale = (std::numeric_limits<float>::max)() / 4;
	double const double_scale = std::ldexp(1.0, 500);
	check(float_scale);
	check(double_scale);
	check(1 / float_scale);
	check(1 / double_scale);
}

namespace {
	struct ear_orientation_counter : atl::geometry::exact {
		std::size_t *calls;

		explicit ear_orientation_counter(std::size_t &count) : calls(&count) {}

		int orientation(point<std::int64_t> a, point<std::int64_t> b, point<std::int64_t> c) const {
			++*calls;
			return atl::geometry::exact{}.orientation(a, b, c);
		}
	};
}

TEST_CASE("ear clipping accepts views and policies and preserves long straight chains", "[geometry]") {
	using atl::geometry::triangulation_of_by_ear_clipping;
	std::array const narrow_polygon{point<atl::int16>{0, 0}, point<atl::int16>{4, 0}, point<atl::int16>{0, 4}};
	auto const narrow_result = triangulation_of_by_ear_clipping(narrow_polygon | std::views::reverse);
	REQUIRE(narrow_result.triangles.size() == 1);
	std::array const floating_polygon{point{0.0, 0.0}, point{4.0, 0.0}, point{0.0, 4.0}};
	REQUIRE(triangulation_of_by_ear_clipping(floating_polygon, atl::geometry::approx{1e-12}).triangles.size() == 1);
	// The apex is the first candidate; cutting it would strand all base vertices.
	std::vector<point<std::int64_t>> polygon{{0, 10}};
	for (std::int64_t x = -256; x <= 256; ++x) polygon.push_back({x, 0});
	std::size_t calls = 0;
	auto const result = triangulation_of_by_ear_clipping(polygon, ear_orientation_counter(calls));
	REQUIRE(calls < 100 * polygon.size()); // Linear predicate work with no reflex vertices.
	verify_one(polygon, result);
}

TEST_CASE("triangulation: hash seed preserves exact triangle and dual order", "[geometry]") {
	std::vector<std::vector<point<std::int64_t>>> polygons{
		{{0, 0}, {2, 0}, {4, 0}, {4, 4}, {2, 4}, {0, 4}, {0, 2}},
		{{0, 0}, {10, 0}, {10, 10}, {2, 10}, {2, 4}, {6, 4}, {6, 6}, {4, 6}, {4, 8}, {8, 8}, {8, 2}, {0, 2}},
	};
	std::vector<point<std::int64_t>> comb;
	for (std::int64_t i = 0; i < 160; ++i) {
		comb.push_back({2 * i, 0});
		comb.push_back({2 * i + 1, 2});
	}
	comb.push_back({320, 0});
	comb.push_back({320, 5});
	comb.push_back({0, 5});
	polygons.push_back(comb);
	auto const saved = atl::detail::hash_seed_value;
	for (auto const &polygon : polygons) {
		auto const expected = atl::geometry::detail::monotone_sweep_triangulation(polygon);
		for (auto const seed : {atl::uint64(0), atl::uint64(1), ~atl::uint64(0), atl::uint64(123456789)}) {
			atl::set_hash_seed(seed);
			auto const actual = atl::geometry::detail::monotone_sweep_triangulation(polygon);
			CHECK(actual.triangles == expected.triangles);
			CHECK(actual.dual_edges == expected.dual_edges);
		}
	}
	atl::detail::hash_seed_value = saved;
}
