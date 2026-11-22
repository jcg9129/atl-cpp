#include <atl/ds/dsu.hpp>
#include <atl/experimental/geometry/arrangement_all_pairs.hpp>
#include <atl/geometry/arrangement.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <vector>

using atl::float64;
using atl::int128;
using atl::geometry::arrangement;
using atl::geometry::arrangement_of;
using atl::geometry::intersects;
using atl::geometry::point;
using atl::geometry::segment;

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

	// exact rational point (nx/d, ny/d), d > 0 — small-coordinate inputs
	// only, so int128 cross-multiplication is safe
	struct rat {
		int128 nx, ny, d;

		friend bool operator<(rat const &l, rat const &r) {
			if (int128 const c = l.nx * r.d - r.nx * l.d; c != 0) return c < 0;
			return l.ny * r.d - r.ny * l.d < 0;
		}
	};

	bool on_segment(sg const s, rat const &p) {
		int128 const a = s.b.y - s.a.y, b = s.a.x - s.b.x;
		int128 const c = -(a * s.a.x + b * s.a.y);
		if (a * p.nx + b * p.ny + c * p.d != 0) return false;
		auto const [lo_x, hi_x] = std::minmax(s.a.x, s.b.x);
		auto const [lo_y, hi_y] = std::minmax(s.a.y, s.b.y);
		return int128(lo_x) * p.d <= p.nx && p.nx <= int128(hi_x) * p.d
			&& int128(lo_y) * p.d <= p.ny && p.ny <= int128(hi_y) * p.d;
	}

	// Independent V / E / C for the Euler certificate: vertices = distinct
	// endpoints and pairwise common points; edges = deduplicated
	// consecutive-vertex pieces along each segment; components over both.
	struct brute_counts {
		std::size_t v, e, c;
	};

	brute_counts brute(std::vector<sg> const &segs) {
		std::map<rat, std::uint32_t> ids;
		auto const id_of = [&](rat const &p) {
			return ids.emplace(p, std::uint32_t(ids.size())).first->second;
		};
		for (auto const &s : segs) {
			id_of({s.a.x, s.a.y, 1});
			id_of({s.b.x, s.b.y, 1});
		}
		for (std::size_t i = 0; i < segs.size(); ++i)
			for (std::size_t j = i + 1; j < segs.size(); ++j) {
				sg const s = segs[i], t = segs[j];
				if (!brute_intersects(s, t)) continue;
				int128 const as = s.b.y - s.a.y, bs = s.a.x - s.b.x;
				int128 const cs = -(as * s.a.x + bs * s.a.y);
				int128 const at = t.b.y - t.a.y, bt = t.a.x - t.b.x;
				int128 const ct = -(at * t.a.x + bt * t.a.y);
				int128 d = as * bt - at * bs;
				if (d == 0) { // parallel/degenerate touch or overlap: the
							  // endpoints landing on the other are the cuts
					for (auto const e : {s.a, s.b})
						if (on_segment(t, {e.x, e.y, 1})) id_of({e.x, e.y, 1});
					for (auto const e : {t.a, t.b})
						if (on_segment(s, {e.x, e.y, 1})) id_of({e.x, e.y, 1});
					continue;
				}
				int128 nx = bs * ct - bt * cs, ny = cs * at - ct * as;
				if (d < 0) {
					d = -d;
					nx = -nx;
					ny = -ny;
				}
				id_of({nx, ny, d});
			}
		std::set<std::pair<std::uint32_t, std::uint32_t>> pieces;
		for (auto const &s : segs) {
			std::vector<std::pair<rat, std::uint32_t>> cuts;
			for (auto const &[p, k] : ids)
				if (on_segment(s, p)) cuts.emplace_back(p, k);
			std::ranges::sort(cuts, [](auto const &l, auto const &r) {
				return l.first < r.first;
			});
			for (std::size_t k = 0; k + 1 < cuts.size(); ++k)
				pieces.emplace(cuts[k].second, cuts[k + 1].second);
		}
		atl::dsu comp(std::uint32_t(ids.size()));
		for (auto const &[a, b] : pieces) comp.unite(a, b);
		std::size_t roots = 0;
		for (std::uint32_t k = 0; k < std::uint32_t(ids.size()); ++k)
			roots += comp.find(k) == k;
		return {ids.size(), pieces.size(), roots};
	}

	// Structural + Euler + area certificate against the brute.
	void verify(std::vector<sg> const &segs) {
		auto const a = arrangement_of(segs);
		std::size_t const halves = a.origin.size();
		REQUIRE(halves == 2 * std::size_t(a.edge_count()));
		REQUIRE(a.next.size() == halves);
		REQUIRE(a.face.size() == halves);
		REQUIRE(a.supporting.size() == a.edge_count());
		// next is a permutation continuing where its argument ends, faces are
		// constant along cycles, twins share their edge's supporting list
		std::vector<std::uint32_t> in_degree(halves, 0);
		for (std::uint32_t h = 0; h < halves; ++h) {
			REQUIRE(a.next[h] < halves);
			REQUIRE(a.origin[a.next[h]] == a.to(h));
			REQUIRE(a.face[a.next[h]] == a.face[h]);
			++in_degree[a.next[h]];
			REQUIRE(a.face[h] < a.faces.size());
		}
		for (auto const d : in_degree) REQUIRE(d == 1);
		for (auto const &sup : a.supporting) REQUIRE(!sup.empty());
		// every supporting claim is geometrically true: both piece endpoints
		// lie on the named input segment (rounded positions are exact here)
		for (std::uint32_t e = 0; e < a.edge_count(); ++e)
			for (auto const s : a.supporting[e])
				for (auto const v : {a.origin[2 * e], a.origin[2 * e + 1]}) {
					auto const p = a.vertices[v];
					// integer-lattice vertices are exact; rational cuts are
					// certified by the brute's V instead
					if (p.x == std::floor(p.x) && p.y == std::floor(p.y))
						REQUIRE(on_segment(segs[s],
										   {std::int64_t(p.x), std::int64_t(p.y), 1}));
				}
		// Euler: F = 1 + C - V + E, against the independent brute
		auto const counts = brute(segs);
		REQUIRE(a.vertices.size() == counts.v);
		REQUIRE(a.edge_count() == counts.e);
		REQUIRE(a.faces.size() == 1 + counts.c - counts.v + counts.e);
		// areas: bounded faces non-negative, everything sums to zero
		float64 total = 0;
		for (std::size_t f = 0; f < a.faces.size(); ++f) {
			total += a.faces[f].area;
			if (f == arrangement::unbounded_face) {
				REQUIRE(a.faces[f].outer == arrangement::none);
				REQUIRE(a.faces[f].area <= 1e-9);
			} else {
				REQUIRE(a.faces[f].outer != arrangement::none);
				REQUIRE(a.faces[f].area >= -1e-9);
			}
		}
		REQUIRE(std::abs(total) <= 1e-6);
		// isolated vertices are exactly those with no incident half-edge
		std::vector<char> touched(a.vertices.size(), 0);
		for (std::uint32_t h = 0; h < halves; ++h) touched[a.origin[h]] = 1;
		std::size_t isolated = 0;
		for (auto const &f : a.faces) isolated += f.isolated.size();
		REQUIRE(isolated
				== std::size_t(std::ranges::count(touched, char(0))));
	}
}

TEST_CASE("arrangement: handcrafted zoo", "[geometry]") {
	SECTION("empty and trivial inputs") {
		auto const empty = arrangement_of(std::vector<sg>{});
		REQUIRE(empty.faces.size() == 1);
		REQUIRE(empty.vertices.empty());

		auto const lone = arrangement_of(std::vector<sg>{{{0, 0}, {4, 0}}});
		REQUIRE(lone.vertices.size() == 2);
		REQUIRE(lone.edge_count() == 1);
		REQUIRE(lone.faces.size() == 1);
		REQUIRE(lone.faces[0].holes.size() == 1); // the slit walk

		auto const pt = arrangement_of(std::vector<sg>{{{2, 2}, {2, 2}}});
		REQUIRE(pt.vertices.size() == 1);
		REQUIRE(pt.faces.size() == 1);
		REQUIRE(pt.faces[0].isolated == std::vector<std::uint32_t>{0});
	}
	SECTION("square: one bounded face") {
		auto const a = arrangement_of(std::vector<sg>{
			{{0, 0}, {4, 0}}, {{4, 0}, {4, 4}}, {{4, 4}, {0, 4}}, {{0, 4}, {0, 0}}});
		REQUIRE(a.faces.size() == 2);
		REQUIRE(std::abs(a.faces[1].area - (16.0)) <= 1e-12);
		REQUIRE(std::abs(a.faces[0].area - (-16.0)) <= 1e-12);
		REQUIRE(a.faces[0].holes.size() == 1);
		REQUIRE(a.faces[1].holes.empty());
		verify({{{0, 0}, {4, 0}}, {{4, 0}, {4, 4}}, {{4, 4}, {0, 4}}, {{0, 4}, {0, 0}}});
	}
	SECTION("nested squares: hole attachment") {
		std::vector<sg> const nested{
			{{0, 0}, {8, 0}}, {{8, 0}, {8, 8}}, {{8, 8}, {0, 8}}, {{0, 8}, {0, 0}}, {{3, 3}, {5, 3}}, {{5, 3}, {5, 5}}, {{5, 5}, {3, 5}}, {{3, 5}, {3, 3}}};
		auto const a = arrangement_of(nested);
		REQUIRE(a.faces.size() == 3);
		// the ring face: area 64 - 4, one hole; the inner square: area 4
		std::multiset<float64> areas;
		for (auto const &f : a.faces) areas.insert(std::round(f.area));
		REQUIRE(areas == std::multiset<float64>{-64.0, 4.0, 60.0});
		for (auto const &f : a.faces)
			if (std::round(f.area) == 60.0) REQUIRE(f.holes.size() == 1);
		verify(nested);
	}
	SECTION("three levels of nesting") {
		std::vector<sg> nested;
		for (std::int64_t r : {12, 6, 2}) {
			nested.push_back({{-r, -r}, {r, -r}});
			nested.push_back({{r, -r}, {r, r}});
			nested.push_back({{r, r}, {-r, r}});
			nested.push_back({{-r, r}, {-r, -r}});
		}
		auto const a = arrangement_of(nested);
		REQUIRE(a.faces.size() == 4);
		std::multiset<float64> areas;
		for (auto const &f : a.faces) areas.insert(std::round(f.area));
		REQUIRE(areas == std::multiset<float64>{-576.0, 16.0, 432.0, 128.0});
		verify(nested);
	}
	SECTION("two siblings inside one square: two holes on one face") {
		std::vector<sg> segs{
			{{0, 0}, {12, 0}}, {{12, 0}, {12, 6}}, {{12, 6}, {0, 6}}, {{0, 6}, {0, 0}}, {{1, 1}, {3, 1}}, {{3, 1}, {3, 3}}, {{3, 3}, {1, 3}}, {{1, 3}, {1, 1}}, {{8, 1}, {10, 1}}, {{10, 1}, {10, 3}}, {{10, 3}, {8, 3}}, {{8, 3}, {8, 1}}};
		auto const a = arrangement_of(segs);
		REQUIRE(a.faces.size() == 4);
		for (auto const &f : a.faces)
			if (std::round(f.area) == 64.0) REQUIRE(f.holes.size() == 2);
		verify(segs);
	}
	SECTION("slit and isolated point inside a face do not split it") {
		std::vector<sg> segs{{{0, 0}, {8, 0}},
							 {{8, 0}, {8, 8}},
							 {{8, 8}, {0, 8}},
							 {{0, 8}, {0, 0}},
							 {{2, 2}, {5, 5}},
							 {{6, 1}, {6, 1}}};
		auto const a = arrangement_of(segs);
		REQUIRE(a.faces.size() == 2);
		for (auto const &f : a.faces)
			if (f.outer != arrangement::none) {
				REQUIRE(std::round(f.area) == 64.0);
				REQUIRE(f.holes.size() == 1);	 // the slit
				REQUIRE(f.isolated.size() == 1); // the point segment
			}
		verify(segs);
	}
	SECTION("collinear overlap merges into shared edges") {
		std::vector<sg> segs{{{0, 0}, {6, 0}}, {{2, 0}, {9, 0}}};
		auto const a = arrangement_of(segs);
		REQUIRE(a.vertices.size() == 4);
		REQUIRE(a.edge_count() == 3);
		// the middle piece (2,0)-(6,0) carries BOTH segments
		bool shared_seen = false;
		for (std::uint32_t e = 0; e < a.edge_count(); ++e)
			if (a.supporting[e].size() == 2) {
				shared_seen = true;
				auto const lo = a.vertices[a.origin[2 * e]];
				auto const hi = a.vertices[a.origin[2 * e + 1]];
				REQUIRE(lo.x == 2.0);
				REQUIRE(hi.x == 6.0);
			}
		REQUIRE(shared_seen);
		verify(segs);
	}
	SECTION("crossing X and a diagonal-cut square") {
		verify({{{0, 0}, {4, 4}}, {{0, 4}, {4, 0}}});
		std::vector<sg> const cut{{{0, 0}, {4, 0}},
								  {{4, 0}, {4, 4}},
								  {{4, 4}, {0, 4}},
								  {{0, 4}, {0, 0}},
								  {{0, 0}, {4, 4}}};
		auto const a = arrangement_of(cut);
		REQUIRE(a.faces.size() == 3);
		verify(cut);
	}
	SECTION("k-star inside a square") {
		std::vector<sg> segs{{{-9, -9}, {9, -9}}, {{9, -9}, {9, 9}}, {{9, 9}, {-9, 9}}, {{-9, 9}, {-9, -9}}};
		for (std::int64_t d = 1; d <= 3; ++d)
			segs.push_back({{-d, -d * d}, {d, d * d}});
		segs.push_back({{0, -7}, {0, 7}});
		verify(segs);
	}
}

TEST_CASE("arrangement: random degenerate-dense grids vs the certificate",
		  "[geometry]") {
	auto seed = GENERATE(take(40, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 60; ++it) {
		std::size_t const n = std::size_t(1 + rng() % 9);
		std::int64_t const span = 2 + std::int64_t(rng() % 5);
		std::vector<sg> segs;
		for (std::size_t k = 0; k < n; ++k) {
			auto const coord = [&] { return std::int64_t(rng() % (span + 1)); };
			sg s{{coord(), coord()}, {coord(), coord()}};
			if (rng() % 7 == 0) s.b = s.a;
			segs.push_back(s);
		}
		if (rng() % 2) { // overlap-rich mode, as in the sweep tests
			std::int64_t const ox = std::int64_t(rng() % 3) - 1,
							   oy = std::int64_t(rng() % 3) - 1;
			std::int64_t const dx = std::int64_t(rng() % 3) - 1,
							   dy = std::int64_t(rng() % 3);
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
}

TEST_CASE("arrangement: double coordinates parity spot check", "[geometry]") {
	using sd = segment<double>;
	auto const a = arrangement_of(std::vector<sd>{
		{{0, 0}, {4, 0}}, {{4, 0}, {4, 4}}, {{4, 4}, {0, 4}}, {{0, 4}, {0, 0}}});
	REQUIRE(a.faces.size() == 2);
	REQUIRE(std::abs(a.faces[1].area - (16.0)) <= 1e-12);
	auto const x = arrangement_of(
		std::vector<sd>{{{0, 0}, {4, 4}}, {{0, 4}, {4, 0}}});
	REQUIRE(x.vertices.size() == 5);
	REQUIRE(x.faces.size() == 1);
}

TEST_CASE("arrangement: all-pairs contender is bit-identical to the sweep",
		  "[geometry]") {
	// same sweep-ordered vertex ids + the shared assembly => the two
	// builders must agree on EVERY output array, not just on invariants
	auto const same = [](std::vector<sg> const &segs) {
		auto const a = arrangement_of(segs);
		auto const b = atl::experimental::arrangement_of_by_all_pairs(segs);
		REQUIRE(a.vertices.size() == b.vertices.size());
		for (std::size_t v = 0; v < a.vertices.size(); ++v) {
			REQUIRE(std::abs(a.vertices[v].x - b.vertices[v].x) <= 1e-12 * (1 + std::abs(a.vertices[v].x)));
			REQUIRE(std::abs(a.vertices[v].y - b.vertices[v].y) <= 1e-12 * (1 + std::abs(a.vertices[v].y)));
		}
		REQUIRE(a.origin == b.origin);
		REQUIRE(a.next == b.next);
		REQUIRE(a.face == b.face);
		REQUIRE(a.supporting == b.supporting);
		REQUIRE(a.faces.size() == b.faces.size());
		for (std::size_t f = 0; f < a.faces.size(); ++f) {
			REQUIRE(a.faces[f].outer == b.faces[f].outer);
			REQUIRE(a.faces[f].holes == b.faces[f].holes);
			REQUIRE(a.faces[f].isolated == b.faces[f].isolated);
			REQUIRE(std::abs(a.faces[f].area - b.faces[f].area) <= 1e-9);
		}
	};
	same({});
	same({{{0, 0}, {4, 0}}});
	same({{{2, 2}, {2, 2}}});
	same({{{0, 0}, {8, 0}}, {{8, 0}, {8, 8}}, {{8, 8}, {0, 8}}, {{0, 8}, {0, 0}}, {{3, 3}, {5, 3}}, {{5, 3}, {5, 5}}, {{5, 5}, {3, 5}}, {{3, 5}, {3, 3}}});
	same({{{0, 0}, {6, 0}}, {{2, 0}, {9, 0}}});
	auto seed = GENERATE(take(20, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 60; ++it) {
		std::size_t const n = std::size_t(1 + rng() % 9);
		std::int64_t const span = 2 + std::int64_t(rng() % 5);
		std::vector<sg> segs;
		for (std::size_t k = 0; k < n; ++k) {
			auto const coord = [&] { return std::int64_t(rng() % (span + 1)); };
			sg s2{{coord(), coord()}, {coord(), coord()}};
			if (rng() % 7 == 0) s2.b = s2.a;
			segs.push_back(s2);
		}
		if (rng() % 2) {
			std::int64_t const ox = std::int64_t(rng() % 3) - 1,
							   oy = std::int64_t(rng() % 3) - 1;
			std::int64_t const dx = std::int64_t(rng() % 3) - 1,
							   dy = std::int64_t(rng() % 3);
			if (dx != 0 || dy != 0)
				for (auto &s2 : segs)
					if (rng() % 2) {
						std::int64_t const t0 = std::int64_t(rng() % 5) - 2;
						std::int64_t const t1 = t0 + std::int64_t(rng() % 4);
						s2 = {{ox + t0 * dx, oy + t0 * dy}, {ox + t1 * dx, oy + t1 * dy}};
					}
		}
		same(segs);
	}
}

TEST_CASE("arrangement: full-range int32 and int16 directions do not overflow", "[geometry]") {
	// spans of 4e9 overflow int32 if the edge directions difference in T
	std::vector<segment<std::int32_t>> const wide{{{-2'000'000'000, 0}, {2'000'000'000, 1}},
												  {{-2'000'000'000, 1}, {2'000'000'000, 0}}};
	auto const a = arrangement_of(wide);
	REQUIRE(a.vertices.size() == 5);
	REQUIRE(a.edge_count() == 4);
	REQUIRE(a.faces.size() == 1);
	std::vector<segment<std::int16_t>> const narrow{{{-30'000, 0}, {30'000, 1}}, {{-30'000, 1}, {30'000, 0}}};
	auto const b = arrangement_of(narrow);
	REQUIRE(b.vertices.size() == 5);
	REQUIRE(b.edge_count() == 4);
}

TEST_CASE("arrangement: all-pairs coordinates at its asserted int32 bound", "[geometry]") {
	std::int32_t const bound = (std::int32_t(1) << 26) - 1;
	std::vector<segment<std::int32_t>> const segments{{{-bound, -bound}, {bound, bound}}, {{-bound, bound}, {bound, -bound}}, {{0, 1}, {0, 1}}};
	auto const a = arrangement_of(segments);
	auto const b = atl::experimental::arrangement_of_by_all_pairs(segments);
	REQUIRE(a.origin == b.origin);
	REQUIRE(a.next == b.next);
	REQUIRE(a.face == b.face);
	REQUIRE(a.supporting == b.supporting);
}

TEST_CASE("arrangement: simultaneous x crossings keep resident order", "[geometry]") {
	verify({{{0, 0}, {2, 2}}, {{0, 2}, {2, 0}}, {{0, 3}, {2, 5}}, {{0, 5}, {2, 3}}});
}
