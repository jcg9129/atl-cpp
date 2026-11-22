#include <atl/experimental/geometry/trapezoid_point_location.hpp>
#include <atl/geometry/arrangement.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/point_location.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/hash.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>
#include "../expect_abort.hpp"

using atl::int128;
using atl::geometry::arrangement;
using atl::geometry::arrangement_of;
using atl::geometry::location;
using atl::geometry::location_kind;
using atl::geometry::locations_of;
using atl::geometry::point;
using atl::geometry::point_location;
using atl::geometry::segment;

namespace {
	using sg = segment<std::int64_t>;
	using pt = point<std::int64_t>;

	struct oracle_point {
		int128 x, y, d;
	};

	int oracle_compare(int128 a, int128 b, int128 c, int128 d) {
		return atl::detail::compare_products(a, b, c, d);
	}

	bool oracle_less(oracle_point const &a, oracle_point const &b) {
		int const x = oracle_compare(a.x, b.d, b.x, a.d);
		return x != 0 ? x < 0 : oracle_compare(a.y, b.d, b.y, a.d) < 0;
	}

	std::vector<oracle_point> oracle_vertices(std::vector<sg> const &segs) {
		std::vector<oracle_point> vertices;
		for (auto const &s : segs) {
			vertices.push_back({s.a.x, s.a.y, 1});
			vertices.push_back({s.b.x, s.b.y, 1});
		}
		for (std::size_t i = 0; i < segs.size(); ++i)
			for (std::size_t j = i + 1; j < segs.size(); ++j) {
				auto const s = segs[i], t = segs[j];
				int128 const dx = int128(s.b.x) - s.a.x, dy = int128(s.b.y) - s.a.y;
				int128 const ex = int128(t.b.x) - t.a.x, ey = int128(t.b.y) - t.a.y;
				int128 d = dx * ey - dy * ex;
				if (d == 0) continue;
				int128 const ox = int128(t.a.x) - s.a.x, oy = int128(t.a.y) - s.a.y;
				int128 u = ox * ey - oy * ex, v = ox * dy - oy * dx;
				if (d < 0) {
					d = -d;
					u = -u;
					v = -v;
				}
				if (u < 0 || d < u || v < 0 || d < v) continue;
				vertices.push_back({int128(s.a.x) * d + dx * u, int128(s.a.y) * d + dy * u, d});
			}
		std::ranges::sort(vertices, oracle_less);
		auto const duplicates = std::ranges::unique(vertices, [](auto const &a, auto const &b)
														requires std::same_as<std::remove_cvref_t<decltype(a)>, oracle_point>
													{
														return !oracle_less(a, b) && !oracle_less(b, a);
													});
		vertices.erase(duplicates.begin(), duplicates.end());
		return vertices;
	}

	location brute(arrangement const &a, std::vector<sg> const &segs, pt const q) {
		auto const vertices = oracle_vertices(segs);
		REQUIRE(vertices.size() == a.vertices.size());
		oracle_point const query{q.x, q.y, 1};
		for (std::size_t v = 0; v < vertices.size(); ++v)
			if (!oracle_less(vertices[v], query) && !oracle_less(query, vertices[v]))
				return {location_kind::vertex, std::uint32_t(v)};
		std::uint32_t best = arrangement::none;
		int128 best_n = 0, best_d = 1, best_a = 0, best_b = 1;
		for (std::uint32_t e = 0; e < a.edge_count(); ++e) {
			auto const s = segs[a.supporting[e][0]];
			int128 A = int128(s.b.y) - s.a.y, B = int128(s.a.x) - s.b.x;
			int128 C = -(A * s.a.x + B * s.a.y);
			if (B < 0 || (B == 0 && A < 0)) {
				A = -A;
				B = -B;
				C = -C;
			}
			int128 const val = A * q.x + B * q.y + C;
			auto const &left = vertices[a.origin[2 * std::size_t(e)]];
			auto const &right = vertices[a.origin[2 * std::size_t(e) + 1]];
			if (val == 0 && !oracle_less(query, left) && !oracle_less(right, query))
				return {location_kind::edge, e};
			if (B == 0 || val <= 0 || !oracle_less(left, query) || !oracle_less(query, right)) continue;
			int128 const n = -(A * q.x + C), d = B;
			int const versus = best == arrangement::none ? 1 : oracle_compare(n, best_d, best_n, d);
			if (versus > 0 || (versus == 0 && oracle_compare(-A, best_b, -best_a, B) > 0)) {
				best = e;
				best_n = n;
				best_d = d;
				best_a = A;
				best_b = B;
			}
		}
		return {location_kind::face, best == arrangement::none ? arrangement::unbounded_face : a.face[2 * std::size_t(best)]};
	}

	std::vector<sg> random_grid(std::mt19937_64 &rng, std::int64_t &span_out) {
		std::size_t const n = std::size_t(1 + rng() % 9);
		std::int64_t const span = 2 + std::int64_t(rng() % 5);
		span_out = span;
		std::vector<sg> segs;
		for (std::size_t k = 0; k < n; ++k) {
			auto const coord = [&] { return std::int64_t(rng() % (span + 1)); };
			sg s{{coord(), coord()}, {coord(), coord()}};
			if (rng() % 7 == 0) s.b = s.a;
			segs.push_back(s);
		}
		return segs;
	}
}

TEST_CASE("point_location: every lattice point matches the exhaustive brute",
		  "[geometry]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 150; ++it) {
		std::int64_t span = 0;
		auto const segs = random_grid(rng, span);
		auto const arr = arrangement_of(segs);
		point_location const loc(arr, segs);
		for (std::int64_t x = -1; x <= span + 1; ++x)
			for (std::int64_t y = -1; y <= span + 1; ++y)
				REQUIRE(loc.location_of({x, y}) == brute(arr, segs, {x, y}));
	}
}

TEST_CASE("point_location: trapezoid-map contender agrees everywhere",
		  "[geometry]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	atl::set_hash_seed(seed); // the map's insertion order replays from the Catch seed
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 150; ++it) {
		std::int64_t span = 0;
		auto const segs = random_grid(rng, span);
		auto const arr = arrangement_of(segs);
		point_location const slabs(arr, segs);
		atl::experimental::point_location_by_trapezoid_map const traps(arr, segs);
		for (std::int64_t x = -1; x <= span + 1; ++x)
			for (std::int64_t y = -1; y <= span + 1; ++y)
				REQUIRE(slabs.location_of({x, y}) == traps.location_of({x, y}));
	}
}

TEST_CASE("point_location: trapezoid contender on long chords up to 2^31", "[geometry]") {
	// a few hundred long crossing chords: long crossed runs, many rational
	// vertices, int256 magnitudes in the map's comparisons
	auto seed = GENERATE(take(4, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	atl::set_hash_seed(seed);
	std::mt19937_64 rng(seed);
	std::int64_t const big = std::int64_t(1) << 31;
	auto const coord = [&] { return std::int64_t(rng() % std::uint64_t(2 * big - 1)) - (big - 1); };
	std::vector<sg> segs;
	std::size_t const n = std::size_t(100 + rng() % 200);
	for (std::size_t k = 0; k < n; ++k) segs.push_back({{coord(), coord()}, {coord(), coord()}});
	// a few shared endpoints and verticals
	for (std::size_t k = 0; k < 10; ++k) segs.push_back({segs[k].a, {segs[k].a.x, coord()}});
	auto const arr = arrangement_of(segs);
	point_location const slabs(arr, segs);
	atl::experimental::point_location_by_trapezoid_map const traps(arr, segs);
	std::vector<point<std::int64_t>> queries;
	for (auto const &s : segs) {
		queries.push_back(s.a);
		queries.push_back(s.b);
		queries.push_back({s.a.x, s.a.y + 1});
	}
	for (int k = 0; k < 3000; ++k) queries.push_back({coord(), coord()});
	for (auto const q : queries) REQUIRE(slabs.location_of(q) == traps.location_of(q));
}

TEST_CASE("point_location: handcrafted classifications", "[geometry]") {
	// nested squares: ring face vs inner face vs unbounded, plus boundary
	std::vector<sg> const nested{
		{{0, 0}, {8, 0}}, {{8, 0}, {8, 8}}, {{8, 8}, {0, 8}}, {{0, 8}, {0, 0}}, {{3, 3}, {5, 3}}, {{5, 3}, {5, 5}}, {{5, 5}, {3, 5}}, {{3, 5}, {3, 3}}};
	auto const arr = arrangement_of(nested);
	point_location const loc(arr, nested);
	std::uint32_t ring = arrangement::none, inner = arrangement::none;
	for (std::uint32_t f = 0; f < arr.faces.size(); ++f) {
		if (std::round(arr.faces[f].area) == 60.0) ring = f;
		if (std::round(arr.faces[f].area) == 4.0) inner = f;
	}
	REQUIRE(loc.location_of({1, 1}) == location{location_kind::face, ring});
	REQUIRE(loc.location_of({4, 4}) == location{location_kind::face, inner});
	REQUIRE(loc.location_of({9, 4})
			== location{location_kind::face, arrangement::unbounded_face});
	REQUIRE(loc.location_of({-1, -1})
			== location{location_kind::face, arrangement::unbounded_face});
	// on a horizontal edge, a vertical edge, and a vertex
	REQUIRE(loc.location_of({4, 0}).kind == location_kind::edge);
	REQUIRE(loc.location_of({0, 4}).kind == location_kind::edge);
	REQUIRE(loc.location_of({8, 8}).kind == location_kind::vertex);
	REQUIRE(loc.location_of({3, 3}).kind == location_kind::vertex);
	// x exactly on a slab boundary but strictly inside a face
	REQUIRE(loc.location_of({3, 1}) == location{location_kind::face, ring});
	REQUIRE(loc.location_of({3, 7}) == location{location_kind::face, ring});

	// trivial arrangements
	auto const empty = arrangement_of(std::vector<sg>{});
	point_location const empty_loc(empty, std::vector<sg>{});
	REQUIRE(empty_loc.location_of({5, 5})
			== location{location_kind::face, arrangement::unbounded_face});

	std::vector<sg> const lone{{{0, 0}, {4, 4}}};
	auto const lone_arr = arrangement_of(lone);
	point_location const lone_loc(lone_arr, lone);
	REQUIRE(lone_loc.location_of({2, 2}).kind == location_kind::edge);
	REQUIRE(lone_loc.location_of({0, 0}).kind == location_kind::vertex);
	REQUIRE(lone_loc.location_of({3, 1})
			== location{location_kind::face, arrangement::unbounded_face});

	std::vector<sg> const dot{{{2, 2}, {2, 2}}};
	auto const dot_arr = arrangement_of(dot);
	point_location const dot_loc(dot_arr, dot);
	REQUIRE(dot_loc.location_of({2, 2}).kind == location_kind::vertex);
	REQUIRE(dot_loc.location_of({2, 3})
			== location{location_kind::face, arrangement::unbounded_face});
}

TEST_CASE("point_location: coordinates near the bound", "[geometry]") {
	std::int64_t const big = (std::int64_t(1) << 31);
	std::vector<sg> const segs{{{-big, -big}, {big, big}}, {{-big, big}, {big, -big}}};
	auto const arr = arrangement_of(segs);
	point_location const loc(arr, segs);
	REQUIRE(loc.location_of({0, 0}).kind == location_kind::vertex);
	// the X has no bounded face: everything off the slits is unbounded
	REQUIRE(loc.location_of({1, 0})
			== location{location_kind::face, arrangement::unbounded_face});
	REQUIRE(loc.location_of({2, 2}).kind == location_kind::edge);
}

TEST_CASE("point_location: double coordinates parity spot check", "[geometry]") {
	using sd = segment<double>;
	std::vector<sd> const square{
		{{0, 0}, {4, 0}}, {{4, 0}, {4, 4}}, {{4, 4}, {0, 4}}, {{0, 4}, {0, 0}}};
	auto const arr = arrangement_of(square);
	point_location const loc(arr, square);
	auto const inside = loc.location_of({2.0, 2.0});
	REQUIRE(inside.kind == location_kind::face);
	REQUIRE(inside.index != arrangement::unbounded_face);
	REQUIRE(loc.location_of({5.0, 5.0})
			== location{location_kind::face, arrangement::unbounded_face});
	REQUIRE(loc.location_of({2.0, 0.0}).kind == location_kind::edge);
	REQUIRE(loc.location_of({4.0, 4.0}).kind == location_kind::vertex);
}

TEST_CASE("locations_of: the offline batch agrees with the online structure",
		  "[geometry]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 150; ++it) {
		std::int64_t span = 0;
		auto const segs = random_grid(rng, span);
		auto const arr = arrangement_of(segs);
		point_location const online(arr, segs);
		std::vector<pt> queries;
		for (std::int64_t x = -1; x <= span + 1; ++x)
			for (std::int64_t y = -1; y <= span + 1; ++y) queries.push_back({x, y});
		auto const batch = locations_of(arr, segs, queries);
		REQUIRE(batch.size() == queries.size());
		for (std::size_t i = 0; i < queries.size(); ++i)
			REQUIRE(batch[i] == online.location_of(queries[i]));
	}
	// trivial batches
	auto const empty = arrangement_of(std::vector<sg>{});
	REQUIRE(locations_of(empty, std::vector<sg>{}, std::vector<pt>{}).empty());
	auto const one = locations_of(empty, std::vector<sg>{}, std::vector<pt>{{3, 3}});
	REQUIRE(one == std::vector<location>{{location_kind::face, arrangement::unbounded_face}});
}

TEST_CASE("point_location: a temporary arrangement is rejected at compile time", "[geometry]") {
	using sgs = std::vector<atl::geometry::segment<int>>;
	static_assert(std::constructible_from<point_location<int>, arrangement const &, sgs const &>);
	static_assert(std::constructible_from<point_location<int>, arrangement &, sgs const &>);
	static_assert(!std::constructible_from<point_location<int>, arrangement, sgs const &>);
	static_assert(!std::constructible_from<point_location<int>, arrangement const &&, sgs const &>);
	SUCCEED();
}

TEST_CASE("point location: 100k sorted isolated vertices and duplicate points", "[geometry]") {
	std::vector<sg> segments;
	segments.reserve(100'000);
	for (std::int64_t i = 0; i < 100'000; ++i) segments.push_back({{i, 0}, {i, 0}});
	auto const arr = arrangement_of(segments);
	point_location const index(arr, segments);
	REQUIRE(arr.vertices.size() == 100'000);
	REQUIRE(index.location_of({99'999, 0}).kind == location_kind::vertex);
	REQUIRE(index.location_of({99'999, 1}).kind == location_kind::face);
	std::vector<pt> const queries{{0, 0}, {99'999, 0}, {99'999, 1}};
	auto const batch = locations_of(arr, segments, queries);
	for (std::size_t i = 0; i < queries.size(); ++i) REQUIRE(batch[i] == index.location_of(queries[i]));
	for (auto &s : segments) s = {{0, 0}, {0, 0}};
	auto const duplicate_arr = arrangement_of(segments);
	REQUIRE(duplicate_arr.vertices.size() == 1);
	REQUIRE(duplicate_arr.edge_count() == 0);
}

TEST_CASE("point location: int64 endpoints and queries at the asserted bound", "[geometry]") {
	std::int64_t const bound = (std::int64_t(1) << 32) - 1;
	std::vector<sg> const segments{{{-bound, -bound}, {bound, bound}}, {{-bound, bound}, {bound, -bound}}};
	auto const arr = arrangement_of(segments);
	point_location const index(arr, segments);
	atl::experimental::point_location_by_trapezoid_map const contender(arr, segments);
	std::vector<pt> const queries{{0, 0}, {-bound, -bound}, {bound, bound}, {0, bound}, {1, 1}};
	auto const batch = locations_of(arr, segments, queries);
	for (std::size_t i = 0; i < queries.size(); ++i) {
		auto const expected = brute(arr, segments, queries[i]);
		REQUIRE(index.location_of(queries[i]) == expected);
		REQUIRE(contender.location_of(queries[i]) == expected);
		REQUIRE(batch[i] == expected);
	}
}

TEST_CASE("point location: query bounds are asserted in both modes", "[geometry]") {
	std::vector<sg> const segments{{{0, 0}, {4, 4}}};
	auto const arr = arrangement_of(segments);
	point_location const index(arr, segments);
	REQUIRE(index.location_of({0, 0}).kind == location_kind::vertex);
	REQUIRE(atl_test::aborts([&] { (void)index.location_of({std::int64_t(1) << 32, 0}); }));
	REQUIRE(atl_test::aborts([&] {
		(void)locations_of(arr, segments, std::vector<pt>{{std::int64_t(1) << 32, 0}});
	}));
}

TEST_CASE("point location: 100k coexisting parallel edges fit a bounded call stack", "[geometry]") {
	std::vector<sg> segments;
	segments.reserve(100'000);
	for (std::int64_t i = 0; i < 100'000; ++i) segments.push_back({{0, 2 * i}, {10, 2 * i}});
	auto const arr = arrangement_of(segments);
	point_location const index(arr, segments);
	REQUIRE(index.location_of({5, 199'998}).kind == location_kind::edge);
	REQUIRE(index.location_of({5, 199'999}) == location{location_kind::face, arrangement::unbounded_face});
	std::vector<pt> const queries{{0, 0}, {5, 0}, {5, 199'998}, {5, 199'999}};
	auto const batch = locations_of(arr, segments, queries);
	for (std::size_t i = 0; i < queries.size(); ++i) REQUIRE(batch[i] == index.location_of(queries[i]));
}

TEST_CASE("point location: floating small grids agree with exact rational classification", "[geometry]") {
	std::mt19937_64 rng(498);
	for (int iteration = 0; iteration < 20; ++iteration) {
		std::int64_t span = 0;
		auto const integer_segments = random_grid(rng, span);
		auto const integer_arr = arrangement_of(integer_segments);
		auto const check = [&]<atl::geometry::coordinate T>()
			requires std::floating_point<T>
		{
			std::vector<segment<T>> segments;
			for (auto const &s : integer_segments) segments.push_back({{T(s.a.x), T(s.a.y)}, {T(s.b.x), T(s.b.y)}});
			auto const arr = arrangement_of(segments);
			REQUIRE(arr.origin == integer_arr.origin);
			REQUIRE(arr.face == integer_arr.face);
			point_location const index(arr, segments);
			std::vector<point<T>> queries;
			for (std::int64_t x = -1; x <= span + 1; ++x)
				for (std::int64_t y = -1; y <= span + 1; ++y) {
					queries.push_back({T(x), T(y)});
					REQUIRE(index.location_of(queries.back()) == brute(integer_arr, integer_segments, {x, y}));
				}
			auto const batch = locations_of(arr, segments, queries);
			for (std::size_t i = 0; i < queries.size(); ++i) REQUIRE(batch[i] == index.location_of(queries[i]));
		};
		check.template operator()<float>();
		check.template operator()<double>();
	}
}
