#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/convex_layers.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <vector>

using atl::uint32;
using atl::geometry::containment;
using atl::geometry::contains;
using atl::geometry::convex_hull;
using atl::geometry::convex_layers;
using atl::geometry::keep_collinear;
using atl::geometry::point;

namespace {
	using lp = point<std::int64_t>;

	// A position is on the boundary iff a line through it and another
	// distinct position supports the entire remainder on one closed side.
	std::vector<std::vector<std::size_t>> brute_layers(std::vector<lp> const &ps) {
		std::vector<std::size_t> remaining(ps.size());
		for (std::size_t i = 0; i < remaining.size(); ++i) remaining[i] = i;
		std::vector<std::vector<std::size_t>> layers;
		while (!remaining.empty()) {
			auto &layer = layers.emplace_back();
			std::vector<std::size_t> rest;
			for (auto const i : remaining) {
				bool is_boundary = false, has_distinct = false;
				for (auto const j : remaining) {
					if (ps[i] == ps[j]) continue;
					has_distinct = true;
					bool has_left = false, has_right = false;
					for (auto const k : remaining) {
						auto const dx = atl::int128(ps[j].x) - ps[i].x, dy = atl::int128(ps[j].y) - ps[i].y;
						auto const determinant = dx * (atl::int128(ps[k].y) - ps[i].y) - dy * (atl::int128(ps[k].x) - ps[i].x);
						has_left |= determinant > 0;
						has_right |= determinant < 0;
					}
					is_boundary |= !has_left || !has_right;
				}
				(is_boundary || !has_distinct ? layer : rest).push_back(i);
			}
			remaining = std::move(rest);
		}
		return layers;
	}

	void check_layers(std::vector<lp> const &ps) {
		auto const layers = convex_layers(ps);
		auto const brute = brute_layers(ps);
		// same partition, layer by layer (membership, not walk order)
		REQUIRE(layers.size() == brute.size());
		std::set<std::size_t> seen;
		for (std::size_t k = 0; k < layers.size(); ++k) {
			std::vector<std::size_t> got(layers[k].begin(), layers[k].end());
			std::ranges::sort(got);
			REQUIRE(got == brute[k]);
			for (std::size_t const i : got) REQUIRE(seen.insert(i).second);
		}
		REQUIRE(seen.size() == ps.size());
		// interiority: deeper points lie strictly inside each layer's hull
		// (whenever that hull has area)
		for (std::size_t k = 0; k < layers.size(); ++k) {
			std::vector<lp> remainder;
			for (std::size_t j = k; j < layers.size(); ++j)
				for (auto const i : layers[j]) remainder.push_back(ps[i]);
			auto const hull = convex_hull(remainder);
			if (hull.size() < 3) continue;
			for (auto const i : layers[k])
				REQUIRE(contains(hull, ps[i]) == containment::boundary);
			for (std::size_t j = k + 1; j < layers.size(); ++j)
				for (auto const i : layers[j])
					REQUIRE(contains(hull, ps[i]) == containment::inside);
		}
	}
}

TEST_CASE("convex_layers: handcrafted", "[geometry]") {
	REQUIRE(convex_layers(std::vector<lp>{}).empty());
	REQUIRE(convex_layers(std::vector<lp>{{5, 5}}).size() == 1);

	auto const same = convex_layers(std::vector<lp>{{2, 2}, {2, 2}, {2, 2}});
	REQUIRE(same.size() == 1);
	REQUIRE(same[0] == std::vector<uint32>{0, 1, 2});

	auto const sq = convex_layers(
		std::vector<lp>{{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}});
	REQUIRE(sq.size() == 2);
	REQUIRE(sq[1] == std::vector<uint32>{4});

	std::vector<lp> nested;
	for (std::int64_t r : {30, 20, 10})
		for (auto const &[sx, sy] :
			 {std::pair{-1, -1}, {1, -1}, {1, 1}, {-1, 1}})
			nested.push_back({sx * r, sy * r});
	auto const rings = convex_layers(nested);
	REQUIRE(rings.size() == 3);
	for (auto const &ring : rings) REQUIRE(ring.size() == 4);

	auto const edge = convex_layers(
		std::vector<lp>{{0, 0}, {10, 0}, {5, 10}, {5, 0}, {5, 3}});
	REQUIRE(edge.size() == 2);
	REQUIRE(edge[1] == std::vector<uint32>{4});

	auto const line = convex_layers(
		std::vector<lp>{{7, 0}, {1, 0}, {5, 0}, {3, 0}});
	REQUIRE(line.size() == 1);
	REQUIRE(line[0] == std::vector<uint32>{1, 3, 2, 0});

	auto const walk = convex_layers(std::vector<lp>{{4, 4}, {0, 0}, {4, 0}, {0, 4}});
	REQUIRE(walk.size() == 1);
	REQUIRE(walk[0] == std::vector<uint32>{1, 2, 0, 3});
}

TEST_CASE("convex_layers: random vs supporting-line brute", "[geometry]") {
	auto seed = GENERATE(take(10, random(std::uint64_t(0),
										 (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	// tiny dense grids: duplicates, collinear runs, degenerate remainders
	for (int it = 0; it < 60; ++it) {
		std::size_t const n = std::size_t(rng() % 41);
		std::int64_t const span = 1 + std::int64_t(rng() % 8);
		std::vector<lp> ps;
		for (std::size_t k = 0; k < n; ++k)
			ps.push_back({std::int64_t(rng() % std::uint64_t(span)),
						  std::int64_t(rng() % std::uint64_t(span))});
		check_layers(ps);
	}
	// scattered clouds: many thin layers
	for (int it = 0; it < 20; ++it) {
		std::size_t const n = 3 + std::size_t(rng() % 60);
		std::vector<lp> ps;
		for (std::size_t k = 0; k < n; ++k)
			ps.push_back({std::int64_t(rng() % 1000) - 500,
						  std::int64_t(rng() % 1000) - 500});
		check_layers(ps);
	}
}

TEST_CASE("convex_layers: double coordinates parity spot check", "[geometry]") {
	using pd = point<double>;
	auto const layers = convex_layers(std::vector<pd>{
		{0, 0}, {3, 0}, {3, 3}, {0, 3}, {1.5, 1.5}, {1.25, 1.5}});
	REQUIRE(layers.size() == 2);
	REQUIRE(layers[0].size() == 4);
	REQUIRE(layers[1].size() == 2);
}

TEST_CASE("convex_layers under approx: slots by exact position, a partition", "[geometry]") {
	using dp = point<double>;
	// near-duplicates 1e-12 apart in x, separated in lex order by the
	// column above them, plus exact duplicates
	std::vector<dp> ps;
	for (int x = 0; x < 5; ++x)
		for (int y = 0; y < 5; ++y) {
			ps.push_back({double(x), double(y)});
			ps.push_back({double(x) + 1e-12, double(y)});
		}
	auto const duplicate_a = ps[3], duplicate_b = ps[17];
	ps.push_back(duplicate_a);
	ps.push_back(duplicate_b);
	auto const layers = convex_layers(ps, atl::geometry::approx{1e-9});
	std::vector<int> seen(ps.size(), 0);
	for (auto const &layer : layers)
		for (uint32 const i : layer) ++seen[i];
	REQUIRE(std::ranges::all_of(seen, [](int const c) { return c == 1; }));
	// exact duplicates peel together
	for (auto const &layer : layers) {
		bool const has3 = std::ranges::find(layer, uint32(3)) != layer.end();
		REQUIRE(has3 == (std::ranges::find(layer, uint32(ps.size() - 2)) != layer.end()));
	}
}

TEST_CASE("convex_layers: integral limits and dyadic floating scales", "[geometry]") {
	std::int64_t const limit = (std::int64_t(1) << 62) - 1;
	std::vector<lp> const points{{-limit, -limit}, {limit, -limit}, {limit, limit}, {-limit, limit}, {0, 0}, {0, -limit}, {0, 0}};
	check_layers(points);
	for (int exponent : {-500, 0, 500}) {
		double const scale = std::ldexp(1.0, exponent);
		std::vector<point<double>> scaled;
		for (auto const q : std::vector<lp>{{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}, {2, 0}})
			scaled.push_back({double(q.x) * scale, double(q.y) * scale});
		auto const layers = convex_layers(scaled);
		REQUIRE(layers == std::vector<std::vector<uint32>>{{0, 5, 1, 2, 3}, {4}});
	}
}
