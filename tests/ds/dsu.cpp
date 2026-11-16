#include <atl/ds/dsu.hpp>
#include <atl/ds/rollback_dsu.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

using atl::uint32;
using atl::uint64;

TEST_CASE("dsu: basic", "[dsu]") {
	atl::dsu d(5);
	REQUIRE(d.elements() == 5);
	REQUIRE(d.count() == 5);
	REQUIRE(!d.same(0, 1));

	REQUIRE(d.unite(0, 1));
	REQUIRE(d.same(0, 1));
	REQUIRE(d.size(0) == 2);
	REQUIRE(d.count() == 4);
	REQUIRE(!d.unite(0, 1));

	REQUIRE(d.unite(2, 3));
	REQUIRE(d.unite(1, 2));
	REQUIRE(d.same(0, 3));
	REQUIRE(d.size(0) == 4);
	REQUIRE(d.count() == 2);
}

// Same union-by-size semantics as rollback_dsu (only the internal representation and
// path compression differ), so connectivity, set sizes, and component count must
// agree operation-for-operation on any stream of unions and queries.
TEST_CASE("dsu: matches rollback_dsu over random unions", "[dsu]") {
	auto seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	uint32 const n = 50;

	atl::dsu a(n);
	atl::rollback_dsu b(n);
	for (int step = 0; step < 3000; ++step) {
		uint32 x = uint32(rng() % n), y = uint32(rng() % n);
		if (rng() % 2) {
			REQUIRE(a.unite(x, y) == b.unite(x, y));
		} else {
			REQUIRE(a.same(x, y) == b.same(x, y));
			REQUIRE(a.size(x) == b.size(x));
			REQUIRE(a.count() == b.count());
		}
	}
}

// an independent oracle, explicit labels relabelled on every merge.
TEST_CASE("dsu: matches a brute-force label oracle; find is idempotent", "[dsu]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 300;
	atl::dsu d(n);
	std::vector<std::size_t> label(n);
	for (std::size_t i = 0; i < n; ++i) label[i] = i;
	std::size_t sets = n;
	for (int step = 0; step < 3000; ++step) {
		std::size_t const a = rng() % n, b = rng() % n;
		if (rng() % 2) {
			bool const merged = label[a] != label[b];
			REQUIRE(d.unite(a, b) == merged);
			if (merged) {
				std::size_t const from = label[b], to = label[a];
				for (auto &l : label)
					if (l == from) l = to;
				--sets;
			}
		} else {
			REQUIRE(d.same(a, b) == (label[a] == label[b]));
			std::size_t const r = d.find(a);
			REQUIRE(d.find(r) == r);
			REQUIRE(d.find(a) == r);
			REQUIRE(label[r] == label[a]);
			REQUIRE(d.size(a) == std::size_t(std::ranges::count(label, label[a])));
		}
		REQUIRE(d.count() == sets);
	}
}

TEST_CASE("dsu: default-constructed and empty", "[dsu]") {
	atl::dsu d;
	REQUIRE(d.elements() == 0);
	REQUIRE(d.count() == 0);
	atl::dsu e(0);
	REQUIRE(e.elements() == 0);
	REQUIRE(e.count() == 0);
}
