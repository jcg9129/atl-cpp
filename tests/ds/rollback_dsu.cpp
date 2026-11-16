#include <atl/ds/persistent_dsu.hpp>
#include <atl/ds/rollback_dsu.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

using atl::uint32;
using atl::uint64;

TEST_CASE("rollback_dsu: basic unite / undo / rollback", "[rollback_dsu]") {
	atl::rollback_dsu d(5);
	REQUIRE(d.elements() == 5);
	REQUIRE(d.count() == 5);
	REQUIRE(!d.same(0, 1));

	REQUIRE(d.unite(0, 1));
	REQUIRE(d.same(0, 1));
	REQUIRE(d.size(0) == 2);
	REQUIRE(d.count() == 4);
	REQUIRE(!d.unite(0, 1));
	REQUIRE(d.time() == 1);

	std::size_t const t = d.time();
	REQUIRE(d.unite(2, 3));
	REQUIRE(d.unite(1, 2));
	REQUIRE(d.same(0, 3));
	REQUIRE(d.size(0) == 4);
	REQUIRE(d.count() == 2);

	d.rollback(t);
	REQUIRE(d.time() == t);
	REQUIRE(d.count() == 4);
	REQUIRE(d.same(0, 1));
	REQUIRE(!d.same(2, 3));
	REQUIRE(!d.same(0, 3));

	d.undo();
	REQUIRE(d.time() == 0);
	REQUIRE(d.count() == 5);
	REQUIRE(!d.same(0, 1));
}

// The persistent DSU is validated against a reference elsewhere, so cross-checking
// rollback_dsu against it transitively validates rollback_dsu. We drive both with
// the same stream of merges, and undo on rollback_dsu mirrors dropping the newest
// persistent version — the two must stay in lock-step across the whole history.
TEST_CASE("rollback_dsu: matches persistent_dsu over a linear history with undo", "[rollback_dsu]") {
	auto seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	uint32 const n = 40;

	atl::rollback_dsu rb(n);
	std::vector<atl::persistent_dsu<>> versions; // versions[i] == state after i live merges
	versions.emplace_back(n);

	auto check = [&] {
		auto const &p = versions.back();
		REQUIRE(rb.count() == p.count());
		REQUIRE(rb.time() == versions.size() - 1);
		for (int t = 0; t < 20; ++t) {
			uint32 a = uint32(rng() % n), b = uint32(rng() % n);
			REQUIRE(rb.same(a, b) == p.same(a, b));
		}
		for (int t = 0; t < 10; ++t) {
			uint32 x = uint32(rng() % n);
			REQUIRE(rb.size(x) == p.size(x));
		}
	};

	check();
	for (int step = 0; step < 3000; ++step) {
		if (rb.time() > 0 && rng() % 3 == 0) {
			rb.undo();
			versions.pop_back();
		} else {
			uint32 a = uint32(rng() % n), b = uint32(rng() % n);
			// Push a new persistent version only when a real merge happened, so both
			// sides agree on how many merges are live (rb records only real merges).
			if (rb.unite(a, b)) versions.push_back(versions.back().unite(a, b));
		}
		check();
	}
}

TEST_CASE("rollback_dsu: bulk rollback returns to an exact earlier state", "[rollback_dsu]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	uint32 const n = 60;

	atl::rollback_dsu d(n);
	for (int warmup = 0; warmup < 30; ++warmup) d.unite(uint32(rng() % n), uint32(rng() % n));

	std::size_t const t = d.time();
	std::size_t const count_at_t = d.count();
	std::vector<std::pair<uint32, uint32>> probes;
	for (int i = 0; i < 50; ++i) {
		uint32 a = uint32(rng() % n), b = uint32(rng() % n);
		probes.push_back({a, b});
	}
	std::vector<bool> connected;
	for (auto const &[a, b] : probes) connected.push_back(d.same(a, b));

	for (int more = 0; more < 40; ++more) d.unite(uint32(rng() % n), uint32(rng() % n));
	d.rollback(t);

	REQUIRE(d.time() == t);
	REQUIRE(d.count() == count_at_t);
	for (std::size_t i = 0; i < probes.size(); ++i) REQUIRE(d.same(probes[i].first, probes[i].second) == connected[i]);
}

// find directly (roots are stable across undo), and the empty structures.
TEST_CASE("rollback_dsu: find, default-constructed and empty", "[rollback_dsu]") {
	atl::rollback_dsu d(6);
	d.unite(0, 1);
	d.unite(2, 3);
	std::size_t const r01 = d.find(0);
	REQUIRE((r01 == 0 || r01 == 1));
	REQUIRE(d.find(1) == r01);
	REQUIRE(d.find(r01) == r01);
	auto const t = d.time();
	d.unite(1, 3);
	REQUIRE(d.find(2) == d.find(0));
	REQUIRE(d.size(2) == 4);
	d.undo();
	REQUIRE(d.time() == t);
	REQUIRE(d.find(0) == r01);
	REQUIRE(d.find(2) != r01);
	REQUIRE(d.size(3) == 2);
	REQUIRE(d.count() == 4);

	atl::rollback_dsu empty;
	REQUIRE(empty.elements() == 0);
	REQUIRE(empty.count() == 0);
	REQUIRE(empty.time() == 0);
	atl::rollback_dsu zero(0);
	REQUIRE(zero.elements() == 0);
	REQUIRE(zero.count() == 0);
}
