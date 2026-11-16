#include <atl/ds/persistent_dsu.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/int128.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <vector>

using atl::int32;
using atl::uint32;
using atl::uint64;

static_assert(atl::detail::persistent_signed_array<atl::persistent_array<int32>>);
static_assert(atl::detail::persistent_signed_array<atl::semi_persistent_array<int32>>);
static_assert(atl::detail::persistent_signed_array<atl::persistent_array<atl::int64>>);
static_assert(!atl::detail::persistent_signed_array<atl::persistent_array<uint32>>);
static_assert(!atl::detail::persistent_signed_array<std::vector<int32>>);

namespace {
	// a backing complete but for default construction (which the empty DSU needs)
	template<bool IsDefaultConstructible>
	struct mock_backing {
		mock_backing()
			requires IsDefaultConstructible
		= default;

		mock_backing(std::size_t, int32) {}

		int32 const &operator[](std::size_t) const { return value; }

		std::size_t size() const { return 0; }

		mock_backing set(std::size_t, int32) const { return *this; }

		int32 value = 0;
	};
}

static_assert(atl::detail::persistent_signed_array<mock_backing<true>>);
static_assert(!atl::detail::persistent_signed_array<mock_backing<false>>);

TEST_CASE("persistent_dsu: basic full persistence", "[persistent_dsu]") {
	atl::persistent_dsu<> d(5);
	REQUIRE(d.count() == 5);
	REQUIRE(!d.same(0, 1));

	auto a = d.unite(0, 1);
	auto b = a.unite(2, 3);
	auto c = a.unite(1, 4); // branch off `a`, not `b`

	REQUIRE(d.count() == 5);
	REQUIRE(!d.same(0, 1));

	REQUIRE(a.same(0, 1));
	REQUIRE(a.count() == 4);

	REQUIRE(b.same(2, 3));
	REQUIRE(b.count() == 3);
	REQUIRE(!b.same(0, 4));

	REQUIRE(c.same(0, 4));
	REQUIRE(c.size(0) == 3);
	REQUIRE(!c.same(2, 3)); // c did not get b's merge
	REQUIRE(c.count() == 3);

	REQUIRE(d.unite(0, 0).count() == 5); // uniting an already-joined pair is a no-op
}

namespace {
	// Reference component labels: comp[i] is i's set id; merging b's set into a's.
	uint32 distinct_labels(std::vector<int> const &comp) {
		return uint32(std::set<int>(comp.begin(), comp.end()).size());
	}

	void ref_unite(std::vector<int> &comp, uint32 a, uint32 b) {
		int ca = comp[a], cb = comp[b];
		if (ca == cb) return;
		for (int &x : comp)
			if (x == cb) x = ca;
	}

	void check_version(auto const &dsu, std::vector<int> const &comp, std::mt19937_64 &rng, uint32 n) {
		REQUIRE(dsu.count() == distinct_labels(comp));
		for (int t = 0; t < 20; ++t) {
			uint32 a = uint32(rng() % n), b = uint32(rng() % n);
			REQUIRE(dsu.same(a, b) == (comp[a] == comp[b]));
		}
		for (int t = 0; t < 10; ++t) {
			uint32 x = uint32(rng() % n);
			REQUIRE(dsu.size(x) == uint32(std::count(comp.begin(), comp.end(), comp[x])));
		}
	}
}

TEST_CASE("persistent_dsu: full persistence vs reference (branch off any version)", "[persistent_dsu]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	uint32 const n = 40;

	std::vector<atl::persistent_dsu<>> vs;
	std::vector<std::vector<int>> comp;
	vs.emplace_back(n);
	{
		std::vector<int> c(n);
		std::iota(c.begin(), c.end(), 0);
		comp.push_back(c);
	}

	for (int step = 0; step < 1200; ++step) {
		std::size_t p = rng() % vs.size(); // branch off an arbitrary past version
		uint32 a = uint32(rng() % n), b = uint32(rng() % n);
		vs.push_back(vs[p].unite(a, b));
		auto c = comp[p];
		ref_unite(c, a, b);
		comp.push_back(c);
	}

	for (std::size_t j = 0; j < vs.size(); ++j) check_version(vs[j], comp[j], rng, n);
}

TEST_CASE("persistent_dsu: partial persistence, linear history", "[persistent_dsu]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	uint32 const n = 40;
	using dsu = atl::persistent_dsu<atl::semi_persistent_array<int32>>;

	std::vector<dsu> vs;
	std::vector<std::vector<int>> comp;
	vs.emplace_back(n);
	{
		std::vector<int> c(n);
		std::iota(c.begin(), c.end(), 0);
		comp.push_back(c);
	}

	auto cur = vs.front();
	auto curc = comp.front();
	for (int step = 0; step < 1500; ++step) {
		uint32 a = uint32(rng() % n), b = uint32(rng() % n);
		cur = cur.unite(a, b); // only ever extend the newest version
		ref_unite(curc, a, b);
		vs.push_back(cur);
		comp.push_back(curc);
	}

	for (std::size_t j = 0; j < vs.size(); ++j) check_version(vs[j], comp[j], rng, n);
}

TEMPLATE_TEST_CASE("persistent_dsu: rvalue unite matches the lvalue one and leaves other versions intact", "[persistent_dsu]",
				   atl::persistent_array<int32>, atl::semi_persistent_array<int32>) {
	using dsu = atl::persistent_dsu<TestType>;
	static_assert(requires { static_cast<dsu (dsu::*)(std::size_t, std::size_t) const &>(&dsu::unite); });
	static_assert(requires { static_cast<dsu (dsu::*)(std::size_t, std::size_t) &&>(&dsu::unite); });

	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + rng() % 60;
	auto same = [n](dsu const &x, dsu const &y) {
		if (x.count() != y.count()) return false;
		for (std::size_t i = 0; i < n; ++i)
			for (std::size_t j = i; j < n; ++j)
				if (x.same(i, j) != y.same(i, j)) return false;
		return true;
	};
	dsu moved(n), copied(n);
	std::vector<std::pair<dsu, dsu>> kept; // (snapshot, lvalue-built twin): the snapshot must not change
	for (int step = 0; step < 200; ++step) {
		if (rng() % 10 == 0) kept.emplace_back(moved, copied);
		std::size_t const a = rng() % n, b = rng() % n;
		moved = std::move(moved).unite(a, b);
		copied = copied.unite(a, b);
		REQUIRE(same(moved, copied));
	}
	for (auto const &[snap, twin] : kept) REQUIRE(same(snap, twin));
	dsu a(n);
	auto b = std::move(a).unite(0, 1);
	a = b;
	REQUIRE(a.same(0, 1));
	auto c = std::move(a).unite(0, 1); // already joined: still a valid version
	REQUIRE(c.count() == n - 1);
}

TEST_CASE("persistent_dsu: published 128-bit backing", "[persistent_dsu]") {
	atl::persistent_dsu<atl::persistent_array<atl::int128>> const base(4);
	auto const left = base.unite(0, 1).unite(2, 3);
	auto const right = base.unite(0, 2);
	REQUIRE(left.count() == 2);
	REQUIRE(left.size(0) == 2);
	REQUIRE(!left.same(0, 2));
	REQUIRE(right.same(0, 2));
	REQUIRE(base.count() == 4);
}
