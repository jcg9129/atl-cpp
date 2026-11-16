#include <atl/ds/hierarchical_bitset.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <vector>
#include "../expect_abort.hpp"

using atl::hierarchical_bitset;
using atl::uint64;

TEST_CASE("hierarchical_bitset: matches std::set under random ops", "[hierarchical_bitset]") {
	auto seed = GENERATE(take(8, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 200000); // spans up to ~3 levels
	hierarchical_bitset t(n);
	std::set<std::size_t> ref;
	for (int i = 0; i < 20000; ++i) {
		std::size_t const x = std::size_t(rng() % n);
		switch (rng() % 5) {
		case 0:
			t.insert(x);
			ref.insert(x);
			break;
		case 1:
			t.erase(x);
			ref.erase(x);
			break;
		case 2:
			REQUIRE(t.contains(x) == (ref.count(x) > 0));
			break;
		case 3: {
			auto it = ref.lower_bound(x);
			REQUIRE(t.next(x) == (it == ref.end() ? hierarchical_bitset::npos : *it));
			break;
		}
		default: {
			auto it = ref.upper_bound(x);
			REQUIRE(t.prev(x) == (it == ref.begin() ? hierarchical_bitset::npos : *std::prev(it)));
		}
		}
		REQUIRE(t.size() == ref.size());
	}
	REQUIRE(t.min() == (ref.empty() ? hierarchical_bitset::npos : *ref.begin()));
	REQUIRE(t.max() == (ref.empty() ? hierarchical_bitset::npos : *ref.rbegin()));
}

TEST_CASE("hierarchical_bitset: basics and edge cases", "[hierarchical_bitset]") {
	hierarchical_bitset t(100);
	REQUIRE(t.empty());
	REQUIRE(t.universe() == 100);
	REQUIRE(t.min() == hierarchical_bitset::npos);
	REQUIRE(t.max() == hierarchical_bitset::npos);
	REQUIRE(t.next(0) == hierarchical_bitset::npos);
	REQUIRE(t.prev(99) == hierarchical_bitset::npos);

	t.insert(50);
	REQUIRE(t.contains(50));
	REQUIRE(t.size() == 1);
	t.insert(50);
	REQUIRE(t.size() == 1);
	REQUIRE(t.next(50) == 50);
	REQUIRE(t.next(51) == hierarchical_bitset::npos);
	REQUIRE(t.prev(50) == 50);
	REQUIRE(t.prev(49) == hierarchical_bitset::npos);
	REQUIRE(t.next(0) == 50);
	REQUIRE(t.prev(99) == 50);
	REQUIRE(t.min() == 50);
	REQUIRE(t.max() == 50);

	t.erase(49);
	REQUIRE(t.size() == 1);
	t.erase(50);
	REQUIRE(t.empty());

	t.insert(0);
	t.insert(99);
	REQUIRE(t.min() == 0);
	REQUIRE(t.max() == 99);
	REQUIRE(t.next(1) == 99);
	REQUIRE(t.prev(98) == 0);

	t.clear();
	REQUIRE(t.empty());
	REQUIRE_FALSE(t.contains(0));
	REQUIRE_FALSE(t.contains(99));
}

TEST_CASE("hierarchical_bitset: bulk range constructor matches element-by-element", "[hierarchical_bitset]") {
	auto seed = GENERATE(take(4, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 200000);
	std::vector<std::size_t> present;
	hierarchical_bitset one_by_one(n);
	for (int i = 0; i < 3000; ++i) {
		std::size_t const x = std::size_t(rng() % n);
		if (!one_by_one.contains(x)) {
			one_by_one.insert(x);
			present.push_back(x);
		}
	}
	std::vector<std::size_t> with_dups = present;
	if (!present.empty()) with_dups.push_back(present[std::size_t(rng() % present.size())]);

	hierarchical_bitset bulk(n, with_dups);
	REQUIRE(bulk.size() == one_by_one.size());
	for (int i = 0; i < 5000; ++i) {
		std::size_t const x = std::size_t(rng() % n);
		REQUIRE(bulk.contains(x) == one_by_one.contains(x));
		REQUIRE(bulk.next(x) == one_by_one.next(x));
		REQUIRE(bulk.prev(x) == one_by_one.prev(x));
	}
}

TEST_CASE("hierarchical_bitset: forward iteration via next", "[hierarchical_bitset]") {
	hierarchical_bitset t(1000);
	for (std::size_t x : {3, 17, 64, 65, 500, 999}) t.insert(x);
	std::vector<std::size_t> got;
	for (std::size_t x = t.min(); x != hierarchical_bitset::npos; x = t.next(x + 1)) // no guard at the last element
		got.push_back(x);
	REQUIRE(got == std::vector<std::size_t>{3, 17, 64, 65, 500, 999});
}

TEST_CASE("hierarchical_bitset: next(universe()) is npos", "[hierarchical_bitset]") {
	for (std::size_t universe : {1, 64, 65, 4096, 4097}) {
		hierarchical_bitset t(universe);
		REQUIRE(t.next(universe) == hierarchical_bitset::npos);
		t.insert(universe - 1);
		REQUIRE(t.next(universe - 1) == universe - 1);
		REQUIRE(t.next(universe) == hierarchical_bitset::npos);
		CHECK(atl_test::aborts([&] { (void)t.next(universe + 1); }));
	}
}

TEST_CASE("hierarchical_bitset: zero universe", "[hierarchical_bitset]") {
	atl::hierarchical_bitset set(0);
	REQUIRE(set.empty());
	REQUIRE(set.next(0) == atl::hierarchical_bitset::npos);
	REQUIRE(set.min() == atl::hierarchical_bitset::npos);
	REQUIRE(set.max() == atl::hierarchical_bitset::npos);
	set.clear();
	static_assert([] { atl::hierarchical_bitset empty(0); return empty.next(0) == atl::hierarchical_bitset::npos; }());
}
