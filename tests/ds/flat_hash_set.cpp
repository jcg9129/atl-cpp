#include <atl/ds/flat_hash_set.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include "../expect_abort.hpp"
#include "move_probe.hpp"

using atl::uint64;

TEST_CASE("flat_hash_set: matches std::set under random ops", "[flat_hash_set]") {
	auto seed = GENERATE(take(4, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::set<uint64> ref;
	atl::flat_hash_set<uint64> s;
	for (int i = 0; i < 100000; ++i) {
		uint64 const k = rng() % 3000;
		switch (rng() % 3) {
		case 0:
			REQUIRE(s.insert(k) == ref.insert(k).second);
			break;
		case 1:
			REQUIRE(s.erase(k) == (ref.erase(k) > 0));
			break;
		default:
			REQUIRE(s.contains(k) == (ref.count(k) > 0));
		}
		REQUIRE(s.size() == ref.size());
	}
	std::set<uint64> const got(s.begin(), s.end());
	REQUIRE(got == ref);
}

TEST_CASE("flat_hash_set: basic operations", "[flat_hash_set]") {
	atl::flat_hash_set<int> s;
	REQUIRE(s.empty());
	REQUIRE(s.insert(5));
	REQUIRE_FALSE(s.insert(5));
	REQUIRE(s.contains(5));
	REQUIRE(s.count(5) == 1);
	REQUIRE(s.count(6) == 0);
	REQUIRE(s.find(5) != s.end());
	REQUIRE(s.find(6) == s.end());
	REQUIRE(s.size() == 1);
	REQUIRE(s.erase(5));
	REQUIRE_FALSE(s.erase(5));
	REQUIRE(s.empty());
}

TEST_CASE("flat_hash_set: clear keeps capacity", "[flat_hash_set]") {
	atl::flat_hash_set<int> s(1000);
	std::size_t const cap = s.capacity();
	for (int i = 0; i < 500; ++i) s.insert(i);
	s.clear();
	REQUIRE(s.empty());
	REQUIRE(s.capacity() == cap); // no reallocation
	REQUIRE_FALSE(s.contains(3));
	s.insert(3);
	REQUIRE(s.contains(3));
}

TEST_CASE("flat_hash_set: grows without losing elements", "[flat_hash_set]") {
	atl::flat_hash_set<int> s;
	for (int i = 0; i < 10000; ++i) REQUIRE(s.insert(i));
	REQUIRE(s.size() == 10000);
	for (int i = 0; i < 10000; ++i) REQUIRE(s.contains(i));
	REQUIRE_FALSE(s.contains(10000));
}

TEST_CASE("flat_hash_set: string keys", "[flat_hash_set]") {
	atl::flat_hash_set<std::string> s;
	s.insert("hello");
	s.insert("world");
	s.insert("hello");
	REQUIRE(s.size() == 2);
	REQUIRE(s.contains("hello"));
	REQUIRE_FALSE(s.contains("nope"));
}

TEST_CASE("flat_hash_set: rvalue keys are moved in, never copied", "[flat_hash_set]") {
	using atl_test::move_probe;

	struct probe_hash {
		std::size_t operator()(move_probe const &p) const noexcept { return std::size_t(atl::uint64(p.v) * 0x9e3779b97f4a7c15ULL); }
	};

	struct probe_eq {
		bool operator()(move_probe const &a, move_probe const &b) const { return a.v == b.v; }
	};

	using probe_set = atl::flat_hash_set<move_probe, probe_hash, probe_eq>;

	{
		probe_set s(8);
		move_probe lv(move_probe::sentinel);
		move_probe::reset();
		REQUIRE(s.insert(lv));
		REQUIRE(move_probe::sentinel_copies == 1);
		move_probe::reset();
		REQUIRE_FALSE(s.insert(lv)); // duplicate: compared, never copied
		REQUIRE(move_probe::sentinel_copies == 0);
	}
	{
		probe_set s(8);
		move_probe::reset();
		REQUIRE(s.insert(move_probe(move_probe::sentinel)));
		REQUIRE(move_probe::sentinel_copies == 0);
	}
	{
		probe_set s(8);
		move_probe::reset();
		REQUIRE(s.emplace(move_probe::sentinel));
		REQUIRE(move_probe::sentinel_copies == 0);
	}
}

TEST_CASE("flat_hash_set: range constructor and insert", "[flat_hash_set]") {
	using atl_test::move_probe;
	std::vector<int> const values{3, 1, 3, 2};
	atl::flat_hash_set<int> s(values);
	REQUIRE(s.size() == 3);
	s.insert(std::views::iota(0, 10));
	REQUIRE(s.size() == 10);
	REQUIRE(s.contains(9));

	struct probe_hash {
		std::size_t operator()(move_probe const &p) const noexcept { return std::size_t(atl::uint64(p.v) * 0x9e3779b97f4a7c15ULL); }
	};

	struct probe_eq {
		bool operator()(move_probe const &a, move_probe const &b) const { return a.v == b.v; }
	};

	std::vector<move_probe> probes;
	probes.emplace_back(move_probe::sentinel);
	move_probe::reset();
	atl::flat_hash_set<move_probe, probe_hash, probe_eq> copied(probes | std::views::all);
	REQUIRE(move_probe::sentinel_copies == 1); // a view: the caller's elements survive
	REQUIRE(probes[0].v == move_probe::sentinel);
	move_probe::reset();
	atl::flat_hash_set<move_probe, probe_hash, probe_eq> moved(std::move(probes));
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(moved.contains(move_probe(move_probe::sentinel)));

	atl::flat_hash_set<std::string> words;
	words.insert(std::vector<std::string>{"a", "b", "a"});
	REQUIRE(words.size() == 2);
	words.insert(std::string("c"));
	REQUIRE(words.size() == 3);
}

// reserve(n) is room for n elements: n inserts never rehash, for n at the 0.75
TEST_CASE("flat_hash_set: reserve(n) then n inserts never rehashes", "[flat_hash_set]") {
	for (std::size_t const n : {std::size_t(1), std::size_t(3), std::size_t(5), std::size_t(6), std::size_t(12),
								std::size_t(1000), std::size_t(786432)}) {
		atl::flat_hash_set<uint64> s;
		s.reserve(n);
		std::size_t const cap = s.capacity();
		for (std::size_t i = 0; i < n; ++i) s.insert(uint64(i) * 7919);
		REQUIRE(s.size() == n);
		REQUIRE(s.capacity() == cap);
	}
}

// live window never grows the table.
TEST_CASE("flat_hash_set: churn under a reservation keeps the capacity", "[flat_hash_set]") {
	atl::flat_hash_set<uint64> s(1'000'000);
	std::size_t const cap = s.capacity();
	for (uint64 i = 0; i < 3'000'000; ++i) {
		s.insert(i);
		if (i >= 100) REQUIRE(s.erase(i - 100));
	}
	REQUIRE(s.size() == 100);
	REQUIRE(s.capacity() == cap);
	for (uint64 i = 3'000'000 - 100; i < 3'000'000; ++i) REQUIRE(s.contains(i));
}

// inserting a key already present never rehashes, even at the growth threshold.
TEST_CASE("flat_hash_set: a duplicate insert at the threshold does not grow", "[flat_hash_set]") {
	atl::flat_hash_set<uint64> s;
	s.reserve(6);
	std::size_t const cap = s.capacity();
	std::size_t n = 0;
	while ((n + 1) * 4 < cap * 3) s.insert(uint64(n++));
	REQUIRE_FALSE(s.insert(uint64(0)));
	REQUIRE(s.capacity() == cap);
}

// structured keys: every key's low 32 bits are zero, which would put all of them on one
TEST_CASE("flat_hash_set: std::hash on packed pair keys", "[flat_hash_set]") {
	atl::flat_hash_set<uint64, std::hash<uint64>> s;
	for (uint64 u = 0; u < 300; ++u)
		for (uint64 v = 0; v < 300; ++v) REQUIRE(s.insert(u << 32 | v << 20));
	REQUIRE(s.size() == 90'000);
	for (uint64 u = 0; u < 300; ++u) REQUIRE(s.contains(u << 32 | 299u << 20));
	REQUIRE_FALSE(s.contains(uint64(300) << 32));
}

namespace {
	// requires-expressions need a template context to fail softly
	template<typename C, typename K>
	concept can_find = requires(C &c, K k) { c.find(k); };
	template<typename C, typename K>
	concept can_contains = requires(C &c, K k) { c.contains(k); };
	template<typename C, typename K>
	concept can_erase = requires(C &c, K k) { c.erase(k); };

	struct plain_string_hash {
		std::size_t operator()(std::string const &s) const noexcept { return std::hash<std::string>{}(s); }
	};
}

TEST_CASE("flat_hash_set: heterogeneous lookup matches the homogeneous path", "[flat_hash_set]") {
	std::mt19937_64 rng(11);
	auto word = [&] {
		std::string w(1 + rng() % 6, 'a');
		for (char &c : w) c = char('a' + rng() % 4);
		return w;
	};
	atl::flat_hash_set<std::string> s;
	for (int i = 0; i < 300; ++i) s.insert(word());
	for (int i = 0; i < 2000; ++i) {
		std::string const key = word();
		std::string_view const view = key;
		bool const is_present = s.contains(key);
		REQUIRE(s.contains(view) == is_present);
		REQUIRE(s.contains(key.c_str()) == is_present);
		REQUIRE(s.count(view) == s.count(key));
		REQUIRE(s.find(view) == s.find(key));
		if (i % 3 == 0) {
			REQUIRE(s.erase(view) == is_present);
			REQUIRE(!s.contains(key));
		}
	}
}

TEST_CASE("flat_hash_set: heterogeneous overloads are opt-in", "[flat_hash_set]") {
	using transparent = atl::flat_hash_set<std::string>;
	using plain = atl::flat_hash_set<std::string, plain_string_hash>;
	using sv = std::string_view;
	STATIC_REQUIRE(can_find<transparent, sv> && can_contains<transparent, sv> && can_erase<transparent, sv>);
	STATIC_REQUIRE(!can_find<plain, sv>);
	STATIC_REQUIRE(!can_contains<plain, sv>);
	STATIC_REQUIRE(!can_erase<plain, sv>);
	STATIC_REQUIRE(!can_erase<transparent, transparent::iterator>);
}

TEST_CASE("flat_hash_set: hasher results need only size_t conversion", "[flat_hash_set]") {
	struct result {
		std::size_t value;

		operator std::size_t() const noexcept { return value; }
	};

	struct hash {
		result operator()(int key) const noexcept { return {std::size_t(key)}; }
	};

	atl::flat_hash_set<int, hash> set;
	for (int i = 0; i < 100; ++i) REQUIRE(set.insert(i));
	for (int i = 0; i < 100; ++i) REQUIRE(set.contains(i));
}

TEST_CASE("flat_hash_set: oversized reservation aborts before arithmetic wraps", "[flat_hash_set]") {
	REQUIRE(atl_test::aborts([] {
		atl::flat_hash_set<int> set;
		set.reserve((std::numeric_limits<std::size_t>::max)());
	}));
}

static_assert([] {
	atl::flat_hash_set<int, atl::identity_hash> set(std::array{1, 2, 2});
	set.emplace(3);
	set.reserve(16);
	if (set.size() != 3 || !set.contains(2) || set.count(4) != 0 || set.find(1) == set.end()) return false;
	if (!set.erase(2)) return false;
	set.clear();
	return set.empty();
}());
