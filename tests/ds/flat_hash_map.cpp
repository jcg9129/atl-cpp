#include <atl/ds/flat_hash_map.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/int128.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include "move_probe.hpp"

using atl::uint64;

TEST_CASE("flat_hash_map: matches std::map under random ops", "[flat_hash_map]") {
	auto seed = GENERATE(take(4, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	std::map<uint64, int> ref;
	atl::flat_hash_map<uint64, int> m;
	for (int i = 0; i < 100000; ++i) {
		uint64 const k = rng() % 3000;
		switch (rng() % 5) {
		case 0: {
			int const v = int(rng());
			m[k] = v;
			ref[k] = v;
			break;
		}
		case 1: {
			int const v = int(rng());
			REQUIRE(m.insert(k, v) == ref.insert({k, v}).second);
			break;
		}
		case 2: {
			int const v = int(rng());
			m.insert_or_assign(k, v);
			ref[k] = v;
			break;
		}
		case 3:
			REQUIRE(m.erase(k) == (ref.erase(k) > 0));
			break;
		default: {
			auto it = m.find(k);
			REQUIRE((it != m.end()) == (ref.count(k) > 0));
			if (it != m.end()) REQUIRE(it->second == ref[k]);
		}
		}
		REQUIRE(m.size() == ref.size());
	}
	std::map<uint64, int> got;
	for (auto const &kv : m) got[kv.first] = kv.second;
	REQUIRE(got == ref);
}

TEST_CASE("flat_hash_map: operator[] and at", "[flat_hash_map]") {
	atl::flat_hash_map<int, int> m;
	REQUIRE(m[7] == 0);
	m[7] = 42;
	REQUIRE(m[7] == 42);
	REQUIRE(m.size() == 1);
	++m[7];
	REQUIRE(m.at(7) == 43);
	REQUIRE_FALSE(m.insert(7, 100));
	REQUIRE(m.at(7) == 43);
	REQUIRE(m.insert(8, 100));
	REQUIRE(m.insert_or_assign(8, 200) == false);
	REQUIRE(m.at(8) == 200);
}

TEST_CASE("flat_hash_map: at_or returns the fallback without inserting", "[flat_hash_map]") {
	atl::flat_hash_map<int, int> m;
	m[7] = 42;
	REQUIRE(m.at_or(7, -1) == 42);
	REQUIRE(m.at_or(99, -1) == -1);
	REQUIRE(m.size() == 1);
	REQUIRE_FALSE(m.contains(99));
	auto const &cm = m;
	REQUIRE(cm.at_or(7, 0) == 42);
	REQUIRE(cm.at_or(8, 0) == 0);
}

TEST_CASE("flat_hash_map: at_or moves an rvalue fallback on a miss", "[flat_hash_map]") {
	using atl_test::move_probe;
	atl::flat_hash_map<int, move_probe> m;
	m.insert_or_assign(1, move_probe(5));

	move_probe::reset();
	move_probe const got = m.at_or(999, move_probe(move_probe::sentinel));
	REQUIRE(got.v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies == 0);

	// a hit never touches the fallback (an lvalue fallback stays uncopied)
	move_probe fallback(move_probe::sentinel);
	move_probe::reset();
	REQUIRE(m.at_or(1, fallback).v == 5);
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(fallback.v == move_probe::sentinel);
}

TEST_CASE("flat_hash_map: string and pair keys", "[flat_hash_map]") {
	atl::flat_hash_map<std::string, int> counts;
	for (char const *w : {"a", "b", "a", "a", "b"}) ++counts[w];
	REQUIRE(counts.at("a") == 3);
	REQUIRE(counts.at("b") == 2);
	REQUIRE(counts.size() == 2);

	atl::flat_hash_map<std::pair<int, int>, int> grid;
	grid[{1, 2}] = 5;
	grid[{1, 2}] += 1;
	REQUIRE(grid.at({1, 2}) == 6);
	REQUIRE_FALSE(grid.contains({2, 1}));
}

TEST_CASE("flat_hash_map: rvalue keys and values are moved in, never copied", "[flat_hash_map]") {
	using atl_test::move_probe;

	struct probe_hash {
		std::size_t operator()(move_probe const &p) const noexcept { return std::size_t(atl::uint64(p.v) * 0x9e3779b97f4a7c15ULL); }
	};

	struct probe_eq {
		bool operator()(move_probe const &a, move_probe const &b) const { return a.v == b.v; }
	};

	atl::flat_hash_map<move_probe, move_probe, probe_hash, probe_eq> m(8);

	move_probe::reset();
	m[move_probe(move_probe::sentinel)] = move_probe(7);
	REQUIRE(move_probe::sentinel_copies == 0);

	move_probe::reset();
	REQUIRE(m.insert(move_probe(move_probe::sentinel + 1), move_probe(move_probe::sentinel)));
	REQUIRE(move_probe::sentinel_copies == 0);

	move_probe::reset();
	REQUIRE_FALSE(m.insert(move_probe(move_probe::sentinel + 1), move_probe(move_probe::sentinel)));
	REQUIRE(move_probe::sentinel_copies == 0);

	move_probe::reset();
	m.insert_or_assign(move_probe(move_probe::sentinel + 2), move_probe(move_probe::sentinel));
	REQUIRE(move_probe::sentinel_copies == 0);

	move_probe lkey(move_probe::sentinel);
	move_probe::reset();
	REQUIRE_FALSE(m.insert(lkey, move_probe(1)));
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(m.size() == 3);
}

TEST_CASE("flat_hash_map: move-only mapped type", "[flat_hash_map]") {
	// T is documented as movable, not copyable — rehash must not copy slots.
	atl::flat_hash_map<int, std::unique_ptr<int>> m;
	for (int i = 0; i < 1000; ++i) m.insert_or_assign(i, std::make_unique<int>(i * i));
	REQUIRE(m.size() == 1000);
	for (int i = 0; i < 1000; ++i) {
		REQUIRE(m.contains(i));
		REQUIRE(*m.at(i) == i * i);
	}
	m[500] = std::make_unique<int>(7);
	REQUIRE(*m.at(500) == 7);
	REQUIRE(m.erase(500));
	REQUIRE_FALSE(m.contains(500));
}

TEST_CASE("flat_hash_map: try_emplace builds the value only on insertion", "[flat_hash_map]") {
	using atl_test::move_probe;
	atl::flat_hash_map<int, move_probe> m(8);
	move_probe::reset();
	REQUIRE(m.try_emplace(1, move_probe::sentinel));
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(m.at(1).v == move_probe::sentinel);
	move_probe value(move_probe::sentinel);
	move_probe::reset();
	REQUIRE_FALSE(m.try_emplace(1, std::move(value)));
	REQUIRE(move_probe::sentinel_copies + move_probe::sentinel_moves == 0);
	REQUIRE(value.v == move_probe::sentinel);
	std::string key = "k";
	atl::flat_hash_map<std::string, std::string> s;
	REQUIRE(s.try_emplace(std::move(key), 3, 'x'));
	REQUIRE(s.at("k") == "xxx");
	REQUIRE(s.emplace("q", "w"));
	REQUIRE_FALSE(s.emplace(std::pair<std::string, std::string>("q", "other")));
	REQUIRE(s.at("q") == "w");
	REQUIRE(s.size() == 2);
}

TEST_CASE("flat_hash_map: range constructor and insert", "[flat_hash_map]") {
	using atl_test::move_probe;
	std::vector<std::pair<int, int>> const entries{{1, 1}, {2, 4}, {1, 9}, {3, 9}};
	atl::flat_hash_map<int, int> m(entries);
	REQUIRE(m.size() == 3);
	REQUIRE(m.at(1) == 1);
	m.insert(std::vector<std::pair<int, int>>{{4, 16}, {2, 0}});
	REQUIRE(m.size() == 4);
	REQUIRE(m.at(2) == 4);
	REQUIRE(m.at(4) == 16);

	std::vector<std::pair<int, move_probe>> probes;
	probes.emplace_back(7, move_probe(move_probe::sentinel));
	atl::flat_hash_map<int, move_probe> copied;
	move_probe::reset();
	copied.insert(std::span(probes)); // a borrowed rvalue range: copied, never gutted
	REQUIRE(move_probe::sentinel_copies == 1);
	REQUIRE(probes[0].second.v == move_probe::sentinel);
	move_probe::reset();
	atl::flat_hash_map<int, move_probe> moved(std::move(probes)); // an owning rvalue: moved
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(moved.at(7).v == move_probe::sentinel);

	std::vector<std::pair<int, std::unique_ptr<int>>> owners;
	owners.emplace_back(5, std::make_unique<int>(25));
	atl::flat_hash_map<int, std::unique_ptr<int>> unique(std::move(owners));
	REQUIRE(*unique.at(5) == 25);
}

// operator[] on a present key never rehashes, even when the table is one insert
// from its growth threshold.
TEST_CASE("flat_hash_map: operator[] on a present key does not grow", "[flat_hash_map]") {
	atl::flat_hash_map<uint64, int> m;
	m.reserve(12);
	std::size_t const cap = m.capacity();
	std::size_t n = 0;
	while ((n + 1) * 4 < cap * 3) m[uint64(n++)] = 1;
	for (std::size_t i = 0; i < n; ++i) ++m[uint64(i)];
	REQUIRE(m.capacity() == cap);
	REQUIRE(m.at(0) == 2);
}

namespace {
	// requires-expressions need a template context to fail softly
	template<typename C, typename K>
	concept can_find = requires(C &c, K k) { c.find(k); };
	template<typename C, typename K>
	concept can_contains = requires(C &c, K k) { c.contains(k); };
	template<typename C, typename K>
	concept can_erase = requires(C &c, K k) { c.erase(k); };
	template<typename C, typename K>
	concept can_at = requires(C &c, K k) { c.at(k); c.at_or(k, 0); };

	std::string random_word(std::mt19937_64 &rng) {
		std::string w(1 + rng() % 6, 'a');
		for (char &c : w) c = char('a' + rng() % 4);
		return w;
	}

	struct plain_string_hash {
		std::size_t operator()(std::string const &s) const noexcept { return std::hash<std::string>{}(s); }
	};
}

TEST_CASE("flat_hash_map: heterogeneous lookup matches the homogeneous path", "[flat_hash_map]") {
	std::mt19937_64 rng(7);
	atl::flat_hash_map<std::string, int> m;
	for (int i = 0; i < 300; ++i) m[random_word(rng)] = i;
	for (int i = 0; i < 2000; ++i) {
		std::string const key = random_word(rng);
		std::string_view const view = key;
		char const *const c_str = key.c_str();
		bool const is_present = m.contains(key);
		REQUIRE(m.contains(view) == is_present);
		REQUIRE(m.contains(c_str) == is_present);
		REQUIRE(m.count(view) == m.count(key));
		REQUIRE((m.find(view) == m.end()) == !is_present);
		REQUIRE(m.at_or(view, -1) == m.at_or(key, -1));
		if (is_present) {
			REQUIRE(&m.at(view) == &m.at(key));
			REQUIRE(m.find(view) == m.find(key));
		}
		if (i % 3 == 0) {
			auto const copy = m;
			bool const erased = m.erase(view);
			REQUIRE(erased == is_present);
			REQUIRE(!m.contains(key));
			REQUIRE(m.size() == copy.size() - (is_present ? 1 : 0));
		}
	}
	REQUIRE(m.contains("abc") == m.contains(std::string("abc")));
}

TEST_CASE("flat_hash_map: heterogeneous overloads are opt-in", "[flat_hash_map]") {
	using transparent = atl::flat_hash_map<std::string, int>;
	using plain = atl::flat_hash_map<std::string, int, plain_string_hash>;
	using plain_eq = atl::flat_hash_map<std::string, int, atl::hash<std::string>, std::equal_to<std::string>>;
	using sv = std::string_view;
	STATIC_REQUIRE(can_find<transparent, sv> && can_contains<transparent, sv> && can_erase<transparent, sv> && can_at<transparent, sv>);
	STATIC_REQUIRE(!can_find<plain, sv>);
	STATIC_REQUIRE(!can_contains<plain, sv>);
	STATIC_REQUIRE(!can_erase<plain, sv>);
	STATIC_REQUIRE(!can_contains<plain_eq, sv>);
	STATIC_REQUIRE(!can_contains<transparent, int>);
	// erase(iterator) is never taken for a key: it stays a compile error here
	STATIC_REQUIRE(!can_erase<transparent, transparent::iterator>);
}

namespace {
	struct mixed_hash {
		using is_transparent = void;

		std::size_t operator()(atl::uint64 x) const noexcept { return std::size_t(x); }

		std::size_t operator()(atl::int32 x) const noexcept { return std::size_t(x); }
	};

	struct mixed_eq {
		using is_transparent = void;

		bool operator()(atl::uint64 a, atl::int32 b) const { return a == uint64(b); }

		bool operator()(atl::uint64 a, atl::uint64 b) const { return a == b; }
	};
}

TEST_CASE("flat_hash_map: a mixed arithmetic probe is not a heterogeneous key", "[flat_hash_map]") {
	// -1 == 2^64 - 1 under the mixed eq: the pair is rejected, the probe converts to Key instead
	STATIC_REQUIRE(!atl::detail::hashed_heterogeneous_key_for<atl::int32, uint64, mixed_hash, mixed_eq>);
	STATIC_REQUIRE(atl::detail::hashed_heterogeneous_key_for<std::string_view, std::string, atl::hash<std::string>,
															 std::equal_to<>>);
}

TEST_CASE("flat_hash_map: software and selected 128-bit keys", "[flat_hash_map]") {
	auto const check = []<atl::integral Key>
		requires(std::numeric_limits<Key>::digits >= 127)
	() {
		atl::flat_hash_map<Key, int> map;
		Key const high = Key(1) << 100;
		REQUIRE(map.insert(high, 7));
		REQUIRE(map.insert(Key(3), 9));
		map.reserve(200);
		REQUIRE(map.at(high) == 7);
		REQUIRE(map.at(Key(3)) == 9);
		REQUIRE(map.erase(high));
		REQUIRE_FALSE(map.contains(high));
	};
	check.operator()<atl::uint128>();
	check.operator()<atl::int128>();
	check.operator()<atl::detail::software_uint128>();
	check.operator()<atl::detail::software_int128>();
	STATIC_REQUIRE(atl::detail::arithmetic_pair<atl::detail::software_int128, unsigned int>);
}

static_assert(!atl::detail::key_value_entry<std::tuple<int>, int, int>);
static_assert(!atl::detail::key_value_entry<std::tuple<>, int, int>);
static_assert(!atl::detail::key_value_entry<int, int, int>);

static_assert([] {
	atl::flat_hash_map<int, int, atl::identity_hash> map(std::array{std::pair{1, 2}, std::pair{1, 3}});
	map.try_emplace(2, 4);
	map.emplace(3, 6);
	map.insert_or_assign(2, 5);
	map.reserve(16);
	if (map.at(1) != 2 || map[2] != 5 || map.at_or(4, -1) != -1 || map.count(3) != 1) return false;
	if (!map.contains(2) || map.find(1) == map.end() || !map.erase(2)) return false;
	map.clear();
	return map.empty();
}());
