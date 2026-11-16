/**
 * @file
 * Race between the library flat hash containers (linear probing), standard
 * unordered containers (node storage), and hash choices (seeded mixing,
 * standard hash with home-slot mixing, or identity). String lookup races the
 * library's transparent view lookup against a string temporary per query.
 * Loops measure throughput of one million random insertions and alternating
 * hit/miss lookups, packed-key insertions (62500 in the identity arm), a
 * reservation boundary, and one million views into 100000 strings of 32-64 chars.
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`,
 * 2026-10-23): transparent lookup takes 143/133 ms versus 237/205 ms by
 * string temporary. Other shapes: not measured yet. Numbers: docs/flat-hash.md.
 */
#include <atl/ds/flat_hash_map.hpp>
#include <atl/ds/flat_hash_set.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using atl::uint64;

TEST_CASE("flat_hash benchmarks", "[flat_hash]") {
	constexpr std::size_t n = 1'000'000;
	std::mt19937_64 rng(12345);
	std::vector<uint64> present(n), absent(n);
	for (auto &x : present) x = rng();
	for (auto &x : absent) x = rng();

	BENCHMARK("flat_hash_set insert (library seeded hash)") {
		atl::flat_hash_set<uint64> s(n);
		for (uint64 k : present) s.insert(k);
		return s.size();
	};
	BENCHMARK("flat_hash_set insert (by identity hash)") {
		atl::flat_hash_set<uint64, atl::identity_hash> s(n);
		for (uint64 k : present) s.insert(k);
		return s.size();
	};
	BENCHMARK("std::unordered_set insert") {
		std::unordered_set<uint64> s;
		s.reserve(n);
		for (uint64 k : present) s.insert(k);
		return s.size();
	};

	atl::flat_hash_set<uint64> fs(n);
	for (uint64 k : present) fs.insert(k);
	std::unordered_set<uint64> us;
	us.reserve(n);
	for (uint64 k : present) us.insert(k);
	{
		atl::flat_hash_set<uint64, atl::identity_hash> identity(n);
		for (uint64 k : present) identity.insert(k);
		REQUIRE(fs.size() == us.size());
		REQUIRE(identity.size() == us.size());
		for (std::size_t i = 0; i < n; ++i) {
			REQUIRE(fs.contains(present[i]) == us.contains(present[i]));
			REQUIRE(fs.contains(absent[i]) == us.contains(absent[i]));
			REQUIRE(identity.contains(present[i]) == us.contains(present[i]));
		}
		{
			atl::flat_hash_map<uint64, std::size_t> reference;
			std::unordered_map<uint64, std::size_t> contender;
			for (std::size_t i = 0; i < n; ++i) {
				reference[present[i]] = i;
				contender[present[i]] = i;
			}
			REQUIRE(reference.size() == contender.size());
			for (auto const &[key, value] : contender) REQUIRE(reference.at(key) == value);
		}
	}
	BENCHMARK("flat_hash_set lookup (library, 50% miss)") {
		std::size_t hits = 0;
		for (std::size_t i = 0; i < n; ++i) hits += fs.contains(i % 2 ? present[i] : absent[i]);
		return hits;
	};
	BENCHMARK("std::unordered_set lookup (50% miss)") {
		std::size_t hits = 0;
		for (std::size_t i = 0; i < n; ++i) hits += us.count(i % 2 ? present[i] : absent[i]);
		return hits;
	};

	BENCHMARK("flat_hash_map insert (library)") {
		atl::flat_hash_map<uint64, std::size_t> m(n);
		for (std::size_t i = 0; i < n; ++i) m[present[i]] = i;
		return m.size();
	};
	BENCHMARK("std::unordered_map insert") {
		std::unordered_map<uint64, std::size_t> m;
		m.reserve(n);
		for (std::size_t i = 0; i < n; ++i) m[present[i]] = i;
		return m.size();
	};
}

// The identity arm isolates mixing cost on random keys and exposes long probes
// on packed keys. The 3 * 2^19 workload tests reservation at the load boundary.
TEST_CASE("flat_hash integer hash mixing and reserve boundary", "[flat_hash][race]") {
	constexpr std::size_t n = 1'000'000;
	std::mt19937_64 rng(777);
	std::vector<uint64> random_keys(n), packed_keys;
	for (auto &x : random_keys) x = rng();
	for (uint64 u = 0; u < 1000; ++u)
		for (uint64 v = 0; v < 1000; ++v) packed_keys.push_back(u << 32 | v << 16);

	{
		atl::flat_hash_set<uint64, std::hash<uint64>> standard(n);
		atl::flat_hash_set<uint64, atl::identity_hash> identity(n);
		atl::flat_hash_set<uint64> reference(n);
		for (auto k : random_keys) {
			standard.insert(k);
			identity.insert(k);
			reference.insert(k);
		}
		REQUIRE(standard.size() == reference.size());
		REQUIRE(identity.size() == reference.size());
		for (auto k : random_keys) {
			REQUIRE(standard.contains(k ^ 1) == reference.contains(k ^ 1));
			REQUIRE(identity.contains(k ^ 1) == reference.contains(k ^ 1));
		}
		standard.clear();
		identity.clear();
		reference.clear();
		for (auto k : packed_keys) {
			standard.insert(k);
			reference.insert(k);
		}
		REQUIRE(standard.size() == reference.size());
		for (std::size_t i = 0; i < packed_keys.size(); i += 16) {
			identity.insert(packed_keys[i]);
			REQUIRE(reference.contains(packed_keys[i]));
		}
		REQUIRE(identity.size() == packed_keys.size() / 16);
	}
	BENCHMARK("insert+find 1e6 random, std::hash (mixed)") {
		atl::flat_hash_set<uint64, std::hash<uint64>> s(n);
		for (uint64 k : random_keys) s.insert(k);
		std::size_t hits = 0;
		for (uint64 k : random_keys) hits += s.contains(k ^ 1);
		return hits + s.size();
	};
	BENCHMARK("insert+find 1e6 random, by identity hash") {
		atl::flat_hash_set<uint64, atl::identity_hash> s(n);
		for (uint64 k : random_keys) s.insert(k);
		std::size_t hits = 0;
		for (uint64 k : random_keys) hits += s.contains(k ^ 1);
		return hits + s.size();
	};
	BENCHMARK("insert+find 1e6 random, atl::hash") {
		atl::flat_hash_set<uint64> s(n);
		for (uint64 k : random_keys) s.insert(k);
		std::size_t hits = 0;
		for (uint64 k : random_keys) hits += s.contains(k ^ 1);
		return hits + s.size();
	};
	BENCHMARK("insert 1e6 packed (u << 32 | v << 16), std::hash (mixed)") {
		atl::flat_hash_set<uint64, std::hash<uint64>> s(n);
		for (uint64 k : packed_keys) s.insert(k);
		return s.size();
	};
	// the unmixed arm degrades to long probe runs: 1/16 of the keys keeps it in bounds
	BENCHMARK("insert 62500 packed (u << 32 | v << 16), by identity hash") {
		atl::flat_hash_set<uint64, atl::identity_hash> s(n);
		for (std::size_t i = 0; i < packed_keys.size(); i += 16) s.insert(packed_keys[i]);
		return s.size();
	};

	constexpr std::size_t boundary = 3 * (std::size_t(1) << 19);
	std::vector<uint64> many(boundary);
	for (auto &x : many) x = rng();
	BENCHMARK("reserve(3*2^19) + 3*2^19 inserts") {
		atl::flat_hash_set<uint64> s;
		s.reserve(boundary);
		for (uint64 k : many) s.insert(k);
		return s.capacity();
	};
}

// lookup of a string_view key in a std::string-keyed table. The homogeneous arm
// builds a std::string temporary per call (32-64 chars, usually a heap allocation);
// the transparent arm hashes and compares the view in place.
TEST_CASE("flat_hash heterogeneous string lookup", "[flat_hash][race]") {
	constexpr std::size_t n = 1'000'000;
	constexpr std::size_t table_size = 100'000;
	std::mt19937_64 rng(385);
	std::vector<std::string> words(table_size);
	for (auto &w : words) {
		w.resize(32 + std::size_t(rng() % 33));
		for (char &c : w) c = char('a' + rng() % 26);
	}
	atl::flat_hash_map<std::string, int> m;
	for (std::size_t i = 0; i < table_size; ++i) m[words[i]] = int(i);
	std::vector<std::string_view> queries(n);
	for (auto &q : queries) q = words[std::size_t(rng() % table_size)];

	for (auto q : queries) REQUIRE(m.at(q) == m.at(std::string(q)));
	BENCHMARK("1e6 string_view lookups, by string temporary") {
		std::size_t sum = 0;
		for (std::string_view q : queries) sum += std::size_t(m.at(std::string(q)));
		return sum;
	};

	BENCHMARK("1e6 string_view lookups, library transparent lookup") {
		std::size_t sum = 0;
		for (std::string_view q : queries) sum += std::size_t(m.at(q));
		return sum;
	};
}
