/**
 * @file
 * Race between `atl::hash` (the library), `hash_by_function_local_seed` (a guarded
 * function-local seed), and `hash_by_three_seeded_rounds` (hash each pair member,
 * then combine, instead of packing the pair into one round). Each loop measures
 * throughput of one million random uint64 or uint32-pair insertions, lookups
 * with alternating hits and misses, or hashes alone.
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`,
 * 2026-10-15): the library takes 29.1/27.6 ms for integer lookup versus
 * 30.6/29.9 ms by function local seed, and 30.0/27.9 ms for pair lookup
 * versus 37.8/39.2 ms by three seeded rounds. Numbers: docs/flat-hash.md.
 */
#include <atl/ds/detail/hash_table.hpp>
#include <atl/ds/flat_hash_set.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/hash.hpp>
#include <catch2/catch.hpp>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace atl;

namespace {
	/** @brief A function-local seed; each access checks static initialization. */
	inline uint64 lazy_seed() {
		static uint64 const seed = [] {
			uint64 s = uint64(std::chrono::steady_clock::now().time_since_epoch().count());
			std::random_device rd;
			s ^= (uint64(rd()) << 32) ^ rd();
			return detail::splitmix64(s);
		}();
		return seed;
	}

	/** @brief `atl::hash<uint64>` on the @ref lazy_seed contender. */
	struct hash_by_function_local_seed {
		std::size_t operator()(uint64 x) const noexcept { return std::size_t(detail::splitmix64(x + lazy_seed())); }
	};

	/** @brief Hash a pair with three seeded rounds instead of the library's packed round. */
	struct hash_by_three_seeded_rounds {
		std::size_t operator()(std::pair<uint32, uint32> const &p) const noexcept {
			return std::size_t(detail::hash_combine(hash<uint32>{}(p.first), hash<uint32>{}(p.second)));
		}
	};
}

TEST_CASE("hash contenders", "[hash]") {
	constexpr std::size_t n = 1'000'000;
	std::mt19937_64 rng(12345);
	std::vector<uint64> present(n), absent(n);
	for (auto &x : present) x = rng();
	for (auto &x : absent) x = rng();

	auto benchmark_integer_hash_operations = [&]<typename Hash>
		requires detail::hasher_for<Hash, uint64> && std::default_initializable<Hash>
	(char const *label) {
		BENCHMARK(std::string("uint64 insert 1e6 | ") + label) {
			flat_hash_set<uint64, Hash> s(n);
			for (uint64 k : present) s.insert(k);
			return s.size();
		};
		flat_hash_set<uint64, Hash> s(n);
		{
			flat_hash_set<uint64> reference(n);
			for (uint64 k : present) {
				s.insert(k);
				reference.insert(k);
			}
			REQUIRE(s.size() == reference.size());
			for (std::size_t i = 0; i < n; ++i) {
				REQUIRE(s.contains(present[i]) == reference.contains(present[i]));
				REQUIRE(s.contains(absent[i]) == reference.contains(absent[i]));
			}
		}
		BENCHMARK(std::string("uint64 lookup 1e6 (50% miss) | ") + label) {
			std::size_t hits = 0;
			for (std::size_t i = 0; i < n; ++i) hits += s.contains(i % 2 ? present[i] : absent[i]);
			return hits;
		};
		BENCHMARK(std::string("uint64 hash only 1e6 | ") + label) {
			uint64 acc = 0;
			Hash const h{};
			for (uint64 k : present) acc += h(k);
			return acc;
		};
	};
	benchmark_integer_hash_operations.operator()<hash_by_function_local_seed>("by function local seed");
	benchmark_integer_hash_operations.operator()<hash<uint64>>("library (static initialization seed)");

	using key = std::pair<uint32, uint32>;
	std::vector<key> pairs(n), other(n);
	for (auto &p : pairs) p = {uint32(rng()), uint32(rng())};
	for (auto &p : other) p = {uint32(rng()), uint32(rng())};
	auto benchmark_pair_hash_operations = [&]<typename Hash>
		requires detail::hasher_for<Hash, key> && std::default_initializable<Hash>
	(char const *label) {
		BENCHMARK(std::string("pair<uint32,uint32> insert 1e6 | ") + label) {
			flat_hash_set<key, Hash> s(n);
			for (auto const &k : pairs) s.insert(k);
			return s.size();
		};
		flat_hash_set<key, Hash> s(n);
		{
			flat_hash_set<key> reference(n);
			for (auto const &k : pairs) {
				s.insert(k);
				reference.insert(k);
			}
			REQUIRE(s.size() == reference.size());
			for (std::size_t i = 0; i < n; ++i) {
				REQUIRE(s.contains(pairs[i]) == reference.contains(pairs[i]));
				REQUIRE(s.contains(other[i]) == reference.contains(other[i]));
			}
		}
		BENCHMARK(std::string("pair<uint32,uint32> lookup 1e6 (50% miss) | ") + label) {
			std::size_t hits = 0;
			for (std::size_t i = 0; i < n; ++i) hits += s.contains(i % 2 ? pairs[i] : other[i]);
			return hits;
		};
	};
	benchmark_pair_hash_operations.operator()<hash_by_three_seeded_rounds>("by three seeded rounds");
	benchmark_pair_hash_operations.operator()<hash<key>>("library (packed pair)");
}
