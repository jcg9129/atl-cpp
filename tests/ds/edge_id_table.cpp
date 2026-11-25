#include <atl/ds/detail/edge_id_table.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/hash_core.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <map>
#include <random>
#include <vector>

using atl::uint32;
using atl::uint64;
using atl::detail::edge_id_table;

TEST_CASE("edge_id_table: insert, query, erase and reuse", "[edge_id_table]") {
	edge_id_table table;
	uint32 const missing = 123;
	REQUIRE(table.lookup_id(0, missing) == missing);
	table.erase_entry(0);
	table.reset_entries();
	table.reserve_entries(0);
	for (uint64 const key : {uint64(0), uint64(1), ~uint64(0)}) {
		table.insert_entry(key, ~uint32(0));
		table.insert_entry(key, 7);
		REQUIRE(table.lookup_id(key, missing) == ~uint32(0));
		table.erase_entry(key);
		REQUIRE(table.lookup_id(key, missing) == missing);
		table.erase_entry(key);
		table.insert_entry(key, 0);
		REQUIRE(table.lookup_id(key, missing) == 0);
	}
	table.reserve_entries(1000);
	REQUIRE(table.lookup_id(~uint64(0), missing) == 0);
	table.reset_entries();
	REQUIRE(table.lookup_id(~uint64(0), missing) == missing);
	table.insert_entry(42, 9);
	REQUIRE(table.lookup_id(42, missing) == 9);
}

TEST_CASE("edge_id_table: structured endpoint pairs grow through capacities", "[edge_id_table]") {
	edge_id_table table;
	for (uint32 u = 0; u < 100; ++u)
		for (uint32 v = 0; v < 100; ++v) table.insert_entry((uint64(u) << 32) | v, u * 100 + v);
	table.reserve_entries(20000);
	table.reserve_entries(10);
	for (uint32 u = 0; u < 100; ++u)
		for (uint32 v = 0; v < 100; ++v) REQUIRE(table.lookup_id((uint64(u) << 32) | v, ~uint32(0)) == u * 100 + v);
}

TEST_CASE("edge_id_table: backward shifts preserve wrapped collision chains", "[edge_id_table]") {
	std::vector<uint64> keys;
	for (uint64 k = 0; keys.size() < 4; ++k)
		if ((atl::detail::edge_id_hash(k) & 7) == 7) keys.push_back(k);
	for (std::size_t removed = 0; removed < keys.size(); ++removed) {
		edge_id_table table;
		for (std::size_t i = 0; i < keys.size(); ++i) table.insert_entry(keys[i], uint32(i));
		table.erase_entry(keys[removed]);
		for (std::size_t i = 0; i < keys.size(); ++i)
			REQUIRE(table.lookup_id(keys[i], ~uint32(0)) == (i == removed ? ~uint32(0) : uint32(i)));
		table.insert_entry(keys[removed], 42);
		REQUIRE(table.lookup_id(keys[removed], ~uint32(0)) == 42);
	}
}

TEST_CASE("edge_id_table: steady size link and cut churn", "[edge_id_table]") {
	edge_id_table table;
	constexpr uint32 live = 256;
	table.reserve_entries(live);
	for (uint32 i = 0; i < live; ++i) table.insert_entry(i, i);
	for (uint32 cycle = 0; cycle < 1000; ++cycle) {
		uint64 const base = uint64(cycle) * live;
		for (uint32 i = 0; i < live; ++i) {
			REQUIRE(table.lookup_id(base + i, ~uint32(0)) == i);
			table.erase_entry(base + i);
			table.insert_entry(base + live + i, i);
			REQUIRE(table.lookup_id(base + i, ~uint32(0)) == ~uint32(0));
		}
		for (uint32 i = 0; i < live; ++i) REQUIRE(table.lookup_id(base + live + i, ~uint32(0)) == i);
	}
}

TEST_CASE("edge_id_table: random mutations agree with ordered map", "[edge_id_table]") {
	edge_id_table table;
	std::map<uint64, uint32> reference;
	std::mt19937_64 rng(17);
	for (uint32 step = 0; step < 50000; ++step) {
		uint64 const key = ((rng() % 32) << 32) | (rng() % 32);
		if (rng() % 2) {
			table.insert_entry(key, step);
			reference.emplace(key, step);
		} else {
			table.erase_entry(key);
			reference.erase(key);
		}
		REQUIRE(table.entry_count() == reference.size());
		auto const it = reference.find(key);
		REQUIRE(table.lookup_id(key, ~uint32(0)) == (it == reference.end() ? ~uint32(0) : it->second));
		if (step % 1000 == 0)
			for (auto const &[k, value] : reference) REQUIRE(table.lookup_id(k, ~uint32(0)) == value);
	}
}

TEST_CASE("edge_id_table: runtime hash exactly matches scalar hash for each seed", "[edge_id_table]") {
	uint64 const saved = atl::detail::hash_seed_value;
	for (uint64 const seed : {uint64(0), uint64(1), ~uint64(0), uint64(123456789)}) {
		atl::set_hash_seed(seed);
		for (uint64 const key : {uint64(0), uint64(1), ~uint64(0), uint64(7) << 32, (uint64(3) << 32) | 9})
			CHECK(atl::detail::edge_id_hash(key) == atl::hash<uint64>{}(key));
		edge_id_table table;
		table.insert_entry(seed, 42);
		CHECK(table.lookup_id(seed, 0) == 42);
	}
	atl::detail::hash_seed_value = saved;
}
