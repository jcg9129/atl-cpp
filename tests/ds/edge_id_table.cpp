#include <atl/ds/detail/edge_id_table.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/hash_core.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <map>
#include <random>
#include <type_traits>
#include <utility>
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
	uint32 const live = GENERATE(256u, 2048u);
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

namespace {
	using owner_pair = std::array<uint32, 2>;

	template<typename Value>
	constexpr Value table_value(uint32 i) {
		if constexpr (std::same_as<Value, owner_pair>)
			return {i, ~i};
		else
			return Value(i);
	}
}

TEMPLATE_TEST_CASE("integer_key_table: all value types, structured growth and churn", "[edge_id_table]", uint32, atl::uint8, owner_pair) {
	using table_type = atl::detail::integer_key_table<TestType>;
	static_assert(std::is_nothrow_default_constructible_v<table_type>);
	static_assert(std::is_nothrow_move_constructible_v<table_type>);
	static_assert(noexcept(std::declval<table_type &>().entry_value(0)));
	static_assert(noexcept(std::declval<table_type const &>().entry_value(0)));
	static_assert(noexcept(std::declval<table_type &>().erase_entry(0)));
	table_type table;
	std::map<uint64, TestType> reference;
	CHECK(table.entry_value(0) == nullptr);
	CHECK(std::as_const(table).entry_value(0) == nullptr);
	for (uint32 u = 0; u < 32; ++u)
		for (uint32 v = 0; v < 32; ++v) {
			uint64 const key = (uint64(u) << 32) | v;
			auto const value = table_value<TestType>(u * 32 + v);
			table.insert_entry(key, value);
			table.insert_entry(key, table_value<TestType>(7));
			reference.emplace(key, value);
		}
	table.reserve_entries(4096);
	for (auto const &[key, value] : reference) REQUIRE(*std::as_const(table).entry_value(key) == value);
	std::mt19937_64 rng(71);
	for (uint32 step = 0; step < 20000; ++step) {
		uint64 const key = ((rng() % 32) << 32) | (rng() % 32);
		auto const value = table_value<TestType>(step);
		if (step % 3 == 0) {
			table.erase_entry(key);
			reference.erase(key);
		} else if (auto const entry = table.entry_value(key)) {
			*entry = value;
			reference.at(key) = value;
		} else {
			table.insert_entry(key, value);
			reference.emplace(key, value);
		}
		auto const found = std::as_const(table).entry_value(key);
		REQUIRE(bool(found) == reference.contains(key));
		if (found) REQUIRE(*found == reference.at(key));
		REQUIRE(table.entry_count() == reference.size());
		if (step % 1000 == 0)
			for (auto const &[k, expected] : reference) REQUIRE(*table.entry_value(k) == expected);
	}
	for (uint64 const key : {uint64(0), ~uint64(0)}) {
		table.erase_entry(key);
		table.insert_entry(key, table_value<TestType>(~uint32(0)));
		REQUIRE(table.lookup_id(key, table_value<TestType>(0)) == table_value<TestType>(~uint32(0)));
	}
	table.reset_entries();
	REQUIRE(table.entry_count() == 0);
	REQUIRE(table.entry_value(0) == nullptr);
	table.insert_entry(0, table_value<TestType>(0));
	REQUIRE(*table.entry_value(0) == table_value<TestType>(0));
}

TEMPLATE_TEST_CASE("integer_key_table: values survive wrapped backward shifts", "[edge_id_table]", uint32, atl::uint8, owner_pair) {
	std::size_t const capacity = GENERATE(8u, 4096u, 8192u);
	std::vector<uint64> keys;
	for (uint64 k = 0; keys.size() < 4; ++k)
		if ((atl::detail::edge_id_hash(k) & (capacity - 1)) == capacity - 1) keys.push_back(k);
	for (std::size_t removed = 0; removed < keys.size(); ++removed) {
		atl::detail::integer_key_table<TestType> table;
		table.reserve_entries(capacity / 2);
		for (std::size_t i = 0; i < keys.size(); ++i) table.insert_entry(keys[i], table_value<TestType>(uint32(i)));
		table.erase_entry(keys[removed]);
		for (std::size_t i = 0; i < keys.size(); ++i) {
			auto const found = table.entry_value(keys[i]);
			REQUIRE(bool(found) == (i != removed));
			if (found) REQUIRE(*found == table_value<TestType>(uint32(i)));
		}
		table.insert_entry(keys[removed], table_value<TestType>(42));
		REQUIRE(*table.entry_value(keys[removed]) == table_value<TestType>(42));
		REQUIRE(table.entry_count() == keys.size());
	}
}

TEMPLATE_TEST_CASE("integer_key_table: threshold migration preserves every entry", "[edge_id_table]", uint32, atl::uint8, owner_pair) {
	bool const reserve = GENERATE(false, true);
	atl::detail::integer_key_table<TestType> table;
	// 1024 entries fit in 2048 slots; the next entry needs exactly 4096 slots.
	for (uint32 i = 0; i < 1024; ++i) table.insert_entry(uint64(i) << 32, table_value<TestType>(i));
	for (uint32 i = 0; i < 1024; i += 3) table.erase_entry(uint64(i) << 32);
	for (uint32 i = 0; i < 1024; i += 3) table.insert_entry(uint64(i) << 32, table_value<TestType>(i));
	if (reserve) table.reserve_entries(1025);
	table.insert_entry(~uint64(0), table_value<TestType>(~uint32(0)));
	auto const verify = [&] {
		REQUIRE(table.entry_count() == 1025);
		for (uint32 i = 0; i < 1024; ++i) {
			auto const value = std::as_const(table).entry_value(uint64(i) << 32);
			REQUIRE(value != nullptr);
			REQUIRE(*value == table_value<TestType>(i));
		}
		REQUIRE(table.lookup_id(~uint64(0), table_value<TestType>(0)) == table_value<TestType>(~uint32(0)));
		REQUIRE(table.entry_value(1) == nullptr);
	};
	verify();
	// Rehash once more with bitmap occupancy on both sides of the migration.
	table.reserve_entries(2049);
	verify();
}

TEMPLATE_TEST_CASE("integer_key_table: reset retains reusable storage at either size", "[edge_id_table]", uint32, atl::uint8, owner_pair) {
	std::size_t const entries = GENERATE(1024u, 2048u, 4096u);
	atl::detail::integer_key_table<TestType> table;
	table.reserve_entries(entries);
	for (uint64 cycle = 0; cycle < 3; ++cycle) {
		for (std::size_t i = 0; i < entries; ++i) table.insert_entry(cycle * entries + i, table_value<TestType>(uint32(i)));
		REQUIRE(table.entry_count() == entries);
		table.reset_entries();
		table.reserve_entries(1);
		REQUIRE(table.entry_count() == 0);
		for (std::size_t i = 0; i < entries; ++i) REQUIRE(table.entry_value(cycle * entries + i) == nullptr);
	}
	table.insert_entry(0, table_value<TestType>(0));
	REQUIRE(*table.entry_value(0) == table_value<TestType>(0));
}
