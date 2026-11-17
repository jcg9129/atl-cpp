/**
 * @file
 * @brief Races full and partial persistent append, list/trie random reads, indexed sets,
 * RRB concat/slice/insert, bulk construction, and list front growth by copying or
 * moving the version handle. Measures complete-script throughput; growth uses 1e6
 * values, reads 2e6 probes, sets 2e5 updates and RRB operations 2000 steps.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-23,
 * -O2 / -O2 -march=native (also -g -DNDEBUG -std=c++23): list front growth
 * by copied handle 32.4 / 33.4 ms, by moved handle 10.2 / 10.5.
 * Other shapes not measured yet; docs/persistent-vector.md and docs/persistent-list.md.
 */

#include <atl/ds/persistent_list.hpp>
#include <atl/ds/persistent_vector.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <format>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
	/** @brief Time a complete workload through a direct, constrained callable. */
	template<typename Run>
		requires std::invocable<Run &>
	void measure(std::string const &label, Run run) {
		BENCHMARK(std::string(label)) { return run(); };
	}
}

using atl::int64;
using atl::uint64;

TEST_CASE("growable persistent vector benchmarks", "[persistent_vector]") {
	auto seed = GENERATE(take(1, random(uint64(0), (std::numeric_limits<uint64>::max)())));

	std::size_t const n = 1000000;
	std::size_t const reads = 2000000;
	std::size_t const updates = 200000;

	std::mt19937_64 rng(seed);
	std::vector<int64> values(n);
	for (auto &x : values) x = int64(rng());
	std::vector<std::size_t> read_index(reads);
	for (auto &i : read_index) i = rng() % n;

	auto label = std::format("n={}", n);

	measure("persistent_vector -- push_back keep-all -- " + label, [&] {
		atl::persistent_vector<int64> v;
		for (int64 x : values) v = v.push_back(x);
		return v[0];
	});
	measure("semi_persistent_vector -- push_back keep-all -- " + label, [&] {
		atl::semi_persistent_vector<int64> v;
		for (int64 x : values) v = v.push_back(x);
		return v[0];
	});

	measure("persistent_vector -- bulk build (O(n)) -- " + label, [&] {
		return atl::persistent_vector<int64>(values).size();
	});

	atl::persistent_vector<int64> full(values);
	atl::persistent_list<int64> list(values);
	for (auto i : read_index) REQUIRE(full[i] == list[i]);
	atl::semi_persistent_vector<int64> partial(values);
	for (auto i : read_index) REQUIRE(full[i] == partial[i]);
	measure(std::format("persistent_vector -- {} random reads", reads), [&] {
		uint64 sum = 0;
		for (auto i : read_index) sum += uint64(full[i]);
		return sum;
	});
	measure(std::format("persistent_list -- {} random reads", reads), [&] {
		uint64 sum = 0;
		for (auto i : read_index) sum += uint64(list[i]);
		return sum;
	});
	measure(std::format("persistent_vector -- {} persistent set", updates), [&] {
		auto v = full;
		for (std::size_t k = 0; k < updates; ++k) v = v.set(read_index[k], int64(k));
		return v[0];
	});

	std::size_t const rrb_ops = 2000;
	measure(std::format("persistent_vector -- {} concat (shared arena) + build", rrb_ops), [&] {
		atl::persistent_vector<int64> f(values);
		auto left = f.slice(0, n / 2);
		auto right = f.slice(n / 2, n);
		atl::persistent_vector<int64> c;
		for (std::size_t k = 0; k < rrb_ops; ++k) c = left.concat(right);
		return c.size();
	});
	measure(std::format("persistent_vector -- {} slice + build", rrb_ops), [&] {
		atl::persistent_vector<int64> f(values);
		atl::persistent_vector<int64> s;
		for (std::size_t k = 0; k < rrb_ops; ++k)
			s = f.slice(read_index[k] % (n / 2), n / 2 + read_index[k] % (n / 2));
		return s.size();
	});
	measure(std::format("persistent_vector -- {} insert-anywhere + build", rrb_ops), [&] {
		atl::persistent_vector<int64> v(values);
		for (std::size_t k = 0; k < rrb_ops; ++k) v = v.insert(read_index[k] % v.size(), int64(k));
		return v[0];
	});
}

TEST_CASE("persistent_list push_front copy vs move of the version", "[persistent_vector][race]") {
	auto seed = GENERATE(take(1, random(uint64(0), (std::numeric_limits<uint64>::max)())));

	std::size_t const n = 1000000;
	std::mt19937_64 rng(seed);
	std::vector<int64> values(n);
	for (auto &x : values) x = int64(rng());
	auto label = std::format("n={}", n);

	measure("persistent_list -- push_front keep-all -- " + label, [&] {
		atl::persistent_list<int64> v;
		for (int64 x : values) v = v.push_front(x);
		return v[0];
	});
	measure("persistent_list -- push_front keep-all, std::move(v) idiom -- " + label, [&] {
		atl::persistent_list<int64> v;
		for (int64 x : values) v = std::move(v).push_front(x);
		return v[0];
	});
}
