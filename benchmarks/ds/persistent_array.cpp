/**
 * @file
 * Race between `atl::persistent_array` (the library, path copying),
 * `atl::semi_persistent_array` (bounded node modifications), build models
 * `persistent_array_by_build_reservation` (up-front reservation or geometric
 * growth), and arena models `reserve_extra_by_growth_policy` (exact or geometric
 * reservation). Loops measure throughput of builds, dependent update chains,
 * random reads, and reservation policies on Radix-16 nodes, including allocation.
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`,
 * 2026-10-15): update chains take 87.1/89.3 ms for the library and 49.5/50.8 ms
 * for semi-persistence; 2000 exact arena reservations take 48.9/49.5 ms versus
 * 0.163/0.158 ms by geometric growth. Numbers: docs/persistent-array.md.
 */
#include <atl/ds/persistent_array.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <format>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

using atl::int64;
using atl::uint64;

TEST_CASE("persistent array benchmarks", "[persistent_array]") {
	auto seed = GENERATE(take(1, random(uint64(0), (std::numeric_limits<uint64>::max)())));

	std::size_t const n = 1u << 20;
	std::size_t const updates = 200000;
	std::size_t const reads = 1000000;

	std::mt19937_64 rng(seed);
	std::vector<int64> initial(n);
	for (auto &x : initial) x = int64(rng());
	std::vector<std::size_t> set_index(updates);
	std::vector<int64> set_value(updates);
	for (std::size_t k = 0; k < updates; ++k) set_index[k] = rng() % n, set_value[k] = int64(rng());
	std::vector<std::size_t> read_index(reads);
	for (auto &i : read_index) i = rng() % n;

	auto label = std::format("n={}, updates={}", n, updates);

	BENCHMARK("persistent_array -- build -- " + label) {
		return atl::persistent_array<int64>(initial).size();
	};
	BENCHMARK("semi_persistent_array -- build -- " + label) {
		return atl::semi_persistent_array<int64>(initial).size();
	};

	// Each sample starts with a fresh pool; build cost is included for both layouts.
	BENCHMARK("persistent_array -- update chain -- " + label) {
		atl::persistent_array<int64> v(initial);
		v.reserve(updates);
		for (std::size_t k = 0; k < updates; ++k) v = v.set(set_index[k], set_value[k]);
		return v[set_index[0]];
	};
	BENCHMARK("semi_persistent_array -- update chain -- " + label) {
		atl::semi_persistent_array<int64> v(initial);
		v.reserve(updates);
		for (std::size_t k = 0; k < updates; ++k) v = v.set(set_index[k], set_value[k]);
		return v[set_index[0]];
	};

	atl::persistent_array<int64> full(initial);
	for (std::size_t k = 0; k < updates; ++k) full = full.set(set_index[k], set_value[k]);
	BENCHMARK(std::format("persistent_array -- {} reads", reads)) {
		uint64 sum = 0;
		for (auto i : read_index) sum += uint64(full[i]);
		return sum;
	};

	atl::semi_persistent_array<int64> semi(initial);
	for (std::size_t k = 0; k < updates; ++k) semi = semi.set(set_index[k], set_value[k]);
	REQUIRE(full.size() == semi.size());
	for (auto i : read_index) REQUIRE(full[i] == semi[i]);
	BENCHMARK(std::format("semi_persistent_array -- {} reads", reads)) {
		uint64 sum = 0;
		for (auto i : read_index) sum += uint64(semi[i]);
		return sum;
	};
}

namespace {
	/** @brief A node-arena model of reserve_extra: exact or geometric. */
	template<bool IsGeometric>
	std::size_t reserve_extra_by_growth_policy(std::size_t ops, std::size_t nodes_per_op) {
		std::vector<std::array<atl::uint32, 16>> nodes(1);
		for (std::size_t k = 0; k < ops; ++k) {
			std::size_t const need = nodes.size() + nodes_per_op;
			if constexpr (IsGeometric) {
				if (need > nodes.capacity()) nodes.reserve((std::max)(need, 2 * nodes.capacity()));
			} else {
				nodes.reserve(need);
			}
			for (std::size_t j = 0; j < nodes_per_op; ++j) nodes.push_back({atl::uint32(k)});
		}
		return nodes.size();
	}

	/**
	 * @brief A copy of the Radix-16 trie build over int64 with node vectors as arenas:
	 *        `ShouldReserve` sizes every arena and level vector up front,
	 *        otherwise they grow geometrically.
	 */
	template<bool ShouldReserve>
	std::size_t persistent_array_by_build_reservation(std::vector<int64> const &values) {
		constexpr std::size_t radix = 16;
		std::vector<std::array<int64, radix>> leaves(1);
		std::vector<std::array<atl::uint32, radix>> branches(1);
		std::size_t const leaf_count = (values.size() + radix - 1) / radix;
		std::vector<atl::uint32> below;
		if constexpr (ShouldReserve) {
			leaves.reserve(leaf_count + 1);
			branches.reserve(leaf_count / (radix - 1) + 8);
			below.reserve(leaf_count);
		}
		for (std::size_t i = 0; i < values.size(); i += radix) {
			std::array<int64, radix> cell{};
			for (std::size_t c = 0; c < radix && i + c < values.size(); ++c) cell[c] = values[i + c];
			below.push_back(atl::uint32(leaves.size()));
			leaves.push_back(cell);
		}
		while (below.size() > 1) {
			std::vector<atl::uint32> parents;
			if constexpr (ShouldReserve) parents.reserve((below.size() + radix - 1) / radix);
			for (std::size_t i = 0; i < below.size(); i += radix) {
				std::array<atl::uint32, radix> child{};
				for (std::size_t c = 0; c < radix && i + c < below.size(); ++c) child[c] = below[i + c];
				parents.push_back(atl::uint32(branches.size()));
				branches.push_back(child);
			}
			below = std::move(parents);
		}
		return leaves.size() + branches.size() + below[0];
	}
}

// Per-operation exact reservation causes quadratic relocation. The library arena grows a
// past-capacity request geometrically; the model arms isolate the policy (64-byte nodes,
// 7 per op, as a Radix-16 path copy at n = 2^20 makes), at a size where the exact policy's
// quadratic relocation still finishes. The library build reserves every arena and level vector
// up front; the model arms are a copy of the build with and without the reserves.
TEST_CASE("persistent_array arena growth and build reservation", "[persistent_array][race]") {
	std::size_t const n = 1u << 20, updates = 200000;
	std::mt19937_64 rng(3);
	std::vector<int64> initial(n);
	for (auto &x : initial) x = int64(rng());
	std::vector<std::size_t> set_index(updates);
	for (auto &i : set_index) i = rng() % n;

	{
		auto expected = initial;
		for (std::size_t k = 0; k < updates; ++k) expected[set_index[k]] = int64(k);
		for (int policy : {0, 1, 2}) {
			atl::persistent_array<int64> v(initial);
			if (policy == 1) v.reserve(updates);
			for (std::size_t k = 0; k < updates; ++k) {
				if (policy == 0) v.reserve(1);
				v = v.set(set_index[k], int64(k));
			}
			for (std::size_t i = 0; i < n; ++i) REQUIRE(v[i] == expected[i]);
		}
	}

	BENCHMARK("update chain n=2^20, 2e5 sets, by per-update reserve(1)") {
		atl::persistent_array<int64> v(initial);
		for (std::size_t k = 0; k < updates; ++k) {
			v.reserve(1);
			v = v.set(set_index[k], int64(k));
		}
		return v[set_index[0]];
	};
	BENCHMARK("update chain n=2^20, 2e5 sets, by up-front reserve") {
		atl::persistent_array<int64> v(initial);
		v.reserve(updates);
		for (std::size_t k = 0; k < updates; ++k) v = v.set(set_index[k], int64(k));
		return v[set_index[0]];
	};
	BENCHMARK("update chain n=2^20, 2e5 sets, by no reserve") {
		atl::persistent_array<int64> v(initial);
		for (std::size_t k = 0; k < updates; ++k) v = v.set(set_index[k], int64(k));
		return v[set_index[0]];
	};
	for (std::size_t const m : {std::size_t(200'000), std::size_t(1) << 20, std::size_t(1) << 22}) {
		std::vector<int64> const vals(initial.begin(), initial.begin() + std::ptrdiff_t((std::min)(m, n)));
		std::vector<int64> big;
		if (m > n) {
			big.resize(m);
			for (auto &x : big) x = int64(rng());
		}
		auto const &src = m > n ? big : vals;
		REQUIRE(persistent_array_by_build_reservation<true>(src) == persistent_array_by_build_reservation<false>(src));
		std::string const tag = std::format("build n={}", m);
		BENCHMARK(tag + ", library") { return atl::persistent_array<int64>(src).size(); };
		BENCHMARK(tag + ", by build reservation (reserved model)") { return persistent_array_by_build_reservation<true>(src); };
		BENCHMARK(tag + ", by build reservation (geometric model)") { return persistent_array_by_build_reservation<false>(src); };
	}
	REQUIRE(reserve_extra_by_growth_policy<true>(2000, 7) == reserve_extra_by_growth_policy<false>(2000, 7));
	BENCHMARK("arena model, 2000 ops x 7 nodes, by exact growth") { return reserve_extra_by_growth_policy<false>(2000, 7); };
	BENCHMARK("arena model, 2000 ops x 7 nodes, by geometric growth") { return reserve_extra_by_growth_policy<true>(2000, 7); };
	BENCHMARK("arena model, 2e5 ops x 7 nodes, by geometric growth") { return reserve_extra_by_growth_policy<true>(200000, 7); };
}
