#pragma once

/**
 * @file
 * @brief Workloads shared by the persistent-heap tuning and race benchmarks:
 *        random version DAGs, a linear push / pop script, a keep-all-versions
 *        shape. Inputs are
 *        generated outside the timed region; each run returns a checksum.
 */

#include <atl/experimental/ds/meldable_heap_variants.hpp>
#include <atl/experimental/ds/persistent_heap_variants.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace persistent_heap_bench {
	using atl::int64;
	using atl::uint32;
	using atl::uint64;

	/** @brief A class-form persistent heap of int64 the workloads can drive. */
	template<typename Heap>
	concept int64_class_heap = std::default_initializable<Heap> && std::copyable<Heap> && requires(Heap const &h, int64 x) {
		{ h.push(x) } -> std::same_as<Heap>;
		{ h.pop() } -> std::same_as<Heap>;
		{ h.meld(h) } -> std::same_as<Heap>;
		{ h.top() } -> std::convertible_to<int64>;
		{ h.empty() } -> std::convertible_to<bool>;
		{ h.size() } -> std::convertible_to<std::size_t>;
	};

	/** @brief A forest-form persistent heap of int64 (versions are `uint32` ids). */
	template<typename Forest>
	concept int64_forest_heap = std::default_initializable<Forest> && requires(Forest &f, uint32 h, int64 x) {
		{ f.push(h, x) } -> std::same_as<uint32>;
		{ f.pop(h) } -> std::same_as<uint32>;
		{ f.meld(h, h) } -> std::same_as<uint32>;
		{ f.top(h) } -> std::convertible_to<int64>;
		{ f.empty(h) } -> std::convertible_to<bool>;
		{ Forest::empty_heap } -> std::convertible_to<uint32>;
	};

	/** @brief One step of a version DAG: `kind` 0 push `value`, 1 pop, 2 meld with version `other`, 3 add `value` to all. */
	struct dag_step {
		uint32 kind, base, other;
		int64 value;
	};

	/**
	 * @brief A random version DAG of `count` steps, each on uniformly random
	 *        existing versions: push / pop / meld in ratio 2 / 1 / 1 (and, with
	 *        `has_add`, one add-to-all in ten); a meld past `size_cap` elements
	 *        and a pop of an empty version become pushes.
	 */
	inline std::vector<dag_step> make_dag(uint64 seed, std::size_t count, uint64 size_cap, bool has_add) {
		std::mt19937_64 rng(seed);
		std::vector<uint64> size{0};
		std::vector<dag_step> steps;
		steps.reserve(count);
		for (std::size_t i = 0; i < count; ++i) {
			uint32 const base = uint32(rng() % size.size()), other = uint32(rng() % size.size());
			uint32 kind = has_add && rng() % 10 == 0 ? 3 : uint32(std::array{0, 0, 1, 2}[rng() % 4]);
			if (kind == 1 && size[base] == 0) kind = 0;
			if (kind == 2 && size[base] + size[other] > size_cap) kind = 0;
			int64 const value = kind == 3 ? int64(rng() % 1000) : int64(rng() % 1000000000);
			steps.push_back({kind, base, other, value});
			size.push_back(kind == 0 ? size[base] + 1 : kind == 1 ? size[base] - 1
													: kind == 2	  ? size[base] + size[other]
																  : size[base]);
		}
		return steps;
	}

	/** @brief Replay a DAG on class-form `Heap`, keeping every version; returns a checksum of the tops. */
	template<int64_class_heap Heap>
	int64 replay_class(std::vector<dag_step> const &steps) {
		std::vector<Heap> versions;
		versions.reserve(steps.size() + 1);
		versions.emplace_back();
		int64 sum = 0;
		for (dag_step const &s : steps) {
			Heap const &base = versions[s.base];
			switch (s.kind) {
			case 0:
				versions.push_back(base.push(s.value));
				break;
			case 1:
				versions.push_back(base.pop());
				break;
			case 2:
				versions.push_back(base.meld(versions[s.other]));
				break;
			default:
				if constexpr (requires { base.add_to_all(s.value); })
					versions.push_back(base.add_to_all(s.value));
				else
					versions.push_back(base);
			}
			if (!versions.back().empty()) sum += versions.back().top();
		}
		return sum;
	}

	/** @brief Replay a DAG (no add-to-all) on forest-form `Forest`. */
	template<int64_forest_heap Forest>
	int64 replay_forest(std::vector<dag_step> const &steps) {
		Forest forest;
		std::vector<uint32> versions;
		versions.reserve(steps.size() + 1);
		versions.push_back(Forest::empty_heap);
		int64 sum = 0;
		for (dag_step const &s : steps) {
			uint32 const base = versions[s.base];
			uint32 made = s.kind == 0 ? forest.push(base, s.value) : s.kind == 1 ? forest.pop(base)
																				 : forest.meld(base, versions[s.other]);
			versions.push_back(made);
			if (!forest.empty(made)) sum += forest.top(made);
		}
		return sum;
	}

	/** @brief One heap used linearly (the move idiom): `values[i] >= 0` pushes it, a negative entry pops. */
	template<int64_class_heap Heap>
	int64 linear_class(std::vector<int64> const &script) {
		Heap h;
		int64 sum = 0;
		for (int64 x : script) {
			if (x >= 0)
				h = std::move(h).push(x);
			else if (!h.empty()) {
				sum += h.top();
				h = std::move(h).pop();
			}
		}
		return sum + int64(h.size());
	}

	/** @brief The same script on a forest-form heap. */
	template<int64_forest_heap Forest>
	int64 linear_forest(std::vector<int64> const &script) {
		Forest forest;
		uint32 h = Forest::empty_heap;
		int64 sum = 0;
		for (int64 x : script) {
			if (x >= 0)
				h = forest.push(h, x);
			else if (!forest.empty(h)) {
				sum += forest.top(h);
				h = forest.pop(h);
			}
		}
		return sum;
	}

	/** @brief The same script on the non-persistent leftist forest (the cost of persistence). */
	inline int64 linear_ephemeral(std::vector<int64> const &script) {
		atl::experimental::meldable_heap_forest_by_rank_leftist<int64> forest(1);
		int64 sum = 0;
		for (int64 x : script) {
			if (x >= 0)
				forest.push(0, x);
			else if (!forest.empty(0))
				sum += forest.pop(0);
		}
		return sum;
	}

	/** @brief Push every value keeping every version, then pop once from `probes` random kept versions. */
	template<int64_class_heap Heap>
	int64 keep_all_class(std::vector<int64> const &values, std::vector<uint32> const &probes) {
		std::vector<Heap> versions;
		versions.reserve(values.size() + 1);
		versions.emplace_back();
		for (int64 x : values) versions.push_back(versions.back().push(x));
		int64 sum = 0;
		for (uint32 p : probes)
			if (Heap const popped = versions[p].pop(); !popped.empty()) sum += popped.top();
		return sum;
	}

	/** @brief The keep-all shape on a forest-form heap. */
	template<int64_forest_heap Forest>
	int64 keep_all_forest(std::vector<int64> const &values, std::vector<uint32> const &probes) {
		Forest forest;
		std::vector<uint32> versions;
		versions.reserve(values.size() + 1);
		versions.push_back(Forest::empty_heap);
		for (int64 x : values) versions.push_back(forest.push(versions.back(), x));
		int64 sum = 0;
		for (uint32 p : probes)
			if (uint32 const popped = forest.pop(versions[p]); !forest.empty(popped)) sum += forest.top(popped);
		return sum;
	}

	/** @brief The row label of a contender. */
	inline std::string name_of(atl::experimental::persistent_heap_algorithm a) {
		using enum atl::experimental::persistent_heap_algorithm;
		return a == rank_leftist ? "by rank leftist" : (a == weight_biased ? "by weight biased" : "by randomized meld");
	}
}
