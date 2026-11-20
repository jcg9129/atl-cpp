/** @file
 * @brief Functional-graph decomposition, endpoint and ordered-fold throughput; not measured yet.
 * Library peeling is compared with path marking; the library cut-LCA indices with
 * full lifting, chain prefixes and reverse-DFS batches. Shapes are cycles, tails,
 * balanced in-trees with short cycles, stars and relabeled random mappings.
 * Online rows time construction, a complete query batch, and construction plus
 * that batch separately; build and combined rows also include freeing the index
 * they build. Offline rows include preprocessing and output allocation.
 * Workloads vary graph/query sizes, step boundaries and repeated/random starts,
 * with scalar xor and noncommutative affine composition. Row suffixes report
 * untimed requested retained/peak heap bytes and monoid-operation counts.
 * No benchmark record exists, so no CPU/compiler/flags/date or winner is claimed.
 */
#include <atl/experimental/graph/functional_graph_variants.hpp>
#include <atl/graph/functional_graph.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <random>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace g = atl::graph;
namespace e = atl::experimental;
using atl::uint32;
using atl::uint64;

namespace allocation_meter {
	/** @brief Requested allocation bytes, excluding allocator overhead and the input arrays. */
	struct statistics {
		std::size_t retained, peak;
	};

	/** @brief Fixed census ledger; never allocates or adds headers to timed allocations. */
	struct entry {
		void *pointer;
		std::size_t bytes;
	};

	std::array<entry, 4096> ledger{};
	bool is_tracking = false;
	std::size_t live = 0, peak = 0, entries = 0;

	/** @brief Begin an untimed allocation census; not constexpr because it instruments dynamic allocation. */
	void begin() {
		live = peak = entries = 0;
		is_tracking = true;
	}

	/** @brief End census while the result/index is alive; not constexpr because it reads runtime allocation counters. */
	statistics end() {
		is_tracking = false;
		return {live, peak};
	}

	/** @brief Allocate without changing the timed allocation layout; not constexpr (malloc). */
	void *allocate(std::size_t bytes, std::size_t alignment) {
		bytes = (std::ranges::max)(bytes, std::size_t(1));
		void *result = nullptr;
		if (alignment <= alignof(std::max_align_t))
			result = std::malloc(bytes);
		else {
			auto const overhead = alignment - 1 + sizeof(void *);
			if (bytes > (std::numeric_limits<std::size_t>::max)() - overhead) throw std::bad_alloc();
			auto space = bytes + overhead;
			void *const allocation = std::malloc(space);
			if (!allocation) throw std::bad_alloc();
			result = static_cast<void **>(allocation) + 1;
			space -= sizeof(void *);
			if (!std::align(alignment, bytes, result, space)) {
				std::free(allocation);
				throw std::bad_alloc();
			}
			static_cast<void **>(result)[-1] = allocation;
		}
		if (!result) throw std::bad_alloc();
		if (is_tracking) {
			if (entries == ledger.size()) {
				std::free(alignment <= alignof(std::max_align_t) ? result : static_cast<void **>(result)[-1]);
				throw std::bad_alloc();
			}
			ledger[entries++] = {result, bytes};
			live += bytes;
			peak = (std::ranges::max)(peak, live);
		}
		return result;
	}

	/** @brief Release storage, searching the fixed ledger only during an untimed census; not constexpr (free). */
	void release(void *pointer, std::size_t alignment = alignof(std::max_align_t)) noexcept {
		if (!pointer) return;
		if (is_tracking)
			for (std::size_t i = entries; i-- > 0;)
				if (ledger[i].pointer == pointer) {
					live -= ledger[i].bytes;
					ledger[i].pointer = nullptr;
					break;
				}
		std::free(alignment <= alignof(std::max_align_t) ? pointer : static_cast<void **>(pointer)[-1]);
	}
}

/** @brief Instrument this benchmark binary's allocations; not constexpr (malloc). */
void *operator new(std::size_t n) { return allocation_meter::allocate(n, alignof(std::max_align_t)); }

/** @brief Instrument array allocations; not constexpr (malloc). */
void *operator new[](std::size_t n) { return allocation_meter::allocate(n, alignof(std::max_align_t)); }

/** @brief Instrument over-aligned allocations; not constexpr (malloc). */
void *operator new(std::size_t n, std::align_val_t alignment) { return allocation_meter::allocate(n, std::size_t(alignment)); }

/** @brief Instrument over-aligned arrays; not constexpr (malloc). */
void *operator new[](std::size_t n, std::align_val_t alignment) { return allocation_meter::allocate(n, std::size_t(alignment)); }

/** @brief Release an instrumented allocation; not constexpr (free). */
void operator delete(void *p) noexcept { allocation_meter::release(p); }

/** @brief Release an instrumented array; not constexpr (free). */
void operator delete[](void *p) noexcept { allocation_meter::release(p); }

/** @brief Release a sized allocation; not constexpr (free). */
void operator delete(void *p, std::size_t) noexcept { allocation_meter::release(p); }

/** @brief Release a sized array; not constexpr (free). */
void operator delete[](void *p, std::size_t) noexcept { allocation_meter::release(p); }

/** @brief Release an aligned allocation; not constexpr (free). */
void operator delete(void *p, std::align_val_t alignment) noexcept { allocation_meter::release(p, std::size_t(alignment)); }

/** @brief Release an aligned array; not constexpr (free). */
void operator delete[](void *p, std::align_val_t alignment) noexcept { allocation_meter::release(p, std::size_t(alignment)); }

/** @brief Release a sized aligned allocation; not constexpr (free). */
void operator delete(void *p, std::size_t, std::align_val_t alignment) noexcept { allocation_meter::release(p, std::size_t(alignment)); }

/** @brief Release a sized aligned array; not constexpr (free). */
void operator delete[](void *p, std::size_t, std::align_val_t alignment) noexcept { allocation_meter::release(p, std::size_t(alignment)); }

namespace {
	/** @brief Allocation and operation census encoded into record.sh's unique row key. */
	[[nodiscard]] std::string census(allocation_meter::statistics stats, uint64 operations = 0) {
		return " retained=" + std::to_string(stats.retained) + "B peak=" + std::to_string(stats.peak)
			 + "B ops=" + std::to_string(operations);
	}

	/** @brief Noncommutative affine transformations modulo a prime. */
	struct affine {
		uint64 a, b;
		bool operator==(affine const &) const = default;
	};

	/** @brief Untimed counting is disabled during time measurements. */
	struct compose {
		uint64 *count = nullptr;

		affine operator()(affine x, affine y) const {
			if (count) ++*count;
			return {(x.a * y.a) % 1009, (x.b * y.a + y.b) % 1009};
		}
	};

	/** @brief Scalar xor monoid with optional operation census. */
	struct xor_operation {
		uint64 *count = nullptr;

		uint64 operator()(uint64 x, uint64 y) const {
			if (count) ++*count;
			return x ^ y;
		}
	};

	template<typename Index>
	concept endpoint_index = requires(Index const &index, uint32 v, uint64 steps) {
		{ index.successor(v, steps) } -> std::same_as<uint32>;
		{ index.size() } -> std::convertible_to<std::size_t>;
	};

	/** @brief Common endpoint output work for every online engine. */
	template<endpoint_index Index>
	[[nodiscard]] uint64 endpoints(Index const &index, std::vector<g::functional_graph_successor_query> const &queries) {
		uint64 checksum = 0;
		for (auto const &[v, k] : queries) checksum += index.successor(v, k);
		return checksum;
	}

	[[nodiscard]] uint64 digest(uint64 value) { return value; }

	[[nodiscard]] uint64 digest(affine value) { return value.a + value.b; }

	template<typename Index>
	concept walk_index = requires(Index const &index, uint32 v, uint64 steps) {
		{ index.size() } -> std::convertible_to<std::size_t>;
		{ index.walk(v, steps).vertex } -> std::convertible_to<uint32>;
		{ digest(index.walk(v, steps).fold) } -> std::same_as<uint64>;
	};

	/** @brief Common ordered-fold output work. */
	template<walk_index Index>
	[[nodiscard]] uint64 walks(Index const &index, std::vector<g::functional_graph_successor_query> const &queries) {
		uint64 checksum = 0;
		for (auto const &[v, k] : queries) {
			auto const result = index.walk(v, k);
			checksum += result.vertex + digest(result.fold);
		}
		return checksum;
	}

	/** @brief Four design shapes, with a star extreme for the balanced-in-tree shape; relabel to disrupt locality. */
	[[nodiscard]] std::vector<uint32> make_graph(uint32 n, unsigned shape) {
		std::mt19937_64 rng(719 + shape);
		std::vector<uint32> successors(n), permutation(n);
		for (uint32 v = 0; v < n; ++v) permutation[v] = v;
		std::ranges::shuffle(permutation, rng);
		for (uint32 v = 0; v < n; ++v) {
			uint32 to = 0;
			if (shape == 0)
				to = (v + 1) % n;
			else if (shape == 1)
				to = (std::ranges::min)(v + 1, n - 1);
			else if (shape == 2) {
				uint32 const group = v / 256, local = v % 256, base = group * 256;
				to = local < 3 ? base + (local + 1) % 3 : base + (local - 1) / 2;
			} else if (shape == 3)
				to = 0;
			else
				to = uint32(rng() % n);
			successors[permutation[v]] = permutation[to];
		}
		return successors;
	}

	/** @brief Small, tail/cycle boundary and large-step queries, with uniform or repeated starts. */
	[[nodiscard]] std::vector<g::functional_graph_successor_query> make_queries(std::vector<uint32> const &successors,
																				std::size_t q, unsigned mode, bool has_repeated_start) {
		std::mt19937_64 rng(319);
		auto const metadata = g::functional_graph_decomposition_of(successors);
		std::vector<g::functional_graph_successor_query> queries;
		queries.reserve(q);
		for (std::size_t i = 0; i < q; ++i) {
			auto const v = has_repeated_start ? 0u : uint32(rng() % successors.size());
			uint64 k = 0;
			if (mode == 0)
				k = rng() % 33;
			else if (mode == 1) {
				uint64 const boundary = i % 2 ? metadata.depth_to_cycle[v] : metadata.cycles[metadata.cycle_id[v]].size();
				k = boundary + (i % 3) - (boundary > 0 ? 1 : 0);
			} else
				k = i % 2 ? g::maximum_functional_graph_steps : (uint64(1) << (32 + i % 28)) + (i % 3) - 1;
			queries.push_back({v, k});
		}
		return queries;
	}

	/** @brief Race build, query and combined endpoint costs, with untimed actual allocation census. */
	template<typename Build, endpoint_index Expected>
		requires std::invocable<Build &> && endpoint_index<std::invoke_result_t<Build &>>
	void race_endpoints(std::string const &tag, Build build, std::vector<g::functional_graph_successor_query> const &queries, Expected const &expected) {
		allocation_meter::begin();
		auto const index = build();
		auto const bytes = allocation_meter::end();
		auto const stats = census(bytes);
		for (auto const &[start, steps] : queries) REQUIRE(index.successor(start, steps) == expected.successor(start, steps));
		BENCHMARK(tag + " build" + stats) { return build().size(); };
		BENCHMARK(tag + " queries" + stats) { return endpoints(index, queries); };
		BENCHMARK(tag + " build+queries" + stats) { return endpoints(build(), queries); };
	}

	/** @brief Race walk-fold engines with census operations separated from timed operations. */
	template<typename Build, walk_index Expected>
		requires std::invocable<Build &, uint64 *> && walk_index<std::invoke_result_t<Build &, uint64 *>>
			  && requires(std::invoke_result_t<Build &, uint64 *> const &index, Expected const &expected) {
					 { index.walk(0, 0).fold == expected.walk(0, 0).fold } -> std::convertible_to<bool>;
				 }
	void race_walks(std::string const &tag, Build build, std::vector<g::functional_graph_successor_query> const &queries, Expected const &expected) {
		uint64 operations = 0;
		allocation_meter::begin();
		auto const measured = build(&operations);
		auto const bytes = allocation_meter::end();
		auto const build_ops = operations;
		operations = 0;
		auto const checksum = walks(measured, queries);
		auto const query_ops = operations;
		auto const index = build(nullptr);
		REQUIRE(walks(index, queries) == checksum);
		for (auto const &[start, steps] : queries) {
			auto const result = index.walk(start, steps), want = expected.walk(start, steps);
			REQUIRE(result.vertex == want.vertex);
			REQUIRE(result.fold == want.fold);
		}
		BENCHMARK(tag + " build" + census(bytes, build_ops)) { return build(nullptr).size(); };
		BENCHMARK(tag + " queries" + census(bytes, query_ops)) { return walks(index, queries); };
		BENCHMARK(tag + " build+queries" + census(bytes, build_ops + query_ops)) { return walks(build(nullptr), queries); };
	}

	/** @brief Walk-fold online and reverse-DFS batch race for one monoid/workload. */
	template<std::copyable Value, typename Operation>
		requires atl::binary_operation<Operation, Value> && std::copyable<Operation> && std::constructible_from<Operation, uint64 *> && std::equality_comparable<Value>
			  && requires(Value const &value) { { digest(value) } -> std::same_as<uint64>; }
	void race_monoid(std::string const &tag, std::vector<uint32> const &successors, std::vector<Value> const &values,
					 Value identity, std::vector<g::functional_graph_successor_query> const &queries) {
		auto const public_build = [&](uint64 *count) { return g::functional_graph_walk_index<Value, Operation>(successors, values, identity, Operation{count}); };
		auto const lift_build = [&](uint64 *count) { return e::functional_graph_walk_index_by_lifting<Value, Operation>(successors, values, identity, Operation{count}); };
		auto const prefix_build = [&](uint64 *count) { return e::functional_graph_walk_index_by_chain_prefix<Value, Operation>(successors, values, identity, Operation{count}); };
		auto const expected = public_build(nullptr);
		race_walks(tag + " library", public_build, queries, expected);
		race_walks(tag + " by lifting", lift_build, queries, expected);
		race_walks(tag + " by chain prefix", prefix_build, queries, expected);
		uint64 operations = 0;
		allocation_meter::begin();
		auto const output = e::functional_graph_walks_by_reverse_dfs(successors, values, queries, identity, Operation{&operations});
		auto const stats = allocation_meter::end();
		for (std::size_t i = 0; i < queries.size(); ++i) {
			auto const result = expected.walk(queries[i].start, queries[i].steps);
			REQUIRE(result.vertex == output[i].vertex);
			REQUIRE(result.fold == output[i].fold);
		}
		BENCHMARK(tag + " by reverse dfs batch" + census(stats, operations)) {
			auto const result = e::functional_graph_walks_by_reverse_dfs(successors, values, queries, identity, Operation{nullptr});
			uint64 checksum = 0;
			for (auto const &walk : result) checksum += walk.vertex + digest(walk.fold);
			return checksum;
		};
	}
}

TEST_CASE("functional graph engine comparison", "[functional_graph]") {
	std::array<std::string, 5> const shapes{"cycle", "tail-loop", "balanced-short-cycles", "star", "random-mapping"};
	for (uint32 const n : {4096u, 65536u}) {
		for (unsigned shape = 0; shape < shapes.size(); ++shape) {
			auto const successors = make_graph(n, shape);
			auto const label = shapes[shape] + " n=" + std::to_string(n);
			allocation_meter::begin();
			auto const peel = g::functional_graph_decomposition_of(successors);
			auto const peel_bytes = allocation_meter::end();
			allocation_meter::begin();
			auto const mark = e::functional_graph_decomposition_by_marking(successors);
			auto const mark_bytes = allocation_meter::end();
			REQUIRE(peel.cycle_id == mark.cycle_id);
			REQUIRE(peel.cycle_position == mark.cycle_position);
			REQUIRE(peel.cycle_root == mark.cycle_root);
			REQUIRE(peel.depth_to_cycle == mark.depth_to_cycle);
			REQUIRE(peel.cycles.members == mark.cycles.members);
			REQUIRE(peel.cycles.offsets == mark.cycles.offsets);
			BENCHMARK(label + " decomposition library" + census(peel_bytes)) { return g::functional_graph_decomposition_of(successors).cycles.members.size(); };
			BENCHMARK(label + " decomposition by marking" + census(mark_bytes)) { return e::functional_graph_decomposition_by_marking(successors).cycles.members.size(); };
			std::vector<uint64> scalar(n);
			std::vector<affine> transforms(n);
			for (uint32 v = 0; v < n; ++v) {
				scalar[v] = uint64(v) * 2311;
				transforms[v] = {v % 1009, (v + 1) % 1009};
			}
			for (std::size_t q : {std::size_t(n / 4), std::size_t(n) * 4})
				for (unsigned mode = 0; mode < 3; ++mode)
					for (bool const has_repeated_start : {false, true}) {
						auto const queries = make_queries(successors, q, mode, has_repeated_start);
						auto const tag = label + " q=" + std::to_string(q) + " k=" + std::array<std::string, 3>{"small", "boundary", "large"}[mode]
									   + (has_repeated_start ? " repeated" : " random-starts");
						auto const expected = g::functional_graph_successor_index(successors);
						race_endpoints(tag + " endpoints library", [&] { return g::functional_graph_successor_index(successors); }, queries, expected);
						race_endpoints(tag + " endpoints by lifting", [&] { return e::functional_graph_successor_index_by_lifting(successors); }, queries, expected);
						allocation_meter::begin();
						auto const output = e::functional_graph_successors_by_reverse_dfs(successors, queries);
						auto const stats = allocation_meter::end();
						for (std::size_t i = 0; i < queries.size(); ++i) REQUIRE(output[i] == expected.successor(queries[i].start, queries[i].steps));
						BENCHMARK(tag + " endpoints by reverse dfs batch" + census(stats)) {
							auto const result = e::functional_graph_successors_by_reverse_dfs(successors, queries);
							return std::accumulate(result.begin(), result.end(), uint64(0));
						};
						race_monoid<uint64, xor_operation>(tag + " xor", successors, scalar, 0, queries);
						race_monoid<affine, compose>(tag + " affine", successors, transforms, affine{1, 0}, queries);
					}
		}
	}
}

TEST_CASE("functional graph allocation census preserves alignment", "[functional_graph]") {
	allocation_meter::begin();
	void *const allocation = allocation_meter::allocate(65, 64);
	bool const is_aligned = reinterpret_cast<std::uintptr_t>(allocation) % 64 == 0;
	auto const retained = allocation_meter::end();
	allocation_meter::release(allocation, 64);
	REQUIRE(is_aligned);
	REQUIRE(retained.retained == 65);
	REQUIRE(retained.peak == 65);
}
