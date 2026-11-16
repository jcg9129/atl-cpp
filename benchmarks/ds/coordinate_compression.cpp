/**
 * @file
 * Race between `atl::compress` and `coordinate_compression::ranks` (the library)
 * and compression by pair sort, index sort, radix digit widths, or fixed keyed
 * sort; ranks contenders use independent searches, sorted merge, rank loops,
 * galloping, or a separate sortedness pre-scan. The library keeps a sorted prefix
 * when a query descent occurs. Loops measure batch throughput, including result
 * allocation and rank checksums, on random, duplicate-heavy, sorted, mostly sorted,
 * small/sparse/oversized batches; rotated rows cycle through 16 inputs.
 * Result (Xeon Platinum 8168, GCC 15.2, `-O2` and `-O2 -march=native`,
 * 2026-11-03): at n=4096 random the recorded library takes 0.075/0.073 ms
 * versus 0.207/0.210 ms by keyed comparison sort. Mostly sorted m=1e6,
 * q=4e6 takes 260/266 ms by sortedness pre-scan versus 58.5/56.0 ms by
 * fused sortedness merge. Record-to-current mappings: docs/coordinate-compression.md.
 */
#include <atl/ds/coordinate_compression.hpp>
#include <atl/experimental/ds/coordinate_compression_variants.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/radix_sort.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	std::vector<int64> make_values(std::size_t n, std::size_t distinct, uint64 seed) {
		std::mt19937_64 rng(seed);
		std::vector<int64> v(n);
		for (auto &x : v) x = int64(rng() % distinct) * 7919 - 1'000'000'000;
		return v;
	}

	uint64 sum_ranks(std::vector<uint32> const &r) {
		uint64 acc = 0;
		for (uint32 x : r) acc += x;
		return acc;
	}

	/** @brief Compute bulk lower-bound ranks by calling the library rank once per query. */
	std::vector<uint32> ranks_by_rank_loop(atl::coordinate_compression<int64> const &cc, std::vector<int64> const &q) {
		std::vector<uint32> out;
		out.reserve(q.size());
		for (int64 x : q) out.push_back(uint32(cc.rank(x)));
		return out;
	}

	/**
	 * @brief The library int64 `compress` with its size dispatch replaced by one fixed sort: radix11
	 *        (`IsRadix`) or the comparison sort of the same sign-flipped keyed records.
	 */
	template<bool IsRadix>
	atl::compressed<int64> compress_by_keyed_sort(std::vector<int64> const &values) {
		constexpr uint64 sign_flip = uint64(1) << 63;
		std::vector<atl::detail::keyed_position<uint64>> keyed;
		keyed.reserve(values.size());
		for (std::size_t i = 0; i < values.size(); ++i) keyed.push_back({uint64(values[i]) ^ sign_flip, uint32(i)});
		if constexpr (IsRadix)
			atl::detail::radix_sort<11>(keyed);
		else
			std::ranges::sort(keyed, {}, &atl::detail::keyed_position<uint64>::key);
		std::vector<uint32> ranks(keyed.size());
		std::vector<int64> universe;
		for (std::size_t i = 0; i < keyed.size(); ++i) {
			if (i == 0 || keyed[i - 1].key != keyed[i].key) universe.push_back(int64(keyed[i].key ^ sign_flip));
			ranks[keyed[i].position] = uint32(universe.size() - 1);
		}
		return {atl::coordinate_compression<int64>(atl::sorted_unique, std::move(universe)), std::move(ranks)};
	}

	/** @brief The library merge path forced: the merge walk for sorted queries, the (query, position) sort and merge for random ones. */
	std::vector<uint32> ranks_by_forced_merge(std::vector<int64> const &universe, std::vector<int64> const &queries, bool const is_sorted) {
		auto comp = std::ranges::less{};
		std::vector<uint32> ranks;
		if (is_sorted) {
			ranks.reserve(queries.size());
			atl::detail::ranks_by_merge(universe, comp, queries, ranks);
		} else {
			atl::detail::ranks_by_sorted_merge(universe, comp, queries, ranks);
		}
		return ranks;
	}

	/** @brief Which query batches @ref ranks_dispatch_rows builds for each query count. */
	struct dispatch_options {
		bool has_sorted_rows = true;		 ///< Also build the sorted batch.
		bool has_mostly_sorted_rows = false; ///< Also build a batch sorted for its first 90%, random after.
		std::size_t inputs = 1;				 ///< Query vectors per shape; each sample takes the next, cyclically.
	};

	/**
	 * @brief Compare all five ranks paths on exact-size universes and deterministic query batches.
	 *
	 * With `options.inputs > 1` every shape has that many distinct query vectors (seeds 17, 18, ...) that the
	 * samples of a row rotate through, so a branch predictor cannot learn one input; every path is checked
	 * against `cc.ranks()` on all of them before timing. One input keeps the original single-input rows.
	 */
	void ranks_dispatch_rows(std::size_t const m, std::vector<std::size_t> const &counts, dispatch_options const options = {}) {
		enum class query_shape { random,
								 sorted,
								 mostly_sorted };
		std::vector<int64> universe(m);
		for (std::size_t i = 0; i < m; ++i) universe[i] = int64(i) * 7919 - 1'000'000'000;
		atl::coordinate_compression<int64> const cc(atl::sorted_unique, universe);
		REQUIRE(cc.size() == m);
		for (std::size_t const q : counts) {
			for (query_shape const shape : {query_shape::random, query_shape::sorted, query_shape::mostly_sorted}) {
				if (shape == query_shape::sorted && !options.has_sorted_rows) continue;
				if (shape == query_shape::mostly_sorted && !options.has_mostly_sorted_rows) continue;
				bool const is_sorted = shape == query_shape::sorted;
				std::vector<std::vector<int64>> batches;
				for (std::size_t k = 0; k < options.inputs; ++k) {
					auto queries = make_values(q, m + 1, 17 + k);
					if (is_sorted) std::ranges::sort(queries);
					if (shape == query_shape::mostly_sorted)
						std::ranges::sort(queries.begin(), queries.begin() + std::ptrdiff_t(q - q / 10));
					batches.push_back(std::move(queries));
				}
				auto comp = std::ranges::less{};
				for (auto const &queries : batches) {
					auto const expected = cc.ranks(queries);
					REQUIRE(ranks_by_rank_loop(cc, queries) == expected);
					std::vector<uint32> check;
					check.reserve(q);
					atl::detail::ranks_by_search(universe, comp, queries, check);
					REQUIRE(check == expected);
					REQUIRE(ranks_by_forced_merge(universe, queries, is_sorted) == expected);
					REQUIRE(atl::experimental::ranks_by_sortedness_prescan(universe, comp, queries) == expected);
				}
				std::size_t next = 0;
				auto const current = [&]() -> std::vector<int64> const & {
					auto const &queries = batches[next];
					next = next + 1 == batches.size() ? 0 : next + 1;
					return queries;
				};
				std::string const tag = "m=" + std::to_string(m) + " q=" + std::to_string(q)
									  + (is_sorted ? " sorted" : shape == query_shape::random ? " random"
																							  : " mostly sorted")
									  + (options.inputs > 1 ? " rotated | " : " | ");
				BENCHMARK(tag + "ranks() (library)") { return sum_ranks(cc.ranks(current())); };
				BENCHMARK(tag + "ranks by search") {
					std::vector<uint32> out;
					out.reserve(q);
					atl::detail::ranks_by_search(universe, comp, current(), out);
					return sum_ranks(out);
				};
				BENCHMARK(tag + (is_sorted ? "ranks by merge" : "ranks by sorted merge")) {
					return sum_ranks(ranks_by_forced_merge(universe, current(), is_sorted));
				};
				BENCHMARK(tag + "ranks by rank loop") { return sum_ranks(ranks_by_rank_loop(cc, current())); };
				BENCHMARK(tag + "ranks by sortedness prescan") {
					return sum_ranks(atl::experimental::ranks_by_sortedness_prescan(universe, comp, current()));
				};
			}
		}
	}
}

TEST_CASE("coordinate_compression: ranks of a batch", "[coordinate_compression]") {
	for (std::size_t const m : {std::size_t(100'000), std::size_t(1'000'000), std::size_t(5'000'000)}) {
		auto const base = make_values(m, m * 4, 1);
		atl::coordinate_compression<int64> const cc(base);
		std::string const tag = "universe " + std::to_string(cc.size()) + " | ";

		struct shape {
			std::string name;
			std::size_t count;
			bool is_sorted;
		};

		for (auto const &sh : {shape{"q=m random", m, false}, shape{"q=m sorted", m, true},
							   shape{"q=m/4 sorted", m / 4, true}, shape{"q=m/16 sorted", m / 16, true},
							   shape{"q=m/64 sorted", m / 64, true}}) {
			auto queries = make_values(sh.count, m * 4, 2);
			if (sh.is_sorted) std::ranges::sort(queries);
			auto const want = ranks_by_rank_loop(cc, queries);
			REQUIRE(cc.ranks(queries) == want);
			std::vector<uint32> check;
			auto comp = std::ranges::less{};
			std::vector<int64> const universe(cc.begin(), cc.end());
			atl::detail::ranks_by_search(universe, comp, queries, check);
			REQUIRE(check == want);
			if (sh.is_sorted) {
				check.clear();
				atl::detail::ranks_by_merge(universe, comp, queries, check);
				REQUIRE(check == want);
			}
			std::string const s = tag + sh.name + " | ";
			BENCHMARK(s + "ranks by rank loop") { return sum_ranks(ranks_by_rank_loop(cc, queries)); };
			BENCHMARK(s + "ranks()") { return sum_ranks(cc.ranks(queries)); };
			BENCHMARK(s + "ranks by search") {
				std::vector<uint32> out;
				out.reserve(queries.size());
				atl::detail::ranks_by_search(universe, comp, queries, out);
				return sum_ranks(out);
			};
			if (sh.is_sorted) {
				BENCHMARK(s + "ranks by merge") {
					std::vector<uint32> out;
					out.reserve(queries.size());
					atl::detail::ranks_by_merge(universe, comp, queries, out);
					return sum_ranks(out);
				};
			}
		}
	}
}

TEST_CASE("coordinate_compression: build and rank every input", "[coordinate_compression]") {
	for (std::size_t const n : {std::size_t(100'000), std::size_t(1'000'000), std::size_t(5'000'000)}) {
		for (std::size_t const distinct : {n * 4, std::size_t(1000)}) {
			auto const v = make_values(n, distinct, 3);
			atl::coordinate_compression<int64> const ref(v);
			auto const want = ranks_by_rank_loop(ref, v);
			auto const a = atl::compress(v);
			auto const b = atl::experimental::compress_by_index_sort(v);
			if (a.ranks != want || b.ranks != want || a.universe.size() != ref.size() || b.universe.size() != ref.size())
				FAIL("compress contenders disagree with build + rank()");
			std::string const s = "n=" + std::to_string(n) + (distinct == 1000 ? " 1000 distinct" : " random") + " | ";
			BENCHMARK(s + "build + ranks by rank loop") {
				atl::coordinate_compression<int64> const cc(v);
				return sum_ranks(ranks_by_rank_loop(cc, v));
			};
			REQUIRE(ref.ranks(v) == want);
			BENCHMARK(s + "build + ranks()") {
				atl::coordinate_compression<int64> const cc(v);
				return sum_ranks(cc.ranks(v));
			};
			BENCHMARK(s + "compress (library)") { return sum_ranks(atl::compress(v).ranks); };
			BENCHMARK(s + "compress by index sort") { return sum_ranks(atl::experimental::compress_by_index_sort(v).ranks); };
		}
	}
}

TEST_CASE("coordinate_compression ranks race", "[coordinate_compression]") {
	for (std::size_t const m : {std::size_t(100'000), std::size_t(1'000'000), std::size_t(5'000'000)}) {
		auto const base = make_values(m, m * 4, 1);
		atl::coordinate_compression<int64> const cc(base);
		std::vector<int64> const universe(cc.begin(), cc.end());
		std::string const tag = "universe " + std::to_string(cc.size()) + " | ";

		struct shape {
			std::string name;
			std::size_t count;
			bool is_sorted;
		};

		for (auto const &sh : {shape{"q=m sorted", m, true}, shape{"q=m/4 sorted", m / 4, true},
							   shape{"q=m/16 sorted", m / 16, true}, shape{"q=m/64 sorted", m / 64, true},
							   shape{"q=m/256 sorted", m / 256, true}, shape{"q=m/1024 sorted", m / 1024, true},
							   shape{"q=m random", m, false}, shape{"q=m/4 random", m / 4, false},
							   shape{"q=m/16 random", m / 16, false}}) {
			auto queries = make_values(sh.count, m * 4, 2);
			if (sh.is_sorted) std::ranges::sort(queries);
			atl::experimental::ranks_policy const contender = sh.is_sorted ? atl::experimental::ranks_policy{0, true}
																		   : atl::experimental::ranks_policy{};
			REQUIRE(cc.ranks(queries) == ranks_by_rank_loop(cc, queries));
			if (atl::experimental::ranks_by_search_or_gallop(universe, std::ranges::less{}, queries, contender) != ranks_by_rank_loop(cc, queries))
				FAIL("contender disagrees with rank()");
			std::string const s = tag + sh.name + " | ";
			BENCHMARK(s + "ranks (library)") { return sum_ranks(cc.ranks(queries)); };
			BENCHMARK(s + (sh.is_sorted ? "ranks by search or gallop" : "ranks by search or gallop")) {
				return sum_ranks(atl::experimental::ranks_by_search_or_gallop(universe, std::ranges::less{}, queries, contender));
			};
		}
	}
}

TEST_CASE("coordinate_compression compress race", "[coordinate_compression]") {
	for (std::size_t const n : {std::size_t(100'000), std::size_t(1'000'000), std::size_t(5'000'000)}) {
		for (std::size_t const distinct : {n * 4, std::size_t(1000)}) {
			auto const v = make_values(n, distinct, 3);
			auto const want = atl::compress(v).ranks;
			if (atl::experimental::compress_by_radix_sort<8>(v).ranks != want
				|| atl::experimental::compress_by_pair_sort(v).ranks != want
				|| atl::experimental::compress_by_radix_sort<16>(v).ranks != want)
				FAIL("radix contenders disagree with compress");
			std::string const s = "n=" + std::to_string(n) + (distinct == 1000 ? " 1000 distinct" : " random") + " | ";
			BENCHMARK(s + "compress radix11 (library)") { return sum_ranks(atl::compress(v).ranks); };
			BENCHMARK(s + "compress by radix sort (8 bits)") { return sum_ranks(atl::experimental::compress_by_radix_sort<8>(v).ranks); };
			BENCHMARK(s + "compress by pair sort") { return sum_ranks(atl::experimental::compress_by_pair_sort(v).ranks); };
			BENCHMARK(s + "compress by radix sort (16 bits)") { return sum_ranks(atl::experimental::compress_by_radix_sort<16>(v).ranks); };
		}
	}
}

TEST_CASE("coordinate_compression compress dispatch crossover", "[coordinate_compression]") {
	std::mt19937_64 rng(19);
	for (std::size_t const n : {std::size_t(1024), std::size_t(2048), std::size_t(3072), std::size_t(4096), std::size_t(16384), std::size_t(32768), std::size_t(65536), std::size_t(131072)}) {
		for (std::size_t const distinct : {n * 4, std::size_t(1000)}) {
			auto values = make_values(n, distinct, 3);
			if (distinct == 1000) {
				for (std::size_t i = 0; i < distinct; ++i) values[i] = int64(i) * 7919 - 1'000'000'000;
				std::ranges::shuffle(values, rng);
			}
			auto const expected = atl::compress(values);
			if (distinct == 1000) REQUIRE(expected.universe.size() == distinct);
			for (auto const &contender : {compress_by_keyed_sort<true>(values), compress_by_keyed_sort<false>(values),
										  atl::experimental::compress_by_pair_sort(values)}) {
				REQUIRE(contender.ranks == expected.ranks);
				REQUIRE(std::ranges::equal(contender.universe, expected.universe));
			}
			std::string const tag = "n=" + std::to_string(n) + (distinct == 1000 ? " 1000 distinct | " : " random | ");
			BENCHMARK(tag + "compress (library)") { return sum_ranks(atl::compress(values).ranks); };
			BENCHMARK(tag + "compress by keyed sort (radix11)") { return sum_ranks(compress_by_keyed_sort<true>(values).ranks); };
			BENCHMARK(tag + "compress by keyed sort (comparison)") { return sum_ranks(compress_by_keyed_sort<false>(values).ranks); };
			BENCHMARK(tag + "compress by pair sort") { return sum_ranks(atl::experimental::compress_by_pair_sort(values).ranks); };
		}
	}
}

TEST_CASE("coordinate_compression compress crossover rotated inputs", "[coordinate_compression]") {
	constexpr std::size_t inputs = 16;
	std::mt19937_64 rng(23);
	for (std::size_t const n : {std::size_t(1024), std::size_t(1536), std::size_t(2048), std::size_t(3072), std::size_t(4096), std::size_t(8192)}) {
		for (std::size_t const distinct : {n * 4, std::size_t(1000)}) {
			std::vector<std::vector<int64>> batches;
			for (std::size_t k = 0; k < inputs; ++k) {
				auto values = make_values(n, distinct, 3 + k);
				if (distinct == 1000) {
					for (std::size_t i = 0; i < distinct; ++i) values[i] = int64(i) * 7919 - 1'000'000'000;
					std::ranges::shuffle(values, rng);
				}
				auto const expected = atl::compress(values);
				if (distinct == 1000) REQUIRE(expected.universe.size() == distinct);
				for (auto const &contender : {compress_by_keyed_sort<true>(values), compress_by_keyed_sort<false>(values)}) {
					REQUIRE(contender.ranks == expected.ranks);
					REQUIRE(std::ranges::equal(contender.universe, expected.universe));
				}
				batches.push_back(std::move(values));
			}
			std::size_t next = 0;
			auto const current = [&]() -> std::vector<int64> const & {
				auto const &values = batches[next];
				next = next + 1 == batches.size() ? 0 : next + 1;
				return values;
			};
			std::string const tag = "n=" + std::to_string(n) + (distinct == 1000 ? " 1000 distinct rotated | " : " random rotated | ");
			BENCHMARK(tag + "compress (library)") { return sum_ranks(atl::compress(current()).ranks); };
			BENCHMARK(tag + "compress by keyed sort (radix11)") { return sum_ranks(compress_by_keyed_sort<true>(current()).ranks); };
			BENCHMARK(tag + "compress by keyed sort (comparison)") { return sum_ranks(compress_by_keyed_sort<false>(current()).ranks); };
		}
	}
}

TEST_CASE("coordinate_compression ranks small and sparse dispatch", "[coordinate_compression]") {
	for (std::size_t const m : {std::size_t(1024), std::size_t(4096), std::size_t(16384), std::size_t(65536)})
		ranks_dispatch_rows(m, {m, m / 16});
	ranks_dispatch_rows(1'000'000, {1'000'000 / 64, 1'000'000 / 256, 1'000'000 / 1024}, {.has_sorted_rows = false});
}

TEST_CASE("coordinate_compression ranks oversized queries m=100000", "[coordinate_compression]") {
	ranks_dispatch_rows(100'000, {200'000, 400'000, 1'600'000});
}

TEST_CASE("coordinate_compression ranks oversized queries m=1000000", "[coordinate_compression]") {
	ranks_dispatch_rows(1'000'000, {2'000'000, 4'000'000, 16'000'000}, {.has_mostly_sorted_rows = true});
}

TEST_CASE("coordinate_compression ranks small universe rotated inputs", "[coordinate_compression]") {
	for (std::size_t const m : {std::size_t(64), std::size_t(256), std::size_t(1024), std::size_t(2048), std::size_t(4096)}) {
		std::vector<std::size_t> counts{m / 64 + (m % 64 != 0), m / 16, m / 4, m};
		std::ranges::sort(counts);
		counts.erase(std::ranges::unique(counts).begin(), counts.end());
		std::erase(counts, std::size_t(0));
		ranks_dispatch_rows(m, counts, {.inputs = 16});
	}
}
