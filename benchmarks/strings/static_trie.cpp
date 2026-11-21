/**
 * @file
 * @brief Frozen-trie build/query throughput: library integer keys and chain shortcuts,
 * by slice comparison (indirect per-level sorting), and dynamic hashed insertion.
 * Shapes: U=1e6 1e5x10, sigma=26 1e5x(5-15), prefix500 2e4, identical 1e5x10.
 * Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off, -O2/native,
 * 2026-10-15: U=1e6 build library 28.5/28.9 ms, slice comparison 53.5/54.2,
 * dynamic insertion 41.0/40.8. Prefix500 library 82.5/84.3 versus 598/602 ms
 * by slice comparison. Full table: docs/static-trie.md.
 */
#include "baseline/static_trie.hpp"
#include <atl/strings/static_trie.hpp>
#include <atl/strings/trie.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using atl::uint32;
using atl::uint64;

TEST_CASE("static_trie build", "[static_trie]") {
	constexpr std::size_t alphabet = 1'000'000;
	std::mt19937_64 rng(1);
	auto const race = [&](std::string const &shape, std::vector<std::vector<uint32>> const &words) {
		atl::static_trie<alphabet> const library(words);
		atl::baseline::static_trie_by_slice_comparison<alphabet> const by_comparison(words);
		REQUIRE(library.node_count() == by_comparison.node_count());
		for (uint32 v = 0; v < library.node_count(); ++v) {
			REQUIRE(library.prefix_count(v) == by_comparison.prefix_count(v));
			REQUIRE(library.word_interval(v) == by_comparison.word_interval(v));
		}
		auto const build_row = [&]<typename Build>(std::string const &method, Build build)
			requires std::is_invocable_r_v<std::size_t, Build &>
		{
			REQUIRE(std::invoke(build) == library.node_count());
			BENCHMARK(shape + " | build " + method) { return std::invoke(build); };
		};
		build_row("library", [&] { return atl::static_trie<alphabet>(words).node_count(); });
		build_row("by slice comparison", [&] { return atl::baseline::static_trie_by_slice_comparison<alphabet>(words).node_count(); });
		build_row("dynamic trie", [&] {
			atl::trie<alphabet> t;
			for (auto const &w : words) t.insert(w);
			return t.node_count();
		});
		BENCHMARK(shape + " | count all words, library") {
			uint64 total = 0;
			for (auto const &w : words) total += library.count(w);
			return total;
		};
	};
	{
		std::vector<std::vector<uint32>> words(100'000, std::vector<uint32>(10));
		for (auto &w : words)
			for (auto &s : w) s = uint32(rng() % alphabet);
		race("U=1e6 1e5x10", words);
	}
	{
		std::vector<std::vector<uint32>> words(100'000);
		for (auto &w : words) {
			w.resize(5 + std::size_t(rng() % 11));
			for (auto &s : w) s = uint32(rng() % 26);
		}
		race("sigma=26 1e5x(5-15)", words);
	}
	{
		std::vector<uint32> prefix(500);
		for (auto &s : prefix) s = uint32(rng() % 26);
		std::vector<std::vector<uint32>> words(20'000, prefix);
		for (auto &w : words)
			for (std::size_t k = 1 + std::size_t(rng() % 20); k-- > 0;) w.push_back(uint32(rng() % 26));
		race("prefix500 2e4", words);
	}
	race("1e5 identical x10", std::vector<std::vector<uint32>>(100'000, std::vector<uint32>(10, 7)));
}
