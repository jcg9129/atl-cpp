/**
 * @file
 * @brief Trie throughput: hashed sibling storage versus global edge hashing and
 * raw-symbol BSTs; erase path scratch versus two descents. Build/count streams
 * use random and sorted U=1e6 1e5x10 words; erase uses binary 2e5x30 updates
 * and U=1e6 1e5x10 insert/erase streams. Raw-symbol sorted timing is excluded:
 * its sibling chain makes that workload quadratic.
 * Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, assertions off,
 * -O2/native, 2026-10-15: random library build/query 38.6/35.8 and 35.2/37.2 ms;
 * global hash 148/77.8 and 60.1/80.8; symbol BST 31.6/24.1 and 31.0/24.2.
 * Binary erase library 225/225 ms, by two descents 237/243. Full table: docs/trie.md.
 */
#include "baseline/trie.hpp"
#include <atl/experimental/strings/trie_variants.hpp>
#include <atl/strings/trie.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using atl::uint32;
using atl::uint64;
namespace ex = atl::experimental;

namespace {
	/** @brief A copyable constructor callback for a counting-trie contender. */
	template<typename Make>
	concept trie_factory = std::invocable<Make &> && std::copy_constructible<Make>
						&& std::move_constructible<std::invoke_result_t<Make &>>
						&& requires(std::invoke_result_t<Make &> &words, std::vector<uint32> const &word) {
							   words.reserve(std::size_t{});
							   words.insert(word);
							   { words.erase(word) } -> std::convertible_to<bool>;
							   { words.count(word) } -> std::convertible_to<uint64>;
							   { words.size() } -> std::convertible_to<std::size_t>;
						   };
}

TEST_CASE("trie contenders", "[trie]") {
	constexpr std::size_t alphabet = 1'000'000, words = 100'000, length = 10;
	std::mt19937_64 rng(1);
	std::vector<std::vector<uint32>> random_words(words, std::vector<uint32>(length));
	for (auto &w : random_words)
		for (auto &symbol : w) symbol = uint32(rng() % alphabet);
	auto sorted_words = random_words;
	std::ranges::sort(sorted_words);
	auto const build = [&]<trie_factory Make>(Make make, std::vector<std::vector<uint32>> const &input) {
		auto t = make();
		t.reserve(words * length);
		for (auto const &w : input) t.insert(w);
		return t;
	};
	auto const query = [&]<typename Trie>(Trie const &t)
		requires requires { { t.count(random_words[0]) } -> std::convertible_to<uint64>; }
	{
		uint64 found = 0;
		for (auto const &w : random_words) found += t.count(w);
		return found;
	};
	auto const race = [&]<trie_factory Make>(Make make, std::string const &shape, std::string const &name,
											 std::vector<std::vector<uint32>> const &input) {
		auto const t = build(make, input);
		REQUIRE(query(t) == words);
		BENCHMARK(shape + " | build " + name) { return build(make, input).size(); };
		BENCHMARK(shape + " | 1e5 queries " + name) { return query(t); };
	};
	race([] { return atl::trie<alphabet>(); }, "random", "library", random_words);
	race([] { return ex::trie_by_global_hash<alphabet>(); }, "random", "by global hash", random_words);
	race([] { return ex::trie_by_symbol_bst<alphabet>(); }, "random", "by symbol BST", random_words);
	race([] { return atl::trie<alphabet>(); }, "sorted", "library", sorted_words);
	race([] { return ex::trie_by_global_hash<alphabet>(); }, "sorted", "by global hash", sorted_words);
}

TEST_CASE("trie erase", "[trie]") {
	std::mt19937_64 rng(2);
	{
		constexpr std::size_t operations = 200'000, bits = 30;
		std::vector<std::vector<uint32>> words(operations, std::vector<uint32>(bits));
		for (auto &w : words)
			for (auto &bit : w) bit = uint32(rng() % 2);
		std::vector<std::size_t> victim(operations);
		for (std::size_t i = 0; i < operations; ++i) victim[i] = std::size_t(rng() % (i + 1));
		auto const run = [&]<trie_factory Make>(Make make) {
			auto t = make();
			t.reserve(operations * bits);
			std::vector<char> is_live(operations, 0);
			uint64 erased = 0;
			for (std::size_t i = 0; i < operations; ++i) {
				t.insert(words[i]);
				is_live[i] = 1;
				if (std::size_t const v = victim[i]; is_live[v] != 0) {
					is_live[v] = 0;
					erased += t.erase(words[v]);
				}
			}
			return erased;
		};
		auto const library = [] { return atl::trie<2>(); };
		auto const two_descents = [] { return atl::baseline::trie_by_two_descents<2>(); };
		REQUIRE(run(library) == run(two_descents));
		BENCHMARK("binary trie<2> 2e5x30 | erase library") { return run(library); };
		BENCHMARK("binary trie<2> 2e5x30 | erase by two descents") { return run(two_descents); };
	}
	{
		constexpr std::size_t alphabet = 1'000'000, words = 100'000, length = 10;
		std::vector<std::vector<uint32>> input(words, std::vector<uint32>(length));
		for (auto &w : input)
			for (auto &symbol : w) symbol = uint32(rng() % alphabet);
		auto const run = [&]<trie_factory Make>(Make make) {
			auto t = make();
			t.reserve(words * length);
			for (auto const &w : input) t.insert(w);
			uint64 erased = 0;
			for (auto const &w : input) erased += t.erase(w);
			return erased;
		};
		auto const library = [] { return atl::trie<alphabet>(); };
		auto const two_descents = [] { return atl::baseline::trie_by_two_descents<alphabet>(); };
		REQUIRE(run(library) == words);
		REQUIRE(run(two_descents) == words);
		BENCHMARK("U=1e6 1e5x10 insert+erase | library") { return run(library); };
		BENCHMARK("U=1e6 1e5x10 insert+erase | by two descents") { return run(two_descents); };
	}
}
