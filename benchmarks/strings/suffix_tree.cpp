/**
 * @file
 * @brief Suffix-tree construction throughput. The library uses a hashed child pool,
 * retargets split entries and reuses sealing scratch; by sibling list scans child
 * lists; by separate lookups repeats pool descents and allocates sealing scratch.
 * Loops reserve then insert random alphabets, Fibonacci, and short-word dictionaries
 * totaling about 1e6 symbols. The sibling list is excluded at sigma=1e6 because
 * linear-degree scans do not finish in a practical time.
 * Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, asserts off,
 * 2026-10-15, -O2 -g -DNDEBUG -std=c++23, with/without -march=native:
 * pool/list random26 560/577 versus 1206/1218 ms; sigma=2 favors the list.
 * Full tables: docs/suffix-tree.md.
 */
#include "baseline/suffix_tree.hpp"
#include <atl/experimental/strings/suffix_tree_variants.hpp>
#include <atl/strings/suffix_tree.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
	/** @brief Measure full-operation throughput with one directly invocable contender. */
	template<typename Operation>
		requires std::invocable<Operation &>
	void measure(std::string const &label, Operation operation) {
		BENCHMARK(std::string(label)) { return std::invoke(operation); };
	}
}

using atl::uint32;
namespace ex = atl::experimental;

// The suffix-tree child-storage race (docs/suffix-tree.md): the library
// hashed child pool vs the first-child/next-sibling list, one random word of
// 10^6 symbols per alphabet size. The list is not timed at sigma = 10^6: every
// root lookup scans an O(degree) list, and it did not finish in the race.
TEST_CASE("suffix_tree contenders", "[suffix_tree]") {
	constexpr std::size_t length = 1'000'000, alphabet = 1'000'000;
	std::mt19937_64 rng(1);
	auto const text = [&](std::size_t sigma) {
		std::vector<uint32> word(length);
		for (auto &symbol : word) symbol = uint32(std::size_t(rng() % sigma));
		return word;
	};

	for (std::size_t const sigma : {std::size_t(2), std::size_t(26), alphabet}) {
		auto const word = text(sigma);
		std::string const shape = "sigma=" + std::to_string(sigma);
		{
			atl::suffix_tree<alphabet> library;
			library.insert(word);
			if (sigma != alphabet) {
				ex::suffix_tree_by_sibling_list<alphabet> list;
				list.insert(word);
				REQUIRE(list.node_count() == library.node_count());
			}
		}
		measure(shape + " | build library", [&] {
			atl::suffix_tree<alphabet> t;
			t.reserve(length);
			t.insert(word);
			return t.node_count();
		});
		if (sigma != alphabet) {
			measure(shape + " | build by sibling list", [&] {
				ex::suffix_tree_by_sibling_list<alphabet> t;
				t.reserve(length);
				t.insert(word);
				return t.node_count();
			});
		}
	}
}

// The library tree uses one pool descent per split and per leaf and keeps
// finish_word's buffer as a member. Compare it against its baseline/ copy, on one 1e6-symbol word at sigma 2, 26 and 1e6, a Fibonacci
// word, 1e6 words of length 1-5 and 1e5 words of length 10 via insert.
TEST_CASE("suffix_tree lookups and word buffer", "[suffix_tree]") {
	constexpr std::size_t length = 1'000'000, alphabet = 1'000'000;
	std::mt19937_64 rng(2);
	auto const race = [&](std::string const &shape, std::vector<std::vector<uint32>> const &words) {
		auto const build = [&]<typename Tree>()
			requires std::default_initializable<Tree> && requires(Tree &structure, std::vector<uint32> const &word) {
				structure.reserve(std::size_t{});
				structure.insert(word);
				{ structure.node_count() } -> std::convertible_to<std::size_t>;
			}
		{
			Tree t;
			t.reserve(length);
			for (auto const &w : words) t.insert(w);
			return t.node_count();
		};
		REQUIRE(build.template operator()<atl::suffix_tree<alphabet>>()
				== build.template operator()<atl::baseline::suffix_tree_by_separate_lookups<alphabet>>());
		measure(shape + " | library", [&] { return build.template operator()<atl::suffix_tree<alphabet>>(); });
		measure(shape + " | by separate lookups", [&] { return build.template operator()<atl::baseline::suffix_tree_by_separate_lookups<alphabet>>(); });
	};
	for (std::size_t const sigma : {std::size_t(2), std::size_t(26), alphabet}) {
		std::vector<uint32> word(length);
		for (auto &symbol : word) symbol = uint32(std::size_t(rng() % sigma));
		race("one word sigma=" + std::to_string(sigma), {word});
	}
	std::vector<uint32> a{0}, b{0, 1};
	while (b.size() < length) {
		auto next = b;
		next.insert(next.end(), a.begin(), a.end());
		a = std::move(b);
		b = std::move(next);
	}
	b.resize(length);
	race("one Fibonacci word", {b});
	std::vector<std::vector<uint32>> short_words;
	for (std::size_t total = 0; total < length;) {
		auto &w = short_words.emplace_back(1 + std::size_t(rng() % 5));
		for (auto &symbol : w) symbol = uint32(std::size_t(rng() % 26));
		total += w.size();
	}
	race("1e6 symbols in words of length 1-5, sigma=26", short_words);
	std::vector<std::vector<uint32>> ten(length / 10, std::vector<uint32>(10));
	for (auto &w : ten)
		for (auto &symbol : w) symbol = uint32(std::size_t(rng() % 26));
	race("1e5 words of length 10, sigma=26", ten);
}
