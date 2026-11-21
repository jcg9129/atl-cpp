/**
 * @file
 * @brief Generalized suffix-automaton construction throughput. The library carries a
 * transition entry into splitting and clones child trees structurally; by transition
 * reinsertion performs lookups and reinserts clone edges. The loop reserves then
 * inserts one million random or unary symbols, or 1e5 ten-symbol words.
 * Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, asserts off,
 * 2026-10-15, -O2 -g -DNDEBUG -std=c++23, with/without -march=native:
 * random26 library 434/449 ms, by transition reinsertion 565/533 ms.
 * Full tables: docs/suffix-automaton.md.
 */
#include "baseline/suffix_automaton.hpp"
#include <atl/strings/suffix_automaton.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
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

// Suffix automaton build (docs/suffix-automaton.md), the library automaton
// against its baseline/ copy. The library engine carries the transition
// entry found by extend into split and copies a clone's sibling BST
// structurally. Shapes: one word of 1e6 symbols at sigma 2, 26 and 1e6, and
// all-a; a generalized dictionary of 1e5 words of length 10 at sigma 26.
TEST_CASE("suffix_automaton build", "[suffix_automaton]") {
	constexpr std::size_t length = 1'000'000, alphabet = 1'000'000;
	std::mt19937_64 rng(1);
	auto const race = [&](std::string const &shape, std::vector<std::vector<uint32>> const &words) {
		auto const build = [&]<typename Automaton>()
			requires std::default_initializable<Automaton> && requires(Automaton &structure, std::vector<uint32> const &word) {
				structure.reserve(std::size_t{});
				structure.insert(word);
				{ structure.transition_count() } -> std::convertible_to<std::size_t>;
			}
		{
			Automaton a;
			a.reserve(length);
			for (auto const &w : words) a.insert(w);
			return a.transition_count();
		};
		REQUIRE(build.template operator()<atl::suffix_automaton<alphabet>>()
				== build.template operator()<atl::baseline::suffix_automaton_by_transition_reinsertion<alphabet>>());
		measure(shape + " | library", [&] { return build.template operator()<atl::suffix_automaton<alphabet>>(); });
		measure(shape + " | by transition reinsertion", [&] { return build.template operator()<atl::baseline::suffix_automaton_by_transition_reinsertion<alphabet>>(); });
	};
	for (std::size_t const sigma : {std::size_t(2), std::size_t(26), alphabet}) {
		std::vector<uint32> word(length);
		for (auto &symbol : word) symbol = uint32(std::size_t(rng() % sigma));
		race("1e6 sigma=" + std::to_string(sigma), {word});
	}
	race("1e6 all-a", {std::vector<uint32>(length, 0)});
	std::vector<std::vector<uint32>> dictionary(length / 10, std::vector<uint32>(10));
	for (auto &w : dictionary)
		for (auto &symbol : w) symbol = uint32(std::size_t(rng() % 26));
	race("dictionary 1e5x10 sigma=26", dictionary);
}
