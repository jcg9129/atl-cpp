/** @file
 * Races reserved palindromic-tree construction throughput on 10^6 symbols
 * (unary and random alphabets 2, 26 and 10^6) and 5*10^6 at alphabet 26.
 * The library inserts in one descent; by find then emplace probes twice for
 * new nodes. Intel Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo,
 * -O2 -g -DNDEBUG, with/without -march=native, 2026-10-15: alphabet 10^6
 * takes 538/528 ms (library), 643/634 ms (by find then emplace).
 * Full tables: docs/palindromic-tree.md. Counter widening requires remeasurement.
 */
#include "baseline/palindromic_tree.hpp"
#include <atl/strings/palindromic_tree.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <vector>

using atl::uint32;

TEST_CASE("palindromic_tree build", "[palindromic_tree]") {
	constexpr std::size_t alphabet = 1'000'000;
	std::mt19937_64 rng(1);
	auto const race = [&](std::string const &shape, std::vector<uint32> const &word) {
		auto const build_library = [&] {
			atl::palindromic_tree<alphabet> t;
			t.reserve(word.size());
			t.insert(word);
			return t;
		};
		auto const build_by_find_then_emplace = [&] {
			atl::baseline::palindromic_tree_by_find_then_emplace<alphabet> t;
			t.reserve(word.size());
			t.insert(word);
			return t;
		};
		auto const library = build_library();
		auto const contender = build_by_find_then_emplace();
		REQUIRE(library.node_count() == contender.node_count());
		REQUIRE(library.occurrence_counts() == contender.occurrence_counts());
		for (uint32 v = 2; v < library.node_count(); ++v) {
			REQUIRE(library.length(v) == contender.length(v));
			REQUIRE(library.parent(v) == contender.parent(v));
			REQUIRE(library.suffix_link(v) == contender.suffix_link(v));
		}
		auto const measure = []<typename Contender>(std::string const &label, Contender build)
			requires std::invocable<Contender &>
		{
			BENCHMARK(std::string(label)) { return std::invoke(build).node_count(); };
		};
		measure(shape + " | library", build_library);
		measure(shape + " | by find then emplace", build_by_find_then_emplace);
	};
	auto const random_word = [&](std::size_t n, std::size_t sigma) {
		std::vector<uint32> word(n);
		for (auto &symbol : word) symbol = uint32(rng() % sigma);
		return word;
	};
	race("1e6 sigma=2", random_word(1'000'000, 2));
	race("1e6 sigma=26", random_word(1'000'000, 26));
	race("1e6 all-a", std::vector<uint32>(1'000'000, 0));
	race("1e6 sigma=1e6", random_word(1'000'000, alphabet));
	race("5e6 sigma=26", random_word(5'000'000, 26));
}
