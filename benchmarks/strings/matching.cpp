/** @file
 * Races border and Z-array throughput on unary, alternating, Fibonacci and random
 * texts of 10^6 and 10^7 symbols, with identity and lowercase projections.
 * The library Z loop skips a known mismatch; by box expansion performs it.
 * Border loops both use repeated projection. Intel Xeon Platinum 8280, GCC
 * 15.2.0, RelWithDebInfo, -O2 -g -DNDEBUG, with/without -march=native,
 * 2026-10-15: Fibonacci Z at 10^7 takes 38.3/42.7 ms (library) versus
 * 58.7/59.1 ms (by box expansion). Full tables: docs/matching.md.
 */
#include "baseline/matching.hpp"
#include <atl/strings/matching.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cctype>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

TEST_CASE("matching arrays", "[matching]") {
	auto const measure = []<typename Contender>(std::string const &label, Contender contender)
		requires std::invocable<Contender &>
	{
		BENCHMARK(std::string(label)) { return std::invoke(contender); };
	};
	std::mt19937_64 rng(1);
	auto const lower = [](char c) { return char(std::tolower(static_cast<unsigned char>(c))); };
	for (std::size_t const n : {std::size_t(1'000'000), std::size_t(10'000'000)}) {
		std::vector<std::pair<std::string, std::string>> texts;
		texts.emplace_back("all-a", std::string(n, 'a'));
		std::string abab(n, 'a');
		for (std::size_t i = 1; i < n; i += 2) abab[i] = 'B';
		texts.emplace_back("abab", abab);
		std::string a = "a", b = "aB";
		while (b.size() < n) {
			std::string next = b + a;
			a = std::move(b);
			b = std::move(next);
		}
		b.resize(n);
		texts.emplace_back("fib", b);
		for (int const sigma : {2, 26}) {
			std::string t(n, 'a');
			for (auto &c : t) c = char('a' + rng() % sigma);
			texts.emplace_back("random" + std::to_string(sigma), t);
		}
		for (auto const &[name, text] : texts) {
			std::string const shape = name + " n=" + std::to_string(n);
			REQUIRE(atl::common_prefix_lengths(text) == atl::baseline::common_prefix_lengths_by_box_expansion(text));
			REQUIRE(atl::common_prefix_lengths(text, lower) == atl::baseline::common_prefix_lengths_by_box_expansion(text, lower));
			REQUIRE(atl::border_lengths(text, lower) == atl::baseline::border_lengths_by_repeated_projection(text, lower));
			measure(shape + " | Z library", [&] { return atl::common_prefix_lengths(text).back(); });
			measure(shape + " | Z by box expansion", [&] { return atl::baseline::common_prefix_lengths_by_box_expansion(text).back(); });
			measure(shape + " | Z tolower library", [&] { return atl::common_prefix_lengths(text, lower).back(); });
			measure(shape + " | Z tolower by box expansion", [&] { return atl::baseline::common_prefix_lengths_by_box_expansion(text, lower).back(); });
			measure(shape + " | borders tolower library", [&] { return atl::border_lengths(text, lower).back(); });
			measure(shape + " | borders tolower by repeated projection", [&] { return atl::baseline::border_lengths_by_repeated_projection(text, lower).back(); });
		}
	}
}
