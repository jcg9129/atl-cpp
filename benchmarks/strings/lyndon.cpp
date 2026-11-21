/** @file
 * Races runs throughput on 10^6-symbol unary, Fibonacci and random texts.
 * The library computes inverted ranks without LCP; by full suffix arrays
 * retains inverted LCP as well. Intel Xeon Platinum 8358, GCC 15.2.0,
 * RelWithDebInfo, -O2 -g -DNDEBUG, with/without -march=native, 2026-10-15:
 * Fibonacci takes 394/385 ms (library) and 407/398 ms (by full suffix arrays).
 * Full tables: docs/lyndon.md. Widened counting positions require remeasurement.
 */
#include "baseline/lyndon.hpp"
#include <atl/strings/lyndon.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using atl::uint32;

TEST_CASE("lyndon runs", "[lyndon]") {
	auto const measure = []<typename Contender>(std::string const &label, Contender contender)
		requires std::invocable<Contender &>
	{
		BENCHMARK(std::string(label)) { return std::invoke(contender); };
	};
	constexpr std::size_t n = 1'000'000;
	std::mt19937_64 rng(1);
	std::vector<std::pair<std::string, std::vector<uint32>>> texts;
	std::vector<uint32> a{0}, b{0, 1};
	while (b.size() < n) {
		auto next = b;
		next.insert(next.end(), a.begin(), a.end());
		a = std::move(b);
		b = std::move(next);
	}
	b.resize(n);
	texts.emplace_back("fib", b);
	texts.emplace_back("all-a", std::vector<uint32>(n, 0));
	for (uint32 const sigma : {2u, 26u}) {
		std::vector<uint32> t(n);
		for (auto &v : t) v = uint32(rng() % sigma);
		texts.emplace_back("random" + std::to_string(sigma), t);
	}
	for (auto const &[name, text] : texts) {
		auto const library = atl::runs(text);
		auto const by_full_suffix_arrays = atl::baseline::runs_by_full_suffix_arrays(text);
		REQUIRE(library.size() == by_full_suffix_arrays.size());
		for (std::size_t i = 0; i < library.size(); ++i) {
			REQUIRE(library[i].period == by_full_suffix_arrays[i].period);
			REQUIRE(library[i].l == by_full_suffix_arrays[i].l);
			REQUIRE(library[i].r == by_full_suffix_arrays[i].r);
		}
		measure(name + " 1e6 | runs library", [&] { return atl::runs(text).size(); });
		measure(name + " 1e6 | runs by full suffix arrays", [&] { return atl::baseline::runs_by_full_suffix_arrays(text).size(); });
	}
}
