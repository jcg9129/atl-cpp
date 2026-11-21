/**
 * @file
 * @brief Suffix-array construction and isolated LCP throughput. The library uses SA-IS,
 * pair compression and in-place sentinel removal; by index compression copies the
 * reduced text and final array; by prefix doubling uses counting-sort rounds.
 * By Kasai follows inverse ranks; by phi scatters, scans text order and gathers.
 * Shapes: random alphabets 2/26, unary, alternating and Fibonacci at 1e6,
 * random26/Fibonacci at 5e6, sparse int64 ranges and string tokens.
 * Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, asserts off,
 * 2026-10-15, -O2 -g -DNDEBUG -std=c++23, with/without -march=native:
 * unary library build 27.0/21.9 ms versus doubling sort 140/141 ms.
 * Phi wins random26 LCP (16.7/16.5 vs 21.2/19.9 ms), loses unary (2.9/2.9 vs 1.8/1.8).
 * Full tables: docs/suffix-array.md.
 */
#include "baseline/suffix_array.hpp"
#include <atl/experimental/strings/suffix_array_variants.hpp>
#include <atl/strings/suffix_array.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <ranges>
#include <span>
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

using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	std::vector<uint32> fibonacci(std::size_t n) {
		std::vector<uint32> a{1}, b{1, 2};
		while (b.size() < n) {
			auto next = b;
			next.insert(next.end(), a.begin(), a.end());
			a = std::move(b);
			b = std::move(next);
		}
		b.resize(n);
		return b;
	}

	/** @brief Kasai's LCP from a finished suffix array: two dependent random accesses per position. */
	std::vector<uint32> lcp_by_kasai(std::vector<uint32> const &s, std::span<uint32 const> sa, std::span<uint32 const> rank) {
		uint32 const n = uint32(sa.size());
		std::vector<uint32> lcp(n - 1);
		for (uint32 i = 0, k = 0; i < n; ++i) {
			if (rank[i] + 1 == n) {
				k = 0;
				continue;
			}
			uint32 const j = sa[rank[i] + 1];
			while (k < n - i && k < n - j && s[i + k] == s[j + k]) ++k;
			lcp[rank[i]] = k;
			if (k > 0) --k;
		}
		return lcp;
	}

	/** @brief The phi (permuted-LCP) method: scatter, text-order pass, gather. */
	std::vector<uint32> lcp_by_phi(std::vector<uint32> const &s, std::span<uint32 const> sa) {
		uint32 const n = uint32(sa.size());
		std::vector<uint32> plcp(n);
		for (uint32 r = 0; r + 1 < n; ++r) plcp[sa[r]] = sa[r + 1];
		plcp[sa[n - 1]] = uint32(-1);
		for (uint32 i = 0, k = 0; i < n; ++i) {
			uint32 const j = plcp[i];
			if (j == uint32(-1)) {
				k = 0;
				continue;
			}
			while (k < n - i && k < n - j && s[i + k] == s[j + k]) ++k;
			plcp[i] = k;
			if (k > 0) --k;
		}
		std::vector<uint32> lcp(n - 1);
		for (uint32 r = 0; r + 1 < n; ++r) lcp[r] = plcp[sa[r]];
		return lcp;
	}
}

// Suffix array build (docs/suffix-array.md). The library build against its
// other copy (baseline/: a copied reduced string per SA-IS level and
// a second buffer for the final array) and the prefix-doubling
// contender (experimental, the sort alone, at 1e6). The LCP phase is timed
// alone, Kasai library vs the phi method, from a finished array.
// Shapes: random sigma 2 and 26, all-a, abab and Fibonacci at 1e6; random 26
// and Fibonacci at 5e6.
TEST_CASE("suffix_array build", "[suffix_array]") {
	std::mt19937_64 rng(1);
	auto const race = [&](std::string const &shape, std::vector<uint32> const &text) {
		atl::suffix_array const library(text);
		REQUIRE(std::ranges::equal(library.lcp_lengths(), atl::baseline::suffix_array_by_index_compression(text).lcp_lengths()));
		measure(shape + " | build library", [&] { return atl::suffix_array(text).size(); });
		measure(shape + " | build by index compression", [&] { return atl::baseline::suffix_array_by_index_compression(text).size(); });
		std::vector<uint32> s = text;
		s.push_back(0);
		uint32 const sigma = *std::ranges::max_element(s) + 1;
		if (text.size() <= 1'000'000) {
			std::vector<uint32> expected(s.size());
			atl::experimental::suffix_sort_by_prefix_doubling(s, expected, sigma);
			REQUIRE(std::ranges::equal(library.positions(), expected | std::views::drop(1)));
		}

		// doubling takes seconds per run at 5e6 on periodic texts; timed at 1e6
		if (text.size() <= 1'000'000) measure(shape + " | sort by prefix doubling", [&] {
			std::vector<uint32> sa(s.size());
			atl::experimental::suffix_sort_by_prefix_doubling(s, sa, sigma);
			return sa[1];
		});
		REQUIRE(lcp_by_kasai(text, library.positions(), library.ranks()) == lcp_by_phi(text, library.positions()));
		measure(shape + " | lcp by Kasai", [&] { return lcp_by_kasai(text, library.positions(), library.ranks()).size(); });
		measure(shape + " | lcp by phi", [&] { return lcp_by_phi(text, library.positions()).size(); });
	};
	auto const random_text = [&](std::size_t n, std::size_t sigma) {
		std::vector<uint32> t(n);
		for (auto &v : t) v = 1 + uint32(std::size_t(rng() % sigma));
		return t;
	};
	race("1e6 random2", random_text(1'000'000, 2));
	race("1e6 random26", random_text(1'000'000, 26));
	race("1e6 all-a", std::vector<uint32>(1'000'000, 1));
	std::vector<uint32> abab(1'000'000);
	for (std::size_t i = 0; i < abab.size(); ++i) abab[i] = 1 + uint32(i % 2);
	race("1e6 abab", abab);
	race("1e6 fib", fibonacci(1'000'000));
	race("5e6 random26", random_text(5'000'000, 26));
	race("5e6 fib", fibonacci(5'000'000));
}

// the compression path for integral symbols whose range defeats the
// offset map, materialized (symbol, index) pairs vs the index sort
// (baseline/), at int64 ranges 32n and 2^40; std::string tokens as control.
TEST_CASE("suffix_array compression", "[suffix_array]") {
	std::mt19937_64 rng(2);
	for (std::size_t const n : {std::size_t(100'000), std::size_t(1'000'000)}) {
		for (uint64 const range : {uint64(32 * n), uint64(1) << 40}) {
			std::vector<int64> text(n);
			for (auto &v : text) v = int64(rng() % range);
			std::string const shape = "n=" + std::to_string(n) + " int64 range " + (range == 32 * n ? "32n" : "2^40");
			REQUIRE(std::ranges::equal(atl::suffix_array(text).positions(), atl::baseline::suffix_array_by_index_compression(text).positions()));
			measure(shape + " | build library", [&] { return atl::suffix_array(text).size(); });
			measure(shape + " | build by index compression", [&] { return atl::baseline::suffix_array_by_index_compression(text).size(); });
		}
	}
	std::vector<std::string> tokens(100'000);
	for (auto &t : tokens) t = "token" + std::to_string(std::size_t(rng() % 5000));
	REQUIRE(std::ranges::equal(atl::suffix_array(tokens).positions(), atl::baseline::suffix_array_by_index_compression(tokens).positions()));
	measure("1e5 string tokens | build library", [&] { return atl::suffix_array(tokens).size(); });
	measure("1e5 string tokens | build by index compression", [&] { return atl::baseline::suffix_array_by_index_compression(tokens).size(); });
}
