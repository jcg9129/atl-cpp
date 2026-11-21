/**
 * @file
 * @brief Stable word-argsort throughput including materialization. The library uses
 * byte MSD radix and wide-symbol stable sort; by stable sort varies bulk/per-symbol
 * append and comparison spelling; by unsigned bytes flips signed-byte order for
 * memcmp; by MSD partition sorts each wide-symbol slice; by LCP merge retains run LCPs.
 * Shapes: 2e5 random2/random26/duplicate words and 2e4 words with a 500-symbol
 * common prefix, as char/int/uint64. Every contender is checked against the library.
 * Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, asserts off,
 * 2026-10-15, -O2 -g -DNDEBUG -std=c++23, with/without -march=native:
 * random26 char library 13.3/13.1 ms versus bulk stable sort 44.4/44.9 ms;
 * random int favors library stable sort over MSD partition, 51.7/51.8 vs 76.3/76.4 ms.
 * Full tables: docs/lexicographic-order.md.
 */
#include <atl/experimental/strings/lexicographic_order_variants.hpp>
#include <atl/strings/lexicographic_order.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
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
using atl::uint64;
namespace ex = atl::experimental;

namespace {
	/** @brief Stable comparison argsort with bulk or per-symbol append and classic or ranges comparison. */
	template<bool IsBulk, bool IsRanges, typename Words>
		requires atl::sortable_word_range<Words const &> && std::ranges::sized_range<Words const &>
	std::vector<uint32> lexicographic_order_by_stable_sort(Words const &words) {
		using symbol = std::ranges::range_value_t<std::ranges::range_value_t<Words>>;
		std::vector<symbol> sym;
		std::vector<std::size_t> word_end{0};
		word_end.reserve(words.size() + 1);
		for (auto const &w : words) {
			if constexpr (IsBulk) {
				std::identity proj;
				atl::detail::append_projected(sym, w, proj);
			} else {
				for (auto const &e : w) sym.push_back(e);
			}
			word_end.push_back(sym.size());
		}
		std::vector<uint32> order(words.size());
		for (std::size_t index = 0; index < order.size(); ++index) order[index] = uint32(index);
		std::ranges::stable_sort(order, [&](uint32 a, uint32 b) {
			auto const *const a0 = sym.data() + word_end[a], *const a1 = sym.data() + word_end[a + 1];
			auto const *const b0 = sym.data() + word_end[b], *const b1 = sym.data() + word_end[b + 1];
			if constexpr (IsRanges)
				return std::ranges::lexicographical_compare(a0, a1, b0, b1);
			else
				return std::lexicographical_compare(a0, a1, b0, b1);
		});
		return order;
	}

	/**
	 * @brief The library sort over `char` words materialized as `unsigned
	 *        char` with the sign bit flipped (when `char` is signed), so the
	 *        word compare is a memcmp on every target while keeping `<` on
	 *        `char` as the order.
	 */
	std::vector<uint32> lexicographic_order_by_unsigned_bytes(std::vector<std::string> const &words) {
		constexpr unsigned char flip = char(-1) < 0 ? 0x80 : 0;
		std::vector<unsigned char> sym;
		std::vector<std::size_t> word_end{0};
		word_end.reserve(words.size() + 1);
		for (auto const &w : words) {
			auto const old = sym.size();
			sym.resize(old + w.size());
			for (std::size_t i = 0; i < w.size(); ++i) sym[old + i] = static_cast<unsigned char>(static_cast<unsigned char>(w[i]) ^ flip);
			word_end.push_back(sym.size());
		}
		std::vector<uint32> order(words.size());
		for (std::size_t index = 0; index < order.size(); ++index) order[index] = uint32(index);
		std::ranges::stable_sort(order, [&](uint32 a, uint32 b) {
			return std::lexicographical_compare(sym.data() + word_end[a], sym.data() + word_end[a + 1],
												sym.data() + word_end[b], sym.data() + word_end[b + 1]);
		});
		return order;
	}

	template<std::integral Symbol>
	std::vector<std::vector<Symbol>> words_of(std::vector<std::string> const &words) {
		std::vector<std::vector<Symbol>> out(words.size());
		for (std::size_t i = 0; i < words.size(); ++i) out[i].assign(words[i].begin(), words[i].end());
		return out;
	}
}

// Word argsort race (docs/lexicographic-order.md): the library sort (MSD
// radix for byte symbols, stable sort otherwise) against the other
// stable sort and its micro-variants (bulk vs per-symbol append,
// compare spellings), the wide-symbol MSD partition and the LCP mergesort.
// Shapes: 2e5 random words
// of length 5-15 over 26 letters; the same over 2 letters; 2e5 words drawn
// from a pool of 1000 (heavy duplicates); 2e4 words sharing a 500-symbol
// prefix with a random 1-20 tail; the random, duplicate and prefix shapes
// again as int and uint64 words.
TEST_CASE("lexicographic_order contenders", "[lexicographic_order]") {
	std::mt19937_64 rng(1);
	auto const random_words = [&](std::size_t count, std::size_t sigma) {
		std::vector<std::string> words(count);
		for (auto &w : words) {
			w.resize(5 + std::size_t(rng() % 11));
			for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
		}
		return words;
	};
	auto const random26 = random_words(200'000, 26);
	auto const random2 = random_words(200'000, 2);
	auto const pool = random_words(1000, 26);
	std::vector<std::string> duplicates(200'000);
	for (auto &w : duplicates) w = pool[std::size_t(rng() % pool.size())];
	std::string prefix(500, 'a');
	for (auto &c : prefix) c = char('a' + std::size_t(rng() % 26));
	std::vector<std::string> shared(20'000);
	for (auto &w : shared) {
		w = prefix;
		w.resize(500 + 1 + std::size_t(rng() % 20));
		for (std::size_t i = 500; i < w.size(); ++i) w[i] = char('a' + std::size_t(rng() % 26));
	}

	auto const race_char = [&](std::string const &shape, std::vector<std::string> const &words) {
		auto const expect = atl::lexicographic_order(words);
		REQUIRE(lexicographic_order_by_stable_sort<true, false>(words) == expect);
		REQUIRE(lexicographic_order_by_stable_sort<false, false>(words) == expect);
		REQUIRE(lexicographic_order_by_stable_sort<true, true>(words) == expect);
		REQUIRE(lexicographic_order_by_unsigned_bytes(words) == expect);
		REQUIRE(ex::lexicographic_order_by_lcp_merge(words) == expect);
		measure(shape + " | library (msd radix)", [&] { return atl::lexicographic_order(words).size(); });
		measure(shape + " | by stable sort, bulk append", [&] { return lexicographic_order_by_stable_sort<true, false>(words).size(); });
		measure(shape + " | by stable sort, per-symbol append", [&] { return lexicographic_order_by_stable_sort<false, false>(words).size(); });
		measure(shape + " | by stable sort, ranges compare", [&] { return lexicographic_order_by_stable_sort<true, true>(words).size(); });
		measure(shape + " | by stable sort, unsigned-byte memcmp", [&] { return lexicographic_order_by_unsigned_bytes(words).size(); });
		measure(shape + " | by LCP merge", [&] { return ex::lexicographic_order_by_lcp_merge(words).size(); });
	};
	race_char("random26 2e5", random26);
	race_char("random2 2e5", random2);
	race_char("duplicates 2e5", duplicates);
	race_char("prefix500 2e4 char", shared);

	auto const race_wide = [&]<typename Words>(std::string const &shape, Words const &words)
		requires atl::sortable_word_range<Words const &> && std::ranges::sized_range<Words const &>
	{
		auto const expect = atl::lexicographic_order(words);
		REQUIRE(lexicographic_order_by_stable_sort<false, false>(words) == expect);
		REQUIRE(ex::lexicographic_order_by_msd_partition(words) == expect);
		REQUIRE(ex::lexicographic_order_by_lcp_merge(words) == expect);
		measure(shape + " | library (stable sort)", [&] { return atl::lexicographic_order(words).size(); });
		measure(shape + " | by stable sort, per-symbol append", [&] { return lexicographic_order_by_stable_sort<false, false>(words).size(); });
		measure(shape + " | by MSD partition", [&] { return ex::lexicographic_order_by_msd_partition(words).size(); });
		measure(shape + " | by LCP merge", [&] { return ex::lexicographic_order_by_lcp_merge(words).size(); });
	};
	race_wide("random26 2e5 int", words_of<int>(random26));
	race_wide("random26 2e5 uint64", words_of<uint64>(random26));
	race_wide("duplicates 2e5 int", words_of<int>(duplicates));
	race_wide("prefix500 2e4 int", words_of<int>(shared));
	race_wide("prefix500 2e4 uint64", words_of<uint64>(shared));
}
