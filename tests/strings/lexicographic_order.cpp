#include <atl/experimental/strings/lexicographic_order_variants.hpp>
#include <atl/strings/lexicographic_order.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <functional>
#include <limits>
#include <random>
#include <ranges>
#include <string>
#include <vector>
#include "input_only.hpp"

using atl::lexicographic_order;

namespace {
	/** @brief Oracle with the library's exact semantics: stable, `<` on projected symbols. */
	template<typename Words, typename Proj = std::identity>
		requires atl::sortable_word_range<Words const &, Proj> && std::ranges::random_access_range<Words const &>
			  && std::ranges::sized_range<Words const &>
	std::vector<std::uint32_t> brute(Words const &w, Proj proj = {}) {
		std::vector<std::uint32_t> order(w.size());
		for (std::size_t index = 0; index < order.size(); ++index) order[index] = std::uint32_t(index);
		std::ranges::stable_sort(order, [&](std::uint32_t a, std::uint32_t b) {
			return std::ranges::lexicographical_compare(w[a], w[b], std::less{}, proj, proj);
		});
		return order;
	}
}

TEST_CASE("lexicographic_order: matches a stable oracle, incl. the MSD variant", "[strings]") {
	auto seed = GENERATE(take(60, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 4);
	std::vector<std::string> words(std::size_t(rng() % 40));
	for (auto &w : words) {
		w.resize(std::size_t(rng() % 10)); // includes empty words
		for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
	}
	auto const expect = brute(words);
	REQUIRE(lexicographic_order(words) == expect);
	REQUIRE(atl::experimental::lexicographic_order_by_msd_partition(words) == expect);
	REQUIRE(atl::experimental::lexicographic_order_by_lcp_merge(words) == expect);
}

TEST_CASE("lexicographic_order: contenders on long shared prefixes and wide symbols", "[strings]") {
	auto seed = GENERATE(take(6, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::string prefix(200 + std::size_t(rng() % 300), 'a');
	for (auto &c : prefix) c = char('a' + std::size_t(rng() % sigma));
	// about 1e3 words: a cut of the shared prefix plus a short random tail,
	// so slices descend hundreds of levels and words end at every depth
	std::vector<std::string> words(1000);
	for (auto &w : words) {
		w = prefix.substr(0, prefix.size() - std::size_t(rng() % 4));
		w.resize(w.size() + std::size_t(rng() % 4));
		for (std::size_t i = prefix.size() - 3; i < w.size(); ++i) w[i] = char('a' + std::size_t(rng() % sigma));
	}
	auto const expect = brute(words);
	REQUIRE(lexicographic_order(words) == expect);
	REQUIRE(atl::experimental::lexicographic_order_by_msd_partition(words) == expect);
	REQUIRE(atl::experimental::lexicographic_order_by_lcp_merge(words) == expect);

	// the generic (non-byte) partition path, signed symbols included
	std::vector<std::vector<std::int64_t>> wide(words.size());
	for (std::size_t i = 0; i < words.size(); ++i)
		for (char const c : words[i]) wide[i].push_back((std::int64_t(c) - 'b') * (std::int64_t(1) << 40));
	REQUIRE(atl::experimental::lexicographic_order_by_msd_partition(wide) == expect);
	REQUIRE(atl::experimental::lexicographic_order_by_lcp_merge(wide) == expect);
}

TEST_CASE("lexicographic_order: stability with heavy duplicates", "[strings]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<std::string> pool(1 + std::size_t(rng() % 4));
	for (auto &w : pool) {
		w.resize(1 + std::size_t(rng() % 5));
		for (auto &c : w) c = char('a' + std::size_t(rng() % 2));
	}
	std::vector<std::string> words(200);
	for (auto &w : words) w = pool[std::size_t(rng() % pool.size())];
	auto const order = lexicographic_order(words);
	// equal words must appear in strictly increasing input-index order
	for (std::size_t i = 1; i < order.size(); ++i) {
		REQUIRE(words[order[i - 1]] <= words[order[i]]);
		if (words[order[i - 1]] == words[order[i]]) REQUIRE(order[i - 1] < order[i]);
	}
	REQUIRE(order == brute(words));
	REQUIRE(atl::experimental::lexicographic_order_by_msd_partition(words) == order);
	REQUIRE(atl::experimental::lexicographic_order_by_lcp_merge(words) == order);
}

TEMPLATE_TEST_CASE_SIG("lexicographic_order: semantics, projections, symbol types", "[strings]", ((int Sort), Sort), 0, 1, 2) {
	auto const lexicographic_order = [](auto &&words, auto... proj) {
		if constexpr (Sort == 0)
			return atl::lexicographic_order(words, proj...);
		else if constexpr (Sort == 1)
			return atl::experimental::lexicographic_order_by_msd_partition(words, proj...);
		else
			return atl::experimental::lexicographic_order_by_lcp_merge(words, proj...);
	};
	SECTION("char order follows the platform's `<` on char, like every atl tool") {
		std::vector<std::string> const words{std::string{"\x80"}, std::string{"a"}};
		auto const order = lexicographic_order(words);
		// signed-char targets put the negative byte first; unsigned-char
		// targets (e.g. aarch64) agree with std::string's memcmp order
		bool const negative_first = char('\x80') < 'a';
		REQUIRE(order
				== (negative_first ? std::vector<std::uint32_t>{0, 1} : std::vector<std::uint32_t>{1, 0}));
		REQUIRE(order == brute(words));
	}
	SECTION("projection: case-insensitive") {
		auto const gs2_lower_projection = [](char c) { return char(std::tolower(c)); };
		std::vector<std::string> const words{"Banana", "apple", "CHERRY"};
		REQUIRE(lexicographic_order(words, gs2_lower_projection) == std::vector<std::uint32_t>{1, 0, 2});
	}
	SECTION("non-char symbols") {
		std::vector<std::vector<int>> const words{{3, 1}, {2, 9, 9}, {3}, {2, 9}};
		REQUIRE(lexicographic_order(words) == std::vector<std::uint32_t>{3, 1, 2, 0});
	}
	SECTION("prefixes sort initial their extensions; edges") {
		std::vector<std::string> const words{"ab", "a", "abc", "", "b"};
		REQUIRE(lexicographic_order(words) == std::vector<std::uint32_t>{3, 1, 0, 2, 4});
		REQUIRE(lexicographic_order(std::vector<std::string>{}).empty());
		REQUIRE(lexicographic_order(std::vector<std::string>{""}) == std::vector<std::uint32_t>{0});
	}
	SECTION("input-only word range") {
		std::vector<std::string> const words{"ba", "b", "a", "ba"};
		REQUIRE(lexicographic_order(test::input_only(words)) == std::vector<std::uint32_t>{2, 1, 0, 3});
	}
}

TEST_CASE("lexicographic_order: forward reserve does not evaluate the projection", "[strings]") {
	std::vector<std::string> const words{"ba", "", "a", "ba"};
	std::forward_list<std::string> forward(words.begin(), words.end());
	CHECK(lexicographic_order(forward) == brute(words));
	auto view = words | std::views::filter([count = 0](auto const &) mutable { return ++count > 0; });
	int calls = 0;
	CHECK(lexicographic_order(view, [&](char c) { ++calls; return c; }) == brute(words));
	CHECK(calls == 5);
	std::vector<std::vector<std::string>> const tokens{{"b", "a"}, {}, {"a"}, {"b", "a"}};
	CHECK(lexicographic_order(tokens) == brute(tokens));
	CHECK(lexicographic_order(test::input_only(tokens)) == brute(tokens));
}

TEST_CASE("lexicographic_order: all-empty collections preserve stable order", "[strings]") {
	std::vector<std::string> const words(64);
	std::vector<atl::uint32> expected(words.size());
	for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = atl::uint32(i);
	REQUIRE(atl::lexicographic_order(words) == expected);
	REQUIRE(atl::experimental::lexicographic_order_by_msd_partition(words) == expected);
	REQUIRE(atl::experimental::lexicographic_order_by_lcp_merge(words) == expected);
	std::vector<std::vector<int>> const wide(64);
	REQUIRE(atl::lexicographic_order(wide) == expected);
	REQUIRE(atl::experimental::lexicographic_order_by_msd_partition(wide) == expected);
	REQUIRE(atl::experimental::lexicographic_order_by_lcp_merge(wide) == expected);
}
