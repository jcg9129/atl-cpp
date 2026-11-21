#include <atl/strings/palindrome_lengths.hpp>
#include <atl/strings/palindromic_tree.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cctype>
#include <cstdint>
#include <limits>
#include <random>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

using atl::palindrome_lengths;

namespace {
	bool is_pal(std::string const &s) { return std::ranges::equal(s, std::views::reverse(s)); }
}

TEST_CASE("palindrome_lengths: radii and queries match a brute oracle", "[strings]") {
	auto seed = GENERATE(take(60, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::string s(std::size_t(rng() % 40), 'a'); // includes the empty text
	for (auto &c : s) c = char('a' + std::size_t(rng() % sigma));
	std::size_t const n = s.size();
	palindrome_lengths const m(s);
	REQUIRE(m.size() == n);
	REQUIRE(m.empty() == s.empty());

	// every substring: O(1) palindrome check vs brute
	std::uint64_t total = 0;
	std::uint32_t longest = 0;
	for (std::uint32_t l = 0; l <= n; ++l)
		for (std::uint32_t r = l; r <= n; ++r) {
			bool const brute = is_pal(s.substr(l, r - l));
			REQUIRE(m.is_palindrome(l, r) == brute);
			if (brute && r > l) {
				++total;
				longest = (std::max)(longest, r - l);
			}
		}
	// per-center maximal lengths vs brute expansion
	for (std::uint32_t i = 0; i < n; ++i) {
		std::uint32_t odd = 1;
		while (i >= odd / 2 + 1 && i + odd / 2 + 1 < n && s[i - odd / 2 - 1] == s[i + odd / 2 + 1]) odd += 2;
		REQUIRE(m.odd_length(i) == odd);
		if (i >= 1) {
			std::uint32_t even = 0;
			while (i >= even / 2 + 1 && i + even / 2 < n && s[i - even / 2 - 1] == s[i + even / 2]) even += 2;
			REQUIRE(m.even_length(i) == even);
		}
	}

	// the aggregate recipes (deliberately not API): folds over the accessors
	std::uint64_t fold_total = 0;
	std::uint32_t fold_longest = 0, fold_pos = 0;
	for (std::uint32_t i = 0; i < n; ++i) {
		fold_total += (m.odd_length(i) + 1) / 2;
		if (m.odd_length(i) > fold_longest) {
			fold_longest = m.odd_length(i);
			fold_pos = i - fold_longest / 2;
		}
		if (i >= 1) {
			fold_total += m.even_length(i) / 2;
			if (m.even_length(i) > fold_longest) {
				fold_longest = m.even_length(i);
				fold_pos = i - fold_longest / 2;
			}
		}
	}
	REQUIRE(fold_total == total);
	REQUIRE(fold_longest == longest);
	if (n > 0) REQUIRE(is_pal(s.substr(fold_pos, fold_longest)));
}

TEST_CASE("palindrome_lengths: agrees with the palindromic_tree across structures", "[strings]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::string s(1 + std::size_t(rng() % 200), 'a');
	for (auto &c : s) c = char('a' + std::size_t(rng() % 2));

	palindrome_lengths const m(s);
	atl::palindromic_tree<26> t;
	t.insert(s, [](char c) { return c - 'a'; });

	// total occurrences: palindromic_tree per-node counts summed == the radius fold
	auto const occ = t.occurrence_counts();
	std::uint64_t eertree_total = 0;
	std::int32_t eertree_longest = 0;
	for (std::uint32_t v = 2; v < t.node_count(); ++v) {
		eertree_total += occ[v];
		eertree_longest = (std::max)(eertree_longest, t.length(v));
	}
	std::uint64_t fold_total = 0;
	std::uint32_t fold_longest = 0;
	for (std::uint32_t i = 0; i < m.size(); ++i) {
		fold_total += (m.odd_length(i) + 1) / 2;
		fold_longest = (std::max)(fold_longest, m.odd_length(i));
		if (i >= 1) {
			fold_total += m.even_length(i) / 2;
			fold_longest = (std::max)(fold_longest, m.even_length(i));
		}
	}
	REQUIRE(fold_total == eertree_total);
	REQUIRE(fold_longest == std::uint32_t(eertree_longest));
}

TEST_CASE("palindrome_lengths: arbitrary equality-comparable symbols and projections", "[strings]") {
	// no integral alphabet: symbols are whole strings
	std::vector<std::string> const tokens{"ab", "cd", "ab", "xy", "ab", "cd", "ab"};
	palindrome_lengths const m(tokens);
	REQUIRE(m.is_palindrome(0, 7)); // the token sequence is a palindrome
	REQUIRE(m.is_palindrome(0, 3));
	REQUIRE_FALSE(m.is_palindrome(0, 2));
	REQUIRE(m.odd_length(3) == 7);

	// projection: case-insensitive characters
	std::string const mixed = "AbBa";
	palindrome_lengths const ci(mixed, [](char c) { return char(std::tolower(static_cast<unsigned char>(c))); });
	REQUIRE(ci.is_palindrome(0, 4));
	palindrome_lengths const cs(mixed);
	REQUIRE_FALSE(cs.is_palindrome(0, 4));

	SECTION("known small cases") {
		palindrome_lengths const e{};
		REQUIRE(e.empty());
		REQUIRE(e.size() == 0);
		REQUIRE(e.is_palindrome(0, 0));

		palindrome_lengths const one(std::string{"z"});
		REQUIRE(one.size() == 1);
		REQUIRE(one.odd_length(0) == 1);
	}
}
