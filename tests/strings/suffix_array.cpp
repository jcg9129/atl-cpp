#include <atl/experimental/strings/suffix_array_variants.hpp>
#include <atl/strings/suffix_array.hpp>
#include <atl/strings/suffix_automaton.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::suffix_array;

namespace {
	std::vector<std::uint32_t> brute_sa(std::string const &s) {
		std::vector<std::uint32_t> sa(s.size());
		for (std::size_t index = 0; index < sa.size(); ++index) sa[index] = std::uint32_t(index);
		std::ranges::sort(sa,
						  [&](std::uint32_t a, std::uint32_t b) { return s.substr(a) < s.substr(b); });
		return sa;
	}

	std::uint32_t brute_lcp(std::string const &s, std::uint32_t a, std::uint32_t b) {
		std::uint32_t k = 0;
		while (a + k < s.size() && b + k < s.size() && s[a + k] == s[b + k]) ++k;
		return k;
	}

	void check_against_brute(std::string const &s) {
		suffix_array const sa(s);
		auto const expect = brute_sa(s);
		REQUIRE(sa.size() == s.size());
		REQUIRE(std::vector<std::uint32_t>(sa.positions().begin(), sa.positions().end()) == expect);
		for (std::uint32_t r = 0; r < s.size(); ++r) REQUIRE(sa.rank(sa.position(r)) == r);
		REQUIRE(sa.lcp_lengths().size() == (s.empty() ? 0 : s.size() - 1));
		for (std::uint32_t r = 0; r + 1 < s.size(); ++r)
			REQUIRE(sa.lcp_lengths()[r] == brute_lcp(s, sa.position(r), sa.position(r + 1)));

		std::vector<std::uint32_t> text(s.size() + 1, 0), base(s.size() + 1);
		for (std::size_t i = 0; i < s.size(); ++i) text[i] = std::uint32_t((unsigned char)s[i]);
		atl::experimental::suffix_sort_by_prefix_doubling(text, base, 256);
		base.erase(base.begin());
		REQUIRE(base == expect);
	}
}

TEST_CASE("suffix_array: exhaustive compact strings match brute force", "[strings]") {
	for (std::size_t sigma : {1, 2, 3}) {
		std::size_t const max_len = sigma <= 2 ? 12 : 8;
		for (std::size_t len = 0; len <= max_len; ++len) {
			std::size_t count = 1;
			for (std::size_t i = 0; i < len; ++i) count *= sigma;
			for (std::size_t code = 0; code < count; ++code) {
				std::string s(len, 'a');
				for (std::size_t i = 0, c = code; i < len; ++i, c /= sigma) s[i] = char('a' + c % sigma);
				check_against_brute(s);
			}
		}
	}
}

TEST_CASE("suffix_array: random strings match brute force", "[strings]") {
	auto seed = GENERATE(take(40, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 26);
	std::string s(1 + std::size_t(rng() % 300), 'a');
	for (auto &c : s) c = char('a' + std::size_t(rng() % sigma));
	check_against_brute(s);
}

TEST_CASE("suffix_array: differential vs the doubling variant, larger inputs", "[strings]") {
	auto seed = GENERATE(take(8, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 4);
	std::size_t const n = 5000 + std::size_t(rng() % 15000);
	std::vector<std::uint32_t> mapped(n);
	for (auto &v : mapped) v = 1 + std::uint32_t(std::size_t(rng() % sigma));
	suffix_array const sa(mapped);
	std::vector<std::uint32_t> text = mapped, base(n + 1);
	text.push_back(0); // the sentinel; mapped symbols are already >= 1
	atl::experimental::suffix_sort_by_prefix_doubling(text, base, std::uint32_t(sigma) + 1);
	base.erase(base.begin()); // the sentinel suffix, rank 0
	REQUIRE(std::vector<std::uint32_t>(sa.positions().begin(), sa.positions().end()) == base);
}

TEST_CASE("suffix_array: differential vs the doubling variant on structured texts", "[strings]") {
	constexpr std::size_t n = 20'000;
	std::mt19937_64 rng(7);
	std::vector<std::vector<std::uint32_t>> texts;
	texts.emplace_back(n, 1);
	auto &abab = texts.emplace_back(n);
	for (std::size_t i = 0; i < n; ++i) abab[i] = 1 + std::uint32_t(i % 2);
	std::vector<std::uint32_t> a{1}, b{1, 2};
	while (b.size() < n) {
		auto next = b;
		next.insert(next.end(), a.begin(), a.end());
		a = std::move(b);
		b = std::move(next);
	}
	b.resize(n);
	texts.push_back(b);
	auto &wide = texts.emplace_back(n);
	for (auto &v : wide) v = 1 + std::uint32_t(std::size_t(rng() % 50'000)); // offset map, sigma > n
	for (auto const &mapped : texts) {
		suffix_array const sa(mapped);
		std::vector<std::uint32_t> text = mapped, base(n + 1);
		text.push_back(0);
		atl::experimental::suffix_sort_by_prefix_doubling(text, base, *std::ranges::max_element(text) + 1);
		base.erase(base.begin());
		REQUIRE(std::vector<std::uint32_t>(sa.positions().begin(), sa.positions().end()) == base);

		for (std::uint32_t r = 0; r + 1 < n; r += 97) {
			std::uint32_t const p = sa.position(r), q = sa.position(r + 1);
			std::uint32_t k = 0;
			while (p + k < n && q + k < n && mapped[p + k] == mapped[q + k]) ++k;
			REQUIRE(sa.lcp_lengths()[r] == k);
		}
	}
}

TEST_CASE("suffix_array: wide-range integral compression matches the compact-alphabet build", "[strings]") {
	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 5);
	std::vector<std::uint32_t> compact(1 + std::size_t(rng() % 3000));
	for (auto &v : compact) v = std::uint32_t(std::size_t(rng() % sigma));
	// an order-preserving spread over a 2^40-wide signed range
	std::vector<std::int64_t> wide(compact.size());
	for (std::size_t i = 0; i < compact.size(); ++i) wide[i] = (std::int64_t(compact[i]) - 2) * (std::int64_t(1) << 40);
	suffix_array const a(compact), b(wide);
	REQUIRE(std::ranges::equal(a.positions(), b.positions()));
	REQUIRE(std::ranges::equal(a.lcp_lengths(), b.lcp_lengths()));
}

TEST_CASE("suffix_array: distinct substrings agree with the suffix automaton", "[strings]") {
	auto seed = GENERATE(take(15, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::string s(1 + std::size_t(rng() % 400), 'a');
	for (auto &c : s) c = char('a' + std::size_t(rng() % 3));

	suffix_array const sa(s);
	std::uint64_t total = std::uint64_t(s.size()) * (s.size() + 1) / 2;
	for (auto const l : sa.lcp_lengths()) total -= l;

	atl::suffix_automaton<26> t;
	t.insert(s, [](char c) { return c - 'a'; });
	REQUIRE(total == t.distinct_substring_count());
}

TEST_CASE("suffix_array: alphabet paths, projections, edges", "[strings]") {
	SECTION("huge-range integrals go through compression") {
		std::vector<std::int64_t> const v{5'000'000'000'000, -7, 5'000'000'000'000, -9'000'000'000'000, -7};
		suffix_array const sa(v);
		// suffixes sorted as the same text over a compact mapped alphabet
		std::vector<std::uint32_t> const mapped{3, 2, 3, 1, 2};
		suffix_array const compact(mapped);
		REQUIRE(std::vector(sa.positions().begin(), sa.positions().end())
				== std::vector(compact.positions().begin(), compact.positions().end()));
		REQUIRE(std::vector(sa.lcp_lengths().begin(), sa.lcp_lengths().end())
				== std::vector(compact.lcp_lengths().begin(), compact.lcp_lengths().end()));
	}
	SECTION("non-integral ordered symbols") {
		std::vector<std::string> const tokens{"beta", "alpha", "beta", "alpha", "gamma"};
		suffix_array const sa(tokens);
		std::vector<std::uint32_t> const mapped{2, 1, 2, 1, 3};
		suffix_array const compact(mapped);
		REQUIRE(std::vector(sa.positions().begin(), sa.positions().end())
				== std::vector(compact.positions().begin(), compact.positions().end()));
	}
	SECTION("projection: case-insensitive") {
		auto const gs2_lower_projection = [](char c) { return char(std::tolower(c)); };
		suffix_array const a(std::string{"AbAb"}, gs2_lower_projection);
		suffix_array const b(std::string{"abab"});
		REQUIRE(std::vector(a.positions().begin(), a.positions().end())
				== std::vector(b.positions().begin(), b.positions().end()));
	}
	SECTION("edges") {
		suffix_array const e{};
		REQUIRE(e.empty());
		REQUIRE(e.positions().empty());
		REQUIRE(e.lcp_lengths().empty());

		suffix_array const one(std::string{"z"});
		REQUIRE(one.size() == 1);
		REQUIRE(one.position(0) == 0);
		REQUIRE(one.lcp_lengths().empty());

		suffix_array const same(std::string(6, 'a')); // aaaaaa: sa = 5,4,3,2,1,0
		for (std::uint32_t r = 0; r < 6; ++r) REQUIRE(same.position(r) == 5 - r);
		for (std::uint32_t r = 0; r < 5; ++r) REQUIRE(same.lcp_lengths()[r] == r + 1);
	}
}

TEST_CASE("suffix_array: the doubling contender handles symbols larger than the text", "[strings]") {
	// Offset mapping can produce symbols up to 8n; subsequent counting
	// rounds need dense ranks to stay inside their n + 1 buckets.
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (int round = 0; round < 50; ++round) {
		std::uint32_t const n = std::uint32_t(1 + std::size_t(rng() % 50)), sigma = 8 * n;
		std::vector<std::uint32_t> text(n);
		for (auto &symbol : text) symbol = std::uint32_t(std::size_t(rng() % sigma));
		std::vector<std::uint32_t> expected(n);
		for (std::uint32_t i = 0; i < n; ++i) expected[i] = i;
		std::ranges::sort(expected, [&](std::uint32_t const a, std::uint32_t const b) {
			return std::ranges::lexicographical_compare(text.begin() + a, text.end(), text.begin() + b, text.end());
		});
		std::vector<std::uint32_t> shifted(n + 1, 0), got(n + 1);
		for (std::uint32_t i = 0; i < n; ++i) shifted[i] = text[i] + 1; // 0 is the sentinel's
		atl::experimental::suffix_sort_by_prefix_doubling(shifted, got, sigma + 1);
		got.erase(got.begin());
		REQUIRE(got == expected);
	}
	std::vector<std::uint32_t> got(3);
	atl::experimental::suffix_sort_by_prefix_doubling(std::vector<std::uint32_t>{6, 4, 0}, got, 7);
	REQUIRE(got == std::vector<std::uint32_t>{2, 1, 0});
}

TEST_CASE("suffix_array: the doubling contender asserts suffix_sort's preconditions", "[strings]") {
	std::vector<std::uint32_t> sa(3);
	std::vector<std::uint32_t> const no_sentinel{2, 1, 1}, ok{2, 1, 0};
	CHECK(atl_test::aborts([&] { atl::experimental::suffix_sort_by_prefix_doubling(no_sentinel, sa, 3); }));
	CHECK(atl_test::aborts([&] {
		atl::experimental::suffix_sort_by_prefix_doubling(std::span<std::uint32_t const>{}, std::span<std::uint32_t>{}, 1);
	}));
	std::vector<std::uint32_t> short_sa(2);
	CHECK(atl_test::aborts([&] { atl::experimental::suffix_sort_by_prefix_doubling(ok, short_sa, 3); }));
	atl::experimental::suffix_sort_by_prefix_doubling(ok, sa, 3);
	CHECK(sa == std::vector<std::uint32_t>{2, 1, 0});
}

TEST_CASE("suffix_array: suffix_sort asserts its sentinel preconditions", "[strings]") {
	std::vector<std::uint32_t> sa(3);
	std::vector<std::uint32_t> const no_sentinel{2, 1, 1}, ok{2, 1, 0};
	CHECK(atl_test::aborts([&] { atl::detail::suffix_sort(no_sentinel, sa, 3); }));
	CHECK(atl_test::aborts([&] { atl::detail::suffix_sort(std::span<std::uint32_t const>{}, std::span<std::uint32_t>{}, 1); }));
	std::vector<std::uint32_t> short_sa(2);
	CHECK(atl_test::aborts([&] { atl::detail::suffix_sort(ok, short_sa, 3); }));
	atl::detail::suffix_sort(ok, sa, 3);
	CHECK(sa == std::vector<std::uint32_t>{2, 1, 0});
}

TEST_CASE("suffix_array: million-symbol unary and random texts", "[strings][deep]") {
	constexpr atl::uint32 length = 1'000'000;
	{
		atl::suffix_array const suffixes(std::string(length, 'a'));
		for (atl::uint32 rank = 0; rank < length; ++rank) {
			REQUIRE(suffixes.position(rank) == length - 1 - rank);
			if (rank + 1 < length) REQUIRE(suffixes.lcp_lengths()[rank] == rank + 1);
		}
	}
	std::mt19937_64 rng(729);
	std::vector<atl::uint32> text(length);
	for (auto &symbol : text) symbol = 1 + atl::uint32(std::size_t(rng() % 26));
	atl::suffix_array const suffixes(text);
	text.push_back(0);
	std::vector<atl::uint32> expected(text.size());
	atl::experimental::suffix_sort_by_prefix_doubling(text, expected, 27);
	REQUIRE(std::ranges::equal(suffixes.positions(), expected | std::views::drop(1)));
	for (atl::uint32 rank = 0; rank + 1 < length; ++rank) {
		auto const a = suffixes.position(rank), b = suffixes.position(rank + 1);
		atl::uint32 common = 0;
		while (common < length - a && common < length - b && text[a + common] == text[b + common]) ++common;
		REQUIRE(suffixes.lcp_lengths()[rank] == common);
	}
}
