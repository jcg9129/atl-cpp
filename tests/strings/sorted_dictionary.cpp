#include <atl/experimental/strings/sorted_dictionary_variants.hpp>
#include <atl/strings/lexicographic_order.hpp>
#include <atl/strings/sorted_dictionary.hpp>
#include <atl/strings/static_trie.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <forward_list>
#include <limits>
#include <optional>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "input_only.hpp"

using atl::sorted_dictionary;

namespace {
	atl::index_interval brute_interval(std::vector<std::string> const &sorted, std::string const &p) {
		std::uint32_t lo = 0;
		while (lo < sorted.size() && !sorted[lo].starts_with(p) && sorted[lo] < p) ++lo;
		std::uint32_t hi = lo;
		while (hi < sorted.size() && sorted[hi].starts_with(p)) ++hi;
		return {lo, hi};
	}
}

TEST_CASE("sorted_dictionary: prefix intervals match brute force and the trie", "[strings]") {
	auto seed = GENERATE(take(40, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::vector<std::string> words(1 + std::size_t(rng() % 40));
	for (auto &w : words) {
		w.resize(std::size_t(rng() % 9)); // includes empty words
		for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
	}
	std::ranges::sort(words);
	sorted_dictionary const dict(words);
	auto const gs2_lower_projection = [](char c) { return c - 'a'; };
	atl::static_trie<26> const trie(words, gs2_lower_projection);

	REQUIRE(dict.prefix_interval(std::string{})
			== atl::index_interval{0, std::uint32_t(words.size())});
	for (int probe = 0; probe < 400; ++probe) {
		std::string p(std::size_t(rng() % 7), 'a');
		for (auto &c : p) c = char('a' + rng() % (sigma + (probe % 2)));
		auto const expect = brute_interval(words, p);
		REQUIRE(dict.prefix_interval(p) == expect);
		// differential vs the trie-walk route
		std::optional cur{trie.root()};
		for (char const c : p)
			if (cur && !(cur = trie.walk(*cur, gs2_lower_projection(c)))) break;
		if (cur) REQUIRE(trie.word_interval(*cur) == expect);
	}
}

TEST_CASE("sorted_dictionary: projections, symbol types, edges", "[strings]") {
	SECTION("query through a different projection") {
		auto const gs2_lower_projection = [](char c) { return char(std::tolower(c)); };
		std::vector<std::string> const words{"ab", "abc", "b"};
		sorted_dictionary const dict(words);
		REQUIRE(dict.prefix_interval(std::string{"AB"}, gs2_lower_projection)
				== atl::index_interval{0, 2});
	}
	SECTION("non-char symbols, cross-type query") {
		std::vector<std::vector<atl::int64>> const words{{-5, 2}, {-5, 2, 9}, {0}, {7, 7}};
		sorted_dictionary const dict(words);
		REQUIRE(dict.prefix_interval(std::vector<atl::int64>{-5, 2})
				== atl::index_interval{0, 2});
		REQUIRE(dict.prefix_interval(std::vector<atl::int64>{7})
				== atl::index_interval{3, 4});
		REQUIRE(dict.prefix_interval(std::vector<atl::int64>{8})
				== atl::index_interval{4, 4});
	}
	SECTION("duplicates and misses") {
		std::vector<std::string> const words{"aa", "aa", "aa", "ab"};
		sorted_dictionary const dict(words);
		REQUIRE(dict.prefix_interval(std::string{"aa"}) == atl::index_interval{0, 3});
		REQUIRE(dict.prefix_interval(std::string{"a"}) == atl::index_interval{0, 4});
		REQUIRE(dict.prefix_interval(std::string{"aab"}) == atl::index_interval{3, 3});
		REQUIRE(dict.prefix_interval(std::string{"b"}) == atl::index_interval{4, 4});
	}
	SECTION("empty dictionary; moves") {
		sorted_dictionary const empty(std::vector<std::string>{});
		REQUIRE(empty.empty());
		REQUIRE(empty.prefix_interval(std::string{"a"}) == atl::index_interval{0, 0});

		sorted_dictionary a(std::vector<std::string>{"x", "xy"});
		auto const b = std::move(a);
		REQUIRE(b.prefix_interval(std::string{"x"}) == atl::index_interval{0, 2});
	}
}

TEST_CASE("sorted_dictionary: bulk materialization paths agree", "[strings]") {
	// identity over contiguous words (memmove path), a projection (transform
	// path) and non-contiguous words all build the same dictionary
	std::vector<std::string> const words{"", "ab", "abc", "abc", "b", "ba"};
	std::vector<std::deque<char>> deques;
	for (auto const &w : words) deques.emplace_back(w.begin(), w.end());
	auto const upper = [](char c) { return char(std::toupper(c)); };
	sorted_dictionary const plain(words);
	sorted_dictionary const projected(words, upper);
	sorted_dictionary const chunked(deques);
	for (std::string const p : {"", "a", "ab", "abc", "abcd", "b", "c"}) {
		std::string up = p;
		for (auto &c : up) c = upper(c);
		REQUIRE(plain.prefix_interval(p) == brute_interval(words, p));
		REQUIRE(projected.prefix_interval(up) == brute_interval(words, p));
		REQUIRE(chunked.prefix_interval(p) == brute_interval(words, p));
	}
}

TEST_CASE("sorted_dictionary: wide symbols match brute force, including wide intervals", "[strings]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	// a shared stem makes long runs of words share the query prefix
	std::vector<std::vector<std::int64_t>> words(1 + std::size_t(rng() % 300));
	for (auto &w : words) {
		w.assign(std::size_t(rng() % 3), 7);
		for (std::size_t k = std::size_t(rng() % 6); k-- > 0;) w.push_back(std::int64_t(std::size_t(rng() % sigma)) - 1);
	}
	std::ranges::sort(words);
	sorted_dictionary const dict(words);
	auto const starts_with = [](std::vector<std::int64_t> const &w, std::vector<std::int64_t> const &p) {
		return p.size() <= w.size() && std::ranges::equal(p, w | std::views::take(p.size()));
	};
	for (int probe = 0; probe < 300; ++probe) {
		std::vector<std::int64_t> p(std::size_t(rng() % 3), 7);
		for (std::size_t k = std::size_t(rng() % 4); k-- > 0;) p.push_back(std::int64_t(std::size_t(rng() % (sigma + 1))) - 1);
		std::uint32_t lo = 0;
		while (lo < words.size() && words[lo] < p && !starts_with(words[lo], p)) ++lo;
		std::uint32_t hi = lo;
		while (hi < words.size() && starts_with(words[hi], p)) ++hi;
		REQUIRE(dict.prefix_interval(p) == atl::index_interval{lo, hi});
	}
}

namespace {
	/** @brief A symbol counting its copies and moves, to see how the build takes it in. */
	struct counted_symbol {
		int v;
		static inline int copies = 0, moves = 0;

		counted_symbol(int x) : v(x) {}

		counted_symbol(counted_symbol const &o) : v(o.v) { ++copies; }

		counted_symbol(counted_symbol &&o) noexcept : v(o.v) {
			++moves;
			o.v = -1;
		}

		counted_symbol &operator=(counted_symbol const &) = default;
		counted_symbol &operator=(counted_symbol &&) = default;
		auto operator<=>(counted_symbol const &) const = default;
	};

	std::vector<std::vector<counted_symbol>> counted_words() {
		std::vector<std::vector<counted_symbol>> words;
		words.push_back({1, 2});
		words.push_back({1, 2, 3});
		words.push_back({2});
		return words;
	}
}

TEST_CASE("sorted_dictionary: an rvalue word list is moved from, an lvalue one copied", "[strings]") {
	auto words = counted_words();
	counted_symbol::copies = counted_symbol::moves = 0;
	sorted_dictionary const copied(words);
	CHECK(counted_symbol::copies == 6);
	CHECK(counted_symbol::moves == 0);
	CHECK(words[0][0].v == 1);

	counted_symbol::copies = counted_symbol::moves = 0;
	sorted_dictionary const moved(std::move(words));
	CHECK(counted_symbol::copies == 0);
	CHECK(counted_symbol::moves == 6);
	CHECK(words[0][0].v == -1);
	std::vector<counted_symbol> const query{1, 2};
	CHECK(moved.prefix_interval(query) == atl::index_interval{0, 2});
	CHECK(moved.prefix_interval(query) == copied.prefix_interval(query));
}

namespace {
	using atl::experimental::sorted_dictionary_by_parts_build;
	using atl::experimental::sorted_dictionary_options;

	/** @brief Every single-knob contender, plus all knobs at once. */
	constexpr std::array variant_options{
		sorted_dictionary_options{},
		sorted_dictionary_options{.is_novector = true},
		sorted_dictionary_options{.is_sym_reserved = true},
		sorted_dictionary_options{.is_linear_upper_scan = true},
		sorted_dictionary_options{.is_descent_lcp = true},
		sorted_dictionary_options{.is_novector = true,
								  .is_sym_reserved = true,
								  .is_linear_upper_scan = true,
								  .is_descent_lcp = true},
	};

	/** @brief The symbol-reserving contender's options. */
	constexpr sorted_dictionary_options reserved_options{.is_sym_reserved = true};

	/** @brief Contender `variant_options[Index]` answers every query as @p library does. */
	template<std::size_t Index, typename Words, typename Symbol, typename Query>
		requires atl::sortable_word_range<Words const &>
	void check_variant(Words const &words, sorted_dictionary<Symbol> const &library, std::vector<Query> const &queries) {
		sorted_dictionary_by_parts_build<Symbol, variant_options[Index]> const dict(words);
		REQUIRE(dict.size() == library.size());
		for (auto const &q : queries) REQUIRE(dict.prefix_interval(q) == library.prefix_interval(q));
	}

	/** @brief Every contender in `variant_options` answers every query as @p library does. */
	template<typename Words, typename Symbol, typename Query, std::size_t... Index>
		requires atl::sortable_word_range<Words const &>
	void check_every_variant(Words const &words, sorted_dictionary<Symbol> const &library,
							 std::vector<Query> const &queries, std::index_sequence<Index...>) {
		(check_variant<Index>(words, library, queries), ...);
	}

	/**
	 * @brief Random sorted words over @p make_symbol, with shared stems so
	 *        intervals run both narrow and past the 16-word upper scan; every
	 *        contender must answer every query as the library class does.
	 */
	template<std::totally_ordered Symbol, typename MakeSymbol>
		requires std::is_invocable_r_v<Symbol, MakeSymbol const &, std::size_t>
	void check_variants(std::mt19937_64 &rng, MakeSymbol const &make_symbol) {
		std::vector<std::vector<Symbol>> words(std::size_t(rng() % 4) == 0 ? std::size_t(rng() % 4) : std::size_t(rng() % 400));
		std::size_t const stem = std::size_t(rng() % 4);
		for (auto &w : words) {
			w.resize(std::size_t(rng() % (stem + 1)));
			for (auto &e : w) e = make_symbol(0);
			for (std::size_t k = std::size_t(rng() % 6); k-- > 0;) w.push_back(make_symbol(1 + std::size_t(rng() % 3)));
		}
		std::ranges::sort(words);
		sorted_dictionary const library(words);
		std::vector<std::vector<Symbol>> queries(200);
		for (auto &q : queries) {
			if (!words.empty() && std::size_t(rng() % 2)) {
				auto const &w = words[std::size_t(rng() % words.size())];
				q.assign(w.begin(), w.begin() + std::ptrdiff_t(rng() % (w.size() + 1)));
				if (!q.empty() && std::size_t(rng() % 4) == 0) q[std::size_t(rng() % q.size())] = make_symbol(1 + std::size_t(rng() % 4));
			} else {
				q.resize(std::size_t(rng() % (stem + 1)));
				for (auto &e : q) e = make_symbol(0);
				for (std::size_t k = std::size_t(rng() % 4); k-- > 0;) q.push_back(make_symbol(1 + std::size_t(rng() % 4)));
			}
		}
		check_every_variant(words, library, queries, std::make_index_sequence<variant_options.size()>{});
	}

	/** @brief The base-4 words "000" .. "333" through contender @p Options, against the library class. */
	template<sorted_dictionary_options Options>
	constexpr bool variant_matches_library() {
		std::array<std::array<char, 3>, 64> words{};
		for (int i = 0; i < 64; ++i) words[i] = {char('0' + i / 16), char('0' + i / 4 % 4), char('0' + i % 4)};
		sorted_dictionary const library(words);
		sorted_dictionary_by_parts_build<char, Options> const dict(words);
		for (std::string_view q : {"", "0", "1", "13", "2", "21", "212", "3", "33", "333", "4", "0000", "/"})
			if (dict.prefix_interval(q) != library.prefix_interval(q)) return false;
		return true;
	}

	/** @brief Every contender in `variant_options` passes @ref variant_matches_library. */
	template<std::size_t... Index>
	constexpr bool every_variant_matches_library(std::index_sequence<Index...>) {
		return (variant_matches_library<variant_options[Index]>() && ...);
	}

	static_assert(every_variant_matches_library(std::make_index_sequence<variant_options.size()>{}));
}

TEST_CASE("sorted_dictionary contenders match the library class", "[strings]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	// symbol 0 builds the shared stems
	check_variants<char>(rng, [](std::size_t s) { return char('a' + s); });
	check_variants<std::uint64_t>(rng, [](std::size_t s) { return std::uint64_t(s) << 40; });
	check_variants<std::string>(rng, [](std::size_t s) { return std::string(1 + s % 2, char('a' + s)); });
}

TEST_CASE("sorted_dictionary contenders: the pre-sized build copies each symbol once, moves none", "[strings]") {
	auto words = counted_words();
	counted_symbol::copies = counted_symbol::moves = 0;
	sorted_dictionary_by_parts_build<counted_symbol, reserved_options> const copied(words);
	CHECK(counted_symbol::copies == 6);
	CHECK(counted_symbol::moves == 0);

	counted_symbol::copies = counted_symbol::moves = 0;
	sorted_dictionary_by_parts_build<counted_symbol, reserved_options> const moved(std::move(words));
	CHECK(counted_symbol::copies == 0);
	CHECK(counted_symbol::moves == 6);
	std::vector<counted_symbol> const query{1, 2};
	CHECK(moved.prefix_interval(query) == copied.prefix_interval(query));
}

TEST_CASE("sorted_dictionary contenders: intervals around the 16-entry linear upper scan", "[strings]") {
	std::vector<std::string_view> const queries{"", "b", "ba", "a", "c", "bb", "baa"};
	for (std::size_t prefix_words = 0; prefix_words <= 2; ++prefix_words)
		for (std::size_t run = 1; run <= 20; ++run)
			for (std::size_t suffix_words = 0; suffix_words <= 2; ++suffix_words) {
				std::vector<std::string> words(prefix_words, "a");
				words.insert(words.end(), run, "ba");
				words.insert(words.end(), suffix_words, "c");
				sorted_dictionary const library(words);
				INFO(prefix_words << ' ' << run << ' ' << suffix_words);
				check_every_variant(words, library, queries, std::make_index_sequence<variant_options.size()>{});
			}
}

TEST_CASE("sorted_dictionary: descent edge sizes and long shared prefixes", "[strings]") {
	for (std::size_t const n : {0u, 1u, 2u, 3u, 4u, 7u, 8u, 9u, 63u, 64u, 65u, 511u, 512u, 513u, 4096u}) {
		std::vector<std::string> words;
		for (std::size_t i = 0; i < n; ++i) words.push_back(std::string(80, 'a') + std::to_string(i / 3));
		std::ranges::sort(words);
		sorted_dictionary const dict(words);
		// The parts-build parts build with RMQ descent certifies the new array indices.
		sorted_dictionary_by_parts_build<char> const oracle(words);
		for (std::string const &q : std::vector<std::string>{"", "a", std::string(80, 'a'), std::string(80, 'a') + "10", std::string(80, 'a') + "100x", "b"}) {
			CHECK(dict.prefix_interval(q) == brute_interval(words, q));
			CHECK(dict.prefix_interval(q) == oracle.prefix_interval(q));
		}
		for (auto const &q : words) CHECK(dict.prefix_interval(q) == brute_interval(words, q));
	}
}

TEST_CASE("sorted_dictionary: reserve respects range category and projection", "[strings]") {
	std::vector<std::string> const words{"", "ab", "abc", "b"};
	sorted_dictionary const single_pass{test::input_only(words)};
	std::forward_list<std::string> forward(words.begin(), words.end());
	sorted_dictionary const multi_pass(forward);
	int calls = 0;
	// A mutable view need not be iterable as const; the size pass must not
	// project symbols or require const iteration of the outer range.
	auto view = words | std::views::filter([count = 0](auto const &) mutable { return ++count > 0; });
	sorted_dictionary const projected(view, [&](char c) { ++calls; return c; });
	CHECK(calls == 6);
	for (std::string_view q : {"", "a", "ab", "abc", "abcd", "b", "c"}) {
		auto const expected = brute_interval(words, std::string(q));
		CHECK(single_pass.prefix_interval(q) == expected);
		CHECK(multi_pass.prefix_interval(q) == expected);
		CHECK(projected.prefix_interval(q) == expected);
	}
}

TEST_CASE("strings GU: queries of another signedness compare by value", "[strings][sorted_dictionary]") {
	std::vector<std::vector<int>> const words{{-1}, {0}, {1}};
	std::vector<unsigned> const zero{0};
	std::vector<unsigned> const largest{(std::numeric_limits<unsigned>::max)()};
	std::vector<std::vector<signed char>> const bytes{{-1}, {0}, {1}};
	std::vector<unsigned char> const byte_zero{0};
	std::vector<unsigned char> const byte_largest{255};

	atl::sorted_dictionary const dictionary(words);
	REQUIRE(dictionary.prefix_interval(zero) == atl::index_interval{1, 2});
	REQUIRE(dictionary.prefix_interval(largest) == atl::index_interval{3, 3});
	atl::sorted_dictionary const byte_dictionary(bytes);
	REQUIRE(byte_dictionary.prefix_interval(byte_zero) == atl::index_interval{1, 2});
	REQUIRE(byte_dictionary.prefix_interval(byte_largest) == atl::index_interval{3, 3});

	atl::experimental::sorted_dictionary_by_parts_build<int> const parts(words);
	REQUIRE(parts.prefix_interval(zero) == atl::index_interval{1, 2});
	REQUIRE(parts.prefix_interval(largest) == atl::index_interval{3, 3});
	atl::experimental::sorted_dictionary_by_inplace_build<int> const inplace(words);
	REQUIRE(inplace.prefix_interval(zero) == atl::index_interval{1, 2});
	REQUIRE(inplace.prefix_interval(largest) == atl::index_interval{3, 3});
	atl::experimental::sorted_dictionary_by_parts_build<signed char> const byte_parts(bytes);
	REQUIRE(byte_parts.prefix_interval(byte_zero) == atl::index_interval{1, 2});
	REQUIRE(byte_parts.prefix_interval(byte_largest) == atl::index_interval{3, 3});
	atl::experimental::sorted_dictionary_by_inplace_build<signed char> const byte_inplace(bytes);
	REQUIRE(byte_inplace.prefix_interval(byte_zero) == atl::index_interval{1, 2});
	REQUIRE(byte_inplace.prefix_interval(byte_largest) == atl::index_interval{3, 3});
}
