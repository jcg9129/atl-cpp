#include <atl/strings/dense_transitions.hpp>
#include <atl/strings/lyndon.hpp>
#include <atl/strings/matching.hpp>
#include <atl/strings/palindrome_lengths.hpp>
#include <atl/strings/palindromic_tree.hpp>
#include <atl/strings/sorted_dictionary.hpp>
#include <atl/strings/static_trie.hpp>
#include <atl/strings/string_hash.hpp>
#include <atl/strings/suffix_array.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/prefetch.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <functional>
#include <list>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
	constexpr bool matching_proof() {
		using std::string_view;
		auto const text = string_view("aabaab");
		auto const borders = atl::border_lengths(text);
		auto const common = atl::common_prefix_lengths(text);
		return borders[5] == 3 && common[3] == 3
			&& atl::occurrences(string_view("aaaaa"), string_view("aaa"), atl::border_lengths(string_view("aaa"))).size() == 3
			&& atl::to_border_lengths(common) == borders && atl::to_common_prefix_lengths(borders) == common
			&& atl::border_lengths(string_view()).empty() && atl::common_prefix_lengths(string_view()).empty()
			&& atl::to_border_lengths(std::array<atl::uint32, 0>{}).empty()
			&& atl::to_common_prefix_lengths(std::array<atl::uint32, 0>{}).empty();
	}

	constexpr bool palindrome_proof() {
		atl::palindrome_lengths const text(std::string_view("abacaba"));
		atl::palindrome_lengths const even(std::string_view("abba"));
		atl::palindrome_lengths const empty;
		return text.is_palindrome(0, 7) && !text.is_palindrome(0, 2) && text.odd_length(3) == 7
			&& text.size() == 7 && !text.empty() && even.even_length(2) == 4
			&& empty.empty() && empty.size() == 0 && empty.is_palindrome(0, 0);
	}

	constexpr bool hash_proof() {
		atl::string_hash const text(std::string_view("abcab"), 131);
		atl::string_hash online(131);
		online.reserve(5);
		online.push_back('a');
		online.push_back('b');
		atl::string_hash const empty(131);
		atl::string_hash const empty_range(std::string_view(), 131);
		return text.equal(0, 2, 3, 5) && !text.equal(0, 2, 2, 4)
			&& text.lcp(0, 3) == 2 && text.lcp(1, 4) == 1 && text.lcp(0, online, 0) == 2
			&& text.concat(text.substring_hash(0, 1), text.substring_hash(1, 3), 2) == text.substring_hash(0, 3)
			&& atl::hash_of(std::string_view("abcab"), 131) == text.full_hash()
			&& online.full_hash() == text.substring_hash(3, 5) && online.size() == 2 && online.base() == 131
			&& empty.full_hash() == 0 && empty_range.full_hash() == 0
			&& empty.concat(text.full_hash(), text.full_hash(), 5) == atl::hash_of(std::string_view("abcababcab"), 131);
	}

	constexpr bool wide_hash_proof() {
		atl::uint128 const wide = (atl::uint128(1) << 100) + 17;
		std::array const symbols{wide};
		atl::uint64 const digit = 1 + atl::uint64(wide % (atl::string_hash::mod - 1));
		return atl::hash_of(symbols, 131) == digit && digit != 18
			&& atl::hash_of(std::array{atl::uint8(255)}, 131) == 256
			&& atl::hash_of(std::array{atl::int8(-1)}, 131) == 256
			&& atl::hash_of(std::array{atl::int16(-1)}, 131) == 65536
			&& atl::hash_of(std::array{atl::uint64(atl::string_hash::mod - 1)}, 131) == atl::hash_of(std::array{atl::uint64(0)}, 131)
			&& atl::hash_of(std::array{atl::uint64(atl::string_hash::mod - 2)}, 131) == atl::string_hash::mod - 1;
	}

	constexpr bool lyndon_proof() {
		auto const repetitions = atl::runs(std::string_view("mississippi"));
		return atl::lyndon_factorization(std::string_view("banana")) == std::vector<atl::uint32>{0, 1, 3, 5, 6}
			&& atl::minimal_rotation_index(std::string_view("cab")) == 1
			&& repetitions.size() == 4 && repetitions[0] == atl::run{1, 2, 4}
			&& atl::lyndon_factorization(std::string_view()) == std::vector<atl::uint32>{0}
			&& atl::minimal_rotation_index(std::string_view()) == 0 && atl::runs(std::string_view()).empty()
			&& atl::runs(std::array{atl::int8(-1), atl::int8(-1)}) == std::vector<atl::run>{{1, 0, 2}}
			&& atl::runs(std::array{std::string_view("x"), std::string_view("x")}) == std::vector<atl::run>{{1, 0, 2}};
	}

	struct equality_symbol {
		int value;
		friend constexpr bool operator==(equality_symbol, equality_symbol) = default;
	};

	constexpr int select_text(atl::comparable_text auto const &) { return 1; }

	constexpr int select_text(atl::sortable_text auto const &) { return 2; }

	static_assert(atl::sortable_text<std::string>);
	static_assert(atl::sortable_text<std::span<atl::uint32 const>>);
	static_assert(atl::sortable_text<std::vector<int>, decltype([](int value) { return -value; })>);
	static_assert(!atl::comparable_text<std::list<char>>);
	static_assert(atl::comparable_text<std::vector<equality_symbol>>);
	static_assert(!atl::sortable_text<std::vector<equality_symbol>>);
	static_assert(equality_symbol{1} == equality_symbol{1} && !(equality_symbol{1} == equality_symbol{2}));
	static_assert(select_text(std::string_view("ab")) == 2);
	static_assert(select_text(std::array{equality_symbol{1}}) == 1);
	static_assert(matching_proof());
	static_assert(palindrome_proof());
	static_assert(hash_proof());
	static_assert(wide_hash_proof());
	static_assert(lyndon_proof());
	static_assert(atl::run{1, 0, 2} < atl::run{2, 0, 4});
}

TEST_CASE("strings GS1: constexpr contracts also hold at runtime", "[strings][constexpr]") {
	REQUIRE(matching_proof());
	REQUIRE(palindrome_proof());
	REQUIRE(hash_proof());
	REQUIRE(wide_hash_proof());
	REQUIRE(lyndon_proof());
}

namespace {
	constexpr bool suffix_array_proof() {
		std::array<atl::uint32, 1> const sentinel{0};
		std::array<atl::uint32, 1> sentinel_order{};
		atl::detail::suffix_sort(sentinel, sentinel_order, 1);
		if (sentinel_order[0] != 0) return false;
		atl::suffix_array const empty;
		atl::suffix_array const text(std::string_view("banana"));
		atl::suffix_array const repeated(std::string_view("abababab"));
		std::array const tokens{std::string_view("b"), std::string_view("a"), std::string_view("c"), std::string_view("a"), std::string_view("b")};
		atl::suffix_array const compressed(tokens);
		std::array<atl::uint32, 5> const positions{3, 1, 4, 0, 2};
		std::array<atl::uint32, 4> const lcp{1, 0, 1, 0};
		return empty.empty() && empty.size() == 0 && empty.ranks().empty() && empty.lcp_lengths().empty()
			&& text.position(0) == 5 && text.position(3) == 0 && text.rank(1) == 2 && text.lcp_lengths()[1] == 3
			&& std::ranges::equal(repeated.positions(), std::array<atl::uint32, 8>{6, 4, 2, 0, 7, 5, 3, 1})
			&& std::ranges::equal(repeated.lcp_lengths(), std::array<atl::uint32, 7>{2, 4, 6, 0, 1, 3, 5})
			&& std::ranges::equal(compressed.positions(), positions) && std::ranges::equal(compressed.lcp_lengths(), lcp)
			&& text.ranks()[1] == 2 && !text.empty();
	}

	constexpr bool dictionary_proof() {
		atl::sorted_dictionary const text(std::array{std::string_view("ab"), std::string_view("abc"), std::string_view("b")});
		if (text.prefix_interval(std::string_view("ab")) != atl::index_interval{0, 2}
			|| text.prefix_interval(std::string_view("c")) != atl::index_interval{3, 3}) return false;
		std::array<std::array<char, 3>, 64> words{};
		for (std::size_t i = 0; i < words.size(); ++i) words[i] = {char('0' + i / 16), char('0' + i / 4 % 4), char('0' + i % 4)};
		atl::sorted_dictionary const dictionary(words);
		auto const query = [&](std::string_view value) { return dictionary.prefix_interval(value); };
		if (dictionary.size() != 64 || dictionary.empty() || query("2") != atl::index_interval{32, 48}
			|| query("13") != atl::index_interval{28, 32} || query("222") != atl::index_interval{42, 43}
			|| query("") != atl::index_interval{0, 64} || query("4") != atl::index_interval{64, 64}
			|| query("1330") != atl::index_interval{32, 32} || query("0") != atl::index_interval{0, 16}) return false;
		for (std::size_t n = 0; n <= 9; ++n) {
			std::vector<std::array<int, 2>> values;
			for (std::size_t i = 0; i < n; ++i) values.push_back({7, int(i / 2)});
			atl::sorted_dictionary source(values);
			auto moved = std::move(source);
			atl::sorted_dictionary assigned(std::array<std::array<int, 2>, 0>{});
			assigned = std::move(moved);
			for (int c = -1; c <= 5; ++c) {
				std::array const q{7, c};
				atl::uint32 lo = 0;
				while (lo < n && values[lo] < q) ++lo;
				atl::uint32 hi = lo;
				while (hi < n && values[hi] == q) ++hi;
				if (assigned.prefix_interval(q) != atl::index_interval{lo, hi}) return false;
			}
		}
		return true;
	}

	constexpr bool symbol_buffer_proof() {
		std::array const words{std::array{atl::uint8(1), atl::uint8(2)}, std::array{atl::uint8(3), atl::uint8(4)}};
		std::vector<atl::uint8> symbols;
		std::identity projection;
		atl::detail::append_projected(symbols, words[0], projection);
		return atl::detail::total_word_size(words) == 4 && symbols == std::vector<atl::uint8>{1, 2}
			&& atl::index_interval{0, 2} == atl::index_interval{0, 2};
	}

	static_assert(suffix_array_proof());
	static_assert(dictionary_proof());
	static_assert(symbol_buffer_proof());
}

TEST_CASE("strings GS2: constexpr suffix and dictionary contracts", "[strings][constexpr]") {
	REQUIRE(suffix_array_proof());
	REQUIRE(dictionary_proof());
	REQUIRE(symbol_buffer_proof());
}

namespace {
	constexpr bool static_trie_proof() {
		using std::string_view;
		auto const sym = [](char c) { return c - 'a'; };
		atl::static_trie<26> const empty;
		if (!empty.empty() || empty.size() != 0 || empty.node_count() != 1 || empty.edge_count() != 0
			|| !empty.children(empty.root()).empty() || empty.word_count(empty.root()) != 0
			|| empty.contains(string_view("a"), sym) || empty.find(string_view(), sym) != empty.root()) return false;
		atl::static_trie<26> const t(std::array{string_view("ab"), string_view("abc"), string_view("b")}, sym);
		auto const a = t.walk(t.root(), 0);
		if (!a) return false;
		auto const ab = t.walk(*a, 1);
		if (!ab || t.word_interval(*ab) != atl::index_interval{0, 2}) return false;
		if (t.count(string_view("ab"), sym) != 1 || t.count_with_prefix(string_view("ab"), sym) != 2
			|| t.contains(string_view("c"), sym) || t.node_count() != 5 || t.find(string_view("ab"), sym) != 2u
			|| t.find(string_view("ba"), sym) || t.subtree_end(1) != 4 || t.subtree_end(t.root()) != 5
			|| t.word_order().size() != 3 || t.word_order()[2] != 2 || t.walk(0, -1) || t.walk(0, atl::uint64(1) << 32)) return false;
		atl::static_trie<26> const duplicates(std::array{string_view("ab"), string_view("ab"), string_view("b")}, sym);
		auto const kids = duplicates.children(duplicates.root());
		atl::static_trie<26> copy = duplicates;
		atl::static_trie<26> assigned;
		assigned = std::move(copy);
		return !duplicates.empty() && duplicates.size() == 3 && duplicates.edge_count() == 3 && kids.size() == 2
			&& kids[0].symbol == 0 && kids[1].symbol == 1 && duplicates.prefix_count(kids[0].target) == 2
			&& duplicates.word_count(kids[0].target) == 0 && duplicates.word_count(duplicates.children(kids[0].target)[0].target) == 2
			&& duplicates.word_count(kids[1].target) == 1 && assigned.count(string_view("ab"), sym) == 2;
	}

	constexpr bool dense_transitions_proof() {
		atl::dense_transitions const t({1, atl::dense_transitions::none, 0, 1}, 2);
		atl::prefetch(&t);
		atl::prefetch<true>(&t);
		atl::dense_transitions const empty;
		atl::dense_transitions empty_copy(empty);
		return t.state_count() == 2 && t.alphabet_size() == 2 && t(0, 0) == 1 && t(0, 1) == atl::dense_transitions::none
			&& t.row(1).size() == 2 && t.row(1)[0] == 0 && empty_copy.state_count() == 0 && empty.alphabet_size() == 1;
	}

	constexpr bool dense_ownership_proof() {
		auto data = std::unique_ptr<atl::uint32[]>(new atl::uint32[4]{1, 0, 0, 1});
		auto original = atl::detail::dense_transitions_builder::adopt(std::move(data), 4, 2);
		auto copy = original;
		atl::dense_transitions assigned;
		assigned = original;
		auto moved = std::move(original);
		if (original.state_count() != 0 || copy.row(0)[1] != 0) return false;
		original = std::move(moved);
		assigned = atl::dense_transitions({0}, 1);
		return moved.state_count() == 0 && original(0, 0) == 1 && copy(1, 0) == 0 && assigned(0, 0) == 0;
	}

	static_assert(static_trie_proof());
	static_assert(dense_transitions_proof());
	static_assert(dense_ownership_proof());
}

TEST_CASE("strings GS3: constexpr frozen trie and dense table contracts", "[strings][constexpr]") {
	REQUIRE(static_trie_proof());
	REQUIRE(dense_transitions_proof());
	REQUIRE(dense_ownership_proof());
}

TEST_CASE("strings GU: palindromic tree roots are constexpr members", "[strings][constexpr]") {
	// The tree has no constant constructor; the roots read no state, so a
	// call on a runtime object is still a constant expression.
	atl::palindromic_tree<2> tree;
	static_assert(tree.odd_root() == 0);
	static_assert(tree.even_root() == 1);
	REQUIRE(tree.odd_root() == 0);
	REQUIRE(tree.even_root() == 1);
	REQUIRE(tree.length(tree.odd_root()) == -1);
	REQUIRE(tree.length(tree.even_root()) == 0);
}
