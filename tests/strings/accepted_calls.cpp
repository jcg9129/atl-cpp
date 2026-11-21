#include <atl/experimental/strings/aho_corasick_variants.hpp>
#include <atl/experimental/strings/lexicographic_order_variants.hpp>
#include <atl/experimental/strings/sorted_dictionary_variants.hpp>
#include <atl/experimental/strings/suffix_tree_variants.hpp>
#include <atl/experimental/strings/trie_variants.hpp>
#include <atl/strings/aho_corasick.hpp>
#include <atl/strings/dense_transitions.hpp>
#include <atl/strings/lexicographic_order.hpp>
#include <atl/strings/lyndon.hpp>
#include <atl/strings/matching.hpp>
#include <atl/strings/palindrome_lengths.hpp>
#include <atl/strings/palindromic_tree.hpp>
#include <atl/strings/sorted_dictionary.hpp>
#include <atl/strings/static_trie.hpp>
#include <atl/strings/string_hash.hpp>
#include <atl/strings/suffix_array.hpp>
#include <atl/strings/suffix_automaton.hpp>
#include <atl/strings/suffix_tree.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/prefetch.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <deque>
#include <functional>
#include <limits>
#include <ranges>
#include <string_view>
#include <vector>
#include "input_only.hpp"

namespace {
	template<typename Character>
		requires std::integral<Character>
	void check_character_calls() {
		std::deque<Character> const text{Character(1), Character(2), Character(1)};
		int offset = 3;
		auto projection = [offset](Character value) mutable { return int(value) + offset; };
		auto const constant_projection = [offset](Character value) { return int(value) + offset; };
		auto const borders = atl::border_lengths(text, std::ref(projection));
		REQUIRE(borders == std::vector<atl::uint32>{0, 0, 1});
		auto const common = atl::common_prefix_lengths(text, constant_projection);
		REQUIRE(atl::to_border_lengths(common) == borders);
		REQUIRE(atl::to_common_prefix_lengths(borders) == common);
		REQUIRE(atl::occurrences(text, text, borders, std::ref(projection)) == std::vector<atl::uint32>{0});
		REQUIRE(atl::lyndon_factorization(text, std::ref(projection)) == std::vector<atl::uint32>{0, 2, 3});
		REQUIRE(atl::minimal_rotation_index(text, constant_projection) == 2);
		REQUIRE(atl::runs(text, std::ref(projection)).empty());
		REQUIRE(atl::palindrome_lengths(text, std::ref(projection)).odd_length(1) == 3);
		atl::string_hash const hash(text, 131, std::ref(projection));
		REQUIRE(hash.full_hash() == atl::hash_of(text, 131, constant_projection));
		atl::string_hash online(131);
		online.push_back(Character(1));
		REQUIRE(online.full_hash() == atl::hash_of(std::array{Character(1)}, 131));
		atl::palindromic_tree<16> tree;
		auto const suffix = tree.insert(text, std::ref(projection));
		REQUIRE(tree.length(suffix) == 3);
		REQUIRE(tree.walk(tree.odd_root(), Character(4)).has_value());
		REQUIRE(tree.length(tree.extend(Character(1))) == 1);
		std::size_t children = 0;
		auto visitor = [&children](atl::uint32, atl::uint32) mutable { ++children; };
		tree.for_each_child(tree.odd_root(), std::ref(visitor));
		REQUIRE(children == 3);
	}
}

TEST_CASE("strings GS1: character types and stateful callable adapters", "[strings][constraints]") {
	check_character_calls<char>();
	check_character_calls<signed char>();
	check_character_calls<unsigned char>();
	check_character_calls<char8_t>();
	check_character_calls<wchar_t>();
	check_character_calls<char16_t>();
	check_character_calls<char32_t>();
	check_character_calls<atl::int16>();
}

TEST_CASE("strings GS1: proxies, single-pass inputs and integer widths", "[strings][constraints]") {
	std::vector<bool> bits{true, false, true};
	REQUIRE(atl::border_lengths(bits)[2] == 1);
	REQUIRE(atl::common_prefix_lengths(bits)[2] == 1);
	REQUIRE(atl::palindrome_lengths(bits).odd_length(1) == 3);
	REQUIRE(atl::lyndon_factorization(bits).back() == 3);
	REQUIRE(atl::minimal_rotation_index(bits) == 1);
	REQUIRE(atl::runs(bits).empty());
	auto const as_integer = [](bool bit) { return atl::uint8(bit); };
	REQUIRE(atl::hash_of(bits, 131, as_integer) == atl::string_hash(bits, 131, as_integer).full_hash());
	atl::palindromic_tree<2> bit_tree;
	REQUIRE(bit_tree.length(bit_tree.insert(bits, as_integer)) == 3);
	std::vector<atl::uint8> const symbols{0, 1, 0};
	test::input_only const input(symbols);
	atl::string_hash const hash(input, 131);
	REQUIRE(hash.full_hash() == atl::hash_of(input, 131));
	atl::palindromic_tree<2> tree;
	REQUIRE(tree.length(tree.insert(input)) == 3);
	std::array const signed_symbols{atl::int128(-7), atl::int128(2), atl::int128(-7)};
	REQUIRE(atl::hash_of(signed_symbols, 131) == atl::string_hash(signed_symbols, 131).full_hash());
	REQUIRE(atl::runs(signed_symbols).empty());
	REQUIRE(atl::palindrome_lengths(std::string_view("aba")).is_palindrome(0, 3));
	std::vector<atl::uint8> const no_symbols;
	test::input_only const empty_input(no_symbols);
	REQUIRE(atl::hash_of(empty_input, 131) == 0);
	REQUIRE(tree.insert(empty_input) == tree.even_root());
	atl::palindromic_tree<0> empty_alphabet;
	REQUIRE(empty_alphabet.node_count() == 2);
}

TEST_CASE("strings GS1: byte values preserve their signed comparison order", "[strings][constraints]") {
	std::array<signed char, 3> const signed_bytes{-1, 127, -1};
	std::array<unsigned char, 3> const unsigned_bytes{255, 127, 255};
	REQUIRE(atl::border_lengths(signed_bytes) == atl::border_lengths(unsigned_bytes));
	REQUIRE(atl::palindrome_lengths(signed_bytes).odd_length(1) == 3);
	REQUIRE(atl::minimal_rotation_index(signed_bytes) == 2);
	REQUIRE(atl::minimal_rotation_index(unsigned_bytes) == 1);
	atl::palindromic_tree<256> tree;
	auto const as_byte = [](signed char value) { return static_cast<unsigned char>(value); };
	REQUIRE(tree.length(tree.insert(signed_bytes, as_byte)) == 3);
}

TEST_CASE("strings GS1: projections receive prvalue range references", "[strings][constraints]") {
	auto values = std::views::iota(0, 3);
	auto projection = [modulus = 2](int &&value) { return value % modulus; };
	std::array const expected{0, 1, 0};
	atl::string_hash const hash(values, 131, std::ref(projection));
	REQUIRE(hash.full_hash() == atl::hash_of(expected, 131));
	REQUIRE(atl::hash_of(values, 131, std::ref(projection)) == hash.full_hash());
	atl::palindromic_tree<2> tree;
	REQUIRE(tree.length(tree.insert(values, std::ref(projection))) == 3);
}

namespace {
	template<std::integral Character>
	void gs2_character_calls() {
		std::deque<Character> const text{Character(1), Character(2), Character(1)};
		auto projection = [offset = 1](Character value) mutable { return int(value) + offset; };
		auto const constant_projection = [offset = 1](Character value) { return int(value) + offset; };
		atl::suffix_array const suffixes(text, std::ref(projection));
		REQUIRE(std::ranges::equal(suffixes.positions(), std::array<atl::uint32, 3>{2, 0, 1}));
		atl::suffix_array const constant_suffixes(text, constant_projection);
		REQUIRE(std::ranges::equal(suffixes.positions(), constant_suffixes.positions()));
		std::array const words{std::deque<Character>{Character(1)}, text, text};
		REQUIRE(atl::lexicographic_order(words, std::ref(projection)) == std::vector<atl::uint32>{0, 1, 2});
		atl::sorted_dictionary const dictionary(words, constant_projection);
		REQUIRE(dictionary.prefix_interval(text, std::ref(projection)) == atl::index_interval{1, 3});
		atl::suffix_automaton<8> automaton;
		atl::suffix_tree<8> tree;
		automaton.insert(text, std::ref(projection));
		tree.insert(text, constant_projection);
		REQUIRE(automaton.contains(text, constant_projection));
		REQUIRE(automaton.walk(automaton.root(), Character(2)).has_value());
		REQUIRE(tree.walk(tree.root(), Character(2)).has_value());
		REQUIRE(automaton.length(automaton.extend(Character(2))) == 1);
		REQUIRE(tree.extend(Character(2)) == 1);
		std::size_t transitions = 0, children = 0;
		auto visit_transition = [&transitions](atl::uint32, atl::uint32) mutable { ++transitions; };
		auto visit_child = [&children](atl::uint32) mutable { ++children; };
		automaton.for_each_transition(automaton.root(), std::ref(visit_transition));
		tree.for_each_child(tree.root(), std::ref(visit_child));
		REQUIRE(transitions == 2);
		REQUIRE(children == 2);
	}
}

TEST_CASE("strings GS2: character types, non-contiguous words and callable adapters", "[strings][constraints]") {
	gs2_character_calls<char>();
	gs2_character_calls<signed char>();
	gs2_character_calls<unsigned char>();
	gs2_character_calls<char8_t>();
	gs2_character_calls<wchar_t>();
	gs2_character_calls<char16_t>();
	gs2_character_calls<char32_t>();
	gs2_character_calls<atl::int16>();
}

TEST_CASE("strings GS2: proxies, prvalues, single-pass words and wide integers", "[strings][constraints]") {
	std::vector<bool> bits{true, false, true};
	atl::suffix_array const suffixes(bits);
	REQUIRE(suffixes.position(0) == 1);
	auto const as_integer = [](bool bit) { return atl::uint8(bit); };
	std::array const words{std::vector<bool>{false}, bits};
	REQUIRE(atl::lexicographic_order(words, as_integer) == std::vector<atl::uint32>{0, 1});
	REQUIRE(atl::lexicographic_order(words) == std::vector<atl::uint32>{0, 1});
	atl::sorted_dictionary const dictionary(words, as_integer);
	REQUIRE(dictionary.prefix_interval(bits, as_integer) == atl::index_interval{1, 2});
	std::vector<std::vector<bool>> const single_pass_words(words.begin(), words.end());
	test::input_only const input(single_pass_words);
	REQUIRE(atl::lexicographic_order(input, as_integer) == std::vector<atl::uint32>{0, 1});
	atl::sorted_dictionary const single_pass(input, as_integer);
	REQUIRE(single_pass.prefix_interval(bits, as_integer) == atl::index_interval{1, 2});
	auto values = std::views::iota(0, 3);
	auto projection = [](int &&value) { return value % 2; };
	atl::suffix_automaton<2> automaton;
	atl::suffix_tree<2> tree;
	automaton.insert(values, std::ref(projection));
	tree.insert(values, std::ref(projection));
	REQUIRE(automaton.contains(values, std::ref(projection)));
	REQUIRE(tree.size() == 3);
	atl::suffix_array const wide(std::array{atl::int128(-7), atl::int128(2), atl::int128(-7)});
	REQUIRE(std::ranges::equal(wide.positions(), std::array<atl::uint32, 3>{2, 0, 1}));
}

TEST_CASE("strings GS2: signed byte boundaries and unsigned-byte projections", "[strings][constraints]") {
	std::array<signed char, 4> const signed_bytes{-128, 127, -1, -128};
	atl::suffix_array const signed_suffixes(signed_bytes);
	REQUIRE(std::ranges::equal(signed_suffixes.positions(), std::array<atl::uint32, 4>{3, 0, 2, 1}));
	auto const as_byte = [](signed char value) { return static_cast<unsigned char>(value); };
	atl::suffix_array const byte_suffixes(signed_bytes, as_byte);
	REQUIRE(std::ranges::equal(byte_suffixes.positions(), std::array<atl::uint32, 4>{1, 3, 0, 2}));
	std::vector<std::vector<signed char>> const words(64, std::vector<signed char>{-128, -1});
	auto const order = atl::lexicographic_order(words);
	REQUIRE(order.front() == 0);
	REQUIRE(order.back() == 63);
	atl::sorted_dictionary const dictionary(words);
	REQUIRE(dictionary.prefix_interval(std::array<signed char, 1>{-128}) == atl::index_interval{0, 64});
	std::deque<atl::int64> const extremes{(std::numeric_limits<atl::int64>::min)(), (std::numeric_limits<atl::int64>::max)(), (std::numeric_limits<atl::int64>::min)()};
	atl::suffix_array const wide(extremes);
	REQUIRE(std::ranges::equal(wide.positions(), std::array<atl::uint32, 3>{2, 0, 1}));
}

namespace {
	template<std::integral Character>
	void gs3_character_calls() {
		std::deque<Character> const word{Character(1), Character(2)};
		int offset = 1;
		auto projection = [offset](Character value) mutable { return atl::uint8(value + offset); };
		auto const constant_projection = [offset](Character value) { return atl::uint8(value + offset); };
		atl::trie<8> dynamic;
		auto const terminal = dynamic.insert(word, std::ref(projection));
		REQUIRE(dynamic.find(word, constant_projection) == terminal);
		REQUIRE(dynamic.count(word, std::ref(projection)) == 1);
		REQUIRE(dynamic.contains(word, constant_projection));
		REQUIRE(dynamic.count_with_prefix(word, constant_projection) == 1);
		REQUIRE(dynamic.walk(dynamic.root(), Character(2)));
		REQUIRE(dynamic.word_count(terminal) == 1);
		REQUIRE(dynamic.prefix_count(terminal) == 1);
		std::size_t child_count = 0;
		auto visitor = [&child_count](atl::uint32, atl::uint32) mutable { ++child_count; };
		dynamic.for_each_child(dynamic.root(), std::ref(visitor));
		auto const const_visitor = [&child_count](atl::uint32, atl::uint32) { ++child_count; };
		dynamic.for_each_child(dynamic.root(), const_visitor);
		REQUIRE(child_count == 2);
		REQUIRE(dynamic.erase(word, std::ref(projection)));
		REQUIRE(dynamic.find(word, constant_projection) == terminal);
		std::vector<std::deque<Character>> const words{word, word};
		atl::static_trie<8> const frozen{test::input_only(words), std::ref(projection)};
		REQUIRE(frozen.count(word, constant_projection) == 2);
		REQUIRE(frozen.contains(word, std::ref(projection)));
		REQUIRE(frozen.find(word, constant_projection));
		REQUIRE(frozen.count_with_prefix(word, std::ref(projection)) == 2);
		REQUIRE(frozen.walk(0, Character(2)));
		atl::aho_corasick<8> const ac{test::input_only(words), constant_projection};
		auto const next = ac.step(ac.root(), Character(2));
		REQUIRE(ac.walk(ac.root(), Character(2)) == next);
		REQUIRE(ac.match_count(ac.step(next, Character(3))) == 2);
		atl::experimental::aho_corasick_by_dense_table<8> const dense(words, std::ref(projection));
		REQUIRE(dense.step(0, Character(2)) == next);
		REQUIRE(dense.walk(0, Character(2)) == next);
		atl::experimental::trie_by_global_hash<8> hashed;
		atl::experimental::trie_by_symbol_bst<8> tree;
		hashed.insert(word, std::ref(projection));
		tree.insert(word, constant_projection);
		REQUIRE(hashed.contains(word, constant_projection));
		REQUIRE(tree.count(word, std::ref(projection)) == 1);
		REQUIRE(hashed.count_with_prefix(word, constant_projection) == 1);
		REQUIRE(tree.count_with_prefix(word, constant_projection) == 1);
		REQUIRE(hashed.walk(0, Character(2)));
		REQUIRE(tree.walk(0, Character(2)));
		REQUIRE(hashed.erase(word, constant_projection));
		REQUIRE(tree.erase(word, std::ref(projection)));
	}
}

TEST_CASE("strings GS3: characters, non-contiguous words and callable adapters", "[strings][constraints]") {
	gs3_character_calls<char>();
	gs3_character_calls<signed char>();
	gs3_character_calls<unsigned char>();
	gs3_character_calls<char8_t>();
	gs3_character_calls<wchar_t>();
	gs3_character_calls<char16_t>();
	gs3_character_calls<char32_t>();
	gs3_character_calls<atl::int16>();
}

TEST_CASE("strings GS3: single-pass words and proxy projections", "[strings][constraints]") {
	std::vector<int> const symbols{1, 0};
	atl::trie<2> dynamic;
	dynamic.insert(test::input_only(symbols));
	REQUIRE(dynamic.count(test::input_only(symbols)) == 1);
	REQUIRE(dynamic.contains(test::input_only(symbols)));
	REQUIRE(dynamic.find(test::input_only(symbols)));
	REQUIRE(dynamic.count_with_prefix(test::input_only(symbols)) == 1);
	std::vector<test::input_only<int>> const words{test::input_only(symbols)};
	atl::static_trie<2> const frozen{test::input_only(words)};
	atl::aho_corasick<2> const ac{test::input_only(words)};
	REQUIRE(frozen.count(test::input_only(symbols)) == 1);
	REQUIRE(ac.match_count(ac.step(ac.step(0, 1), 0)) == 1);
	std::vector<bool> bits{true, false};
	auto const bit_value = [](bool value) { return atl::uint8(value); };
	dynamic.insert(bits, bit_value);
	REQUIRE(dynamic.count(bits, bit_value) == 2);
	REQUIRE(dynamic.erase(bits, bit_value));
	auto const proxy_value = [](std::vector<bool>::reference &&value) { return atl::uint8(bool(value)); };
	REQUIRE(dynamic.count(bits, proxy_value) == 1);
	std::vector<std::vector<bool>> const bit_words{bits};
	REQUIRE(atl::static_trie<2>(bit_words, bit_value).contains(bits, bit_value));
	atl::aho_corasick<2> const bit_ac(bit_words, bit_value);
	REQUIRE(bit_ac.match_count(bit_ac.step(bit_ac.step(0, true), false)) == 1);
	std::string_view const ascii("ab");
	atl::trie<256> bytes;
	bytes.insert(ascii);
	REQUIRE(bytes.count(ascii) == 1);
	std::array const high_byte{static_cast<unsigned char>(255)};
	bytes.insert(high_byte);
	REQUIRE(bytes.count(high_byte) == 1);
}

namespace {
	/** @brief A word with a member `begin` and an `end` found only by argument-dependent lookup. */
	struct adl_end_word {
		int symbols[1]{0};

		int *begin() { return symbols; }

		friend int *end(adl_end_word &word) { return word.symbols + 1; }
	};

	/** @brief A word collection with the same member-begin, ADL-end shape. */
	struct adl_end_words {
		std::vector<std::vector<int>> items;

		auto begin() { return items.begin(); }

		friend auto end(adl_end_words &words) { return words.items.end(); }
	};

	/** @brief A projected symbol that is default constructible but not assignable. */
	struct frozen_symbol {
		int const value = 0;
		auto operator<=>(frozen_symbol const &) const = default;
	};
}

TEST_CASE("strings GU: ranges whose end is found by argument-dependent lookup", "[strings][constraints]") {
	adl_end_word word;
	atl::trie<2> dynamic;
	dynamic.insert(word);
	REQUIRE(dynamic.count(word) == 1);
	REQUIRE(dynamic.contains(word));
	REQUIRE(dynamic.find(word));
	REQUIRE(dynamic.count_with_prefix(word) == 1);
	REQUIRE(dynamic.erase(word));
	REQUIRE_FALSE(dynamic.contains(word));

	std::vector<adl_end_word> inner(2);
	atl::static_trie<2> const frozen(inner);
	REQUIRE(frozen.count(word) == 2);
	atl::aho_corasick<2> const ac(inner);
	REQUIRE(ac.match_count(ac.step(ac.root(), 0)) == 2);
	atl::experimental::aho_corasick_by_dense_table<2> const dense(inner);
	REQUIRE(dense.match_count(dense.step(0, 0)) == 2);

	adl_end_words outer{{{1}, {0}, {0, 1}}};
	atl::static_trie<2> const outer_frozen(outer);
	REQUIRE(outer_frozen.count(std::array{0, 1}) == 1);
	atl::aho_corasick<2> const outer_ac(outer);
	REQUIRE(outer_ac.node_count() == 4);
	REQUIRE(atl::lexicographic_order(outer) == std::vector<atl::uint32>{1, 2, 0});
	REQUIRE(atl::experimental::lexicographic_order_by_msd_partition(outer) == std::vector<atl::uint32>{1, 2, 0});
	REQUIRE(atl::experimental::lexicographic_order_by_lcp_merge(outer) == std::vector<atl::uint32>{1, 2, 0});
	adl_end_words sorted{{{0}, {0, 1}, {1}}};
	atl::sorted_dictionary const dictionary(sorted);
	REQUIRE(dictionary.prefix_interval(std::array{0}) == atl::index_interval{0, 2});
	atl::experimental::sorted_dictionary_by_parts_build<int> const parts(sorted);
	REQUIRE(parts.prefix_interval(std::array{0}) == atl::index_interval{0, 2});
	atl::experimental::sorted_dictionary_by_inplace_build<int> const inplace(sorted);
	REQUIRE(inplace.prefix_interval(std::array{0}) == atl::index_interval{0, 2});

	atl::suffix_automaton<2> automaton;
	automaton.insert(word);
	REQUIRE(automaton.contains(word));
	atl::suffix_tree<2> tree;
	REQUIRE(tree.insert(word) == 0);
	atl::experimental::suffix_tree_by_sibling_list<2> list;
	REQUIRE(list.insert(word) == 0);
	atl::palindromic_tree<2> palindromes;
	REQUIRE(palindromes.length(palindromes.insert(word)) == 1);
	REQUIRE(atl::hash_of(word) == atl::hash_of(std::array{0}));
	REQUIRE(atl::string_hash(word).full_hash() == atl::hash_of(std::array{0}));
	atl::experimental::trie_by_global_hash<2> hashed;
	atl::experimental::trie_by_symbol_bst<2> bst;
	hashed.insert(word);
	bst.insert(word);
	REQUIRE(hashed.contains(word));
	REQUIRE(bst.count(word) == 1);
	REQUIRE(hashed.erase(word));
	REQUIRE(bst.erase(word));
}

TEST_CASE("strings GU: input-only words are erased in one pass", "[strings][constraints]") {
	std::vector<int> const symbols{1, 0, 1};
	atl::trie<2> dynamic;
	dynamic.insert(test::input_only(symbols));
	dynamic.insert(test::input_only(symbols));
	REQUIRE(dynamic.erase(test::input_only(symbols)));
	REQUIRE(dynamic.count(test::input_only(symbols)) == 1);
	REQUIRE(dynamic.count_with_prefix(std::array{1, 0}) == 1);
	REQUIRE(dynamic.erase(test::input_only(symbols)));
	REQUIRE_FALSE(dynamic.erase(test::input_only(symbols)));
	REQUIRE(dynamic.empty());
	std::vector<int> const outside{1, 2};
	REQUIRE_FALSE(dynamic.erase(test::input_only(outside)));
	REQUIRE(dynamic.size() == 0);
}

TEST_CASE("strings GU: projected symbols that are not assignable", "[strings][constraints]") {
	auto const project = [](int value) { return frozen_symbol{value}; };
	std::vector<std::vector<int>> const words{{2, 1}, {1}, {1, 2}};
	REQUIRE(atl::lexicographic_order(words, project) == std::vector<atl::uint32>{1, 2, 0});
	REQUIRE(atl::experimental::lexicographic_order_by_msd_partition(words, project) == std::vector<atl::uint32>{1, 2, 0});
	REQUIRE(atl::experimental::lexicographic_order_by_lcp_merge(words, project) == std::vector<atl::uint32>{1, 2, 0});
	std::vector<std::vector<int>> const sorted{{1}, {1, 2}, {2, 1}};
	atl::sorted_dictionary const dictionary(sorted, project);
	REQUIRE(dictionary.prefix_interval(std::array{1}, project) == atl::index_interval{0, 2});
	atl::experimental::sorted_dictionary_by_parts_build<frozen_symbol> const parts(sorted, project);
	REQUIRE(parts.prefix_interval(std::array{2}, project) == atl::index_interval{2, 3});
	atl::experimental::sorted_dictionary_by_inplace_build<frozen_symbol> const inplace(sorted, project);
	REQUIRE(inplace.prefix_interval(std::array{2}, project) == atl::index_interval{2, 3});
	std::vector<std::vector<frozen_symbol>> const frozen_words{{frozen_symbol{2}}, {frozen_symbol{1}}};
	REQUIRE(atl::lexicographic_order(frozen_words) == std::vector<atl::uint32>{1, 0});
	atl::sorted_dictionary const identity_dictionary(std::vector<std::vector<frozen_symbol>>{{frozen_symbol{1}}, {frozen_symbol{2}}});
	REQUIRE(identity_dictionary.prefix_interval(std::array{frozen_symbol{2}}) == atl::index_interval{1, 2});
}

TEST_CASE("strings GU: Boolean symbols in the LCP mergesort contender", "[strings][constraints]") {
	std::array<std::array<bool, 1>, 2> const words{{{false}, {true}}};
	REQUIRE(atl::experimental::lexicographic_order_by_lcp_merge(words) == std::vector<atl::uint32>{0, 1});
	std::array<std::array<bool, 1>, 2> const reversed{{{true}, {false}}};
	REQUIRE(atl::experimental::lexicographic_order_by_lcp_merge(reversed) == std::vector<atl::uint32>{1, 0});
	std::vector<std::vector<bool>> const longer{{true, false}, {true}, {false, true, true}, {true, false}};
	REQUIRE(atl::experimental::lexicographic_order_by_lcp_merge(longer) == atl::lexicographic_order(longer));
	REQUIRE(atl::experimental::lexicographic_order_by_msd_partition(longer) == atl::lexicographic_order(longer));
}
