#include <atl/config.hpp>
#include <atl/experimental/strings/aho_corasick_variants.hpp>
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
#include <atl/utils/prefetch.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <list>
#include <string_view>
#include <vector>
#include "../expect_abort.hpp"
#include "input_only.hpp"

namespace {
	template<typename Text>
	concept can_border = requires(Text &text) { atl::border_lengths(text); };
	template<typename Text>
	concept can_common = requires(Text &text) { atl::common_prefix_lengths(text); };
	template<typename Lengths>
	concept can_convert_border = requires(Lengths &lengths) { atl::to_border_lengths(lengths); };
	template<typename Lengths>
	concept can_convert_common = requires(Lengths &lengths) { atl::to_common_prefix_lengths(lengths); };
	template<typename Borders>
	concept can_match = requires(std::string_view text, Borders &borders) { atl::occurrences(text, text, borders); };
	template<typename Text>
	concept can_factor = requires(Text &text) { atl::lyndon_factorization(text); };
	template<typename Text>
	concept can_rotate = requires(Text &text) { atl::minimal_rotation_index(text); };
	template<typename Text>
	concept can_runs = requires(Text &text) { atl::runs(text); };
	template<typename Text>
	concept can_palindromes = requires(Text &text) { atl::palindrome_lengths(text); };
	template<typename Text>
	concept can_hash = requires(Text &text) { atl::string_hash(text, 131); atl::hash_of(text, 131); };
	template<typename Symbol>
	concept can_push_hash = requires(atl::string_hash &hash, Symbol symbol) { hash.push_back(symbol); };
	template<typename Symbol>
	concept can_extend = requires(atl::palindromic_tree<2> &tree, Symbol symbol) { tree.extend(symbol); tree.walk(0, symbol); };
	template<typename Text>
	concept can_insert = requires(atl::palindromic_tree<2> &tree, Text &text) { tree.insert(text); };
	template<typename Visit>
	concept can_visit = requires(atl::palindromic_tree<2> &tree, Visit visit) { tree.for_each_child(0, visit); };
	template<std::size_t Alphabet>
	concept can_tree = requires { typename atl::palindromic_tree<Alphabet>; };

	template<typename Element>
	concept can_input_only = requires { typename test::input_only<Element>; };
	static_assert(can_input_only<atl::uint8>);
	static_assert(!can_input_only<bool> && !can_input_only<int const>);
	static_assert(std::ranges::input_range<test::input_only<int>>);
	static_assert(!std::ranges::forward_range<test::input_only<int>>);

	static_assert(!can_border<std::list<char>> && !can_common<std::list<char>>);
	static_assert(!can_convert_border<std::vector<int>> && !can_convert_common<std::vector<int>>);
	static_assert(!can_match<std::vector<int>>);
	static_assert(!can_factor<std::list<char>> && !can_rotate<std::list<char>> && !can_runs<std::list<char>>);
	static_assert(!can_palindromes<std::list<char>>);
	static_assert(!can_hash<std::vector<double>> && !can_hash<std::vector<bool>>);
	static_assert(!can_push_hash<bool> && !can_push_hash<double>);
	static_assert(!can_extend<double> && !can_insert<std::vector<double>>);
	static_assert(!can_visit<decltype([](int) {})>);
	static_assert(sizeof(std::size_t) == 4 || !can_tree<(std::numeric_limits<std::size_t>::max)()>);
}

TEST_CASE("strings GS1: asserted bounds reject invalid calls", "[strings][constraints]") {
	atl::string_hash const hash(std::string_view("abc"), 131);
	REQUIRE(atl_test::aborts([&] { (void)hash.equal(2, 1, 0, 0); }));
	REQUIRE(atl_test::aborts([&] { (void)hash.equal(0, 0, 0, 4); }));
	REQUIRE(atl_test::aborts([&] { (void)hash.concat(atl::string_hash::mod, 0, 0); }));
	REQUIRE(atl_test::aborts([] { atl::string_hash value{131}; value.reserve((std::numeric_limits<std::size_t>::max)()); }));
	REQUIRE(atl_test::aborts([] { atl::palindromic_tree<2> tree; tree.extend(-1); }));
	REQUIRE(atl_test::aborts([] { atl::palindromic_tree<2> tree; tree.reserve((std::numeric_limits<std::size_t>::max)()); }));
	atl::palindromic_tree<2> tree;
	REQUIRE_FALSE(tree.walk(tree.odd_root(), -1));
	REQUIRE_FALSE(tree.walk(tree.odd_root(), atl::uint64(1) << 32));
}

namespace {
	template<typename Text>
	concept gs2_can_suffix_array = requires(Text &text) { atl::suffix_array(text); };
	template<typename Words>
	concept gs2_can_order = requires(Words &words) { atl::lexicographic_order(words); };
	template<typename Words>
	concept gs2_can_dictionary = requires(Words &words) { atl::sorted_dictionary(words); };
	template<typename Words>
	concept gs2_can_char_dictionary = requires(Words &words) { atl::sorted_dictionary<char>(words); };
	template<typename Query>
	concept gs2_can_query = requires(atl::sorted_dictionary<char> const &dictionary, Query &query) { dictionary.prefix_interval(query); };
	template<typename Text>
	concept gs2_can_insert = requires(atl::suffix_automaton<2> &automaton, atl::suffix_tree<2> &tree, Text &text) { automaton.insert(text); automaton.contains(text); tree.insert(text); };
	template<typename Symbol>
	concept gs2_can_extend = requires(atl::suffix_automaton<2> &automaton, atl::suffix_tree<2> &tree, Symbol symbol) { automaton.extend(symbol); automaton.walk(0, symbol); tree.extend(symbol); tree.walk(0, symbol); };
	template<typename Visit>
	concept gs2_can_visit_automaton = requires(atl::suffix_automaton<2> &automaton, Visit visit) { automaton.for_each_transition(0, visit); };
	template<typename Visit>
	concept gs2_can_visit_tree = requires(atl::suffix_tree<2> &tree, Visit visit) { tree.for_each_child(0, visit); };
	template<std::size_t Alphabet>
	concept gs2_can_alphabet = requires { typename atl::suffix_automaton<Alphabet>; typename atl::suffix_tree<Alphabet>; };
	static_assert(!gs2_can_suffix_array<std::list<char>>);
	static_assert(!gs2_can_order<std::vector<std::list<char>>> && !gs2_can_dictionary<std::vector<std::list<char>>>);
	static_assert(!gs2_can_char_dictionary<std::vector<std::vector<int>>>);
	static_assert(!gs2_can_query<std::list<char>> && !gs2_can_query<std::vector<std::string_view>>);
	static_assert(!gs2_can_insert<std::vector<double>> && !gs2_can_extend<double>);
	static_assert(!gs2_can_visit_automaton<decltype([](int) {})> && !gs2_can_visit_tree<decltype([]() {})>);
	static_assert(sizeof(std::size_t) == 4 || !gs2_can_alphabet<(std::numeric_limits<std::size_t>::max)()>);
	static_assert(!atl::sortable_text<std::string_view, decltype([]() {})>);
}

TEST_CASE("strings GS2: asserted suffix structure bounds", "[strings][constraints]") {
	REQUIRE(atl_test::aborts([] { atl::suffix_array const suffixes{std::string_view("a")}; (void)suffixes.position(1); }));
	REQUIRE(atl_test::aborts([] { atl::suffix_array const suffixes; (void)suffixes.rank(0); }));
	REQUIRE(atl_test::aborts([] { atl::suffix_automaton<2> automaton; automaton.extend(-1); }));
	REQUIRE(atl_test::aborts([] { atl::suffix_tree<2> tree; tree.extend(2); }));
	REQUIRE(atl_test::aborts([] { atl::suffix_automaton<2> automaton; automaton.reserve((std::numeric_limits<std::size_t>::max)()); }));
	REQUIRE(atl_test::aborts([] { atl::suffix_tree<2> tree; tree.reserve((std::numeric_limits<std::size_t>::max)()); }));
	atl::suffix_automaton<2> automaton;
	atl::suffix_tree<2> tree;
	REQUIRE_FALSE(automaton.walk(0, -1));
	REQUIRE_FALSE(tree.walk(0, atl::uint64(1) << 32));
}

namespace {
	template<typename Word, typename Proj = std::identity>
	concept gs3_can_word = requires(atl::trie<2> &t, Word &word, Proj proj) {
		t.insert(word, proj);
		t.count(word, proj);
		t.contains(word, proj);
		t.find(word, proj);
		t.count_with_prefix(word, proj);
	};
	template<typename Word>
	concept gs3_can_erase = requires(atl::trie<2> &t, Word &word) { t.erase(word); };
	template<typename Words, typename Proj = std::identity>
	concept gs3_can_build = requires(Words &words, Proj proj) { atl::static_trie<2>(words, proj); atl::aho_corasick<2>(words, proj); };
	template<typename Symbol>
	concept gs3_can_symbol = requires(atl::trie<2> &t, atl::static_trie<2> const &s, atl::aho_corasick<2> const &a, Symbol symbol) {
		t.walk(0, symbol);
		s.walk(0, symbol);
		a.walk(0, symbol);
		a.step(0, symbol);
	};
	template<typename Visit>
	concept gs3_can_visit = requires(atl::trie<2> const &t, Visit visit) { t.for_each_child(0, visit); };
	template<std::size_t Alphabet>
	concept gs3_can_alphabet = requires {
		typename atl::trie<Alphabet>;
		typename atl::static_trie<Alphabet>;
		typename atl::aho_corasick<Alphabet>;
		typename atl::experimental::trie_by_global_hash<Alphabet>;
		typename atl::experimental::trie_by_symbol_bst<Alphabet>;
	};
	static_assert(!gs3_can_word<std::vector<double>> && !gs3_can_word<int>);
	static_assert(!gs3_can_word<std::vector<int>, decltype([]() { return 0; })>);
	static_assert(!gs3_can_word<std::vector<int>, decltype([](int) {})>);
	static_assert(gs3_can_erase<test::input_only<int>>);
	static_assert(!gs3_can_build<std::vector<int>> && !gs3_can_build<std::vector<std::vector<double>>>);
	static_assert(!gs3_can_build<std::vector<std::vector<int>>, decltype([]() { return 0; })>);
	static_assert(!gs3_can_symbol<double> && !gs3_can_visit<decltype([](int) {})>);
	static_assert(gs3_can_word<std::list<char>> && gs3_can_symbol<signed char>);
	static_assert(sizeof(std::size_t) == 4 || !gs3_can_alphabet<(std::numeric_limits<std::size_t>::max)()>);
}

TEST_CASE("strings GS3: asserted trie, table and automaton bounds", "[strings][constraints]") {
	REQUIRE(atl_test::aborts([] { atl::trie<2> t; t.reserve((std::numeric_limits<std::size_t>::max)()); }));
	REQUIRE(atl_test::aborts([] { atl::trie<2> t; t.insert(std::array{-1}); }));
	REQUIRE(atl_test::aborts([] { atl::trie<2> t; (void)t.walk(1, 0); }));
	REQUIRE(atl_test::aborts([] { atl::static_trie<2> const t; (void)t.children(atl::uint32(-1)); }));
	REQUIRE(atl_test::aborts([] { atl::static_trie<2> const t(std::array<std::array<int, 1>, 1>{{{2}}}); }));
	REQUIRE(atl_test::aborts([] { atl::aho_corasick<2> const ac; (void)ac.step(1, 0); }));
	REQUIRE(atl_test::aborts([] { atl::aho_corasick<2> const ac; (void)ac.transition_table(0); }));
	REQUIRE(atl_test::aborts([] { atl::aho_corasick<2> const ac; (void)ac.transition_table(3); }));
	REQUIRE(atl_test::aborts([] { atl::aho_corasick<2> const ac(std::array<std::array<int, 1>, 1>{{{-1}}}); }));
	REQUIRE(atl_test::aborts([] { atl::dense_transitions const table({0}, 0); }));
	REQUIRE(atl_test::aborts([] { atl::dense_transitions const table({1}, 1); }));
	atl::trie<0> empty_alphabet;
	empty_alphabet.insert(std::array<int, 0>{});
	REQUIRE(empty_alphabet.size() == 1);
	atl::aho_corasick<0> const ac;
	REQUIRE(ac.step(ac.root(), 0) == ac.root());
}

namespace {
	template<std::size_t Alphabet>
	concept gu_can_full_table = requires(atl::aho_corasick<Alphabet> const &automaton) { automaton.transition_table(); };
	static_assert(gu_can_full_table<1> && gu_can_full_table<2> && !gu_can_full_table<0>);

	/** @brief Wide symbols whose low 64 bits are a valid symbol must stay out of the alphabet. */
	template<typename Wide, typename SignedWide>
	void check_wide_symbols() {
		Wide const high_zero = Wide(1) << 64;					 // low 64 bits: symbol 0
		Wide const high_one = (Wide(1) << 64) + 1;				 // low 64 bits: symbol 1
		SignedWide const negative_zero = -(SignedWide(1) << 64); // low 64 bits: symbol 0
		SignedWide const minus_one = -1;

		atl::trie<1> dynamic;
		dynamic.insert(std::array<int, 1>{0});
		REQUIRE(dynamic.contains(std::array<int, 1>{0}));
		REQUIRE_FALSE(dynamic.contains(std::array<Wide, 1>{high_zero}));
		REQUIRE_FALSE(dynamic.contains(std::array<SignedWide, 1>{negative_zero}));
		REQUIRE(dynamic.count(std::array<Wide, 1>{high_zero}) == 0);
		REQUIRE(dynamic.count_with_prefix(std::array<Wide, 1>{high_zero}) == 0);
		REQUIRE_FALSE(dynamic.find(std::array<Wide, 1>{high_zero}));
		REQUIRE_FALSE(dynamic.erase(std::array<Wide, 1>{high_zero}));
		REQUIRE_FALSE(dynamic.walk(dynamic.root(), high_zero));
		REQUIRE_FALSE(dynamic.walk(dynamic.root(), negative_zero));
		REQUIRE(dynamic.contains(std::array<int, 1>{0}));
		REQUIRE(dynamic.walk(dynamic.root(), Wide(0)));

		atl::static_trie<1> const frozen(std::array<std::array<int, 1>, 1>{{{0}}});
		REQUIRE(frozen.contains(std::array<int, 1>{0}));
		REQUIRE_FALSE(frozen.contains(std::array<Wide, 1>{high_zero}));
		REQUIRE_FALSE(frozen.find(std::array<SignedWide, 1>{negative_zero}));
		REQUIRE_FALSE(frozen.walk(frozen.root(), high_zero));
		REQUIRE_FALSE(frozen.walk(frozen.root(), minus_one));

		atl::aho_corasick<2> const automaton(std::array<std::array<int, 1>, 1>{{{1}}});
		auto const matched = automaton.step(automaton.root(), 1);
		REQUIRE(matched != automaton.root());
		REQUIRE(automaton.step(matched, high_one) == automaton.root());
		REQUIRE(automaton.step(automaton.root(), high_one) == automaton.root());
		REQUIRE(automaton.step(matched, Wide(1)) == automaton.step(matched, 1));
		REQUIRE_FALSE(automaton.walk(automaton.root(), high_one));
		REQUIRE(automaton.walk(automaton.root(), Wide(1)));

		atl::suffix_automaton<2> suffixes;
		suffixes.extend(1);
		REQUIRE_FALSE(suffixes.walk(suffixes.root(), high_one));
		REQUIRE(suffixes.walk(suffixes.root(), Wide(1)));
		REQUIRE_FALSE(suffixes.contains(std::array<Wide, 1>{high_one}));
		REQUIRE(suffixes.contains(std::array<Wide, 1>{Wide(1)}));

		atl::suffix_tree<2> tree;
		tree.extend(1);
		REQUIRE_FALSE(tree.walk(tree.root(), high_one));
		REQUIRE(tree.walk(tree.root(), Wide(1)));

		atl::palindromic_tree<2> palindromes;
		palindromes.extend(1);
		REQUIRE_FALSE(palindromes.walk(palindromes.odd_root(), high_one));
		REQUIRE(palindromes.walk(palindromes.odd_root(), Wide(1)));

		REQUIRE(atl_test::aborts([&] { atl::trie<1> t; t.insert(std::array<Wide, 1>{high_zero}); }));
		REQUIRE(atl_test::aborts([&] { atl::static_trie<1> const t(std::array<std::array<Wide, 1>, 1>{{{high_zero}}}); }));
		REQUIRE(atl_test::aborts([&] { atl::aho_corasick<2> const t(std::array<std::array<Wide, 1>, 1>{{{high_one}}}); }));
		REQUIRE(atl_test::aborts([&] { atl::suffix_automaton<2> t; t.extend(high_one); }));
		REQUIRE(atl_test::aborts([&] { atl::suffix_tree<2> t; t.extend(high_one); }));
		REQUIRE(atl_test::aborts([&] { atl::palindromic_tree<2> t; t.extend(high_one); }));
		REQUIRE(atl_test::aborts([&] { atl::trie<1> t; t.insert(std::array<SignedWide, 1>{negative_zero}); }));
	}

	/** @brief Runs the wide-symbol checks where the standard library treats 128-bit types as integral. */
	template<typename Probe = atl::uint128>
	void check_wide_symbols_if_integral() {
		if constexpr (std::integral<Probe> && std::integral<atl::int128>) check_wide_symbols<atl::uint128, atl::int128>();
	}
}

TEST_CASE("strings GU: wide integral symbols never alias a smaller symbol", "[strings][constraints]") {
#if ATL_HAS_INT128
	check_wide_symbols_if_integral();
#else
	SUCCEED("no native 128-bit integers");
#endif
}
