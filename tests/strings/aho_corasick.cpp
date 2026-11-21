#include <atl/experimental/strings/aho_corasick_variants.hpp>
#include <atl/strings/aho_corasick.hpp>
#include <atl/strings/dense_transitions.hpp>
#include <atl/strings/matching.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "input_only.hpp"

using atl::aho_corasick;

namespace {
	auto const gs3_aho_projection = [](char c) { return c - 'a'; };

	template<std::size_t U>
		requires(U <= (atl::uint64(1) << 32))
	std::string prefix_of(aho_corasick<U> const &ac, std::uint32_t v) {
		std::string s;
		for (; v != ac.root(); v = ac.parent(v)) s.push_back(char('a' + ac.symbol(v)));
		std::ranges::reverse(s);
		return s;
	}
}

TEST_CASE("aho_corasick: structure matches the prefix-set brute force", "[strings]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::vector<std::string> words(1 + std::size_t(rng() % 6));
	for (auto &w : words) {
		w.resize(std::size_t(rng() % 9)); // includes empty patterns
		for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
	}
	aho_corasick<26> const ac(words, gs3_aho_projection);

	std::set<std::string> prefixes;
	for (auto const &w : words)
		for (std::size_t j = 0; j <= w.size(); ++j) prefixes.insert(w.substr(0, j));
	REQUIRE(ac.node_count() == prefixes.size());
	REQUIRE(ac.size() == words.size());

	for (std::size_t i = 0; i < words.size(); ++i) REQUIRE(prefix_of(ac, ac.word_node(i)) == words[i]);

	auto const order = ac.nodes_by_depth();
	REQUIRE(order.size() == ac.node_count());
	REQUIRE(order[0] == ac.root());
	std::vector<std::size_t> position(order.size());
	for (std::size_t i = 0; i < order.size(); ++i) position[order[i]] = i;

	for (std::uint32_t v = 0; v < ac.node_count(); ++v) {
		auto const p = prefix_of(ac, v);
		REQUIRE(prefixes.count(p) == 1);

		auto const multiplicity = std::ranges::count(words, p);
		REQUIRE(ac.word_count(v) == std::size_t(multiplicity));
		std::size_t suffix_matches = 0;
		for (auto const &w : words)
			if (w.size() <= p.size() && p.compare(p.size() - w.size(), w.size(), w) == 0) ++suffix_matches;
		REQUIRE(ac.match_count(v) == suffix_matches);
		if (v == ac.root()) continue;
		// suffix link: the longest proper suffix that is a pattern prefix
		std::string best;
		for (std::size_t d = 1; d < p.size(); ++d)
			if (auto suf = p.substr(d); prefixes.count(suf)) {
				best = suf;
				break;
			}
		REQUIRE(prefix_of(ac, ac.suffix_link(v)) == best);
		// BFS order: parents first in both trees
		REQUIRE(position[ac.parent(v)] < position[v]);
		REQUIRE(position[ac.suffix_link(v)] < position[v]);
	}
}

TEST_CASE("aho_corasick: text scan finds every occurrence", "[strings]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::vector<std::string> words(1 + std::size_t(rng() % 6));
	for (auto &w : words) {
		w.resize(1 + std::size_t(rng() % 7));
		for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
	}
	std::string text(std::size_t(rng() % 60), 'a');
	for (auto &c : text) c = char('a' + std::size_t(rng() % sigma));
	aho_corasick<26> const ac(words, gs3_aho_projection);

	// per-position matches + per-pattern totals via the documented recipe
	std::vector<std::uint64_t> hits(ac.node_count(), 0);
	auto state = ac.root();
	for (std::size_t i = 0; i < text.size(); ++i) {
		state = ac.step(state, gs3_aho_projection(text[i]));
		++hits[state];
		std::size_t brute_here = 0;
		for (auto const &w : words)
			if (w.size() <= i + 1 && text.compare(i + 1 - w.size(), w.size(), w) == 0) ++brute_here;
		REQUIRE(ac.match_count(state) == brute_here);
	}
	auto const order = ac.nodes_by_depth();
	for (auto i = order.size(); i-- > 1;) hits[ac.suffix_link(order[i])] += hits[order[i]];
	for (std::size_t i = 0; i < words.size(); ++i) {
		auto const brute = atl::occurrences(text, words[i], atl::border_lengths(words[i])).size();
		REQUIRE(hits[ac.word_node(i)] == brute);
	}
}

TEST_CASE("aho_corasick: edges, sparse alphabet, out-of-alphabet symbols", "[strings]") {
	SECTION("single pattern behaves like KMP") {
		aho_corasick<26> const ac(std::vector<std::string>{"aba"}, gs3_aho_projection);
		auto v = ac.root();
		std::string const text = "ababa";
		std::vector<std::size_t> ends;
		for (std::size_t i = 0; i < text.size(); ++i) {
			v = ac.step(v, gs3_aho_projection(text[i]));
			if (ac.match_count(v)) ends.push_back(i);
		}
		REQUIRE(ends == std::vector<std::size_t>{2, 4}); // overlapping "aba"s
	}
	SECTION("out-of-alphabet text symbols reset the scan") {
		aho_corasick<26> const ac(std::vector<std::string>{"ab"}, gs3_aho_projection);
		auto v = ac.step(ac.root(), 0);
		REQUIRE(v != ac.root());
		REQUIRE(ac.step(v, 700) == ac.root());
		REQUIRE(ac.step(v, -3) == ac.root());
	}
	SECTION("duplicate and empty patterns count with multiplicity") {
		aho_corasick<26> const ac(std::vector<std::string>{"a", "a", ""}, gs3_aho_projection);
		REQUIRE(ac.size() == 3);
		REQUIRE(ac.word_count(ac.root()) == 1); // the empty pattern
		REQUIRE(ac.match_count(ac.root()) == 1);
		auto const v = ac.step(ac.root(), 0);
		REQUIRE(ac.word_count(v) == 2);
		REQUIRE(ac.match_count(v) == 3); // "a" twice + the empty pattern
	}
	SECTION("large sparse alphabet with projection") {
		constexpr std::size_t U = 1'000'000;
		std::vector<std::vector<std::uint32_t>> const words{{999'999, 5}, {5, 999'999}, {999'999}};
		aho_corasick<U> const ac(words);
		std::vector<std::uint32_t> const text{5, 999'999, 5, 999'999, 999'999};
		std::uint64_t total = 0;
		auto v = ac.root();
		for (auto const c : text) total += ac.match_count(v = ac.step(v, c));
		// occurrences: {5,999999}@1, {999999,5}@2, {999999}@1,3,4 -> 6
		REQUIRE(total == 6);
	}
}

TEST_CASE("aho_corasick: output links and words_at enumerate every match", "[strings]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::vector<std::string> words(1 + std::size_t(rng() % 6));
	for (auto &w : words) {
		w.resize(std::size_t(rng() % 7)); // includes empty patterns
		for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
	}
	std::string text(std::size_t(rng() % 60), 'a');
	for (auto &c : text) c = char('a' + std::size_t(rng() % sigma));
	aho_corasick<26> const ac(words, gs3_aho_projection);

	// node structure: walk, length, words_at, output_link against the prefixes
	for (std::uint32_t v = 0; v < ac.node_count(); ++v) {
		auto const p = prefix_of(ac, v);
		REQUIRE(ac.length(v) == p.size());
		for (char c = 'a'; c < char('a' + sigma); ++c) {
			auto const child = ac.walk(v, gs3_aho_projection(c));
			bool const is_prefix = std::ranges::any_of(words, [&](auto const &w) { return w.starts_with(p + c); });
			REQUIRE(child.has_value() == is_prefix);
			if (child) REQUIRE(ac.parent(*child) == v);
		}
		auto const at = ac.words_at(v);
		REQUIRE(at.size() == ac.word_count(v));
		REQUIRE(std::ranges::is_sorted(at));
		for (auto const i : at) REQUIRE(words[i] == p);
		std::optional<std::uint32_t> want;
		for (std::size_t d = 1; d <= p.size() && !want; ++d)
			for (std::uint32_t u = 0; u < ac.node_count(); ++u)
				if (ac.word_count(u) > 0 && prefix_of(ac, u) == p.substr(d)) want = u;
		REQUIRE(ac.output_link(v) == want);
	}

	// enumerated (end position, pattern) pairs equal the brute
	std::multiset<std::pair<std::size_t, std::uint32_t>> found, brute;
	auto state = ac.root();
	for (std::size_t i = 0; i < text.size(); ++i) {
		state = ac.step(state, gs3_aho_projection(text[i]));
		for (std::optional<std::uint32_t> u = ac.word_count(state) ? state : ac.output_link(state); u;
			 u = ac.output_link(*u))
			for (auto const id : ac.words_at(*u)) found.emplace(i, id);
		for (std::uint32_t id = 0; id < words.size(); ++id)
			if (words[id].size() <= i + 1 && text.compare(i + 1 - words[id].size(), words[id].size(), words[id]) == 0)
				brute.emplace(i, id);
	}
	REQUIRE(found == brute);
}

TEST_CASE("aho_corasick: output links skip long non-matching link chains", "[strings]") {
	// patterns {a, a^m c} on a^n: every position's suffix-link chain is ~m
	// long, its output chain one node
	constexpr std::size_t m = 50;
	std::vector<std::string> const words{"a", std::string(m, 'a') + 'c'};
	aho_corasick<26> const ac(words, gs3_aho_projection);
	auto state = ac.root();
	for (std::size_t i = 0; i < 2 * m; ++i) {
		state = ac.step(state, gs3_aho_projection('a'));
		REQUIRE(ac.length(state) == (std::min)(i + 1, m));
		std::size_t hops = 0, matches = 0;
		for (std::optional<std::uint32_t> u = ac.word_count(state) ? state : ac.output_link(state); u;
			 u = ac.output_link(*u), ++hops)
			matches += ac.words_at(*u).size();
		REQUIRE(hops == 1);
		REQUIRE(matches == 1);
	}
}

TEST_CASE("aho_corasick: deep failure chains and input-only pattern ranges", "[strings]") {
	// patterns a^1..a^k: from a^j a 'b' fails through the whole link chain
	// back to the root; every (state, symbol) step must equal the brute
	// longest-suffix-that-is-a-prefix answer
	constexpr std::size_t k = 40;
	std::vector<std::string> words;
	for (std::size_t j = 1; j <= k; ++j) words.emplace_back(j, 'a');
	words.push_back(std::string(k / 2, 'a') + "b");
	aho_corasick<26> const ac(words, gs3_aho_projection);
	std::set<std::string> prefixes;
	for (auto const &w : words)
		for (std::size_t l = 0; l <= w.size(); ++l) prefixes.insert(w.substr(0, l));
	for (std::uint32_t v = 0; v < ac.node_count(); ++v)
		for (char const c : {'a', 'b', 'c'}) {
			std::string const t = prefix_of(ac, v) + c;
			std::string expect;
			for (std::size_t l = 0; l <= t.size(); ++l)
				if (prefixes.count(t.substr(l))) {
					expect = t.substr(l);
					break;
				}
			REQUIRE(prefix_of(ac, ac.step(v, gs3_aho_projection(c))) == expect);
		}

	// an input-only range of patterns builds the same automaton
	aho_corasick<26> const streamed{test::input_only(words), gs3_aho_projection};
	REQUIRE(streamed.node_count() == ac.node_count());
	REQUIRE(streamed.size() == ac.size());
	for (std::uint32_t v = 0; v < ac.node_count(); ++v) {
		REQUIRE(prefix_of(streamed, v) == prefix_of(ac, v));
		REQUIRE(streamed.match_count(v) == ac.match_count(v));
	}
}

TEST_CASE("aho_corasick: default construction and empty", "[strings]") {
	aho_corasick<26> const none;
	REQUIRE(none.empty());
	REQUIRE(none.size() == 0);
	REQUIRE(none.node_count() == 1);
	REQUIRE(none.step(none.root(), 3) == none.root());
	REQUIRE(none.match_count(none.root()) == 0);
	REQUIRE(none.words_at(none.root()).empty());
	REQUIRE_FALSE(none.output_link(none.root()));
	REQUIRE_FALSE(none.walk(none.root(), 0));
	REQUIRE(none.nodes_by_depth().size() == 1);

	aho_corasick<26> const some(std::vector<std::string>{"ab"}, gs3_aho_projection);
	REQUIRE_FALSE(some.empty());
	REQUIRE_FALSE(some.walk(some.root(), 700)); // outside the alphabet
}

namespace {
	/** @brief Random pattern sets over [0, sigma): plain random, a^k runs, overlapping shifts, empty patterns. */
	std::vector<std::vector<std::uint32_t>> random_patterns(std::mt19937_64 &rng, std::uint32_t sigma) {
		std::vector<std::vector<std::uint32_t>> words;
		std::size_t const count = 1 + std::size_t(rng() % 8);
		for (std::size_t i = 0; i < count; ++i) {
			std::vector<std::uint32_t> w(std::size_t(rng() % 12));
			switch (std::size_t(rng() % 4)) {
			case 0:
				break; // a^k (all zeros)
			case 1:
				for (auto &c : w) c = std::uint32_t(std::size_t(rng() % sigma));
				break;
			case 2: // a periodic word: many overlapping self-occurrences
				for (std::size_t j = 0; j < w.size(); ++j) w[j] = std::uint32_t(j % (1 + std::size_t(rng() % 3)) % sigma);
				break;
			default: // a shift of the adjacent pattern: overlaps across patterns
				if (!words.empty() && !words.back().empty())
					w.assign(words.back().begin() + 1, words.back().end()), w.push_back(std::uint32_t(std::size_t(rng() % sigma)));
			}
			words.push_back(std::move(w));
		}
		return words;
	}
}

TEST_CASE("aho_corasick: transition_table equals step on every (state, symbol)", "[strings]") {
	auto seed = GENERATE(take(200, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	constexpr std::size_t U = 6;
	std::uint32_t const sigma = 1 + std::uint32_t(std::size_t(rng() % U));
	auto const words = random_patterns(rng, sigma);
	aho_corasick<U> const ac(words);

	auto const full = ac.transition_table();
	REQUIRE(full.state_count() == ac.node_count());
	REQUIRE(full.alphabet_size() == U);
	for (std::uint32_t v = 0; v < ac.node_count(); ++v) {
		REQUIRE(full.row(v).size() == U);
		for (std::uint32_t c = 0; c < U; ++c) {
			REQUIRE(full(v, c) == ac.step(v, c));
			REQUIRE(full.row(v)[c] == full(v, c));
		}
	}
	// a narrower table is the same function on its symbols, even when
	// patterns use symbols past it
	std::uint32_t const narrow = 1 + std::uint32_t(std::size_t(rng() % U));
	auto const part = ac.transition_table(narrow);
	REQUIRE(part.state_count() == ac.node_count());
	REQUIRE(part.alphabet_size() == narrow);
	for (std::uint32_t v = 0; v < ac.node_count(); ++v)
		for (std::uint32_t c = 0; c < narrow; ++c) REQUIRE(part(v, c) == ac.step(v, c));

	// the dense engine: same node ids, same structure, same transitions
	atl::experimental::aho_corasick_by_dense_table<U> const dense(words);
	auto const dense_part = dense.transition_table(narrow);
	REQUIRE(dense_part.state_count() == ac.node_count());
	REQUIRE(dense_part.alphabet_size() == narrow);
	for (std::uint32_t v = 0; v < ac.node_count(); ++v)
		for (std::uint32_t c = 0; c < narrow; ++c) REQUIRE(dense_part(v, c) == part(v, c));
	REQUIRE(dense.node_count() == ac.node_count());
	REQUIRE(dense.size() == ac.size());
	auto const dense_table = dense.transition_table();
	for (std::uint32_t v = 0; v < ac.node_count(); ++v) {
		REQUIRE(dense.length(v) == ac.length(v));
		REQUIRE(dense.word_count(v) == ac.word_count(v));
		REQUIRE(dense.match_count(v) == ac.match_count(v));
		REQUIRE(dense.output_link(v) == ac.output_link(v));
		REQUIRE(std::ranges::equal(dense.words_at(v), ac.words_at(v)));
		if (v != ac.root()) {
			REQUIRE(dense.suffix_link(v) == ac.suffix_link(v));
			REQUIRE(dense.parent(v) == ac.parent(v));
			REQUIRE(dense.symbol(v) == ac.symbol(v));
		}
		for (std::uint32_t c = 0; c < U; ++c) {
			REQUIRE(dense.step(v, c) == ac.step(v, c));
			REQUIRE(dense.walk(v, c) == ac.walk(v, c));
			REQUIRE(dense_table(v, c) == full(v, c));
		}
		REQUIRE(dense.step(v, U) == ac.root());
		REQUIRE_FALSE(dense.walk(v, U));
	}
	for (std::size_t i = 0; i < words.size(); ++i) REQUIRE(dense.word_node(i) == ac.word_node(i));
	auto const order = dense.nodes_by_depth();
	REQUIRE(order.size() == ac.node_count());
	std::vector<std::size_t> position(order.size());
	for (std::size_t i = 0; i < order.size(); ++i) position[order[i]] = i;
	for (std::uint32_t v = 1; v < ac.node_count(); ++v) {
		REQUIRE(position[dense.parent(v)] < position[v]);
		REQUIRE(position[dense.suffix_link(v)] < position[v]);
	}
}

TEST_CASE("aho_corasick: transition_table on deep chains and degenerate automata", "[strings]") {
	SECTION("a^k: every miss falls through the whole chain") {
		constexpr std::size_t k = 300;
		aho_corasick<26> const ac(std::vector<std::vector<std::uint32_t>>{std::vector<std::uint32_t>(k, 0), {0, 1}});
		auto const t = ac.transition_table();
		for (std::uint32_t v = 0; v < ac.node_count(); ++v)
			for (std::uint32_t c = 0; c < 26; ++c) REQUIRE(t(v, c) == ac.step(v, c));
	}
	SECTION("a table past the prefetching threshold (2^18 cells)") {
		std::mt19937_64 rng(544);
		std::vector<std::vector<std::uint32_t>> words(2000, std::vector<std::uint32_t>(10));
		for (auto &w : words)
			for (auto &c : w) c = std::uint32_t(std::size_t(rng() % 26));
		aho_corasick<26> const ac(words);
		auto const t = ac.transition_table();
		REQUIRE(t.state_count() * t.alphabet_size() >= (std::size_t(1) << 18));
		for (std::uint32_t v = 0; v < ac.node_count(); ++v)
			for (std::uint32_t c = 0; c < 26; ++c) REQUIRE(t(v, c) == ac.step(v, c));
	}
	SECTION("no patterns, and only the empty pattern") {
		for (auto const &ac : {aho_corasick<4>(), aho_corasick<4>(std::vector<std::string>{""}, gs3_aho_projection)}) {
			auto const t = ac.transition_table();
			REQUIRE(t.state_count() == 1);
			REQUIRE(std::ranges::equal(t.row(ac.root()), std::vector<std::uint32_t>(4, 0)));
		}
		atl::experimental::aho_corasick_by_dense_table<4> const none;
		REQUIRE(none.node_count() == 1);
		REQUIRE(none.step(none.root(), 3) == none.root());
		REQUIRE(none.nodes_by_depth().size() == 1);
		REQUIRE(none.words_at(none.root()).empty());
	}
	SECTION("a default dense_transitions is empty") {
		atl::dense_transitions const t;
		REQUIRE(t.state_count() == 0);
		REQUIRE(t.alphabet_size() == 1);
	}
	SECTION("a partial table keeps the no-transition sentinel") {
		atl::dense_transitions const t({1, atl::dense_transitions::none, 0, 1}, 2);
		REQUIRE(t.state_count() == 2);
		REQUIRE(t(0, 0) == 1);
		REQUIRE(t(0, 1) == atl::dense_transitions::none);
		REQUIRE(t.row(1)[0] == 0);
	}
}

namespace {
	/** @brief Every (state, symbol) cell of @p t equals the library_table table @p library_table. */
	void require_same_table(atl::dense_transitions const &t, atl::dense_transitions const &library_table) {
		REQUIRE(t.state_count() == library_table.state_count());
		REQUIRE(t.alphabet_size() == library_table.alphabet_size());
		for (std::uint32_t v = 0; v < library_table.state_count(); ++v) {
			REQUIRE(t.row(v).size() == library_table.alphabet_size());
			for (std::uint32_t c = 0; c < library_table.alphabet_size(); ++c) REQUIRE(t(v, c) == library_table(v, c));
		}
	}

	/** @brief Every transition_table_by_row_options instantiation at @p sigma and prefetch threshold @p min_cells equals the library_table table. */
	template<std::size_t U>
		requires(U <= (atl::uint64(1) << 32))
	void require_tuned_tables(aho_corasick<U> const &ac, std::size_t const sigma, std::size_t const min_cells) {
		using atl::experimental::row_prefetch;
		using atl::experimental::transition_table_by_row_options;
		auto const library_table = ac.transition_table(sigma);
		require_same_table(transition_table_by_row_options<false, row_prefetch::none>(ac, sigma, min_cells), library_table);
		require_same_table(transition_table_by_row_options<false, row_prefetch::first_line>(ac, sigma, min_cells), library_table);
		require_same_table(transition_table_by_row_options<false, row_prefetch::all_lines>(ac, sigma, min_cells), library_table);
		require_same_table(transition_table_by_row_options<false, row_prefetch::all_lines_write>(ac, sigma, min_cells), library_table);
		require_same_table(transition_table_by_row_options<true, row_prefetch::none>(ac, sigma, min_cells), library_table);
		require_same_table(transition_table_by_row_options<true, row_prefetch::first_line>(ac, sigma, min_cells), library_table);
		require_same_table(transition_table_by_row_options<true, row_prefetch::all_lines>(ac, sigma, min_cells), library_table);
		require_same_table(transition_table_by_row_options<true, row_prefetch::all_lines_write>(ac, sigma, min_cells), library_table);
	}
}

// Transition-table build contenders: the uninitialised-storage, threshold and
// row-prefetch variants build exactly the library_table table.
TEST_CASE("aho_corasick: transition_table_by_row_options variants equal the library_table table", "[strings]") {
	auto seed = GENERATE(take(100, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	constexpr std::size_t U = 40; // rows of 40 cells span several cache lines
	std::uint32_t const sigma = 1 + std::uint32_t(std::size_t(rng() % U));
	auto const words = random_patterns(rng, sigma);
	aho_corasick<U> const ac(words);
	std::size_t const narrow = 1 + std::size_t(rng() % U);
	for (std::size_t const min_cells : {std::size_t(0), std::size_t(1) << 18, (std::numeric_limits<std::size_t>::max)()}) {
		require_tuned_tables(ac, U, min_cells);
		require_tuned_tables(ac, narrow, min_cells);
		require_tuned_tables(ac, 1, min_cells);
	}
	require_tuned_tables(aho_corasick<U>(), U, 0); // the single-state automaton
	require_tuned_tables(aho_corasick<U>(std::vector<std::vector<std::uint32_t>>{{}}), 3, 0);
	std::vector<std::uint32_t> const chain(50, 0);
	require_tuned_tables(aho_corasick<U>(std::vector<std::vector<std::uint32_t>>{chain}), U, 0); // deep failure chain
	// rows past max_prefetched_lines (256 cells) prefetch only their head
	constexpr std::size_t wide = 300;
	std::vector<std::vector<std::uint32_t>> wide_words(20, std::vector<std::uint32_t>(1 + std::size_t(rng() % 6)));
	for (auto &w : wide_words)
		for (auto &c : w) c = std::uint32_t(std::size_t(rng() % wide));
	aho_corasick<wide> const wide_ac(wide_words);
	for (std::size_t const s : {std::size_t(255), std::size_t(256), std::size_t(257), wide}) require_tuned_tables(wide_ac, s, 0);
}

TEST_CASE("dense_transitions: overwrite snapshots retain value semantics", "[strings]") {
	atl::dense_transitions copied;
	{
		aho_corasick<3> const ac(std::vector<std::vector<std::uint32_t>>{{}, {0, 1}, {1}, {2, 0}});
		auto original = ac.transition_table();
		copied = original;
		auto copy(original);
		require_same_table(copy, original);
		auto moved(std::move(original));
		CHECK(original.state_count() == 0);
		require_same_table(copy, moved);
		original = std::move(moved);
		CHECK(moved.state_count() == 0);
		require_same_table(copy, original);
		original = atl::dense_transitions({0}, 1);
		CHECK(copy.state_count() > original.state_count());
	}
	CHECK(copied(0, 0) != 0);
	auto moved(std::move(copied));
	copied = moved;
	require_same_table(copied, moved);
}

TEST_CASE("aho_corasick: overwrite fill around the prefetch threshold", "[strings]") {
	for (std::size_t const states : {65535u, 65536u, 65537u}) {
		aho_corasick<4> const ac(std::vector<std::vector<std::uint32_t>>{std::vector<std::uint32_t>(states - 1, 0)});
		auto const table = ac.transition_table();
		CHECK(table.state_count() == states);
		for (std::uint32_t v = 0; v < states; ++v) {
			CHECK(table(v, 0) == (std::min)(v + 1, std::uint32_t(states - 1)));
			for (std::uint32_t c = 1; c < 4; ++c) CHECK(table(v, c) == 0);
		}
	}
}

TEST_CASE("aho_corasick: million-symbol paths and failures use loops", "[strings]") {
	std::vector<std::vector<unsigned char>> const words(1, std::vector<unsigned char>(1'000'000, 0));
	aho_corasick<2> const ac(words);
	REQUIRE(ac.length(ac.word_node(0)) == words[0].size());
	REQUIRE(ac.step(ac.word_node(0), 1) == ac.root());
	auto const table = ac.transition_table();
	REQUIRE(table(ac.word_node(0), 0) == ac.word_node(0));
	REQUIRE(table(ac.word_node(0), 1) == ac.root());
}
