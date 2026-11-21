#include <atl/strings/palindromic_tree.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <string>
#include <utility>
#include <vector>

using atl::palindromic_tree;

namespace {
	bool is_pal(std::string const &s) { return std::ranges::equal(s, std::views::reverse(s)); }

	// brute-force palindromic-substring statistics over a word collection
	struct oracle {
		std::vector<std::string> words;
		std::set<std::string> pals; // distinct nonempty palindromic substrings

		void insert(std::string const &w) {
			words.push_back(w);
			for (std::size_t i = 0; i < w.size(); ++i)
				for (std::size_t len = 1; i + len <= w.size(); ++len)
					if (auto sub = w.substr(i, len); is_pal(sub)) pals.insert(std::move(sub));
		}

		long long occurrences(std::string const &p) const {
			long long total = 0;
			for (auto const &w : words)
				for (std::size_t i = 0; i + p.size() <= w.size(); ++i)
					if (w.compare(i, p.size(), p) == 0) ++total;
			return total;
		}

		static std::string longest_pal_suffix(std::string const &prefix) {
			for (std::size_t len = prefix.size(); len > 0; --len)
				if (auto suf = prefix.substr(prefix.size() - len); is_pal(suf)) return suf;
			return {};
		}
	};

	auto const palindromic_test_symbol = [](char c) { return c - 'a'; };

	// locate the node of palindrome p by descending center-outward
	template<std::size_t U>
		requires(U <= (atl::uint64(1) << 32))
	std::optional<typename atl::palindromic_tree<U>::node_id> locate(palindromic_tree<U> const &t, std::string const &p) {
		std::optional cur{p.size() % 2 ? t.odd_root() : t.even_root()};
		for (std::size_t i = p.size() / 2; cur && i < p.size(); ++i) cur = t.walk(*cur, palindromic_test_symbol(p[i]));
		return cur;
	}
}

TEST_CASE("palindromic_tree: palindrome statistics match a brute oracle", "[strings]") {
	auto seed = GENERATE(take(40, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	palindromic_tree<26> t;
	if (std::size_t(rng() % 2) != 0) t.reserve(std::size_t(rng() % 64)); // reserving must not change the structure
	oracle ref;

	std::size_t const words = 1 + std::size_t(rng() % 4);
	for (std::size_t i = 0; i < words; ++i) {
		std::string w(std::size_t(rng() % 15), 'a'); // includes the empty word
		for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
		// streaming build; extend must return the longest palindromic suffix
		// of every prefix
		for (std::size_t j = 0; j < w.size(); ++j) {
			auto const v = t.extend(palindromic_test_symbol(w[j]));
			REQUIRE(v == t.last());
			auto const expect = oracle::longest_pal_suffix(w.substr(0, j + 1));
			REQUIRE(t.length(v) == atl::int32(expect.size()));
			REQUIRE(locate(t, expect) == std::optional{v});
		}
		t.finish_word();
		REQUIRE(t.last() == t.even_root());
		ref.insert(w);
	}

	REQUIRE(t.palindrome_count() == ref.pals.size());
	REQUIRE(t.node_count() == ref.pals.size() + 2);

	auto const occ = t.occurrence_counts();
	for (auto const &p : ref.pals) {
		CAPTURE(p);
		auto const v = locate(t, p);
		REQUIRE(v.has_value());
		REQUIRE(*v >= 2);
		REQUIRE(t.length(*v) == atl::int32(p.size()));
		REQUIRE(t.symbol(*v) == std::uint32_t(palindromic_test_symbol(p.front())));
		REQUIRE(occ[*v] == std::uint64_t(ref.occurrences(p)));

		if (p.size() >= 3) REQUIRE(locate(t, p.substr(1, p.size() - 2)) == std::optional{t.parent(*v)});
		if (p.size() == 2) REQUIRE(t.parent(*v) == t.even_root());
		if (p.size() == 1) REQUIRE(t.parent(*v) == t.odd_root());

		auto const proper = oracle::longest_pal_suffix(p.substr(1));
		if (proper.empty())
			REQUIRE(t.suffix_link(*v) == t.even_root());
		else
			REQUIRE(locate(t, proper) == std::optional{t.suffix_link(*v)});

		std::size_t suffixes = 0;
		for (std::size_t d = 1; d <= p.size(); ++d)
			if (is_pal(p.substr(p.size() - d))) ++suffixes;
		REQUIRE(t.palindromic_suffix_count(*v) == suffixes);

		REQUIRE(t.parent(*v) < *v);
		REQUIRE(t.suffix_link(*v) < *v);
	}

	// non-palindromes and absent palindromes are not locatable
	for (int probe = 0; probe < 100; ++probe) {
		std::string p(1 + std::size_t(rng() % 8), 'a');
		for (auto &c : p) c = char('a' + std::size_t(rng() % sigma));
		if (!is_pal(p)) continue;
		REQUIRE(locate(t, p).has_value() == (ref.pals.count(p) != 0));
	}
}

TEST_CASE("palindromic_tree: whole-word insert equals streaming and shares nodes", "[strings]") {
	auto seed = GENERATE(take(15, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<std::string> words(1 + std::size_t(rng() % 4));
	for (auto &w : words) {
		w.resize(std::size_t(rng() % 18));
		for (auto &c : w) c = char('a' + std::size_t(rng() % 2));
	}

	palindromic_tree<26> inserted;
	palindromic_tree<26> streamed;
	for (auto const &w : words) {
		auto const a = inserted.insert(w, palindromic_test_symbol);
		typename palindromic_tree<26>::node_id b = streamed.even_root();
		for (char const c : w) b = streamed.extend(palindromic_test_symbol(c));
		auto const ended = streamed.finish_word();
		if (!w.empty()) REQUIRE(ended == b);
		REQUIRE(a == ended); // identical build order -> identical ids
	}
	REQUIRE(inserted.node_count() == streamed.node_count());
	REQUIRE(inserted.palindrome_count() == streamed.palindrome_count());
	REQUIRE(inserted.occurrence_counts() == streamed.occurrence_counts());
}

TEST_CASE("palindromic_tree: known small cases", "[strings]") {
	SECTION("empty tree") {
		palindromic_tree<26> t;
		REQUIRE(t.node_count() == 2);
		REQUIRE(t.palindrome_count() == 0);
		REQUIRE(t.length(t.odd_root()) == -1);
		REQUIRE(t.length(t.even_root()) == 0);
		REQUIRE(t.suffix_link(t.even_root()) == t.odd_root());
	}
	SECTION("aaaa: chain with occurrence counts 4..1") {
		palindromic_tree<26> t;
		t.insert(std::string{"aaaa"}, palindromic_test_symbol);
		REQUIRE(t.palindrome_count() == 4);
		auto const occ = t.occurrence_counts();
		std::string p;
		for (int len = 1; len <= 4; ++len) {
			p.push_back('a');
			auto const v = locate(t, p);
			REQUIRE(v.has_value());
			REQUIRE(occ[*v] == std::uint64_t(5 - len));
			REQUIRE(t.palindromic_suffix_count(*v) == std::uint32_t(len));
		}
	}
	SECTION("abba: the even-root textbook case") {
		palindromic_tree<26> t;
		t.insert(std::string{"abba"}, palindromic_test_symbol);
		REQUIRE(t.palindrome_count() == 4);
		auto const v = locate(t, std::string{"abba"});
		REQUIRE(v.has_value());
		REQUIRE(t.parent(*v) == *locate(t, std::string{"bb"}));
		REQUIRE(t.suffix_link(*v) == *locate(t, std::string{"a"}));
	}
	SECTION("generalized: shared palindromes across words") {
		palindromic_tree<26> t;
		t.insert(std::string{"aba"}, palindromic_test_symbol);
		auto const initial_nodes = t.node_count();
		t.insert(std::string{"aba"}, palindromic_test_symbol); // exact duplicate: no new nodes
		REQUIRE(t.node_count() == initial_nodes);
		auto const occ = t.occurrence_counts();
		REQUIRE(occ[*locate(t, std::string{"aba"})] == 2);
		REQUIRE(occ[*locate(t, std::string{"a"})] == 4);
		t.insert(std::string{"bab"}, palindromic_test_symbol); // shares a and b, adds bab
		REQUIRE(t.node_count() == initial_nodes + 1);
	}
	SECTION("palindromes never cross a word boundary") {
		palindromic_tree<26> t;
		t.insert(std::string{"ab"}, palindromic_test_symbol);
		t.insert(std::string{"ba"}, palindromic_test_symbol);
		// "ab" + "ba" would contain abba / bb as one string; as two words not
		REQUIRE_FALSE(locate(t, std::string{"bb"}).has_value());
		REQUIRE(t.palindrome_count() == 2);
	}
	SECTION("clear resets to the empty tree") {
		palindromic_tree<26> t;
		t.insert(std::string{"racecar"}, palindromic_test_symbol);
		t.clear();
		REQUIRE(t.node_count() == 2);
		t.insert(std::string{"aa"}, palindromic_test_symbol);
		REQUIRE(t.palindrome_count() == 2);
	}
}

TEST_CASE("palindromic_tree: large sparse alphabet with projection", "[strings]") {
	auto seed = GENERATE(take(5, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	constexpr std::size_t U = 1'000'000;
	palindromic_tree<U> t;

	std::vector<std::uint32_t> symbols(3);
	for (auto &s : symbols) s = std::uint32_t(std::size_t(rng() % U));
	std::vector<std::vector<std::uint32_t>> words(2);
	std::set<std::vector<std::uint32_t>> pals;
	for (auto &w : words) {
		w.resize(5 + std::size_t(rng() % 30));
		for (auto &c : w) c = symbols[std::size_t(rng() % symbols.size())];
		t.insert(w);
		for (std::size_t i = 0; i < w.size(); ++i)
			for (std::size_t len = 1; i + len <= w.size(); ++len) {
				std::vector<std::uint32_t> sub(w.begin() + i, w.begin() + i + len);
				if (std::ranges::equal(sub, std::views::reverse(sub))) pals.insert(std::move(sub));
			}
	}
	REQUIRE(t.palindrome_count() == pals.size());
	// walk rejects out-of-alphabet symbols instead of asserting
	REQUIRE_FALSE(t.walk(t.odd_root(), std::uint64_t(U)).has_value());
}

TEST_CASE("palindromic_tree: series links and child enumeration match brute force", "[strings]") {
	auto seed = GENERATE(take(40, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	palindromic_tree<26> t;
	oracle ref;
	for (std::size_t w = 1 + std::size_t(rng() % 3); w-- > 0;) {
		std::string s(std::size_t(rng() % 30), 'a');
		for (auto &c : s) c = char('a' + std::size_t(rng() % 2)); // sigma 2: long arithmetic series
		t.insert(s, palindromic_test_symbol);
		ref.insert(s);
	}
	auto const proper = [](std::string const &p) { return oracle::longest_pal_suffix(p.substr(1)); };
	auto const diff = [&](std::string const &p) { return p.size() - proper(p).size(); };
	auto const series = t.series_links();
	REQUIRE(series.size() == t.node_count());
	REQUIRE(series[t.odd_root()] == t.odd_root());
	REQUIRE(series[t.even_root()] == t.even_root());
	for (auto const &p : ref.pals) {
		auto const v = locate(t, p);
		REQUIRE(v);
		std::string u = proper(p);
		while (!u.empty() && diff(u) == diff(p)) u = proper(u);
		REQUIRE(series[*v] == (u.empty() ? t.even_root() : *locate(t, u)));
	}

	std::vector<std::string> spelled(t.node_count());
	for (auto const &p : ref.pals) spelled[*locate(t, p)] = p;
	for (std::uint32_t v = 0; v < t.node_count(); ++v) {
		std::set<std::pair<std::uint32_t, std::uint32_t>> seen;
		t.for_each_child(v, [&](std::uint32_t sym, std::uint32_t child) { seen.emplace(sym, child); });
		std::set<std::pair<std::uint32_t, std::uint32_t>> want;
		for (char c = 'a'; c <= 'b'; ++c) {
			std::string const wrapped = v == t.odd_root() ? std::string(1, c) : c + spelled[v] + c;
			if (ref.pals.contains(wrapped)) want.emplace(std::uint32_t(c - 'a'), *locate(t, wrapped));
		}
		REQUIRE(seen == want);
	}
}

TEST_CASE("palindromic_tree: million-symbol unary word uses bounded stack", "[strings][deep]") {
	constexpr std::size_t length = 1'000'000;
	palindromic_tree<1> tree;
	tree.reserve(length);
	for (std::size_t i = 0; i < length; ++i) tree.extend(0);
	REQUIRE(tree.palindrome_count() == length);
	REQUIRE(tree.length(tree.last()) == atl::int32(length));
	auto const counts = tree.occurrence_counts();
	auto const series = tree.series_links();
	REQUIRE(counts[2] == length);
	REQUIRE(series[tree.last()] == tree.even_root());
	std::size_t children = 0;
	tree.for_each_child(tree.odd_root(), [&](atl::uint32, atl::uint32) { ++children; });
	REQUIRE(children == 1);
	tree.finish_word();
	REQUIRE(tree.last() == tree.even_root());
}

TEST_CASE("palindromic_tree: repeated words accumulate independent seeds", "[strings]") {
	palindromic_tree<1> tree;
	std::vector<unsigned char> const word(100, 0);
	for (int repeat = 0; repeat < 1000; ++repeat) tree.insert(word);
	auto const counts = tree.occurrence_counts();
	for (atl::uint32 v = 2; v < tree.node_count(); ++v)
		REQUIRE(counts[v] == atl::uint64(1000) * (101 - atl::uint64(tree.length(v))));
}
