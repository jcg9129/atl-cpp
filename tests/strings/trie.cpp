#include <atl/experimental/strings/trie_variants.hpp>
#include <atl/strings/trie.hpp>
#include <atl/utils/hash.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>
#include "../expect_abort.hpp"

using atl::trie;

namespace {

	struct oracle {
		std::map<std::vector<int>, long long> m;

		void insert(std::vector<int> const &w) { ++m[w]; }

		bool erase(std::vector<int> const &w) {
			auto it = m.find(w);
			if (it == m.end() || it->second == 0) return false;
			--it->second;
			return true;
		}

		long long count(std::vector<int> const &w) const {
			auto it = m.find(w);
			return it == m.end() ? 0 : it->second;
		}

		long long with_prefix(std::vector<int> const &p) const {
			long long s = 0;
			for (auto const &[k, v] : m)
				if (k.size() >= p.size() && std::ranges::equal(p, std::views::take(k, std::ptrdiff_t(p.size())))) s += v;
			return s;
		}
	};
}

TEST_CASE("trie: matches a multiset oracle under mixed insert/erase/query", "[strings]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	constexpr std::size_t U = 8;
	trie<U> t;
	oracle ref;

	auto rand_word = [&] {
		std::vector<int> w(std::size_t(rng() % 5)); // includes the empty word
		for (auto &c : w) c = int(std::size_t(rng() % U));
		return w;
	};

	for (int op = 0; op < 4000; ++op) {
		auto w = rand_word();
		if (std::size_t(rng() % 3) == 0) {
			REQUIRE(t.erase(w) == ref.erase(w));
		} else {
			t.insert(w);
			ref.insert(w);
		}
		auto p = rand_word();
		REQUIRE(t.count(w) == std::size_t(ref.count(w)));
		REQUIRE(t.contains(p) == (ref.count(p) > 0));
		REQUIRE(t.count_with_prefix(p) == std::size_t(ref.with_prefix(p)));
	}
	REQUIRE(t.size() == std::size_t(ref.with_prefix({})));
}

TEST_CASE("trie: adversarial sorted insertion stays cheap (no BST degeneration)", "[strings]") {
	// distinct increasing first symbols would chain a raw-symbol BST; the hashed
	// ordering must keep this fast. Correctness is the observable proxy here.
	constexpr std::size_t U = 200'000;
	trie<U> t;
	for (int i = 0; i < 100'000; ++i) t.insert(std::vector<int>{i, 0, 0});
	REQUIRE(t.size() == 100'000);
	REQUIRE(t.count_with_prefix(std::vector<int>{}) == 100'000);
	REQUIRE(t.count(std::vector<int>{50'000, 0, 0}) == 1);
	REQUIRE(t.contains(std::vector<int>{99'999, 0, 0}));
	REQUIRE_FALSE(t.contains(std::vector<int>{100'000, 0, 0}));
}

TEST_CASE("trie: projections, node-handle walk, edge cases", "[strings]") {
	trie<128> words;
	for (auto const *w : {"tea", "ten", "ten", "to", "in"})
		words.insert(std::string(w), [](char ch) { return int(ch); });

	REQUIRE(words.count(std::string("ten"), [](char ch) { return int(ch); }) == 2);
	REQUIRE(words.count_with_prefix(std::string("te"), [](char ch) { return int(ch); }) == 3); // tea, ten, ten
	REQUIRE(words.size() == 5);

	auto n = words.root();
	auto a = words.walk(n, int('t'));
	REQUIRE(a);
	auto b = words.walk(*a, int('e'));
	REQUIRE(b);
	REQUIRE(words.prefix_count(*b) == 3);
	REQUIRE(words.word_count(*b) == 0);			   // "te" is not itself a stored word
	REQUIRE_FALSE(words.walk(*b, int('z')));	   // no child
	REQUIRE_FALSE(words.walk(words.root(), 9999)); // symbol outside the alphabet

	trie<4> e;
	REQUIRE(e.empty());
	e.insert(std::vector<int>{});
	REQUIRE(e.count(std::vector<int>{}) == 1);
	REQUIRE(e.count_with_prefix(std::vector<int>{}) == 1);
	REQUIRE(e.word_count(e.root()) == 1);
	REQUIRE(e.erase(std::vector<int>{}));
	REQUIRE(e.count(std::vector<int>{}) == 0);
	REQUIRE_FALSE(e.erase(std::vector<int>{}));

	words.clear();
	REQUIRE(words.size() == 0);
	REQUIRE(words.count_with_prefix(std::string("te"), [](char ch) { return int(ch); }) == 0);
}

TEST_CASE("trie: insert returns the node, find and for_each_child agree with walk", "[strings]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	constexpr std::size_t U = 6;
	trie<U> t;
	oracle ref;
	std::map<std::vector<int>, std::uint32_t> node_of; // every prefix of an inserted word
	for (int i = 0; i < 300; ++i) {
		std::vector<int> w(std::size_t(rng() % 6)); // includes the empty word
		for (auto &c : w) c = int(std::size_t(rng() % U));
		auto const n = t.insert(w);
		ref.insert(w);

		std::optional cur{t.root()};
		std::vector<int> prefix;
		node_of[prefix] = *cur;
		for (int const c : w) {
			cur = t.walk(*cur, c);
			REQUIRE(cur);
			prefix.push_back(c);
			node_of[prefix] = *cur;
		}
		REQUIRE(n == *cur);
		REQUIRE(t.word_count(n) == std::size_t(ref.count(w)));
	}
	for (auto const &[w, n] : node_of) {
		REQUIRE(t.find(w) == std::optional<std::uint32_t>(n));
		REQUIRE(t.word_count(*t.find(w)) == t.count(w));

		std::map<int, std::uint32_t> seen;
		t.for_each_child(n, [&](std::uint32_t sym, std::uint32_t child) {
			REQUIRE(seen.emplace(int(sym), child).second);
		});
		std::map<int, std::uint32_t> want;
		for (int c = 0; c < int(U); ++c) {
			auto longer = w;
			longer.push_back(c);
			if (auto const it = node_of.find(longer); it != node_of.end()) want.emplace(c, it->second);
		}
		REQUIRE(seen == want);
	}
	REQUIRE_FALSE(t.find(std::vector<int>(7, 0)));	 // longer than any word
	REQUIRE_FALSE(t.find(std::vector<int>{int(U)})); // outside the alphabet
}

TEST_CASE("trie: pinning the hash seed makes structure reproducible", "[strings]") {
	atl::set_hash_seed(12345);
	trie<1000> a;
	atl::set_hash_seed(12345);
	trie<1000> b;
	std::mt19937 rng(1);
	for (int i = 0; i < 2000; ++i) {
		std::vector<int> w(1 + std::size_t(rng() % 4));
		for (auto &c : w) c = int(std::size_t(rng() % 1000));
		a.insert(w);
		b.insert(w);
	}
	REQUIRE(a.node_count() == b.node_count()); // same seed -> identical shape
}

TEST_CASE("trie: symbols outside the alphabet are absent, not aliased", "[strings]") {
	trie<26> t;
	t.insert(std::vector<int>{5});
	t.insert(std::vector<int>{5, 7});
	std::int64_t const wrap = std::int64_t(1) << 32;
	for (std::int64_t const alias : {wrap + 5, -wrap + 5, std::int64_t(26), std::int64_t(-1)}) {
		std::vector<std::int64_t> const word{alias};
		REQUIRE(t.count(word) == 0);
		REQUIRE_FALSE(t.contains(word));
		REQUIRE(t.count_with_prefix(word) == 0);
		REQUIRE_FALSE(t.erase(word));
		REQUIRE(t.count(std::vector<std::int64_t>{5, alias}) == 0);
	}
	REQUIRE(t.count(std::vector<int>{5}) == 1); // nothing was erased through an alias
	REQUIRE(t.count(std::vector<std::int64_t>{5, 7}) == 1);
	REQUIRE(t.size() == 2);
}

TEMPLATE_TEST_CASE("experimental tries: node handles are asserted, symbols never alias", "[trie]",
				   atl::experimental::trie_by_global_hash<16>, atl::experimental::trie_by_symbol_bst<16>) {
	TestType t;
	t.insert(std::vector<int>{1, 2});
	auto const bad = typename TestType::node_id(t.node_count());
	CHECK(atl_test::aborts([&] { (void)t.walk(bad, 1); }));
	CHECK(atl_test::aborts([&] { (void)t.word_count(bad); }));
	CHECK(atl_test::aborts([&] { (void)t.prefix_count(bad); }));
	// 2^32 + 1 narrows to 1, which is on the stored path
	REQUIRE(t.count_with_prefix(std::vector<std::uint64_t>{(std::uint64_t(1) << 32) + 1}) == 0);
	REQUIRE(t.count(std::vector<std::uint64_t>{(std::uint64_t(1) << 32) + 1, 2}) == 0);
	REQUIRE(t.count_with_prefix(std::vector<std::uint64_t>{1}) == 1);
}

TEMPLATE_TEST_CASE("experimental tries: multiset oracle under mixed operations", "[strings]",
				   atl::experimental::trie_by_global_hash<8>, atl::experimental::trie_by_symbol_bst<8>) {
	TestType words;
	oracle reference;
	std::mt19937 rng(71);
	for (std::size_t i = 0; i < 4000; ++i) {
		std::vector<int> word(std::size_t(rng() % 7));
		for (auto &symbol : word) symbol = int(std::size_t(rng() % 8));
		if (std::size_t(rng() % 3) == 0) {
			REQUIRE(words.erase(word) == reference.erase(word));
		} else {
			words.insert(word);
			reference.insert(word);
		}
		REQUIRE(words.count(word) == std::size_t(reference.count(word)));
		REQUIRE(words.count_with_prefix(word) == std::size_t(reference.with_prefix(word)));
		REQUIRE(words.size() == std::size_t(reference.with_prefix({})));
	}
}

TEST_CASE("trie: million-symbol paths use heap scratch", "[strings]") {
	std::vector<unsigned char> const word(1'000'000, 0);
	trie<1> words;
	words.reserve(word.size());
	auto const terminal = words.insert(word);
	REQUIRE(words.node_count() == word.size() + 1);
	REQUIRE(words.erase(word));
	REQUIRE(words.find(word) == terminal);
	REQUIRE(words.word_count(terminal) == 0);
}
