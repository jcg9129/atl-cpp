#include <atl/strings/static_trie.hpp>
#include <atl/strings/trie.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using atl::static_trie;
using atl::trie;
using atl::uint32;
using atl::uint64;

TEST_CASE("static_trie: agrees with the dynamic trie on every query", "[strings]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	constexpr std::size_t U = 7;
	auto rand_word = [&] {
		std::vector<int> w(std::size_t(rng() % 6)); // includes the empty word
		for (auto &c : w) c = int(std::size_t(rng() % U));
		return w;
	};

	std::vector<std::vector<int>> words(std::size_t(rng() % 60));
	for (auto &w : words) w = rand_word();

	static_trie<U> frozen(words);
	trie<U> dyn;
	for (auto const &w : words) dyn.insert(w);

	REQUIRE(frozen.size() == dyn.size());
	REQUIRE(frozen.node_count() == dyn.node_count()); // same structure, different layout

	for (int probe = 0; probe < 400; ++probe) {
		auto const p = rand_word();
		REQUIRE(frozen.count(p) == dyn.count(p));
		REQUIRE(frozen.contains(p) == dyn.contains(p));
		REQUIRE(frozen.count_with_prefix(p) == dyn.count_with_prefix(p));
	}

	for (int probe = 0; probe < 100; ++probe) {
		auto sn = frozen.root();
		auto dn = dyn.root();
		for (int step = 0; step < 6; ++step) {
			int const s = int(std::size_t(rng() % U));
			auto const sc = frozen.walk(sn, s);
			auto const dc = dyn.walk(dn, s);
			REQUIRE(sc.has_value() == dc.has_value());
			if (!sc) break;
			sn = *sc;
			dn = *dc;
			REQUIRE(frozen.word_count(sn) == dyn.word_count(dn));
			REQUIRE(frozen.prefix_count(sn) == dyn.prefix_count(dn));
		}
	}
}

TEST_CASE("static_trie: shared prefixes and duplicates build single-child chains", "[strings]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	constexpr std::size_t U = 3;
	auto rand_symbols = [&](std::size_t len) {
		std::vector<int> w(len);
		for (auto &c : w) c = int(std::size_t(rng() % U));
		return w;
	};
	std::vector<std::vector<int>> stems(1 + std::size_t(rng() % 3));
	for (auto &s : stems) s = rand_symbols(std::size_t(rng() % 12));
	std::vector<std::vector<int>> words(1 + std::size_t(rng() % 40));
	for (auto &w : words) {
		w = stems[std::size_t(rng() % stems.size())];
		w.resize(w.size() - std::size_t(rng() % (w.size() / 4 + 1))); // some words end inside a stem
		auto const tail = rand_symbols(std::size_t(rng() % 4));
		w.insert(w.end(), tail.begin(), tail.end());
	}
	for (std::size_t i = std::size_t(rng() % 5); i-- > 0;) words.push_back(words[std::size_t(rng() % words.size())]);

	static_trie<U> const frozen(words);
	trie<U> dyn;
	for (auto const &w : words) dyn.insert(w);
	REQUIRE(frozen.node_count() == dyn.node_count());
	for (auto const &w : words)
		for (std::size_t len = 0; len <= w.size(); ++len) {
			std::vector<int> const p(w.begin(), w.begin() + std::ptrdiff_t(len));
			REQUIRE(frozen.count(p) == dyn.count(p));
			REQUIRE(frozen.count_with_prefix(p) == dyn.count_with_prefix(p));
		}

	auto const order = frozen.word_order();
	for (std::size_t i = 1; i < order.size(); ++i) REQUIRE(!(words[order[i]] < words[order[i - 1]]));
	for (uint32 v = 0; v < frozen.node_count(); ++v)
		for (auto const &e : frozen.children(v)) REQUIRE(e.target > v);
}

TEST_CASE("static_trie: lexicographic order — sorted children, preorder ids", "[strings]") {
	auto seed = GENERATE(take(10, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<std::string> words(2 + std::size_t(rng() % 40));
	for (auto &w : words) {
		w.resize(std::size_t(rng() % 8));
		for (auto &c : w) c = char('a' + std::size_t(rng() % 4));
	}
	static_trie<26> t(words, [](char c) { return c - 'a'; });

	for (uint32 n = 0; n < t.node_count(); ++n) {
		auto const ch = t.children(n);
		for (std::size_t i = 1; i < ch.size(); ++i) REQUIRE(ch[i - 1].symbol < ch[i].symbol);
	}

	// enumerating by sorted children yields the words in lexicographic order,
	// with multiplicity — and visits node ids in increasing (preorder) order
	std::vector<std::string> enumerated;
	uint32 expected_id = 0;
	std::vector<std::pair<uint32, std::string>> pending{{t.root(), {}}};
	while (!pending.empty()) {
		auto [n, prefix] = std::move(pending.back());
		pending.pop_back();
		REQUIRE(n == expected_id++);
		for (std::size_t k = 0; k < t.word_count(n); ++k) enumerated.push_back(prefix);
		auto const children = t.children(n);
		for (auto it = children.rbegin(); it != children.rend(); ++it)
			pending.emplace_back(it->target, prefix + char('a' + it->symbol));
	}
	auto sorted = words;
	std::ranges::sort(sorted);
	REQUIRE(enumerated == sorted);
	REQUIRE(expected_id == t.node_count()); // every node visited exactly once
}

TEST_CASE("static_trie: word_interval matches prefix ranges over the sorted words", "[strings]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::vector<std::string> words(1 + std::size_t(rng() % 30));
	for (auto &w : words) {
		w.resize(std::size_t(rng() % 8)); // includes empty words
		for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
	}
	auto const lower = [](char c) { return c - 'a'; };
	atl::static_trie<26> const t(words, lower);

	std::vector<std::string> sorted = words;
	std::ranges::sort(sorted);
	auto const brute_interval = [&](std::string const &p) {
		std::uint32_t lo = 0;
		while (lo < sorted.size() && !sorted[lo].starts_with(p) && sorted[lo] < p) ++lo;
		std::uint32_t hi = lo;
		while (hi < sorted.size() && sorted[hi].starts_with(p)) ++hi;
		return atl::index_interval{lo, hi};
	};

	REQUIRE(t.word_interval(t.root()) == atl::index_interval{0, std::uint32_t(words.size())});
	for (int probe = 0; probe < 300; ++probe) {
		std::string p(std::size_t(rng() % 6), 'a');
		for (auto &c : p) c = char('a' + std::size_t(rng() % sigma));
		std::optional cur{t.root()};
		for (char const c : p)
			if (cur && !(cur = t.walk(*cur, lower(c)))) break;
		auto const [blo, bhi] = brute_interval(p);
		if (cur) {
			REQUIRE(t.word_interval(*cur) == atl::index_interval{blo, bhi});
			REQUIRE(std::size_t(bhi - blo) == t.prefix_count(*cur));
		} else {
			REQUIRE(blo == bhi);
		}
	}
}

TEST_CASE("static_trie: edge cases and a large sparse alphabet", "[strings]") {
	static_trie<10> empty;
	REQUIRE(empty.empty());
	REQUIRE(empty.node_count() == 1);
	REQUIRE(empty.count_with_prefix(std::vector<int>{}) == 0);
	REQUIRE_FALSE(empty.walk(empty.root(), 3));

	std::vector<std::vector<int>> words{{}, {}, {1, 2}, {1, 2}, {1, 2}, {1}};
	static_trie<10> t(words);
	REQUIRE(t.size() == 6);
	REQUIRE(t.count(std::vector<int>{}) == 2);
	REQUIRE(t.count(std::vector<int>{1, 2}) == 3);
	REQUIRE(t.count(std::vector<int>{1}) == 1);
	REQUIRE(t.count_with_prefix(std::vector<int>{1}) == 4);
	REQUIRE(t.count_with_prefix(std::vector<int>{}) == 6);
	REQUIRE(t.count(std::vector<int>{2}) == 0);

	// sparse symbols over U = 10^6: no per-node alphabet cost anywhere
	std::mt19937_64 rng(5);
	constexpr std::size_t U = 1'000'000;
	std::vector<std::vector<uint32>> sparse(2000);
	for (auto &w : sparse) {
		w.resize(1 + std::size_t(rng() % 6));
		for (auto &c : w) c = uint32(std::size_t(rng() % U));
	}
	static_trie<U> big(sparse);
	trie<U> dyn;
	for (auto const &w : sparse) dyn.insert(w);
	REQUIRE(big.node_count() == dyn.node_count());
	for (auto const &w : sparse) {
		REQUIRE(big.count(w) == dyn.count(w));
		REQUIRE(big.count_with_prefix(w) == dyn.count_with_prefix(w));
	}
	// out-of-alphabet symbols are simply absent on queries
	REQUIRE(big.count(std::vector<uint64>{uint64(U) + 5}) == 0);
}

TEST_CASE("static_trie: find, word_order and subtree_end", "[strings]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::vector<std::string> words(std::size_t(rng() % 30));
	for (auto &w : words) {
		w.resize(std::size_t(rng() % 8)); // includes empty words
		for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
	}
	auto const lower = [](char c) { return c - 'a'; };
	static_trie<26> const t(words, lower);

	// word_order is a permutation, and the words read through it are sorted
	auto const order = t.word_order();
	REQUIRE(order.size() == words.size());
	std::vector<uint32> seen(order.begin(), order.end());
	std::ranges::sort(seen);
	for (uint32 i = 0; i < seen.size(); ++i) REQUIRE(seen[i] == i);
	for (std::size_t i = 1; i < order.size(); ++i) REQUIRE(words[order[i - 1]] <= words[order[i]]);

	// every node: find spells it back, its interval maps to exactly the input
	// words with its prefix, and [n, subtree_end(n)) is its DFS-reachable set
	std::vector<std::string> spelled(t.node_count());
	std::vector<uint32> stack{t.root()};
	while (!stack.empty()) {
		uint32 const n = stack.back();
		stack.pop_back();
		for (auto const &e : t.children(n)) {
			spelled[e.target] = spelled[n] + char('a' + e.symbol);
			stack.push_back(e.target);
		}
	}
	for (uint32 n = 0; n < t.node_count(); ++n) {
		REQUIRE(t.find(spelled[n], lower) == std::optional<uint32>(n));

		auto const [begin, end] = t.word_interval(n);
		std::vector<uint32> via_order(order.begin() + begin, order.begin() + end);
		std::ranges::sort(via_order);
		std::vector<uint32> want;
		for (uint32 i = 0; i < words.size(); ++i)
			if (words[i].starts_with(spelled[n])) want.push_back(i);
		REQUIRE(via_order == want);

		std::vector<uint32> reach;
		std::vector<uint32> dfs{n};
		while (!dfs.empty()) {
			uint32 const v = dfs.back();
			dfs.pop_back();
			reach.push_back(v);
			for (auto const &e : t.children(v)) dfs.push_back(e.target);
		}
		std::ranges::sort(reach);
		REQUIRE(reach.size() == t.subtree_end(n) - n);
		for (std::size_t k = 0; k < reach.size(); ++k) REQUIRE(reach[k] == n + k);
	}
	REQUIRE_FALSE(t.find(std::string(9, 'a'), lower)); // longer than any word
	REQUIRE_FALSE(t.find(std::string("d"), lower));	   // outside the used symbols

	static_trie<26> const empty;
	REQUIRE(empty.word_order().empty());
	REQUIRE(empty.subtree_end(empty.root()) == 1);
	REQUIRE(empty.find(std::string{}, lower) == std::optional<uint32>(0));
}

TEST_CASE("static_trie: million-symbol paths use iterative construction", "[strings]") {
	std::vector<std::vector<unsigned char>> const input(2, std::vector<unsigned char>(1'000'000, 0));
	atl::static_trie<1> const words(input);
	REQUIRE(words.node_count() == input[0].size() + 1);
	REQUIRE(words.count(input[0]) == 2);
	REQUIRE(words.subtree_end(words.root()) == words.node_count());
}
