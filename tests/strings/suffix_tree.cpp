#include <atl/experimental/strings/suffix_tree_variants.hpp>
#include <atl/strings/suffix_array.hpp>
#include <atl/strings/suffix_automaton.hpp>
#include <atl/strings/suffix_tree.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

using atl::suffix_tree;

namespace {
	auto const gs2_lower_projection = [](char c) { return c - 'a'; };

	template<std::size_t U>
		requires(U <= (atl::uint64(1) << 32))
	std::string label_of(suffix_tree<U> const &t, std::uint32_t v) {
		std::string out;
		while (v != t.root()) {
			auto const e = t.edge(v);
			auto const w = t.word(e.word);
			std::string part;
			for (std::uint32_t i = 0; i < e.length; ++i) part.push_back(char('a' + w[e.offset + i]));
			out = part + out;
			v = t.parent(v);
		}
		return out;
	}

	long long occurrences(std::vector<std::string> const &words, std::string const &p) {
		long long total = 0;
		for (auto const &w : words)
			for (std::size_t i = 0; i + p.size() <= w.size(); ++i)
				if (w.compare(i, p.size(), p) == 0) ++total;
		return total;
	}

	long long suffix_matches(std::vector<std::string> const &words, std::string const &p) {
		long long total = 0;
		for (auto const &w : words)
			if (p.size() <= w.size() && w.compare(w.size() - p.size(), p.size(), p) == 0) ++total;
		return total;
	}
}

TEST_CASE("suffix_tree: full structure matches brute force", "[strings]") {
	auto seed = GENERATE(take(40, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::vector<std::string> words(1 + std::size_t(rng() % 4));
	for (auto &w : words) {
		w.resize(1 + std::size_t(rng() % 12));
		for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
	}
	suffix_tree<26> t;
	for (auto const &w : words) t.insert(w, gs2_lower_projection);

	auto const occ = t.occurrence_counts();
	std::uint64_t distinct = 0;
	for (std::uint32_t v = 1; v < t.node_count(); ++v) {
		auto const p = label_of(t, v);
		REQUIRE(p.size() == t.length(v));
		distinct += t.edge(v).length;
		REQUIRE(std::int64_t(t.suffix_count(v)) == suffix_matches(words, p));
		REQUIRE(std::int64_t(occ[v]) == occurrences(words, p));
		if (t.suffix_count(v) > 0) {
			auto const [sw, so] = t.suffix(v);
			REQUIRE(words[sw].substr(so) == p);
		}
		if (auto const link = t.suffix_link(v)) REQUIRE(label_of(t, *link) == p.substr(1));
	}
	// distinct substrings: cross-structure oracle vs the suffix automaton
	atl::suffix_automaton<26> sam;
	for (auto const &w : words) sam.insert(w, gs2_lower_projection);
	REQUIRE(distinct == sam.distinct_substring_count());

	// every suffix of every word descends from the root and ends marked
	for (auto const &w : words)
		for (std::size_t start = 0; start < w.size(); ++start) {
			std::uint32_t v = t.root();
			std::size_t i = start;
			while (i < w.size()) {
				auto const child = t.walk(v, gs2_lower_projection(w[i]));
				REQUIRE(child.has_value());
				auto const e = t.edge(*child);
				auto const stored = t.word(e.word);
				std::size_t k = 0;
				for (; k < e.length && i < w.size(); ++k, ++i)
					REQUIRE(stored[e.offset + k] == std::uint32_t(gs2_lower_projection(w[i])));
				REQUIRE((i < w.size() ? k == e.length : true)); // full edges until the end
				v = *child;
			}
			REQUIRE(t.length(v) == w.size() - start);
			REQUIRE(t.suffix_count(v) > 0);
		}

	// nodes_by_length is topological for the parent tree
	auto const order = t.nodes_by_length();
	std::vector<std::size_t> position(order.size());
	for (std::size_t i = 0; i < order.size(); ++i) position[order[i]] = i;
	for (std::uint32_t v = 1; v < t.node_count(); ++v) REQUIRE(position[t.parent(v)] < position[v]);
}

TEST_CASE("suffix_tree: extend returns the online longest repeated suffix", "[strings]") {
	auto seed = GENERATE(take(25, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 3);
	std::vector<std::string> words(1 + std::size_t(rng() % 3));
	for (auto &w : words) {
		w.resize(1 + std::size_t(rng() % 14));
		for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
	}
	suffix_tree<26> t;
	std::vector<std::string> finished;
	for (auto const &w : words) {
		std::string prefix;
		for (char const c : w) {
			prefix.push_back(c);
			auto const got = t.extend(gs2_lower_projection(c));
			// brute: longest suffix of `prefix` occurring in a finished word,
			// or in `prefix` itself at an other start
			std::uint32_t best = 0;
			for (std::size_t len = 1; len <= prefix.size(); ++len) {
				auto const suf = prefix.substr(prefix.size() - len);
				bool found = prefix.find(suf) + len < prefix.size() + 1 && prefix.find(suf) != prefix.size() - len;
				for (auto const &f : finished) found = found || f.find(suf) != std::string::npos;
				if (found) best = std::uint32_t(len);
			}
			REQUIRE(got == best);
		}
		t.finish_word();
		finished.push_back(w);
	}
}

TEST_CASE("suffix_tree: duplicates, streaming parity, mid-word queries, edges", "[strings]") {
	SECTION("duplicate words add counts, not nodes") {
		suffix_tree<26> t;
		t.insert(std::string{"abab"}, gs2_lower_projection);
		auto const initial = t.node_count();
		t.insert(std::string{"abab"}, gs2_lower_projection);
		REQUIRE(t.node_count() == initial);
		for (std::uint32_t v = 1; v < t.node_count(); ++v)
			if (t.suffix_count(v) > 0) REQUIRE(t.suffix_count(v) % 2 == 0);
	}
	SECTION("streaming equals insert") {
		std::mt19937_64 rng(7);
		std::vector<std::string> words(3);
		for (auto &w : words) {
			w.resize(1 + std::size_t(rng() % 12));
			for (auto &c : w) c = char('a' + std::size_t(rng() % 2));
		}
		suffix_tree<26> a, b;
		for (auto const &w : words) {
			a.insert(w, gs2_lower_projection);
			for (char const c : w) b.extend(gs2_lower_projection(c));
			b.finish_word();
		}
		REQUIRE(a.node_count() == b.node_count());
		REQUIRE(a.occurrence_counts() == b.occurrence_counts());
	}
	SECTION("mid-word queries are legal") {
		suffix_tree<26> t;
		t.insert(std::string{"banana"}, gs2_lower_projection);
		t.extend(gs2_lower_projection('b'));
		t.extend(gs2_lower_projection('a'));
		REQUIRE(t.walk(t.root(), gs2_lower_projection('b')).has_value()); // implicit tree visible
		REQUIRE(t.size() == 8);
		t.finish_word();
		REQUIRE(t.word_count() == 2);
	}
	SECTION("empty word and single symbols") {
		suffix_tree<26> t;
		REQUIRE(t.insert(std::string{}, gs2_lower_projection) == 0);
		REQUIRE(t.node_count() == 1);
		t.insert(std::string{"z"}, gs2_lower_projection);
		REQUIRE(t.node_count() == 2);
		REQUIRE(t.suffix_count(1) == 1);
	}
	SECTION("large sparse alphabet") {
		constexpr std::size_t U = 1'000'000;
		suffix_tree<U> t;
		std::vector<std::uint32_t> const w{999'999, 5, 999'999, 5};
		t.insert(w);
		auto const v = t.walk(t.root(), 999'999u);
		REQUIRE(v.has_value());
		auto const occ = t.occurrence_counts();
		REQUIRE(occ[t.root()] == 4); // total suffixes
	}
}

TEST_CASE("suffix_tree: clear, for_each_child, reserve", "[strings]") {
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<std::string> words(1 + std::size_t(rng() % 3));
	for (auto &w : words) {
		w.resize(std::size_t(rng() % 15));
		for (auto &c : w) c = char('a' + std::size_t(rng() % 3));
	}
	suffix_tree<26> fresh;
	fresh.reserve(64);
	for (auto const &w : words) fresh.insert(w, gs2_lower_projection);

	// clear mid-word, then rebuild: the same tree as a fresh one
	suffix_tree<26> reused;
	reused.insert(std::string{"cabbage"}, gs2_lower_projection);
	reused.extend(gs2_lower_projection('b'));
	reused.clear();
	REQUIRE(reused.node_count() == 1);
	REQUIRE(reused.word_count() == 0);
	REQUIRE(reused.size() == 0);
	for (auto const &w : words) reused.insert(w, gs2_lower_projection);
	REQUIRE(reused.node_count() == fresh.node_count());
	REQUIRE(reused.word_count() == fresh.word_count());
	std::vector<atl::uint64> const occ = reused.occurrence_counts();
	REQUIRE(occ == fresh.occurrence_counts());
	for (std::uint32_t v = 0; v < fresh.node_count(); ++v) {
		REQUIRE(label_of(reused, v) == label_of(fresh, v));
		REQUIRE(reused.suffix_count(v) == fresh.suffix_count(v));
	}

	// children: exactly the nodes whose parent is v, each found by walk
	for (std::uint32_t v = 0; v < fresh.node_count(); ++v) {
		std::vector<std::uint32_t> seen;
		fresh.for_each_child(v, [&](std::uint32_t c) { seen.push_back(c); });
		std::ranges::sort(seen);
		std::vector<std::uint32_t> want;
		for (std::uint32_t u = 1; u < fresh.node_count(); ++u)
			if (fresh.parent(u) == v) want.push_back(u);
		REQUIRE(seen == want);
		for (std::uint32_t const c : seen) {
			auto const e = fresh.edge(c);
			REQUIRE(fresh.walk(v, fresh.word(e.word)[e.offset]) == std::optional<std::uint32_t>(c));
		}
	}
}

TEST_CASE("suffix_tree: the list-storage variant matches node by node", "[strings]") {
	// same Ukkonen steps, different child storage: identical ids, labels,
	// links and counts, at every point of a multi-word build
	auto seed = GENERATE(take(20, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 1 + std::size_t(rng() % 4);
	suffix_tree<26> library;
	atl::experimental::suffix_tree_by_sibling_list<26> list;
	for (std::size_t words = 1 + std::size_t(rng() % 4); words-- > 0;) {
		std::size_t const length = std::size_t(rng() % 40);
		for (std::size_t i = 0; i < length; ++i) {
			std::uint32_t const c = std::uint32_t(std::size_t(rng() % sigma));
			REQUIRE(list.extend(c) == library.extend(c));
		}
		REQUIRE(list.finish_word() == library.finish_word());
	}
	REQUIRE(list.node_count() == library.node_count());
	REQUIRE(list.occurrence_counts() == library.occurrence_counts());
	REQUIRE(list.nodes_by_length() == library.nodes_by_length());
	for (std::uint32_t v = 0; v < library.node_count(); ++v) {
		REQUIRE(list.length(v) == library.length(v));
		REQUIRE(list.suffix_count(v) == library.suffix_count(v));
		if (library.suffix_count(v) > 0) {
			REQUIRE(list.suffix(v).word == library.suffix(v).word);
			REQUIRE(list.suffix(v).offset == library.suffix(v).offset);
		}
		for (std::uint32_t c = 0; c < sigma; ++c) REQUIRE(list.walk(v, c) == library.walk(v, c));
		if (v == 0) continue;
		REQUIRE(list.parent(v) == library.parent(v));
		REQUIRE(list.suffix_link(v) == library.suffix_link(v));
		auto const le = list.edge(v);
		auto const se = library.edge(v);
		REQUIRE((le.word == se.word && le.offset == se.offset && le.length == se.length));
	}
}

TEST_CASE("suffix_tree: million-symbol unary path uses heap traversal storage", "[strings][deep]") {
	constexpr atl::uint32 length = 1'000'000;
	atl::suffix_tree<2> tree;
	tree.reserve(length);
	for (atl::uint32 i = 0; i < length; ++i) tree.extend(0);
	REQUIRE(tree.finish_word() == 0);
	REQUIRE(tree.node_count() == std::size_t(length) + 1);
	auto const counts = tree.occurrence_counts();
	auto const order = tree.nodes_by_length();
	REQUIRE(counts[tree.root()] == length);
	for (atl::uint32 depth = 1; depth <= length; ++depth) {
		auto const node = order[depth];
		REQUIRE(tree.length(node) == depth);
		REQUIRE(counts[node] == length - depth + 1);
		REQUIRE(tree.suffix_count(node) == 1);
	}
}

TEST_CASE("suffix_tree: million random symbols agree with suffix-array geometry", "[strings][deep]") {
	constexpr atl::uint32 length = 1'000'000;
	std::mt19937_64 rng(730);
	std::vector<atl::uint32> text(length);
	for (auto &symbol : text) symbol = atl::uint32(std::size_t(rng() % 26));
	atl::suffix_array const suffixes(text);
	atl::uint64 distinct = atl::uint64(length) * (atl::uint64(length) + 1) / 2;
	for (auto common : suffixes.lcp_lengths()) distinct -= common;
	atl::suffix_tree<26> tree;
	tree.reserve(length);
	tree.insert(text);
	atl::uint64 edge_symbols = 0;
	for (atl::uint32 node = 1; node < tree.node_count(); ++node) edge_symbols += tree.edge(node).length;
	REQUIRE(edge_symbols == distinct);
	REQUIRE(tree.occurrence_counts()[tree.root()] == length);
}
