#include <atl/strings/suffix_automaton.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

using atl::suffix_automaton;

namespace {

	struct oracle {
		std::vector<std::string> words;
		std::set<std::string> subs; // distinct nonempty substrings

		void insert(std::string const &w) {
			words.push_back(w);
			for (std::size_t i = 0; i < w.size(); ++i)
				for (std::size_t len = 1; i + len <= w.size(); ++len) subs.insert(w.substr(i, len));
		}

		bool contains(std::string const &s) const { return s.empty() || subs.count(s) != 0; }

		// occurrences counted per word instance, overlaps included
		long long occurrences(std::string const &s) const {
			long long total = 0;
			for (auto const &w : words)
				for (std::size_t i = 0; i + s.size() <= w.size(); ++i)
					if (w.compare(i, s.size(), s) == 0) ++total;
			return total;
		}

		bool is_suffix(std::string const &s) const {
			return std::ranges::any_of(words, [&](std::string const &w) {
				return w.size() >= s.size() && w.compare(w.size() - s.size(), s.size(), s) == 0;
			});
		}
	};

	auto const gs2_lower_projection = [](char c) { return c - 'a'; };

	template<std::size_t U>
		requires(U <= (atl::uint64(1) << 32))
	std::optional<typename atl::suffix_automaton<U>::state_id>
	walk_string(suffix_automaton<U> const &sa, std::string const &s) {
		std::optional cur{sa.root()};
		for (char const c : s)
			if (cur && !(cur = sa.walk(*cur, gs2_lower_projection(c)))) return std::nullopt;
		return cur;
	}
}

TEST_CASE("suffix_automaton: substring statistics match a brute oracle", "[strings]") {
	auto seed = GENERATE(take(40, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const sigma = 2 + std::size_t(rng() % 4);
	suffix_automaton<26> sa;
	oracle ref;

	std::size_t const words = 1 + std::size_t(rng() % 5);
	for (std::size_t i = 0; i < words; ++i) {
		std::string w(std::size_t(rng() % 13), 'a'); // includes the empty word
		for (auto &c : w) c = char('a' + std::size_t(rng() % sigma));
		auto const end = sa.insert(w, gs2_lower_projection);
		auto const walked = walk_string(sa, w);
		REQUIRE(walked.has_value());
		REQUIRE(*walked == end);
		ref.insert(w);
	}

	REQUIRE(sa.distinct_substring_count() == ref.subs.size());

	auto const occ = sa.occurrence_counts();
	auto const terminal = sa.terminals();
	for (auto const &s : ref.subs) {
		CAPTURE(s);
		REQUIRE(sa.contains(s, gs2_lower_projection));
		auto const v = walk_string(sa, s);
		REQUIRE(v.has_value());
		REQUIRE(occ[*v] == std::uint64_t(ref.occurrences(s)));
		REQUIRE(terminal[*v] == ref.is_suffix(s));
		REQUIRE(sa.shortest_length(*v) <= s.size());
		REQUIRE(s.size() <= sa.length(*v));
		REQUIRE(sa.shortest_length(*v) == sa.length(sa.suffix_link(*v)) + 1);
	}

	for (int probe = 0; probe < 200; ++probe) {
		std::string s(1 + std::size_t(rng() % 14), 'a');
		for (auto &c : s) c = char('a' + std::size_t(rng() % sigma));
		REQUIRE(sa.contains(s, gs2_lower_projection) == ref.contains(s));
		REQUIRE(walk_string(sa, s).has_value() == ref.contains(s));
	}

	// states_by_length is topological for the link tree: parents first
	auto const order = sa.states_by_length();
	REQUIRE(order.size() == sa.state_count());
	REQUIRE(order[0] == sa.root());
	std::vector<std::size_t> position(order.size());
	for (std::size_t i = 0; i < order.size(); ++i) position[order[i]] = i;
	for (std::size_t i = 1; i < order.size(); ++i) {
		REQUIRE(sa.length(order[i - 1]) <= sa.length(order[i]));
		REQUIRE(position[sa.suffix_link(order[i])] < position[order[i]]);
	}
	// transitions all lead to longer classes; count matches the pool
	std::size_t transitions = 0;
	for (std::size_t v = 0; v < sa.state_count(); ++v)
		sa.for_each_transition(std::uint32_t(v), [&](std::uint32_t, std::uint32_t target) {
			++transitions;
			REQUIRE(sa.length(target) > sa.length(std::uint32_t(v)));
		});
	REQUIRE(transitions == sa.transition_count());
}

TEST_CASE("suffix_automaton: reserve, and clones of wide states", "[strings]") {
	auto seed = GENERATE(take(15, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	// a compact vocabulary of chunks over 26 letters: repeated chunks force
	// splits of states with many outgoing transitions
	std::vector<std::string> chunks(4 + std::size_t(rng() % 4));
	for (auto &c : chunks) {
		c.resize(1 + std::size_t(rng() % 4));
		for (auto &x : c) x = char('a' + std::size_t(rng() % 26));
	}
	std::vector<std::string> words(1 + std::size_t(rng() % 4));
	std::size_t total = 0;
	for (auto &w : words) {
		for (std::size_t k = std::size_t(rng() % 8); k-- > 0;) w += chunks[std::size_t(rng() % chunks.size())];
		total += w.size();
	}
	suffix_automaton<26> reserved, plain;
	reserved.reserve(total);
	oracle ref;
	for (auto const &w : words) {
		reserved.insert(w, gs2_lower_projection);
		plain.insert(w, gs2_lower_projection);
		ref.insert(w);
	}
	REQUIRE(reserved.state_count() == plain.state_count());
	REQUIRE(reserved.transition_count() == plain.transition_count());
	REQUIRE(reserved.distinct_substring_count() == ref.subs.size());
	auto const occ = reserved.occurrence_counts();
	for (auto const &s : ref.subs) {
		auto const v = walk_string(reserved, s);
		REQUIRE(v.has_value());
		REQUIRE(occ[*v] == std::uint64_t(ref.occurrences(s)));
	}
	for (int probe = 0; probe < 200; ++probe) {
		std::string s = chunks[std::size_t(rng() % chunks.size())] + chunks[std::size_t(rng() % chunks.size())];
		REQUIRE(reserved.contains(s, gs2_lower_projection) == ref.contains(s));
	}
}

TEST_CASE("suffix_automaton: streaming extend/finish_word equals whole-word insert", "[strings]") {
	auto seed = GENERATE(take(15, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<std::string> words(1 + std::size_t(rng() % 5));
	for (auto &w : words) {
		w.resize(std::size_t(rng() % 15));
		for (auto &c : w) c = char('a' + std::size_t(rng() % 3));
	}

	suffix_automaton<26> inserted;
	suffix_automaton<26> streamed;
	for (auto const &w : words) {
		inserted.insert(w, gs2_lower_projection);
		for (char const c : w) {
			auto const cur = streamed.extend(gs2_lower_projection(c));
			REQUIRE(cur == streamed.last());
		}
		streamed.finish_word();
		REQUIRE(streamed.last() == streamed.root());
	}

	REQUIRE(inserted.state_count() == streamed.state_count());
	REQUIRE(inserted.transition_count() == streamed.transition_count());
	REQUIRE(inserted.distinct_substring_count() == streamed.distinct_substring_count());
	REQUIRE(inserted.occurrence_counts() == streamed.occurrence_counts());

	for (int probe = 0; probe < 100; ++probe) {
		std::string s(1 + std::size_t(rng() % 12), 'a');
		for (auto &c : s) c = char('a' + std::size_t(rng() % 3));
		REQUIRE(inserted.contains(s, gs2_lower_projection) == streamed.contains(s, gs2_lower_projection));
	}
}

TEST_CASE("suffix_automaton: known compact cases", "[strings]") {
	SECTION("empty automaton") {
		suffix_automaton<26> sa;
		REQUIRE(sa.state_count() == 1);
		REQUIRE(sa.transition_count() == 0);
		REQUIRE(sa.distinct_substring_count() == 0);
		REQUIRE(sa.contains(std::string{}, gs2_lower_projection));
		REQUIRE_FALSE(sa.contains(std::string{"a"}, gs2_lower_projection));
		REQUIRE(sa.terminals() == std::vector<bool>{true});
	}
	SECTION("inserting the empty word records no end state") {
		suffix_automaton<26> sa;
		REQUIRE(sa.insert(std::string{}, gs2_lower_projection) == sa.root());
		REQUIRE(sa.state_count() == 1);
		REQUIRE(sa.terminals() == std::vector<bool>{true});
	}
	SECTION("abcbc: the textbook clone") {
		suffix_automaton<26> sa;
		sa.insert(std::string{"abcbc"}, gs2_lower_projection);
		// distinct substrings of abcbc: 5+4+3+2+1 - |{bc, c, b}| duplicates = 12
		REQUIRE(sa.distinct_substring_count() == 12);
		auto const occ = sa.occurrence_counts();
		REQUIRE(occ[*walk_string(sa, std::string{"bc"})] == 2);
		REQUIRE(occ[*walk_string(sa, std::string{"abc"})] == 1);
	}
	SECTION("aaaa: one chain, occurrence counts 4..1") {
		suffix_automaton<26> sa;
		sa.insert(std::string{"aaaa"}, gs2_lower_projection);
		REQUIRE(sa.state_count() == 5);
		auto const occ = sa.occurrence_counts();
		auto const terminal = sa.terminals();
		std::string s;
		for (int len = 1; len <= 4; ++len) {
			s.push_back('a');
			REQUIRE(occ[*walk_string(sa, s)] == std::uint64_t(5 - len));
			REQUIRE(terminal[*walk_string(sa, s)]);
		}
	}
	SECTION("duplicate words double occurrence counts") {
		suffix_automaton<26> sa;
		sa.insert(std::string{"aba"}, gs2_lower_projection);
		sa.insert(std::string{"aba"}, gs2_lower_projection);
		auto const occ = sa.occurrence_counts();
		REQUIRE(occ[*walk_string(sa, std::string{"aba"})] == 2);
		REQUIRE(occ[*walk_string(sa, std::string{"a"})] == 4);
	}
	SECTION("clear resets to the empty automaton") {
		suffix_automaton<26> sa;
		sa.insert(std::string{"banana"}, gs2_lower_projection);
		sa.clear();
		REQUIRE(sa.state_count() == 1);
		REQUIRE(sa.distinct_substring_count() == 0);
		sa.insert(std::string{"ab"}, gs2_lower_projection);
		REQUIRE(sa.distinct_substring_count() == 3);
	}
}

TEST_CASE("suffix_automaton: large sparse alphabet with projection", "[strings]") {
	auto seed = GENERATE(take(5, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	constexpr std::size_t U = 1'000'000;
	suffix_automaton<U> sa;

	// few distinct symbols scattered over [0, U) so substrings actually repeat
	std::vector<std::uint32_t> symbols(6);
	for (auto &s : symbols) s = std::uint32_t(std::size_t(rng() % U));
	std::vector<std::vector<std::uint32_t>> words(3);
	for (auto &w : words) {
		w.resize(5 + std::size_t(rng() % 40));
		for (auto &c : w) c = symbols[std::size_t(rng() % symbols.size())];
		sa.insert(w);
	}

	auto substring_of_some = [&](std::vector<std::uint32_t> const &probe) {
		return std::ranges::any_of(words, [&](auto const &w) {
			return !probe.empty() && std::search(w.begin(), w.end(), probe.begin(), probe.end()) != w.end();
		});
	};
	for (int probe_round = 0; probe_round < 300; ++probe_round) {
		std::vector<std::uint32_t> probe(1 + std::size_t(rng() % 6));
		for (auto &c : probe) c = symbols[std::size_t(rng() % symbols.size())];
		REQUIRE(sa.contains(probe) == substring_of_some(probe));
	}
	// walk rejects out-of-alphabet symbols instead of asserting
	REQUIRE_FALSE(sa.walk(sa.root(), std::uint64_t(U)).has_value());
}

TEST_CASE("suffix_automaton: longest common substring via the classic walk", "[strings]") {
	auto seed = GENERATE(take(15, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	std::string a(5 + std::size_t(rng() % 30), 'a'), b(5 + std::size_t(rng() % 30), 'a');
	for (auto &c : a) c = char('a' + std::size_t(rng() % 3));
	for (auto &c : b) c = char('a' + std::size_t(rng() % 3));

	suffix_automaton<26> sa;
	sa.insert(a, gs2_lower_projection);
	std::size_t best = 0, len = 0;
	auto v = sa.root();
	for (char const c : b) {
		while (v != sa.root() && !sa.walk(v, gs2_lower_projection(c))) {
			v = sa.suffix_link(v);
			len = sa.length(v);
		}
		if (auto const next = sa.walk(v, gs2_lower_projection(c))) {
			v = *next;
			++len;
		}
		best = (std::max)(best, len);
	}

	std::size_t brute = 0;
	for (std::size_t i = 0; i < b.size(); ++i)
		for (std::size_t l = brute + 1; i + l <= b.size(); ++l)
			if (a.find(b.substr(i, l)) != std::string::npos)
				brute = l;
			else
				break;
	REQUIRE(best == brute);
}
