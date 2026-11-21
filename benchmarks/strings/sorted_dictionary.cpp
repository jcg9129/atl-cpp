/**
 * @file
 * @brief Sorted-dictionary construction and query-stream throughput. By query options
 * isolates upper gallops, miss exits, scan bounds and offset widths; by parts build
 * returns materialized arrays and varies vectorization, reserve, lower-descent LCPs
 * and linear upper scans; by inplace build isolates the upper scan with member-built
 * arrays. Library lower searches use fixed descent LCPs and upper RMQ gallops.
 * Shapes: 1e6 char words with 5e5 hits/misses/whole words; 2e4 uint64 words sharing
 * 500 symbols with 2e4 misses/unique hits/wide intervals; 2e4 token words x50.
 * Intel Xeon Platinum 8168 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, asserts off,
 * 2026-11-03, -O2 -g -DNDEBUG -std=c++23, with/without -march=native:
 * library char build 97.7/97.1 ms; parts build with descent LCPs 201/204 ms.
 * Upper linear scans have no consistent win; other micro-variant rows are not measured yet.
 * Full tables: docs/sorted-dictionary.md.
 */
#include <atl/ds/direct_rmq.hpp>
#include <atl/experimental/strings/sorted_dictionary_variants.hpp>
#include <atl/strings/lexicographic_order.hpp>
#include <atl/strings/sorted_dictionary.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
	/** @brief Measure full-operation throughput with one directly invocable contender. */
	template<typename Operation>
		requires std::invocable<Operation &>
	void measure(std::string const &label, Operation operation) {
		BENCHMARK(std::string(label)) { return std::invoke(operation); };
	}
}

using atl::uint32;
using atl::uint64;

namespace {
	/** @brief The micro-variants of atl::sorted_dictionary raced here, one switch each. */
	struct options {
		bool is_gallop = false;		/**< @brief upper end by galloping the LCP minima */
		bool is_miss_skip = false;	/**< @brief no upper-end search on a miss */
		bool is_equality = false;	/**< @brief advance on `==`, one `<` at the mismatch */
		bool is_narrow_end = false; /**< @brief uint32 word offsets */
		bool is_tail_exit = false;	/**< @brief end = n with no fold when x prefixes the last word */
		bool is_bounded = false;	/**< @brief the upper search stays below the first word the lower descent saw beyond x */
		bool is_limit_scan = false; /**< @brief symbol scans test one precomputed bound, min(m, len) */

		/** @brief Switch-wise equality. */
		constexpr bool operator==(options const &) const = default;
	};

	/**
	 * @brief A by query options of atl::sorted_dictionary's build and query, with
	 *        the variants switched by @p Options; all switches off is the
	 *        two-boundary implementation with all query shortcuts disabled.
	 */
	template<std::totally_ordered Symbol, options Options>
	class sorted_dictionary_by_query_options {
		using offset = std::conditional_t<Options.is_narrow_end, uint32, std::size_t>;
		std::vector<Symbol> sym_;
		std::vector<offset> word_end_{0};
		std::vector<uint32> neighbor_lcp_;
		atl::direct_rmq<uint32> rmq_;

		[[nodiscard]] static bool is_same(Symbol const &a, Symbol const &b) {
			if constexpr (Options.is_equality)
				return a == b;
			else
				return !(a < b) && !(b < a);
		}

	public:
		template<typename Words>
			requires atl::sortable_word_range<Words const &, std::identity> && std::ranges::sized_range<Words const &>
		explicit sorted_dictionary_by_query_options(Words const &words) {
			word_end_.reserve(words.size() + 1);
			std::identity proj;
			for (auto const &w : words) {
				atl::detail::append_projected(sym_, w, proj);
				atl_assert(sym_.size() < (uint64(1) << 32));
				word_end_.push_back(offset(sym_.size()));
			}
			uint32 const n = uint32(word_end_.size() - 1);
			neighbor_lcp_.reserve(n == 0 ? 0 : n - 1);
			for (uint32 i = 0; i + 1 < n; ++i) {
				std::size_t const a = word_end_[i], b = word_end_[i + 1];
				std::size_t const la = b - a, lb = word_end_[i + 2] - b;
				std::size_t h = 0;
				if constexpr (Options.is_limit_scan) {
					std::size_t const limit = (std::min)(la, lb);
					while (h < limit && is_same(sym_[a + h], sym_[b + h])) ++h;
				} else {
					while (h < la && h < lb && is_same(sym_[a + h], sym_[b + h])) ++h;
				}
				neighbor_lcp_.push_back(uint32(h));
			}
			rmq_ = atl::direct_rmq<uint32>(neighbor_lcp_);
		}

		sorted_dictionary_by_query_options(sorted_dictionary_by_query_options const &) = delete;

		template<typename Query>
			requires atl::sortable_text<Query const &, std::identity> && std::totally_ordered_with<atl::text_symbol_t<Query const &>, Symbol>
		atl::index_interval prefix_interval(Query const &x) const {
			auto const first = std::ranges::begin(x);
			atl_assert(std::ranges::size(x) < (uint64(1) << 32));
			uint32 const m = uint32(std::ranges::size(x));
			uint32 const n = uint32(word_end_.size() - 1);
			if (n == 0) return {0, 0};
			auto const extend = [&](uint32 i, uint32 h) -> std::pair<uint32, int> {
				std::size_t const off = word_end_[i];
				std::size_t const len = word_end_[i + 1] - off;
				if constexpr (Options.is_limit_scan) {
					uint32 const limit = uint32(std::min<std::size_t>(m, len));
					if constexpr (Options.is_equality) {
						while (h < limit && sym_[off + h] == first[h]) ++h;
						if (h == m) return {h, 0};
						if (h == len) return {h, -1};
						return {h, sym_[off + h] < first[h] ? -1 : +1};
					} else {
						for (; h < limit; ++h) {
							auto const &q = first[h];
							if (sym_[off + h] < q) return {h, -1};
							if (q < sym_[off + h]) return {h, +1};
						}
						return h == m ? std::pair{h, 0} : std::pair{h, -1};
					}
				} else if constexpr (Options.is_equality) {
					while (h < m && h < len && sym_[off + h] == first[h]) ++h;
					if (h == m) return {h, 0};
					if (h == len) return {h, -1};
					return {h, sym_[off + h] < first[h] ? -1 : +1};
				} else {
					for (; h < m && h < len; ++h) {
						auto const &q = first[h];
						if (sym_[off + h] < q) return {h, -1};
						if (q < sym_[off + h]) return {h, +1};
					}
					return h == m ? std::pair{h, 0} : std::pair{h, -1};
				}
			};
			auto const e0 = extend(0, 0);
			auto const en = n > 1 ? extend(n - 1, 0) : e0;
			// the first word known to compare greater than every x-prefixed word (n when none is known)
			uint32 out = en.second > 0 ? n - 1 : n;
			// returns the boundary and lcp(x, word at the boundary) when < n
			auto const boundary = [&](int threshold) -> std::pair<uint32, uint32> {
				if (e0.second >= threshold) return {0, e0.first};
				if (en.second < threshold) return {n, 0};
				uint32 l = 0, r = n - 1;
				uint32 llcp = e0.first, rlcp = en.first;
				while (r - l > 1) {
					uint32 const mid = l + (r - l) / 2;
					if (llcp >= rlcp) {
						uint32 const h = rmq_.fold(l, mid);
						if (h < llcp) {
							r = mid;
							rlcp = h;
						} else if (h > llcp) {
							l = mid;
						} else if (auto const [lc, s] = extend(mid, llcp); s >= threshold) {
							r = mid;
							rlcp = lc;
						} else {
							l = mid;
							llcp = lc;
						}
					} else {
						uint32 const h = rmq_.fold(mid, r);
						if (h < rlcp) {
							l = mid;
							llcp = h;
						} else if (h > rlcp) {
							r = mid;
						} else if (auto const [lc, s] = extend(mid, rlcp); s >= threshold) {
							r = mid;
							rlcp = lc;
						} else {
							l = mid;
							llcp = lc;
						}
					}
					if constexpr (Options.is_bounded)
						if (rlcp < m) out = r; // word r diverges from x, so it sorts suffix_words x
				}
				return {r, rlcp};
			};
			if constexpr (!Options.is_miss_skip && !Options.is_gallop) {
				return {boundary(0).first, boundary(1).first};
			} else {
				auto const [begin, begin_lcp] = boundary(0);
				if (Options.is_miss_skip && (begin == n || begin_lcp < m)) return {begin, begin};
				if constexpr (!Options.is_gallop) {
					return {begin, boundary(1).first};
				} else {
					if (begin == n || begin_lcp < m) return {begin, begin};
					if constexpr (Options.is_tail_exit)
						if (en.second == 0) return {begin, n};
					if constexpr (Options.is_bounded) {
						uint32 lo = begin, hi = out; // word lo is in; hi == n or word hi is out
						for (uint32 step = 1; step <= 16 && step < hi - begin; step *= 2) {
							if (rmq_.fold(begin, begin + step) < m) {
								hi = begin + step;
								break;
							}
							lo = begin + step;
						}
						while (hi - lo > 1) {
							uint32 const mid = lo + (hi - lo) / 2;
							if (rmq_.fold(begin, mid) >= m)
								lo = mid;
							else
								hi = mid;
						}
						return {begin, hi};
					} else {
						uint32 lo = begin, hi = begin + 1, step = 1; // word lo is in, word hi (or n) is out
						while (hi < n && rmq_.fold(begin, hi) >= m) {
							lo = hi;
							step *= 2;
							hi = step > 16 ? n : uint32(std::min(uint64(n), uint64(begin) + step));
						}
						while (hi - lo > 1) {
							uint32 const mid = lo + (hi - lo) / 2;
							if (rmq_.fold(begin, mid) >= m)
								lo = mid;
							else
								hi = mid;
						}
						return {begin, hi};
					}
				}
			}
		}
	};

	template<std::totally_ordered Symbol>
	std::vector<std::vector<Symbol>> sorted(std::vector<std::vector<Symbol>> words) {
		std::ranges::sort(words);
		return words;
	}

	/** @brief The brute oracle: insertion point of @p x, then the count of words it prefixes. */
	template<std::totally_ordered Symbol>
	atl::index_interval brute_interval(std::vector<std::vector<Symbol>> const &words, std::vector<Symbol> const &x) {
		std::size_t const begin = std::size_t(std::ranges::count_if(words, [&](auto const &w) { return w < x; }));
		std::size_t end = begin;
		while (end < words.size() && words[end].size() >= x.size() && std::ranges::equal(words[end].begin(), words[end].begin() + std::ptrdiff_t(x.size()), x.begin(), x.end()))
			++end;
		return {uint32(begin), uint32(end)};
	}

	/** @brief One raced variant: its row label and its switches. */
	struct variant {
		char const *name;
		options switches;
	};

	/**
	 * @brief Query variants for @p Symbol: the two-boundary implementation,
	 *        galloping with miss skips and narrow offsets, the parts-build
	 *        combination, and that combination with one shortcut at a time removed.
	 */
	template<std::totally_ordered Symbol>
	constexpr auto variants() {
		constexpr bool is_eq = !(std::integral<Symbol> && sizeof(Symbol) == 1); // the library dispatch
		constexpr options library{.is_gallop = true,
								  .is_miss_skip = true,
								  .is_equality = is_eq,
								  .is_narrow_end = true,
								  .is_tail_exit = true,
								  .is_bounded = true,
								  .is_limit_scan = true};
		auto without = [&](bool options::*part) {
			options o = library;
			o.*part = false;
			return o;
		};
		options no_gallop = without(&options::is_gallop);
		no_gallop.is_tail_exit = no_gallop.is_bounded = false; // both refine the gallop
		return std::array{
			variant{"by two-boundary search", {}},
			variant{"gallop + miss skip + narrow offsets, 2026-10-15", {.is_gallop = true, .is_miss_skip = true, .is_equality = is_eq, .is_narrow_end = true}},
			variant{"by query options", library},
			variant{"without gallop, miss skip kept", no_gallop},
			variant{"without tail exit", without(&options::is_tail_exit)},
			variant{"without gallop bound", without(&options::is_bounded)},
			variant{"without equality scan", without(&options::is_equality)},
			variant{"size_t offsets", without(&options::is_narrow_end)},
			variant{"without scan bound", without(&options::is_limit_scan)},
		};
	}

	/** @brief Whether variant @p I repeats the switches of an other one (then it is not timed twice). */
	template<std::totally_ordered Symbol, std::size_t I>
	constexpr bool is_repeat() {
		constexpr auto list = variants<Symbol>();
		for (std::size_t j = 0; j < I; ++j)
			if (list[j].switches == list[I].switches) return true;
		return false;
	}

	/** @brief One dictionary per variant of @p Symbol, all built over the same words. */
	template<std::totally_ordered Symbol, typename Indices = std::make_index_sequence<variants<Symbol>().size()>>
	struct contenders;

	/** @brief The contenders over the variant indices @p I. */
	template<std::totally_ordered Symbol, std::size_t... I>
	struct contenders<Symbol, std::index_sequence<I...>> {
		std::tuple<sorted_dictionary_by_query_options<Symbol, variants<Symbol>()[I].switches>...> all;

		explicit contenders(std::vector<std::vector<Symbol>> const &words) : all((void(I), words)...) {}

		/** @brief Calls `f(dictionary, integral_constant<size_t, I>{})` for each distinct variant. */
		template<typename F>
			requires(std::invocable<F &, sorted_dictionary_by_query_options<Symbol, variants<Symbol>()[I].switches> const &, std::integral_constant<std::size_t, I>> && ...)
		void for_each(F &&f) const {
			(
				[&] {
					if constexpr (!is_repeat<Symbol, I>()) std::invoke(f, std::get<I>(all), std::integral_constant<std::size_t, I>{});
				}(),
				...);
		}
	};

	/** @brief A raced shape: sorted words and named query streams over them. */
	template<std::totally_ordered Symbol>
	struct dictionary_shape {
		std::vector<std::vector<Symbol>> words;
		std::vector<std::pair<std::string, std::vector<std::vector<Symbol>>>> streams;
	};

	/** @brief 1e6 sorted char words of length 5-15; 5e5 prefix hits and 5e5 one-symbol misses. */
	dictionary_shape<char> char_shape(std::mt19937_64 &rng) {
		std::vector<std::vector<char>> words(1'000'000);
		for (auto &word : words) {
			word.resize(5 + std::size_t(rng() % 11));
			for (auto &c : word) c = char('a' + std::size_t(rng() % 26));
		}
		words = sorted(std::move(words));
		std::vector<std::vector<char>> hits(500'000), misses(500'000);
		for (auto &q : hits) {
			auto const &word = words[std::size_t(rng() % words.size())];
			q.assign(word.begin(), word.begin() + std::ptrdiff_t(1 + std::size_t(rng() % word.size())));
		}
		for (auto &q : misses) {
			auto const &word = words[std::size_t(rng() % words.size())];
			q.assign(word.begin(), word.end());
			q[std::size_t(rng() % q.size())] = 'z' + 1;
		}
		return {std::move(words), {{"5e5 hits", std::move(hits)}, {"5e5 misses", std::move(misses)}}};
	}

	/** @brief 2e4 uint64 words sharing a 500-symbol prefix; all-miss, unique-hit and wide-interval streams. */
	dictionary_shape<uint64> uint64_shape(std::mt19937_64 &rng) {
		std::vector<uint64> prefix(500);
		for (auto &s : prefix) s = rng();
		std::vector<std::vector<uint64>> words(20'000);
		for (auto &word : words) {
			word = prefix;
			word.resize(500 + 1 + std::size_t(rng() % 20));
			for (std::size_t i = 500; i < word.size(); ++i) word[i] = std::size_t(rng() % 4);
		}
		words = sorted(std::move(words));
		std::vector<std::vector<uint64>> miss(20'000), unique(20'000), wide(20'000);
		for (auto &q : miss) {
			q = words[std::size_t(rng() % words.size())];
			q[std::size_t(rng() % q.size())] = 5;
		}
		for (auto &q : unique) q = words[std::size_t(rng() % words.size())];
		for (auto &q : wide) {
			auto const &word = words[std::size_t(rng() % words.size())];
			q.assign(word.begin(), word.begin() + std::ptrdiff_t(500 + std::size_t(rng() % 3)));
		}
		return {std::move(words), {{"2e4 misses", std::move(miss)}, {"2e4 unique hits", std::move(unique)}, {"2e4 wide", std::move(wide)}}};
	}

	/** @brief 2e4 token words (std::string symbols of 10-30 chars, shared prefixes) by 50 tokens; 2e4 prefix queries. */
	dictionary_shape<std::string> token_shape(std::mt19937_64 &rng) {
		std::vector<std::string> tokens(64);
		for (auto &t : tokens) {
			t = "tok_";
			t.resize(10 + std::size_t(rng() % 21));
			for (std::size_t i = 4; i < t.size(); ++i) t[i] = char('a' + std::size_t(rng() % 26));
		}
		std::vector<std::vector<std::string>> words(20'000);
		for (auto &word : words) {
			word.resize(50);
			for (std::size_t i = 0; i < word.size(); ++i) word[i] = tokens[i < 20 ? i : std::size_t(rng() % 4)];
		}
		words = sorted(std::move(words));
		std::vector<std::vector<std::string>> queries(20'000);
		for (auto &q : queries) {
			auto const &word = words[std::size_t(rng() % words.size())];
			q.assign(word.begin(), word.begin() + std::ptrdiff_t(20 + std::size_t(rng() % 31)));
		}
		return {std::move(words), {{"2e4 queries", std::move(queries)}}};
	}
}

/**
 * sorted_dictionary micro-variant race (docs/sorted-dictionary.md). Shapes:
 * 1e6 sorted char words of length 5-15 with 5e5 prefix queries (hits cut from
 * random words, misses with one altered symbol); 2e4 uint64 words sharing a
 * 500-symbol prefix, 2e4 queries split into all-miss, unique-hit and
 * wide-interval streams; 2e4 token words (std::string symbols of 10-30 chars
 * with shared prefixes) by 50 tokens, 2e4 queries.
 */
TEST_CASE("sorted_dictionary contenders", "[sorted_dictionary]") {
	std::mt19937_64 rng(1);

	// brute oracle for every variant and the library class, run independently of timing
	auto const oracle = [&]<std::totally_ordered Symbol, typename MakeSymbol>(MakeSymbol const &make_symbol)
		requires std::is_invocable_r_v<Symbol, MakeSymbol const &>
	{
		for (int round = 0; round < 300; ++round) {
			std::vector<std::vector<Symbol>> words(std::size_t(rng() % 60));
			std::size_t const max_len = 1 + std::size_t(rng() % 8);
			for (auto &w : words) {
				w.resize(std::size_t(rng() % (max_len + 1)));
				for (auto &e : w) e = make_symbol();
			}
			words = sorted(std::move(words));
			contenders<Symbol> const local(words);
			atl::sorted_dictionary<Symbol> const library(words);
			for (int probe = 0; probe < 60; ++probe) {
				std::vector<Symbol> x;
				if (!words.empty() && std::size_t(rng() % 2)) {
					auto const &w = words[std::size_t(rng() % words.size())];
					x.assign(w.begin(), w.begin() + std::ptrdiff_t(rng() % (w.size() + 1)));
					if (!x.empty() && std::size_t(rng() % 3) == 0) x[std::size_t(rng() % x.size())] = make_symbol();
				} else {
					x.resize(std::size_t(rng() % (max_len + 2)));
					for (auto &s : x) s = make_symbol();
				}
				auto const expect = brute_interval(words, x);
				REQUIRE(library.prefix_interval(x) == expect);
				local.for_each([&]<typename Dictionary, std::size_t I>(Dictionary const &dict, std::integral_constant<std::size_t, I>)
								   requires requires(std::vector<Symbol> const &query) { dict.prefix_interval(query); }
							   { REQUIRE(dict.prefix_interval(x) == expect); });
			}
		}
	};
	oracle.template operator()<char>([&] { return char('a' + std::size_t(rng() % 3)); });
	oracle.template operator()<uint64>([&] { return uint64(std::size_t(rng() % 3)) << 40; });
	oracle.template operator()<std::string>([&] { return std::string(1 + std::size_t(rng() % 2), char('a' + std::size_t(rng() % 2))); });

	auto const race = [&]<std::totally_ordered Symbol>(std::string const &shape, std::vector<std::vector<Symbol>> const &words,
													   std::vector<std::pair<std::string, std::vector<std::vector<Symbol>>>> const &streams) {
		constexpr auto list = variants<Symbol>();
		atl::sorted_dictionary const reference(words);
		contenders<Symbol> const local(words);
		// every variant agrees with the library class on the timed streams
		for (auto const &[_, stream] : streams)
			for (auto const &q : stream) {
				auto const expect = reference.prefix_interval(q);
				local.for_each([&]<typename Dictionary, std::size_t I>(Dictionary const &dict, std::integral_constant<std::size_t, I>)
								   requires requires(std::vector<Symbol> const &query) { dict.prefix_interval(query); }
							   { REQUIRE(dict.prefix_interval(q) == expect); });
			}

		measure(shape + " | build library", [&] { return atl::sorted_dictionary(words).size(); });
		local.for_each([&]<typename Dictionary, std::size_t I>(Dictionary const &, std::integral_constant<std::size_t, I>)
						   requires std::is_object_v<Dictionary>
					   {
						   measure(shape + " | build " + list[I].name, [&] {
							   return sorted_dictionary_by_query_options<Symbol, list[I].switches>(words).prefix_interval(words[0]).end;
						   });
					   });
		// Repetition ruled out a cold allocator as the parts build gap's cause.
		measure(shape + " | build library, again", [&] { return atl::sorted_dictionary(words).size(); });
		auto const queries = [&]<typename Dictionary>(Dictionary const &dict, std::vector<std::vector<Symbol>> const &stream)
			requires requires(std::vector<Symbol> const &query) { dict.prefix_interval(query); }
		{
			uint64 total = 0;
			for (auto const &q : stream) {
				auto const [begin, end] = dict.prefix_interval(q);
				total += end - begin;
			}
			return total;
		};
		for (auto const &[stream_name, stream] : streams) {
			std::string const row = shape + " " + stream_name + " | ";
			measure(row + "library", [&] { return queries(reference, stream); });
			local.for_each([&]<typename Dictionary, std::size_t I>(Dictionary const &dict, std::integral_constant<std::size_t, I>)
							   requires requires(std::vector<Symbol> const &query) { dict.prefix_interval(query); }
						   {
							   measure(row + list[I].name, [&] { return queries(dict, stream); });
						   });
		}
	};

	{
		auto const [words, streams] = char_shape(rng);
		race.template operator()<char>("char 1e6 words", words, streams);
	}
	{
		auto const [words, streams] = uint64_shape(rng);
		race.template operator()<uint64>("uint64 prefix500 2e4 words", words, streams);
	}
	{
		auto const [words, streams] = token_shape(rng);
		race.template operator()<std::string>("tokens 2e4x50", words, streams);
	}
}

// Retained baselines against the combined library winners.
// The parts build and isolated descent rows preserve the old build layout;
// the linear-scan rows retain the rejected paths.
TEST_CASE("sorted_dictionary build layout, LCP descent and linear scan", "[sorted_dictionary]") {
	using atl::experimental::sorted_dictionary_by_inplace_build;
	using atl::experimental::sorted_dictionary_by_parts_build;
	using atl::experimental::sorted_dictionary_options;
	std::mt19937_64 rng(2);

	auto const contend = [&]<std::totally_ordered Symbol>(std::string const &name, dictionary_shape<Symbol> const &s) {
		auto const &words = s.words;
		constexpr options local_copy = variants<Symbol>()[2].switches;
		// every contender agrees with the library class on the timed streams
		atl::sorted_dictionary const reference(words);
		sorted_dictionary_by_parts_build<Symbol, {.is_linear_upper_scan = true}> const linear(words);
		sorted_dictionary_by_parts_build<Symbol, {.is_descent_lcp = true}> const descent(words);
		sorted_dictionary_by_parts_build<Symbol, {.is_linear_upper_scan = true, .is_descent_lcp = true}> const both(words);
		sorted_dictionary_by_inplace_build<Symbol> const copy(words);
		sorted_dictionary_by_inplace_build<Symbol, {.is_linear_upper_scan = true}> const copy_linear(words);
		REQUIRE(copy.size() == reference.size());
		REQUIRE(copy_linear.size() == reference.size());
		for (auto const &[_, stream] : s.streams)
			for (auto const &q : stream) {
				auto const expect = reference.prefix_interval(q);
				REQUIRE(copy.prefix_interval(q) == expect);
				REQUIRE(copy_linear.prefix_interval(q) == expect);
				REQUIRE(linear.prefix_interval(q) == expect);
				REQUIRE(descent.prefix_interval(q) == expect);
				REQUIRE(both.prefix_interval(q) == expect);
			}

		std::string const build = name + " | build sorted_dictionary ";
		measure(build + "library", [&] { return atl::sorted_dictionary(words).size(); });
		measure(build + "library, queried once", [&] { return atl::sorted_dictionary(words).prefix_interval(words[0]).end; });
		measure(build + "by parts build, RMQ descent", [&] { return sorted_dictionary_by_parts_build<Symbol>(words).size(); });
		measure(build + "by parts build, novector", [&] { return sorted_dictionary_by_parts_build<Symbol, {.is_novector = true}>(words).size(); });
		measure(build + "by query options", [&] { return sorted_dictionary_by_query_options<Symbol, local_copy>(words).prefix_interval(words[0]).end; });
		measure(build + "by parts build, unconditional symbol reserve", [&] { return sorted_dictionary_by_parts_build<Symbol, {.is_sym_reserved = true}>(words).size(); });
		measure(build + "by parts build, descent lcp", [&] { return sorted_dictionary_by_parts_build<Symbol, {.is_descent_lcp = true}>(words).size(); });
		// Repetition ruled out a cold allocator as the parts build gap's cause.
		measure(build + "library, again", [&] { return atl::sorted_dictionary(words).size(); });
		measure(build + "by inplace build", [&] { return sorted_dictionary_by_inplace_build<Symbol>(words).size(); });

		auto const queries = [&]<typename Dictionary>(Dictionary const &dict, std::vector<std::vector<Symbol>> const &stream)
			requires requires(std::vector<Symbol> const &query) { dict.prefix_interval(query); }
		{
			uint64 total = 0;
			for (auto const &q : stream) {
				auto const [begin, end] = dict.prefix_interval(q);
				total += end - begin;
			}
			return total;
		};
		for (auto const &[stream_name, stream] : s.streams) {
			std::string const row = name + " " + stream_name + " | sorted_dictionary ";
			measure(row + "library", [&] { return queries(reference, stream); });
			measure(row + "by parts build, linear upper scan", [&] { return queries(linear, stream); });
			measure(row + "by parts build, descent lcp", [&] { return queries(descent, stream); });
			measure(row + "by parts build, linear upper scan + descent lcp", [&] { return queries(both, stream); });
			measure(row + "by inplace build", [&] { return queries(copy, stream); });
			measure(row + "by inplace build, linear upper scan", [&] { return queries(copy_linear, stream); });
		}
	};

	{
		auto s = char_shape(rng);
		std::vector<std::vector<char>> whole(500'000);
		for (auto &q : whole) q = s.words[std::size_t(rng() % s.words.size())];
		s.streams.emplace_back("5e5 whole words", std::move(whole));
		contend.template operator()<char>("char 1e6 words", s);
	}
	contend.template operator()<uint64>("uint64 prefix500 2e4 words", uint64_shape(rng));
	contend.template operator()<std::string>("tokens 2e4x50", token_shape(rng));
}
