/**
 * @file
 * @brief Aho–Corasick build/scan throughput, output enumeration and dense-table
 * construction. Dense storage replaces suffix fallback with a load; table fills
 * vary zeroing, row-line hints and payload prefetch. Random dictionaries use
 * sigma 2/8/26 and U=1e6, 1e3..1e5 words of lengths 10/20/30, plus unary chains.
 * Scans process 1e6 symbols; table rows measure a whole allocation and fill.
 * Xeon Platinum 8358, GCC 15.2.0, RelWithDebInfo, assertions off, -O2/native,
 * 2026-11-02: sigma=26 1e5x10 table library 72.3/71.9 ms, zeroed first-line
 * 76.4/75.2, overwrite first-line 73.9/74.5. Full table: docs/aho-corasick.md.
 * Build/scan, step/walk fills and dense-engine races are not measured yet.
 */
#include <atl/experimental/strings/aho_corasick_variants.hpp>
#include <atl/strings/aho_corasick.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

using atl::uint32;
using atl::uint64;

namespace {
	/** @brief Occurrences counted by enumerating every match down the output links. */
	template<typename Automaton>
		requires requires(Automaton const &ac, uint32 v) {
			{ ac.root() } -> std::convertible_to<uint32>;
			{ ac.step(v, v) } -> std::convertible_to<uint32>;
			{ ac.match_count(v) } -> std::convertible_to<uint64>;
			{ ac.word_count(v) } -> std::convertible_to<uint64>;
			{ ac.output_link(v) } -> std::same_as<std::optional<uint32>>;
			{ ac.words_at(v).size() } -> std::convertible_to<std::size_t>;
		}
	uint64 count_matches_by_output_links(Automaton const &ac, std::vector<uint32> const &text) {
		uint64 total = 0;
		uint32 v = ac.root();
		for (uint32 const c : text) {
			v = ac.step(v, c);
			std::optional<uint32> u = ac.word_count(v) != 0 ? std::optional<uint32>(v) : ac.output_link(v);
			for (; u; u = ac.output_link(*u)) total += ac.words_at(*u).size();
		}
		return total;
	}

	/** @brief Occurrences counted through the precomputed per-state match counts. */
	template<typename Automaton>
		requires requires(Automaton const &ac, uint32 v) {
			{ ac.root() } -> std::convertible_to<uint32>;
			{ ac.step(v, v) } -> std::convertible_to<uint32>;
			{ ac.match_count(v) } -> std::convertible_to<uint64>;
			{ ac.word_count(v) } -> std::convertible_to<uint64>;
			{ ac.output_link(v) } -> std::same_as<std::optional<uint32>>;
			{ ac.words_at(v).size() } -> std::convertible_to<std::size_t>;
		}
	uint64 count_matches_by_match_counts(Automaton const &ac, std::vector<uint32> const &text) {
		uint64 total = 0;
		uint32 v = ac.root();
		for (uint32 const c : text) total += ac.match_count(v = ac.step(v, c));
		return total;
	}
}

TEST_CASE("aho_corasick build and scan", "[aho_corasick]") {
	constexpr std::size_t alphabet = 1'000'000, patterns = 100'000, length = 10, n = 1'000'000;
	std::mt19937_64 rng(1);
	for (std::size_t const sigma : {std::size_t(2), std::size_t(26), alphabet}) {
		std::vector<std::vector<uint32>> words(patterns, std::vector<uint32>(length));
		for (auto &w : words)
			for (auto &c : w) c = uint32(rng() % sigma);
		std::vector<uint32> text(n);
		for (auto &c : text) c = uint32(rng() % sigma);
		std::string const shape = "sigma=" + std::to_string(sigma);
		BENCHMARK(shape + " | build 1e5x10") { return atl::aho_corasick<alphabet>(words).node_count(); };
		atl::aho_corasick<alphabet> const ac(words);
		BENCHMARK(shape + " | scan 1e6, match_count") { return count_matches_by_match_counts(ac, text); };
	}

	constexpr std::size_t m = 1000;
	std::vector<uint32> a_m_c(m + 1, 0);
	a_m_c[m] = 2;
	atl::aho_corasick<26> const ac(std::vector<std::vector<uint32>>{{0}, a_m_c});
	std::vector<uint32> all_a(n, 0), periodic(n);
	for (std::size_t i = 0; i < n; ++i) periodic[i] = i % (m + 1) == m ? 2 : 0;
	REQUIRE(count_matches_by_output_links(ac, all_a) == n);
	REQUIRE(count_matches_by_output_links(ac, periodic) == count_matches_by_match_counts(ac, periodic));
	BENCHMARK("{a, a^1e3 c} text a^1e6 | enumerate via output links") { return count_matches_by_output_links(ac, all_a); };
	BENCHMARK("{a, a^1e3 c} text a^1e6 | match_count") { return count_matches_by_match_counts(ac, all_a); };
	BENCHMARK("{a, a^1e3 c} text (a^1e3 c)^* | enumerate via output links") { return count_matches_by_output_links(ac, periodic); };
	BENCHMARK("{a, a^1e3 c} text (a^1e3 c)^* | match_count") { return count_matches_by_match_counts(ac, periodic); };
}

TEST_CASE("aho_corasick transition table", "[aho_corasick]") {
	constexpr std::size_t sigma = 26;
	for (std::size_t const k : {std::size_t(2'000), std::size_t(10'000), std::size_t(20'000)}) {
		atl::aho_corasick<sigma> const ac(std::vector<std::vector<uint32>>{std::vector<uint32>(k, 0)});
		std::size_t const states = ac.node_count();
		/** @brief Fill each cell by independent suffix-link fallback. */
		auto const transition_table_by_step = [&] {
			std::vector<uint32> go(states * sigma);
			for (uint32 v = 0; v < states; ++v)
				for (uint32 c = 0; c < sigma; ++c) go[v * sigma + c] = ac.step(v, c);
			return go;
		};
		/** @brief Fill each cell in BFS order using public trie-edge lookup. */
		auto const transition_table_by_walk = [&] {
			std::vector<uint32> go(states * sigma);
			for (uint32 const v : ac.nodes_by_depth())
				for (uint32 c = 0; c < sigma; ++c) {
					if (auto const child = ac.walk(v, c))
						go[v * sigma + c] = *child;
					else
						go[v * sigma + c] = v == ac.root() ? ac.root() : go[ac.suffix_link(v) * sigma + c];
				}
			return go;
		};
		auto const table = ac.transition_table();
		auto const walked = transition_table_by_walk();
		REQUIRE(std::ranges::equal(walked, std::span(&table.row(0)[0], states * sigma)));
		std::string const shape = "a^" + std::to_string(k) + " sigma 26";
		if (k <= 10'000) {
			REQUIRE(transition_table_by_step() == walked);
			BENCHMARK(shape + " | table by step") { return transition_table_by_step().size(); };
		}
		BENCHMARK(shape + " | table by walk") { return transition_table_by_walk().size(); };
		BENCHMARK(shape + " | transition_table library") { return ac.transition_table().state_count(); };
		// the floor under any fill: allocating and zeroing the table alone
		BENCHMARK(shape + " | zeroed allocation only") { return std::vector<uint32>(states * sigma).size(); };
	}
}

namespace {
	/** @brief A forward scan through a dense table: one load per symbol. */
	template<typename Automaton>
		requires requires(Automaton const &ac, uint32 v) {
			{ ac.root() } -> std::convertible_to<uint32>;
			{ ac.step(v, v) } -> std::convertible_to<uint32>;
			{ ac.match_count(v) } -> std::convertible_to<uint64>;
			{ ac.word_count(v) } -> std::convertible_to<uint64>;
			{ ac.output_link(v) } -> std::same_as<std::optional<uint32>>;
			{ ac.words_at(v).size() } -> std::convertible_to<std::size_t>;
		}
	uint64 count_matches_by_dense_table(Automaton const &ac, atl::dense_transitions const &table,
										std::vector<uint32> const &text) {
		uint64 total = 0;
		uint32 v = ac.root();
		for (uint32 const c : text) total += ac.match_count(v = table(v, c));
		return total;
	}

	/** @brief The race at one alphabet size, over a cache-resident and a large dictionary. */
	template<std::size_t Sigma>
		requires(Sigma >= 1 && Sigma < (uint64(1) << 32))
	void dense_race(std::mt19937_64 &rng) {
		constexpr std::size_t length = 10, n = 1'000'000;
		for (std::size_t const patterns : {std::size_t(1'000), std::size_t(10'000), std::size_t(100'000)}) {
			std::vector<std::vector<uint32>> words(patterns, std::vector<uint32>(length));
			for (auto &w : words)
				for (auto &c : w) c = uint32(rng() % Sigma);
			std::vector<uint32> text(n);
			for (auto &c : text) c = uint32(rng() % Sigma);
			atl::aho_corasick<Sigma> const sparse(words);
			atl::experimental::aho_corasick_by_dense_table<Sigma> const dense(words);
			auto const table = sparse.transition_table();
			uint64 const expect = count_matches_by_match_counts(sparse, text);
			REQUIRE(count_matches_by_match_counts(dense, text) == expect);
			REQUIRE(count_matches_by_dense_table(sparse, table, text) == expect);
			std::string const shape = "sigma=" + std::to_string(Sigma) + " " + std::to_string(patterns) + "x10 (" + std::to_string(sparse.node_count()) + " nodes)";
			BENCHMARK(shape + " | build sparse") { return atl::aho_corasick<Sigma>(words).node_count(); };
			BENCHMARK(shape + " | build by dense table") {
				return atl::experimental::aho_corasick_by_dense_table<Sigma>(words).node_count();
			};
			BENCHMARK(shape + " | transition_table library") { return sparse.transition_table().state_count(); };
			BENCHMARK(shape + " | zeroed allocation only") {
				return std::vector<uint32>(sparse.node_count() * Sigma).size();
			};
			BENCHMARK(shape + " | scan 1e6 sparse step") { return count_matches_by_match_counts(sparse, text); };
			BENCHMARK(shape + " | scan 1e6 by dense table") { return count_matches_by_match_counts(dense, text); };
			BENCHMARK(shape + " | scan 1e6 sparse + table") { return count_matches_by_dense_table(sparse, table, text); };
		}
	}
}

TEST_CASE("aho_corasick dense engine race", "[aho_corasick]") {
	std::mt19937_64 rng(2);
	dense_race<2>(rng);
	dense_race<26>(rng);
}

namespace {
	/** @brief The table-build contender rows for @p ac, library and row options. */
	template<std::size_t Sigma>
		requires(Sigma >= 1 && Sigma < (uint64(1) << 32))
	void table_variant_rows(atl::aho_corasick<Sigma> const &ac, std::string const &name) {
		using atl::experimental::row_prefetch;
		using atl::experimental::transition_table_by_row_options;
		auto const library_table = ac.transition_table();
		constexpr std::size_t always = 0, never = (std::numeric_limits<std::size_t>::max)();
		auto const same = [&](atl::dense_transitions const &t) {
			for (uint32 v = 0; v < library_table.state_count(); ++v)
				if (!std::ranges::equal(t.row(v), library_table.row(v))) return false;
			return true;
		};
		std::string const shape = name + " (" + std::to_string(ac.node_count()) + " nodes, " + std::to_string(ac.node_count() * Sigma) + " cells)";
		auto const row = [&]<std::invocable Build>(std::string const &method, Build build)
			requires std::same_as<std::invoke_result_t<Build &>, atl::dense_transitions>
		{
			REQUIRE(same(std::invoke(build)));
			BENCHMARK(shape + " | transition_table " + method) { return std::invoke(build).state_count(); };
		};
		row("library", [&] { return ac.transition_table(); });
		row("by row options, zeroed first line", [&] { return transition_table_by_row_options<false, row_prefetch::first_line>(ac, Sigma); });
		row("by row options, overwrite first line", [&] { return transition_table_by_row_options<true, row_prefetch::first_line>(ac, Sigma); });
		row("by row options, never prefetch", [&] { return transition_table_by_row_options<false, row_prefetch::none>(ac, Sigma, never); });
		row("by row options, always first line", [&] { return transition_table_by_row_options<false, row_prefetch::first_line>(ac, Sigma, always); });
		row("by row options, all lines", [&] { return transition_table_by_row_options<false, row_prefetch::all_lines>(ac, Sigma, always); });
		row("by row options, all lines write", [&] { return transition_table_by_row_options<false, row_prefetch::all_lines_write>(ac, Sigma, always); });
		row("by row options, overwrite all lines write", [&] { return transition_table_by_row_options<true, row_prefetch::all_lines_write>(ac, Sigma, always); });
	}

	/** @brief @ref table_variant_rows on @p patterns random words of @p length symbols below @p Sigma. */
	template<std::size_t Sigma>
		requires(Sigma >= 1 && Sigma < (uint64(1) << 32))
	void random_table_variant_rows(std::mt19937_64 &rng, std::size_t const patterns, std::size_t const length) {
		std::vector<std::vector<uint32>> words(patterns, std::vector<uint32>(length));
		for (auto &w : words)
			for (auto &c : w) c = uint32(rng() % Sigma);
		table_variant_rows(atl::aho_corasick<Sigma>(words),
						   "sigma=" + std::to_string(Sigma) + " " + std::to_string(patterns) + "x" + std::to_string(length));
	}
}

TEST_CASE("aho_corasick table build variants", "[aho_corasick]") {
	std::mt19937_64 rng(3);
	random_table_variant_rows<2>(rng, 1'000, 10);
	random_table_variant_rows<2>(rng, 10'000, 10);
	random_table_variant_rows<2>(rng, 100'000, 10);
	random_table_variant_rows<2>(rng, 10'000, 20);
	random_table_variant_rows<2>(rng, 100'000, 20);
	random_table_variant_rows<2>(rng, 100'000, 30);
	random_table_variant_rows<8>(rng, 1'000, 10);
	random_table_variant_rows<8>(rng, 10'000, 10);
	random_table_variant_rows<26>(rng, 1'000, 10);
	random_table_variant_rows<26>(rng, 10'000, 10);
	random_table_variant_rows<26>(rng, 100'000, 10);
	for (std::size_t const k : {std::size_t(2'000), std::size_t(20'000)})
		table_variant_rows(atl::aho_corasick<26>(std::vector<std::vector<uint32>>{std::vector<uint32>(k, 0)}),
						   "a^" + std::to_string(k) + " sigma 26");
}
