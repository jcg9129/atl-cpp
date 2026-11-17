/**
 * @file
 * @brief Races array and online Li Chao envelopes. Inline flags change slot layout; endpoint
 * carry and coordinate-search variants change descent or lookup. Online variants use
 * materialized paths, two-bound clips, whole-first queries, clip overwrite or an optional
 * dominated-segment exit. Measures interleaved insertion/query latency including construction,
 * plus separate build/insert/query throughput phases, on wide/dense domains, lines,
 * random/full/suffix/improving segments and fixed-first-point queries.
 * Xeon Platinum 8168, GCC 15.2, RelWithDebInfo, assertions off, 2026-11-03,
 * -O2 / -O2 -march=native (also -g -DNDEBUG -std=c++23): wide random segments
 * take 28.8 / 30.6 ms with the library exit versus 629 / 665 without it.
 * The older coordinate-search records use Xeon Platinum 8280, same compiler and flags,
 * 2026-11-01; full tables and phase interpretation are in docs/li-chao-tree.md.
 */

#include <atl/ds/dynamic_li_chao_tree.hpp>
#include <atl/ds/li_chao_tree.hpp>
#include <atl/ds/linear_function.hpp>
#include <atl/experimental/ds/dynamic_li_chao_tree_variants.hpp>
#include <atl/experimental/ds/li_chao_tree_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using atl::int64;

namespace {
	/** @brief One step of the traffic: an insert (line or segment `[l, r)`) or a query at `l`. */
	struct operation {
		bool is_query, is_line;
		int64 a, b, l, r; // a query asks at x = l
	};

	/**
	 * @brief The segment ends of a traffic: anywhere, the whole domain, or
	 *        `[l, hi)`; `full_queries_at_lo` is the whole domain with every
	 *        query at `x = lo`.
	 */
	enum class segment_shape { random,
							   full,
							   full_queries_at_lo,
							   suffix };

	/**
	 * @brief `count` inserts then `count` operations alternating insert and
	 *        query, line or segment insertion interleaved with point minima:
	 *        |a|, |x| <= bound, |b| <= 1e18 / 2. Every insert is a segment
	 *        (`is_line` false); @ref as_lines sends them through the line overload.
	 */
	std::vector<operation> traffic(std::size_t count, int64 lo, int64 hi, segment_shape shape, std::mt19937_64 &rng) {
		auto const coordinate = [&] { return lo + int64(rng() % std::uint64_t(hi - lo)); };
		std::vector<operation> ops;
		for (std::size_t i = 0; i < 2 * count; ++i) {
			bool const is_query = i >= count && i % 2 == 1;
			int64 const a = int64(rng() % 2'000'000'001) - 1'000'000'000;
			int64 const b = int64(rng() % 1'000'000'000'000'000'001) - 500'000'000'000'000'000;
			int64 l = coordinate(), r = coordinate();
			if (l > r) std::swap(l, r);
			if (is_query) {
				int64 const x = shape == segment_shape::full_queries_at_lo ? lo : l;
				ops.push_back({true, false, a, b, x, x});
				continue;
			}
			switch (shape) {
			case segment_shape::random:
				ops.push_back({false, false, a, b, l, r + 1});
				break;
			case segment_shape::full:
			case segment_shape::full_queries_at_lo:
				ops.push_back({false, false, a, b, lo, hi});
				break;
			case segment_shape::suffix:
				ops.push_back({false, false, a, b, l, hi});
				break;
			}
		}
		return ops;
	}

	/** @brief `ops` with every insert sent through the line overload `insert(a, b)` instead. */
	std::vector<operation> as_lines(std::vector<operation> ops) {
		for (auto &op : ops) op.is_line = !op.is_query;
		return ops;
	}

	/** @brief Replay `ops` on `tree`, accumulating answers modulo 2^64 so the queries are not optimized away. */
	template<typename Tree>
		requires requires(Tree &tree, int64 x) {
			tree.insert(x, x);
			tree.insert(x, x, x, x);
			{ tree.query(x) } -> std::same_as<std::optional<int64>>;
		}
	atl::uint64 run(Tree &tree, std::vector<operation> const &ops) {
		atl::uint64 checksum = 0;
		for (auto const &op : ops) {
			if (op.is_query)
				checksum += atl::uint64(tree.query(op.l).value_or(0));
			else if (op.is_line)
				tree.insert(op.a, op.b);
			else
				tree.insert(op.a, op.b, op.l, op.r);
		}
		return checksum;
	}

	/** @brief The sorted distinct query points of `ops`: the array tree's coordinates. */
	std::vector<int64> query_points(std::vector<operation> const &ops) {
		std::vector<int64> points;
		for (auto const &op : ops)
			if (op.is_query) points.push_back(op.l);
		std::ranges::sort(points);
		points.erase(std::ranges::unique(points).begin(), points.end());
		return points;
	}

	constexpr std::size_t count = 200'000;
	constexpr int64 bound = 1'000'000'000, dense = 1'000'000, power = int64(1) << 20;

	/** @brief Every traffic of the benchmark, drawn in a fixed order from one seed. */
	struct all_traffic {
		std::mt19937_64 rng{1};
		std::vector<operation> wide_full_at_lo = traffic(count, -bound, bound + 1, segment_shape::full_queries_at_lo, rng);
		std::vector<operation> wide_segments = traffic(count, -bound, bound + 1, segment_shape::random, rng);
		std::vector<operation> dense_full_at_lo = traffic(count, 0, dense, segment_shape::full_queries_at_lo, rng);
		std::vector<operation> dense_segments = traffic(count, 0, dense, segment_shape::random, rng);
		std::vector<operation> control_full = traffic(count, 0, power, segment_shape::full, rng);
		std::vector<operation> dense_suffixes = traffic(count, 0, dense, segment_shape::suffix, rng);
		std::vector<operation> wide_suffixes = traffic(count, -bound, bound + 1, segment_shape::suffix, rng);
		std::vector<operation> wide_full = traffic(count, -bound, bound + 1, segment_shape::full, rng);
		std::vector<operation> dense_full = traffic(count, 0, dense, segment_shape::full, rng);
	};
}

TEST_CASE("li_chao_tree", "[li_chao_tree]") {
	all_traffic const traffics;
	auto const envelope_by_query_coordinates = [](std::vector<operation> const &ops) {
		atl::li_chao_tree<int64> tree(query_points(ops));
		return run(tree, ops);
	};
	auto const envelope_by_inline_flags_on_coordinates = [](std::vector<operation> const &ops) {
		atl::experimental::li_chao_tree_by_inline_flags<int64> tree(query_points(ops));
		return run(tree, ops);
	};
	auto const envelope_by_dense_range = [](int64 lo, int64 hi, std::vector<operation> const &ops) {
		atl::li_chao_tree<int64> tree(lo, hi);
		return run(tree, ops);
	};
	auto const envelope_by_inline_flags_on_range = [](int64 lo, int64 hi, std::vector<operation> const &ops) {
		atl::experimental::li_chao_tree_by_inline_flags<int64> tree(lo, hi);
		return run(tree, ops);
	};
	auto const dynamic = [](int64 lo, int64 hi, std::vector<operation> const &ops) {
		atl::dynamic_li_chao_tree<int64> tree(lo, hi);
		return run(tree, ops);
	};

	{
		auto const &ops = traffics.wide_full_at_lo;
		BENCHMARK("full-range segments |x|<=1e9 (queries at x=lo) / array over query coordinates") { return envelope_by_query_coordinates(ops); };
		BENCHMARK("full-range segments |x|<=1e9 (queries at x=lo) / array by inline flags") {
			return envelope_by_inline_flags_on_coordinates(ops);
		};
		BENCHMARK("full-range segments |x|<=1e9 (queries at x=lo) / dynamic node pool") { return dynamic(-bound, bound + 1, ops); };
	}
	{
		auto const &ops = traffics.wide_segments;
		BENCHMARK("segments |x|<=1e9 / array over query coordinates") { return envelope_by_query_coordinates(ops); };
		BENCHMARK("segments |x|<=1e9 / array by inline flags") { return envelope_by_inline_flags_on_coordinates(ops); };
		BENCHMARK("segments |x|<=1e9 / dynamic node pool") { return dynamic(-bound, bound + 1, ops); };
	}
	{
		auto const &ops = traffics.dense_full_at_lo;
		BENCHMARK("full-range segments x in [0,1e6) (queries at x=lo) / array over the range") {
			return envelope_by_dense_range(0, dense, ops);
		};
		BENCHMARK("full-range segments x in [0,1e6) (queries at x=lo) / array by inline flags") {
			return envelope_by_inline_flags_on_range(0, dense, ops);
		};
		BENCHMARK("construction only x in [0,1e6) / array over the range") {
			atl::li_chao_tree<int64> tree(0, dense);
			return tree.size();
		};
		BENCHMARK("full-range segments x in [0,1e6) (queries at x=lo) / dynamic node pool") { return dynamic(0, dense, ops); };
	}
	{
		auto const &ops = traffics.dense_segments;
		BENCHMARK("segments x in [0,1e6) / array over the range") { return envelope_by_dense_range(0, dense, ops); };
		BENCHMARK("segments x in [0,1e6) / array by inline flags") { return envelope_by_inline_flags_on_range(0, dense, ops); };
		BENCHMARK("segments x in [0,1e6) / dynamic node pool") { return dynamic(0, dense, ops); };
	}

	for (auto const *ops : {&traffics.wide_full_at_lo, &traffics.wide_segments, &traffics.wide_full, &traffics.wide_suffixes}) {
		auto const expected = envelope_by_query_coordinates(*ops);
		REQUIRE(envelope_by_inline_flags_on_coordinates(*ops) == expected);
		REQUIRE(dynamic(-bound, bound + 1, *ops) == expected);
	}
	for (auto const *ops : {&traffics.dense_full_at_lo, &traffics.dense_segments, &traffics.dense_full, &traffics.dense_suffixes}) {
		auto const expected = envelope_by_dense_range(0, dense, *ops);
		REQUIRE(envelope_by_inline_flags_on_range(0, dense, *ops) == expected);
		REQUIRE(dynamic(0, dense, *ops) == expected);
	}

	// queries at random points: lines through insert(a, b), then the same as full-range segments
	{
		auto const &ops = traffics.wide_full;
		auto const lines = as_lines(ops);
		BENCHMARK("lines |x|<=1e9 / array over query coordinates") { return envelope_by_query_coordinates(lines); };
		BENCHMARK("lines |x|<=1e9 / dynamic node pool") { return dynamic(-bound, bound + 1, lines); };
		BENCHMARK("full-range segments |x|<=1e9 / array over query coordinates") { return envelope_by_query_coordinates(ops); };
		BENCHMARK("full-range segments |x|<=1e9 / dynamic node pool") { return dynamic(-bound, bound + 1, ops); };
	}
	{
		auto const &ops = traffics.dense_full;
		auto const lines = as_lines(ops);
		BENCHMARK("lines x in [0,1e6) / array over the range") { return envelope_by_dense_range(0, dense, lines); };
		BENCHMARK("lines x in [0,1e6) / dynamic node pool") { return dynamic(0, dense, lines); };
		BENCHMARK("full-range segments x in [0,1e6) / array over the range") { return envelope_by_dense_range(0, dense, ops); };
		BENCHMARK("full-range segments x in [0,1e6) / dynamic node pool") { return dynamic(0, dense, ops); };
	}
	// control: 2^20 points split into one canonical node even without the padding extension
	{
		auto const &ops = traffics.control_full;
		REQUIRE(envelope_by_dense_range(0, power, ops) == dynamic(0, power, ops));
		BENCHMARK("full-range segments x in [0,2^20) / array over the range") { return envelope_by_dense_range(0, power, ops); };
		BENCHMARK("full-range segments x in [0,2^20) / dynamic node pool") { return dynamic(0, power, ops); };
	}
	{
		auto const &ops = traffics.dense_suffixes;
		BENCHMARK("suffix segments [l,1e6) x in [0,1e6) / array over the range") { return envelope_by_dense_range(0, dense, ops); };
		BENCHMARK("suffix segments [l,1e6) x in [0,1e6) / dynamic node pool") { return dynamic(0, dense, ops); };
	}
	{
		auto const &ops = traffics.wide_suffixes;
		BENCHMARK("suffix segments [l,1e9] |x|<=1e9 / array over query coordinates") { return envelope_by_query_coordinates(ops); };
		BENCHMARK("suffix segments [l,1e9] |x|<=1e9 / dynamic node pool") { return dynamic(-bound, bound + 1, ops); };
	}
}

namespace {
	/** @brief The dynamic tree contenders' rows on one shape over `[lo, hi)`. */
	void contender_rows(std::string const &shape, int64 lo, int64 hi, std::vector<operation> const &ops) {
		namespace experimental = atl::experimental;
		atl::dynamic_li_chao_tree<int64> library(lo, hi);
		auto const expected = run(library, ops);
		experimental::dynamic_li_chao_tree_by_materialized_paths<int64> materialized(lo, hi);
		experimental::dynamic_li_chao_tree_by_two_bound_clips<int64> clipped(lo, hi);
		REQUIRE(run(materialized, ops) == expected);
		REQUIRE(run(clipped, ops) == expected);
		BENCHMARK(shape + " / by materialized paths") {
			experimental::dynamic_li_chao_tree_by_materialized_paths<int64> tree(lo, hi);
			return run(tree, ops);
		};
		BENCHMARK(shape + " / by two bound clips") {
			experimental::dynamic_li_chao_tree_by_two_bound_clips<int64> tree(lo, hi);
			return run(tree, ops);
		};
	}
}

TEST_CASE("dynamic_li_chao_tree segment contenders", "[li_chao_tree][dynamic]") {
	all_traffic const traffics;
	contender_rows("segments |x|<=1e9", -bound, bound + 1, traffics.wide_segments);
	contender_rows("suffix segments [l,1e9] |x|<=1e9", -bound, bound + 1, traffics.wide_suffixes);
	contender_rows("segments x in [0,1e6)", 0, dense, traffics.dense_segments);
	contender_rows("suffix segments [l,1e6) x in [0,1e6)", 0, dense, traffics.dense_suffixes);
	contender_rows("lines |x|<=1e9", -bound, bound + 1, as_lines(traffics.wide_full));
}

// the dense lines rows split into phases: construction (the 32 MB slot
// array for the array tree), inserts, and queries on the built trees
TEST_CASE("li_chao_tree phases", "[li_chao_tree]") {
	all_traffic const traffics;
	auto const lines = as_lines(traffics.dense_full);
	std::vector<operation> inserts, queries;
	for (auto const &op : lines) (op.is_query ? queries : inserts).push_back(op);
	BENCHMARK("phases lines x in [0,1e6) / array: construction + inserts") {
		atl::li_chao_tree<int64> tree(0, dense);
		return run(tree, inserts);
	};
	BENCHMARK("phases lines x in [0,1e6) / dynamic: inserts") {
		atl::dynamic_li_chao_tree<int64> tree(0, dense);
		return run(tree, inserts);
	};
	atl::li_chao_tree<int64> array(0, dense);
	run(array, inserts);
	atl::dynamic_li_chao_tree<int64> dynamic(0, dense);
	run(dynamic, inserts);
	BENCHMARK("phases lines x in [0,1e6) / array: queries") { return run(array, queries); };
	BENCHMARK("phases lines x in [0,1e6) / dynamic: queries") { return run(dynamic, queries); };
	BENCHMARK("phases lines x in [0,1e6) / array: construction") {
		atl::li_chao_tree<int64> tree(0, dense);
		return tree.size();
	};
	BENCHMARK("phases lines x in [0,1e6) / array: the inserts again") { return run(array, inserts); };
	BENCHMARK("phases lines x in [0,1e6) / dynamic: the inserts again") { return run(dynamic, inserts); };
}

namespace {
	using atl::experimental::coordinate_search;
	using li_chao_tree_by_endpoint_carry = atl::experimental::li_chao_tree_by_endpoint_carry_and_coordinate_search<int64, std::less<>, atl::linear_function<int64>, true, coordinate_search::standard>;
	using li_chao_tree_by_branchless_search = atl::experimental::li_chao_tree_by_endpoint_carry_and_coordinate_search<int64, std::less<>, atl::linear_function<int64>, false, coordinate_search::branchless>;
	using li_chao_tree_by_eytzinger_search = atl::experimental::li_chao_tree_by_endpoint_carry_and_coordinate_search<int64, std::less<>, atl::linear_function<int64>, false, coordinate_search::eytzinger>;

	/** @brief The rows of one traffic over query coordinates: library, and each contender (build + replay, as the rows above). */
	template<typename Tree>
		requires std::constructible_from<Tree, std::vector<int64>>
	atl::uint64 over_points(std::vector<operation> const &ops) {
		Tree tree(query_points(ops));
		return run(tree, ops);
	}

	/** @brief The same rows for every tree: "<shape> library", "<shape> carry endpoints", "<shape> branchless search", "<shape> eytzinger search". */
	void coordinate_rows(std::string const &shape, std::vector<operation> const &ops) {
		auto const expected = over_points<atl::li_chao_tree<int64>>(ops);
		REQUIRE(over_points<li_chao_tree_by_endpoint_carry>(ops) == expected);
		REQUIRE(over_points<li_chao_tree_by_branchless_search>(ops) == expected);
		REQUIRE(over_points<li_chao_tree_by_eytzinger_search>(ops) == expected);
		BENCHMARK(shape + " library") { return over_points<atl::li_chao_tree<int64>>(ops); };
		BENCHMARK(shape + " by endpoint carry") { return over_points<li_chao_tree_by_endpoint_carry>(ops); };
		BENCHMARK(shape + " by branchless search") { return over_points<li_chao_tree_by_branchless_search>(ops); };
		BENCHMARK(shape + " by Eytzinger search") { return over_points<li_chao_tree_by_eytzinger_search>(ops); };
	}

	/** @brief The phases of the coordinate-mode lines row for one tree type, each row separately. */
	template<typename Tree>
		requires std::constructible_from<Tree, std::vector<int64>>
	void phase_rows(std::string const &name, std::vector<int64> const &points, std::vector<operation> const &inserts,
					std::vector<operation> const &queries) {
		BENCHMARK("coordinate phases / build " + name) { return Tree(points).size(); };
		BENCHMARK("coordinate phases / build + inserts " + name) {
			Tree tree(points);
			return run(tree, inserts);
		};
		Tree built(points);
		run(built, inserts);
		BENCHMARK("coordinate phases / queries " + name) { return run(built, queries); };
		Tree const empty(points);
		BENCHMARK("coordinate phases / queries on an empty tree (the search alone) " + name) {
			atl::uint64 checksum = 0;
			for (auto const &op : queries) checksum += empty.query(op.l).has_value();
			return checksum;
		};
	}
}

TEST_CASE("li_chao_tree coordinate contenders", "[li_chao_tree]") {
	all_traffic const traffics;
	coordinate_rows("full-range segments |x|<=1e9 (queries at x=lo) / array over query coordinates", traffics.wide_full_at_lo);
	coordinate_rows("segments |x|<=1e9 / array over query coordinates", traffics.wide_segments);
	coordinate_rows("lines |x|<=1e9 / array over query coordinates", as_lines(traffics.wide_full));
	coordinate_rows("full-range segments |x|<=1e9 / array over query coordinates", traffics.wide_full);
	coordinate_rows("suffix segments [l,1e9] |x|<=1e9 / array over query coordinates", traffics.wide_suffixes);
	// the dense range does no search: only the carrying descent differs
	{
		auto const &ops = traffics.dense_segments;
		BENCHMARK("segments x in [0,1e6) / array over the range library") {
			atl::li_chao_tree<int64> tree(0, dense);
			return run(tree, ops);
		};
		BENCHMARK("segments x in [0,1e6) / array over the range by endpoint carry") {
			li_chao_tree_by_endpoint_carry tree(0, dense);
			return run(tree, ops);
		};
		auto const lines = as_lines(traffics.dense_full);
		BENCHMARK("lines x in [0,1e6) / array over the range library") {
			atl::li_chao_tree<int64> tree(0, dense);
			return run(tree, lines);
		};
		BENCHMARK("lines x in [0,1e6) / array over the range by endpoint carry") {
			li_chao_tree_by_endpoint_carry tree(0, dense);
			return run(tree, lines);
		};
	}
}

// the 28.6 ms (coordinates) against 7.4 ms (dense range) of the lines rows,
// split into build / inserts / queries, and the search alone
TEST_CASE("li_chao_tree coordinate phases", "[li_chao_tree]") {
	all_traffic const traffics;
	auto const lines = as_lines(traffics.wide_full);
	std::vector<operation> inserts, queries;
	for (auto const &op : lines) (op.is_query ? queries : inserts).push_back(op);
	auto const points = query_points(lines);
	BENCHMARK("coordinate phases / sort and deduplicate the query points") { return query_points(lines).size(); };
	phase_rows<atl::li_chao_tree<int64>>("library", points, inserts, queries);
	phase_rows<li_chao_tree_by_endpoint_carry>("by endpoint carry", points, inserts, queries);
	phase_rows<li_chao_tree_by_branchless_search>("by branchless search", points, inserts, queries);
	phase_rows<li_chao_tree_by_eytzinger_search>("by Eytzinger search", points, inserts, queries);
}

namespace {
	/** @brief The node-pool tree with whole-first queries and lost-clip overwrite switches. */
	template<bool IsQueryWholeFirst, bool OverwritesLostClips>
	using dynamic_li_chao_tree_by_query_order_and_clip_overwrite = atl::experimental::dynamic_li_chao_tree_by_query_order_and_clip_overwrite<int64, std::less<>, atl::linear_function<int64>,
																																			 IsQueryWholeFirst, OverwritesLostClips>;

	/** @brief One row of `Tree` (a dynamic tree over `[lo, hi)`) replaying `ops`. */
	template<typename Tree>
		requires std::constructible_from<Tree, int64, int64>
	void tree_row(std::string name, int64 lo, int64 hi, std::vector<operation> const &ops) {
		atl::dynamic_li_chao_tree<int64> library(lo, hi);
		Tree checked(lo, hi);
		REQUIRE(run(checked, ops) == run(library, ops));
		BENCHMARK(std::move(name)) {
			Tree tree(lo, hi);
			return run(tree, ops);
		};
	}

	/** @brief The library algorithm with the dominated-segment exit kept or removed. */
	template<bool HasDominatedSegmentExit>
	using dynamic_li_chao_tree_by_optional_exit = atl::experimental::dynamic_li_chao_tree_by_optional_dominated_exit<int64, std::less<>, atl::linear_function<int64>,
																													 HasDominatedSegmentExit>;

	/** @brief The two optional-exit rows (exit kept, exit removed) of a shape, each checked against the library checksum. */
	void optional_exit_rows(std::string const &shape, int64 lo, int64 hi, std::vector<operation> const &ops) {
		atl::dynamic_li_chao_tree<int64> library(lo, hi);
		atl::uint64 const expected = run(library, ops);
		{
			dynamic_li_chao_tree_by_optional_exit<true> copy(lo, hi);
			REQUIRE(run(copy, ops) == expected);
		}
		{
			dynamic_li_chao_tree_by_optional_exit<false> copy(lo, hi);
			REQUIRE(run(copy, ops) == expected);
		}
		tree_row<dynamic_li_chao_tree_by_optional_exit<true>>(shape + " / by optional dominated exit enabled", lo, hi, ops);
		tree_row<dynamic_li_chao_tree_by_optional_exit<false>>(shape + " / by optional dominated exit disabled", lo, hi, ops);
	}

	/** @brief The query-loop rows on a lines shape: library, materialized paths, whole-first. */
	void line_rows(std::string const &shape, int64 lo, int64 hi, std::vector<operation> const &ops) {
		tree_row<atl::dynamic_li_chao_tree<int64>>(shape + " / dynamic_li_chao_tree library", lo, hi, ops);
		tree_row<atl::experimental::dynamic_li_chao_tree_by_materialized_paths<int64>>(shape + " / dynamic_li_chao_tree by materialized paths", lo, hi,
																					   ops);
		tree_row<dynamic_li_chao_tree_by_query_order_and_clip_overwrite<true, false>>(shape + " / dynamic_li_chao_tree by whole first query", lo, hi, ops);
	}

	/** @brief Every contender's row on a segments shape, the library row first. */
	void segment_rows(std::string const &shape, int64 lo, int64 hi, std::vector<operation> const &ops) {
		tree_row<atl::dynamic_li_chao_tree<int64>>(shape + " / dynamic_li_chao_tree library", lo, hi, ops);
		tree_row<dynamic_li_chao_tree_by_query_order_and_clip_overwrite<true, false>>(shape + " / dynamic_li_chao_tree by whole first query", lo, hi, ops);
		tree_row<dynamic_li_chao_tree_by_query_order_and_clip_overwrite<false, false>>(shape + " / dynamic_li_chao_tree by omitted dominated exit", lo, hi, ops);
		tree_row<dynamic_li_chao_tree_by_query_order_and_clip_overwrite<false, true>>(shape + " / dynamic_li_chao_tree by clip overwrite", lo, hi, ops);
		optional_exit_rows(shape, lo, hi, ops);
	}
}

TEST_CASE("dynamic_li_chao_tree query order and dominated-segment pruning", "[li_chao_tree][dynamic]") {
	all_traffic const traffics;
	std::mt19937_64 rng(2);
	line_rows("lines |x|<=1e9", -bound, bound + 1, as_lines(traffics.wide_full));
	line_rows("lines x in [0,1e6)", 0, dense, as_lines(traffics.dense_full));
	line_rows("lines 1e6 |x|<=1e9", -bound, bound + 1, as_lines(traffic(5 * count, -bound, bound + 1, segment_shape::full, rng)));
	segment_rows("segments |x|<=1e9", -bound, bound + 1, traffics.wide_segments);
	segment_rows("suffix segments [l,1e9] |x|<=1e9", -bound, bound + 1, traffics.wide_suffixes);
	segment_rows("segments x in [0,1e6)", 0, dense, traffics.dense_segments);
	segment_rows("suffix segments [l,1e6) x in [0,1e6)", 0, dense, traffics.dense_suffixes);
	std::vector<operation> improving;
	improving.reserve(2 * count);
	for (std::size_t i = 0; i < count; ++i) {
		atl_assert(i < std::size_t(std::numeric_limits<int64>::max()));
		int64 const step = int64(i) + 1;
		int64 const l = step % (dense / 2), r = dense - l;
		improving.push_back({false, false, 0, -step, l, r});
		improving.push_back({true, false, 0, 0, l, l});
	}
	tree_row<atl::dynamic_li_chao_tree<int64>>("improving segments x in [0,1e6) / library", 0, dense, improving);
	tree_row<dynamic_li_chao_tree_by_query_order_and_clip_overwrite<false, false>>("improving segments x in [0,1e6) / without dominated-segment exit", 0, dense, improving);
	optional_exit_rows("improving segments x in [0,1e6)", 0, dense, improving);
}
