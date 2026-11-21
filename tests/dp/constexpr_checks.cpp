#include <atl/dp/convex_piecewise_linear.hpp>
#include <atl/dp/convex_piecewise_linear_forest.hpp>
#include <atl/dp/exact_count_optimum.hpp>
#include <atl/dp/interval_merge.hpp>
#include <atl/dp/optimal_partition.hpp>
#include <atl/dp/row_minima.hpp>
#include <atl/dp/segmentation.hpp>
#include <atl/experimental/dp/convex_piecewise_linear_forest_variants.hpp>
#include <atl/experimental/dp/convex_piecewise_linear_variants.hpp>
#include <atl/experimental/dp/leaf_path_balance.hpp>
#include <atl/experimental/dp/leaf_path_balance_variants.hpp>
#include <atl/experimental/dp/optimal_partition_variants.hpp>
#include <atl/experimental/dp/segmentation_variants.hpp>
#include <atl/experimental/dp/tree_transport.hpp>
#include <atl/experimental/dp/tree_transport_variants.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <utility>
#include <vector>

namespace dp_optimization_checks {
	using atl::int64;
	using atl::uint32;

	// (i - 2j)^2 on 0..5 x 0..3 is Monge: row minima at j = round(i / 2), ties to the left
	constexpr auto halved_distance_cost = [](uint32 i, uint32 j) {
		int64 const d = int64(i) - 2 * int64(j);
		return d * d;
	};

	// segments of squared length: unit segments are best
	constexpr auto squared_length_cost = [](uint32 j, uint32 i) { return int64(i - j) * int64(i - j); };

	// squared length plus 3 per segment: three segments of two are best
	constexpr auto priced_length_cost = [](uint32 j, uint32 i) { return int64(i - j) * int64(i - j) + 3; };

	// unit stones: merging [l, r) costs r - l
	constexpr auto stone_cost = [](uint32 l, uint32 r) { return int64(r - l); };

	constexpr bool row_minima_proof() {
		auto const a = atl::totally_monotone_row_minima(6, 4, halved_distance_cost);
		auto const b = atl::monotone_row_minima(6, 4, halved_distance_cost);
		uint32 const want[6] = {0, 0, 1, 1, 2, 2};
		for (uint32 i = 0; i < 6; ++i)
			if (a.column[i] != want[i] || b.column[i] != want[i] || a.value[i] != b.value[i]) return false;
		return true;
	}

	constexpr bool segmentation_proof() {
		auto const cuts = atl::optimal_segmentation(6, priced_length_cost).boundaries(); // 3 * 7
		return atl::optimal_segmentation(6, squared_length_cost).value[6] == 6
			&& cuts == std::vector<uint32>{0, 2, 4, 6};
	}

	constexpr bool segmentation_contenders_proof() {
		return atl::experimental::optimal_segmentation_by_halving_divide_and_conquer(6, squared_length_cost).value[6] == 6
			&& atl::experimental::optimal_segmentation_by_halving_smawk(6, squared_length_cost).value[6] == 6
			&& atl::experimental::optimal_segmentation_by_smawk_blocks(6, squared_length_cost).value[6] == 6;
	}

	constexpr bool partition_proof() {
		auto const exact = atl::optimal_partition_cuts(6, 2, priced_length_cost); // two segments of three: 2 * 12
		return atl::optimal_partition(6, 3, squared_length_cost)[2] == 12
			&& atl::optimal_partition(6, 6, squared_length_cost)[5] == 6 && exact.value == 24
			&& exact.boundaries == std::vector<uint32>{0, 3, 6};
	}

	constexpr bool partition_by_knuth_bound_proof() {
		return atl::experimental::optimal_partition_by_knuth_bound(6, 3, squared_length_cost)[2] == 12;
	}

	constexpr bool partition_contenders_proof() {
		return atl::experimental::optimal_partition_by_divide_and_conquer(6, 3, squared_length_cost)[2] == 12
			&& atl::experimental::optimal_partition_by_floored_divide_and_conquer(6, 3, squared_length_cost)[2] == 12
			&& atl::experimental::optimal_partition_by_smawk(6, 3, squared_length_cost)[2] == 12
			&& atl::experimental::optimal_partition_by_smawk(6, 6, squared_length_cost)[5] == 6;
	}

	constexpr bool interval_merge_proof() {
		auto const table = atl::optimal_interval_merge(4, stone_cost); // ((1 1)(1 1)) costs 2 + 2 + 4
		auto const steps = atl::optimal_merge_order(table);			   // [0, 2), [2, 4), then [0, 4)
		// maximizing: the table keeps std::greater, and the tree is read off under it
		auto const negated = atl::optimal_interval_merge(4, [](uint32 l, uint32 r) { return -int64(r - l); }, std::greater<>{});
		auto const negated_steps = atl::optimal_merge_order(negated);
		return table.value(0, 4) == 8 && steps.size() == 3 && steps[2].left == 0 && steps[2].split == 2
			&& steps[2].right == 4 && negated.value(0, 4) == -8 && negated_steps.size() == 3 && negated_steps[2].split == 2;
	}

	// f(c) = (c - 3)^2 on c = 0 .. 6, penalized by price * c, fewest count on ties
	constexpr auto parabola_solver = [](int64 price) {
		std::pair<int64, int64> best{9, 0};
		for (int64 c = 1; c <= 6; ++c) {
			std::pair<int64, int64> const candidate{(c - 3) * (c - 3) + price * c, c};
			if (candidate < best) best = candidate;
		}
		return best;
	};

	constexpr bool exact_count_proof() {
		auto const two = atl::exact_count_optimum(int64(2), int64(-20), int64(20), parabola_solver);
		return two.value == 1 && two.price == 1 && two.count == 2
			&& atl::exact_count_optimum(int64(6), int64(-20), int64(20), parabola_solver).value == 9;
	}

	static_assert(segmentation_proof());
	static_assert(partition_by_knuth_bound_proof());
	static_assert(interval_merge_proof());
	static_assert(exact_count_proof());
	// These reach a recursion (row minima's divide and conquer, SMAWK, the
	// halving drivers) that passes the caller's locals down by reference,
	// which MSVC's constant evaluation is known to misread; the runtime
	// TEST_CASE below runs the same checks there.
#if !defined(_MSC_VER) || defined(__clang__)
	static_assert(row_minima_proof());
	static_assert(segmentation_contenders_proof());
	static_assert(partition_proof());
	static_assert(partition_contenders_proof());
#endif
}

namespace convex_piecewise_linear_checks {
	using atl::int64;
	namespace ex = atl::experimental;

	// |x - 1| + |x - 4| + max(0, x - 6), then a window [0, 2] and a translation
	template<typename Function>
	constexpr bool unit_slope_trick_proof() {
		Function f;
		f.add_absolute(1);
		f.add_absolute(4);
		f.add_rising_hinge(6);
		if (f.minimum() != 3 || f(7) != 10 || *f.minimizers().lower != 1 || *f.minimizers().upper != 4) return false;
		f.take_window_minimum(0, 2);
		f.translate(-1);
		f.add_falling_hinge(10);
		f.take_prefix_minimum();
		return *f.minimizers().lower == 5 && !f.minimizers().upper && f.minimum() == 8 && f(0) == 13;
	}

	// weighted hinges, a clip of both far ends and a segment of negative slope
	template<typename Function>
	constexpr bool weighted_slope_trick_proof() {
		Function f;
		f.add_falling_hinge(10);
		f.add_falling_hinge(20);
		f.add_falling_hinge(30);
		f.add_rising_hinge(0, 2); // minimum 50 on [10, 20]
		if (f.minimum() != 50 || *f.minimizers().lower != 10 || *f.minimizers().upper != 20) return false;
		auto const range = f.clip_slopes(-2, 2); // splits the weight-2 breakpoint at 0
		auto const crossing = f.insert_segment(-1, 3);
		if (f.minimum() != 47) return false;
		auto const breakpoints = f.take_breakpoints();
		return range.lower == 0 && !range.upper && crossing == 10 && f.minimum() == 0 && breakpoints.size() == 4;
	}

	// the unit form of the weighted proof: a clip of both far ends, a segment of negative slope and the breakpoints
	template<typename Function>
	constexpr bool unit_clip_and_segment_proof() {
		Function f;
		f.add_falling_hinge(10);
		f.add_falling_hinge(20);
		f.add_falling_hinge(30);
		f.add_rising_hinge(0);
		f.add_rising_hinge(0); // minimum 50 on [10, 20]
		if (f.minimum() != 50 || *f.minimizers().lower != 10 || *f.minimizers().upper != 20) return false;
		auto const range = f.clip_slopes(-2, 2);
		auto const crossing = f.insert_segment(-1, 3);
		if (f.minimum() != 47 || f.leftmost_slope() != -2) return false;
		auto const breakpoints = f.take_breakpoints();
		return range.lower == 0 && !range.upper && crossing == 10 && breakpoints.size() == 4;
	}

	// the queries and the rest of the vocabulary on int16 positions and weights: 3|x - 2| + 5
	constexpr bool narrow_queries_proof() {
		atl::convex_piecewise_linear<atl::int16, atl::int16> f;
		f.reserve(4);
		f.add_absolute(2, 3);
		f.add_constant(5);
		if (f(0) != 11 || f(4) != 11 || f.leftmost_slope() != -3 || f.rightmost_slope() != 3) return false;
		auto const lower = f.take_suffix_minimum(); // flat left of 2
		if (lower.lower != 2 || lower.upper || lower.clamp(-7) != 2 || lower.clamp(9) != 9 || f(-100) != 5) return false;
		auto const crossing = f.insert_segment(3, 1); // the rightmost slope: nothing moves
		auto const breakpoints = f.take_breakpoints();
		return crossing == 2 && breakpoints.size() == 1 && breakpoints[0].position == 2 && breakpoints[0].weight == 3;
	}

	// the unit-heaps and breakpoint-pool contenders on the unit shapes they share
	template<typename Function>
	constexpr bool contender_proof() {
		Function f;
		f.add_absolute(1);
		f.add_absolute(4);
		f.add_rising_hinge(6);
		if (f.minimum() != 3 || *f.minimizers().lower != 1 || *f.minimizers().upper != 4) return false;
		auto const window = f.take_window_minimum(0, 2);
		f.translate(-1);
		f.add_falling_hinge(10);
		auto const prefix = f.take_prefix_minimum();
		return window == 1 && prefix.upper == 5 && *f.minimizers().lower == 5 && !f.minimizers().upper && f.minimum() == 8;
	}

	static_assert(unit_slope_trick_proof<atl::convex_piecewise_linear<int64>>());
	static_assert(unit_slope_trick_proof<ex::convex_piecewise_linear_by_floyd_replace_top<int64>>());
	static_assert(unit_slope_trick_proof<ex::convex_piecewise_linear_by_whole_heap_scan<int64>>());
	static_assert(weighted_slope_trick_proof<atl::convex_piecewise_linear<int64, int64>>());
	static_assert(weighted_slope_trick_proof<ex::convex_piecewise_linear_by_floyd_replace_top<int64, int64>>());
	static_assert(weighted_slope_trick_proof<ex::convex_piecewise_linear_by_whole_heap_scan<int64, int64>>());
	static_assert(weighted_slope_trick_proof<ex::convex_piecewise_linear_by_breakpoint_pool<int64>>());
	static_assert(unit_clip_and_segment_proof<atl::convex_piecewise_linear<int64>>());
	static_assert(unit_clip_and_segment_proof<ex::convex_piecewise_linear_by_floyd_replace_top<int64>>());
	static_assert(unit_clip_and_segment_proof<ex::convex_piecewise_linear_by_whole_heap_scan<int64>>());
	static_assert(narrow_queries_proof());
	static_assert(contender_proof<ex::convex_piecewise_linear_by_unit_heaps<int64>>());
	static_assert(contender_proof<ex::convex_piecewise_linear_by_breakpoint_pool<int64>>());
}

namespace convex_piecewise_linear_forest_checks {
	using atl::int64;
	using atl::uint32;
	namespace ex = atl::experimental;

	// 2|x - 3| summed with 2 max(0, 5 - x), translated, clipped, a segment, a window and the breakpoints
	constexpr bool forest_clip_and_segment_proof() {
		atl::convex_piecewise_linear_forest<int64, int64> forest(1);
		auto const f = forest[forest.id(0)];
		auto const g = forest[forest.add_function()];
		f.add_absolute(3, 2);
		g.add_falling_hinge(5, 2);
		forest.merge(f.id(), g.id());
		f.translate(1);							 // 2|x - 4| + 2 max(0, 6 - x)
		auto const minimizers = f.minimizers();	 // slopes -4, 0 on [4, 6], 2
		auto const range = f.clip_slopes(-3, 1); // eats 1 at 4 (left), 1 at 6 (right)
		auto const crossing = f.insert_segment(0, 2);
		auto const window = f.take_window_minimum(-1, 1);
		auto const breakpoints = f.take_breakpoints();
		return minimizers.lower == 4 && minimizers.upper == 6 && range.lower == 4 && range.upper == 6
			&& range.clamp(10) == 6 && crossing == 6 && window.has_value() && breakpoints.size() == 2;
	}

	// both branches of insert_segment, prefix and suffix minima, ids and the breakpoints of a second function
	constexpr bool forest_segment_branches_and_ids_proof() {
		atl::convex_piecewise_linear_forest<int64, int64> forest(2);
		auto const f = forest[forest.id(0)];
		auto const g = forest[forest.id(1)];
		f.add_absolute(0);								// breakpoint 0 of weight 2: slopes -1, 1
		f.add_rising_hinge(4, 3);						// {0: 2, 4: 3}: slopes -1, 4
		auto const right_part = f.insert_segment(2, 5); // moves the lighter right side: {0: 2, 4: 1, 9: 2}
		auto const left_part = f.insert_segment(1, 3);	// keeps the lighter left part, moves the rest: {0: 2, 7: 1, 12: 2}
		bool const is_slope_kept = f.leftmost_slope() == -1 && f.rightmost_slope() == 4;
		auto const prefix = f.take_prefix_minimum(); // eats 4 from the right, down to the split breakpoint at 0
		bool const is_prefix_flat = f.leftmost_slope() == -1 && f.rightmost_slope() == 0;
		auto const suffix = f.take_suffix_minimum();
		g.add_rising_hinge(2);
		g.add_rising_hinge(2, 3);
		g.add_falling_hinge(6, 2);
		auto const window = g.take_window_minimum(-1, 1); // the lighter left side is moved
		auto const breakpoints = g.take_breakpoints();
		return forest.function_count() == 2 && f.id() == forest.id(0) && g.id().index() == 1 && right_part == 4
			&& left_part == 0 && is_slope_kept && prefix.upper == 0 && !prefix.lower && is_prefix_flat
			&& suffix.lower == 0 && !suffix.upper && f.leftmost_slope() == 0 && f.rightmost_slope() == 0
			&& window.has_value() && breakpoints.size() == 3 && breakpoints[0].position == 1
			&& breakpoints[0].weight == 2 && breakpoints[1].position == 3 && breakpoints[2].position == 7
			&& breakpoints[2].weight == 2 && g.leftmost_slope() == 0 && g.rightmost_slope() == 0;
	}

	// the read-only view of a const forest
	constexpr bool forest_const_view_proof() {
		atl::convex_piecewise_linear_forest<int64, int64> forest(1);
		forest[forest.id(0)].add_absolute(3, 2);
		forest[forest.id(0)].add_falling_hinge(-4, 5);
		auto const &view = std::as_const(forest);
		auto const read = view[view.id(0)];
		return view.function_count() == 1 && read.id() == forest.id(0) && read.leftmost_slope() == -7
			&& read.rightmost_slope() == 2 && forest[forest.id(0)].minimizers().lower == 3;
	}

	// the insert_segment contender on a raw pool: both sides moved, and the crossing
	constexpr bool insert_segment_by_fewer_entries_proof() {
		atl::detail::breakpoint_pool<int64, int64> pool(1);
		ex::fewer_entries_scratch<int64, int64> scratch;
		// {0: 2, 4: 3}, slopes -1 .. 4: slope 2 reaches the crossing at 4 first from the right
		pool.push(0, 0, 2);
		pool.push(0, 4, 3);
		auto const right = ex::insert_segment_by_fewer_entries(pool, 0, -1, 4, 2, 5, scratch); // {0: 2, 4: 1, 9: 2}
		auto const left = ex::insert_segment_by_fewer_entries(pool, 0, -1, 4, 0, 3, scratch);  // {0: 1, 3: 1, 7: 1, 12: 2}
		std::vector<atl::weighted_breakpoint<int64, int64>> out;
		pool.take_all(0, out);
		return right == 4 && left == 0 && out.size() == 4 && out[0].position == 0 && out[0].weight == 1
			&& out[1].position == 3 && out[2].position == 7 && out[3].position == 12 && out[3].weight == 2;
	}

	// the library pool and both contenders: push, translate, meld, take_all, clear and refill
	template<typename Pool>
	constexpr bool pool_proof() {
		Pool pool(2);
		pool.push(0, 5, 2);
		pool.push(0, 3, 1);
		pool.push(1, 9, 4);
		pool.translate(0, 1);
		pool.meld(0, 1);
		std::vector<atl::weighted_breakpoint<int64, int64>> out;
		pool.take_all(0, out);
		bool const is_taken = out.size() == 3 && out[0].position == 4 && out[1].position == 6 && out[2].position == 9;
		pool.push(0, 1, 1);
		pool.clear(0);
		pool.push(0, 2, 7);
		out.clear();
		pool.take_all(0, out);
		return is_taken && out.size() == 1 && out[0].position == 2 && out[0].weight == 7;
	}

	// the free-cost weighted sample: edges (5, 4), (15, 15) on the root, (3, 2), (5, 1) below the first
	constexpr bool leaf_path_balance_proof() {
		std::array<uint32, 5> const parent{0, 0, 0, 1, 1};
		std::array<int64, 5> const cost{0, 5, 15, 3, 5}, weight{0, 4, 15, 2, 1};
		auto const result = ex::balance_leaf_paths(parent, cost, weight);
		auto const bounded = ex::balance_leaf_paths(parent, cost, weight, 0);
		auto const by_fewer_entries = ex::balance_leaf_paths_by_fewer_entries(parent, cost, weight, 0);
		return result.total_cost == 19 && result.path_sum == 15 && result.new_cost[3] == 10 && result.new_cost[4] == 10
			&& bounded.total_cost == 19 && by_fewer_entries.total_cost == 19 && by_fewer_entries.path_sum == bounded.path_sum;
	}

	// two transport networks: a star of three (15) and a tree of six (9), on every engine
	template<atl::experimental::transport_heap_forest HeapForest>
	constexpr bool tree_transport_proof() {
		std::array<uint32, 3> const parent{0, 0, 0};
		std::array<int64, 3> const cost{0, 5, 5}, supply{2, 5, 1}, demand{1, 0, 3};
		std::array<uint32, 6> const parent2{0, 0, 0, 0, 1, 1};
		std::array<int64, 6> const cost2{0, 2, 5, 1, 5, 1}, supply2{0, 1, 2, 2, 0, 0}, demand2{0, 0, 1, 1, 1, 1};
		return ex::tree_transport_cost_on<HeapForest>(parent, cost, supply, demand) == 15
			&& ex::tree_transport_cost_on<HeapForest>(parent2, cost2, supply2, demand2) == 9;
	}

	// the two loops that differ from the example's in more than the engine
	constexpr bool tree_transport_loops_proof() {
		std::array<uint32, 6> const parent{0, 0, 0, 0, 1, 1};
		std::array<int64, 6> const cost{0, 2, 5, 1, 5, 1}, supply{0, 1, 2, 2, 0, 0}, demand{0, 0, 1, 1, 1, 1};
		return ex::tree_transport_cost(parent, cost, supply, demand) == 9
			&& ex::tree_transport_cost_by_pop_and_push(parent, cost, supply, demand) == 9
			&& ex::tree_transport_cost_by_length_records(parent, cost, supply, demand) == 9;
	}

	using segment = ex::transport_segment;
	using segment_order = ex::transport_segment_order;

	static_assert(forest_clip_and_segment_proof());
	static_assert(forest_segment_branches_and_ids_proof());
	static_assert(forest_const_view_proof());
	static_assert(insert_segment_by_fewer_entries_proof());
	static_assert(pool_proof<atl::detail::breakpoint_pool<int64, int64>>());
	static_assert(pool_proof<ex::breakpoint_pool_by_repeated_pops<int64, int64>>());
	static_assert(pool_proof<ex::breakpoint_pool_by_tunable_pairing_pops<int64, int64>>());
	static_assert(leaf_path_balance_proof());
	static_assert(tree_transport_proof<ex::transport_heap_forest_default>());
	static_assert(tree_transport_proof<ex::meldable_heap_forest_by_skew<segment, segment_order, int64>>());
	static_assert(tree_transport_proof<ex::meldable_heap_forest_by_rank_leftist<segment, segment_order, int64>>());
	static_assert(tree_transport_proof<ex::meldable_heap_forest_by_binary<segment, segment_order, int64>>());
	static_assert(tree_transport_loops_proof());
}

TEST_CASE("dp optimization solvers constant-evaluate", "[dp][constexpr]") {
	using namespace dp_optimization_checks;
	CHECK(row_minima_proof());
	CHECK(segmentation_proof());
	CHECK(segmentation_contenders_proof());
	CHECK(partition_proof());
	CHECK(partition_by_knuth_bound_proof());
	CHECK(partition_contenders_proof());
	CHECK(interval_merge_proof());
	CHECK(exact_count_proof());
}

TEST_CASE("convex_piecewise_linear constant-evaluates", "[dp][constexpr][convex_piecewise_linear]") {
	using namespace convex_piecewise_linear_checks;
	CHECK(unit_slope_trick_proof<atl::convex_piecewise_linear<int64>>());
	CHECK(weighted_slope_trick_proof<atl::convex_piecewise_linear<int64, int64>>());
	CHECK(weighted_slope_trick_proof<ex::convex_piecewise_linear_by_breakpoint_pool<int64>>());
	CHECK(unit_clip_and_segment_proof<atl::convex_piecewise_linear<int64>>());
	CHECK(narrow_queries_proof());
	CHECK(contender_proof<ex::convex_piecewise_linear_by_unit_heaps<int64>>());
}

TEST_CASE("convex_piecewise_linear_forest and its worked examples constant-evaluate",
		  "[dp][constexpr][convex_piecewise_linear_forest]") {
	using namespace convex_piecewise_linear_forest_checks;
	CHECK(forest_clip_and_segment_proof());
	CHECK(forest_segment_branches_and_ids_proof());
	CHECK(forest_const_view_proof());
	CHECK(insert_segment_by_fewer_entries_proof());
	CHECK(pool_proof<atl::detail::breakpoint_pool<int64, int64>>());
	CHECK(leaf_path_balance_proof());
	CHECK(tree_transport_proof<ex::transport_heap_forest_default>());
	CHECK(tree_transport_loops_proof());
}
