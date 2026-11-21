#include <atl/dp/concepts.hpp>
#include <atl/dp/convex_piecewise_linear.hpp>
#include <atl/dp/convex_piecewise_linear_forest.hpp>
#include <atl/dp/exact_count_optimum.hpp>
#include <atl/dp/interval_merge.hpp>
#include <atl/dp/optimal_partition.hpp>
#include <atl/dp/row_minima.hpp>
#include <atl/dp/segmentation.hpp>
#include <atl/experimental/dp/convex_piecewise_linear_variants.hpp>
#include <atl/experimental/dp/optimal_partition_variants.hpp>
#include <atl/experimental/dp/segmentation_variants.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using atl::int64;
using atl::uint32;

// a named namespace: unused members of an unnamed one draw unneeded-declaration warnings
namespace dp_accepted_values {
	/** @brief A value ordered only by a member function, so its comparator is a member-function pointer. */
	struct member_ordered {
		int64 value = 0;

		[[nodiscard]] constexpr bool less(member_ordered const &other) const { return value < other.value; }

		friend constexpr member_ordered operator+(member_ordered a, member_ordered b) { return {a.value + b.value}; }
	};
}

namespace {
	int64 free_segment_cost(uint32 j, uint32 i) {
		return int64(i - j) * int64(i - j) + 1;
	}

	int64 free_matrix_cost(uint32 i, uint32 j) {
		return (int64(i) - 2 * int64(j)) * (int64(i) - 2 * int64(j));
	}

	/**
	 * @brief Every row-minima and segment-DP entry point on costs returning
	 *        `Value`, called with each accepted callable form: a const
	 *        lambda, a mutable stateful lambda as an lvalue, the same
	 *        through std::ref, an rvalue lambda and a comparator lambda.
	 *        Narrow types (`int16`, `uint8`) promote to `int` in the
	 *        library's sums and must be converted back there.
	 */
	template<typename Value>
		requires std::semiregular<Value> && std::totally_ordered<Value>
	void check_value_type() {
		int offset = 1; // not const: the lambdas below must really capture it
		auto const matrix = [offset](uint32 i, uint32 j) {
			uint32 const d = i > 2 * j ? i - 2 * j : 2 * j - i;
			return Value(Value(d) * Value(d) + Value(offset));
		};
		auto const segment = [offset](uint32 j, uint32 i) { return Value(Value(i - j) * Value(i - j) + Value(offset)); };
		std::size_t calls = 0;
		auto counted_segment = [&calls, segment](uint32 j, uint32 i) mutable {
			++calls;
			return segment(j, i);
		};
		auto counted_matrix = [&calls, matrix](uint32 i, uint32 j) mutable {
			++calls;
			return matrix(i, j);
		};
		auto const less = [](Value const &a, Value const &b) { return a < b; };

		std::vector<uint32> const columns{0, 0, 1, 1, 2, 2};
		REQUIRE(atl::monotone_row_minima(6, 4, matrix).column == columns);
		REQUIRE(atl::totally_monotone_row_minima(6, 4, matrix, less).column == columns);
		REQUIRE(atl::monotone_row_minima(6, 4, std::ref(counted_matrix)).column == columns);
		REQUIRE(atl::totally_monotone_row_minima(6, 4, counted_matrix).column == columns);
		REQUIRE(atl::totally_monotone_row_minima(6, 4, [](uint32, uint32) { return Value(1); }).column
				== std::vector<uint32>(6, 0));

		// six items, segments cost length^2 + 1: unit segments win (12), three of two cost 15
		REQUIRE(atl::optimal_segmentation(6, segment).value[6] == Value(12));
		REQUIRE(atl::optimal_segmentation(6, counted_segment, less).value[6] == Value(12));
		REQUIRE(atl::optimal_segmentation(6, std::ref(counted_segment)).value[6] == Value(12));
		REQUIRE(atl::experimental::optimal_segmentation_by_halving_divide_and_conquer(6, segment).value[6] == Value(12));
		REQUIRE(atl::experimental::optimal_segmentation_by_halving_smawk(6, std::ref(counted_segment)).value[6] == Value(12));
		REQUIRE(atl::experimental::optimal_segmentation_by_smawk_blocks(6, counted_segment).value[6] == Value(12));

		REQUIRE(atl::optimal_partition(6, 3, segment) == std::vector<Value>{Value(37), Value(20), Value(15)});
		REQUIRE(atl::optimal_partition(6, 3, std::ref(counted_segment), less)[2] == Value(15));
		auto const rvalue_curve = atl::optimal_partition(6, 3, [offset](uint32 j, uint32 i) {
			return Value(Value(i - j) * Value(i - j) + Value(offset));
		});
		REQUIRE(rvalue_curve[1] == Value(20));
		auto const cuts = atl::optimal_partition_cuts(6, 3, counted_segment);
		REQUIRE(cuts.value == Value(15));
		REQUIRE(cuts.boundaries == std::vector<uint32>{0, 2, 4, 6});
		REQUIRE(atl::experimental::optimal_partition_by_divide_and_conquer(6, 3, segment)[2] == Value(15));
		REQUIRE(atl::experimental::optimal_partition_by_knuth_bound(6, 3, std::ref(counted_segment))[2] == Value(15));
		REQUIRE(atl::experimental::optimal_partition_by_floored_divide_and_conquer(6, 3, counted_segment, less)[2] == Value(15));
		REQUIRE(atl::experimental::optimal_partition_by_smawk(6, 3, segment)[2] == Value(15));

		// four unit stones: ((1 1)(1 1)) costs 2 + 2 + 4
		auto stones = [](uint32 l, uint32 r) { return Value(r - l); };
		auto const table = atl::optimal_interval_merge(4, stones, less);
		REQUIRE(table.value(0, 4) == Value(8));
		REQUIRE(atl::optimal_interval_merge(4, std::ref(stones)).value(0, 4) == Value(8));
		// the table keeps the comparator it was built with, and the tree is read off under it
		static_assert(std::same_as<decltype(table.comp), std::remove_const_t<decltype(less)>>);
		auto const steps = atl::optimal_merge_order(table);
		REQUIRE(steps.size() == 3);
		REQUIRE(steps[2].split == 2);
		REQUIRE(calls > 0);
	}

	/**
	 * @brief @ref atl::exact_count_optimum with `Value` totals, `Price`
	 *        prices and `Count` counts, the solver returning a std::tuple
	 *        and passed as an lvalue, an rvalue and through std::ref.
	 */
	template<typename Value, typename Price, typename Count>
		requires std::semiregular<Value> && std::totally_ordered<Value>
	void check_exact_count_types() {
		// f(c) = (c - 3)^2 on c = 0 .. 6, penalized by price * c, fewest count on ties
		auto solve = [](Price price) {
			std::tuple<Value, Count> best{Value(9), Count(0)};
			for (int c = 1; c <= 6; ++c) {
				auto const total = Value(Value((c - 3) * (c - 3)) + Value(price) * Value(c));
				if (total < std::get<0>(best)) best = {total, Count(c)};
			}
			return best;
		};
		auto const r = atl::exact_count_optimum(Count(2), Price(-20), Price(20), solve);
		static_assert(std::same_as<decltype(r.count), Count>);
		REQUIRE(r.value == Value(1));
		REQUIRE(r.price == Price(1));
		REQUIRE(r.count == Count(2));
		REQUIRE(atl::exact_count_optimum(Count(6), Price(-20), Price(20), std::ref(solve)).value == Value(9));
		REQUIRE(atl::exact_count_optimum(Count(0), Price(-20), Price(20), [solve](Price p) { return solve(p); }).value == Value(9));
		// maximizing -f: the mirror image, penalized by -price * c
		auto solve_max = [](Price price) {
			std::pair<Value, Count> best{Value(-9), Count(0)};
			for (int c = 1; c <= 6; ++c) {
				auto const total = Value(Value(-(c - 3) * (c - 3)) - Value(price) * Value(c));
				if (best.first < total) best = {total, Count(c)};
			}
			return best;
		};
		REQUIRE(atl::exact_count_optimum(Count(4), Price(-20), Price(20), solve_max, std::greater<>{}).value == Value(-1));
	}
}

TEST_CASE("dp optimization entry points accept every callable form and value type", "[dp][accepted_calls]") {
	check_value_type<int64>();
	check_value_type<atl::int16>();
	check_value_type<atl::uint8>();
	check_value_type<double>();
	check_value_type<atl::int128>();
	REQUIRE(atl::optimal_segmentation(6, free_segment_cost).value[6] == 12);
	REQUIRE(atl::optimal_partition(6, 2, &free_segment_cost)[1] == 20);
	REQUIRE(atl::totally_monotone_row_minima(6, 4, free_matrix_cost).column == std::vector<uint32>{0, 0, 1, 1, 2, 2});
	// a Boolean cost: the solvers that store values in a plain vector accept it (the
	// raw-array ones reject it, see boundary_constraints.cpp)
	auto const flag = [](uint32 i, uint32 j) { return i != j; };
	REQUIRE(atl::monotone_row_minima(2, 2, flag).column == std::vector<uint32>{0, 1});
	REQUIRE(atl::optimal_segmentation(2, flag).value[2] == true);
}

TEST_CASE("dp comparators are called through std::invoke", "[dp][accepted_calls]") {
	using dp_accepted_values::member_ordered;
	auto const by_member = &member_ordered::less;
	auto const matrix = [](uint32 i, uint32 j) { return member_ordered{(int64(i) - 2 * int64(j)) * (int64(i) - 2 * int64(j))}; };
	auto const segment = [](uint32 j, uint32 i) { return member_ordered{int64(i - j) * int64(i - j) + 1}; };
	std::vector<uint32> const columns{0, 0, 1, 1, 2, 2};
	REQUIRE(atl::monotone_row_minima(1, 1, matrix, by_member).column == std::vector<uint32>{0});
	REQUIRE(atl::monotone_row_minima(6, 4, matrix, by_member).column == columns);
	REQUIRE(atl::totally_monotone_row_minima(6, 4, matrix, by_member).column == columns);
	REQUIRE(atl::optimal_segmentation(6, segment, by_member).value[6].value == 12);
	REQUIRE(atl::experimental::optimal_segmentation_by_halving_divide_and_conquer(6, segment, by_member).value[6].value == 12);
	REQUIRE(atl::experimental::optimal_segmentation_by_halving_smawk(6, segment, by_member).value[6].value == 12);
	REQUIRE(atl::experimental::optimal_segmentation_by_smawk_blocks(6, segment, by_member).value[6].value == 12);
	REQUIRE(atl::optimal_partition(6, 3, segment, by_member)[2].value == 15);
	REQUIRE(atl::optimal_partition_cuts(6, 3, segment, by_member).value.value == 15);
	REQUIRE(atl::experimental::optimal_partition_by_divide_and_conquer(6, 3, segment, by_member)[2].value == 15);
	REQUIRE(atl::experimental::optimal_partition_by_knuth_bound(6, 3, segment, by_member)[2].value == 15);
	REQUIRE(atl::experimental::optimal_partition_by_floored_divide_and_conquer(6, 3, segment, by_member)[2].value == 15);
	REQUIRE(atl::experimental::optimal_partition_by_smawk(6, 3, segment, by_member)[2].value == 15);
	auto const stones = [](uint32 l, uint32 r) { return member_ordered{int64(r - l)}; };
	auto const table = atl::optimal_interval_merge(4, stones, by_member);
	REQUIRE(table.value(0, 4).value == 8);
	REQUIRE(atl::optimal_merge_order(table).size() == 3);
}

TEST_CASE("exact_count_optimum accepts narrow, wide and floating value types", "[dp][accepted_calls]") {
	check_exact_count_types<int64, int64, int64>();
	check_exact_count_types<atl::int16, atl::int16, atl::int16>();
	check_exact_count_types<atl::int128, int64, atl::int8>();
	check_exact_count_types<atl::int128, atl::int128, int64>();
	check_exact_count_types<double, int, unsigned>();
}

namespace {
	/**
	 * @brief Every member of @ref atl::convex_piecewise_linear (and the
	 *        configuration contender's) with `Value` positions and `Slope`
	 *        storage, on 2|x - 3| + max(0, x - 5) + 1 or its unit form.
	 *        Narrow types (`int16`, `int8`) promote to `int` in the class's
	 *        sums and must be converted back there.
	 */
	template<atl::piecewise_linear_value Value, atl::piecewise_linear_slope Slope, typename Function>
		requires std::same_as<typename Function::value_type, Value>
	void check_piecewise_linear_members() {
		Function f;
		f.reserve(8);
		if constexpr (requires { requires Function::is_unit; }) {
			f.add_absolute(Value(3));
			f.add_absolute(Value(3));
			f.add_rising_hinge(Value(5));
			f.add_falling_hinge(Value(-50));
			f.add_rising_hinge(Value(-50));
		} else {
			f.add_absolute(Value(3), 2);
			f.add_rising_hinge(Value(5));
			f.add_falling_hinge(Value(-50), 1);
			f.add_rising_hinge(Value(-50), 1);
		}
		// 2|x - 3| + max(0, x - 5) + |x + 50|, then + 1
		f.add_constant(Value(1));
		REQUIRE(f.minimum() == Value(54));
		REQUIRE(f(Value(7)) == Value(1 + 8 + 2 + 57));
		REQUIRE(f.minimizers().lower == Value(3));
		REQUIRE(f.minimizers().upper == Value(3));
		REQUIRE(f.minimizers().clamp(Value(9)) == Value(3));
		REQUIRE(f.leftmost_slope() == -3);
		REQUIRE(f.rightmost_slope() == 4);
		f.translate(Value(1));
		REQUIRE(f.take_window_minimum(Value(0), Value(2)) == Value(4));
		REQUIRE(f.insert_segment(1, Value(2)) == Value(6));
		auto const clipped = f.clip_slopes(-2, 3);
		REQUIRE(clipped.lower == Value(-49));
		REQUIRE(clipped.upper.has_value());
		REQUIRE(f.take_prefix_minimum().upper == Value(6));
		REQUIRE(f.take_suffix_minimum().lower == Value(4));
		REQUIRE(f.take_breakpoints().empty());
	}

	template<atl::piecewise_linear_value Value, atl::piecewise_linear_slope Slope>
	void check_piecewise_linear_types() {
		check_piecewise_linear_members<Value, Slope, atl::convex_piecewise_linear<Value, Slope>>();
		check_piecewise_linear_members<Value, Slope,
									   atl::experimental::convex_piecewise_linear_by_configuration<Value, Slope, true>>();
		check_piecewise_linear_members<Value, Slope,
									   atl::experimental::convex_piecewise_linear_by_configuration<Value, Slope, false>>();
	}
}

TEST_CASE("convex_piecewise_linear accepts narrow, wide and floating positions and slopes", "[dp][accepted_calls]") {
	check_piecewise_linear_types<int64, atl::unit_weight>();
	check_piecewise_linear_types<int64, int64>();
	check_piecewise_linear_types<atl::int16, atl::unit_weight>();
	check_piecewise_linear_types<atl::int16, atl::int16>();
	check_piecewise_linear_types<atl::int32, atl::int8>();
	check_piecewise_linear_types<atl::int128, atl::int128>();
	check_piecewise_linear_types<atl::int128, atl::unit_weight>();
	check_piecewise_linear_types<double, int64>();
	check_piecewise_linear_types<float, atl::unit_weight>();
	check_piecewise_linear_members<int64, int64, atl::experimental::convex_piecewise_linear_by_breakpoint_pool<int64>>();
	// the vocabulary types on their own
	atl::slope_clip_interval<atl::int16> const window{atl::int16(-2), std::nullopt};
	REQUIRE(window.clamp(atl::int16(-5)) == atl::int16(-2));
	REQUIRE(window.clamp(atl::int16(30)) == atl::int16(30));
	atl::weighted_breakpoint<double, atl::int8> const breakpoint{0.5, atl::int8(3)};
	REQUIRE(breakpoint.weight == 3);
}

namespace {
	/**
	 * @brief Every member of @ref atl::convex_piecewise_linear_forest and its
	 *        function references with `Position` coordinates and `Slope`
	 *        weights, on 2|x - 3| + max(0, x - 5) + |x + 50|. Narrow types
	 *        (`int16`, `int8`) promote to `int` in the forest's sums.
	 */
	template<atl::piecewise_linear_value Position, atl::signed_integral Slope>
	void check_forest_members() {
		using forest_type = atl::convex_piecewise_linear_forest<Position, Slope>;
		static_assert(std::same_as<typename forest_type::position_type, Position>);
		static_assert(std::same_as<typename forest_type::slope_type, Slope>);
		forest_type forest(1);
		forest.reserve(8);
		auto const f = forest[forest.id(0)];
		auto const g = forest[forest.add_function()];
		f.add_absolute(Position(3), Slope(2));
		f.add_rising_hinge(Position(5));
		g.add_falling_hinge(Position(-50), Slope(1));
		g.add_rising_hinge(Position(-50), Slope(1));
		forest.merge(f.id(), g.id());
		// the read-only view of a const forest
		forest_type const &view = forest;
		auto const read = view[f.id()];
		static_assert(std::same_as<decltype(read), typename forest_type::const_function_reference const>);
		REQUIRE(view.function_count() == 2);
		REQUIRE(view.id(0) == f.id());
		REQUIRE(read.id() == f.id());
		REQUIRE(read.leftmost_slope() == Slope(-3));
		REQUIRE(read.rightmost_slope() == Slope(4));
		REQUIRE(forest.function_count() == 2);
		REQUIRE(f.leftmost_slope() == Slope(-3));
		REQUIRE(f.rightmost_slope() == Slope(4));
		REQUIRE(f.minimizers().lower == Position(3));
		REQUIRE(f.minimizers().upper == Position(3));
		REQUIRE(f.minimizers().clamp(Position(9)) == Position(3));
		f.translate(Position(1));
		REQUIRE(f.take_window_minimum(Position(0), Position(2)) == Position(4));
		REQUIRE(f.insert_segment(Slope(1), Position(2)) == Position(6));
		auto const clipped = f.clip_slopes(Slope(-2), Slope(3));
		REQUIRE(clipped.lower == Position(-49));
		REQUIRE(clipped.upper == Position(10));
		REQUIRE(f.take_prefix_minimum().upper == Position(6));
		REQUIRE(f.take_suffix_minimum().lower == Position(4));
		REQUIRE(f.take_breakpoints().empty());
		REQUIRE(g.take_breakpoints().empty());
	}
}

TEST_CASE("convex_piecewise_linear_forest accepts narrow, wide and floating positions and slopes", "[dp][accepted_calls]") {
	check_forest_members<int64, int64>();
	check_forest_members<atl::int16, atl::int16>();
	check_forest_members<atl::int32, atl::int8>();
	check_forest_members<atl::int128, atl::int128>();
	check_forest_members<atl::int128, int64>();
	check_forest_members<double, int64>();
	check_forest_members<float, atl::int32>();
	// breakpoint sinks: a capturing lambda, a const one and a std::reference_wrapper
	atl::detail::breakpoint_pool<int64, int64> pool(1);
	pool.push(0, 4, 3);
	pool.push(0, 9, 2);
	int64 eaten = 0;
	auto const count = [&eaten](int64 const &, int64 const &weight) { eaten += weight; };
	REQUIRE(pool.eat<true>(0, 2, count) == 4);
	auto counting = [&eaten](int64, int64 weight) mutable { eaten += weight; };
	REQUIRE(pool.eat<false>(0, 1, std::ref(counting)) == 9);
	int64 visited = 0;
	pool.visit(0, [&visited](int64 const &position, int64 const &) { visited += position; });
	REQUIRE(eaten == 3);
	REQUIRE(visited == 4 + 9);
}
