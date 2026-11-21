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
#include <atl/experimental/dp/tree_transport.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using atl::int64;
using atl::uint32;

// a named namespace: unused members of an unnamed one draw unneeded-declaration warnings
namespace dp_boundary_values {
	/** @brief A value with an order but no addition. */
	struct ordered_only {
		int64 value = 0;

		friend constexpr auto operator<=>(ordered_only const &, ordered_only const &) = default;
	};

	/** @brief A semiregular value with neither order nor addition. */
	struct unordered {
		int64 value = 0;
	};

	/** @brief An ordered value with addition and subtraction but no multiplication: positions only, no slope scaling. */
	struct additive_only {
		int64 value = 0;

		friend constexpr auto operator<=>(additive_only const &, additive_only const &) = default;

		friend constexpr additive_only operator+(additive_only a, additive_only b) { return {a.value + b.value}; }

		friend constexpr additive_only operator-(additive_only a, additive_only b) { return {a.value - b.value}; }

		constexpr additive_only &operator+=(additive_only b) { return *this = *this + b; }

		constexpr additive_only &operator-=(additive_only b) { return *this = *this - b; }
	};

}

namespace {
	using dp_boundary_values::additive_only;
	using dp_boundary_values::ordered_only;
	using dp_boundary_values::unordered;

	// one concept per entry point, so a rejection is checked for each on its own
	template<typename Cost, typename Compare = std::less<>>
	concept can_monotone_row_minima = requires(Cost &cost, Compare comp) { atl::monotone_row_minima(3, 3, cost, comp); };

	template<typename Cost, typename Compare = std::less<>>
	concept can_totally_monotone_row_minima =
		requires(Cost &cost, Compare comp) { atl::totally_monotone_row_minima(3, 3, cost, comp); };

	/** @brief Whether both row-minima entry points accept the call. */
	template<typename Cost, typename Compare = std::less<>>
	constexpr bool every_row_minima_accepts =
		can_monotone_row_minima<Cost, Compare> && can_totally_monotone_row_minima<Cost, Compare>;

	/** @brief Whether each row-minima entry point, checked separately, rejects the call. */
	template<typename Cost, typename Compare = std::less<>>
	constexpr bool every_row_minima_rejects =
		!can_monotone_row_minima<Cost, Compare> && !can_totally_monotone_row_minima<Cost, Compare>;

	template<typename Cost, typename Compare = std::less<>>
	concept can_optimal_segmentation = requires(Cost &cost, Compare comp) { atl::optimal_segmentation(3, cost, comp); };

	template<typename Cost, typename Compare = std::less<>>
	concept can_optimal_partition = requires(Cost &cost, Compare comp) { atl::optimal_partition(3, 2, cost, comp); };

	template<typename Cost, typename Compare = std::less<>>
	concept can_optimal_partition_cuts = requires(Cost &cost, Compare comp) { atl::optimal_partition_cuts(3, 2, cost, comp); };

	template<typename Cost, typename Compare = std::less<>>
	concept can_optimal_interval_merge = requires(Cost &cost, Compare comp) { atl::optimal_interval_merge(3, cost, comp); };

	template<typename Cost, typename Compare = std::less<>>
	concept can_segmentation_by_halving_divide_and_conquer = requires(Cost &cost, Compare comp) {
		atl::experimental::optimal_segmentation_by_halving_divide_and_conquer(3, cost, comp);
	};

	template<typename Cost, typename Compare = std::less<>>
	concept can_segmentation_by_halving_smawk =
		requires(Cost &cost, Compare comp) { atl::experimental::optimal_segmentation_by_halving_smawk(3, cost, comp); };

	template<typename Cost, typename Compare = std::less<>>
	concept can_segmentation_by_smawk_blocks =
		requires(Cost &cost, Compare comp) { atl::experimental::optimal_segmentation_by_smawk_blocks(3, cost, comp); };

	template<typename Cost, typename Compare = std::less<>>
	concept can_partition_by_divide_and_conquer =
		requires(Cost &cost, Compare comp) { atl::experimental::optimal_partition_by_divide_and_conquer(3, 2, cost, comp); };

	template<typename Cost, typename Compare = std::less<>>
	concept can_partition_by_knuth_bound =
		requires(Cost &cost, Compare comp) { atl::experimental::optimal_partition_by_knuth_bound(3, 2, cost, comp); };

	template<typename Cost, typename Compare = std::less<>>
	concept can_partition_by_floored_divide_and_conquer = requires(Cost &cost, Compare comp) {
		atl::experimental::optimal_partition_by_floored_divide_and_conquer(3, 2, cost, comp);
	};

	template<typename Cost, typename Compare = std::less<>>
	concept can_partition_by_smawk =
		requires(Cost &cost, Compare comp) { atl::experimental::optimal_partition_by_smawk(3, 2, cost, comp); };

	/** @brief Whether every segment-DP entry point accepts the call. */
	template<typename Cost, typename Compare = std::less<>>
	constexpr bool every_segment_accepts =
		can_optimal_segmentation<Cost, Compare> && can_optimal_partition<Cost, Compare>
		&& can_optimal_partition_cuts<Cost, Compare> && can_optimal_interval_merge<Cost, Compare>
		&& can_segmentation_by_halving_divide_and_conquer<Cost, Compare> && can_segmentation_by_halving_smawk<Cost, Compare>
		&& can_segmentation_by_smawk_blocks<Cost, Compare> && can_partition_by_divide_and_conquer<Cost, Compare>
		&& can_partition_by_knuth_bound<Cost, Compare> && can_partition_by_floored_divide_and_conquer<Cost, Compare>
		&& can_partition_by_smawk<Cost, Compare>;

	/** @brief Whether each segment-DP entry point, checked separately, rejects the call. */
	template<typename Cost, typename Compare = std::less<>>
	constexpr bool every_segment_rejects =
		!can_optimal_segmentation<Cost, Compare> && !can_optimal_partition<Cost, Compare>
		&& !can_optimal_partition_cuts<Cost, Compare> && !can_optimal_interval_merge<Cost, Compare>
		&& !can_segmentation_by_halving_divide_and_conquer<Cost, Compare> && !can_segmentation_by_halving_smawk<Cost, Compare>
		&& !can_segmentation_by_smawk_blocks<Cost, Compare> && !can_partition_by_divide_and_conquer<Cost, Compare>
		&& !can_partition_by_knuth_bound<Cost, Compare> && !can_partition_by_floored_divide_and_conquer<Cost, Compare>
		&& !can_partition_by_smawk<Cost, Compare>;

	template<typename Solve, typename Price = int64, typename Compare = std::less<>>
	concept can_count = requires(Solve &solve, Price price, Compare comp) { atl::exact_count_optimum(1, price, price, solve, comp); };

	template<typename Value, typename Compare = std::less<>>
	concept can_merge_order =
		requires(atl::interval_merge_result<Value, Compare> const &table) { atl::optimal_merge_order(table); };

	template<typename Value, typename Compare = std::less<>>
	concept can_make_merge_table = requires { typename atl::interval_merge_result<Value, Compare>; };

	using good_cost = decltype([](uint32, uint32) { return int64(0); });
	using unary_cost = decltype([](uint32) { return int64(0); });
	using void_cost = decltype([](uint32, uint32) {});
	using move_only_cost = decltype([](uint32, uint32) { return std::unique_ptr<int>(); });
	using unordered_cost = decltype([](uint32, uint32) { return unordered{}; });
	using ordered_only_cost = decltype([](uint32, uint32) { return ordered_only{}; });
	using mutable_cost = decltype([n = 0](uint32, uint32) mutable { return int64(++n); });
	using string_compare = decltype([](std::string const &a, std::string const &b) { return a < b; });
	using bool_cost = decltype([](uint32 i, uint32 j) { return i < j; });

	using good_solver = decltype([](int64 p) { return std::pair<int64, int64>(p, 0); });
	using single_value_solver = decltype([](int64 p) { return p; });
	using floating_count_solver = decltype([](int64 p) { return std::pair<int64, double>(p, 0.0); });
	using string_value_solver = decltype([](int64) { return std::pair<std::string, int64>(); });
	using triple_solver = decltype([](int64 p) { return std::tuple<int64, int64, int64>(p, 0, 0); });
}

static_assert(every_row_minima_accepts<good_cost>);
static_assert(every_row_minima_accepts<good_cost, std::greater<>>);
static_assert(every_row_minima_accepts<ordered_only_cost>);
static_assert(every_row_minima_accepts<mutable_cost>);
static_assert(every_row_minima_rejects<mutable_cost const>);
static_assert(every_row_minima_rejects<unary_cost>);
static_assert(every_row_minima_rejects<void_cost>);
static_assert(every_row_minima_rejects<move_only_cost>);
static_assert(every_row_minima_rejects<unordered_cost>);
static_assert(every_row_minima_rejects<good_cost, string_compare>);
// a Boolean cost: the divide and conquer stores into a vector, SMAWK writes through data()
static_assert(can_monotone_row_minima<bool_cost>);
static_assert(!can_totally_monotone_row_minima<bool_cost>);

static_assert(every_segment_accepts<good_cost>);
static_assert(every_segment_accepts<good_cost, std::greater<>>);
static_assert(every_segment_accepts<mutable_cost>);
static_assert(every_segment_rejects<mutable_cost const>);
static_assert(every_segment_rejects<ordered_only_cost>);
static_assert(every_segment_rejects<unary_cost>);
static_assert(every_segment_rejects<move_only_cost>);
static_assert(every_segment_rejects<good_cost, string_compare>);
// a Boolean cost: rejected where the layers or the table are raw arrays (or SMAWK runs)
static_assert(can_optimal_segmentation<bool_cost>);
static_assert(can_segmentation_by_halving_divide_and_conquer<bool_cost>);
static_assert(can_segmentation_by_smawk_blocks<bool_cost>);
static_assert(!can_segmentation_by_halving_smawk<bool_cost>);
static_assert(!can_optimal_partition<bool_cost>);
static_assert(!can_optimal_partition_cuts<bool_cost>);
static_assert(!can_optimal_interval_merge<bool_cost>);
static_assert(!can_partition_by_divide_and_conquer<bool_cost>);
static_assert(!can_partition_by_knuth_bound<bool_cost>);
static_assert(!can_partition_by_floored_divide_and_conquer<bool_cost>);
static_assert(!can_partition_by_smawk<bool_cost>);

static_assert(can_merge_order<int64>);
static_assert(can_merge_order<double, std::greater<>>);
static_assert(!can_merge_order<ordered_only>);
static_assert(!can_make_merge_table<int64, string_compare>);

static_assert(can_count<good_solver>);
static_assert(can_count<good_solver, int64, std::greater<>>);
static_assert(!can_count<good_solver, double>);
static_assert(!can_count<single_value_solver>);
static_assert(!can_count<floating_count_solver>);
static_assert(!can_count<string_value_solver>);
static_assert(!can_count<triple_solver>);
static_assert(!can_count<good_solver, int64, string_compare>);

namespace {
	template<typename Value, typename Slope = atl::unit_weight>
	concept can_make_piecewise_linear = requires { typename atl::convex_piecewise_linear<Value, Slope>; };

	template<typename Value, typename Slope = int64>
	concept can_make_breakpoint_pool_contender =
		requires { typename atl::experimental::convex_piecewise_linear_by_breakpoint_pool<Value, Slope>; };

	template<typename Function>
	concept takes_hinge_weight = requires(Function f) { f.add_rising_hinge(1, 2); };
}

static_assert(atl::piecewise_linear_value<int64>);
static_assert(atl::piecewise_linear_value<atl::int128>);
static_assert(atl::piecewise_linear_value<double>);
static_assert(atl::piecewise_linear_value<additive_only>);
static_assert(!atl::piecewise_linear_value<atl::uint64>);
static_assert(!atl::piecewise_linear_value<atl::uint32>);
static_assert(!atl::piecewise_linear_value<atl::uint128>);
static_assert(!atl::piecewise_linear_value<ordered_only>);
static_assert(!atl::piecewise_linear_value<std::string>);
static_assert(atl::scalable_by<int64, atl::int8>);
static_assert(!atl::scalable_by<additive_only, int64>);
static_assert(!atl::scalable_by<int64, atl::uint32>);
static_assert(!atl::scalable_by<int64, double>);
static_assert(atl::piecewise_linear_slope<atl::unit_weight>);
static_assert(atl::piecewise_linear_slope<atl::int128>);
static_assert(!atl::piecewise_linear_slope<atl::uint64>);
static_assert(!atl::piecewise_linear_slope<bool>);
static_assert(!atl::piecewise_linear_slope<double>);

static_assert(can_make_piecewise_linear<int64>);
static_assert(can_make_piecewise_linear<atl::int16, atl::int16>);
static_assert(can_make_piecewise_linear<double, int64>);
static_assert(!can_make_piecewise_linear<atl::uint64>);
static_assert(!can_make_piecewise_linear<atl::uint128, int64>);
static_assert(!can_make_piecewise_linear<int64, atl::uint32>);
static_assert(!can_make_piecewise_linear<int64, double>);
static_assert(!can_make_piecewise_linear<additive_only>);
static_assert(!can_make_piecewise_linear<std::string>);
static_assert(can_make_breakpoint_pool_contender<int64>);
static_assert(!can_make_breakpoint_pool_contender<atl::uint64>);
static_assert(!can_make_breakpoint_pool_contender<int64, atl::unit_weight>);

// unit mode drops the weight argument; weighted mode takes it
static_assert(!takes_hinge_weight<atl::convex_piecewise_linear<int64>>);
static_assert(takes_hinge_weight<atl::convex_piecewise_linear<int64, int64>>);
static_assert(!takes_hinge_weight<atl::experimental::convex_piecewise_linear_by_floyd_replace_top<int64>>);

namespace {
	template<typename Position, typename Slope = int64>
	concept can_make_forest = requires { typename atl::convex_piecewise_linear_forest<Position, Slope>; };

	/** @brief A sink taking only a position. */
	struct position_only_sink {
		void operator()(int64 const &) const {}
	};

	/** @brief A sink that would write through its arguments. */
	struct mutating_sink {
		void operator()(int64 &, int64 &) const {}
	};

	/** @brief A heap forest without the lazy add the transport loop needs. */
	using transport_heaps_without_add =
		atl::meldable_heap_forest<atl::experimental::transport_segment, atl::experimental::transport_segment_order>;
}

static_assert(can_make_forest<int64>);
static_assert(can_make_forest<atl::int16, atl::int16>);
static_assert(can_make_forest<double, int64>);
static_assert(can_make_forest<atl::int128, atl::int128>);
static_assert(can_make_forest<additive_only>);
static_assert(!can_make_forest<atl::uint64>);

namespace {
	template<typename Reference>
	concept can_walk_minimizers = requires(Reference const &f) { f.minimizers(); };
}

// the const view offers only the O(1) queries; minimizers walks the pool and needs a non-const forest
static_assert(can_walk_minimizers<atl::convex_piecewise_linear_forest<int64, int64>::function_reference>);
static_assert(!can_walk_minimizers<atl::convex_piecewise_linear_forest<int64, int64>::const_function_reference>);
static_assert(!can_make_forest<atl::uint128, int64>);
static_assert(!can_make_forest<std::string>);
static_assert(!can_make_forest<ordered_only>);
static_assert(!can_make_forest<int64, atl::uint32>);
static_assert(!can_make_forest<int64, double>);
static_assert(!can_make_forest<int64, atl::unit_weight>);
static_assert(!can_make_forest<int64, bool>);

static_assert(atl::breakpoint_sink<void (*)(int64, int64), int64, int64>);
static_assert(atl::breakpoint_sink<std::function<void(double, int64)>, double, int64>);
static_assert(!atl::breakpoint_sink<position_only_sink, int64, int64>);
static_assert(!atl::breakpoint_sink<mutating_sink, int64, int64>);

static_assert(atl::experimental::transport_heap_forest<atl::experimental::transport_heap_forest_default>);
static_assert(!atl::experimental::transport_heap_forest<transport_heaps_without_add>);
static_assert(!atl::experimental::transport_heap_forest<std::vector<atl::experimental::transport_segment>>);

TEST_CASE("dp optimization entry points reject calls their concepts exclude", "[dp][boundary_constraints]") {
	// the checks are the static_asserts above; this keeps the unit a test
	auto const cost = [](uint32, uint32) { return int64(0); };
	REQUIRE(atl::optimal_segmentation(2, cost).value.size() == 3);
}
