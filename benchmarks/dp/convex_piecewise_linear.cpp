/**
 * @file
 * @brief Slope-trick race: atl::convex_piecewise_linear in its two storage
 * modes (unit: bare positions; weighted, Slope = int64: position-weight
 * entries) against the experimental contenders, each differing in one thing:
 * by priority_queue (std::priority_queue heaps, a pop and a push where the
 * library replaces the top in place), by unit heaps (unit breakpoints without
 * the weighted machinery), by breakpoint pool (the forest's two-ended pairing
 * heaps, far ends reachable), by Floyd replace_top (bottom-up replacement of
 * the top) and by whole-heap scan (the farthest right breakpoint found over
 * the whole heap, not its leaves).
 *
 * Throughput of whole runs: n = 1e6 operations with a_i uniform in [0, 1e9)
 * (prefix minimum + |x - a|, window [-1000, 1000] + |x - a|, random hinges),
 * weighted forms with weights in [1, 1000], moving cost (add w|x - a|, then
 * clip slopes to [-speed, speed], n = 2e5, weights in [1, 10]) and 2000
 * segments at the rightmost slope after 2e4 hinges (about 1e4 right
 * entries). Contenders agree on every result before timing.
 *
 * Tuning race, Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions
 * off, 2026-11-01, plain -O2 / -march=native: the leaf scan beats the
 * whole-heap scan on rightmost segments (unit 7.3 / 9.5 ms against 12.2 /
 * 17.7); Floyd's replace_top costs unit random hinges 2% / 6% (68.5 / 68.7
 * against 67.1 / 64.7) and is not taken. The contender race (first test
 * case) has no current record; its 2026-10-23 verdict is in the experimental
 * header. Full tables: docs/convex-piecewise-linear.md.
 */
#include <atl/dp/convex_piecewise_linear.hpp>
#include <atl/experimental/dp/convex_piecewise_linear_variants.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <random>
#include <vector>

namespace ex = atl::experimental;
using atl::int64;

namespace {
	/** @brief The unit slope-trick interface the unit workloads drive. */
	template<typename Function>
	concept slope_trick_function = std::default_initializable<Function> && requires(Function f, int64 x) {
		f.take_prefix_minimum();
		f.take_window_minimum(x, x);
		f.add_absolute(x);
		f.add_rising_hinge(x);
		f.add_falling_hinge(x);
		{ f.minimum() } -> std::convertible_to<int64>;
	};

	/** @brief The weighted interface with slope clipping the weighted workloads drive. */
	template<typename Function>
	concept weighted_slope_trick_function = slope_trick_function<Function> && requires(Function f, int64 x) {
		f.add_absolute(x, x);
		f.add_rising_hinge(x, x);
		f.add_falling_hinge(x, x);
		f.clip_slopes(x, x);
	};

	/** @brief The rightmost-slope segment the segment workloads drive. */
	template<typename Function>
	concept segment_function = requires(Function f, Function const cf, int64 x) {
		{ *f.insert_segment(cf.rightmost_slope(), x) } -> std::convertible_to<int64>;
	};

	/** @brief Make a sequence nondecreasing at the least total change: prefix minimum, then add |x - a_i|. */
	template<slope_trick_function Function>
	int64 nondecreasing(std::vector<int64> const &a) {
		Function f;
		for (int64 const x : a) {
			f.take_prefix_minimum();
			f.add_absolute(x);
		}
		return f.minimum();
	}

	/** @brief Consecutive positions at most `step` apart: window minimum over [-step, step], then add |x - a_i|. */
	template<slope_trick_function Function>
	int64 bounded_steps(std::vector<int64> const &a, int64 step) {
		Function f;
		for (int64 const x : a) {
			f.take_window_minimum(-step, step);
			f.add_absolute(x);
		}
		return f.minimum();
	}

	/** @brief Random hinges only: both heaps grow to ~n/2. */
	template<slope_trick_function Function>
	int64 hinges(std::vector<int64> const &a) {
		Function f;
		for (int64 const x : a)
			if (x & 1)
				f.add_rising_hinge(x);
			else
				f.add_falling_hinge(x);
		return f.minimum();
	}

	/** @brief Weighted nondecreasing fit: prefix minimum, then add w_i |x - a_i|. */
	template<weighted_slope_trick_function Function>
	int64 weighted_nondecreasing(std::vector<int64> const &a, std::vector<int64> const &w) {
		Function f;
		for (std::size_t i = 0; i < a.size(); ++i) {
			f.take_prefix_minimum();
			f.add_absolute(a[i], w[i]);
		}
		return f.minimum();
	}

	/** @brief Random weighted hinges: both sides grow, a crossing hinge moves several breakpoints. */
	template<weighted_slope_trick_function Function>
	int64 weighted_hinges(std::vector<int64> const &a, std::vector<int64> const &w) {
		Function f;
		for (std::size_t i = 0; i < a.size(); ++i)
			if (a[i] & 1)
				f.add_rising_hinge(a[i], w[i]);
			else
				f.add_falling_hinge(a[i], w[i]);
		return f.minimum();
	}

	/**
	 * @brief Moving at a cost per unit distance: add w_i |x - a_i|, then let
	 *        the position move at cost `speed` per unit (clip slopes into
	 *        [-speed, speed]): a far-end clip on both sides every step.
	 */
	template<weighted_slope_trick_function Function>
	int64 moving_cost(std::vector<int64> const &a, std::vector<int64> const &w, int64 speed) {
		Function f;
		for (std::size_t i = 0; i < a.size(); ++i) {
			f.add_absolute(a[i], w[i]);
			f.clip_slopes(-speed, speed);
		}
		return f.minimum();
	}

	/** @brief Many weighted hinges, then `inserts` segments at the rightmost slope: each finds the farthest breakpoint right of the minimum (O(m)). */
	template<weighted_slope_trick_function Function>
		requires segment_function<Function>
	int64 segments_at_rightmost(std::vector<int64> const &a, std::vector<int64> const &w, std::size_t inserts) {
		Function f;
		for (std::size_t i = 0; i < a.size(); ++i)
			if (a[i] & 1)
				f.add_rising_hinge(a[i], w[i]);
			else
				f.add_falling_hinge(a[i], w[i]);
		int64 checksum = 0;
		for (std::size_t i = 0; i < inserts; ++i) checksum += *f.insert_segment(f.rightmost_slope(), 0) + f.minimum();
		return checksum;
	}

	/** @brief The unit version of @ref segments_at_rightmost. */
	template<slope_trick_function Function>
		requires segment_function<Function>
	int64 unit_segments_at_rightmost(std::vector<int64> const &a, std::size_t inserts) {
		Function f;
		for (int64 const x : a)
			if (x & 1)
				f.add_rising_hinge(x);
			else
				f.add_falling_hinge(x);
		int64 checksum = 0;
		for (std::size_t i = 0; i < inserts; ++i) checksum += *f.insert_segment(f.rightmost_slope(), 0) + f.minimum();
		return checksum;
	}
}

TEST_CASE("convex_piecewise_linear contenders", "[convex_piecewise_linear]") {
	std::mt19937_64 rng(1);
	std::size_t const n = 1'000'000;
	std::vector<int64> a(n), w(n), light_w(n);
	for (auto &x : a) x = int64(rng() % 1'000'000'000);
	for (auto &x : w) x = 1 + int64(rng() % 1000);
	for (auto &x : light_w) x = 1 + int64(rng() % 10);
	std::vector<int64> const a_short(a.begin(), a.begin() + 200'000), w_short(light_w.begin(), light_w.begin() + 200'000);
	using unit_mode = atl::convex_piecewise_linear<int64>;
	using weighted_mode = atl::convex_piecewise_linear<int64, int64>;
	using by_breakpoint_pool = ex::convex_piecewise_linear_by_breakpoint_pool<int64>;
	using by_unit_heaps = ex::convex_piecewise_linear_by_unit_heaps<int64>;
	using by_priority_queue = ex::convex_piecewise_linear_by_priority_queue<int64>;

	int64 const want_nondecreasing = nondecreasing<unit_mode>(a);
	REQUIRE(nondecreasing<weighted_mode>(a) == want_nondecreasing);
	REQUIRE(nondecreasing<by_breakpoint_pool>(a) == want_nondecreasing);
	REQUIRE(nondecreasing<by_unit_heaps>(a) == want_nondecreasing);
	REQUIRE(nondecreasing<by_priority_queue>(a) == want_nondecreasing);
	int64 const want_window = bounded_steps<unit_mode>(a, 1000);
	REQUIRE(bounded_steps<weighted_mode>(a, 1000) == want_window);
	REQUIRE(bounded_steps<by_breakpoint_pool>(a, 1000) == want_window);
	REQUIRE(bounded_steps<by_unit_heaps>(a, 1000) == want_window);
	REQUIRE(bounded_steps<by_priority_queue>(a, 1000) == want_window);
	int64 const want_hinges = hinges<unit_mode>(a);
	REQUIRE(hinges<weighted_mode>(a) == want_hinges);
	REQUIRE(hinges<by_breakpoint_pool>(a) == want_hinges);
	REQUIRE(hinges<by_unit_heaps>(a) == want_hinges);
	REQUIRE(hinges<by_priority_queue>(a) == want_hinges);
	REQUIRE(weighted_nondecreasing<by_breakpoint_pool>(a, w) == weighted_nondecreasing<weighted_mode>(a, w));
	REQUIRE(weighted_hinges<by_breakpoint_pool>(a, w) == weighted_hinges<weighted_mode>(a, w));
	REQUIRE(moving_cost<by_breakpoint_pool>(a_short, w_short, 100) == moving_cost<weighted_mode>(a_short, w_short, 100));

	BENCHMARK("nondecreasing n=1e6 / library, unit mode") { return nondecreasing<unit_mode>(a); };
	BENCHMARK("nondecreasing n=1e6 / library, weighted mode") { return nondecreasing<weighted_mode>(a); };
	BENCHMARK("nondecreasing n=1e6 / by breakpoint pool") { return nondecreasing<by_breakpoint_pool>(a); };
	BENCHMARK("nondecreasing n=1e6 / by unit heaps") { return nondecreasing<by_unit_heaps>(a); };
	BENCHMARK("nondecreasing n=1e6 / by priority_queue") { return nondecreasing<by_priority_queue>(a); };
	BENCHMARK("window n=1e6 / library, unit mode") { return bounded_steps<unit_mode>(a, 1000); };
	BENCHMARK("window n=1e6 / library, weighted mode") { return bounded_steps<weighted_mode>(a, 1000); };
	BENCHMARK("window n=1e6 / by breakpoint pool") { return bounded_steps<by_breakpoint_pool>(a, 1000); };
	BENCHMARK("window n=1e6 / by unit heaps") { return bounded_steps<by_unit_heaps>(a, 1000); };
	BENCHMARK("window n=1e6 / by priority_queue") { return bounded_steps<by_priority_queue>(a, 1000); };
	BENCHMARK("hinges n=1e6 / library, unit mode") { return hinges<unit_mode>(a); };
	BENCHMARK("hinges n=1e6 / library, weighted mode") { return hinges<weighted_mode>(a); };
	BENCHMARK("hinges n=1e6 / by breakpoint pool") { return hinges<by_breakpoint_pool>(a); };
	BENCHMARK("hinges n=1e6 / by unit heaps") { return hinges<by_unit_heaps>(a); };
	BENCHMARK("hinges n=1e6 / by priority_queue") { return hinges<by_priority_queue>(a); };
	BENCHMARK("weighted nondecreasing n=1e6 / library, weighted mode") { return weighted_nondecreasing<weighted_mode>(a, w); };
	BENCHMARK("weighted nondecreasing n=1e6 / by breakpoint pool") { return weighted_nondecreasing<by_breakpoint_pool>(a, w); };
	BENCHMARK("weighted hinges n=1e6 / library, weighted mode") { return weighted_hinges<weighted_mode>(a, w); };
	BENCHMARK("weighted hinges n=1e6 / by breakpoint pool") { return weighted_hinges<by_breakpoint_pool>(a, w); };
	BENCHMARK("moving speed 100 n=2e5 / library, weighted mode") { return moving_cost<weighted_mode>(a_short, w_short, 100); };
	BENCHMARK("moving speed 100 n=2e5 / by breakpoint pool") { return moving_cost<by_breakpoint_pool>(a_short, w_short, 100); };
	BENCHMARK("moving speed 1e4 n=2e5 / library, weighted mode") { return moving_cost<weighted_mode>(a_short, w_short, 10'000); };
	BENCHMARK("moving speed 1e4 n=2e5 / by breakpoint pool") { return moving_cost<by_breakpoint_pool>(a_short, w_short, 10'000); };
}

TEST_CASE("convex_piecewise_linear tuning contenders", "[convex_piecewise_linear]") {
	std::mt19937_64 rng(1);
	std::size_t const n = 1'000'000;
	std::vector<int64> a(n), w(n);
	for (auto &x : a) x = int64(rng() % 1'000'000'000);
	for (auto &x : w) x = 1 + int64(rng() % 1000);
	std::vector<int64> const a_few(a.begin(), a.begin() + 20'000), w_few(w.begin(), w.begin() + 20'000);
	using unit_mode = atl::convex_piecewise_linear<int64>;
	using weighted_mode = atl::convex_piecewise_linear<int64, int64>;
	using unit_by_floyd = ex::convex_piecewise_linear_by_floyd_replace_top<int64>;
	using unit_by_scan = ex::convex_piecewise_linear_by_whole_heap_scan<int64>;
	using weighted_by_floyd = ex::convex_piecewise_linear_by_floyd_replace_top<int64, int64>;
	using weighted_by_scan = ex::convex_piecewise_linear_by_whole_heap_scan<int64, int64>;

	REQUIRE(nondecreasing<unit_by_floyd>(a) == nondecreasing<unit_mode>(a));
	REQUIRE(bounded_steps<unit_by_floyd>(a, 1000) == bounded_steps<unit_mode>(a, 1000));
	REQUIRE(hinges<unit_by_floyd>(a) == hinges<unit_mode>(a));
	REQUIRE(weighted_nondecreasing<weighted_by_floyd>(a, w) == weighted_nondecreasing<weighted_mode>(a, w));
	REQUIRE(weighted_hinges<weighted_by_floyd>(a, w) == weighted_hinges<weighted_mode>(a, w));
	REQUIRE(unit_segments_at_rightmost<unit_by_scan>(a_few, 2000) == unit_segments_at_rightmost<unit_mode>(a_few, 2000));
	REQUIRE(segments_at_rightmost<weighted_by_scan>(a_few, w_few, 2000) == segments_at_rightmost<weighted_mode>(a_few, w_few, 2000));

	BENCHMARK("nondecreasing n=1e6 / library, unit mode") { return nondecreasing<unit_mode>(a); };
	BENCHMARK("nondecreasing n=1e6 / unit mode by Floyd replace_top") { return nondecreasing<unit_by_floyd>(a); };
	BENCHMARK("window n=1e6 / library, unit mode") { return bounded_steps<unit_mode>(a, 1000); };
	BENCHMARK("window n=1e6 / unit mode by Floyd replace_top") { return bounded_steps<unit_by_floyd>(a, 1000); };
	BENCHMARK("hinges n=1e6 / library, unit mode") { return hinges<unit_mode>(a); };
	BENCHMARK("hinges n=1e6 / unit mode by Floyd replace_top") { return hinges<unit_by_floyd>(a); };
	BENCHMARK("weighted nondecreasing n=1e6 / library, weighted mode") { return weighted_nondecreasing<weighted_mode>(a, w); };
	BENCHMARK("weighted nondecreasing n=1e6 / weighted mode by Floyd replace_top") { return weighted_nondecreasing<weighted_by_floyd>(a, w); };
	BENCHMARK("weighted hinges n=1e6 / library, weighted mode") { return weighted_hinges<weighted_mode>(a, w); };
	BENCHMARK("weighted hinges n=1e6 / weighted mode by Floyd replace_top") { return weighted_hinges<weighted_by_floyd>(a, w); };
	BENCHMARK("segments at rightmost slope m=1e4 x2000 / library, unit mode") { return unit_segments_at_rightmost<unit_mode>(a_few, 2000); };
	BENCHMARK("segments at rightmost slope m=1e4 x2000 / unit mode by whole-heap scan") { return unit_segments_at_rightmost<unit_by_scan>(a_few, 2000); };
	BENCHMARK("segments at rightmost slope m=1e4 x2000 / library, weighted mode") { return segments_at_rightmost<weighted_mode>(a_few, w_few, 2000); };
	BENCHMARK("segments at rightmost slope m=1e4 x2000 / weighted mode by whole-heap scan") { return segments_at_rightmost<weighted_by_scan>(a_few, w_few, 2000); };
}
