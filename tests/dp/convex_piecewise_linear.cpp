#include <atl/dp/convex_piecewise_linear.hpp>
#include <atl/experimental/dp/convex_piecewise_linear_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::int64;

namespace {
	/**
	 * @brief The certifying oracle of the slope-trick family (the library
	 *        class and the experimental contenders): a function on the
	 *        integers kept as its values on `[-width, width]` and extended
	 *        linearly past both ends. Exact for a convex piecewise-linear
	 *        function whose breakpoints stay well inside.
	 */
	struct explicit_function {
		static constexpr int64 width = 600;
		std::vector<int64> values = std::vector<int64>(2 * width + 1, 0);

		[[nodiscard]] int64 operator()(int64 x) const {
			if (x < -width) return values[0] + (values[0] - values[1]) * (-width - x);
			if (x > width) return values[2 * width] + (values[2 * width] - values[2 * width - 1]) * (x - width);
			return values[std::size_t(x + width)];
		}

		/** @brief The slope on `[x, x + 1]`. */
		[[nodiscard]] int64 slope(int64 x) const { return (*this)(x + 1) - (*this)(x); }

		template<std::regular_invocable<int64> Map>
		void rebuild(Map const &map) {
			std::vector<int64> next(values.size());
			for (int64 x = -width; x <= width; ++x) next[std::size_t(x + width)] = map(x);
			values = std::move(next);
		}
	};

	/** @brief The vocabulary every engine of the family shares, on `int64` positions. */
	template<typename Function>
	concept slope_trick_engine = std::default_initializable<Function> && requires(Function f, Function const cf, int64 x) {
		f.add_absolute(x);
		f.add_constant(x);
		f.translate(x);
		f.take_prefix_minimum();
		f.take_suffix_minimum();
		f.take_window_minimum(x, x);
		f.clip_slopes(-x, x);
		f.insert_segment(x, x);
		f.take_breakpoints();
		{ cf.minimum() } -> std::convertible_to<int64>;
		{ cf.leftmost_slope() } -> std::convertible_to<int64>;
		{ cf.rightmost_slope() } -> std::convertible_to<int64>;
		{ cf(x) } -> std::convertible_to<int64>;
		cf.minimizers();
	};

	/** @brief Whether `Function` stores unit breakpoints (its hinges take no weight). */
	template<typename Function>
	constexpr bool is_unit_engine = requires { requires Function::is_unit; };

	/** @brief `clamp(p, low, high)` with an absent `p` read as minus infinity. */
	int64 clamp_crossing(std::optional<int64> const &p, int64 low, int64 high) {
		return p ? std::clamp(*p, low, high) : low;
	}

	/** @brief Every operation of the shared vocabulary against the oracle, one random operation sequence per seed. */
	template<slope_trick_engine Function>
	void check_against_oracle(unsigned seed) {
		std::mt19937_64 rng(seed);
		auto uniform = [&](int64 lo, int64 hi) { return std::uniform_int_distribution<int64>(lo, hi)(rng); };
		Function f;
		explicit_function g;
		int const steps = int(uniform(1, 40));
		for (int step = 0; step < steps; ++step) {
			explicit_function const input = g;
			auto const input_minimizers = f.minimizers();
			int64 const a = uniform(-20, 20);
			// mostly unit weights, sometimes heavier or zero
			int64 const w = is_unit_engine<Function> || uniform(0, 3) != 0 ? 1 : uniform(0, 4);
			int64 const kind = uniform(0, 11);
			INFO("seed=" << seed << " step=" << step << " kind=" << kind);
			if (kind == 0) {
				if constexpr (is_unit_engine<Function>)
					f.add_rising_hinge(a);
				else
					f.add_rising_hinge(a, w);
				g.rebuild([&](int64 x) { return input(x) + w * std::max<int64>(0, x - a); });
			} else if (kind == 1) {
				if constexpr (is_unit_engine<Function>)
					f.add_falling_hinge(a);
				else
					f.add_falling_hinge(a, w);
				g.rebuild([&](int64 x) { return input(x) + w * std::max<int64>(0, a - x); });
			} else if (kind == 2) {
				if constexpr (is_unit_engine<Function>)
					f.add_absolute(a);
				else
					f.add_absolute(a, w);
				g.rebuild([&](int64 x) { return input(x) + w * (x < a ? a - x : x - a); });
			} else if (kind == 3) {
				f.add_absolute(a); // the default weight
				g.rebuild([&](int64 x) { return input(x) + (x < a ? a - x : x - a); });
			} else if (kind == 4) {
				f.add_constant(a);
				g.rebuild([&](int64 x) { return input(x) + a; });
			} else if (kind == 5) {
				// the slope far left is <= 0, so no y left of the table beats it
				auto const range = f.take_prefix_minimum();
				for (std::size_t x = 1; x < g.values.size(); ++x) g.values[x] = std::min(g.values[x], g.values[x - 1]);
				REQUIRE(!range.lower);
				REQUIRE(range.upper == input_minimizers.upper);
				for (int64 x = -100; x <= 100; x += 3) {
					REQUIRE(range.clamp(x) <= x);
					REQUIRE(input(range.clamp(x)) == g(x));
				}
			} else if (kind == 6) {
				auto const range = f.take_suffix_minimum();
				for (std::size_t x = g.values.size() - 1; x-- > 0;)
					g.values[x] = std::min(g.values[x], g.values[x + 1]);
				REQUIRE(!range.upper);
				REQUIRE(range.lower == input_minimizers.lower);
				for (int64 x = -100; x <= 100; x += 3) {
					REQUIRE(range.clamp(x) >= x);
					REQUIRE(input(range.clamp(x)) == g(x));
				}
			} else if (kind == 7) {
				int64 const d = uniform(-3, 3);
				f.translate(d);
				g.rebuild([&](int64 x) { return input(x - d); });
			} else if (kind == 8) {
				int64 const lo = uniform(-4, 4), hi = lo + uniform(0, 5);
				auto const crossing = f.take_window_minimum(lo, hi);
				g.rebuild([&](int64 x) {
					int64 best = input(x - lo);
					for (int64 d = lo + 1; d <= hi; ++d) best = std::min(best, input(x - d));
					return best;
				});
				// the smallest minimizer is the documented choice among the tied ones
				REQUIRE(crossing == input_minimizers.lower);
				for (int64 x = -100; x <= 100; x += 3) REQUIRE(input(clamp_crossing(crossing, x - hi, x - lo)) == g(x));
			} else if (kind == 9) {
				int64 const leftmost = f.leftmost_slope(), rightmost = f.rightmost_slope();
				int64 const low = uniform(leftmost - 2, 0), high = uniform(0, rightmost + 2);
				auto const range = f.clip_slopes(low, high);
				auto const kernel = [&](int64 t) { return t >= 0 ? high * t : low * t; };
				g.rebuild([&](int64 x) {
					int64 best = std::numeric_limits<int64>::max();
					for (int64 y = -explicit_function::width; y <= explicit_function::width; ++y)
						best = std::min(best, input(y) + kernel(x - y));
					return best;
				});
				REQUIRE(range.lower.has_value() == (leftmost < low));
				REQUIRE(range.upper.has_value() == (high < rightmost));
				for (int64 x = -100; x <= 100; x += 3) {
					int64 const y = range.clamp(x);
					REQUIRE(input(y) + kernel(x - y) == g(x));
				}
			} else {
				int64 const leftmost = f.leftmost_slope(), rightmost = f.rightmost_slope();
				int64 const slope = uniform(leftmost - 1, rightmost), length = uniform(0, 4);
				auto const crossing = f.insert_segment(slope, length);
				g.rebuild([&](int64 x) {
					int64 best = std::numeric_limits<int64>::max();
					for (int64 t = 0; t <= length; ++t) best = std::min(best, input(x - t) + slope * t);
					return best;
				});
				REQUIRE(crossing.has_value() == (leftmost < slope));
				if (crossing) {
					REQUIRE(input.slope(*crossing - 1) <= slope);
					REQUIRE(input.slope(*crossing) >= slope);
					// of the tied points, the one nearest the minimum
					if (slope == 0)
						REQUIRE(crossing == input_minimizers.lower);
					else if (slope > 0)
						REQUIRE(input.slope(*crossing - 1) < slope);
					else
						REQUIRE(input.slope(*crossing) > slope);
				}
				for (int64 x = -100; x <= 100; x += 3) {
					int64 const y = clamp_crossing(crossing, x - length, x);
					REQUIRE(input(y) + slope * (x - y) == g(x));
				}
			}
			int64 const width = explicit_function::width;
			REQUIRE(f.leftmost_slope() == g.slope(-width));
			REQUIRE(f.rightmost_slope() == g.slope(width - 1));
			int64 const minimum = *std::ranges::min_element(g.values);
			REQUIRE(f.minimum() == minimum);
			auto const range = f.minimizers();
			if (range.lower) {
				REQUIRE(g(*range.lower) == minimum);
				REQUIRE(g(*range.lower - 1) > minimum);
			} else {
				REQUIRE(g(-width) == minimum);
			}
			if (range.upper) {
				REQUIRE(g(*range.upper) == minimum);
				REQUIRE(g(*range.upper + 1) > minimum);
			} else {
				REQUIRE(g(width) == minimum);
			}
		}
		bool is_exact = true;
		for (int64 x = -300; x <= 300; ++x) is_exact &= f(x) == g(x);
		REQUIRE(is_exact);
		// the breakpoints describe the oracle's slopes; then f is the zero function
		int64 const leftmost = f.leftmost_slope(), rightmost = f.rightmost_slope();
		auto const breakpoints = f.take_breakpoints();
		std::size_t next = 0;
		int64 slope = leftmost, total_weight = 0;
		for (std::size_t i = 0; i < breakpoints.size(); ++i) {
			REQUIRE(breakpoints[i].weight > 0);
			total_weight += breakpoints[i].weight;
			if (i > 0) REQUIRE(breakpoints[i - 1].position < breakpoints[i].position);
		}
		REQUIRE(total_weight == rightmost - leftmost);
		for (int64 x = -explicit_function::width + 5; x < explicit_function::width - 5; ++x) {
			while (next < breakpoints.size() && breakpoints[next].position <= x) slope += breakpoints[next++].weight;
			REQUIRE(slope == g.slope(x));
		}
		REQUIRE(f.minimum() == 0);
		REQUIRE(f.leftmost_slope() == 0);
		REQUIRE(f.rightmost_slope() == 0);
		REQUIRE(f(17) == 0);
		REQUIRE(f.take_breakpoints().empty());
	}
}

TEST_CASE("convex_piecewise_linear matches an explicit function", "[convex_piecewise_linear]") {
	static_assert(is_unit_engine<atl::convex_piecewise_linear<int64>>);
	static_assert(!is_unit_engine<atl::convex_piecewise_linear<int64, int64>>);
	for (unsigned seed = 0; seed < 800; ++seed) check_against_oracle<atl::convex_piecewise_linear<int64>>(seed);
}

TEST_CASE("convex_piecewise_linear, weighted, matches an explicit function", "[convex_piecewise_linear]") {
	for (unsigned seed = 0; seed < 800; ++seed) check_against_oracle<atl::convex_piecewise_linear<int64, int64>>(seed);
}

TEST_CASE("convex_piecewise_linear contenders match an explicit function", "[convex_piecewise_linear]") {
	for (unsigned seed = 0; seed < 800; ++seed)
		check_against_oracle<atl::experimental::convex_piecewise_linear_by_breakpoint_pool<int64>>(seed);
	for (unsigned seed = 0; seed < 200; ++seed) {
		check_against_oracle<atl::experimental::convex_piecewise_linear_by_whole_heap_scan<int64>>(seed);
		check_against_oracle<atl::experimental::convex_piecewise_linear_by_floyd_replace_top<int64, int64>>(seed);
	}
}

TEST_CASE("convex_piecewise_linear: make a sequence nondecreasing, weighted", "[convex_piecewise_linear]") {
	// min sum w_i |b_i - a_i| over nondecreasing b: dp_i(x) = prefix-min of
	// dp_{i-1} plus w_i |x - a_i|, checked against the O(n * V) table DP
	std::mt19937_64 rng(42);
	for (int iter = 0; iter < 400; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 15);
		bool const is_unit = iter % 2 == 0;
		atl::convex_piecewise_linear<int64, int64> f;
		atl::experimental::convex_piecewise_linear_by_breakpoint_pool<int64> h;
		atl::convex_piecewise_linear<int64> u;
		std::vector<int64> dp(12, 0);
		for (std::size_t i = 0; i < n; ++i) {
			int64 const a = int64(rng() % 12), w = is_unit ? 1 : int64(rng() % 5);
			f.take_prefix_minimum();
			f.add_absolute(a, w);
			h.take_prefix_minimum();
			h.add_absolute(a, w);
			if (is_unit) {
				u.take_prefix_minimum();
				u.add_absolute(a);
			}
			int64 running = dp[0];
			for (int64 x = 0; x < 12; ++x) {
				running = std::min(running, dp[std::size_t(x)]);
				dp[std::size_t(x)] = running + w * (x < a ? a - x : x - a);
			}
		}
		CHECK(f.minimum() == *std::ranges::min_element(dp));
		CHECK(h.minimum() == f.minimum());
		if (is_unit) CHECK(u.minimum() == f.minimum());
	}
}

TEST_CASE("convex_piecewise_linear edge cases", "[convex_piecewise_linear]") {
	// the empty function: zero everywhere, unbounded minimizers, every operation a no-op
	atl::convex_piecewise_linear<int64> f;
	CHECK(f.minimum() == 0);
	CHECK(!f.minimizers().lower);
	CHECK(!f.minimizers().upper);
	CHECK(f(12345) == 0);
	CHECK(!f.take_prefix_minimum().upper);
	CHECK(!f.take_suffix_minimum().lower);
	CHECK(!f.take_window_minimum(-5, 5));
	auto const empty_clip = f.clip_slopes(-3, 3);
	CHECK((!empty_clip.lower && !empty_clip.upper));
	auto const flattened = f.clip_slopes(0, 0);
	CHECK((!flattened.lower && !flattened.upper));
	CHECK(!f.insert_segment(0, 4));
	CHECK(f.take_breakpoints().empty());
	f.translate(7);
	CHECK(f(-3) == 0);
	f.add_absolute(3);
	f.add_absolute(3);
	CHECK(f.leftmost_slope() == -2);
	CHECK(f.rightmost_slope() == 2);
	CHECK(f(0) == 6);
	CHECK(*f.minimizers().lower == 3);
	CHECK(*f.minimizers().upper == 3);
	// the window widens the bottom to [3 + 2, 3 + 6]
	CHECK(f.take_window_minimum(2, 6) == 3);
	CHECK(*f.minimizers().lower == 5);
	CHECK(*f.minimizers().upper == 9);
	CHECK(f(3) == 4);
	CHECK(f(11) == 4);
	CHECK(f.take_suffix_minimum().lower == 5);
	CHECK(!f.minimizers().lower);
	CHECK(f(-100) == 0);
	// a weighted hinge moves the minimum across several breakpoints at once
	atl::convex_piecewise_linear<int64, int64> h;
	h.add_falling_hinge(10);
	h.add_falling_hinge(20);
	h.add_falling_hinge(30);  // slopes -3, -2, -1, 0 past 30
	h.add_rising_hinge(0, 2); // + 2 max(0, x)
	CHECK(h.minimum() == 30 + 20 - 0 - 0);
	CHECK(h.minimizers().lower == 10);
	CHECK(h.minimizers().upper == 20);
	CHECK(h(15) == 50);
	CHECK(h(20) == 50);
	CHECK(h(25) == 55);
	// zero weights change nothing
	h.add_absolute(-7, 0);
	CHECK(h.minimum() == 50);
	CHECK(h.leftmost_slope() == -3);
	// large coordinates: no overflow with breakpoints near +-2^61
	atl::convex_piecewise_linear<int64> g;
	int64 const big = int64(1) << 61;
	g.add_rising_hinge(big);
	g.add_falling_hinge(-big);
	CHECK(g(0) == 0);
	CHECK(g(big + 5) == 5);
	CHECK(g(-big - 5) == 5);
}

TEST_CASE("convex_piecewise_linear: ties between optimal arguments", "[convex_piecewise_linear]") {
	// |x - 3| + |x - 7|: minimum 4 on [3, 7], slopes -2, 0, 2
	auto const make = [] {
		atl::convex_piecewise_linear<int64> f;
		f.add_absolute(3);
		f.add_absolute(7);
		return f;
	};
	auto f = make();
	CHECK(f.take_window_minimum(0, 2) == 3); // the smallest minimizer
	f = make();
	CHECK(f.insert_segment(0, 5) == 3);
	f = make();
	CHECK(f.take_prefix_minimum().upper == 7); // the largest minimizer
	f = make();
	CHECK(f.take_suffix_minimum().lower == 3);
	// a stretch of slope exactly s qualifies whole: p is its end nearest the minimum
	atl::convex_piecewise_linear<int64, int64> h;
	h.add_absolute(3, 1);
	h.add_rising_hinge(7, 1); // slopes -1, 1 on [3, 7], 2
	CHECK(h.insert_segment(1, 4) == 3);
	h = {};
	h.add_absolute(3, 1);
	h.add_falling_hinge(-2, 1); // slopes -2, -1 on [-2, 3], 1
	CHECK(h.insert_segment(-1, 4) == 3);
	CHECK(h.minimum() == -4);
}

TEST_CASE("convex_piecewise_linear at the bounds of its value and slope types", "[convex_piecewise_linear]") {
	// positions and values at the edge of int64: every documented quantity fits
	constexpr int64 top = std::numeric_limits<int64>::max(), bottom = std::numeric_limits<int64>::lowest();
	atl::convex_piecewise_linear<int64> f;
	f.add_rising_hinge(top - 10);
	f.add_falling_hinge(bottom + 10);
	CHECK(f(top) == 10);
	CHECK(f(bottom) == 10);
	CHECK(f(0) == 0);
	CHECK(f.minimizers().lower == bottom + 10);
	CHECK(f.minimizers().upper == top - 10);
	// weights summing to the largest int8 slope
	atl::convex_piecewise_linear<int64, atl::int8> w;
	w.add_rising_hinge(0, 100);
	w.add_rising_hinge(5, 27);
	w.add_falling_hinge(0, 127);
	CHECK(w.rightmost_slope() == 127);
	CHECK(w.leftmost_slope() == -127);
	CHECK(w(10) == 1000 + 27 * 5);
	CHECK(w(-1) == 127);
	auto const clipped = w.clip_slopes(-127, 100);
	CHECK(!clipped.lower);
	CHECK(clipped.upper == 5);
	CHECK(w(10) == 1000);
	// int128 values past the int64 range
	atl::convex_piecewise_linear<atl::int128, int64> wide;
	atl::int128 const far_point = atl::int128(top) * 4;
	wide.add_absolute(far_point, 3);
	CHECK(wide(0) == far_point * 3);
	CHECK(wide.minimizers().lower == far_point);
}

TEST_CASE("convex_piecewise_linear over double", "[convex_piecewise_linear]") {
	atl::convex_piecewise_linear<double, int64> f;
	f.add_absolute(0.5);
	f.add_absolute(2.5, 2);
	CHECK(f.minimum() == 2.0);
	CHECK(*f.minimizers().lower == 2.5);
	CHECK(*f.minimizers().upper == 2.5);
	f.translate(0.25);
	CHECK(f(0.0) == 0.75 + 2 * 2.75);
	auto const range = f.clip_slopes(-1, 1);
	CHECK(range.lower == 0.75);
	CHECK(range.upper == 2.75);
}

TEST_CASE("convex_piecewise_linear preconditions", "[convex_piecewise_linear]") {
	atl::convex_piecewise_linear<int64, int64> f;
	f.add_absolute(0, 2);
	CHECK(atl_test::aborts([&] { f.take_window_minimum(3, 2); }));
	CHECK(atl_test::aborts([&] { f.add_rising_hinge(1, -1); }));
	CHECK(atl_test::aborts([&] { f.clip_slopes(1, 2); }));
	CHECK(atl_test::aborts([&] { f.clip_slopes(-2, -1); }));
	CHECK(atl_test::aborts([&] { f.insert_segment(0, -1); }));
	CHECK(atl_test::aborts([&] { f.insert_segment(3, 1); }));
}

TEST_CASE("convex_piecewise_linear_by_priority_queue and by_unit_heaps agree with the library", "[convex_piecewise_linear]") {
	std::mt19937_64 rng(43);
	for (int iter = 0; iter < 200; ++iter) {
		atl::convex_piecewise_linear<int64> f;
		atl::experimental::convex_piecewise_linear_by_priority_queue<int64> g;
		atl::experimental::convex_piecewise_linear_by_unit_heaps<int64> h;
		for (int step = 0; step < 200; ++step) {
			int64 const a = int64(rng() % 1001) - 500;
			switch (rng() % 7) {
			case 0:
				f.add_rising_hinge(a), g.add_rising_hinge(a), h.add_rising_hinge(a);
				break;
			case 1:
				f.add_falling_hinge(a), g.add_falling_hinge(a), h.add_falling_hinge(a);
				break;
			case 2:
				f.add_absolute(a), g.add_absolute(a), h.add_absolute(a);
				break;
			case 3:
				f.take_prefix_minimum(), g.take_prefix_minimum(), h.take_prefix_minimum();
				break;
			case 4:
				f.take_suffix_minimum(), g.take_suffix_minimum(), h.take_suffix_minimum();
				break;
			case 5:
				f.translate(a % 7), g.translate(a % 7), h.translate(a % 7);
				break;
			default:
				f.take_window_minimum(-(a & 3), a & 7), g.take_window_minimum(-(a & 3), a & 7),
					h.take_window_minimum(-(a & 3), a & 7);
				break;
			}
			auto const x = f.minimizers(), y = g.minimizers(), z = h.minimizers();
			REQUIRE(f.minimum() == g.minimum());
			REQUIRE(f.minimum() == h.minimum());
			REQUIRE(x.lower == y.lower);
			REQUIRE(x.upper == y.upper);
			REQUIRE(x.lower == z.lower);
			REQUIRE(x.upper == z.upper);
		}
	}
}

TEST_CASE("convex_piecewise_linear_by_breakpoint_pool agrees with the library on long runs", "[convex_piecewise_linear]") {
	// wide coordinates, weights and every operation: the minimum and the minimizers agree throughout
	std::mt19937_64 rng(44);
	for (int iter = 0; iter < 100; ++iter) {
		atl::convex_piecewise_linear<int64, int64> f;
		atl::experimental::convex_piecewise_linear_by_breakpoint_pool<int64> g;
		for (int step = 0; step < 400; ++step) {
			int64 const a = int64(rng() % 2001) - 1000, w = 1 + int64(rng() % 5);
			switch (rng() % 9) {
			case 0:
				f.add_rising_hinge(a, w), g.add_rising_hinge(a, w);
				break;
			case 1:
				f.add_falling_hinge(a, w), g.add_falling_hinge(a, w);
				break;
			case 2:
				f.add_absolute(a, w), g.add_absolute(a, w);
				break;
			case 3:
				REQUIRE(f.take_prefix_minimum().upper == g.take_prefix_minimum().upper);
				break;
			case 4:
				REQUIRE(f.take_suffix_minimum().lower == g.take_suffix_minimum().lower);
				break;
			case 5:
				REQUIRE(f.take_window_minimum(-(a & 3), a & 7) == g.take_window_minimum(-(a & 3), a & 7));
				break;
			case 6: {
				int64 const low = -int64(rng() % 30), high = int64(rng() % 30);
				auto const x = f.clip_slopes(low, high), y = g.clip_slopes(low, high);
				REQUIRE(x.lower == y.lower);
				REQUIRE(x.upper == y.upper);
				break;
			}
			case 7: {
				int64 const slope = f.leftmost_slope() + int64(rng() % std::uint64_t(f.rightmost_slope() - f.leftmost_slope() + 2)) - 1;
				auto const x = f.insert_segment(slope, a & 15);
				auto const y = g.insert_segment(slope, a & 15);
				REQUIRE(x == y);
				break;
			}
			default:
				f.translate(a % 7), g.translate(a % 7);
				break;
			}
			auto const x = f.minimizers(), y = g.minimizers();
			REQUIRE(f.minimum() == g.minimum());
			REQUIRE(f.leftmost_slope() == g.leftmost_slope());
			REQUIRE(f.rightmost_slope() == g.rightmost_slope());
			REQUIRE(x.lower == y.lower);
			REQUIRE(x.upper == y.upper);
		}
		int64 const probe = int64(rng() % 2001) - 1000;
		REQUIRE(f(probe) == g(probe));
		auto const p = f.take_breakpoints(), q = g.take_breakpoints();
		REQUIRE(p.size() == q.size());
		for (std::size_t i = 0; i < p.size(); ++i) {
			REQUIRE(p[i].position == q[i].position);
			REQUIRE(p[i].weight == q[i].weight);
		}
	}
}

namespace {
	/** @brief A hinge on `f`: unit (no weight argument) or of weight `w`. */
	template<slope_trick_engine Function>
	void add_hinge(Function &f, int kind, int64 a, int64 w) {
		if constexpr (Function::is_unit) {
			if (kind == 0)
				f.add_rising_hinge(a);
			else if (kind == 1)
				f.add_falling_hinge(a);
			else
				f.add_absolute(a);
		} else {
			if (kind == 0)
				f.add_rising_hinge(a, w);
			else if (kind == 1)
				f.add_falling_hinge(a, w);
			else
				f.add_absolute(a, w);
		}
	}

	/** @brief Drive the library class and a configuration contender through the same random operations; every observable agrees. */
	template<atl::piecewise_linear_slope Slope, bool UsesFloydReplaceTop>
	void check_configuration_agrees(std::uint64_t seed, int steps, int64 spread) {
		using library = atl::convex_piecewise_linear<int64, Slope>;
		using contender = atl::experimental::convex_piecewise_linear_by_configuration<int64, Slope, UsesFloydReplaceTop>;
		std::mt19937_64 rng(seed);
		library f;
		contender g;
		for (int step = 0; step < steps; ++step) {
			int64 const a = int64(rng() % std::uint64_t(2 * spread + 1)) - spread, w = 1 + int64(rng() % 5);
			switch (int const kind = int(rng() % 8); kind) {
			case 0:
			case 1:
			case 2:
				add_hinge(f, kind, a, w);
				add_hinge(g, kind, a, w);
				break;
			case 3:
				REQUIRE(f.take_prefix_minimum().upper == g.take_prefix_minimum().upper);
				break;
			case 4:
				REQUIRE(f.take_window_minimum(-(a & 3), a & 7) == g.take_window_minimum(-(a & 3), a & 7));
				break;
			case 5: {
				// half the time exactly the rightmost slope (the farthest-breakpoint path)
				int64 const span = f.rightmost_slope() - f.leftmost_slope();
				int64 const slope = (rng() & 1) ? f.rightmost_slope() : f.leftmost_slope() + int64(rng() % std::uint64_t(span + 2)) - 1;
				REQUIRE(f.insert_segment(slope, a & 15) == g.insert_segment(slope, a & 15));
				break;
			}
			case 6:
				if constexpr (!library::is_unit) {
					int64 const low = -int64(rng() % 30), high = int64(rng() % 30);
					auto const x = f.clip_slopes(low, high), y = g.clip_slopes(low, high);
					REQUIRE(x.lower == y.lower);
					REQUIRE(x.upper == y.upper);
				}
				break;
			default:
				f.translate(a % 7), g.translate(a % 7);
				break;
			}
			REQUIRE(f.minimum() == g.minimum());
			REQUIRE(f.leftmost_slope() == g.leftmost_slope());
			REQUIRE(f.rightmost_slope() == g.rightmost_slope());
			REQUIRE(f.minimizers().lower == g.minimizers().lower);
			REQUIRE(f.minimizers().upper == g.minimizers().upper);
		}
		REQUIRE(f(spread / 3) == g(spread / 3));
		auto const p = f.take_breakpoints(), q = g.take_breakpoints();
		REQUIRE(p.size() == q.size());
		for (std::size_t i = 0; i < p.size(); ++i) REQUIRE((p[i].position == q[i].position && p[i].weight == q[i].weight));
	}
}

TEST_CASE("convex_piecewise_linear_by_configuration agrees with the library", "[convex_piecewise_linear]") {
	// narrow spreads force ties and tiny heaps, wide ones deep heaps; both replace_top forms, both modes
	for (std::uint64_t seed = 1; seed <= 60; ++seed)
		for (int64 const spread : {2, 30, 1000}) {
			int const steps = seed % 4 == 0 ? 3 : 250;
			check_configuration_agrees<atl::unit_weight, true>(seed, steps, spread);
			check_configuration_agrees<atl::unit_weight, false>(seed, steps, spread);
			check_configuration_agrees<int64, true>(seed, steps, spread);
			check_configuration_agrees<int64, false>(seed, steps, spread);
		}
}

TEST_CASE("convex_piecewise_linear: rightmost lookup at edge heap sizes", "[convex_piecewise_linear]") {
	auto const check = []<atl::piecewise_linear_slope Slope>() {
		for (int64 n : {0, 1, 2, 3, 4, 7, 8, 9, 19'999, 20'000, 20'001}) {
			atl::convex_piecewise_linear<int64, Slope> f;
			for (int64 i = 0; i < n; ++i) {
				if constexpr (decltype(f)::is_unit)
					f.add_rising_hinge(i / 2);
				else
					f.add_rising_hinge(i / 2, 2);
			}
			f.translate(17);
			auto const value_at_30 = f(30);
			auto const crossing = f.insert_segment(f.rightmost_slope(), 9);
			if (n == 0)
				CHECK(!crossing);
			else
				CHECK(crossing == (n - 1) / 2 + 17);
			CHECK(f(30) == value_at_30);
		}
	};
	check.template operator()<atl::unit_weight>();
	check.template operator()<int64>();
	atl::convex_piecewise_linear<double> other_type;
	other_type.add_rising_hinge(3.5);
	CHECK(other_type.insert_segment(1, 2.0) == 3.5);
}

namespace {
	/** @brief The engines with the whole vocabulary, unit and weighted, on `Value` positions. */
	template<typename Value>
	using full_engines = std::tuple<atl::convex_piecewise_linear<Value>, atl::convex_piecewise_linear<Value, int64>,
									atl::experimental::convex_piecewise_linear_by_configuration<Value, atl::unit_weight, true>,
									atl::experimental::convex_piecewise_linear_by_configuration<Value, int64, false>,
									atl::experimental::convex_piecewise_linear_by_breakpoint_pool<Value, int64>>;

	/** @brief The unit engines with only the hinge, clip and read interface. */
	template<typename Value>
	using unit_only_engines = std::tuple<atl::experimental::convex_piecewise_linear_by_priority_queue<Value>,
										 atl::experimental::convex_piecewise_linear_by_unit_heaps<Value>>;

	/** @brief `check(Function{})` for every type of the tuple `Engines`. */
	template<typename Engines, typename Check>
	void for_each_engine(Check const &check) {
		[&]<std::size_t... I>(std::index_sequence<I...>) {
			(check(std::tuple_element_t<I, Engines>{}), ...);
		}(std::make_index_sequence<std::tuple_size_v<Engines>>{});
	}
}

TEST_CASE("convex_piecewise_linear forms distances, products and sums wide", "[convex_piecewise_linear]") {
	constexpr int64 top = std::numeric_limits<int64>::max(), bottom = std::numeric_limits<int64>::lowest();
	// f(0) = INT64_MIN + (0 - INT64_MIN) = 0: the distance leaves int64, f(0) does not
	for_each_engine<full_engines<int64>>([&](auto f) {
		f.add_constant(bottom);
		f.add_rising_hinge(bottom);
		CHECK(f(0) == 0);
		CHECK(f(bottom) == bottom);
	});
	// a hinge crossing the minimum: INT64_MIN + (INT64_MAX - (-1)) = 0
	auto const crossing = [&](auto f) {
		f.add_constant(bottom);
		f.add_falling_hinge(top);
		f.add_rising_hinge(-1);
		CHECK(f.minimum() == 0);
		CHECK(f.minimizers().lower == -1);
		CHECK(f.minimizers().upper == top);
	};
	for_each_engine<full_engines<int64>>(crossing);
	for_each_engine<unit_only_engines<int64>>(crossing);
	// the falling hinge's mirror image: INT64_MIN + (1 - INT64_MIN) leaves int64 on the way
	auto const falling = [&](auto f) {
		f.add_constant(bottom + 1);
		f.add_rising_hinge(bottom + 1);
		f.add_falling_hinge(1);
		CHECK(f.minimum() == 1);
	};
	for_each_engine<full_engines<int64>>(falling);
	for_each_engine<unit_only_engines<int64>>(falling);
	// a segment below every slope: INT64_MAX + (-INT64_MAX) * 2 = -INT64_MAX
	for_each_engine<full_engines<int64>>([&](auto f) {
		f.add_constant(top);
		CHECK(f.insert_segment(-top, 2) == std::nullopt);
		CHECK(f.minimum() == -top);
	});
	// a segment through the left side: INT64_MAX + -(INT64_MAX - 1) * 2 = 2 - INT64_MAX
	auto const through_left = [&](auto f) {
		f.add_constant(top);
		f.add_falling_hinge(0, top);
		CHECK(f.insert_segment(-(top - 1), 2) == 0);
		CHECK(f.minimum() == 2 - top);
		CHECK(f(2) == 2 - top);
	};
	through_left(atl::convex_piecewise_linear<int64, int64>{});
	through_left(atl::experimental::convex_piecewise_linear_by_configuration<int64, int64, true>{});
	through_left(atl::experimental::convex_piecewise_linear_by_breakpoint_pool<int64, int64>{});
	// int16 values promote through int32: 30000 - (-30000) leaves int16
	atl::convex_piecewise_linear<atl::int16, atl::int16> narrow;
	narrow.add_constant(atl::int16(-30000));
	narrow.add_falling_hinge(atl::int16(30000));
	narrow.add_rising_hinge(atl::int16(-30000));
	CHECK(narrow.minimum() == atl::int16(30000));
	CHECK(narrow(atl::int16(0)) == atl::int16(30000));
}

TEST_CASE("convex_piecewise_linear reads a coordinate argument before changing", "[convex_piecewise_linear]") {
	// a == f.minimum() refers to the state the hinge changes: it must act as the value at the call
	auto const rising = [](auto f) {
		f.add_falling_hinge(10);
		f.add_rising_hinge(f.minimum());
		CHECK(f.minimizers().lower == 0);
		CHECK(f.minimizers().upper == 10);
		CHECK(f.minimum() == 10);
	};
	for_each_engine<full_engines<int64>>(rising);
	for_each_engine<unit_only_engines<int64>>(rising);
	auto const absolute = [](auto f) {
		f.add_falling_hinge(10);
		f.add_absolute(f.minimum()); // |x - 0| on top of max(0, 10 - x)
		CHECK(f.minimizers().lower == 0);
		CHECK(f.minimizers().upper == 10);
		CHECK(f.minimum() == 10);
	};
	for_each_engine<full_engines<int64>>(absolute);
	for_each_engine<unit_only_engines<int64>>(absolute);
	for_each_engine<full_engines<int64>>([](auto f) {
		f.add_falling_hinge(10);
		f.add_constant(3);
		f.translate(1);
		// the length refers to the minimum, which the segment changes after reading it
		(void)f.insert_segment(-1, f.minimum());
		CHECK(f.minimum() == 0);
		CHECK(f.minimizers().lower == 14);
	});
	// weighted
	for_each_engine<full_engines<int64>>([](auto f) {
		if constexpr (!is_unit_engine<decltype(f)>) {
			f.add_falling_hinge(10, 2);
			f.add_absolute(f.minimum(), 3);
			CHECK(f.minimum() == 20);
			CHECK(f.minimizers().lower == 0);
			CHECK(f.minimizers().upper == 0);
		}
	});
}

TEST_CASE("convex_piecewise_linear: an emptied side forgets its offset", "[convex_piecewise_linear]") {
	// (1 - 1e20) + 1e20 rounds to 0: a stale offset would lose the new breakpoint
	auto const right = [](auto f) {
		f.translate(1e20);
		(void)f.take_prefix_minimum();
		f.add_rising_hinge(1.0);
		CHECK(f.minimizers().upper == 1.0);
	};
	auto const left = [](auto f) {
		f.translate(-1e20);
		(void)f.take_suffix_minimum();
		f.add_falling_hinge(2.0);
		CHECK(f.minimizers().lower == 2.0);
	};
	for_each_engine<full_engines<double>>(right);
	for_each_engine<unit_only_engines<double>>(right);
	for_each_engine<full_engines<double>>(left);
	for_each_engine<unit_only_engines<double>>(left);
}

TEST_CASE("convex_piecewise_linear asserts that combined breakpoint weights fit", "[convex_piecewise_linear]") {
	// side weights of 127 each fit int8, but the breakpoint at 0 carries 254
	auto const check = [](auto f) {
		f.add_absolute(0, 127);
		CHECK(f.rightmost_slope() == 127);
		CHECK(atl_test::aborts([&] { (void)f.take_breakpoints(); }));
	};
	check(atl::convex_piecewise_linear<int64, atl::int8>{});
	check(atl::experimental::convex_piecewise_linear_by_configuration<int64, atl::int8, true>{});
	check(atl::experimental::convex_piecewise_linear_by_breakpoint_pool<int64, atl::int8>{});
	// within the type, the weights combine
	atl::convex_piecewise_linear<int64, atl::int8> f;
	f.add_absolute(0, 63);
	auto const breakpoints = f.take_breakpoints();
	REQUIRE(breakpoints.size() == 1);
	CHECK(breakpoints[0].weight == 126);
}
