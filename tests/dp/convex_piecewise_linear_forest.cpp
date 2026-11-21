#include <atl/dp/convex_piecewise_linear_forest.hpp>
#include <atl/experimental/dp/convex_piecewise_linear_forest_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

using atl::int64;

namespace {
	/**
	 * @brief The oracle: a function on the integers kept as its values on
	 *        `[-width, width]`, extended linearly past both ends. Exact for a
	 *        convex piecewise-linear function whose breakpoints stay well
	 *        inside; values are known only up to the constants the forest
	 *        does not track, so only slopes are compared.
	 */
	struct explicit_function {
		static constexpr int64 width = 150;
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

	using forest_type = atl::convex_piecewise_linear_forest<int64, int64>;

	/** @brief Drain `function` and compare its slopes with the oracle's on the comparison window. */
	void check_shape(forest_type::function_reference function, explicit_function const &oracle) {
		int64 const leftmost = function.leftmost_slope(), rightmost = function.rightmost_slope();
		auto const breakpoints = function.take_breakpoints();
		int64 total = 0;
		for (std::size_t i = 0; i < breakpoints.size(); ++i) {
			REQUIRE(breakpoints[i].weight > 0);
			if (i > 0) REQUIRE(breakpoints[i - 1].position < breakpoints[i].position);
			REQUIRE(std::abs(breakpoints[i].position) < explicit_function::width - 10);
			total += breakpoints[i].weight;
		}
		REQUIRE(leftmost + total == rightmost);
		std::size_t next = 0;
		int64 slope = leftmost;
		for (int64 x = -explicit_function::width + 5; x < explicit_function::width - 5; ++x) {
			while (next < breakpoints.size() && breakpoints[next].position <= x) slope += breakpoints[next++].weight;
			REQUIRE(slope == oracle.slope(x));
		}
		REQUIRE(function.leftmost_slope() == 0);
		REQUIRE(function.rightmost_slope() == 0);
	}
}

TEST_CASE("convex_piecewise_linear_forest: random operations against explicit tables", "[dp][convex_piecewise_linear_forest]") {
	constexpr std::size_t functions = 4;
	auto const clamp_crossing = [](std::optional<int64> const &p, int64 low, int64 high) {
		return p ? std::clamp(*p, low, high) : low;
	};
	for (unsigned seed = 0; seed < 400; ++seed) {
		std::mt19937_64 rng(seed);
		auto uniform = [&](int64 lo, int64 hi) { return std::uniform_int_distribution<int64>(lo, hi)(rng); };
		forest_type forest(functions - 1);
		auto const added = forest.add_function(); // before id(functions - 1): that id is valid only now
		REQUIRE(added == forest.id(functions - 1));
		REQUIRE(forest.function_count() == functions);
		std::vector<explicit_function> oracle(functions);
		for (int step = 0; step < 30; ++step) {
			std::size_t const index = std::size_t(uniform(0, functions - 1));
			auto const f = forest[forest.id(index)];
			REQUIRE(f.id().index() == index);
			explicit_function const old = oracle[index];
			explicit_function &now = oracle[index];
			int64 const leftmost = f.leftmost_slope(), rightmost = f.rightmost_slope();
			int const kind = int(uniform(0, 11));
			INFO("seed=" << seed << " step=" << step << " kind=" << kind);
			if (kind <= 2) {
				int64 const a = uniform(-25, 25), w = uniform(0, 4);
				bool const is_default = w == 1 && uniform(0, 1) == 0;
				if (kind == 0) is_default ? f.add_rising_hinge(a) : f.add_rising_hinge(a, w);
				if (kind == 1) is_default ? f.add_falling_hinge(a) : f.add_falling_hinge(a, w);
				if (kind == 2) is_default ? f.add_absolute(a) : f.add_absolute(a, w);
				now.rebuild([&](int64 x) {
					int64 const up = x > a ? x - a : 0, down = x < a ? a - x : 0;
					return old(x) + w * (kind == 0 ? up : kind == 1 ? down
																	: up + down);
				});
			} else if (kind == 3) {
				int64 const d = uniform(-2, 2);
				f.translate(d);
				now.rebuild([&](int64 x) { return old(x - d); });
			} else if (kind == 4) {
				std::size_t const other = std::size_t(uniform(0, functions - 1));
				forest.merge(f.id(), forest.id(other));
				if (other != index) {
					explicit_function const melded = oracle[other];
					now.rebuild([&](int64 x) { return old(x) + melded(x); });
					oracle[other] = {};
				}
			} else if (kind == 5) {
				int64 const low = uniform(leftmost - 2, rightmost);
				int64 const high = uniform(std::max(low, leftmost), rightmost + 2);
				auto const range = f.clip_slopes(low, high);
				auto const kernel = [&](int64 t) { return t >= 0 ? high * t : low * t; };
				now.rebuild([&](int64 x) {
					int64 best = std::numeric_limits<int64>::max();
					for (int64 y = -explicit_function::width; y <= explicit_function::width; ++y)
						best = std::min(best, old(y) + kernel(x - y));
					return best;
				});
				REQUIRE(range.lower.has_value() == (leftmost < low));
				REQUIRE(range.upper.has_value() == (high < rightmost));
				for (int64 x = -100; x <= 100; x += 7) {
					int64 const y = range.clamp(x);
					REQUIRE(old(y) + kernel(x - y) == now(x));
				}
			} else if (kind == 6) {
				int64 const slope = uniform(leftmost - 1, rightmost), length = uniform(0, 4);
				auto const crossing = f.insert_segment(slope, length);
				now.rebuild([&](int64 x) {
					int64 best = std::numeric_limits<int64>::max();
					for (int64 t = 0; t <= length; ++t) best = std::min(best, old(x - t) + slope * t);
					return best;
				});
				REQUIRE(crossing.has_value() == (leftmost < slope));
				if (crossing) {
					REQUIRE(old.slope(*crossing - 1) <= slope);
					REQUIRE(old.slope(*crossing) >= slope);
				}
				for (int64 x = -100; x <= 100; x += 7) {
					int64 const y = clamp_crossing(crossing, x - length, x);
					REQUIRE(old(y) + slope * (x - y) == now(x));
				}
			} else if (kind == 7 && leftmost <= 0) {
				auto const range = f.take_prefix_minimum();
				for (std::size_t x = 1; x < now.values.size(); ++x)
					now.values[x] = std::min(now.values[x], now.values[x - 1]);
				REQUIRE(!range.lower);
				REQUIRE(range.upper.has_value() == (rightmost > 0));
				for (int64 x = -100; x <= 100; x += 7) {
					REQUIRE(range.clamp(x) <= x);
					REQUIRE(old(range.clamp(x)) == now(x));
				}
			} else if (kind == 8 && rightmost >= 0) {
				auto const range = f.take_suffix_minimum();
				for (std::size_t x = now.values.size() - 1; x-- > 0;)
					now.values[x] = std::min(now.values[x], now.values[x + 1]);
				REQUIRE(!range.upper);
				REQUIRE(range.lower.has_value() == (leftmost < 0));
				for (int64 x = -100; x <= 100; x += 7) {
					REQUIRE(range.clamp(x) >= x);
					REQUIRE(old(range.clamp(x)) == now(x));
				}
			} else if (kind == 9 && rightmost >= 0) {
				int64 const lo = uniform(-3, 3), hi = lo + uniform(0, 4);
				auto const crossing = f.take_window_minimum(lo, hi);
				now.rebuild([&](int64 x) {
					int64 best = old(x - lo);
					for (int64 d = lo + 1; d <= hi; ++d) best = std::min(best, old(x - d));
					return best;
				});
				REQUIRE(crossing.has_value() == (leftmost < 0));
				for (int64 x = -100; x <= 100; x += 7)
					REQUIRE(old(clamp_crossing(crossing, x - hi, x - lo)) == now(x));
			} else if (kind == 11) {
				// a point clip: every breakpoint goes, both ends exact
				int64 const slope = uniform(leftmost, rightmost);
				auto const range = f.clip_slopes(slope, slope);
				now.rebuild([&](int64 x) {
					int64 best = std::numeric_limits<int64>::max();
					for (int64 y = -explicit_function::width; y <= explicit_function::width; ++y)
						best = std::min(best, old(y) + slope * (x - y));
					return best;
				});
				REQUIRE(range.lower.has_value() == (leftmost < slope));
				REQUIRE(range.upper.has_value() == (slope < rightmost));
				if (range.lower) {
					REQUIRE(old.slope(*range.lower - 1) < slope);
					REQUIRE(old.slope(*range.lower) >= slope);
				}
				if (range.upper) {
					REQUIRE(old.slope(*range.upper - 1) <= slope);
					REQUIRE(old.slope(*range.upper) > slope);
				}
				REQUIRE(f.leftmost_slope() == slope);
				REQUIRE(f.rightmost_slope() == slope);
			} else if (kind == 10 && leftmost <= 0 && rightmost >= 0) {
				// non-destructive: the shape is checked again at the end
				auto const range = f.minimizers();
				int64 const width = explicit_function::width;
				int64 const minimum = *std::ranges::min_element(now.values);
				if (range.lower) {
					REQUIRE(now(*range.lower) == minimum);
					REQUIRE(now(*range.lower - 1) > minimum);
				} else {
					REQUIRE(now(-width) == minimum);
				}
				if (range.upper) {
					REQUIRE(now(*range.upper) == minimum);
					REQUIRE(now(*range.upper + 1) > minimum);
				} else {
					REQUIRE(now(width) == minimum);
				}
			}
		}
		for (std::size_t i = 0; i < functions; ++i) check_shape(forest[forest.id(i)], oracle[i]);
	}
}

TEST_CASE("convex_piecewise_linear_forest: clip_slopes(0, 0) gives the minimizers", "[dp][convex_piecewise_linear_forest]") {
	forest_type forest(1);
	auto const f = forest[forest.id(0)];
	f.add_absolute(3, 2); // 2|x - 3|
	f.add_rising_hinge(7);
	f.add_falling_hinge(5, 2); // slopes -4, 0 on [3, 5], 2, 3 past 7
	auto const nondestructive = f.minimizers();
	REQUIRE(nondestructive.lower == 3);
	REQUIRE(nondestructive.upper == 5);
	REQUIRE(f.leftmost_slope() == -4);
	auto const minimizers = f.clip_slopes(0, 0);
	REQUIRE(minimizers.lower == 3);
	REQUIRE(minimizers.upper == 5);
	REQUIRE(f.leftmost_slope() == 0);
	REQUIRE(f.rightmost_slope() == 0);
	REQUIRE(f.take_breakpoints().empty());
	CHECK(!f.minimizers().lower);
	CHECK(!f.minimizers().upper);
}

TEST_CASE("convex_piecewise_linear_forest: point clips, heavy on either side",
		  "[dp][convex_piecewise_linear_forest]") {
	// everything goes: nothing may reach a later function or a merge
	forest_type forest(2);
	auto const f = forest[forest.id(0)], g = forest[forest.id(1)];
	for (int64 a = 0; a < 50; ++a) f.add_falling_hinge(a);
	f.add_rising_hinge(100, 2);
	auto const range = f.clip_slopes(1, 1); // slopes -50 .. -1, 0 on [49, 100], then 2
	REQUIRE(range.lower == 100);
	REQUIRE(range.upper == 100);
	auto const h = forest[forest.add_function()];
	REQUIRE(h.take_breakpoints().empty());
	h.add_absolute(7);
	for (int64 a = 0; a < 50; ++a) g.add_rising_hinge(a);
	g.add_falling_hinge(-5, 3);
	auto const other = g.clip_slopes(-1, -1); // slope -3, 0 past -5, rising from 0
	REQUIRE(other.lower == -5);
	REQUIRE(other.upper == -5);
	forest.merge(h.id(), f.id());
	forest.merge(h.id(), g.id());
	REQUIRE(h.leftmost_slope() == 1 - 1 - 1);
	REQUIRE(h.rightmost_slope() == 1 + 1 - 1);
	auto const breakpoints = h.take_breakpoints();
	REQUIRE(breakpoints.size() == 1);
	REQUIRE(breakpoints[0].position == 7);
	REQUIRE(breakpoints[0].weight == 2);
	for (int i = 0; i < 10; ++i) REQUIRE(forest[forest.add_function()].take_breakpoints().empty());
}

TEST_CASE("convex_piecewise_linear_forest: minimizers from either end", "[dp][convex_piecewise_linear_forest]") {
	// the walk starts from the lighter end; both must agree with the shape
	forest_type forest(2);
	auto const heavy_right = forest[forest.id(0)], heavy_left = forest[forest.id(1)];
	heavy_right.add_falling_hinge(0); // slope -1 left of 0
	heavy_right.add_rising_hinge(4, 5);
	heavy_right.add_rising_hinge(9, 5);
	heavy_left.add_rising_hinge(10);
	heavy_left.add_falling_hinge(6, 5);
	heavy_left.add_falling_hinge(1, 5);
	for (int repeat = 0; repeat < 2; ++repeat) {
		auto const a = heavy_right.minimizers(), b = heavy_left.minimizers();
		CHECK(a.lower == 0);
		CHECK(a.upper == 4);
		CHECK(b.lower == 6);
		CHECK(b.upper == 10);
	}
	// a single point: the slope jumps across 0
	forest_type single(1);
	auto const g = single[single.id(0)];
	g.add_absolute(2, 3);
	g.add_absolute(-2);
	CHECK(g.minimizers().lower == 2);
	CHECK(g.minimizers().upper == 2);
	auto const breakpoints = g.take_breakpoints();
	REQUIRE(breakpoints.size() == 2);
}

TEST_CASE("convex_piecewise_linear_forest: one breakpoint eaten from both ends", "[dp][convex_piecewise_linear_forest]") {
	forest_type forest(2);
	auto const f = forest[forest.id(0)], g = forest[forest.id(1)];
	f.add_absolute(10, 5); // weight 10 at 10
	auto const range = f.clip_slopes(-2, 3);
	REQUIRE(range.lower == 10);
	REQUIRE(range.upper == 10);
	g.add_absolute(-4);
	forest.merge(g.id(), f.id());
	forest.merge(g.id(), g.id()); // self-merge: no-op
	g.translate(6);
	REQUIRE(g.leftmost_slope() == -3);
	REQUIRE(g.rightmost_slope() == 4);
	auto const breakpoints = g.take_breakpoints();
	REQUIRE(breakpoints.size() == 2);
	REQUIRE(breakpoints[0].position == 2);
	REQUIRE(breakpoints[0].weight == 2);
	REQUIRE(breakpoints[1].position == 16);
	REQUIRE(breakpoints[1].weight == 5);
	// a drained function is the zero function and takes new breakpoints
	g.add_rising_hinge(1, 1);
	REQUIRE(g.take_breakpoints().size() == 1);
	REQUIRE(f.take_breakpoints().empty());
}

TEST_CASE("convex_piecewise_linear_forest: ids", "[dp][convex_piecewise_linear_forest]") {
	forest_type forest(3);
	CHECK(forest.id(2).index() == 2);
	CHECK(forest.id(0) == forest_type::function_id{});
	CHECK(forest.id(0) < forest.id(1));
	auto const added = forest.add_function();
	CHECK(added == forest.id(3));
	CHECK(forest[added].id() == added);
	CHECK(forest.function_count() == 4);
	// a reference stays valid across add_function
	auto const f = forest[forest.id(1)];
	for (int i = 0; i < 100; ++i) forest.add_function();
	f.add_absolute(5);
	CHECK(forest[forest.id(1)].rightmost_slope() == 1);
	// an empty forest grows from nothing
	forest_type empty;
	CHECK(empty.function_count() == 0);
	auto const first = empty.add_function();
	CHECK(first == forest_type::function_id{});
	empty[first].add_falling_hinge(-2, 3);
	CHECK(empty[first].leftmost_slope() == -3);
	CHECK(empty[first].minimizers().lower == -2);
	CHECK(!empty[first].minimizers().upper);
}

TEST_CASE("convex_piecewise_linear_forest: large weights and coordinates", "[dp][convex_piecewise_linear_forest]") {
	forest_type forest(1);
	auto const f = forest[forest.id(0)];
	int64 const far = int64(1) << 61;
	f.add_absolute(far, 1'000'000);
	f.add_absolute(-far, 999'999);
	auto const range = f.clip_slopes(-1, 1);
	REQUIRE(range.lower == -far);
	REQUIRE(range.upper == far);
	auto const breakpoints = f.take_breakpoints();
	REQUIRE(breakpoints.size() == 1); // the left end ate the whole breakpoint at -far
	REQUIRE(breakpoints[0].position == far);
	REQUIRE(breakpoints[0].weight == 2);
}

TEST_CASE("convex_piecewise_linear_forest: preconditions", "[dp][convex_piecewise_linear_forest]") {
	forest_type forest(1);
	auto const f = forest[forest.id(0)];
	f.add_rising_hinge(0, 3); // slopes 0 .. 3
	CHECK(atl_test::aborts([&] { f.clip_slopes(2, 1); }));
	CHECK(atl_test::aborts([&] { f.clip_slopes(-5, -1); }));
	CHECK(atl_test::aborts([&] { f.clip_slopes(4, 5); }));
	CHECK(atl_test::aborts([&] { f.add_absolute(1, -1); }));
	CHECK(atl_test::aborts([&] { (void)forest.id(1); }));
	CHECK(atl_test::aborts([&] { f.insert_segment(1, -1); }));
	CHECK(atl_test::aborts([&] { f.insert_segment(4, 1); }));
	CHECK(atl_test::aborts([&] { f.take_window_minimum(2, 1); }));
	forest_type other(2);
	CHECK(atl_test::aborts([&] { forest.merge(forest.id(0), other.id(1)); }));
	CHECK(atl_test::aborts([&] { (void)forest[other.id(1)]; }));
	f.translate(-10);
	f.clip_slopes(1, 3); // slopes 1 .. 3: no minimum
	CHECK(atl_test::aborts([&] { (void)f.minimizers(); }));
	CHECK(atl_test::aborts([&] { f.take_prefix_minimum(); }));
}

namespace {
	using atl::uint32;
	using atl::weighted_breakpoint;

	/** @brief Random pushes, translations, melds, clears and take_all calls on `Pool`, against one multimap per set. */
	template<typename Pool>
	void check_pool_against_oracle(atl::uint64 seed, std::size_t sets, int steps) {
		std::mt19937_64 rng(seed);
		Pool pool(sets);
		std::vector<std::multimap<int64, int64>> oracle(sets);
		for (int step = 0; step < steps; ++step) {
			std::size_t const a = std::size_t(rng() % sets), b = std::size_t(rng() % sets);
			switch (rng() % 8) {
			case 0:
			case 1:
			case 2: {
				int64 const position = int64(rng() % 40) - 20, weight = int64(rng() % 5) + 1;
				pool.push(a, position, weight);
				oracle[a].emplace(position, weight);
				break;
			}
			case 3: {
				int64 const d = int64(rng() % 11) - 5;
				pool.translate(a, d);
				std::multimap<int64, int64> moved;
				for (auto const &[position, weight] : oracle[a]) moved.emplace(position + d, weight);
				oracle[a] = std::move(moved);
				break;
			}
			case 4:
				pool.meld(a, b);
				if (a != b) oracle[a].merge(oracle[b]);
				break;
			case 5:
				pool.clear(a);
				oracle[a].clear();
				break;
			default: {
				std::vector<weighted_breakpoint<int64, int64>> taken;
				pool.take_all(a, taken);
				std::vector<weighted_breakpoint<int64, int64>> expected;
				for (auto const &[position, weight] : oracle[a]) {
					if (!expected.empty() && expected.back().position == position)
						expected.back().weight += weight;
					else
						expected.push_back({position, weight});
				}
				oracle[a].clear();
				REQUIRE(taken.size() == expected.size());
				for (std::size_t i = 0; i < taken.size(); ++i) {
					REQUIRE(taken[i].position == expected[i].position);
					REQUIRE(taken[i].weight == expected[i].weight);
				}
			}
			}
		}
		for (std::size_t set = 0; set < sets; ++set) {
			std::vector<weighted_breakpoint<int64, int64>> taken;
			pool.take_all(set, taken);
			int64 total = 0;
			for (auto const &t : taken) total += t.weight;
			int64 expected = 0;
			for (auto const &[position, weight] : oracle[set]) expected += weight;
			REQUIRE(total == expected);
		}
	}
}

TEST_CASE("breakpoint_pool and its contenders match a multimap oracle", "[dp][convex_piecewise_linear_forest]") {
	auto const seed = GENERATE(take(12, random(atl::uint64(0), std::numeric_limits<atl::uint64>::max())));
	for (std::size_t const sets : {1, 2, 5, 30}) {
		check_pool_against_oracle<atl::detail::breakpoint_pool<int64, int64>>(seed, sets, 600);
		check_pool_against_oracle<atl::experimental::breakpoint_pool_by_tunable_pairing_pops<int64, int64>>(seed, sets, 600);
		check_pool_against_oracle<atl::experimental::breakpoint_pool_by_repeated_pops<int64, int64>>(seed, sets, 600);
	}
}

TEST_CASE("breakpoint_pool: release handles stale entries and refills", "[dp][convex_piecewise_linear_forest]") {
	for (int const n : {0, 1, 2, 3, 31, 32, 33, 1000}) {
		atl::detail::breakpoint_pool<int64, int64> pool(3);
		pool.push(2, 99, 7);
		for (int round = 0; round < 3; ++round) {
			for (int i = 0; i < n; ++i) pool.push(std::size_t(i % 2), i / 2, 2);
			pool.translate(0, 5);
			pool.meld(0, 1);
			if (n) {
				pool.eat<true>(0, 1, [](int64, int64) {});
				pool.eat<false>(0, 1, [](int64, int64) {});
			}
			if (round % 2 == 0) {
				pool.clear(0);
				pool.clear(0);
			} else {
				std::vector<atl::weighted_breakpoint<int64, int64>> out;
				pool.take_all(0, out);
				int64 total = 0;
				for (auto const &point : out) total += point.weight;
				REQUIRE(total == (n ? 2 * n - 2 : 0));
				REQUIRE(std::ranges::is_sorted(out, {}, &atl::weighted_breakpoint<int64, int64>::position));
			}
			pool.push(0, -3, 4);
			std::vector<atl::weighted_breakpoint<int64, int64>> out;
			pool.take_all(0, out);
			REQUIRE(out.size() == 1);
			REQUIRE(out[0].position == -3);
			REQUIRE(out[0].weight == 4);
		}
		std::vector<atl::weighted_breakpoint<int64, int64>> out;
		pool.take_all(2, out);
		REQUIRE(out.size() == 1);
		REQUIRE(out[0].position == 99);
		REQUIRE(out[0].weight == 7);
	}
}

namespace {
	/** @brief `take_all` onto an output whose tail is unrelated to the set, on `Pool`. */
	template<typename Pool>
	void check_take_all_tail() {
		Pool pool(1);
		pool.push(0, 0, 1);
		// a tail right of the set's breakpoint: a different position, appended, not combined
		std::vector<weighted_breakpoint<int64, int64>> out{{10, 1}};
		pool.take_all(0, out);
		REQUIRE(out.size() == 2);
		CHECK(out[0].position == 10);
		CHECK(out[0].weight == 1);
		CHECK(out[1].position == 0);
		CHECK(out[1].weight == 1);
		// a tail at the set's first position: combined
		pool.push(0, 3, 2);
		pool.push(0, 5, 1);
		std::vector<weighted_breakpoint<int64, int64>> joined{{3, 4}};
		pool.take_all(0, joined);
		REQUIRE(joined.size() == 2);
		CHECK(joined[0].position == 3);
		CHECK(joined[0].weight == 6);
		CHECK(joined[1].position == 5);
	}
}

TEST_CASE("breakpoint_pool: take_all combines equal positions only", "[dp][convex_piecewise_linear_forest]") {
	check_take_all_tail<atl::detail::breakpoint_pool<int64, int64>>();
	check_take_all_tail<atl::experimental::breakpoint_pool_by_tunable_pairing_pops<int64, int64>>();
	check_take_all_tail<atl::experimental::breakpoint_pool_by_repeated_pops<int64, int64>>();
	// a combined weight past Slope is caught
	atl::detail::breakpoint_pool<int64, atl::int8> narrow(1);
	narrow.push(0, 7, atl::int8(100));
	narrow.push(0, 7, atl::int8(100));
	std::vector<weighted_breakpoint<int64, atl::int8>> out;
	CHECK(atl_test::aborts([&] { narrow.take_all(0, out); }));
}

TEST_CASE("convex_piecewise_linear_forest: read-only access through a const forest", "[dp][convex_piecewise_linear_forest]") {
	forest_type forest(2);
	auto const f = forest[forest.id(0)];
	f.add_absolute(3, 2);
	f.add_falling_hinge(-4, 5);
	forest_type const &view = forest;
	auto const read = view[forest.id(0)];
	CHECK(read.id() == forest.id(0));
	CHECK(read.leftmost_slope() == -7);
	CHECK(read.rightmost_slope() == 2);
	CHECK(f.minimizers().lower == 3);
	CHECK(f.minimizers().upper == 3);
	CHECK(view.function_count() == 2);
	CHECK(view[forest.id(1)].leftmost_slope() == 0);
	// the walk behind minimizers leaves the function as it was
	explicit_function oracle;
	oracle.rebuild([](int64 x) { return 2 * std::abs(x - 3) + 5 * std::max<int64>(0, -4 - x); });
	check_shape(f, oracle);
	CHECK(atl_test::aborts([&] { (void)view[forest_type{3}.id(2)]; }));
}

namespace {
	/** @brief Drain `set` of a raw pool with end slopes `leftmost`, `rightmost` and compare its slopes with the oracle's. */
	void check_pool_shape(atl::detail::breakpoint_pool<int64, int64> &pool, int64 leftmost, int64 rightmost,
						  explicit_function const &oracle) {
		std::vector<weighted_breakpoint<int64, int64>> breakpoints;
		pool.take_all(0, breakpoints);
		int64 total = 0;
		for (auto const &b : breakpoints) total += b.weight;
		REQUIRE(leftmost + total == rightmost);
		std::size_t next = 0;
		int64 slope = leftmost;
		for (int64 x = -explicit_function::width + 5; x < explicit_function::width - 5; ++x) {
			while (next < breakpoints.size() && breakpoints[next].position <= x) slope += breakpoints[next++].weight;
			REQUIRE(slope == oracle.slope(x));
		}
	}
}

TEST_CASE("insert_segment_by_fewer_entries matches explicit tables", "[dp][convex_piecewise_linear_forest]") {
	for (unsigned seed = 0; seed < 400; ++seed) {
		std::mt19937_64 rng(seed);
		auto uniform = [&](int64 lo, int64 hi) { return std::uniform_int_distribution<int64>(lo, hi)(rng); };
		atl::detail::breakpoint_pool<int64, int64> pool(1);
		atl::experimental::fewer_entries_scratch<int64, int64> scratch;
		int64 leftmost = 0, rightmost = 0;
		explicit_function now;
		for (int step = 0; step < 40; ++step) {
			explicit_function const old = now;
			int const kind = int(uniform(0, 3));
			INFO("seed=" << seed << " step=" << step << " kind=" << kind);
			if (kind <= 1) {
				// a rising or falling hinge of weight 1 .. 3, as the forest adds them
				int64 const a = uniform(-25, 25), w = uniform(1, 3);
				pool.push(0, a, w);
				(kind == 0 ? rightmost : leftmost) += kind == 0 ? w : -w;
				now.rebuild([&](int64 x) { return old(x) + w * (kind == 0 ? std::max<int64>(0, x - a) : std::max<int64>(0, a - x)); });
			} else if (kind == 2) {
				int64 const d = uniform(-2, 2);
				pool.translate(0, d);
				now.rebuild([&](int64 x) { return old(x - d); });
			} else {
				int64 const slope = uniform(leftmost - 1, rightmost), length = uniform(0, 4);
				auto const crossing =
					atl::experimental::insert_segment_by_fewer_entries(pool, 0, leftmost, rightmost, slope, length, scratch);
				now.rebuild([&](int64 x) {
					int64 best = std::numeric_limits<int64>::max();
					for (int64 t = 0; t <= length; ++t) best = std::min(best, old(x - t) + slope * t);
					return best;
				});
				REQUIRE(crossing.has_value() == (leftmost < slope));
				if (crossing) {
					REQUIRE(old.slope(*crossing - 1) <= slope);
					REQUIRE(old.slope(*crossing) >= slope);
				}
				for (int64 x = -100; x <= 100; x += 7) {
					int64 const y = crossing ? std::clamp(*crossing, x - length, x) : x - length;
					REQUIRE(old(y) + slope * (x - y) == now(x));
				}
			}
		}
		check_pool_shape(pool, leftmost, rightmost, now);
	}
	// the preconditions
	atl::detail::breakpoint_pool<int64, int64> pool(1);
	atl::experimental::fewer_entries_scratch<int64, int64> scratch;
	pool.push(0, 0, 2);
	CHECK(atl_test::aborts([&] { (void)atl::experimental::insert_segment_by_fewer_entries(pool, 0, -1, 1, 0, -1, scratch); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::insert_segment_by_fewer_entries(pool, 0, -1, 1, 2, 1, scratch); }));
}
