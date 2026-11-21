#include <atl/dp/exact_count_optimum.hpp>
#include <atl/dp/optimal_partition.hpp>
#include <atl/dp/segmentation.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <bit>
#include <catch2/catch.hpp>
#include <compare>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"
#include "monge_costs.hpp"

using atl::int64;
using atl::uint32;

namespace {
	/**
	 * @brief The penalized 1D/1D DP by brute force: the minimum over all
	 *        cuts of `[0, n)` of the segment costs plus `price` per segment,
	 *        ties to fewer segments.
	 */
	template<std::regular_invocable<uint32, uint32> Cost>
	std::pair<int64, int64> penalized_cuts(std::size_t n, Cost const &cost, int64 price) {
		std::vector<std::pair<int64, int64>> dp(n + 1, {0, 0});
		for (std::size_t i = 1; i <= n; ++i) {
			std::pair<int64, int64> best{0, -1};
			for (std::size_t j = 0; j < i; ++j) {
				std::pair<int64, int64> const candidate{dp[j].first + cost(uint32(j), uint32(i)) + price,
														dp[j].second + 1};
				if (best.second < 0 || candidate < best) best = candidate;
			}
			dp[i] = best;
		}
		return dp[n];
	}

	/** @brief A value with a segment count, ordered lexicographically: fewer segments break ties. */
	struct counted {
		int64 value = 0, count = 0;

		friend constexpr counted operator+(counted const &a, counted const &b) {
			return {a.value + b.value, a.count + b.count};
		}

		friend constexpr auto operator<=>(counted const &, counted const &) = default;
	};
}

TEST_CASE("exact_count_optimum matches optimal_partition on random Monge costs", "[exact_count_optimum]") {
	std::mt19937_64 rng(31);
	for (int iter = 0; iter < 1500; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 13);
		// density 0 makes the curve collinear over long stretches, the case the
		// value-from-the-penalized-optimum rule exists for
		int64 const density = iter % 4 == 0 ? 0 : (iter % 4 == 1 ? 1 : 6);
		auto const a = atl_test::random_monge(n + 1, n + 1, rng, density, iter % 2 == 0 ? 0 : 12);
		auto const cost = [&a](uint32 j, uint32 i) { return a[j][i]; };
		auto const curve = atl::optimal_partition(n, n, cost);
		auto solve = [&](int64 price) { return penalized_cuts(n, cost, price); };
		for (std::size_t k = 1; k <= n; ++k) {
			INFO("n=" << n << " k=" << k << " iter=" << iter);
			auto const r = atl::exact_count_optimum(int64(k), int64(-1'000'000), int64(1'000'000), solve);
			CHECK(r.value == curve[k - 1]);
			// the price is the smallest one at which k ties for the penalized optimum
			CHECK(solve(r.price).first == curve[k - 1] + r.price * int64(k));
			if (k < n) CHECK(-r.price == curve[k] - curve[k - 1]);
			// the reported count is the solver's at that price, at most k
			CHECK(r.count == solve(r.price).second);
			CHECK(r.count <= int64(k));
		}
	}
}

TEST_CASE("exact_count_optimum over optimal_segmentation with a counted value", "[exact_count_optimum]") {
	// the penalized DP itself is a quadrangle-inequality DP: (cost + price, 1)
	// keeps the inequality, and the lexicographic order breaks ties to fewer
	// segments, so the decision queue solves each probe in O(n log n)
	std::mt19937_64 rng(32);
	for (int iter = 0; iter < 300; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 40);
		std::vector<int64> s(n + 1, 0);
		for (std::size_t i = 1; i <= n; ++i) s[i] = s[i - 1] + int64(rng() % (iter % 3 == 0 ? 2 : 50));
		auto const cost = [&s](uint32 j, uint32 i) {
			int64 const d = s[i] - s[j];
			return d * d;
		};
		auto const curve = atl::optimal_partition(n, n, cost);
		auto solve = [&](int64 price) {
			auto const r = atl::optimal_segmentation(n, [&](uint32 j, uint32 i) { return counted{cost(j, i) + price, 1}; });
			return std::pair{r.value[n].value, r.value[n].count};
		};
		std::size_t const k = 1 + std::size_t(rng() % n);
		CHECK(atl::exact_count_optimum(int64(k), int64(-1), s[n] * s[n] + 1, solve).value == curve[k - 1]);
	}
}

TEST_CASE("exact_count_optimum maximizes with std::greater", "[exact_count_optimum]") {
	// exactly k pairwise non-adjacent elements of maximum sum: a concave curve
	// (a flow), checked against all subsets
	std::mt19937_64 rng(33);
	for (int iter = 0; iter < 400; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 12);
		std::vector<int64> a(n);
		for (auto &x : a) x = int64(rng() % 21) - 5;
		std::vector<std::pair<bool, int64>> best((n + 1) / 2 + 1, {false, 0});
		for (uint32 mask = 0; mask < (1u << n); ++mask) {
			if (mask & (mask >> 1)) continue;
			int64 sum = 0;
			for (std::size_t i = 0; i < n; ++i)
				if (mask >> i & 1) sum += a[i];
			auto &slot = best[std::size_t(std::popcount(mask))];
			if (!slot.first || sum > slot.second) slot = {true, sum};
		}
		// maximize sum - price * count, ties to fewer: order (value, -count)
		auto solve = [&](int64 price) {
			std::pair<int64, int64> skip{0, 0}, take{0, 0}; // best ending with a[i] not taken / taken
			auto const better = [](std::pair<int64, int64> const &x, std::pair<int64, int64> const &y) {
				return x.first != y.first ? x.first > y.first : x.second < y.second;
			};
			bool has_take = false;
			for (std::size_t i = 0; i < n; ++i) {
				std::pair<int64, int64> const next_take{skip.first + a[i] - price, skip.second + 1};
				std::pair<int64, int64> next_skip = skip;
				if (has_take && better(take, next_skip)) next_skip = take;
				skip = next_skip, take = next_take, has_take = true;
			}
			return better(take, skip) ? take : skip;
		};
		std::size_t const k = std::size_t(rng() % best.size());
		INFO("n=" << n << " k=" << k);
		CHECK(atl::exact_count_optimum(int64(k), int64(-100), int64(100), solve, std::greater<>{}).value == best[k].second);
	}
}

TEST_CASE("exact_count_optimum on a fully collinear curve", "[exact_count_optimum]") {
	// every cut costs n + 5k: all counts tie at price -5, and no price makes
	// the solver report a middle count
	std::size_t const n = 9;
	auto const cost = [](uint32 j, uint32 i) { return int64(i - j) + 5; };
	auto solve = [&](int64 price) { return penalized_cuts(n, cost, price); };
	for (std::size_t k = 1; k <= n; ++k) {
		auto const r = atl::exact_count_optimum(int64(k), int64(-50), int64(50), solve);
		CHECK(r.value == int64(n + 5 * k));
		CHECK(r.price == (k < n ? -5 : -50));
		// inside the stretch the solver reports its fewest-items end: a tie, detected without another call
		CHECK(r.count == (k < n ? 1 : int64(n)));
	}
	// a one-price range is fine when the count is reached there
	CHECK(atl::exact_count_optimum(int64(1), int64(0), int64(0), solve).value == int64(n + 5));
}

TEST_CASE("exact_count_optimum accepts unsigned prices and narrow counts", "[exact_count_optimum]") {
	std::size_t const n = 6;
	auto const cost = [](uint32 j, uint32 i) { return int64(i - j) * int64(i - j); };
	auto solve = [&](atl::uint32 price) {
		auto const r = penalized_cuts(n, cost, int64(price));
		return std::pair{r.first, int(r.second)};
	};
	// 36, 18, 12, 10, 8, 6: slopes -18, -6, -2, -2, -2
	CHECK(atl::exact_count_optimum(3, atl::uint32(0), atl::uint32(100), solve).value == 12);
	CHECK(atl::exact_count_optimum(4, atl::uint32(0), atl::uint32(100), solve).value == 10);
	CHECK(atl::exact_count_optimum(6, atl::uint32(0), atl::uint32(100), solve).value == 6);
	CHECK(atl::exact_count_optimum(1, atl::uint32(0), atl::uint32(100), solve).value == 36);
}

TEST_CASE("exact_count_optimum survives a price range spanning the whole type", "[exact_count_optimum]") {
	auto solve = [](int64 price) {
		// f(c) = (c - 3)^2 on c = 0 .. 6, penalized by price * c, fewest count on ties; the
		// penalized values are int128, so every probed price is exact (|price| * 6 < 2^66)
		std::pair<atl::int128, int64> best{0, -1};
		for (int64 c = 0; c <= 6; ++c) {
			std::pair<atl::int128, int64> const candidate{atl::int128((c - 3) * (c - 3)) + atl::int128(price) * c, c};
			if (best.second < 0 || candidate < best) best = candidate;
		}
		return best;
	};
	auto const r = atl::exact_count_optimum(int64(2), std::numeric_limits<int64>::min(),
											std::numeric_limits<int64>::max(), solve);
	CHECK(r.value == atl::int128(1));
	CHECK(r.price == 1);
	CHECK(r.count == 2);
}

TEST_CASE("exact_count_optimum forms price * count wide", "[exact_count_optimum]") {
	// the only feasible count is 2^63, of value 0: at price -1 the solver returns
	// {INT64_MIN, 2^63}, and -1 * 2^63 does not fit int64 although the answer does
	atl::uint64 const count = atl::uint64(1) << 63;
	auto solve = [count](int64) { return std::pair<int64, atl::uint64>{std::numeric_limits<int64>::min(), count}; };
	auto const r = atl::exact_count_optimum(count, int64(-1), int64(-1), solve);
	CHECK(r.value == 0);
	CHECK(r.price == -1);
	CHECK(r.count == count);
}

TEST_CASE("exact_count_optimum's price is the slope with the direction's sign", "[exact_count_optimum]") {
	// maximize f(c) = 3c over c = 0 .. 2: the solver maximizes 3c - price * c, ties to fewer
	auto solve = [](int64 price) {
		std::pair<int64, int64> best{0, 0};
		for (int64 c = 1; c <= 2; ++c)
			if (3 * c - price * c > best.first) best = {3 * c - price * c, c};
		return best;
	};
	auto const r = atl::exact_count_optimum(int64(1), int64(0), int64(4), solve, std::greater<>{});
	CHECK(r.value == 3);
	// maximizing, the slope just right of the count is +price: f(2) - f(1) = 3
	CHECK(r.price == 3);
	CHECK(r.count == 0); // c = 0, 1, 2 all tie at price 3: the fewest is reported
}

TEST_CASE("exact_count_optimum rejects a price range that misses the count", "[exact_count_optimum]") {
	std::size_t const n = 6;
	auto const cost = [](uint32 j, uint32 i) { return int64(i - j) + 5; };
	auto solve = [&](int64 price) { return penalized_cuts(n, cost, price); };
	// hi too low: even at hi every segment pays, the solver cuts all n
	CHECK(atl_test::aborts([&] { (void)atl::exact_count_optimum(int64(2), int64(-50), int64(-10), solve); }));
	// lo too high: at lo the solver already takes a single segment
	CHECK(atl_test::aborts([&] { (void)atl::exact_count_optimum(int64(4), int64(10), int64(50), solve); }));
	CHECK(atl_test::aborts([&] { (void)atl::exact_count_optimum(int64(4), int64(1), int64(0), solve); }));
	// infeasible counts, whatever the range: no cut has 0 or n + 1 segments
	CHECK(atl_test::aborts([&] { (void)atl::exact_count_optimum(int64(0), int64(-50), int64(50), solve); }));
	CHECK(atl_test::aborts([&] { (void)atl::exact_count_optimum(int64(n + 1), int64(-50), int64(50), solve); }));
}
