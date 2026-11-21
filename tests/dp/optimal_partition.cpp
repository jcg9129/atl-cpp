#include <atl/dp/concepts.hpp>
#include <atl/dp/optimal_partition.hpp>
#include <atl/experimental/dp/optimal_partition_variants.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <vector>
#include "../expect_abort.hpp"
#include "monge_costs.hpp"

using atl::int64;
using atl::uint32;

namespace {
	/**
	 * @brief The layered O(k n^2) recurrence; nullopt marks an infeasible
	 *        state. The certifying oracle of the partition family (the
	 *        library and the experimental contenders alike).
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires std::invocable<Cost const &, uint32, uint32> && std::strict_weak_order<Compare const &, int64, int64>
	std::vector<int64> brute(std::size_t n, std::size_t segments, Cost const &cost, Compare comp = {}) {
		std::vector<std::optional<int64>> previous(n + 1), current(n + 1);
		previous[0] = 0;
		std::vector<int64> best;
		for (std::size_t k = 1; k <= segments; ++k) {
			for (std::size_t i = 0; i <= n; ++i) {
				current[i].reset();
				for (std::size_t j = 0; j < i; ++j)
					if (previous[j]) {
						int64 const v = *previous[j] + cost(uint32(j), uint32(i));
						if (!current[i] || comp(v, *current[i])) current[i] = v;
					}
			}
			std::swap(previous, current);
			best.push_back(*previous[n]);
		}
		return best;
	}

	/**
	 * @brief The cuts @ref atl::optimal_partition_cuts documents, by the
	 *        O(k n^2) recurrence: the leftmost best last cut of exactly `k`
	 *        segments, then the leftmost best cut of each prefix it ends.
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires std::invocable<Cost const &, uint32, uint32> && std::strict_weak_order<Compare const &, int64, int64>
	std::vector<uint32> brute_cuts(std::size_t n, std::size_t segments, Cost const &cost, Compare comp = {}) {
		std::vector<std::vector<std::optional<int64>>> dp(segments + 1, std::vector<std::optional<int64>>(n + 1));
		std::vector<std::vector<uint32>> opt(segments + 1, std::vector<uint32>(n + 1, 0));
		dp[0][0] = 0;
		for (std::size_t k = 1; k <= segments; ++k)
			for (std::size_t i = 1; i <= n; ++i)
				for (std::size_t j = 0; j < i; ++j)
					if (dp[k - 1][j]) {
						int64 const v = *dp[k - 1][j] + cost(uint32(j), uint32(i));
						if (!dp[k][i] || comp(v, *dp[k][i])) {
							dp[k][i] = v;
							opt[k][i] = uint32(j);
						}
					}
		std::vector<uint32> cuts(segments + 1, 0);
		cuts[segments] = uint32(n);
		for (std::size_t k = segments; k >= 1; --k) cuts[k - 1] = opt[k][cuts[k]];
		return cuts;
	}
}

TEST_CASE("optimal_partition matches the layered recurrence on random Monge costs", "[partition]") {
	std::mt19937_64 rng(21);
	for (int iter = 0; iter < 3000; ++iter) {
		std::size_t const n = std::size_t(rng() % 14), segments = n == 0 ? 0 : std::size_t(rng() % (n + 1));
		auto const a = atl_test::random_monge(n + 1, n + 1, rng, iter % 3 == 0 ? 1 : 5, iter % 2 == 0 ? 0 : 12);
		auto const cost = [&a](uint32 j, uint32 i) { return a[j][i]; };
		INFO("n=" << n << " segments=" << segments);
		auto const want = brute(n, segments, cost);
		CHECK(atl::optimal_partition(n, segments, cost) == want);
		CHECK(atl::experimental::optimal_partition_by_knuth_bound(n, segments, cost) == want);
		CHECK(atl::experimental::optimal_partition_by_divide_and_conquer(n, segments, cost) == want);
		CHECK(atl::experimental::optimal_partition_by_floored_divide_and_conquer(n, segments, cost) == want);
		CHECK(atl::experimental::optimal_partition_by_smawk(n, segments, cost) == want);
	}
}

TEST_CASE("optimal_partition maximizes with std::greater", "[partition]") {
	std::mt19937_64 rng(22);
	for (int iter = 0; iter < 1000; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 12), segments = 1 + std::size_t(rng() % n);
		auto a = atl_test::random_monge(n + 1, n + 1, rng, 4, 8);
		for (auto &row : a)
			for (auto &x : row) x = -x;
		auto const cost = [&a](uint32 j, uint32 i) { return a[j][i]; };
		auto const want = brute(n, segments, cost, std::greater<>{});
		CHECK(atl::optimal_partition(n, segments, cost, std::greater<>{}) == want);
		CHECK(atl::experimental::optimal_partition_by_knuth_bound(n, segments, cost, std::greater<>{}) == want);
		CHECK(atl::experimental::optimal_partition_by_floored_divide_and_conquer(n, segments, cost, std::greater<>{}) == want);
		CHECK(atl::experimental::optimal_partition_by_smawk(n, segments, cost, std::greater<>{}) == want);
	}
}

TEST_CASE("optimal_partition with a stateful two-pointer cost", "[partition]") {
	// distinct values in [j, i) -- maximized: splitting never loses distinct
	// values, so this is a concave-type (inverse-Monge) cost. The
	// cost keeps a sliding window and moves its ends to each query.
	std::mt19937_64 rng(23);
	for (int iter = 0; iter < 200; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 40), segments = 1 + std::size_t(rng() % std::min<std::size_t>(n, 6));
		std::vector<uint32> a(n);
		for (auto &x : a) x = uint32(rng() % 6);

		struct window {
			std::vector<uint32> const *items;
			std::vector<uint32> count = std::vector<uint32>(6, 0);
			uint32 lo = 0, hi = 0, distinct = 0;

			int64 operator()(uint32 j, uint32 i) {
				auto add = [&](uint32 x) { distinct += count[(*items)[x]]++ == 0; };
				auto drop = [&](uint32 x) { distinct -= --count[(*items)[x]] == 0; };
				while (hi < i) add(hi++);
				while (lo > j) add(--lo);
				while (hi > i) drop(--hi);
				while (lo < j) drop(lo++);
				return distinct;
			}
		} cost{&a};

		auto const plain = [&a](uint32 j, uint32 i) {
			std::vector<char> seen(6, 0);
			int64 d = 0;
			for (uint32 x = j; x < i; ++x) d += !seen[a[x]]++;
			return d;
		};
		auto const want = brute(n, segments, plain, std::greater<>{});
		CHECK(atl::optimal_partition(n, segments, cost, std::greater<>{}) == want);
		CHECK(atl::experimental::optimal_partition_by_knuth_bound(n, segments, cost, std::greater<>{}) == want);
		CHECK(atl::experimental::optimal_partition_by_floored_divide_and_conquer(n, segments, cost, std::greater<>{}) == want);
		// out of order, so the window moves far more, but the values still agree
		CHECK(atl::experimental::optimal_partition_by_smawk(n, segments, cost, std::greater<>{}) == want);
	}
}

namespace {
	/**
	 * @brief A per-layer engine choice drawn at random -- divide and
	 *        conquer, Knuth unlimited, or a random budget up to the
	 *        divide-and-conquer estimate (Knuth when the previous layer fit
	 *        it, falling back when this one does not) -- so every hand-over
	 *        between the engines is exercised.
	 */
	struct random_choice {
		std::mt19937_64 *rng;

		std::uint64_t operator()(std::uint64_t divide_estimate) const {
			switch ((*rng)() % 3) {
			case 0:
				return 0;
			case 1:
				return UINT64_MAX;
			default:
				return 1 + (*rng)() % divide_estimate;
			}
		}
	};

	/** @brief optimal_partition with the engine choice @ref random_choice. */
	template<typename Cost>
		requires atl::additive_cost<Cost, std::less<>>
	std::vector<int64> mixed_curve(uint32 n, uint32 segments, Cost &cost, std::mt19937_64 &rng) {
		std::vector<int64> best;
		if (segments == 0) return best;
		std::vector<int64> previous(n + 1), current(n + 1);
		for (uint32 i = 1; i <= n; ++i) previous[i] = cost(0, i);
		best.push_back(previous[n]);
		std::vector<uint32> argmins(2 * (n + 1));
		auto argmin_row = [&argmins, n](uint32 k) { return argmins.data() + (k & 1) * (n + 1); };
		auto after_layer = [&best, n](std::vector<int64> const &layer) { best.push_back(layer[n]); };
		std::less<> const comp;
		atl::detail::partition_layers(n, segments, true, cost, comp, previous, current, argmin_row, after_layer,
									  random_choice{&rng});
		return best;
	}

	/** @brief optimal_partition_cuts' value and cuts with the engine choice @ref random_choice. */
	template<typename Cost>
		requires atl::additive_cost<Cost, std::less<>>
	atl::partition_cuts_result<int64> mixed_cuts(uint32 n, uint32 segments, Cost &cost, std::mt19937_64 &rng) {
		std::size_t const row = n + 1;
		std::vector<int64> previous(row), current(row);
		std::vector<uint32> choice((segments - 1) * row);
		for (uint32 i = 1; i <= n - (segments - 1); ++i) previous[i] = cost(0, i);
		auto argmin_row = [&choice, row](uint32 k) { return choice.data() + (k - 2) * row; };
		auto after_layer = [](std::vector<int64> const &) {};
		std::less<> const comp;
		atl::detail::partition_layers(n, segments, false, cost, comp, previous, current, argmin_row, after_layer,
									  random_choice{&rng});
		atl::partition_cuts_result<int64> result{previous[n], std::vector<uint32>(segments + 1)};
		uint32 i = n;
		result.boundaries[segments] = i;
		for (uint32 k = segments; k >= 2; --k) result.boundaries[k - 1] = i = choice[(k - 2) * row + i];
		return result;
	}
}

TEST_CASE("optimal_partition engines hand layers over in any order", "[partition]") {
	std::mt19937_64 rng(24);
	for (int iter = 0; iter < 3000; ++iter) {
		uint32 const n = 1 + uint32(rng() % 40), segments = 1 + uint32(rng() % n);
		auto const a = atl_test::random_monge(n + 1, n + 1, rng, iter % 3 == 0 ? 1 : 5, iter % 2 == 0 ? 0 : 12);
		auto cost = [&a](uint32 j, uint32 i) { return a[j][i]; };
		INFO("n=" << n << " segments=" << segments);
		auto const want = brute(n, segments, cost);
		REQUIRE(mixed_curve(n, segments, cost, rng) == want);
		auto const cuts = mixed_cuts(n, segments, cost, rng);
		REQUIRE(cuts.value == want[segments - 1]);
		// leftmost argmins whatever the engine: the same cuts as the library
		CHECK(cuts.boundaries == atl::optimal_partition_cuts(n, segments, cost).boundaries);
	}
}

TEST_CASE("optimal_partition on larger convex segment costs, every engine", "[partition]") {
	// (S_i - S_j - L)^2 over random prefix sums: convex in the segment sum, so Monge
	std::mt19937_64 rng(25);
	for (std::size_t const n : {60u, 200u}) {
		for (int iter = 0; iter < 4; ++iter) {
			std::vector<int64> s(n + 1, 0);
			for (std::size_t i = 1; i <= n; ++i) s[i] = s[i - 1] + int64(rng() % (iter % 2 == 0 ? 3 : 1000));
			int64 const target = s[n] / int64(1 + rng() % n);
			auto const cost = [&](uint32 j, uint32 i) {
				int64 const d = s[i] - s[j] - target;
				return d * d;
			};
			auto const want = brute(n, n, cost);
			for (std::size_t const segments : {std::size_t(1), std::size_t(2), std::size_t(7), n / 3, n - 1, n}) {
				INFO("n=" << n << " segments=" << segments);
				std::vector<int64> const prefix(want.begin(), want.begin() + std::ptrdiff_t(segments));
				CHECK(atl::optimal_partition(n, segments, cost) == prefix);
				CHECK(atl::experimental::optimal_partition_by_knuth_bound(n, segments, cost) == prefix);
				CHECK(atl::experimental::optimal_partition_by_floored_divide_and_conquer(n, segments, cost) == prefix);
				CHECK(atl::experimental::optimal_partition_by_smawk(n, segments, cost) == prefix);
				CHECK(atl::optimal_partition_cuts(n, segments, cost).value == want[segments - 1]);
			}
		}
	}
}

TEST_CASE("optimal_partition contenders match the library on wider random Monge costs", "[partition]") {
	std::mt19937_64 rng(26);
	for (int iter = 0; iter < 400; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 80);
		std::size_t const segments = iter % 4 == 0 ? n : 1 + std::size_t(rng() % (iter % 4 == 1 ? std::min<std::size_t>(n, 4) : n));
		auto const a = atl_test::random_monge(n + 1, n + 1, rng, iter % 3 == 0 ? 1 : 5, iter % 2 == 0 ? 0 : 12);
		auto const cost = [&a](uint32 j, uint32 i) { return a[j][i]; };
		INFO("n=" << n << " segments=" << segments);
		auto const want = atl::optimal_partition(n, segments, cost);
		CHECK(atl::experimental::optimal_partition_by_floored_divide_and_conquer(n, segments, cost) == want);
		CHECK(atl::experimental::optimal_partition_by_smawk(n, segments, cost) == want);
	}
}

TEST_CASE("optimal_partition contenders off the quadrangle inequality", "[partition]") {
	// random costs break the precondition: every engine still returns real
	// partition costs (never better than the optimum), and SMAWK does go
	// wrong somewhere, so its stated precondition is load-bearing
	std::mt19937_64 rng(27);
	int smawk_misses = 0;
	for (int iter = 0; iter < 2000; ++iter) {
		std::size_t const n = 2 + std::size_t(rng() % 12), segments = 1 + std::size_t(rng() % n);
		std::vector<std::vector<int64>> a(n + 1, std::vector<int64>(n + 1));
		for (auto &row : a)
			for (auto &x : row) x = int64(rng() % 50);
		auto const cost = [&a](uint32 j, uint32 i) { return a[j][i]; };
		INFO("n=" << n << " segments=" << segments);
		auto const want = brute(n, segments, cost);
		auto const smawk = atl::experimental::optimal_partition_by_smawk(n, segments, cost);
		auto const floored = atl::experimental::optimal_partition_by_floored_divide_and_conquer(n, segments, cost);
		REQUIRE(smawk.size() == segments);
		REQUIRE(floored.size() == segments);
		for (std::size_t k = 0; k < segments; ++k) {
			CHECK(smawk[k] >= want[k]);
			CHECK(floored[k] >= want[k]);
		}
		smawk_misses += smawk != want;
	}
	CHECK(smawk_misses > 0);
}

TEST_CASE("optimal_partition edge cases", "[partition]") {
	auto const cost = [](uint32 j, uint32 i) { return int64(i - j) * int64(i - j); };
	CHECK(atl::optimal_partition(0, 0, cost).empty());
	CHECK(atl::optimal_partition(5, 0, cost).empty());
	CHECK(atl::optimal_partition(1, 1, cost) == std::vector<int64>{1});
	// 6 items in k segments of squared length: as balanced as possible
	CHECK(atl::optimal_partition(6, 6, cost) == std::vector<int64>{36, 18, 12, 10, 8, 6});
	for (std::size_t const n : {0u, 1u, 2u, 6u})
		for (std::size_t segments = 0; segments <= n; ++segments) {
			auto const want = atl::optimal_partition(n, segments, cost);
			CHECK(atl::experimental::optimal_partition_by_floored_divide_and_conquer(n, segments, cost) == want);
			CHECK(atl::experimental::optimal_partition_by_smawk(n, segments, cost) == want);
		}
}

TEST_CASE("optimal_partition rejects more segments than items", "[partition]") {
	auto const cost = [](uint32, uint32) { return 0; };
	CHECK(atl_test::aborts([&] { (void)atl::optimal_partition(3, 4, cost); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::optimal_partition_by_floored_divide_and_conquer(3, 4, cost); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::optimal_partition_by_smawk(3, 4, cost); }));
}

TEST_CASE("optimal_partition_cuts rebuilds an exactly-k optimum", "[partition]") {
	std::mt19937_64 rng(23);
	for (int iter = 0; iter < 3000; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 13), segments = 1 + std::size_t(rng() % n);
		bool const is_max = iter % 4 == 3;
		auto a = atl_test::random_monge(n + 1, n + 1, rng, iter % 3 == 0 ? 1 : 5, iter % 2 == 0 ? 0 : 12);
		if (is_max)
			for (auto &row : a)
				for (auto &x : row) x = -x;
		auto const cost = [&a](uint32 j, uint32 i) { return a[j][i]; };
		auto const r = is_max ? atl::optimal_partition_cuts(n, segments, cost, std::greater<>{})
							  : atl::optimal_partition_cuts(n, segments, cost);
		int64 const want = is_max ? brute(n, segments, cost, std::greater<>{})[segments - 1]
								  : brute(n, segments, cost)[segments - 1];
		INFO("n=" << n << " segments=" << segments);
		REQUIRE(r.value == want);
		// ties: the documented leftmost cuts, not just any optimal ones
		CHECK(r.boundaries == (is_max ? brute_cuts(n, segments, cost, std::greater<>{}) : brute_cuts(n, segments, cost)));
		REQUIRE(r.boundaries.size() == segments + 1);
		REQUIRE(r.boundaries.front() == 0);
		REQUIRE(r.boundaries.back() == n);
		int64 total = 0;
		for (std::size_t t = 0; t < segments; ++t) {
			REQUIRE(r.boundaries[t] < r.boundaries[t + 1]);
			total += cost(r.boundaries[t], r.boundaries[t + 1]);
		}
		CHECK(total == want);
	}
}

TEST_CASE("optimal_partition_cuts preconditions", "[partition]") {
	auto const cost = [](uint32 j, uint32 i) { return int64(i - j) * int64(i - j); };
	CHECK(atl_test::aborts([&] { (void)atl::optimal_partition_cuts(3, 0, cost); }));
	CHECK(atl_test::aborts([&] { (void)atl::optimal_partition_cuts(3, 4, cost); }));
	auto const one = atl::optimal_partition_cuts(5, 1, cost);
	CHECK(one.value == 25);
	CHECK(one.boundaries == std::vector<uint32>{0, 5});
	auto const all = atl::optimal_partition_cuts(5, 5, cost);
	CHECK(all.value == 5);
	CHECK(all.boundaries == std::vector<uint32>{0, 1, 2, 3, 4, 5});
}

TEST_CASE("optimal_partition_cuts with a segment count close to n", "[partition]") {
	std::mt19937_64 rng(24);
	for (int iter = 0; iter < 300; ++iter) {
		std::size_t const n = 1 + std::size_t(rng() % 90), slack = std::size_t(rng() % std::min<std::size_t>(n, 6)), segments = n - slack;
		auto const a = atl_test::random_monge(n + 1, n + 1, rng, 5, 12);
		auto const cost = [&a](uint32 j, uint32 i) { return a[j][i]; };
		auto const r = atl::optimal_partition_cuts(n, segments, cost);
		INFO("n=" << n << " segments=" << segments);
		REQUIRE(r.value == atl::optimal_partition(n, segments, cost)[segments - 1]);
		REQUIRE(r.boundaries.size() == segments + 1);
		REQUIRE(r.boundaries.front() == 0);
		REQUIRE(r.boundaries.back() == n);
		int64 total = 0;
		for (std::size_t t = 0; t < segments; ++t) {
			REQUIRE(r.boundaries[t] < r.boundaries[t + 1]);
			total += cost(r.boundaries[t], r.boundaries[t + 1]);
		}
		CHECK(total == r.value);
	}
}

TEST_CASE("optimal_partition forms candidate totals wide", "[optimal_partition]") {
	// dp_1[1] + cost(1, 3) = 2 INT64_MAX leaves int64; the best two segments, [0, 2) [2, 3), cost 0
	auto const cost = [](uint32 j, uint32 i) { return (j == 0 && i == 2) || (j == 2 && i == 3) ? int64(0) : std::numeric_limits<int64>::max(); };
	std::vector<int64> const want{std::numeric_limits<int64>::max(), 0};
	CHECK(atl::optimal_partition(3, 2, cost) == want);
	auto const cuts = atl::optimal_partition_cuts(3, 2, cost);
	CHECK(cuts.value == 0);
	CHECK(cuts.boundaries == std::vector<uint32>{0, 2, 3});
	CHECK(atl::experimental::optimal_partition_by_divide_and_conquer(3, 2, cost) == want);
	CHECK(atl::experimental::optimal_partition_by_knuth_bound(3, 2, cost) == want);
	CHECK(atl::experimental::optimal_partition_by_floored_divide_and_conquer(3, 2, cost) == want);
	CHECK(atl::experimental::optimal_partition_by_smawk(3, 2, cost) == want);
	// a middle layer by divide and conquer and by SMAWK: three segments of four items
	auto const four = [](uint32 j, uint32 i) { return i == j + 1 ? int64(0) : std::numeric_limits<int64>::max(); };
	CHECK(atl::optimal_partition(4, 4, four)[3] == 0);
	CHECK(atl::experimental::optimal_partition_by_divide_and_conquer(4, 4, four)[3] == 0);
	CHECK(atl::experimental::optimal_partition_by_smawk(4, 4, four)[3] == 0);
}
