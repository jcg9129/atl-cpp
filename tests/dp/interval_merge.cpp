#include <atl/dp/interval_merge.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>
#include "../expect_abort.hpp"
#include "monge_costs.hpp"

using atl::int64;
using atl::uint32;

namespace {
	/** @brief The certifying oracle: the O(n^3) recurrence over every split. */
	template<typename Cost, typename Compare = std::less<>>
		requires std::invocable<Cost const &, uint32, uint32> && std::strict_weak_order<Compare const &, int64, int64>
	atl_test::matrix brute(std::size_t n, Cost const &cost, Compare comp = {}) {
		atl_test::matrix dp(n + 1, std::vector<int64>(n + 1, 0));
		for (std::size_t len = 2; len <= n; ++len)
			for (std::size_t l = 0; l + len <= n; ++l) {
				std::size_t const r = l + len;
				int64 best = dp[l][l + 1] + dp[l + 1][r];
				for (std::size_t m = l + 2; m < r; ++m)
					if (comp(dp[l][m] + dp[m][r], best)) best = dp[l][m] + dp[m][r];
				dp[l][r] = best + cost(uint32(l), uint32(r));
			}
		return dp;
	}

	template<typename Compare>
	void check(std::size_t n, atl::interval_merge_result<int64, Compare> const &got, atl_test::matrix const &want) {
		REQUIRE(got.items == n);
		REQUIRE(got.table.size() == n * (n + 1) / 2);
		for (std::size_t l = 0; l < n; ++l)
			for (std::size_t r = l + 1; r <= n; ++r) {
				INFO("n=" << n << " [" << l << ", " << r << ")");
				CHECK(got.value(l, r) == want[l][r]);
			}
	}
}

TEST_CASE("optimal_interval_merge matches the O(n^3) recurrence", "[interval_merge]") {
	std::mt19937_64 rng(31);
	for (int iter = 0; iter < 3000; ++iter) {
		std::size_t const n = std::size_t(rng() % 14);
		auto const w = atl_test::random_interval_cost(n, rng, iter % 3 == 0 ? 1 : 6);
		auto const cost = [&w](uint32 l, uint32 r) { return w[l][r]; };
		check(n, atl::optimal_interval_merge(n, cost), brute(n, cost));
	}
}

TEST_CASE("optimal_interval_merge maximizes with std::greater", "[interval_merge]") {
	std::mt19937_64 rng(32);
	for (int iter = 0; iter < 1000; ++iter) {
		std::size_t const n = std::size_t(rng() % 12);
		auto w = atl_test::random_interval_cost(n, rng, 5);
		for (auto &row : w)
			for (auto &x : row) x = -x;
		auto const cost = [&w](uint32 l, uint32 r) { return w[l][r]; };
		check(n, atl::optimal_interval_merge(n, cost, std::greater<>{}), brute(n, cost, std::greater<>{}));
	}
}

TEST_CASE("optimal_interval_merge on stone merging at scale", "[interval_merge]") {
	// the merge cost of [l, r) is the weight of its stones
	std::mt19937_64 rng(33);
	std::size_t const n = 300;
	std::vector<int64> prefix(n + 1, 0);
	for (std::size_t i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + 1 + int64(rng() % 1000);
	auto const cost = [&prefix](uint32 l, uint32 r) { return prefix[r] - prefix[l]; };
	check(n, atl::optimal_interval_merge(n, cost), brute(n, cost));
}

TEST_CASE("optimal_interval_merge edge cases", "[interval_merge]") {
	std::size_t calls = 0;
	auto counting = [&calls](uint32 l, uint32 r) {
		++calls;
		return int64(r - l);
	};
	CHECK(atl::optimal_interval_merge(0, counting).table.empty());
	auto const one = atl::optimal_interval_merge(1, counting);
	CHECK(one.value(0, 1) == 0);
	CHECK(calls == 0);
	// four unit stones: ((1 1)(1 1)) costs 2 + 2 + 4
	auto const four = atl::optimal_interval_merge(4, counting);
	CHECK(four.value(0, 4) == 8);
	CHECK(calls == 6); // once per segment of two items or more
}

TEST_CASE("interval_merge_result asserts its index range", "[interval_merge]") {
	auto const cost = [](uint32, uint32) { return int64(0); };
	auto const table = atl::optimal_interval_merge(3, cost);
	CHECK(atl_test::aborts([&] { (void)table.value(2, 2); }));
	CHECK(atl_test::aborts([&] { (void)table.value(0, 4); }));
}

namespace {
	/** @brief Replay `steps` from single items: each merges two live segments; the last leaves `[0, n)`; returns the summed cost. */
	template<std::invocable<uint32, uint32> Cost>
	int64 replay(std::size_t n, std::vector<atl::merge_step> const &steps, Cost const &cost) {
		REQUIRE(steps.size() == (n < 2 ? 0 : n - 1));
		std::vector<std::size_t> end_of(n + 1, 0); // end_of[l] = r while [l, r) is live
		for (std::size_t l = 0; l < n; ++l) end_of[l] = l + 1;
		int64 total = 0;
		for (auto const &[left, split, right] : steps) {
			REQUIRE(left < split);
			REQUIRE(split < right);
			REQUIRE(end_of[left] == split);
			REQUIRE(end_of[split] == right);
			end_of[left] = right;
			end_of[split] = 0;
			total += cost(left, right);
		}
		if (n >= 1) REQUIRE(end_of[0] == n);
		return total;
	}
}

namespace {
	/** @brief Whether every step splits at the leftmost best `m` of the brute-force table, as documented. */
	template<typename Compare = std::less<>>
		requires std::strict_weak_order<Compare const &, int64, int64>
	bool takes_leftmost_splits(std::vector<atl::merge_step> const &steps, atl_test::matrix const &dp, Compare comp = {}) {
		for (auto const &[left, split, right] : steps) {
			uint32 best = left + 1;
			for (uint32 m = left + 2; m < right; ++m)
				if (comp(dp[left][m] + dp[m][right], dp[left][best] + dp[best][right])) best = m;
			if (split != best) return false;
		}
		return true;
	}
}

TEST_CASE("optimal_merge_order rebuilds an optimal merge tree from the table", "[interval_merge]") {
	std::mt19937_64 rng(33);
	for (int iter = 0; iter < 3000; ++iter) {
		std::size_t const n = std::size_t(rng() % 14);
		bool const is_max = iter % 4 == 3;
		auto w = atl_test::random_interval_cost(n, rng, iter % 3 == 0 ? 1 : 6);
		if (is_max)
			for (auto &row : w)
				for (auto &x : row) x = -x;
		auto const cost = [&w](uint32 l, uint32 r) { return w[l][r]; };
		INFO("n=" << n);
		if (is_max) {
			auto const table = atl::optimal_interval_merge(n, cost, std::greater<>{});
			static_assert(std::same_as<decltype(table), atl::interval_merge_result<int64, std::greater<>> const>);
			auto const steps = atl::optimal_merge_order(table); // under the stored std::greater
			int64 const total = replay(n, steps, cost);
			auto const want = brute(n, cost, std::greater<>{});
			if (n >= 1) CHECK(total == want[0][n]);
			CHECK(takes_leftmost_splits(steps, want, std::greater<>{}));
		} else {
			auto const table = atl::optimal_interval_merge(n, cost);
			auto const steps = atl::optimal_merge_order(table);
			int64 const total = replay(n, steps, cost);
			auto const want = brute(n, cost);
			if (n >= 1) CHECK(total == want[0][n]);
			CHECK(takes_leftmost_splits(steps, want));
		}
	}
}

namespace {
	/**
	 * @brief Four items whose merges cost `-big`, `0`, `big` for two, three and
	 *        four items: the best tree merges two pairs (`-2 big` before the
	 *        last merge, `-big` in all), which only a wide child sum sees.
	 */
	template<typename Value>
	void check_wide_child_sums(Value big) {
		auto const cost = [big](uint32 l, uint32 r) {
			uint32 const length = r - l;
			return length == 2 ? Value(-big) : length == 3 ? Value(0)
														   : big;
		};
		auto const table = atl::optimal_interval_merge(4, cost);
		CHECK(table.value(0, 2) == Value(-big));
		CHECK(table.value(0, 3) == Value(-big));
		CHECK(table.value(0, 4) == Value(-big));
		auto const steps = atl::optimal_merge_order(table);
		REQUIRE(steps.size() == 3);
		CHECK(steps.back().split == 2);
	}
}

TEST_CASE("optimal_interval_merge forms child sums wide", "[interval_merge]") {
	// int16: -30000 + -30000 leaves int16; int64: -6e18 + -6e18 leaves int64
	check_wide_child_sums<atl::int16>(atl::int16(30000));
	check_wide_child_sums<int64>(int64(6'000'000'000'000'000'000));
	// the maximizing mirror image
	auto const cost = [](uint32 l, uint32 r) {
		uint32 const length = r - l;
		return length == 2 ? int64(6'000'000'000'000'000'000) : length == 3 ? int64(0)
																			: int64(-6'000'000'000'000'000'000);
	};
	auto const table = atl::optimal_interval_merge(4, cost, std::greater<>{});
	CHECK(table.value(0, 4) == int64(6'000'000'000'000'000'000));
	CHECK(atl::optimal_merge_order(table).back().split == 2);
}
