#include <atl/dp/segmentation.hpp>
#include <atl/experimental/dp/segmentation_variants.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <vector>
#include "../expect_abort.hpp"
#include "monge_costs.hpp"

namespace ex = atl::experimental;
using atl::int64;
using atl::uint32;

namespace {
	/** @brief The certifying oracle: the O(n^2) recurrence, smallest best decision. */
	template<typename Cost, typename Compare = std::less<>>
		requires std::invocable<Cost const &, uint32, uint32> && std::strict_weak_order<Compare const &, int64, int64>
	atl::segmentation_result<int64> brute(std::size_t n, Cost const &cost, Compare comp = {}) {
		atl::segmentation_result<int64> r{std::vector<int64>(n + 1, 0), std::vector<uint32>(n + 1, 0)};
		for (std::size_t i = 1; i <= n; ++i) {
			r.value[i] = cost(0, uint32(i));
			for (std::size_t j = 1; j < i; ++j)
				if (int64 const v = r.value[j] + cost(uint32(j), uint32(i)); comp(v, r.value[i])) {
					r.value[i] = v;
					r.decision[i] = uint32(j);
				}
		}
		return r;
	}

	template<typename Cost, typename Compare = std::less<>>
		requires std::invocable<Cost const &, uint32, uint32> && std::strict_weak_order<Compare const &, int64, int64>
	void check_all(std::size_t n, Cost const &cost, atl::segmentation_result<int64> const &want, Compare comp = {}) {
		auto const check = [&](char const *name, atl::segmentation_result<int64> const &got) {
			INFO(name << " n=" << n);
			CHECK(got.value == want.value);
			CHECK(got.decision == want.decision);
		};
		check("library", atl::optimal_segmentation(n, cost, comp));
		check("by halving divide and conquer", ex::optimal_segmentation_by_halving_divide_and_conquer(n, cost, comp));
		check("by halving SMAWK", ex::optimal_segmentation_by_halving_smawk(n, cost, comp));
		check("by SMAWK blocks", ex::optimal_segmentation_by_smawk_blocks(n, cost, comp));
	}
}

TEST_CASE("segmentation contenders match the O(n^2) recurrence on random Monge costs", "[segmentation]") {
	std::mt19937_64 rng(11);
	for (int iter = 0; iter < 4000; ++iter) {
		std::size_t const n = std::size_t(rng() % 16);
		auto const a = atl_test::random_monge(n + 1, n + 1, rng, iter % 3 == 0 ? 1 : 5, iter % 2 == 0 ? 0 : 12);
		auto const cost = [&a](uint32 j, uint32 i) { return a[j][i]; };
		check_all(n, cost, brute(n, cost));
	}
}

TEST_CASE("segmentation maximizes an inverse-Monge cost with std::greater", "[segmentation]") {
	std::mt19937_64 rng(12);
	for (int iter = 0; iter < 1500; ++iter) {
		std::size_t const n = std::size_t(rng() % 14);
		auto a = atl_test::random_monge(n + 1, n + 1, rng, 4, 8);
		for (auto &row : a)
			for (auto &x : row) x = -x;
		auto const cost = [&a](uint32 j, uint32 i) { return a[j][i]; };
		check_all(n, cost, brute(n, cost, std::greater<>{}), std::greater<>{});
	}
}

TEST_CASE("segmentation on a convex length cost at scale", "[segmentation]") {
	// w(j, i) = (S_i - S_j - L)^2 over prefix sums of positive weights: the
	// classic "cut into lines near a target length" shape
	std::mt19937_64 rng(13);
	for (std::size_t const n : {1u, 2u, 700u, 3000u}) {
		std::vector<int64> s(n + 1, 0);
		for (std::size_t i = 1; i <= n; ++i) s[i] = s[i - 1] + 1 + int64(rng() % 20);
		auto const cost = [&s](uint32 j, uint32 i) {
			int64 const x = s[i] - s[j] - 60;
			return x * x;
		};
		check_all(n, cost, brute(n, cost));
	}
}

TEST_CASE("segmentation edge cases", "[segmentation]") {
	auto const cost = [](uint32 j, uint32 i) { return int64(i - j) * int64(i - j); };
	auto const empty = atl::optimal_segmentation(0, cost);
	CHECK(empty.value == std::vector<int64>{0});
	CHECK(empty.decision == std::vector<uint32>{0});
	auto const one = atl::optimal_segmentation(1, cost);
	CHECK(one.value == std::vector<int64>{0, 1});
	// a squared length is best cut into unit segments
	auto const five = atl::optimal_segmentation(5, cost);
	CHECK(five.value == std::vector<int64>{0, 1, 2, 3, 4, 5});
	CHECK(five.decision == std::vector<uint32>{0, 0, 1, 2, 3, 4});
	// a constant cost per segment ties everywhere, so the smallest j wins:
	// one segment
	auto const flat = atl::optimal_segmentation(4, [](uint32, uint32) { return int64(7); });
	CHECK(flat.decision == std::vector<uint32>{0, 0, 0, 0, 0});
	// values near the top of the range: no infinity is formed
	auto const high = atl::optimal_segmentation(3, [](uint32, uint32) { return int64(1) << 61; });
	CHECK(high.value[3] == int64(1) << 61);
}

TEST_CASE("segmentation rejects a size past the index range", "[segmentation]") {
	auto const cost = [](uint32, uint32) { return 0; };
	// 2^32 - 1 on 64-bit targets, SIZE_MAX on 32-bit ones: past the bound on both
	CHECK(atl_test::aborts([&] { (void)atl::optimal_segmentation(std::size_t(std::numeric_limits<uint32>::max()), cost); }));
	CHECK(atl_test::aborts([&] { (void)atl::optimal_segmentation(std::size_t(std::numeric_limits<uint32>::max()) - 1, cost); }));
}

TEST_CASE("segmentation boundaries rebuild an optimal segmentation", "[segmentation]") {
	std::mt19937_64 rng(13);
	for (int iter = 0; iter < 3000; ++iter) {
		std::size_t const n = std::size_t(rng() % 16);
		auto const a = atl_test::random_monge(n + 1, n + 1, rng, iter % 3 == 0 ? 1 : 5, iter % 2 == 0 ? 0 : 12);
		auto const cost = [&a](uint32 j, uint32 i) { return a[j][i]; };
		auto const r = atl::optimal_segmentation(n, cost);
		auto const cuts = r.boundaries();
		REQUIRE(cuts.front() == 0);
		REQUIRE(cuts.back() == n);
		int64 total = 0;
		for (std::size_t t = 0; t + 1 < cuts.size(); ++t) {
			REQUIRE(cuts[t] < cuts[t + 1]);
			total += cost(cuts[t], cuts[t + 1]);
		}
		CHECK(total == brute(n, cost).value[n]);
	}
	CHECK(atl::segmentation_result<int64>{}.boundaries().empty());
}

TEST_CASE("optimal_segmentation forms candidate totals wide", "[segmentation]") {
	// dp[1] + cost(1, 2) = 2 INT64_MAX leaves int64; the answer cost(0, 2) = 0 does not
	auto const cost = [](uint32 j, uint32 i) { return j == 0 && i == 2 ? int64(0) : std::numeric_limits<int64>::max(); };
	auto const check = [](auto const &r) {
		CHECK(r.value[1] == std::numeric_limits<int64>::max());
		CHECK(r.value[2] == 0);
		CHECK(r.decision[2] == 0);
	};
	check(atl::optimal_segmentation(2, cost));
	check(ex::optimal_segmentation_by_halving_divide_and_conquer(2, cost));
	check(ex::optimal_segmentation_by_halving_smawk(2, cost));
	check(ex::optimal_segmentation_by_smawk_blocks(2, cost));
	// maximizing: the mirror image at INT64_MIN
	auto const negated = [](uint32 j, uint32 i) { return j == 0 && i == 2 ? int64(0) : std::numeric_limits<int64>::min(); };
	CHECK(atl::optimal_segmentation(2, negated, std::greater<>{}).value[2] == 0);
}
