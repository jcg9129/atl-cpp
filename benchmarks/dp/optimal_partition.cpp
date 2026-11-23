/**
 * @file
 * @brief atl::optimal_partition (per layer, divide and conquer until Knuth's
 * windows get small, then Knuth's layered bound) against contenders that
 * change only how a layer is solved: Knuth's bound on every layer, divide
 * and conquer on every layer, divide and conquer floored at the previous
 * layer's argmins, and SMAWK on every layer but the last. Throughput of one
 * whole curve (exactly 1 .. K segments) for n = 1e3, 5e3, 2e4 and K = 2, 20,
 * 100, n/10, n, over prefix sums S: "cheap" = (S_i - S_j)^2, "pow1.5" = |S_i
 * - S_j - L|^1.5 (a sqrt). Also: the switch budget's calibration, the
 * constant-factor steps of the Knuth engine, and two large workloads (a
 * convex cost, and a stateful two-pointer cost).
 *
 * Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-11-01, plain -O2 / -march=native, "layer engines" rows: the floor
 * ties the library within 0.92-1.16x with no consistent direction (n = 2e4,
 * K = n, cheap: 1473 / 1460 ms against 1538 / 1521), SMAWK ties at K = 2 and
 * is 1.35-5.6x slower from K = 20 up (n = 2e4, K = 2000, cheap: 1344 / 1474
 * against 385 / 384). Full table: docs/dp-optimization.md. The race against
 * Knuth's bound and divide and conquer on every layer, the calibration, the
 * Knuth tuning steps and the large workloads have no current record (not
 * measured yet); the experimental header keeps the race's verdict.
 */
#include <atl/dp/concepts.hpp>
#include <atl/dp/optimal_partition.hpp>
#include <atl/experimental/dp/optimal_partition_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <format>
#include <functional>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	/** @brief Random prefix sums `s[0 .. n]` with steps in `[0, 1000)`. */
	std::vector<int64> prefix_sums(std::size_t n, std::mt19937_64 &rng) {
		std::vector<int64> s(n + 1, 0);
		for (std::size_t i = 1; i <= n; ++i) s[i] = s[i - 1] + int64(rng() % 1000);
		return s;
	}

	/** @brief optimal_partition's curve with another per-layer engine choice (the wrapper adapts the choice). */
	template<atl::detail::partition_layer_choice Choose, typename Cost>
		requires atl::additive_cost<Cost, std::less<>>
	std::vector<int64> optimal_partition_by_choice(Choose const &choose, uint32 n, uint32 segments, Cost &cost) {
		std::vector<int64> best;
		std::vector<int64> previous(n + 1), current(n + 1);
		for (uint32 i = 1; i <= n; ++i) previous[i] = cost(0, i);
		best.push_back(previous[n]);
		std::vector<uint32> argmins(2 * (n + 1));
		auto argmin_row = [&argmins, n](uint32 k) { return argmins.data() + (k & 1) * (n + 1); };
		auto after_layer = [&best, n](std::vector<int64> const &layer) { best.push_back(layer[n]); };
		std::less<> const comp;
		atl::detail::partition_layers(n, segments, true, cost, comp, previous, current, argmin_row, after_layer,
									  choose);
		return best;
	}

	/**
	 * @brief Knuth's layered bound on every layer as usually written: a full
	 *        `(segments + 1) x (n + 2)` table of `std::size_t` argmins and a
	 *        full table of values.
	 */
	template<typename Cost>
		requires atl::additive_cost<Cost, std::less<>>
	std::vector<int64> optimal_partition_by_knuth_tables(std::size_t n, std::size_t segments, Cost &cost) {
		std::vector<std::vector<std::size_t>> opt(segments + 1, std::vector<std::size_t>(n + 2, 0));
		std::vector<std::vector<int64>> dp(segments + 1, std::vector<int64>(n + 1, 0));
		for (std::size_t i = 1; i <= n; ++i) dp[1][i] = cost(0, uint32(i));
		std::vector<int64> best{dp[1][n]};
		for (std::size_t k = 2; k <= segments; ++k) {
			for (std::size_t i = n; i >= (k == segments ? n : k); --i) {
				std::size_t const lo = (std::max)(opt[k - 1][i], k - 1);
				std::size_t const hi = (std::max)(lo, i == n ? n - 1 : (std::min)(opt[k][i + 1], i - 1));
				std::size_t arg = lo;
				int64 value = dp[k - 1][lo] + cost(uint32(lo), uint32(i));
				for (std::size_t j = lo + 1; j <= hi; ++j) {
					int64 const v = dp[k - 1][j] + cost(uint32(j), uint32(i));
					if (v < value) value = v, arg = j;
				}
				dp[k][i] = value;
				opt[k][i] = arg;
			}
			best.push_back(dp[k][n]);
		}
		return best;
	}

	/**
	 * @brief Knuth's layered bound on every layer with two rolling rows of
	 *        argmins (of type `Index`) and of values, the upper bound
	 *        reloaded from the argmin row.
	 */
	template<std::unsigned_integral Index, typename Cost>
		requires atl::additive_cost<Cost, std::less<>>
	std::vector<int64> optimal_partition_by_knuth_rolling_rows(std::size_t n, std::size_t segments, Cost &cost) {
		std::vector<Index> below(n + 2, 0), argmin(n + 2, 0);
		std::vector<int64> previous(n + 1, 0), current(n + 1, 0);
		for (std::size_t i = 1; i <= n; ++i) previous[i] = cost(0, uint32(i));
		std::vector<int64> best{previous[n]};
		Index const last = Index(n);
		for (Index k = 2; k <= Index(segments); ++k) {
			for (Index i = last; i >= (k == Index(segments) ? last : k); --i) {
				Index const lo = (std::max)(below[i], Index(k - 1));
				Index const hi = (std::max)(lo, i == last ? Index(last - 1) : (std::min)(argmin[i + 1], Index(i - 1)));
				Index arg = lo;
				int64 value = previous[lo] + cost(uint32(lo), uint32(i));
				for (Index j = lo + 1; j <= hi; ++j) {
					int64 const v = previous[j] + cost(uint32(j), uint32(i));
					if (v < value) value = v, arg = j;
				}
				current[i] = value;
				argmin[i] = arg;
			}
			std::swap(previous, current);
			std::swap(below, argmin);
			best.push_back(previous[n]);
		}
		return best;
	}

	/**
	 * @brief Distinct values in the queried segment `[j, i)`, kept in a
	 *        sliding window that moves its ends to each query (a stateful
	 *        two-pointer cost).
	 */
	struct distinct_window {
		std::vector<uint32> const *items;
		std::vector<uint32> count;
		uint32 lo = 0, hi = 0;
		int64 value = 0;

		int64 operator()(uint32 j, uint32 i) {
			std::vector<uint32> const &a = *items;
			while (hi < i) value += count[a[hi++]]++ == 0;
			while (lo > j) value += count[a[--lo]]++ == 0;
			while (hi > i) value -= --count[a[--hi]] == 0;
			while (lo < j) value -= --count[a[lo++]] == 0;
			return value;
		}
	};
}

// Rows too slow to run at all (K n log n past ~2e9 evaluations) are left out.
TEST_CASE("optimal_partition race", "[partition][race]") {
	std::mt19937_64 rng(2);
	std::vector<int64> const s = prefix_sums(20'000, rng);
	for (uint32 const n : {1'000u, 5'000u, 20'000u}) {
		int64 const target = s[n] / 50;
		auto cheap = [&s](uint32 j, uint32 i) {
			int64 const d = s[i] - s[j];
			return d * d;
		};
		auto pow15 = [&s, target](uint32 j, uint32 i) {
			double const d = std::abs(double(s[i] - s[j] - target));
			return int64(d * std::sqrt(d));
		};
		std::vector<uint32> segment_counts{2, 20, 100, n / 10, n};
		if (n == 1'000) segment_counts = {2, 20, 100, 1'000};
		for (uint32 const segments : segment_counts) {
			auto run = [&]<typename Cost>
				requires atl::additive_cost<Cost, std::less<>>
			(char const *name, Cost &cost) {
				auto const label = [&](char const *who) { return std::format("{} {} n={} K={}", who, name, n, segments); };
				auto const want = atl::optimal_partition(n, segments, cost);
				REQUIRE(atl::experimental::optimal_partition_by_knuth_bound(n, segments, cost) == want);
				BENCHMARK(label("library")) { return atl::optimal_partition(n, segments, cost).back(); };
				BENCHMARK(label("by Knuth bound")) {
					return atl::experimental::optimal_partition_by_knuth_bound(n, segments, cost).back();
				};
				if (double(segments) * n * std::log2(n) < 2e9)
					BENCHMARK(label("by divide and conquer")) {
						return atl::experimental::optimal_partition_by_divide_and_conquer(n, segments, cost).back();
					};
			};
			run("cheap", cheap);
			run("pow1.5", pow15);
		}
	}
}

// The floored divide and conquer and SMAWK (opt-in: out-of-order evaluation)
// on the race's shapes.
TEST_CASE("optimal_partition layer engines", "[partition][engines]") {
	std::mt19937_64 rng(2);
	std::vector<int64> const s = prefix_sums(20'000, rng);
	for (uint32 const n : {1'000u, 5'000u, 20'000u}) {
		int64 const target = s[n] / 50;
		auto cheap = [&s](uint32 j, uint32 i) {
			int64 const d = s[i] - s[j];
			return d * d;
		};
		auto pow15 = [&s, target](uint32 j, uint32 i) {
			double const d = std::abs(double(s[i] - s[j] - target));
			return int64(d * std::sqrt(d));
		};
		std::vector<uint32> segment_counts{2, 20, 100, n / 10, n};
		if (n == 1'000) segment_counts = {2, 20, 100, 1'000};
		for (uint32 const segments : segment_counts) {
			auto run = [&]<typename Cost>
				requires atl::additive_cost<Cost, std::less<>>
			(char const *name, Cost &cost) {
				auto const label = [&](char const *who) {
					return std::format("optimal_partition {} {} n={} K={}", who, name, n, segments);
				};
				auto const want = atl::optimal_partition(n, segments, cost);
				// The cheap cost (d * d) is exactly Monge, so every contender returns the same optimum. The pow1.5
				// cost truncates d^1.5 to an integer and is only approximately Monge: divide and conquer and SMAWK
				// may then pick argmins that differ by rounding units, so the totals agree only to a few units.
				bool const is_exact_monge = std::string_view(name) == "cheap";
				auto const agrees = [&](std::vector<int64> const &got) {
					if (is_exact_monge) return got == want;
					if (got.size() != want.size()) return false;
					for (std::size_t k = 0; k < got.size(); ++k) {
						int64 const slack = (std::max)(int64(4), want[k] / 1'000'000'000);
						if (got[k] - want[k] > slack || want[k] - got[k] > slack) return false;
					}
					return true;
				};
				REQUIRE(agrees(atl::experimental::optimal_partition_by_floored_divide_and_conquer(n, segments, cost)));
				REQUIRE(agrees(atl::experimental::optimal_partition_by_smawk(n, segments, cost)));
				BENCHMARK(label("library")) { return atl::optimal_partition(n, segments, cost).back(); };
				BENCHMARK(label("by floored divide and conquer")) {
					return atl::experimental::optimal_partition_by_floored_divide_and_conquer(n, segments, cost).back();
				};
				BENCHMARK(label("by SMAWK")) { return atl::experimental::optimal_partition_by_smawk(n, segments, cost).back(); };
			};
			run("cheap", cheap);
			run("pow1.5", pow15);
		}
	}
}

// Where to switch: Knuth's bound once the previous layer's windows total at
// most `numerator / denominator` times rows * bit_width(rows), every variant
// through the same instantiation; "never" keeps divide and conquer on every
// layer but the last (one row either way).
TEST_CASE("optimal_partition switch calibration", "[partition][calibration]") {
	std::mt19937_64 rng(2);
	std::vector<int64> const s = prefix_sums(20'000, rng);
	for (uint32 const n : {5'000u, 20'000u}) {
		int64 const target = s[n] / 50;
		auto cheap = [&s](uint32 j, uint32 i) {
			int64 const d = s[i] - s[j];
			return d * d;
		};
		auto pow15 = [&s, target](uint32 j, uint32 i) {
			double const d = std::abs(double(s[i] - s[j] - target));
			return int64(d * std::sqrt(d));
		};
		for (uint32 const segments : {20u, 50u, 100u, 500u}) {
			auto run = [&]<typename Cost>
				requires atl::additive_cost<Cost, std::less<>>
			(char const *name, Cost &cost) {
				auto const label = [&](char const *who) { return std::format("{} {} n={} K={}", who, name, n, segments); };
				auto const never = [](uint64) { return uint64(0); };
				BENCHMARK(label("budget never")) { return optimal_partition_by_choice(never, n, segments, cost).back(); };
				for (auto const &[numerator, denominator] : {std::pair<uint64, uint64>{1, 4}, {1, 2}, {1, 1}, {2, 1}}) {
					atl::detail::adaptive_partition_choice const choose{numerator, denominator};
					BENCHMARK(label(std::format("budget {}/{}", numerator, denominator).c_str())) {
						return optimal_partition_by_choice(choose, n, segments, cost).back();
					};
				}
			};
			run("cheap", cheap);
			run("pow1.5", pow15);
		}
	}
}

// What each constant-factor step of the Knuth engine buys, every layer by
// Knuth: wide windows (K = 20) and narrow ones (K = n).
TEST_CASE("optimal_partition Knuth tuning", "[partition][tuning]") {
	std::mt19937_64 rng(3);
	std::size_t const n = 3'000;
	std::vector<int64> const s = prefix_sums(n, rng);
	auto cheap = [&s](uint32 j, uint32 i) {
		int64 const d = s[i] - s[j];
		return d * d;
	};
	for (std::size_t const segments : {std::size_t(20), n}) {
		auto const label = [&](char const *who) { return std::format("{} n={} K={}", who, n, segments); };
		auto const want = atl::experimental::optimal_partition_by_knuth_bound(n, segments, cheap);
		REQUIRE(optimal_partition_by_knuth_tables(n, segments, cheap) == want);
		REQUIRE(optimal_partition_by_knuth_rolling_rows<std::size_t>(n, segments, cheap) == want);
		REQUIRE(optimal_partition_by_knuth_rolling_rows<uint32>(n, segments, cheap) == want);
		BENCHMARK(label("by Knuth tables (size_t)")) { return optimal_partition_by_knuth_tables(n, segments, cheap).back(); };
		BENCHMARK(label("by Knuth rolling rows (size_t)")) {
			return optimal_partition_by_knuth_rolling_rows<std::size_t>(n, segments, cheap).back();
		};
		BENCHMARK(label("by Knuth rolling rows (uint32)")) {
			return optimal_partition_by_knuth_rolling_rows<uint32>(n, segments, cheap).back();
		};
		BENCHMARK(label("by Knuth bound (upper bound in a register)")) {
			return atl::experimental::optimal_partition_by_knuth_bound(n, segments, cheap).back();
		};
	}
}

// optimal_partition on large batches: a cheap convex cost, and a stateful
// two-pointer cost (distinct values in [j, i), maximized) whose window moves with the scans.
TEST_CASE("optimal_partition", "[partition]") {
	std::mt19937_64 rng(1);
	{
		std::size_t const n = 100'000, k = 20;
		std::vector<int64> s(n + 1, 0);
		for (std::size_t i = 1; i <= n; ++i) s[i] = s[i - 1] + int64(rng() % 1000);
		auto const cost = [&](uint32 j, uint32 i) {
			int64 const d = s[i] - s[j];
			return d * d;
		};
		REQUIRE(atl::experimental::optimal_partition_by_divide_and_conquer(n, k, cost) == atl::optimal_partition(n, k, cost));
		BENCHMARK("library: squared segment sum n=1e5 k=20") { return atl::optimal_partition(n, k, cost).back(); };
		BENCHMARK("by divide and conquer: squared segment sum n=1e5 k=20") {
			return atl::experimental::optimal_partition_by_divide_and_conquer(n, k, cost).back();
		};
	}
	{
		std::size_t const n = 35'000, k = 50;
		std::vector<uint32> a(n);
		for (auto &x : a) x = uint32(rng() % n);
		auto const library = [&] {
			distinct_window cost{&a, std::vector<uint32>(n, 0)};
			return atl::optimal_partition(n, k, cost, std::greater<>{}).back();
		};
		auto const by_divide_and_conquer = [&] {
			distinct_window cost{&a, std::vector<uint32>(n, 0)};
			return atl::experimental::optimal_partition_by_divide_and_conquer(n, k, cost, std::greater<>{}).back();
		};
		REQUIRE(library() == by_divide_and_conquer());
		BENCHMARK("library: distinct values (two-pointer) n=35000 k=50") { return library(); };
		BENCHMARK("by divide and conquer: distinct values (two-pointer) n=35000 k=50") { return by_divide_and_conquer(); };
	}
}
