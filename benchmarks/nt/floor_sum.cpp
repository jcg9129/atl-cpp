/** @file
 * @brief Throughput of 1e5 reduced floor sums: library narrows divisions when
 * values shrink, wide divisions keep 128 bits throughout, entry dispatch
 * chooses 64/128 bits once. Shapes: n,m<1e9; n,m<1e18; n<1e18,m<1e9.
 * Separate rows measure first-residue searches and a staircase sum monoid.
 * Xeon Platinum 8358, GCC 15.2, RelWithDebInfo, 2026-10-15, -O2 / -O2
 * -march=native: n,m<1e18 library 64.2/64.2 ms, wide 92.3/93.1 ms,
 * entry dispatch 107/108 ms. Full table: docs/floor-sum.md.
 */
#include <atl/experimental/nt/floor_sum_variants.hpp>
#include <atl/nt/floor_sum.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using atl::int128;
using atl::int64;
using atl::uint64;

namespace {
	struct query {
		int64 n, m, a, b;
	};

	std::vector<query> make_queries(int64 n_bound, int64 m_bound, int count, uint64 seed) {
		atl_assert(n_bound >= 1 && m_bound >= 1 && count >= 0);
		std::mt19937_64 rng(seed);
		std::vector<query> queries(static_cast<std::size_t>(count));
		for (auto &q : queries) {
			q.n = int64(rng() % uint64(n_bound));
			q.m = int64(rng() % uint64(m_bound)) + 1;
			q.a = int64(rng() % uint64(q.m));
			q.b = int64(rng() % uint64(q.m));
		}
		return queries;
	}
}

TEST_CASE("floor_sum: reduced-loop width", "[floor_sum]") {
	constexpr int count = 100'000;

	struct shape {
		std::string label;
		int64 n_bound, m_bound;
	};

	std::array<shape, 3> const shapes{{{"n,m<1e9", 1'000'000'000, 1'000'000'000},
									   {"n,m<1e18", 1'000'000'000'000'000'000, 1'000'000'000'000'000'000},
									   {"n<1e18,m<1e9", 1'000'000'000'000'000'000, 1'000'000'000}}};
	for (auto const &s : shapes) {
		auto const queries = make_queries(s.n_bound, s.m_bound, count, 42);
		for (auto const &q : queries) {
			int128 const want = atl::floor_sum(q.n, q.m, q.a, q.b);
			REQUIRE(atl::experimental::floor_sum_by_wide_divisions(q.n, q.m, q.a, q.b) == want);
			REQUIRE(atl::experimental::floor_sum_by_entry_dispatch(q.n, q.m, q.a, q.b) == want);
		}
		auto const run = [&]<typename Sum>(Sum sum)
			requires std::is_invocable_r_v<int128, Sum &, int64, int64, int64, int64>
		{
			atl::uint128 acc = 0;
			for (auto const &q : queries) acc += atl::uint128(sum(q.n, q.m, q.a, q.b));
			return uint64(acc);
		};
		BENCHMARK("library " + s.label) {
			return run([](int64 n, int64 m, int64 a, int64 b) { return atl::floor_sum(n, m, a, b); });
		};
		BENCHMARK("by wide divisions " + s.label) {
			return run([](int64 n, int64 m, int64 a, int64 b) {
				return atl::experimental::floor_sum_by_wide_divisions(n, m, a, b);
			});
		};
		BENCHMARK("by entry dispatch " + s.label) {
			return run([](int64 n, int64 m, int64 a, int64 b) {
				return atl::experimental::floor_sum_by_entry_dispatch(n, m, a, b);
			});
		};
	}
}

TEST_CASE("floor_sum: the other Euclid-like recursions", "[floor_sum]") {
	constexpr int count = 100'000;
	std::mt19937_64 rng(7);
	std::vector<std::array<uint64, 5>> residue_queries(count);
	for (auto &q : residue_queries) {
		uint64 const m = rng() % 1'000'000'000'000'000'000 + 1, lo = rng() % m;
		q = {m, rng() % m, rng() % m, lo, (std::min)(m - 1, lo + rng() % 1000)};
	}
	for (auto const &q : residue_queries) {
		auto const result = atl::first_residue_in_range(q[0], q[1], q[2], q[3], q[4]);
		uint64 const divisor = std::gcd(q[0], q[1]);
		atl::uint128 const first =
			atl::uint128(q[3]) + (atl::uint128(q[2] % divisor) + divisor - q[3] % divisor) % divisor;
		REQUIRE(result.has_value() == (first <= q[4]));
		if (result) {
			uint64 const residue = uint64((atl::uint128(q[1]) * *result + q[2]) % q[0]);
			REQUIRE((q[3] <= residue && residue <= q[4]));
		}
	}

	BENCHMARK("first_residue_in_range m<1e18") {
		uint64 acc = 0;
		for (auto const &q : residue_queries)
			acc += atl::first_residue_in_range(q[0], q[1], q[2], q[3], q[4]).value_or(1);
		return acc;
	};

	struct element {
		atl::uint128 rights, ups, sum;
	};

	auto const combine = [](element const &l, element const &r) {
		return element{l.rights + r.rights, l.ups + r.ups, l.sum + r.sum + l.ups * r.rights};
	};
	auto const queries = make_queries(1'000'000'000, 1'000'000'000, count, 43);
	for (auto const &q : queries) {
		auto const got = atl::floor_staircase_product(uint64(q.n), uint64(q.m), uint64(q.a), uint64(q.b),
													  element{0, 1, 0}, element{1, 0, 0}, element{0, 0, 0}, combine);
		REQUIRE(got.sum
				== atl::uint128(atl::floor_sum(q.n, q.m, q.a, q.b) + (atl::int128(q.a) * q.n + q.b) / q.m - q.b / q.m));
	}

	BENCHMARK("floor_staircase_product Σf n,m<1e9") {
		atl::uint128 acc = 0;
		for (auto const &q : queries)
			acc += atl::floor_staircase_product(uint64(q.n), uint64(q.m), uint64(q.a), uint64(q.b), element{0, 1, 0},
												element{1, 0, 0}, element{0, 0, 0}, combine)
					   .sum;
		return uint64(acc);
	};
}
