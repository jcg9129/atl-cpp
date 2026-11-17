/**
 * @file
 * @brief Races static min/gcd folds: library sparse_table overlaps windows, disjoint_sparse_table
 * joins disjoint prefix/suffix folds, direct_rmq uses block masks, segment_tree traverses nodes.
 * Measures construction and query throughput separately: n=1e6/1e7, 1e7 uniform queries
 * and lengths <=64/1000. Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions off,
 * 2026-10-15, -O2 / -O2 -march=native (also -g -DNDEBUG -std=c++23):
 * n=1e6 uniform min queries take 173 / 149 ms for sparse_table, 373 / 311 for direct_rmq.
 * Complete measurements are in docs/sparse-table.md.
 */

#include <atl/ds/direct_rmq.hpp>
#include <atl/ds/segment_tree.hpp>
#include <atl/ds/sparse_table.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <bit>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using atl::uint32;
using atl::uint64;

// Static range folds, the chooser in docs/sparse-table.md: sparse_table vs
// disjoint_sparse_table vs direct_rmq vs segment_tree on min and gcd. The contenders that
// appended sparse-table rows once or used 32-wide direct_rmq blocks both
// lost in the 2026-10-15 measurements and were removed.
namespace {
	struct min_op {
		uint32 operator()(uint32 a, uint32 b) const { return std::min(a, b); }
	};

	struct gcd_op {
		uint32 operator()(uint32 a, uint32 b) const { return std::gcd(a, b); }
	};

	/** @brief q queries over [0, n): uniform, or with lengths in [1, max_len]. */
	std::vector<std::pair<uint32, uint32>> make_queries(std::size_t n, std::size_t q, std::size_t max_len, uint64 seed) {
		std::mt19937_64 rng(seed);
		std::vector<std::pair<uint32, uint32>> qs(q);
		for (auto &[l, r] : qs)
			if (max_len == 0) {
				std::size_t a = rng() % n, b = rng() % n;
				if (a > b) std::swap(a, b);
				l = uint32(a), r = uint32(b + 1);
			} else {
				std::size_t const len = 1 + rng() % max_len, a = rng() % (n - len + 1);
				l = uint32(a), r = uint32(a + len);
			}
		return qs;
	}

	template<class F>
		requires std::is_invocable_r_v<uint32, F &, uint32, uint32>
	uint64 sum_over(std::vector<std::pair<uint32, uint32>> const &qs, F f) {
		uint64 acc = 0;
		for (auto const &[l, r] : qs) acc += f(l, r);
		return acc;
	}
}

TEST_CASE("static range folds: chooser", "[sparse_table]") {
	for (std::size_t const n : {std::size_t(1'000'000), std::size_t(10'000'000)}) {
		std::string const tag = n == 1'000'000 ? "n=1e6" : "n=1e7";
		std::mt19937_64 rng(5);
		std::vector<uint32> a(n), g(n);
		for (auto &x : a) x = uint32(rng());
		for (auto &x : g) x = uint32(6 * (1 + rng() % 100'000));

		atl::sparse_table<uint32, min_op> const sp(a);
		atl::disjoint_sparse_table<uint32, min_op> const dsp(a);
		atl::direct_rmq const rmq(a);
		atl::segment_tree<uint32, min_op> const seg(a, std::numeric_limits<uint32>::max());

		auto const check = make_queries(n, 100'000, 0, 99);
		for (auto const &[l, r] : check) {
			uint32 const want = sp.fold(l, r);
			if (dsp.fold(l, r) != want || rmq.fold(l, r) != want || seg.fold(l, r) != want)
				FAIL("range-min contenders disagree");
		}

		bool const is_large = n != 1'000'000;
		if (!is_large) {
			BENCHMARK(tag + " build | sparse_table min") { return atl::sparse_table<uint32, min_op>(a).fold(0, 1); };
			BENCHMARK(tag + " build | disjoint_sparse_table min") { return atl::disjoint_sparse_table<uint32, min_op>(a).fold(0, 1); };
			BENCHMARK(tag + " build | direct_rmq") { return atl::direct_rmq(a).position(0, 1); };
			BENCHMARK(tag + " build | segment_tree min") {
				return atl::segment_tree<uint32, min_op>(a, std::numeric_limits<uint32>::max()).fold();
			};
		}

		// n=1e7 (a 1 GB table per sparse structure) only on the uniform shape, to bound the run
		for (std::size_t const max_len : {std::size_t(0), std::size_t(64), std::size_t(1000)}) {
			if (is_large && max_len != 0) continue;
			auto const qs = make_queries(n, 10'000'000, max_len, 17);
			std::string const s =
				tag + " 1e7 queries " + (max_len == 0 ? "uniform" : "len<=" + std::to_string(max_len)) + " | ";
			BENCHMARK(s + "sparse_table min") {
				return sum_over(qs, [&](uint32 l, uint32 r) { return sp.fold(l, r); });
			};
			BENCHMARK(s + "disjoint_sparse_table min") {
				return sum_over(qs, [&](uint32 l, uint32 r) { return dsp.fold(l, r); });
			};
			BENCHMARK(s + "direct_rmq") {
				return sum_over(qs, [&](uint32 l, uint32 r) { return rmq.fold(l, r); });
			};
			if (max_len == 0) {
				BENCHMARK(s + "segment_tree min") {
					return sum_over(qs, [&](uint32 l, uint32 r) { return seg.fold(l, r); });
				};
			}
		}

		if (n != 1'000'000) continue;
		atl::disjoint_sparse_table<uint32, gcd_op> const dg(g);
		atl::segment_tree<uint32, gcd_op> const sg(g, uint32(0));
		auto const qs = make_queries(n, 10'000'000, 0, 23);
		BENCHMARK(tag + " build | disjoint_sparse_table gcd") { return atl::disjoint_sparse_table<uint32, gcd_op>(g).fold(0, 1); };
		BENCHMARK(tag + " build | segment_tree gcd") { return atl::segment_tree<uint32, gcd_op>(g, uint32(0)).fold(); };
		BENCHMARK(tag + " 1e7 queries uniform | disjoint_sparse_table gcd") {
			return sum_over(qs, [&](uint32 l, uint32 r) { return dg.fold(l, r); });
		};
		BENCHMARK(tag + " 1e7 queries uniform | segment_tree gcd") {
			return sum_over(qs, [&](uint32 l, uint32 r) { return sg.fold(l, r); });
		};
	}
}
