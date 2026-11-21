#pragma once
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file
 * @brief The prefix-doubling suffix sort that lost the race to
 *        atl::suffix_array's SA-IS.
 */
/** @brief Alternative string algorithms retained for comparison and certification. */
namespace atl::experimental {
	/**
	 * @brief Prefix-doubling suffix sort with per-level counting sorts — the
	 *        raced baseline for atl::suffix_array's SA-IS (and its
	 *        differential oracle in the tests).
	 *
	 * RACE RESULT (x86-64, Xeon Platinum 8358, plain -O2, record
	 * benchmarks/strings/suffix_array.cpp 2026-10-15): doubling's
	 * sort alone against the library full build (SA-IS plus the rank inverse
	 * and Kasai LCP, about 20 ms of it on random text) at 10^6 symbols:
	 * random2 89.8 vs 87.0 ms, random26 85.7 vs 88.0, all-a 140 vs 27.0,
	 * abab 142 vs 33.7, Fibonacci 273 vs 45.0. On random text doubling's
	 * sort only ties the whole library build, so SA-IS sorts about 20 ms
	 * faster; on repetitive text, where the doubling rounds run to log n,
	 * doubling is 4-6x slower than the whole build. -march=native gives the same picture
	 * (all-a 141 vs 21.9, Fibonacci 277 vs 42.2).
	 *
	 * Same out-span contract as `atl::detail::suffix_sort`, so either engine
	 * drops into the other's call site: @p s ends with a unique smallest
	 * sentinel 0, and @p sa receives every suffix including the sentinel's
	 * (at rank 0).
	 * @param s the dense-mapped, sentinel-terminated text, values in
	 *        `[0, sigma)`
	 * @param sa the output: start positions in lexicographic order
	 * @param sigma the alphabet bound
	 * @pre `s` is nonempty, `s.size() == sa.size() < 2^31`, `s.back() == 0`
	 *      and no other symbol is 0, every s[i] < sigma
	 * @complexity O((n + sigma) log n)
	 */
	constexpr void suffix_sort_by_prefix_doubling(std::span<uint32 const> s, std::span<uint32> sa, uint32 sigma) {
		atl_assert(!s.empty() && s.size() == sa.size() && s.back() == 0);
		atl_assert(s.size() < (uint64(1) << 31)); // the doubling step half reaches n
		uint32 const n = uint32(s.size());
		std::vector<uint32> rank(s.begin(), s.end()), next_rank(n), count, tmp(n);
		// one stable counting sort of src (already ordered by the secondary
		// key) into sa, by the primary key
		auto const stable_sort_by = [&]<typename Key>(Key key, uint32 range)
			requires std::is_invocable_r_v<uint32, Key &, uint32>
		{
			count.assign(range, 0);
			for (uint32 i = 0; i < n; ++i) ++count[key(tmp[i])];
			uint32 sum = 0;
			for (auto &c : count) {
				uint32 const here = c;
				c = sum;
				sum += here;
			}
			for (uint32 i = 0; i < n; ++i) sa[count[key(tmp[i])]++] = tmp[i];
		};
		for (std::size_t index = 0; index < tmp.size(); ++index) tmp[index] = uint32(index);
		stable_sort_by([&](uint32 i) { return rank[i]; }, sigma);
		// name the symbols densely: every later round counting-sorts ranks
		// into n + 1 buckets, which raw symbols (up to sigma - 1) can exceed
		next_rank[sa[0]] = 0;
		for (uint32 r = 1, names = 0; r < n; ++r) {
			if (rank[sa[r - 1]] != rank[sa[r]]) ++names;
			next_rank[sa[r]] = names;
		}
		rank.swap(next_rank);
		for (uint32 half = 1;; half *= 2) {
			// sort by (rank[i], rank[i + half]) with ONE counting pass: the
			// order by the SECOND key needs no counting — it is the preceding
			// order shifted left by half, tail suffixes (empty second key)
			// first
			uint32 idx = 0;
			for (uint32 i = n - (std::min)(half, n); i < n; ++i) tmp[idx++] = i;
			for (uint32 r = 0; r < n; ++r)
				if (sa[r] >= half) tmp[idx++] = sa[r] - half;
			stable_sort_by([&](uint32 i) { return rank[i]; }, n + 1);
			auto const pair_of = [&](uint32 i) {
				return std::pair{rank[i], i + half < n ? rank[i + half] + 1 : 0};
			};
			uint32 names = 0;
			next_rank[sa[0]] = 0;
			for (uint32 r = 1; r < n; ++r) {
				if (pair_of(sa[r - 1]) < pair_of(sa[r])) ++names;
				next_rank[sa[r]] = names;
			}
			rank.swap(next_rank);
			if (names + 1 == n) break;
		}
	}
}
