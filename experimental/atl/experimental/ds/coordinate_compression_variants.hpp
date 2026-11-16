/**
 * @file
 * @brief Compression by pair sort, indirect index sort, or configurable radix digits;
 * bulk ranks by search/gallop dispatch or a separate sortedness pre-scan.
 * Pair sort is the certifying comparison oracle in tests. The pre-scan sorts the
 * entire query batch after a descent; the library preserves its answered prefix.
 * Result (benchmarks/ds/coordinate_compression.cpp, Xeon Platinum 8168, GCC 15.2,
 * -O2 / -O2 -march=native, 2026-11-03): mostly sorted m=1e6 q=4e6 takes
 * 260/266 ms by pre-scan versus 58.5/56.0 ms by fused merge. Other race records:
 * docs/coordinate-compression.md; index sort, gallop and radix8/16 not measured yet.
 */
#pragma once

#include <atl/ds/coordinate_compression.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/radix_sort.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental {
	/**
	 * @brief Pair sort of (value, position) records, the certifying comparison baseline.
	 * @note Projected results support static_cast to stored values. Algorithms reference the comparator,
	 *       whose storage is directly constructed from its rvalue argument.
	 * Pair sorting at n=131072 random takes 10.3/10.4 ms versus
	 * 4.0/3.9 ms for the recorded library (-O2 / -O2 -march=native,
	 * Xeon Platinum 8168, GCC 15.2, 2026-11-03, benchmarks/ds/coordinate_compression.cpp).
	 */
	template<std::ranges::input_range R, typename Proj = std::identity, typename Compare = std::ranges::less>
		requires std::invocable<Proj &, detail::forwarded_element_t<R>>
			  && std::is_constructible_v<Compare, Compare &&>
			  && requires { static_cast<detail::projected_value_t<R, Proj>>(std::invoke(std::declval<Proj &>(), std::declval<detail::forwarded_element_t<R>>())); }
			  && std::movable<detail::projected_value_t<R, Proj>>
			  && std::strict_weak_order<Compare const &, detail::projected_value_t<R, Proj> const &,
										detail::projected_value_t<R, Proj> const &>
	[[nodiscard]] constexpr compressed<detail::projected_value_t<R, Proj>, Compare> compress_by_pair_sort(R &&values, Proj proj = {}, Compare comp = {}) {
		using value_type = detail::projected_value_t<R, Proj>;
		std::vector<std::pair<value_type, uint32>> keyed;
		if constexpr (std::ranges::sized_range<R>) {
			atl_assert(std::ranges::size(values) <= (std::numeric_limits<uint32>::max)());
			keyed.reserve(std::ranges::size(values));
		}
		uint32 position = 0;
		for (auto &&e : values) {
			atl_assert(position < (std::numeric_limits<uint32>::max)()); // non-sized ranges are not covered above
			keyed.emplace_back(static_cast<value_type>(std::invoke(proj, detail::forward_element<R>(std::forward<decltype(e)>(e)))), position++);
		}
		std::ranges::sort(keyed, std::reference_wrapper<Compare const>{comp}, &std::pair<value_type, uint32>::first);
		std::vector<uint32> ranks(keyed.size());
		std::vector<value_type> universe;
		for (std::size_t i = 0; i < keyed.size(); ++i) {
			if (universe.empty() || comp(universe.back(), keyed[i].first)) universe.push_back(std::move(keyed[i].first));
			atl_assert(!universe.empty() && universe.size() <= (std::numeric_limits<uint32>::max)());
			ranks[keyed[i].second] = uint32(universe.size() - 1);
		}
		return {coordinate_compression<value_type, Compare>(sorted_unique, std::move(universe), Compare(std::move(comp))),
				std::move(ranks)};
	}

	/**
	 * @brief Same contract as @ref atl::compress, built from an index sort.
	 * @note Projected results support static_cast to stored values. Algorithms reference the comparator,
	 *       whose storage is directly constructed from its rvalue argument.
	 * @pre `values` has fewer than 2^32 elements (checked under `ATL_ASSERT`).
	 * @complexity O(n log n)
	 */
	template<std::ranges::input_range R, typename Proj = std::identity, typename Compare = std::ranges::less>
		requires std::invocable<Proj &, detail::forwarded_element_t<R>>
			  && std::is_constructible_v<Compare, Compare &&>
			  && requires { static_cast<detail::projected_value_t<R, Proj>>(std::invoke(std::declval<Proj &>(), std::declval<detail::forwarded_element_t<R>>())); }
			  && std::movable<detail::projected_value_t<R, Proj>>
			  && std::strict_weak_order<Compare const &, detail::projected_value_t<R, Proj> const &,
										detail::projected_value_t<R, Proj> const &>
	constexpr compressed<detail::projected_value_t<R, Proj>, Compare> compress_by_index_sort(R &&values, Proj proj = {},
																							 Compare comp = {}) {
		using value_type = detail::projected_value_t<R, Proj>;
		std::vector<value_type> input;
		if constexpr (std::ranges::sized_range<R>) {
			atl_assert(std::ranges::size(values) <= (std::numeric_limits<uint32>::max)());
			input.reserve(std::ranges::size(values));
		}
		for (auto &&e : values) {
			atl_assert(input.size() < (std::numeric_limits<uint32>::max)());
			input.push_back(static_cast<value_type>(std::invoke(proj, detail::forward_element<R>(std::forward<decltype(e)>(e)))));
		}
		atl_assert(input.size() <= (std::numeric_limits<uint32>::max)()); // non-sized ranges are not covered above
		std::vector<uint32> order;
		order.reserve(input.size());
		// libc++ at the supported floor has no ranges::iota algorithm.
		for (uint32 i = 0; i < input.size(); ++i) order.push_back(i);
		std::ranges::sort(order, std::reference_wrapper<Compare const>{comp}, [&](uint32 i) -> value_type const & { return input[i]; });
		std::vector<uint32> ranks(input.size());
		std::vector<value_type> universe;
		for (uint32 const i : order) {
			if (universe.empty() || comp(universe.back(), input[i])) universe.push_back(std::move(input[i]));
			ranks[i] = uint32(universe.size() - 1);
		}
		return {coordinate_compression<value_type, Compare>(sorted_unique, std::move(universe), Compare(std::move(comp))),
				std::move(ranks)};
	}

	/**
	 * @brief Same contract as @ref atl::compress for integral values under `<`, sorted by an LSD radix
	 *        sort instead of a comparison sort of (value, position) pairs.
	 *
	 * Digit-width race: not measured yet.
	 * Each value maps to an order-preserving unsigned key (the sign bit flipped for signed types), so
	 * the sort is `bits / DigitBits` linear passes over (key, position) records, a pass skipped
	 * when every key shares its digit: it could win once n log n comparisons outweigh those passes,
	 * which is the large-n end of the race. The universe decodes back from the sorted keys.
	 * @tparam DigitBits Bits per radix pass: 8 or 16; the library uses 11-bit passes.
	 * @note The forwarded projection result must support static_cast to the integral stored value.
	 * @pre `values` has fewer than 2^32 elements (checked under `ATL_ASSERT`).
	 * @complexity O((n + 2^DigitBits) * bits / DigitBits)
	 */
	template<int DigitBits = 8, std::ranges::input_range R, typename Proj = std::identity>
		requires(DigitBits == 8 || DigitBits == 16) && std::invocable<Proj &, detail::forwarded_element_t<R>>
			 && requires { static_cast<detail::projected_value_t<R, Proj>>(std::invoke(std::declval<Proj &>(), std::declval<detail::forwarded_element_t<R>>())); }
			 && (atl::integral<detail::projected_value_t<R, Proj>> || std::integral<detail::projected_value_t<R, Proj>>)
			 && (!std::same_as<detail::projected_value_t<R, Proj>, bool>)
	[[nodiscard]] constexpr compressed<detail::projected_value_t<R, Proj>> compress_by_radix_sort(R &&values, Proj proj = {}) {
		using value_type = detail::projected_value_t<R, Proj>;
		using key_type = atl::traits::make_unsigned_t<value_type>;
		constexpr key_type sign_flip = (atl::traits::is_signed_integral_v<value_type> || std::is_signed_v<value_type>) ? key_type(key_type(1) << (sizeof(key_type) * 8 - 1)) : 0;
		std::vector<detail::keyed_position<key_type>> keyed;
		if constexpr (std::ranges::sized_range<R>) {
			atl_assert(std::ranges::size(values) <= (std::numeric_limits<uint32>::max)());
			keyed.reserve(std::ranges::size(values));
		}
		uint32 position = 0;
		for (auto &&e : values) {
			atl_assert(position < (std::numeric_limits<uint32>::max)()); // non-sized ranges are not covered above
			auto const x = static_cast<value_type>(std::invoke(proj, detail::forward_element<R>(std::forward<decltype(e)>(e))));
			keyed.push_back({key_type(key_type(x) ^ sign_flip), position++});
		}
		detail::radix_sort<DigitBits>(keyed);
		std::vector<uint32> ranks(keyed.size());
		std::vector<value_type> universe;
		for (std::size_t i = 0; i < keyed.size(); ++i) {
			if (i == 0 || keyed[i - 1].key != keyed[i].key) universe.push_back(value_type(key_type(keyed[i].key ^ sign_flip)));
			ranks[keyed[i].position] = uint32(universe.size() - 1);
		}
		return {coordinate_compression<value_type>(sorted_unique, std::move(universe)), std::move(ranks)};
	}

	/**
	 * @brief The knobs of @ref ranks_by_search_or_gallop; the defaults select search/merge thresholds for
	 *        @ref atl::coordinate_compression::ranks dispatch.
	 */
	struct ranks_policy {
		/** @brief Whether `q * divisor >= m`, exactly and without overflow (a 0 divisor reaches only m = 0). */
		static constexpr bool reaches(std::size_t q, std::size_t divisor, std::size_t m) {
			return divisor == 0 ? m == 0 : q >= m / divisor + (m % divisor != 0);
		}

		/** @brief Sorted queries take the merge walk when `q * merge_divisor >= m` (default: 64). */
		std::size_t merge_divisor = 64;
		/** @brief Sorted queries below the merge threshold gallop instead of binary-searching. */
		bool is_galloping = false;
	};

	/**
	 * @brief Same result as @ref atl::detail::ranks_by_merge for queries sorted under @p comp, each
	 *        rank found by an exponential then a binary search forward from the previous one.
	 *
	 * Galloping race: not measured yet.
	 * A query `g` slots past the previous one costs O(log g) comparisons that stay near the previous
	 * hit: for sparse sorted queries it could beat both the q independent root-to-leaf binary searches
	 * (which re-walk the cold top of the universe) and the merge walk (which touches all m slots).
	 * @pre `queries` sorted under @p comp: asserted as far as it decides a rank (each query is still
	 *      after the universe value below the previous rank).
	 * @complexity O(q log(m / q + 1)) for q sorted queries
	 */
	template<typename Value, typename Compare, std::ranges::input_range R>
		requires std::indirect_strict_weak_order<Compare &, Value const *, std::ranges::iterator_t<R>>
	constexpr void ranks_by_gallop(std::vector<Value> const &universe, Compare &comp, R &&queries, std::vector<uint32> &out) {
		atl_assert(universe.size() <= (std::numeric_limits<uint32>::max)());
		std::size_t const m = universe.size();
		std::size_t j = 0;
		for (auto const &x : queries) {
			atl_assert(out.size() < (std::numeric_limits<uint32>::max)());
			atl_assert(j == 0 || comp(universe[j - 1], x));
			if (j < m && comp(universe[j], x)) {
				// invariant: universe[low] is before x; the answer lies in (low, low + step]
				std::size_t low = j, step = 1;
				while (step < m - low && comp(universe[low + step], x)) {
					low += step;
					step *= 2;
				}
				auto const first = universe.begin() + std::ptrdiff_t(low + 1);
				auto const last = universe.begin() + std::ptrdiff_t(low + (std::min)(step, m - low));
				j = std::size_t(std::ranges::lower_bound(first, last, x, std::reference_wrapper<Compare>{comp}) - universe.begin());
			}
			out.push_back(uint32(j));
		}
	}

	/**
	 * @brief @ref atl::coordinate_compression::ranks over the strictly increasing @p universe with the
	 *        dispatch set by @p policy; sized forward ranges only.
	 *
	 * Search/gallop dispatch race: not measured yet.
	 * Sortedness is checked whenever a sorted-query path can be taken: from the merge threshold, or
	 * always when galloping. Unsorted queries retain one binary search each.
	 * @pre `queries` has fewer than 2^32 elements (checked under `ATL_ASSERT`).
	 */
	template<typename Value, typename Compare, std::ranges::forward_range R>
		requires std::ranges::sized_range<R>
			  && std::indirect_strict_weak_order<Compare &, Value const *, std::ranges::iterator_t<R>>
			  && std::copyable<std::ranges::range_value_t<R>>
	[[nodiscard]] constexpr std::vector<uint32> ranks_by_search_or_gallop(std::vector<Value> const &universe, Compare comp, R &&queries,
																		  ranks_policy policy = {}) {
		std::size_t const q = std::ranges::size(queries), m = universe.size();
		atl_assert(q <= (std::numeric_limits<uint32>::max)());
		std::vector<uint32> out;
		bool const is_merging = ranks_policy::reaches(q, policy.merge_divisor, m);
		bool const is_query_sorted = (is_merging || policy.is_galloping) && std::ranges::is_sorted(queries, std::reference_wrapper<Compare>{comp});
		if (is_query_sorted) {
			out.reserve(q);
			if (is_merging)
				detail::ranks_by_merge(universe, comp, queries, out);
			else
				ranks_by_gallop(universe, comp, queries, out);

		} else {
			out.reserve(q);
			detail::ranks_by_search(universe, comp, queries, out);
		}
		return out;
	}

	/**
	 * @brief The bulk-ranks baseline with a separate sortedness pre-scan.
	 *
	 * Retained as the benchmark baseline. The 2026-11-03 measurements using
	 * benchmarks/ds/coordinate_compression.cpp, Xeon Platinum 8168,
	 * -O2 / -march=native, mostly sorted m=1e6 q=4e6: 260/266 ms versus fused 58.5/56 ms.
	 * It discards the sorted prefix on a descent and sorts the whole batch.
	 * The library keeps the prefix ranks.
	 * @pre `queries` has fewer than 2^32 elements; `universe` is strictly increasing under `comp`.
	 * @complexity O(q log m), O(q + m) for numerous sorted queries, O(q log q + m) for sort-merge.
	 */
	template<typename Value, typename Compare, std::ranges::input_range R>
		requires std::indirect_strict_weak_order<Compare &, Value const *, std::ranges::iterator_t<R>>
	[[nodiscard]] constexpr std::vector<uint32> ranks_by_sortedness_prescan(std::vector<Value> const &universe, Compare comp, R &&queries) {
		atl_assert(universe.size() <= (std::numeric_limits<uint32>::max)());
		std::vector<uint32> out;
		if constexpr (std::ranges::sized_range<R>) {
			atl_assert(std::ranges::size(queries) <= (std::numeric_limits<uint32>::max)());
			out.reserve(std::ranges::size(queries));
		}
		if constexpr (std::ranges::forward_range<R>) {
			std::size_t const min_merge_count = universe.size() / 64 + (universe.size() % 64 != 0);
			std::size_t count = 0;
			bool is_sorted = true;
			if constexpr (std::ranges::sized_range<R>) {
				count = std::ranges::size(queries);
				if (count < min_merge_count) {
					detail::ranks_by_search(universe, comp, queries, out);
					return out;
				}
				is_sorted = std::ranges::is_sorted(queries, std::reference_wrapper<Compare>{comp});
			} else {
				auto it = std::ranges::begin(queries);
				auto const last = std::ranges::end(queries);
				if (it != last) {
					count = 1;
					for (auto previous = it++; it != last; previous = it++, ++count) {
						atl_assert(count < (std::numeric_limits<uint32>::max)());
						if (comp(*it, *previous)) is_sorted = false;
					}
				}
				if (count < min_merge_count) {
					detail::ranks_by_search(universe, comp, queries, out);
					return out;
				}
			}
			if (is_sorted) {
				detail::ranks_by_merge(universe, comp, queries, out);
				return out;
			}
			if constexpr (std::is_trivially_copyable_v<std::ranges::range_value_t<R>>
						  && std::copyable<std::ranges::range_value_t<R>>
						  && std::constructible_from<std::ranges::range_value_t<R>, std::ranges::range_reference_t<R>>) {
				detail::ranks_by_sorted_merge(universe, comp, queries, out);
				return out;
			}
		}
		detail::ranks_by_search(universe, comp, queries, out);
		return out;
	}
}
