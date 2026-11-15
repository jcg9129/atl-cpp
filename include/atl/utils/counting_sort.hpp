#pragma once
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		template<typename R, typename Proj>
		using counting_key_t =
			std::remove_cvref_t<std::invoke_result_t<Proj &, std::ranges::range_reference_t<R>>>;
	}

	/**
	 * @brief A range is `counting_sortable` under projection `Proj` when it is a
	 *        sized random-access range whose elements can be permuted in place and
	 *        whose projected value is an integral key of at most 64 bits (bucket
	 *        offsets are computed in `uint64`). `Proj` is invoked as an lvalue on
	 *        `std::ranges::range_reference_t<R>`.
	 */
	template<typename R, typename Proj = std::identity>
	concept counting_sortable = std::ranges::random_access_range<R> && std::ranges::sized_range<R>
							 && std::permutable<std::ranges::iterator_t<R>>
							 && std::integral<detail::counting_key_t<R, Proj>>
							 && sizeof(detail::counting_key_t<R, Proj>) <= sizeof(uint64);

	namespace detail {
		/**
		 * @brief A bucket key for @ref bucket_by_key: `key(i)` maps item index `i`
		 *        to its bucket number.
		 */
		template<typename Key>
		concept bucket_key_function =
			std::invocable<Key const &, std::size_t>
			&& std::convertible_to<std::invoke_result_t<Key const &, std::size_t>, std::size_t>;

		/**
		 * @brief The stable bucket placement under every counting sort: items
		 *        `0 .. order.size() - 1` with keys `key(i)` in `[0, bucket_count)`
		 *        are grouped by key into caller-owned arrays. Afterwards bucket
		 *        `b` is `order[offsets[b] .. offsets[b + 1])`, holding the indices
		 *        of the items with key `b` in increasing index order;
		 *        `offsets[bucket_count] == order.size()`. This is the CSR row-start
		 *        table plus the gather permutation, so a caller that knows its
		 *        bucket count (vertex ids, depths) needs no min/max scan.
		 *
		 * Counts accumulate into `offsets[b]` as running ends and the scatter
		 * walks the items backwards, decrementing to the bucket starts: stable,
		 * and the offsets come out final with no cursor copy or shift pass.
		 *
		 * @param bucket_count Number of buckets.
		 * @param key Item index to bucket number; called twice per item, so it
		 *        must return the same bucket both times.
		 * @param offsets Output, `bucket_count + 1` entries.
		 * @param order Output, one entry per item; its size is the item count.
		 * @pre `offsets.size() == bucket_count + 1`, at most `2^32 - 1` items,
		 *      and every `key(i) < bucket_count` (all asserted).
		 * @complexity O(n + bucket_count)
		 */
		template<bucket_key_function Key>
		constexpr void bucket_by_key(std::size_t bucket_count, Key const &key, std::span<uint32> offsets,
									 std::span<uint32> order) {
			std::size_t const n = order.size();
			atl_assert(offsets.size() == bucket_count + 1);
			atl_assert(uint64(n) <= uint64(uint32(-1)));
			std::ranges::fill(offsets, uint32(0));
			for (std::size_t i = 0; i < n; ++i) {
				// Valid bucket keys fit size_t, including on 32-bit targets.
				std::size_t const b = static_cast<std::size_t>(key(i));
				atl_assert(b < bucket_count);
				++offsets[b];
			}
			uint32 end = 0;
			for (std::size_t b = 0; b < bucket_count; ++b) {
				end += offsets[b];
				offsets[b] = end;
			}
			offsets[bucket_count] = uint32(n);
			for (std::size_t i = n; i-- > 0;) order[--offsets[static_cast<std::size_t>(key(i))]] = uint32(i);
		}

		/**
		 * @brief Reusable `uint32` scratch for `counting_sort`; held thread-local so
		 *        repeated sorts allocate only when a buffer must grow.
		 */
		struct counting_sort_scratch {
			std::vector<uint32> count, perm;
		};

		inline counting_sort_scratch &counting_sort_buffers() {
			static thread_local counting_sort_scratch s;
			return s;
		}

		/**
		 * @brief Fill `s.perm` with the stable ascending-by-key gather permutation
		 *        of `r` (`perm[k]` = source index of sorted position `k`); returns `n`.
		 */
		template<typename R, typename Proj>
			requires counting_sortable<R, Proj>
		std::size_t counting_permutation(R &&r, Proj &proj, counting_sort_scratch &s) {
			std::size_t const n = static_cast<std::size_t>(std::ranges::size(r));
			atl_assert(uint64(n) <= uint64(uint32(-1))); // the scratch holds uint32 indices
			s.perm.resize(n);
			if (n == 0) return 0;
			auto const first = std::ranges::begin(r);
			using key_t = counting_key_t<R, Proj>;
			auto const key = [&](std::size_t i) -> key_t {
				return std::invoke(proj, first[std::ptrdiff_t(i)]);
			};
			key_t lo = key(0), hi = lo; // min/max in the key's own type — correct order for any integral
			for (std::size_t i = 1; i < n; ++i) {
				key_t const k = key(i);
				lo = k < lo ? k : lo;
				hi = k > hi ? k : hi;
			}
			// span and per-key offset as modular uint64 differences: well-defined
			// for every integral key type (no signed-overflow UB even at the type's
			// extremes), and exact once the span is known small (asserted below)
			uint64 const span = uint64(hi) - uint64(lo);
			atl_assert(span < (uint64(1) << 31)); // small-span assumption drives the bucket count
			std::size_t const bucket_count = std::size_t(span) + 1;
			s.count.resize(bucket_count + 1);
			bucket_by_key(
				bucket_count, [&](std::size_t i) { return std::size_t(uint64(key(i)) - uint64(lo)); },
				std::span<uint32>(s.count), std::span<uint32>(s.perm));
			return n;
		}
	}

	/**
	 * @brief Stable in-place ascending sort of @p r by the integer key
	 *        `proj(element)`, in **O(n + span)** where `span = max key − min
	 *        key`. Same result as `std::ranges::stable_sort(r, {}, proj)`, but
	 *        linear when the key span is small (say ≤ 10⁶) — 2× to 6× faster
	 *        than `std::ranges::sort` at n = 10⁶ (x86-64 Xeon 8280, see
	 *        docs/counting-sort.md). A free function like `std::ranges::sort`:
	 *        nothing to construct; the index scratch is reused internally across
	 *        calls (@ref counting_sort_clear releases it), and the elements are
	 *        moved through one element-typed buffer per call.
	 *
	 * To argsort — obtain a stable permutation, or sort parallel arrays — sort an
	 * index range by an indexing projection:
	 * @code
	 *   #include <numeric>
	 *   std::vector<uint32> idx(n);
	 *   std::iota(idx.begin(), idx.end(), 0u); // Xcode 16.4 lacks ranges::iota.
	 *   atl::counting_sort(idx, [&](uint32 i) { return key[i]; });
	 * @endcode
	 *
	 * @pre The projected keys span `max - min < 2^31` (asserted: it sizes the
	 *      bucket array) and `size(r) < 2^32` (asserted: indices are `uint32`).
	 * @pre `proj` is equality-preserving: it is called three times per element
	 *      (min/max scan, count, scatter) and must return the same key each time.
	 * @pre Not reentrant on one thread: neither `proj` nor an element's move may
	 *      call counting_sort, which shares the thread's index scratch.
	 * @complexity O(n + span) time; O(n + span) extra memory (the index scratch,
	 *             kept between calls, plus one element buffer per call).
	 * @note Not constexpr (thread-local scratch, element buffer on the heap).
	 */
	template<typename R, typename Proj = std::identity>
		requires counting_sortable<R, Proj>
	void counting_sort(R &&r, Proj proj = {}) {
		auto &s = detail::counting_sort_buffers();
		std::size_t const n = detail::counting_permutation(r, proj, s);
		auto const first = std::ranges::begin(r);
		// gather out of place into an element buffer, then move back: 1.5x to 5x
		// faster than realising the permutation in place by walking its cycles,
		// whose swaps land at random positions (benchmarks/utils/counting_sort.cpp)
		std::vector<std::ranges::range_value_t<R>> sorted;
		sorted.reserve(n);
		for (std::size_t k = 0; k < n; ++k)
			sorted.push_back(std::ranges::iter_move(first + std::ptrdiff_t(s.perm[k])));
		std::ranges::move(sorted, first);
	}

	/**
	 * @brief Release the reusable counting-sort scratch on the current thread (frees
	 *        the internal buffers). Rarely needed: only to reclaim memory.
	 * @note Not constexpr (thread-local scratch).
	 */
	inline void counting_sort_clear() { detail::counting_sort_buffers() = detail::counting_sort_scratch{}; }
}
