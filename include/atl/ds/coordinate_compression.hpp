#pragma once
#include <atl/ds/detail/forward_element.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/tags.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/radix_sort.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief Append to @p out the lower-bound rank in @p universe of every query, one binary search each.
		 * @complexity O(q log m)
		 */
		template<typename Value, typename Compare, std::ranges::input_range R>
			requires std::indirect_strict_weak_order<Compare &, Value const *, std::ranges::iterator_t<R>>
		constexpr void ranks_by_search(std::vector<Value> const &universe, Compare &comp, R &&queries,
									   std::vector<uint32> &out) {
			atl_assert(universe.size() <= (std::numeric_limits<uint32>::max)());
			for (auto const &x : queries) {
				atl_assert(out.size() < (std::numeric_limits<uint32>::max)()); // central check: queries may be non-sized
				out.push_back(uint32(std::ranges::lower_bound(universe, x, std::reference_wrapper<Compare>{comp}) - universe.begin()));
			}
		}

		/**
		 * @brief Same result as @ref ranks_by_search for queries already sorted under @p comp: one pointer
		 *        advanced monotonically through @p universe.
		 * @complexity O(q + m)
		 */
		template<typename Value, typename Compare, std::ranges::input_range R>
			requires std::indirect_strict_weak_order<Compare &, Value const *, std::ranges::iterator_t<R>>
		constexpr void ranks_by_merge(std::vector<Value> const &universe, Compare &comp, R &&queries,
									  std::vector<uint32> &out) {
			atl_assert(universe.size() <= (std::numeric_limits<uint32>::max)());
			std::size_t j = 0;
			std::size_t const m = universe.size();
			for (auto const &x : queries) {
				atl_assert(out.size() < (std::numeric_limits<uint32>::max)()); // central check: queries may be non-sized
				while (j < m && comp(universe[j], x)) ++j;
				out.push_back(uint32(j));
			}
		}

		/**
		 * @brief The smallest input that @ref atl::compress radix-sorts instead of comparison-sorting.
		 *
		 * Radix passes cost 2^11 counters each whatever the input size. Measurements using
		 * benchmarks/ds/coordinate_compression.cpp on Xeon Platinum 8168 show radix winning at every measured size from 1024: radix/comparison ms are 0.019/0.026
		 * at 1024, 0.038/0.086 at 2048 and 0.057/0.149 at 3072. The Xeon Platinum 8358
		 * repeated-identical-input race loses 38-58% at 1024 (0.018/0.012 ms); 2048 was not run.
		 * Keep 1024 on comparison sort pending a clean re-race.
		 */
		inline constexpr std::size_t radix_compress_min_size = std::size_t(1) << 11;

		/**
		 * @brief Append lower-bound ranks after the ranks already in
		 *        @p out: the (query, position) pairs of @p queries are sorted under @p comp and one merge walk from
		 *        the start of @p universe writes each rank to its slot after the existing ones.
		 *
		 * The ranks of an already answered sorted prefix stay intact.
		 * @complexity O(q log q + m)
		 */
		template<typename Value, typename Compare, std::ranges::forward_range R>
			requires std::indirect_strict_weak_order<Compare &, Value const *, std::ranges::iterator_t<R>>
				  && std::copyable<std::ranges::range_value_t<R>>
				  && std::constructible_from<std::ranges::range_value_t<R>, std::ranges::range_reference_t<R>>
		constexpr void ranks_by_sorted_merge_suffix(std::vector<Value> const &universe, Compare &comp, R &&queries,
													std::vector<uint32> &out) {
			using query_type = std::ranges::range_value_t<R>;
			atl_assert(universe.size() <= (std::numeric_limits<uint32>::max)());
			std::size_t const offset = out.size();
			atl_assert(offset <= (std::numeric_limits<uint32>::max)());
			std::vector<std::pair<query_type, uint32>> keyed;
			if constexpr (std::ranges::sized_range<R>) keyed.reserve(std::ranges::size(queries));
			for (auto const &x : queries) {
				atl_assert(keyed.size() < (std::numeric_limits<uint32>::max)() - offset);
				keyed.emplace_back(x, uint32(keyed.size()));
			}
			std::ranges::sort(keyed, std::reference_wrapper<Compare>{comp}, &std::pair<query_type, uint32>::first);
			out.resize(offset + keyed.size());
			std::size_t j = 0;
			std::size_t const m = universe.size();
			for (auto const &[x, position] : keyed) {
				while (j < m && comp(universe[j], x)) ++j;
				out[offset + position] = uint32(j);
			}
		}

		/**
		 * @brief Same result as @ref ranks_by_search for unsorted queries: (query, position)
		 *        pairs sorted under @p comp, then one merge walk writes each rank to its position.
		 *
		 * Trades q cache-missing binary searches for an in-cache sort of the queries and one sequential
		 * pass over the universe; wins for the measured numerous unsorted batches.
		 * @pre `out` is empty.
		 * @complexity O(q log q + m)
		 */
		template<typename Value, typename Compare, std::ranges::forward_range R>
			requires std::indirect_strict_weak_order<Compare &, Value const *, std::ranges::iterator_t<R>>
				  && std::copyable<std::ranges::range_value_t<R>>
				  && std::constructible_from<std::ranges::range_value_t<R>, std::ranges::range_reference_t<R>>
		constexpr void ranks_by_sorted_merge(std::vector<Value> const &universe, Compare &comp, R &&queries,
											 std::vector<uint32> &out) {
			atl_assert(out.empty());
			ranks_by_sorted_merge_suffix(universe, comp, std::forward<R>(queries), out);
		}

		/**
		 * @brief The value type a range of `R` elements seen through `Proj` compresses to.
		 *
		 * Under `std::identity` this is the range's value type, not the decayed reference: a proxy
		 * reference such as libc++'s `std::vector<bool>` const reference (`__bit_const_reference`,
		 * not assignable) would otherwise become the stored type and fail `std::movable`.
		 */
		template<std::ranges::input_range R, typename Proj>
			requires std::invocable<Proj &, forwarded_element_t<R>>
		using projected_value_t =
			std::conditional_t<std::same_as<Proj, std::identity>, std::ranges::range_value_t<R>,
							   std::remove_cvref_t<std::invoke_result_t<Proj &, forwarded_element_t<R>>>>;
	}

	/**
	 * @brief Coordinate compression: the sorted distinct universe of a set of
	 *        values, mapping each value to its rank and back.
	 *
	 * Built once from a range (compose ranges/views to gather values from several
	 * sources), then queried: @ref rank is the compressed index — usable directly
	 * as an index into a structure of @ref size slots when present — and @ref operator[]
	 * decompresses. `rank`/`rank_upper` are defined for values *not* in the
	 * universe too (lower/upper-bound semantics), so half-open query endpoints
	 * `[rank(l), rank(r))` work without the endpoints having been collected.
	 *
	 * @tparam T The value type.
	 * @tparam Compare A const-callable strict weak order over `T` (default `<`);
	 *                 "before" means ordered before under it. Its semantic laws are unchecked.
	 * @note Algorithms reference the stored comparator without copying it; move-only comparators are supported.
	 * @note Construction uses O(n) temporary storage; queries preserve references and iterators.
	 */
	template<std::movable T, typename Compare = std::ranges::less>
		requires std::strict_weak_order<Compare const &, T const &, T const &>
	class coordinate_compression {
		std::vector<T> values_;
		[[no_unique_address]] Compare comp_;

	public:
		/** @brief The stored coordinate type. */
		using value_type = T;
		/** @brief The universe size and rank type. */
		using size_type = std::size_t;

		/** @brief An empty universe: every rank is 0. Requires a default-initializable comparator. */
		constexpr coordinate_compression() = default;

		/**
		 * @brief Build the universe from @p values, each seen through @p proj (as
		 *        in the std::ranges algorithms); duplicates are dropped. Elements
		 *        are moved out only of an owning rvalue range — never through a
		 *        borrowed range or a view such as `std::span(v)` or
		 *        `v | views::filter(...)` (see @ref detail::forward_element).
		 * @tparam R Input range; forwarded projected elements must support static_cast<T> (explicit conversions are accepted).
		 * @tparam Proj Projection callable on each forwarded element.
		 * @param values Coordinates to collect.
		 * @param proj Projection, invoked once per element.
		 * @param comp Const-callable strict weak order, directly constructible from its rvalue argument.
		 * @complexity O(n log n)
		 */
		template<std::ranges::input_range R, typename Proj = std::identity>
			requires std::invocable<Proj &, detail::forwarded_element_t<R>>
				  && requires { static_cast<T>(std::invoke(std::declval<Proj &>(), std::declval<detail::forwarded_element_t<R>>())); }
				  && std::is_constructible_v<Compare, Compare &&>
		constexpr explicit coordinate_compression(R &&values, Proj proj = {}, Compare comp = {}) : comp_(std::move(comp)) {
			if constexpr (std::ranges::sized_range<R>) values_.reserve(std::ranges::size(values));
			for (auto &&e : values) values_.push_back(static_cast<T>(std::invoke(proj, detail::forward_element<R>(std::forward<decltype(e)>(e)))));
			std::ranges::sort(values_, std::reference_wrapper<Compare const>{comp_});
			auto const dup =
				std::ranges::unique(values_, [&](T const &a, T const &b) { return !comp_(a, b) && !comp_(b, a); });
			values_.erase(dup.begin(), dup.end());
		}

		/**
		 * @brief Adopt @p universe, which must already be strictly increasing under
		 *        `Compare` (sorted and without duplicates); no sort is done.
		 * @param universe Sorted distinct coordinates to adopt.
		 * @param comp Const-callable strict weak order, directly constructed from its rvalue argument.
		 * @pre `universe` is strictly increasing (checked under `ATL_ASSERT`).
		 * @complexity O(1), O(n) under `ATL_ASSERT`
		 */
		constexpr coordinate_compression(sorted_unique_t, std::vector<T> universe, Compare comp = {})
			requires std::is_constructible_v<Compare, Compare &&>
			: values_(std::move(universe)), comp_(std::move(comp)) {
			atl_assert(std::ranges::adjacent_find(values_, [&](T const &a, T const &b) { return !comp_(a, b); })
					   == values_.end());
		}

		/**
		 * @brief The number of distinct values before @p x — the compressed index
		 *        of `x` when present, and still meaningful when absent
		 *        (lower-bound semantics).
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr std::size_t rank(T const &x) const {
			return std::size_t(std::ranges::lower_bound(values_, x, std::reference_wrapper<Compare const>{comp_}) - values_.begin());
		}

		/**
		 * @brief The number of distinct values not after @p x (upper-bound
		 *        semantics) — e.g. the exclusive end for an inclusive right
		 *        endpoint, without the `rank(x + 1)` overflow trap.
		 * @complexity O(log n)
		 */
		[[nodiscard]] constexpr std::size_t rank_upper(T const &x) const {
			return std::size_t(std::ranges::upper_bound(values_, x, std::reference_wrapper<Compare const>{comp_}) - values_.begin());
		}

		/**
		 * @brief The @ref rank of every element of @p queries, in order: a bulk form
		 *        that merge-walks numerous sorted queries and sorts only the suffix after
		 *        a descent, restoring input order.
		 *
		 * Sortedness is detected during the merge walk, without a separate pre-scan. At the first
		 * descent, prefix ranks are kept and only the remaining suffix is sorted and merged for
		 * trivially copyable, copyable query values constructible from their references; other
		 * query types search the suffix. Non-forward input ranges always search per element.
		 * The walk starts at `q >= ceil(n / 64)`; smaller batches search. Sized ranges are never
		 * traversed to count; non-sized forward ranges take one counting pass before dispatch.
		 *
		 * Measurements using benchmarks/ds/coordinate_compression.cpp,
		 * Xeon Platinum 8168, ms at -O2 / -march=native, mostly sorted
		 * m=1e6: q=2e6 112/113 -> 32.2/31.2, q=4e6 260/266 -> 58.5/56,
		 * q=16e6 1292/1305 -> 254/249 (-71 to -80%). Former -O2 dispatch lost to plain search
		 * at q=4e6 (260 vs 202) and q=16e6 (1292 vs 687). Accepted Xeon Platinum 8168 costs:
		 * random m=1e6 q=m/64 1.6/1.7 -> 1.8/1.9 ms (+12%); random m=16384 q=1024 +4%/+2%;
		 * sorted m=1e5 +8%/+11% at -O2 (display resolution), -6% to -18% native.
		 *
		 * @param queries Values comparable with `T` under `Compare`; never moved from.
		 * @return `queries.size()` ranks, each below or equal to `size()`.
		 * @pre `queries` has at most 2^32 - 1 elements (checked under `ATL_ASSERT`).
		 * @complexity O(q log n), or for `q * 64 >= n` O(q + n) on sorted queries and
		 * O(q log q + n) on the sort-and-merge path.
		 */
		template<std::ranges::input_range R>
			requires std::indirect_strict_weak_order<Compare const &, T const *, std::ranges::iterator_t<R>>
		[[nodiscard]] constexpr std::vector<uint32> ranks(R &&queries) const {
			atl_assert(values_.size() <= (std::numeric_limits<uint32>::max)());
			std::vector<uint32> out;
			if constexpr (std::ranges::sized_range<R>) {
				atl_assert(std::ranges::size(queries) <= (std::numeric_limits<uint32>::max)());
				out.reserve(std::ranges::size(queries));
			}
			if constexpr (std::ranges::forward_range<R>) {
				std::size_t const count = [&] {
					if constexpr (std::ranges::sized_range<R>)
						return std::size_t(std::ranges::size(queries));
					else
						return std::size_t(std::ranges::distance(queries));
				}();
				atl_assert(count <= (std::numeric_limits<uint32>::max)());
				std::size_t const min_merge_count = values_.size() / 64 + (values_.size() % 64 != 0);
				if (count < min_merge_count) {
					detail::ranks_by_search(values_, comp_, queries, out);
					return out;
				}
				if constexpr (!std::ranges::sized_range<R>) out.reserve(count);
				std::size_t j = 0;
				auto it = std::ranges::begin(queries);
				auto const last = std::ranges::end(queries);
				auto previous = it; // strict ordering makes the first query's self-comparison false
				for (; it != last; previous = it++) {
					auto const &x = *it;
					if (comp_(x, *previous)) break;
					while (j < values_.size() && comp_(values_[j], x)) ++j;
					out.push_back(uint32(j));
				}
				if (it == last) return out;
				std::ranges::subrange const rest(it, last);
				if constexpr (std::is_trivially_copyable_v<std::ranges::range_value_t<R>>
							  && std::copyable<std::ranges::range_value_t<R>>
							  && std::constructible_from<std::ranges::range_value_t<R>, std::ranges::range_reference_t<R>>)
					detail::ranks_by_sorted_merge_suffix(values_, comp_, rest, out);
				else
					detail::ranks_by_search(values_, comp_, rest, out);
			} else {
				detail::ranks_by_search(values_, comp_, queries, out);
			}
			return out;
		}

		/** @brief Whether @p x is in the universe. @complexity O(log n) */
		[[nodiscard]] constexpr bool contains(T const &x) const {
			auto const it = std::ranges::lower_bound(values_, x, std::reference_wrapper<Compare const>{comp_});
			return it != values_.end() && !comp_(x, *it);
		}

		/** @brief The @p i-th distinct value (inverse of @ref rank). @pre `i < size()` */
		constexpr T const &operator[](std::size_t i) const {
			atl_assert(i < values_.size());
			return values_[i];
		}

		/** @brief Number of distinct values. */
		constexpr std::size_t size() const noexcept { return values_.size(); }

		/** @brief Whether the universe has no values. */
		constexpr bool empty() const noexcept { return values_.empty(); }

		/** @brief The distinct values in order, iterable (e.g. for building auxiliary tables). */
		constexpr auto begin() const noexcept { return values_.begin(); }

		/** @brief Past-the-end iterator over the distinct values. */
		constexpr auto end() const noexcept { return values_.end(); }
	};

	/**
	 * @brief The universe of a set of values together with every input value's rank in it.
	 * @tparam T The value type.
	 * @tparam Compare The strict weak order of the universe.
	 */
	template<std::movable T, typename Compare = std::ranges::less>
		requires std::strict_weak_order<Compare const &, T const &, T const &>
	struct compressed {
		coordinate_compression<T, Compare> universe; ///< The sorted distinct values.
		std::vector<uint32> ranks;					 ///< `ranks[i]` is the rank of the i-th input value.
	};

	/**
	 * @brief Compress @p values and return, besides the universe, the rank of every input value in input order:
	 *        the common "compress then map everything" step in one pass, with no per-value search.
	 *
	 * Sorts integral keys under `<` (except bool) by fixed 11-bit LSD radix passes from 2^11 values,
	 * otherwise comparison-sorts (value, input position) pairs; the ranks fall out of a linear walk over the sorted run, so
	 * each value costs a sort slot instead of an extra `log n` cache-missing search. Elements are moved only
	 * out of an owning rvalue range, as in the @ref coordinate_compression constructor.
	 *
	 * @tparam R Input range.
	 * @tparam Proj Projection; identity stores the range value type, other projections their decayed result.
	 *              The result must support static_cast to the stored value; explicit conversions are accepted.
	 * @tparam Compare Const-callable strict weak order over projected values; directly constructible from its rvalue argument.
	 * @param values Coordinates to compress.
	 * @param proj Projection invoked once per forwarded element.
	 * @param comp Ordering of the universe.
	 * @return Sorted distinct universe and input-order ranks.
	 * @pre `values` has at most 2^32 - 1 elements (checked under `ATL_ASSERT`).
	 * @complexity O(n) for integral keys under `<` (except bool) from 2^11 values, otherwise O(n log n).
	 */
	template<std::ranges::input_range R, typename Proj = std::identity, typename Compare = std::ranges::less>
		requires std::invocable<Proj &, detail::forwarded_element_t<R>>
			  && std::is_constructible_v<Compare, Compare &&>
			  && requires { static_cast<detail::projected_value_t<R, Proj>>(std::invoke(std::declval<Proj &>(), std::declval<detail::forwarded_element_t<R>>())); }
			  && std::movable<detail::projected_value_t<R, Proj>>
			  && std::strict_weak_order<Compare const &, detail::projected_value_t<R, Proj> const &,
										detail::projected_value_t<R, Proj> const &>
	[[nodiscard]] constexpr compressed<detail::projected_value_t<R, Proj>, Compare> compress(R &&values, Proj proj = {}, Compare comp = {}) {
		using value_type = detail::projected_value_t<R, Proj>;
		if constexpr ((atl::integral<value_type> || std::integral<value_type>) && !std::same_as<value_type, bool>
					  && (std::same_as<Compare, std::ranges::less> || std::same_as<Compare, std::less<value_type>>
						  || std::same_as<Compare, std::less<void>>)) {
			using unsigned_type = atl::traits::make_unsigned_t<value_type>;
			using key_type = std::conditional_t<(sizeof(unsigned_type) < sizeof(uint32)), uint32, unsigned_type>;
			constexpr key_type sign_flip = atl::traits::is_signed_integral_v<value_type> || std::is_signed_v<value_type> ? key_type(key_type(1) << (sizeof(value_type) * 8 - 1)) : 0;
			std::vector<detail::keyed_position<key_type>> keyed;
			if constexpr (std::ranges::sized_range<R>) {
				atl_assert(std::ranges::size(values) <= (std::numeric_limits<uint32>::max)());
				keyed.reserve(std::ranges::size(values));
			}
			uint32 position = 0;
			for (auto &&e : values) {
				atl_assert(position < (std::numeric_limits<uint32>::max)()); // non-sized ranges are not covered above
				auto const x = static_cast<value_type>(std::invoke(proj, detail::forward_element<R>(std::forward<decltype(e)>(e))));
				keyed.push_back({key_type(key_type(unsigned_type(x)) ^ sign_flip), position++});
			}
			if (keyed.size() >= detail::radix_compress_min_size)
				detail::radix_sort<11>(keyed);
			else
				std::ranges::sort(keyed, {}, &detail::keyed_position<key_type>::key);
			std::vector<uint32> ranks(keyed.size());
			std::vector<value_type> universe;
			for (std::size_t i = 0; i < keyed.size(); ++i) {
				if (i == 0 || keyed[i - 1].key != keyed[i].key) universe.push_back(value_type(unsigned_type(keyed[i].key ^ sign_flip)));
				atl_assert(!universe.empty() && universe.size() <= (std::numeric_limits<uint32>::max)());
				ranks[keyed[i].position] = uint32(universe.size() - 1);
			}
			return {coordinate_compression<value_type, Compare>(sorted_unique, std::move(universe), Compare(std::move(comp))), std::move(ranks)};
		} else {
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
	}

	/** @brief Deduce coordinates from the projected range values and retain the comparator type. */
	template<std::ranges::input_range R, typename Proj = std::identity, typename Compare = std::ranges::less>
		requires std::invocable<Proj &, detail::forwarded_element_t<R>>
			  && std::movable<detail::projected_value_t<R, Proj>>
			  && std::is_constructible_v<Compare, Compare &&>
			  && requires { static_cast<detail::projected_value_t<R, Proj>>(std::invoke(std::declval<Proj &>(), std::declval<detail::forwarded_element_t<R>>())); }
			  && std::strict_weak_order<Compare const &, detail::projected_value_t<R, Proj> const &,
										detail::projected_value_t<R, Proj> const &>
	coordinate_compression(R &&, Proj = {}, Compare = {})
		-> coordinate_compression<detail::projected_value_t<R, Proj>, Compare>;
}
