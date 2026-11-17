#pragma once

#include <atl/types/alias.hpp>
#include <atl/types/utils.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Constant-time range-minimum queries over a static array.
	 *
	 * After O(n) construction, returns in O(1) the index of the smallest element
	 * in a range, where "smallest" is defined by `Compare` (default std::less).
	 *
	 * Holds a non-owning view (std::span) of the array, so the array must outlive
	 * the direct_rmq and must not be modified. Construction requires a
	 * `borrowed_range` — an lvalue container, or a borrowed view such as
	 * `std::span` / `std::string_view` — so a temporary owning range (which would
	 * leave a dangling view) is rejected at compile time.
	 *
	 * The queries mirror @ref atl::sparse_table, `position` returns the index of
	 * the Compare-smallest element, `fold` the element itself.
	 *
	 * @tparam T       The element type.
	 * @tparam Compare A strict-weak-ordering comparator (as std::sort expects).
	 *         The call shape is enforced by concept; the ordering *law* remains a
	 *         precondition on the caller (a concept cannot check it).
	 *
	 * @note More precisely, preprocessing and storage are O(n + m log m),
	 *       m = ceil(n / 64); the supported size cap bounds the block table to 27 rows.
	 *       Supports up to 2^32 - 1 elements: the sparse table over block minima
	 *       stores positions as uint32 to halve its footprint (a measurable
	 *       constant-factor win on query throughput).
	 *
	 * @link https://en.wikipedia.org/wiki/Range_minimum_query
	 */
	template<typename T, typename Compare = std::less<>>
		requires std::strict_weak_order<Compare const &, T const &, T const &>
	class direct_rmq {
		std::span<T const> data_;
		[[no_unique_address]] Compare comp_;
		std::size_t total_blocks_ = 0;
		std::vector<uint32> blocks_rmq_;
		std::vector<uint64> masks_;

		/** @brief Validate 32-bit positions before allocating either directory. */
		static constexpr std::size_t checked_size(std::size_t n) {
			atl_assert(n <= (std::numeric_limits<uint32>::max)());
			return n;
		}

		constexpr std::size_t argmin(std::size_t x, std::size_t y) const { return comp_(data_[y], data_[x]) ? y : x; }

		constexpr std::size_t in_block_query(std::size_t l, std::size_t r) const {
			atl_assert(r - l < block_size);
			uint64 rmask = masks_[r];
			if (std::size_t pos = r - l + 1; pos != block_size) {
				rmask &= ((1ULL << pos) - 1);
			}
			atl_assert(rmask != 0);
			std::size_t pos = std::size_t(std::bit_width(rmask));
			return (r - pos + 1);
		}

		constexpr static auto block_size = num_bits<uint64>;

	public:
		/** @brief An empty structure (no queries are valid); assign a built one before use. */
		constexpr direct_rmq() = default;

		/**
		 * @brief Build the structure over `a` (a borrowed contiguous range of `T`).
		 * @param a The array to query; must outlive the direct_rmq and not change.
		 * @param comp The comparison function to use.
		 * @complexity O(n)
		 */
		template<std::ranges::contiguous_range Range>
			requires std::ranges::borrowed_range<Range> && std::ranges::sized_range<Range>
					  && std::same_as<std::ranges::range_value_t<Range>, T>
					  && std::convertible_to<std::ranges::range_reference_t<Range>, T const &>
		constexpr explicit direct_rmq(Range &&a, Compare comp = {})
			: data_(std::ranges::data(a), checked_size(std::ranges::size(a))),
			  comp_(std::move(comp)),
			  total_blocks_(data_.size() / block_size + (data_.size() % block_size != 0)),
			  blocks_rmq_(std::bit_width(static_cast<uint64>(total_blocks_)) * total_blocks_),
			  masks_(data_.size()) {
			std::size_t n = data_.size();
			atl_assert(n <= (std::numeric_limits<uint32>::max)());
			for (std::size_t i = 0, block = 0; i < n;) {
				std::size_t end = i + (std::min)(n - i, std::size_t(block_size));
				uint32 &min_pos = (blocks_rmq_[block++] = uint32(i));
				while (++i < end)
					if (comp_(data_[i], data_[min_pos])) min_pos = uint32(i);
			}
			auto cur_lvl = blocks_rmq_.begin();
			for (std::size_t m, mh = 1; (m = mh << 1) <= total_blocks_; mh = m) {
				auto next_lvl = std::next(cur_lvl, std::ptrdiff_t(total_blocks_));
				for (std::size_t i = 0; i + mh < total_blocks_; ++i)
					next_lvl[std::ptrdiff_t(i)] =
						uint32(argmin(cur_lvl[std::ptrdiff_t(i)], cur_lvl[std::ptrdiff_t(i + mh)]));
				cur_lvl = next_lvl;
			}
			uint64 mask = 0;
			for (std::size_t i = 0; i < n; ++i) {
				mask = mask << 1;
				for (std::size_t k = std::countr_zero(mask); k != block_size; k = std::countr_zero(mask)) {
					// pop only STRICTLY greater elements: equal ones stay, so the
					// leftmost of a run of minima is the one every query reports
					if (!comp_(data_[i], data_[i - k])) break;
					atl_assert((mask >> k) & 1);
					mask ^= uint64(1) << k;
					atl_assert(((mask >> k) & 1) == 0);
				}
				mask |= 1;
				masks_[i] = mask;
			}
		}

		/**
		 * @brief The index of the Compare-smallest element in the range [l, r);
		 *        among equal minima, the LEFTMOST.
		 * @pre `l < r <= size()`
		 * @complexity O(1)
		 */
		constexpr std::size_t position(std::size_t l, std::size_t r) const {
			atl_assert(l < r);
			atl_assert(r <= data_.size());

			std::size_t x = l / block_size, y = --r / block_size;
			std::size_t block_span = y - x;
			if (block_span == 0) return in_block_query(l, r);
			if (block_span == 1) return argmin(in_block_query(l, x * block_size + block_size - 1),
											   in_block_query(y * block_size, r));
			std::size_t level = std::size_t(std::bit_width(static_cast<uint64>(block_span) - 1) - 1);
			return argmin(argmin(in_block_query(l, x * block_size + block_size - 1),
								 argmin(blocks_rmq_[total_blocks_ * level + x + 1], blocks_rmq_[total_blocks_ * level + y - (std::size_t(1) << level)])),
						  in_block_query(y * block_size, r));
		}

		/**
		 * @brief The Compare-smallest element in the range [l, r) (the maximum
		 *        under `std::greater`), as a reference into the viewed array.
		 * @pre `l < r <= size()`
		 * @complexity O(1)
		 */
		constexpr T const &fold(std::size_t l, std::size_t r) const { return data_[position(l, r)]; }

		/** @brief Number of elements in the borrowed array. */
		constexpr std::size_t size() const noexcept { return data_.size(); }

		/** @brief Whether the borrowed array is empty. */
		constexpr bool empty() const noexcept { return data_.empty(); }
	};

	/** @brief Deduce stored element, operation and dimensions from construction arguments. */
	template<std::ranges::contiguous_range Range, typename Compare = std::less<>>
		requires std::ranges::borrowed_range<Range> && std::ranges::sized_range<Range>
	explicit direct_rmq(Range &&, Compare = Compare()) -> direct_rmq<std::ranges::range_value_t<Range>, Compare>;
}
