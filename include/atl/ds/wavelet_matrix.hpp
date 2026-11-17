#pragma once
#include <atl/ds/detail/bit_vector.hpp>
#include <atl/ds/detail/forward_element.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Wavelet matrix: static order statistics over a sequence of
	 *        unsigned integers in O(log max) per query and about 1.5 bits per
	 *        value-bit of memory.
	 *
	 * One rank-supported bit vector per value bit, built by
	 * stably partitioning the sequence on that bit (highest first). Every
	 * query is a top-down walk mapping a position interval through ranks —
	 * contiguous word reads, no pointers, no per-query allocation.
	 *
	 * This is the memory-lean, cache-friendly answer to *static* range order
	 * statistics (k-th smallest, "how many < x", "how many in [lo, hi)").

	 * See @ref atl::persistent_segment_tree for order statistics over prefix count versions.
	 * @tparam T The stored unsigned integer type.
	 */
	template<unsigned_integral T>
	class wavelet_matrix {
		std::vector<detail::bit_vector> levels_;
		std::vector<uint32> zeros_;
		std::size_t n_ = 0;

		/**
		 * @brief Whether the source value @p v is representable in `T` (a narrowing
		 * static_cast would otherwise truncate it silently).
		 */
		template<typename Source>
			requires(integral<Source> || std::integral<Source>)
		static constexpr bool fits(Source v) {
			if constexpr (std::same_as<Source, bool>)
				return true;
			else {
				if constexpr ((traits::is_signed_integral_v<Source> || std::is_signed_v<Source>))
					if (v < 0) return false;
				return traits::make_unsigned_t<Source>(v) <= (std::numeric_limits<T>::max)();
			}
		}

		/** @brief Validate the source size before converting to size_t on a 32-bit target. */
		template<integral Size>
		static constexpr std::size_t checked_size(Size n) {
			atl_assert(n <= (std::numeric_limits<uint32>::max)());
			return std::size_t(n);
		}

		/** @brief Width through 64-bit limbs, including class-based uint128 values. */
		static constexpr int value_width(T value) {
			if constexpr (sizeof(T) <= sizeof(uint64))
				return int(std::bit_width(uint64(value)));
			else {
				uint64 const high = uint64(value >> 64);
				return high != 0 ? 64 + int(std::bit_width(high)) : int(std::bit_width(uint64(value)));
			}
		}

	public:
		/** @brief Stored unsigned integer type. */
		using value_type = T;
		/** @brief Position and count type. */
		using size_type = std::size_t;

		/** @brief Construct an empty sequence. */
		constexpr wavelet_matrix() = default;

		/**
		 * @brief Build from the sequence; the number of levels is the bit
		 *        width of the maximum value. Elements are consumed by value.
		 * @pre `size(values)` fits `uint32` (positions are stored 32-bit), and
		 *      every integral value is representable in `T` (no negative or
		 *      too-wide value), checked per element.
		 * @complexity O(n log max)
		 */
		template<std::ranges::sized_range R>
			requires requires { static_cast<T>(std::declval<detail::forwarded_element_t<R>>()); }
		constexpr explicit wavelet_matrix(R &&values) : n_(checked_size(std::ranges::size(values))) {
			atl_assert(n_ <= (std::numeric_limits<uint32>::max)());
			std::vector<T> cur;
			cur.reserve(n_);
			for (auto &&v : values) {
				if constexpr (integral<std::remove_cvref_t<decltype(v)>> || std::integral<std::remove_cvref_t<decltype(v)>>) atl_assert(fits(v));
				cur.push_back(static_cast<T>(detail::forward_element<R>(v)));
			}
			T mx = 0;
			for (T const v : cur) mx = mx < v ? v : mx;
			int const width = mx == 0 ? 1 : value_width(mx);
			levels_.resize(width);
			zeros_.resize(width);
			// one stable-partition pass per level, branch-free on the data bit: each value
			// goes to both the zeros cursor and the ones cursor and only the matching one
			// advances; the level's bits are assembled a word at a time in a register
			std::vector<T> next(n_), ones(n_);
			for (int lvl = width - 1; lvl >= 0; --lvl) {
				detail::bit_vector &bv = levels_[lvl] = detail::bit_vector(n_);
				std::size_t lo = 0, hi = 0;
				uint64 word = 0;
				for (std::size_t i = 0; i < n_; ++i) {
					T const v = cur[i];
					uint64 const bit = uint64(v >> lvl & 1);
					next[lo] = v;
					ones[hi] = v;
					lo += std::size_t(bit ^ 1);
					hi += std::size_t(bit);
					word |= bit << (i % 64);
					if (i % 64 == 63) bv.set_word(i / 64, word), word = 0;
				}
				if (n_ % 64) bv.set_word(n_ / 64, word);
				bv.finish();
				zeros_[lvl] = uint32(lo);
				std::ranges::copy(ones.begin(), ones.begin() + std::ptrdiff_t(hi), next.begin() + std::ptrdiff_t(lo));
				cur.swap(next);
			}
		}

		/** @brief Number of stored values. */
		constexpr std::size_t size() const noexcept { return n_; }

		/** @brief Whether the sequence has no values. */
		constexpr bool empty() const noexcept { return n_ == 0; }

		/** @brief The value at position @p i. @pre `i < size()` @complexity O(log max) */
		constexpr T at(std::size_t i) const {
			atl_assert(i < n_);
			T v = 0;
			for (std::size_t lvl = levels_.size(); lvl-- > 0;) {
				detail::bit_vector const &bv = levels_[lvl];
				if (bv.at_unchecked(i)) {
					v |= T(1) << lvl;
					i = zeros_[lvl] + bv.rank1_unchecked(i);
				} else {
					i = bv.rank0_unchecked(i);
				}
			}
			return v;
		}

		/**
		 * @brief The @p k-th smallest value (0-indexed) among positions `[l, r)`.
		 * @pre `l < r <= size()` and `k < r - l` @complexity O(log max)
		 */
		constexpr T kth(std::size_t l, std::size_t r, std::size_t k) const {
			atl_assert(l < r && r <= n_ && k < r - l);
			T v = 0;
			for (std::size_t lvl = levels_.size(); lvl-- > 0;) {
				detail::bit_vector const &bv = levels_[lvl];
				uint32 const l0 = bv.rank0_unchecked(l), r0 = bv.rank0_unchecked(r);
				if (uint32 const z = r0 - l0; k < z) {
					l = l0;
					r = r0;
				} else {
					k -= z;
					v |= T(1) << lvl;
					l = zeros_[lvl] + (uint32(l) - l0);
					r = zeros_[lvl] + (uint32(r) - r0);
				}
			}
			return v;
		}

		/**
		 * @brief Number of values `< upper` among positions `[l, r)`. The bound is
		 * exclusive, so values equal to `(std::numeric_limits<T>::max)()` are
		 * never counted; the count of values in `[lo, max]` (closed above) is
		 * `(r - l) - count_less(l, r, lo)`. @pre `l <= r <= size()`
		 * @complexity O(log max)
		 */
		constexpr std::size_t count_less(std::size_t l, std::size_t r, T upper) const {
			atl_assert(l <= r && r <= n_);
			if (levels_.size() < std::size_t(value_width(upper))) return r - l;
			std::size_t res = 0;
			for (std::size_t lvl = levels_.size(); lvl-- > 0;) {
				detail::bit_vector const &bv = levels_[lvl];
				uint32 const l0 = bv.rank0_unchecked(l), r0 = bv.rank0_unchecked(r);
				if (upper >> lvl & 1) {
					res += r0 - l0; // every 0-side value has a smaller prefix
					l = zeros_[lvl] + (uint32(l) - l0);
					r = zeros_[lvl] + (uint32(r) - r0);
				} else {
					l = l0;
					r = r0;
				}
			}
			return res;
		}

		/**
		 * @brief Number of values in `[lo, hi)` among positions `[l, r)`. The bound
		 * is half-open, so values equal to `(std::numeric_limits<T>::max)()`
		 * are never counted — use `(r - l) - count_less(l, r, lo)` for
		 * `[lo, max]`. @pre `l <= r <= size()` and `lo <= hi`
		 * @complexity O(log max)
		 */
		constexpr std::size_t count(std::size_t l, std::size_t r, T lo, T hi) const {
			atl_assert(lo <= hi);
			return count_less(l, r, hi) - count_less(l, r, lo);
		}
	};

	/** @brief Deduce stored element, operation and dimensions from construction arguments. */
	template<std::ranges::sized_range R>
	wavelet_matrix(R &&) -> wavelet_matrix<std::ranges::range_value_t<R>>;
}
