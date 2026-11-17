#pragma once
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <bit>
#include <cstddef>
#include <limits>
#include <vector>

namespace atl {
	template<unsigned_integral T>
	class wavelet_matrix;
}

namespace atl::detail {
	/**
	 * @brief Static bit sequence with O(1) rank: the building block of the
	 *        wavelet matrix.
	 *
	 * Bits are packed into 64-bit words with one cumulative popcount per word
	 * (uint32), so the whole structure is about 1.5 bits per stored bit and a rank is
	 * one word load, one mask, one popcount. Built once via @ref set +
	 * @ref finish; immutable afterwards.
	 */
	class bit_vector {
		std::vector<uint64> words_;
		/** @brief After finish(), rank_[w] counts ones strictly before word w; padding bits and the sentinel word are zero. */
		std::vector<uint32> rank_;
		std::size_t n_ = 0;

		/**
		 * @brief Number of words for @p n bits, plus one zero padding word so
		 *        a rank at `i == n` on a word boundary reads a valid word.
		 * @pre `n` fits `uint32` (the per-word cumulative ranks are 32-bit),
		 *      checked before anything is allocated.
		 */
		static constexpr std::size_t word_count(std::size_t n) {
			atl_assert(n <= (std::numeric_limits<uint32>::max)());
			return n / 64 + (n % 64 != 0) + 1;
		}

		// Wavelet queries check their input once; rank remapping preserves valid positions at every level.
		template<unsigned_integral T>
		friend class atl::wavelet_matrix;

		constexpr bool at_unchecked(std::size_t i) const { return (words_[i / 64] >> (i % 64)) & 1; }

		constexpr uint32 rank1_unchecked(std::size_t i) const {
			return rank_[i / 64] + uint32(std::popcount(words_[i / 64] & ((uint64(1) << (i % 64)) - 1)));
		}

		constexpr uint32 rank0_unchecked(std::size_t i) const { return uint32(i) - rank1_unchecked(i); }

	public:
		constexpr bit_vector() = default;

		/** @brief An all-zero sequence of @p n bits. @pre `n` fits `uint32`. */
		constexpr explicit bit_vector(std::size_t n) : words_(word_count(n), 0), n_(n) {}

		/**
		 * @brief Set bit @p i to 1. @pre `i < size()` and @ref finish not
		 *        yet called.
		 */
		constexpr void set(std::size_t i) {
			atl_assert(i < n_);
			words_[i / 64] |= uint64(1) << (i % 64);
		}

		/**
		 * @brief Overwrite the 64-bit word @p w (bits `64 w` to `64 w + 63`) with
		 *        @p bits, for builders that assemble whole words in a register.
		 *        Bits at or past `size()` must be zero. @pre `w < ⌈size() / 64⌉` and
		 *        @ref finish not yet called.
		 */
		constexpr void set_word(std::size_t w, uint64 bits) {
			atl_assert(w < (n_ / 64 + (n_ % 64 != 0)));
			words_[w] = bits;
		}

		/** @brief Precompute the per-word cumulative ranks. Call once, after all @ref set. */
		constexpr void finish() {
			if (words_.empty()) words_.push_back(0);
			rank_.resize(words_.size() + 1);
			uint32 acc = 0;
			for (std::size_t w = 0; w < words_.size(); ++w) {
				rank_[w] = acc;
				acc += uint32(std::popcount(words_[w]));
			}
			rank_[words_.size()] = acc;
		}

		/** @brief Bit @p i. @pre `i < size()` */
		constexpr bool at(std::size_t i) const {
			atl_assert(i < n_);
			return at_unchecked(i);
		}

		/**
		 * @brief Number of 1-bits in `[0, i)`. @pre `i <= size()` and @ref finish
		 * called. @complexity O(1)
		 */
		constexpr uint32 rank1(std::size_t i) const {
			atl_assert(i <= n_ && !rank_.empty());
			return rank1_unchecked(i);
		}

		/**
		 * @brief Number of 0-bits in `[0, i)`. @pre `i <= size()` and @ref finish
		 * called. @complexity O(1)
		 */
		constexpr uint32 rank0(std::size_t i) const { return uint32(i) - rank1(i); }

		/** @brief Total number of 0-bits. */
		constexpr uint32 zeros() const { return rank0(n_); }

		constexpr std::size_t size() const noexcept { return n_; }
	};
}
