#pragma once
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <vector>

namespace atl {
	/**
	 * @brief Ordered set of integers over a bounded universe `[0, universe)`, with
	 *        predecessor / successor queries — a hierarchical 64-ary bitset.
	 *
	 * Level 0 packs the universe into 64-bit words (bit `x` set ⇔ `x` is present); each
	 * higher level keeps one summary bit per word below it (set ⇔ that word is nonzero).
	 * There are max(1, ceil(log_64(universe))) levels: at most six for a 32-bit
	 * universe. Storage is O(1 + universe / 64) words. Membership is O(1);
	 * insertion, erasure and neighbor queries take O(1 + log_64(universe))
	 * word operations. Construction and clear take O(1 + universe / 64).
	 *
	 * Positions lie in [0, universe); next also accepts universe(). Neighbor
	 * queries return npos when no matching element exists. A zero universe is valid.
	 * Allocation failure propagates from the backing vectors.
	 * @note All operations are constexpr when their backing allocations can be constant-evaluated.

	 */
	class hierarchical_bitset {
	public:
		/** @brief Returned by @ref next / @ref prev / @ref min / @ref max when none exists. */
		static constexpr std::size_t npos = std::size_t(-1);

		/** @brief Construct an empty set over the universe `[0, universe)`. */
		constexpr explicit hierarchical_bitset(std::size_t universe) : universe_(universe) {
			std::size_t m = universe ? universe : 1;
			offset_.push_back(0);
			std::size_t total = 0;
			do {
				std::size_t const words = m / 64 + (m % 64 != 0);
				total += words;
				offset_.push_back(total);
				m = words;
			} while (m > 1);
			levels_ = offset_.size() - 1;
			words_.assign(total, 0);
		}

		/**
		 * @brief Build a set over `[0, universe)` initially containing @p elements (a
		 *        range of positions), bottom-up.
		 *
		 * Sets the leaf bits and then fills every summary level in one pass — O(|elements|
		 * + universe / 64), much faster than inserting the elements one at a time (which
		 * re-propagates the summaries per element). Duplicates are harmless.
		 * @tparam Range Input range with size-convertible values and positions directly constructible from named element lvalues.
		 */
		template<std::ranges::input_range Range>
			requires std::convertible_to<std::ranges::range_value_t<Range>, std::size_t>
				  && requires(std::remove_reference_t<std::ranges::range_reference_t<Range>> &e) { std::size_t(e); }
		constexpr hierarchical_bitset(std::size_t universe, Range &&elements) : hierarchical_bitset(universe) {
			for (auto &&e : elements) {
				std::size_t const x = std::size_t(e);
				atl_assert(x < universe_);
				words_[x >> 6] |= uint64(1) << (x & 63);
			}
			build_levels();
		}

		/** @brief Whether @p x is in the set. @pre `x < universe()` */
		[[nodiscard]] constexpr bool contains(std::size_t x) const {
			atl_assert(x < universe_);
			return words_[x >> 6] >> (x & 63) & 1;
		}

		/** @brief Add @p x (no-op if already present). @pre `x < universe()` */
		constexpr void insert(std::size_t x) {
			atl_assert(x < universe_);
			uint64 &leaf = words_[x >> 6];
			uint64 const bit = uint64(1) << (x & 63);
			if (leaf & bit) return;
			++size_;
			// a word that was already nonzero has its summary bit set on every level above
			bool const was_empty = leaf == 0;
			leaf |= bit;
			if (!was_empty) return;
			for (std::size_t h = 1; h < levels_; ++h) {
				x >>= 6;
				uint64 &word = words_[offset_[h] + (x >> 6)];
				uint64 const old = word;
				word = old | uint64(1) << (x & 63);
				if (old) return;
			}
		}

		/** @brief Remove @p x (no-op if absent). @pre `x < universe()` */
		constexpr void erase(std::size_t x) {
			atl_assert(x < universe_);
			if (!contains(x)) return;
			--size_;
			for (std::size_t h = 0; h < levels_; ++h) {
				std::size_t const w = offset_[h] + (x >> 6);
				words_[w] &= ~(uint64(1) << (x & 63));
				if (words_[w]) break; // word still nonzero → higher summaries unchanged
				x >>= 6;
			}
		}

		/**
		 * @brief Smallest present element `>= x`, or @ref npos. @pre `x <= universe()`,
		 * so the strict successor `next(x + 1)` needs no guard at the last
		 * element (`next(universe())` is @ref npos)
		 */
		[[nodiscard]] constexpr std::size_t next(std::size_t x) const {
			// x == universe_ reaches npos: either its word is past the level's
			// last one, or every bit from x on is zero padding
			atl_assert(x <= universe_);
			for (std::size_t h = 0; h < levels_; ++h) {
				std::size_t const w = x >> 6;
				if (w >= offset_[h + 1] - offset_[h]) return npos; // past this level's last word
				uint64 const d = words_[offset_[h] + w] >> (x & 63);
				if (d) {
					x += std::size_t(std::countr_zero(d));
					for (std::size_t g = h; g-- > 0;) {
						x <<= 6;
						x += std::size_t(std::countr_zero(words_[offset_[g] + (x >> 6)]));
					}
					return x;
				}
				x = w + 1;
			}
			return npos;
		}

		/** @brief Largest present element `<= x`, or @ref npos. @pre `x < universe()` */
		[[nodiscard]] constexpr std::size_t prev(std::size_t x) const {
			atl_assert(x < universe_);
			for (std::size_t h = 0; h < levels_; ++h) {
				uint64 const d = words_[offset_[h] + (x >> 6)] << (63 - (x & 63));
				if (d) {
					x -= std::size_t(std::countl_zero(d));
					for (std::size_t g = h; g-- > 0;) {
						x <<= 6;
						x += 63 - std::size_t(std::countl_zero(words_[offset_[g] + (x >> 6)]));
					}
					return x;
				}
				if ((x >> 6) == 0) return npos; // no earlier word → no predecessor
				x = (x >> 6) - 1;
			}
			return npos;
		}

		/** @brief Smallest present element, or @ref npos if empty. */
		constexpr std::size_t(min)() const { return empty() ? npos : next(0); }

		/** @brief Largest present element, or @ref npos if empty. */
		constexpr std::size_t(max)() const { return empty() ? npos : prev(universe_ - 1); }

		/** @brief Empty the set, keeping the allocated capacity. */
		constexpr void clear() {
			if (size_ == 0) return;
			std::ranges::fill(words_, uint64(0));
			size_ = 0;
		}

		/** @brief Number of present positions. */
		constexpr std::size_t size() const noexcept { return size_; }

		/** @brief Whether no positions are present. */
		constexpr bool empty() const noexcept { return size_ == 0; }

		/** @brief Exclusive upper bound of valid positions. */
		constexpr std::size_t universe() const noexcept { return universe_; }

	private:
		/** @brief Recompute `size_` and every summary level from an already-populated level 0. */
		constexpr void build_levels() {
			std::size_t count = 0;
			for (std::size_t w = 0; w < offset_[1]; ++w) count += std::size_t(std::popcount(words_[w]));
			size_ = count;
			for (std::size_t h = 0; h + 1 < levels_; ++h)
				for (std::size_t w = 0, wn = offset_[h + 1] - offset_[h]; w < wn; ++w)
					if (words_[offset_[h] + w]) words_[offset_[h + 1] + (w >> 6)] |= uint64(1) << (w & 63);
		}

		std::size_t universe_;
		std::size_t size_ = 0;
		std::size_t levels_ = 0;
		std::vector<std::size_t> offset_; // offset_[h] = first word of level h; offset_[levels_] = total
		std::vector<uint64> words_;
	};
}
