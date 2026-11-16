#pragma once
#include <atl/bits/bitlow.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <array>
#include <bit>
#include <climits>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <utility>

/**
 * @file
 * @brief Linear bases over GF(2) of fixed-width unsigned words: `xor_basis`
 *        (the span of every inserted word, kept in reduced row echelon form)
 *        and `stamped_xor_basis` (each pivot keeps the newest stamp, so
 *        queries can be restricted to the words stamped at or after a
 *        threshold: the range-of-an-array and offline-deletion idiom).
 *
 * References: Chen Xinyang, "On the Linear (XOR) Basis in OI" (2025 national
 * team papers), sections 2 and 3.4; Chang Ruinian, "Linear algebra" (2022),
 * section 8.2 (the latest-deletion-time basis, which is the same structure
 * with deletion times as stamps). Implemented from the descriptions.
 */

namespace atl {
	namespace detail {
		/** @brief Storage width of an unsigned word, including software 128-bit words. */
		template<unsigned_integral Word>
		inline constexpr std::size_t word_bits = sizeof(Word) * CHAR_BIT;

		/**
		 * @brief The index of the highest set bit of @p x.
		 * @pre `x != 0`
		 */
		template<unsigned_integral Word>
		constexpr std::size_t top_bit_index(Word x) noexcept {
			if constexpr (sizeof(Word) <= sizeof(uint64))
				return std::size_t(std::bit_width(uint64(x))) - 1;
			else {
				// Standard bit operations accept native limbs, including for software words.
				uint64 const high = uint64(x >> 64);
				return high != 0 ? 64 + std::size_t(std::bit_width(high)) - 1
								 : std::size_t(std::bit_width(uint64(x))) - 1;
			}
		}

		template<unsigned_integral Word>
		constexpr std::size_t popcount(Word x) noexcept {
			if constexpr (sizeof(Word) <= sizeof(uint64))
				return std::size_t(std::popcount(uint64(x)));
			else
				return std::size_t(std::popcount(uint64(x))) + std::size_t(std::popcount(uint64(x >> 64)));
		}

		/**
		 * @brief The index of the lowest set bit of @p x.
		 * @pre `x != 0`
		 */
		template<unsigned_integral Word>
		constexpr std::size_t low_bit_index(Word x) noexcept {
			if constexpr (sizeof(Word) <= sizeof(uint64))
				return std::size_t(std::countr_zero(uint64(x)));
			else {
				uint64 const low = uint64(x);
				return low != 0 ? std::size_t(std::countr_zero(low))
								: 64 + std::size_t(std::countr_zero(uint64(x >> 64)));
			}
		}
	}

	/**
	 * @brief An input range whose references implicitly convert to the chosen word type.
	 * @tparam R The candidate range, including reference qualification.
	 * @tparam Word The conversion destination.
	 */
	template<typename R, typename Word>
	concept word_range =
		std::ranges::input_range<R> && std::convertible_to<std::ranges::range_reference_t<R>, Word>;

	/**
	 * @brief The linear span over GF(2) of a set of `Word`s (bit vectors of
	 *        width `sizeof(Word) * 8`, addition is XOR), kept as a basis in
	 *        REDUCED row echelon form.
	 *
	 * Slot `b` holds the basis vector whose highest set bit is `b`, or 0; the
	 * occupied slots form `pivot_mask()`. Reduced means no basis vector has
	 * another vector's pivot bit set. That invariant is what makes every query
	 * O(1 + rank) word operations without an elimination dependency chain: reducing a word is one
	 * XOR per pivot bit it contains (the XORs never touch other pivot bits),
	 * the maximum is the XOR of the vectors whose pivot bit the start lacks,
	 * and the k-th smallest representable value is the XOR of the vectors
	 * picked by the bits of k. The reduced basis of a span is unique, so `==`
	 * compares spans.
	 *
	 * A failed insert (the common case once the rank saturates) costs one XOR
	 * per pivot bit of the word; a successful one additionally clears the new
	 * pivot bit from the older vectors, O(rank), and happens at most
	 * `sizeof(Word) * 8` times between calls to `clear()`.
	 *
	 * @tparam Word A cv-unqualified unsigned integer recognized by `atl::unsigned_integral`.
	 * @complexity O(bits) storage; word operations take constant time.
	 */
	template<unsigned_integral Word>
		requires std::same_as<Word, std::remove_cv_t<Word>>
	class xor_basis {
	public:
		/** @brief The vector width in bits, and the maximum rank. */
		static constexpr std::size_t bits = detail::word_bits<Word>;

	private:
		std::array<Word, bits> vectors_{};
		Word pivots_ = 0;
		std::size_t rank_ = 0;

		/** @brief XOR of the basis vectors at the set bits of @p selector,
		 *         which must lie within `pivots_`. */
		constexpr Word combination(Word selector) const noexcept {
			Word sum = 0;
			while (selector != 0) {
				sum ^= vectors_[detail::low_bit_index(selector)];
				selector &= selector - 1;
			}
			return sum;
		}

	public:
		/** @brief Construct the zero space. @complexity O(bits). */
		constexpr xor_basis() noexcept = default;

		/**
		 * @brief The basis of the span of @p values.
		 * @tparam R An input range satisfying `word_range<R, Word>`.
		 * @param values Words whose span is retained, consumed once.
		 * @complexity O(bits + |values| (1 + rank) + rank^2) time; O(bits) space.
		 */
		template<word_range<Word> R>
		constexpr explicit xor_basis(R &&values) {
			for (auto &&value : values) insert(Word(value));
		}

		/**
		 * @brief Add @p x to the spanning set.
		 * @param x The word to add; zero and dependent words are accepted.
		 * @return Whether @p x was independent of the words inserted so far
		 *         (the rank grew); 0 is never independent.
		 * @complexity O(1 + popcount(x & pivot_mask())), plus O(rank) when the
		 *             insert succeeds
		 */
		constexpr bool insert(Word x) noexcept {
			x = reduce(x);
			if (x == 0) return false;
			std::size_t const b = detail::top_bit_index(x);
			Word const bit = Word(Word(1) << b);
			// only vectors with a higher pivot can hold bit b (x has no pivot
			// bits, so XOR-ing it in keeps the other vectors reduced)
			Word higher = Word(pivots_ & Word(~(bit | (bit - 1))));
			while (higher != 0) {
				std::size_t const at = detail::low_bit_index(higher);
				if (vectors_[at] & bit) vectors_[at] ^= x;
				higher &= higher - 1;
			}
			vectors_[b] = x;
			pivots_ |= bit;
			++rank_;
			return true;
		}

		/**
		 * @brief Add every vector of @p other to the spanning set, leaving the
		 *        basis of the sum of the two spans.
		 * @param other The other span; self-merge is allowed.
		 * @complexity O(1 + other.rank() * bits).
		 */
		constexpr void merge(xor_basis const &other) noexcept {
			Word rest = other.pivots_;
			while (rest != 0 && rank_ < bits) {
				insert(other.vectors_[detail::low_bit_index(rest)]);
				rest &= rest - 1;
			}
		}

		/**
		 * @brief The smallest value of `x ^ v` over the span: @p x with every
		 *        pivot bit cleared. It is 0 exactly when @p x is in the span.
		 * @param x The initial word.
		 * @return The minimum XOR with any representable word.
		 * @complexity O(1 + popcount(x & pivot_mask())).
		 */
		constexpr Word reduce(Word x) const noexcept { return Word(x ^ combination(Word(x & pivots_))); }

		/**
		 * @brief Test membership in the span, including zero.
		 * @param x The candidate word.
		 * @return Whether some subset XOR equals @p x.
		 * @complexity O(1 + rank).
		 */
		constexpr bool contains(Word x) const noexcept { return reduce(x) == 0; }

		/**
		 * @brief Minimize XOR with a representable word.
		 * @param start The initial word.
		 * @return The smallest `start ^ v` over the span, as with `reduce`.
		 * @complexity O(1 + rank).
		 */
		constexpr Word min_xor(Word start) const noexcept { return reduce(start); }

		/**
		 * @brief The largest value of `start ^ v` over the span; `max_xor()`
		 *        is the largest representable value.
		 * @param start The initial word, defaulting to zero.
		 * @return The maximum XOR with a representable word.
		 * @complexity O(1 + rank)
		 */
		constexpr Word max_xor(Word start = 0) const noexcept {
			return Word(start ^ combination(Word(pivots_ & Word(~start))));
		}

		/**
		 * @brief The smallest nonzero representable value: the basis vector
		 *        with the lowest pivot.
		 * @pre `rank() > 0` (asserted with `ATL_ASSERT`).
		 * @return The least nonzero word in the span.
		 * @complexity O(1)
		 */
		constexpr Word min_nonzero() const noexcept {
			atl_assert(rank_ > 0);
			return vectors_[detail::low_bit_index(pivots_)];
		}

		/**
		 * @brief The @p k-th smallest representable value, 0-indexed, so
		 *        `kth_smallest(0) == 0` and `kth_smallest(2^rank - 1) ==
		 *        max_xor()`.
		 * @param k The zero-based index, represented in Word.
		 * @pre `k < 2^rank()` (asserted with `ATL_ASSERT`; every Word fits at full rank).
		 * @return The word at index @p k in increasing unsigned order.
		 * @complexity O(1 + rank)
		 */
		constexpr Word kth_smallest(Word k) const noexcept {
			atl_assert(rank_ == bits || (k >> rank_) == 0);
			Word sum = 0, rest = pivots_;
			// branch-free select: the bits of k are random, a branch on them
			// mispredicts half the time
			for (; k != 0; k >>= 1) {
				sum ^= vectors_[detail::low_bit_index(rest)] & Word(Word(k & 1) * Word(~Word(0)));
				rest &= rest - 1;
			}
			return sum;
		}

		/**
		 * @brief How many representable values are strictly less than
		 *        @p bound (the inverse of `kth_smallest`).
		 * @param bound Exclusive upper bound.
		 * @return A count in [0, bound], which always fits Word, even at full rank.
		 * @complexity O(1 + rank)
		 */
		constexpr Word count_below(Word bound) const noexcept {
			// The representable values in increasing order are indexed by their
			// pivot bits read as a number (bit b of a value depends only on the
			// pivots >= b). c is the value whose pivot bits equal bound's; it
			// first differs from bound at a non-pivot bit t. Every value whose
			// pivot bits above t are smaller than bound's is below bound, and
			// so is every value sharing them when bound has bit t set (and c
			// does not). Sums of independent popcounts, no dependent chain.
			Word const c = combination(Word(bound & pivots_));
			Word const differ = Word(c ^ bound);
			Word above = Word(bound & pivots_), count = 0;
			if (differ != 0) {
				std::size_t const t = detail::top_bit_index(differ);
				Word const below = Word((Word(1) << t) - 1);
				above &= Word(~below);
				if ((bound >> t) & 1) count = Word(Word(1) << detail::popcount(Word(pivots_ & below)));
			}
			for (; above != 0; above &= above - 1) {
				Word const lower = Word(lowest_one(above) - 1);
				count += Word(Word(1) << detail::popcount(Word(pivots_ & lower)));
			}
			return count;
		}

		/**
		 * @brief The number of representable values, `2^rank()` (0 included).
		 * @pre `rank() < 64` (asserted with `ATL_ASSERT`).
		 * @return The count, including zero.
		 * @complexity O(1).
		 */
		constexpr uint64 representable_count() const noexcept {
			atl_assert(rank_ < 64);
			return uint64(1) << rank_;
		}

		/** @brief The dimension of the span. @return The rank in [0, bits]. @complexity O(1). */
		constexpr std::size_t rank() const noexcept { return rank_; }

		/** @brief Test for the zero space. @return Whether rank is zero. @complexity O(1). */
		constexpr bool empty() const noexcept { return rank_ == 0; }

		/** @brief The occupied pivot bits: bit `b` is set iff some basis vector
		 *         has `b` as its highest set bit. @return The mask. @complexity O(1). */
		constexpr Word pivot_mask() const noexcept { return pivots_; }

		/**
		 * @brief The basis vector with highest set bit @p b, or 0 when there is
		 *        none.
		 * @param b The bit position, indexed from zero.
		 * @pre `b < bits` (asserted with `ATL_ASSERT`).
		 * @return The vector at slot @p b, or zero for an empty slot.
		 * @complexity O(1).
		 */
		constexpr Word vector(std::size_t b) const noexcept {
			atl_assert(b < bits);
			return vectors_[b];
		}

		/** @brief Reset to the zero space. @complexity O(bits). */
		constexpr void clear() noexcept { *this = xor_basis(); }

		/** @brief Compare spans via their unique reduced bases. @return Whether the spans coincide. @complexity O(bits). */
		constexpr bool operator==(xor_basis const &) const noexcept = default;
	};

	/**
	 * @brief Deduce the unsigned word type from the range value type.
	 * @tparam R An input range with an unsigned integral value type.
	 */
	template<std::ranges::input_range R>
		requires unsigned_integral<std::ranges::range_value_t<R>>
	xor_basis(R &&) -> xor_basis<std::ranges::range_value_t<R>>;

	/**
	 * @brief A GF(2) linear basis whose vectors carry stamps, answering
	 *        queries about the span of only the inserted words stamped at or
	 *        after a threshold.
	 *
	 * Insertion runs the echelon elimination, but at every occupied pivot it
	 * keeps whichever of the two words has the newer (larger) stamp in the
	 * slot and carries on eliminating with the older one. Afterwards, for
	 * every threshold `from`, the slots stamped `>= from` are a basis of the
	 * span of the inserted words stamped `>= from`, in any insertion order.
	 *
	 * Range queries over an array: insert `a[i]` with stamp `i` for `i < r`;
	 * then `max_xor(l)`, `contains(x, l)`, `rank(l)` answer for `a[l, r)`.
	 * Sort offline queries by `r`, or keep a copy per prefix (the object is a
	 * fixed-size value, trivially copyable when Stamp is) to answer online. Offline
	 * deletion: stamp each word with its deletion time and query with the
	 * current time plus one.
	 *
	 * The vectors stay in plain (not reduced) echelon form: reducing would mix
	 * stamps. `basis_from` hands back a reduced `xor_basis` for the queries
	 * this class does not repeat (k-th smallest, counts, min).
	 *
	 * @tparam Word  A cv-unqualified vector type, as in `xor_basis`.
	 * @tparam Stamp A totally ordered, semiregular stamp; defaults to uint32.
	 * @note Bounds assume constant-time stamp operations. No exception guarantee
	 *       is provided if stamp construction, assignment, comparison or swap throws.
	 * @complexity O(bits) words and stamps of storage.
	 */
	template<unsigned_integral Word, typename Stamp = uint32>
		requires std::same_as<Word, std::remove_cv_t<Word>> && std::totally_ordered<Stamp> && std::semiregular<Stamp>
	class stamped_xor_basis {
	public:
		/** @brief The vector width in bits, and the maximum rank. */
		static constexpr std::size_t bits = detail::word_bits<Word>;

	private:
		std::array<Word, bits> vectors_{};
		std::array<Stamp, bits> stamps_{};
		std::size_t rank_ = 0;

	public:
		/** @brief Construct the zero space. @complexity O(bits). */
		constexpr stamped_xor_basis() = default;

		/**
		 * @brief Add @p x with stamp @p stamp to the spanning set.
		 * @param x The word to add.
		 * @param stamp Its stamp; ties and arbitrary insertion order are accepted.
		 * @return Whether @p x was independent of every word inserted so far
		 *         (the rank of the whole span grew).
		 * @complexity O(bits)
		 */
		constexpr bool insert(Word x, Stamp stamp) {
			while (x != 0) {
				std::size_t const b = detail::top_bit_index(x);
				if (vectors_[b] == 0) {
					vectors_[b] = x;
					stamps_[b] = std::move(stamp);
					++rank_;
					return true;
				}
				if (stamps_[b] < stamp) {
					std::swap(vectors_[b], x);
					std::swap(stamps_[b], stamp);
				}
				x ^= vectors_[b]; // the older word, eliminated by the newer
			}
			return false;
		}

		/**
		 * @brief Whether @p x is the XOR of some subset of the words stamped
		 *        `>= from`.
		 * @param x The candidate word.
		 * @param from Inclusive lower stamp bound.
		 * @return Whether the restricted span contains @p x (always true for zero).
		 * @complexity O(bits)
		 */
		constexpr bool contains(Word x, Stamp const &from) const {
			while (x != 0) {
				std::size_t const b = detail::top_bit_index(x);
				if (vectors_[b] == 0 || stamps_[b] < from) return false;
				x ^= vectors_[b];
			}
			return true;
		}

		/**
		 * @brief The largest value of `start ^ v` over the span of the words
		 *        stamped `>= from`.
		 * @param from Inclusive lower stamp bound.
		 * @param start The initial word, defaulting to zero.
		 * @return The maximum XOR over the restricted span.
		 * @complexity O(bits)
		 */
		constexpr Word max_xor(Stamp const &from, Word start = 0) const {
			for (std::size_t b = bits; b-- > 0;)
				if (vectors_[b] != 0 && !(stamps_[b] < from) && !((start >> b) & 1)) start ^= vectors_[b];
			return start;
		}

		/**
		 * @brief The dimension of the span of the words stamped `>= from`.
		 * @param from Inclusive lower stamp bound.
		 * @return The restricted rank in [0, bits].
		 * @complexity O(bits)
		 */
		constexpr std::size_t rank(Stamp const &from) const {
			std::size_t count = 0;
			for (std::size_t b = 0; b < bits; ++b) count += vectors_[b] != 0 && !(stamps_[b] < from);
			return count;
		}

		/** @brief Dimension of the full span. @return The rank in [0, bits]. @complexity O(1). */
		constexpr std::size_t rank() const noexcept { return rank_; }

		/**
		 * @brief The reduced basis of the span of the words stamped `>= from`.
		 * @param from Inclusive lower stamp bound.
		 * @return An independent reduced basis value.
		 * @complexity O(bits^2)
		 */
		constexpr xor_basis<Word> basis_from(Stamp const &from) const {
			xor_basis<Word> basis;
			for (std::size_t b = 0; b < bits; ++b)
				if (vectors_[b] != 0 && !(stamps_[b] < from)) basis.insert(vectors_[b]);
			return basis;
		}

		/**
		 * @brief The echelon vector with highest set bit @p b, or 0.
		 * @param b The bit position, indexed from zero.
		 * @pre `b < bits` (asserted with `ATL_ASSERT`).
		 * @return The vector at slot @p b, or zero for an empty slot.
		 * @complexity O(1).
		 */
		constexpr Word vector(std::size_t b) const noexcept {
			atl_assert(b < bits);
			return vectors_[b];
		}

		/**
		 * @brief The stamp of the vector with highest set bit @p b
		 *        (meaningful only when `vector(b) != 0`).
		 * @param b The bit position, indexed from zero.
		 * @pre `b < bits` (asserted with `ATL_ASSERT`).
		 * @return A reference into this object; insertion, clear or assignment may
		 *         change its value, and destruction invalidates it.
		 * @complexity O(1).
		 */
		constexpr Stamp const &stamp(std::size_t b) const noexcept {
			atl_assert(b < bits);
			return stamps_[b];
		}

		/** @brief Reset to the zero space. @complexity O(bits). */
		constexpr void clear() { *this = stamped_xor_basis(); }
	};
}
