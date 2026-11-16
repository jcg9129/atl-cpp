#pragma once
#include <atl/bits/xor_basis.hpp>
#include <atl/types/concepts.hpp>
#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

/**
 * @file
 * @brief GF(2) basis contenders that lost the race to the library's reduced
 *        row echelon `atl::xor_basis` (atl/bits/xor_basis.hpp).
 *
 * Shared contract, the library class's core: `insert` (returns whether the
 * rank grew), `contains`, `max_xor(start)`, `rank`, and `reduced()` handing
 * back the equivalent library basis for the order queries. Both keep the
 * textbook plain echelon form (slot b holds some vector with top bit b), so
 * every insert and query is a dependent chain: each XOR decides which slot
 * the next step reads.
 *
 * The certifying oracle is the subset-XOR closure in tests/bits/xor_basis.cpp
 * (`brute_span`); `xor_basis_by_echelon_jump` is the readable textbook reference.
 */
namespace atl::experimental {
	/**
	 * @brief Plain echelon basis; elimination jumps straight to the next
	 *        highest set bit (`bit_width`) instead of scanning every slot.
	 *
	 * LOST the race (benchmarks/bits/xor_basis.cpp, measured
	 * 2026-10-15, x86-64 Xeon 8358, 1e6 operations, asserts off, ms, plain -O2 /
	 * -march=native): insert on random full-rank uint64 140 / 129 vs the library
	 * 33.6 / 23.8; uint32 rank 20 of 30 bits 48.9 / 42.1 vs 14.3 / 12.2;
	 * max_xor on full-rank uint64 267 / 268 vs 33.6 / 22.5; contains 129 /
	 * 121 vs 32.9 / 22.4. Every step's XOR decides the slot the next step
	 * reads, while the reduced form's XORs are independent and only touch
	 * the word's pivot bits.
	 * Default construction creates the zero space in O(bits) time.
	 * @tparam Word A cv-unqualified unsigned integer recognized by `atl::unsigned_integral`.
	 * @complexity O(1 + rank) insert and contains; O(bits) max_xor; O(bits) storage.
	 */
	template<unsigned_integral Word>
		requires std::same_as<Word, std::remove_cv_t<Word>>
	class xor_basis_by_echelon_jump {
	public:
		/** @brief The vector width in bits. */
		static constexpr std::size_t bits = atl::detail::word_bits<Word>;

	private:
		std::array<Word, bits> vectors_{};
		std::size_t rank_ = 0;

	public:
		/**
		 * @brief Extend the span with a word, including zero or a dependent word.
		 * @param x The word to insert.
		 * @return Whether the rank grew.
		 * @complexity O(bits).
		 */
		constexpr bool insert(Word x) noexcept {
			while (x != 0) {
				std::size_t const b = atl::detail::top_bit_index(x);
				if (vectors_[b] == 0) {
					vectors_[b] = x;
					++rank_;
					return true;
				}
				x ^= vectors_[b];
			}
			return false;
		}

		/**
		 * @brief Test membership in the span, including zero.
		 * @param x The candidate word.
		 * @return Whether the span contains @p x.
		 * @complexity O(bits).
		 */
		constexpr bool contains(Word x) const noexcept {
			while (x != 0) {
				std::size_t const b = atl::detail::top_bit_index(x);
				if (vectors_[b] == 0) return false;
				x ^= vectors_[b];
			}
			return true;
		}

		/**
		 * @brief Maximize XOR with a representable word.
		 * @param start The initial word, defaulting to zero.
		 * @return The largest `start ^ v` over the span.
		 * @complexity O(bits).
		 */
		constexpr Word max_xor(Word start = 0) const noexcept {
			for (std::size_t b = bits; b-- > 0;)
				if (!((start >> b) & 1)) start ^= vectors_[b];
			return start;
		}

		/** @brief The dimension of the span. @return The rank in [0, bits]. @complexity O(1). */
		constexpr std::size_t rank() const noexcept { return rank_; }

		/**
		 * @brief Convert to the promoted representation without changing the span.
		 * @return An independent reduced basis value.
		 * @complexity O(bits^2) time and O(bits) extra space.
		 */
		constexpr atl::xor_basis<Word> reduced() const noexcept {
			atl::xor_basis<Word> basis;
			for (Word const v : vectors_)
				if (v != 0) basis.insert(v);
			return basis;
		}
	};

	/**
	 * @brief Plain echelon basis with the classic elimination loop: test every
	 *        bit from the top, whether or not it is set.
	 *
	 * LOST the race, behind `xor_basis_by_echelon_jump` too (benchmarks/bits/xor_basis.cpp,
	 * 2026-10-15, x86-64 Xeon 8358, 1e6 operations, asserts off): insert on
	 * random full-rank uint64 287 / 279 ms, contains 267 / 267 ms (plain
	 * -O2 / -march=native). Scanning tests unset bits as well as set bits;
	 * the data-dependent branches are costly on these workloads.
	 * Default construction creates the zero space in O(bits) time.
	 * @tparam Word A cv-unqualified unsigned integer recognized by `atl::unsigned_integral`.
	 * @complexity O(bits) insert, contains and max_xor; O(bits) storage.
	 */
	template<unsigned_integral Word>
		requires std::same_as<Word, std::remove_cv_t<Word>>
	class xor_basis_by_echelon_scan {
	public:
		/** @brief The vector width in bits. */
		static constexpr std::size_t bits = atl::detail::word_bits<Word>;

	private:
		std::array<Word, bits> vectors_{};
		std::size_t rank_ = 0;

	public:
		/**
		 * @brief Extend the span with a word, including zero or a dependent word.
		 * @param x The word to insert.
		 * @return Whether the rank grew.
		 * @complexity O(bits).
		 */
		constexpr bool insert(Word x) noexcept {
			for (std::size_t b = bits; b-- > 0;) {
				if (!((x >> b) & 1)) continue;
				if (vectors_[b] == 0) {
					vectors_[b] = x;
					++rank_;
					return true;
				}
				x ^= vectors_[b];
			}
			return false;
		}

		/**
		 * @brief Test membership in the span, including zero.
		 * @param x The candidate word.
		 * @return Whether the span contains @p x.
		 * @complexity O(bits).
		 */
		constexpr bool contains(Word x) const noexcept {
			for (std::size_t b = bits; b-- > 0;)
				if ((x >> b) & 1) x ^= vectors_[b];
			return x == 0;
		}

		/**
		 * @brief Maximize XOR with a representable word.
		 * @param start The initial word, defaulting to zero.
		 * @return The largest `start ^ v` over the span.
		 * @complexity O(bits).
		 */
		constexpr Word max_xor(Word start = 0) const noexcept {
			for (std::size_t b = bits; b-- > 0;)
				if (!((start >> b) & 1)) start ^= vectors_[b];
			return start;
		}

		/** @brief The dimension of the span. @return The rank in [0, bits]. @complexity O(1). */
		constexpr std::size_t rank() const noexcept { return rank_; }

		/**
		 * @brief Convert to the promoted representation without changing the span.
		 * @return An independent reduced basis value.
		 * @complexity O(bits^2) time and O(bits) extra space.
		 */
		constexpr atl::xor_basis<Word> reduced() const noexcept {
			atl::xor_basis<Word> basis;
			for (Word const v : vectors_)
				if (v != 0) basis.insert(v);
			return basis;
		}
	};

	/**
	 * @brief `xor_basis::count_below` as a greedy descent over the pivots
	 *        from the top: k gains bit i when the value with the chosen high
	 *        part of k and bit i set, low part zero, is still below bound.
	 *
	 * lost to the library's pivot-pattern count (benchmarks/bits/xor_basis.cpp,
	 * 2026-10-15, x86-64 Xeon 8358, asserts off, 1e6 queries at
	 * rank 50): 208 / 130 ms vs 107 / 48.2 ms, plain -O2 / -march=native.
	 * Each step's comparison waits for the previous step's value.
	 * @tparam Word A cv-unqualified unsigned integer recognized by `atl::unsigned_integral`.
	 * @param basis The reduced basis to query.
	 * @param bound Exclusive upper bound.
	 * @return The number of representable words less than bound; it fits Word.
	 * @complexity O(1 + rank) time and O(1) extra space.
	 */
	template<unsigned_integral Word>
		requires std::same_as<Word, std::remove_cv_t<Word>>
	constexpr Word count_below_by_greedy_descent(atl::xor_basis<Word> const &basis, Word bound) noexcept {
		if (bound == 0) return 0;
		Word k = 0, value = 0, rest = basis.pivot_mask();
		for (std::size_t i = basis.rank(); i-- > 0;) {
			std::size_t const b = atl::detail::top_bit_index(rest);
			rest ^= Word(Word(1) << b);
			Word const take = Word(Word((value ^ basis.vector(b)) < bound) * Word(~Word(0)));
			value ^= basis.vector(b) & take;
			k |= Word(Word(1) << i) & take;
		}
		return Word(k + 1);
	}
}
