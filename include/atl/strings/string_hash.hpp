#pragma once
#include <atl/nt/static_modulus.hpp>
#include <atl/strings/comparable_text.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/traits/integral.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/hash.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief A range @ref atl::string_hash can hash under `Proj`: an input range
	 *        whose projected symbol is an integer (`char` and `int128`
	 *        included, `bool` excluded).
	 * @tparam R An input range.
	 * @tparam Proj A projection invocable on its range-reference type.
	 */
	template<typename R, typename Proj = std::identity>
	concept hashable_text = std::ranges::input_range<R> && detail::byte_hashable<text_symbol_t<R, Proj>>;

	namespace detail {
		/** @brief The string-hash modulus, the Mersenne prime 2⁶¹ − 1. */
		inline constexpr uint64 string_hash_mod = (uint64(1) << 61) - 1;

		using string_hash_ops = static_modulus<string_hash_mod>;

		/**
		 * @brief A symbol's polynomial digit: a full-width reduction into the
		 *        nonzero residues `[1, mod − 1]` (no blank symbol, int128 not
		 *        truncated). A nonnegative value encodes alike whatever its
		 *        integer type; a negative one goes through its own type's
		 *        unsigned form.
		 */
		template<byte_hashable Symbol>
		[[nodiscard]] constexpr uint64 string_hash_digit(Symbol x) {
			return 1 + uint64(traits::make_unsigned_t<Symbol>(x) % (string_hash_mod - 1));
		}
	}

	/**
	 * @brief Online polynomial hashes modulo 2^61 - 1, with constant-time substring hashes.
	 *
	 * Symbols encode as `1 + unsigned(symbol) % (mod - 1)`, using the symbol's
	 * own unsigned type, including full-width int128. Nonnegative equal values
	 * encode equally across types; negative values can encode differently at
	 * different widths. Distinct values can have identical digits.
	 * Hash equality is probabilistic: equal hashes do not certify equal content.
	 * With distinct encoded sequences and a uniformly random base in [2, mod - 2],
	 * a nonzero difference polynomial of degree d has collision probability at
	 * most d / (mod - 3). The seeded default is not a cryptographic guarantee.
	 * Compare instances only when their bases and symbol encodings agree.
	 * @pre Length + 1 is representable in size_t and fits both vectors' max_size().
	 * @note Explicit bases permit constant evaluation; default_base() uses runtime state.
	 * Moved-from instances and instances with a failed append allocation may only
	 * be destroyed or assigned to.
	 * @complexity O(n) build time and storage (two uint64 tables), O(1) amortized append.
	 */
	class string_hash {
	public:
		/** @brief A hash value: an element of `[0, 2⁶¹ − 1)`. */
		using hash_type = uint64;

		/** @brief The modulus, the Mersenne prime 2⁶¹ − 1. */
		static constexpr uint64 mod = detail::string_hash_mod;

	private:
		using mod_ops = detail::string_hash_ops;

		std::vector<uint64> prefix_; // prefix_[i] = right-aligned hash of the first i elements
		std::vector<uint64> power_;	 // power_[k]  = base_^k mod
		uint64 base_;

		/** @brief `base_^k`: the table when it reaches, else by squaring. */
		[[nodiscard]] constexpr uint64 power_of(std::size_t k) const {
			return k < power_.size() ? power_[k] : mod_ops::power(base_, k);
		}

	public:
		/**
		 * @brief The empty sequence, ready for @ref atl::string_hash::push_back.
		 * @param base The polynomial base, reduced modulo mod.
		 * @pre base modulo mod is at least two (asserted).
		 */
		constexpr explicit string_hash(uint64 base = default_base()) : prefix_{0}, power_{1}, base_(base % mod) {
			atl_assert(base_ >= 2); // 0 and 1 are degenerate bases
		}

		/**
		 * @brief Build the rolling hash of @p seq, each element seen through
		 *        @p proj.
		 * @tparam R An input range with integer projected symbols.
		 * @tparam Proj A projection invocable on range references.
		 * @param seq  A range of integer symbols (e.g. a `std::string`).
		 * @param base The polynomial base (reduced mod `mod`); defaults to the
		 *             process-wide @ref atl::string_hash::default_base. Reduced base must be at least two (asserted).
		 * @param proj Projection to an integer symbol.
		 * @complexity O(n)
		 */
		template<typename R, typename Proj = std::identity>
			requires hashable_text<R, Proj>
		constexpr explicit string_hash(R &&seq, uint64 base = default_base(), Proj proj = {}) : string_hash(base) {
			if constexpr (std::ranges::sized_range<R>) {
				auto const length = std::ranges::size(seq);
				atl_assert(length < (std::numeric_limits<std::size_t>::max)());
				reserve(std::size_t(length));
			}
			for (auto &&e : atl::detail::input_elements(seq)) push_back(std::invoke(proj, std::forward<decltype(e)>(e)));
		}

		/** @brief Number of symbols hashed. */
		constexpr std::size_t size() const noexcept { return prefix_.size() - 1; }

		/** @brief Pre-allocate for n symbols in total.
		 * @param n Total symbol capacity.
		 * @pre n < SIZE_MAX (asserted); n + 1 fits both vectors' max_size().
		 * @complexity O(n) worst-case relocation; no change to size(). */
		constexpr void reserve(std::size_t n) {
			atl_assert(n < (std::numeric_limits<std::size_t>::max)());
			prefix_.reserve(n + 1);
			power_.reserve(n + 1);
		}

		/**
		 * @brief Append one symbol to the hashed sequence.
		 * @tparam Symbol An integer symbol, excluding bool.
		 * @param sym Symbol encoded through its own unsigned integer type.
		 * @complexity O(1) amortized
		 */
		template<detail::byte_hashable Symbol>
		constexpr void push_back(Symbol sym) {
			atl_assert(size() < (std::numeric_limits<std::size_t>::max)() - 1);
			prefix_.push_back(mod_ops::add(mod_ops::mul(prefix_.back(), base_), detail::string_hash_digit(sym)));
			power_.push_back(mod_ops::mul(power_.back(), base_));
		}

		/**
		 * @brief Hash of the half-open subrange `[l, r)`.
		 * @param l Inclusive start.
		 * @param r Exclusive end.
		 * @return A reduced residue in [0, mod), zero for an empty interval.
		 * @pre `l <= r <= size()`
		 * @complexity O(1)
		 */
		[[nodiscard]] constexpr hash_type substring_hash(std::size_t l, std::size_t r) const {
			atl_assert(l <= r && r < prefix_.size());
			return mod_ops::sub(prefix_[r], mod_ops::mul(prefix_[l], power_[r - l]));
		}

		/**
		 * @brief Hash of the whole sequence — the value to key an `unordered_map` by.
		 * @complexity O(1)
		 */
		[[nodiscard]] constexpr hash_type full_hash() const { return substring_hash(0, size()); }

		/**
		 * @brief Whether `[l1, r1)` and `[l2, r2)` have equal lengths and hashes.
		 * @param l1 Inclusive first start.
		 * @param r1 Exclusive first end.
		 * @param l2 Inclusive second start.
		 * @param r2 Exclusive second end.
		 * @pre Both half-open intervals lie in [0, size()] (asserted).
		 * @note Hash collisions can produce false positives.
		 * @complexity O(1)
		 */
		[[nodiscard]] constexpr bool equal(std::size_t l1, std::size_t r1, std::size_t l2, std::size_t r2) const {
			atl_assert(l1 <= r1 && r1 <= size() && l2 <= r2 && r2 <= size());
			return r1 - l1 == r2 - l2 && substring_hash(l1, r1) == substring_hash(l2, r2);
		}

		/**
		 * @brief The hash of the concatenation `left · right`, from the hashes
		 *        of its parts (under this instance's base) and the length of
		 *        `right`.
		 * @param left Reduced hash of the left part under this base.
		 * @param right Reduced hash of the right part under this base.
		 * @param right_length Number of symbols in the right part.
		 * @pre left and right are reduced hashes below mod (asserted).
		 * @complexity O(1) when `right_length <= size()`, else O(log right_length)
		 */
		[[nodiscard]] constexpr hash_type concat(hash_type left, hash_type right, std::size_t right_length) const {
			atl_assert(left < mod && right < mod);
			return mod_ops::add(mod_ops::mul(left, power_of(right_length)), right);
		}

		/**
		 * @brief Hash-based estimate of the longest common prefix of the suffixes starting
		 *        at @p i and @p j.
		 * @note Collisions can overestimate the length.
		 * @param i First suffix start.
		 * @param j Second suffix start.
		 * @pre `i, j <= size()`
		 * @complexity O(log(size() + 1)), a binary search on hash equality
		 */
		[[nodiscard]] constexpr std::size_t lcp(std::size_t i, std::size_t j) const { return lcp(i, *this, j); }

		/**
		 * @brief Hash-based estimate of the longest common prefix of this sequence's suffix
		 *        at @p i and @p other's suffix at @p j.
		 * @note Collisions can overestimate the length.
		 * @param i Suffix start in this sequence.
		 * @param other Sequence hashed with the same base and symbol encoding.
		 * @param j Suffix start in other.
		 * @pre `i <= size()`, `j <= other.size()`, and both instances share a
		 *      base (asserted)
		 * @complexity O(log(min(size() - i, other.size() - j) + 1))
		 */
		[[nodiscard]] constexpr std::size_t lcp(std::size_t i, string_hash const &other, std::size_t j) const {
			atl_assert(base_ == other.base_); // hashes under different bases do not compare
			atl_assert(i <= size() && j <= other.size());
			std::size_t lo = 0, hi = (std::min)(size() - i, other.size() - j);
			while (lo < hi) {
				std::size_t const mid = lo + (hi - lo + 1) / 2;
				if (substring_hash(i, i + mid) == other.substring_hash(j, j + mid))
					lo = mid;
				else
					hi = mid - 1;
			}
			return lo;
		}

		/** @brief The base this instance uses; record it to reproduce a run. */
		[[nodiscard]] constexpr uint64 base() const noexcept { return base_; }

		/**
		 * @brief The process-wide default base: a value in `[2, mod − 2]`
		 *        derived from the process hash seed, so it is random per run
		 *        (resisting anti-hash tests), shared by every instance whatever
		 *        its element type, and pinned by @ref atl::set_hash_seed — call that
		 *        prior to constructing the instances you intend to compare.
		 * @note Not constexpr (reads the process hash seed).
		 */
		[[nodiscard]] static uint64 default_base() {
			// a salt keeps the base independent of the other seeded hashes' keys
			return 2 + detail::splitmix64(detail::hash_seed() ^ 0x5bd1e9955bd1e995ULL) % (mod - 3);
		}
	};

	/**
	 * @brief The polynomial hash of the whole of @p seq under @p base — equal
	 *        to `string_hash(seq, base, proj).full_hash()`, without the
	 *        16 B/symbol tables.
	 * @tparam R An input range with integer projected symbols.
	 * @tparam Proj A projection invocable on range references.
	 * @param seq Input symbols.
	 * @param base Polynomial base, reduced modulo mod.
	 * @param proj Projection to an integer other than bool.
	 * @pre base modulo mod is at least two (asserted).
	 * @return Reduced whole-sequence hash, zero for an empty sequence.
	 * @complexity O(n), O(1) extra space
	 */
	template<typename R, typename Proj = std::identity>
		requires hashable_text<R, Proj>
	[[nodiscard]] constexpr string_hash::hash_type hash_of(R &&seq, uint64 base = string_hash::default_base(), Proj proj = {}) {
		using mod_ops = detail::string_hash_ops;
		uint64 const b = base % string_hash::mod;
		atl_assert(b >= 2); // 0 and 1 are degenerate bases
		uint64 h = 0;
		for (auto &&e : atl::detail::input_elements(seq)) h = mod_ops::add(mod_ops::mul(h, b), detail::string_hash_digit(std::invoke(proj, std::forward<decltype(e)>(e))));
		return h;
	}
}
