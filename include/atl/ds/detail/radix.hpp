#pragma once
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <bit>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace atl::detail {
	/** @brief A power-of-two branching factor in [2, 2^16]; child slots fit uint16. */
	template<std::size_t Radix>
	concept valid_radix = Radix >= 2 && Radix <= (std::size_t{1} << 16) && (Radix & (Radix - 1)) == 0;

	/** @brief Base-Radix paths for unsigned indices, including native and software uint128. */
	template<std::size_t Radix>
		requires valid_radix<Radix>
	struct radix_index {
		static constexpr std::size_t bits = std::countr_zero(Radix);
		static constexpr std::size_t mask = Radix - 1;

		/** @brief Smallest h with Radix^(h+1) >= n; a height-0 tree is one leaf. */
		static constexpr std::size_t height_for(std::size_t n) {
			return n <= Radix ? 0 : (std::size_t(std::bit_width(n - 1)) - 1) / bits;
		}

		/** @brief The same height for an atl unsigned integer of any supported width. */
		template<atl::unsigned_integral Index>
			requires std::same_as<Index, std::remove_cv_t<Index>>
		static constexpr std::size_t height_for(Index n) {
			if constexpr (sizeof(Index) <= sizeof(std::size_t)) {
				return height_for(static_cast<std::size_t>(n));
			} else {
				if (n <= 1) return 0;
				Index remaining = Index(n - 1);
				std::size_t height = 0;
				while ((remaining >>= bits) != 0) ++height;
				return height;
			}
		}

		/** @brief Child slot at level; the shift must remain within the index width.
		 * @pre level <= (numeric_limits<size_t>::digits - 1) / bits.
		 */
		static constexpr std::size_t digit(std::size_t index, std::size_t level) {
			atl_assert(level <= (std::numeric_limits<std::size_t>::digits - 1) / bits);
			return (index >> (bits * level)) & mask;
		}

		/** @brief Child slot for an unsigned index wider than size_t (e.g. uint128), narrowed only after
		 *        masking. Narrower indices convert to size_t and use the overload above.
		 * @pre level <= (numeric_limits<Index>::digits - 1) / bits.
		 */
		template<atl::unsigned_integral Index>
			requires std::same_as<Index, std::remove_cv_t<Index>> && (sizeof(Index) > sizeof(std::size_t))
		static constexpr std::size_t digit(Index index, std::size_t level) {
			atl_assert(level <= (std::numeric_limits<Index>::digits - 1) / bits);
			return static_cast<std::size_t>((index >> (bits * level)) & Index(mask));
		}
	};
}
