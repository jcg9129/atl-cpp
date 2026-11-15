#pragma once
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::detail {
	/**
	 * @brief An unsigned sort key and the position of the element it was
	 *        computed from. Key must be an unqualified atl unsigned integer (not bool
	 *        or a character type), so records remain assignable. What @ref radix_sort permutes, so the caller
	 *        reorders its own (possibly large) elements once afterwards.
	 */
	template<atl::unsigned_integral Key>
		requires std::same_as<Key, std::remove_cv_t<Key>>
	struct keyed_position {
		Key key;
		uint32 position;
	};

	/**
	 * @brief Stable LSD radix sort of keyed positions by key, @p DigitBits a
	 *        pass, a pass skipped when every key shares its digit.
	 * @pre Fewer than `2^32` items (asserted): the bucket offsets are `uint32`.
	 * @complexity O((n + 2^DigitBits) * bits / DigitBits) time, O(n +
	 *             2^DigitBits) extra memory
	 */
	template<int DigitBits, atl::unsigned_integral Key>
		requires(DigitBits > 0 && DigitBits <= 16 && std::same_as<Key, std::remove_cv_t<Key>>)
	constexpr void radix_sort(std::vector<keyed_position<Key>> &keys) {
		std::size_t const n = keys.size();
		if (n == 0) return;
		atl_assert(n <= std::numeric_limits<uint32>::max());
		constexpr Key mask = Key((uint32(1) << DigitBits) - 1);
		std::vector<keyed_position<Key>> scratch(n);
		std::vector<uint32> count(std::size_t(1) << DigitBits);
		for (int shift = 0; shift < int(sizeof(Key)) * 8; shift += DigitBits) {
			std::ranges::fill(count, 0);
			for (auto const &k : keys) ++count[(k.key >> shift) & mask];
			if (count[(keys[0].key >> shift) & mask] == n) continue;
			uint32 total = 0;
			for (auto &c : count) total += std::exchange(c, total);
			for (auto const &k : keys) scratch[count[(k.key >> shift) & mask]++] = k;
			keys.swap(scratch);
		}
	}
}
