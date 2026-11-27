#pragma once

#include <atl/nt/detail/prime_mod_int_base.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <ranges>
#include <vector>

namespace atl {
	/**
	 * @brief Invert every element, preserving order and length; empty input gives empty output.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param values The elements to invert.
	 * @return The elementwise inverses, computed from prefix products with one field inversion.
	 * @pre Every element is nonzero (asserted).
	 * @complexity O(m) field operations and memory for m elements, plus one modular inversion when m > 0.
	 */
	template<prime_mod_int_range Range>
	[[nodiscard]] constexpr auto batch_inverse(Range const &values)
		-> std::vector<std::ranges::range_value_t<Range const>> {
		using Coefficient = std::ranges::range_value_t<Range const>;
		std::vector<Coefficient> result, prefixes;
		if constexpr (std::ranges::sized_range<Range const>) {
			auto const size = std::size_t(std::ranges::size(values));
			result.reserve(size);
			prefixes.reserve(size);
		}
		Coefficient product(1);
		for (auto const &value : values) {
			atl_assert(value != Coefficient(0));
			prefixes.push_back(product);
			product *= value;
			result.push_back(value);
		}
		if (result.empty()) return result;
		Coefficient inverse_product = product.inverse();
		for (std::size_t i = result.size(); i > 0; --i) {
			Coefficient const value = result[i - 1];
			result[i - 1] = inverse_product * prefixes[i - 1];
			inverse_product *= value;
		}
		return result;
	}
}
