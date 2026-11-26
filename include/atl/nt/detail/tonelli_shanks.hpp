#pragma once

#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <optional>
#include <type_traits>

namespace atl::detail {
	/** @brief Canonical square root for a nonzero reduced residue modulo an odd prime. */
	template<typename Arithmetic>
	constexpr std::optional<uint64> tonelli_shanks(uint64 value, uint64 prime, Arithmetic const &arithmetic) {
		// value, non_residue < prime fit the engine's word
		using word = std::remove_cvref_t<decltype(arithmetic.one())>;
		auto const one = arithmetic.one();
		auto const value_form = arithmetic.to_form(word(value));
		if (arithmetic.power(value_form, (prime - 1) / 2) != one) return std::nullopt;
		if (prime % 4 == 3) {
			auto const root = arithmetic.from_form(arithmetic.power(value_form, prime / 4 + 1));
			return (std::ranges::min)(uint64(root), prime - root);
		}
		int order = std::countr_zero(prime - 1);
		auto const odd_part = (prime - 1) >> order;
		uint64 non_residue = 2;
		while (arithmetic.power(arithmetic.to_form(word(non_residue)), (prime - 1) / 2) == one) {
			atl_assert(non_residue < prime - 1);
			++non_residue;
		}
		auto generator = arithmetic.power(arithmetic.to_form(word(non_residue)), odd_part);
		auto remainder = arithmetic.power(value_form, odd_part);
		auto root = arithmetic.power(value_form, odd_part / 2 + 1);
		while (remainder != one) {
			int exponent = 0;
			for (auto power = remainder; power != one; power = arithmetic.mul(power, power)) ++exponent;
			// The remainder lies in the subgroup of squares of the current generator.
			atl_assert(exponent > 0 && exponent < order);
			auto correction = generator;
			for (int i = 0; i < order - exponent - 1; ++i) correction = arithmetic.mul(correction, correction);
			root = arithmetic.mul(root, correction);
			generator = arithmetic.mul(correction, correction);
			remainder = arithmetic.mul(remainder, generator);
			order = exponent;
		}
		auto const canonical = arithmetic.from_form(root);
		return (std::ranges::min)(uint64(canonical), prime - canonical);
	}
}
