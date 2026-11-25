#pragma once

#include <atl/nt/prime_mod_int_core.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <ranges>
#include <vector>

namespace atl {
	/**
	 * @brief Formal derivative of a series given by its coefficients in increasing degree order.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param values The m input coefficients.
	 * @return The `max(m - 1, 0)` coefficients of the derivative, without trimming trailing zeros. Degrees are
	 *         taken in the coefficient field, so degrees that are multiples of the characteristic give zero.
	 * @complexity O(m) field operations and output memory.
	 */
	template<prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_derivative(Range const &values)
		-> std::vector<std::ranges::range_value_t<Range const>> {
		using Coefficient = std::ranges::range_value_t<Range const>;
		std::vector<Coefficient> result;
		if constexpr (std::ranges::sized_range<Range const>) {
			auto const size = std::ranges::size(values);
			if (size != 0) result.reserve(std::size_t(size - 1));
		}
		auto first = std::ranges::begin(values);
		auto const last = std::ranges::end(values);
		if (first == last) return result;
		++first;
		Coefficient degree(1);
		for (; first != last; ++first) {
			result.push_back(*first * degree);
			degree += Coefficient(1);
		}
		return result;
	}

	/**
	 * @brief Formal integral with constant term zero.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param values The m input coefficients.
	 * @return The `m + 1` coefficients of the integral, without trimming; `{0}` for empty input.
	 * @pre `m < Coefficient::mod()`, since the last coefficient divides by m (asserted while consuming the input).
	 * @complexity O(m) field operations and memory; the integer inverses come from the linear recurrence
	 *             `1/i = -(p / i) * (1 / (p mod i))`, so there is no inversion per term.
	 */
	template<prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_integral(Range const &values)
		-> std::vector<std::ranges::range_value_t<Range const>> {
		using Coefficient = std::ranges::range_value_t<Range const>;
		std::vector<Coefficient> result{Coefficient(0)}, inverses{Coefficient(0)};
		if constexpr (std::ranges::sized_range<Range const>) {
			auto const size = std::ranges::size(values);
			atl_assert(size < Coefficient::mod());
			result.reserve(std::size_t(size) + 1);
			inverses.reserve(std::size_t(size) + 1);
		}
		for (auto const &value : values) {
			std::size_t const degree = result.size();
			atl_assert(degree < Coefficient::mod());
			Coefficient const inverse = degree == 1 ? Coefficient(1)
													: -Coefficient(Coefficient::mod() / degree) * inverses[std::size_t(Coefficient::mod() % degree)];
			inverses.push_back(inverse);
			result.push_back(value * inverse);
		}
		return result;
	}
}
