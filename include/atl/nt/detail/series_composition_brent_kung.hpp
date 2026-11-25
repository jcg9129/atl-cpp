#pragma once

/** @file
 * @brief The optional Brent-Kung composition strategy and baby-step block sizing.
 */

#include <atl/nt/detail/power_series_native.hpp>
#include <atl/nt/detail/series_composition_extensions_native.hpp>
#include <atl/nt/ntt_engine_core.hpp>
#include <atl/nt/prime_mod_int_core.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace atl::detail {
	/** @brief Ceiling square root for baby-step/giant-step block counts without floating-point narrowing.
	 * @pre length <= maximum_series_length.
	 */
	[[nodiscard]] constexpr std::size_t composition_block_size(std::size_t length) {
		atl_assert(length <= maximum_series_length);
		std::size_t block = 1;
		while (block * block < length) {
			atl_assert(block < maximum_series_length && block < (std::numeric_limits<std::size_t>::max)() / (block + 1));
			++block;
		}
		return block;
	}

	/** @brief Brent–Kung (1978), first algorithm, with a classical rectangular matrix product; race pending.
	 * @pre Positive precision, nonempty outer and zero-constant inner; lengths capped.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_brent_kung(std::span<Coefficient const> outer,
																			std::span<Coefficient const> inner) {
		atl_assert(!outer.empty() && !inner.empty() && outer.size() <= maximum_series_length && inner.size() <= maximum_series_length);
		atl_assert(inner[0] == Coefficient(0));
		auto const length = inner.size(), block = composition_block_size(outer.size());
		std::vector<std::vector<Coefficient>> baby(block + 1, std::vector<Coefficient>(length));
		baby[0][0] = Coefficient(1);
		for (std::size_t i = 1; i <= block; ++i) baby[i] = series_product<Engine, Coefficient>(baby[i - 1], inner, length);
		auto const rows = (outer.size() + block - 1) / block;
		std::vector<std::vector<Coefficient>> matrix(rows, std::vector<Coefficient>(length));
		for (std::size_t row = 0; row < rows; ++row)
			for (std::size_t i = 0; i < block && row * block + i < outer.size(); ++i)
				for (std::size_t j = i; j < length; ++j) matrix[row][j] += outer[row * block + i] * baby[i][j];
		std::vector<Coefficient> result(length);
		for (std::size_t row = rows; row > 0; --row) {
			if (row != rows) result = series_product<Engine, Coefficient>(result, baby[block], length);
			for (std::size_t j = 0; j < length; ++j) result[j] += matrix[row - 1][j];
		}
		return result;
	}

	template<>
	struct composition_strategy<composition_algorithm::brent_kung> {
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		static constexpr std::vector<Coefficient> compose(std::span<Coefficient const> outer, std::span<Coefficient const> inner, std::size_t, std::size_t, std::size_t) {
			return composition_brent_kung<Engine, Coefficient>(outer, inner);
		}
	};
}
