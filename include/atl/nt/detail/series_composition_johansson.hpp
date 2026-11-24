#pragma once

#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/detail/series_composition_extensions.hpp>
#include <atl/nt/factorial_table.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/power_series_native.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/nt/series_composition_brent_kung.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <span>
#include <vector>

namespace atl::detail {
	/** @brief Johansson fast Lagrange inversion (2015), baby-step/giant-step coefficient products; race pending.
	 * Only the diagonal coefficient [x^(k-1)]h^k is formed for each k, using h=x/f.
	 * @pre A capped precision-sized reversion prefix with length <= p.
	 * @complexity O(sqrt(n) M(n) + n^2) time and O(n sqrt(n)) storage.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_inverse_johansson(std::span<Coefficient const> series) {
		composition_check_inverse<Coefficient>(series);
		auto const length = series.size();
		atl_assert(length <= Coefficient::mod());
		if (length < 2) return std::vector<Coefficient>(length);
		auto const precision = length - 1, block = composition_block_size(precision);
		auto const reciprocal = series_inverse<Engine>(series.subspan(1), precision);
		std::vector<std::vector<Coefficient>> baby(block + 1, std::vector<Coefficient>(precision));
		baby[0][0] = Coefficient(1);
		for (std::size_t i = 1; i <= block; ++i) baby[i] = series_product<Engine, Coefficient>(baby[i - 1], reciprocal, precision);
		std::vector<Coefficient> giant = baby[0], result(length);
		factorial_table<Coefficient> const table(precision);
		for (std::size_t base = 0; base < length; base += block) {
			for (std::size_t offset = 0; offset < block && base + offset < length; ++offset) {
				auto const degree = base + offset;
				if (degree == 0) continue;
				for (std::size_t i = 0; i < degree; ++i) result[degree] += giant[i] * baby[offset][degree - 1 - i];
				result[degree] *= table.inverse_integer(degree);
			}
			if (block < length - base) giant = series_product<Engine, Coefficient>(giant, baby[block], precision);
		}
		return result;
	}

}
