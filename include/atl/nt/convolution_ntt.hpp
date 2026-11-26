#pragma once

/** @file
 * @brief Modular convolution by a direct NTT, without fallback algorithms.
 */
#include <atl/nt/detail/convolution_base.hpp>
#include <atl/nt/detail/ntt_engine_base.hpp>
#include <atl/nt/mod_int_static.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery_static.hpp>
#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {

	/**
	 * @brief Polynomial product by a direct NTT, without reconstruction or floating-point fallbacks.
	 * @tparam Engine Raw NTT engine; constant evaluation uses scalar_ntt.
	 * @pre The modulus models detail::ntt_modulus. For nonempty inputs, lengths are at most
	 *      2^(size_t digits - 2) and detail::planned_ntt_size(n, m) divides Z::mod()-1 (asserted,
	 *      including for schoolbook products). A sufficient bound is bit_ceil(n+m-1) dividing Z::mod()-1.
	 * @return Empty for an empty operand; otherwise n+m-1 coefficients.
	 * @complexity O((n+m) log(n+m)) time and O(n+m) storage; tiny products use schoolbook multiplication.
	 */
	template<ntt_engine Engine = default_ntt_engine, mod_int_range A, mod_int_range B>
		requires std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>
			  && mod_int_range<A const> && mod_int_range<B const>
			  && detail::ntt_modulus<std::ranges::range_value_t<A>>
	constexpr auto convolution_ntt(A const &a, B const &b) {
		using Z = std::ranges::range_value_t<A>;
		std::size_t const n = std::ranges::size(a), m = std::ranges::size(b);
		auto const first_a = std::ranges::begin(a);
		auto const first_b = std::ranges::begin(b);
		if (n == 0 || m == 0) return std::vector<Z>{};
		constexpr std::size_t limit = std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 2);
		atl_assert(n <= limit && m <= limit);
		std::size_t const need = n + m - 1;
		auto const naive = [&] {
			std::vector<Z> c(need);
			for (std::size_t i = 0; i < n; ++i)
				for (std::size_t j = 0; j < m; ++j) c[i + j] += Z(first_a[i]) * Z(first_b[j]);
			return c;
		};

		atl_assert((Z::mod() - 1) % detail::planned_ntt_size(n, m) == 0);
		if ((std::min)(n, m) <= 48) return naive(); // measured crossover
		bool const is_square = detail::is_same_sequence(a, b);
		using UM = decltype(Z::mod());
		auto const at_a = [&](std::size_t i) { return Z(first_a[i]).value(); };
		auto const at_b = [&](std::size_t i) { return Z(first_b[i]).value(); };
		std::vector<UM> raw;
		if consteval {
			raw = detail::raw_convolution<scalar_ntt, Z>(n, at_a, m, at_b, is_square);
		} else {
			raw = detail::raw_convolution<Engine, Z>(n, at_a, m, at_b, is_square);
		}
		std::vector<Z> c(need);
		for (std::size_t i = 0; i < need; ++i) c[i] = Z(raw[i]);
		return c;
	}

}
