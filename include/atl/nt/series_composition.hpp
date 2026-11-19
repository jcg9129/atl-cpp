#pragma once

#include <atl/nt/detail/polynomial_points.hpp>
#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/detail/series_composition.hpp>
#include <atl/nt/detail/series_composition_extensions.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief The composition `outer(inner(x))` modulo `x^length`.
	 *
	 * The whole outer range is a finite polynomial and is used in full, also when `inner[0]` is nonzero (it is
	 * Taylor-shifted by `inner[0]` first). An empty outer gives zeros.
	 * @tparam Engine NTT engine used for the products; engines that do not support the modulus fall back to the
	 *         convolution driver's scalar tiers, and constant evaluation uses scalar transforms,
	 *         or schoolbook products where the field has none.
	 * @tparam Outer Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @tparam Inner Input range with the same coefficient type; read only up to `length` coefficients, absent ones are zero.
	 * @param outer Coefficients of the outer polynomial in increasing degree order.
	 * @param inner Coefficients of the inner series in increasing degree order.
	 * @param length Number of output coefficients.
	 * @return Exactly `length` coefficients, trailing zeros included.
	 * @pre The outer length and `length` are at most `maximum_series_length` (asserted).
	 * @complexity O(M(m) log(m + 1) + M(n) log(n + 1)) time and O(m + n log(n + 1)) memory for outer length `m` and
	 *             precision `n`, where M(k) is the cost of a length-k product.
	 * @note Transpose of the bivariate power projection, with Horner's rule for tiny shapes; race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Outer, prime_mod_int_range Inner>
		requires std::same_as<series_coefficient_t<Outer>, series_coefficient_t<Inner>>
	[[nodiscard]] constexpr auto series_composition(Outer const &outer, Inner const &inner, std::size_t length)
		-> std::vector<series_coefficient_t<Outer>> {
		using Coefficient = series_coefficient_t<Outer>;
		atl_assert(length <= maximum_series_length);
		auto polynomial = detail::points_copy(outer);
		auto prefix = detail::series_prefix(inner, length);
		return detail::composition_dispatch<Engine, detail::composition_algorithm::kinoshita_li, Coefficient>(std::move(polynomial), std::move(prefix), length);
	}

	/**
	 * @brief The power projection `b_i = sum_j weights[j] [x^j] series(x)^i` for `i < count`.
	 *
	 * The whole weight range defines the precision `n`; the series is read modulo `x^n`. This is the transpose of
	 * composition in its outer polynomial. There is no restriction on the constant term or the characteristic;
	 * empty weights give `count` zeros.
	 * @tparam Engine As for `series_composition`.
	 * @tparam Weights Input range of static prime-modulus integers.
	 * @tparam Series Input range with the same coefficient type; absent coefficients are zero.
	 * @param weights Coefficients `weights[j]` in increasing degree order; their number is the precision `n`.
	 * @param series Coefficients of the series in increasing degree order.
	 * @param count Number of output coefficients; may be smaller or larger than `n`.
	 * @return Exactly `count` coefficients.
	 * @pre The weight length and `count` are at most `maximum_series_length` (asserted).
	 * @complexity O(M(n) log(n + 1) + M(count)) time and O(n + count) memory; O(count M(n)) for `count <= 8`.
	 * @note Bivariate Bostan-Mori with paired spectra, repeated products for small counts; race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Weights, prime_mod_int_range Series>
		requires std::same_as<series_coefficient_t<Weights>, series_coefficient_t<Series>>
	[[nodiscard]] constexpr auto series_power_projection(Weights const &weights, Series const &series, std::size_t count)
		-> std::vector<series_coefficient_t<Weights>> {
		using Coefficient = series_coefficient_t<Weights>;
		atl_assert(count <= maximum_series_length);
		auto const functional = detail::points_copy(weights);
		auto const prefix = detail::series_prefix(series, functional.size());
		if (count <= detail::composition_projection_cutoff)
			return detail::composition_project_repeated<Engine, Coefficient>(functional, prefix, count);
		return detail::composition_project<Engine, true, Coefficient>(functional, prefix, count);
	}

	/**
	 * @brief The compositional inverse: the unique `g` with `g[0] = 0` and `f(g(x)) = x` modulo `x^length`.
	 *
	 * Precision zero returns empty and precision one returns `{0}` without looking at `f[1]`.
	 * @tparam Engine As for `series_composition`.
	 * @tparam Series Input range of static prime-modulus integers; absent coefficients are zero.
	 * @param series Coefficients of `f` in increasing degree order.
	 * @param length Number of output coefficients, the precision.
	 * @return Exactly `length` coefficients, with a zero constant term.
	 * @pre `length <= maximum_series_length`; for a positive length `f[0] == 0`, from length two `f[1] != 0`
	 *      (all asserted). There is no `length <= p` restriction.
	 * @complexity O(M(n) log(n + 1)) time for precision `n`.
	 * @note Lagrange inversion through one power projection when `n <= p`, Newton's method with one composition per
	 *       step beyond the characteristic; race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Series>
	[[nodiscard]] constexpr auto series_compositional_inverse(Series const &series, std::size_t length)
		-> std::vector<series_coefficient_t<Series>> {
		using Coefficient = series_coefficient_t<Series>;
		auto const prefix = detail::series_prefix(series, length);
		if (length <= Coefficient::mod()) return detail::composition_inverse_lagrange<Engine, Coefficient>(prefix);
		return detail::composition_inverse_newton<Engine, Coefficient>(prefix);
	}
}
