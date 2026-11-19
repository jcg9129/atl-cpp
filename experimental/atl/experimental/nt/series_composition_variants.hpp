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

/**
 * @file
 * @brief Contenders for composition, power projection and compositional inverse of `<atl/nt/series_composition.hpp>`.
 *
 * Each one has the template parameters, argument order, result and preconditions of the library function it races,
 * unless its own comment says otherwise. All races are pending: no measurement decides any of them yet
 * (benchmarks/nt/series_composition.cpp).
 */

namespace atl::experimental {
	/**
	 * @brief Composition by the Brent–Kung baby-step/giant-step method with a classical rectangular matrix product.
	 *
	 * O(sqrt(m) M(n) + m n) time and O(n sqrt(m)) storage for outer length `m` and precision `n`. Race pending against `atl::series_composition`.
	 * @pre Same contracts as `atl::series_composition`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Outer, prime_mod_int_range Inner>
		requires std::same_as<atl::series_coefficient_t<Outer>, atl::series_coefficient_t<Inner>>
	[[nodiscard]] constexpr auto series_composition_by_brent_kung(Outer const &outer, Inner const &inner, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Outer>> {
		using Coefficient = atl::series_coefficient_t<Outer>;
		atl_assert(length <= maximum_series_length);
		auto polynomial = atl::detail::points_copy(outer);
		auto prefix = atl::detail::series_prefix(inner, length);
		return atl::detail::composition_dispatch<Engine, atl::detail::composition_algorithm::brent_kung, Coefficient>(std::move(polynomial), std::move(prefix), length);
	}

	/**
	 * @brief Composition by Horner's rule with fast truncated products, one per outer coefficient; race pending.
	 * @pre Same contracts as `atl::series_composition`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Outer, prime_mod_int_range Inner>
		requires std::same_as<atl::series_coefficient_t<Outer>, atl::series_coefficient_t<Inner>>
	[[nodiscard]] constexpr auto series_composition_by_fast_horner(Outer const &outer, Inner const &inner, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Outer>> {
		using Coefficient = atl::series_coefficient_t<Outer>;
		atl_assert(length <= maximum_series_length);
		auto polynomial = atl::detail::points_copy(outer);
		auto prefix = atl::detail::series_prefix(inner, length);
		return atl::detail::composition_dispatch<Engine, atl::detail::composition_algorithm::horner, Coefficient>(std::move(polynomial), std::move(prefix), length);
	}

	/**
	 * @brief Composition by Horner's rule with schoolbook truncated products; cubic for dense inputs; race pending.
	 * @pre Same contracts as `atl::series_composition`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Outer, prime_mod_int_range Inner>
		requires std::same_as<atl::series_coefficient_t<Outer>, atl::series_coefficient_t<Inner>>
	[[nodiscard]] constexpr auto series_composition_by_schoolbook_horner(Outer const &outer, Inner const &inner, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Outer>> {
		using Coefficient = atl::series_coefficient_t<Outer>;
		atl_assert(length <= maximum_series_length);
		auto polynomial = atl::detail::points_copy(outer);
		auto prefix = atl::detail::series_prefix(inner, length);
		return atl::detail::composition_dispatch<Engine, atl::detail::composition_algorithm::schoolbook, Coefficient>(std::move(polynomial), std::move(prefix), length);
	}

	/**
	 * @brief The library's composition with its two Horner crossovers as arguments, for the cutoff sweep; race pending.
	 * @param horner_cutoff Outer lengths up to this use fast-product Horner instead of the transposed projection.
	 * @param schoolbook_cutoff Precisions up to this use schoolbook Horner.
	 * @pre Same contracts as `atl::series_composition`; both cutoffs positive and below `maximum_series_length`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Outer, prime_mod_int_range Inner>
		requires std::same_as<atl::series_coefficient_t<Outer>, atl::series_coefficient_t<Inner>>
	[[nodiscard]] constexpr auto series_composition_by_cutoffs(Outer const &outer, Inner const &inner, std::size_t length,
															   std::size_t horner_cutoff, std::size_t schoolbook_cutoff)
		-> std::vector<atl::series_coefficient_t<Outer>> {
		using Coefficient = atl::series_coefficient_t<Outer>;
		atl_assert(length <= maximum_series_length);
		auto polynomial = atl::detail::points_copy(outer);
		auto prefix = atl::detail::series_prefix(inner, length);
		return atl::detail::composition_dispatch<Engine, atl::detail::composition_algorithm::kinoshita_li, Coefficient>(std::move(polynomial), std::move(prefix), length, horner_cutoff, schoolbook_cutoff);
	}

	/**
	 * @brief Power projection by bivariate Bostan–Mori with plain padded products, two per step; race pending.
	 * @pre Same contracts as `atl::series_power_projection`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Weights, prime_mod_int_range Series>
		requires std::same_as<atl::series_coefficient_t<Weights>, atl::series_coefficient_t<Series>>
	[[nodiscard]] constexpr auto series_power_projection_by_padded_bostan_mori(Weights const &weights, Series const &series, std::size_t count)
		-> std::vector<atl::series_coefficient_t<Weights>> {
		using Coefficient = atl::series_coefficient_t<Weights>;
		atl_assert(count <= maximum_series_length);
		auto const functional = atl::detail::points_copy(weights);
		auto const prefix = atl::detail::series_prefix(series, functional.size());
		return atl::detail::composition_project<Engine, false, Coefficient>(functional, prefix, count);
	}

	/**
	 * @brief Power projection by repeated truncated multiplication by the series and one dot product per power; race pending.
	 *
	 * The library uses this for counts up to a small cutoff.
	 * @pre Same contracts as `atl::series_power_projection`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Weights, prime_mod_int_range Series>
		requires std::same_as<atl::series_coefficient_t<Weights>, atl::series_coefficient_t<Series>>
	[[nodiscard]] constexpr auto series_power_projection_by_repeated_products(Weights const &weights, Series const &series, std::size_t count)
		-> std::vector<atl::series_coefficient_t<Weights>> {
		using Coefficient = atl::series_coefficient_t<Weights>;
		atl_assert(count <= maximum_series_length);
		auto const functional = atl::detail::points_copy(weights);
		auto const prefix = atl::detail::series_prefix(series, functional.size());
		return atl::detail::composition_project_repeated<Engine, Coefficient>(functional, prefix, count);
	}

	/**
	 * @brief Power projection by bivariate Bostan–Mori with the paired-spectra schedule at every count; race pending.
	 *
	 * The library dispatches small counts to repeated products instead.
	 * @pre Same contracts as `atl::series_power_projection`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Weights, prime_mod_int_range Series>
		requires std::same_as<atl::series_coefficient_t<Weights>, atl::series_coefficient_t<Series>>
	[[nodiscard]] constexpr auto series_power_projection_by_bostan_mori(Weights const &weights, Series const &series, std::size_t count)
		-> std::vector<atl::series_coefficient_t<Weights>> {
		using Coefficient = atl::series_coefficient_t<Weights>;
		atl_assert(count <= maximum_series_length);
		auto const functional = atl::detail::points_copy(weights);
		auto const prefix = atl::detail::series_prefix(series, functional.size());
		return atl::detail::composition_project<Engine, true, Coefficient>(functional, prefix, count);
	}

	/**
	 * @brief Reversion by Newton's method with one composition per step; valid in every characteristic; race pending.
	 *
	 * The library uses it beyond the characteristic.
	 * @pre Same contracts as `atl::series_compositional_inverse`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Series>
	[[nodiscard]] constexpr auto series_compositional_inverse_by_composition_newton(Series const &series, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Series>> {
		using Coefficient = atl::series_coefficient_t<Series>;
		auto const prefix = atl::detail::series_prefix(series, length);
		return atl::detail::composition_inverse_newton<Engine, Coefficient>(prefix);
	}

	/**
	 * @brief Reversion by Lagrange inversion with Johansson's baby-step/giant-step coefficient products; race pending.
	 * @pre Same contracts as `atl::series_compositional_inverse`; `length <= p`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Series>
	[[nodiscard]] constexpr auto series_compositional_inverse_by_baby_giant_lagrange(Series const &series, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Series>> {
		using Coefficient = atl::series_coefficient_t<Series>;
		auto const prefix = atl::detail::series_prefix(series, length);
		return atl::detail::composition_inverse_johansson<Engine, Coefficient>(prefix);
	}

	/**
	 * @brief Reversion by the direct coefficient recurrence, cubic, for tiny precisions; race pending.
	 * @pre Same contracts as `atl::series_compositional_inverse`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Series>
	[[nodiscard]] constexpr auto series_compositional_inverse_by_coefficient_recurrence(Series const &series, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Series>> {
		using Coefficient = atl::series_coefficient_t<Series>;
		auto const prefix = atl::detail::series_prefix(series, length);
		return atl::detail::composition_inverse_recurrence<Coefficient>(prefix);
	}

	/**
	 * @brief Reversion by Lagrange inversion through one power projection and a normalized series power, at every length `<= p`.
	 *
	 * This is the library's choice up to the characteristic; the contender asserts the bound. Race pending.
	 * @pre Same contracts as `atl::series_compositional_inverse`; `length <= p`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Series>
	[[nodiscard]] constexpr auto series_compositional_inverse_by_projection_lagrange(Series const &series, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Series>> {
		using Coefficient = atl::series_coefficient_t<Series>;
		auto const prefix = atl::detail::series_prefix(series, length);
		return atl::detail::composition_inverse_lagrange<Engine, Coefficient>(prefix);
	}
}
