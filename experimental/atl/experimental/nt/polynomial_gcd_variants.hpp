#pragma once

#include <atl/nt/detail/polynomial_gcd.hpp>
#include <atl/nt/detail/polynomial_points.hpp>
#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/detail/power_series_extensions.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/polynomial_gcd.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Contenders for the polynomial gcd family of `<atl/nt/polynomial_gcd.hpp>` and a fast `berlekamp_massey`.
 *
 * Each one has the template parameters, argument order, result and preconditions of the library function it races,
 * unless its own comment says otherwise. All races are pending: no measurement decides any of them yet
 * (benchmarks/nt/polynomial_gcd.cpp); the library defaults (half-gcd, Euclidean leaf size 64) are provisional.
 */

namespace atl::experimental {
	/** @brief `polynomial_gcd` with the spectrum of each transition-matrix entry cached and the two products of an output entry summed before its inverse transform; race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<atl::series_coefficient_t<First>, atl::series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_gcd_by_fused_matrix_products(First const &first, Second const &second) -> std::vector<atl::series_coefficient_t<First>> {
		return atl::detail::polynomial_gcd_dispatch<Engine, atl::detail::polynomial_gcd_algorithm::transformed_half_gcd>(first, second, atl::detail::polynomial_half_gcd_cutoff);
	}

	/** @brief `polynomial_extended_gcd` with fused spectral matrix products, as `polynomial_gcd_by_fused_matrix_products`; race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<atl::series_coefficient_t<First>, atl::series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_extended_gcd_by_fused_matrix_products(First const &first, Second const &second) -> polynomial_extended_gcd_result<atl::series_coefficient_t<First>> {
		return atl::detail::polynomial_extended_gcd_dispatch<Engine, atl::detail::polynomial_gcd_algorithm::transformed_half_gcd>(first, second, atl::detail::polynomial_half_gcd_cutoff);
	}

	/** @brief `polynomial_gcd` by the plain Euclidean algorithm with fast polynomial division, without half-gcd; quadratic in the worst case; race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<atl::series_coefficient_t<First>, atl::series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_gcd_by_fast_division_euclid(First const &first, Second const &second) -> std::vector<atl::series_coefficient_t<First>> {
		return atl::detail::polynomial_gcd_dispatch<Engine, atl::detail::polynomial_gcd_algorithm::euclid>(first, second, atl::detail::polynomial_half_gcd_cutoff);
	}

	/** @brief `polynomial_extended_gcd` by Euclid with fast division, as `polynomial_gcd_by_fast_division_euclid`; race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<atl::series_coefficient_t<First>, atl::series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_extended_gcd_by_fast_division_euclid(First const &first, Second const &second) -> polynomial_extended_gcd_result<atl::series_coefficient_t<First>> {
		return atl::detail::polynomial_extended_gcd_dispatch<Engine, atl::detail::polynomial_gcd_algorithm::euclid>(first, second, atl::detail::polynomial_half_gcd_cutoff);
	}

	/** @brief `polynomial_gcd` by the plain Euclidean algorithm with schoolbook division; quadratic; race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<atl::series_coefficient_t<First>, atl::series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_gcd_by_schoolbook_euclid(First const &first, Second const &second) -> std::vector<atl::series_coefficient_t<First>> {
		return atl::detail::polynomial_gcd_dispatch<Engine, atl::detail::polynomial_gcd_algorithm::schoolbook>(first, second, atl::detail::polynomial_half_gcd_cutoff);
	}

	/** @brief `polynomial_extended_gcd` by Euclid with schoolbook division, as `polynomial_gcd_by_schoolbook_euclid`; race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<atl::series_coefficient_t<First>, atl::series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_extended_gcd_by_schoolbook_euclid(First const &first, Second const &second) -> polynomial_extended_gcd_result<atl::series_coefficient_t<First>> {
		return atl::detail::polynomial_extended_gcd_dispatch<Engine, atl::detail::polynomial_gcd_algorithm::schoolbook>(first, second, atl::detail::polynomial_half_gcd_cutoff);
	}

	/**
	 * @brief `polynomial_gcd` by half-gcd with ordinary (unfused) matrix products and the Euclidean leaf size as an
	 *        argument; race pending.
	 * @param cutoff Operands up to this length are reduced by Euclidean steps.
	 * @pre `1 <= cutoff <= maximum_series_length` (asserted).
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<atl::series_coefficient_t<First>, atl::series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_gcd_by_leaf_cutoff_plain_products(First const &first, Second const &second, std::size_t cutoff) -> std::vector<atl::series_coefficient_t<First>> {
		return atl::detail::polynomial_gcd_dispatch<Engine, atl::detail::polynomial_gcd_algorithm::half_gcd>(first, second, cutoff);
	}

	/**
	 * @brief `polynomial_gcd` by half-gcd with the fused spectral matrix products of
	 *        `polynomial_gcd_by_fused_matrix_products` and the Euclidean leaf size as an argument; race pending.
	 * @param cutoff Operands up to this length are reduced by Euclidean steps.
	 * @pre `1 <= cutoff <= maximum_series_length` (asserted).
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<atl::series_coefficient_t<First>, atl::series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_gcd_by_leaf_cutoff_fused_products(First const &first, Second const &second, std::size_t cutoff) -> std::vector<atl::series_coefficient_t<First>> {
		return atl::detail::polynomial_gcd_dispatch<Engine, atl::detail::polynomial_gcd_algorithm::transformed_half_gcd>(first, second, cutoff);
	}

	/**
	 * @brief `polynomial_extended_gcd` as `polynomial_gcd_by_leaf_cutoff_plain_products`; race pending.
	 * @pre `1 <= cutoff <= maximum_series_length` (asserted).
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<atl::series_coefficient_t<First>, atl::series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_extended_gcd_by_leaf_cutoff_plain_products(First const &first, Second const &second, std::size_t cutoff) -> polynomial_extended_gcd_result<atl::series_coefficient_t<First>> {
		return atl::detail::polynomial_extended_gcd_dispatch<Engine, atl::detail::polynomial_gcd_algorithm::half_gcd>(first, second, cutoff);
	}

	/**
	 * @brief `polynomial_extended_gcd` as `polynomial_gcd_by_leaf_cutoff_fused_products`; race pending.
	 * @pre `1 <= cutoff <= maximum_series_length` (asserted).
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<atl::series_coefficient_t<First>, atl::series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_extended_gcd_by_leaf_cutoff_fused_products(First const &first, Second const &second, std::size_t cutoff) -> polynomial_extended_gcd_result<atl::series_coefficient_t<First>> {
		return atl::detail::polynomial_extended_gcd_dispatch<Engine, atl::detail::polynomial_gcd_algorithm::transformed_half_gcd>(first, second, cutoff);
	}

	/** @brief `pade_approximant` by extended Euclid with schoolbook division, quadratic; race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto pade_approximant_by_quadratic_euclid(Range const &series, std::size_t numerator_degree, std::size_t denominator_degree) -> std::optional<rational_function<atl::series_coefficient_t<Range>>> {
		return atl::detail::pade_dispatch<Engine, atl::detail::polynomial_gcd_algorithm::schoolbook>(series, numerator_degree, denominator_degree);
	}

	/**
	 * @brief The shortest linear recurrence of a sequence from half-gcd on the reversed sequence, a fast counterpart of `berlekamp_massey`; race pending.
	 *
	 * As with `berlekamp_massey`, a prefix too short to determine the recurrence can select a different shortest
	 * recurrence.
	 * @tparam Engine As for `polynomial_gcd`.
	 * @tparam Sequence Input range of static prime-modulus integers.
	 * @param sequence The terms in order.
	 * @return The coefficients `c` with `s[i] = sum c[j] s[i - 1 - j]`; empty for an all-zero sequence.
	 * @pre The length is at most `maximum_series_length` (asserted).
	 * @complexity O(M(n) log(n + 1)) field work and O(n) memory for `n` terms.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Sequence>
	[[nodiscard]] constexpr auto berlekamp_massey_by_half_gcd(Sequence const &sequence) -> std::vector<atl::series_coefficient_t<Sequence>> {
		using Coefficient = atl::series_coefficient_t<Sequence>;
		auto reversed = atl::detail::points_copy(sequence);
		std::size_t const length = reversed.size();
		atl_assert(length <= maximum_series_length);
		if (length == 0) return {};
		std::ranges::reverse(reversed);
		atl::detail::series_normalize_polynomial(reversed);
		if (reversed.empty()) return {};
		if (reversed.size() == 1) return std::vector<Coefficient>(length);
		std::vector<Coefficient> power(length + 1);
		power.back() = Coefficient(1);
		auto state = atl::detail::gcd_reduce<Engine, atl::detail::polynomial_gcd_algorithm::half_gcd, 1>(std::move(power), std::move(reversed), (length + 1) / 2, atl::detail::polynomial_half_gcd_cutoff);
		// A valid characteristic polynomial needs its remainder degree below its own degree, even after an abnormal drop.
		while (!state.remainders.later.empty() && state.remainders.later.size() >= state.cofactors[1].size()) {
			auto division = atl::detail::gcd_divide<Engine, atl::detail::polynomial_gcd_algorithm::half_gcd>(state.remainders.earlier, state.remainders.later);
			auto next = std::move(state.cofactors[0]);
			atl::detail::gcd_accumulate<Coefficient>(next, atl::detail::gcd_product<Engine, Coefficient>(division.quotient, state.cofactors[1]), true);
			state.cofactors[0] = std::move(state.cofactors[1]);
			state.cofactors[1] = std::move(next);
			state.remainders.earlier = std::move(state.remainders.later);
			state.remainders.later = std::move(division.remainder);
		}
		auto const &characteristic = state.cofactors[1];
		atl_assert(!characteristic.empty());
		std::vector<Coefficient> result(characteristic.size() - 1);
		Coefficient const scale = -Coefficient(1) / characteristic.back();
		for (std::size_t i = 0; i < result.size(); ++i) result[i] = characteristic[result.size() - 1 - i] * scale;
		return result;
	}
}
