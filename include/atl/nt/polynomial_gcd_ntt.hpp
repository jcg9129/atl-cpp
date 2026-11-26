#pragma once

/** @file
 * @brief Reduced polynomial_gcd definitions for native fields with direct NTT order at least 2^23.
 */

#include <atl/nt/detail/ntt_engine_base.hpp>
#include <atl/nt/detail/polynomial_gcd_native_base.hpp>
#include <atl/nt/detail/power_series_extensions_native_base.hpp>
#include <atl/nt/detail/power_series_native_base.hpp>
#include <atl/nt/detail/prime_mod_int_base.hpp>
#include <atl/nt/power_series_ntt.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Monic gcd of two polynomials and the cofactors of their Bezout identity.
	 *
	 * `first * first_cofactor + second * second_cofactor == gcd`. Every polynomial is normalized: trailing zeros are
	 * stripped and the zero polynomial is the empty vector.
	 * @tparam Coefficient Static prime-modulus integer type.
	 */
	template<static_prime_mod_int Coefficient>
	struct polynomial_extended_gcd_result {
		/** @brief The monic gcd; empty when both inputs are zero. */
		std::vector<Coefficient> gcd;
		/** @brief Multiplier of the first input in the Bezout identity. */
		std::vector<Coefficient> first_cofactor;
		/** @brief Multiplier of the second input in the Bezout identity. */
		std::vector<Coefficient> second_cofactor;

		/**
		 * @brief Structural comparison: the gcds and both cofactors are equal member by member.
		 * @note This is not equivalence of Bezout identities: two valid results for the same inputs can have
		 *       different cofactors and compare unequal.
		 */
		friend bool operator==(polynomial_extended_gcd_result const &, polynomial_extended_gcd_result const &) = default;
	};

	/**
	 * @brief A rational function `numerator / denominator` in lowest terms with denominator constant term one.
	 * @tparam Coefficient Static prime-modulus integer type.
	 */
	template<static_prime_mod_int Coefficient>
	struct rational_function {
		/** @brief Normalized numerator. */
		std::vector<Coefficient> numerator;
		/** @brief Normalized denominator; its constant coefficient is one. */
		std::vector<Coefficient> denominator;
	};

	namespace detail {
		/** @brief Monic gcd by the selected Euclidean contender (the last nonzero remainder, scaled). */
		template<ntt_engine Engine, polynomial_gcd_algorithm Algorithm, prime_mod_int_range First, prime_mod_int_range Second>
			requires std::same_as<series_coefficient_t<First>, series_coefficient_t<Second>>
		[[nodiscard]] constexpr auto polynomial_gcd_dispatch(First const &first, Second const &second, std::size_t cutoff) -> std::vector<series_coefficient_t<First>> {
			using Coefficient = series_coefficient_t<First>;
			auto state = gcd_reduce<Engine, Algorithm, 0>(series_polynomial(first), series_polynomial(second), 0, cutoff);
			auto result = std::move(state.remainders.earlier);
			if (!result.empty()) gcd_scale(result, Coefficient(1) / result.back());
			return result;
		}

		/** @brief Monic gcd with both cofactors by the selected Euclidean contender; the identity is scaled with the gcd. */
		template<ntt_engine Engine, polynomial_gcd_algorithm Algorithm, prime_mod_int_range First, prime_mod_int_range Second>
			requires std::same_as<series_coefficient_t<First>, series_coefficient_t<Second>>
		[[nodiscard]] constexpr auto polynomial_extended_gcd_dispatch(First const &first, Second const &second, std::size_t cutoff) -> polynomial_extended_gcd_result<series_coefficient_t<First>> {
			using Coefficient = series_coefficient_t<First>;
			auto state = gcd_reduce<Engine, Algorithm, 2>(series_polynomial(first), series_polynomial(second), 0, cutoff);
			if (state.remainders.earlier.empty()) return {};
			Coefficient const scale = Coefficient(1) / state.remainders.earlier.back();
			gcd_scale(state.remainders.earlier, scale);
			gcd_scale(state.cofactors[0], scale);
			gcd_scale(state.cofactors[1], scale);
			return {std::move(state.remainders.earlier), std::move(state.cofactors[0]), std::move(state.cofactors[1])};
		}

		/** @brief Pade approximant by extended Euclid on `x^precision` and the series, stopped at the numerator degree bound.
		 * @pre `numerator_degree + denominator_degree + 1 <= maximum_series_length`, checked without overflow (asserted).
		 */
		template<ntt_engine Engine, polynomial_gcd_algorithm Algorithm, prime_mod_int_range Range>
		[[nodiscard]] constexpr auto pade_dispatch(Range const &series, std::size_t numerator_degree, std::size_t denominator_degree) -> std::optional<rational_function<series_coefficient_t<Range>>> {
			using Coefficient = series_coefficient_t<Range>;
			atl_assert(numerator_degree < maximum_series_length && denominator_degree < maximum_series_length);
			atl_assert(numerator_degree < maximum_series_length - denominator_degree);
			std::size_t const precision = numerator_degree + denominator_degree + 1;
			auto input = series_prefix(series, precision);
			series_normalize_polynomial(input);
			std::vector<Coefficient> power(precision + 1);
			power.back() = Coefficient(1);
			auto state = gcd_reduce<Engine, Algorithm, 1>(std::move(power), std::move(input), numerator_degree + 1, polynomial_half_gcd_cutoff);
			auto numerator = std::move(state.remainders.later);
			auto denominator = std::move(state.cofactors[1]);
			if (denominator.empty() || denominator[0] == Coefficient(0) || denominator.size() > denominator_degree + 1) return std::nullopt;
			Coefficient const scale = Coefficient(1) / denominator[0];
			gcd_scale(numerator, scale);
			gcd_scale(denominator, scale);
			return rational_function<Coefficient>{std::move(numerator), std::move(denominator)};
		}
	}

	/**
	 * @brief Monic gcd of two polynomials over a prime field.
	 * @tparam Engine NTT engine used for the products; engines that do not support the modulus fall back to the
	 *         convolution driver's tiers, and constant evaluation uses scalar transforms,
	 *         or schoolbook products where the field has none.
	 * @tparam First Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @tparam Second Input range with the same coefficient type as `First`.
	 * @param first Coefficients in increasing degree order; trailing zeros are ignored and the input is not modified.
	 * @param second As `first`.
	 * @return The monic gcd, empty for two zero inputs; for one zero input, the other made monic.
	 * @pre The normalized lengths are at most `maximum_series_length` (asserted); trailing zeros may exceed it.
	 * @complexity O(M(n) log(n + 1)) field work and O(n) memory for `n` the larger length. Race pending.
	 * @note Recursive half-gcd with a provisional Euclidean leaf size of 64.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<series_coefficient_t<First>, series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_gcd(First const &first, Second const &second) -> std::vector<series_coefficient_t<First>> {
		return detail::polynomial_gcd_dispatch<Engine, detail::polynomial_gcd_algorithm::half_gcd>(first, second, detail::polynomial_half_gcd_cutoff);
	}

	/**
	 * @brief Monic gcd and minimal Bezout cofactors of two polynomials over a prime field.
	 *
	 * With `g` the gcd and `d` the degree of `g`, the cofactor of each input has degree strictly below the degree of the
	 * other input minus `d` whenever that difference is positive. When it is zero, one input divides the other and
	 * only the divisor has a nonzero cofactor, a constant; for equal degrees that is the second input. `(f, 0)` gives
	 * `(f / lc(f), {1 / lc(f)}, {})`, `(0, f)` the same with the cofactors interchanged and `(0, 0)` three empty
	 * polynomials.
	 * @tparam Engine As for `polynomial_gcd`.
	 * @tparam First Input range of static prime-modulus integers.
	 * @tparam Second Input range with the same coefficient type as `First`.
	 * @param first Coefficients in increasing degree order; trailing zeros are ignored.
	 * @param second As `first`.
	 * @return A `polynomial_extended_gcd_result`; every polynomial is normalized.
	 * @pre The normalized lengths are at most `maximum_series_length` (asserted).
	 * @complexity O(M(n) log(n + 1)) field work and O(n) memory for `n` the larger length. Race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range First, prime_mod_int_range Second>
		requires std::same_as<series_coefficient_t<First>, series_coefficient_t<Second>>
	[[nodiscard]] constexpr auto polynomial_extended_gcd(First const &first, Second const &second) -> polynomial_extended_gcd_result<series_coefficient_t<First>> {
		return detail::polynomial_extended_gcd_dispatch<Engine, detail::polynomial_gcd_algorithm::half_gcd>(first, second, detail::polynomial_half_gcd_cutoff);
	}

	/**
	 * @brief Inverse of a polynomial modulo another, or `nullopt` when their gcd is not one.
	 *
	 * The inverse is reduced (degree below the modulus degree). A value that is a multiple of the modulus, including
	 * zero, has no inverse.
	 * @tparam Engine As for `polynomial_gcd`.
	 * @tparam Value Input range of static prime-modulus integers.
	 * @tparam Modulus Input range with the same coefficient type as `Value`.
	 * @param value Coefficients of the polynomial to invert; it is first reduced modulo `modulus`.
	 * @param modulus Coefficients of the modulus; trailing zeros are ignored.
	 * @return The normalized inverse, or `nullopt`.
	 * @pre The normalized lengths are at most `maximum_series_length` and the modulus has degree at least one (both asserted).
	 * @complexity O(M(n) log(n + 1)) field work and O(n) memory for `n` the larger length; only the cofactor of the value
	 *             is accumulated. Race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Value, prime_mod_int_range Modulus>
		requires std::same_as<series_coefficient_t<Value>, series_coefficient_t<Modulus>>
	[[nodiscard]] constexpr auto polynomial_inverse_mod(Value const &value, Modulus const &modulus) -> std::optional<std::vector<series_coefficient_t<Value>>> {
		using Coefficient = series_coefficient_t<Value>;
		auto divisor = detail::series_polynomial(modulus);
		atl_assert(divisor.size() >= 2);
		auto residue = polynomial_division<Engine>(value, divisor).remainder;
		auto state = detail::gcd_reduce<Engine, detail::polynomial_gcd_algorithm::half_gcd, 1>(std::move(divisor), std::move(residue), 0, detail::polynomial_half_gcd_cutoff);
		if (state.remainders.earlier.size() != 1) return std::nullopt;
		auto result = std::move(state.cofactors[0]);
		detail::gcd_scale(result, Coefficient(1) / state.remainders.earlier[0]);
		return result;
	}

	/**
	 * @brief The Pade approximant `P / Q` of a power series: lowest terms, `deg P <= numerator_degree`,
	 *        `deg Q <= denominator_degree`, `Q[0] = 1` and `P == Q * series` modulo `x^(numerator_degree + denominator_degree + 1)`.
	 * @tparam Engine As for `polynomial_gcd`.
	 * @tparam Range Input range of static prime-modulus integers.
	 * @param series Series coefficients in increasing degree order; only the first
	 *        `numerator_degree + denominator_degree + 1` are read and absent ones are zero.
	 * @param numerator_degree Degree bound of the numerator.
	 * @param denominator_degree Degree bound of the denominator.
	 * @return A `rational_function`, or `nullopt` when no such denominator exists.
	 * @pre `numerator_degree + denominator_degree + 1 <= maximum_series_length` (asserted, without overflow).
	 * @complexity O(M(n) log(n + 1)) field work and O(n) memory for `n = numerator_degree + denominator_degree + 1`. Race pending.
	 * @note Half-gcd stopped at the numerator degree bound. A nonzero denominator constant also certifies lowest terms,
	 *       since a common factor would divide `x^n`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto pade_approximant(Range const &series, std::size_t numerator_degree, std::size_t denominator_degree) -> std::optional<rational_function<series_coefficient_t<Range>>> {
		return detail::pade_dispatch<Engine, detail::polynomial_gcd_algorithm::half_gcd>(series, numerator_degree, denominator_degree);
	}
}
