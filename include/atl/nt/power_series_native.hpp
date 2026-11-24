#pragma once

/** @file
 * @brief The power_series API for prime moduli up to 32 bits with native double-width coefficient multiplication.
 */

#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/detail/power_series_extensions.hpp>
#include <atl/nt/factorial_table.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/nt/series_calculus.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Reciprocal `1 / f` modulo `x^length`.
	 * @tparam Engine NTT engine used for the products; engines that do not support the modulus fall back to the
	 *         convolution driver's scalar tiers, and constant evaluation uses scalar transforms,
	 *         or schoolbook products where the field has none.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param series Coefficients of `f` in increasing degree order; missing coefficients are zero and coefficients
	 *        at indices `>= length` are not read.
	 * @param length Number of output coefficients.
	 * @return Exactly `length` coefficients, trailing zeros included.
	 * @pre `length <= maximum_series_length`; for positive `length`, `f[0] != 0` (both asserted). Works beyond
	 *      the characteristic.
	 * @complexity O(M(length)) field work and O(length) memory, where M(n) is the cost of a length-n product.
	 * @note Recurrence seed up to a provisional cutoff, then high-half Newton doubling that reuses one transform
	 *       per step; race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_inverse(Range const &series, std::size_t length)
		-> std::vector<series_coefficient_t<Range>> {
		auto const prefix = detail::series_prefix(series, length);
		return detail::series_inverse_newton<Engine, series_coefficient_t<Range>>(prefix, length);
	}

	/**
	 * @brief Quotient `numerator / denominator` modulo `x^length`.
	 * @tparam Engine As for `series_inverse`.
	 * @tparam Numerator Input range of static prime-modulus integers.
	 * @tparam Denominator Input range with the same coefficient type as `Numerator`.
	 * @param numerator Coefficients in increasing degree order; missing coefficients are zero.
	 * @param denominator Coefficients in increasing degree order; missing coefficients are zero.
	 * @param length Number of output coefficients.
	 * @return Exactly `length` coefficients.
	 * @pre `length <= maximum_series_length`; for positive `length`, the denominator's constant is nonzero (asserted).
	 * @complexity O(M(length)) field work and O(length) memory.
	 * @note Above the Newton cutoff the denominator is inverted only to half precision and the last step is
	 *       fused; race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Numerator, prime_mod_int_range Denominator>
		requires std::same_as<series_coefficient_t<Numerator>, series_coefficient_t<Denominator>>
	[[nodiscard]] constexpr auto series_division(Numerator const &numerator, Denominator const &denominator, std::size_t length)
		-> std::vector<series_coefficient_t<Numerator>> {
		auto const numerator_prefix = detail::series_prefix(numerator, length);
		auto const denominator_prefix = detail::series_prefix(denominator, length);
		return detail::series_division_fused<Engine, series_coefficient_t<Numerator>>(numerator_prefix, denominator_prefix, length);
	}

	/**
	 * @brief Formal logarithm `log f` modulo `x^length`, with constant term zero.
	 * @tparam Engine NTT engine used for the products; engines that do not support the modulus fall back to the
	 *         convolution driver's scalar tiers, and constant evaluation uses scalar transforms,
	 *         or schoolbook products where the field has none.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param series Coefficients of `f` in increasing degree order; missing coefficients are zero.
	 * @param length Number of output coefficients.
	 * @return Exactly `length` coefficients.
	 * @pre `length <= maximum_series_length`; for positive `length`, `f[0] == 1` and `length <= p` for the modulus p
	 *      (asserted). The result needs division by every integer below `length`.
	 * @complexity O(M(length)) field work and O(length) memory.
	 * @note Computed as the integral of `f' / f` by fused division; race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_logarithm(Range const &series, std::size_t length)
		-> std::vector<series_coefficient_t<Range>> {
		using Coefficient = series_coefficient_t<Range>;
		atl_assert(length <= maximum_series_length);
		if (length == 0) return {};
		atl_assert(length <= Coefficient::mod());
		auto const prefix = detail::series_prefix(series, length);
		atl_assert(prefix[0] == Coefficient(1));
		auto const derivative = series_derivative(prefix);
		auto const denominator = std::span(prefix).first(length - 1);
		return series_integral(detail::series_division_fused<Engine, Coefficient>(derivative, denominator, length - 1));
	}

	/**
	 * @brief Formal exponential `exp f` modulo `x^length`, with constant term one.
	 * @tparam Engine NTT engine used for the products; engines that do not support the modulus fall back to the
	 *         convolution driver's scalar tiers, and constant evaluation uses scalar transforms,
	 *         or schoolbook products where the field has none.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param series Coefficients of `f` in increasing degree order; missing coefficients are zero, so the empty
	 *        series gives the constant one.
	 * @param length Number of output coefficients.
	 * @return Exactly `length` coefficients.
	 * @pre `length <= maximum_series_length`; for positive `length`, `f[0] == 0` and `length <= p` for the modulus p
	 *      (asserted).
	 * @complexity O(M(length)) field work and O(length) memory.
	 * @note Newton doubling that carries the reciprocal of the exponential along; race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_exponential(Range const &series, std::size_t length)
		-> std::vector<series_coefficient_t<Range>> {
		using Coefficient = series_coefficient_t<Range>;
		atl_assert(length <= maximum_series_length);
		if (length == 0) return {};
		atl_assert(length <= Coefficient::mod());
		auto const prefix = detail::series_prefix(series, length);
		return detail::series_exponential_coupled<Engine, Coefficient>(prefix);
	}

	/**
	 * @brief Integer power `f^exponent` modulo `x^length`.
	 * @tparam Engine NTT engine used for the products; engines that do not support the modulus fall back to the
	 *         convolution driver's scalar tiers, and constant evaluation uses scalar transforms,
	 *         or schoolbook products where the field has none.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param series Coefficients of `f` in increasing degree order; missing coefficients are zero.
	 * @param exponent Full 64-bit exponent. Exponent zero gives the constant one, also for the zero series; the zero
	 *        series to a positive exponent gives zeros.
	 * @param length Number of output coefficients.
	 * @return Exactly `length` coefficients.
	 * @pre `length <= maximum_series_length` (asserted). There is no bound by the characteristic.
	 * @complexity O(M(length) + log(exponent + 1)) field work when the precision left after removing the valuation
	 *             is at most p; otherwise binary powering, O(M(length) log(exponent + 1)). O(length) memory.
	 * @note The valuation shift is checked against the length before it is formed, so any exponent is safe. Past
	 *       the leading power of x, `exp(exponent * log f)` is used where it is valid; race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_power(Range const &series, std::uint64_t exponent, std::size_t length)
		-> std::vector<series_coefficient_t<Range>> {
		using Coefficient = series_coefficient_t<Range>;
		return detail::series_power_normalized(detail::series_prefix(series, length), exponent, [](std::span<Coefficient const> unit, std::uint64_t integer_exponent) {
			if (unit.size() > Coefficient::mod()) return detail::series_power_binary<Engine, Coefficient>(unit, integer_exponent);
			auto logarithm = series_logarithm<Engine>(unit, unit.size());
			for (auto &value : logarithm) value *= Coefficient(integer_exponent);
			return series_exponential<Engine>(logarithm, unit.size());
		});
	}

	/**
	 * @brief Canonical square root `g` with `g^2 = f` modulo `x^length`, or nullopt when there is none.
	 * @tparam Engine NTT engine used for the products; engines that do not support the modulus fall back to the
	 *         convolution driver's scalar tiers, and constant evaluation uses scalar transforms,
	 *         or schoolbook products where the field has none.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param series Coefficients of `f` in increasing degree order; missing coefficients are zero.
	 * @param length Number of output coefficients.
	 * @return Exactly `length` coefficients, or nullopt. An all-zero input gives zeros. In odd characteristic the
	 *         input needs an even valuation `2s` and a leading coefficient that is a square; the root's leading
	 *         coefficient is the smaller scalar root (`atl::square_root_mod`), and the last `s` coefficients, which
	 *         the input does not determine, are zero. In characteristic two the result is `g[i] = f[2i]` and every
	 *         odd-indexed coefficient of the input below `length` must vanish.
	 * @pre `length <= maximum_series_length` (asserted). There is no bound by the characteristic.
	 * @complexity O(M(length)) field work and O(length) memory, plus one scalar square root in odd characteristic.
	 * @note Coupled Newton iteration on the shared driver; race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_square_root(Range const &series, std::size_t length)
		-> std::optional<std::vector<series_coefficient_t<Range>>> {
		using Coefficient = series_coefficient_t<Range>;
		return detail::series_square_root_normalized(detail::series_prefix(series, length), [](std::span<Coefficient const> unit) {
			return detail::series_square_root_coupled<Engine, Coefficient>(unit);
		});
	}

	/**
	 * @brief Polynomial division with remainder: `f = q g + r` with `deg r < deg g`.
	 * @tparam Engine As for `series_inverse`.
	 * @tparam Numerator Input range of static prime-modulus integers.
	 * @tparam Denominator Input range with the same coefficient type as `Numerator`.
	 * @param numerator The whole dividend; trailing zeros are ignored.
	 * @param denominator The whole divisor; trailing zeros are ignored.
	 * @return The unique quotient and remainder, both normalized (the zero polynomial is empty).
	 * @pre The divisor is not the zero polynomial, also for a zero dividend, and the normalized input lengths are at
	 *      most `maximum_series_length` (asserted); trailing zeros may exceed it.
	 * @complexity O(M(n)) field work and O(n) memory for n the dividend length; schoolbook below a provisional cutoff.
	 * @note Reversed-series inverse and a low product; race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Numerator, prime_mod_int_range Denominator>
		requires std::same_as<series_coefficient_t<Numerator>, series_coefficient_t<Denominator>>
	[[nodiscard]] constexpr auto polynomial_division(Numerator const &numerator, Denominator const &denominator)
		-> polynomial_division_result<series_coefficient_t<Numerator>> {
		using Coefficient = series_coefficient_t<Numerator>;
		auto dividend = detail::series_polynomial(numerator);
		auto divisor = detail::series_polynomial(denominator);
		atl_assert(!divisor.empty());
		if (divisor.size() <= detail::series_newton_cutoff || dividend.size() < divisor.size()
			|| dividend.size() - divisor.size() < detail::series_newton_cutoff)
			return detail::series_polynomial_long_division(std::move(dividend), std::move(divisor));
		std::size_t const count = dividend.size() - divisor.size() + 1;
		auto reversed_dividend = dividend, reversed_divisor = divisor;
		std::ranges::reverse(reversed_dividend);
		std::ranges::reverse(reversed_divisor);
		auto const inverse = series_inverse<Engine>(reversed_divisor, count);
		auto quotient = detail::series_product<Engine, Coefficient>(reversed_dividend, inverse, count);
		std::ranges::reverse(quotient);
		auto remainder = detail::series_product<Engine, Coefficient>(quotient, divisor, divisor.size() - 1);
		for (std::size_t i = 0; i < remainder.size(); ++i) remainder[i] = dividend[i] - remainder[i];
		detail::series_normalize_polynomial(quotient);
		detail::series_normalize_polynomial(remainder);
		return {std::move(quotient), std::move(remainder)};
	}

	/**
	 * @brief Reciprocal modulo `x^length` by a triangular solve that visits only the nonzero terms.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param series Coefficients in increasing degree order; missing coefficients are zero.
	 * @param length Number of output coefficients; there is no transform cap.
	 * @return Exactly `length` coefficients, the same as `series_inverse`; no bound by the characteristic.
	 * @pre For positive `length`, `f[0] != 0` (asserted).
	 * @complexity O(length * (s + 1)) field work and O(length + s) memory, for s nonconstant nonzero terms.
	 */
	template<prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_inverse_sparse(Range const &series, std::size_t length)
		-> std::vector<series_coefficient_t<Range>> {
		using Coefficient = series_coefficient_t<Range>;
		auto const prefix = detail::series_copy_prefix(series, length);
		return detail::series_sparse_quotient<Coefficient>(std::vector<Coefficient>{1}, prefix, length);
	}

	/**
	 * @brief Quotient modulo `x^length` by a triangular solve that visits only the denominator's nonzero terms.
	 * @tparam Numerator Input range of static prime-modulus integers.
	 * @tparam Denominator Input range with the same coefficient type as `Numerator`.
	 * @param numerator Coefficients in increasing degree order; missing coefficients are zero.
	 * @param denominator Coefficients in increasing degree order; missing coefficients are zero.
	 * @param length Number of output coefficients; there is no transform cap.
	 * @return Exactly `length` coefficients, the same as `series_division`; no bound by the characteristic.
	 * @pre For positive `length`, the denominator's constant is nonzero (asserted).
	 * @complexity O(length * (s + 1)) field work and O(length + s) memory, for s nonconstant nonzero denominator terms.
	 */
	template<prime_mod_int_range Numerator, prime_mod_int_range Denominator>
		requires std::same_as<series_coefficient_t<Numerator>, series_coefficient_t<Denominator>>
	[[nodiscard]] constexpr auto series_division_sparse(Numerator const &numerator, Denominator const &denominator, std::size_t length)
		-> std::vector<series_coefficient_t<Numerator>> {
		using Coefficient = series_coefficient_t<Numerator>;
		auto const dividend = detail::series_copy_prefix(numerator, length);
		auto const divisor = detail::series_copy_prefix(denominator, length);
		return detail::series_sparse_quotient<Coefficient>(dividend, divisor, length);
	}

	/**
	 * @brief Logarithm modulo `x^length` from `f (log f)' = f'`, visiting only the nonzero terms of `f`.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param series Coefficients in increasing degree order; missing coefficients are zero.
	 * @param length Number of output coefficients; there is no transform cap.
	 * @return Exactly `length` coefficients, the same as `series_logarithm`.
	 * @pre For positive `length`, `f[0] == 1` and `length <= p` (asserted).
	 * @complexity O(length * (s + 1)) field work and O(length + s) memory, for s nonconstant nonzero terms.
	 */
	template<prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_logarithm_sparse(Range const &series, std::size_t length)
		-> std::vector<series_coefficient_t<Range>> {
		using Coefficient = series_coefficient_t<Range>;
		if (length == 0) return {};
		atl_assert(length <= Coefficient::mod());
		auto const prefix = detail::series_copy_prefix(series, length);
		atl_assert(prefix[0] == Coefficient(1));
		auto const terms = detail::series_sparse_terms<Coefficient>(prefix);
		factorial_table<Coefficient> const table(length - 1);
		std::vector<Coefficient> result(length);
		for (std::size_t degree = 1; degree < length; ++degree) {
			Coefficient residual = Coefficient(degree) * prefix[degree];
			for (auto const &term : terms) {
				if (term.degree >= degree) break;
				residual -= term.value * Coefficient(degree - term.degree) * result[degree - term.degree];
			}
			result[degree] = residual * table.inverse_integer(degree);
		}
		return result;
	}

	/**
	 * @brief Exponential modulo `x^length` from `g' = f' g`, visiting only the nonzero terms of `f`.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param series Coefficients in increasing degree order; missing coefficients are zero.
	 * @param length Number of output coefficients; there is no transform cap.
	 * @return Exactly `length` coefficients, the same as `series_exponential`.
	 * @pre For positive `length`, `f[0] == 0` and `length <= p` (asserted).
	 * @complexity O(length * (s + 1)) field work and O(length + s) memory, for s nonconstant nonzero terms.
	 */
	template<prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_exponential_sparse(Range const &series, std::size_t length)
		-> std::vector<series_coefficient_t<Range>> {
		using Coefficient = series_coefficient_t<Range>;
		if (length == 0) return {};
		atl_assert(length <= Coefficient::mod());
		auto const prefix = detail::series_copy_prefix(series, length);
		atl_assert(prefix[0] == Coefficient(0));
		auto const terms = detail::series_sparse_terms<Coefficient>(prefix);
		factorial_table<Coefficient> const table(length - 1);
		std::vector<Coefficient> result(length);
		result[0] = Coefficient(1);
		for (std::size_t degree = 1; degree < length; ++degree) {
			for (auto const &term : terms) {
				if (term.degree > degree) break;
				result[degree] += Coefficient(term.degree) * term.value * result[degree - term.degree];
			}
			result[degree] *= table.inverse_integer(degree);
		}
		return result;
	}

	/**
	 * @brief Integer power modulo `x^length` from the differential equation `f g' = exponent f' g` after removing
	 *        the valuation, visiting only the nonzero terms.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param series Coefficients in increasing degree order; missing coefficients are zero.
	 * @param length Number of output coefficients; there is no transform cap.
	 * @param exponent Full 64-bit exponent, with the semantics of `series_power`.
	 * @return Exactly `length` coefficients, the same as `series_power`.
	 * @pre Write `f = x^v u` with `u(0) != 0`. When `length > 0`, `exponent > 0`, `f` is nonzero below `x^length`
	 *      and `exponent * v < length`, the recurrence runs over the `length - exponent * v` coefficients of the
	 *      unit power and divides by every integer below that count, so `length - exponent * v <= p` for the
	 *      modulus p (asserted). Every other input (length zero, exponent zero, the zero series, or a shift
	 *      `exponent * v >= length` that leaves only zeros) has no bound by the characteristic.
	 * @complexity O(length * (s + 1) + log(exponent + 1)) field work and O(length + s) memory.
	 */
	template<prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_power_sparse(Range const &series, std::uint64_t exponent, std::size_t length)
		-> std::vector<series_coefficient_t<Range>> {
		using Coefficient = series_coefficient_t<Range>;
		return detail::series_power_normalized(detail::series_copy_prefix(series, length), exponent, [](std::span<Coefficient const> unit, std::uint64_t integer_exponent) {
			atl_assert(unit.size() <= Coefficient::mod());
			return detail::series_power_ode<Coefficient>(unit, Coefficient(integer_exponent));
		});
	}

	/**
	 * @brief Canonical square root modulo `x^length` visiting only the nonzero terms.
	 * @tparam Range Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param series Coefficients in increasing degree order; missing coefficients are zero.
	 * @param length Number of output coefficients; there is no transform cap.
	 * @return The same result as `series_square_root`: the same scalar root, zero tail and nullopt cases.
	 * @pre Write `f = x^v u` with `u(0) != 0`. In odd characteristic, when `f` is nonzero below `x^length`, `v` is
	 *      even and `u(0)` is a square, the root is `x^(v/2)` times the root of `u`, and the recurrence runs over
	 *      the `length - v` coefficients of `u` and divides by every integer below that count, so
	 *      `length - v <= p` for the modulus p (asserted). The zero series and the nullopt cases have no bound;
	 *      characteristic two has no bound.
	 * @complexity O(length * (s + 1)) field work and O(length + s) memory, plus one scalar square root.
	 */
	template<prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_square_root_sparse(Range const &series, std::size_t length)
		-> std::optional<std::vector<series_coefficient_t<Range>>> {
		using Coefficient = series_coefficient_t<Range>;
		return detail::series_square_root_normalized(detail::series_copy_prefix(series, length), [](std::span<Coefficient const> unit) {
			atl_assert(unit.size() <= Coefficient::mod());
			return detail::series_power_ode<Coefficient>(unit, Coefficient(1) / Coefficient(2));
		});
	}
}
