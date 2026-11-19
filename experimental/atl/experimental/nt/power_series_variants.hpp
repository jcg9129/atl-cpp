#pragma once

#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/detail/power_series_extensions.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/power_series.hpp>
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

/**
 * @file
 * @brief Contenders for the formal power series functions of `atl/nt/power_series.hpp`.
 *
 * Every function is named `<function>_by_<method>` after the library function it races and has the same template
 * parameters, argument order, result and preconditions as that function (differences are stated). The races are
 * pending: no contender has been measured, and the library functions are provisional choices.
 */

namespace atl::experimental {
	/**
	 * @brief Contender for `atl::series_inverse`: the defining quadratic recurrence, with no Newton step. Race pending.
	 * @complexity O(length^2) field operations.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_inverse_by_recurrence(Range const &series, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Range>> {
		auto const prefix = atl::detail::series_prefix(series, length);
		return atl::detail::series_inverse_recurrence<atl::series_coefficient_t<Range>>(prefix, length);
	}

	/**
	 * @brief Contender for `atl::series_inverse`: Newton `g (2 - f g)` with two ordinary products per step instead
	 *        of the high-half step that reuses one transform. Race pending.
	 * @complexity O(M(length)) field operations.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_inverse_by_plain_newton(Range const &series, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Range>> {
		using Coefficient = atl::series_coefficient_t<Range>;
		auto const prefix = atl::detail::series_prefix(series, length);
		auto inverse = atl::detail::series_inverse_recurrence<Coefficient>(prefix, (std::min)(length, atl::detail::series_newton_cutoff));
		while (inverse.size() < length) {
			atl_assert(inverse.size() <= maximum_series_length / 2);
			std::size_t const next = (std::min)(length, 2 * inverse.size());
			auto error = atl::detail::series_product<Engine, Coefficient>(std::span(prefix).first(next), inverse, next);
			for (auto &value : error) value = -value;
			error[0] += Coefficient(2);
			inverse = atl::detail::series_product<Engine, Coefficient>(inverse, error, next);
		}
		return inverse;
	}

	/**
	 * @brief Contender for `atl::series_inverse`: the same high-half Newton step, with the doubling loop supplied
	 *        by the shared driver `atl::detail::series_newton_lift` instead of a hand-written loop. Race pending.
	 * @complexity O(M(length)) field operations.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_inverse_by_newton_driver(Range const &series, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Range>> {
		using Coefficient = atl::series_coefficient_t<Range>;
		auto const prefix = atl::detail::series_prefix(series, length);
		auto inverse = atl::detail::series_inverse_recurrence<Coefficient>(prefix, (std::min)(length, atl::detail::series_newton_cutoff));
		atl::detail::series_newton_lift(inverse, inverse.size(), length, [&](auto &current, std::size_t, std::size_t next) {
			current = atl::detail::series_inverse_extend<Engine, Coefficient>(prefix, std::move(current), next);
		});
		return inverse;
	}

	/**
	 * @brief Contender for `atl::series_division`: the full inverse followed by one truncated product, instead of
	 *        the fused last step. Race pending.
	 * @complexity O(M(length)) field operations.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Numerator, prime_mod_int_range Denominator>
		requires std::same_as<atl::series_coefficient_t<Numerator>, atl::series_coefficient_t<Denominator>>
	[[nodiscard]] constexpr auto series_division_by_inverse_product(Numerator const &numerator, Denominator const &denominator, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Numerator>> {
		using Coefficient = atl::series_coefficient_t<Numerator>;
		auto const prefix = atl::detail::series_prefix(numerator, length);
		auto const inverse = atl::series_inverse<Engine>(denominator, length);
		return atl::detail::series_product<Engine, Coefficient>(prefix, inverse, length);
	}

	/**
	 * @brief Contender for `atl::series_logarithm`: derivative, full inverse, product and integral, instead of the
	 *        fused division of the derivative by the series. Race pending.
	 * @complexity O(M(length)) field operations.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_logarithm_by_inverse_product(Range const &series, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Range>> {
		using Coefficient = atl::series_coefficient_t<Range>;
		atl_assert(length <= maximum_series_length && length <= Coefficient::mod());
		auto const prefix = atl::detail::series_prefix(series, length);
		return atl::detail::series_logarithm_product<Engine, Coefficient>(prefix);
	}

	/**
	 * @brief Contender for `atl::series_exponential`: the defining quadratic recurrence, with no Newton step. Race pending.
	 * @complexity O(length^2) field operations.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_exponential_by_recurrence(Range const &series, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Range>> {
		using Coefficient = atl::series_coefficient_t<Range>;
		atl_assert(length <= maximum_series_length && length <= Coefficient::mod());
		auto const prefix = atl::detail::series_prefix(series, length);
		return atl::detail::series_exponential_recurrence<Coefficient>(prefix, length);
	}

	/**
	 * @brief Contender for `atl::series_exponential`: Newton `g (1 + f - log g)` that rebuilds the reciprocal for
	 *        `log g` at every step, instead of carrying it along. Race pending.
	 * @complexity O(M(length)) field operations.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_exponential_by_plain_newton(Range const &series, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Range>> {
		using Coefficient = atl::series_coefficient_t<Range>;
		atl_assert(length <= maximum_series_length && length <= Coefficient::mod());
		auto const prefix = atl::detail::series_prefix(series, length);
		auto exponential = atl::detail::series_exponential_recurrence<Coefficient>(prefix, (std::min)(length, atl::detail::series_newton_cutoff));
		while (exponential.size() < length) {
			atl_assert(exponential.size() <= maximum_series_length / 2);
			std::size_t const next = (std::min)(length, 2 * exponential.size());
			auto padded = exponential;
			padded.resize(next);
			auto correction = atl::detail::series_logarithm_product<Engine, Coefficient>(padded);
			for (std::size_t i = 0; i < next; ++i) correction[i] = prefix[i] - correction[i];
			correction[0] += Coefficient(1);
			exponential = atl::detail::series_product<Engine, Coefficient>(exponential, correction, next);
		}
		return exponential;
	}

	/**
	 * @brief Contender for `atl::series_exponential`: the same coupled Newton step, with the doubling loop supplied
	 *        by the shared driver `atl::detail::series_newton_lift` instead of a hand-written loop. Race pending.
	 * @complexity O(M(length)) field operations.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_exponential_by_newton_driver(Range const &series, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Range>> {
		using Coefficient = atl::series_coefficient_t<Range>;
		atl_assert(length <= maximum_series_length && length <= Coefficient::mod());
		auto const prefix = atl::detail::series_prefix(series, length);
		auto exponential = atl::detail::series_exponential_recurrence<Coefficient>(prefix, (std::min)(length, atl::detail::series_newton_cutoff));
		if (exponential.size() == length) return exponential;
		auto inverse = atl::detail::series_inverse_recurrence<Coefficient>(exponential, exponential.size());
		std::size_t const seed = exponential.size();

		struct state_type {
			std::vector<Coefficient> value;
			std::vector<Coefficient> reciprocal;
		} state{std::move(exponential), std::move(inverse)};

		atl::detail::series_newton_lift(state, seed, length, [&](auto &current, std::size_t middle, std::size_t next) {
			std::size_t const width = next - middle;
			current.reciprocal = atl::detail::series_inverse_extend<Engine, Coefficient>(current.value, std::move(current.reciprocal), next);
			auto const derivative = series_derivative(current.value);
			auto const logarithm = series_integral(atl::detail::series_product<Engine, Coefficient>(derivative, current.reciprocal, next - 1));
			std::vector<Coefficient> delta(width);
			for (std::size_t i = 0; i < width; ++i) delta[i] = prefix[middle + i] - logarithm[middle + i];
			auto const correction = atl::detail::series_product<Engine, Coefficient>(current.value, delta, width);
			current.value.resize(next);
			std::ranges::copy(correction, std::span(current.value).subspan(middle).begin());
			if (next < length) {
				auto const inverse_correction = atl::detail::series_product<Engine, Coefficient>(std::span(current.reciprocal).first(middle), delta, width);
				for (std::size_t i = 0; i < width; ++i) current.reciprocal[middle + i] -= inverse_correction[i];
			}
		});
		return std::move(state.value);
	}

	/**
	 * @brief Contender for `atl::series_power`: binary powering with truncation after every product, with the same
	 *        valuation and scalar handling, instead of `exp(exponent * log f)`. Race pending.
	 * @complexity O(M(length) log(exponent + 1)) field operations.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_power_by_binary_powering(Range const &series, std::uint64_t exponent, std::size_t length)
		-> std::vector<atl::series_coefficient_t<Range>> {
		using Coefficient = atl::series_coefficient_t<Range>;
		return atl::detail::series_power_normalized(atl::detail::series_prefix(series, length), exponent, [](std::span<Coefficient const> unit, std::uint64_t integer_exponent) {
			return atl::detail::series_power_binary<Engine, Coefficient>(unit, integer_exponent);
		});
	}

	/**
	 * @brief Contender for `atl::series_square_root`: plain Newton `y <- (y + f / y) / 2` that divides by the public
	 *        `atl::series_division` at every step, instead of carrying the reciprocal along. Race pending.
	 * @complexity O(M(length)) field operations.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_square_root_by_plain_newton(Range const &series, std::size_t length)
		-> std::optional<std::vector<atl::series_coefficient_t<Range>>> {
		using Coefficient = atl::series_coefficient_t<Range>;
		return atl::detail::series_square_root_normalized(atl::detail::series_prefix(series, length), [](std::span<Coefficient const> unit) {
			std::vector<Coefficient> root{Coefficient(1)};
			Coefficient const half = Coefficient(1) / Coefficient(2);
			while (root.size() < unit.size()) {
				atl_assert(root.size() <= maximum_series_length / 2);
				std::size_t const middle = root.size(), next = (std::min)(unit.size(), 2 * middle);
				auto const quotient = atl::series_division<Engine>(unit.first(next), root, next);
				root.resize(next);
				for (std::size_t i = middle; i < next; ++i) root[i] = quotient[i] * half;
			}
			return root;
		});
	}

	/**
	 * @brief Contender for `atl::series_square_root`: `exp(log(f) / 2)` of the unit part, as a reference.
	 *        Race pending.
	 * @pre As `atl::series_square_root`, and in odd characteristic `length <= p` (asserted).
	 * @complexity O(M(length)) field operations.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Range>
	[[nodiscard]] constexpr auto series_square_root_by_logarithm(Range const &series, std::size_t length)
		-> std::optional<std::vector<atl::series_coefficient_t<Range>>> {
		using Coefficient = atl::series_coefficient_t<Range>;
		if constexpr (Coefficient::mod() != 2) atl_assert(length <= Coefficient::mod());
		return atl::detail::series_square_root_normalized(atl::detail::series_prefix(series, length), [](std::span<Coefficient const> unit) {
			auto logarithm = atl::series_logarithm<Engine>(unit, unit.size());
			Coefficient const half = Coefficient(1) / Coefficient(2);
			for (auto &value : logarithm) value *= half;
			return atl::series_exponential<Engine>(logarithm, unit.size());
		});
	}

	/**
	 * @brief Contender for `atl::polynomial_division`: schoolbook long division at every size, instead of the
	 *        reversed-series inverse and a low product. Race pending.
	 * @complexity O(n m) field operations for dividend length n and divisor length m.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Numerator, prime_mod_int_range Denominator>
		requires std::same_as<atl::series_coefficient_t<Numerator>, atl::series_coefficient_t<Denominator>>
	[[nodiscard]] constexpr auto polynomial_division_by_schoolbook(Numerator const &numerator, Denominator const &denominator)
		-> polynomial_division_result<atl::series_coefficient_t<Numerator>> {
		return atl::detail::series_polynomial_long_division(atl::detail::series_polynomial(numerator), atl::detail::series_polynomial(denominator));
	}
}
