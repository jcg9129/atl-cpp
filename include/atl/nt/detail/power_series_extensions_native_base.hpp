#pragma once

#include <atl/nt/detail/factorial_table_base.hpp>
#include <atl/nt/detail/ntt_engine_base.hpp>
#include <atl/nt/detail/power_series_native_base.hpp>
#include <atl/nt/detail/prime_mod_int_base.hpp>
#include <atl/nt/detail/square_root_mod_static.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Quotient and remainder of a polynomial division, both normalized: the zero polynomial is an empty
	 *        vector and nonzero polynomials have a nonzero last coefficient.
	 * @tparam Coefficient Static prime-modulus integer type.
	 */
	template<static_prime_mod_int Coefficient>
	struct polynomial_division_result {
		/** @brief Polynomial quotient, in increasing degree order. */
		std::vector<Coefficient> quotient;
		/** @brief Remainder of smaller degree than the denominator. */
		std::vector<Coefficient> remainder;
	};

	namespace detail {
		/** @brief Strip trailing zeros, representing the zero polynomial by an empty vector. */
		template<static_prime_mod_int Coefficient>
		constexpr void series_normalize_polynomial(std::vector<Coefficient> &polynomial) {
			while (!polynomial.empty() && polynomial.back() == Coefficient(0)) polynomial.pop_back();
		}

		/**
		 * @brief Read a whole polynomial, normalized; trailing zeros are dropped without being stored.
		 * @pre The normalized length is at most `maximum_series_length` (asserted); trailing zeros may exceed it.
		 */
		template<prime_mod_int_range Range>
		[[nodiscard]] constexpr auto series_polynomial(Range const &polynomial) -> std::vector<series_coefficient_t<Range>> {
			using Coefficient = series_coefficient_t<Range>;
			std::vector<Coefficient> result;
			std::size_t zeros = 0;
			for (auto const &value : polynomial) {
				if (value == Coefficient(0)) {
					atl_assert(zeros < (std::numeric_limits<std::size_t>::max)());
					++zeros;
				} else {
					atl_assert(result.size() < maximum_series_length);
					atl_assert(zeros < maximum_series_length - result.size());
					result.resize(result.size() + zeros, Coefficient(0));
					result.push_back(value);
					zeros = 0;
				}
			}
			return result;
		}

		/** @brief A nonconstant nonzero coefficient with its degree. */
		template<static_prime_mod_int Coefficient>
		struct series_sparse_term {
			/** @brief Positive degree. */
			std::size_t degree;
			/** @brief Nonzero coefficient. */
			Coefficient value;
		};

		/** @brief The nonconstant nonzero terms in increasing degree order. */
		template<static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr auto series_sparse_terms(std::span<Coefficient const> series) -> std::vector<series_sparse_term<Coefficient>> {
			std::vector<series_sparse_term<Coefficient>> terms;
			for (std::size_t i = 1; i < series.size(); ++i)
				if (series[i] != Coefficient(0)) terms.push_back({i, series[i]});
			return terms;
		}

		/**
		 * @brief Solve `denominator * result = numerator` modulo `x^length`, visiting only the stored nonzero terms.
		 * @pre For positive `length` the denominator is nonempty with a nonzero constant term (asserted).
		 */
		template<static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_sparse_quotient(std::span<Coefficient const> numerator, std::span<Coefficient const> denominator, std::size_t length) {
			if (length == 0) return {};
			atl_assert(!denominator.empty() && denominator[0] != Coefficient(0));
			auto const terms = series_sparse_terms<Coefficient>(denominator);
			Coefficient const inverse = Coefficient(1) / denominator[0];
			std::vector<Coefficient> result(length);
			std::ranges::copy(numerator.first((std::min)(numerator.size(), length)), result.begin());
			for (std::size_t degree = 0; degree < length; ++degree) {
				Coefficient residual = result[degree];
				for (auto const &term : terms) {
					if (term.degree > degree) break;
					residual -= term.value * result[degree - term.degree];
				}
				result[degree] = residual * inverse;
			}
			return result;
		}

		/**
		 * @brief `f^exponent` for a unit `f`, from the differential equation `f g' = exponent f' g`.
		 * @pre `series[0] == 1` and `series.size() <= Coefficient::mod()` (asserted).
		 */
		template<static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_power_ode(std::span<Coefficient const> series, Coefficient exponent) {
			atl_assert(!series.empty() && series[0] == Coefficient(1) && series.size() <= Coefficient::mod());
			auto const terms = series_sparse_terms<Coefficient>(series);
			factorial_table<Coefficient> const table(series.size() - 1);
			std::vector<Coefficient> result(series.size());
			result[0] = Coefficient(1);
			for (std::size_t degree = 1; degree < series.size(); ++degree) {
				for (auto const &term : terms) {
					if (term.degree > degree) break;
					result[degree] += ((exponent + Coefficient(1)) * Coefficient(term.degree) - Coefficient(degree)) * term.value * result[degree - term.degree];
				}
				result[degree] *= table.inverse_integer(degree);
			}
			return result;
		}

		/**
		 * @brief `f^exponent` modulo `x^size` by binary powering with truncation after every product.
		 * @pre `series.size() <= maximum_series_length` (asserted).
		 */
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_power_binary(std::span<Coefficient const> series, std::uint64_t exponent) {
			atl_assert(series.size() <= maximum_series_length);
			std::vector<Coefficient> result(series.size()), base(series.begin(), series.end());
			if (result.empty()) return result;
			result[0] = Coefficient(1);
			while (exponent != 0) {
				if ((exponent & 1) != 0) result = series_product<Engine, Coefficient>(result, base, series.size());
				exponent >>= 1;
				if (exponent != 0) base = series_product<Engine, Coefficient>(base, base, series.size());
			}
			return result;
		}

		/** @brief A callable computing the power of a unit series with a full 64-bit integer exponent. */
		template<typename Operation, typename Coefficient>
		concept series_power_operation = static_prime_mod_int<Coefficient> && requires(Operation operation, std::span<Coefficient const> series, std::uint64_t exponent) {
			{ std::invoke(operation, series, exponent) } -> std::same_as<std::vector<Coefficient>>;
		};

		/**
		 * @brief Integer power with the valuation and the leading scalar split off before `operation` evaluates the unit power.
		 *
		 * Exponent zero gives one, a zero series to a positive exponent gives zeros, and a valuation shift that
		 * reaches `length` gives zeros; the comparison against `(length - 1) / valuation` precedes the
		 * multiplication so the shift cannot overflow. The scalar keeps the full integer exponent.
		 */
		template<static_prime_mod_int Coefficient, typename Operation>
			requires series_power_operation<Operation, Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_power_normalized(std::vector<Coefficient> series, std::uint64_t exponent, Operation operation) {
			std::size_t const length = series.size();
			std::vector<Coefficient> result(length);
			if (length == 0) return result;
			if (exponent == 0) {
				result[0] = Coefficient(1);
				return result;
			}
			std::size_t valuation = 0;
			while (valuation < length && series[valuation] == Coefficient(0)) ++valuation;
			if (valuation == length) return result;
			if (valuation != 0 && exponent > (length - 1) / valuation) return result;
			std::size_t const shift = valuation == 0 ? 0 : valuation * std::size_t(exponent);
			std::size_t const precision = length - shift;
			Coefficient const leading = series[valuation], inverse = Coefficient(1) / leading;
			for (std::size_t i = 0; i < precision; ++i) series[i] = series[valuation + i] * inverse;
			series.resize(precision);
			auto const unit = std::invoke(operation, std::span<Coefficient const>(series), exponent);
			Coefficient const scale = leading.power(exponent);
			for (std::size_t i = 0; i < precision; ++i) result[shift + i] = unit[i] * scale;
			return result;
		}

		/** @brief A root and its reciprocal, both correct at the current Newton precision. */
		template<static_prime_mod_int Coefficient>
		struct series_root_state {
			/** @brief Root approximation. */
			std::vector<Coefficient> root;
			/** @brief Reciprocal of the root approximation. */
			std::vector<Coefficient> inverse;
		};

		/**
		 * @brief Root of a unit series by coupled Newton on the shared driver.
		 * @pre The series is nonempty with constant term one, the characteristic is odd, and
		 *      `series.size() <= maximum_series_length` (asserted). There is no bound by the characteristic.
		 */
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_square_root_coupled(std::span<Coefficient const> series) {
			atl_assert(!series.empty() && series[0] == Coefficient(1) && Coefficient::mod() != 2);
			atl_assert(series.size() <= maximum_series_length);
			std::size_t const seed = (std::min)(series.size(), series_newton_cutoff);
			Coefficient const half = Coefficient(1) / Coefficient(2);
			series_root_state<Coefficient> state{std::vector<Coefficient>(seed), {}};
			state.root[0] = Coefficient(1);
			for (std::size_t degree = 1; degree < seed; ++degree) {
				Coefficient residual = series[degree];
				for (std::size_t i = 1; i < degree; ++i) residual -= state.root[i] * state.root[degree - i];
				state.root[degree] = residual * half;
			}
			if (seed == series.size()) return std::move(state.root);
			state.inverse = series_inverse_recurrence<Coefficient>(state.root, seed);
			series_newton_lift(state, seed, series.size(), [&](auto &current, std::size_t middle, std::size_t next) {
				// The residual vanishes below middle, so the old reciprocal suffices for the correction.
				auto square = series_product<Engine, Coefficient>(current.root, current.root, next);
				for (std::size_t i = middle; i < next; ++i) square[i] = (series[i] - square[i]) * half;
				auto const correction = series_product<Engine, Coefficient>(std::span(square).subspan(middle), current.inverse, next - middle);
				current.root.resize(next);
				std::ranges::copy(correction, std::span(current.root).subspan(middle).begin());
				if (next < series.size()) current.inverse = series_inverse_extend<Engine, Coefficient>(current.root, std::move(current.inverse), next);
			});
			return std::move(state.root);
		}

		/** @brief A callable computing the root of a unit series. */
		template<typename Operation, typename Coefficient>
		concept series_root_operation = static_prime_mod_int<Coefficient> && requires(Operation operation, std::span<Coefficient const> series) {
			{ std::invoke(operation, series) } -> std::same_as<std::vector<Coefficient>>;
		};

		/**
		 * @brief Canonical square root: the valuation and leading scalar are split off before `operation` computes the unit root.
		 *
		 * Returns nullopt for an odd valuation, a non-residue leading scalar, or (characteristic two) a nonzero
		 * odd-indexed coefficient. The coefficients the input does not determine are zero. Characteristic two
		 * never calls `operation`: the root takes every even-indexed coefficient.
		 */
		template<static_prime_mod_int Coefficient, typename Operation>
			requires series_root_operation<Operation, Coefficient>
		[[nodiscard]] constexpr std::optional<std::vector<Coefficient>> series_square_root_normalized(std::vector<Coefficient> series, Operation operation) {
			std::size_t const length = series.size();
			std::vector<Coefficient> result(length);
			if constexpr (Coefficient::mod() == 2) {
				for (std::size_t i = 0; i < length; ++i) {
					if (i % 2 == 0)
						result[i / 2] = series[i];
					else if (series[i] != Coefficient(0))
						return std::nullopt;
				}
				return result;
			} else {
				std::size_t valuation = 0;
				while (valuation < length && series[valuation] == Coefficient(0)) ++valuation;
				if (valuation == length) return result;
				if (valuation % 2 != 0) return std::nullopt;
				auto const scalar = square_root_mod(series[valuation]);
				if (!scalar) return std::nullopt;
				Coefficient const inverse = Coefficient(1) / series[valuation];
				std::size_t const precision = length - valuation;
				for (std::size_t i = 0; i < precision; ++i) series[i] = series[valuation + i] * inverse;
				series.resize(precision);
				auto const unit = std::invoke(operation, std::span<Coefficient const>(series));
				for (std::size_t i = 0; i < precision; ++i) result[valuation / 2 + i] = unit[i] * *scalar;
				return result;
			}
		}

		/**
		 * @brief Schoolbook long division of whole polynomials, also the base case of the fast division.
		 * @pre The denominator is nonzero and both normalized lengths are at most `maximum_series_length` (asserted).
		 */
		template<static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr polynomial_division_result<Coefficient> series_polynomial_long_division(std::vector<Coefficient> numerator, std::vector<Coefficient> denominator) {
			series_normalize_polynomial(numerator);
			series_normalize_polynomial(denominator);
			atl_assert(numerator.size() <= maximum_series_length && denominator.size() <= maximum_series_length);
			atl_assert(!denominator.empty());
			if (numerator.size() < denominator.size()) return {{}, std::move(numerator)};
			std::vector<Coefficient> quotient(numerator.size() - denominator.size() + 1);
			Coefficient const inverse = Coefficient(1) / denominator.back();
			for (std::size_t remaining = quotient.size(); remaining > 0; --remaining) {
				std::size_t const degree = remaining - 1;
				quotient[degree] = numerator[degree + denominator.size() - 1] * inverse;
				for (std::size_t i = 0; i < denominator.size(); ++i) numerator[degree + i] -= quotient[degree] * denominator[i];
			}
			numerator.resize(denominator.size() - 1);
			series_normalize_polynomial(numerator);
			series_normalize_polynomial(quotient);
			return {std::move(quotient), std::move(numerator)};
		}
	}
}
