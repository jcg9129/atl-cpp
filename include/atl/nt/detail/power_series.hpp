#pragma once

#include <atl/nt/convolution.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/nt/series_calculus.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Largest coefficient count the transform-backed power-series functions accept.
	 *
	 * The bound keeps the three-prime reconstruction of the convolution driver exact for every 32-bit modulus,
	 * and keeps every doubling step's size representable.
	 */
	inline constexpr std::size_t maximum_series_length = std::size_t{1} << 22;

	namespace detail {
		/** @brief Provisional crossover from the quadratic recurrences to Newton doubling; race pending. */
		inline constexpr std::size_t series_newton_cutoff = 64;

		/** @brief The first `length` coefficients of the input, padded with zeros; coefficients beyond are not read. No length cap. */
		template<prime_mod_int_range Range>
		[[nodiscard]] constexpr auto series_copy_prefix(Range const &series, std::size_t length)
			-> std::vector<series_coefficient_t<Range>> {
			std::vector<series_coefficient_t<Range>> result(length);
			if (length == 0) return result;
			auto first = std::ranges::begin(series);
			auto const last = std::ranges::end(series);
			for (std::size_t i = 0; i < length && first != last; ++i, ++first) result[i] = *first;
			return result;
		}

		/** @brief `series_copy_prefix` for the transform-backed paths. @pre `length <= maximum_series_length` (asserted). */
		template<prime_mod_int_range Range>
		[[nodiscard]] constexpr auto series_prefix(Range const &series, std::size_t length)
			-> std::vector<series_coefficient_t<Range>> {
			atl_assert(length <= maximum_series_length);
			return series_copy_prefix(series, length);
		}

		/** @brief A step that mutates a Newton state at two successive precisions. */
		template<typename Step, typename State>
		concept series_newton_step = std::invocable<Step &, State &, std::size_t, std::size_t>;

		/**
		 * @brief Newton driver: lift a state correct modulo `x^middle` to `x^length` by doubling.
		 *
		 * Calls `step(state, middle, next)` with `next = min(length, 2 * middle)`; the step must write only the
		 * coefficient interval `[middle, next)` and leave the lower part unchanged. Auxiliary series in the state
		 * obey the same precision invariant. Only the last step may be shorter than a doubling.
		 * @pre `middle <= length <= maximum_series_length`; unless `middle == length`, `middle` is a positive power
		 *      of two (asserted).
		 */
		template<typename State, typename Step>
			requires series_newton_step<Step, State>
		constexpr void series_newton_lift(State &state, std::size_t middle, std::size_t length, Step step) {
			atl_assert(middle <= length && length <= maximum_series_length);
			atl_assert(middle == length || std::has_single_bit(middle));
			while (middle < length) {
				atl_assert(middle <= (std::numeric_limits<std::size_t>::max)() / 2);
				auto const next = (std::min)(length, 2 * middle);
				std::invoke(step, state, middle, next);
				middle = next;
			}
		}

		/**
		 * @brief The first `length` coefficients of a product, padded with zeros, through the convolution driver.
		 *
		 * Moduli above 32 bits, and any modulus whose storage type has no native double-width multiply, are
		 * multiplied by 16-bit limbs through `convolution_uint64`, because the driver's three-prime reconstruction
		 * bound does not hold for full-width residues. Constant evaluation of that case is schoolbook.
		 * @pre Both input lengths and `length` are at most `maximum_series_length` (asserted).
		 */
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_product(std::span<Coefficient const> left,
																		std::span<Coefficient const> right, std::size_t length) {
			atl_assert(length <= maximum_series_length);
			atl_assert(left.size() <= maximum_series_length && right.size() <= maximum_series_length);
			left = left.first((std::min)(left.size(), length));
			right = right.first((std::min)(right.size(), length));
			if (left.empty() || right.empty()) return std::vector<Coefficient>(length);
			if constexpr (Coefficient::mod() <= (std::numeric_limits<std::uint32_t>::max)()
						  && traits::has_widen_mul_v<decltype(Coefficient::mod())>) {
				auto result = convolution<Engine>(left, right);
				result.resize(length);
				return result;
			} else {
				std::vector<Coefficient> result(length);
				if consteval {
					for (std::size_t i = 0; i < left.size(); ++i)
						for (std::size_t j = 0; j < right.size() && j < length - i; ++j) result[i + j] += left[i] * right[j];
				} else {
					std::vector<std::uint64_t> left_limb(left.size()), right_limb(right.size());
					Coefficient left_place(1);
					for (unsigned left_shift = 0; left_shift < 64; left_shift += 16) {
						for (std::size_t i = 0; i < left.size(); ++i) left_limb[i] = (std::uint64_t(left[i].value()) >> left_shift) & 65535;
						Coefficient place = left_place;
						for (unsigned right_shift = 0; right_shift < 64; right_shift += 16) {
							for (std::size_t i = 0; i < right.size(); ++i) right_limb[i] = (std::uint64_t(right[i].value()) >> right_shift) & 65535;
							auto const product = convolution_uint64<Engine>(left_limb, right_limb);
							for (std::size_t i = 0; i < (std::min)(length, product.size()); ++i) result[i] += Coefficient(product[i]) * place;
							place *= Coefficient(65536);
						}
						left_place *= Coefficient(65536);
					}
				}
				return result;
			}
		}

		/**
		 * @brief Inverse modulo `x^length` by the defining recurrence; also the Newton seed.
		 * @pre `length <= maximum_series_length`; for positive `length`, the constant term is nonzero (asserted).
		 */
		template<static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_inverse_recurrence(std::span<Coefficient const> series, std::size_t length) {
			atl_assert(length <= maximum_series_length);
			if (length == 0) return {};
			atl_assert(!series.empty() && series[0] != Coefficient(0));
			std::vector<Coefficient> result(length);
			result[0] = Coefficient(1) / series[0];
			for (std::size_t degree = 1; degree < length; ++degree) {
				Coefficient sum(0);
				for (std::size_t i = 1; i <= degree && i < series.size(); ++i) sum += series[i] * result[degree - i];
				result[degree] = -sum * result[0];
			}
			return result;
		}

		/**
		 * @brief Extend an inverse from `m` to `length <= 2m` coefficients, keeping its low half.
		 *
		 * The cyclic product of `f[0..length)` and `g[0..m)` aliases only below `m`, so the error's high half can
		 * be shifted down and multiplied by `-g`. One transform of `g` serves both products: three forward and two
		 * inverse transforms of size `2m`. Moduli without a suitable transform use two ordinary products.
		 * @pre `inverse` is correct modulo `x^m`; `m` is a power of two and `m < length <= 2m` (asserted).
		 */
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_inverse_extend(std::span<Coefficient const> series,
																			   std::vector<Coefficient> inverse, std::size_t length) {
			std::size_t const middle = inverse.size();
			atl_assert(length <= maximum_series_length);
			atl_assert(std::has_single_bit(middle) && middle < length && middle <= maximum_series_length / 2);
			std::size_t const size = 2 * middle;
			atl_assert(length <= size);
			std::size_t const width = length - middle;
			if constexpr (ntt_modulus<Coefficient> && traits::has_widen_mul_v<decltype(Coefficient::mod())>) {
				if ((Coefficient::mod() - 1) % size == 0) {
					std::vector<Coefficient> transformed_inverse(size), error(size);
					std::ranges::copy(inverse, transformed_inverse.begin());
					std::ranges::copy(series.first((std::min)(series.size(), length)), error.begin());
					ntt<Engine>(transformed_inverse);
					ntt<Engine>(error);
					for (std::size_t i = 0; i < size; ++i) error[i] *= transformed_inverse[i];
					inverse_ntt<Engine>(error);
					for (std::size_t i = 0; i < width; ++i) error[i] = error[middle + i];
					std::ranges::fill(std::span(error).subspan(width), Coefficient(0));
					ntt<Engine>(error);
					for (std::size_t i = 0; i < size; ++i) error[i] *= transformed_inverse[i];
					inverse_ntt<Engine>(error);
					inverse.resize(length);
					for (std::size_t i = 0; i < width; ++i) inverse[middle + i] = -error[i];
					return inverse;
				}
			}
			auto const product = series_product<Engine, Coefficient>(series, inverse, length);
			auto const correction = series_product<Engine, Coefficient>(std::span(product).subspan(middle), inverse, width);
			inverse.resize(length);
			for (std::size_t i = 0; i < width; ++i) inverse[middle + i] = -correction[i];
			return inverse;
		}

		/**
		 * @brief Inverse modulo `x^length`: recurrence seed, then high-half Newton doubling; race pending.
		 * @pre As `series_inverse_recurrence`.
		 */
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_inverse_newton(std::span<Coefficient const> series, std::size_t length) {
			atl_assert(length <= maximum_series_length);
			auto inverse = series_inverse_recurrence<Coefficient>(series, (std::min)(length, series_newton_cutoff));
			while (inverse.size() < length) {
				atl_assert(inverse.size() <= maximum_series_length / 2);
				auto const next = (std::min)(length, 2 * inverse.size());
				inverse = series_inverse_extend<Engine>(series, std::move(inverse), next);
			}
			return inverse;
		}

		/**
		 * @brief Division modulo `x^length` that, above the Newton cutoff, inverts the denominator only to
		 *        `m = bit_floor(length - 1)` (at or below the cutoff it uses the full inverse).
		 *
		 * The quotient's low part is `numerator * inverse`; the high part is the high half of
		 * `numerator - denominator * low` times that same inverse. The last three products replace constructing
		 * the full inverse; race pending.
		 * @pre Both inputs have exactly `length` coefficients and the denominator constant is nonzero for
		 *      positive `length` (asserted).
		 */
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_division_fused(std::span<Coefficient const> numerator,
																			   std::span<Coefficient const> denominator, std::size_t length) {
			atl_assert(length <= maximum_series_length);
			atl_assert(numerator.size() == length && denominator.size() == length);
			if (length == 0) return {};
			atl_assert(denominator[0] != Coefficient(0));
			std::size_t const middle = length <= series_newton_cutoff ? length : std::bit_floor(length - 1);
			auto const inverse = series_inverse_newton<Engine, Coefficient>(denominator, middle);
			auto quotient = series_product<Engine, Coefficient>(numerator.first(middle), inverse, middle);
			if (middle == length) return quotient;
			auto residual = series_product<Engine, Coefficient>(denominator, quotient, length);
			for (std::size_t i = middle; i < length; ++i) residual[i] = numerator[i] - residual[i];
			auto const correction = series_product<Engine, Coefficient>(std::span(residual).subspan(middle), inverse, length - middle);
			quotient.resize(length);
			std::ranges::copy(correction, std::span(quotient).subspan(middle).begin());
			return quotient;
		}

		/**
		 * @brief Logarithm by derivative, full inverse, product and integral.
		 * @pre The input has constant term one and `series.size() <= Coefficient::mod()` (asserted).
		 */
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_logarithm_product(std::span<Coefficient const> series) {
			atl_assert(series.size() <= maximum_series_length && series.size() <= Coefficient::mod());
			if (series.empty()) return {};
			atl_assert(series[0] == Coefficient(1));
			auto const derivative = series_derivative(series);
			auto const inverse = series_inverse_newton<Engine, Coefficient>(series, series.size() - 1);
			return series_integral(series_product<Engine, Coefficient>(derivative, inverse, series.size() - 1));
		}

		/**
		 * @brief Exponential modulo `x^length` by the defining recurrence.
		 * @pre For positive `length`: constant term zero and `length <= Coefficient::mod()` (asserted).
		 */
		template<static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_exponential_recurrence(std::span<Coefficient const> series, std::size_t length) {
			atl_assert(length <= maximum_series_length && length <= Coefficient::mod());
			if (length == 0) return {};
			atl_assert(!series.empty() && series[0] == Coefficient(0));
			std::vector<Coefficient> result(length);
			result[0] = Coefficient(1);
			for (std::size_t degree = 1; degree < length; ++degree) {
				Coefficient sum(0);
				for (std::size_t i = 1; i <= degree && i < series.size(); ++i) sum += Coefficient(i) * series[i] * result[degree - i];
				result[degree] = sum / Coefficient(degree);
			}
			return result;
		}

		/**
		 * @brief Exponential by Newton doubling that carries the reciprocal `1/g` along; race pending.
		 *
		 * Each step extends the old reciprocal, forms `log g`, and takes `delta = f - log g` on the new interval.
		 * Then `g` becomes `g * (1 + delta)` and `1/g` becomes `(1/g) * (1 - delta)`; both are exact because
		 * `delta^2` vanishes at the new precision.
		 * @pre The prefix has constant term zero and `series.size() <= Coefficient::mod()` (asserted).
		 */
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		[[nodiscard]] constexpr std::vector<Coefficient> series_exponential_coupled(std::span<Coefficient const> series) {
			std::size_t const length = series.size();
			atl_assert(length <= maximum_series_length && length <= Coefficient::mod());
			auto exponential = series_exponential_recurrence<Coefficient>(series, (std::min)(length, series_newton_cutoff));
			if (length <= series_newton_cutoff) return exponential;
			auto inverse = series_inverse_recurrence<Coefficient>(exponential, exponential.size());
			while (exponential.size() < length) {
				std::size_t const middle = exponential.size();
				atl_assert(middle <= maximum_series_length / 2);
				std::size_t const next = (std::min)(length, 2 * middle), width = next - middle;
				inverse = series_inverse_extend<Engine, Coefficient>(exponential, std::move(inverse), next);
				auto const derivative = series_derivative(exponential);
				auto const logarithm = series_integral(series_product<Engine, Coefficient>(derivative, inverse, next - 1));
				std::vector<Coefficient> delta(width);
				for (std::size_t i = 0; i < width; ++i) delta[i] = series[middle + i] - logarithm[middle + i];
				auto const correction = series_product<Engine, Coefficient>(exponential, delta, width);
				exponential.resize(next);
				std::ranges::copy(correction, std::span(exponential).subspan(middle).begin());
				if (next < length) {
					auto const inverse_correction = series_product<Engine, Coefficient>(std::span(inverse).first(middle), delta, width);
					for (std::size_t i = 0; i < width; ++i) inverse[middle + i] -= inverse_correction[i];
				}
			}
			return exponential;
		}
	}
}
