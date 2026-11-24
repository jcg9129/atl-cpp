#pragma once

#include <atl/nt/convolution_mod.hpp>
#include <atl/nt/detail/polynomial_points.hpp>
#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/detail/power_series_extensions.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/power_series_native.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace atl::detail {
	/** @brief Provisional Euclidean leaf size of the half-gcd recursion; race pending (swept over 32..256). */
	inline constexpr std::size_t polynomial_half_gcd_cutoff = 64;

	/** @brief Polynomial Euclidean contenders; race pending. */
	enum class polynomial_gcd_algorithm {
		/** @brief Recursive high-half transitions with ordinary products. */
		half_gcd,
		/** @brief Recursive transitions with the spectra of each matrix entry cached and the two products of an output entry summed before its inverse transform. */
		transformed_half_gcd,
		/** @brief Euclid with fast polynomial division. */
		euclid,
		/** @brief Euclid with schoolbook division. */
		schoolbook
	};

	/** @brief Two consecutive Euclidean remainders; `earlier` is the one of larger degree and is divided by `later` next. */
	template<static_prime_mod_int Coefficient>
	struct polynomial_remainders {
		std::vector<Coefficient> earlier;
		std::vector<Coefficient> later;
	};

	/** @brief Row-major two-by-two matrix of polynomials. */
	template<static_prime_mod_int Coefficient>
	using polynomial_transition = std::array<std::vector<Coefficient>, 4>;

	/** @brief Normalized product modulo `x^maximum_series_length`; the transition identities cancel any higher terms.
	 * @pre Both lengths are at most `maximum_series_length` (asserted).
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr auto gcd_product(std::span<Coefficient const> first, std::span<Coefficient const> second) -> std::vector<Coefficient> {
		if (first.empty() || second.empty()) return {};
		atl_assert(first.size() <= maximum_series_length && second.size() <= maximum_series_length);
		auto result = series_product<Engine, Coefficient>(first, second, (std::min)(maximum_series_length, first.size() + second.size() - 1));
		series_normalize_polynomial(result);
		return result;
	}

	/** @brief `target += source` (or `-=`), then normalize `target`.
	 * @pre Both lengths are at most `maximum_series_length` (asserted).
	 */
	template<static_prime_mod_int Coefficient>
	constexpr void gcd_accumulate(std::vector<Coefficient> &target, std::span<Coefficient const> source, bool should_subtract = false) {
		atl_assert(source.size() <= maximum_series_length && target.size() <= maximum_series_length);
		target.resize(std::max(target.size(), source.size()));
		for (std::size_t i = 0; i < source.size(); ++i) target[i] += should_subtract ? -source[i] : source[i];
		series_normalize_polynomial(target);
	}

	/** @brief Product of rectangular polynomial matrices; with `ShouldTransform` and a transform-friendly field, each input
	 *         spectrum is computed once and every output entry sums its pointwise products before one inverse transform.
	 * @pre Every input entry has length at most `maximum_series_length` (asserted). Outputs are reduced modulo that power of `x`.
	 */
	template<ntt_engine Engine, bool ShouldTransform, std::size_t Rows, std::size_t Inner, std::size_t Columns, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr auto gcd_matrix_product(std::array<std::vector<Coefficient>, Rows * Inner> const &left,
													std::array<std::vector<Coefficient>, Inner * Columns> const &right) -> std::array<std::vector<Coefficient>, Rows * Columns> {
		std::size_t length = 0;
		for (std::size_t row = 0; row < Rows; ++row)
			for (std::size_t column = 0; column < Columns; ++column)
				for (std::size_t index = 0; index < Inner; ++index) {
					auto const &first = left[row * Inner + index];
					auto const &second = right[index * Columns + column];
					atl_assert(first.size() <= maximum_series_length && second.size() <= maximum_series_length);
					if (first.empty() || second.empty()) continue;
					length = (std::max)(length, first.size() + second.size() - 1);
				}
		std::array<std::vector<Coefficient>, Rows * Columns> result;
		if (length == 0) return result;
		if constexpr (ShouldTransform && points_transform_coefficient<Coefficient>) {
			auto const size = std::bit_ceil(length);
			if (length > polynomial_half_gcd_cutoff && size <= maximum_series_length && points_can_transform<Coefficient>(size)) {
				auto first_spectra = left;
				auto second_spectra = right;
				for (auto &entry : first_spectra) {
					entry.resize(size);
					ntt<Engine>(entry);
				}
				for (auto &entry : second_spectra) {
					entry.resize(size);
					ntt<Engine>(entry);
				}
				for (std::size_t row = 0; row < Rows; ++row)
					for (std::size_t column = 0; column < Columns; ++column) {
						auto &entry = result[row * Columns + column];
						entry.resize(size);
						for (std::size_t index = 0; index < Inner; ++index)
							for (std::size_t i = 0; i < size; ++i)
								entry[i] += first_spectra[row * Inner + index][i] * second_spectra[index * Columns + column][i];
						inverse_ntt<Engine>(entry);
						entry.resize(length);
						series_normalize_polynomial(entry);
					}
				return result;
			}
		}
		for (std::size_t row = 0; row < Rows; ++row)
			for (std::size_t column = 0; column < Columns; ++column)
				for (std::size_t index = 0; index < Inner; ++index)
					gcd_accumulate<Coefficient>(result[row * Columns + column], gcd_product<Engine, Coefficient>(left[row * Inner + index], right[index * Columns + column]));
		return result;
	}

	template<ntt_engine Engine, bool ShouldTransform, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr auto gcd_apply(polynomial_transition<Coefficient> const &transition, std::vector<Coefficient> first, std::vector<Coefficient> second) -> polynomial_remainders<Coefficient> {
		auto result = gcd_matrix_product<Engine, ShouldTransform, 2, 2, 1, Coefficient>(transition, {std::move(first), std::move(second)});
		return {std::move(result[0]), std::move(result[1])};
	}

	/** @brief Left-compose one exact Euclidean step with the quotient `q`: the matrix `[[0, 1], [1, -q]]` times `transition`. */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	constexpr void gcd_transition_step(polynomial_transition<Coefficient> &transition, std::span<Coefficient const> quotient) {
		for (std::size_t column = 0; column < 2; ++column) {
			auto next = std::move(transition[column]);
			gcd_accumulate<Coefficient>(next, gcd_product<Engine, Coefficient>(quotient, transition[2 + column]), true);
			transition[column] = std::move(transition[2 + column]);
			transition[2 + column] = std::move(next);
		}
	}

	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr auto gcd_high(std::vector<Coefficient> const &polynomial, std::size_t shift) -> std::vector<Coefficient> {
		if (shift >= polynomial.size()) return {};
		return {polynomial.begin() + std::ptrdiff_t(shift), polynomial.end()};
	}

	/** @brief Half-gcd from high coefficients, following the degree invariant of von zur Gathen and Gerhard, chapter 11.
	 * The returned unimodular matrix stops when the second degree first falls below `ceil(deg(first) / 2)`.
	 * The arguments are taken by value: every caller passes temporaries, which are moved into the recursion.
	 * @pre Inputs are normalized, `first` is nonzero, `deg(second) < deg(first)` and `1 <= cutoff <= maximum_series_length` (asserted).
	 */
	template<ntt_engine Engine, bool ShouldTransform, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr auto gcd_half(std::vector<Coefficient> first, std::vector<Coefficient> second, std::size_t cutoff) -> polynomial_transition<Coefficient> {
		atl_assert(!first.empty() && second.size() < first.size());
		atl_assert(first.size() <= maximum_series_length && cutoff > 0 && cutoff <= maximum_series_length);
		atl_assert(first.back() != Coefficient(0) && (second.empty() || second.back() != Coefficient(0)));
		std::size_t const middle = first.size() / 2;
		polynomial_transition<Coefficient> transition{{{1}, {}, {}, {1}}};
		if (second.size() <= middle) return transition;
		if (first.size() <= cutoff || middle == 0) {
			auto earlier = std::move(first), later = std::move(second);
			while (later.size() > middle) {
				auto division = polynomial_division<Engine>(earlier, later);
				gcd_transition_step<Engine, Coefficient>(transition, division.quotient);
				earlier = std::move(later);
				later = std::move(division.remainder);
			}
			return transition;
		}
		transition = gcd_half<Engine, ShouldTransform>(gcd_high(first, middle), gcd_high(second, middle), cutoff);
		auto reduced = gcd_apply<Engine, ShouldTransform>(transition, std::move(first), std::move(second));
		if (reduced.later.size() <= middle) return transition;
		auto division = polynomial_division<Engine>(reduced.earlier, reduced.later);
		gcd_transition_step<Engine, Coefficient>(transition, division.quotient);
		if (division.remainder.size() <= middle) return transition;
		// The boundary quotient can skip many degrees, so the next high shift comes from the actual degree.
		atl_assert(reduced.later.size() - 1 <= 2 * middle);
		std::size_t const shift = 2 * middle - (reduced.later.size() - 1);
		auto const tail = gcd_half<Engine, ShouldTransform>(gcd_high(reduced.later, shift), gcd_high(division.remainder, shift), cutoff);
		transition = gcd_matrix_product<Engine, ShouldTransform, 2, 2, 2, Coefficient>(tail, transition);
		return transition;
	}

	/** @brief Division for a Euclidean contender: schoolbook for `schoolbook`, fast otherwise.
	 * @pre `first` may be the sentinel `x^maximum_series_length` (length `maximum_series_length + 1`) with a divisor of
	 *      length at least two; otherwise both lengths are at most `maximum_series_length` (asserted).
	 */
	template<ntt_engine Engine, polynomial_gcd_algorithm Algorithm, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr auto gcd_divide(std::vector<Coefficient> const &first, std::vector<Coefficient> const &second) -> polynomial_division_result<Coefficient> {
		if (first.size() == maximum_series_length + 1) {
			atl_assert(second.size() >= 2 && second.size() <= maximum_series_length);
			atl_assert(first.back() == Coefficient(1));
			atl_assert(std::ranges::all_of(std::span(first).first(maximum_series_length), [](Coefficient value) { return value == Coefficient(0); }));
			std::size_t const count = first.size() - second.size() + 1;
			auto reversed = second;
			std::ranges::reverse(reversed);
			auto quotient = series_inverse<Engine>(reversed, count);
			std::ranges::reverse(quotient);
			auto remainder = series_product<Engine, Coefficient>(quotient, second, second.size() - 1);
			for (auto &value : remainder) value = -value;
			series_normalize_polynomial(remainder);
			series_normalize_polynomial(quotient);
			return {std::move(quotient), std::move(remainder)};
		}
		if constexpr (Algorithm == polynomial_gcd_algorithm::schoolbook)
			return series_polynomial_long_division(first, second);
		else
			return polynomial_division<Engine>(first, second);
	}

	/** @brief Reduced remainders and, for `Columns` of one or two, the cofactors that express them in the inputs. */
	template<static_prime_mod_int Coefficient, std::size_t Columns>
		requires(Columns <= 2)
	struct polynomial_euclidean_state {
		polynomial_remainders<Coefficient> remainders;
		/** @brief Row-major 2 by `Columns` cofactors of the original inputs; empty for the gcd alone. */
		std::array<std::vector<Coefficient>, 2 * Columns> cofactors;
	};

	/** @brief Euclid until `second.size() <= bound`; zero columns never accumulate cofactors.
	 * One column tracks only the second input (modular inverse and Pade); two track both inputs.
	 * @pre `bound <= maximum_series_length` and `1 <= cutoff <= maximum_series_length`. Inputs are normalized, with lengths
	 *      within the cap except an optional first input `x^maximum_series_length` (all asserted).
	 */
	template<ntt_engine Engine, polynomial_gcd_algorithm Algorithm, std::size_t Columns, static_prime_mod_int Coefficient>
		requires(Columns <= 2)
	[[nodiscard]] constexpr auto gcd_reduce(std::vector<Coefficient> first, std::vector<Coefficient> second, std::size_t bound, std::size_t cutoff) -> polynomial_euclidean_state<Coefficient, Columns> {
		atl_assert(bound <= maximum_series_length && cutoff > 0 && cutoff <= maximum_series_length);
		atl_assert(first.size() <= maximum_series_length + 1 && second.size() <= maximum_series_length);
		atl_assert((first.empty() || first.back() != Coefficient(0)) && (second.empty() || second.back() != Coefficient(0)));
		if (first.size() == maximum_series_length + 1) {
			atl_assert(first.back() == Coefficient(1) && std::ranges::all_of(std::span(first).first(maximum_series_length), [](Coefficient value) { return value == Coefficient(0); }));
		}
		polynomial_euclidean_state<Coefficient, Columns> state{{std::move(first), std::move(second)}, {}};
		if constexpr (Columns == 1) state.cofactors[1] = {1};
		if constexpr (Columns == 2) {
			state.cofactors[0] = {1};
			state.cofactors[3] = {1};
		}
		auto &earlier = state.remainders.earlier;
		auto &later = state.remainders.later;
		while (!later.empty() && later.size() > bound) {
			if constexpr (Algorithm == polynomial_gcd_algorithm::half_gcd || Algorithm == polynomial_gcd_algorithm::transformed_half_gcd) {
				if (earlier.size() <= maximum_series_length && earlier.size() > later.size() && earlier.size() > cutoff) {
					constexpr bool should_transform = Algorithm == polynomial_gcd_algorithm::transformed_half_gcd;
					std::size_t const half = earlier.size() / 2;
					std::size_t const target = (std::max)(bound, half);
					if (later.size() > target && target < earlier.size() - 1) {
						std::size_t const shift = 2 * target - (earlier.size() - 1);
						auto const transition = gcd_half<Engine, should_transform>(gcd_high(earlier, shift), gcd_high(later, shift), cutoff);
						auto reduced = gcd_apply<Engine, should_transform>(transition, std::move(earlier), std::move(later));
						earlier = std::move(reduced.earlier);
						later = std::move(reduced.later);
						if constexpr (Columns != 0)
							state.cofactors = gcd_matrix_product<Engine, should_transform, 2, 2, Columns, Coefficient>(transition, state.cofactors);
						if (later.empty() || later.size() <= bound) break;
					}
				}
			}
			auto division = gcd_divide<Engine, Algorithm>(earlier, later);
			if constexpr (Columns != 0) {
				for (std::size_t column = 0; column < Columns; ++column) {
					auto next = std::move(state.cofactors[column]);
					gcd_accumulate<Coefficient>(next, gcd_product<Engine, Coefficient>(division.quotient, state.cofactors[Columns + column]), true);
					state.cofactors[column] = std::move(state.cofactors[Columns + column]);
					state.cofactors[Columns + column] = std::move(next);
				}
			}
			earlier = std::move(later);
			later = std::move(division.remainder);
		}
		return state;
	}

	template<static_prime_mod_int Coefficient>
	constexpr void gcd_scale(std::vector<Coefficient> &polynomial, Coefficient scalar) {
		for (auto &value : polynomial) value *= scalar;
	}
}
