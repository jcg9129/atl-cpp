#pragma once

#include <atl/nt/detail/factorial_table_base.hpp>
#include <atl/nt/detail/ntt_engine_base.hpp>
#include <atl/nt/detail/power_series_native_base.hpp>
#include <atl/nt/detail/prime_mod_int_base.hpp>
#include <atl/nt/detail/series_calculus_base.hpp>
#include <atl/nt/detail/series_composition_native_base.hpp>
#include <atl/nt/polynomial_points_ntt.hpp>
#include <atl/nt/power_series_ntt.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace atl::detail {
	/** @brief The algorithms `composition_dispatch` can run after its structural cases; the choice is race pending. */
	enum class composition_algorithm {
		/** @brief Transpose of bivariate power projection. */
		kinoshita_li,
		/** @brief Baby-step/giant-step with classical matrix products. */
		brent_kung,
		/** @brief Horner with truncated fast products. */
		horner,
		/** @brief Horner with schoolbook products. */
		schoolbook
	};

	template<composition_algorithm Algorithm>
	struct composition_strategy;

	template<>
	struct composition_strategy<composition_algorithm::kinoshita_li> {
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		static constexpr std::vector<Coefficient> compose(std::span<Coefficient const> outer, std::span<Coefficient const> inner, std::size_t length, std::size_t horner_cutoff, std::size_t schoolbook_cutoff) {
			if (length <= schoolbook_cutoff) return composition_horner<Engine, true, Coefficient>(outer, inner, length);
			if (outer.size() <= horner_cutoff) return composition_horner<Engine, false, Coefficient>(outer, inner, length);
			return composition_kinoshita_li<Engine, Coefficient>(outer, inner);
		}
	};

	template<>
	struct composition_strategy<composition_algorithm::horner> {
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		static constexpr std::vector<Coefficient> compose(std::span<Coefficient const> outer, std::span<Coefficient const> inner, std::size_t length, std::size_t, std::size_t) {
			return composition_horner<Engine, false, Coefficient>(outer, inner, length);
		}
	};

	template<>
	struct composition_strategy<composition_algorithm::schoolbook> {
		template<ntt_engine Engine, static_prime_mod_int Coefficient>
		static constexpr std::vector<Coefficient> compose(std::span<Coefficient const> outer, std::span<Coefficient const> inner, std::size_t length, std::size_t, std::size_t) {
			return composition_horner<Engine, true, Coefficient>(outer, inner, length);
		}
	};

	/** @brief Shared finite-polynomial normalization and structural dispatch for all composition contenders.
	 * @pre Outer and requested lengths capped; inner has exactly length entries; positive cutoffs below the cap.
	 */
	template<ntt_engine Engine, composition_algorithm Algorithm, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_dispatch(std::vector<Coefficient> outer,
																		  std::vector<Coefficient> inner, std::size_t length,
																		  std::size_t horner_cutoff = composition_horner_cutoff, std::size_t schoolbook_cutoff = composition_schoolbook_cutoff) {
		atl_assert(outer.size() <= maximum_series_length && length <= maximum_series_length && inner.size() == length);
		atl_assert(horner_cutoff > 0 && horner_cutoff < maximum_series_length && schoolbook_cutoff > 0 && schoolbook_cutoff < maximum_series_length);
		std::vector<Coefficient> result(length);
		if (length == 0 || outer.empty()) return result;
		auto const constant = inner[0];
		inner[0] = Coefficient(0);
		std::size_t valuation = length, nonzero_count = 0;
		for (std::size_t i = 1; i < length; ++i) {
			if (inner[i] == Coefficient(0)) continue;
			valuation = (std::min)(valuation, i);
			++nonzero_count;
		}
		if (nonzero_count == 0) {
			result[0] = points_horner<Coefficient>(outer, constant);
			return result;
		}
		if (constant != Coefficient(0)) outer = polynomial_taylor_shift<Engine>(outer, constant);
		outer.resize((std::min)(outer.size(), (length - 1) / valuation + 1));
		if (nonzero_count == 1) {
			Coefficient power(1);
			for (std::size_t i = 0; i < outer.size(); ++i) {
				result[i * valuation] = outer[i] * power;
				power *= inner[valuation];
			}
			return result;
		}
		return composition_strategy<Algorithm>::template compose<Engine, Coefficient>(outer, inner, length, horner_cutoff, schoolbook_cutoff);
	}

	/** @brief Assert the reversion contract once, with no linear-coefficient check for precision one.
	 * @pre Prefix length at most maximum_series_length; zero constant for positive precision, unit linear term from precision two.
	 */
	template<static_prime_mod_int Coefficient>
	constexpr void composition_check_inverse(std::span<Coefficient const> series) {
		atl_assert(series.size() <= maximum_series_length);
		atl_assert(series.empty() || series[0] == Coefficient(0));
		atl_assert(series.size() < 2 || series[1] != Coefficient(0));
	}

	/** @brief Reversion by one composition per Newton step; valid in every characteristic; race pending.
	 * h=f(g) gives f'(g)=h'/g', avoiding a second composition. Derivatives are needed only to the correction's width.
	 * @pre A capped precision-sized prefix satisfying the reversion contract.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_inverse_newton(std::span<Coefficient const> series) {
		composition_check_inverse<Coefficient>(series);
		auto const length = series.size();
		if (length < 2) return std::vector<Coefficient>(length);
		std::vector<Coefficient> result{Coefficient(0), Coefficient(1) / series[1]};
		series_newton_lift(result, 2, length, [&](std::vector<Coefficient> &inverse, std::size_t middle, std::size_t next) {
			std::vector<Coefficient> outer(series.begin(), series.begin() + next), inner = inverse;
			inner.resize(next);
			auto composed = composition_dispatch<Engine, composition_algorithm::kinoshita_li, Coefficient>(std::move(outer), std::move(inner), next);
			auto const width = next - middle;
			auto const derivative = series_derivative(inverse);
			auto const composed_derivative = series_derivative(composed);
			auto numerator = series_product<Engine, Coefficient>(std::span(composed).subspan(middle), derivative, width);
			std::vector<Coefficient> denominator(composed_derivative.begin(), composed_derivative.begin() + width);
			auto const correction = series_division<Engine>(numerator, denominator, width);
			inverse.resize(next);
			for (std::size_t i = 0; i < width; ++i) inverse[middle + i] = -correction[i];
		});
		return result;
	}

	/** @brief Projection/Lagrange reversion followed by normalized series power; race pending.
	 * For d=n-1, [x^d]f^i=(i/d)[t^(d-i)](t/g)^d. Reversing/scaling these moments recovers (t/g)^d.
	 * @pre A capped precision-sized reversion prefix with length <= p.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_inverse_lagrange(std::span<Coefficient const> series) {
		composition_check_inverse<Coefficient>(series);
		auto const length = series.size();
		atl_assert(length <= Coefficient::mod());
		if (length < 2) return std::vector<Coefficient>(length);
		auto const degree = length - 1;
		std::vector<Coefficient> weights(length);
		weights.back() = Coefficient(1);
		auto const projected = composition_project<Engine, true, Coefficient>(weights, series, length);
		factorial_table<Coefficient> const table(degree);
		std::vector<Coefficient> power(degree);
		for (std::size_t i = 0; i < degree; ++i) power[i] = Coefficient(degree) * table.inverse_integer(degree - i) * projected[degree - i];
		auto const inverse_constant = Coefficient(1) / power[0];
		for (auto &value : power) value *= inverse_constant;
		auto const exponent = (Coefficient(-1) / Coefficient(degree)).value();
		auto unit = series_power<Engine>(power, std::uint64_t(exponent), degree);
		auto const scale = Coefficient(1) / series[1];
		std::vector<Coefficient> result(length);
		for (std::size_t i = 0; i < degree; ++i) result[i + 1] = scale * unit[i];
		return result;
	}

	/** @brief Direct triangular coefficient recurrence for tiny reversion; race pending.
	 * Each new coefficient cancels f(g)'s next residual; the general recurrence is cubic, not quadratic.
	 * @pre A capped precision-sized prefix satisfying the reversion contract.
	 * @complexity O(n^3) time and O(n^2) storage; no characteristic restriction.
	 */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_inverse_recurrence(std::span<Coefficient const> series) {
		composition_check_inverse<Coefficient>(series);
		auto const length = series.size();
		std::vector<Coefficient> result(length);
		if (length < 2) return result;
		auto const scale = Coefficient(1) / series[1];
		std::vector<std::vector<Coefficient>> powers(length, std::vector<Coefficient>(length));
		powers[0][0] = Coefficient(1);
		for (std::size_t degree = 1; degree < length; ++degree) {
			Coefficient residual(0);
			for (std::size_t exponent = 2; exponent <= degree; ++exponent) {
				for (std::size_t i = 1; i < degree; ++i) powers[exponent][degree] += result[i] * powers[exponent - 1][degree - i];
				residual += series[exponent] * powers[exponent][degree];
			}
			result[degree] = (Coefficient(degree == 1 ? 1 : 0) - residual) * scale;
			powers[1][degree] = result[degree];
		}
		return result;
	}
}
