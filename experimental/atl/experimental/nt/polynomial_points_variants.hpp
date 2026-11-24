#pragma once
#include <atl/nt/polynomial_points.hpp>

#include <atl/nt/detail/polynomial_points.hpp>
#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/factorial_table.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Contenders for the polynomial point operations of `<atl/nt/polynomial_points.hpp>`.
 *
 * Each one has the template parameters, argument order, result and preconditions of the library function it races,
 * unless its own comment says otherwise. All races are pending: no measurement decides any of them yet
 * (benchmarks/nt/polynomial_points.cpp).
 */

namespace atl::experimental {
	/**
	 * @brief Taylor shift by splitting the polynomial in halves and combining them with a power of `x + shift`.
	 *
	 * Needs no factorials, so it is the library's fallback beyond the characteristic; the contender uses it for
	 * every length. Race pending against `atl::polynomial_taylor_shift`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial>
	[[nodiscard]] constexpr auto polynomial_taylor_shift_by_divide_and_conquer(Polynomial const &polynomial, atl::series_coefficient_t<Polynomial> shift) {
		using Coefficient = atl::series_coefficient_t<Polynomial>;
		return atl::detail::points_shift_divide<Engine, Coefficient>(atl::detail::points_copy(polynomial), shift);
	}

	/** @brief Taylor shift by quadratic repeated multiplication by `x + shift` (the certifying contender); race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial>
	[[nodiscard]] constexpr auto polynomial_taylor_shift_by_horner(Polynomial const &polynomial, atl::series_coefficient_t<Polynomial> shift) {
		using Coefficient = atl::series_coefficient_t<Polynomial>;
		return atl::detail::points_shift_horner<Coefficient>(atl::detail::points_copy(polynomial), shift);
	}

	/** @brief Polynomial from roots by multiplying the linear factors one after another, quadratic; race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Roots>
	[[nodiscard]] constexpr auto polynomial_from_roots_by_sequential_product(Roots const &roots) {
		using Coefficient = atl::series_coefficient_t<Roots>;
		return atl::detail::points_roots_sequential<Coefficient>(atl::detail::points_copy(roots));
	}

	/** @brief Polynomial product merging the two shortest factors first (a heap by length) instead of a balanced tree; race pending. */
	template<ntt_engine Engine = default_ntt_engine, atl::detail::points_polynomial_range Polynomials>
	[[nodiscard]] constexpr auto polynomial_product_by_smallest_first(Polynomials const &polynomials) {
		using Coefficient = atl::series_coefficient_t<std::ranges::range_value_t<Polynomials const>>;
		auto factors = atl::detail::points_factors(polynomials);
		if (factors.empty()) return std::vector<Coefficient>{Coefficient(1)};
		auto const compare = [](std::vector<Coefficient> const &left, std::vector<Coefficient> const &right) { return left.size() > right.size(); };
		std::ranges::make_heap(factors, compare);
		while (factors.size() > 1) {
			std::ranges::pop_heap(factors, compare);
			auto left = std::move(factors.back());
			factors.pop_back();
			std::ranges::pop_heap(factors, compare);
			auto right = std::move(factors.back());
			factors.pop_back();
			if (left.empty() || right.empty()) return std::vector<Coefficient>{};
			atl_assert(left.size() <= maximum_series_length && right.size() <= maximum_series_length - left.size() + 1);
			factors.push_back(atl::detail::series_product<Engine, Coefficient>(left, right, left.size() + right.size() - 1));
			std::ranges::push_heap(factors, compare);
		}
		return std::move(factors.front());
	}

	/** @brief Polynomial product multiplying the factors one after another with schoolbook products, quadratic; race pending. */
	template<ntt_engine Engine = default_ntt_engine, atl::detail::points_polynomial_range Polynomials>
	[[nodiscard]] constexpr auto polynomial_product_by_sequential_schoolbook(Polynomials const &polynomials) {
		using Coefficient = atl::series_coefficient_t<std::ranges::range_value_t<Polynomials const>>;
		auto const factors = atl::detail::points_factors(polynomials);
		std::vector<Coefficient> result{Coefficient(1)};
		for (auto const &factor : factors) {
			if (factor.empty()) return std::vector<Coefficient>{};
			atl_assert(result.size() <= maximum_series_length && factor.size() <= maximum_series_length - result.size() + 1);
			std::vector<Coefficient> product(result.size() + factor.size() - 1);
			for (std::size_t i = 0; i < result.size(); ++i)
				for (std::size_t j = 0; j < factor.size(); ++j) product[i + j] += result[i] * factor[j];
			result = std::move(product);
		}
		return result;
	}

	/** @brief Multipoint evaluation by a remainder tree: polynomial division by each child product, instead of transposed middle products; race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial, prime_mod_int_range Points>
		requires std::same_as<atl::series_coefficient_t<Polynomial>, atl::series_coefficient_t<Points>>
	[[nodiscard]] constexpr auto polynomial_evaluation_by_remainder_tree(Polynomial const &polynomial, Points const &points) {
		using Coefficient = atl::series_coefficient_t<Points>;
		return atl::detail::points_tree_storage<Coefficient, Engine>(atl::detail::points_copy(points)).evaluate(atl::detail::points_copy(polynomial), true);
	}

	/** @brief Multipoint evaluation by Horner's rule at each point, quadratic (the certifying contender); race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial, prime_mod_int_range Points>
		requires std::same_as<atl::series_coefficient_t<Polynomial>, atl::series_coefficient_t<Points>>
	[[nodiscard]] constexpr auto polynomial_evaluation_by_horner(Polynomial const &polynomial, Points const &points) {
		using Coefficient = atl::series_coefficient_t<Points>;
		auto const coefficients = atl::detail::points_copy(polynomial);
		auto result = atl::detail::points_copy(points);
		for (auto &point : result) point = atl::detail::points_horner<Coefficient>(coefficients, point);
		return result;
	}

	/**
	 * @brief The library's transposed multipoint evaluation with a selectable Horner leaf size, for the cutoff sweep; race pending.
	 * @pre Those of `atl::polynomial_evaluation`; `1 <= cutoff < maximum_series_length`.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial, prime_mod_int_range Points>
		requires std::same_as<atl::series_coefficient_t<Polynomial>, atl::series_coefficient_t<Points>>
	[[nodiscard]] constexpr auto polynomial_evaluation_by_leaf_cutoff(Polynomial const &polynomial, Points const &points, std::size_t cutoff) {
		using Coefficient = atl::series_coefficient_t<Points>;
		return atl::detail::points_tree_storage<Coefficient, Engine>(atl::detail::points_copy(points), cutoff).evaluate(atl::detail::points_copy(polynomial));
	}

	/** @brief Interpolation whose derivative evaluation uses the remainder tree instead of middle products; race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Points, prime_mod_int_range Values>
		requires std::same_as<atl::series_coefficient_t<Points>, atl::series_coefficient_t<Values>>
	[[nodiscard]] constexpr auto polynomial_interpolation_by_remainder_tree(Points const &points, Values const &values) {
		using Coefficient = atl::series_coefficient_t<Points>;
		return atl::detail::points_tree_storage<Coefficient, Engine>(atl::detail::points_copy(points)).interpolate(atl::detail::points_copy(values), true);
	}

	/**
	 * @brief Interpolation by quadratic Newton divided differences and conversion from the Newton basis; race pending.
	 * @pre Equal input lengths <= maximum_series_length; points are distinct.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Points, prime_mod_int_range Values>
		requires std::same_as<atl::series_coefficient_t<Points>, atl::series_coefficient_t<Values>>
	[[nodiscard]] constexpr auto polynomial_interpolation_by_newton_differences(Points const &points, Values const &values) {
		using Coefficient = atl::series_coefficient_t<Points>;
		return atl::detail::points_interpolation_newton<Coefficient>(atl::detail::points_copy(points), atl::detail::points_copy(values));
	}

	/** @brief Geometric-progression evaluation by Horner's rule at each point, quadratic; race pending. */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial>
	[[nodiscard]] constexpr auto polynomial_geometric_evaluation_by_horner(Polynomial const &polynomial, atl::series_coefficient_t<Polynomial> first, atl::series_coefficient_t<Polynomial> ratio, std::size_t count) {
		using Coefficient = atl::series_coefficient_t<Polynomial>;
		atl_assert(count <= maximum_series_length);
		auto const coefficients = atl::detail::points_copy(polynomial);
		std::vector<Coefficient> result(count);
		for (auto &value : result) {
			value = atl::detail::points_horner<Coefficient>(coefficients, first);
			first *= ratio;
		}
		return result;
	}

	/**
	 * @brief Sample shift by three convolutions: samples to differences, translated falling factorials, then samples.
	 *
	 * Computes at most one field period of outputs and repeats it. Race pending against `atl::sample_point_shift`.
	 * @pre Sample count <= p; input and output lengths <= maximum_series_length.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Samples>
	[[nodiscard]] constexpr auto sample_point_shift_by_falling_factorials(Samples const &samples, atl::series_coefficient_t<Samples> shift, std::size_t count) {
		using Coefficient = atl::series_coefficient_t<Samples>;
		auto const values = atl::detail::points_copy(samples);
		atl_assert(values.size() <= Coefficient::mod() && count <= maximum_series_length);
		std::vector<Coefficient> result(count);
		if (values.empty() || count == 0) return result;
		std::size_t const length = values.size();
		std::size_t const outputs = Coefficient::mod() < count ? std::size_t(Coefficient::mod()) : count;
		factorial_table<Coefficient> const table((std::max)(length, outputs) - 1);
		std::vector<Coefficient> scaled(length), signed_inverse(length), falling(length);
		Coefficient product(1);
		for (std::size_t i = 0; i < length; ++i) {
			scaled[i] = values[i] * table.inverse_factorial(i);
			signed_inverse[i] = (i % 2 == 0 ? Coefficient(1) : Coefficient(-1)) * table.inverse_factorial(i);
			falling[i] = product * table.inverse_factorial(i);
			product *= shift - Coefficient(i);
		}
		auto differences = atl::detail::series_product<Engine, Coefficient>(scaled, signed_inverse, length);
		for (std::size_t i = 0; i < length; ++i) differences[i] *= table.factorial(i);
		std::ranges::reverse(differences);
		auto translated = atl::detail::series_product<Engine, Coefficient>(differences, falling, length);
		std::ranges::reverse(translated);
		for (std::size_t i = 0; i < length; ++i) translated[i] *= table.inverse_factorial(i);
		std::vector<Coefficient> inverse_factorials(outputs);
		for (std::size_t i = 0; i < outputs; ++i) inverse_factorials[i] = table.inverse_factorial(i);
		auto const convolution = atl::detail::series_product<Engine, Coefficient>(translated, inverse_factorials, outputs);
		for (std::size_t i = 0; i < count; ++i) {
			std::size_t const index = i % outputs;
			result[i] = convolution[index] * table.factorial(index);
		}
		return result;
	}

	/**
	 * @brief Sample shift by evaluating the barycentric formula directly at every target, quadratic.
	 *
	 * Targets that coincide with a source node return the known sample. Race pending.
	 * @pre Sample count <= p; input and output lengths <= maximum_series_length.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Samples>
	[[nodiscard]] constexpr auto sample_point_shift_by_direct_barycentric(Samples const &samples, atl::series_coefficient_t<Samples> shift, std::size_t count) {
		using Coefficient = atl::series_coefficient_t<Samples>;
		auto const values = atl::detail::points_copy(samples);
		atl_assert(values.size() <= Coefficient::mod() && count <= maximum_series_length);
		std::vector<Coefficient> result(count);
		if (values.empty()) return result;
		factorial_table<Coefficient> const table(values.size() - 1);
		for (std::size_t j = 0; j < count; ++j) {
			Coefficient const point = shift + Coefficient(j);
			if (point.value() < values.size()) {
				result[j] = values[std::size_t(point.value())];
				continue;
			}
			Coefficient product(1), sum(0);
			for (std::size_t i = 0; i < values.size(); ++i) {
				Coefficient weight = values[i] * table.inverse_factorial(i) * table.inverse_factorial(values.size() - 1 - i);
				if ((values.size() - 1 - i) % 2 != 0) weight = -weight;
				sum += weight / (point - Coefficient(i));
				product *= point - Coefficient(i);
			}
			result[j] = product * sum;
		}
		return result;
	}
}
