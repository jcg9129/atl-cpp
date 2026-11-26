#pragma once

/** @file
 * @brief Reduced polynomial_points definitions for native fields with direct NTT order at least 2^23.
 */

#include <atl/nt/detail/ntt_engine_base.hpp>
#include <atl/nt/detail/polynomial_points_native_base.hpp>
#include <atl/nt/detail/power_series_native_base.hpp>
#include <atl/nt/detail/prime_mod_int_base.hpp>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <vector>

namespace atl {
	/**
	 * @brief Coefficients of `f(x + shift)`, with the length of the input in every characteristic.
	 * @tparam Engine NTT engine used for the products; engines that do not support the modulus fall back to the
	 *         convolution driver's scalar tiers, and constant evaluation uses scalar transforms,
	 *         or schoolbook products where the field has none.
	 * @tparam Polynomial Input range of static prime-modulus integers; a single pass through a const reference suffices.
	 * @param polynomial Coefficients of `f` in increasing degree order.
	 * @param shift The translation.
	 * @return Exactly `polynomial.size()` coefficients, trailing zeros included.
	 * @pre The input length is at most `maximum_series_length` (asserted).
	 * @complexity O(M(n)) for `n <= p`, O(M(n) log n) for `n > p`, where `n` is the input length (the coefficient
	 *             count, trailing zeros included) and p the modulus.
	 * @note The method is chosen by `n` alone: for `n <= p` one factorial-scaled convolution (it needs `i!` and
	 *       `1 / i!` for `i < n`, which exist exactly when `n <= p`); for `n > p` divide and conquer on halves
	 *       combined with a power of `x + shift`, with quadratic leaves. Race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial>
	[[nodiscard]] constexpr auto polynomial_taylor_shift(Polynomial const &polynomial, series_coefficient_t<Polynomial> shift) {
		using Coefficient = series_coefficient_t<Polynomial>;
		return detail::points_shift_factorial<Engine, Coefficient>(detail::points_copy(polynomial), shift);
	}

	/**
	 * @brief The monic polynomial with the given roots, the product of `x - root`.
	 * @tparam Engine As for `polynomial_taylor_shift`.
	 * @tparam Roots Input range of static prime-modulus integers.
	 * @param roots The roots, repeats allowed.
	 * @return Exactly `roots.size() + 1` coefficients; `{1}` when there are no roots.
	 * @pre The number of roots is below `maximum_series_length` (asserted).
	 * @complexity O(M(n) log n) time and O(n) auxiliary memory for `n` roots. Race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Roots>
	[[nodiscard]] constexpr auto polynomial_from_roots(Roots const &roots) {
		using Coefficient = series_coefficient_t<Roots>;
		return detail::points_roots_balanced<Engine, Coefficient>(detail::points_copy(roots));
	}

	/**
	 * @brief Product of a list of polynomials, with trailing zeros removed.
	 * @tparam Engine As for `polynomial_taylor_shift`.
	 * @tparam Polynomials Input range whose elements are ranges of static prime-modulus integers of one type.
	 * @param polynomials The factors; each is read once and trailing zeros of a factor are ignored.
	 * @return The normalized product: `{1}` for an empty list, empty when any factor is zero.
	 * @pre Every factor's length, trailing zeros included, is at most `maximum_series_length`, and the total
	 *      degree of a nonzero product is below it (asserted).
	 * @complexity O(M(d) log k) for `k` factors of total degree `d`. Race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, detail::points_polynomial_range Polynomials>
	[[nodiscard]] constexpr auto polynomial_product(Polynomials const &polynomials) {
		using Coefficient = series_coefficient_t<std::ranges::range_value_t<Polynomials const>>;
		auto const factors = detail::points_factors(polynomials);
		return detail::points_product_balanced<Engine, Coefficient>(factors);
	}

	/**
	 * @brief A fixed list of evaluation points with its product tree, reusable for many polynomials.
	 *
	 * The tree owns the points, the products of their linear factors and the transforms of every child product,
	 * and is immutable after construction. Evaluation and interpolation share it, so building once pays off from
	 * the second operation on.
	 * @tparam Coefficient Static prime-modulus integer type of the points and values.
	 * @tparam Engine NTT engine used for the transforms and products.
	 * @complexity Construction O(M(n) log n) time and O(n log n) storage for `n` points; each operation
	 *             O(M(n) log n + M(d)) for an input of length `d`. Race pending.
	 */
	template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
	class polynomial_point_tree {
		/** @brief Products and spectra owned exclusively by this tree. */
		detail::points_tree_storage<Coefficient, Engine> storage_;

	public:
		/**
		 * @brief Build the tree over the points, in their original order.
		 * @tparam Points Input range of `Coefficient`.
		 * @param points The points; duplicates are allowed for evaluation but not for interpolation.
		 * @pre The point count is at most `maximum_series_length` (asserted).
		 */
		template<prime_mod_int_range Points>
			requires std::same_as<Coefficient, series_coefficient_t<Points>>
		constexpr explicit polynomial_point_tree(Points const &points) : storage_(detail::points_copy(points)) {}

		/** @brief Number of fixed points. */
		constexpr std::size_t point_count() const { return storage_.point_count(); }

		/**
		 * @brief The fixed points, exactly as given to the constructor: caller order, duplicates included, no padding.
		 * @return A view of `point_count()` points owned by the tree, valid until the tree is destroyed or assigned
		 *         to; the tree has no other mutating operation.
		 * @note Deleted on rvalues, where the view would dangle at once.
		 */
		[[nodiscard]] constexpr std::span<Coefficient const> points() const & { return storage_.points(); }

		/** @brief Deleted: the view of a temporary tree would dangle. */
		std::span<Coefficient const> points() const && = delete;

		/**
		 * @brief Evaluate a polynomial at every fixed point.
		 * @tparam Polynomial Input range of `Coefficient`.
		 * @param polynomial Coefficients in increasing degree order; a polynomial longer than the point count is
		 *        first reduced modulo the root product.
		 * @return One value per point, in point order, duplicates included.
		 * @pre The polynomial length is at most `maximum_series_length` (asserted).
		 */
		template<prime_mod_int_range Polynomial>
			requires std::same_as<Coefficient, series_coefficient_t<Polynomial>>
		[[nodiscard]] constexpr std::vector<Coefficient> evaluate(Polynomial const &polynomial) const {
			return storage_.evaluate(detail::points_copy(polynomial));
		}

		/**
		 * @brief The polynomial of degree below `point_count()` taking the given value at every point.
		 * @tparam Values Input range of `Coefficient`.
		 * @param values One value per point, in point order.
		 * @return Exactly `point_count()` coefficients, trailing zeros included; empty for no points.
		 * @pre `values.size() == point_count()` and the points are distinct; distinctness is checked through the
		 *      derivative of the root product being nonzero at every point (both asserted).
		 */
		template<prime_mod_int_range Values>
			requires std::same_as<Coefficient, series_coefficient_t<Values>>
		[[nodiscard]] constexpr std::vector<Coefficient> interpolate(Values const &values) const {
			return storage_.interpolate(detail::points_copy(values));
		}
	};

	/**
	 * @brief Evaluate a polynomial at arbitrary points, one value per point.
	 * @tparam Engine As for `polynomial_taylor_shift`.
	 * @tparam Polynomial Input range of static prime-modulus integers.
	 * @tparam Points Input range with the same coefficient type.
	 * @param polynomial Coefficients in increasing degree order.
	 * @param points The points, in the order of the output; duplicates are allowed.
	 * @return Exactly `points.size()` values.
	 * @pre Both input lengths are at most `maximum_series_length` (asserted).
	 * @complexity O(M(n) log n + M(d)) time and O(n log n + d) memory for `n` points and a polynomial of length `d`.
	 *             Race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial, prime_mod_int_range Points>
		requires std::same_as<series_coefficient_t<Polynomial>, series_coefficient_t<Points>>
	[[nodiscard]] constexpr auto polynomial_evaluation(Polynomial const &polynomial, Points const &points) {
		return polynomial_point_tree<series_coefficient_t<Points>, Engine>(points).evaluate(polynomial);
	}

	/**
	 * @brief The polynomial of degree below `n` through `n` point/value pairs.
	 * @tparam Engine As for `polynomial_taylor_shift`.
	 * @tparam Points Input range of static prime-modulus integers.
	 * @tparam Values Input range with the same coefficient type.
	 * @param points The distinct interpolation points.
	 * @param values The value at each point, in the same order.
	 * @return Exactly `n` coefficients, trailing zeros included; empty for `n = 0`.
	 * @pre Equal input lengths, at most `maximum_series_length`; the points are distinct, checked through the
	 *      derivative of the root product being nonzero at every point (all asserted).
	 * @complexity O(M(n) log n) time and O(n log n) memory. Race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Points, prime_mod_int_range Values>
		requires std::same_as<series_coefficient_t<Points>, series_coefficient_t<Values>>
	[[nodiscard]] constexpr auto polynomial_interpolation(Points const &points, Values const &values) {
		return polynomial_point_tree<series_coefficient_t<Points>, Engine>(points).interpolate(values);
	}

	/**
	 * @brief Values of a polynomial along a geometric progression, `f(first * ratio^j)` for `j < count`.
	 *
	 * A zero `first` or a zero `ratio` is handled directly, so the chirp-z method only inverts a nonzero ratio.
	 * @tparam Engine As for `polynomial_taylor_shift`.
	 * @tparam Polynomial Input range of static prime-modulus integers.
	 * @param polynomial Coefficients in increasing degree order.
	 * @param first The first point.
	 * @param ratio The ratio of consecutive points.
	 * @param count Number of values.
	 * @return Exactly `count` values.
	 * @pre The polynomial length and `count` are at most `maximum_series_length` (asserted).
	 * @complexity O(M(n + count)) time and O(n + count) memory for a polynomial of length `n`. Race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Polynomial>
	[[nodiscard]] constexpr auto polynomial_geometric_evaluation(Polynomial const &polynomial, series_coefficient_t<Polynomial> first, series_coefficient_t<Polynomial> ratio, std::size_t count) {
		using Coefficient = series_coefficient_t<Polynomial>;
		return detail::points_geometric<Engine, Coefficient>(detail::points_copy(polynomial), first, ratio, count);
	}

	/**
	 * @brief Samples `f(shift), ..., f(shift + count - 1)` of a polynomial from its samples `f(0), ..., f(n - 1)`.
	 *
	 * A target that coincides with a source node, including after wrapping modulo `p`, returns the known sample.
	 * @tparam Engine As for `polynomial_taylor_shift`.
	 * @tparam Samples Input range of static prime-modulus integers.
	 * @param samples `f(0), ..., f(n - 1)`; empty samples describe the zero polynomial.
	 * @param shift The first target, a residue modulo `p`.
	 * @param count Number of output samples; may exceed `p`, in which case the sequence repeats.
	 * @return Exactly `count` samples.
	 * @pre `n <= p`, the sample length and `count` are at most `maximum_series_length` (asserted); the degree of
	 *      `f` is below `n` (documented only).
	 * @complexity O(M(n + count)) time and O(n + count) memory. Race pending.
	 */
	template<ntt_engine Engine = default_ntt_engine, prime_mod_int_range Samples>
	[[nodiscard]] constexpr auto sample_point_shift(Samples const &samples, series_coefficient_t<Samples> shift, std::size_t count) {
		using Coefficient = series_coefficient_t<Samples>;
		return detail::points_sample_middle<Engine, Coefficient>(detail::points_copy(samples), shift, count);
	}
}
