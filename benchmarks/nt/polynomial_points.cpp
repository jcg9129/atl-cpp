/**
 * @file
 * @brief Polynomial point operations: the library functions of `<atl/nt/polynomial_points.hpp>` against their contenders.
 *
 * Raced, one operation at a time, each with the one thing the contender does differently:
 * - `polynomial_taylor_shift`: library = factorial-scaled convolution; by divide and conquer = halves combined with a power of
 *   `x + shift`; by Horner = quadratic repeated multiplication.
 * - `polynomial_from_roots`: library = balanced product tree with schoolbook leaves; by sequential product = one factor at a time.
 * - `polynomial_product` of unequal factors: library = balanced tree; by smallest first = merge the two shortest factors first;
 *   by sequential schoolbook = one factor at a time.
 * - `polynomial_evaluation`: library = transposed middle-product descent; by remainder tree = polynomial division by each child
 *   product; by Horner = quadratic; by leaf cutoff = the library descent with the Horner leaf size swept over 16..128.
 * - `polynomial_interpolation`: library = derivative weights from the middle-product descent; by remainder tree = derivative
 *   weights from the remainder tree; by Newton differences = quadratic divided differences.
 * - `polynomial_geometric_evaluation`: library = chirp-z with a middle product; by Horner = quadratic.
 * - `sample_point_shift`: library = barycentric weights with a middle product; by falling factorials = three convolutions;
 *   by direct barycentric = the formula at every target, quadratic.
 * - `polynomial_point_tree`: evaluation and interpolation with the tree built once, against the library functions that build it per call.
 *
 * Every contender is checked against the library result before any timing. Lengths n = m: 2^10, 2^12, 2^16, 2^17 and 500000
 * modulo 998244353, and 2^16 modulo 1000000007; the quadratic contenders run only up to 2^12. The unequal-factor shape has factors
 * of degree 1..17 with total degree n. Each sample is one call, including allocation; the loop measures latency.
 *
 * Result: not measured yet (race pending). The page docs/polynomial-points.md has the table once a record exists.
 */
#include <atl/experimental/nt/polynomial_points_variants.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/polynomial_points.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
	namespace ex = atl::experimental;

	/** @brief The quadratic contenders are never run above this length. */
	constexpr std::size_t quadratic_limit = std::size_t{1} << 12;

	/** @brief Check a contender against the expected result, then time it under the label. */
	template<typename Result, typename Contender>
		requires std::is_invocable_r_v<Result, Contender &>
	void time_contender(std::string const &label, Result const &expected, Contender contender) {
		REQUIRE(contender() == expected);
		BENCHMARK(std::string(label)) { return contender(); };
	}

	template<atl::static_prime_mod_int Coefficient>
	void race(std::size_t length) {
		using Vector = std::vector<Coefficient>;
		std::mt19937_64 random(7001);
		Vector polynomial(length), points(length);
		for (auto &value : polynomial) value = Coefficient(random());
		for (std::size_t i = 0; i < length; ++i) points[i] = Coefficient(i);
		auto const shape = " | p=" + std::to_string(Coefficient::mod()) + " n=m=" + std::to_string(length);
		bool const is_quadratic_size = length <= quadratic_limit;
		Coefficient const shift(713), first(3), ratio(5);

		auto const shifted = atl::polynomial_taylor_shift(polynomial, shift);
		time_contender("polynomial_taylor_shift (library)" + shape, shifted, [&] { return atl::polynomial_taylor_shift(polynomial, shift); });
		time_contender("polynomial_taylor_shift by divide and conquer" + shape, shifted, [&] { return ex::polynomial_taylor_shift_by_divide_and_conquer(polynomial, shift); });
		if (is_quadratic_size)
			time_contender("polynomial_taylor_shift by Horner" + shape, shifted, [&] { return ex::polynomial_taylor_shift_by_horner(polynomial, shift); });

		auto const roots = atl::polynomial_from_roots(points);
		time_contender("polynomial_from_roots (library)" + shape, roots, [&] { return atl::polynomial_from_roots(points); });
		if (is_quadratic_size)
			time_contender("polynomial_from_roots by sequential product" + shape, roots, [&] { return ex::polynomial_from_roots_by_sequential_product(points); });

		std::vector<Vector> factors;
		std::size_t remaining = length;
		while (remaining != 0) {
			std::size_t const degree = std::min(remaining, 1 + factors.size() * 7 % 17);
			Vector factor(degree + 1);
			for (auto &value : factor) value = Coefficient(random());
			factor.back() = Coefficient(1);
			factors.push_back(std::move(factor));
			remaining -= degree;
		}
		auto const product = atl::polynomial_product(factors);
		time_contender("polynomial_product (library)" + shape, product, [&] { return atl::polynomial_product(factors); });
		time_contender("polynomial_product by smallest first" + shape, product, [&] { return ex::polynomial_product_by_smallest_first(factors); });
		if (is_quadratic_size)
			time_contender("polynomial_product by sequential schoolbook" + shape, product, [&] { return ex::polynomial_product_by_sequential_schoolbook(factors); });

		auto const values = atl::polynomial_evaluation(polynomial, points);
		time_contender("polynomial_evaluation (library)" + shape, values, [&] { return atl::polynomial_evaluation(polynomial, points); });
		time_contender("polynomial_evaluation by remainder tree" + shape, values, [&] { return ex::polynomial_evaluation_by_remainder_tree(polynomial, points); });
		if (is_quadratic_size)
			time_contender("polynomial_evaluation by Horner" + shape, values, [&] { return ex::polynomial_evaluation_by_horner(polynomial, points); });
		for (std::size_t cutoff : {16, 24, 32, 48, 64, 96, 128})
			time_contender("polynomial_evaluation by leaf cutoff=" + std::to_string(cutoff) + shape, values, [&] { return ex::polynomial_evaluation_by_leaf_cutoff(polynomial, points, cutoff); });

		auto const interpolated = atl::polynomial_interpolation(points, values);
		REQUIRE(interpolated == polynomial);
		time_contender("polynomial_interpolation (library)" + shape, interpolated, [&] { return atl::polynomial_interpolation(points, values); });
		time_contender("polynomial_interpolation by remainder tree" + shape, interpolated, [&] { return ex::polynomial_interpolation_by_remainder_tree(points, values); });
		if (is_quadratic_size)
			time_contender("polynomial_interpolation by Newton differences" + shape, interpolated, [&] { return ex::polynomial_interpolation_by_newton_differences(points, values); });

		atl::polynomial_point_tree<Coefficient> const tree(points);
		time_contender("polynomial_point_tree::evaluate (tree built once)" + shape, values, [&] { return tree.evaluate(polynomial); });
		time_contender("polynomial_point_tree::interpolate (tree built once)" + shape, interpolated, [&] { return tree.interpolate(values); });

		auto const geometric = atl::polynomial_geometric_evaluation(polynomial, first, ratio, length);
		Vector geometric_points(length);
		Coefficient point = first;
		for (auto &entry : geometric_points) {
			entry = point;
			point *= ratio;
		}
		REQUIRE(atl::polynomial_evaluation(polynomial, geometric_points) == geometric);
		time_contender("polynomial_geometric_evaluation (library)" + shape, geometric, [&] { return atl::polynomial_geometric_evaluation(polynomial, first, ratio, length); });
		if (is_quadratic_size)
			time_contender("polynomial_geometric_evaluation by Horner" + shape, geometric, [&] { return ex::polynomial_geometric_evaluation_by_horner(polynomial, first, ratio, length); });

		Vector const &samples = values;
		auto const sample_shift = atl::sample_point_shift(samples, shift, length);
		Vector shifted_points(length);
		for (std::size_t i = 0; i < length; ++i) shifted_points[i] = shift + Coefficient(i);
		REQUIRE(atl::polynomial_evaluation(polynomial, shifted_points) == sample_shift);
		time_contender("sample_point_shift (library)" + shape, sample_shift, [&] { return atl::sample_point_shift(samples, shift, length); });
		time_contender("sample_point_shift by falling factorials" + shape, sample_shift, [&] { return ex::sample_point_shift_by_falling_factorials(samples, shift, length); });
		if (is_quadratic_size)
			time_contender("sample_point_shift by direct barycentric" + shape, sample_shift, [&] { return ex::sample_point_shift_by_direct_barycentric(samples, shift, length); });
	}
}

TEST_CASE("polynomial_points: library versus contenders (race pending)", "[polynomial_points]") {
	for (std::size_t length : {std::size_t{1} << 10, quadratic_limit, std::size_t{1} << 16, std::size_t{1} << 17, std::size_t{500000}})
		race<atl::mod_int32<998244353>>(length);
	race<atl::mod_int32<1000000007>>(std::size_t{1} << 16);
}
