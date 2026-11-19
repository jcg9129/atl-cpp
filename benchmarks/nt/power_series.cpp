/**
 * @file
 * @brief Race of the formal power series functions of `atl/nt/power_series.hpp` against their contenders.
 *
 * What is raced, and the one thing each contender does differently:
 * - `series_inverse` (library: high-half Newton step that reuses one transform) against
 *   `series_inverse_by_recurrence` (quadratic recurrence only), `series_inverse_by_plain_newton` (two ordinary
 *   products per step) and `series_inverse_by_newton_driver` (same step, doubling loop from the shared driver).
 * - `series_division` (library: inverse to half precision and a fused last step) against
 *   `series_division_by_inverse_product` (full inverse, then one product).
 * - `series_logarithm` (library: fused division of the derivative by the series) against
 *   `series_logarithm_by_inverse_product` (derivative, full inverse, product, integral).
 * - `series_exponential` (library: Newton step carrying the reciprocal along) against
 *   `series_exponential_by_recurrence` (quadratic recurrence only), `series_exponential_by_plain_newton`
 *   (rebuilds the reciprocal in every step) and `series_exponential_by_newton_driver` (same step, shared driver loop).
 * - `series_power` (library: valuation, then `exp(exponent * log f)`) against `series_power_by_binary_powering`
 *   (truncated binary powering) and `series_power_sparse` (differential equation, on dense input).
 * - `series_square_root` (library: coupled Newton on the shared driver) against `series_square_root_by_plain_newton`
 *   (public division in every step), `series_square_root_by_logarithm` (`exp(log(f) / 2)`) and
 *   `series_square_root_sparse` (differential equation, on dense input).
 * - `polynomial_division` (library: reversed inverse and a low product) against `polynomial_division_by_schoolbook`.
 * - Each dense entry point against its sparse entry point (`series_inverse_sparse`, ...) on inputs with s nonconstant
 *   nonzero terms.
 *
 * What the loop measures: the throughput of one whole call, per Catch2 sample, on fixed seeded dense inputs
 * (or sparse inputs for the sparse race), after REQUIRE-checking that every contender agrees with the library
 * function. Shapes: lengths 2^10, 2^12, 2^16, 2^20 and 500000 for the prime 998244353 and 2^16 for 1000000007;
 * powers with exponents 2, 3, 32 and 10^18; polynomial divisions with (degree f, degree g) = (2n, n), (n, n/2)
 * and (n, 64) for n = 2^10, 2^16 and 500000; sparse shapes s = 2, 8 and 64 at lengths 2^16 and 500000. The
 * quadratic contenders run only up to `quadratic_limit` = 2^12, because beyond it they cannot finish in race time.
 *
 * Result: not measured yet; the race is pending, so every library choice and the recurrence/Newton cutoff are
 * provisional.
 */
#include <atl/experimental/nt/power_series_variants.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/power_series.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
	/** @brief Largest length at which the O(n^2) recurrences and sparse entry points on dense input are timed. */
	constexpr std::size_t quadratic_limit = std::size_t{1} << 12;

	/** @brief Time one contender: each Catch2 sample runs the whole call. */
	template<std::invocable Contender>
	void time_contender(std::string const &name, std::string const &shape, Contender const &contender) {
		BENCHMARK(name + " | " + shape) { return contender(); };
	}

	/** @brief Inverse, division, logarithm and exponential contenders on a fixed seeded dense input. */
	template<atl::static_prime_mod_int Coefficient>
	void race_newton(std::size_t length) {
		std::mt19937_64 random(3);
		std::vector<Coefficient> unit(length), zero(length), numerator(length);
		for (auto &value : unit) value = Coefficient(random());
		for (auto &value : zero) value = Coefficient(random());
		for (auto &value : numerator) value = Coefficient(random());
		unit[0] = Coefficient(1);
		zero[0] = Coefficient(0);
		auto const shape = "p=" + std::to_string(Coefficient::mod()) + " n=" + std::to_string(length);
		bool const is_quadratic_size = length <= quadratic_limit;
		auto const inverse = atl::series_inverse(unit, length);
		REQUIRE(atl::experimental::series_inverse_by_plain_newton(unit, length) == inverse);
		REQUIRE(atl::experimental::series_inverse_by_newton_driver(unit, length) == inverse);
		auto const quotient = atl::series_division(numerator, unit, length);
		REQUIRE(atl::experimental::series_division_by_inverse_product(numerator, unit, length) == quotient);
		auto const logarithm = atl::series_logarithm(unit, length);
		REQUIRE(atl::experimental::series_logarithm_by_inverse_product(unit, length) == logarithm);
		auto const exponential = atl::series_exponential(zero, length);
		REQUIRE(atl::experimental::series_exponential_by_plain_newton(zero, length) == exponential);
		REQUIRE(atl::experimental::series_exponential_by_newton_driver(zero, length) == exponential);
		if (is_quadratic_size) {
			REQUIRE(atl::experimental::series_inverse_by_recurrence(unit, length) == inverse);
			REQUIRE(atl::experimental::series_exponential_by_recurrence(zero, length) == exponential);
			time_contender("series_inverse_by_recurrence", shape, [&] { return atl::experimental::series_inverse_by_recurrence(unit, length); });
		}
		time_contender("series_inverse_by_plain_newton", shape, [&] { return atl::experimental::series_inverse_by_plain_newton(unit, length); });
		time_contender("series_inverse_by_newton_driver", shape, [&] { return atl::experimental::series_inverse_by_newton_driver(unit, length); });
		time_contender("series_inverse (library)", shape, [&] { return atl::series_inverse(unit, length); });
		time_contender("series_division_by_inverse_product", shape, [&] { return atl::experimental::series_division_by_inverse_product(numerator, unit, length); });
		time_contender("series_division (library)", shape, [&] { return atl::series_division(numerator, unit, length); });
		time_contender("series_logarithm_by_inverse_product", shape, [&] { return atl::experimental::series_logarithm_by_inverse_product(unit, length); });
		time_contender("series_logarithm (library)", shape, [&] { return atl::series_logarithm(unit, length); });
		if (is_quadratic_size)
			time_contender("series_exponential_by_recurrence", shape, [&] { return atl::experimental::series_exponential_by_recurrence(zero, length); });
		time_contender("series_exponential_by_plain_newton", shape, [&] { return atl::experimental::series_exponential_by_plain_newton(zero, length); });
		time_contender("series_exponential_by_newton_driver", shape, [&] { return atl::experimental::series_exponential_by_newton_driver(zero, length); });
		time_contender("series_exponential (library)", shape, [&] { return atl::series_exponential(zero, length); });
	}

	/** @brief Power and square root contenders on identical seeded dense inputs, with agreement checked before timing. */
	template<atl::static_prime_mod_int Coefficient>
	void race_power_root(std::size_t length) {
		std::mt19937_64 random(7);
		std::vector<Coefficient> unit(length);
		for (auto &value : unit) value = Coefficient(random());
		unit[0] = Coefficient(1);
		auto const shape = "p=" + std::to_string(Coefficient::mod()) + " n=" + std::to_string(length);
		for (std::uint64_t exponent : {2ULL, 3ULL, 32ULL, 1000000000000000000ULL}) {
			auto const power = atl::series_power(unit, exponent, length);
			REQUIRE(atl::experimental::series_power_by_binary_powering(unit, exponent, length) == power);
			if (length <= quadratic_limit) REQUIRE(atl::series_power_sparse(unit, exponent, length) == power);
			auto const power_shape = shape + " exponent=" + std::to_string(exponent);
			time_contender("series_power (library)", power_shape, [&] { return atl::series_power(unit, exponent, length); });
			time_contender("series_power_by_binary_powering", power_shape, [&] { return atl::experimental::series_power_by_binary_powering(unit, exponent, length); });
			if (length <= quadratic_limit)
				time_contender("series_power_sparse on dense input", power_shape, [&] { return atl::series_power_sparse(unit, exponent, length); });
		}
		auto const root = atl::series_square_root(unit, length);
		REQUIRE(atl::experimental::series_square_root_by_plain_newton(unit, length) == root);
		REQUIRE(atl::experimental::series_square_root_by_logarithm(unit, length) == root);
		if (length <= quadratic_limit) REQUIRE(atl::series_square_root_sparse(unit, length) == root);
		time_contender("series_square_root (library)", shape, [&] { return atl::series_square_root(unit, length); });
		time_contender("series_square_root_by_plain_newton", shape, [&] { return atl::experimental::series_square_root_by_plain_newton(unit, length); });
		time_contender("series_square_root_by_logarithm", shape, [&] { return atl::experimental::series_square_root_by_logarithm(unit, length); });
		if (length <= quadratic_limit)
			time_contender("series_square_root_sparse on dense input", shape, [&] { return atl::series_square_root_sparse(unit, length); });
	}

	/** @brief Sparse entry points against the dense ones; the denominator's sparsity determines the division work. */
	template<atl::static_prime_mod_int Coefficient>
	void race_sparse(std::size_t length, std::size_t terms) {
		std::mt19937_64 random(8);
		std::vector<Coefficient> unit(length), zero(length), numerator(length);
		unit[0] = Coefficient(1);
		for (std::size_t i = 1; i <= terms; ++i) {
			auto const degree = i * (length - 1) / terms;
			unit[degree] = Coefficient(random());
			zero[degree] = unit[degree];
		}
		for (auto &value : numerator) value = Coefficient(random());
		auto const shape = "p=" + std::to_string(Coefficient::mod()) + " n=" + std::to_string(length) + " s=" + std::to_string(terms);
		REQUIRE(atl::series_inverse_sparse(unit, length) == atl::series_inverse(unit, length));
		REQUIRE(atl::series_division_sparse(numerator, unit, length) == atl::series_division(numerator, unit, length));
		REQUIRE(atl::series_logarithm_sparse(unit, length) == atl::series_logarithm(unit, length));
		REQUIRE(atl::series_exponential_sparse(zero, length) == atl::series_exponential(zero, length));
		REQUIRE(atl::series_square_root_sparse(unit, length) == atl::series_square_root(unit, length));
		for (std::uint64_t exponent : {2ULL, 3ULL, 32ULL, 1000000000000000000ULL}) {
			REQUIRE(atl::series_power_sparse(unit, exponent, length) == atl::series_power(unit, exponent, length));
			auto const power_shape = shape + " exponent=" + std::to_string(exponent);
			time_contender("series_power (library) on sparse input", power_shape, [&] { return atl::series_power(unit, exponent, length); });
			time_contender("series_power_sparse", power_shape, [&] { return atl::series_power_sparse(unit, exponent, length); });
		}
		time_contender("series_inverse (library) on sparse input", shape, [&] { return atl::series_inverse(unit, length); });
		time_contender("series_inverse_sparse", shape, [&] { return atl::series_inverse_sparse(unit, length); });
		time_contender("series_division (library) on sparse input", shape, [&] { return atl::series_division(numerator, unit, length); });
		time_contender("series_division_sparse", shape, [&] { return atl::series_division_sparse(numerator, unit, length); });
		time_contender("series_logarithm (library) on sparse input", shape, [&] { return atl::series_logarithm(unit, length); });
		time_contender("series_logarithm_sparse", shape, [&] { return atl::series_logarithm_sparse(unit, length); });
		time_contender("series_exponential (library) on sparse input", shape, [&] { return atl::series_exponential(zero, length); });
		time_contender("series_exponential_sparse", shape, [&] { return atl::series_exponential_sparse(zero, length); });
		time_contender("series_square_root (library) on sparse input", shape, [&] { return atl::series_square_root(unit, length); });
		time_contender("series_square_root_sparse", shape, [&] { return atl::series_square_root_sparse(unit, length); });
	}

	/** @brief Polynomial division shapes, certified by reconstruction before any timing. */
	template<atl::static_prime_mod_int Coefficient>
	void race_polynomial(std::size_t n) {
		std::mt19937_64 random(9);
		for (auto const &[numerator_degree, denominator_degree] : {std::pair{2 * n, n}, std::pair{n, n / 2}, std::pair{n, std::size_t{64}}}) {
			std::vector<Coefficient> numerator(numerator_degree + 1), denominator(denominator_degree + 1);
			for (auto &value : numerator) value = Coefficient(random());
			for (auto &value : denominator) value = Coefficient(random());
			numerator.back() = denominator.back() = Coefficient(1);
			auto const division = atl::polynomial_division(numerator, denominator);
			auto reconstruction = atl::detail::series_product<atl::default_ntt_engine, Coefficient>(division.quotient, denominator, numerator.size());
			for (std::size_t i = 0; i < division.remainder.size(); ++i) reconstruction[i] += division.remainder[i];
			REQUIRE(reconstruction == numerator);
			REQUIRE(division.remainder.size() < denominator.size());
			auto const shape = "p=" + std::to_string(Coefficient::mod()) + " deg(f)=" + std::to_string(numerator_degree) + " deg(g)=" + std::to_string(denominator_degree);
			if (numerator_degree <= quadratic_limit) {
				auto const schoolbook = atl::experimental::polynomial_division_by_schoolbook(numerator, denominator);
				REQUIRE(schoolbook.quotient == division.quotient);
				REQUIRE(schoolbook.remainder == division.remainder);
				time_contender("polynomial_division_by_schoolbook", shape, [&] { return atl::experimental::polynomial_division_by_schoolbook(numerator, denominator); });
			}
			time_contender("polynomial_division (library)", shape, [&] { return atl::polynomial_division(numerator, denominator); });
		}
	}
}

TEST_CASE("power_series: inverse, division, logarithm and exponential contenders (race pending)", "[power_series]") {
	for (std::size_t length : {std::size_t{1} << 10, std::size_t{1} << 12, std::size_t{1} << 16, std::size_t{1} << 20, std::size_t{500000}})
		race_newton<atl::mod_int32<998244353>>(length);
	race_newton<atl::mod_int32<1000000007>>(std::size_t{1} << 16);
}

TEST_CASE("power_series: power and square root contenders (race pending)", "[power_series]") {
	for (std::size_t length : {std::size_t{1} << 10, std::size_t{1} << 12, std::size_t{1} << 16, std::size_t{1} << 20, std::size_t{500000}})
		race_power_root<atl::mod_int32<998244353>>(length);
	race_power_root<atl::mod_int32<1000000007>>(std::size_t{1} << 16);
}

TEST_CASE("power_series: sparse entry points and polynomial division (race pending)", "[power_series]") {
	for (std::size_t length : {std::size_t{1} << 16, std::size_t{500000}})
		for (std::size_t terms : {2, 8, 64}) race_sparse<atl::mod_int32<998244353>>(length, terms);
	for (std::size_t length : {std::size_t{1} << 10, std::size_t{1} << 16, std::size_t{500000}}) {
		race_polynomial<atl::mod_int32<998244353>>(length);
		race_polynomial<atl::mod_int32<1000000007>>(length);
	}
}
