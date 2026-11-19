/**
 * @file
 * @brief Composition, power projection and compositional inverse: the library functions of `<atl/nt/series_composition.hpp>`
 *        against their contenders.
 *
 * Raced, one operation at a time, each with the one thing the contender does differently:
 * - `series_composition`: library = transpose of the bivariate power projection (Kinoshita-Li) with Horner for tiny shapes;
 *   by Brent-Kung = baby and giant powers with a classical matrix product; by fast Horner = one truncated fast product per
 *   outer coefficient; by schoolbook Horner = the same with schoolbook products; by cutoffs = the library with its two Horner
 *   crossovers as arguments.
 * - `series_power_projection`: library = bivariate Bostan-Mori with the paired-spectra schedule, repeated products for counts up to
 *   8; by padded Bostan-Mori = plain padded products; by Bostan-Mori = the paired-spectra schedule at every count; by repeated
 *   products = multiplication by the series and one dot product per power.
 * - `series_compositional_inverse`: library = Lagrange inversion through one power projection; by composition Newton = one
 *   composition per Newton step; by baby giant Lagrange = Johansson's baby and giant powers; by coefficient recurrence = the
 *   direct cubic recurrence.
 *
 * Every contender is checked against the library result before any timing; the composition and the projection are also checked
 * against each other through the transposition identity sum(weights[i] * composed[i]) = sum(outer[i] * projected[i]).
 * Lengths n (outer length = precision = number of weights): 32, 64, 2^10, 2^12, 2^13, 2^14, 2^16 and 250000 modulo 998244353,
 * and 2^13 modulo 1000000007. Brent-Kung and baby giant Lagrange run only up to 2^14, the quadratic contenders (fast Horner,
 * repeated products) up to 2^12, the cubic ones (schoolbook Horner, coefficient recurrence) up to 64. The cutoff sweep times the
 * contenders at sizes cutoff-1, cutoff and cutoff+1 for cutoffs 4..64. Each sample is one call, including allocation; the loop
 * measures latency.
 *
 * Result: not measured yet (race pending). The page docs/series-composition.md has the table once a record exists.
 */
#include <atl/experimental/nt/series_composition_variants.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/nt/series_composition.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {
	namespace ex = atl::experimental;

	/** @brief The quadratic contenders are never run above this length. */
	constexpr std::size_t quadratic_limit = std::size_t{1} << 12;
	/** @brief The classical-matrix contenders are never run above this length. */
	constexpr std::size_t matrix_limit = std::size_t{1} << 14;
	/** @brief The cubic contenders (general recurrence, dense schoolbook Horner) are never run above this length. */
	constexpr std::size_t cubic_limit = 64;

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
		std::mt19937_64 random(7002);
		Vector outer(length), inner(length), weights(length);
		for (auto &value : outer) value = Coefficient(random());
		for (auto &value : inner) value = Coefficient(random());
		for (auto &value : weights) value = Coefficient(random());
		inner[0] = Coefficient(0);
		inner[1] = Coefficient(3);
		auto const shape = " | p=" + std::to_string(Coefficient::mod()) + " n=" + std::to_string(length);

		auto const composed = atl::series_composition(outer, inner, length);
		auto const projected = atl::series_power_projection(weights, inner, length);
		auto const inverse = atl::series_compositional_inverse(inner, length);
		Coefficient dot_composed(0), dot_projected(0);
		for (std::size_t i = 0; i < length; ++i) {
			dot_composed += composed[i] * weights[i];
			dot_projected += projected[i] * outer[i];
		}
		REQUIRE(dot_composed == dot_projected);

		time_contender("series_composition (library)" + shape, composed, [&] { return atl::series_composition(outer, inner, length); });
		if (length <= matrix_limit)
			time_contender("series_composition by Brent-Kung" + shape, composed, [&] { return ex::series_composition_by_brent_kung(outer, inner, length); });
		if (length <= quadratic_limit)
			time_contender("series_composition by fast Horner" + shape, composed, [&] { return ex::series_composition_by_fast_horner(outer, inner, length); });
		if (length <= cubic_limit)
			time_contender("series_composition by schoolbook Horner" + shape, composed, [&] { return ex::series_composition_by_schoolbook_horner(outer, inner, length); });

		time_contender("series_power_projection (library)" + shape, projected, [&] { return atl::series_power_projection(weights, inner, length); });
		time_contender("series_power_projection by padded Bostan-Mori" + shape, projected, [&] { return ex::series_power_projection_by_padded_bostan_mori(weights, inner, length); });
		time_contender("series_power_projection by Bostan-Mori" + shape, projected, [&] { return ex::series_power_projection_by_bostan_mori(weights, inner, length); });
		if (length <= quadratic_limit)
			time_contender("series_power_projection by repeated products" + shape, projected, [&] { return ex::series_power_projection_by_repeated_products(weights, inner, length); });

		time_contender("series_compositional_inverse (library)" + shape, inverse, [&] { return atl::series_compositional_inverse(inner, length); });
		time_contender("series_compositional_inverse by composition Newton" + shape, inverse, [&] { return ex::series_compositional_inverse_by_composition_newton(inner, length); });
		if (length <= matrix_limit)
			time_contender("series_compositional_inverse by baby giant Lagrange" + shape, inverse, [&] { return ex::series_compositional_inverse_by_baby_giant_lagrange(inner, length); });
		if (length <= cubic_limit)
			time_contender("series_compositional_inverse by coefficient recurrence" + shape, inverse, [&] { return ex::series_compositional_inverse_by_coefficient_recurrence(inner, length); });
	}

	void sweep_cutoffs() {
		using Field = atl::mod_int32<998244353>;
		using Vector = std::vector<Field>;
		for (std::size_t cutoff : {4, 8, 12, 16, 24, 32, 48, 64}) {
			for (std::size_t n : {cutoff - 1, cutoff, cutoff + 1}) {
				Vector inner(n, Field(2)), outer(n, Field(3)), weights(n, Field(5));
				inner[0] = Field(0);
				auto const composed = atl::series_composition(outer, inner, n);
				auto const projected = atl::series_power_projection(weights, inner, n);
				auto const shape = " | cutoff=" + std::to_string(cutoff) + " n=" + std::to_string(n);
				time_contender("series_composition by cutoffs" + shape, composed, [&] { return ex::series_composition_by_cutoffs(outer, inner, n, cutoff, cutoff); });
				time_contender("series_power_projection by repeated products" + shape, projected, [&] { return ex::series_power_projection_by_repeated_products(weights, inner, n); });
				time_contender("series_power_projection by Bostan-Mori" + shape, projected, [&] { return ex::series_power_projection_by_bostan_mori(weights, inner, n); });
			}
		}
	}
}

TEST_CASE("series_composition: library versus contenders (race pending)", "[series_composition]") {
	for (std::size_t length : {std::size_t{32}, std::size_t{64}, std::size_t{1} << 10, quadratic_limit, std::size_t{1} << 13,
							   std::size_t{1} << 14, std::size_t{1} << 16, std::size_t{250000}})
		race<atl::mod_int32<998244353>>(length);
	race<atl::mod_int32<1000000007>>(std::size_t{1} << 13);
	sweep_cutoffs();
}
