/**
 * @file
 * @brief Polynomial gcd family: the library functions of `<atl/nt/polynomial_gcd.hpp>` against their contenders.
 *
 * Raced, one operation at a time, each with the one thing the contender does differently:
 * - `polynomial_gcd` and `polynomial_extended_gcd`: library = recursive half-gcd with ordinary matrix products and a Euclidean
 *   leaf size of 64; by fused matrix products = the same recursion with each matrix entry transformed once and the two products
 *   of an output entry summed before its inverse transform; by fast-division Euclid = plain Euclid with fast polynomial
 *   division, no half-gcd; by schoolbook Euclid = plain Euclid with schoolbook division; by leaf cutoff plain products and by
 *   leaf cutoff fused products = the half-gcd with the leaf size swept over 32, 48, 64, 96, 128, 192 and 256, with ordinary
 *   or with fused matrix products.
 * - `polynomial_inverse_mod`: the library function only (one cofactor accumulated).
 * - `pade_approximant`: library = stopped half-gcd; by quadratic Euclid = stopped Euclid with schoolbook division.
 * - Shortest linear recurrence: `berlekamp_massey` (library, quadratic) against `berlekamp_massey_by_half_gcd` (half-gcd on the
 *   reversed sequence).
 *
 * Every contender is checked against the library result, or against the Bezout and approximation identities, before any timing.
 * Inputs of length n are random, coprime or with a planted common factor of degree n / 3: n = 2^10, 2^12, 2^14, 2^16, 2^17 and
 * 500000 modulo 998244353, and 2^16 modulo 1000000007; the quadratic contenders run only up to 2^12. The recurrence race uses
 * random sequences of length 2^10, 2^14 and 2^16 (the library `berlekamp_massey` up to 2^14); the leaf-size sweep uses random
 * inputs of length 256, 1024 and 4096. Each sample is one call, including allocation; the loop measures latency.
 *
 * Result: not measured yet (race pending). The page docs/polynomial-gcd.md has the table once a record exists.
 */
#include <atl/experimental/nt/polynomial_gcd_variants.hpp>
#include <atl/nt/detail/polynomial_gcd.hpp>
#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/linear_recurrence.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/polynomial_gcd.hpp>
#include <atl/nt/power_series.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {
	namespace ex = atl::experimental;

	/** @brief The quadratic contenders are never run above this length. */
	constexpr std::size_t quadratic_limit = std::size_t{1} << 12;

	/** @brief Check the Bezout identity with independent product identities. */
	template<atl::static_prime_mod_int Coefficient>
	void certify(std::vector<Coefficient> const &first, std::vector<Coefficient> const &second, atl::polynomial_extended_gcd_result<Coefficient> const &result) {
		auto identity = atl::detail::gcd_product<atl::default_ntt_engine, Coefficient>(first, result.first_cofactor);
		auto const other = atl::detail::gcd_product<atl::default_ntt_engine, Coefficient>(second, result.second_cofactor);
		atl::detail::gcd_accumulate<Coefficient>(identity, other);
		REQUIRE(identity == result.gcd);
	}

	/** @brief Check a contender against the expected result, then time it under the label. */
	template<typename Result, typename Contender>
		requires std::is_invocable_r_v<Result, Contender &>
	void time_contender(std::string const &label, Result const &expected, Contender contender) {
		REQUIRE(contender() == expected);
		BENCHMARK(std::string(label)) { return contender(); };
	}

	/** @brief Certify the Bezout identity and the gcd of an extended-gcd contender, then time it under the label. */
	template<atl::static_prime_mod_int Coefficient, typename Contender>
		requires std::is_invocable_r_v<atl::polynomial_extended_gcd_result<Coefficient>, Contender &>
	void time_extended_gcd(std::string const &label, std::vector<Coefficient> const &first, std::vector<Coefficient> const &second,
						   std::vector<Coefficient> const &expected_gcd, Contender contender) {
		auto const result = contender();
		REQUIRE(result.gcd == expected_gcd);
		certify(first, second, result);
		BENCHMARK(std::string(label)) { return contender(); };
	}

	/** @brief Check a Pade contender against the library approximant, then time it under the label. */
	template<atl::static_prime_mod_int Coefficient, typename Contender>
		requires std::is_invocable_r_v<std::optional<atl::rational_function<Coefficient>>, Contender &>
	void time_pade(std::string const &label, std::optional<atl::rational_function<Coefficient>> const &expected, Contender contender) {
		auto const result = contender();
		REQUIRE(result.has_value() == expected.has_value());
		if (expected) {
			REQUIRE(result->numerator == expected->numerator);
			REQUIRE(result->denominator == expected->denominator);
		}
		BENCHMARK(std::string(label)) { return contender(); };
	}

	/** @brief Every gcd-family function on random coprime inputs, or inputs with a planted common factor; race pending. */
	template<atl::static_prime_mod_int Coefficient>
	void race(std::size_t length, bool has_planted_gcd) {
		using Vector = std::vector<Coefficient>;
		std::mt19937_64 random(7102);
		std::size_t const factor_degree = has_planted_gcd ? length / 3 : 0;
		Vector first(length - factor_degree), second(length - factor_degree), factor(factor_degree + 1);
		do {
			for (auto &value : first) value = Coefficient(random());
			for (auto &value : second) value = Coefficient(random());
			first.back() = second.back() = Coefficient(1);
		} while (atl::polynomial_gcd(first, second).size() != 1);
		for (auto &value : factor) value = Coefficient(random());
		factor.back() = Coefficient(1);
		first = atl::detail::gcd_product<atl::default_ntt_engine, Coefficient>(first, factor);
		second = atl::detail::gcd_product<atl::default_ntt_engine, Coefficient>(second, factor);
		auto const shape = " | p=" + std::to_string(Coefficient::mod()) + " n=" + std::to_string(length) + (has_planted_gcd ? " planted" : " coprime");
		bool const is_quadratic_size = length <= quadratic_limit;

		auto const expected = atl::polynomial_gcd(first, second);
		REQUIRE(expected == factor);
		time_contender("polynomial_gcd (library)" + shape, expected, [&] { return atl::polynomial_gcd(first, second); });
		time_contender("polynomial_gcd by fused matrix products" + shape, expected, [&] { return ex::polynomial_gcd_by_fused_matrix_products(first, second); });
		if (is_quadratic_size) {
			time_contender("polynomial_gcd by fast-division Euclid" + shape, expected, [&] { return ex::polynomial_gcd_by_fast_division_euclid(first, second); });
			time_contender("polynomial_gcd by schoolbook Euclid" + shape, expected, [&] { return ex::polynomial_gcd_by_schoolbook_euclid(first, second); });
		}

		time_extended_gcd<Coefficient>("polynomial_extended_gcd (library)" + shape, first, second, expected, [&] { return atl::polynomial_extended_gcd(first, second); });
		time_extended_gcd<Coefficient>("polynomial_extended_gcd by fused matrix products" + shape, first, second, expected, [&] { return ex::polynomial_extended_gcd_by_fused_matrix_products(first, second); });
		if (is_quadratic_size) {
			time_extended_gcd<Coefficient>("polynomial_extended_gcd by fast-division Euclid" + shape, first, second, expected, [&] { return ex::polynomial_extended_gcd_by_fast_division_euclid(first, second); });
			time_extended_gcd<Coefficient>("polynomial_extended_gcd by schoolbook Euclid" + shape, first, second, expected, [&] { return ex::polynomial_extended_gcd_by_schoolbook_euclid(first, second); });
		}

		auto const inverse = atl::polynomial_inverse_mod(first, second);
		REQUIRE(inverse.has_value() == !has_planted_gcd);
		if (inverse) REQUIRE(atl::polynomial_division(atl::detail::gcd_product<atl::default_ntt_engine, Coefficient>(first, *inverse), second).remainder == Vector{1});
		time_contender("polynomial_inverse_mod (library)" + shape, inverse, [&] { return atl::polynomial_inverse_mod(first, second); });

		std::size_t const numerator_degree = (length - 1) / 2, denominator_degree = length - 1 - numerator_degree;
		auto const pade = atl::pade_approximant(first, numerator_degree, denominator_degree);
		REQUIRE(pade.has_value());
		auto cross = atl::detail::series_product<atl::default_ntt_engine, Coefficient>(first, pade->denominator, length);
		auto numerator = pade->numerator;
		numerator.resize(length);
		REQUIRE(cross == numerator);
		time_pade<Coefficient>("pade_approximant (library)" + shape, pade, [&] { return atl::pade_approximant(first, numerator_degree, denominator_degree); });
		if (is_quadratic_size)
			time_pade<Coefficient>("pade_approximant by quadratic Euclid" + shape, pade, [&] { return ex::pade_approximant_by_quadratic_euclid(first, numerator_degree, denominator_degree); });
	}

	/** @brief The half-gcd leaf size swept, with plain and with fused matrix products; race pending. */
	void sweep() {
		using C = atl::mod_int32<998244353>;
		using V = std::vector<C>;
		std::mt19937_64 random(23);
		for (std::size_t length : {256, 1024, 4096}) {
			V first(length), second(length);
			for (auto &value : first) value = C(random());
			for (auto &value : second) value = C(random());
			auto const expected = atl::polynomial_gcd(first, second);
			for (std::size_t cutoff : {32, 48, 64, 96, 128, 192, 256}) {
				auto const shape = " | n=" + std::to_string(length) + " cutoff=" + std::to_string(cutoff);
				time_contender("polynomial_gcd by leaf cutoff plain products" + shape, expected, [&] { return ex::polynomial_gcd_by_leaf_cutoff_plain_products(first, second, cutoff); });
				time_contender("polynomial_gcd by leaf cutoff fused products" + shape, expected, [&] { return ex::polynomial_gcd_by_leaf_cutoff_fused_products(first, second, cutoff); });
				time_extended_gcd<C>("polynomial_extended_gcd by leaf cutoff plain products" + shape, first, second, expected, [&] { return ex::polynomial_extended_gcd_by_leaf_cutoff_plain_products(first, second, cutoff); });
				time_extended_gcd<C>("polynomial_extended_gcd by leaf cutoff fused products" + shape, first, second, expected, [&] { return ex::polynomial_extended_gcd_by_leaf_cutoff_fused_products(first, second, cutoff); });
			}
		}
	}

	/** @brief Shortest recurrence by half-gcd against the library quadratic `berlekamp_massey` where affordable; race pending. */
	void recurrence_race(std::size_t length) {
		using C = atl::mod_int32<998244353>;
		using V = std::vector<C>;
		std::mt19937_64 random(29);
		V sequence(length);
		for (auto &value : sequence) value = C(random());
		auto const fast = ex::berlekamp_massey_by_half_gcd(sequence);
		V characteristic(fast.size() + 1);
		characteristic.back() = C(1);
		for (std::size_t i = 0; i < fast.size(); ++i) characteristic[fast.size() - 1 - i] = -fast[i];
		auto reversed = sequence;
		std::ranges::reverse(reversed);
		auto const residual = atl::detail::series_product<atl::default_ntt_engine, C>(reversed, characteristic, length);
		for (std::size_t i = fast.size(); i < length; ++i) REQUIRE(residual[i] == C(0));
		auto const shape = " | n=" + std::to_string(length);
		if (length <= (std::size_t{1} << 14)) {
			REQUIRE(fast == atl::berlekamp_massey(sequence));
			time_contender("berlekamp_massey (library)" + shape, fast, [&] { return atl::berlekamp_massey(sequence); });
		}
		time_contender("berlekamp_massey by half-gcd" + shape, fast, [&] { return ex::berlekamp_massey_by_half_gcd(sequence); });
	}
}

TEST_CASE("polynomial_gcd: library versus contenders (race pending)", "[polynomial_gcd]") {
	for (std::size_t length : {std::size_t{1} << 10, std::size_t{1} << 12, std::size_t{1} << 14, std::size_t{1} << 16, std::size_t{1} << 17, std::size_t{500000}})
		for (bool has_planted_gcd : {false, true}) race<atl::mod_int32<998244353>>(length, has_planted_gcd);
	for (bool has_planted_gcd : {false, true}) race<atl::mod_int32<1000000007>>(std::size_t{1} << 16, has_planted_gcd);
	for (std::size_t length : {std::size_t{1} << 10, std::size_t{1} << 14, std::size_t{1} << 16}) recurrence_race(length);
	sweep();
}
