#include <atl/experimental/nt/polynomial_gcd_variants.hpp>
#include <atl/nt/linear_recurrence.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/polynomial_gcd.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

namespace gcd_tests {
	/** @brief Independent zero representation. */
	template<atl::static_prime_mod_int Coefficient>
	void trim(std::vector<Coefficient> &value) {
		while (!value.empty() && value.back() == Coefficient(0)) value.pop_back();
	}

	/** @brief Independent schoolbook polynomial multiplication. */
	template<atl::static_prime_mod_int Coefficient>
	auto product(std::vector<Coefficient> const &left, std::vector<Coefficient> const &right) -> std::vector<Coefficient> {
		if (left.empty() || right.empty()) return {};
		std::vector<Coefficient> result(left.size() + right.size() - 1);
		for (std::size_t i = 0; i < left.size(); ++i)
			for (std::size_t j = 0; j < right.size(); ++j) result[i + j] += left[i] * right[j];
		trim(result);
		return result;
	}

	/** @brief Independent descending schoolbook remainder, sharing no contender code. */
	template<atl::static_prime_mod_int Coefficient>
	auto remainder(std::vector<Coefficient> value, std::vector<Coefficient> divisor) -> std::vector<Coefficient> {
		trim(value);
		trim(divisor);
		REQUIRE(!divisor.empty());
		while (value.size() >= divisor.size()) {
			std::size_t const offset = value.size() - divisor.size();
			Coefficient const scale = value.back() / divisor.back();
			for (std::size_t j = 0; j < divisor.size(); ++j) value[offset + j] -= scale * divisor[j];
			trim(value);
		}
		return value;
	}

	/** @brief Independent monic Euclid oracle. */
	template<atl::static_prime_mod_int Coefficient>
	auto oracle(std::vector<Coefficient> first, std::vector<Coefficient> second) -> std::vector<Coefficient> {
		trim(first);
		trim(second);
		while (!second.empty()) {
			auto next = remainder(first, second);
			first = second;
			second = next;
		}
		if (!first.empty()) {
			Coefficient const leading = first.back();
			for (auto &value : first) value /= leading;
		}
		return first;
	}

	/** @brief Certify a normalized Bezout identity and the minimal degree bounds. */
	template<atl::static_prime_mod_int Coefficient>
	void bezout(std::vector<Coefficient> first, std::vector<Coefficient> second, atl::polynomial_extended_gcd_result<Coefficient> const &result) {
		auto left = product(first, result.first_cofactor), right = product(second, result.second_cofactor);
		left.resize(std::max(left.size(), right.size()));
		for (std::size_t i = 0; i < right.size(); ++i) left[i] += right[i];
		trim(left);
		trim(first);
		trim(second);
		REQUIRE(left == result.gcd);
		for (auto const *value : {&result.gcd, &result.first_cofactor, &result.second_cofactor})
			REQUIRE((value->empty() || value->back() != Coefficient(0)));
		if (!first.empty() && !second.empty() && !result.gcd.empty()) {
			if (second.size() > result.gcd.size()) REQUIRE(result.first_cofactor.size() <= second.size() - result.gcd.size());
			if (first.size() > result.gcd.size()) REQUIRE(result.second_cofactor.size() <= first.size() - result.gcd.size());
		}
	}

	/** @brief All gcd paths against an independent oracle, with planted factors and trailing zeros. */
	template<atl::static_prime_mod_int Coefficient>
	void family() {
		using Vector = std::vector<Coefficient>;
		using Gcd = Vector (*)(Vector const &, Vector const &);
		using Extended = atl::polynomial_extended_gcd_result<Coefficient> (*)(Vector const &, Vector const &);
		std::mt19937_64 random(7101);
		for (std::size_t length : {0, 1, 2, 5, 31, 65, 137}) {
			for (std::size_t trial = 0; trial < 8; ++trial) {
				std::size_t const second_length = length == 0 ? 0 : std::size_t(random() % (length + 1));
				std::size_t const factor_length = trial == 0 ? 0 : std::size_t(random() % (length + 1)) + 1;
				Vector first(length), second(second_length), factor(factor_length);
				for (auto *polynomial : {&first, &second, &factor})
					for (auto &value : *polynomial) value = Coefficient(random());
				if (trial != 0) {
					factor.back() = Coefficient(1);
					first = product(first, factor);
					second = product(second, factor);
				}
				auto const expected = oracle(first, second);
				first.resize(first.size() + trial % 3);
				second.resize(second.size() + trial % 4);
				auto const first_copy = first, second_copy = second;
				for (Gcd solver : {atl::polynomial_gcd<atl::default_ntt_engine, Vector, Vector>,
								   atl::experimental::polynomial_gcd_by_fused_matrix_products<atl::default_ntt_engine, Vector, Vector>,
								   atl::experimental::polynomial_gcd_by_fast_division_euclid<atl::default_ntt_engine, Vector, Vector>,
								   atl::experimental::polynomial_gcd_by_schoolbook_euclid<atl::default_ntt_engine, Vector, Vector>}) REQUIRE(solver(first, second) == expected);
				for (Extended solver : {atl::polynomial_extended_gcd<atl::default_ntt_engine, Vector, Vector>,
										atl::experimental::polynomial_extended_gcd_by_fused_matrix_products<atl::default_ntt_engine, Vector, Vector>,
										atl::experimental::polynomial_extended_gcd_by_fast_division_euclid<atl::default_ntt_engine, Vector, Vector>,
										atl::experimental::polynomial_extended_gcd_by_schoolbook_euclid<atl::default_ntt_engine, Vector, Vector>}) {
					auto const result = solver(first, second);
					REQUIRE(result.gcd == expected);
					bezout(first, second, result);
				}
				for (std::size_t cutoff : {1, 2, 32, 64, 128, 256}) {
					REQUIRE(atl::experimental::polynomial_gcd_by_leaf_cutoff_plain_products(first, second, cutoff) == expected);
					REQUIRE(atl::experimental::polynomial_gcd_by_leaf_cutoff_fused_products(first, second, cutoff) == expected);
					for (auto const &result : {atl::experimental::polynomial_extended_gcd_by_leaf_cutoff_plain_products(first, second, cutoff),
											   atl::experimental::polynomial_extended_gcd_by_leaf_cutoff_fused_products(first, second, cutoff)}) {
						REQUIRE(result.gcd == expected);
						bezout(first, second, result);
					}
				}
				REQUIRE(first == first_copy);
				REQUIRE(second == second_copy);
				trim(second);
				if (second.size() >= 2) {
					auto const inverse = atl::polynomial_inverse_mod(first, second);
					REQUIRE(inverse.has_value() == (expected.size() == 1));
					if (inverse) {
						REQUIRE(inverse->size() < second.size());
						REQUIRE(remainder(product(first, *inverse), second) == Vector{1});
					}
				}
			}
		}
		for (Vector const &value : {Vector{}, Vector{0}, Vector{3, 1, 2}}) {
			auto const left = atl::polynomial_extended_gcd(value, Vector{}), right = atl::polynomial_extended_gcd(Vector{}, value);
			bezout(value, Vector{}, left);
			bezout(Vector{}, value, right);
		}
	}

	/** @brief Certify the approximation through every required coefficient. */
	template<atl::static_prime_mod_int Coefficient>
	void approximation(std::vector<Coefficient> const &series, std::size_t numerator_degree, std::size_t denominator_degree,
					   atl::rational_function<Coefficient> const &result) {
		REQUIRE(result.numerator.size() <= numerator_degree + 1);
		REQUIRE(result.denominator.size() <= denominator_degree + 1);
		REQUIRE(!result.denominator.empty());
		REQUIRE(result.denominator[0] == Coefficient(1));
		REQUIRE(oracle(result.numerator, result.denominator) == std::vector<Coefficient>{1});
		auto cross = product(series, result.denominator);
		std::size_t const precision = numerator_degree + denominator_degree + 1;
		cross.resize(precision);
		auto numerator = result.numerator;
		numerator.resize(precision);
		REQUIRE(cross == numerator);
	}

	/** @brief Enumerate all denominators over a tiny field; independent existence certificate. */
	template<atl::static_prime_mod_int Coefficient>
	void tiny_pade() {
		using Vector = std::vector<Coefficient>;
		std::size_t const prime = Coefficient::mod();
		for (std::size_t numerator_degree = 0; numerator_degree <= 2; ++numerator_degree)
			for (std::size_t denominator_degree = 0; denominator_degree <= 2; ++denominator_degree) {
				std::size_t const precision = numerator_degree + denominator_degree + 1;
				std::size_t possibilities = 1, denominators = 1;
				for (std::size_t i = 0; i < precision; ++i) possibilities *= prime;
				for (std::size_t i = 0; i < denominator_degree; ++i) denominators *= prime;
				for (std::size_t code = 0; code < possibilities; ++code) {
					Vector series(precision);
					auto digits = code;
					for (auto &value : series) {
						value = Coefficient(digits % prime);
						digits /= prime;
					}
					bool has_approximant = false;
					for (std::size_t denominator_code = 0; denominator_code < denominators; ++denominator_code) {
						Vector denominator(denominator_degree + 1);
						denominator[0] = Coefficient(1);
						digits = denominator_code;
						for (std::size_t i = 1; i <= denominator_degree; ++i) {
							denominator[i] = Coefficient(digits % prime);
							digits /= prime;
						}
						auto cross = product(series, denominator);
						cross.resize(precision);
						bool is_valid = true;
						for (std::size_t i = numerator_degree + 1; i < precision; ++i) is_valid = is_valid && cross[i] == Coefficient(0);
						has_approximant = has_approximant || is_valid;
					}
					auto const result = atl::pade_approximant(series, numerator_degree, denominator_degree);
					REQUIRE(result.has_value() == has_approximant);
					auto const quadratic = atl::experimental::pade_approximant_by_quadratic_euclid(series, numerator_degree, denominator_degree);
					REQUIRE(quadratic.has_value() == has_approximant);
					if (result) {
						approximation(series, numerator_degree, denominator_degree, *result);
						REQUIRE(result->numerator == quadratic->numerator);
						REQUIRE(result->denominator == quadratic->denominator);
					}
				}
			}
	}

	/** @brief Reversed Euclid returns the same minimum order and a valid recurrence, including ambiguous prefixes. */
	template<atl::static_prime_mod_int Coefficient>
	void recurrence_and_pade() {
		using Vector = std::vector<Coefficient>;
		std::mt19937_64 random(19);
		for (std::size_t length : {0, 1, 2, 3, 17, 65, 136, 257}) {
			for (std::size_t trial = 0; trial < 12; ++trial) {
				Vector series(length);
				for (auto &value : series) value = Coefficient(trial == 0 ? 0 : random());
				if (trial == 1 && length != 0) {
					std::ranges::fill(series, Coefficient(0));
					series.back() = Coefficient(1);
				}
				auto const fast = atl::experimental::berlekamp_massey_by_half_gcd(series), slow = atl::berlekamp_massey(series);
				REQUIRE(fast.size() == slow.size());
				if (fast.size() <= length / 2) REQUIRE(fast == slow);
				for (std::size_t i = fast.size(); i < length; ++i) {
					Coefficient predicted(0);
					for (std::size_t j = 0; j < fast.size(); ++j) predicted += fast[j] * series[i - j - 1];
					REQUIRE(predicted == series[i]);
				}
				std::size_t const numerator_degree = length / 3, denominator_degree = length / 2;
				auto const result = atl::pade_approximant(series, numerator_degree, denominator_degree);
				auto const quadratic = atl::experimental::pade_approximant_by_quadratic_euclid(series, numerator_degree, denominator_degree);
				REQUIRE(result.has_value() == quadratic.has_value());
				if (result) {
					approximation(series, numerator_degree, denominator_degree, *result);
					REQUIRE(result->numerator == quadratic->numerator);
					REQUIRE(result->denominator == quadratic->denominator);
				}
			}
		}
	}
}

TEST_CASE("polynomial_gcd: all prime tiers and independent Euclid", "[polynomial_gcd]") {
	gcd_tests::family<atl::mod_int32<998244353>>();
	gcd_tests::family<atl::mod_int32<1000000007>>();
	gcd_tests::family<atl::mod_int32<7>>();
	gcd_tests::family<atl::mod_int32<13>>();
	gcd_tests::family<atl::mod_int32<2>>();
	gcd_tests::family<atl::mod_int64<18446744073709551557ULL>>();
}

TEST_CASE("polynomial_gcd: Pade existence and fast minimal recurrences", "[polynomial_gcd]") {
	gcd_tests::tiny_pade<atl::mod_int32<2>>();
	gcd_tests::tiny_pade<atl::mod_int32<3>>();
	gcd_tests::recurrence_and_pade<atl::mod_int32<998244353>>();
	gcd_tests::recurrence_and_pade<atl::mod_int32<1000000007>>();
	gcd_tests::recurrence_and_pade<atl::mod_int32<7>>();
	gcd_tests::recurrence_and_pade<atl::mod_int32<13>>();
	gcd_tests::recurrence_and_pade<atl::mod_int32<2>>();
	gcd_tests::recurrence_and_pade<atl::mod_int64<18446744073709551557ULL>>();
}

TEST_CASE("polynomial_gcd: abnormal quotient degrees, divisibility and single pass inputs", "[polynomial_gcd]") {
	using Coefficient = atl::mod_int32<998244353>;
	using Vector = std::vector<Coefficient>;
	Vector first{2, 3, 1}, second{3, 1};
	for (std::size_t degree : {2, 7, 3, 11, 2, 40, 3, 17, 29, 5}) {
		Vector quotient(degree + 1, Coefficient(1));
		auto next = gcd_tests::product(first, quotient);
		next.resize(std::max(next.size(), second.size()));
		for (std::size_t i = 0; i < second.size(); ++i) next[i] += second[i];
		second = first;
		first = next;
	}
	auto const expected = gcd_tests::oracle(first, second);
	for (std::size_t cutoff : {1, 32, 64, 128, 256}) {
		for (auto const &result : {atl::experimental::polynomial_extended_gcd_by_leaf_cutoff_plain_products(first, second, cutoff),
								   atl::experimental::polynomial_extended_gcd_by_leaf_cutoff_fused_products(first, second, cutoff)}) {
			REQUIRE(result.gcd == expected);
			gcd_tests::bezout(first, second, result);
		}
	}
	auto const divisible = gcd_tests::product(first, second);
	gcd_tests::bezout(divisible, first, atl::polynomial_extended_gcd(divisible, first));
	gcd_tests::bezout(first, divisible, atl::polynomial_extended_gcd(first, divisible));
	REQUIRE(!atl::polynomial_inverse_mod(divisible, first));
	std::forward_list<Coefficient> const input(first.begin(), first.end());
	REQUIRE(atl::polynomial_gcd(input, second) == expected);
}

TEST_CASE("polynomial_gcd: public and detail contracts", "[polynomial_gcd]") {
	using C = atl::mod_int32<7>;
	using V = std::vector<C>;
	using E = atl::default_ntt_engine;
	V oversized(atl::maximum_series_length + 1);
	oversized.back() = C(1);
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_gcd(oversized, V{}); }));
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_extended_gcd(V{}, oversized); }));
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_inverse_mod(oversized, V{1, 1}); }));
	for (V const &modulus : {V{}, V{0}, V{1}, V{3, 0}}) CHECK(atl_test::aborts([&] { (void)atl::polynomial_inverse_mod(V{}, modulus); }));
	for (std::size_t cutoff : {std::size_t{0}, atl::maximum_series_length + 1}) {
		CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_gcd_by_leaf_cutoff_plain_products(V{}, V{}, cutoff); }));
		CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_gcd_by_leaf_cutoff_fused_products(V{}, V{}, cutoff); }));
		CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_extended_gcd_by_leaf_cutoff_plain_products(V{}, V{}, cutoff); }));
		CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_extended_gcd_by_leaf_cutoff_fused_products(V{}, V{}, cutoff); }));
	}
	CHECK(atl_test::aborts([] { (void)atl::pade_approximant(V{}, atl::maximum_series_length, 0); }));
	CHECK(atl_test::aborts([] { (void)atl::pade_approximant(V{}, 0, atl::maximum_series_length); }));
	CHECK(atl_test::aborts([] { (void)atl::pade_approximant(V{}, atl::maximum_series_length - 1, 1); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::gcd_half<E, false>(V{}, V{}, 1); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::gcd_half<E, false>(V{1}, V{1}, 1); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::gcd_reduce<E, atl::detail::polynomial_gcd_algorithm::euclid, 0>(V{}, V{}, atl::maximum_series_length + 1, 1); }));
	oversized.back() = C(0);
	REQUIRE(atl::polynomial_gcd(oversized, V{1}) == V{1});
}

namespace gcd_tests {
	/** @brief Count direct transforms without conflating driver convolution calls. */
	struct counting_engine : atl::ntt_engine_base<counting_engine> {
		/** @brief Forward transform count. */
		inline static std::size_t forwards = 0;
		/** @brief Inverse transform count. */
		inline static std::size_t inverses = 0;

		/** @brief Delegate the scalar field capability. */
		template<atl::static_mod_int Coefficient>
		static constexpr bool supports() { return atl::scalar_ntt::supports<Coefficient>(); }

		/** @brief Count a forward. */
		template<atl::static_mod_int Coefficient>
		static void forward(decltype(Coefficient::mod()) *values, std::uint32_t levels) {
			++forwards;
			atl::scalar_ntt::forward<Coefficient>(values, levels);
		}

		/** @brief Count an inverse. */
		template<atl::static_mod_int Coefficient>
		static void inverse(decltype(Coefficient::mod()) *values, std::uint32_t levels) {
			++inverses;
			atl::scalar_ntt::inverse<Coefficient>(values, levels);
		}
	};
}

TEST_CASE("polynomial_gcd: fused matrix transform schedule and larger dense identities", "[polynomial_gcd]") {
	using C = atl::mod_int32<998244353>;
	using V = std::vector<C>;
	using E = gcd_tests::counting_engine;
	atl::detail::polynomial_transition<C> left, right;
	for (auto &entry : left) entry.assign(50, C(3));
	for (auto &entry : right) entry.assign(50, C(5));
	auto const expected = atl::detail::gcd_matrix_product<atl::default_ntt_engine, false, 2, 2, 2, C>(left, right);
	E::forwards = E::inverses = 0;
	REQUIRE(atl::detail::gcd_matrix_product<E, true, 2, 2, 2, C>(left, right) == expected);
	REQUIRE(E::forwards == 8);
	REQUIRE(E::inverses == 4);
	V const first(50, C(2)), second(50, C(7));
	auto const applied = atl::detail::gcd_apply<atl::default_ntt_engine, false>(left, first, second);
	E::forwards = E::inverses = 0;
	auto const fused = atl::detail::gcd_apply<E, true>(left, first, second);
	REQUIRE(fused.earlier == applied.earlier);
	REQUIRE(fused.later == applied.later);
	REQUIRE(E::forwards == 6);
	REQUIRE(E::inverses == 2);
	std::mt19937_64 random(312);
	for (std::size_t length : {1025, 4097}) {
		V earlier(length), later(length - 1);
		for (auto &value : earlier) value = C(random());
		for (auto &value : later) value = C(random());
		auto const expected_gcd = gcd_tests::oracle(earlier, later);
		REQUIRE(atl::polynomial_gcd(earlier, later) == expected_gcd);
		auto const result = atl::experimental::polynomial_extended_gcd_by_fused_matrix_products(earlier, later);
		REQUIRE(result.gcd == expected_gcd);
		gcd_tests::bezout(earlier, later, result);
	}
}

TEST_CASE("polynomial_gcd: internal product, size and sentinel guards", "[polynomial_gcd]") {
	using C = atl::mod_int32<7>;
	using V = std::vector<C>;
	using E = atl::default_ntt_engine;
	using A = atl::detail::polynomial_gcd_algorithm;
	V oversized(atl::maximum_series_length + 1);
	CHECK(atl_test::aborts([&] { (void)atl::detail::gcd_product<E, C>(oversized, V{1}); }));
	CHECK(atl_test::aborts([&] { V target; atl::detail::gcd_accumulate<C>(target, oversized); }));
	CHECK(atl_test::aborts([&] { V target = oversized; atl::detail::gcd_accumulate<C>(target, V{}); }));
	CHECK(atl_test::aborts([&] {
		atl::detail::polynomial_transition<C> transition;
		transition[0] = oversized;
		(void)atl::detail::gcd_matrix_product<E, true, 2, 2, 2, C>(transition, transition);
	}));
	CHECK(atl_test::aborts([&] { (void)atl::detail::gcd_half<E, false>(oversized, V{}, 1); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::gcd_half<E, false>(V{1}, V{}, 0); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::gcd_half<E, false>(V{1}, V{}, atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::gcd_reduce<E, A::euclid, 0>(V{}, oversized, 0, 1); }));
	CHECK(atl_test::aborts([&] { V larger(atl::maximum_series_length + 2); (void)atl::detail::gcd_reduce<E, A::euclid, 0>(larger, V{}, 0, 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::gcd_divide<E, A::half_gcd>(oversized, V{1}); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::gcd_divide<E, A::half_gcd>(oversized, V{1, 1}); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::gcd_half<E, false>(V{1, 0}, V{}, 1); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::gcd_half<E, false>(V{1, 0, 1}, V{1, 0}, 1); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::gcd_reduce<E, A::euclid, 0>(V{0}, V{}, 0, 1); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::gcd_reduce<E, A::euclid, 0>(V{1}, V{0}, 0, 1); }));
	oversized.back() = C(1);
	oversized[0] = C(1);
	CHECK(atl_test::aborts([&] { (void)atl::detail::gcd_reduce<E, A::euclid, 0>(oversized, V{}, 0, 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::gcd_divide<E, A::half_gcd>(oversized, V{1, 1}); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::berlekamp_massey_by_half_gcd(oversized); }));
}

TEST_CASE("polynomial_gcd: Pade accepts the full precision cap", "[polynomial_gcd]") {
	using C = atl::mod_int32<7>;
	using V = std::vector<C>;
	V const series(atl::maximum_series_length, C(1));
	auto const result = atl::pade_approximant(series, atl::maximum_series_length - 2, 1);
	REQUIRE(result.has_value());
	REQUIRE(result->numerator == V{1});
	REQUIRE(result->denominator == V{1, 6});
}

TEST_CASE("polynomial_gcd: documented cofactor conventions", "[polynomial_gcd]") {
	using C = atl::mod_int32<7>;
	using V = std::vector<C>;
	V const f{2, 3, 1}, divisor{1, 1}, scaled{4, 6, 2};
	auto const check = [](V const &first, V const &second, V const &gcd, V const &first_cofactor, V const &second_cofactor) {
		auto const result = atl::polynomial_extended_gcd(first, second);
		CHECK(result.gcd == gcd);
		CHECK(result.first_cofactor == first_cofactor);
		CHECK(result.second_cofactor == second_cofactor);
	};
	check(f, V{}, f, V{1}, V{});
	check(V{}, f, f, V{}, V{1});
	check(V{}, V{}, V{}, V{}, V{});
	check(divisor, f, divisor, V{1}, V{});
	check(f, divisor, divisor, V{}, V{1});
	check(f, scaled, f, V{}, V{4});
	check(scaled, f, f, V{}, V{1});
	check(f, V{3, 4, 1}, divisor, V{6}, V{1});
	auto const zero_series = atl::pade_approximant(V{}, 1, 1);
	REQUIRE(zero_series.has_value());
	CHECK(zero_series->numerator.empty());
	CHECK(zero_series->denominator == V{1});
	auto const truncated = atl::pade_approximant(V{1, 1}, 0, 3);
	REQUIRE(truncated.has_value());
	CHECK(truncated->numerator == V{1});
	CHECK(truncated->denominator == V{1, 6, 1, 6});
	auto const inverse = atl::polynomial_inverse_mod(V{3}, divisor);
	REQUIRE(inverse.has_value());
	CHECK(*inverse == V{5});
	CHECK(!atl::polynomial_inverse_mod(V{}, divisor).has_value());
}

TEST_CASE("polynomial_gcd: extended results compare member by member", "[polynomial_gcd]") {
	using C = atl::mod_int32<7>;
	using V = std::vector<C>;
	using Result = atl::polynomial_extended_gcd_result<C>;
	V const f{2, 3, 1}, g{3, 4, 1};
	auto const result = atl::polynomial_extended_gcd(f, g);
	CHECK(result == Result{{1, 1}, {6}, {1}});
	CHECK(result == atl::experimental::polynomial_extended_gcd_by_schoolbook_euclid(f, g));
	CHECK_FALSE(result != Result{{1, 1}, {6}, {1}});
	CHECK(result != Result{{1, 1}, {6}, {}});
	CHECK(result != Result{{1, 1}, {}, {1}});
	CHECK(result != Result{{2, 2}, {5}, {2}});
	// Another Bezout identity for the same gcd, shifted by (x + 3, -(x + 2)): valid, yet structurally different.
	Result const other{{1, 1}, {2, 1}, {6, 6}};
	auto identity = gcd_tests::product(f, other.first_cofactor);
	auto const second_term = gcd_tests::product(g, other.second_cofactor);
	for (std::size_t i = 0; i < second_term.size(); ++i) identity[i] += second_term[i];
	gcd_tests::trim(identity);
	REQUIRE(identity == other.gcd);
	CHECK(result != other);
}
