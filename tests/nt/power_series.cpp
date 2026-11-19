#include <atl/experimental/nt/linear_recurrence_variants.hpp>
#include <atl/experimental/nt/power_series_variants.hpp>
#include <atl/nt/convolution.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/power_series.hpp>
#include <atl/nt/series_calculus.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"
#include "input_only.hpp"

namespace {
	/** @brief Independent truncated schoolbook multiplication for the defining identities. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> schoolbook(std::vector<Coefficient> const &a, std::vector<Coefficient> const &b, std::size_t n) {
		std::vector<Coefficient> result(n);
		for (std::size_t i = 0; i < a.size() && i < n; ++i)
			for (std::size_t j = 0; j < b.size() && j < n - i; ++j) result[i + j] += a[i] * b[j];
		return result;
	}

	/** @brief Solve b*q=a as a lower triangular coefficient system, independently of all library kernels. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> quotient_oracle(std::vector<Coefficient> const &a, std::vector<Coefficient> const &b, std::size_t n) {
		std::vector<Coefficient> result(n);
		for (std::size_t k = 0; k < n; ++k) {
			Coefficient residual = k < a.size() ? a[k] : Coefficient(0);
			for (std::size_t j = 1; j <= k && j < b.size(); ++j) residual -= b[j] * result[k - j];
			result[k] = residual / b[0];
		}
		return result;
	}

	/** @brief Solve f'=f*(log f)' coefficient by coefficient, with log f constant zero. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> logarithm_oracle(std::vector<Coefficient> const &f, std::size_t n) {
		std::vector<Coefficient> result(n);
		for (std::size_t k = 1; k < n; ++k) {
			Coefficient residual = k < f.size() ? Coefficient(k) * f[k] : Coefficient(0);
			for (std::size_t j = 1; j < k; ++j)
				if (k - j < f.size()) residual -= Coefficient(j) * result[j] * f[k - j];
			result[k] = residual / Coefficient(k);
		}
		return result;
	}

	/** @brief Solve g'=f'*g coefficient by coefficient, with g constant one. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> exponential_oracle(std::vector<Coefficient> const &f, std::size_t n) {
		std::vector<Coefficient> result(n);
		if (n != 0) result[0] = Coefficient(1);
		for (std::size_t k = 1; k < n; ++k) {
			for (std::size_t j = 0; j < k; ++j)
				if (k - j < f.size()) result[k] += Coefficient(k - j) * f[k - j] * result[j];
			result[k] /= Coefficient(k);
		}
		return result;
	}

	/** @brief All contenders versus independent definitions, including short and overlong inputs. */
	template<atl::static_prime_mod_int Coefficient>
	void check_oracles(std::size_t n, std::mt19937_64 &random) {
		CAPTURE(n, Coefficient::mod());
		for (int shape = 0; shape < 3; ++shape) {
			std::size_t const input_size = shape == 0 ? n / 2 + 1 : shape == 1 ? n + 5
																			   : 1;
			std::vector<Coefficient> f(input_size), a(input_size);
			for (auto &value : f) value = Coefficient(random());
			for (auto &value : a) value = Coefficient(random());
			f[0] = Coefficient(3);
			if (f[0] == Coefficient(0)) f[0] = Coefficient(1);
			auto const saved_f = f, saved_a = a;
			auto const inverse = quotient_oracle(std::vector<Coefficient>{1}, f, n);
			REQUIRE(atl::series_inverse(f, n) == inverse);
			REQUIRE(atl::experimental::series_inverse_by_recurrence(f, n) == inverse);
			REQUIRE(atl::experimental::series_inverse_by_plain_newton(f, n) == inverse);
			REQUIRE(atl::series_inverse(atl_test::input_only<Coefficient>(f), n) == inverse);
			// The Kitamasa contender's own Newton inverse is an independent implementation; its convolution needs a 32-bit modulus.
			if constexpr (Coefficient::mod() <= (std::numeric_limits<std::uint32_t>::max)()) REQUIRE(atl::experimental::detail::series_inverse(f, n) == inverse);
			auto identity = std::vector<Coefficient>(n);
			if (n) identity[0] = Coefficient(1);
			REQUIRE(schoolbook(f, inverse, n) == identity);
			auto const quotient = quotient_oracle(a, f, n);
			REQUIRE(atl::series_division(a, f, n) == quotient);
			REQUIRE(atl::experimental::series_division_by_inverse_product(a, f, n) == quotient);
			REQUIRE(atl::series_division(atl_test::input_only<Coefficient>(a), atl_test::input_only<Coefficient>(f), n) == quotient);
			REQUIRE(f == saved_f);
			REQUIRE(a == saved_a);
			if (n > Coefficient::mod()) continue;
			f[0] = Coefficient(1);
			auto const saved_unit = f;
			auto const logarithm = logarithm_oracle(f, n);
			REQUIRE(atl::series_logarithm(f, n) == logarithm);
			REQUIRE(atl::experimental::series_logarithm_by_inverse_product(f, n) == logarithm);
			REQUIRE(atl::series_logarithm(atl_test::input_only<Coefficient>(f), n) == logarithm);
			REQUIRE(f == saved_unit);
			auto const derivative_product = schoolbook(f, atl::series_derivative(logarithm), n == 0 ? 0 : n - 1);
			auto expected_derivative = atl::series_derivative(f);
			expected_derivative.resize(n == 0 ? 0 : n - 1);
			REQUIRE(derivative_product == expected_derivative);
			a[0] = Coefficient(0);
			auto const saved_zero = a;
			auto const exponential = exponential_oracle(a, n);
			REQUIRE(atl::series_exponential(a, n) == exponential);
			REQUIRE(atl::experimental::series_exponential_by_recurrence(a, n) == exponential);
			REQUIRE(atl::experimental::series_exponential_by_plain_newton(a, n) == exponential);
			REQUIRE(atl::series_exponential(atl_test::input_only<Coefficient>(a), n) == exponential);
			REQUIRE(a == saved_zero);
			REQUIRE(atl::series_derivative(exponential) == schoolbook(atl::series_derivative(a), exponential, n == 0 ? 0 : n - 1));
			a.resize(n);
			f.resize(n);
			REQUIRE(atl::series_logarithm(exponential, n) == a);
			REQUIRE(atl::series_exponential(logarithm, n) == f);
		}
	}

	/** @brief Large checks by product and differential identities, without quadratic work. */
	template<atl::static_prime_mod_int Coefficient>
	void check_large(std::size_t n) {
		std::mt19937_64 random(3);
		std::vector<Coefficient> f(n), a(n);
		for (auto &value : f) value = Coefficient(random());
		for (auto &value : a) value = Coefficient(random());
		f[0] = Coefficient(1);
		a[0] = Coefficient(0);
		auto product = [&](atl::mod_int_range auto const &left, atl::mod_int_range auto const &right, std::size_t size) {
			auto result = atl::convolution(left, right);
			result.resize(size);
			return result;
		};
		auto const inverse = atl::series_inverse(f, n);
		REQUIRE(inverse == atl::experimental::series_inverse_by_plain_newton(f, n));
		std::vector<Coefficient> identity(n);
		identity[0] = Coefficient(1);
		REQUIRE(product(f, inverse, n) == identity);
		auto const quotient = atl::series_division(a, f, n);
		REQUIRE(quotient == atl::experimental::series_division_by_inverse_product(a, f, n));
		REQUIRE(product(quotient, f, n) == a);
		auto const logarithm = atl::series_logarithm(f, n);
		REQUIRE(logarithm == atl::experimental::series_logarithm_by_inverse_product(f, n));
		REQUIRE(product(f, atl::series_derivative(logarithm), n - 1) == atl::series_derivative(f));
		auto const exponential = atl::series_exponential(a, n);
		REQUIRE(exponential == atl::experimental::series_exponential_by_plain_newton(a, n));
		REQUIRE(product(exponential, atl::series_derivative(a), n - 1) == atl::series_derivative(exponential));
		REQUIRE(atl::series_logarithm(exponential, n) == a);
		REQUIRE(atl::series_exponential(logarithm, n) == f);
	}

	/** @brief Instrumented engine checks the reuse count without depending on timings. */
	struct counting_engine : atl::ntt_engine_base<counting_engine> {
		inline static int forwards = 0, inverses = 0;

		/** @brief Use the scalar engine's modulus contract. */
		template<atl::static_mod_int Coefficient>
		static constexpr bool supports() { return atl::scalar_ntt::supports<Coefficient>(); }

		/** @brief Count and perform one forward transform. */
		template<atl::static_mod_int Coefficient>
		static void forward(decltype(Coefficient::mod()) *values, atl::uint32 levels) {
			++forwards;
			atl::scalar_ntt::forward<Coefficient>(values, levels);
		}

		/** @brief Count and perform one inverse transform. */
		template<atl::static_mod_int Coefficient>
		static void inverse(decltype(Coefficient::mod()) *values, atl::uint32 levels) {
			++inverses;
			atl::scalar_ntt::inverse<Coefficient>(values, levels);
		}
	};

	/** @brief An engine rejecting every modulus must fall back to scalar, without calling its transforms. */
	struct rejecting_engine : atl::ntt_engine_base<rejecting_engine> {
		/** @brief Reject this modulus. */
		template<atl::static_mod_int Coefficient>
		static constexpr bool supports() { return false; }

		/** @brief Any call would expose a missing capability fallback. */
		template<atl::static_mod_int Coefficient>
		static void forward(decltype(Coefficient::mod()) *, atl::uint32) { FAIL("unsupported engine called"); }

		/** @brief Any call would expose a missing capability fallback. */
		template<atl::static_mod_int Coefficient>
		static void inverse(decltype(Coefficient::mod()) *, atl::uint32) { FAIL("unsupported engine called"); }
	};
}

TEST_CASE("power_series: independent definitions and Newton boundaries", "[power_series]") {
	std::mt19937_64 random(42);
	for (std::size_t n : {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 66, 127, 128, 129, 255, 256, 257}) {
		check_oracles<atl::mod_int32<998244353>>(n, random);
		check_oracles<atl::mod_int32<1000000007>>(n, random);
	}
	for (int trial = 0; trial < 24; ++trial) {
		check_oracles<atl::mod_int32<998244353>>(std::size_t(random() % 300), random);
		check_oracles<atl::mod_int32<1000000007>>(std::size_t(random() % 300), random);
	}
	for (std::size_t n = 0; n <= 15; ++n) {
		check_oracles<atl::mod_int32<7>>(n, random);
		check_oracles<atl::mod_int32<13>>(n, random);
	}
	for (std::size_t n : {0, 1, 2, 3, 65, 129}) check_oracles<atl::mod_int64<18446744073709551557ull>>(n, random);
	for (std::size_t n : {65, 129, 257}) check_oracles<atl::mod_int32<4294967291u>>(n, random);
	check_oracles<atl::mod_int64<998244353>>(129, random);
	for (std::size_t n : {0, 1, 2, 3, 65}) check_oracles<atl::mod_int32<2>>(n, random);
}

TEST_CASE("power_series: large identities", "[power_series]") {
	for (std::size_t n : {65535, 65536, 65539}) check_large<atl::mod_int32<998244353>>(n);
	check_large<atl::mod_int32<1000000007>>(65539);
}

TEST_CASE("power_series: engine threading, prefixes, empty inputs and compatibility", "[power_series]") {
	using Coefficient = atl::mod_int32<998244353>;
	std::vector<Coefficient> f{1, 3, 2};
	counting_engine::forwards = counting_engine::inverses = 0;
	auto const inverse = atl::series_inverse<counting_engine>(f, 129);
	REQUIRE(counting_engine::forwards == 6);
	REQUIRE(counting_engine::inverses == 4);
	REQUIRE(atl::series_inverse<rejecting_engine>(f, 129) == inverse);
	REQUIRE(atl::series_inverse(std::forward_list<Coefficient>{1, 3, 2}, 129) == inverse);
	std::vector<Coefficient> const empty;
	REQUIRE(atl::series_exponential(empty, 3) == std::vector<Coefficient>{1, 0, 0});
	REQUIRE(atl::series_division(empty, f, 3) == std::vector<Coefficient>(3));
	REQUIRE(atl::series_logarithm(std::vector<Coefficient>{1}, 3) == std::vector<Coefficient>(3));
	REQUIRE(atl::series_inverse(std::vector<Coefficient>{2}, 3) == std::vector<Coefficient>{Coefficient(2).inverse(), 0, 0});
	REQUIRE(atl::series_inverse(empty, 0).empty());
	REQUIRE(atl::series_division(f, empty, 0).empty());
	REQUIRE(atl::series_logarithm(empty, 0).empty());
	REQUIRE(atl::series_exponential(f, 0).empty());
	REQUIRE(atl::experimental::series_inverse_by_recurrence(empty, 0).empty());
	REQUIRE(atl::experimental::series_inverse_by_plain_newton(empty, 0).empty());
	REQUIRE(atl::experimental::series_division_by_inverse_product(f, empty, 0).empty());
	REQUIRE(atl::experimental::series_logarithm_by_inverse_product(empty, 0).empty());
	REQUIRE(atl::experimental::series_exponential_by_recurrence(f, 0).empty());
	REQUIRE(atl::experimental::series_exponential_by_plain_newton(f, 0).empty());
	for (std::size_t n : {1, 2, 65, 129}) {
		auto longer = f;
		longer.resize(n + 2, Coefficient(17));
		auto shorter = longer;
		shorter.resize(n);
		REQUIRE(atl::series_inverse(longer, n) == atl::series_inverse(shorter, n));
		REQUIRE(atl::series_division(longer, longer, n) == atl::series_division(shorter, shorter, n));
		REQUIRE(atl::series_logarithm(longer, n) == atl::series_logarithm(shorter, n));
		longer[0] = shorter[0] = Coefficient(0);
		REQUIRE(atl::series_exponential(longer, n) == atl::series_exponential(shorter, n));
	}
	std::vector<Coefficient> const recurrence{1, 1}, initial{0, 1};
	REQUIRE(atl::experimental::linear_recurrence_term_by_convolution_kitamasa(recurrence, initial, 20) == Coefficient(6765));
}

TEST_CASE("power_series: asserted contracts for all contenders", "[power_series]") {
	using Coefficient = atl::mod_int32<7>;
	using Vector = std::vector<Coefficient>;
	using Unary = Vector (*)(Vector const &, std::size_t);
	using Binary = Vector (*)(Vector const &, Vector const &, std::size_t);
	std::array<Unary, 3> const inverses{atl::series_inverse<>, atl::experimental::series_inverse_by_recurrence<>, atl::experimental::series_inverse_by_plain_newton<>};
	std::array<Unary, 2> const logarithms{atl::series_logarithm<>, atl::experimental::series_logarithm_by_inverse_product<>};
	std::array<Unary, 3> const exponentials{atl::series_exponential<>, atl::experimental::series_exponential_by_recurrence<>, atl::experimental::series_exponential_by_plain_newton<>};
	std::array<Binary, 2> const divisions{atl::series_division<>, atl::experimental::series_division_by_inverse_product<>};
	for (auto solver : inverses) {
		CHECK(atl_test::aborts([&] { (void)solver({}, 1); }));
		CHECK(atl_test::aborts([&] { (void)solver({0, 1}, 1); }));
		CHECK(atl_test::aborts([&] { (void)solver({1}, atl::maximum_series_length + 1); }));
	}
	for (auto solver : logarithms) {
		CHECK(atl_test::aborts([&] { (void)solver({}, 1); }));
		CHECK(atl_test::aborts([&] { (void)solver({2}, 1); }));
		CHECK(atl_test::aborts([&] { (void)solver({1}, 8); }));
		CHECK(atl_test::aborts([&] { (void)solver({1}, atl::maximum_series_length + 1); }));
	}
	for (auto solver : exponentials) {
		CHECK(atl_test::aborts([&] { (void)solver({1}, 1); }));
		CHECK(atl_test::aborts([&] { (void)solver({}, 8); }));
		CHECK(atl_test::aborts([&] { (void)solver({}, atl::maximum_series_length + 1); }));
	}
	for (auto solver : divisions) {
		CHECK(atl_test::aborts([&] { (void)solver({1}, {}, 1); }));
		CHECK(atl_test::aborts([&] { (void)solver({0, 1}, {0, 1}, 2); }));
		CHECK(atl_test::aborts([&] { (void)solver({}, {1}, atl::maximum_series_length + 1); }));
	}
	CHECK(atl_test::aborts([] { (void)atl::series_logarithm(std::vector<atl::mod_int32<13>>{1}, 14); }));
	CHECK(atl_test::aborts([] { (void)atl::series_exponential(std::vector<atl::mod_int32<13>>{}, 14); }));
}

TEST_CASE("power_series: internal kernel guards", "[power_series]") {
	using Coefficient = atl::mod_int32<7>;
	using Vector = std::vector<Coefficient>;
	using Engine = atl::default_ntt_engine;
	constexpr auto limit = atl::maximum_series_length;
	Vector const unit{1}, zero{0}, empty;
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_product<Engine, Coefficient>(unit, unit, limit + 1); }));
	{
		Vector const oversized(limit + 1);
		CHECK(atl_test::aborts([&] { (void)atl::detail::series_product<Engine, Coefficient>(oversized, unit, 1); }));
		CHECK(atl_test::aborts([&] { (void)atl::detail::series_product<Engine, Coefficient>(unit, oversized, 1); }));
		CHECK(atl_test::aborts([&] { (void)atl::detail::series_logarithm_product<Engine, Coefficient>(oversized); }));
	}
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_inverse_recurrence<Coefficient>(unit, limit + 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_inverse_newton<Engine, Coefficient>(unit, limit + 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_inverse_extend<Engine, Coefficient>(unit, unit, limit + 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_inverse_extend<Engine, Coefficient>(unit, empty, 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_inverse_extend<Engine, Coefficient>(unit, Vector(3), 4); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_inverse_extend<Engine, Coefficient>(unit, unit, 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_inverse_extend<Engine, Coefficient>(unit, unit, 3); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_division_fused<Engine, Coefficient>(unit, unit, limit + 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_division_fused<Engine, Coefficient>(empty, unit, 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_division_fused<Engine, Coefficient>(unit, empty, 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_division_fused<Engine, Coefficient>(unit, zero, 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_logarithm_product<Engine, Coefficient>(zero); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_logarithm_product<Engine, Coefficient>(Vector(8, 1)); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_exponential_recurrence<Coefficient>(zero, limit + 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_exponential_recurrence<Coefficient>(zero, 8); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_exponential_recurrence<Coefficient>(empty, 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_exponential_coupled<Engine, Coefficient>(Vector(8)); }));
}

namespace {
	/** @brief Taylor composition by Horner and independent schoolbook products, for precision at most 64. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> taylor_horner(std::vector<Coefficient> const &u, std::vector<Coefficient> const &derivatives) {
		std::vector<Coefficient> value(u.size());
		for (std::size_t k = derivatives.size(); k > 0; --k) {
			value = schoolbook(value, u, u.size());
			if (!value.empty()) value[0] += derivatives[k - 1];
		}
		return value;
	}

	/** @brief Taylor coefficients F^(k)(c)/k! for inverse, exponential, and integer powers. */
	template<atl::static_prime_mod_int Coefficient>
	void check_taylor(std::size_t n, std::mt19937_64 &random) {
		using Vector = std::vector<Coefficient>;
		using Unary = Vector (*)(Vector const &, std::size_t);
		using Power = Vector (*)(Vector const &, std::uint64_t, std::size_t);
		std::array<Unary, 5> const inverses{atl::series_inverse<>, atl::experimental::series_inverse_by_recurrence<>, atl::experimental::series_inverse_by_plain_newton<>, atl::experimental::series_inverse_by_newton_driver<>, atl::series_inverse_sparse};
		std::array<Unary, 5> const exponentials{atl::series_exponential<>, atl::experimental::series_exponential_by_recurrence<>, atl::experimental::series_exponential_by_plain_newton<>, atl::experimental::series_exponential_by_newton_driver<>, atl::series_exponential_sparse};
		std::array<Power, 3> const powers{atl::series_power<>, atl::experimental::series_power_by_binary_powering<>, atl::series_power_sparse};
		Vector u(n), inverse_coefficients(n), exponential_coefficients(n), power_coefficients(n);
		for (std::size_t i = 1; i < n; ++i) u[i] = Coefficient(random());
		Coefficient const c(3);
		Coefficient inverse_coefficient = Coefficient(1) / c, inverse_factorial(1);
		for (std::size_t k = 0; k < n; ++k) {
			if (k != 0) inverse_factorial /= Coefficient(k);
			inverse_coefficients[k] = inverse_coefficient;
			exponential_coefficients[k] = inverse_factorial;
			inverse_coefficient /= -c;
		}
		auto input = u;
		if (n != 0) input[0] = c;
		auto const inverse = taylor_horner(u, inverse_coefficients);
		for (auto solver : inverses) REQUIRE(solver(input, n) == inverse);
		auto const exponential = taylor_horner(u, exponential_coefficients);
		for (auto solver : exponentials) REQUIRE(solver(u, n) == exponential);
		for (std::uint64_t exponent : {0ULL, 1ULL, 2ULL, 3ULL, 32ULL, 1000000000000000000ULL}) {
			Coefficient binomial(1);
			for (std::size_t k = 0; k < n; ++k) {
				if (k > exponent)
					power_coefficients[k] = Coefficient(0);
				else {
					if (k != 0) binomial *= Coefficient(exponent - (k - 1)) / Coefficient(k);
					power_coefficients[k] = binomial * c.power(exponent - k);
				}
			}
			auto const power = taylor_horner(u, power_coefficients);
			for (auto solver : powers) REQUIRE(solver(input, exponent, n) == power);
		}
	}

	/** @brief Independent descending elimination, with no library division or normalization helper. */
	template<atl::static_prime_mod_int Coefficient>
	atl::polynomial_division_result<Coefficient> polynomial_oracle(std::vector<Coefficient> numerator, std::vector<Coefficient> denominator) {
		while (!denominator.empty() && denominator.back() == Coefficient(0)) denominator.pop_back();
		while (!numerator.empty() && numerator.back() == Coefficient(0)) numerator.pop_back();
		std::vector<Coefficient> quotient;
		if (numerator.size() >= denominator.size()) quotient.resize(numerator.size() - denominator.size() + 1);
		while (numerator.size() >= denominator.size()) {
			auto const degree = numerator.size() - denominator.size();
			Coefficient const scale = numerator.back() / denominator.back();
			quotient[degree] += scale;
			for (std::size_t j = 0; j < denominator.size(); ++j) numerator[degree + j] -= scale * denominator[j];
			while (!numerator.empty() && numerator.back() == Coefficient(0)) numerator.pop_back();
		}
		return {quotient, numerator};
	}

	/** @brief Compare all power and root contenders against independent products, on sparse and dense inputs. */
	template<atl::static_prime_mod_int Coefficient>
	void check_powers_and_roots(std::size_t n, std::size_t terms, std::mt19937_64 &random) {
		using Vector = std::vector<Coefficient>;
		Vector f(n + 3);
		for (std::size_t i = 0; i < terms; ++i) f[i % f.size()] = Coefficient(random());
		f[0] = Coefficient(3);
		if (f[0] == Coefficient(0)) f[0] = Coefficient(1);
		Vector identity(n);
		if (n != 0) identity[0] = Coefficient(1);
		for (std::uint64_t exponent : {0ULL, 1ULL, 2ULL, 3ULL, 8ULL}) {
			auto expected = identity;
			for (std::uint64_t k = 0; k < exponent; ++k) expected = schoolbook(expected, f, n);
			REQUIRE(atl::series_power(f, exponent, n) == expected);
			REQUIRE(atl::experimental::series_power_by_binary_powering(f, exponent, n) == expected);
			if (n <= Coefficient::mod()) REQUIRE(atl::series_power_sparse(f, exponent, n) == expected);
		}
		REQUIRE(atl::series_inverse_sparse(f, n) == quotient_oracle(Vector{1}, f, n));
		REQUIRE(atl::series_division_sparse(f, f, n) == identity);
		REQUIRE(atl::experimental::series_inverse_by_newton_driver(f, n) == atl::series_inverse(f, n));
		if (n <= Coefficient::mod()) {
			f[0] = Coefficient(1);
			REQUIRE(atl::series_logarithm_sparse(f, n) == logarithm_oracle(f, n));
			f[0] = Coefficient(0);
			REQUIRE(atl::series_exponential_sparse(f, n) == exponential_oracle(f, n));
			REQUIRE(atl::experimental::series_exponential_by_newton_driver(f, n) == exponential_oracle(f, n));
		}
		for (std::size_t valuation : {0, 1, 2, 3, 4, 6}) {
			Vector root(n + 7);
			for (std::size_t i = valuation; i < root.size(); ++i) root[i] = Coefficient(random());
			root[valuation] = Coefficient(1);
			auto const square = schoolbook(root, root, n);
			auto const actual = atl::series_square_root(square, n);
			REQUIRE(actual.has_value());
			REQUIRE(schoolbook(*actual, *actual, n) == square);
			REQUIRE(atl::experimental::series_square_root_by_plain_newton(square, n) == actual);
			if (n <= Coefficient::mod() || Coefficient::mod() == 2) {
				REQUIRE(atl::series_square_root_sparse(square, n) == actual);
				REQUIRE(atl::experimental::series_square_root_by_logarithm(square, n) == actual);
			}
			if constexpr (Coefficient::mod() != 2) {
				for (std::size_t i = n > valuation ? n - valuation : 0; i < n; ++i) REQUIRE((*actual)[i] == Coefficient(0));
			}
		}
	}
}

TEST_CASE("power_series: powers against independent Taylor expansions", "[power_series]") {
	std::mt19937_64 random(4);
	for (std::size_t n : {0, 1, 2, 7, 16, 63, 64}) {
		check_taylor<atl::mod_int32<998244353>>(n, random);
		check_taylor<atl::mod_int32<1000000007>>(n, random);
	}
}

TEST_CASE("power_series: powers, roots and sparse shapes against products", "[power_series]") {
	std::mt19937_64 random(5);
	for (std::size_t n : {0, 1, 2, 3, 7, 16, 64, 65, 127, 128, 129, 257})
		for (std::size_t s : {1, 2, 8, 300}) check_powers_and_roots<atl::mod_int32<998244353>>(n, s, random);
	for (std::size_t n : {2, 7, 9, 65, 129}) {
		check_powers_and_roots<atl::mod_int32<7>>(n, 8, random);
		check_powers_and_roots<atl::mod_int32<2>>(n, 8, random);
	}
	check_powers_and_roots<atl::mod_int32<1000000007>>(129, 300, random);
	check_powers_and_roots<atl::mod_int64<18446744073709551557ULL>>(65, 8, random);
}

TEST_CASE("power_series: valuations, root choice and precision", "[power_series]") {
	using Coefficient = atl::mod_int32<7>;
	using Vector = std::vector<Coefficient>;
	using Root = std::optional<Vector> (*)(Vector const &, std::size_t);
	std::array<Root, 4> const roots{atl::series_square_root<>, atl::experimental::series_square_root_by_plain_newton<>, atl::experimental::series_square_root_by_logarithm<>, atl::series_square_root_sparse};
	for (auto solver : roots) {
		REQUIRE(solver({}, 0) == std::optional<Vector>{Vector{}});
		REQUIRE(solver({}, 7) == std::optional<Vector>{Vector(7)});
		for (std::size_t valuation : {0, 1, 2, 3}) {
			for (int scalar = 1; scalar < 7; ++scalar) {
				Vector f(7);
				f[valuation] = Coefficient(scalar);
				auto const actual = solver(f, 7);
				int scalar_root = 0;
				while (scalar_root < 7 && scalar_root * scalar_root % 7 != scalar) ++scalar_root;
				REQUIRE(actual.has_value() == (valuation % 2 == 0 && scalar_root < 7));
				if (actual) REQUIRE((*actual)[valuation / 2] == Coefficient(scalar_root));
			}
		}
		Vector f{0, 0, 4, 4, 1, 0, 0, 6, 6};
		auto const actual = solver(f, 7);
		REQUIRE(actual == std::optional<Vector>{Vector{0, 2, 1, 0, 0, 0, 0}});
		f.resize(7);
		REQUIRE(solver(f, 7) == actual);
	}
	using Bit = atl::mod_int32<2>;
	std::vector<Bit> const even{1, 0, 1, 0, 1, 0, 1, 0, 1};
	REQUIRE(atl::series_square_root(even, 9) == std::optional{std::vector<Bit>{1, 1, 1, 1, 1, 0, 0, 0, 0}});
	REQUIRE_FALSE(atl::series_square_root(std::vector<Bit>{0, 1}, 9));
	REQUIRE_FALSE(atl::series_square_root_sparse(std::vector<Bit>{0, 1}, 9));
	for (std::size_t valuation : {0, 1, 2, 3}) {
		Vector f(4);
		f[valuation] = Coefficient(2);
		auto const expected = valuation == 0 ? Vector{Coefficient(2).power(1000000000000000000ULL), 0, 0, 0} : Vector(4);
		REQUIRE(atl::series_power(f, 1000000000000000000ULL, 4) == expected);
		REQUIRE(atl::experimental::series_power_by_binary_powering(f, 1000000000000000000ULL, 4) == expected);
		REQUIRE(atl::series_power_sparse(f, 1000000000000000000ULL, 4) == expected);
		REQUIRE(atl::series_power(f, std::numeric_limits<std::uint64_t>::max(), 4) == (valuation == 0 ? Vector{Coefficient(2).power(std::numeric_limits<std::uint64_t>::max()), 0, 0, 0} : Vector(4)));
	}
	REQUIRE(atl::series_power(Vector{}, 0, 4) == Vector{1, 0, 0, 0});
	REQUIRE(atl::series_power(Vector{}, 1, 4) == Vector(4));
	REQUIRE(atl::series_power(Vector{1, 1}, 1000000000000000000ULL, 10) == atl::experimental::series_power_by_binary_powering(Vector{1, 1}, 1000000000000000000ULL, 10));
	REQUIRE(atl::series_power(std::forward_list<Coefficient>{0, 2, 1}, 2, 7) == Vector{0, 0, 4, 4, 1, 0, 0});
	REQUIRE(atl::series_square_root(std::forward_list<Coefficient>{0, 0, 4, 4, 1}, 7) == roots[0](Vector{0, 0, 4, 4, 1}, 7));
}

TEST_CASE("power_series: polynomial division independent elimination", "[power_series]") {
	using Coefficient = atl::mod_int32<998244353>;
	using Vector = std::vector<Coefficient>;
	std::mt19937_64 random(6);
	for (std::size_t n : {0, 1, 10, 64, 65, 129, 257}) {
		for (std::size_t m : {1, 2, 64, 65, 128}) {
			Vector f(n), g(m);
			for (auto &value : f) value = Coefficient(random());
			for (auto &value : g) value = Coefficient(random());
			g.back() = Coefficient(3);
			auto const expected = polynomial_oracle(f, g);
			f.resize(n + 3);
			g.resize(m + 3);
			auto const actual = atl::polynomial_division(f, g);
			REQUIRE(actual.quotient == expected.quotient);
			REQUIRE(actual.remainder == expected.remainder);
			auto const other = atl::experimental::polynomial_division_by_schoolbook(f, g);
			REQUIRE(other.quotient == expected.quotient);
			REQUIRE(other.remainder == expected.remainder);
			auto reconstructed = schoolbook(actual.quotient, g, f.size());
			for (std::size_t i = 0; i < actual.remainder.size(); ++i) reconstructed[i] += actual.remainder[i];
			REQUIRE(reconstructed == f);
		}
	}
	REQUIRE(atl::polynomial_division(Vector{0, 0}, Vector{1, 0, 0}).quotient.empty());
	REQUIRE(atl::polynomial_division(std::forward_list<Coefficient>{1, 2, 1, 0}, std::forward_list<Coefficient>{1, 1, 0}).quotient == Vector{1, 1});
}

TEST_CASE("power_series: shared driver schedule and transform counts", "[power_series]") {
	using Coefficient = atl::mod_int32<998244353>;
	std::vector<std::pair<std::size_t, std::size_t>> schedule;
	atl::detail::series_newton_lift(schedule, 64, 257, [](auto &state, std::size_t middle, std::size_t next) { state.emplace_back(middle, next); });
	REQUIRE(schedule == std::vector<std::pair<std::size_t, std::size_t>>{{64, 128}, {128, 256}, {256, 257}});
	std::vector<Coefficient> const f{1, 2, 1};
	counting_engine::forwards = counting_engine::inverses = 0;
	REQUIRE(atl::series_square_root<counting_engine>(f, 512) == atl::series_square_root(f, 512));
	// Two reciprocal-maintaining steps (6/4 each), then a final square/product (3/2).
	REQUIRE(counting_engine::forwards == 15);
	REQUIRE(counting_engine::inverses == 10);
	REQUIRE(atl::series_square_root<rejecting_engine>(f, 512) == atl::series_square_root(f, 512));
}

TEST_CASE("power_series: asserted contracts of powers, roots, division and sparse entries", "[power_series]") {
	using Coefficient = atl::mod_int32<7>;
	using Vector = std::vector<Coefficient>;
	using Unary = Vector (*)(Vector const &, std::size_t);
	using Power = Vector (*)(Vector const &, std::uint64_t, std::size_t);
	using Root = std::optional<Vector> (*)(Vector const &, std::size_t);
	for (Unary solver : {atl::series_inverse_sparse<Vector>, atl::experimental::series_inverse_by_newton_driver<atl::default_ntt_engine, Vector>}) {
		CHECK(atl_test::aborts([&] { (void)solver({}, 1); }));
		CHECK(atl_test::aborts([&] { (void)solver({0, 1}, 2); }));
	}
	CHECK(atl_test::aborts([] { (void)atl::series_division_sparse(Vector{1}, Vector{}, 1); }));
	CHECK(atl_test::aborts([] { (void)atl::series_logarithm_sparse(Vector{2}, 1); }));
	CHECK(atl_test::aborts([] { (void)atl::series_logarithm_sparse(Vector{1}, 8); }));
	for (Unary solver : {atl::series_exponential_sparse<Vector>, atl::experimental::series_exponential_by_newton_driver<atl::default_ntt_engine, Vector>}) {
		CHECK(atl_test::aborts([&] { (void)solver({1}, 1); }));
		CHECK(atl_test::aborts([&] { (void)solver({}, 8); }));
	}
	for (Power solver : {atl::series_power<atl::default_ntt_engine, Vector>, atl::experimental::series_power_by_binary_powering<atl::default_ntt_engine, Vector>})
		CHECK(atl_test::aborts([&] { (void)solver({}, 0, atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([] { (void)atl::series_power_sparse(Vector{1, 1}, 2, 8); }));
	for (Root solver : {atl::series_square_root<atl::default_ntt_engine, Vector>, atl::experimental::series_square_root_by_plain_newton<atl::default_ntt_engine, Vector>, atl::experimental::series_square_root_by_logarithm<atl::default_ntt_engine, Vector>})
		CHECK(atl_test::aborts([&] { (void)solver({}, atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([] { (void)atl::series_square_root_sparse(Vector{1, 1}, 8); }));
	CHECK(atl_test::aborts([] { (void)atl::experimental::series_square_root_by_logarithm(Vector{}, 8); }));
	CHECK(atl_test::aborts([] { (void)atl::experimental::series_inverse_by_newton_driver(Vector{}, atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([] { (void)atl::experimental::series_exponential_by_newton_driver(Vector{}, atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([] { (void)atl::polynomial_division(Vector{}, Vector{0, 0}); }));
	CHECK(atl_test::aborts([] { (void)atl::experimental::polynomial_division_by_schoolbook(Vector{}, Vector{}); }));
	Vector oversized(atl::maximum_series_length + 1);
	oversized.back() = Coefficient(1);
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_division(oversized, Vector{1}); }));
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_division(Vector{}, oversized); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_division_by_schoolbook(oversized, Vector{1}); }));
	oversized.back() = Coefficient(0);
	REQUIRE(atl::polynomial_division(oversized, Vector{1}).quotient.empty());
	std::size_t state = 0;
	CHECK(atl_test::aborts([&] { atl::detail::series_newton_lift(state, 0, 1, [](auto &, auto, auto) {}); }));
	CHECK(atl_test::aborts([&] { atl::detail::series_newton_lift(state, 3, 7, [](auto &, auto, auto) {}); }));
	CHECK(atl_test::aborts([&] { atl::detail::series_newton_lift(state, 4, 3, [](auto &, auto, auto) {}); }));
	CHECK(atl_test::aborts([&] { atl::detail::series_newton_lift(state, 1, atl::maximum_series_length + 1, [](auto &, auto, auto) {}); }));
}

namespace {
	/** @brief Sparse power and root at the recurrence-length bound `n - shift <= p`, against repeated schoolbook products. */
	template<atl::static_prime_mod_int Coefficient>
	void check_sparse_shift_bounds() {
		using Vector = std::vector<Coefficient>;
		std::size_t const p = std::size_t(Coefficient::mod());
		for (std::size_t valuation : {0, 1, 2, 3}) {
			Vector f(valuation + p + 2);
			f[valuation] = Coefficient(3);
			f[valuation + 1] = Coefficient(1);
			f[valuation + 4] = Coefficient(5);
			for (std::uint64_t exponent : {1ULL, 2ULL, 3ULL}) {
				std::size_t const shift = valuation * std::size_t(exponent);
				for (std::size_t n : {shift + p - 1, shift + p}) {
					CAPTURE(Coefficient::mod(), valuation, exponent, n);
					Vector expected(n);
					expected[0] = Coefficient(1);
					for (std::uint64_t k = 0; k < exponent; ++k) expected = schoolbook(expected, f, n);
					REQUIRE(atl::series_power_sparse(f, exponent, n) == expected);
				}
				CHECK(atl_test::aborts([&] { (void)atl::series_power_sparse(f, exponent, shift + p + 1); }));
			}
		}
		for (std::size_t half : {0, 1, 2}) {
			Vector root(2 * half + p + 2);
			root[half] = Coefficient(2);
			root[half + 1] = Coefficient(1);
			root[half + 3] = Coefficient(4);
			for (std::size_t n : {2 * half + p - 1, 2 * half + p}) {
				CAPTURE(Coefficient::mod(), half, n);
				auto const square = schoolbook(root, root, n);
				auto const actual = atl::series_square_root_sparse(square, n);
				REQUIRE(actual.has_value());
				REQUIRE(schoolbook(*actual, *actual, n) == square);
				REQUIRE(actual == atl::series_square_root(square, n));
			}
			auto const square = schoolbook(root, root, 2 * half + p + 1);
			CHECK(atl_test::aborts([&] { (void)atl::series_square_root_sparse(square, 2 * half + p + 1); }));
		}
	}
}

TEST_CASE("power_series: sparse power and root bounded by the recurrence length after the shift", "[power_series]") {
	check_sparse_shift_bounds<atl::mod_int32<7>>();
	check_sparse_shift_bounds<atl::mod_int32<11>>();
	using Coefficient = atl::mod_int32<7>;
	using Vector = std::vector<Coefficient>;
	// Rejected by a plain `length <= p` bound, valid by the recurrence length.
	Vector const cube_base{0, 0, 1, 1};
	Vector cube(10);
	cube[6] = Coefficient(1);
	cube[7] = Coefficient(3);
	cube[8] = Coefficient(3);
	cube[9] = Coefficient(1);
	REQUIRE(atl::series_power_sparse(cube_base, 3, 10) == cube);
	REQUIRE(atl::series_power_sparse(Vector{0, 1, 1}, 2, 9) == Vector{0, 0, 1, 2, 1, 0, 0, 0, 0});
	CHECK(atl_test::aborts([] { (void)atl::series_power_sparse(Vector{0, 1, 1}, 2, 10); }));
	Vector expected_one(20);
	expected_one[0] = Coefficient(1);
	REQUIRE(atl::series_power_sparse(Vector{}, 0, 20) == expected_one);
	REQUIRE(atl::series_power_sparse(Vector{0, 1, 1}, 0, 20) == expected_one);
	REQUIRE(atl::series_power_sparse(Vector{}, 5, 20) == Vector(20));
	REQUIRE(atl::series_power_sparse(Vector{0, 1, 1}, 20, 20) == Vector(20));
	REQUIRE(atl::series_power_sparse(Vector{0, 1, 1}, std::numeric_limits<std::uint64_t>::max(), 20) == Vector(20));
	REQUIRE(atl::series_power_sparse(Vector{1, 1}, 3, 0).empty());
	REQUIRE(atl::series_square_root_sparse(Vector{}, 20) == std::optional<Vector>{Vector(20)});
	REQUIRE_FALSE(atl::series_square_root_sparse(Vector{0, 1, 1}, 20).has_value());
	REQUIRE_FALSE(atl::series_square_root_sparse(Vector{3, 1}, 20).has_value());
	auto const shifted_root = atl::series_square_root_sparse(Vector{0, 0, 0, 0, 1, 2, 1}, 11);
	REQUIRE(shifted_root.has_value());
	REQUIRE(*shifted_root == Vector{0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0});
}

TEST_CASE("power_series: sparse entries exceed the transform cap", "[power_series]") {
	using Coefficient = atl::mod_int32<998244353>;
	using Vector = std::vector<Coefficient>;
	std::size_t const length = atl::maximum_series_length + 1;
	auto check_constant = [&](Vector const &value, Coefficient constant) {
		REQUIRE(value.size() == length);
		REQUIRE(value[0] == constant);
		REQUIRE(std::ranges::all_of(std::span(value).subspan(1), [](auto coefficient) { return coefficient == Coefficient(0); }));
	};
	check_constant(atl::series_inverse_sparse(Vector{1}, length), Coefficient(1));
	check_constant(atl::series_division_sparse(Vector{2}, Vector{1}, length), Coefficient(2));
	check_constant(atl::series_logarithm_sparse(Vector{1}, length), Coefficient(0));
	check_constant(atl::series_exponential_sparse(Vector{}, length), Coefficient(1));
	check_constant(atl::series_power_sparse(Vector{1}, 3, length), Coefficient(1));
	auto const root = atl::series_square_root_sparse(Vector{1}, length);
	REQUIRE(root.has_value());
	check_constant(*root, Coefficient(1));
}

TEST_CASE("power_series: internal guards of powers, roots and division", "[power_series]") {
	using Coefficient = atl::mod_int32<7>;
	using Vector = std::vector<Coefficient>;
	using Engine = atl::default_ntt_engine;
	Vector const unit{1}, zero{0}, empty;
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_power_ode<Coefficient>(empty, Coefficient(1)); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_power_ode<Coefficient>(zero, Coefficient(1)); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_power_ode<Coefficient>(Vector(8, 1), Coefficient(1)); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_square_root_coupled<Engine, Coefficient>(empty); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_square_root_coupled<Engine, Coefficient>(zero); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::series_square_root_coupled<Engine, atl::mod_int32<2>>(std::vector<atl::mod_int32<2>>{1}); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_sparse_quotient<Coefficient>(unit, empty, 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_sparse_quotient<Coefficient>(unit, zero, 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_polynomial_long_division(unit, empty); }));
	Vector oversized(atl::maximum_series_length + 1, Coefficient(1));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_power_binary<Engine, Coefficient>(oversized, 0); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_square_root_coupled<Engine, Coefficient>(oversized); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_polynomial_long_division(oversized, unit); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::series_polynomial_long_division(unit, oversized); }));
}
