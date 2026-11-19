#include <atl/nt/mod_int.hpp>
#include <atl/nt/series_calculus.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <forward_list>
#include <random>
#include <vector>
#include "../expect_abort.hpp"
#include "input_only.hpp"

/** @brief Check definitions by repeated addition, then both derivative/integral identities. */
template<atl::static_prime_mod_int Coefficient>
void check_calculus(std::size_t length) {
	std::mt19937 generator(42);
	std::vector<Coefficient> values(length);
	for (auto &value : values) value = Coefficient(generator());
	auto const derivative = atl::series_derivative(values);
	REQUIRE(derivative.size() == (length == 0 ? 0 : length - 1));
	for (std::size_t i = 1; i < length; ++i) {
		Coefficient expected(0);
		for (std::size_t j = 0; j < i; ++j) expected += values[i];
		REQUIRE(derivative[i - 1] == expected);
	}
	REQUIRE(atl::series_derivative(atl_test::input_only<Coefficient>(values)) == derivative);
	if (length >= Coefficient::mod()) return;
	auto const integral = atl::series_integral(values);
	REQUIRE(integral.size() == length + 1);
	REQUIRE(integral[0] == Coefficient(0));
	for (std::size_t i = 1; i <= length; ++i) {
		Coefficient repeated(0);
		for (std::size_t j = 0; j < i; ++j) repeated += integral[i];
		REQUIRE(repeated == values[i - 1]);
	}
	REQUIRE(atl::series_derivative(integral) == values);
	REQUIRE(atl::series_integral(atl_test::input_only<Coefficient>(values)) == integral);
	if (!values.empty()) {
		values[0] = 0;
		REQUIRE(atl::series_integral(derivative) == values);
	}
}

TEST_CASE("series_calculus: definitions, boundaries and round trips", "[series_calculus]") {
	for (std::size_t size : {0, 1}) check_calculus<atl::mod_int32<2>>(size);
	check_calculus<atl::mod_int32<2>>(7);
	check_calculus<atl::mod_int32<3>>(2);
	check_calculus<atl::mod_int32<17>>(16);
	check_calculus<atl::mod_int32<998244353>>(150);
	check_calculus<atl::mod_int64<18446744073709551557ull>>(50);
	using Coefficient = atl::mod_int32<7>;
	REQUIRE(atl::series_derivative(std::vector<Coefficient>{1, 0, 0}) == std::vector<Coefficient>{0, 0});
	REQUIRE(atl::series_integral(std::vector<Coefficient>{1, 0, 0}) == std::vector<Coefficient>{0, 1, 0, 0});
	REQUIRE(atl::series_derivative(std::forward_list<Coefficient>{}).empty());
	REQUIRE(atl::series_integral(std::forward_list<Coefficient>{}) == std::vector<Coefficient>{0});
}

TEST_CASE("series_calculus: asserted length bounds", "[series_calculus]") {
	using Coefficient = atl::mod_int32<2>;
	std::vector<Coefficient> const too_long(2, 1);
	CHECK(atl_test::aborts([&] { (void)atl::series_integral(too_long); }));
	CHECK(atl_test::aborts([&] { (void)atl::series_integral(atl_test::input_only<Coefficient>(too_long)); }));
}
