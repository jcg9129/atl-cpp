#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/factorial_table.hpp>
#include <atl/nt/mod_int.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <vector>
#include "../expect_abort.hpp"

static_assert(atl::static_prime_mod_int<atl::mod_int32<2>>);
static_assert(atl::static_prime_mod_int<atl::mod_int32<998244353>>);
static_assert(atl::static_prime_mod_int<atl::mod_int64<18446744073709551557ull>>);
static_assert(!atl::static_prime_mod_int<atl::mod_int32<1>>);
static_assert(!atl::static_prime_mod_int<atl::mod_int32<9>>);
static_assert(!atl::static_prime_mod_int<atl::dynamic_mod_int<>>);
static_assert(!atl::static_prime_mod_int<int>);

/** @brief Check factorial products and Pascal's addition-only binomial definition independently. */
template<atl::static_prime_mod_int Coefficient>
void check_factorial_table(std::size_t maximum) {
	atl::factorial_table<Coefficient> const table(maximum);
	REQUIRE(table.maximum_index() == maximum);
	Coefficient product(1);
	std::vector<Coefficient> binomial_row(maximum + 1);
	binomial_row[0] = 1;
	for (std::size_t n = 0; n <= maximum; ++n) {
		if (n != 0) product *= Coefficient(n);
		REQUIRE(table.factorial(n) == product);
		REQUIRE(table.inverse_factorial(n) * product == Coefficient(1));
		if (n != 0) REQUIRE(table.inverse_integer(n) * Coefficient(n) == Coefficient(1));
		for (std::size_t k = 0; k <= n; ++k) REQUIRE(table.binomial(n, k) == binomial_row[k]);
		REQUIRE(table.binomial(n, n + 1) == Coefficient(0));
		REQUIRE(table.binomial(n, (std::numeric_limits<std::size_t>::max)()) == Coefficient(0));
		if (n < maximum)
			for (std::size_t k = n + 1; k > 0; --k) binomial_row[k] += binomial_row[k - 1];
	}
}

TEST_CASE("factorial_table: products and Pascal oracle", "[factorial_table]") {
	check_factorial_table<atl::mod_int32<2>>(0);
	check_factorial_table<atl::mod_int32<2>>(1);
	check_factorial_table<atl::mod_int32<3>>(2);
	check_factorial_table<atl::mod_int32<17>>(16);
	check_factorial_table<atl::mod_int32<998244353>>(120);
	check_factorial_table<atl::mod_int64<18446744073709551557ull>>(60);
}

TEST_CASE("factorial_table: asserted boundaries", "[factorial_table]") {
	using Coefficient = atl::mod_int32<7>;
	CHECK(atl_test::aborts([] { atl::factorial_table<Coefficient> const table{7}; }));
	CHECK(atl_test::aborts([] { atl::factorial_table<Coefficient> const table{(std::numeric_limits<std::size_t>::max)()}; }));
	CHECK(atl_test::aborts([] { atl::factorial_table<atl::mod_int64<18446744073709551557ull>> const table{(std::numeric_limits<std::size_t>::max)()}; }));
	atl::factorial_table<Coefficient> const table(3);
	CHECK(atl_test::aborts([&] { (void)table.factorial(4); }));
	CHECK(atl_test::aborts([&] { (void)table.inverse_factorial(4); }));
	CHECK(atl_test::aborts([&] { (void)table.inverse_integer(0); }));
	CHECK(atl_test::aborts([&] { (void)table.inverse_integer(4); }));
	CHECK(atl_test::aborts([&] { (void)table.binomial(4, 0); }));
	CHECK(atl_test::aborts([&] { (void)table.binomial(4, 5); }));
	atl::factorial_table<Coefficient> const empty(0);
	CHECK(atl_test::aborts([&] { (void)empty.inverse_integer(1); }));
}

/** @brief Growing a table in steps must give exactly the entries of one table built at the final size. */
template<atl::static_prime_mod_int Coefficient>
void check_extension(std::vector<std::size_t> const &steps) {
	atl::factorial_table<Coefficient> grown(steps.front());
	std::size_t maximum = steps.front();
	for (std::size_t step : steps) {
		grown.extend(step);
		maximum = (std::max)(maximum, step);
		REQUIRE(grown.maximum_index() == maximum);
	}
	atl::factorial_table<Coefficient> const whole(maximum);
	for (std::size_t i = 0; i <= maximum; ++i) {
		CAPTURE(Coefficient::mod(), i);
		REQUIRE(grown.factorial(i) == whole.factorial(i));
		REQUIRE(grown.inverse_factorial(i) == whole.inverse_factorial(i));
		if (i != 0) REQUIRE(grown.inverse_integer(i) == whole.inverse_integer(i));
		for (std::size_t k = 0; k <= i; ++k) REQUIRE(grown.binomial(i, k) == whole.binomial(i, k));
	}
}

TEST_CASE("factorial_table: extend grows only the suffix", "[factorial_table]") {
	check_extension<atl::mod_int32<2>>({0, 1, 0});
	check_extension<atl::mod_int32<7>>({0, 1, 2, 3, 4, 5, 6});
	check_extension<atl::mod_int32<7>>({2, 6, 3});
	check_extension<atl::mod_int32<17>>({0, 16});
	check_extension<atl::mod_int32<998244353>>({5, 3, 17, 64, 65, 200, 1000});
	check_extension<atl::mod_int64<18446744073709551557ull>>({1, 30, 31, 60});
	std::vector<std::size_t> one_at_a_time(301);
	for (std::size_t i = 0; i < one_at_a_time.size(); ++i) one_at_a_time[i] = i;
	check_extension<atl::mod_int32<998244353>>(one_at_a_time);
}

TEST_CASE("factorial_table: asserted extension boundaries", "[factorial_table]") {
	using Coefficient = atl::mod_int32<7>;
	atl::factorial_table<Coefficient> table(3);
	table.extend(6);
	REQUIRE(table.maximum_index() == 6);
	table.extend(6);
	table.extend(0);
	REQUIRE(table.maximum_index() == 6);
	CHECK(atl_test::aborts([&] { table.extend(7); }));
	CHECK(atl_test::aborts([&] { table.extend((std::numeric_limits<std::size_t>::max)()); }));
	atl::factorial_table<atl::mod_int64<18446744073709551557ull>> wide(2);
	CHECK(atl_test::aborts([&] { wide.extend((std::numeric_limits<std::size_t>::max)()); }));
}
