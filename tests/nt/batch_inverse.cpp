#include <atl/nt/batch_inverse.hpp>
#include <atl/nt/mod_int.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <forward_list>
#include <ranges>
#include <span>
#include <vector>
#include "../expect_abort.hpp"
#include "input_only.hpp"

using test_mint = atl::mod_int32<17>;
static_assert(atl::prime_mod_int_range<std::vector<test_mint>>);
static_assert(atl::prime_mod_int_range<std::span<test_mint const>>);
static_assert(atl::prime_mod_int_range<atl_test::input_only<test_mint>>);
static_assert(!std::ranges::forward_range<atl_test::input_only<test_mint> const>);
static_assert(!atl::prime_mod_int_range<std::vector<int>>);
static_assert(!atl::prime_mod_int_range<std::vector<atl::mod_int32<15>>>);
static_assert(!atl::prime_mod_int_range<std::ranges::basic_istream_view<test_mint, char>>);

/** @brief Compare inverses with exhaustive multiplication in a small prime field. */
template<atl::static_prime_mod_int Field>
void check_batch() {
	std::vector<Field> values;
	std::vector<Field> expected;
	for (unsigned i = 1; i < Field::mod(); ++i) {
		values.emplace_back(i);
		for (unsigned j = 1; j < Field::mod(); ++j)
			if (i * j % Field::mod() == 1) expected.emplace_back(j);
	}
	REQUIRE(atl::batch_inverse(values) == expected);
	REQUIRE(atl::batch_inverse(atl_test::input_only<Field>(values)) == expected);
	REQUIRE(atl::batch_inverse(std::forward_list<Field>(values.begin(), values.end())) == expected);
}

TEST_CASE("batch_inverse: exhaustive definition and const input ranges", "[batch_inverse]") {
	check_batch<atl::mod_int32<2>>();
	check_batch<atl::mod_int32<3>>();
	check_batch<test_mint>();
	REQUIRE(atl::batch_inverse(std::vector<test_mint>{}).empty());
	REQUIRE(atl::batch_inverse(std::forward_list<test_mint>{}).empty());
	REQUIRE(atl::batch_inverse(std::vector<test_mint>{1}) == std::vector<test_mint>{1});
	REQUIRE(atl::batch_inverse(std::vector<test_mint>{3, 3, 3}) == std::vector<test_mint>{6, 6, 6});
	using Wide = atl::mod_int64<18446744073709551557ull>;
	std::vector<Wide> const values{1, 2, -1, -2, 123456789};
	auto const result = atl::batch_inverse(values);
	for (std::size_t i = 0; i < values.size(); ++i) REQUIRE(values[i] * result[i] == Wide(1));
}

TEST_CASE("batch_inverse: asserted preconditions", "[batch_inverse]") {
	for (auto const &values : {std::vector<test_mint>{0}, {0, 1}, {1, 0}, {1, 0, 1}}) {
		CHECK(atl_test::aborts([&] { (void)atl::batch_inverse(values); }));
		CHECK(atl_test::aborts([&] { (void)atl::batch_inverse(atl_test::input_only<test_mint>(values)); }));
	}
}
