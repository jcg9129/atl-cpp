#include <atl/experimental/nt/sieve_variants.hpp>
#include <atl/nt/basic.hpp>
#include <atl/nt/sieve.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>
#include "../expect_abort.hpp"

using atl::int32;
using atl::int64;
using atl::uint32;
using atl::uint64;

TEST_CASE("Prime ranges", "[sieve]") {
	REQUIRE(atl::prime_range(100)
			== std::vector<uint32>({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
									79, 83, 89, 97}));

	REQUIRE(atl::prime_range(97)
			== std::vector<uint32>({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
									79, 83, 89}));
}

TEST_CASE("sieve primes and is_prime", "[sieve]") {
	std::size_t const N = 1000;
	atl::sieve sv(N);

	REQUIRE(sv.primes() == atl::prime_range(N));
	for (std::size_t i = 0; i < N; ++i) {
		REQUIRE(sv.is_prime(i) == atl::is_prime(i));
	}
}

TEST_CASE("sieve factorization matches factorization", "[sieve]") {
	std::size_t const N = 2000;
	atl::sieve sv(N);

	for (std::size_t i = 1; i < N; ++i) {
		REQUIRE(std::ranges::equal(sv.factorization(i), atl::factorization(i)));
		if (i >= 2) REQUIRE(sv.min_prime_factor(i) == atl::factorization(i).front().prime);

		std::vector<uint32> distinct;
		for (auto [p, e] : atl::factorization(i)) distinct.push_back(uint32(p));
		REQUIRE(std::ranges::equal(sv.prime_factors(i), distinct));
	}
}

TEST_CASE("sieve divisors", "[sieve]") {
	std::size_t const N = 1000;
	atl::sieve sv(N);

	for (std::size_t i = 1; i < N; ++i) {
		auto divs = sv.divisors(i);

		REQUIRE(std::ranges::is_sorted(divs));
		REQUIRE(std::ranges::adjacent_find(divs) == divs.end());
		for (uint64 d : divs) REQUIRE(i % d == 0);

		uint64 brute = 0;
		for (uint64 d = 1; d <= i; ++d)
			if (i % d == 0) ++brute;
		REQUIRE(divs.size() == brute);
	}
}

TEST_CASE("sieve: a bound past the uint32 range is asserted", "[sieve]") {
	if constexpr (std::numeric_limits<std::size_t>::digits > 32) {
		std::size_t const too_big = static_cast<std::size_t>(uint64(1) << 32);
		CHECK(atl_test::aborts([&] { (void)atl::sieve_info(too_big); }));
		CHECK(atl_test::aborts([&] { (void)atl::prime_range(too_big); }));
		CHECK(atl_test::aborts([&] { atl::sieve s(too_big); }));
	}
}

TEST_CASE("sieve_info records compare as values", "[sieve]") {
	using info = atl::sieve_item_info;
	std::vector<info> const expected{
		{0, 0, 0},
		{1, 0, 1},
		{2, 1, 2},
		{3, 1, 3},
		{2, 2, 4},
		{5, 1, 5},
		{2, 1, 2},
		{7, 1, 7},
		{2, 3, 8},
		{3, 2, 9},
		{2, 1, 2},
		{11, 1, 11},
		{2, 2, 4}};
	REQUIRE(atl::sieve_info(13) == expected);
}

TEST_CASE("prime_range: the segmented sieve matches the linear sieve at every size and block edge", "[sieve]") {
	atl::sieve const s(400'000);
	auto expected = [&](std::size_t n) {
		std::vector<uint32> want;
		for (uint32 p : s.primes())
			if (p < n) want.push_back(p);
		return want;
	};
	for (std::size_t n = 0; n < 3000; ++n) REQUIRE(atl::prime_range(n) == expected(n));
	for (std::size_t n : {std::size_t(131071), std::size_t(131072), std::size_t(131073), std::size_t(131074),
						  std::size_t(262143), std::size_t(262145), std::size_t(361), std::size_t(362),
						  std::size_t(398161), std::size_t(398162), std::size_t(399999)})
		REQUIRE(atl::prime_range(n) == expected(n));
}

TEST_CASE("sieve contenders agree with the linear table oracle", "[sieve]") {
	for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(13),
						  std::size_t(1000), std::size_t(131071), std::size_t(131074)}) {
		atl::sieve const table(n);
		REQUIRE(atl::experimental::sieve_info_by_least_prime_comparison(n) == atl::sieve_info(n));
		REQUIRE(atl::experimental::prime_range_by_unsegmented_sieve(n) == table.primes());
	}
}
