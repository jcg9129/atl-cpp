#include <atl/experimental/nt/square_root_mod_variants.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/square_root_mod.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include "../expect_abort.hpp"

using std::uint64_t;

/** @brief Independent product oracle by double-and-add on plain residues. */
constexpr uint64_t reference_product(uint64_t left, uint64_t right, uint64_t prime) {
	uint64_t result = 0;
	while (right != 0) {
		if (right % 2 != 0) result = result >= prime - left ? result - (prime - left) : result + left;
		right /= 2;
		left = left >= prime - left ? left - (prime - left) : left + left;
	}
	return result;
}

/** @brief Euler-criterion oracle using plain residues and independent exponentiation. */
constexpr uint64_t reference_power(uint64_t value, uint64_t exponent, uint64_t prime) {
	uint64_t result = 1;
	while (exponent != 0) {
		if (exponent % 2 != 0) result = reference_product(result, value, prime);
		exponent /= 2;
		value = reference_product(value, value, prime);
	}
	return result;
}

TEST_CASE("square_root_mod: Tonelli-Shanks and Cipolla against exhaustive search", "[square_root_mod]") {
	for (uint64_t prime : {2, 3, 5, 7, 13, 17, 41, 97, 193, 257}) {
		for (uint64_t value = 0; value < prime; ++value) {
			std::optional<uint64_t> expected;
			for (uint64_t candidate = 0; candidate < prime; ++candidate)
				if (candidate * candidate % prime == value) {
					expected = candidate;
					break;
				}
			REQUIRE(atl::square_root_mod(value, prime) == expected);
			REQUIRE(atl::experimental::square_root_mod_by_cipolla(value, prime) == expected);
		}
	}
}

/** @brief Test field overloads independently of the raw interface using known squared representatives. */
template<atl::static_prime_mod_int Coefficient>
void check_field() {
	uint64_t const prime = Coefficient::mod();
	for (uint64_t candidate : {uint64_t(0), uint64_t(1), uint64_t(2), prime / 2, prime - 1}) {
		candidate %= prime;
		Coefficient const square(reference_product(candidate, candidate, prime));
		Coefficient const expected(std::ranges::min(candidate, prime - candidate));
		REQUIRE(atl::square_root_mod(square) == expected);
		REQUIRE(atl::square_root_mod(square)->value() == atl::square_root_mod(square.value(), prime));
		REQUIRE(atl::experimental::square_root_mod_by_cipolla(square) == expected);
	}
	std::mt19937_64 random(prime);
	for (int iteration = 0; iteration < 128; ++iteration) {
		uint64_t const value = prime < 128 ? uint64_t(iteration) % prime : random() % prime;
		auto const field_root = atl::square_root_mod(Coefficient(value));
		auto const integer_root = atl::square_root_mod(value, prime);
		REQUIRE(field_root.has_value() == integer_root.has_value());
		if (field_root) {
			REQUIRE(field_root->value() == *integer_root);
			REQUIRE(*integer_root <= prime / 2);
			REQUIRE(reference_product(*integer_root, *integer_root, prime) == value);
		}
	}
	for (uint64_t value = 1; value < std::ranges::min(prime, uint64_t(50)); ++value) {
		if (reference_power(value, (prime - 1) / 2, prime) == 1) continue;
		REQUIRE_FALSE(atl::square_root_mod(Coefficient(value)));
		REQUIRE_FALSE(atl::square_root_mod(value, prime));
		REQUIRE_FALSE(atl::experimental::square_root_mod_by_cipolla(Coefficient(value)));
	}
}

TEST_CASE("square_root_mod: static field overloads", "[square_root_mod]") {
	check_field<atl::mod_int32<2>>();
	check_field<atl::mod_int32<3>>();
	check_field<atl::mod_int32<17>>();
	check_field<atl::mod_int32<41>>();
	check_field<atl::mod_int32<97>>();
	check_field<atl::mod_int32<257>>();
	check_field<atl::mod_int32<998244353>>();
	check_field<atl::mod_int32<1000000007>>();
	check_field<atl::mod_int32<4294967291u>>();
	check_field<atl::mod_int64<4294967291ull>>();
	check_field<atl::mod_int64<4294967311ull>>();
	check_field<atl::mod_int64<2305843009213693951ull>>();
	check_field<atl::mod_int64<4179340454199820289ull>>();
	check_field<atl::mod_int64<18446744069414584321ull>>();
	check_field<atl::mod_int64<18446744073709551557ull>>();
}

TEST_CASE("square_root_mod: large primes, Euler criterion and overflow boundaries", "[square_root_mod]") {
	std::mt19937_64 random(6);
	for (uint64_t prime : {uint64_t(998244353), uint64_t(1000000007), uint64_t(4179340454199820289ull),
						   uint64_t(4611686018427387847ull), uint64_t(18446744069414584321ull), uint64_t(18446744073709551557ull)}) {
		REQUIRE(atl::is_prime(prime));
		for (uint64_t candidate : {uint64_t(0), uint64_t(1), prime / 2, prime - 2, prime - 1}) {
			auto const square = reference_product(candidate, candidate, prime);
			auto const expected = std::ranges::min(candidate, prime - candidate);
			REQUIRE(atl::square_root_mod(square, prime) == expected);
			REQUIRE(atl::experimental::square_root_mod_by_cipolla(square, prime) == expected);
		}
		for (int iteration = 0; iteration < 200; ++iteration) {
			uint64_t const value = random() % prime;
			bool const has_root = value == 0 || reference_power(value, (prime - 1) / 2, prime) == 1;
			for (auto root : {atl::square_root_mod(value, prime), atl::experimental::square_root_mod_by_cipolla(value, prime)}) {
				REQUIRE(root.has_value() == has_root);
				if (root) {
					REQUIRE(*root <= prime / 2);
					REQUIRE(reference_product(*root, *root, prime) == value);
				}
			}
		}
	}
}

TEST_CASE("square_root_mod: asserted preconditions for each contender", "[square_root_mod]") {
	using Solver = std::optional<uint64_t> (*)(uint64_t, uint64_t);
	for (auto solver : {Solver(atl::square_root_mod), Solver(atl::experimental::square_root_mod_by_cipolla)}) {
		for (uint64_t composite : {0, 1, 4, 9, 561})
			CHECK(atl_test::aborts([&] { (void)solver(0, composite); }));
		CHECK(atl_test::aborts([&] { (void)solver(2, 2); }));
		CHECK(atl_test::aborts([&] { (void)solver(7, 7); }));
		CHECK(atl_test::aborts([&] { (void)solver(std::numeric_limits<uint64_t>::max(), 18446744073709551557ull); }));
	}
}
