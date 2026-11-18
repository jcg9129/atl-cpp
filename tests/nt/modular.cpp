#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <limits>
#include <random>
#include "../expect_abort.hpp"

using atl::uint64;

TEST_CASE("mul_mod_binary matches the widening mul_mod", "[modular]") {
	// Double-and-add is an independent oracle for the native or software widened product.
	auto seed = GENERATE(take(1, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);

	uint64 const moduli[] = {
		2,
		1000000007ull,
		(uint64(1) << 62) + 1,
		(std::numeric_limits<uint64>::max)() - 4,
	};

	for (uint64 m : moduli) {
		for (int t = 0; t < 3000; ++t) {
			uint64 a = rng() % m, b = rng() % m;
			REQUIRE(atl::mul_mod_binary(a, b, m) == atl::mul_mod_unchecked(a, b, m));
		}
	}
}

TEST_CASE("pow_mod matches a reference and satisfies Fermat", "[modular]") {
	using atl::uint32;

	REQUIRE(atl::pow_mod<uint64>(2, 10, 1000000007) == 1024);
	REQUIRE(atl::pow_mod<uint64>(7, 0, 13) == 1);
	REQUIRE(atl::pow_mod<uint64>(5, 3, 1) == 0);

	auto ref = [](uint64 b, uint64 e, uint64 m) {
		b %= m;
		uint64 r = 1 % m;
		for (uint64 i = 0; i < e; ++i) r = atl::mul_mod_unchecked(r, b, m);
		return r;
	};
	std::mt19937_64 rng(12345);
	for (uint64 m : {uint64(2), uint64(998244353), (uint64(1) << 61) - 1}) {
		for (int t = 0; t < 2000; ++t) {
			uint64 b = rng(), e = rng() % 200;
			REQUIRE(atl::pow_mod<uint64>(b, e, m) == ref(b, e, m));
		}
	}

	uint32 const p = 1000000007u;
	for (uint32 a : {2u, 3u, 123456u, 999999u})
		REQUIRE(atl::pow_mod<uint32>(a, p - 1, p) == 1);
}

TEST_CASE("Barrett reducers cover the full modulus range", "[modular][boundary]") {
	std::mt19937_64 rng(79);
	for (atl::uint32 modulus : {1u, 3u, 2147483647u, 4294967291u, 4294967295u}) {
		atl::barrett const reducer(modulus);
		REQUIRE(reducer.mod() == modulus);
		REQUIRE(reducer.mul(modulus - 1, modulus - 1) == 1u % modulus);
		for (int i = 0; i < 2000; ++i) {
			auto const a = atl::uint32(rng() % modulus), b = atl::uint32(rng() % modulus);
			REQUIRE(reducer.mul(a, b) == atl::uint32(atl::uint64(a) * b % modulus));
		}
	}
	for (uint64 modulus : {uint64(1), uint64(4294967291u), (uint64(1) << 61) - 1, uint64(18446744073709551557ull),
						   ~uint64(0)}) {
		atl::barrett64 const reducer(modulus);
		REQUIRE(reducer.mod() == modulus);
		REQUIRE(reducer.mul(modulus - 1, modulus - 1) == 1u % modulus);
		REQUIRE(atl::mul_mod_unchecked(modulus - 1, modulus - 1, modulus) == 1u % modulus);
		REQUIRE(atl::mul_mod_binary(modulus - 1, modulus - 1, modulus) == 1u % modulus);
		for (int i = 0; i < 2000; ++i) {
			uint64 const a = rng() % modulus, b = rng() % modulus;
			REQUIRE(reducer.mul(a, b) == uint64(atl::uint128(a) * b % modulus));
		}
	}
}

TEST_CASE("modular runtime preconditions are asserted", "[modular][boundary]") {
	CHECK(atl_test::aborts([] { (void)atl::barrett{0}; }));
	CHECK(atl_test::aborts([] { (void)atl::barrett64{0}; }));
	CHECK(atl_test::aborts([] { (void)atl::barrett{7}.mul(7, 0); }));
	CHECK(atl_test::aborts([] { (void)atl::barrett64{7}.mul(0, 7); }));
	CHECK(atl_test::aborts([] { (void)atl::reduce_mod(1, 0u); }));
	CHECK(atl_test::aborts([] { (void)atl::pow_mod(1u, 0, 0u); }));
	CHECK(atl_test::aborts([] { (void)atl::inverse_mod(1u, 0u); }));
	CHECK(atl_test::aborts([] { (void)atl::inverse_mod(2u, 4u); }));
}
