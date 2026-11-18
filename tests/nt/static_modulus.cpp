#include <atl/config.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/static_modulus.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <limits>
#include <random>

using atl::uint32;
using atl::uint64;

TEST_CASE("static_modulus: Mersenne detection", "[static_modulus]") {
	STATIC_REQUIRE(atl::static_modulus<uint32(2147483647)>::is_mersenne);
	STATIC_REQUIRE(atl::static_modulus<(uint64(1) << 61) - 1>::is_mersenne);
	STATIC_REQUIRE(!atl::static_modulus<uint32(998244353)>::is_mersenne);
	STATIC_REQUIRE(!atl::static_modulus<uint64(1000000007)>::is_mersenne);
}

TEMPLATE_TEST_CASE_SIG("static_modulus: 32-bit ops match reference", "[static_modulus]", ((uint32 M), M),
					   998244353, 2147483647, 3) {
	using mod = atl::static_modulus<M>;
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (int t = 0; t < 20000; ++t) {
		uint32 a = uint32(rng() % M), b = uint32(rng() % M);
		REQUIRE(mod::add(a, b) == (a + b) % M);
		REQUIRE(mod::sub(a, b) == (a + M - b) % M);
		REQUIRE(mod::mul(a, b) == uint32((uint64(a) * b) % M));
	}
}

TEST_CASE("static_modulus: 64-bit mul matches reference (2^61-1)", "[static_modulus]") {
	using mod = atl::static_modulus<(uint64(1) << 61) - 1>;
	constexpr uint64 M = mod::mod;
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (int t = 0; t < 20000; ++t) {
		uint64 a = rng() % M, b = rng() % M;
		REQUIRE(mod::mul(a, b) == uint64((atl::uint128(a) * b) % M));
	}
}

TEST_CASE("static_modulus: 128-bit mul falls back (no truncation)", "[static_modulus]") {
	// There is no 256-bit product type; double-and-add must retain the complete mathematical product.
	constexpr atl::uint128 M = (atl::uint128(1) << 100) + 277;
	using mod = atl::static_modulus<M>;
	std::mt19937_64 rng(2024);
	for (int t = 0; t < 5000; ++t) {
		atl::uint128 a = ((atl::uint128(rng()) << 64) | rng()) % M;
		atl::uint128 b = ((atl::uint128(rng()) << 64) | rng()) % M;
		REQUIRE((mod::mul(a, b) == atl::mul_mod_binary<atl::uint128>(a, b, M)));
	}
}

TEST_CASE("static_modulus: power matches naive", "[static_modulus]") {
	using mod = atl::static_modulus<uint32(2147483647)>;
	constexpr uint32 M = mod::mod;
	std::mt19937_64 rng(12345);
	for (int t = 0; t < 2000; ++t) {
		uint32 a = uint32(rng() % M);
		uint64 e = rng() % 40;
		uint32 naive = 1;
		for (uint64 i = 0; i < e; ++i) naive = uint32((uint64(naive) * a) % M);
		REQUIRE(mod::power(a, e) == naive);
	}
}

TEST_CASE("static_modulus: agrees with mod_int", "[static_modulus]") {
	constexpr uint32 M = 2147483647;
	using mod = atl::static_modulus<M>;
	std::mt19937_64 rng(999);
	for (int t = 0; t < 20000; ++t) {
		uint32 a = uint32(rng() % M), b = uint32(rng() % M);
		atl::mod_int32<M> ma = a, mb = b;
		REQUIRE(mod::mul(a, b) == (ma * mb).value());
		REQUIRE(mod::add(a, b) == (ma + mb).value());
		REQUIRE(mod::sub(a, b) == (ma - mb).value());
		REQUIRE(mod::neg(a) == (-ma).value());
		atl::mod_int32<M> inc = ma, dec = ma;
		++inc, --dec;
		REQUIRE(mod::incr(a) == inc.value());
		REQUIRE(mod::decr(a) == dec.value());
		if (a != 0) REQUIRE(mod::inverse(a) == ma.inverse().value());
	}
}

TEST_CASE("static_modulus: 64-bit non-Mersenne moduli multiply exactly", "[static_modulus]") {
	std::mt19937_64 rng(17);
	auto check = [&]<uint64 M>()
		requires(M > 0)
	{
		for (int i = 0; i < 20000; ++i) {
			uint64 const a = rng() % M, b = rng() % M;
			REQUIRE(atl::static_modulus<M>::mul(a, b) == uint64(atl::uint128(a) * b % M));
		}
		REQUIRE(atl::static_modulus<M>::mul(M - 1, M - 1) == uint64(atl::uint128(M - 1) * (M - 1) % M));
	};
	check.operator()<1'000'000'000'000'000'000ull>();
	check.operator()<1'000'000'000'000'000'003ull>();
	check.operator()<18446744073709551557ull>();
	check.operator()<~uint64(0)>();
	check.operator()<(uint64(1) << 63) + 1>();
	STATIC_REQUIRE(atl::static_modulus<1'000'000'000'000'000'000ull>::mul(999'999'999'999'999'999ull, 2)
				   == 999'999'999'999'999'998ull);
}
