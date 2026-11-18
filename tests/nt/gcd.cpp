#include <atl/nt/gcd.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <limits>
#include <numeric>
#include <random>

using atl::int128;
using atl::int32;
using atl::int64;
using atl::uint32;
using atl::uint64;

TEST_CASE("Test extended gcd", "[gcd]") {
	for (int32 a = -100; a < 100; ++a) {
		for (int32 b = -100; b < 100; ++b) {
			auto [g, x, y] = atl::extended_gcd(a, b);
			REQUIRE(g == static_cast<uint32>(std::abs(std::gcd(a, b))));
			REQUIRE(int64(a) * x + int64(b) * y == int64(g));
		}
	}
	for (int64 i = 0; i < 100; ++i) {
		for (int64 j = 0; j < 100; ++j) {
			int64 a = (std::numeric_limits<int64>::min)() + i;
			int64 b = (std::numeric_limits<int64>::max)() - j;
			auto [g, x, y] = atl::extended_gcd(a, b);
			REQUIRE(int128(a) % g == 0);
			REQUIRE(int128(b) % g == 0);
			REQUIRE(int128(a) * x + int128(b) * y == g);
		}
	}
}

TEST_CASE("Test inverse mod", "[gcd]") {
	for (uint32 m = 2; m < 100; ++m) {
		for (uint32 a = 1; a < m; ++a) {
			if (std::gcd(a, m) == 1) {
				auto inv = atl::inverse_mod(a, m);
				REQUIRE(int64(a) * inv % m == 1);
			}
		}
	}
}

TEST_CASE("mod_int inverse", "[gcd]") {
	auto seed = GENERATE(take(1, random(uint64(0), (std::numeric_limits<atl::uint64>::max)())));

	uint32 const mod = 1000000007;
	using Z = atl::mod_int<uint32, mod>;

	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
	std::uniform_int_distribution<> dist(1, mod - 1);

	for (int i = 0; i < 1000; ++i) {
		auto a = Z(dist(rng));
		auto b = 1 / a;
		auto c = a * b;
		REQUIRE(c == Z(1));
	}
}
