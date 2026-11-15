#include <atl/types/detail/int256.hpp>
#include <catch2/catch.hpp>
#include <cstdint>
#include <random>

#if ATL_HAS_INT128
using atl::int128;
using atl::uint64;
using atl::detail::compare_products;
using atl::detail::int256;

namespace {
	int128 random_int128(std::mt19937_64 &rng, int const bits) {
		// magnitude < 2^bits, either sign
		atl::uint128 magnitude = 0;
		for (int have = 0; have < bits; have += 64)
			magnitude = (magnitude << 64) | rng();
		magnitude &= (atl::uint128(1) << bits) - 1;
		int128 const v = int128(magnitude);
		return rng() % 2 ? v : -v;
	}
}

TEST_CASE("int256: products that fit int128 reproduce it exactly", "[types]") {
	auto seed = GENERATE(take(10, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 20000; ++it) {
		int128 const a = random_int128(rng, 63), b = random_int128(rng, 63);
		int128 const truth = a * b; // fits: |a*b| < 2^126
		int256 const wide = int256::product(a, b);
		// low two limbs must equal the int128 value, upper limbs its sign fill
		atl::uint128 const low = atl::uint128(wide.limb[0]) | (atl::uint128(wide.limb[1]) << 64);
		REQUIRE(low == atl::uint128(truth));
		uint64 const fill = truth < 0 ? ~uint64(0) : 0;
		REQUIRE(wide.limb[2] == fill);
		REQUIRE(wide.limb[3] == fill);
		REQUIRE(wide.sign() == (truth > 0 ? 1 : truth < 0 ? -1
														  : 0));
	}
}

TEST_CASE("int256: algebraic identities at full width", "[types]") {
	auto seed = GENERATE(take(10, random(std::uint64_t(0),
										 std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 20000; ++it) {
		int128 const a = random_int128(rng, 126), b = random_int128(rng, 126);
		int128 const c = random_int128(rng, 125);
		// commutativity, sign symmetry, negation
		REQUIRE(int256::product(a, b) == int256::product(b, a));
		REQUIRE(int256::product(-a, b) == -int256::product(a, b));
		REQUIRE(int256::product(a, -b) == -int256::product(a, b));
		// distributivity through the accumulator: a*(b + c) = a*b + a*c
		// (guard the b + c overflow by halving)
		int128 const bh = b / 2, ch = c / 2;
		REQUIRE(int256::product(a, bh + ch)
				== int256::product(a, bh) + int256::product(a, ch));
		// subtraction inverse and zero sign
		int256 const p = int256::product(a, b);
		REQUIRE((p - p).sign() == 0);
		REQUIRE((p - p) == int256{});
		// compare_products consistency with the sign of the difference
		REQUIRE(compare_products(a, b, a, b) == 0);
		REQUIRE(compare_products(a, b, b, a) == 0);
	}
}

TEST_CASE("int256: ordering of adjacent huge products", "[types]") {
	// (2^126 - 1)^2 vs (2^126 - 1)^2 - tiny: adjacent magnitudes at the top
	int128 const big = (int128(1) << 126) - 1;
	REQUIRE(compare_products(big, big, big, big - 1) == 1);
	REQUIRE(compare_products(big, big - 1, big, big) == -1);
	REQUIRE(compare_products(big, big, -big, -big) == 0);
	REQUIRE(compare_products(-big, big, big, -big) == 0);
	REQUIRE(compare_products(big, -big, big, big) == -1);
	// carries across every limb boundary
	int128 const p2_64 = int128(1) << 64;
	int256 x = int256::product(p2_64, p2_64); // 2^128: limb[2] == 1
	REQUIRE(x.limb[2] == 1);
	REQUIRE((x - x).sign() == 0);
	x += int256::product(int128(1), int128(-1)); // 2^128 - 1: borrows down
	REQUIRE(x.limb[2] == 0);
	REQUIRE(x.limb[1] == ~uint64(0));
	REQUIRE(x.limb[0] == ~uint64(0));
	REQUIRE(x.sign() == 1);
}

TEST_CASE("int256: products at the int128 extremes", "[int256]") {
	using atl::int128;
	using atl::uint128;
	using atl::detail::int256;
	int128 const min = int128(uint128(1) << 127), max = int128((uint128(1) << 127) - 1);
	int256 const min_min = int256::product(min, min); // 2^254
	REQUIRE((min_min.limb[0] | min_min.limb[1] | min_min.limb[2]) == 0);
	REQUIRE(min_min.limb[3] == atl::uint64(1) << 62);
	int256 const min_neg1 = int256::product(min, -1); // 2^127
	REQUIRE(min_neg1.limb[0] == 0);
	REQUIRE(min_neg1.limb[1] == atl::uint64(1) << 63);
	REQUIRE((min_neg1.limb[2] | min_neg1.limb[3]) == 0);
	REQUIRE(int256::product(min, 1) == int256::product(-1, max) - int256::product(1, 1)); // -max - 1
	REQUIRE(int256::product(max, min).sign() == -1);
	REQUIRE(int256::product(min, max) == int256::product(max, min));
	REQUIRE((int256::product(max, max) - int256::product(max, max)).sign() == 0);
	REQUIRE(atl::detail::compare_products(min, max, max, min) == 0);
	REQUIRE(atl::detail::compare_products(min, min, max, max) == 1);
	REQUIRE(atl::detail::compare_products(min, 1, max, -1) == -1);
}

#else
TEST_CASE("int256: unavailable without native int128", "[types]") {
	STATIC_REQUIRE(ATL_HAS_INT128 == 0);
}
#endif
