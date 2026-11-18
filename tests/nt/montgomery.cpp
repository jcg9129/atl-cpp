#include <atl/nt/montgomery.hpp>
#include <atl/nt/static_modulus.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <limits>
#include <random>

using atl::uint32;
using atl::uint64;

TEMPLATE_TEST_CASE_SIG("montgomery: agrees with static_modulus (32-bit)", "[montgomery]", ((uint32 M), M), 998244353,
					   1000000007) {
	using mg = atl::montgomery<M>;
	using sm = atl::static_modulus<M>;
	auto seed = GENERATE(take(3, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	REQUIRE(mg::from_form(mg::one()) == 1);
	for (int t = 0; t < 20000; ++t) {
		uint32 a = uint32(rng() % M), b = uint32(rng() % M);
		uint32 fa = mg::to_form(a), fb = mg::to_form(b);
		REQUIRE(mg::from_form(fa) == a);
		REQUIRE(mg::from_form(mg::mul(fa, fb)) == sm::mul(a, b));
		REQUIRE(mg::from_form(mg::add(fa, fb)) == sm::add(a, b));
		REQUIRE(mg::from_form(mg::sub(fa, fb)) == sm::sub(a, b));
		REQUIRE(mg::from_form(mg::neg(fa)) == sm::neg(a));
		REQUIRE(mg::from_form(mg::incr(fa)) == sm::incr(a));
		REQUIRE(mg::from_form(mg::decr(fa)) == sm::decr(a));
		if (a != 0) REQUIRE(mg::mul(fa, mg::inverse(fa)) == mg::one());
	}
	for (int t = 0; t < 2000; ++t) {
		uint32 a = uint32(rng() % M);
		uint64 e = rng() % 64;
		REQUIRE(mg::from_form(mg::power(mg::to_form(a), e)) == sm::power(a, e));
	}
}

TEMPLATE_TEST_CASE_SIG("montgomery: agrees with static_modulus (64-bit)", "[montgomery]", ((uint64 M), M),
					   1000000000000000003, 9223372036854775783) {
	using mg = atl::montgomery<M>;
	using sm = atl::static_modulus<M>;
	auto seed = GENERATE(take(3, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	REQUIRE(mg::from_form(mg::one()) == 1);
	for (int t = 0; t < 20000; ++t) {
		uint64 a = rng() % M, b = rng() % M;
		uint64 fa = mg::to_form(a), fb = mg::to_form(b);
		REQUIRE(mg::from_form(fa) == a);
		REQUIRE(mg::from_form(mg::mul(fa, fb)) == sm::mul(a, b));
		REQUIRE(mg::from_form(mg::add(fa, fb)) == sm::add(a, b));
		REQUIRE(mg::from_form(mg::sub(fa, fb)) == sm::sub(a, b));
		REQUIRE(mg::from_form(mg::neg(fa)) == sm::neg(a));
		if (a != 0) REQUIRE(mg::mul(fa, mg::inverse(fa)) == mg::one());
	}
	for (int t = 0; t < 2000; ++t) {
		uint64 a = rng() % M;
		uint64 e = rng() % 64;
		REQUIRE(mg::from_form(mg::power(mg::to_form(a), e)) == sm::power(a, e));
	}
}

// Subtraction must avoid forming a + Mod when the modulus exceeds half the word range.
TEMPLATE_TEST_CASE_SIG("montgomery: moduli above half the type range", "[montgomery]", ((auto M), M),
					   uint32(4294967291u), uint64(18446744073709551557ull)) {
	using mg = atl::montgomery<M>;
	using sm = atl::static_modulus<M>;
	using value_type = decltype(M);
	std::mt19937_64 rng(42);
	for (int t = 0; t < 20000; ++t) {
		value_type a = value_type(rng() % M), b = value_type(rng() % M);
		if (t % 4 == 0) a = value_type(M - 1 - a % 16), b = value_type(b % 16);
		value_type fa = mg::to_form(a), fb = mg::to_form(b);
		REQUIRE(mg::from_form(fa) == a);
		REQUIRE(mg::from_form(mg::sub(fa, fb)) == sm::sub(a, b));
		REQUIRE(mg::from_form(mg::sub(fb, fa)) == sm::sub(b, a));
		REQUIRE(mg::from_form(mg::mul(fa, fb)) == sm::mul(a, b));
	}
	REQUIRE(sm::sub(0, M - 1) == 1);
	REQUIRE(sm::sub(1, M - 1) == 2);
}

TEST_CASE("Montgomery lazy products respect their largest modulus", "[montgomery][boundary]") {
	auto check = []<auto Modulus>()
		requires atl::unsigned_integral<decltype(Modulus)> && atl::traits::has_widen_mul_v<decltype(Modulus)>
			  && (Modulus > 0) && ((Modulus & 1) != 0)
			  && (Modulus <= (std::numeric_limits<decltype(Modulus)>::max)() / 4)
	{
		using mg = atl::montgomery<Modulus>;
		using word = decltype(Modulus);
		using wide = atl::traits::widen_mul_type_t<word>;
		for (word a : {word(0), word(Modulus - 1), word(2 * Modulus - 1)}) {
			for (word b : {word(0), word(Modulus - 1), word(2 * Modulus - 1)}) {
				word const product = mg::mul_lazy(a, b);
				REQUIRE(product < 2 * Modulus);
				REQUIRE(product % Modulus == mg::mul(a % Modulus, b % Modulus));
				REQUIRE(mg::reduce_lazy(wide(a) * b) == product);
			}
			word const b = mg::to_form(Modulus - 1);
			word const product = mg::mul_lazy_fixed(a, b, mg::companion(b));
			REQUIRE(product < 2 * Modulus);
			REQUIRE(product % Modulus == mg::mul(a % Modulus, b));
		}
	};
	check.operator()<uint32(1073741823)>();
	check.operator()<(uint64(1) << 62) - 1>();
}
