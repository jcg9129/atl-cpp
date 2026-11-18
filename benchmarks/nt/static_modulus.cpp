/**
 * @file
 * @brief Dependent multiply-chain latency: constant-modulus reciprocal or Mersenne
 * fold, mod_int engine dispatch, runtime-modulus widening, and by wide remainder.
 * Chains have 5,000,000 steps at 998244353, 2^31-1, 2^61-1 and 10^18.
 * Two exponentiation rows sum 100,000 powers using distinct 64-bit type spellings.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions disabled, 2026-10-15:
 * -O2 -g -DNDEBUG -std=c++23: the 10^18 reciprocal takes 30.4 ms against
 * 133 ms by wide remainder; adding -march=native gives 27.5 against 137 ms.
 * All rows and both flag sets appear in docs/modular-arithmetic.md.
 */
#include <atl/nt/mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/static_modulus.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <type_traits>

namespace {
	template<typename Value, typename Multiply>
		requires std::is_invocable_r_v<Value, Multiply &, Value &, Value &>
			  && std::is_move_constructible_v<Value>
			  && requires(Value &a, Value &b, Multiply &multiply) { a = multiply(a, b); }
	Value multiply_chain(Value a, Value b, std::size_t steps, Multiply multiply) {
		for (std::size_t i = 0; i < steps; ++i) a = multiply(a, b);
		return a;
	}

	/** @brief Reduce the exact 128-bit product with constant remainder division. */
	atl::uint64 mul_mod_by_wide_remainder(atl::uint64 a, atl::uint64 b) {
		return atl::uint64(atl::uint128(a) * b % 1'000'000'000'000'000'000ull);
	}

	template<atl::unsigned_integral Integer>
	Integer sum_modular_powers(Integer modulus) {
		Integer sum = 0;
		for (Integer exp = 0; exp < 100000; ++exp)
			sum += atl::pow_mod<Integer>(3, atl::uint64(exp | (Integer(1) << 40)), modulus);
		return sum;
	}
}

TEST_CASE("static_modulus multiply-chain latency", "[static_modulus]") {
	constexpr std::size_t steps = 5'000'000;
	{
		using atl::uint32;
		constexpr uint32 modulus = 998244353;
		using Z = atl::mod_int32<modulus>;
		uint32 volatile runtime_modulus = modulus;
		auto reciprocal = [](uint32 a, uint32 b) { return atl::static_modulus<modulus>::mul(a, b); };
		auto multiply = [](Z a, Z b) { return a * b; };
		auto const expected = multiply_chain(uint32(3), uint32(12345), 128, reciprocal);
		REQUIRE(multiply_chain(Z(3), Z(12345), 128, multiply).value() == expected);
		REQUIRE(multiply_chain(
					uint32(3), uint32(12345), 128,
					[&](uint32 a, uint32 b) { return atl::mul_mod_unchecked(a, b, uint32(runtime_modulus)); })
				== expected);
		BENCHMARK("998244353 -- library static_modulus") {
			return multiply_chain(uint32(3), uint32(12345), steps, reciprocal);
		};
		BENCHMARK("998244353 -- library mod_int") { return multiply_chain(Z(3), Z(12345), steps, multiply).value(); };
		BENCHMARK("998244353 -- library mul_mod (runtime modulus)") {
			uint32 const m = runtime_modulus;
			return multiply_chain(uint32(3), uint32(12345), steps, [m](uint32 a, uint32 b) {
				return atl::mul_mod_unchecked(a, b, m);
			});
		};
	}
	{
		using atl::uint32;
		constexpr uint32 modulus = 2147483647;
		using Z = atl::mod_int32<modulus>;
		auto fold = [](uint32 a, uint32 b) { return atl::static_modulus<modulus>::mul(a, b); };
		auto multiply = [](Z a, Z b) { return a * b; };
		REQUIRE(multiply_chain(Z(3), Z(12345), 128, multiply).value()
				== multiply_chain(uint32(3), uint32(12345), 128, fold));
		BENCHMARK("2^31-1 -- library static_modulus (fold)") {
			return multiply_chain(uint32(3), uint32(12345), steps, fold);
		};
		BENCHMARK("2^31-1 -- library mod_int") { return multiply_chain(Z(3), Z(12345), steps, multiply).value(); };
	}
	{
		using atl::uint64;
		constexpr uint64 modulus = (uint64(1) << 61) - 1;
		using Z = atl::mod_int64<modulus>;
		auto fold = [](uint64 a, uint64 b) { return atl::static_modulus<modulus>::mul(a, b); };
		auto multiply = [](Z a, Z b) { return a * b; };
		REQUIRE(multiply_chain(Z(3), Z(12345), 128, multiply).value()
				== multiply_chain(uint64(3), uint64(12345), 128, fold));
		BENCHMARK("2^61-1 -- library static_modulus (fold)") {
			return multiply_chain(uint64(3), uint64(12345), steps, fold);
		};
		BENCHMARK("2^61-1 -- library mod_int64") { return multiply_chain(Z(3), Z(12345), steps, multiply).value(); };
	}
	{
		using atl::uint64;
		constexpr uint64 modulus = 1'000'000'000'000'000'000ull;
		using Z = atl::mod_int64<modulus>;
		auto reciprocal = [](uint64 a, uint64 b) { return atl::static_modulus<modulus>::mul(a, b); };
		auto remainder = [](uint64 a, uint64 b) { return mul_mod_by_wide_remainder(a, b); };
		auto multiply = [](Z a, Z b) { return a * b; };
		auto const expected = multiply_chain(uint64(3), uint64(123456789012345ull), 128, reciprocal);
		REQUIRE(multiply_chain(uint64(3), uint64(123456789012345ull), 128, remainder) == expected);
		REQUIRE(multiply_chain(Z(3), Z(123456789012345ull), 128, multiply).value() == expected);
		BENCHMARK("1e18 -- by wide remainder") {
			return multiply_chain(uint64(3), uint64(123456789012345ull), steps, remainder);
		};
		BENCHMARK("1e18 -- library static_modulus (barrett64)") {
			return multiply_chain(uint64(3), uint64(123456789012345ull), steps, reciprocal);
		};
		BENCHMARK("1e18 -- library mod_int64") {
			return multiply_chain(Z(3), Z(123456789012345ull), steps, multiply).value();
		};
	}
	{
		unsigned long long volatile runtime_modulus = 1'000'000'000'000'000'003ull;
		REQUIRE(sum_modular_powers(runtime_modulus) == sum_modular_powers(atl::uint64(runtime_modulus)));
		BENCHMARK("1e18+3 -- library pow_mod<unsigned long long> x1e5") { return sum_modular_powers(runtime_modulus); };
		BENCHMARK("1e18+3 -- library pow_mod<uint64> x1e5") {
			return sum_modular_powers(atl::uint64(runtime_modulus));
		};
	}
}
