/**
 * @file
 * @brief Dependent multiply-chain latency: static_modulus uses a reciprocal,
 * montgomery uses REDC in Montgomery form, and mod_int64 dispatches to REDC.
 * Each row takes 5,000,000 dependent steps with moduli 10^18+3 or 998244353.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions disabled, 2026-10-15:
 * -O2 -g -DNDEBUG -std=c++23 gives 30.1/18.8/18.9 ms at 10^18+3 and
 * 17.1/18.4 ms at 998244353. Adding -march=native gives 27.1/18.5/18.5 ms
 * and 17.1/18.4 ms respectively. See docs/modular-arithmetic.md for the table.
 */
#include <atl/nt/mod_int.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/static_modulus.hpp>
#include <atl/types/alias.hpp>
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
}

TEST_CASE("montgomery multiply-chain latency", "[montgomery]") {
	constexpr std::size_t steps = 5'000'000;
	{
		using atl::uint64;
		constexpr uint64 modulus = 1'000'000'000'000'000'003ull;
		using mg = atl::montgomery<modulus>;
		using Z = atl::mod_int64<modulus>;
		auto reciprocal = [](uint64 a, uint64 b) { return atl::static_modulus<modulus>::mul(a, b); };
		auto redc = [](uint64 a, uint64 b) { return mg::mul(a, b); };
		auto multiply = [](Z a, Z b) { return a * b; };
		auto const expected = multiply_chain(uint64(3), uint64(12345), 128, reciprocal);
		REQUIRE(mg::from_form(multiply_chain(mg::to_form(3), mg::to_form(12345), 128, redc)) == expected);
		REQUIRE(multiply_chain(Z(3), Z(12345), 128, multiply).value() == expected);
		BENCHMARK("1e18+3 -- library static_modulus (reciprocal)") {
			return multiply_chain(uint64(3), uint64(12345), steps, reciprocal);
		};
		BENCHMARK("1e18+3 -- library montgomery (REDC)") {
			return mg::from_form(multiply_chain(mg::to_form(3), mg::to_form(12345), steps, redc));
		};
		BENCHMARK("1e18+3 -- library mod_int64 (Montgomery)") {
			return multiply_chain(Z(3), Z(12345), steps, multiply).value();
		};
	}
	{
		using atl::uint32;
		constexpr uint32 modulus = 998244353;
		using mg = atl::montgomery<modulus>;
		auto reciprocal = [](uint32 a, uint32 b) { return atl::static_modulus<modulus>::mul(a, b); };
		auto redc = [](uint32 a, uint32 b) { return mg::mul(a, b); };
		REQUIRE(mg::from_form(multiply_chain(mg::to_form(3), mg::to_form(12345), 128, redc))
				== multiply_chain(uint32(3), uint32(12345), 128, reciprocal));
		BENCHMARK("998244353 -- library static_modulus (reciprocal)") {
			return multiply_chain(uint32(3), uint32(12345), steps, reciprocal);
		};
		BENCHMARK("998244353 -- library montgomery (REDC)") {
			return mg::from_form(multiply_chain(mg::to_form(3), mg::to_form(12345), steps, redc));
		};
	}
}
