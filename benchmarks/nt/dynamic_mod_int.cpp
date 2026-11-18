/**
 * @file
 * @brief Construction throughput for 10,000,000 random 64-bit values modulo
 * 998244353: library constructors, reduce_mod division, and by reciprocal
 * using a local reciprocal. Signed construction reduces magnitude then sign.
 * The sum loop measures independent reductions with an unsigned checksum.
 * Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions disabled, 2026-10-15:
 * -O2 -g -DNDEBUG -std=c++23: unsigned 75.1/15.5/19.7 ms for division/local
 * reciprocal/library; signed 118/58.7 ms for division/library. Adding
 * -march=native gives 75.2/15.5/15.4 and 117/57.5 ms respectively.
 * The full table is in docs/modular-arithmetic.md.
 */
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/detail/wide_arithmetic.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

namespace {
	/**
	 * @brief Reduce a full 64-bit value with ceil(2^64/m); a quotient estimate
	 * at most one too large needs one conditional add. @pre `1 < m < 2^32`.
	 */
	atl::uint32 reduce_mod_by_reciprocal(atl::uint64 value, atl::uint32 modulus, atl::uint64 reciprocal) {
		atl::uint64 const quotient = atl::detail::multiply_high(value, reciprocal);
		atl::uint64 const product = quotient * modulus;
		return atl::uint32(value - product + (value < product ? modulus : 0));
	}

	template<atl::integral Integer, typename Reduce>
		requires std::is_invocable_r_v<atl::uint32, Reduce &, Integer &>
	atl::uint32 sum_reduced_values(std::span<Integer const> inputs, Reduce reduce) {
		atl::uint32 sum = 0;
		for (Integer value : inputs) sum += reduce(value);
		return sum;
	}
}

TEST_CASE("dynamic_mod_int construction throughput", "[dynamic_mod_int]") {
	using atl::int64;
	using atl::uint32;
	using atl::uint64;
	using Z = atl::dynamic_mod_int<uint32>;
	uint32 volatile runtime_modulus = 998244353;
	Z::set_mod(runtime_modulus);
	uint32 const modulus = Z::mod();
	uint64 const reciprocal = ~uint64(0) / modulus + 1;
	std::mt19937_64 rng(3);
	std::vector<uint64> unsigned_inputs(10'000'000);
	for (auto &value : unsigned_inputs) value = rng();
	std::vector<int64> signed_inputs(unsigned_inputs.begin(), unsigned_inputs.end());
	for (std::size_t i = 0; i < 100000; ++i) {
		REQUIRE(Z(unsigned_inputs[i]).value() == atl::reduce_mod<uint64, uint32>(unsigned_inputs[i], modulus));
		REQUIRE(reduce_mod_by_reciprocal(unsigned_inputs[i], modulus, reciprocal) == Z(unsigned_inputs[i]).value());
		REQUIRE(Z(signed_inputs[i]).value() == atl::reduce_mod<int64, uint32>(signed_inputs[i], modulus));
	}
	BENCHMARK("1e7 Z(uint64) | library reduce_mod (division)") {
		return sum_reduced_values(std::span<uint64 const>(unsigned_inputs), [modulus](uint64 value) {
			return atl::reduce_mod(value, modulus);
		});
	};
	BENCHMARK("1e7 Z(uint64) | by reciprocal") {
		return sum_reduced_values(std::span<uint64 const>(unsigned_inputs), [modulus, reciprocal](uint64 value) {
			return reduce_mod_by_reciprocal(value, modulus, reciprocal);
		});
	};
	BENCHMARK("1e7 Z(uint64) | library constructor") {
		return sum_reduced_values(std::span<uint64 const>(unsigned_inputs), [](uint64 value) {
			return Z(value).value();
		});
	};
	BENCHMARK("1e7 Z(int64) | library reduce_mod (division)") {
		return sum_reduced_values(std::span<int64 const>(signed_inputs), [modulus](int64 value) {
			return atl::reduce_mod(value, modulus);
		});
	};
	BENCHMARK("1e7 Z(int64) | library constructor (magnitude reciprocal)") {
		return sum_reduced_values(std::span<int64 const>(signed_inputs), [](int64 value) { return Z(value).value(); });
	};
}
