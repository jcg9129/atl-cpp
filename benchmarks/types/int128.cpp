/**
 * @file
 * Cost table for 128-bit arithmetic, not a race between implementations: it
 * shows what the portable software `uint128` costs next to the native
 * `__uint128_t` (when the compiler has it) and next to the 64-bit word
 * primitives the software type is built from. Nothing here is shipped or
 * rejected on these numbers.
 *
 * Rows, all run over the same 4096 random operand sets:
 * - `software uint128` and `native __uint128_t` rows do the same work on each
 *   type: add, multiply (low 128 bits), 128/64 division by a word, 128/128
 *   division, and multiply-high (the top 128 bits of the 256-bit product;
 *   natively assembled from four limb products by `multiply_high_by_native_limbs`);
 * - `word primitive` rows time the building blocks on 64-bit words:
 *   64x64 -> 128 multiply, high word of that product, add with carry, subtract
 *   with borrow, division with a word quotient (`high < divisor`), and general
 *   128/64 division.
 *
 * Each row folds the results of independent operand sets into a checksum, so it
 * measures throughput over the whole set (4096 operations per sample).
 *
 * Result: not measured yet; there are no recorded numbers. Table: docs/int128.md.
 */
#include <atl/types/int128.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstdint>
#include <random>
#include <vector>

namespace {
	using namespace atl::detail;
	using unsigned_integer = software_uint128;

	/** @brief Independent operands for full-width and word-sized divisor benchmarks. */
	struct arithmetic_input {
		unsigned_integer left;	///< Full-width dividend.
		unsigned_integer right; ///< Full-width multiplier or divisor.
		std::uint64_t divisor;	///< Nonzero word divisor.
	};

	/** @brief Select a common benchmark loop at compile time. */
	enum class operation { add,
						   multiply,
						   divide_word,
						   divide_integer };

	/** @brief Unsigned 128-bit types whose accumulation cannot overflow a signed built-in. */
	template<typename Integer>
	concept unsigned_wide_integer = std::same_as<Integer, unsigned_integer>
#if defined(__SIZEOF_INT128__)
								 || std::same_as<Integer, __uint128_t>
#endif
		;

	/** @brief Time identical arithmetic workloads for a software or native integer. */
	template<unsigned_wide_integer Integer, operation Operation>
	Integer integer_workload(std::vector<arithmetic_input> const &inputs) {
		Integer checksum = 0;
		for (auto const &input : inputs) {
			Integer const left = (Integer(input.left.high) << 64) | Integer(input.left.low);
			Integer const right = (Integer(input.right.high) << 64) | Integer(input.right.low);
			if constexpr (Operation == operation::add) checksum += left + right;
			if constexpr (Operation == operation::multiply) checksum += left * right;
			if constexpr (Operation == operation::divide_word) checksum += left / Integer(input.divisor);
			if constexpr (Operation == operation::divide_integer) checksum += left / right;
		}
		return checksum;
	}

#if defined(__SIZEOF_INT128__)
	/** @brief Top 128 bits of the 256-bit product of two native values, assembled from four 64x64 limb products. */
	__uint128_t multiply_high_by_native_limbs(__uint128_t left, __uint128_t right) {
		auto const bottom = __uint128_t(std::uint64_t(left)) * std::uint64_t(right);
		auto const cross_left = __uint128_t(std::uint64_t(left)) * std::uint64_t(right >> 64);
		auto const cross_right = __uint128_t(std::uint64_t(left >> 64)) * std::uint64_t(right);
		auto const top = (left >> 64) * (right >> 64);
		auto const middle = (bottom >> 64) + std::uint64_t(cross_left) + __uint128_t(std::uint64_t(cross_right));
		return top + (cross_left >> 64) + (cross_right >> 64) + (middle >> 64);
	}
#endif
}

TEST_CASE("portable 128-bit arithmetic", "[types][int128]") {
	std::mt19937_64 random(812);
	std::vector<arithmetic_input> inputs;
	inputs.reserve(4096);
	for (int index = 0; index < 4096; ++index) inputs.push_back({{random(), random() | (std::uint64_t(1) << 63)}, {random(), (random() >> 32) | 1}, random() | 1});

#if defined(__SIZEOF_INT128__)
	auto const widen = [](unsigned_integer value) { return (__uint128_t(value.high) << 64) | value.low; };
	REQUIRE(widen(integer_workload<unsigned_integer, operation::add>(inputs)) == integer_workload<__uint128_t, operation::add>(inputs));
	REQUIRE(widen(integer_workload<unsigned_integer, operation::multiply>(inputs)) == integer_workload<__uint128_t, operation::multiply>(inputs));
	REQUIRE(widen(integer_workload<unsigned_integer, operation::divide_word>(inputs)) == integer_workload<__uint128_t, operation::divide_word>(inputs));
	REQUIRE(widen(integer_workload<unsigned_integer, operation::divide_integer>(inputs)) == integer_workload<__uint128_t, operation::divide_integer>(inputs));
	for (auto const &input : inputs) {
		auto const software = multiply_high_wide({input.left.low, input.left.high}, {input.right.low, input.right.high});
		REQUIRE(widen({software.low, software.high}) == multiply_high_by_native_limbs(widen(input.left), widen(input.right)));
	}
#endif

	BENCHMARK("software uint128 | add") { return integer_workload<unsigned_integer, operation::add>(inputs); };
	BENCHMARK("software uint128 | multiply (low 128 bits)") { return integer_workload<unsigned_integer, operation::multiply>(inputs); };
	BENCHMARK("software uint128 | divide by word (128/64)") { return integer_workload<unsigned_integer, operation::divide_word>(inputs); };
	BENCHMARK("software uint128 | divide by 128-bit (128/128)") { return integer_workload<unsigned_integer, operation::divide_integer>(inputs); };
	BENCHMARK("software uint128 | multiply-high (top 128 of 256 bits)") {
		unsigned_integer checksum{};
		for (auto const &input : inputs) {
			auto const result = multiply_high_wide({input.left.low, input.left.high}, {input.right.low, input.right.high});
			checksum += unsigned_integer{result.low, result.high};
		}
		return checksum;
	};
#if defined(__SIZEOF_INT128__)
	BENCHMARK("native __uint128_t | add") { return integer_workload<__uint128_t, operation::add>(inputs); };
	BENCHMARK("native __uint128_t | multiply (low 128 bits)") { return integer_workload<__uint128_t, operation::multiply>(inputs); };
	BENCHMARK("native __uint128_t | divide by word (128/64)") { return integer_workload<__uint128_t, operation::divide_word>(inputs); };
	BENCHMARK("native __uint128_t | divide by 128-bit (128/128)") { return integer_workload<__uint128_t, operation::divide_integer>(inputs); };
	BENCHMARK("native __uint128_t | multiply-high (top 128 of 256 bits)") {
		__uint128_t checksum = 0;
		for (auto const &input : inputs) checksum += multiply_high_by_native_limbs((__uint128_t(input.left.high) << 64) | input.left.low, (__uint128_t(input.right.high) << 64) | input.right.low);
		return checksum;
	};
#endif
	BENCHMARK("word primitive | multiply 64x64 -> 128") {
		std::uint64_t checksum = 0;
		for (auto const &input : inputs) {
			auto const result = multiply_wide(input.left.low, input.right.low);
			checksum += result.low ^ result.high;
		}
		return checksum;
	};
	BENCHMARK("word primitive | multiply-high (top 64 of 64x64)") {
		std::uint64_t checksum = 0;
		for (auto const &input : inputs) checksum += multiply_high(input.left.low, input.right.low);
		return checksum;
	};
	BENCHMARK("word primitive | add with carry") {
		std::uint64_t checksum = 0;
		for (auto const &input : inputs) {
			auto const result = add_with_carry(input.left.low, input.right.low, input.left.high & 1);
			checksum += result.value ^ result.carry;
		}
		return checksum;
	};
	BENCHMARK("word primitive | subtract with borrow") {
		std::uint64_t checksum = 0;
		for (auto const &input : inputs) {
			auto const result = subtract_with_borrow(input.left.low, input.right.low, input.left.high & 1);
			checksum += result.value ^ result.borrow;
		}
		return checksum;
	};
	BENCHMARK("word primitive | divide with word quotient (high < divisor)") {
		std::uint64_t checksum = 0;
		for (auto const &input : inputs) {
			auto const result = divide_wide_bounded(input.left.low, input.divisor - 1, input.divisor);
			checksum += result.quotient ^ result.remainder;
		}
		return checksum;
	};
	BENCHMARK("word primitive | divide by word (general 128/64)") {
		std::uint64_t checksum = 0;
		for (auto const &input : inputs) {
			auto const result = divide_wide(input.left.low, input.left.high, input.divisor);
			checksum += result.quotient.low ^ result.quotient.high ^ result.remainder;
		}
		return checksum;
	};
}
