/**
 * @file
 * Race between two implementations of `compare_products` (the exact sign of
 * `a*b - c*d` for signed 128-bit operands, via 256-bit products).
 *
 * They differ in how they handle the operand signs:
 * - `atl::detail::compare_products` (the library) multiplies the two's-complement
 *   bits as unsigned values, corrects the high half by masks, and subtracts the
 *   two products, with no negation and no branch on the signs;
 * - `compare_products_by_sign_magnitude` multiplies the absolute values,
 *   negates the 256-bit product when the signs differ, and forms the
 *   difference as `x + (-y)`.
 *
 * The benchmark compares 1e6 independent quadruples of random mixed-sign
 * ~100-bit operands, so it measures throughput.
 *
 * Result (Xeon Platinum 8280, GCC 15.2, 2026-10-15): the library is faster on
 * both flag sets. `by_sign_magnitude` is 9.5% slower with `-O2 -march=native`
 * (36.8 ms against 33.6 ms) and 22% slower with plain `-O2` (38.2 ms against
 * 31.2 ms). Numbers: docs/int128.md.
 */
#include <atl/types/alias.hpp>
#include <atl/types/detail/int256.hpp>
#include <catch2/catch.hpp>
#include <random>
#include <vector>

using namespace atl;

namespace {
	/** @brief Exact 256-bit product of two signed values; multiplies the magnitudes and negates the result when the signs differ. */
	detail::int256 product_by_sign_magnitude(int128 a, int128 b) {
		bool const is_negative = (a < 0) != (b < 0);
		uint128 const ua = a < 0 ? uint128(0) - uint128(a) : uint128(a);
		uint128 const ub = b < 0 ? uint128(0) - uint128(b) : uint128(b);
		uint64 const a0 = uint64(ua), a1 = uint64(ua >> 64);
		uint64 const b0 = uint64(ub), b1 = uint64(ub >> 64);
		uint128 const p00 = uint128(a0) * b0;
		uint128 const p01 = uint128(a0) * b1;
		uint128 const p10 = uint128(a1) * b0;
		uint128 const p11 = uint128(a1) * b1;
		detail::int256 r;
		r.limb[0] = uint64(p00);
		uint128 const mid = uint128(uint64(p01)) + uint64(p10) + uint64(p00 >> 64);
		r.limb[1] = uint64(mid);
		uint128 const high = (p01 >> 64) + (p10 >> 64) + uint64(p11) + uint64(mid >> 64);
		r.limb[2] = uint64(high);
		r.limb[3] = uint64(p11 >> 64) + uint64(high >> 64);
		return is_negative ? -r : r;
	}

	/** @brief Same result as `detail::compare_products`; uses the sign-magnitude product and forms the difference as `x + (-y)`. */
	int compare_products_by_sign_magnitude(int128 a, int128 b, int128 c, int128 d) {
		detail::int256 x = product_by_sign_magnitude(a, b);
		x += -product_by_sign_magnitude(c, d);
		return x.sign();
	}
}

TEST_CASE("int256 compare_products", "[int256]") {
	std::mt19937_64 rng(9);
	auto operand = [&] {
		int128 const v = int128((uint128(rng() >> 28) << 64) | rng()); // ~100 bits
		return rng() & 1 ? -v : v;
	};
	std::size_t const n = 1'000'000;
	std::vector<int128> q(4 * n);
	for (auto &x : q) x = operand();
	for (std::size_t i = 0; i < n; ++i) {
		REQUIRE(detail::compare_products(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3])
				== compare_products_by_sign_magnitude(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]));
		REQUIRE(detail::compare_products(q[4 * i], q[4 * i + 1], q[4 * i], q[4 * i + 1]) == 0);
	}
	BENCHMARK("1e6 compare_products, mixed signs | by sign magnitude") {
		int acc = 0;
		for (std::size_t i = 0; i < n; ++i)
			acc += compare_products_by_sign_magnitude(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
		return acc;
	};
	BENCHMARK("1e6 compare_products, mixed signs | library") {
		int acc = 0;
		for (std::size_t i = 0; i < n; ++i)
			acc += detail::compare_products(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
		return acc;
	};
}
