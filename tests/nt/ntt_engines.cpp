// clang-format off
// Pair with ntt_engines_include_order.cpp: standard and atl headers precede the engines here.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#include <atl/nt/convolution.hpp>
#include <atl/nt/detail/ntt_simd.hpp>
#include <atl/nt/detail/ntt_conv8.hpp>
#include <catch2/catch.hpp>
#include "../expect_abort.hpp"
// clang-format on

using namespace atl;
using Z = mod_int<uint32, 998244353>;
using mont = montgomery<Z::mod()>;

// a 64-bit static modulus no 32-bit vector engine can serve
using Z64 = mod_int<uint64, (uint64(1) << 61) - 1>;
static_assert(simd_ntt::supports<Z>() && conv8_ntt::supports<Z>());
static_assert(!simd_ntt::supports<Z64>() && !conv8_ntt::supports<Z64>());
static_assert(std::is_same_v<detail::engine_or_scalar_t<simd_ntt, Z64>, scalar_ntt>);
static_assert(std::is_same_v<detail::engine_or_scalar_t<conv8_ntt, Z>, conv8_ntt>);

namespace {
	template<typename Engine, typename Element>
	concept raw_engine_for = requires(decltype(Element::mod()) *p) {
		Engine::template forward<Element>(p, 0);
		Engine::template inverse<Element>(p, 0);
		Engine::template convolve<Element>(p, p, 0);
	};
	static_assert(raw_engine_for<scalar_ntt, Z>);
	static_assert(raw_engine_for<simd_ntt, Z>);
	static_assert(raw_engine_for<conv8_ntt, Z>);
	static_assert(!raw_engine_for<scalar_ntt, mod_int<uint32, 129>>);
	static_assert(!raw_engine_for<simd_ntt, mod_int<uint32, 129>>);
	static_assert(!raw_engine_for<conv8_ntt, mod_int<uint32, 129>>);
	static_assert(raw_engine_for<scalar_ntt, mod_int<uint64, 3>>);
	static_assert(!raw_engine_for<simd_ntt, mod_int<uint64, 3>>);
	static_assert(!raw_engine_for<conv8_ntt, mod_int<uint64, 3>>);

	std::vector<uint32> random_residues(std::mt19937_64 &rng, std::size_t n) {
		std::vector<uint32> v(n);
		for (auto &x : v) x = uint32(rng() % Z::mod());
		return v;
	}
}

TEST_CASE("ntt engines: simd transforms match scalar bit for bit", "[nt]") {
	auto seed = GENERATE(take(6, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (uint32 h = 0; h <= 11; ++h) {
		auto a = random_residues(rng, std::size_t(1) << h);
		auto b = a;
		scalar_ntt::forward<Z>(a.data(), h);
		simd_ntt::forward<Z>(b.data(), h);
		REQUIRE(a == b);
		scalar_ntt::inverse<Z>(a.data(), h);
		simd_ntt::inverse<Z>(b.data(), h);
		REQUIRE(a == b);
	}
}

TEST_CASE("ntt engines: conv8_block_simd matches its scalar spec", "[nt]") {
	auto seed = GENERATE(take(4, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (int it = 0; it < 500; ++it) {
		auto a = random_residues(rng, 8), b = random_residues(rng, 8);
		auto a_spec = a;
		uint32 const w_mont = mont::to_form(uint32(rng() % Z::mod()));
		conv8_ntt::conv8_block_simd<Z>(a.data(), b.data(), w_mont);
		conv8_ntt::conv8_block_reference<Z>(a_spec.data(), b.data(), w_mont);
		for (int i = 0; i < 8; ++i) {
			uint32 x = a[i]; // block8 leaves lazy residues, the spec canonical ones
			x = x >= Z::mod() ? x - Z::mod() : x;
			REQUIRE(x == a_spec[i]);
		}
	}
}

TEST_CASE("ntt engines: fused conv8 convolve matches scalar convolve", "[nt]") {
	auto seed = GENERATE(take(4, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (uint32 h = 3; h <= 10; ++h) {
		auto a = random_residues(rng, std::size_t(1) << h);
		auto b = random_residues(rng, std::size_t(1) << h);
		auto a2 = a, b2 = b;
		conv8_ntt::convolve<Z>(a.data(), b.data(), h);
		scalar_ntt::convolve<Z>(a2.data(), b2.data(), h);
		REQUIRE(a == a2);
	}
}

TEST_CASE("ntt engines: convolution<Engine> agrees across engines and shapes", "[nt]") {
	auto seed = GENERATE(take(4, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	auto both = [&](std::size_t n, std::size_t m) {
		std::vector<Z> x, y;
		for (std::size_t i = 0; i < n; ++i) x.push_back(Z(uint32(rng() % Z::mod())));
		for (std::size_t i = 0; i < m; ++i) y.push_back(Z(uint32(rng() % Z::mod())));
		auto const want = convolution<scalar_ntt>(x, y);
		REQUIRE(convolution<simd_ntt>(x, y) == want);
		REQUIRE(convolution<conv8_ntt>(x, y) == want);
	};
	both(49, 49);	// just past the naive cutoff
	both(300, 300); // balanced
	both(1000, 60); // lopsided (overlap-add path: raw transforms, no fusion)
	both(5000, 49); // very lopsided at the cutoff boundary
	both(1 << 11, (1 << 11) - 3);
}

TEST_CASE("ntt engines: square matches convolve of a copy", "[nt]") {
	auto seed = GENERATE(take(4, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (uint32 h = 3; h <= 11; ++h) {
		auto const a = random_residues(rng, std::size_t(1) << h);
		auto want = a, copy = a;
		scalar_ntt::convolve<Z>(want.data(), copy.data(), h);
		auto got = a;
		scalar_ntt::square<Z>(got.data(), h);
		REQUIRE(got == want);
		got = a;
		simd_ntt::square<Z>(got.data(), h);
		REQUIRE(got == want);
		got = a;
		conv8_ntt::square<Z>(got.data(), h);
		REQUIRE(got == want);
		got = a, copy = a;
		simd_ntt::convolve<Z>(got.data(), copy.data(), h);
		REQUIRE(got == want);
	}
}

TEST_CASE("ntt engines: convolution(a, a) agrees across engines", "[nt]") {
	auto seed = GENERATE(take(4, random(std::uint64_t(0), (std::numeric_limits<std::uint64_t>::max)())));
	std::mt19937_64 rng(seed);
	for (std::size_t n : {49, 300, 1000, 2048}) {
		std::vector<Z> x;
		for (std::size_t i = 0; i < n; ++i) x.push_back(Z(uint32(rng() % Z::mod())));
		auto const copy = x;
		auto const want = convolution<scalar_ntt>(x, copy);
		REQUIRE(convolution<scalar_ntt>(x, x) == want);
		REQUIRE(convolution<simd_ntt>(x, x) == want);
		REQUIRE(convolution<conv8_ntt>(x, x) == want);
	}
}

TEST_CASE("ntt engines: modulus and level boundaries", "[nt]") {
	using Tiny = mod_int<uint32, 3>;
	uint32 values[]{1, 2};
	scalar_ntt::forward<Tiny>(values, 1);
	scalar_ntt::inverse<Tiny>(values, 1);
	REQUIRE(values[0] % 3 == 2);
	REQUIRE(values[1] % 3 == 1);
	CHECK(atl_test::aborts([] { uint32 x[2]{}; scalar_ntt::forward<Tiny>(x, 2); }));
	CHECK(atl_test::aborts([] { uint32 x[2]{}; simd_ntt::inverse<Tiny>(x, 2); }));
	CHECK(atl_test::aborts([] { uint32 x[2]{}; simd_ntt::forward<Tiny>(x, 1, 2); }));
	CHECK(atl_test::aborts([] { uint32 x[8]{}; conv8_ntt::square<Z>(x, 2); }));
}

TEST_CASE("ntt engines: fused sums near the modulus ceiling", "[nt]") {
	using CeilingZ = mod_int<uint32, 1004535809>;
	using CeilingMont = montgomery<CeilingZ::mod()>;
	std::vector<uint32> a(8, 2 * CeilingZ::mod() - 1), b = a, expected = a;
	uint32 const root = CeilingMont::to_form(CeilingZ::mod() - 1);
	conv8_ntt::conv8_block_simd<CeilingZ>(a.data(), b.data(), root);
	conv8_ntt::conv8_block_reference<CeilingZ>(expected.data(), b.data(), root);
	for (std::size_t i = 0; i < a.size(); ++i) REQUIRE(a[i] % CeilingZ::mod() == expected[i]);
}
