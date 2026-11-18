#include <atl/experimental/nt/bitwise_convolution_variants.hpp>
#include <atl/nt/bitwise_convolution.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <bit>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <random>
#include <span>
#include <type_traits>
#include <vector>
#include "../expect_abort.hpp"

using atl::int64;
using atl::uint32;
using atl::uint64;
namespace ex = atl::experimental;
using bitwise_mint = atl::mod_int<uint32, 998244353>;
using Even = atl::mod_int<uint32, 1000>;

static_assert(atl::commutative_ring<bitwise_mint>);
static_assert(atl::halvable_ring<bitwise_mint>);
static_assert(atl::halvable_ring<int64>);
static_assert(atl::halvable_ring<double>);
static_assert(atl::commutative_ring<uint32>);
static_assert(!atl::halvable_ring<uint32>); // no 1/2^n modulo 2^32
static_assert(atl::halvable_ring<atl::dynamic_mod_int<>>);
static_assert(atl::set_function_range<std::vector<bitwise_mint> &>);
static_assert(atl::set_function_range<std::span<int64>>);
static_assert(!atl::set_function_range<std::vector<bitwise_mint> const &>);

namespace {
	template<atl::commutative_ring T>
		requires std::constructible_from<T, uint64>
	T random_element(std::mt19937_64 &rng) {
		if constexpr (atl::modular_integer<T>)
			return T(uint64(rng()));
		else if constexpr (std::floating_point<T>)
			return T(int64(rng() % 201) - 100);
		else if constexpr (std::is_signed_v<T>)
			return T(int64(rng() % 2001) - 1000);
		else
			return T(rng());
	}

	template<atl::commutative_ring T>
		requires std::constructible_from<T, uint64>
	std::vector<T> random_vector(std::mt19937_64 &rng, std::size_t size) {
		std::vector<T> v(size);
		for (auto &x : v) x = random_element<T>(rng);
		return v;
	}

	/** @brief f[S] summed over the masks T with pred(S, T), the transforms'
	 *         definitions. */
	template<atl::additive_group T, typename Weight>
		requires std::is_invocable_r_v<int, Weight &, std::size_t, std::size_t>
	std::vector<T> naive_transform(std::vector<T> const &f, Weight weight) {
		std::vector<T> g(f.size());
		for (std::size_t s = 0; s < f.size(); ++s)
			for (std::size_t t = 0; t < f.size(); ++t) {
				int const w = weight(s, t);
				if (w == 1)
					g[s] += f[t];
				else if (w == -1)
					g[s] -= f[t];
			}
		return g;
	}

	int is_sub(std::size_t s, std::size_t t) { return (t & s) == t ? 1 : 0; }

	int is_super(std::size_t s, std::size_t t) { return (t & s) == s ? 1 : 0; }

	int sub_sign(std::size_t s, std::size_t t) { return (t & s) == t ? (std::popcount(s ^ t) % 2 ? -1 : 1) : 0; }

	int super_sign(std::size_t s, std::size_t t) { return (t & s) == s ? (std::popcount(s ^ t) % 2 ? -1 : 1) : 0; }

	int hadamard_sign(std::size_t s, std::size_t t) { return std::popcount(s & t) % 2 ? -1 : 1; }

	template<std::equality_comparable T>
	void check_equal(std::vector<T> const &got, std::vector<T> const &want) {
		REQUIRE(got.size() == want.size());
		for (std::size_t i = 0; i < got.size(); ++i) REQUIRE(got[i] == want[i]);
	}
}

TEMPLATE_TEST_CASE("bitwise transforms against their definitions", "[bitwise_convolution]", bitwise_mint, int64, uint32) {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (std::size_t bits = 0; bits <= 13; ++bits) { // past the 11-bit cache block, odd and even remainders
		if (bits > 8 && rng() % 3 != 0) continue;
		auto const f = random_vector<TestType>(rng, std::size_t(1) << bits);
		auto const run = [&]<typename Transform>(Transform transform)
			requires std::invocable<Transform &, std::vector<TestType> &>
		{
			auto g = f;
			transform(g);
			return g;
		};
		auto const zeta = run([]<atl::set_function_range Range>(Range &g) { atl::subset_zeta(g); });
		auto const sup = run([]<atl::set_function_range Range>(Range &g) { atl::superset_zeta(g); });
		auto const hadamard = run([]<atl::set_function_range Range>(Range &g) { atl::walsh_hadamard(g); });
		if (bits <= 9) {
			check_equal(zeta, naive_transform(f, is_sub));
			check_equal(sup, naive_transform(f, is_super));
			check_equal(hadamard, naive_transform(f, hadamard_sign));
			check_equal(run([]<atl::set_function_range Range>(Range &g) { atl::subset_mobius(g); }), naive_transform(f, sub_sign));
			check_equal(run([]<atl::set_function_range Range>(Range &g) { atl::superset_mobius(g); }), naive_transform(f, super_sign));
		}
		auto back = zeta;
		atl::subset_mobius(back);
		check_equal(back, f);
		back = sup;
		atl::superset_mobius(back);
		check_equal(back, f);
		if constexpr (atl::halvable_ring<TestType>) {
			back = hadamard;
			atl::inverse_walsh_hadamard(back);
			check_equal(back, f);
		}
		auto const engines = [&](auto butterfly, std::vector<TestType> const &want) {
			auto a = f, b = f, c = f;
			ex::transform_by_bit_outer_radix2(a.begin(), bits, butterfly);
			ex::transform_by_radix4(b.begin(), bits, butterfly);
			ex::transform_by_blocked_radix4(c.begin(), bits, butterfly);
			check_equal(a, want);
			check_equal(b, want);
			check_equal(c, want);
		};
		engines(atl::detail::subset_zeta_butterfly{}, zeta);
		engines(atl::detail::walsh_hadamard_butterfly{}, hadamard);
	}
}

TEMPLATE_TEST_CASE("bitwise convolutions against their definitions", "[bitwise_convolution]", bitwise_mint, int64, uint32) {
	auto seed = GENERATE(take(20, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	// unequal and non-power-of-two lengths exercise the padding
	std::size_t const n = rng() % 70, m = rng() % 70;
	auto const a = random_vector<TestType>(rng, n), b = random_vector<TestType>(rng, m);
	check_equal(atl::or_convolution(a, b),
				ex::bitwise_convolution_by_definition(a, b, [](std::size_t i, std::size_t j) { return i | j; }));
	check_equal(atl::and_convolution(a, b),
				ex::bitwise_convolution_by_definition(a, b, [](std::size_t i, std::size_t j) { return i & j; }));
	if constexpr (atl::halvable_ring<TestType>)
		check_equal(atl::xor_convolution(a, b),
					ex::bitwise_convolution_by_definition(a, b, [](std::size_t i, std::size_t j) { return i ^ j; }));
	auto const subset = ex::subset_convolution_by_definition(a, b);
	check_equal(atl::subset_convolution(a, b), subset);
	check_equal(ex::subset_convolution_by_rank_major(a, b), subset);
}

TEST_CASE("bitwise convolutions: floating point, spans, edge cases", "[bitwise_convolution]") {
	std::mt19937_64 rng(5);
	auto const a = random_vector<double>(rng, 32), b = random_vector<double>(rng, 32);
	auto const got = atl::xor_convolution(a, b);
	auto const want = ex::bitwise_convolution_by_definition(a, b, [](std::size_t i, std::size_t j) { return i ^ j; });
	for (std::size_t i = 0; i < 32; ++i) REQUIRE(std::abs(got[i] - want[i]) < 1e-6);

	std::vector<bitwise_mint> const empty;
	REQUIRE(atl::or_convolution(empty, empty).empty());
	REQUIRE(atl::subset_convolution(empty, empty).empty());
	REQUIRE(atl::xor_convolution(empty, empty).empty());
	std::vector<bitwise_mint> const one{bitwise_mint(3)}, two{bitwise_mint(5)};
	check_equal(atl::subset_convolution(one, two), std::vector<bitwise_mint>{bitwise_mint(15)});
	check_equal(atl::xor_convolution(one, two), std::vector<bitwise_mint>{bitwise_mint(15)});
	check_equal(atl::or_convolution(empty, two), std::vector<bitwise_mint>{bitwise_mint(0)}); // padded to 1

	std::vector<int64> buffer{1, 2, 3, 4, 99};
	atl::subset_zeta(std::span(buffer).first(4));
	check_equal(buffer, std::vector<int64>{1, 3, 4, 10, 99});

	std::vector<uint32> const big{~uint32(0), 2}, bounded{2, ~uint32(0)};
	check_equal(atl::subset_convolution(big, bounded), std::vector<uint32>{~uint32(0) * 2, ~uint32(0) * ~uint32(0) + 4});

	using D = atl::dynamic_mod_int<uint32, 61>;
	D::set_mod(1000000007);
	std::vector<D> const da{D(1), D(2), D(3), D(4)}, db{D(5), D(6), D(7), D(8)};
	auto const dh = atl::xor_convolution(da, db);
	auto const dw = ex::bitwise_convolution_by_definition(da, db, [](std::size_t i, std::size_t j) { return i ^ j; });
	check_equal(dh, dw);
}

TEST_CASE("bitwise convolutions: n = 16 subset convolution against rank-major", "[bitwise_convolution]") {
	std::mt19937_64 rng(16);
	auto const a = random_vector<bitwise_mint>(rng, 1 << 16), b = random_vector<bitwise_mint>(rng, 1 << 16);
	auto const h = atl::subset_convolution(a, b);
	check_equal(h, ex::subset_convolution_by_rank_major(a, b));
	for (int probe = 0; probe < 20; ++probe) { // spot checks against the definition
		std::size_t const s = rng() % (1 << 16);
		bitwise_mint want{};
		for (std::size_t l = s;; l = (l - 1) & s) {
			want += a[l] * b[s ^ l];
			if (l == 0) break;
		}
		REQUIRE(h[s] == want);
	}
}

TEST_CASE("bitwise convolutions: preconditions", "[bitwise_convolution]") {
	std::vector<int64> three{1, 2, 3};
	CHECK(atl_test::aborts([&] { atl::subset_zeta(three); }));
	CHECK(atl_test::aborts([&] { atl::walsh_hadamard(three); }));
	std::vector<int64> empty;
	CHECK(atl_test::aborts([&] { atl::subset_mobius(empty); }));
	std::vector<Even> even{Even(1), Even(2)};
	CHECK(atl_test::aborts([&] { atl::inverse_walsh_hadamard(even); }));
	std::vector<int64> not_an_image{1, 0}; // transform image of nothing: odd sum
	CHECK(atl_test::aborts([&] { atl::inverse_walsh_hadamard(not_an_image); }));
}

TEST_CASE("inverse Walsh scaling with a narrow signed element", "[bitwise_convolution]") {
	std::vector<signed char> f(128);
	atl::inverse_walsh_hadamard(f);
	CHECK(f == std::vector<signed char>(128));
	std::vector<atl::int128> wide{1, 2, 3, 4};
	atl::walsh_hadamard(wide);
	atl::inverse_walsh_hadamard(wide);
	CHECK(wide == std::vector<atl::int128>{1, 2, 3, 4});
}
