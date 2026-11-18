#include <atl/experimental/nt/convolution_variants.hpp>
#include <atl/nt/convolution.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/fft.hpp>
#include <algorithm>
#include <bit>
#include <catch2/catch.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <span>
#include <vector>
#include "../expect_abort.hpp"

using namespace atl;
namespace ex = atl::experimental;
using convolution_mint = mod_int<uint32, 998244353>;
using M7 = mod_int<uint32, 1000000007>;

static_assert(static_mod_int<convolution_mint>);
static_assert(static_mod_int<M7>);
static_assert(!static_mod_int<uint32>);
static_assert(mod_int_range<std::vector<convolution_mint>>);
static_assert(mod_int_range<std::span<convolution_mint const>>);
static_assert(!mod_int_range<std::vector<uint32>>);
static_assert(integral_range<std::vector<uint32>>);
static_assert(!integral_range<std::vector<convolution_mint>>);
static_assert(detail::primitive_root<convolution_mint>() == convolution_mint(3));
static_assert(cplx_range<std::vector<cplx> &>);
static_assert(!cplx_range<std::vector<cplx> const &>);
static_assert(!static_mod_int<dynamic_mod_int<>>); // modulus not a constant expression

TEST_CASE("convolution: every engine against the oracle", "[nt]") {
	auto seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = rng() % 70, m = rng() % 70;
	std::vector<convolution_mint> a(n), b(m);
	for (auto &x : a) x = convolution_mint(uint32(rng() % convolution_mint::mod()));
	for (auto &x : b) x = convolution_mint(uint32(rng() % convolution_mint::mod()));
	auto const want = ex::convolution_by_definition(a, b);

	auto const check = [&]<mod_int_range Result>(Result const &got) {
		REQUIRE(got.size() == want.size());
		for (std::size_t i = 0; i < want.size(); ++i) REQUIRE(got[i] == want[i]);
	};
	check(convolution(a, b));
	check(ex::convolution_by_radix2(a, b));
	check(ex::convolution_by_dif(a, b));
	check(ex::convolution_by_lazy_montgomery(a, b));
	check(ex::convolution_by_radix4(a, b));
}

static_assert(ntt_engine<ex::ntt_by_lazy_montgomery>);
static_assert(ntt_engine<ex::ntt_by_radix4>);

TEST_CASE("convolution: experimental engines through the library driver", "[nt]") {
	auto seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	// past the naive cutoff; odd and even heights, balanced and lopsided (overlap-add)
	std::size_t const n = 49 + rng() % 600, m = 49 + rng() % (rng() % 2 ? 600 : 30);
	std::vector<convolution_mint> a(n), b(m);
	for (auto &x : a) x = convolution_mint(uint32(rng() % convolution_mint::mod()));
	for (auto &x : b) x = convolution_mint(uint32(rng() % convolution_mint::mod()));
	auto const want = ex::convolution_by_definition(a, b);
	REQUIRE(convolution<ex::ntt_by_lazy_montgomery>(a, b) == want);
	REQUIRE(convolution<ex::ntt_by_radix4>(a, b) == want);
}

TEST_CASE("convolution: unfriendly modulus (FFT balanced-split path)", "[nt]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 49 + rng() % 200, m = 49 + rng() % 200;
	std::vector<M7> a(n), b(m);
	for (auto &x : a) x = M7(uint32(rng() % M7::mod()));
	for (auto &x : b) x = M7(uint32(rng() % M7::mod()));
	auto const want = ex::convolution_by_definition(a, b);
	auto const got = convolution(a, b);
	REQUIRE(got.size() == want.size());
	for (std::size_t i = 0; i < want.size(); ++i) REQUIRE(got[i] == want[i]);
}

TEST_CASE("convolution: huge modulus falls back to Garner", "[nt]") {
	using ZBig = mod_int<uint32, 4294967291u>; // 2^32 - 5, prime, > 2^30
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 49 + rng() % 150, m = 49 + rng() % 150;
	std::vector<ZBig> a(n), b(m);
	for (auto &x : a) x = ZBig(uint32(rng() % ZBig::mod()));
	for (auto &x : b) x = ZBig(uint32(rng() % ZBig::mod()));
	auto const want = ex::convolution_by_definition(a, b);
	auto const got = convolution(a, b);
	REQUIRE(got.size() == want.size());
	for (std::size_t i = 0; i < want.size(); ++i) REQUIRE(got[i] == want[i]);
}

TEST_CASE("convolution: a square takes the one-forward path in every tier", "[nt]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto check = [&]<static_mod_int M>(std::size_t n) {
		std::vector<M> a(n);
		for (auto &x : a) x = M(uint32(rng() % M::mod()));
		auto const want = ex::convolution_by_definition(a, a);
		REQUIRE(convolution(a, a) == want);
		std::span<M const> const view(a);
		REQUIRE(convolution(view, a) == want); // same storage through another range type
		REQUIRE(convolution(a, view.first(n - 1)) == ex::convolution_by_definition(a, std::vector<M>(a.begin(), a.end() - 1)));
	};
	check.operator()<convolution_mint>(49 + rng() % 300);			  // direct NTT
	check.operator()<M7>(49 + rng() % 300);							  // FFT split
	check.operator()<mod_int<uint32, 4294967291u>>(49 + rng() % 150); // Garner
	std::vector<uint32> c(49 + rng() % 300);
	for (auto &x : c) x = uint32(rng() % 100000);
	std::vector<uint64> want(2 * c.size() - 1, 0);
	for (std::size_t i = 0; i < c.size(); ++i)
		for (std::size_t j = 0; j < c.size(); ++j) want[i + j] += uint64(c[i]) * c[j];
	REQUIRE(convolution_uint64(c, c) == want);
}

TEST_CASE("convolution: a lopsided product over a bounded-headroom modulus stays direct", "[nt]") {
	using Z7 = mod_int<uint32, 7340033>; // 7·2^20 + 1
	std::mt19937_64 rng(5);
	std::vector<Z7> a((1 << 20) - 10), b(70);
	for (auto &x : a) x = Z7(uint32(rng() % Z7::mod()));
	for (auto &x : b) x = Z7(uint32(rng() % Z7::mod()));
	// the padded size 2^21 exceeds the headroom; the overlap-add size 256 does not
	REQUIRE(detail::planned_ntt_size(a.size(), b.size()) == 256);
	auto const got = convolution(a, b);
	REQUIRE(got.size() == a.size() + b.size() - 1);
	for (std::size_t k : {std::size_t(0), std::size_t(69), std::size_t(5000), got.size() - 1}) {
		Z7 want(0);
		for (std::size_t j = 0; j < b.size(); ++j)
			if (k >= j && k - j < a.size()) want += a[k - j] * b[j];
		REQUIRE(got[k] == want);
	}
}

TEST_CASE("convolution: lopsided products in the FFT-split tier run overlap-add", "[nt]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const m = 49 + rng() % 100, n = 4 * std::bit_ceil(2 * m) + rng() % 3000;
	REQUIRE(detail::planned_ntt_size(n, m) < std::bit_ceil(n + m - 1));
	std::vector<M7> a(n), b(m);
	for (auto &x : a) x = M7(uint32(rng() % M7::mod()));
	for (auto &x : b) x = M7(uint32(rng() % M7::mod()));
	REQUIRE(convolution(a, b) == ex::convolution_by_definition(a, b));
	REQUIRE(convolution(b, a) == ex::convolution_by_definition(a, b));
	for (auto &x : a) x = M7(M7::mod() - 1);
	for (auto &x : b) x = M7(M7::mod() - 1);
	REQUIRE(convolution(a, b) == ex::convolution_by_definition(a, b));
}

TEST_CASE("fft / inverse_fft: round-trip and real convolution", "[nt]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = std::size_t(1) << (3 + rng() % 8);
	std::vector<cplx> a(n);
	for (auto &z : a) z = {double(rng() % 1000) - 500, double(rng() % 1000) - 500};
	auto b = a;
	fft(b);
	inverse_fft(b);
	for (std::size_t i = 0; i < n; ++i) {
		REQUIRE(std::abs(b[i].re - a[i].re) < 1e-6);
		REQUIRE(std::abs(b[i].im - a[i].im) < 1e-6);
	}
}

TEST_CASE("convolution_uint64: exact counts", "[nt]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 300, m = 1 + rng() % 300;
	std::vector<uint32> a(n), b(m);
	for (auto &x : a) x = uint32(rng() % 100000);
	for (auto &x : b) x = uint32(rng() % 100000);
	auto const got = convolution_uint64(a, b);
	std::vector<uint64> want(n + m - 1, 0);
	for (std::size_t i = 0; i < n; ++i)
		for (std::size_t j = 0; j < m; ++j) want[i + j] += uint64(a[i]) * b[j];
	REQUIRE(got == want);
}

TEST_CASE("ntt / inverse_ntt: transform pair round-trips", "[nt]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = std::size_t(1) << (rng() % 10);
	std::vector<convolution_mint> a(n);
	for (auto &x : a) x = convolution_mint(uint32(rng() % convolution_mint::mod()));
	auto b = a;
	ntt(b);
	inverse_ntt(b);
	REQUIRE(a == b);
	// pointwise product of spectra = cyclic convolution
	auto fa = a;
	ntt(fa);
	for (std::size_t i = 0; i < n; ++i) fa[i] *= fa[i];
	inverse_ntt(fa);
	std::vector<convolution_mint> cyclic(n);
	for (std::size_t i = 0; i < n; ++i)
		for (std::size_t j = 0; j < n; ++j) cyclic[(i + j) % n] += a[i] * a[j];
	REQUIRE(fa == cyclic);
}

TEST_CASE("convolution: canonical", "[nt]") {
	std::vector<convolution_mint> f{convolution_mint(3), convolution_mint(1), convolution_mint(4), convolution_mint(1), convolution_mint(5)};
	REQUIRE(convolution(f, std::vector<convolution_mint>{convolution_mint(1)}) == f);
	REQUIRE(convolution(std::vector<convolution_mint>{}, f).empty());
	std::vector<convolution_mint> one_x{convolution_mint(1), convolution_mint(1)};
	REQUIRE(convolution(one_x, one_x) == std::vector<convolution_mint>{convolution_mint(1), convolution_mint(2), convolution_mint(1)});
}

TEST_CASE("convolution: an odd COMPOSITE modulus takes the non-NTT tiers", "[nt]") {
	// A composite can have enough 2-adic headroom without a root of that order.
	auto const check = []<static_mod_int Element>(Element zero, std::size_t const n, std::size_t const m, std::mt19937_64 &rng) {
		using Y = decltype(zero);
		std::vector<Y> a(n), b(m);
		for (auto &x : a) x = Y(rng());
		for (auto &x : b) x = Y(rng());
		std::vector<Y> expected(n + m - 1);
		for (std::size_t i = 0; i < n; ++i)
			for (std::size_t j = 0; j < m; ++j) expected[i + j] += a[i] * b[j];
		auto const got = convolution(a, b);
		REQUIRE(got.size() == expected.size());
		for (std::size_t k = 0; k < expected.size(); ++k) REQUIRE(got[k].value() == expected[k].value());
	};
	auto seed = GENERATE(take(6, random(uint64(0), ~uint64(0))));
	std::mt19937_64 rng(seed);
	std::size_t const n = 49 + rng() % 150, m = 49 + rng() % 150;
	check(mod_int<uint32, 129>(), 60, 60, rng); // 128 | 129 - 1
	check(mod_int<uint32, 129>(), n, m, rng);
	check(mod_int<uint32, 3 * 257>(), n, m, rng);	  // 256 | 771 - 1 is false: plain composite
	check(mod_int<uint32, 65537u * 3u>(), n, m, rng); // large odd composite
	check(mod_int<uint32, 4097>(), n, m, rng);		  // 4097 = 17 * 241, 4096 | mod - 1
}

namespace {
	template<typename Values>
	concept transformable = requires(Values &values) {
		ntt(values);
		inverse_ntt(values);
	};
}

TEST_CASE("ntt: a modulus the engine cannot run over is rejected at compile time", "[nt]") {
	static_assert(transformable<std::vector<convolution_mint>>);
	static_assert(transformable<std::vector<mod_int<uint32, 7340033>>>);
	static_assert(!transformable<std::vector<mod_int<uint64, 4179340454199820289ull>>>); // 29 * 2^57 + 1
	static_assert(!transformable<std::vector<mod_int<uint32, 3221225473u>>>);			 // 3 * 2^30 + 1: over 30 bits
	static_assert(!transformable<std::vector<mod_int<uint32, 129>>>);					 // composite
	static_assert(!transformable<std::vector<mod_int<uint32, 1000000008>>>);			 // even
	std::vector<convolution_mint> values{convolution_mint(1), convolution_mint(2), convolution_mint(3), convolution_mint(4)};
	auto const original = values;
	ntt(values);
	inverse_ntt(values);
	for (std::size_t i = 0; i < values.size(); ++i) REQUIRE(values[i].value() == original[i].value());
}

TEST_CASE("convolution: the Garner tier asserts its coefficient bound", "[nt]") {
	// a 64-bit modulus goes straight to Garner, whose result is only the true
	// coefficient mod M while that coefficient stays below p0·p1·p2 ~ 7.8e25
	using ZWide = mod_int<uint64, 18446744073709551557ull>;
	std::vector<ZWide> bounded(64, ZWide(uint64(1) << 20)), huge(64, ZWide(uint64(1) << 62));
	auto const got = convolution(bounded, bounded);
	REQUIRE(got[63] == ZWide(uint64(64) << 40));
	CHECK(atl_test::aborts([&] { (void)convolution(huge, huge); }));
}

namespace {
	template<typename M>
	concept experimental_ntt_ready = requires(std::vector<M> &v, std::vector<M> const &c) {
		typename ex::ntt_rate_tables<M>;
		ex::ntt_by_dif(v, false);
		ex::convolution_by_dif(c, c);
		ex::convolution_by_lazy_montgomery(c, c);
		ex::convolution_by_radix4(c, c);
	};

	template<typename M>
	concept experimental_ntt_rejected = !requires { typename ex::ntt_rate_tables<M>; }
									 && !requires(std::vector<M> &v) { ex::ntt_by_dif(v, false); }
									 && !requires(std::vector<M> const &c) { ex::convolution_by_dif(c, c); }
									 && !requires(std::vector<M> const &c) { ex::convolution_by_lazy_montgomery(c, c); }
									 && !requires(std::vector<M> const &c) { ex::convolution_by_radix4(c, c); };
}

static_assert(experimental_ntt_ready<convolution_mint>);
static_assert(experimental_ntt_rejected<mod_int<uint32, 129>>);			// composite
static_assert(experimental_ntt_rejected<mod_int<uint32, 3221225473u>>); // 3 * 2^30 + 1: over 30 bits

TEST_CASE("convolution: FFT split at the size and magnitude bounds", "[nt]") {
	using LimitZ = mod_int<uint32, uint32(1) << 30>;
	int const height = GENERATE(19, 20); // FFT limit, then Garner fallback
	std::size_t const n = std::size_t(1) << (height - 1);
	// Centered value 2^29 - 2^14 splits into (-2^14, 2^14).
	LimitZ const coefficient((uint32(1) << 29) - (uint32(1) << 14));
	std::vector<LimitZ> a(n, coefficient), b(n, coefficient);
	REQUIRE(detail::planned_ntt_size(n, n) == (std::size_t(1) << height));
	auto const product = convolution(a, b);
	auto const square = convolution(a, a);
	std::ranges::fill(b, -coefficient);
	auto const negative_product = convolution(a, b);
	bool has_exact_product = true, has_exact_square = true, has_exact_negative_product = true;
	for (std::size_t k = 0; k < product.size(); ++k) {
		LimitZ const expected = coefficient * coefficient * LimitZ(k < n ? k + 1 : 2 * n - 1 - k);
		has_exact_product = has_exact_product && product[k] == expected;
		has_exact_negative_product = has_exact_negative_product && negative_product[k] == -expected;
		has_exact_square = has_exact_square && square[k] == expected;
	}
	CHECK(has_exact_product);
	CHECK(has_exact_square);
	CHECK(has_exact_negative_product);
}

TEST_CASE("convolution: coefficient and size preconditions", "[nt]") {
	CHECK(atl_test::aborts([] { (void)convolution_uint64(std::vector<int>{-1}, std::vector<int>{1}); }));
	CHECK(atl_test::aborts([] { (void)convolution_uint64(std::vector<uint128>{uint128(1) << 64}, std::vector<int>{1}); }));
	CHECK(atl_test::aborts([] { (void)convolution_uint64(std::vector<uint64>{(std::numeric_limits<uint64>::max)()}, std::vector<uint64>{2}); }));
	CHECK(atl_test::aborts([] { std::vector<convolution_mint> f(3); ntt(f); }));
	CHECK(atl_test::aborts([] { std::vector<mod_int<uint32, 3>> f(4); inverse_ntt(f); }));
	CHECK(atl_test::aborts([] { (void)detail::planned_ntt_size(std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 1), 1); }));
	std::vector<uint128> const a{uint128(7), uint128(11)};
	CHECK(convolution_uint64(a, a) == std::vector<uint64>{49, 154, 121});
	CHECK(convolution_uint64(std::vector<uint128>{uint128((std::numeric_limits<uint64>::max)())}, std::vector<int>{0}) == std::vector<uint64>{0});
}

TEST_CASE("convolution: exact CRT coefficient boundaries", "[nt]") {
	constexpr uint64 pair_product = uint64(detail::crt_primes[1]) * detail::crt_primes[2];
	CHECK(convolution_uint64(std::vector<uint64>{pair_product - 1}, std::vector<unsigned>{1}) == std::vector<uint64>{pair_product - 1});
	using WideElement = mod_int<uint64, 18446744073709551557ull>;
	std::vector<WideElement> const a{uint64(detail::crt_primes[0]) * detail::crt_primes[1]};
	std::vector<WideElement> const at_limit{detail::crt_primes[2]};
	std::vector<WideElement> const within_limit{detail::crt_primes[2] - 1};
	CHECK_FALSE(detail::garner_fits(a, at_limit));
	CHECK(detail::garner_fits(a, within_limit));
}

TEST_CASE("convolution_uint64: adaptive CRT against widened definition", "[nt]") {
	constexpr uint64 pair_product = uint64(detail::crt_primes[1]) * detail::crt_primes[2];
	constexpr uint64 largest = (std::numeric_limits<uint64>::max)();
	auto const check = [](std::vector<uint64> const &a, std::vector<uint64> const &b) {
		std::vector<uint128> expected(a.size() + b.size() - 1);
		for (std::size_t i = 0; i < a.size(); ++i)
			for (std::size_t j = 0; j < b.size(); ++j) expected[i + j] += uint128(a[i]) * b[j];
		auto const got = convolution_uint64(a, b);
		REQUIRE(got.size() == expected.size());
		for (std::size_t i = 0; i < got.size(); ++i) CHECK(uint128(got[i]) == expected[i]);
	};
	check({pair_product - 1}, {1});
	check({pair_product}, {1});
	check({largest}, {1});
	check({largest - 1, 1}, {1});
	check(std::vector<uint64>(49, (pair_product - 1) / 49), std::vector<uint64>(49, 1));
	check(std::vector<uint64>(49, pair_product / 49 + 1), std::vector<uint64>(49, 1));
	check(std::vector<uint64>(49, largest / 49), std::vector<uint64>(49, 1));
	std::vector<uint64> const square(49, 600000000);
	check(square, square);
	check(std::vector<uint64>(49, largest / 49), std::vector<uint64>(400, 1));
	std::mt19937_64 rng(1703);
	for (unsigned trial = 0; trial < 4; ++trial) {
		std::vector<uint64> a(49 + trial), b(61);
		for (auto &value : a) value = rng() % 500000000;
		for (auto &value : b) value = rng() % 500000000;
		check(a, b);
	}
	CHECK(atl_test::aborts([] {
		std::vector<uint64> const a(49, (std::numeric_limits<uint64>::max)());
		(void)convolution_uint64(a, a);
	}));
}
