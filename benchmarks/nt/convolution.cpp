/**
 * @file
 * @brief Polynomial-product throughput: balanced and square 2^19 inputs,
 * lopsided 2^20 by 2^6, batches of 20000 tiny products, arbitrary moduli and
 * exact uint64 with two or three primes. Each sample includes scratch allocation
 * and returns a coefficient. The three-prime shape is not measured yet.
 * Contenders isolate bit reversal (by radix2), incremental rates (by DIF),
 * deferred reduction (by lazy Montgomery), fused levels (by radix4), vector
 * pointwise/scaling, per-call rates with scalar block reduction, one padded FFT
 * split, and a prescaled overlap-add spectrum. Library engines run directly.
 * Source: this file; Xeon Platinum 8280, GCC 15.2, RelWithDebInfo, assertions
 * off, 2026-10-15, -O2 -g -DNDEBUG -std=c++23, with/without -march=native.
 * Balanced library scalar: 82.8 / 81.0 ms; conv8: 21.3 / 21.1 ms (plain /
 * native). Vector engines use targeted AVX2 with either flag set.
 * Full tables: docs/convolution.md.
 */
#include <atl/experimental/nt/convolution_variants.hpp>
#include <atl/nt/convolution.hpp>
#include <atl/nt/detail/ntt_conv8.hpp>
#include <atl/nt/detail/ntt_simd.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <bit>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <random>
#include <type_traits>
#include <vector>

using namespace atl;
namespace ex = atl::experimental;
using Z = mod_int<uint32, 998244353>;
using M7 = mod_int<uint32, 1000000007>;
using Z7 = mod_int<uint32, 7340033>; // 7·2^20 + 1: 2^20 of 2-adic headroom

namespace {
	template<typename Coefficient>
		requires std::is_invocable_r_v<Z, Coefficient &>
	Z sum_product_coefficients(Coefficient coefficient) {
		Z sum;
		for (int i = 0; i < 20000; ++i) sum += coefficient();
		return sum;
	}
}

namespace contenders {
	/** @brief simd_ntt with a vectorized pointwise product and scaling. */
	struct ntt_by_vector_pointwise : ntt_engine_base<ntt_by_vector_pointwise> {
		template<static_mod_int M>
		static constexpr bool supports() {
			return simd_ntt::supports<M>();
		}

		template<static_mod_int M>
			requires(supports<M>())
		static void forward(uint32 *f, uint32 h) {
			simd_ntt::forward<M>(f, h);
		}

		template<static_mod_int M>
			requires(supports<M>())
		static void inverse(uint32 *f, uint32 h) {
			simd_ntt::inverse<M>(f, h);
		}

		template<static_mod_int M>
			requires(supports<M>())
		ATL_SIMD_TARGET static void convolve(uint32 *fa, uint32 *fb, uint32 h) {
			detail::check_ntt_levels<M>(h);
			atl_assert(h >= 3);
			std::size_t const size = std::size_t(1) << h;
			simd_ntt::forward<M>(fa, h);
			simd_ntt::forward<M>(fb, h);
			for (std::size_t i = 0; i < size; i += 8)
				detail::store8(fa + i, detail::mul_lazy8<M>(detail::load8(fa + i), detail::load8(fb + i)));
			simd_ntt::inverse<M>(fa, h);
			detail::scale_tight<M>(fa, size, detail::inverse_scale<M>(h));
		}
	};

	/** @brief Fused eight-point convolution with per-call rates and scalar block reduction. */
	struct convolution_by_scalar_block_reduce : ntt_engine_base<convolution_by_scalar_block_reduce> {
		template<static_mod_int M>
		static constexpr bool supports() {
			return simd_ntt::supports<M>();
		}

		template<static_mod_int M>
			requires(supports<M>())
		static void forward(uint32 *f, uint32 h) {
			simd_ntt::forward<M>(f, h);
		}

		template<static_mod_int M>
			requires(supports<M>())
		static void inverse(uint32 *f, uint32 h) {
			simd_ntt::inverse<M>(f, h);
		}

		template<static_mod_int M>
			requires(supports<M>())
		ATL_SIMD_TARGET static void block(uint32 *a, uint32 *b, uint32 w_mont) {
			using mont = montgomery<M::mod()>;
			constexpr uint32 mod = M::mod();
			constexpr uint32 mod_inv = mont::companion(1u);
			auto const redc = [](uint64 x) ATL_SIMD_TARGET -> uint32 {
				uint32 const q = uint32(x) * mod_inv;
				uint32 const hi = uint32((uint64(q) * mod) >> 32);
				return uint32(x >> 32) + mod - hi;
			};
			uint32 b_canon[8];
			alignas(32) uint32 window[16];
			for (int i = 0; i < 8; ++i) {
				uint32 x = a[i];
				x = x >= mod ? x - mod : x;
				uint32 y = b[i];
				b_canon[i] = y >= mod ? y - mod : y;
				window[8 + i] = x;
				uint32 aw = redc(uint64(x) * w_mont);
				window[i] = aw >= mod ? aw - mod : aw;
			}
			detail::u32x8 acc_even{}, acc_odd{};
			for (int i = 0; i < 8; ++i) {
				detail::u32x8 const bi = detail::splat8(b_canon[i]);
				detail::u32x8 const win = detail::load8(window + 8 - i);
				acc_even = detail::add64(acc_even, detail::mul_even64(bi, win));
				acc_odd = detail::add64(acc_odd, detail::mul_even64(bi, detail::shr64_32(win)));
			}
			for (std::size_t k = 0; k < 4; ++k) {
				uint64 const sum_even = uint64(acc_even[2 * k]) | (uint64(acc_even[2 * k + 1]) << 32);
				uint64 const sum_odd = uint64(acc_odd[2 * k]) | (uint64(acc_odd[2 * k + 1]) << 32);
				uint32 e = redc(sum_even), o = redc(sum_odd);
				if (e >= 2 * mod) e -= 2 * mod;
				if (o >= 2 * mod) o -= 2 * mod;
				a[2 * std::size_t(k)] = e;
				a[2 * std::size_t(k) + 1] = o;
			}
		}

		template<static_mod_int M>
			requires(supports<M>())
		ATL_SIMD_TARGET static void convolve(uint32 *fa, uint32 *fb, uint32 h) {
			detail::check_ntt_levels<M>(h);
			atl_assert(h >= 3);
			using mont = montgomery<M::mod()>;
			constexpr uint32 mod = M::mod();
			std::size_t const size = std::size_t(1) << h;
			simd_ntt::forward<M>(fa, h, 3);
			simd_ntt::forward<M>(fb, h, 3);
			std::size_t const blocks = size >> 3;
			uint32 const block_levels = uint32(std::countr_zero(blocks));
			uint32 const zeta = power(detail::primitive_root<M>(), (mod - 1) / uint32(blocks)).value();
			uint32 block_rate[32];
			for (uint32 j = 0; j < block_levels; ++j)
				block_rate[j] = mont::to_form(
					power(M(zeta), (uint32(1) << (block_levels - 1 - j)) + (uint32(1) << (block_levels - j)))
						.value());
			uint32 w_mont = mont::one();
			for (std::size_t b = 0; b < blocks; ++b) {
				block<M>(fa + 8 * b, fb + 8 * b, w_mont);
				if (b + 1 < blocks) w_mont = mont::mul(w_mont, block_rate[std::countr_zero(~b)]);
			}
			simd_ntt::inverse<M>(fa, h, 3);
			uint32 const rr = mont::to_form(mont::to_form(uint32(1)));
			uint32 const scale = mont::mul(mont::to_form(power(M(uint32(blocks)), mod - 2).value()), rr);
			for (std::size_t i = 0; i < size; ++i) {
				uint32 const x = mont::mul_lazy(fa[i], scale);
				fa[i] = x >= mod ? x - mod : x;
			}
		}
	};

	/** @brief A balanced FFT split using one padded transform. */
	template<static_mod_int M>
		requires(sizeof(decltype(M::mod())) == 4 && M::mod() <= (uint64(1) << 30))
	std::vector<M> convolution_by_padded_fft_split(std::vector<M> const &a, std::vector<M> const &b) {
		std::size_t const n = a.size(), m = b.size(), need = n + m - 1, size = std::bit_ceil(need);
		auto const split = [](uint32 v) {
			int64 const sv = v > M::mod() / 2 ? int64(v) - int64(M::mod()) : int64(v);
			int64 const hi = (sv + (int64(1) << 14)) >> 15;
			return cplx{double(sv - hi * (int64(1) << 15)), double(hi)};
		};
		std::vector<cplx> in1(size), in2(size);
		for (std::size_t i = 0; i < n; ++i) in1[i] = split(a[i].value());
		for (std::size_t i = 0; i < m; ++i) in2[i] = split(b[i].value());
		fft(in1);
		fft(in2);
		std::vector<cplx> out1(size), out2(size);
		for (std::size_t j = 0; j < size; ++j) {
			std::size_t const k = (size - j) & (size - 1);
			cplx const pa = (in1[j] + in1[k].conj()) * cplx{0.5, 0};
			cplx const qa = (in1[j] - in1[k].conj()) * cplx{0, -0.5};
			out1[j] = (pa * in2[j]).conj();
			out2[j] = (qa * in2[j]).conj();
		}
		fft(out1);
		fft(out2);
		double const inv = 1.0 / double(size);
		int64 const mod = int64(M::mod());
		auto const norm = [&](int64 x) { return uint64(((x % mod) + mod) % mod); };
		std::vector<M> c(need);
		for (std::size_t i = 0; i < need; ++i) {
			int64 const ll = std::llround(out1[i].re * inv);
			int64 const lh_hl = std::llround(-out1[i].im * inv) + std::llround(out2[i].re * inv);
			int64 const hh = std::llround(-out2[i].im * inv);
			uint64 const sum = norm(ll) + (norm(lh_hl) << 15) % uint64(mod) + (norm(hh) << 30) % uint64(mod);
			c[i] = M(sum % uint64(mod));
		}
		return c;
	}

	/** @brief The overlap-add tier with the 1/size scale folded into `fs` once (a tie). */
	template<ntt_engine Eng>
	std::vector<uint32> convolution_by_prescaled_spectrum(std::vector<Z> const &a, std::vector<Z> const &b) {
		using mont = montgomery<Z::mod()>;
		constexpr uint32 mod = Z::mod();
		std::size_t const n = a.size(), m = b.size(), need = n + m - 1;
		std::size_t const shortest = (std::min)(n, m), longest = (std::max)(n, m);
		std::vector<Z> const &s = n <= m ? a : b, &l = n <= m ? b : a;
		std::size_t const size = std::bit_ceil(2 * shortest);
		uint32 const h = uint32(std::countr_zero(size));
		uint32 const rr = mont::to_form(mont::to_form(uint32(1)));
		uint32 const scale = mont::mul(mont::to_form(power(Z(uint32(size)), mod - 2).value()), rr);
		std::size_t const step = size - shortest + 1;
		std::vector<uint32> fs(size, 0), chunk(size), acc(need + size, 0);
		for (std::size_t i = 0; i < shortest; ++i) fs[i] = s[i].value();
		Eng::template forward<Z>(fs.data(), h);
		for (uint32 &x : fs) x = mont::mul_lazy(x, scale);
		for (std::size_t at = 0; at < longest; at += step) {
			std::size_t const len = (std::min)(step, longest - at);
			for (std::size_t i = 0; i < len; ++i) chunk[i] = l[at + i].value();
			std::ranges::fill(chunk.begin() + len, chunk.end(), 0u);
			Eng::template forward<Z>(chunk.data(), h);
			for (std::size_t i = 0; i < size; ++i) chunk[i] = mont::mul_lazy(chunk[i], fs[i]);
			Eng::template inverse<Z>(chunk.data(), h);
			for (std::size_t i = 0; i < size; ++i) {
				uint32 x = chunk[i];
				if (x >= mod) x -= mod;
				uint32 const sum = acc[at + i] + x;
				acc[at + i] = sum >= mod ? sum - mod : sum;
			}
		}
		acc.resize(need);
		return acc;
	}
}

TEST_CASE("convolution contenders", "[convolution]") {
	std::mt19937 rng(1);
	auto make_input = [&]<static_mod_int M = Z>(std::size_t n) {
		std::vector<M> v(n);
		for (auto &x : v) x = M(uint32(rng() % M::mod()));
		return v;
	};

	{
		auto const a = make_input(1 << 19), b = make_input(1 << 19);
		auto const expected = convolution(a, b);
		REQUIRE(expected == ex::convolution_by_lazy_montgomery(a, b));
		REQUIRE(expected == ex::convolution_by_radix2(a, b));
		REQUIRE(expected == ex::convolution_by_dif(a, b));
		REQUIRE(expected == ex::convolution_by_radix4(a, b));
		REQUIRE(expected == convolution<ex::ntt_by_lazy_montgomery>(a, b));
		REQUIRE(expected == convolution<ex::ntt_by_radix4>(a, b));
		REQUIRE(expected == convolution<simd_ntt>(a, b));
		REQUIRE(expected == convolution<conv8_ntt>(a, b));
		REQUIRE(expected == convolution<contenders::convolution_by_scalar_block_reduce>(a, b));
		REQUIRE(expected == convolution<contenders::ntt_by_vector_pointwise>(a, b));
		auto const a_copy = a;
		auto const square = convolution(a, a_copy);
		REQUIRE(square == convolution(a, a));
		REQUIRE(square == convolution<simd_ntt>(a, a));
		REQUIRE(square == convolution<conv8_ntt>(a, a));
		REQUIRE(square == convolution<ex::ntt_by_radix4>(a, a));
		BENCHMARK("balanced 2^19 | by radix2") { return ex::convolution_by_radix2(a, b)[1]; };
		BENCHMARK("balanced 2^19 | by DIF") { return ex::convolution_by_dif(a, b)[1]; };
		BENCHMARK("balanced 2^19 | by lazy Montgomery") { return ex::convolution_by_lazy_montgomery(a, b)[1]; };
		BENCHMARK("balanced 2^19 | by radix4") { return ex::convolution_by_radix4(a, b)[1]; };
		BENCHMARK("balanced 2^19 | library") { return convolution(a, b)[1]; };
		BENCHMARK("balanced 2^19 | lazy Montgomery engine") { return convolution<ex::ntt_by_lazy_montgomery>(a, b)[1]; };
		BENCHMARK("balanced 2^19 | radix-4 engine") {
			return convolution<ex::ntt_by_radix4>(a, b)[1];
		};
		BENCHMARK("balanced 2^19 | by vector pointwise") {
			return convolution<contenders::ntt_by_vector_pointwise>(a, b)[1];
		};
		BENCHMARK("balanced 2^19 | simd engine") { return convolution<simd_ntt>(a, b)[1]; };
		BENCHMARK("balanced 2^19 | by scalar block reduce") { return convolution<contenders::convolution_by_scalar_block_reduce>(a, b)[1]; };
		BENCHMARK("balanced 2^19 | conv8 engine") { return convolution<conv8_ntt>(a, b)[1]; };
		BENCHMARK("square 2^19 | library") { return convolution(a, a)[1]; };
		BENCHMARK("square 2^19 | radix-4 engine") { return convolution<ex::ntt_by_radix4>(a, a)[1]; };
		BENCHMARK("square 2^19 | simd engine") { return convolution<simd_ntt>(a, a)[1]; };
		BENCHMARK("square 2^19 | conv8 engine") { return convolution<conv8_ntt>(a, a)[1]; };
	}
	{
		auto const a = make_input(1 << 20), b = make_input(1 << 6);
		auto const want = convolution(a, b);
		std::vector<uint32> expected;
		for (Z x : want) expected.push_back(x.value());
		REQUIRE(want == ex::convolution_by_lazy_montgomery(a, b));
		REQUIRE(want == convolution<ex::ntt_by_lazy_montgomery>(a, b));
		REQUIRE(want == convolution<ex::ntt_by_radix4>(a, b));
		REQUIRE(want == convolution<simd_ntt>(a, b));
		REQUIRE(expected == contenders::convolution_by_prescaled_spectrum<scalar_ntt>(a, b));
		BENCHMARK("lopsided 2^20 x 2^6 | by lazy Montgomery") { return ex::convolution_by_lazy_montgomery(a, b)[1]; };
		BENCHMARK("lopsided 2^20 x 2^6 | by prescaled spectrum") {
			return contenders::convolution_by_prescaled_spectrum<scalar_ntt>(a, b)[1];
		};
		BENCHMARK("lopsided 2^20 x 2^6 | library") { return convolution(a, b)[1]; };
		BENCHMARK("lopsided 2^20 x 2^6 | lazy Montgomery engine") {
			return convolution<ex::ntt_by_lazy_montgomery>(a, b)[1];
		};
		BENCHMARK("lopsided 2^20 x 2^6 | radix-4 engine") {
			return convolution<ex::ntt_by_radix4>(a, b)[1];
		};
		BENCHMARK("lopsided 2^20 x 2^6 | simd engine") { return convolution<simd_ntt>(a, b)[1]; };
	}
	{
		// 2^20 x 2^6 over 7340033: the padded size 2^21 exceeds the headroom,
		// the planned overlap-add size 128 does not
		auto const a = make_input.operator()<Z7>(1 << 20), b = make_input.operator()<Z7>(1 << 6);
		REQUIRE(convolution(a, b) == contenders::convolution_by_padded_fft_split(a, b));
		BENCHMARK("lopsided 2^20 x 2^6 mod 7340033 | by padded FFT split") {
			return contenders::convolution_by_padded_fft_split(a, b)[1];
		};
		BENCHMARK("lopsided 2^20 x 2^6 mod 7340033 | library") { return convolution(a, b)[1]; };
		auto const c = make_input.operator()<M7>(1 << 20), d = make_input.operator()<M7>(1 << 6);
		REQUIRE(convolution(c, d) == contenders::convolution_by_padded_fft_split(c, d));
		BENCHMARK("lopsided 2^20 x 2^6 mod 1e9+7 | by padded FFT split") {
			return contenders::convolution_by_padded_fft_split(c, d)[1];
		};
		BENCHMARK("lopsided 2^20 x 2^6 mod 1e9+7 | library (overlap-add fft split)") {
			return convolution(c, d)[1];
		};
	}
	{
		auto const a = make_input(48), b = make_input(48);
		REQUIRE(convolution(a, b) == ex::convolution_by_definition(a, b));
		BENCHMARK("batch 20000 x (48x48) | by definition") {
			return sum_product_coefficients([&] { return ex::convolution_by_definition(a, b)[1]; });
		};
		BENCHMARK("batch 20000 x (48x48) | library") {
			return sum_product_coefficients([&] { return convolution(a, b)[1]; });
		};
		auto const c = make_input(64), d = make_input(64);
		auto const expected = ex::convolution_by_definition(c, d);
		REQUIRE(convolution(c, d) == expected);
		REQUIRE(convolution<simd_ntt>(c, d) == expected);
		REQUIRE(convolution<conv8_ntt>(c, d) == expected);
		REQUIRE(convolution<contenders::convolution_by_scalar_block_reduce>(c, d) == expected);
		BENCHMARK("batch 20000 x (64x64) | library") {
			return sum_product_coefficients([&] { return convolution(c, d)[1]; });
		};
		BENCHMARK("batch 20000 x (64x64) | simd engine") {
			return sum_product_coefficients([&] { return convolution<simd_ntt>(c, d)[1]; });
		};
		BENCHMARK("batch 20000 x (64x64) | by scalar block reduce") {
			return sum_product_coefficients([&] { return convolution<contenders::convolution_by_scalar_block_reduce>(c, d)[1]; });
		};
		BENCHMARK("batch 20000 x (64x64) | conv8 engine") {
			return sum_product_coefficients([&] { return convolution<conv8_ntt>(c, d)[1]; });
		};
	}
	{
		auto const a = make_input.operator()<M7>(1 << 19), b = make_input.operator()<M7>(1 << 19);
		auto const a_copy = a;
		REQUIRE(convolution(a, a) == convolution(a, a_copy));
		BENCHMARK("1e9+7 2^19 | library (fft split)") { return convolution(a, b)[1]; };
		BENCHMARK("1e9+7 square 2^19 | library (fft split)") { return convolution(a, a)[1]; };
		using ZBig = mod_int<uint32, 4294967291u>;
		auto const ga = make_input.operator()<ZBig>(1 << 19), gb = make_input.operator()<ZBig>(1 << 19);
		auto const ga_copy = ga;
		REQUIRE(convolution(ga, ga) == convolution(ga, ga_copy));
		REQUIRE(convolution(ga, gb) == convolution<conv8_ntt>(ga, gb));
		REQUIRE(convolution(ga, ga) == convolution<conv8_ntt>(ga, ga));
		BENCHMARK("2^32-5 2^19 | library (garner)") { return convolution(ga, gb)[1]; };
		BENCHMARK("2^32-5 square 2^19 | library (garner)") { return convolution(ga, ga)[1]; };
		BENCHMARK("2^32-5 2^19 | library Garner with conv8") { return convolution<conv8_ntt>(ga, gb)[1]; };
		BENCHMARK("2^32-5 square 2^19 | library Garner with conv8") { return convolution<conv8_ntt>(ga, ga)[1]; };
	}
	for (bool const has_three_prime_bound : {false, true}) {
		std::size_t const n = std::size_t(1) << 19;
		std::vector<uint32> a(n), b(n);
		for (auto &x : a) x = has_three_prime_bound ? 4000000u : uint32(rng() % 1000);
		for (auto &x : b) x = has_three_prime_bound ? 3000000u : uint32(rng() % 1000);
		auto const a_copy = a;
		REQUIRE(convolution_uint64(a, a) == convolution_uint64(a, a_copy));
		if (has_three_prime_bound) {
			std::vector<uint64> expected(2 * n - 1), expected_square(2 * n - 1);
			for (std::size_t k = 0; k < expected.size(); ++k) {
				std::size_t const pairs = k < n ? k + 1 : 2 * n - 1 - k;
				expected[k] = uint64(uint128(pairs) * a[0] * b[0]);
				expected_square[k] = uint64(uint128(pairs) * a[0] * a[0]);
			}
			REQUIRE(convolution_uint64(a, b) == expected);
			REQUIRE(convolution_uint64(a, a) == expected_square);
		}
		BENCHMARK(has_three_prime_bound ? "uint64 three-prime 2^19 | library" : "uint64 2^19 | library") { return convolution_uint64(a, b)[1]; };
		BENCHMARK(has_three_prime_bound ? "uint64 three-prime square 2^19 | library" : "uint64 square 2^19 | library") { return convolution_uint64(a, a)[1]; };
	}
}
