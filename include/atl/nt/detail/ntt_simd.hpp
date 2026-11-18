#pragma once
#include <atl/nt/montgomery.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>

#if !defined(ATL_SIMD_PORTABLE) && !defined(ATL_FORCE_FALLBACKS)
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define ATL_SIMD_AVX2 1
#include <immintrin.h>
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define ATL_SIMD_NEON 1
#include <arm_neon.h>
#endif
#endif

// Function attributes work on GCC and Clang; Clang does not honour GCC's
// target pragmas. Keep vector helpers and their hot-loop callers on one ABI.
#if defined(ATL_SIMD_AVX2)
#define ATL_SIMD_TARGET __attribute__((target("avx2")))
#else
#define ATL_SIMD_TARGET
#endif

namespace atl::detail {
#if defined(ATL_SIMD_AVX2) || defined(ATL_SIMD_NEON)
	using u32x8 [[gnu::vector_size(32)]] = uint32;
#else
	struct u32x8 {
		uint32 lanes[8]{};

		uint32 &operator[](std::size_t i) { return lanes[i]; }

		uint32 const &operator[](std::size_t i) const { return lanes[i]; }

		friend u32x8 operator+(u32x8 a, u32x8 b) {
			for (std::size_t i = 0; i < 8; ++i) a[i] += b[i];
			return a;
		}

		friend u32x8 operator-(u32x8 a, u32x8 b) {
			for (std::size_t i = 0; i < 8; ++i) a[i] -= b[i];
			return a;
		}

		friend u32x8 operator*(u32x8 a, u32x8 b) {
			for (std::size_t i = 0; i < 8; ++i) a[i] *= b[i];
			return a;
		}
	};
#endif

	/** @brief Unaligned 8-lane load (memcpy compiles to one vector move). */
	ATL_SIMD_TARGET inline u32x8 load8(uint32 const *p) {
		u32x8 v{};
#if defined(ATL_SIMD_AVX2) || defined(ATL_SIMD_NEON)
		std::memcpy(&v, p, sizeof(v));
#else
		for (std::size_t i = 0; i < 8; ++i) v[i] = p[i];
#endif
		return v;
	}

	/** @brief Unaligned 8-lane store. */
	ATL_SIMD_TARGET inline void store8(uint32 *p, u32x8 v) {
#if defined(ATL_SIMD_AVX2) || defined(ATL_SIMD_NEON)
		std::memcpy(p, &v, sizeof(v));
#else
		for (std::size_t i = 0; i < 8; ++i) p[i] = v[i];
#endif
	}

	ATL_SIMD_TARGET inline u32x8 splat8(uint32 x) {
#if defined(ATL_SIMD_AVX2) || defined(ATL_SIMD_NEON)
		return u32x8{x, x, x, x, x, x, x, x};
#else
		u32x8 r{};
		for (std::size_t i = 0; i < 8; ++i) r[i] = x;
		return r;
#endif
	}

	/**
	 * @brief Lane-wise unsigned min (vpminud / NEON umin; GCC lowers the select).
	 * GCC vector compare yields all-ones (-1) on true, so the mask is direct.
	 */
	ATL_SIMD_TARGET inline u32x8 min8(u32x8 a, u32x8 b) {
#if defined(ATL_SIMD_AVX2) || defined(ATL_SIMD_NEON)
		u32x8 const m = (u32x8)(a < b);
		return (a & m) | (b & ~m);
#else
		for (std::size_t i = 0; i < 8; ++i) a[i] = (std::min)(a[i], b[i]);
		return a;
#endif
	}

	/** @brief high 32 bits of `a[i] * b[i]` (full 64-bit product) for each lane */
#if defined(ATL_SIMD_AVX2)
	ATL_SIMD_TARGET inline u32x8 mulhi8(u32x8 a, u32x8 b) {
		__m256i const lanes_a = (__m256i)a, lanes_b = (__m256i)b;
		__m256i const lo = _mm256_srli_epi64(_mm256_mul_epu32(lanes_a, lanes_b), 32);
		__m256i const hi = _mm256_mul_epu32(_mm256_srli_epi64(lanes_a, 32), _mm256_srli_epi64(lanes_b, 32));
		// even lanes carry hi(prod even) in low dword; odd lanes carry hi(prod odd)
		return (u32x8)_mm256_blend_epi32(lo, hi, 0xAA);
	}
#elif defined(ATL_SIMD_NEON)
	ATL_SIMD_TARGET inline u32x8 mulhi8(u32x8 av, u32x8 bv) {
		struct v2 {
			uint32x4_t x, y;
		}; // two 128-bit halves, reinterpreted (no lane shuffles)

		auto const a = std::bit_cast<v2>(av);
		auto const b = std::bit_cast<v2>(bv);
		uint32x4_t const h0 =
			vuzp2q_u32(vreinterpretq_u32_u64(vmull_u32(vget_low_u32(a.x), vget_low_u32(b.x))),
					   vreinterpretq_u32_u64(vmull_high_u32(a.x, b.x)));
		uint32x4_t const h1 =
			vuzp2q_u32(vreinterpretq_u32_u64(vmull_u32(vget_low_u32(a.y), vget_low_u32(b.y))),
					   vreinterpretq_u32_u64(vmull_high_u32(a.y, b.y)));
		return std::bit_cast<u32x8>(v2{h0, h1});
	}
#else
	ATL_SIMD_TARGET inline u32x8 mulhi8(u32x8 a, u32x8 b) {
		u32x8 r{};
		for (int i = 0; i < 8; ++i) r[i] = uint32((uint64(a[i]) * b[i]) >> 32);
		return r;
	}
#endif

	/**
	 * @brief Lane-wise `montgomery::mul_lazy:` `a·b·R⁻¹` in `[0, 2·mod)`.
	 * @pre every lane of `a` and `b` below `2·mod`.
	 */
	template<static_mod_int Z>
		requires(sizeof(decltype(Z::mod())) == 4 && scalar_ntt::supports<Z>())
	ATL_SIMD_TARGET inline u32x8 mul_lazy8(u32x8 a, u32x8 b) {
		using mont = montgomery<Z::mod()>;
		u32x8 const q = a * b * splat8(mont::companion(1u));
		return mulhi8(a, b) + splat8(Z::mod()) - mulhi8(q, splat8(Z::mod()));
	}

	/**
	 * @brief conv8's final scaling, `f[i] = f[i]·scale·R⁻¹`
	 *        tightened to `[0, mod)` over `size` lazy residues.
	 */
	template<static_mod_int Z>
		requires(sizeof(decltype(Z::mod())) == 4 && scalar_ntt::supports<Z>())
	ATL_SIMD_TARGET inline void scale_tight(uint32 *f, std::size_t size, uint32 scale) {
		using mont = montgomery<Z::mod()>;
		constexpr uint32 mod = Z::mod();
		u32x8 const vs = splat8(scale), vmod = splat8(mod);
		std::size_t i = 0;
		for (; i + 8 <= size; i += 8) {
			u32x8 const x = mul_lazy8<Z>(load8(f + i), vs);
			store8(f + i, min8(x, x - vmod));
		}
		for (; i < size; ++i) {
			uint32 const x = mont::mul_lazy(f[i], scale);
			f[i] = x >= mod ? x - mod : x;
		}
	}

	/**
	 * @brief The scale that `scale_tight` needs following a pointwise Montgomery
	 *        product and an unscaled inverse of `2^levels` points: `2^-levels`
	 *        times `R` twice over (one undoes the product's `R⁻¹`, one the
	 *        scaling's own).
	 * @pre `2^levels` divides `mod - 1`.
	 */
	template<static_mod_int Z>
		requires(sizeof(decltype(Z::mod())) == 4 && scalar_ntt::supports<Z>())
	inline uint32 inverse_scale(uint32 levels) {
		using mont = montgomery<Z::mod()>;
		constexpr uint32 mod = Z::mod();
		// 2^k · (mod - (mod - 1) / 2^k) = 1 (mod mod): the inverse in closed form
		uint32 const inverse_size = mod - ((mod - 1) >> levels);
		return mont::mul(mont::to_form(inverse_size), mont::to_form(mont::to_form(uint32(1))));
	}
}

namespace atl {
	/**
	 * @brief Eight-lane lazy-Montgomery DIF/DIT engine with targeted AVX2 on
	 *        GCC/Clang x86, baseline NEON on AArch64, or portable C++ lanes.
	 * @note Not constexpr (uses cached twiddles and optional intrinsics).
	 * @pre Raw buffers and levels satisfy the contract of atl::scalar_ntt.
	 * @pre On x86 with GCC/Clang, the CPU supports AVX2; otherwise execution
	 *      stops with an illegal instruction. MSVC, ATL_SIMD_PORTABLE and
	 *      ATL_FORCE_FALLBACKS use portable lanes. The default atl::scalar_ntt
	 *      has no AVX2 requirement.
	 */
	struct simd_ntt : ntt_engine_base<simd_ntt> {
		/** @brief Whether Z has a supported prime modulus and 32-bit representation. */
		template<static_mod_int Z>
		static constexpr bool supports() {
			using U = decltype(Z::mod());
			return sizeof(U) == 4 && scalar_ntt::supports<Z>();
		}

		/**
		 * @brief `stop` leaves the finest `stop` levels undone (coarse-only forward),
		 * for the stop-at-8 fused convolution; `stop == 0` is the full transform.
		 * @pre `stop <= h`; raw buffers satisfy atl::scalar_ntt's contract.
		 * @note Not constexpr (uses cached twiddles and optional intrinsics).
		 */
		template<static_mod_int Z>
			requires(supports<Z>())
		ATL_SIMD_TARGET static void forward(uint32 *f, uint32 h, uint32 stop = 0) {
			detail::check_ntt_levels<Z>(h);
			atl_assert(stop <= h);
			using mont = montgomery<Z::mod()>;
			constexpr uint32 mod = Z::mod();
			auto const &t = scalar_ntt::twiddle_table<Z>();
			detail::u32x8 const vmod = detail::splat8(mod), vmod2 = detail::splat8(2 * mod);
			for (uint32 len = 0; len + stop < h; ++len) {
				std::size_t const p = std::size_t(1) << (h - len - 1);
				uint32 rot = mont::one(), rotq = mont::companion(rot);
				for (std::size_t s = 0; s < (std::size_t(1) << len); ++s) {
					uint32 *lo = f + (s << (h - len));
					uint32 *hi = lo + p;
					if (p >= 8) {
						detail::u32x8 const vr = detail::splat8(rot), vrq = detail::splat8(rotq);
						for (std::size_t i = 0; i < p; i += 8) {
							detail::u32x8 const l = detail::load8(lo + i), x = detail::load8(hi + i);
							detail::u32x8 const r =
								detail::mulhi8(x, vr) + vmod - detail::mulhi8(x * vrq, vmod);
							detail::store8(lo + i, detail::min8(l + r, l + r - vmod2));
							detail::store8(hi + i, detail::min8(l - r, l + vmod2 - r));
						}
					} else {
						for (std::size_t i = 0; i < p; ++i) {
							uint32 const l = lo[i], r = mont::mul_lazy_fixed(hi[i], rot, rotq);
							lo[i] = (std::min)(l + r, l + r - 2 * mod);
							hi[i] = (std::min)(l - r, l + 2 * mod - r);
						}
					}
					if (s + 1 != (std::size_t(1) << len)) {
						rot = mont::mul(rot, t.step[std::countr_zero(~s)]);
						rotq = mont::companion(rot);
					}
				}
			}
		}

		/**
		 * @brief `stop` undoes only the coarse levels (`h-stop-1 .. 0`), skipping the
		 * finest `stop` levels; `stop == 0` performs the full inverse.
		 * @pre `stop <= h`; raw buffers satisfy atl::scalar_ntt's contract.
		 * @note Not constexpr (uses cached twiddles and optional intrinsics).
		 */
		template<static_mod_int Z>
			requires(supports<Z>())
		ATL_SIMD_TARGET static void inverse(uint32 *f, uint32 h, uint32 stop = 0) {
			detail::check_ntt_levels<Z>(h);
			atl_assert(stop <= h);
			using mont = montgomery<Z::mod()>;
			constexpr uint32 mod = Z::mod();
			auto const &t = scalar_ntt::twiddle_table<Z>();
			detail::u32x8 const vmod2 = detail::splat8(2 * mod);
			for (uint32 len = h - stop; len-- > 0;) {
				std::size_t const p = std::size_t(1) << (h - len - 1);
				uint32 rot = mont::one(), rotq = mont::companion(rot);
				for (std::size_t s = 0; s < (std::size_t(1) << len); ++s) {
					uint32 *lo = f + (s << (h - len));
					uint32 *hi = lo + p;
					if (p >= 8) {
						detail::u32x8 const vr = detail::splat8(rot), vrq = detail::splat8(rotq),
											vmod = detail::splat8(mod);
						for (std::size_t i = 0; i < p; i += 8) {
							detail::u32x8 const l = detail::load8(lo + i), r = detail::load8(hi + i);
							detail::store8(lo + i, detail::min8(l + r, l + r - vmod2));
							detail::u32x8 const d = l + vmod2 - r;
							detail::store8(hi + i,
										   detail::mulhi8(d, vr) + vmod - detail::mulhi8(d * vrq, vmod));
						}
					} else {
						for (std::size_t i = 0; i < p; ++i) {
							uint32 const l = lo[i], r = hi[i];
							lo[i] = (std::min)(l + r, l + r - 2 * mod);
							hi[i] = mont::mul_lazy_fixed(l + 2 * mod - r, rot, rotq);
						}
					}
					if (s + 1 != (std::size_t(1) << len)) {
						rot = mont::mul(rot, t.inverse_step[std::countr_zero(~s)]);
						rotq = mont::companion(rot);
					}
				}
			}
		}
	};
}
