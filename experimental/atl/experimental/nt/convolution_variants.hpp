#pragma once
#include <atl/nt/convolution.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <bit>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Convolution contenders for the algorithm race.
 *
 * Shared contract: `f(a, b)` over vectors of an NTT-friendly `mod_int`
 * returns the cyclic-free product of sizes |a|+|b|-1; every contender is
 * cross-checked against the naive oracle in tests (small sizes only).
 */
namespace atl::experimental {
	using atl::mod_int_range;
	using atl::static_mod_int;

	namespace detail {
		using atl::detail::ntt_modulus;
		using atl::detail::primitive_root;
	}

	/**
	 * @brief The definitional product: the oracle.
	 * @complexity O(|a| |b|)
	 */
	template<mod_int_range A, mod_int_range B>
		requires std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>
	auto convolution_by_definition(A const &a, B const &b) {
		using Z = std::ranges::range_value_t<A>;
		std::size_t const n = std::ranges::size(a), m = std::ranges::size(b);
		if (n == 0 || m == 0) return std::vector<Z>{};
		std::vector<Z> c(n + m - 1);
		for (std::size_t i = 0; i < n; ++i)
			for (std::size_t j = 0; j < m; ++j) c[i + j] += a[i] * b[j];
		return c;
	}

	/**
	 * @brief Textbook iterative radix-2 Cooley-Tukey with an explicit
	 *        bit-reversal permutation, arithmetic through mod_int: the race
	 *        baseline.
	 * @pre |a| is a power of two dividing mod - 1
	 * @complexity O(n log n)
	 */
	template<mod_int_range R>
		requires detail::ntt_modulus<std::ranges::range_value_t<R>> && std::ranges::output_range<R, std::ranges::range_value_t<R>>
	void ntt_by_radix2(R &&a, bool should_invert) {
		using Z = std::ranges::range_value_t<R>;
		std::size_t const n = std::ranges::size(a);
		atl_assert(std::has_single_bit(n) && (Z::mod() - 1) % n == 0);
		for (std::size_t i = 1, j = 0; i < n; ++i) {
			std::size_t bit = n >> 1;
			for (; j & bit; bit >>= 1) j ^= bit;
			j ^= bit;
			if (i < j) std::swap(a[i], a[j]);
		}
		for (std::size_t len = 2; len <= n; len <<= 1) {
			Z w = power(detail::primitive_root<Z>(), (Z::mod() - 1) / len);
			if (should_invert) w = power(w, Z::mod() - 2);
			for (std::size_t i = 0; i < n; i += len) {
				Z cur(1);
				for (std::size_t k = 0; k < len / 2; ++k) {
					Z const even = a[i + k], odd = a[i + k + len / 2] * cur;
					a[i + k] = even + odd;
					a[i + k + len / 2] = even - odd;
					cur *= w;
				}
			}
		}
		if (should_invert) {
			Z const scale = power(Z(n), Z::mod() - 2);
			for (auto &x : a) x *= scale;
		}
	}

	/**
	 * @brief Product through `ntt_by_radix2`.
	 * @pre bit_ceil(|a| + |b| - 1) divides mod - 1
	 * @complexity O(n log n), n = bit_ceil(|a| + |b| - 1)
	 */
	template<mod_int_range A, mod_int_range B>
		requires std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>
			  && detail::ntt_modulus<std::ranges::range_value_t<A>>
	auto convolution_by_radix2(A const &a, B const &b) {
		using Z = std::ranges::range_value_t<A>;
		if (std::ranges::empty(a) || std::ranges::empty(b)) return std::vector<Z>{};
		std::size_t const need = std::ranges::size(a) + std::ranges::size(b) - 1;
		std::size_t const n = std::bit_ceil(need);
		atl_assert((Z::mod() - 1) % n == 0);
		std::vector<Z> fa(std::ranges::begin(a), std::ranges::end(a));
		std::vector<Z> fb(std::ranges::begin(b), std::ranges::end(b));
		fa.resize(n);
		fb.resize(n);
		ntt_by_radix2(fa, false);
		ntt_by_radix2(fb, false);
		for (std::size_t i = 0; i < n; ++i) fa[i] *= fb[i];
		ntt_by_radix2(fa, true);
		fa.resize(need);
		return fa;
	}

	/**
	 * @brief The root ladder and the incremental "rate" twiddles of the
	 *        DIF/DIT contenders: `rate[i]` advances the per-block twiddle
	 *        across a block whose index has i trailing ones, so it costs one
	 *        multiply instead of a root table.
	 *
	 * `root[k]` is a primitive 2^k-th root of unity (`iroot[k]` its inverse);
	 * the radix-4 engine reads its own 3-level rates off the same ladder.
	 * The modulus must be an `ntt_modulus`: its 2-adic order
	 * indexes the 30-entry tables.
	 */
	template<static_mod_int Z>
		requires detail::ntt_modulus<Z>
	struct ntt_rate_tables {
		Z root[30], iroot[30], rate[30], irate[30];
		uint32 levels;

		ntt_rate_tables() {
			auto const mod = Z::mod();
			levels = uint32(std::countr_zero(mod - 1));
			root[levels] = power(detail::primitive_root<Z>(), (mod - 1) >> levels);
			iroot[levels] = power(root[levels], mod - 2);
			for (uint32 i = levels; i-- > 0;) {
				root[i] = root[i + 1] * root[i + 1];
				iroot[i] = iroot[i + 1] * iroot[i + 1];
			}
			Z prod(1), iprod(1);
			for (uint32 i = 0; i + 2 <= levels; ++i) {
				rate[i] = root[i + 2] * prod;
				irate[i] = iroot[i + 2] * iprod;
				prod *= iroot[i + 2];
				iprod *= root[i + 2];
			}
		}
	};

	/**
	 * @brief DIF forward / DIT inverse with incremental `ntt_rate_tables`
	 *        twiddles.
	 *
	 * The forward transform leaves the spectrum bit-reversed and the inverse
	 * consumes it that way, so the O(n) permutation pass disappears.
	 * Arithmetic still through mod_int — this contender isolates the
	 * structural win.
	 * @pre |a| is a power of two dividing mod - 1
	 * @complexity O(n log n)
	 */
	template<mod_int_range R>
		requires detail::ntt_modulus<std::ranges::range_value_t<R>> && std::ranges::output_range<R, std::ranges::range_value_t<R>>
	void ntt_by_dif(R &&a, bool should_invert) {
		using Z = std::ranges::range_value_t<R>;
		static ntt_rate_tables<Z> const tables;
		std::size_t const n = std::ranges::size(a);
		atl_assert(std::has_single_bit(n) && (Z::mod() - 1) % n == 0);
		uint32 const h = uint32(std::countr_zero(n));
		if (!should_invert) {
			for (uint32 len = 0; len < h; ++len) {
				std::size_t const p = std::size_t(1) << (h - len - 1);
				Z rot(1);
				for (std::size_t s = 0; s < (std::size_t(1) << len); ++s) {
					std::size_t const offset = s << (h - len);
					for (std::size_t i = 0; i < p; ++i) {
						Z const l = a[i + offset];
						Z const r = a[i + offset + p] * rot;
						a[i + offset] = l + r;
						a[i + offset + p] = l - r;
					}
					if (s + 1 != (std::size_t(1) << len))
						rot *= tables.rate[std::countr_zero(~s)];
				}
			}
		} else {
			for (uint32 len = h; len-- > 0;) {
				std::size_t const p = std::size_t(1) << (h - len - 1);
				Z rot(1);
				for (std::size_t s = 0; s < (std::size_t(1) << len); ++s) {
					std::size_t const offset = s << (h - len);
					for (std::size_t i = 0; i < p; ++i) {
						Z const l = a[i + offset];
						Z const r = a[i + offset + p];
						a[i + offset] = l + r;
						a[i + offset + p] = (l - r) * rot;
					}
					if (s + 1 != (std::size_t(1) << len))
						rot *= tables.irate[std::countr_zero(~s)];
				}
			}
			Z const scale = power(Z(uint32(n)), Z::mod() - 2);
			for (auto &x : a) x *= scale;
		}
	}

	/**
	 * @brief Product through `ntt_by_dif`.
	 * @pre bit_ceil(|a| + |b| - 1) divides mod - 1
	 * @complexity O(n log n), n = bit_ceil(|a| + |b| - 1)
	 */
	template<mod_int_range A, mod_int_range B>
		requires std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>
			  && detail::ntt_modulus<std::ranges::range_value_t<A>>
	auto convolution_by_dif(A const &a, B const &b) {
		using Z = std::ranges::range_value_t<A>;
		if (std::ranges::empty(a) || std::ranges::empty(b)) return std::vector<Z>{};
		std::size_t const need = std::ranges::size(a) + std::ranges::size(b) - 1;
		std::size_t const n = std::bit_ceil(need);
		atl_assert((Z::mod() - 1) % n == 0);
		std::vector<Z> fa(std::ranges::begin(a), std::ranges::end(a));
		std::vector<Z> fb(std::ranges::begin(b), std::ranges::end(b));
		fa.resize(n);
		fb.resize(n);
		ntt_by_dif(fa, false);
		ntt_by_dif(fb, false);
		for (std::size_t i = 0; i < n; ++i) fa[i] *= fb[i];
		ntt_by_dif(fa, true);
		fa.resize(need);
		return fa;
	}

	namespace detail {
		/**
		 * @brief One padded single-prime transform through `Engine::convolve`,
		 *        with none of the library driver's tuning (no naive cutoff, no
		 *        overlap-add): the engine-only measurement the `*_convolution`
		 *        race wrappers stand for.
		 * @pre bit_ceil(|a| + |b| - 1) divides mod - 1
		 * @complexity O(n log n), n = bit_ceil(|a| + |b| - 1)
		 */
		template<ntt_engine Engine, mod_int_range A, mod_int_range B>
			requires std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>
				  && ntt_modulus<std::ranges::range_value_t<A>>
		auto engine_convolution(A const &a, B const &b) {
			using Z = std::ranges::range_value_t<A>;
			using U = decltype(Z::mod());
			if (std::ranges::empty(a) || std::ranges::empty(b)) return std::vector<Z>{};
			std::size_t const need = std::ranges::size(a) + std::ranges::size(b) - 1;
			std::size_t const n = std::bit_ceil(need);
			atl_assert((Z::mod() - 1) % n == 0);
			std::vector<U> fa(n, 0), fb(n, 0);
			for (std::size_t i = 0; i < std::ranges::size(a); ++i) fa[i] = a[i].value();
			for (std::size_t i = 0; i < std::ranges::size(b); ++i) fb[i] = b[i].value();
			Engine::template convolve<Z>(fa.data(), fb.data(), uint32(std::countr_zero(n)));
			std::vector<Z> out(need);
			for (std::size_t i = 0; i < need; ++i) out[i] = Z(fa[i]);
			return out;
		}
	}

	/**
	 * @brief The DIF/DIT structure on raw residue arrays with Harvey-style
	 *        lazy reduction, as an `atl::ntt_engine`.
	 *
	 * Data lives in [0, 2p) plain form, twiddles tight in Montgomery form (a
	 * plain x Montgomery product REDCs back to plain). Forward butterflies
	 * pay two conditional subtracts; the inverse's subtract feeds its
	 * multiply unreduced (< 4p against a tight twiddle stays under Mod·R),
	 * absorbing one. The library `atl::scalar_ntt` is this engine plus
	 * branchless subtracts and Shoup-style fixed-operand products.
	 *
	 * Through the library driver, balanced 2^19 takes 82.2 / 83.7 ms versus
	 * scalar_ntt's 82.8 / 81.0 ms (plain -O2 / -march=native): neither engine
	 * wins both configurations. Source: benchmarks/nt/convolution.cpp,
	 * Xeon 8280, GCC 15.2, RelWithDebInfo, assertions off, 2026-10-15.
	 */
	struct ntt_by_lazy_montgomery : ntt_engine_base<ntt_by_lazy_montgomery> {
		template<static_mod_int Z>
		static constexpr bool supports() {
			return detail::ntt_modulus<Z>;
		}

		/** @brief Per-modulus rate tables in tight Montgomery form, built once. */
		template<static_mod_int Z>
			requires detail::ntt_modulus<Z>
		struct twiddles {
			using U = decltype(Z::mod());
			U rate[30], irate[30];

			twiddles() {
				using mont = montgomery<Z::mod()>;
				ntt_rate_tables<Z> const t;
				for (int i = 0; i < 30; ++i) {
					rate[i] = mont::to_form(t.rate[i].value());
					irate[i] = mont::to_form(t.irate[i].value());
				}
			}
		};

		/** @brief The shared `twiddles` for `Z`, built on first use. */
		template<static_mod_int Z>
			requires detail::ntt_modulus<Z>
		static twiddles<Z> const &twiddle_table() {
			static twiddles<Z> const t;
			return t;
		}

		/**
		 * @brief Forward transform of `2^h` residues in place, natural order to
		 *        bit-reversed; inputs and outputs in `[0, 2·mod)`.
		 */
		template<static_mod_int Z>
			requires detail::ntt_modulus<Z>
		static void forward(decltype(Z::mod()) *f, uint32 h) {
			atl::detail::check_ntt_levels<Z>(h);
			using U = decltype(Z::mod());
			using mont = montgomery<Z::mod()>;
			constexpr U mod = Z::mod();
			auto const &t = twiddle_table<Z>();
			for (uint32 len = 0; len < h; ++len) {
				std::size_t const p = std::size_t(1) << (h - len - 1);
				U rot = mont::one();
				for (std::size_t s = 0; s < (std::size_t(1) << len); ++s) {
					U *lo = f + (s << (h - len));
					U *hi = lo + p;
					for (std::size_t i = 0; i < p; ++i) {
						U const l = lo[i];
						U const r = mont::mul_lazy(hi[i], rot);
						U const add = l + r;
						U const sub = l + 2 * mod - r;
						lo[i] = add >= 2 * mod ? add - 2 * mod : add;
						hi[i] = sub >= 2 * mod ? sub - 2 * mod : sub;
					}
					if (s + 1 != (std::size_t(1) << len))
						rot = mont::mul(rot, t.rate[std::countr_zero(~s)]);
				}
			}
		}

		/**
		 * @brief Inverse transform of `2^h` residues in place, bit-reversed to
		 *        natural order, without the `1/n` scaling; inputs and outputs in
		 *        `[0, 2·mod)`.
		 */
		template<static_mod_int Z>
			requires detail::ntt_modulus<Z>
		static void inverse(decltype(Z::mod()) *f, uint32 h) {
			atl::detail::check_ntt_levels<Z>(h);
			using U = decltype(Z::mod());
			using mont = montgomery<Z::mod()>;
			constexpr U mod = Z::mod();
			auto const &t = twiddle_table<Z>();
			for (uint32 len = h; len-- > 0;) {
				std::size_t const p = std::size_t(1) << (h - len - 1);
				U rot = mont::one();
				for (std::size_t s = 0; s < (std::size_t(1) << len); ++s) {
					U *lo = f + (s << (h - len));
					U *hi = lo + p;
					for (std::size_t i = 0; i < p; ++i) {
						U const l = lo[i];
						U const r = hi[i];
						U const add = l + r;
						lo[i] = add >= 2 * mod ? add - 2 * mod : add;
						hi[i] = mont::mul_lazy(l + 2 * mod - r, rot); // < 4p x tight
					}
					if (s + 1 != (std::size_t(1) << len))
						rot = mont::mul(rot, t.irate[std::countr_zero(~s)]);
				}
			}
		}
	};

	/**
	 * @brief Product through `ntt_by_lazy_montgomery`, one padded transform (the race
	 *        entry; `convolution<ntt_by_lazy_montgomery>` runs it through the library
	 *        driver instead).
	 * @pre bit_ceil(|a| + |b| - 1) divides mod - 1
	 * @complexity O(n log n), n = bit_ceil(|a| + |b| - 1)
	 */
	template<mod_int_range A, mod_int_range B>
		requires std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>
			  && detail::ntt_modulus<std::ranges::range_value_t<A>>
	auto convolution_by_lazy_montgomery(A const &a, B const &b) {
		return detail::engine_convolution<ntt_by_lazy_montgomery>(a, b);
	}

	/**
	 * @brief The lazy core with radix-4 passes, as an `atl::ntt_engine:`
	 *        two levels fused per sweep (an extra radix-2 level when the
	 *        height is odd), halving the passes over the array — the lever
	 *        once 2^20 elements stop fitting in cache.
	 *
	 * Bounds per line are folded just enough to stay under Mod·R against
	 * tight twiddles.
	 *
	 * Through the library driver, balanced 2^19 takes 84.4 / 83.1 ms versus
	 * scalar_ntt's 82.8 / 81.0 ms (plain -O2 / -march=native), so fewer
	 * memory passes do not repay the larger butterfly here. Source:
	 * benchmarks/nt/convolution.cpp, Xeon 8280, GCC 15.2, RelWithDebInfo,
	 * assertions off, 2026-10-15.
	 */
	struct ntt_by_radix4 : ntt_engine_base<ntt_by_radix4> {
		template<static_mod_int Z>
		static constexpr bool supports() {
			return detail::ntt_modulus<Z>;
		}

		/**
		 * @brief Per-modulus tables in tight Montgomery form, built once: the
		 *        radix-2 rates for the odd tail, the 3-level rates of the
		 *        fused passes, and the fourth root of unity.
		 */
		template<static_mod_int Z>
			requires detail::ntt_modulus<Z>
		struct twiddles {
			using U = decltype(Z::mod());
			U rate2[30], irate2[30], rate3[30], irate3[30], imag, iimag;

			twiddles() {
				using mont = montgomery<Z::mod()>;
				ntt_rate_tables<Z> const t;
				for (int i = 0; i < 30; ++i) {
					rate2[i] = mont::to_form(t.rate[i].value());
					irate2[i] = mont::to_form(t.irate[i].value());
				}
				imag = mont::to_form(t.root[2].value());
				iimag = mont::to_form(t.iroot[2].value());
				Z prod(1), iprod(1);
				for (uint32 i = 0; i + 3 <= t.levels; ++i) {
					rate3[i] = mont::to_form((t.root[i + 3] * prod).value());
					irate3[i] = mont::to_form((t.iroot[i + 3] * iprod).value());
					prod *= t.iroot[i + 3];
					iprod *= t.root[i + 3];
				}
			}
		};

		/** @brief The shared `twiddles` for `Z`, built on first use. */
		template<static_mod_int Z>
			requires detail::ntt_modulus<Z>
		static twiddles<Z> const &twiddle_table() {
			static twiddles<Z> const t;
			return t;
		}

		/**
		 * @brief Forward transform of `2^h` residues in place, natural order to
		 *        bit-reversed; inputs and outputs in `[0, 2·mod)`.
		 */
		template<static_mod_int Z>
			requires detail::ntt_modulus<Z>
		static void forward(decltype(Z::mod()) *f, uint32 h) {
			atl::detail::check_ntt_levels<Z>(h);
			using U = decltype(Z::mod());
			using mont = montgomery<Z::mod()>;
			constexpr U mod = Z::mod();
			auto const &tb = twiddle_table<Z>();
			uint32 len = 0;
			while (len < h) {
				if (h - len == 1) {
					U rot = mont::one();
					for (std::size_t s = 0; s < (std::size_t(1) << len); ++s) {
						U *x = f + (s << 1);
						U const l = x[0], r = mont::mul_lazy(x[1], rot);
						U const add = l + r, sub = l + 2 * mod - r;
						x[0] = add >= 2 * mod ? add - 2 * mod : add;
						x[1] = sub >= 2 * mod ? sub - 2 * mod : sub;
						if (s + 1 != (std::size_t(1) << len))
							rot = mont::mul(rot, tb.rate2[std::countr_zero(~s)]);
					}
					len += 1;
					continue;
				}
				std::size_t const p = std::size_t(1) << (h - len - 2);
				U rot = mont::one();
				for (std::size_t s = 0; s < (std::size_t(1) << len); ++s) {
					U const rot2 = mont::mul(rot, rot);
					U const rot3 = mont::mul(rot2, rot);
					U *f0 = f + (s << (h - len));
					U *f1 = f0 + p, *f2 = f0 + 2 * p, *f3 = f0 + 3 * p;
					for (std::size_t i = 0; i < p; ++i) {
						uint64 const a0 = f0[i];									 // < 2p
						uint64 const a1 = mont::mul_lazy(f1[i], rot);				 // < 2p
						uint64 const a2 = mont::mul_lazy(f2[i], rot2);				 // < 2p
						uint64 const a3 = mont::mul_lazy(f3[i], rot3);				 // < 2p
						U const d13 = mont::mul_lazy(U(a1 + 2 * mod - a3), tb.imag); // < 4p x tight
						f0[i] = fold<Z>(a0 + a1 + a2 + a3);
						f1[i] = fold<Z>(a0 + a2 + 4 * uint64(mod) - a1 - a3);
						f2[i] = fold<Z>(a0 + 2 * mod - a2 + d13);
						f3[i] = fold<Z>(a0 + 4 * uint64(mod) - a2 - d13);
					}
					if (s + 1 != (std::size_t(1) << len))
						rot = mont::mul(rot, tb.rate3[std::countr_zero(~s)]);
				}
				len += 2;
			}
		}

		/**
		 * @brief Inverse transform of `2^h` residues in place, bit-reversed to
		 *        natural order, without the `1/n` scaling; inputs and outputs in
		 *        `[0, 2·mod)`.
		 */
		template<static_mod_int Z>
			requires detail::ntt_modulus<Z>
		static void inverse(decltype(Z::mod()) *f, uint32 h) {
			atl::detail::check_ntt_levels<Z>(h);
			using U = decltype(Z::mod());
			using mont = montgomery<Z::mod()>;
			constexpr U mod = Z::mod();
			auto const &tb = twiddle_table<Z>();
			uint32 len = h;
			while (len > 0) {
				if (len & 1) { // the forward odd tail (p = 1) unwinds first
					U rot = mont::one();
					for (std::size_t s = 0; s < (std::size_t(1) << (len - 1)); ++s) {
						U *x = f + (s << 1);
						U const l = x[0], r = x[1];
						U const add = l + r;
						x[0] = add >= 2 * mod ? add - 2 * mod : add;
						x[1] = mont::mul_lazy(l + 2 * mod - r, rot);
						if (s + 1 != (std::size_t(1) << (len - 1)))
							rot = mont::mul(rot, tb.irate2[std::countr_zero(~s)]);
					}
					len -= 1;
					continue;
				}
				len -= 2;
				std::size_t const p = std::size_t(1) << (h - len - 2);
				U rot = mont::one();
				for (std::size_t s = 0; s < (std::size_t(1) << len); ++s) {
					U const rot2 = mont::mul(rot, rot);
					U const rot3 = mont::mul(rot2, rot);
					U *f0 = f + (s << (h - len));
					U *f1 = f0 + p, *f2 = f0 + 2 * p, *f3 = f0 + 3 * p;
					for (std::size_t i = 0; i < p; ++i) {
						uint64 const a0 = f0[i], a1 = f1[i], a2 = f2[i], a3 = f3[i];
						U const d23 = mont::mul_lazy(U(a2 + 2 * mod - a3), tb.iimag);
						uint64 const s01 = a0 + a1;			  // < 4p
						uint64 const d01 = a0 + 2 * mod - a1; // < 4p
						f0[i] = fold<Z>(s01 + a2 + a3);
						f1[i] = mont::mul_lazy(fold<Z>(d01 + d23), rot);
						f2[i] = mont::mul_lazy(fold<Z>(s01 + 4 * uint64(mod) - a2 - a3), rot2);
						f3[i] = mont::mul_lazy(fold<Z>(d01 + 2 * uint64(mod) - d23), rot3);
					}
					if (s + 1 != (std::size_t(1) << len))
						rot = mont::mul(rot, tb.irate3[std::countr_zero(~s)]);
				}
			}
		}

	private:
		/** @brief Folds a lazy sum below 8·mod back into `[0, 2·mod)`. */
		template<detail::ntt_modulus Z>
		static decltype(Z::mod()) fold(uint64 x) {
			constexpr uint64 mod = Z::mod();
			if (x >= 4 * mod) x -= 4 * mod;
			if (x >= 2 * mod) x -= 2 * mod;
			return decltype(Z::mod())(x);
		}
	};

	/**
	 * @brief Product through `ntt_by_radix4`, one padded transform (the race
	 *        entry; `convolution<ntt_by_radix4>` runs it through the library
	 *        driver instead).
	 * @pre bit_ceil(|a| + |b| - 1) divides mod - 1
	 * @complexity O(n log n), n = bit_ceil(|a| + |b| - 1)
	 */
	template<mod_int_range A, mod_int_range B>
		requires std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>
			  && detail::ntt_modulus<std::ranges::range_value_t<A>>
	auto convolution_by_radix4(A const &a, B const &b) {
		return detail::engine_convolution<ntt_by_radix4>(a, b);
	}
}
