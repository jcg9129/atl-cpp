#pragma once
#include <atl/nt/detail/primality_static.hpp>
#include <atl/nt/mod_int_static.hpp>
#include <atl/nt/montgomery_static.hpp>
#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace atl {
	namespace detail {
		template<typename Z>
		concept ntt_modulus = static_mod_int<Z>
						   && (sizeof(decltype(Z::mod())) == 4 || sizeof(decltype(Z::mod())) == 8)
						   && traits::has_widen_mul_v<decltype(Z::mod())>
						   && Z::mod() > 2 && Z::mod() < (uint64(1) << 30)
						   && Z::mod() <= (std::numeric_limits<decltype(Z::mod())>::max)() / 4
						   && static_is_prime<Z::mod()>();

		template<ntt_modulus Z>
		constexpr void check_ntt_levels(uint32 h) {
			atl_assert(h < std::numeric_limits<std::size_t>::digits
					   && h <= uint32(std::countr_zero(Z::mod() - 1)));
		}

		/**
		 * @brief The smallest primitive root of a prime modulus, found by trial: a
		 *        generator fails for some prime factor q of mod-1 iff
		 *        g^((mod-1)/q) == 1. Trial division suffices (mod < 2^32) and runs
		 *        at compile time.
		 */
		template<ntt_modulus Z>
		constexpr Z primitive_root() {
			using U = decltype(Z::mod());
			U const mod = Z::mod();
			U factors[32] = {};
			int count = 0;
			U m = mod - 1;
			for (U p = 2; p * p <= m; ++p)
				if (m % p == 0) {
					factors[count++] = p;
					while (m % p == 0) m /= p;
				}
			if (m > 1) factors[count++] = m;
			for (U g = 2;; ++g) {
				bool is_generator = true;
				for (int i = 0; i < count && is_generator; ++i)
					is_generator = power(Z(g), (mod - 1) / factors[i]) != Z(1);
				if (is_generator) return Z(g);
			}
		}
	}

	/**
	 * @brief Interface for single-prime raw-residue transforms and cyclic products.
	 * @tparam E Engine type providing static supports, forward, inverse and convolve templates.
	 * The signature probe uses `mod_int<uint32, 998244353>`; `supports<Z>()`
	 * determines per-modulus capability. Forward maps natural to bit-reversed
	 * order; inverse consumes that order without scaling. Both use `[0, 2p)`
	 * residues. Convolve returns canonical, fully scaled residues in its first
	 * buffer and overwrites its second buffer. The driver uses atl::scalar_ntt
	 * when the selected engine cannot serve a direct-NTT modulus.
	 */
	template<typename E>
	concept ntt_engine = requires(decltype(mod_int<uint32, 998244353>::mod()) *f, uint32 h) {
		{ E::template supports<mod_int<uint32, 998244353>>() } -> std::convertible_to<bool>;
		E::template forward<mod_int<uint32, 998244353>>(f, h);
		E::template inverse<mod_int<uint32, 998244353>>(f, h);
		requires(!E::template supports<mod_int<uint32, 998244353>>() || requires {
			E::template convolve<mod_int<uint32, 998244353>>(f, f, h);
		});
	};

	/**
	 * @brief CRTP base giving an engine the default single-prime `convolve`
	 *        from its own `forward`/`inverse`: two forward transforms, a lazy
	 *        pointwise product, one inverse, and the `1/n` (plus Montgomery)
	 *        scaling back to `[0, mod)`. Engines that fuse the pointwise into
	 *        the transform domain override `convolve` and skip this.
	 */
	template<typename Derived>
	struct ntt_engine_base {
	private:
		ntt_engine_base() = default;
		friend Derived;

	public:
		/**
		 * @brief Cyclic convolution of `fa` and `fb` (both `2^h` residues in
		 *        `[0, mod)`), fully scaled, into `fa` in `[0, mod)`; `fb` is
		 *        clobbered.
		 */
		template<static_mod_int Z>
			requires detail::ntt_modulus<Z> && (Derived::template supports<Z>())
		static constexpr void convolve(decltype(Z::mod()) *fa, decltype(Z::mod()) *fb, uint32 h) {
			detail::check_ntt_levels<Z>(h);
			using mont = montgomery<Z::mod()>;
			std::size_t const size = std::size_t(1) << h;
			Derived::template forward<Z>(fa, h);
			Derived::template forward<Z>(fb, h);
			for (std::size_t i = 0; i < size; ++i) fa[i] = mont::mul_lazy(fa[i], fb[i]);
			Derived::template inverse<Z>(fa, h);
			scale_down<Z>(fa, h);
		}

		/**
		 * @brief Cyclic convolution of `f` (`2^h` residues in `[0, mod)`) with
		 *        itself, fully scaled, into `f` in `[0, mod)`: one forward
		 *        transform where `convolve` runs two.
		 */
		template<static_mod_int Z>
			requires detail::ntt_modulus<Z> && (Derived::template supports<Z>())
		static constexpr void square(decltype(Z::mod()) *f, uint32 h) {
			detail::check_ntt_levels<Z>(h);
			using mont = montgomery<Z::mod()>;
			std::size_t const size = std::size_t(1) << h;
			Derived::template forward<Z>(f, h);
			for (std::size_t i = 0; i < size; ++i) f[i] = mont::mul_lazy(f[i], f[i]);
			Derived::template inverse<Z>(f, h);
			scale_down<Z>(f, h);
		}

	private:
		/**
		 * @brief Multiplies the `2^h` lazy residues of an unscaled inverse by
		 *        `1/2^h` (undoing the pointwise product's Montgomery factor too),
		 *        tightening them to `[0, mod)`.
		 */
		template<detail::ntt_modulus Z>
		static constexpr void scale_down(decltype(Z::mod()) *f, uint32 h) {
			using U = decltype(Z::mod());
			using mont = montgomery<Z::mod()>;
			constexpr U mod = Z::mod();
			std::size_t const size = std::size_t(1) << h;
			// 2^h · (mod - (mod - 1) / 2^h) = 1 (mod mod): the inverse in closed form
			U const inverse_size = mod - ((mod - 1) >> h);
			U const rr = mont::to_form(mont::to_form(U(1)));
			U const scale = mont::mul(mont::to_form(inverse_size), rr);
			for (std::size_t i = 0; i < size; ++i) {
				U const x = mont::mul_lazy(f[i], scale);
				f[i] = x >= mod ? x - mod : x;
			}
		}
	};

	/**
	 * @brief The portable scalar NTT engine (no compiler intrinsics — compiles
	 *        and runs anywhere; the baseline every vector engine is measured
	 *        against). Lazy-reduction DIF forward / DIT inverse: no bit-reversal
	 *        pass, twiddles maintained incrementally through tight-Montgomery
	 *        "step" tables; data lives as plain residues in `[0, 2*mod)` with
	 *        Harvey-style deferred reduction.
	 * @pre Z uses 32- or 64-bit unsigned residues and p is an odd prime below 2^30
	 *      (constrained). Buffers contain
	 *      `2^h` residues and h does not exceed `countr_zero(p - 1)` (asserted).
	 *      Residue ranges are documented on each operation; pointer validity
	 *      and nonaliasing for a two-buffer convolution are caller obligations.
	 * @complexity Transforms and products take O(h 2^h) time and O(1) auxiliary
	 *             space; per-modulus cached tables have O(log p) entries.
	 */
	struct scalar_ntt : ntt_engine_base<scalar_ntt> {
		/** @brief Whether Z's constant modulus is an odd prime below 2^30.
		 * @tparam Z Static modular integer type.
		 * @return Whether raw scalar transforms support Z.
		 */
		template<static_mod_int Z>
		static constexpr bool supports() {
			return detail::ntt_modulus<Z>;
		}

		/** @brief Per-modulus incremental twiddle ("step") tables, built once. */
		template<detail::ntt_modulus Z>
		struct twiddles {
			/** @brief The unsigned residue representation. */
			using U = decltype(Z::mod());
			/** @brief The transform modulus. */
			static constexpr U mod = Z::mod();
			static_assert(mod % 2 == 1, "NTT needs an odd (prime) modulus");
			U step[30] = {};		 ///< Forward increment factors in tight Montgomery form.
			U inverse_step[30] = {}; ///< Inverse increment factors in tight Montgomery form.

			/** @brief Builds increment factors from a primitive root. */
			constexpr twiddles() {
				using mont = montgomery<mod>;
				uint32 const levels = uint32(std::countr_zero(mod - 1));
				Z root[30], iroot[30];
				root[levels] = power(detail::primitive_root<Z>(), (mod - 1) >> levels);
				iroot[levels] = power(root[levels], mod - 2);
				for (uint32 i = levels; i-- > 0;) {
					root[i] = root[i + 1] * root[i + 1];
					iroot[i] = iroot[i + 1] * iroot[i + 1];
				}
				Z prod(1), iprod(1);
				for (uint32 i = 0; i + 2 <= levels; ++i) {
					step[i] = mont::to_form((root[i + 2] * prod).value());
					inverse_step[i] = mont::to_form((iroot[i + 2] * iprod).value());
					prod *= iroot[i + 2];
					iprod *= root[i + 2];
				}
			}
		};

		/** @brief Shared immutable per-modulus twiddle table.
		 * @note Not constexpr (returns a function-local static); constexpr
		 *       transforms build a local table instead.
		 */
		template<detail::ntt_modulus Z>
		static twiddles<Z> const &twiddle_table() {
			static twiddles<Z> const t;
			return t;
		}

		/**
		 * @brief Forward transform of `2^h` residues in place, natural order to
		 *        bit-reversed; inputs and outputs in `[0, 2·mod)`.
		 */
		template<detail::ntt_modulus Z>
		static constexpr void forward(decltype(Z::mod()) *f, uint32 h) {
			detail::check_ntt_levels<Z>(h);
			// Constant evaluation cannot read the cached function-local static.
			if consteval {
				forward_with<Z>(f, h, twiddles<Z>());
			} else {
				forward_with<Z>(f, h, twiddle_table<Z>());
			}
		}

		/**
		 * @brief Inverse transform of `2^h` residues in place, bit-reversed to
		 *        natural order, without the `1/n` scaling; inputs and outputs in
		 *        `[0, 2·mod)`.
		 */
		template<detail::ntt_modulus Z>
		static constexpr void inverse(decltype(Z::mod()) *f, uint32 h) {
			detail::check_ntt_levels<Z>(h);
			if consteval {
				inverse_with<Z>(f, h, twiddles<Z>());
			} else {
				inverse_with<Z>(f, h, twiddle_table<Z>());
			}
		}

	private:
		template<detail::ntt_modulus Z>
		static constexpr void forward_with(decltype(Z::mod()) *f, uint32 h, twiddles<Z> const &t) {
			using U = decltype(Z::mod());
			using mont = montgomery<Z::mod()>;
			constexpr U mod = Z::mod();
			for (uint32 len = 0; len < h; ++len) {
				std::size_t const p = std::size_t(1) << (h - len - 1);
				U rot = mont::one();
				U rotq = mont::companion(rot);
				for (std::size_t s = 0; s < (std::size_t(1) << len); ++s) {
					U *lo = f + (s << (h - len));
					U *hi = lo + p;
					for (std::size_t i = 0; i < p; ++i) {
						U const l = lo[i];
						U const r = mont::mul_lazy_fixed(hi[i], rot, rotq);
						// min on unsigned wrap = branchless conditional subtract
						lo[i] = (std::min)(l + r, l + r - 2 * mod);
						hi[i] = (std::min)(l - r, l + 2 * mod - r);
					}
					if (s + 1 != (std::size_t(1) << len)) {
						rot = mont::mul(rot, t.step[std::countr_zero(~s)]);
						rotq = mont::companion(rot);
					}
				}
			}
		}

		template<detail::ntt_modulus Z>
		static constexpr void inverse_with(decltype(Z::mod()) *f, uint32 h, twiddles<Z> const &t) {
			using U = decltype(Z::mod());
			using mont = montgomery<Z::mod()>;
			constexpr U mod = Z::mod();
			for (uint32 len = h; len-- > 0;) {
				std::size_t const p = std::size_t(1) << (h - len - 1);
				U rot = mont::one();
				U rotq = mont::companion(rot);
				for (std::size_t s = 0; s < (std::size_t(1) << len); ++s) {
					U *lo = f + (s << (h - len));
					U *hi = lo + p;
					for (std::size_t i = 0; i < p; ++i) {
						U const l = lo[i];
						U const r = hi[i];
						lo[i] = (std::min)(l + r, l + r - 2 * mod);
						hi[i] = mont::mul_lazy_fixed(l + 2 * mod - r, rot, rotq);
					}
					if (s + 1 != (std::size_t(1) << len)) {
						rot = mont::mul(rot, t.inverse_step[std::countr_zero(~s)]);
						rotq = mont::companion(rot);
					}
				}
			}
		}
	};

	namespace detail {
		/**
		 * @brief The engine actually used for modulus `Z`: the requested engine when
		 *        it `supports<Z>()`, else the portable scalar fallback (e.g. a
		 *        32-bit-only vector engine handed a 64-bit modulus). Every driver
		 *        entry resolves its engine through this one alias so the fallback
		 *        rule lives in one place.
		 */
		template<ntt_engine Engine, static_mod_int Z>
		using engine_or_scalar_t =
			std::conditional_t<Engine::template supports<Z>(), Engine, scalar_ntt>;
	}
}

namespace atl {
#if defined(ATL_NTT_ENGINE)
	/** @brief Configured transform engine; its header must already be included. */
	using default_ntt_engine = ATL_NTT_ENGINE;
#else
	/** @brief Portable scalar engine used by the convolution driver by default. */
	using default_ntt_engine = scalar_ntt;
#endif
}
