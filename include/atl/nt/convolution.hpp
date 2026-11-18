#pragma once
#include <atl/nt/basic.hpp>
#include <atl/nt/fft.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief What the convolution entries accept: a sized random-access range of
	 *        static-modulus modular integers.
	 */
	template<typename R>
	concept mod_int_range = std::ranges::random_access_range<R> && std::ranges::sized_range<R>
						 && static_mod_int<std::ranges::range_value_t<R>>;

	/**
	 * @brief Integral ranges for the exact-integer entry.
	 */
	template<typename R>
	concept integral_range = std::ranges::random_access_range<R> && std::ranges::sized_range<R>
						  && integral<std::ranges::range_value_t<R>>;

	namespace detail {
		/**
		 * @brief Whether `a` and `b` are provably the same sequence — the same
		 *        contiguous storage with the same length, or the same object —
		 *        so a convolution of the two is a square. Equal contents in
		 *        different storage are not detected (the check stays O(1)).
		 */
		template<std::ranges::sized_range A, std::ranges::sized_range B>
		constexpr bool is_same_sequence(A const &a, B const &b) {
			if constexpr (std::ranges::contiguous_range<A const> && std::ranges::contiguous_range<B const>
						  && std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>) {
				return std::ranges::size(a) == std::ranges::size(b)
					&& std::ranges::data(a) == std::ranges::data(b);
			} else if constexpr (std::same_as<A, B>) {
				return std::addressof(a) == std::addressof(b);
			} else {
				return false;
			}
		}

		/**
		 * @brief `Engine::square` when the engine provides it, else a copy and
		 *        `Engine::convolve` (the @ref atl::ntt_engine concept does not require
		 *        `square`).
		 */
		template<ntt_engine Engine, ntt_modulus Z>
		constexpr void engine_square(std::vector<decltype(Z::mod())> &f, uint32 h) {
			if constexpr (requires { Engine::template square<Z>(f.data(), h); }) {
				Engine::template square<Z>(f.data(), h);
			} else {
				std::vector<decltype(Z::mod())> g(f);
				Engine::template convolve<Z>(f.data(), g.data(), h);
			}
		}

		/**
		 * @brief An indexed accessor `get(i)` yielding raw residues convertible to
		 *        `Z`'s unsigned representation — how the convolution kernels read
		 *        their inputs without materialising them.
		 */
		template<typename Get, typename Z>
		concept residue_accessor =
			static_mod_int<Z> && std::invocable<Get const &, std::size_t>
			&& std::convertible_to<std::invoke_result_t<Get const &, std::size_t>,
								   decltype(Z::mod())>;

		/**
		 * @brief The transform size `raw_convolution` runs for inputs of `n`
		 *        and `m` entries: `bit_ceil(2·min(n, m))` when that is at most a
		 *        quarter of the padded `bit_ceil(n + m - 1)` (overlap-add), the
		 *        padded size otherwise. A modulus serves the direct tier iff its
		 *        2-adic headroom covers this size.
		 */
		constexpr std::size_t planned_ntt_size(std::size_t n, std::size_t m) {
			atl_assert(n != 0 && m != 0);
			constexpr std::size_t limit = std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 2);
			atl_assert(n <= limit && m <= limit);
			std::size_t const padded = std::bit_ceil(n + m - 1);
			std::size_t const short_size = std::bit_ceil(2 * (std::min)(n, m));
			return short_size <= padded / 4 ? short_size : padded;
		}

		/**
		 * @brief One friendly-prime convolution of raw residue arrays (planned
		 *        sizes; result truncated to `need`), the tuned composition: naive
		 *        under the measured cutoff, overlap-add blocking for lopsided
		 *        inputs, one padded transform otherwise. With `is_square` the two
		 *        inputs are the same sequence and one forward transform serves
		 *        both.
		 */
		template<ntt_engine Engine, ntt_modulus Z, residue_accessor<Z> GetA,
				 residue_accessor<Z> GetB>
		constexpr std::vector<decltype(Z::mod())> raw_convolution(std::size_t n, GetA const &at_a,
																  std::size_t m, GetB const &at_b,
																  bool is_square = false) {
			using U = decltype(Z::mod());
			constexpr U mod = Z::mod();
			using mont = montgomery<mod>;
			using Eng = detail::engine_or_scalar_t<Engine, Z>;
			std::size_t const size = planned_ntt_size(n, m);
			std::size_t const need = n + m - 1;

			std::size_t const shortest = (std::min)(n, m);
			std::size_t const short_size = std::bit_ceil(2 * shortest);
			bool const is_lopsided = size != std::bit_ceil(need);
			atl_assert((mod - 1) % size == 0);
			uint32 const h = uint32(std::countr_zero(size));

			auto const load_short = [&](std::vector<U> &f) {
				for (std::size_t i = 0; i < shortest; ++i) f[i] = n <= m ? at_a(i) : at_b(i);
			};
			auto const at_long = [&](std::size_t i) { return n <= m ? at_b(i) : at_a(i); };
			std::size_t const longest = (std::max)(n, m);

			if (!is_lopsided) {
				std::vector<U> fa(size, 0);
				for (std::size_t i = 0; i < longest; ++i) fa[i] = at_long(i);
				if (is_square) {
					engine_square<Eng, Z>(fa, h);
				} else {
					std::vector<U> fb(size, 0);
					load_short(fb);
					Eng::template convolve<Z>(fa.data(), fb.data(), h);
				}
				fa.resize(need);
				return fa;
			}
			// overlap-add: chunks of (short_size - shortest + 1) long-side entries
			// against one short-side transform, each landing exactly short_size wide.
			// Raw transforms only (the fused convolve is for the balanced case)
			U const rr = mont::to_form(mont::to_form(U(1)));
			// 2^h · (mod - (mod - 1) / 2^h) = 1 (mod mod): the inverse in closed form
			U const scale = mont::mul(mont::to_form(U(mod - ((mod - 1) >> h))), rr);
			std::size_t const step = short_size - shortest + 1;
			std::vector<U> fs(short_size, 0), chunk(short_size), acc(need + short_size, 0);
			load_short(fs);
			Eng::template forward<Z>(fs.data(), h);
			for (std::size_t at = 0; at < longest; at += step) {
				std::size_t const len = (std::min)(step, longest - at);
				for (std::size_t i = 0; i < len; ++i) chunk[i] = at_long(at + i);
				std::ranges::fill(chunk.begin() + len, chunk.end(), U(0));
				Eng::template forward<Z>(chunk.data(), h);
				for (std::size_t i = 0; i < short_size; ++i) chunk[i] = mont::mul_lazy(chunk[i], fs[i]);
				Eng::template inverse<Z>(chunk.data(), h);
				for (std::size_t i = 0; i < short_size; ++i) {
					U x = mont::mul_lazy(chunk[i], scale);
					if (x >= mod) x -= mod;
					U const sum = acc[at + i] + x;
					acc[at + i] = sum >= mod ? sum - mod : sum;
				}
			}
			acc.resize(need);
			return acc;
		}

		/**
		 * @brief The three NTT primes behind the CRT paths (all with primitive root
		 *        3; product ~ 7.8·10^25). Their 2-adic headroom is 2^25, 2^26 and
		 *        2^23 (998244353 = 119·2^23 + 1), so a CRT transform is capped at
		 *        2^23 points.
		 */
		inline constexpr uint32 crt_primes[3] = {167772161, 469762049, 998244353};

		/**
		 * @brief Whether every true coefficient of `a * b` stays below the product
		 *        of `crt_primes`, determined from the inputs' maxima: the
		 *        precondition of the Garner tier.
		 */
		template<mod_int_range A, mod_int_range B>
		bool garner_fits(A const &a, B const &b) {
			using ZA = std::ranges::range_value_t<A>;
			using ZB = std::ranges::range_value_t<B>;
			uint64 top_a = 0, top_b = 0;
			for (auto const &x : a) top_a = (std::max)(top_a, uint64(ZA(x).value()));
			for (auto const &x : b) top_b = (std::max)(top_b, uint64(ZB(x).value()));
			uint128 const bound = uint128(crt_primes[0]) * crt_primes[1] * crt_primes[2];
			return top_a == 0 || top_b == 0
				|| uint128(top_a) <= (bound - 1) / (std::min)(std::ranges::size(a), std::ranges::size(b)) / top_b;
		}

		/**
		 * @brief Balanced 15-bit FFT split with overlap-add for lopsided inputs.
		 * @pre Nonzero lengths, modulus <= 2^30, planned size <= 2^21 (asserted).
		 * Centered residues split into signed halves bounded by 2^14. Anchored
		 * roots limit accumulated phase error; integer recovery rounds each of
		 * the ll, lh, hl and hh products separately.
		 */
		template<static_mod_int Z, residue_accessor<Z> GetA, residue_accessor<Z> GetB>
			requires(Z::mod() <= (uint64(1) << 30))
		std::vector<Z> fft_split_convolution(std::size_t n, GetA const &at_a, std::size_t m,
											 GetB const &at_b, bool is_square = false) {
			using U = decltype(Z::mod());
			atl_assert(n != 0 && m != 0 && Z::mod() <= (uint64(1) << 30));
			atl_assert(planned_ntt_size(n, m) <= (std::size_t(1) << 21));
			std::size_t const need = n + m - 1;
			std::size_t const size = planned_ntt_size(n, m);
			std::size_t const shortest = (std::min)(n, m), longest = (std::max)(n, m);
			std::size_t const step = size - shortest + 1;
			auto const at_short = [&](std::size_t i) { return n <= m ? at_a(i) : at_b(i); };
			auto const at_long = [&](std::size_t i) { return n <= m ? at_b(i) : at_a(i); };
			auto const split = [](U v) {
				int64 const sv = v > Z::mod() / 2 ? int64(v) - int64(Z::mod()) : int64(v);
				int64 const hi = (sv + (int64(1) << 14)) >> 15;
				return cplx{double(sv - hi * (int64(1) << 15)), double(hi)};
			};
			std::vector<cplx> in1(size), in2;
			if (!is_square) {
				in2.resize(size);
				for (std::size_t i = 0; i < shortest; ++i) in2[i] = split(at_short(i));
				fft(in2);
			}
			std::vector<cplx> const &rhs = is_square ? in1 : in2;
			std::vector<cplx> out1(size), out2(size);
			double const inv = 1.0 / double(size);
			uint64 const mod = Z::mod();
			auto const norm = [&](int64 x) { return uint64(((x % int64(mod)) + int64(mod)) % int64(mod)); };
			std::vector<uint64> acc(need + size, 0);
			for (std::size_t at = 0; at < longest; at += step) {
				std::size_t const len = (std::min)(step, longest - at);
				for (std::size_t i = 0; i < len; ++i) in1[i] = split(at_long(at + i));
				std::ranges::fill(in1.begin() + std::ptrdiff_t(len), in1.end(), cplx{0, 0});
				fft(in1);
				for (std::size_t j = 0; j < size; ++j) {
					std::size_t const k = (size - j) & (size - 1);
					cplx const pa = (in1[j] + in1[k].conj()) * cplx{0.5, 0};
					cplx const qa = (in1[j] - in1[k].conj()) * cplx{0, -0.5};
					out1[j] = (pa * rhs[j]).conj();
					out2[j] = (qa * rhs[j]).conj();
				}
				fft(out1);
				fft(out2);
				for (std::size_t i = 0, valid = (std::min)(size, len + shortest - 1); i < valid; ++i) {
					int64 const ll = std::llround(out1[i].re * inv);
					int64 const lh_hl = std::llround(-out1[i].im * inv) + std::llround(out2[i].re * inv);
					int64 const hh = std::llround(-out2[i].im * inv);
					uint64 const sum = norm(ll) + (norm(lh_hl) << 15) % mod + (norm(hh) << 30) % mod;
					acc[at + i] = (acc[at + i] + sum) % mod;
				}
			}
			std::vector<Z> c(need);
			for (std::size_t i = 0; i < need; ++i) c[i] = Z(acc[i]);
			return c;
		}
	}

	/**
	 * @brief In-place number-theoretic transform of a power-of-two-sized
	 *        range of `mod_int`s, in the bit-reversed convention: @ref atl::ntt
	 *        takes natural order to bit-reversed, @ref atl::inverse_ntt takes
	 *        bit-reversed back to natural order (scaling by 1/n included).
	 *        Pointwise spectrum products correspond to cyclic convolution.
	 * @tparam Engine Raw NTT engine; constant evaluation uses atl::scalar_ntt.
	 * @param a Writable sized random-access range of static modular integers.
	 * @complexity O(n log n) time and O(n) auxiliary storage.
	 * @pre `size` is a power of two dividing `Z::mod() - 1` (asserted). The
	 *      modulus must be an odd PRIME of at most 30 bits — a constraint,
	 *      so a modulus outside that range or a composite is rejected.
	 */
	template<ntt_engine Engine = default_ntt_engine, mod_int_range R>
		requires detail::ntt_modulus<std::ranges::range_value_t<R>>
			  && std::ranges::output_range<R, std::ranges::range_value_t<R>>
	constexpr void ntt(R &&a) {
		using Z = std::ranges::range_value_t<R>;
		using U = decltype(Z::mod());
		using Eng = detail::engine_or_scalar_t<Engine, Z>;
		std::size_t const n = std::ranges::size(a);
		auto const first = std::ranges::begin(a);
		atl_assert(std::has_single_bit(n) && (Z::mod() - 1) % n == 0);
		std::vector<U> raw(n);
		for (std::size_t i = 0; i < n; ++i) raw[i] = Z(first[i]).value();
		uint32 const h = uint32(std::countr_zero(n));
		if consteval {
			scalar_ntt::forward<Z>(raw.data(), h);
		} else {
			Eng::template forward<Z>(raw.data(), h);
		}
		for (std::size_t i = 0; i < n; ++i) first[i] = Z(raw[i] >= Z::mod() ? raw[i] - Z::mod() : raw[i]);
	}

	/**
	 * @brief The inverse of @ref atl::ntt (bit-reversed spectrum in, natural order and
	 *        1/n scaling out).
	 * @tparam Engine Raw NTT engine; constant evaluation uses atl::scalar_ntt.
	 * @param a Writable sized random-access range of static modular integers.
	 * @pre Nonzero power-of-two length dividing `Z::mod() - 1` (asserted).
	 * @complexity O(n log n) time and O(n) auxiliary storage.
	 */
	template<ntt_engine Engine = default_ntt_engine, mod_int_range R>
		requires detail::ntt_modulus<std::ranges::range_value_t<R>>
			  && std::ranges::output_range<R, std::ranges::range_value_t<R>>
	constexpr void inverse_ntt(R &&a) {
		using Z = std::ranges::range_value_t<R>;
		using U = decltype(Z::mod());
		using mont = montgomery<Z::mod()>;
		using Eng = detail::engine_or_scalar_t<Engine, Z>;
		std::size_t const n = std::ranges::size(a);
		auto const first = std::ranges::begin(a);
		atl_assert(std::has_single_bit(n) && (Z::mod() - 1) % n == 0);
		std::vector<U> raw(n);
		for (std::size_t i = 0; i < n; ++i) raw[i] = Z(first[i]).value();
		uint32 const h = uint32(std::countr_zero(n));
		if consteval {
			scalar_ntt::inverse<Z>(raw.data(), h);
		} else {
			Eng::template inverse<Z>(raw.data(), h);
		}
		U const scale = mont::to_form(power(Z(U(n)), Z::mod() - 2).value());
		for (std::size_t i = 0; i < n; ++i) {
			U x = mont::mul_lazy(raw[i], scale);
			if (x >= Z::mod()) x -= Z::mod();
			first[i] = Z(x);
		}
	}

	/**
	 * @brief Polynomial product over a static modulus: `c[k] = sum a[i] b[k-i]`.
	 * @tparam Engine Raw NTT engine, with scalar fallback for unsupported moduli.
	 * @param a First sized random-access range of static modular integers.
	 * @param b Second range, with the same value type as a.
	 * @return Empty if either input is empty; otherwise n+m-1 coefficients.
	 * @pre Nonempty input lengths are at most `2^(size_t digits - 2)` (asserted).
	 *      Direct NTTs require a prime below 2^30 and sufficient 2-adic order.
	 *      The balanced FFT split uses modulus <= 2^30 and planned size <= 2^21;
	 *      round-to-nearest arithmetic without fast-math is required.
	 *      Garner requires the planned size <= 2^23 and
	 *      `min(n,m) max(a) max(b) < 167772161*469762049*998244353` (asserted).
	 * @complexity O((n+m) log(n+m)) time, O(n+m) storage. Constant evaluation
	 *             uses O(nm) for moduli without a suitable direct transform.
	 * @note Identical storage uses one forward transform per prime or FFT split.
	 *       Constant evaluation uses scalar transforms or naive multiplication.
	 */
	template<ntt_engine Engine = default_ntt_engine, mod_int_range A, mod_int_range B>
		requires std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>
			  && mod_int_range<A const> && mod_int_range<B const>
	constexpr auto convolution(A const &a, B const &b) {
		using Z = std::ranges::range_value_t<A>;
		std::size_t const n = std::ranges::size(a), m = std::ranges::size(b);
		auto const first_a = std::ranges::begin(a);
		auto const first_b = std::ranges::begin(b);
		if (n == 0 || m == 0) return std::vector<Z>{};
		constexpr std::size_t limit = std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 2);
		atl_assert(n <= limit && m <= limit);
		std::size_t const need = n + m - 1;
		auto const naive = [&] {
			std::vector<Z> c(need);
			for (std::size_t i = 0; i < n; ++i)
				for (std::size_t j = 0; j < m; ++j) c[i + j] += Z(first_a[i]) * Z(first_b[j]);
			return c;
		};

		if ((std::min)(n, m) <= 48) return naive(); // measured crossover
		bool const is_square = detail::is_same_sequence(a, b);
		using UM = decltype(Z::mod());
		if constexpr (detail::ntt_modulus<Z>) {
			// Lazy butterflies require 4p <= R; primality supplies the roots and
			// inverse length. Do not instantiate this tier for other moduli.
			if ((Z::mod() - 1) % detail::planned_ntt_size(n, m) == 0) {
				auto const at_a = [&](std::size_t i) { return Z(first_a[i]).value(); };
				auto const at_b = [&](std::size_t i) { return Z(first_b[i]).value(); };
				std::vector<UM> raw;
				if consteval {
					raw = detail::raw_convolution<scalar_ntt, Z>(n, at_a, m, at_b, is_square);
				} else {
					raw = detail::raw_convolution<Engine, Z>(n, at_a, m, at_b, is_square);
				}
				std::vector<Z> c(need);
				for (std::size_t i = 0; i < need; ++i) c[i] = Z(raw[i]);
				return c;
			}
		}
		if consteval {
			return naive();
		} else {
			if constexpr (Z::mod() <= (decltype(Z::mod())(1) << 30)) {
				if (detail::planned_ntt_size(n, m) <= (std::size_t(1) << 21))
					return detail::fft_split_convolution<Z>(
						n, [&](std::size_t i) { return Z(first_a[i]).value(); }, m,
						[&](std::size_t i) { return Z(first_b[i]).value(); }, is_square);
			}
			atl_assert(detail::garner_fits(a, b));
			std::vector<uint32> r[3];
			[&]<std::size_t... K>(std::index_sequence<K...>) {
				((r[K] = detail::raw_convolution<Engine, mod_int<uint32, detail::crt_primes[K]>>(
					  n, [&](std::size_t i) { return uint32(Z(first_a[i]).value() % detail::crt_primes[K]); },
					  m, [&](std::size_t i) { return uint32(Z(first_b[i]).value() % detail::crt_primes[K]); },
					  is_square)),
				 ...);
			}(std::make_index_sequence<3>{});
			constexpr uint64 p0 = detail::crt_primes[0], p1 = detail::crt_primes[1],
							 p2 = detail::crt_primes[2];
			// Fixed 32-bit residues keep inverse products within uint64.
			constexpr uint64 inv_p0_p1 =
				power(mod_int<uint32, detail::crt_primes[1]>(uint32(p0 % p1)), p1 - 2).value();
			constexpr uint64 inv_p01_p2 =
				power(mod_int<uint32, detail::crt_primes[2]>(uint32(p0 * p1 % p2)), p2 - 2).value();
			Z const zp0(p0), zp01(p0 * p1 % uint64(Z::mod()));
			std::vector<Z> c(need);
			for (std::size_t i = 0; i < need; ++i) {
				uint64 const v0 = r[0][i];
				uint64 const v1 = (r[1][i] + p1 - v0 % p1) * inv_p0_p1 % p1;
				uint64 const t01 = (v0 + p0 * v1) % p2;
				uint64 const v2 = (r[2][i] + p2 - t01) * inv_p01_p2 % p2;
				c[i] = Z(v0) + zp0 * Z(v1) + zp01 * Z(v2);
			}
			return c;
		}
	}

	/**
	 * @brief Exact nonnegative integer polynomial product as uint64 coefficients.
	 * @tparam Engine Raw NTT engine; constant evaluation uses atl::scalar_ntt.
	 * @param a First sized random-access integral range.
	 * @param b Second sized random-access integral range.
	 * @return Empty if either input is empty; otherwise n+m-1 coefficients.
	 * @pre Inputs are nonnegative and fit uint64, input lengths are at most
	 *      `2^(size_t digits - 2)`, and the sufficient coefficient bound
	 *      `B = min(n,m) max(a) max(b) <= UINT64_MAX` holds (asserted).
	 *      Above the naive cutoff the planned
	 *      transform size is at most 2^23 (asserted).
	 * @complexity O((n+m) log(n+m)) time and O(n+m) auxiliary storage.
	 * @note Uses two-prime CRT when `B < 469762049*998244353`, otherwise
	 *       three-prime Garner with 167772161 as the third prime. Selection
	 *       uses an exact bound in every build, including without ATL_ASSERT.
	 *       Reconstruction uses portable atl::uint128.
	 */
	template<ntt_engine Engine = default_ntt_engine, integral_range A, integral_range B>
		requires integral_range<A const> && integral_range<B const>
	constexpr std::vector<uint64> convolution_uint64(A const &a, B const &b) {
		std::size_t const n = std::ranges::size(a), m = std::ranges::size(b);
		auto const first_a = std::ranges::begin(a);
		auto const first_b = std::ranges::begin(b);
		if (n == 0 || m == 0) return {};
		constexpr std::size_t limit = std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 2);
		atl_assert(n <= limit && m <= limit);
		std::size_t const need = n + m - 1;
		constexpr uint64 p1 = detail::crt_primes[1], p2 = detail::crt_primes[2];
		bool has_two_prime_bound = false;
		{
			uint128 top_a = 0, top_b = 0;
			for (std::size_t i = 0; i < n; ++i) {
				using Value = std::ranges::range_value_t<A>;
				Value const value = Value(first_a[i]);
				if constexpr (signed_integral<Value>) atl_assert(value >= 0);
				top_a = (std::max)(top_a, uint128(value));
			}
			for (std::size_t i = 0; i < m; ++i) {
				using Value = std::ranges::range_value_t<B>;
				Value const value = Value(first_b[i]);
				if constexpr (signed_integral<Value>) atl_assert(value >= 0);
				top_b = (std::max)(top_b, uint128(value));
			}
			atl_assert(top_a <= (std::numeric_limits<uint64>::max)() && top_b <= (std::numeric_limits<uint64>::max)());
			uint128 const product = top_a * top_b;
			uint128 const lower = uint128(uint64(product)) * (std::min)(n, m);
			// B = upper * 2^64 + uint64(lower). Each partial product fits uint128;
			// upper < 2^128 even for 64-bit size_t and maximal uint64 inputs.
			uint128 const upper = (product >> 64) * (std::min)(n, m) + (lower >> 64);
			has_two_prime_bound = upper == 0 && uint64(lower) < p1 * p2;
			atl_assert(upper == 0);
		}
		if ((std::min)(n, m) <= 48) {
			std::vector<uint64> c(need, 0);
			for (std::size_t i = 0; i < n; ++i)
				for (std::size_t j = 0; j < m; ++j) c[i + j] += uint64(first_a[i]) * uint64(first_b[j]);
			return c;
		}
		bool const is_square = detail::is_same_sequence(a, b);
		auto const run = [&]<uint32 P>(std::integral_constant<uint32, P>) {
			auto const at_a = [&](std::size_t i) { return uint32(uint64(first_a[i]) % P); };
			auto const at_b = [&](std::size_t i) { return uint32(uint64(first_b[i]) % P); };
			if consteval {
				return detail::raw_convolution<scalar_ntt, mod_int<uint32, P>>(n, at_a, m, at_b, is_square);
			} else {
				return detail::raw_convolution<Engine, mod_int<uint32, P>>(n, at_a, m, at_b, is_square);
			}
		};
		auto const r1 = run(std::integral_constant<uint32, uint32(p1)>{});
		auto const r2 = run(std::integral_constant<uint32, uint32(p2)>{});
		constexpr uint64 inv_p1_p2 =
			power(mod_int<uint32, detail::crt_primes[2]>(uint32(p1 % p2)), p2 - 2).value();
		std::vector<uint64> c(need);
		if (has_two_prime_bound) {
			for (std::size_t i = 0; i < need; ++i) {
				uint64 const v1 = r1[i];
				uint64 const v2 = (r2[i] + p2 - v1 % p2) * inv_p1_p2 % p2;
				c[i] = v1 + p1 * v2;
			}
		} else {
			constexpr uint64 p3 = detail::crt_primes[0];
			auto const r3 = run(std::integral_constant<uint32, uint32(p3)>{});
			constexpr uint64 inv_p12_p3 =
				power(mod_int<uint32, detail::crt_primes[0]>(uint32(p1 * p2 % p3)), p3 - 2).value();
			for (std::size_t i = 0; i < need; ++i) {
				uint64 const v1 = r1[i];
				uint64 const v2 = (r2[i] + p2 - v1 % p2) * inv_p1_p2 % p2;
				uint64 const t12 = v1 + p1 * v2;
				uint64 const v3 = (r3[i] + p3 - t12 % p3) * inv_p12_p3 % p3;
				// t12 < p1*p2 < 2^59; v3 < p3 < 2^28. Widening first
				// keeps the reconstruction below p1*p2*p3 < 2^87.
				c[i] = uint64(uint128(t12) + uint128(p1 * p2) * v3);
			}
		}
		return c;
	}
}
