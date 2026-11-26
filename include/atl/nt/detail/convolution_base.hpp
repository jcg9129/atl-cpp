#pragma once

/** @file
 * @brief Shared direct transform kernels and in-place transforms.
 */
#include <atl/nt/detail/ntt_engine_base.hpp>
#include <atl/nt/mod_int_static.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery_static.hpp>
#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
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

}
