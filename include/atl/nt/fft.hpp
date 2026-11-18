#pragma once
#include <atl/utils/assert.hpp>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief Complex value with direct real/imaginary arithmetic for the FFT.
	 */
	struct cplx {
		double re = 0; ///< Real component.
		double im = 0; ///< Imaginary component.

		/** @brief Componentwise sum. */
		friend constexpr cplx operator+(cplx a, cplx b) { return {a.re + b.re, a.im + b.im}; }

		/** @brief Componentwise difference. */
		friend constexpr cplx operator-(cplx a, cplx b) { return {a.re - b.re, a.im - b.im}; }

		/** @brief Complex product without special handling for nonfinite values. */
		friend constexpr cplx operator*(cplx a, cplx b) {
			return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
		}

		/** @brief Complex conjugate. */
		constexpr cplx conj() const { return {re, -im}; }
	};

	/** @brief A writable sized random-access range of @ref atl::cplx, the transforms' currency. */
	template<typename R>
	concept cplx_range = std::ranges::random_access_range<R> && std::ranges::sized_range<R>
					  && std::ranges::output_range<R, cplx> && std::same_as<std::ranges::range_value_t<R>, cplx>;

	namespace detail {
		/**
		 * @brief Cached 2k-th roots in rt[k..2k), grown to the requested size.
		 * Anchoring every 64th root with trigonometry limits multiplication-chain
		 * phase error without the cost of trigonometry for every root.
		 * Callers serialize growth of this mutable cache.
		 */
		inline std::vector<cplx> const &fft_roots(std::size_t n) {
			static std::vector<cplx> rt{{1, 0}, {1, 0}};
			static std::size_t built = 2;
			if (built < n) {
				rt.resize(n);
				for (std::size_t k = built; k < n; k *= 2) {
					long double const base = std::acos(-1.0L) / double(k);
					cplx const step{double(std::cos((long double)base)),
									double(std::sin((long double)base))};
					for (std::size_t i = k; i < 2 * k; ++i) {
						if ((i - k) % 64 == 0) {
							long double const angle = base * double(i - k);
							rt[i] = {double(std::cos(angle)), double(std::sin(angle))};
						} else {
							rt[i] = rt[i - 1] * step;
						}
					}
				}
				built = n;
			}
			return rt;
		}
	}

	/**
	 * @brief In-place complex FFT of a power-of-two-sized range, natural
	 *        order in and out (bit-reversal inside). Approximate by nature:
	 *        error grows with size and magnitude — integer users go through
	 *        @ref atl::convolution / `convolution_uint64`, which manage the
	 *        precision budget; this is for the scenarios that genuinely
	 *        want doubles (signals, geometry sums, custom splits).
	 * @pre `std::ranges::size(a)` is a power of two (asserted).
	 * @param a Mutable complex samples; length n is nonzero.
	 * @complexity O(n log n) time, O(n) shared cached roots.
	 * @note Not constexpr (uses trigonometry and a shared root cache).
	 * @note Approximate arithmetic; integer samples have no general exact-rounding
	 *       guarantee. The balanced split in atl::convolution limits each transform
	 *       to 2^21 points and signed halves to magnitude 2^14. It requires
	 *       round-to-nearest arithmetic without fast-math. See the guide for the
	 *       tested integer envelope.
	 */
	template<cplx_range R>
	void fft(R &&a) {
		std::size_t const n = std::ranges::size(a);
		auto const first = std::ranges::begin(a);
		atl_assert(std::has_single_bit(n));
		auto const &rt = detail::fft_roots(n);
		for (std::size_t i = 1, j = 0; i < n; ++i) {
			std::size_t bit = n >> 1;
			for (; j & bit; bit >>= 1) j ^= bit;
			j ^= bit;
			if (i < j) {
				cplx const temp = first[i];
				first[i] = cplx(first[j]);
				first[j] = temp;
			}
		}
		for (std::size_t k = 1; k < n; k *= 2)
			for (std::size_t i = 0; i < n; i += 2 * k)
				for (std::size_t j = 0; j < k; ++j) {
					cplx const z = rt[j + k] * first[i + j + k];
					first[i + j + k] = first[i + j] - z;
					first[i + j] = first[i + j] + z;
				}
	}

	/**
	 * @brief The inverse of @ref atl::fft (conjugation trick + 1/n scaling).
	 * @pre `std::ranges::size(a)` is a power of two (asserted).
	 * @param a Mutable complex samples; length n is nonzero.
	 * @complexity O(n log n) time, O(n) shared cached roots.
	 * @note Not constexpr (uses trigonometry and a shared root cache).
	 * @note Approximate arithmetic; integer samples have no general exact-rounding
	 *       guarantee. The balanced split in atl::convolution limits each transform
	 *       to 2^21 points and signed halves to magnitude 2^14. It requires
	 *       round-to-nearest arithmetic without fast-math. See the guide for the
	 *       tested integer envelope.
	 */
	template<cplx_range R>
	void inverse_fft(R &&a) {
		std::size_t const n = std::ranges::size(a);
		auto const first = std::ranges::begin(a);
		for (std::size_t i = 0; i < n; ++i) {
			cplx const z = first[i];
			first[i] = z.conj();
		}
		fft(a);
		double const inv = 1.0 / double(n);
		for (std::size_t i = 0; i < n; ++i) {
			cplx const z = first[i];
			first[i] = {z.re * inv, -z.im * inv};
		}
	}
}
