#pragma once

/** @file
 * @brief The online_convolution API for prime moduli up to 32 bits with native double-width coefficient multiplication.
 */

#include <atl/nt/detail/online_convolution.hpp>
#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <concepts>
#include <cstddef>

namespace atl {
	/**
	 * @brief Fully relaxed Cauchy product of two streams: `push(a[k], b[k])` returns `sum(i + j = k) a[i] b[j]`.
	 *
	 * Each coefficient is returned as soon as both inputs of its index have arrived, so a pushed value may depend on
	 * earlier returns (online algorithms such as exponentials of series defined by recurrences).
	 * @tparam Coefficient Static prime-modulus integer type.
	 * @tparam Engine NTT engine used for the block products; engines that do not support the modulus fall back to the
	 *         convolution driver's tiers, and constant evaluation uses scalar transforms,
	 *         or schoolbook products where the field has none.
	 * @complexity For `n` pushes: O(log^2(n + 1)) amortized field work per push and O(n log(n + 1)) worst case per push
	 *             in fields with a transform; in general O(sum over dyadic `s <= n` of M(s) / s) amortized and O(M(n))
	 *             worst case. O(n) memory. Race pending.
	 * @note Dyadic rectangles with the transforms of the stationary blocks kept; blocks up to a provisional size of 32 are
	 *       multiplied directly.
	 */
	template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
	class online_convolution {
		/** @brief The dyadic schedule and its retained block spectra. */
		detail::online_product_state<Coefficient, Engine, detail::online_product_algorithm::cached, false> state_;

	public:
		/**
		 * @brief Append the next element of both streams and return the product coefficient of that index.
		 * @param left The next coefficient of the first stream.
		 * @param right The next coefficient of the second stream.
		 * @return `sum(i + j = k) a[i] b[j]` for `k = size() - 1` after the push.
		 * @pre `size() < maximum_series_length` (asserted).
		 */
		[[nodiscard]] constexpr Coefficient push(Coefficient left, Coefficient right) { return state_.push(left, right); }

		/**
		 * @brief Reserve storage so that pushes until `size() == count` do not reallocate the streams or the
		 *        pending outputs.
		 * @param count Total number of coefficients that will have been pushed, counting earlier pushes. Only
		 *        capacity changes: later results and `size()` are as without the call.
		 * @pre `count <= maximum_series_length` (asserted).
		 * @throws std::bad_alloc (or what the allocator throws) if the storage cannot grow; the logical state is
		 *         then unchanged, though some buffers may already have grown.
		 * @note Not covered: the retained block spectra (one pair of length `2s` per dyadic scale `s`, allocated
		 *       when that scale is first used) and the temporaries of each block product.
		 */
		constexpr void reserve(std::size_t count) { state_.reserve(count); }

		/** @brief Number of pushes so far. */
		[[nodiscard]] constexpr std::size_t size() const { return state_.size(); }
	};

	/**
	 * @brief Semi-relaxed Cauchy product of a stream with a whole, fixed kernel: `push(a[k])` returns `sum(i + j = k) a[i] kernel[j]`.
	 *
	 * Kernel coefficients beyond its length are zero, also after the stream outgrows it. The kernel is copied and the
	 * input range is not modified; the stationary kernel blocks are transformed once, in the constructor, where the field
	 * has a transform.
	 * @tparam Coefficient Static prime-modulus integer type.
	 * @tparam Engine As for `online_convolution`.
	 * @complexity For `n` pushes: as `online_convolution`; O(n + kernel length) memory. Construction costs
	 *             O(M(kernel length)) field work. Race pending.
	 */
	template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
	class online_fixed_convolution {
		/** @brief The fixed kernel, stream, and dyadic caches. */
		detail::online_product_state<Coefficient, Engine, detail::online_product_algorithm::cached, true> state_;

	public:
		/**
		 * @brief Copy a kernel; trailing zeros are ignored.
		 * @tparam Range Input range of `Coefficient`; a single pass through a const reference suffices.
		 * @param kernel Coefficients in increasing degree order.
		 * @pre The normalized kernel length is at most `maximum_series_length` (asserted).
		 */
		template<prime_mod_int_range Range>
			requires std::same_as<series_coefficient_t<Range>, Coefficient>
		constexpr explicit online_fixed_convolution(Range const &kernel) : state_(kernel) {}

		/**
		 * @brief Append the next stream element and return its convolution coefficient with the kernel.
		 * @param value The next coefficient of the stream.
		 * @return `sum(i + j = k) a[i] kernel[j]` for `k = size() - 1` after the push.
		 * @pre `size() < maximum_series_length` (asserted).
		 */
		[[nodiscard]] constexpr Coefficient push(Coefficient value) { return state_.push(value); }

		/**
		 * @brief Reserve storage so that pushes until `size() == count` do not reallocate the stream, the zero padding of the kernel or the
		 *        pending outputs.
		 * @param count Total number of coefficients that will have been pushed, counting earlier pushes. Only
		 *        capacity changes: later results and `size()` are as without the call.
		 * @pre `count <= maximum_series_length` (asserted).
		 * @throws std::bad_alloc (or what the allocator throws) if the storage cannot grow; the logical state is
		 *         then unchanged, though some buffers may already have grown.
		 * @note Not covered: the retained block spectra (one pair of length `2s` per dyadic scale `s`, allocated
		 *       when that scale is first used; the kernel's own spectra are built by the constructor) and the temporaries of each block product.
		 */
		constexpr void reserve(std::size_t count) { state_.reserve(count); }

		/** @brief Number of pushes so far. */
		[[nodiscard]] constexpr std::size_t size() const { return state_.size(); }
	};
}
