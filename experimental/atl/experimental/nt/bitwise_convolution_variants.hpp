#pragma once
#include <atl/nt/bitwise_convolution.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <vector>

/**
 * @file
 * @brief Contenders and oracles for atl/nt/bitwise_convolution.hpp.
 *
 * Transform engines share the library engine's contract,
 * `transform(first, bits, butterfly)`: apply the 2x2 butterfly along every
 * bit of the 2^bits elements at `first`. They differ only in loop order.
 *
 * Blocked radix-2 has the lowest total across the four n = 20 shapes in
 * both flag sets. Subset product by rank major loses at 1110 / 1111 ms
 * versus the library's 770 / 733 ms (plain -O2 / -march=native).
 * Source: benchmarks/nt/bitwise_convolution.cpp, Xeon 8358, GCC 15.2,
 * RelWithDebInfo, assertions off, 2026-10-15.
 *
 * The by-definition functions are the certifying oracles of the family: the
 * definitions summed directly, O(4^n) or O(3^n).
 */
namespace atl::experimental {
	/**
	 * @brief The textbook order: one full sweep per bit, radix 2.
	 *
	 * LOST narrowly: see the file comment.
	 * @complexity O(n 2^n), n sweeps over the array
	 */
	template<std::random_access_iterator It, atl::detail::butterfly_of<std::iter_value_t<It>> Butterfly>
		requires std::invocable<Butterfly const &, std::iter_reference_t<It>, std::iter_reference_t<It>>
	constexpr void transform_by_bit_outer_radix2(It f, std::size_t bits, Butterfly const &butterfly) {
		atl::detail::radix2_passes(f, std::size_t(1) << bits, 0, bits, butterfly);
	}

	/**
	 * @brief Two bits per sweep (a 4-point step) from the lowest bit up, no
	 *        cache blocking.
	 *
	 * LOST: see the file comment (best only at n = 16).
	 * @complexity O(n 2^n), n/2 sweeps
	 */
	template<std::random_access_iterator It, atl::detail::butterfly_of<std::iter_value_t<It>> Butterfly>
		requires std::invocable<Butterfly const &, std::iter_reference_t<It>, std::iter_reference_t<It>>
	constexpr void transform_by_radix4(It f, std::size_t bits, Butterfly const &butterfly) {
		std::size_t const size = std::size_t(1) << bits;
		std::size_t bit = 0;
		if (bits % 2 == 1) atl::detail::radix2_passes(f, size, 0, 1, butterfly), bit = 1;
		for (; bit < bits; bit += 2) {
			std::size_t const quarter = std::size_t(1) << bit;
			for (std::size_t block = 0; block < size; block += 4 * quarter)
				for (std::size_t i = block; i < block + quarter; ++i) {
					auto &&a0 = f[i];
					auto &&a1 = f[i + quarter];
					auto &&a2 = f[i + 2 * quarter];
					auto &&a3 = f[i + 3 * quarter];
					butterfly(a0, a1);
					butterfly(a2, a3);
					butterfly(a0, a2);
					butterfly(a1, a3);
				}
		}
	}

	/**
	 * @brief The low bits finished inside each L1-sized block (the library
	 *        block size), then two bits per sweep (a 4-point step) for the
	 *        rest, halving the passes over memory.
	 *
	 * LOST: see the file comment.
	 * @complexity O(n 2^n), 1 + (n - 11) / 2 sweeps
	 */
	template<std::random_access_iterator It, atl::detail::butterfly_of<std::iter_value_t<It>> Butterfly>
		requires std::invocable<Butterfly const &, std::iter_reference_t<It>, std::iter_reference_t<It>>
	constexpr void transform_by_blocked_radix4(It f, std::size_t bits, Butterfly const &butterfly) {
		std::size_t const low = (std::min)(bits, atl::detail::transform_block_bits);
		std::size_t const size = std::size_t(1) << bits;
		for (std::size_t block = 0; block < size; block += std::size_t(1) << low)
			atl::detail::radix2_passes(f + std::ptrdiff_t(block), std::size_t(1) << low, 0, low, butterfly);
		std::size_t bit = low;
		if ((bits - low) % 2 == 1) atl::detail::radix2_passes(f, size, bit, bit + 1, butterfly), ++bit;
		for (; bit < bits; bit += 2) {
			std::size_t const quarter = std::size_t(1) << bit;
			for (std::size_t block = 0; block < size; block += 4 * quarter)
				for (std::size_t i = block; i < block + quarter; ++i) {
					auto &&a0 = f[i];
					auto &&a1 = f[i + quarter];
					auto &&a2 = f[i + 2 * quarter];
					auto &&a3 = f[i + 3 * quarter];
					butterfly(a0, a1);
					butterfly(a2, a3);
					butterfly(a0, a2);
					butterfly(a1, a3);
				}
		}
	}

	/**
	 * @brief Subset convolution with the ranked arrays laid out rank-major
	 *        (n + 1 arrays of 2^n, each transformed by the library engine),
	 *        full rank ranges everywhere: the textbook layout.
	 *
	 * LOST to the library mask-major layout with popcount-restricted rank
	 * ranges (same record, mod_int 998244353, n = 20): 1110 / 1111 ms vs
	 * 770 / 733 ms, plain -O2 / -march=native. The product strides 2^n
	 * between ranks, and every phase touches all n + 1 ranks.
	 * @complexity O(n^2 2^n)
	 */
	template<std::ranges::random_access_range A, atl::ring_range<std::ranges::range_value_t<A>> B>
		requires atl::ring_range<A, std::ranges::range_value_t<A>> && atl::commutative_ring<std::ranges::range_value_t<A>>
	constexpr std::vector<std::ranges::range_value_t<A>> subset_convolution_by_rank_major(A const &a, B const &b) {
		using T = std::ranges::range_value_t<A>;
		auto [x, y] = atl::detail::padded_pair<T>(a, b);
		if (x.empty()) return x;
		std::size_t const size = x.size();
		std::size_t const bits = std::size_t(std::countr_zero(size));
		std::vector<std::vector<T>> ra(bits + 1, std::vector<T>(size)), rb(bits + 1, std::vector<T>(size));
		for (std::size_t s = 0; s < size; ++s) {
			ra[std::size_t(std::popcount(s))][s] = x[s];
			rb[std::size_t(std::popcount(s))][s] = y[s];
		}
		for (std::size_t k = 0; k <= bits; ++k) {
			atl::detail::butterfly_transform(ra[k].begin(), bits, atl::detail::subset_zeta_butterfly{});
			atl::detail::butterfly_transform(rb[k].begin(), bits, atl::detail::subset_zeta_butterfly{});
		}
		std::vector<std::vector<T>> rh(bits + 1, std::vector<T>(size));
		for (std::size_t k = 0; k <= bits; ++k)
			for (std::size_t i = 0; i <= k; ++i)
				for (std::size_t s = 0; s < size; ++s) rh[k][s] += ra[i][s] * rb[k - i][s];
		for (std::size_t k = 0; k <= bits; ++k)
			atl::detail::butterfly_transform(rh[k].begin(), bits, atl::detail::subset_mobius_butterfly{});
		for (std::size_t s = 0; s < size; ++s) x[s] = rh[std::size_t(std::popcount(s))][s];
		return x;
	}

	/**
	 * @brief The definition of a bitwise convolution: `h[op(i, j)] += a[i]
	 *        b[j]` over all pairs, with the library padding rule.
	 * @complexity O(4^n)
	 */
	template<atl::commutative_ring T, typename Op>
		requires std::regular_invocable<Op const &, std::size_t, std::size_t>
			  && std::convertible_to<std::invoke_result_t<Op const &, std::size_t, std::size_t>, std::size_t>
	constexpr std::vector<T> bitwise_convolution_by_definition(std::vector<T> const &a, std::vector<T> const &b, Op op) {
		std::size_t const longest = (std::max)(a.size(), b.size());
		std::vector<T> h(longest == 0 ? 0 : std::bit_ceil(longest));
		for (std::size_t i = 0; i < a.size(); ++i)
			for (std::size_t j = 0; j < b.size(); ++j) h[op(i, j)] += a[i] * b[j];
		return h;
	}

	/**
	 * @brief The definition of the subset convolution: every split of every
	 *        mask into a submask and its complement.
	 * @complexity O(3^n)
	 */
	template<atl::commutative_ring T>
	constexpr std::vector<T> subset_convolution_by_definition(std::vector<T> const &a, std::vector<T> const &b) {
		std::size_t const longest = (std::max)(a.size(), b.size());
		std::vector<T> h(longest == 0 ? 0 : std::bit_ceil(longest));
		auto const at = [](std::vector<T> const &v, std::size_t i) { return i < v.size() ? v[i] : T{}; };
		for (std::size_t s = 0; s < h.size(); ++s)
			for (std::size_t l = s;; l = (l - 1) & s) {
				h[s] += at(a, l) * at(b, s ^ l);
				if (l == 0) break;
			}
		return h;
	}
}
