#pragma once
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Transforms of functions on the subsets of an n-element set (arrays of
 *        size 2^n indexed by bitmask) and the products they diagonalize:
 *        zeta / Möbius over subsets and supersets, the Walsh–Hadamard
 *        transform, and the OR, AND, XOR and subset (disjoint union)
 *        convolutions.
 *
 * Every transform is a tensor power of one 2x2 map. The subset product
 * uses ranked OR convolution. References: Liu Chengao (2018), "DFT" (national team papers);
 * Lv Kaifeng (2015), "Set power series", §5.1.1. Implemented from the descriptions.
 */

namespace atl {
	/**
	 * @brief An element the zeta and Möbius transforms can run over: a regular
	 *        type with `T{}` as zero, `+`, `-` and their compound forms. Exact
	 *        for modular integers and for plain integers (unsigned ones work
	 *        modulo 2^width).
	 */
	template<typename T>
	concept additive_group = std::regular<T> && requires(T a, T const b) {
		{ b + b } -> std::convertible_to<T>;
		{ b - b } -> std::convertible_to<T>;
		{ a += b } -> std::same_as<T &>;
		{ a -= b } -> std::same_as<T &>;
	};

	/**
	 * @brief An element the convolutions can run over: an `atl::additive_group`
	 *        with a commutative `*`.
	 */
	template<typename T>
	concept commutative_ring = additive_group<T> && requires(T a, T const b) {
		{ b * b } -> std::convertible_to<T>;
		{ a *= b } -> std::same_as<T &>;
	};

	/**
	 * @brief A ring in which the inverse Walsh–Hadamard transform's division
	 *        by 2^n is available: modular integers (odd modulus, asserted),
	 *        signed integers (exact division, the input must be a transform
	 *        image) and floating point. Unsigned integers are excluded: modulo
	 *        2^width the division by 2^n is not defined.
	 */
	template<typename T>
	concept halvable_ring =
		commutative_ring<T> && (modular_integer<T> || signed_integral<T> || std::floating_point<T>);

	/**
	 * @brief A mutable sized random-access range of @ref atl::additive_group
	 *        elements, transformed in place.
	 */
	template<typename R>
	concept set_function_range = std::ranges::random_access_range<R> && std::ranges::sized_range<R>
							  && additive_group<std::ranges::range_value_t<R>>
							  && std::ranges::output_range<R, std::ranges::range_value_t<R>>
							  && std::same_as<std::ranges::range_reference_t<R>, std::ranges::range_value_t<R> &>;

	/**
	 * @brief A sized random-access range of elements of the ring @p T, read by
	 *        the convolutions.
	 */
	template<typename R, typename T>
	concept ring_range = std::ranges::random_access_range<R const> && std::ranges::sized_range<R const>
					  && std::same_as<std::ranges::range_value_t<R>, T>
					  && std::indirectly_copyable<std::ranges::iterator_t<R const>, T *>;

	namespace detail {
		/**
		 * @brief A 2x2 butterfly over `T`: invoked on the pair (f[S], f[S | bit])
		 *        and replacing them in place.
		 */
		template<typename Butterfly, typename T>
		concept butterfly_of = std::invocable<Butterfly const &, T &, T &>;

		/** @brief (lo, hi) -> (lo, hi + lo): the subset zeta along one bit. */
		struct subset_zeta_butterfly {
			template<additive_group T>
			constexpr void operator()(T &lo, T &hi) const {
				hi += lo;
			}
		};

		/** @brief (lo, hi) -> (lo, hi - lo): the subset Möbius along one bit. */
		struct subset_mobius_butterfly {
			template<additive_group T>
			constexpr void operator()(T &lo, T &hi) const {
				hi -= lo;
			}
		};

		/** @brief (lo, hi) -> (lo + hi, hi): the superset zeta along one bit. */
		struct superset_zeta_butterfly {
			template<additive_group T>
			constexpr void operator()(T &lo, T &hi) const {
				lo += hi;
			}
		};

		/** @brief (lo, hi) -> (lo - hi, hi): the superset Möbius along one bit. */
		struct superset_mobius_butterfly {
			template<additive_group T>
			constexpr void operator()(T &lo, T &hi) const {
				lo -= hi;
			}
		};

		/** @brief (lo, hi) -> (lo + hi, lo - hi): Walsh–Hadamard along one bit. */
		struct walsh_hadamard_butterfly {
			template<additive_group T>
			constexpr void operator()(T &lo, T &hi) const {
				T const sum = T(lo + hi);
				hi = T(lo - hi);
				lo = sum;
			}
		};

		/**
		 * @brief The cache block of the transform engine: bits below it are
		 *        finished one 2^11-element block at a time, while the block
		 *        is in L1 (16 KiB of 64-bit elements).
		 */
		inline constexpr std::size_t transform_block_bits = 11;

		/**
		 * @brief The radix-2 passes along bits [low, high) of the @p size
		 *        elements at @p f (a multiple of 2^high).
		 */
		template<std::random_access_iterator It, butterfly_of<std::iter_value_t<It>> Butterfly>
			requires std::invocable<Butterfly const &, std::iter_reference_t<It>, std::iter_reference_t<It>>
		constexpr void radix2_passes(It f, std::size_t size, std::size_t low, std::size_t high,
									 Butterfly const &butterfly) {
			for (std::size_t bit = low; bit < high; ++bit) {
				std::size_t const half = std::size_t(1) << bit;
				for (std::size_t block = 0; block < size; block += 2 * half)
					for (std::size_t i = block; i < block + half; ++i) butterfly(f[i], f[i + half]);
			}
		}

		/**
		 * @brief Apply @p butterfly along every bit of the 2^bits elements at
		 *        @p f, so the array goes through the tensor power of the 2x2
		 *        map: the low bits finished inside each L1-sized block, then
		 *        one radix-2 sweep per remaining bit. The order of the bits
		 *        does not matter, the per-bit maps commute.
		 *
		 * Raced at n = 20 against plain bit-outer, radix-4 and blocked +
		 * radix-4 (experimental/atl/experimental/nt/bitwise_convolution_variants.hpp):
		 * the four orders sit within about 10% of each other on every shape,
		 * and this one has the lowest total in both flag configurations.
		 */
		template<std::random_access_iterator It, butterfly_of<std::iter_value_t<It>> Butterfly>
			requires std::invocable<Butterfly const &, std::iter_reference_t<It>, std::iter_reference_t<It>>
		constexpr void butterfly_transform(It f, std::size_t bits, Butterfly const &butterfly) {
			std::size_t const low = (std::min)(bits, transform_block_bits);
			std::size_t const size = std::size_t(1) << bits;
			for (std::size_t block = 0; block < size; block += std::size_t(1) << low)
				radix2_passes(f + std::ptrdiff_t(block), std::size_t(1) << low, 0, low, butterfly);
			radix2_passes(f, size, low, bits, butterfly);
		}

		/**
		 * @brief The number of bits of a set-function array, asserting its
		 *        size is a power of two.
		 */
		constexpr std::size_t set_function_bits(std::size_t size) {
			atl_assert(std::has_single_bit(size));
			return std::size_t(std::countr_zero(size));
		}

		template<set_function_range R, butterfly_of<std::ranges::range_value_t<R>> Butterfly>
		constexpr void transform_range(R &&f, Butterfly const &butterfly) {
			std::size_t const bits = set_function_bits(std::size_t(std::ranges::size(f)));
			butterfly_transform(std::ranges::begin(f), bits, butterfly);
		}

		/**
		 * @brief Multiply every element of @p f by 1/2^bits, the scale of the
		 *        inverse Walsh–Hadamard transform.
		 */
		template<std::random_access_iterator It>
			requires halvable_ring<std::iter_value_t<It>> && std::indirectly_writable<It, std::iter_value_t<It>>
		constexpr void scale_by_inverse_power_of_two(It f, std::size_t bits) {
			using T = std::iter_value_t<It>;
			std::size_t const size = std::size_t(1) << bits;
			if constexpr (modular_integer<T>) {
				atl_assert(T::mod() % 2 == 1);
				T const scale = T(uint64(2)).inverse().power(bits);
				for (std::size_t i = 0; i < size; ++i) f[i] *= scale;
			} else if constexpr (std::floating_point<T>) {
				T const scale = T(1) / T(size);
				for (std::size_t i = 0; i < size; ++i) f[i] *= scale;
			} else {
				for (std::size_t i = 0; i < size; ++i) {
					if (uint128(size) <= uint128((std::numeric_limits<T>::max)())) {
						atl_assert(f[i] % T(size) == 0);
						f[i] /= T(size);
						continue;
					}
					// Divide in representable steps even when 2^bits exceeds T's range.
					for (std::size_t bit = 0; bit < bits; ++bit) {
						atl_assert(f[i] % T(2) == 0);
						f[i] /= T(2);
					}
				}
			}
		}

		/**
		 * @brief Copies of @p a and @p b zero-padded to the common length
		 *        `bit_ceil(max(|a|, |b|))`; both empty when the inputs are.
		 */
		template<commutative_ring T, ring_range<T> A, ring_range<T> B>
		constexpr std::pair<std::vector<T>, std::vector<T>> padded_pair(A const &a, B const &b) {
			std::size_t const longest = (std::max)(std::size_t(std::ranges::size(a)), std::size_t(std::ranges::size(b)));
			atl_assert(longest <= (std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 1)));
			std::size_t const size = longest == 0 ? 0 : std::bit_ceil(longest);
			std::vector<T> x(size), y(size);
			std::ranges::copy(a, x.begin());
			std::ranges::copy(b, y.begin());
			return {std::move(x), std::move(y)};
		}

		/**
		 * @brief The convolution diagonalized by @p forward / @p inverse:
		 *        transform both, multiply pointwise, transform back.
		 */
		template<commutative_ring T, ring_range<T> A, ring_range<T> B, butterfly_of<T> Forward, butterfly_of<T> Inverse>
		constexpr std::vector<T> diagonal_convolution(A const &a, B const &b, Forward const &forward,
													  Inverse const &inverse) {
			auto [x, y] = padded_pair<T>(a, b);
			if (x.empty()) return x;
			std::size_t const bits = std::size_t(std::countr_zero(x.size()));
			butterfly_transform(x.begin(), bits, forward);
			butterfly_transform(y.begin(), bits, forward);
			for (std::size_t i = 0; i < x.size(); ++i) x[i] *= y[i];
			butterfly_transform(x.begin(), bits, inverse);
			return x;
		}
	}

	/**
	 * @brief Subset zeta transform, in place: `f[S] <- sum of f[T] over T ⊆ S`.
	 * @param f Mutable mask-indexed values; T{} is the additive identity.
	 * @pre Arithmetic obeys the documented group laws with no signed overflow.
	 *      `|f|` is a power of two (asserted)
	 * @complexity O(n 2^n) additions for `|f| = 2^n`
	 */
	template<set_function_range R>
	constexpr void subset_zeta(R &&f) {
		detail::transform_range(f, detail::subset_zeta_butterfly{});
	}

	/**
	 * @brief Subset Möbius transform, the inverse of `subset_zeta:`
	 *        `f[S] <- sum of (-1)^|S \ T| f[T] over T ⊆ S`.
	 * @param f Mutable mask-indexed values; T{} is the additive identity.
	 * @pre Arithmetic obeys the documented group laws with no signed overflow.
	 *      `|f|` is a power of two (asserted)
	 * @complexity O(n 2^n)
	 */
	template<set_function_range R>
	constexpr void subset_mobius(R &&f) {
		detail::transform_range(f, detail::subset_mobius_butterfly{});
	}

	/**
	 * @brief Superset zeta transform, in place: `f[S] <- sum of f[T] over T ⊇ S`.
	 * @param f Mutable mask-indexed values; T{} is the additive identity.
	 * @pre Arithmetic obeys the documented group laws with no signed overflow.
	 *      `|f|` is a power of two (asserted)
	 * @complexity O(n 2^n)
	 */
	template<set_function_range R>
	constexpr void superset_zeta(R &&f) {
		detail::transform_range(f, detail::superset_zeta_butterfly{});
	}

	/**
	 * @brief Superset Möbius transform, the inverse of @ref atl::superset_zeta.
	 * @param f Mutable mask-indexed values; T{} is the additive identity.
	 * @pre Arithmetic obeys the documented group laws with no signed overflow.
	 *      `|f|` is a power of two (asserted)
	 * @complexity O(n 2^n)
	 */
	template<set_function_range R>
	constexpr void superset_mobius(R &&f) {
		detail::transform_range(f, detail::superset_mobius_butterfly{});
	}

	/**
	 * @brief Walsh–Hadamard transform, in place, unscaled:
	 *        `f[S] <- sum of (-1)^|S ∩ T| f[T] over all T`. Applying it twice
	 *        multiplies by 2^n.
	 * @param f Mutable mask-indexed values; T{} is the additive identity.
	 * @pre Arithmetic obeys the documented group laws with no signed overflow.
	 *      `|f|` is a power of two (asserted)
	 * @complexity O(n 2^n)
	 */
	template<set_function_range R>
	constexpr void walsh_hadamard(R &&f) {
		detail::transform_range(f, detail::walsh_hadamard_butterfly{});
	}

	/**
	 * @brief The inverse Walsh–Hadamard transform: the transform again, then
	 *        division by 2^n.
	 * @param f Mutable mask-indexed values; T{} is the additive identity.
	 * @pre Arithmetic obeys the documented group laws with no signed overflow.
	 *      `|f|` is a power of two (asserted); a modular element type has an
	 *      odd modulus (asserted); for signed integers every entry of the
	 *      unscaled result is divisible by 2^n, which holds when `f` is the
	 *      transform of an integer array (asserted)
	 * @complexity O(n 2^n)
	 */
	template<set_function_range R>
		requires halvable_ring<std::ranges::range_value_t<R>>
	constexpr void inverse_walsh_hadamard(R &&f) {
		detail::transform_range(f, detail::walsh_hadamard_butterfly{});
		detail::scale_by_inverse_power_of_two(std::ranges::begin(f),
											  std::size_t(std::countr_zero(std::size_t(std::ranges::size(f)))));
	}

	/**
	 * @brief OR convolution: `h[S] = sum of a[L] b[R] over L | R = S`.
	 *
	 * The inputs are zero-padded to the common length `bit_ceil(max(|a|,
	 * |b|))`, which is the length of the result (empty when both are empty).
	 * @param a First const-readable mask-indexed range.
	 * @param b Second range of the same value type.
	 * @return Zero-padded product; empty if both inputs are empty.
	 * @pre Ring laws hold and signed intermediate arithmetic does not overflow.
	 * @complexity O(n 2^n) time, O(2^n) auxiliary storage.
	 */
	template<std::ranges::random_access_range A, ring_range<std::ranges::range_value_t<A>> B>
		requires ring_range<A, std::ranges::range_value_t<A>> && commutative_ring<std::ranges::range_value_t<A>>
	constexpr std::vector<std::ranges::range_value_t<A>> or_convolution(A const &a, B const &b) {
		return detail::diagonal_convolution<std::ranges::range_value_t<A>>(a, b, detail::subset_zeta_butterfly{},
																		   detail::subset_mobius_butterfly{});
	}

	/**
	 * @brief AND convolution: `h[S] = sum of a[L] b[R] over L & R = S`, with
	 *        the padding of @ref atl::or_convolution.
	 * @param a First const-readable mask-indexed range.
	 * @param b Second range of the same value type.
	 * @return Zero-padded product; empty if both inputs are empty.
	 * @pre Ring laws hold and signed intermediate arithmetic does not overflow.
	 * @complexity O(n 2^n) time, O(2^n) auxiliary storage.
	 */
	template<std::ranges::random_access_range A, ring_range<std::ranges::range_value_t<A>> B>
		requires ring_range<A, std::ranges::range_value_t<A>> && commutative_ring<std::ranges::range_value_t<A>>
	constexpr std::vector<std::ranges::range_value_t<A>> and_convolution(A const &a, B const &b) {
		return detail::diagonal_convolution<std::ranges::range_value_t<A>>(
			a, b, detail::superset_zeta_butterfly{}, detail::superset_mobius_butterfly{});
	}

	/**
	 * @brief XOR convolution: `h[S] = sum of a[L] b[R] over L ^ R = S`, with
	 *        the padding of @ref atl::or_convolution. Needs the division by 2^n of
	 *        @ref atl::inverse_walsh_hadamard; for signed integers the transform
	 *        domain has the conservative bound `2^(3n) max|a| max|b|`, which must
	 *        not overflow.
	 * @param a First const-readable mask-indexed range.
	 * @param b Second range of the same value type.
	 * @return Zero-padded product; empty if both inputs are empty.
	 * @pre Ring laws hold and signed intermediate arithmetic does not overflow.
	 * @complexity O(n 2^n) time, O(2^n) auxiliary storage.
	 */
	template<std::ranges::random_access_range A, ring_range<std::ranges::range_value_t<A>> B>
		requires ring_range<A, std::ranges::range_value_t<A>> && halvable_ring<std::ranges::range_value_t<A>>
	constexpr std::vector<std::ranges::range_value_t<A>> xor_convolution(A const &a, B const &b) {
		using T = std::ranges::range_value_t<A>;
		auto h = detail::diagonal_convolution<T>(a, b, detail::walsh_hadamard_butterfly{},
												 detail::walsh_hadamard_butterfly{});
		if (!h.empty()) detail::scale_by_inverse_power_of_two(h.begin(), std::size_t(std::countr_zero(h.size())));
		return h;
	}

	/**
	 * @brief Subset convolution: `h[S] = sum of a[L] b[S \ L] over L ⊆ S`, the
	 *        product over disjoint unions, with the padding of
	 *        @ref atl::or_convolution.
	 *
	 * Ranked zeta transform: row S holds the polynomial in z whose degree-|S|
	 * coefficient is a[S]; the OR convolution of the rows, read at degree
	 * |S|, is h[S]. Rows are laid out mask-major (each mask's n+1 ranks
	 * contiguous), so the per-mask polynomial product is local, and every
	 * step touches only the ranks that can be nonzero (zeta) or that the
	 * final read needs (Möbius), which halves each of the three phases.
	 * Needs no division, so it runs over any commutative ring.
	 * @param a First const-readable mask-indexed range.
	 * @param b Second range of the same value type.
	 * @return Zero-padded disjoint-subset product; empty if both inputs are empty.
	 * @pre Ring laws hold, signed intermediates do not overflow, and the padded
	 *      length times (n+1) fits size_t (asserted).
	 * @complexity O(n^2 2^n) time, O(n 2^n) space
	 */
	template<std::ranges::random_access_range A, ring_range<std::ranges::range_value_t<A>> B>
		requires ring_range<A, std::ranges::range_value_t<A>> && commutative_ring<std::ranges::range_value_t<A>>
	constexpr std::vector<std::ranges::range_value_t<A>> subset_convolution(A const &a, B const &b) {
		using T = std::ranges::range_value_t<A>;
		auto [x, y] = detail::padded_pair<T>(a, b);
		if (x.empty()) return x;
		std::size_t const size = x.size();
		std::size_t const bits = std::size_t(std::countr_zero(size));
		std::size_t const width = bits + 1;
		atl_assert(size <= (std::numeric_limits<std::size_t>::max)() / width);
		std::vector<T> ranked_a(size * width), ranked_b(size * width);
		for (std::size_t s = 0; s < size; ++s) {
			std::size_t const rank = std::size_t(std::popcount(s));
			ranked_a[s * width + rank] = x[s];
			ranked_b[s * width + rank] = y[s];
		}
		// Ranks above |S| are still zero, so they need not be propagated.
		for (std::size_t bit = 0; bit < bits; ++bit) {
			std::size_t const half = std::size_t(1) << bit;
			for (std::size_t block = 0; block < size; block += 2 * half)
				for (std::size_t s = block; s < block + half; ++s) {
					std::size_t const top = std::size_t(std::popcount(s));
					T const *const lo_a = ranked_a.data() + s * width;
					T const *const lo_b = ranked_b.data() + s * width;
					T *const hi_a = ranked_a.data() + (s + half) * width;
					T *const hi_b = ranked_b.data() + (s + half) * width;
					for (std::size_t k = 0; k <= top; ++k) hi_a[k] += lo_a[k], hi_b[k] += lo_b[k];
				}
		}
		// Only k >= |S| can reach a final read through Möbius at a superset T;
		// the factors vanish above rank |S|.
		std::vector<T> product(width);
		for (std::size_t s = 0; s < size; ++s) {
			std::size_t const rank = std::size_t(std::popcount(s));
			T *const row_a = ranked_a.data() + s * width;
			T const *const row_b = ranked_b.data() + s * width;
			for (std::size_t k = rank; k <= bits; ++k) {
				T sum{};
				for (std::size_t i = k - (std::min)(k, rank); i <= (std::min)(k, rank); ++i) sum += row_a[i] * row_b[k - i];
				product[k] = sum;
			}
			for (std::size_t k = rank; k <= bits; ++k) row_a[k] = product[k];
		}
		// Only ranks above |S| can contribute to a read at a strict superset.
		for (std::size_t bit = 0; bit < bits; ++bit) {
			std::size_t const half = std::size_t(1) << bit;
			for (std::size_t block = 0; block < size; block += 2 * half)
				for (std::size_t s = block; s < block + half; ++s) {
					T const *const lo = ranked_a.data() + s * width;
					T *const hi = ranked_a.data() + (s + half) * width;
					for (std::size_t k = std::size_t(std::popcount(s)) + 1; k <= bits; ++k) hi[k] -= lo[k];
				}
		}
		for (std::size_t s = 0; s < size; ++s) x[s] = ranked_a[s * width + std::size_t(std::popcount(s))];
		return x;
	}
}
