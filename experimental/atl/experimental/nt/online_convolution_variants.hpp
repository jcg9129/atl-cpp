#pragma once

#include <atl/nt/detail/online_convolution.hpp>
#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <concepts>
#include <cstddef>

/**
 * @file
 * @brief Contenders for `atl::online_convolution` and `atl::online_fixed_convolution` of `<atl/nt/online_convolution.hpp>`.
 *
 * Each class has the template parameters, `push` and `size` of the library class it races, plus a constructor argument
 * `cutoff`: blocks up to that size are multiplied directly (the library uses a provisional 32). All races are pending:
 * no measurement decides any of them yet (benchmarks/nt/online_convolution.cpp, cutoffs swept over 16..128).
 * @pre For every class: `0 < cutoff <= maximum_series_length / 2` (asserted) and at most `maximum_series_length` pushes.
 */

namespace atl::experimental {
	/** @brief `online_convolution` with the library schedule (spectra of the stationary blocks kept) with the direct block size as an argument; race pending. */
	template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
	class online_convolution_by_block_cutoff {
		/** @brief The stream and the selected schedule. */
		atl::detail::online_product_state<Coefficient, Engine, atl::detail::online_product_algorithm::cached, false> state_;

	public:
		/** @brief Start an empty stream and choose the direct block size. */
		constexpr explicit online_convolution_by_block_cutoff(std::size_t cutoff = atl::detail::online_convolution_cutoff) : state_(cutoff) {}

		/** @brief Append the next element of both streams and return the product coefficient of that index. */
		[[nodiscard]] constexpr Coefficient push(Coefficient left, Coefficient right) { return state_.push(left, right); }

		/** @brief Number of pushes so far. */
		[[nodiscard]] constexpr std::size_t size() const { return state_.size(); }
	};

	/** @brief `online_fixed_convolution` with the library schedule (spectra of the stationary blocks kept) with the direct block size as an argument; race pending. */
	template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
	class online_fixed_convolution_by_block_cutoff {
		/** @brief The stream and the selected schedule. */
		atl::detail::online_product_state<Coefficient, Engine, atl::detail::online_product_algorithm::cached, true> state_;

	public:
		/** @brief Copy the kernel (trailing zeros ignored) and choose the direct block size. */
		template<prime_mod_int_range Range>
			requires std::same_as<atl::series_coefficient_t<Range>, Coefficient>
		constexpr explicit online_fixed_convolution_by_block_cutoff(Range const &kernel, std::size_t cutoff = atl::detail::online_convolution_cutoff) : state_(kernel, cutoff) {}

		/** @brief Append the next stream element and return its coefficient of the product with the kernel. */
		[[nodiscard]] constexpr Coefficient push(Coefficient value) { return state_.push(value); }

		/** @brief Number of pushes so far. */
		[[nodiscard]] constexpr std::size_t size() const { return state_.size(); }
	};

	/** @brief `online_convolution` with the library schedule, transforming the stationary blocks again for every rectangle instead of keeping their spectra; race pending. */
	template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
	class online_convolution_by_uncached_blocks {
		/** @brief The stream and the selected schedule. */
		atl::detail::online_product_state<Coefficient, Engine, atl::detail::online_product_algorithm::uncached, false> state_;

	public:
		/** @brief Start an empty stream and choose the direct block size. */
		constexpr explicit online_convolution_by_uncached_blocks(std::size_t cutoff = atl::detail::online_convolution_cutoff) : state_(cutoff) {}

		/** @brief Append the next element of both streams and return the product coefficient of that index. */
		[[nodiscard]] constexpr Coefficient push(Coefficient left, Coefficient right) { return state_.push(left, right); }

		/** @brief Number of pushes so far. */
		[[nodiscard]] constexpr std::size_t size() const { return state_.size(); }
	};

	/** @brief `online_fixed_convolution` with the library schedule, transforming the stationary blocks again for every rectangle instead of keeping their spectra; race pending. */
	template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
	class online_fixed_convolution_by_uncached_blocks {
		/** @brief The stream and the selected schedule. */
		atl::detail::online_product_state<Coefficient, Engine, atl::detail::online_product_algorithm::uncached, true> state_;

	public:
		/** @brief Copy the kernel (trailing zeros ignored) and choose the direct block size. */
		template<prime_mod_int_range Range>
			requires std::same_as<atl::series_coefficient_t<Range>, Coefficient>
		constexpr explicit online_fixed_convolution_by_uncached_blocks(Range const &kernel, std::size_t cutoff = atl::detail::online_convolution_cutoff) : state_(kernel, cutoff) {}

		/** @brief Append the next stream element and return its coefficient of the product with the kernel. */
		[[nodiscard]] constexpr Coefficient push(Coefficient value) { return state_.push(value); }

		/** @brief Number of pushes so far. */
		[[nodiscard]] constexpr std::size_t size() const { return state_.size(); }
	};

	/** @brief `online_convolution` with the defining quadratic sum at every push, without blocks (`cutoff` is accepted and unused); race pending. */
	template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
	class online_convolution_by_direct_sums {
		/** @brief The stream and the selected schedule. */
		atl::detail::online_product_state<Coefficient, Engine, atl::detail::online_product_algorithm::direct, false> state_;

	public:
		/** @brief Start an empty stream and choose the direct block size. */
		constexpr explicit online_convolution_by_direct_sums(std::size_t cutoff = atl::detail::online_convolution_cutoff) : state_(cutoff) {}

		/** @brief Append the next element of both streams and return the product coefficient of that index. */
		[[nodiscard]] constexpr Coefficient push(Coefficient left, Coefficient right) { return state_.push(left, right); }

		/** @brief Number of pushes so far. */
		[[nodiscard]] constexpr std::size_t size() const { return state_.size(); }
	};

	/** @brief `online_fixed_convolution` with the defining quadratic sum at every push, without blocks (`cutoff` is accepted and unused); race pending. */
	template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
	class online_fixed_convolution_by_direct_sums {
		/** @brief The stream and the selected schedule. */
		atl::detail::online_product_state<Coefficient, Engine, atl::detail::online_product_algorithm::direct, true> state_;

	public:
		/** @brief Copy the kernel (trailing zeros ignored) and choose the direct block size. */
		template<prime_mod_int_range Range>
			requires std::same_as<atl::series_coefficient_t<Range>, Coefficient>
		constexpr explicit online_fixed_convolution_by_direct_sums(Range const &kernel, std::size_t cutoff = atl::detail::online_convolution_cutoff) : state_(kernel, cutoff) {}

		/** @brief Append the next stream element and return its coefficient of the product with the kernel. */
		[[nodiscard]] constexpr Coefficient push(Coefficient value) { return state_.push(value); }

		/** @brief Number of pushes so far. */
		[[nodiscard]] constexpr std::size_t size() const { return state_.size(); }
	};

	/** @brief `online_convolution` with the library schedule with recursive Karatsuba for the block products instead of transforms; race pending. */
	template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
	class online_convolution_by_karatsuba_blocks {
		/** @brief The stream and the selected schedule. */
		atl::detail::online_product_state<Coefficient, Engine, atl::detail::online_product_algorithm::karatsuba, false> state_;

	public:
		/** @brief Start an empty stream and choose the direct block size. */
		constexpr explicit online_convolution_by_karatsuba_blocks(std::size_t cutoff = atl::detail::online_convolution_cutoff) : state_(cutoff) {}

		/** @brief Append the next element of both streams and return the product coefficient of that index. */
		[[nodiscard]] constexpr Coefficient push(Coefficient left, Coefficient right) { return state_.push(left, right); }

		/** @brief Number of pushes so far. */
		[[nodiscard]] constexpr std::size_t size() const { return state_.size(); }
	};

	/** @brief `online_fixed_convolution` with the library schedule with recursive Karatsuba for the block products instead of transforms; race pending. */
	template<static_prime_mod_int Coefficient, ntt_engine Engine = default_ntt_engine>
	class online_fixed_convolution_by_karatsuba_blocks {
		/** @brief The stream and the selected schedule. */
		atl::detail::online_product_state<Coefficient, Engine, atl::detail::online_product_algorithm::karatsuba, true> state_;

	public:
		/** @brief Copy the kernel (trailing zeros ignored) and choose the direct block size. */
		template<prime_mod_int_range Range>
			requires std::same_as<atl::series_coefficient_t<Range>, Coefficient>
		constexpr explicit online_fixed_convolution_by_karatsuba_blocks(Range const &kernel, std::size_t cutoff = atl::detail::online_convolution_cutoff) : state_(kernel, cutoff) {}

		/** @brief Append the next stream element and return its coefficient of the product with the kernel. */
		[[nodiscard]] constexpr Coefficient push(Coefficient value) { return state_.push(value); }

		/** @brief Number of pushes so far. */
		[[nodiscard]] constexpr std::size_t size() const { return state_.size(); }
	};
}
