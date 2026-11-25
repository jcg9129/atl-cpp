#pragma once

/** @file
 * @brief Wide-residue series product policy, installed before wide coefficient instantiation.
 */

#include <atl/nt/convolution_uint64.hpp>
#include <atl/nt/detail/power_series_native.hpp>
#include <atl/nt/ntt_engine_core.hpp>
#include <atl/nt/prime_mod_int_core.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace atl::detail {
	/**
	 * @brief Wide-residue series products by 16-bit limbs through exact integer convolution.
	 *
	 * Moduli above 32 bits, and storage types without native double-width multiplication, cannot use the
	 * modular driver's three-prime reconstruction bound. Constant evaluation uses schoolbook products.
	 */
	template<static_prime_mod_int Coefficient>
		requires(!native_series_coefficient<Coefficient>)
	struct series_product_policy<Coefficient> {
		template<ntt_engine Engine>
		static constexpr std::vector<Coefficient> multiply(std::span<Coefficient const> left, std::span<Coefficient const> right, std::size_t length) {
			std::vector<Coefficient> result(length);
			if consteval {
				for (std::size_t i = 0; i < left.size(); ++i)
					for (std::size_t j = 0; j < right.size() && j < length - i; ++j) result[i + j] += left[i] * right[j];
			} else {
				std::vector<std::uint64_t> left_limb(left.size()), right_limb(right.size());
				Coefficient left_place(1);
				for (unsigned left_shift = 0; left_shift < 64; left_shift += 16) {
					for (std::size_t i = 0; i < left.size(); ++i) left_limb[i] = (std::uint64_t(left[i].value()) >> left_shift) & 65535;
					Coefficient place = left_place;
					for (unsigned right_shift = 0; right_shift < 64; right_shift += 16) {
						for (std::size_t i = 0; i < right.size(); ++i) right_limb[i] = (std::uint64_t(right[i].value()) >> right_shift) & 65535;
						auto const product = convolution_uint64<Engine>(left_limb, right_limb);
						for (std::size_t i = 0; i < (std::min)(length, product.size()); ++i) result[i] += Coefficient(product[i]) * place;
						place *= Coefficient(65536);
					}
					left_place *= Coefficient(65536);
				}
			}
			return result;
		}
	};
}
