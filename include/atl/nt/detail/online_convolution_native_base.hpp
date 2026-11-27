#pragma once

#include <atl/nt/detail/ntt_engine_base.hpp>
#include <atl/nt/detail/polynomial_points_native_base.hpp>
#include <atl/nt/detail/power_series_extensions_native_base.hpp>
#include <atl/nt/detail/power_series_native_base.hpp>
#include <atl/nt/detail/prime_mod_int_base.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

namespace atl::detail {
	/** @brief Provisional size up to which a dyadic block is multiplied directly; race pending (swept over 16..128). */
	inline constexpr std::size_t online_convolution_cutoff = 32;

	/** @brief Relaxed multiplication contenders; race pending. */
	enum class online_product_algorithm {
		/** @brief Dyadic rectangles; the spectra of the stationary blocks are kept until the stream ends. */
		cached,
		/** @brief Dyadic rectangles; the stationary blocks are transformed again for every rectangle. */
		uncached,
		/** @brief The defining quadratic Cauchy sum at every push. */
		direct,
		/** @brief Dyadic rectangles multiplied by recursive Karatsuba instead of transforms. */
		karatsuba
	};

	/** @brief Spectra (length `2s`) of the stationary dyadic blocks `[s, 2s)` of the left and right input; `right` is the kernel block of a fixed stream. */
	template<static_prime_mod_int Coefficient>
	struct online_block_spectra {
		std::vector<Coefficient> left;
		std::vector<Coefficient> right;
	};

	/** @brief Karatsuba product of two equal power-of-two blocks, independent of the transform kernels.
	 * @pre Equal nonzero power-of-two lengths, at most `maximum_series_length / 2` (asserted).
	 */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr auto online_karatsuba(std::span<Coefficient const> left, std::span<Coefficient const> right) -> std::vector<Coefficient> {
		atl_assert(left.size() == right.size() && std::has_single_bit(left.size()) && left.size() <= maximum_series_length / 2);
		std::size_t const size = left.size();
		std::vector<Coefficient> result(2 * size - 1);
		if (size <= 16) {
			for (std::size_t i = 0; i < size; ++i)
				for (std::size_t j = 0; j < size; ++j) result[i + j] += left[i] * right[j];
			return result;
		}
		std::size_t const half = size / 2;
		auto const lower = online_karatsuba<Coefficient>(left.first(half), right.first(half));
		auto const upper = online_karatsuba<Coefficient>(left.subspan(half), right.subspan(half));
		std::vector<Coefficient> left_sum(half), right_sum(half);
		for (std::size_t i = 0; i < half; ++i) {
			left_sum[i] = left[i] + left[half + i];
			right_sum[i] = right[i] + right[half + i];
		}
		auto middle = online_karatsuba<Coefficient>(left_sum, right_sum);
		for (std::size_t i = 0; i < middle.size(); ++i) {
			middle[i] -= lower[i] + upper[i];
			result[i] += lower[i];
			result[size + i] += upper[i];
			result[half + i] += middle[i];
		}
		return result;
	}

	/** @brief The relaxed schedule: a dyadic partition of the Cauchy product, with the stream and the pending outputs.
	 *
	 * Every pair `(i, j)` with `i, j >= 1` belongs to the unique scale `s` with `min(i, j)` in `[s, 2s)`. When the
	 * count `t` of pushed inputs is divisible by `s` (and `t >= 2s`), the stationary block `[s, 2s)` times the newly
	 * completed block `[t - s, t)` starts at output `t`, so no output is read before its inputs have arrived. Pairs
	 * with an index of zero are added at once. For a fixed kernel the right side is known in full and is padded with
	 * zeros as the stream grows.
	 */
	template<static_prime_mod_int Coefficient, ntt_engine Engine, online_product_algorithm Algorithm, bool IsFixed>
	class online_product_state {
		std::vector<Coefficient> left_;
		/** @brief The streamed right input, or the kernel (normalized, padded with zeros as the stream grows). */
		std::vector<Coefficient> right_;
		/** @brief Output coefficients with their contributions scheduled so far. */
		std::vector<Coefficient> output_;
		/** @brief Stationary block spectra indexed by `log2(s)`. */
		std::vector<online_block_spectra<Coefficient>> spectra_;
		/** @brief Blocks up to this size are multiplied directly. */
		std::size_t cutoff_;

		/** @brief Transform of a block zero-padded to twice its length.
		 * @pre `0 < block.size() <= maximum_series_length / 2` (asserted).
		 */
		[[nodiscard]] constexpr auto transform(std::span<Coefficient const> block) const -> std::vector<Coefficient>
			requires points_transform_coefficient<Coefficient>
		{
			atl_assert(!block.empty() && block.size() <= maximum_series_length / 2);
			std::vector<Coefficient> result(2 * block.size());
			std::ranges::copy(block, result.begin());
			ntt<Engine>(result);
			return result;
		}

		/** @brief Add the rectangles of scale `block_size` that start at output `size()`, fusing their two orientations before one inverse transform.
		 * @pre `block_size` is a power of two, at most half of `size()`, and divides it (asserted).
		 */
		constexpr void schedule(std::size_t block_size) {
			std::size_t const count = left_.size();
			atl_assert(std::has_single_bit(block_size) && block_size <= count / 2 && count % block_size == 0);
			std::size_t const offset = count - block_size;
			// Nothing past the cap is ever returned, so the last block has no outputs left to feed.
			if (count == maximum_series_length) return;
			bool const is_diagonal = offset == block_size;
			auto const left = std::span<Coefficient const>(left_), right = std::span<Coefficient const>(right_);
			auto const left_prefix = left.subspan(block_size, block_size), right_prefix = right.subspan(block_size, block_size);
			auto const left_block = left.subspan(offset, block_size), right_block = right.subspan(offset, block_size);
			if (std::ranges::all_of(left_prefix, [](Coefficient value) { return value == Coefficient(0); })
				&& std::ranges::all_of(right_prefix, [](Coefficient value) { return value == Coefficient(0); })) return;
			std::size_t const length = (std::min)(2 * block_size - 1, maximum_series_length - count);
			if (output_.size() < count + length) output_.resize(count + length);
			if (block_size <= cutoff_) {
				for (std::size_t i = 0; i < block_size; ++i)
					for (std::size_t j = 0; j < block_size && i + j < length; ++j) {
						output_[count + i + j] += left_prefix[i] * right_block[j];
						if (!is_diagonal) output_[count + i + j] += right_prefix[i] * left_block[j];
					}
				return;
			}
			if constexpr (Algorithm == online_product_algorithm::karatsuba) {
				auto const first = online_karatsuba<Coefficient>(left_prefix, right_block);
				for (std::size_t i = 0; i < length; ++i) output_[count + i] += first[i];
				if (!is_diagonal) {
					auto const second = online_karatsuba<Coefficient>(right_prefix, left_block);
					for (std::size_t i = 0; i < length; ++i) output_[count + i] += second[i];
				}
				return;
			}
			if constexpr (points_transform_coefficient<Coefficient>) {
				if (points_can_transform<Coefficient>(2 * block_size)) {
					std::size_t const level = std::size_t(std::countr_zero(block_size));
					if (spectra_.size() <= level) spectra_.resize(level + 1);
					auto &cache = spectra_[level];
					auto first = transform(left_block);
					auto second = [&] {
						if constexpr (IsFixed && Algorithm == online_product_algorithm::cached)
							if (is_diagonal && !cache.right.empty()) return cache.right;
						return transform(right_block);
					}();
					if (is_diagonal) {
						if constexpr (Algorithm == online_product_algorithm::cached) {
							cache.left = first;
							cache.right = second;
						}
						for (std::size_t i = 0; i < first.size(); ++i) first[i] *= second[i];
					} else {
						if constexpr (Algorithm == online_product_algorithm::uncached) {
							cache.left = transform(left_prefix);
							cache.right = transform(right_prefix);
						}
						for (std::size_t i = 0; i < first.size(); ++i) first[i] = first[i] * cache.right[i] + second[i] * cache.left[i];
					}
					inverse_ntt<Engine>(first);
					for (std::size_t i = 0; i < length; ++i) output_[count + i] += first[i];
					if constexpr (Algorithm == online_product_algorithm::uncached) {
						cache.left.clear();
						cache.right.clear();
					}
					return;
				}
			}
			auto const first = series_product<Engine, Coefficient>(left_prefix, right_block, length);
			for (std::size_t i = 0; i < length; ++i) output_[count + i] += first[i];
			if (!is_diagonal) {
				auto const second = series_product<Engine, Coefficient>(right_prefix, left_block, length);
				for (std::size_t i = 0; i < length; ++i) output_[count + i] += second[i];
			}
		}

	public:
		/** @brief Start an empty fully relaxed stream.
		 * @pre `0 < cutoff <= maximum_series_length / 2` (asserted).
		 */
		constexpr explicit online_product_state(std::size_t cutoff = online_convolution_cutoff)
			requires(!IsFixed)
			: cutoff_(cutoff) {
			atl_assert(cutoff > 0 && cutoff <= maximum_series_length / 2);
		}

		/** @brief Start a semi-relaxed stream against a whole kernel, copied and normalized.
		 * @pre `0 < cutoff <= maximum_series_length / 2` and the normalized kernel length is at most `maximum_series_length` (asserted).
		 */
		template<prime_mod_int_range Range>
			requires IsFixed && std::same_as<series_coefficient_t<Range>, Coefficient>
		constexpr explicit online_product_state(Range const &kernel, std::size_t cutoff = online_convolution_cutoff)
			: right_(series_polynomial(kernel)), cutoff_(cutoff) {
			atl_assert(cutoff > 0 && cutoff <= maximum_series_length / 2);
			if constexpr (Algorithm == online_product_algorithm::cached && points_transform_coefficient<Coefficient>) {
				for (std::size_t block = 1; block <= right_.size() / 2; block *= 2) {
					if (block <= cutoff_ || !points_can_transform<Coefficient>(2 * block)) continue;
					std::size_t const level = std::size_t(std::countr_zero(block));
					if (spectra_.size() <= level) spectra_.resize(level + 1);
					spectra_[level].right = transform(std::span<Coefficient const>(right_).subspan(block, block));
				}
			}
		}

		/** @brief Append the next input(s) and return the Cauchy coefficient of index `size() - 1`; `right` is ignored for a fixed kernel.
		 * @pre `size() < maximum_series_length` (asserted).
		 */
		[[nodiscard]] constexpr Coefficient push(Coefficient left, Coefficient right = Coefficient(0)) {
			atl_assert(left_.size() < maximum_series_length);
			left_.push_back(left);
			std::size_t const count = left_.size(), degree = count - 1;
			if constexpr (IsFixed) {
				if (right_.size() < count) right_.resize(count);
			} else
				right_.push_back(right);
			if constexpr (Algorithm == online_product_algorithm::direct) {
				Coefficient result(0);
				for (std::size_t i = 0; i < count; ++i) result += left_[i] * right_[degree - i];
				return result;
			} else {
				if (output_.size() < count) output_.resize(count);
				output_[degree] += left_[degree] * right_[0];
				if (degree != 0) output_[degree] += left_[0] * right_[degree];
				Coefficient const result = output_[degree];
				for (std::size_t block = 1; block <= count / 2 && count % block == 0; block *= 2) schedule(block);
				return result;
			}
		}

		/**
		 * @brief Reserve the stream, kernel padding, pending outputs and spectrum index for `count` pushes in total.
		 *
		 * The pending outputs reach at most `min(2 count - 1, maximum_series_length)` entries: a push at count `t`
		 * schedules blocks of at most `t / 2`, whose outputs end below `2t - 1` and never past the cap.
		 * @pre `count <= maximum_series_length` (asserted).
		 */
		constexpr void reserve(std::size_t count) {
			atl_assert(count <= maximum_series_length);
			left_.reserve(count);
			right_.reserve((std::max)(right_.size(), count));
			if constexpr (Algorithm != online_product_algorithm::direct) {
				if (count != 0) output_.reserve((std::min)(2 * count - 1, maximum_series_length));
				spectra_.reserve(std::size_t(std::bit_width(count / 2)));
			}
		}

		/** @brief Number of pushes so far. */
		[[nodiscard]] constexpr std::size_t size() const { return left_.size(); }
	};
}
