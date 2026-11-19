#pragma once

#include <atl/nt/convolution.hpp>
#include <atl/nt/detail/polynomial_points.hpp>
#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace atl::detail {
	/** @brief Provisional tiny Horner cutoff; race pending. */
	inline constexpr std::size_t composition_schoolbook_cutoff = 8;
	/** @brief Provisional outer-degree crossover to transposition; race pending. */
	inline constexpr std::size_t composition_horner_cutoff = 16;
	/** @brief Provisional small-count projection crossover; race pending. */
	inline constexpr std::size_t composition_projection_cutoff = 8;
	/**
	 * @brief Working rectangles have only a constant-factor expansion over the public precision cap.
	 *
	 * Composition at precision n never needs more than 8n: at halving k the width is `ceil(n / 2^k)` and the
	 * numerator and denominator have at most `2^k` and `2^k + 1` rows, whatever the outer length (rows are also
	 * truncated to it), so every packed rectangle, product and fallback product is below `8n`. Transform sizes are
	 * rounded up only while they stay within `maximum_series_length`; larger products are blocked. Hence there is
	 * no joint bound on outer length times precision, and the public length cap keeps every level within this limit.
	 */
	inline constexpr std::size_t composition_work_limit = 32 * maximum_series_length;
	static_assert(maximum_series_length <= (std::numeric_limits<std::size_t>::max)() / 64);
	static_assert(8 * maximum_series_length <= composition_work_limit);

	/** @brief Low product for packed bivariate work, blocking before calling the capped FPS kernels.
	 * @pre Operand and requested lengths do not exceed composition_work_limit; positive block_length <= maximum_series_length/2.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_product(std::span<Coefficient const> left,
																		 std::span<Coefficient const> right, std::size_t length, std::size_t block_length = maximum_series_length / 2) {
		atl_assert(length <= composition_work_limit && left.size() <= composition_work_limit && right.size() <= composition_work_limit);
		atl_assert(block_length > 0 && block_length <= maximum_series_length / 2);
		left = left.first((std::min)(left.size(), length));
		right = right.first((std::min)(right.size(), length));
		if (length <= maximum_series_length && left.size() <= block_length && right.size() <= block_length)
			return series_product<Engine, Coefficient>(left, right, length);
		std::vector<Coefficient> result(length);
		auto const block = block_length;
		for (std::size_t i = 0; i < left.size(); i += block) {
			for (std::size_t j = 0; j < right.size() && j < length - i; j += block) {
				auto const left_block = left.subspan(i, (std::min)(block, left.size() - i));
				auto const right_block = right.subspan(j, (std::min)(block, right.size() - j));
				auto const size = (std::min)(length - i - j, left_block.size() + right_block.size() - 1);
				auto const product = series_product<Engine, Coefficient>(left_block, right_block, size);
				for (std::size_t k = 0; k < size; ++k) result[i + j + k] += product[k];
			}
		}
		return result;
	}

	/** @brief Insert an empty x-block between y-rows so univariate products cannot carry into the next row.
	 * @pre Positive width; a nonempty rectangle has an integral number of rows; packed length is within the work cap.
	 */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_pack(std::span<Coefficient const> rectangle, std::size_t width) {
		atl_assert(width > 0 && width <= maximum_series_length && rectangle.size() % width == 0);
		if (rectangle.empty()) return {};
		auto const rows = rectangle.size() / width;
		atl_assert(rows <= composition_work_limit / (2 * width));
		std::vector<Coefficient> result((2 * rows - 1) * width);
		for (std::size_t row = 0; row < rows; ++row)
			std::ranges::copy(rectangle.subspan(row * width, width), result.begin() + 2 * row * width);
		return result;
	}

	/** @brief Extract one x-parity from a packed product, retaining only the required rectangle.
	 * @pre Positive width and row count, parity 0 or 1, sufficient packed input, and bounded output rectangle.
	 */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_extract(std::span<Coefficient const> product,
																		 std::size_t width, std::size_t rows, std::size_t parity) {
		atl_assert(width > 0 && width <= maximum_series_length && parity <= 1 && rows > 0);
		auto const next = (width + 1) / 2;
		atl_assert(rows <= composition_work_limit / (2 * width));
		atl_assert(product.size() >= (rows - 1) * 2 * width + width);
		std::vector<Coefficient> result(next * rows);
		for (std::size_t row = 0; row < rows; ++row)
			for (std::size_t i = 0; i < next && 2 * i + parity < width; ++i)
				result[row * next + i] = product[row * 2 * width + 2 * i + parity];
		return result;
	}

	/** @brief Transform doubling when the coefficient support fits the lower half; otherwise a full transform.
	 * The first half of the bit-reversed spectrum evaluates at the even roots. A twisted half transform supplies the odd roots.
	 * @pre size is a supported power of two; input length <= size <= maximum_series_length.
	 */
	template<ntt_engine Engine, ntt_modulus Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_transform(std::span<Coefficient const> input, std::size_t size) {
		atl_assert(std::has_single_bit(size) && input.size() <= size && size <= maximum_series_length && (Coefficient::mod() - 1) % size == 0);
		std::vector<Coefficient> result(size);
		std::ranges::copy(input, result.begin());
		if (size >= 2 && input.size() <= size / 2) {
			auto const half = size / 2;
			Coefficient const root = primitive_root<Coefficient>().power((Coefficient::mod() - 1) / size);
			Coefficient power(1);
			for (std::size_t i = 0; i < half; ++i) {
				result[half + i] = result[i] * power;
				power *= root;
			}
			ntt<Engine>(std::span(result).first(half));
			ntt<Engine>(std::span(result).subspan(half));
		} else {
			ntt<Engine>(result);
		}
		return result;
	}

	/** @brief One Graeffe step and its operation-local denominator spectrum, reusable by transposition. */
	template<static_prime_mod_int Coefficient>
	struct composition_graeffe_result {
		/** @brief Selected-parity numerator rectangle; empty when only preparing a denominator. */
		std::vector<Coefficient> numerator;
		/** @brief Even denominator rectangle. */
		std::vector<Coefficient> denominator;
		/** @brief Full bit-reversed transform of the incoming packed denominator, or empty on fallback. */
		std::vector<Coefficient> spectrum;
	};

	/** @brief Graeffe halving of x and doubling of y, after Kinoshita–Li (2024), Algorithm 1, implemented from the rational identity.
	 * At roots w,-w, Q(-w) is an adjacent spectral entry. Q(w)Q(-w) and the
	 * selected numerator parity need only half-size inverse transforms. The fallback path divides by no two.
	 * @pre width > 1; rectangles have width columns; denominator has one more row than numerator unless y-truncated;
	 * count is positive and at most maximum_series_length; packed work fits composition_work_limit.
	 */
	template<ntt_engine Engine, bool IsOptimized, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr composition_graeffe_result<Coefficient> composition_graeffe(
		std::span<Coefficient const> numerator, std::span<Coefficient const> denominator, std::size_t width, std::size_t count) {
		atl_assert(width > 1 && width <= maximum_series_length && count > 0 && count <= maximum_series_length);
		atl_assert(!denominator.empty() && denominator.size() % width == 0 && numerator.size() % width == 0);
		auto packed_denominator = composition_pack<Coefficient>(denominator, width);
		auto packed_numerator = composition_pack<Coefficient>(numerator, width);
		atl_assert(packed_denominator.size() <= composition_work_limit / 2 && packed_numerator.size() <= packed_denominator.size());
		auto const denominator_rows = (std::min)(count, 2 * (denominator.size() / width) - 1);
		auto const numerator_rows = numerator.empty() ? 0 : (std::min)(count, numerator.size() / width + denominator.size() / width - 1);
		auto const size = std::bit_ceil(2 * packed_denominator.size() - 1);
		composition_graeffe_result<Coefficient> result;
		if constexpr (IsOptimized && points_transform_coefficient<Coefficient>) {
			if (size <= maximum_series_length && points_can_transform<Coefficient>(size)) {
				packed_denominator = composition_transform<Engine, Coefficient>(packed_denominator, size);
				std::vector<Coefficient> even(size / 2), selected;
				for (std::size_t i = 0; i < size / 2; ++i) even[i] = packed_denominator[2 * i] * packed_denominator[2 * i + 1];
				inverse_ntt<Engine>(even);
				// Extraction has already halved all packed exponents, including the y stride.
				auto const next = (width + 1) / 2;
				result.denominator.resize(next * denominator_rows);
				for (std::size_t row = 0; row < denominator_rows; ++row)
					for (std::size_t i = 0; i < next; ++i) result.denominator[row * next + i] = even[row * width + i];
				if (!numerator.empty()) {
					packed_numerator = composition_transform<Engine, Coefficient>(packed_numerator, size);
					selected.resize(size / 2);
					bool const is_odd = (width - 1) % 2 != 0;
					Coefficient const half = Coefficient(1) / Coefficient(2);
					std::vector<Coefficient> odd_scale(size / 2, half);
					if (is_odd) {
						auto const inverse_root = primitive_root<Coefficient>().power((Coefficient::mod() - 1) - (Coefficient::mod() - 1) / size);
						for (std::size_t block = 1; block < size / 2; block *= 2) {
							auto const factor = inverse_root.power(size / (4 * block));
							for (std::size_t i = 0; i < block; ++i) odd_scale[block + i] = odd_scale[i] * factor;
						}
					}
					for (std::size_t i = 0; i < size / 2; ++i) {
						auto const at_positive_root = packed_numerator[2 * i] * packed_denominator[2 * i + 1];
						auto const at_negative_root = packed_numerator[2 * i + 1] * packed_denominator[2 * i];
						selected[i] = is_odd ? (at_positive_root - at_negative_root) * odd_scale[i] : (at_positive_root + at_negative_root) * half;
					}
					inverse_ntt<Engine>(selected);
					result.numerator.resize(next * numerator_rows);
					for (std::size_t row = 0; row < numerator_rows; ++row)
						for (std::size_t i = 0; i < next && 2 * i + (width - 1) % 2 < width; ++i)
							result.numerator[row * next + i] = selected[row * width + i];
				}
				result.spectrum = std::move(packed_denominator);
				return result;
			}
		}
		auto reflected = packed_denominator;
		for (std::size_t i = 1; i < reflected.size(); i += 2) reflected[i] = -reflected[i];
		auto const denominator_product = composition_product<Engine, Coefficient>(packed_denominator, reflected,
																				  (denominator_rows - 1) * 2 * width + width);
		result.denominator = composition_extract<Coefficient>(denominator_product, width, denominator_rows, 0);
		if (!numerator.empty()) {
			auto const numerator_product = composition_product<Engine, Coefficient>(packed_numerator, reflected,
																					(numerator_rows - 1) * 2 * width + width);
			result.numerator = composition_extract<Coefficient>(numerator_product, width, numerator_rows, (width - 1) % 2);
		}
		return result;
	}

	/** @brief Initial denominator rectangle 1-yG, truncated in y before allocation.
	 * @pre series is nonempty, its length and positive count are at most maximum_series_length.
	 */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_denominator(std::span<Coefficient const> series, std::size_t count) {
		atl_assert(!series.empty() && series.size() <= maximum_series_length && count > 0 && count <= maximum_series_length);
		std::vector<Coefficient> result(series.size() * (std::min)(count, std::size_t{2}));
		result[0] = Coefficient(1);
		if (count > 1)
			for (std::size_t i = 0; i < series.size(); ++i) result[series.size() + i] = -series[i];
		return result;
	}

	/** @brief Bivariate Bostan–Mori projection, with an optimized transform schedule or plain padded products; race pending.
	 * @pre Equal weight/series lengths at most maximum_series_length; count at most that cap.
	 */
	template<ntt_engine Engine, bool IsOptimized, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_project(std::span<Coefficient const> weights,
																		 std::span<Coefficient const> series, std::size_t count) {
		atl_assert(weights.size() == series.size() && weights.size() <= maximum_series_length && count <= maximum_series_length);
		if (weights.empty() || count == 0) return std::vector<Coefficient>(count);
		auto denominator = composition_denominator<Coefficient>(series, count);
		std::vector<Coefficient> numerator(weights.rbegin(), weights.rend());
		for (std::size_t width = weights.size(); width > 1; width = (width + 1) / 2) {
			auto step = composition_graeffe<Engine, IsOptimized, Coefficient>(numerator, denominator, width, count);
			numerator = std::move(step.numerator);
			denominator = std::move(step.denominator);
		}
		auto const inverse = series_inverse_newton<Engine, Coefficient>(denominator, count);
		return series_product<Engine, Coefficient>(numerator, inverse, count);
	}

	/** @brief Repeated truncated multiplication and dot products for small counts; race pending.
	 * @pre Equal weight/series lengths at most maximum_series_length; count at most that cap.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_project_repeated(std::span<Coefficient const> weights,
																				  std::span<Coefficient const> series, std::size_t count) {
		atl_assert(weights.size() == series.size() && weights.size() <= maximum_series_length && count <= maximum_series_length);
		std::vector<Coefficient> result(count), power(weights.size());
		if (weights.empty()) return result;
		power[0] = Coefficient(1);
		for (std::size_t i = 0; i < count; ++i) {
			for (std::size_t j = 0; j < weights.size(); ++j) result[i] += weights[j] * power[j];
			if (i + 1 < count) power = series_product<Engine, Coefficient>(power, series, weights.size());
		}
		return result;
	}

	/** @brief Reverse the linear numerator steps while keeping the denominator fixed: the transpose of projection.
	 * @pre width and count positive and capped; numerator_rows is positive and <= count;
	 * denominator is the width-column Graeffe rectangle and outer has count entries.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_transpose(std::span<Coefficient const> outer,
																		   std::span<Coefficient const> denominator, std::size_t width, std::size_t numerator_rows) {
		auto const count = outer.size();
		atl_assert(width > 0 && width <= maximum_series_length && count > 0 && count <= maximum_series_length);
		atl_assert(numerator_rows > 0 && numerator_rows <= count && !denominator.empty() && denominator.size() % width == 0);
		if (width == 1) {
			auto const inverse = series_inverse_newton<Engine, Coefficient>(denominator, count);
			std::vector<Coefficient> reversed(outer.rbegin(), outer.rend());
			auto const product = series_product<Engine, Coefficient>(reversed, inverse, count);
			std::vector<Coefficient> result(numerator_rows);
			for (std::size_t i = 0; i < numerator_rows; ++i) result[i] = product[count - 1 - i];
			return result;
		}
		auto step = composition_graeffe<Engine, true, Coefficient>({}, denominator, width, count);
		auto const next_rows = (std::min)(count, numerator_rows + denominator.size() / width - 1);
		auto const adjoint = composition_transpose<Engine, Coefficient>(outer, step.denominator, (width + 1) / 2, next_rows);
		auto reflected = composition_pack<Coefficient>(denominator, width);
		for (std::size_t i = 1; i < reflected.size(); i += 2) reflected[i] = -reflected[i];
		atl_assert(numerator_rows <= composition_work_limit / (2 * width));
		auto const input_length = (2 * numerator_rows - 1) * width;
		auto const full_length = input_length + reflected.size() - 1;
		atl_assert(full_length <= composition_work_limit);
		std::vector<Coefficient> reversed(full_length);
		auto const next = (width + 1) / 2, parity = (width - 1) % 2;
		for (std::size_t row = 0; row < next_rows; ++row)
			for (std::size_t i = 0; i < next && 2 * i + parity < width; ++i)
				reversed[full_length - 1 - (row * 2 * width + 2 * i + parity)] = adjoint[row * next + i];
		// The cached-spectrum path multiplies in place, so `reversed` then holds the full reversed product.
		bool is_transformed = false;
		if constexpr (points_transform_coefficient<Coefficient>) {
			if (!step.spectrum.empty()) {
				atl_assert(full_length <= step.spectrum.size());
				reversed.resize(step.spectrum.size());
				ntt<Engine>(reversed);
				for (std::size_t i = 0; i < reversed.size(); ++i) reversed[i] *= step.spectrum[i ^ 1];
				inverse_ntt<Engine>(reversed);
				is_transformed = true;
			}
		}
		std::vector<Coefficient> product, middle;
		if (!is_transformed) {
			if (full_length <= maximum_series_length) {
				middle = points_middle_product<Engine, Coefficient>(reversed, reflected);
				std::ranges::reverse(middle);
			} else {
				product = composition_product<Engine, Coefficient>(reversed, reflected, full_length);
			}
		}
		auto const &full = is_transformed ? reversed : product;
		std::vector<Coefficient> result(numerator_rows * width);
		for (std::size_t row = 0; row < numerator_rows; ++row)
			for (std::size_t i = 0; i < width; ++i) {
				auto const position = row * 2 * width + i;
				result[row * width + i] = middle.empty() ? full[full_length - 1 - position] : middle[position];
			}
		return result;
	}

	/** @brief Kinoshita–Li composition obtained by transposing the projection map in its weights; race pending.
	 * @pre Positive length, nonempty outer, and zero-constant inner; all lengths capped.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_kinoshita_li(std::span<Coefficient const> outer,
																			  std::span<Coefficient const> inner) {
		atl_assert(!outer.empty() && !inner.empty() && outer.size() <= maximum_series_length && inner.size() <= maximum_series_length);
		atl_assert(inner[0] == Coefficient(0));
		auto const denominator = composition_denominator<Coefficient>(inner, outer.size());
		auto result = composition_transpose<Engine, Coefficient>(outer, denominator, inner.size(), 1);
		std::ranges::reverse(result);
		return result;
	}

	/** @brief Horner composition with either fast or schoolbook truncated multiplication; race pending.
	 * @pre Output, outer and inner lengths are capped; inner has length entries and constant zero.
	 */
	template<ntt_engine Engine, bool IsSchoolbook, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_horner(std::span<Coefficient const> outer,
																		std::span<Coefficient const> inner, std::size_t length) {
		atl_assert(length <= maximum_series_length && outer.size() <= maximum_series_length && inner.size() == length);
		atl_assert(length == 0 || inner[0] == Coefficient(0));
		std::vector<Coefficient> result(length);
		if (length == 0) return result;
		for (std::size_t remaining = outer.size(); remaining > 0; --remaining) {
			if constexpr (IsSchoolbook) {
				std::vector<Coefficient> product(length);
				for (std::size_t i = 0; i < length; ++i)
					for (std::size_t j = 1; j < length - i; ++j) product[i + j] += result[i] * inner[j];
				result = std::move(product);
			} else {
				result = series_product<Engine, Coefficient>(result, inner, length);
			}
			result[0] += outer[remaining - 1];
		}
		return result;
	}

	/** @brief Ceiling square root for baby-step/giant-step block counts without floating-point narrowing.
	 * @pre length <= maximum_series_length.
	 */
	[[nodiscard]] constexpr std::size_t composition_block_size(std::size_t length) {
		atl_assert(length <= maximum_series_length);
		std::size_t block = 1;
		while (block * block < length) {
			atl_assert(block < maximum_series_length && block < (std::numeric_limits<std::size_t>::max)() / (block + 1));
			++block;
		}
		return block;
	}

	/** @brief Brent–Kung (1978), first algorithm, with a classical rectangular matrix product; race pending.
	 * @pre Positive precision, nonempty outer and zero-constant inner; lengths capped.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> composition_brent_kung(std::span<Coefficient const> outer,
																			std::span<Coefficient const> inner) {
		atl_assert(!outer.empty() && !inner.empty() && outer.size() <= maximum_series_length && inner.size() <= maximum_series_length);
		atl_assert(inner[0] == Coefficient(0));
		auto const length = inner.size(), block = composition_block_size(outer.size());
		std::vector<std::vector<Coefficient>> baby(block + 1, std::vector<Coefficient>(length));
		baby[0][0] = Coefficient(1);
		for (std::size_t i = 1; i <= block; ++i) baby[i] = series_product<Engine, Coefficient>(baby[i - 1], inner, length);
		auto const rows = (outer.size() + block - 1) / block;
		std::vector<std::vector<Coefficient>> matrix(rows, std::vector<Coefficient>(length));
		for (std::size_t row = 0; row < rows; ++row)
			for (std::size_t i = 0; i < block && row * block + i < outer.size(); ++i)
				for (std::size_t j = i; j < length; ++j) matrix[row][j] += outer[row * block + i] * baby[i][j];
		std::vector<Coefficient> result(length);
		for (std::size_t row = rows; row > 0; --row) {
			if (row != rows) result = series_product<Engine, Coefficient>(result, baby[block], length);
			for (std::size_t j = 0; j < length; ++j) result[j] += matrix[row - 1][j];
		}
		return result;
	}
}
