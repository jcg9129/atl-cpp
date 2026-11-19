#pragma once

#include <atl/nt/batch_inverse.hpp>
#include <atl/nt/convolution.hpp>
#include <atl/nt/detail/power_series.hpp>
#include <atl/nt/detail/power_series_extensions.hpp>
#include <atl/nt/factorial_table.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/power_series.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/types/traits/widen_mul.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace atl::detail {
	/** @brief Provisional schoolbook leaf size for point operations; race pending. */
	inline constexpr std::size_t polynomial_points_cutoff = 32;

	/**
	 * @brief Copy a whole input range, keeping its length (trailing zeros are not normalized away).
	 * @pre The length is at most maximum_series_length (asserted).
	 */
	template<prime_mod_int_range Range>
	[[nodiscard]] constexpr auto points_copy(Range const &input) -> std::vector<series_coefficient_t<Range>> {
		std::vector<series_coefficient_t<Range>> result;
		for (auto const &value : input) {
			atl_assert(result.size() < maximum_series_length);
			result.push_back(value);
		}
		return result;
	}

	/** @brief A field/storage pair supported by the cyclic kernels without double-width software emulation. */
	template<typename Coefficient>
	concept points_transform_coefficient = ntt_modulus<Coefficient> && traits::has_widen_mul_v<decltype(Coefficient::mod())>;

	/** @brief Whether the selected field admits a cyclic transform of this length.
	 * @pre length is a power of two at most maximum_series_length.
	 */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr bool points_can_transform(std::size_t length) {
		atl_assert(std::has_single_bit(length) && length <= maximum_series_length);
		if constexpr (points_transform_coefficient<Coefficient>)
			return (Coefficient::mod() - 1) % length == 0;
		else
			return false;
	}

	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr Coefficient points_horner(std::span<Coefficient const> polynomial, Coefficient point) {
		Coefficient value(0);
		for (std::size_t i = polynomial.size(); i > 0; --i) value = value * point + polynomial[i - 1];
		return value;
	}

	/**
	 * @brief Product coefficients [right.size()-1, left.size()), without padding to the full product length.
	 * A cyclic transform of bit_ceil(left.size()) can alias only below the requested interval.
	 * Other fields split the long operand into two halves and use the existing exact product tiers.
	 * @pre 1 <= right.size() <= left.size() <= maximum_series_length.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> points_middle_product(std::span<Coefficient const> left, std::span<Coefficient const> right) {
		atl_assert(!right.empty() && right.size() <= left.size() && left.size() <= maximum_series_length);
		std::size_t const start = right.size() - 1, count = left.size() - start;
		if ((std::min)(count, right.size()) <= polynomial_points_cutoff) {
			std::vector<Coefficient> result(count);
			for (std::size_t i = 0; i < count; ++i)
				for (std::size_t j = 0; j < right.size(); ++j) result[i] += left[start + i - j] * right[j];
			return result;
		}
		std::size_t const size = std::bit_ceil(left.size());
		if constexpr (points_transform_coefficient<Coefficient>) {
			if (points_can_transform<Coefficient>(size)) {
				std::vector<Coefficient> transformed_left(size), transformed_right(size);
				std::ranges::copy(left, transformed_left.begin());
				std::ranges::copy(right, transformed_right.begin());
				ntt<Engine>(transformed_left);
				ntt<Engine>(transformed_right);
				for (std::size_t i = 0; i < size; ++i) transformed_left[i] *= transformed_right[i];
				inverse_ntt<Engine>(transformed_left);
				return std::vector<Coefficient>(transformed_left.begin() + start, transformed_left.begin() + left.size());
			}
		}
		// Splitting both operands bounds each ordinary transform by bit_ceil(left.size()).
		std::size_t const block = size / 2;
		std::vector<Coefficient> result(count);
		for (std::size_t left_offset = 0; left_offset < left.size(); left_offset += block) {
			for (std::size_t right_offset = 0; right_offset < right.size(); right_offset += block) {
				auto const part_left = left.subspan(left_offset, (std::min)(block, left.size() - left_offset));
				auto const part_right = right.subspan(right_offset, (std::min)(block, right.size() - right_offset));
				std::size_t const offset = left_offset + right_offset, length = part_left.size() + part_right.size() - 1;
				if (offset >= left.size() || offset + length <= start) continue;
				auto const product = series_product<Engine, Coefficient>(part_left, part_right, length);
				for (std::size_t degree = (std::max)(start, offset); degree < (std::min)(left.size(), offset + length); ++degree)
					result[degree - start] += product[degree - offset];
			}
		}
		return result;
	}

	/** @brief Quadratic shift by repeated multiplication by x + shift; works in every characteristic. */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> points_shift_horner(std::span<Coefficient const> polynomial, Coefficient shift) {
		atl_assert(polynomial.size() <= maximum_series_length);
		std::vector<Coefficient> result(polynomial.size());
		for (std::size_t remaining = polynomial.size(); remaining > 0; --remaining) {
			std::size_t const degree = polynomial.size() - remaining;
			for (std::size_t i = degree; i > 0; --i) result[i] = result[i - 1] + shift * result[i];
			result[0] = result[0] * shift + polynomial[remaining - 1];
		}
		return result;
	}

	/** @brief Divide-and-conquer shift, combining shifted halves with (x + shift)^middle; race pending. */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> points_shift_divide(std::span<Coefficient const> polynomial, Coefficient shift) {
		atl_assert(polynomial.size() <= maximum_series_length);
		if (polynomial.size() <= polynomial_points_cutoff) return points_shift_horner<Coefficient>(polynomial, shift);
		std::size_t const middle = polynomial.size() / 2;
		auto lower = points_shift_divide<Engine, Coefficient>(polynomial.first(middle), shift);
		auto upper = points_shift_divide<Engine, Coefficient>(polynomial.subspan(middle), shift);
		std::vector<Coefficient> power{Coefficient(1)}, base{shift, Coefficient(1)};
		std::size_t exponent = middle;
		while (exponent != 0) {
			if (exponent % 2 != 0) power = series_product<Engine, Coefficient>(power, base, power.size() + base.size() - 1);
			exponent /= 2;
			if (exponent != 0) base = series_product<Engine, Coefficient>(base, base, 2 * base.size() - 1);
		}
		auto result = series_product<Engine, Coefficient>(upper, power, polynomial.size());
		for (std::size_t i = 0; i < lower.size(); ++i) result[i] += lower[i];
		return result;
	}

	/** @brief Factorial-scaled convolution shift, with a characteristic-independent fallback; race pending. */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> points_shift_factorial(std::span<Coefficient const> polynomial, Coefficient shift) {
		atl_assert(polynomial.size() <= maximum_series_length);
		std::size_t const length = polynomial.size();
		if (length > Coefficient::mod()) return points_shift_divide<Engine, Coefficient>(polynomial, shift);
		if (length == 0) return {};
		factorial_table<Coefficient> const table(length - 1);
		std::vector<Coefficient> left(length), right(length);
		Coefficient power(1);
		for (std::size_t i = 0; i < length; ++i) {
			left[length - 1 - i] = polynomial[i] * table.factorial(i);
			right[i] = power * table.inverse_factorial(i);
			power *= shift;
		}
		auto const product = series_product<Engine, Coefficient>(left, right, length);
		for (std::size_t i = 0; i < length; ++i) left[i] = product[length - 1 - i] * table.inverse_factorial(i);
		return left;
	}

	/** @brief Quadratic Newton divided differences and nested conversion, also the small interpolation base case.
	 * @pre Equal lengths <= maximum_series_length; points are distinct.
	 */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> points_interpolation_newton(std::span<Coefficient const> nodes, std::span<Coefficient const> values) {
		atl_assert(nodes.size() <= maximum_series_length && nodes.size() == values.size());
		std::vector<Coefficient> differences(values.begin(), values.end());
		std::size_t const count = nodes.size();
		for (std::size_t width = 1; width < count; ++width) {
			for (std::size_t i = count; i > width; --i) {
				Coefficient const denominator = nodes[i - 1] - nodes[i - 1 - width];
				atl_assert(denominator != Coefficient(0));
				differences[i - 1] = (differences[i - 1] - differences[i - 2]) / denominator;
			}
		}
		std::vector<Coefficient> result(count);
		for (std::size_t remaining = count; remaining > 0; --remaining) {
			for (std::size_t i = count - remaining; i > 0; --i) result[i] = result[i - 1] - nodes[remaining - 1] * result[i];
			result[0] = differences[remaining - 1] - nodes[remaining - 1] * result[0];
		}
		return result;
	}

	/** @brief Schoolbook product of linear root factors, retaining its monic top coefficient. */
	template<static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> points_roots_sequential(std::span<Coefficient const> roots) {
		atl_assert(roots.size() < maximum_series_length);
		std::vector<Coefficient> result(roots.size() + 1);
		result[0] = Coefficient(1);
		std::size_t degree = 0;
		for (auto root : roots) {
			result[degree + 1] = result[degree];
			for (std::size_t i = degree; i > 0; --i) result[i] = result[i - 1] - root * result[i];
			result[0] *= -root;
			++degree;
		}
		return result;
	}

	/** @brief Balanced multiplication with schoolbook root leaves; race pending. */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> points_roots_balanced(std::span<Coefficient const> roots) {
		atl_assert(roots.size() < maximum_series_length);
		if (roots.size() <= polynomial_points_cutoff) return points_roots_sequential<Coefficient>(roots);
		auto const left = points_roots_balanced<Engine, Coefficient>(roots.first(roots.size() / 2));
		auto const right = points_roots_balanced<Engine, Coefficient>(roots.subspan(roots.size() / 2));
		return series_product<Engine, Coefficient>(left, right, roots.size() + 1);
	}

	/** @brief A const range of prime-field polynomial ranges with one coefficient type. */
	template<typename Polynomials>
	concept points_polynomial_range = std::ranges::input_range<Polynomials const>
								   && prime_mod_int_range<std::ranges::range_value_t<Polynomials const>>;

	/** @brief Consume and normalize factors, checking the total nonzero product length. */
	template<points_polynomial_range Polynomials>
	[[nodiscard]] constexpr auto points_factors(Polynomials const &polynomials) {
		using Coefficient = series_coefficient_t<std::ranges::range_value_t<Polynomials const>>;
		std::vector<std::vector<Coefficient>> factors;
		bool has_zero = false;
		for (auto const &polynomial : polynomials) {
			auto factor = points_copy(polynomial);
			series_normalize_polynomial(factor);
			has_zero = has_zero || factor.empty();
			factors.push_back(std::move(factor));
		}
		if (has_zero) return std::vector<std::vector<Coefficient>>{{}};
		std::size_t degree = 0;
		for (auto const &factor : factors) {
			atl_assert(factor.size() - 1 < maximum_series_length - degree);
			degree += factor.size() - 1;
		}
		return factors;
	}

	/** @brief Balanced tree product of normalized factors; race pending. */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> points_product_balanced(std::span<std::vector<Coefficient> const> factors) {
		if (factors.empty()) return {Coefficient(1)};
		if (factors.size() == 1) return factors.front();
		auto const left = points_product_balanced<Engine, Coefficient>(factors.first(factors.size() / 2));
		auto const right = points_product_balanced<Engine, Coefficient>(factors.subspan(factors.size() / 2));
		if (left.empty() || right.empty()) return {};
		atl_assert(left.size() <= maximum_series_length && right.size() <= maximum_series_length - left.size() + 1);
		return series_product<Engine, Coefficient>(left, right, left.size() + right.size() - 1);
	}

	/** @brief Chirp-z evaluation via triangular exponents and just the middle product; race pending.
	 * @pre Input and output lengths <= maximum_series_length; 2 <= working_length_limit <= maximum_series_length.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> points_geometric(std::span<Coefficient const> polynomial, Coefficient first, Coefficient ratio, std::size_t count, std::size_t working_length_limit = maximum_series_length) {
		atl_assert(working_length_limit >= 2 && working_length_limit <= maximum_series_length);
		atl_assert(polynomial.size() <= maximum_series_length && count <= maximum_series_length);
		std::vector<Coefficient> result(count);
		if (count == 0 || polynomial.empty()) return result;
		if (first == Coefficient(0)) {
			std::ranges::fill(result, polynomial[0]);
			return result;
		}
		if (ratio == Coefficient(0)) {
			std::ranges::fill(result, polynomial[0]);
			result[0] = points_horner<Coefficient>(polynomial, first);
			return result;
		}
		// Blocking preserves the public cap even when n + count - 1 exceeds it.
		std::size_t const block = (std::min)(polynomial.size(), working_length_limit / 2);
		Coefficient const inverse_ratio = Coefficient(1) / ratio;
		for (std::size_t offset = 0; offset < polynomial.size(); offset += block) {
			std::size_t const width = (std::min)(block, polynomial.size() - offset);
			std::size_t const output_block = working_length_limit - width + 1;
			Coefficient target = first;
			for (std::size_t start = 0; start < count; start += output_block) {
				std::size_t const outputs = (std::min)(output_block, count - start), length = width + outputs - 1;
				std::vector<Coefficient> left(length), right(width), inverse_chirp((std::max)(width, outputs));
				Coefficient chirp(1), step(1), inverse(1), inverse_step(1), power(1);
				for (std::size_t i = 0; i < length; ++i) {
					left[i] = chirp;
					chirp *= step;
					step *= ratio;
					if (i < inverse_chirp.size()) {
						inverse_chirp[i] = inverse;
						inverse *= inverse_step;
						inverse_step *= inverse_ratio;
					}
					if (i < width) {
						right[width - 1 - i] = polynomial[offset + i] * power * inverse_chirp[i];
						power *= target;
					}
				}
				auto const middle = points_middle_product<Engine, Coefficient>(left, right);
				Coefficient scale = target.power(offset), scale_step = ratio.power(offset);
				for (std::size_t j = 0; j < outputs; ++j) {
					result[start + j] += middle[j] * inverse_chirp[j] * scale;
					scale *= scale_step;
				}
				target *= ratio.power(outputs);
			}
		}
		return result;
	}

	/**
	 * @brief Factorial-weighted barycentric shift using middle products and one batch inversion; race pending.
	 * Zero denominators are replaced before inversion and their outputs patched from the source samples.
	 * @pre samples.size() <= p; input and output lengths <= maximum_series_length;
	 * 2 <= working_length_limit <= maximum_series_length.
	 */
	template<ntt_engine Engine, static_prime_mod_int Coefficient>
	[[nodiscard]] constexpr std::vector<Coefficient> points_sample_middle(std::span<Coefficient const> samples, Coefficient shift, std::size_t count, std::size_t working_length_limit = maximum_series_length) {
		atl_assert(working_length_limit >= 2 && working_length_limit <= maximum_series_length);
		atl_assert(samples.size() <= Coefficient::mod());
		atl_assert(samples.size() <= maximum_series_length && count <= maximum_series_length);
		std::vector<Coefficient> result(count);
		std::size_t const length = samples.size();
		if (length == 0 || count == 0) return result;
		// A whole field of source nodes leaves no non-node targets.
		if (length == Coefficient::mod()) {
			for (std::size_t j = 0; j < count; ++j) result[j] = samples[std::size_t((shift + Coefficient(j)).value())];
			return result;
		}
		factorial_table<Coefficient> const table(length - 1);
		std::vector<Coefficient> weights(length);
		for (std::size_t i = 0; i < length; ++i) {
			weights[i] = samples[i] * table.inverse_factorial(i) * table.inverse_factorial(length - 1 - i);
			if ((length - 1 - i) % 2 != 0) weights[i] = -weights[i];
		}
		std::vector<Coefficient> denominators(length + count - 1);
		for (std::size_t i = 0; i < denominators.size(); ++i) {
			denominators[i] = shift + Coefficient(i) - Coefficient(length - 1);
			if (denominators[i] == Coefficient(0)) denominators[i] = Coefficient(1);
		}
		auto const inverses = batch_inverse(denominators);
		// Blocking both axes avoids tiny output blocks when the sample count reaches the series cap.
		std::size_t const block = (std::min)(length, working_length_limit / 2);
		for (std::size_t offset = 0; offset < length; offset += block) {
			std::size_t const width = (std::min)(block, length - offset);
			std::size_t const output_block = working_length_limit - width + 1;
			for (std::size_t start = 0; start < count; start += output_block) {
				std::size_t const outputs = (std::min)(output_block, count - start);
				auto const window = std::span(inverses).subspan(length - offset - width + start, width + outputs - 1);
				auto const middle = points_middle_product<Engine, Coefficient>(window, std::span(weights).subspan(offset, width));
				for (std::size_t j = 0; j < outputs; ++j) result[start + j] += middle[j];
			}
		}
		// Patched factors are units, so inverse prefixes recover every non-colliding sliding product.
		std::vector<Coefficient> prefixes(denominators.size() + 1, Coefficient(1));
		for (std::size_t i = 0; i < denominators.size(); ++i) prefixes[i + 1] = prefixes[i] * denominators[i];
		Coefficient inverse_prefix(1);
		for (std::size_t j = 0; j < count; ++j) {
			Coefficient const point = shift + Coefficient(j);
			if (point.value() < length)
				result[j] = samples[std::size_t(point.value())];
			else
				result[j] *= prefixes[j + length] * inverse_prefix;
			inverse_prefix *= inverses[j];
		}
		return result;
	}

	/** @brief Fixed point products and per-node reversed child spectra, shared by both tree directions. */
	template<static_prime_mod_int Coefficient, ntt_engine Engine>
	class points_tree_storage {
		std::vector<Coefficient> points_;
		/** @brief Reversed monic node polynomials; a node of `k` points has `k + 1` coefficients, the constant term 1 first. */
		std::vector<std::vector<Coefficient>> products_;
		/** @brief Left child spectra at the parent's cyclic length; empty where the field has no transform of that length. */
		std::vector<std::vector<Coefficient>> left_spectra_;
		/** @brief Right child spectra at the parent's cyclic length; empty exactly when `left_spectra_` is. */
		std::vector<std::vector<Coefficient>> right_spectra_;
		std::size_t cutoff_;

		constexpr void build(std::size_t node, std::size_t begin, std::size_t end) {
			std::size_t const count = end - begin;
			if (count <= cutoff_) {
				products_[node] = points_roots_sequential<Coefficient>(std::span(points_).subspan(begin, count));
				std::ranges::reverse(products_[node]);
				return;
			}
			std::size_t const middle = begin + count / 2;
			build(2 * node, begin, middle);
			build(2 * node + 1, middle, end);
			auto const &left = products_[2 * node];
			auto const &right = products_[2 * node + 1];
			if constexpr (points_transform_coefficient<Coefficient>) {
				std::size_t const size = std::bit_ceil(count);
				if (points_can_transform<Coefficient>(size)) {
					left_spectra_[node].resize(size);
					right_spectra_[node].resize(size);
					std::ranges::copy(left, left_spectra_[node].begin());
					std::ranges::copy(right, right_spectra_[node].begin());
					ntt<Engine>(left_spectra_[node]);
					ntt<Engine>(right_spectra_[node]);
					auto product = left_spectra_[node];
					for (std::size_t i = 0; i < size; ++i) product[i] *= right_spectra_[node][i];
					inverse_ntt<Engine>(product);
					product.resize(count + 1);
					// At a full cyclic length only the top coefficient wraps, into the known constant one.
					product[0] = Coefficient(1);
					product[count] = left.back() * right.back();
					products_[node] = std::move(product);
					return;
				}
			}
			products_[node] = series_product<Engine, Coefficient>(left, right, (std::min)(count + 1, maximum_series_length));
			if (count == maximum_series_length) products_[node].push_back(left.back() * right.back());
		}

		/** @brief Transposed descent: the sibling reversed polynomial selects the child's middle interval. */
		constexpr void descend(std::size_t node, std::size_t begin, std::size_t end, std::vector<Coefficient> state, std::span<Coefficient> values) const {
			std::size_t const count = end - begin;
			if (count <= cutoff_) {
				auto remainder = series_product<Engine, Coefficient>(state, std::span(products_[node]).first(count), count);
				std::ranges::reverse(remainder);
				for (std::size_t i = begin; i < end; ++i) values[i] = points_horner<Coefficient>(remainder, points_[i]);
				return;
			}
			std::size_t const middle = begin + count / 2, left_count = middle - begin, right_count = end - middle;
			std::vector<Coefficient> left, right;
			if constexpr (points_transform_coefficient<Coefficient>) {
				if (!left_spectra_[node].empty()) {
					std::size_t const size = left_spectra_[node].size();
					state.resize(size);
					ntt<Engine>(state);
					auto spectrum = state;
					for (std::size_t i = 0; i < size; ++i) spectrum[i] *= right_spectra_[node][i];
					inverse_ntt<Engine>(spectrum);
					left.assign(spectrum.begin() + right_count, spectrum.begin() + count);
					for (std::size_t i = 0; i < size; ++i) state[i] *= left_spectra_[node][i];
					inverse_ntt<Engine>(state);
					right.assign(state.begin() + left_count, state.begin() + count);
				} else {
					left = points_middle_product<Engine, Coefficient>(state, products_[2 * node + 1]);
					right = points_middle_product<Engine, Coefficient>(state, products_[2 * node]);
				}
			} else {
				left = points_middle_product<Engine, Coefficient>(state, products_[2 * node + 1]);
				right = points_middle_product<Engine, Coefficient>(state, products_[2 * node]);
			}
			descend(2 * node, begin, middle, std::move(left), values);
			descend(2 * node + 1, middle, end, std::move(right), values);
		}

		constexpr void descend_remainders(std::size_t node, std::size_t begin, std::size_t end, std::vector<Coefficient> polynomial, std::span<Coefficient> values) const {
			if (end - begin <= cutoff_) {
				for (std::size_t i = begin; i < end; ++i) values[i] = points_horner<Coefficient>(polynomial, points_[i]);
				return;
			}
			std::size_t const middle = begin + (end - begin) / 2;
			auto left_divisor = products_[2 * node], right_divisor = products_[2 * node + 1];
			std::ranges::reverse(left_divisor);
			std::ranges::reverse(right_divisor);
			auto left = polynomial_division<Engine>(polynomial, left_divisor).remainder;
			auto right = polynomial_division<Engine>(polynomial, right_divisor).remainder;
			descend_remainders(2 * node, begin, middle, std::move(left), values);
			descend_remainders(2 * node + 1, middle, end, std::move(right), values);
		}

		/** @brief Upward weighted combination in reversed coordinates, reusing the descent spectra. */
		[[nodiscard]] constexpr std::vector<Coefficient> ascend(std::size_t node, std::size_t begin, std::size_t end, std::span<Coefficient const> weights) const {
			std::size_t const count = end - begin;
			if (count <= cutoff_) {
				std::vector<Coefficient> result(count);
				for (std::size_t i = begin; i < end; ++i) {
					Coefficient quotient(1);
					for (std::size_t j = 0; j < count; ++j) {
						result[j] += weights[i] * quotient;
						if (j + 1 < count) quotient = products_[node][j + 1] + points_[i] * quotient;
					}
				}
				return result;
			}
			std::size_t const middle = begin + count / 2;
			auto left = ascend(2 * node, begin, middle, weights);
			auto right = ascend(2 * node + 1, middle, end, weights);
			if constexpr (points_transform_coefficient<Coefficient>) {
				if (!left_spectra_[node].empty()) {
					std::size_t const size = left_spectra_[node].size();
					left.resize(size);
					right.resize(size);
					ntt<Engine>(left);
					ntt<Engine>(right);
					for (std::size_t i = 0; i < size; ++i) left[i] = left[i] * right_spectra_[node][i] + right[i] * left_spectra_[node][i];
					inverse_ntt<Engine>(left);
					left.resize(count);
					return left;
				}
			}
			left = series_product<Engine, Coefficient>(left, products_[2 * node + 1], count);
			right = series_product<Engine, Coefficient>(right, products_[2 * node], count);
			for (std::size_t i = 0; i < count; ++i) left[i] += right[i];
			return left;
		}

	public:
		/** @brief Build one immutable tree; cutoff is exposed internally for the pending leaf race.
		 * @pre points.size() <= maximum_series_length; 1 <= cutoff < maximum_series_length.
		 */
		constexpr explicit points_tree_storage(std::vector<Coefficient> points, std::size_t cutoff = polynomial_points_cutoff) : points_(std::move(points)), cutoff_(cutoff) {
			atl_assert(points_.size() <= maximum_series_length);
			atl_assert(cutoff >= 1 && cutoff < maximum_series_length);
			if (points_.empty()) return;
			std::size_t const leaves = std::bit_ceil((points_.size() + cutoff - 1) / cutoff);
			products_.resize(2 * leaves);
			left_spectra_.resize(2 * leaves);
			right_spectra_.resize(2 * leaves);
			build(1, 0, points_.size());
		}

		constexpr std::size_t point_count() const { return points_.size(); }

		constexpr std::span<Coefficient const> points() const { return points_; }

		/** @brief Evaluate arbitrary-length input, first reducing it at the root when necessary.
		 * With `uses_remainders` the descent divides by the child products instead of using middle products.
		 * @pre Polynomial length <= maximum_series_length.
		 */
		[[nodiscard]] constexpr std::vector<Coefficient> evaluate(std::vector<Coefficient> polynomial, bool uses_remainders = false) const {
			atl_assert(polynomial.size() <= maximum_series_length);
			std::size_t const count = points_.size();
			std::vector<Coefficient> values(count);
			if (count == 0) return values;
			if (count <= cutoff_) {
				for (std::size_t i = 0; i < count; ++i) values[i] = points_horner<Coefficient>(polynomial, points_[i]);
				return values;
			}
			if (polynomial.size() > count) {
				auto divisor = products_[1];
				std::ranges::reverse(divisor);
				polynomial = polynomial_division<Engine>(polynomial, divisor).remainder;
			}
			if (uses_remainders) {
				descend_remainders(1, 0, count, std::move(polynomial), values);
				return values;
			}
			polynomial.resize(count);
			std::ranges::reverse(polynomial);
			auto const inverse = series_inverse<Engine>(std::span(products_[1]).first(count), count);
			auto state = series_product<Engine, Coefficient>(polynomial, inverse, count);
			descend(1, 0, count, std::move(state), values);
			return values;
		}

		/** @brief Interpolate using derivative weights and a shared upward combination.
		 * @pre values.size() == point_count(); derivative values are nonzero (distinct points).
		 */
		[[nodiscard]] constexpr std::vector<Coefficient> interpolate(std::span<Coefficient const> values, bool uses_remainders = false) const {
			std::size_t const count = points_.size();
			atl_assert(values.size() == count);
			if (count == 0) return {};
			if (count <= cutoff_) return points_interpolation_newton<Coefficient>(points_, values);
			std::vector<Coefficient> derivative(count);
			for (std::size_t i = 1; i <= count; ++i) derivative[i - 1] = Coefficient(i) * products_[1][count - i];
			auto weights = batch_inverse(evaluate(std::move(derivative), uses_remainders));
			for (std::size_t i = 0; i < count; ++i) weights[i] *= values[i];
			auto result = ascend(1, 0, count, weights);
			std::ranges::reverse(result);
			return result;
		}
	};
}
