#pragma once

#include <atl/nt/convolution_mod.hpp>
#include <atl/nt/exact_field.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief A sized random-access range of @ref atl::exact_field values for the
	 *        recurrence functions to read sequences and polynomials.
	 */
	template<typename R>
	concept field_range = std::ranges::random_access_range<R> && std::ranges::sized_range<R>
					   && exact_field<std::ranges::range_value_t<R>>
					   && std::convertible_to<std::ranges::range_reference_t<R>, std::ranges::range_value_t<R>>;

	namespace detail {
		template<exact_field T>
		constexpr std::vector<T> polynomial_product_schoolbook(std::vector<T> const &a, std::vector<T> const &b) {
			if (a.empty() || b.empty()) return {};
			std::vector<T> c(a.size() + b.size() - 1, T(0));
			for (std::size_t i = 0; i < a.size(); ++i)
				for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = c[i + j] + a[i] * b[j];
			return c;
		}

		/**
		 * @brief The product of two polynomials (coefficient vectors, constant
		 *        term first): @ref atl::convolution for a static modulus (which
		 *        multiplies naively below its measured crossover), the schoolbook
		 *        product otherwise.
		 */
		template<exact_field T>
		constexpr std::vector<T> polynomial_product(std::vector<T> const &a, std::vector<T> const &b) {
			if constexpr (static_mod_int<T>) {
				return atl::convolution(a, b);
			} else {
				return polynomial_product_schoolbook(a, b);
			}
		}

		/**
		 * @brief Kitamasa: `x^k mod f`, f the characteristic polynomial, by
		 *        square-and-multiply with schoolbook products and long division;
		 *        then `a[k] = sum r_i a[i]`.
		 * @pre `k >= c.size()`, `a.size() >= c.size()`.
		 * @complexity O(d² log k)
		 */
		template<exact_field T>
		constexpr T kitamasa_by_schoolbook(std::vector<T> const &c, std::vector<T> const &a, uint64 k) {
			std::size_t const d = c.size();
			if (d == 0) return T(0);
			if (k < d && k < a.size()) return a[std::size_t(k)];
			// reduce a polynomial of degree < 2d by x^d = sum c[j] x^{d-1-j}
			auto const reduce = [&](std::vector<T> &r) {
				for (std::size_t top = r.size(); top-- > d;) {
					T const lead = r[top];
					if (lead == T(0)) continue;
					for (std::size_t j = 0; j < d; ++j) r[top - 1 - j] = r[top - 1 - j] + lead * c[j];
				}
				r.resize(d, T(0));
			};
			std::vector<T> r{T(1)};
			for (int bit = int(std::bit_width(k)) - 1; bit >= 0; --bit) {
				r = polynomial_product_schoolbook(r, r);
				if ((k >> bit) & 1) r.insert(r.begin(), T(0));
				if (r.size() > d) reduce(r);
			}
			r.resize(d, T(0));
			T result = T(0);
			for (std::size_t i = 0; i < d; ++i) result = result + r[i] * a[i];
			return result;
		}

		/**
		 * @brief Bostan–Mori rounds by polynomial products: `[x^k] p/q` with
		 *        `p.size() == max(d, 1)`, `q.size() == d + 1`, `q[0]` invertible.
		 * @complexity O(M(d) log k)
		 */
		template<exact_field T>
		constexpr T bostan_mori_by_products(std::vector<T> p, std::vector<T> q, uint64 k) {
			std::vector<T> q_negated(q.size());
			while (k > 0) {
				for (std::size_t i = 0; i < q.size(); ++i) q_negated[i] = i % 2 ? T(0) - q[i] : q[i];
				auto const u = polynomial_product(p, q_negated);
				auto const v = polynomial_product(q, q_negated);
				std::size_t const parity = std::size_t(k & 1);
				for (std::size_t i = 0; i < p.size(); ++i) p[i] = 2 * i + parity < u.size() ? u[2 * i + parity] : T(0);
				for (std::size_t i = 0; i < q.size(); ++i) q[i] = v[2 * i];
				k >>= 1;
			}
			return p[0] / q[0];
		}

		/**
		 * @brief From this order on, the transform-domain route; below it the
		 *        schoolbook routes win (benchmarks/nt/linear_recurrence.cpp,
		 *        2026-10-15, Xeon 8280: at d = 32 transforms 0.325 ms vs schoolbook
		 *        Kitamasa 0.192, at d = 64 0.692 vs 0.678 (tie), at d = 100 0.688
		 *        vs 1.7; plain -O2).
		 */
		inline constexpr std::size_t bostan_mori_transform_cutoff = 64;

		/**
		 * @brief Bostan–Mori rounds in the transform domain, three length-N
		 *        transforms' worth per round instead of two full convolutions.
		 *
		 * With N = bit_ceil(2d + 2) and the engine's bit-reversed layout, entry
		 * `j` and `j ^ 1` of a transform hold the values at `w` and `-w`, so
		 * `Q(-x)` costs no transform: `U = P(x)Q(-x)` and `V = Q(x)Q(-x)` are
		 * pointwise products. `V` is even and its values pair up equal, so the
		 * entries `2m` are already the half-length transform of `V`'s even part.
		 * `U`'s even / odd part at `w²` is `(U(w) ± U(-w)) / 2` (divided by `w`
		 * for the odd part), again a half-length transform, so two half-length
		 * inverses finish the round. The `w` of each entry is read off the
		 * transform of `x` itself, so nothing depends on the engine's choice of
		 * root.
		 *
		 * @pre `p.size() == d`, `q.size() == d + 1`, `d >= 1`, N divides `mod - 1`.
		 * @complexity O(d log d log k)
		 */
		template<ntt_modulus T>
		constexpr T bostan_mori_by_transforms(std::vector<T> p, std::vector<T> q, uint64 k) {
			std::size_t const d = q.size() - 1, n = std::bit_ceil(2 * d + 2), half = n / 2;
			// entry 2m of the transform of x is w_m; entry 2m+1 is -w_m
			std::vector<T> root(n, T(0));
			root[1] = T(1);
			atl::ntt(root);
			T const one_half = T(2).inverse();
			std::vector<T> odd_scale(half);
			for (std::size_t m = 0; m < half; ++m) odd_scale[m] = one_half * root[2 * m].inverse();
			std::vector<T> fp(n), fq(n), even(half), odd(half);
			while (k > 0) {
				std::ranges::fill(std::ranges::copy(p, fp.begin()).out, fp.end(), T(0));
				std::ranges::fill(std::ranges::copy(q, fq.begin()).out, fq.end(), T(0));
				atl::ntt(fp);
				atl::ntt(fq);
				bool const is_odd = (k & 1) != 0;
				for (std::size_t m = 0; m < half; ++m) {
					T const u0 = fp[2 * m] * fq[2 * m + 1], u1 = fp[2 * m + 1] * fq[2 * m];
					even[m] = fq[2 * m] * fq[2 * m + 1];
					odd[m] = is_odd ? (u0 - u1) * odd_scale[m] : (u0 + u1) * one_half;
				}
				atl::inverse_ntt(even);
				atl::inverse_ntt(odd);
				std::ranges::copy(even.begin(), even.begin() + std::ptrdiff_t(d + 1), q.begin());
				std::ranges::copy(odd.begin(), odd.begin() + std::ptrdiff_t(d), p.begin());
				k >>= 1;
			}
			return p[0] / q[0];
		}
	}

	/**
	 * @brief The shortest linear recurrence a sequence satisfies: coefficients
	 *        `c` (length `d`) with `s[i] = c[0]·s[i-1] + … + c[d-1]·s[i-d]` for
	 *        every `d <= i < s.size()` (Berlekamp–Massey).
	 *
	 * A prefix of at least twice the generating sequence's minimal order
	 * determines its minimal recurrence. A shorter prefix need not predict
	 * unseen terms. An all-zero sequence gives `d = 0`.
	 *
	 * @tparam Sequence A const-readable sized random-access range of exact-field values.
	 * @param sequence The first terms `s[0..N)`.
	 * @return A coefficient vector, empty for an empty or all-zero input.
	 * @note Dynamic-modulus instantiations use runtime state and cannot be constant-evaluated.
	 * @complexity O(N²) field operations, at most one division per sample
	 */
	template<field_range Sequence>
		requires field_range<Sequence const>
	[[nodiscard]] constexpr auto berlekamp_massey(Sequence const &sequence) {
		using T = std::ranges::range_value_t<Sequence>;
		std::size_t const n = std::ranges::size(sequence);
		// connection polynomials, constant term 1: s[i] + sum_{j>=1} C[j] s[i-j] = 0
		std::vector<T> current{T(1)}, anchor{T(1)};
		std::size_t length = 0, shift = 1;
		T anchor_discrepancy = T(1);
		for (std::size_t i = 0; i < n; ++i) {
			T discrepancy = std::ranges::begin(sequence)[std::ptrdiff_t(i)];
			for (std::size_t j = 1; j <= length; ++j) discrepancy = discrepancy + current[j] * std::ranges::begin(sequence)[std::ptrdiff_t(i - j)];
			if (discrepancy == T(0)) {
				++shift;
				continue;
			}
			T const factor = discrepancy / anchor_discrepancy;
			bool const is_lengthening = 2 * length <= i;
			std::vector<T> saved = is_lengthening ? current : std::vector<T>{};
			if (current.size() < anchor.size() + shift) current.resize(anchor.size() + shift, T(0));
			for (std::size_t j = 0; j < anchor.size(); ++j) current[j + shift] = current[j + shift] - factor * anchor[j];
			if (is_lengthening) {
				length = i + 1 - length;
				anchor = std::move(saved);
				anchor_discrepancy = discrepancy;
				shift = 1;
			} else {
				++shift;
			}
		}
		std::vector<T> coefficients(length);
		for (std::size_t j = 1; j <= length; ++j) coefficients[j - 1] = j < current.size() ? T(0) - current[j] : T(0);
		return coefficients;
	}

	/**
	 * @brief `[x^k] P(x) / Q(x)`, the k-th coefficient of a rational power
	 *        series, by Bostan–Mori: `P/Q = P(x)Q(-x) / Q(x)Q(-x)`, whose
	 *        denominator is even, so keeping the coefficients of `P(x)Q(-x)` of
	 *        parity `k` halves `k` each round.
	 *
	 * @param numerator `P`, constant term first.
	 * @param denominator `Q`, constant term first.
	 * @param k The coefficient index.
	 * @return The coefficient of x^k, including zero for an empty numerator.
	 * @pre max(numerator.size(), denominator.size() - 1) <= SIZE_MAX / 4 - 1 (asserted).
	 * @pre `denominator` is nonempty with an invertible constant term
	 *      (asserted nonzero).
	 * From d = 64 on, over an NTT-friendly modulus, the rounds run in the
	 * transform domain (see detail::bostan_mori_by_transforms); otherwise each
	 * round is two polynomial products.
	 *
	 * @complexity O(d + M(d) log(k + 1)) time, O(d + 1) storage;
	 * d = max(numerator.size(), denominator.size() - 1), M is polynomial product cost.
	 * Static moduli use convolution; other fields use O(d²) schoolbook products.
	 * @note Dynamic-modulus instantiations use runtime state and cannot be constant-evaluated.
	 */
	template<field_range Numerator, field_range Denominator>
		requires field_range<Numerator const> && field_range<Denominator const>
			  && std::same_as<std::ranges::range_value_t<Numerator>, std::ranges::range_value_t<Denominator>>
	[[nodiscard]] constexpr auto rational_series_coefficient(Numerator const &numerator, Denominator const &denominator, uint64 k) {
		using T = std::ranges::range_value_t<Numerator>;
		atl_assert(std::ranges::size(denominator) > 0 && !(*std::ranges::begin(denominator) == T(0)));
		std::vector<T> p;
		p.reserve(std::ranges::size(numerator));
		for (auto &&value : numerator) p.emplace_back(std::forward<decltype(value)>(value));
		std::vector<T> q;
		q.reserve(std::ranges::size(denominator));
		for (auto &&value : denominator) q.emplace_back(std::forward<decltype(value)>(value));
		std::size_t const d = (std::max)(p.size(), q.size() - 1);
		atl_assert(d <= (std::numeric_limits<std::size_t>::max)() / 4 - 1);
		p.resize(std::max<std::size_t>(d, 1), T(0));
		q.resize(d + 1, T(0));
		if constexpr (detail::ntt_modulus<T>) {
			if (d >= (std::max<std::size_t>)(detail::bostan_mori_transform_cutoff, 1) && (T::mod() - 1) % std::bit_ceil(2 * d + 2) == 0)
				return detail::bostan_mori_by_transforms(std::move(p), std::move(q), k);
		}
		return detail::bostan_mori_by_products(std::move(p), std::move(q), k);
	}

	/**
	 * @brief The k-th term of the linear recurrence `a[i] = c[0]·a[i-1] + … +
	 *        c[d-1]·a[i-d]` from its first `d` terms, for k up to 2^64 - 1.
	 *
	 * Size-dispatched, by the race (benchmarks/nt/linear_recurrence.cpp,
	 * 2026-10-15, x86-64 Xeon 8280): from order 64 on, over an NTT-friendly
	 * modulus, Bostan–Mori in the transform domain on `A(x) = P(x)/Q(x)`,
	 * `Q = 1 - c[0]x - … - c[d-1]x^d`, `P = A·Q mod x^d` (d = 1e5, k = 1e18:
	 * 1204 ms vs 2270 by convolution products and 2549 for Kitamasa with
	 * convolutions, plain -O2); over another static modulus Bostan–Mori by
	 * `atl::convolution` products; below order 64, or with no convolution
	 * available (a run-time modulus), Kitamasa's `x^k mod` characteristic
	 * polynomial with schoolbook products (d = 10: 0.018 ms vs 0.025 for
	 * schoolbook Bostan–Mori). Losers: experimental/nt/linear_recurrence_variants.hpp.
	 *
	 * @param coefficients `c`, length `d`.
	 * @param k The zero-based term index.
	 * @return The k-th term; zero for an empty recurrence.
	 * @pre d <= SIZE_MAX / 4 - 1 (asserted).
	 * @param initial The first terms `a[0..)`; only the first `d` are read.
	 * @pre `initial.size() >= coefficients.size()` (asserted).
	 * @complexity O(d + d log(d + 1) log(k + 1)) with suitable NTT roots,
	 * O(d + M(d) log(k + 1)) with convolution, O(d + d² log(k + 1)) otherwise; O(d) storage.
	 * @note Dynamic-modulus instantiations use runtime state and cannot be constant-evaluated.
	 */
	template<field_range Coefficients, field_range Initial>
		requires field_range<Coefficients const> && field_range<Initial const>
			  && std::same_as<std::ranges::range_value_t<Coefficients>, std::ranges::range_value_t<Initial>>
	[[nodiscard]] constexpr auto linear_recurrence_term(Coefficients const &coefficients, Initial const &initial, uint64 k) {
		using T = std::ranges::range_value_t<Coefficients>;
		std::size_t const d = std::ranges::size(coefficients);
		atl_assert(d <= (std::numeric_limits<std::size_t>::max)() / 4 - 1);
		atl_assert(std::ranges::size(initial) >= d);
		if (k < d) return T(std::ranges::begin(initial)[std::ptrdiff_t(k)]);
		std::vector<T> c;
		c.reserve(d);
		for (auto &&value : coefficients) c.emplace_back(std::forward<decltype(value)>(value));
		std::vector<T> a(std::ranges::begin(initial), std::ranges::begin(initial) + std::ptrdiff_t(d));
		if (d == 0) return T(0);
		bool is_transform_sized = false;
		if constexpr (detail::ntt_modulus<T>)
			is_transform_sized = d >= detail::bostan_mori_transform_cutoff
							  && (T::mod() - 1) % std::bit_ceil(2 * d + 2) == 0;
		// no convolution (a run-time modulus): schoolbook Kitamasa beat
		// schoolbook Bostan-Mori at every order raced (153 vs 182 ms at d = 1000)
		if (!is_transform_sized && (!static_mod_int<T> || d < detail::bostan_mori_transform_cutoff))
			return detail::kitamasa_by_schoolbook(c, a, k);
		std::vector<T> q(d + 1);
		q[0] = T(1);
		for (std::size_t i = 0; i < d; ++i) q[i + 1] = T(0) - c[i];
		auto p = detail::polynomial_product(a, q);
		p.resize(d);
		if constexpr (detail::ntt_modulus<T>)
			if (is_transform_sized) return detail::bostan_mori_by_transforms(std::move(p), std::move(q), k);
		return detail::bostan_mori_by_products(std::move(p), std::move(q), k);
	}
}
