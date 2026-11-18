#pragma once
#include <atl/nt/convolution.hpp>
#include <atl/nt/gaussian_elimination.hpp>
#include <atl/nt/linear_recurrence.hpp>
#include <atl/nt/matrix.hpp>
#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

/**
 * @file
 * @brief k-th term of a linear recurrence: the contenders raced against the
 *        size-dispatched `atl::linear_recurrence_term` (schoolbook Kitamasa
 *        below order 64, transform-domain Bostan–Mori from 64 on).
 *
 * Race: benchmarks/nt/linear_recurrence.cpp, measured 2026-10-15
 * with 10 samples per row (x86-64 Xeon Platinum 8280, mod
 * 998244353, k = 1e18). A short k lets Kitamasa skip full-size rounds,
 * so every raced order uses the same large index.
 *
 * Shared contract: `f(c, a, k)` with recurrence coefficients `c` (length d,
 * `a[i] = sum c[j] a[i-1-j]`), at least d initial terms `a`, and the index
 * `k`, returns `a[k]`. The readable oracle in tests is
 * @ref atl::experimental::linear_recurrence_term_by_stepping.
 * @pre Initial terms contain at least d entries; polynomial sizes fit storage.
 */
namespace atl::experimental {
	using atl::exact_field;
	using atl::static_mod_int;

	/**
	 * @brief The definition, run forward: the oracle.
	 * @complexity O(k d)
	 */
	template<exact_field T>
	constexpr T linear_recurrence_term_by_stepping(std::vector<T> const &c, std::vector<T> a, uint64 k) {
		std::size_t const d = c.size();
		atl_assert(a.size() >= d);
		atl_assert(d <= (std::numeric_limits<std::size_t>::max)() / 4 - 1);
		if (d == 0) return T(0);
		atl_assert(k < std::vector<T>().max_size());
		a.resize(d);
		while (a.size() <= k) {
			T next = T(0);
			for (std::size_t j = 0; j < d; ++j) next = next + c[j] * a[a.size() - 1 - j];
			a.push_back(next);
		}
		return a[std::size_t(k)];
	}

	/**
	 * @brief The companion matrix raised to `k` (atl::power over atl::matrix),
	 *        applied to the initial terms.
	 *
	 * Lost: 0.082 ms at d = 10 (plain -O2; 0.065 -march=native) against the
	 * library schoolbook Kitamasa's 0.018: O(d³) per squaring.
	 * @complexity O(d³ log k)
	 */
	template<exact_field T>
	constexpr T linear_recurrence_term_by_matrix_power(std::vector<T> const &c, std::vector<T> const &a, uint64 k) {
		std::size_t const d = c.size();
		atl_assert(a.size() >= d);
		atl_assert(d <= (std::numeric_limits<std::size_t>::max)() / 4 - 1);
		if (k < d) return a[std::size_t(k)];
		if (d == 0) return T(0);
		atl::matrix<T> companion(d, d, T(0));
		for (std::size_t j = 0; j < d; ++j) companion(0, uint32(j)) = c[j];
		for (std::size_t i = 1; i < d; ++i) companion(uint32(i), uint32(i - 1)) = T(1);
		// state (a[i+d-1], ..., a[i]) advances by one per multiplication
		std::vector<T> state(d);
		for (std::size_t i = 0; i < d; ++i) state[i] = a[d - 1 - i];
		auto const advanced = atl::power(companion, k - (d - 1)) * std::span<T const>(state);
		return advanced[0];
	}

	namespace detail {
		template<exact_field T>
		constexpr std::vector<T> schoolbook_product(std::vector<T> const &a, std::vector<T> const &b) {
			if (a.empty() || b.empty()) return {};
			std::vector<T> c(a.size() + b.size() - 1, T(0));
			for (std::size_t i = 0; i < a.size(); ++i)
				for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = c[i + j] + a[i] * b[j];
			return c;
		}

		/** @brief `[x^0..x^n)` of `1/q`, by Newton doubling over `polynomial_product`. */
		template<exact_field T>
		constexpr std::vector<T> series_inverse(std::vector<T> const &q, std::size_t n) {
			std::vector<T> g{T(1) / q[0]};
			for (std::size_t m = 1; m < n; m *= 2) {
				std::size_t const next = (std::min)(2 * m, n);
				std::vector<T> head(q.begin(), q.begin() + std::ptrdiff_t((std::min)(q.size(), next)));
				auto e = atl::detail::polynomial_product(head, g); // q g = 1 + O(x^m)
				e.resize(next, T(0));
				for (auto &x : e) x = T(0) - x;
				e[0] = e[0] + T(2);
				g = atl::detail::polynomial_product(g, e);
				g.resize(next, T(0));
			}
			g.resize(n, T(0));
			return g;
		}
	}

	/**
	 * @brief Schoolbook Kitamasa
	 *        (`atl::detail::kitamasa_by_schoolbook`) forced for every order, so
	 *        the race can place the cutoff.
	 * @complexity O(d² log k)
	 */
	template<exact_field T>
	constexpr T linear_recurrence_term_by_schoolbook_kitamasa(std::vector<T> const &c, std::vector<T> const &a, uint64 k) {
		atl_assert(a.size() >= c.size());
		if (k < c.size()) return a[std::size_t(k)];
		return atl::detail::kitamasa_by_schoolbook(c, a, k);
	}

	/**
	 * @brief Kitamasa with convolution products and Newton-inverse division
	 *        (the reversed characteristic polynomial's inverse mod x^{d-1},
	 *        computed once): three products per bit against Bostan–Mori's two.
	 *
	 * Lost at every order: 15.7 ms vs 6.4 at d = 1000 and 2549 vs 1204 at
	 * d = 1e5 against the library transform-domain Bostan–Mori (plain -O2).
	 * @complexity O(M(d) log k)
	 */
	template<static_mod_int T>
		requires exact_field<T>
	constexpr T linear_recurrence_term_by_convolution_kitamasa(std::vector<T> const &c, std::vector<T> const &a, uint64 k) {
		std::size_t const d = c.size();
		atl_assert(a.size() >= d);
		atl_assert(d <= (std::numeric_limits<std::size_t>::max)() / 4 - 1);
		if (k < d) return a[std::size_t(k)];
		if (d == 0) return T(0);
		if (d == 1) return a[0] * c[0].power(k);
		std::vector<T> f(d + 1); // monic, constant term first
		f[d] = T(1);
		for (std::size_t j = 0; j < d; ++j) f[d - 1 - j] = T(0) - c[j];
		std::vector<T> reversed_f(f.rbegin(), f.rend());
		auto const inverse = detail::series_inverse(reversed_f, d - 1);
		auto const reduce = [&](std::vector<T> &r) {
			if (r.size() <= d) return;
			r.resize(2 * d - 1, T(0));
			std::vector<T> top(r.rbegin(), r.rbegin() + std::ptrdiff_t(d - 1));
			auto q = atl::convolution(top, inverse);
			q.resize(d - 1);
			std::ranges::reverse(q);
			auto const qf = atl::convolution(q, f);
			for (std::size_t i = 0; i < d; ++i) r[i] -= qf[i];
			r.resize(d);
		};
		std::vector<T> r{T(1)};
		for (int bit = int(std::bit_width(k)) - 1; bit >= 0; --bit) {
			r = atl::convolution(r, r);
			reduce(r);
			if ((k >> bit) & 1) {
				r.insert(r.begin(), T(0));
				if (r.size() > d) { // fold the one term at x^d
					T const lead = r[d];
					for (std::size_t j = 0; j < d; ++j) r[d - 1 - j] += lead * c[j];
					r.resize(d);
				}
			}
		}
		r.resize(d, T(0));
		T result = T(0);
		for (std::size_t i = 0; i < d; ++i) result += r[i] * a[i];
		return result;
	}

	/**
	 * @brief Bostan–Mori with schoolbook products throughout.
	 *
	 * Lost to schoolbook Kitamasa at every order: 0.025 vs 0.018 ms (d = 10),
	 * 1.9 vs 1.7 (d = 100), 182 vs 153 (d = 1000), plain -O2: its two
	 * products per round cost more than one squaring plus a reduction.
	 * @complexity O(d² log k)
	 */
	template<exact_field T>
	constexpr T linear_recurrence_term_by_schoolbook_bostan_mori(std::vector<T> const &c, std::vector<T> const &a, uint64 k) {
		std::size_t const d = c.size();
		atl_assert(a.size() >= d);
		atl_assert(d <= (std::numeric_limits<std::size_t>::max)() / 4 - 1);
		if (k < d) return a[std::size_t(k)];
		if (d == 0) return T(0);
		std::vector<T> q(d + 1);
		q[0] = T(1);
		for (std::size_t i = 0; i < d; ++i) q[i + 1] = T(0) - c[i];
		auto p = detail::schoolbook_product(std::vector<T>(a.begin(), a.begin() + std::ptrdiff_t(d)), q);
		p.resize(d);
		std::vector<T> q_negated(d + 1);
		while (k > 0) {
			for (std::size_t i = 0; i <= d; ++i) q_negated[i] = i % 2 ? T(0) - q[i] : q[i];
			auto const u = detail::schoolbook_product(p, q_negated);
			auto const v = detail::schoolbook_product(q, q_negated);
			std::size_t const parity = std::size_t(k & 1);
			for (std::size_t i = 0; i < d; ++i) p[i] = u[2 * i + parity];
			for (std::size_t i = 0; i <= d; ++i) q[i] = v[2 * i];
			k >>= 1;
		}
		return p[0] / q[0];
	}

	/**
	 * @brief Bostan–Mori by two `atl::convolution` products per round (the
	 *        library's product path, which still serves non-NTT moduli
	 *        and orders below the transform cutoff), forced for every order.
	 *
	 * Lost where the transform route applies: 1.3 vs 0.688 ms (d = 100),
	 * 12.4 vs 6.4 (d = 1000), 2270 vs 1204 (d = 1e5); below d = 64 it trails
	 * schoolbook Kitamasa (0.197 vs 0.192 at d = 32, 0.026 vs 0.018 at d = 10),
	 * plain -O2. Used for a static modulus the transforms cannot take.
	 * @complexity O(M(d) log k)
	 */
	template<static_mod_int T>
		requires exact_field<T>
	constexpr T linear_recurrence_term_by_convolution_bostan_mori(std::vector<T> const &c, std::vector<T> const &a, uint64 k) {
		std::size_t const d = c.size();
		atl_assert(a.size() >= d);
		atl_assert(d <= (std::numeric_limits<std::size_t>::max)() / 4 - 1);
		if (k < d) return a[std::size_t(k)];
		if (d == 0) return T(0);
		std::vector<T> q(d + 1);
		q[0] = T(1);
		for (std::size_t i = 0; i < d; ++i) q[i + 1] = T(0) - c[i];
		auto p = atl::convolution(std::vector<T>(a.begin(), a.begin() + std::ptrdiff_t(d)), q);
		p.resize(d);
		return atl::detail::bostan_mori_by_products(std::move(p), std::move(q), k);
	}

	/**
	 * @brief Transform-domain Bostan–Mori
	 *        (`atl::detail::bostan_mori_by_transforms`) forced for every order,
	 *        so the race can place the cutoff.
	 * @pre `Z::mod() - 1` is divisible by `bit_ceil(2d + 2)`.
	 * @complexity O(d log d log k)
	 */
	template<atl::detail::ntt_modulus T>
	constexpr T linear_recurrence_term_by_transform_bostan_mori(std::vector<T> const &c, std::vector<T> const &a, uint64 k) {
		std::size_t const d = c.size();
		atl_assert(a.size() >= d);
		atl_assert(d <= (std::numeric_limits<std::size_t>::max)() / 4 - 1);
		if (k < d) return a[std::size_t(k)];
		if (d == 0) return T(0);
		std::vector<T> q(d + 1);
		q[0] = T(1);
		for (std::size_t i = 0; i < d; ++i) q[i + 1] = T(0) - c[i];
		auto p = atl::convolution(std::vector<T>(a.begin(), a.begin() + std::ptrdiff_t(d)), q);
		p.resize(d);
		return atl::detail::bostan_mori_by_transforms(std::move(p), std::move(q), k);
	}
}
