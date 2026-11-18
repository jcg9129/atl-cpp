#pragma once
#include <atl/nt/matrix.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

/**
 * @file
 * @brief Square-matrix product contenders over a static modulus of at most 32
 *        bits, raced against the library `atl::detail::lazy_product`.
 *
 * Shared contract: `f(a, b, n)` multiplies two row-major `n x n` matrices of
 * residues and returns the row-major product (square only: the race shape;
 * the library kernel takes `rows, inner, columns`).
 *
 * Race (benchmarks/nt/matrix.cpp, 2026-10-14, Xeon 8280, mod
 * 998244353, ms at n = 200 / 300 / 500): library lazy i-k-j 5.3 / 17.4 / 80.4
 * plain -O2, 2.1 / 7.1 / 32.1 -march=native. The winner is the simplest lazy
 * loop; no contender beats it by more than noise in either configuration. Every contender is checked
 * against @ref atl::experimental::matrix_product_by_montgomery_ikj (the readable oracle: the generic
 * semiring loop, one Montgomery multiply-add per term) in tests.
 * @pre Both input spans contain n² entries; n² fits vector storage.
 */
namespace atl::experimental {
	using atl::static_mod_int;

	namespace detail {
		/** @brief How many residue products a 64-bit accumulator below `mod` can take. */
		template<static_mod_int Z>
			requires(Z::mod() <= (std::numeric_limits<uint32>::max)())
		constexpr uint64 lazy_batch() {
			constexpr uint64 mod = Z::mod(), square = (mod - 1) * (mod - 1);
			return square == 0 ? (std::numeric_limits<uint64>::max)()
							   : ((std::numeric_limits<uint64>::max)() - (mod - 1)) / square;
		}
	}

	/**
	 * @brief i-k-j over `Z` itself: one Montgomery multiply and one modular add
	 *        per term. The oracle; also what `matrix` runs for moduli wider
	 *        than 32 bits.
	 *
	 * Lost: 12.1 / 40.9 / 190 ms (-O2) and 12.2 / 41.0 / 189 (native) at
	 * n = 200 / 300 / 500, 2.3-5.9x behind the library kernel: a Montgomery
	 * reduction per term, which does not vectorize.
	 * @complexity O(n³)
	 */
	template<static_mod_int Z>
	constexpr std::vector<Z> matrix_product_by_montgomery_ikj(std::span<Z const> a, std::span<Z const> b, uint32 n) {
		atl_assert(n == 0 || std::size_t(n) <= std::vector<Z>().max_size() / n);
		atl_assert(a.size() == std::size_t(n) * n && b.size() == a.size());
		if (n == 0) return {};
		std::vector<Z> c(std::size_t(n) * n);
		for (uint32 i = 0; i < n; ++i)
			for (uint32 k = 0; k < n; ++k) {
				Z const x = a[std::size_t(i) * n + k];
				for (uint32 j = 0; j < n; ++j) c[std::size_t(i) * n + j] += x * b[std::size_t(k) * n + j];
			}
		return c;
	}

	/**
	 * @brief i-j-k against a transposed right factor, over `Z`: each entry is a
	 *        contiguous dot product with Montgomery multiply-adds.
	 *
	 * Lost: 12.0 / 40.4 / 188 ms (-O2), 12.1 / 41.0 / 189 (native); the
	 * loop order is irrelevant when the per-term reduction dominates.
	 * @complexity O(n³)
	 */
	template<static_mod_int Z>
	constexpr std::vector<Z> matrix_product_by_montgomery_transpose(std::span<Z const> a, std::span<Z const> b, uint32 n) {
		atl_assert(n == 0 || std::size_t(n) <= std::vector<Z>().max_size() / n);
		atl_assert(a.size() == std::size_t(n) * n && b.size() == a.size());
		if (n == 0) return {};
		std::vector<Z> bt(std::size_t(n) * n), c(std::size_t(n) * n);
		for (uint32 i = 0; i < n; ++i)
			for (uint32 j = 0; j < n; ++j) bt[std::size_t(j) * n + i] = b[std::size_t(i) * n + j];
		for (uint32 i = 0; i < n; ++i)
			for (uint32 j = 0; j < n; ++j) {
				Z sum{};
				for (uint32 k = 0; k < n; ++k) sum += a[std::size_t(i) * n + k] * bt[std::size_t(j) * n + k];
				c[std::size_t(i) * n + j] = sum;
			}
		return c;
	}

	/**
	 * @brief i-j-k against a transposed right factor on plain residues: each
	 *        entry a 64-bit dot product reduced every `lazy_batch` terms.
	 *
	 * Lost: ties with -O2 (5.3 / 17.5 / 80.3 ms vs the library 5.3 / 17.4 /
	 * 80.4) but 3.5 / 11.6 / 55.4 vs 2.1 / 7.1 / 32.1 with -march=native: a
	 * horizontal reduction per entry vectorizes worse than i-k-j's
	 * independent accumulators, and it pays an O(n²) transpose.
	 * @complexity O(n³)
	 */
	template<static_mod_int Z>
		requires(Z::mod() <= (std::numeric_limits<uint32>::max)())
	constexpr std::vector<Z> matrix_product_by_lazy_transpose(std::span<Z const> a, std::span<Z const> b, uint32 n) {
		atl_assert(n == 0 || std::size_t(n) <= std::vector<Z>().max_size() / n);
		atl_assert(a.size() == std::size_t(n) * n && b.size() == a.size());
		if (n == 0) return {};
		constexpr uint64 mod = Z::mod(), batch = detail::lazy_batch<Z>();
		std::vector<uint32> lhs(a.size()), rhs_t(b.size());
		for (std::size_t i = 0; i < a.size(); ++i) lhs[i] = uint32(a[i].value());
		for (uint32 i = 0; i < n; ++i)
			for (uint32 j = 0; j < n; ++j) rhs_t[std::size_t(j) * n + i] = uint32(b[std::size_t(i) * n + j].value());
		std::vector<Z> c(std::size_t(n) * n);
		for (uint32 i = 0; i < n; ++i) {
			uint32 const *x = lhs.data() + std::size_t(i) * n;
			for (uint32 j = 0; j < n; ++j) {
				uint32 const *y = rhs_t.data() + std::size_t(j) * n;
				uint64 sum = 0;
				for (uint32 k0 = 0; k0 < n;) {
					uint32 const k1 = k0 + uint32((std::min<uint64>)(batch, n - k0));
					uint64 part = 0;
					for (uint32 k = k0; k < k1; ++k) part += uint64(x[k]) * y[k];
					sum = (sum + part % mod) % mod;
					k0 = k1;
				}
				c[std::size_t(i) * n + j] = Z::unchecked(sum);
			}
		}
		return c;
	}

	/**
	 * @brief The library lazy i-k-j kernel tiled: a block of `Tile` output
	 *        columns at a time, so the accumulator tile and the touched slice
	 *        of the right factor stay in L1/L2 across the k loop.
	 *
	 * Lost (no gain for the extra code): 5.2 / 18.2 / 81.5 ms (-O2), 1.9 /
	 * 7.0 / 32.1 (native) vs the library 5.3 / 17.4 / 80.4 and 2.1 / 7.1 /
	 * 32.1. At n <= 500 one accumulator row (4 KB) and B (<= 1 MB) already
	 * sit in L1/L2.
	 * @complexity O(n³)
	 */
	template<static_mod_int Z, uint32 Tile = 256>
		requires(Z::mod() <= (std::numeric_limits<uint32>::max)()) && (Tile > 0)
	constexpr std::vector<Z> matrix_product_by_lazy_tiles(std::span<Z const> a, std::span<Z const> b, uint32 n) {
		atl_assert(n == 0 || std::size_t(n) <= std::vector<Z>().max_size() / n);
		atl_assert(a.size() == std::size_t(n) * n && b.size() == a.size());
		if (n == 0) return {};
		constexpr uint64 mod = Z::mod(), batch = detail::lazy_batch<Z>();
		std::vector<uint32> lhs(a.size()), rhs(b.size());
		for (std::size_t i = 0; i < a.size(); ++i) lhs[i] = uint32(a[i].value());
		for (std::size_t i = 0; i < b.size(); ++i) rhs[i] = uint32(b[i].value());
		std::vector<Z> c(std::size_t(n) * n);
		std::array<uint64, Tile> acc{};
		for (uint32 j0 = 0; j0 < n; j0 += (std::min)(Tile, n - j0)) {
			uint32 const width = (std::min)(Tile, n - j0);
			for (uint32 i = 0; i < n; ++i) {
				std::ranges::fill(std::span(acc).first(width), uint64(0));
				uint32 const *x = lhs.data() + std::size_t(i) * n;
				for (uint32 k0 = 0; k0 < n;) {
					uint32 const k1 = k0 + uint32((std::min<uint64>)(batch, n - k0));
					for (uint32 k = k0; k < k1; ++k) {
						uint64 const xk = x[k];
						uint32 const *y = rhs.data() + std::size_t(k) * n + j0;
						for (uint32 j = 0; j < width; ++j) acc[j] += xk * y[j];
					}
					k0 = k1;
					if (k0 < n)
						for (uint32 j = 0; j < width; ++j) acc[j] %= mod;
				}
				for (uint32 j = 0; j < width; ++j) c[std::size_t(i) * n + j0 + j] = Z::unchecked(acc[j] % mod);
			}
		}
		return c;
	}

	/**
	 * @brief Lazy i-k-j with a branch-free per-term bound instead of periodic
	 *        `%`: with each add, an accumulator at or above `8 mod²` drops by
	 *        `8 mod²` (a conditional subtract the compiler turns into a
	 *        vector compare-and-mask), so no 64-bit division runs in the loop.
	 *
	 * Lost: 7.2 / 24.0 / 110 ms (-O2), 2.5 / 9.5 / 42.7 (native): a compare
	 * and select on every term costs more than one `%` per 18 terms.
	 * @complexity O(n³)
	 */
	template<static_mod_int Z>
		requires(Z::mod() < (uint64(1) << 30))
	constexpr std::vector<Z> matrix_product_by_lazy_subtract(std::span<Z const> a, std::span<Z const> b, uint32 n) {
		atl_assert(n == 0 || std::size_t(n) <= std::vector<Z>().max_size() / n);
		atl_assert(a.size() == std::size_t(n) * n && b.size() == a.size());
		if (n == 0) return {};
		constexpr uint64 mod = Z::mod(), bound = 8 * mod * mod;
		std::vector<uint32> lhs(a.size()), rhs(b.size());
		for (std::size_t i = 0; i < a.size(); ++i) lhs[i] = uint32(a[i].value());
		for (std::size_t i = 0; i < b.size(); ++i) rhs[i] = uint32(b[i].value());
		std::vector<Z> c(std::size_t(n) * n);
		std::vector<uint64> accumulator(n);
		for (uint32 i = 0; i < n; ++i) {
			std::ranges::fill(accumulator, uint64(0));
			uint64 *acc = accumulator.data();
			uint32 const *x = lhs.data() + std::size_t(i) * n;
			for (uint32 k = 0; k < n; ++k) {
				uint64 const xk = x[k];
				uint32 const *y = rhs.data() + std::size_t(k) * n;
				for (uint32 j = 0; j < n; ++j) {
					uint64 const s = acc[j] + xk * y[j];
					acc[j] = s >= bound ? s - bound : s;
				}
			}
			for (uint32 j = 0; j < n; ++j) c[std::size_t(i) * n + j] = Z::unchecked(acc[j] % mod);
		}
		return c;
	}
}
