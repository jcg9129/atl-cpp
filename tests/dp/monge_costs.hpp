#pragma once
#include <atl/types/alias.hpp>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace atl_test {
	using matrix = std::vector<std::vector<atl::int64>>;

	/**
	 * @brief A random Monge matrix: `A[i][j] = u[i] + v[j] + sum_{a <= i, b >=
	 *        j} D[a][b]` with `D >= 0`, so `A[i][j] + A[i+1][j+1] - A[i][j+1]
	 *        - A[i+1][j] = -D[i+1][j] <= 0`. Small ranges make ties common.
	 */
	inline matrix random_monge(std::size_t rows, std::size_t cols, std::mt19937_64 &rng, atl::int64 density,
							   atl::int64 offset) {
		auto const draw = [&rng](atl::int64 range) { return atl::int64(rng() % std::uint64_t(range + 1)); };
		matrix s(rows, std::vector<atl::int64>(cols + 1, 0));
		for (std::size_t i = 0; i < rows; ++i)
			for (std::size_t j = cols; j-- > 0;)
				s[i][j] = s[i][j + 1] + draw(density) + (i > 0 ? s[i - 1][j] - s[i - 1][j + 1] : 0);
		matrix a(rows, std::vector<atl::int64>(cols));
		std::vector<atl::int64> u(rows), v(cols);
		for (auto &x : u) x = draw(offset) - offset / 2;
		for (auto &x : v) x = draw(offset) - offset / 2;
		for (std::size_t i = 0; i < rows; ++i)
			for (std::size_t j = 0; j < cols; ++j) a[i][j] = u[i] + v[j] + s[i][j];
		return a;
	}

	/**
	 * @brief A random segment cost on boundaries `0 .. n` satisfying the
	 *        quadrangle inequality AND monotonicity on inclusion (Knuth's
	 *        two conditions): `w[l][r] = sum_{l <= x <= y < r} D[x][y]` with
	 *        `D >= 0` -- the sub-segments of `[a, c)` and `[b, d)` are
	 *        covered by those of `[a, d)` and `[b, c)`.
	 */
	inline matrix random_interval_cost(std::size_t n, std::mt19937_64 &rng, atl::int64 density) {
		matrix d(n + 1, std::vector<atl::int64>(n + 1, 0));
		for (std::size_t x = 0; x < n; ++x)
			for (std::size_t y = x; y < n; ++y) d[x][y] = atl::int64(rng() % std::uint64_t(density + 1));
		// w[l][r] = w[l][r-1] + sum_{l <= x <= r-1} D[x][r-1]
		matrix w(n + 1, std::vector<atl::int64>(n + 1, 0));
		for (std::size_t l = 0; l <= n; ++l) {
			for (std::size_t r = l + 1; r <= n; ++r) {
				atl::int64 column = 0;
				for (std::size_t x = l; x < r; ++x) column += d[x][r - 1];
				w[l][r] = w[l][r - 1] + column;
			}
		}
		return w;
	}
}
