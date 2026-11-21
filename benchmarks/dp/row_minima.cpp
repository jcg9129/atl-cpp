/**
 * @file
 * @brief Row minima race: atl::totally_monotone_row_minima (SMAWK, O(rows +
 * cols) evaluations, totally monotone matrices) against
 * atl::monotone_row_minima (divide and conquer, O((rows + cols) log rows),
 * monotone matrices). Throughput of one whole call on square, wide and tall
 * shapes (1e5 and 1e6 rows and columns) of A[i][j] = f(x_i - y_j) over sorted
 * points, which is Monge for convex f: "cheap" = squared distance (two loads,
 * a multiply), "pow1.5" = |d|^1.5 (a sqrt per entry). Results are
 * cross-checked before timing. Xeon Platinum 8280, GCC 15.2.0,
 * RelWithDebInfo, assertions off, 2026-10-15, plain -O2 / -march=native:
 * SMAWK wins or ties every shape: 1.7-2.3x on the wide shape (1e5 x 1e6
 * cheap: 10.6 / 10.4 ms against 21.5 / 17.6) and 1.16-1.26x with the pow1.5
 * cost elsewhere; on square and tall cheap shapes it leads by at most 1.14x
 * at -O2 and ties within 3% with -march=native.
 * Full table: docs/dp-optimization.md.
 */
#include <atl/dp/row_minima.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

using atl::float64;
using atl::int64;
using atl::uint32;

TEST_CASE("row minima contenders", "[row_minima]") {
	std::mt19937_64 rng(1);
	auto points = [&rng](std::size_t n) {
		std::vector<int64> v(n);
		for (auto &x : v) x = int64(rng() % 1'000'000'000);
		std::ranges::sort(v);
		return v;
	};

	auto race = [&](std::string const &shape, std::size_t rows, std::size_t cols) {
		auto const x = points(rows), y = points(cols);
		auto const cheap = [&](uint32 i, uint32 j) {
			int64 const d = x[i] - y[j];
			return d * d;
		};
		auto const pow15 = [&](uint32 i, uint32 j) {
			float64 const d = std::abs(float64(x[i] - y[j]));
			return d * std::sqrt(d);
		};
		REQUIRE(atl::totally_monotone_row_minima(rows, cols, cheap).column == atl::monotone_row_minima(rows, cols, cheap).column);
		REQUIRE(atl::totally_monotone_row_minima(rows, cols, pow15).value == atl::monotone_row_minima(rows, cols, pow15).value);

		BENCHMARK(shape + " cheap | totally_monotone_row_minima") { return atl::totally_monotone_row_minima(rows, cols, cheap).column[0]; };
		BENCHMARK(shape + " cheap | monotone_row_minima") {
			return atl::monotone_row_minima(rows, cols, cheap).column[0];
		};
		BENCHMARK(shape + " pow1.5 | totally_monotone_row_minima") { return atl::totally_monotone_row_minima(rows, cols, pow15).column[0]; };
		BENCHMARK(shape + " pow1.5 | monotone_row_minima") {
			return atl::monotone_row_minima(rows, cols, pow15).column[0];
		};
	};

	race("1e5 x 1e5", 100'000, 100'000);
	race("1e6 x 1e6", 1'000'000, 1'000'000);
	race("1e5 x 1e6", 100'000, 1'000'000);
	race("1e6 x 1e5", 1'000'000, 100'000);
}
