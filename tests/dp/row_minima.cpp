#include <atl/dp/row_minima.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"
#include "monge_costs.hpp"

using atl::int64;
using atl::uint32;

namespace {
	using atl_test::matrix;
	using atl_test::random_monge;

	/**
	 * @brief A monotone but not totally monotone matrix: each row has a
	 *        unique minimum at a nondecreasing column, everything else is
	 *        random and larger.
	 */
	matrix random_monotone(std::size_t rows, std::size_t cols, std::mt19937_64 &rng) {
		std::vector<std::size_t> at(rows);
		for (auto &x : at) x = std::size_t(rng() % cols);
		std::ranges::sort(at);
		matrix a(rows, std::vector<int64>(cols));
		for (std::size_t i = 0; i < rows; ++i)
			for (std::size_t j = 0; j < cols; ++j) a[i][j] = j == at[i] ? int64(rng() % 5) : 10 + int64(rng() % 20);
		return a;
	}

	template<typename Compare = std::less<>>
		requires std::strict_weak_order<Compare const &, int64, int64>
	void check(matrix const &a, std::size_t cols, atl::row_minima_result<int64> const &got, Compare comp = {}) {
		REQUIRE(got.column.size() == a.size());
		REQUIRE(got.value.size() == a.size());
		for (std::size_t i = 0; i < a.size(); ++i) {
			std::size_t best = 0;
			for (std::size_t j = 1; j < cols; ++j)
				if (comp(a[i][j], a[i][best])) best = j;
			CHECK(got.column[i] == best);
			CHECK(got.value[i] == a[i][best]);
		}
	}
}

static_assert(atl::matrix_cost<decltype([](uint32, uint32) { return 0; }), std::less<>>);
static_assert(atl::matrix_cost<decltype([](uint32 i, uint32 j) mutable { return double(i + j); }), std::greater<>>);
static_assert(!atl::matrix_cost<decltype([](uint32) { return 0; }), std::less<>>);
static_assert(!atl::matrix_cost<decltype([](uint32, uint32) {}), std::less<>>);
static_assert(atl::additive_cost<decltype([](uint32, uint32) { return int64(0); }), std::less<>>);

TEST_CASE("row minima of random Monge matrices match a brute force", "[row_minima]") {
	std::mt19937_64 rng(7);
	for (int iter = 0; iter < 3000; ++iter) {
		std::size_t const rows = std::size_t(rng() % 12), cols = 1 + std::size_t(rng() % 12);
		int64 const density = iter % 3 == 0 ? 1 : 6, offset = iter % 2 == 0 ? 0 : 20;
		matrix const a = random_monge(rows, cols, rng, density, offset);
		auto const cost = [&a](uint32 i, uint32 j) { return a[i][j]; };
		check(a, cols, atl::monotone_row_minima(rows, cols, cost));
		check(a, cols, atl::totally_monotone_row_minima(rows, cols, cost));
	}
}

TEST_CASE("row maxima of inverse-Monge matrices via std::greater", "[row_minima]") {
	std::mt19937_64 rng(8);
	for (int iter = 0; iter < 1000; ++iter) {
		std::size_t const rows = 1 + std::size_t(rng() % 10), cols = 1 + std::size_t(rng() % 10);
		matrix a = random_monge(rows, cols, rng, 3, 10);
		for (auto &row : a)
			for (auto &x : row) x = -x;
		auto const cost = [&a](uint32 i, uint32 j) { return a[i][j]; };
		check(a, cols, atl::monotone_row_minima(rows, cols, cost, std::greater<>{}), std::greater<>{});
		check(a, cols, atl::totally_monotone_row_minima(rows, cols, cost, std::greater<>{}), std::greater<>{});
	}
}

TEST_CASE("monotone_row_minima needs only monotone minima", "[row_minima]") {
	// SMAWK's stricter precondition is load-bearing: on matrices that are
	// monotone but not totally monotone it misses some rows
	std::mt19937_64 rng(9);
	int smawk_misses = 0;
	for (int iter = 0; iter < 1000; ++iter) {
		std::size_t const rows = 1 + std::size_t(rng() % 15), cols = 1 + std::size_t(rng() % 15);
		matrix const a = random_monotone(rows, cols, rng);
		auto const cost = [&a](uint32 i, uint32 j) { return a[i][j]; };
		auto const divided = atl::monotone_row_minima(rows, cols, cost);
		check(a, cols, divided);
		smawk_misses += atl::totally_monotone_row_minima(rows, cols, cost).column != divided.column;
	}
	CHECK(smawk_misses > 0);
}

TEST_CASE("row minima on a million rows stay within a shallow stack", "[row_minima]") {
	// both recursions halve the rows, so 1e6 rows nest about 20 frames deep,
	// well inside a 1 MiB stack; A[i][j] = (i - 300000 j)^2 is Monge, and a
	// row halfway between two columns takes the left one
	std::size_t const rows = 1'000'000, cols = 4;
	auto const cost = [](uint32 i, uint32 j) {
		int64 const d = int64(i) - 300'000 * int64(j);
		return d * d;
	};
	auto const tm = atl::totally_monotone_row_minima(rows, cols, cost);
	auto const dc = atl::monotone_row_minima(rows, cols, cost);
	REQUIRE(tm.column == dc.column);
	for (std::size_t i = 0; i < rows; i += 9973) {
		auto const want = uint32((std::min)((i + 149'999) / 300'000, cols - 1));
		CHECK(tm.column[i] == want);
	}
}

TEST_CASE("row minima on large shapes and evaluation counts", "[row_minima]") {
	// A[i][j] = (x_i - y_j)^2 with sorted x, y is Monge.
	std::mt19937_64 rng(10);
	for (auto [rows, cols] : {std::pair<std::size_t, std::size_t>{3000, 50}, {50, 3000}, {2000, 2000}, {1, 5000}, {5000, 1}}) {
		std::vector<int64> x(rows), y(cols);
		for (auto &v : x) v = int64(rng() % 100000);
		for (auto &v : y) v = int64(rng() % 100000);
		std::ranges::sort(x);
		std::ranges::sort(y);
		std::size_t evaluations = 0;
		auto cost = [&](uint32 i, uint32 j) {
			++evaluations;
			return (x[i] - y[j]) * (x[i] - y[j]);
		};
		auto const tm = atl::totally_monotone_row_minima(rows, cols, cost);
		CHECK(evaluations <= 6 * (rows + cols));
		auto const dc = atl::monotone_row_minima(rows, cols, cost);
		CHECK(tm.column == dc.column);
		CHECK(tm.value == dc.value);
		for (std::size_t i = 0; i < rows; i += 97) {
			std::size_t best = 0;
			for (std::size_t j = 1; j < cols; ++j)
				if (cost(uint32(i), uint32(j)) < cost(uint32(i), uint32(best))) best = j;
			CHECK(tm.column[i] == best);
		}
	}
}

TEST_CASE("row minima edge shapes", "[row_minima]") {
	auto const cost = [](uint32 i, uint32 j) { return int64(i) - int64(j); };
	CHECK(atl::monotone_row_minima(0, 0, cost).column.empty());
	CHECK(atl::totally_monotone_row_minima(0, 5, cost).value.empty());
	auto const one = atl::totally_monotone_row_minima(1, 1, cost);
	CHECK(one.column == std::vector<uint32>{0});
	CHECK(one.value == std::vector<int64>{0});
	// all-equal matrix: every row takes column 0
	auto const flat = [](uint32, uint32) { return 3; };
	CHECK(atl::totally_monotone_row_minima(7, 9, flat).column == std::vector<uint32>(7, 0));
	CHECK(atl::monotone_row_minima(7, 9, flat).column == std::vector<uint32>(7, 0));
}

TEST_CASE("row minima reject a matrix with rows but no columns", "[row_minima]") {
	auto const cost = [](uint32, uint32) { return 0; };
	CHECK(atl_test::aborts([&] { (void)atl::monotone_row_minima(3, 0, cost); }));
	CHECK(atl_test::aborts([&] { (void)atl::totally_monotone_row_minima(3, 0, cost); }));
}
