#include <atl/nt/bit_matrix.hpp>
#include <atl/types/alias.hpp>
#include <bit>
#include <catch2/catch.hpp>
#include <limits>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;

namespace {
	std::vector<bool> apply_matrix(atl::bit_matrix const &a, std::vector<bool> const &x) {
		std::vector<bool> out(a.rows());
		for (uint32 i = 0; i < a.rows(); ++i) {
			bool sum = false;
			for (uint32 j = 0; j < a.columns(); ++j) sum ^= a.test(i, j) && x[j];
			out[i] = sum;
		}
		return out;
	}

	atl::bit_matrix random_bits(uint32 rows, uint32 columns, std::mt19937_64 &rng, uint32 density) {
		atl::bit_matrix m(rows, columns);
		for (uint32 i = 0; i < rows; ++i)
			for (uint32 j = 0; j < columns; ++j)
				if (rng() % 100 < density) m.set(i, j);
		return m;
	}
}

TEST_CASE("bit_matrix: access", "[bit_matrix]") {
	atl::bit_matrix m(3, 130);
	m.set(2, 129);
	m.set(0, 64);
	m.flip(1, 0);
	REQUIRE(m.test(2, 129));
	REQUIRE(m.test(0, 64));
	REQUIRE(m.test(1, 0));
	REQUIRE(!m.test(1, 1));
	m.set(2, 129, false);
	m.flip(1, 0);
	atl::bit_matrix expected(3, 130);
	expected.set(0, 64);
	REQUIRE(m == expected);
	REQUIRE(m.row_words(0).size() == 3);
	REQUIRE(atl::rank(atl::bit_matrix()) == 0);
}

TEST_CASE("bit_matrix: rank and solve vs enumeration", "[bit_matrix]") {
	std::mt19937_64 rng(20);
	for (int round = 0; round < 3000; ++round) {
		uint32 const rows = uint32(rng() % 9), n = uint32(rng() % 11);
		auto const a = random_bits(rows, n, rng, uint32(rng() % 100));
		std::vector<bool> b(rows);
		for (uint32 i = 0; i < rows; ++i) b[i] = rng() & 1;
		uint64 solutions = 0, kernel = 0;
		std::vector<bool> x(n);
		for (uint64 code = 0; code < (uint64(1) << n); ++code) {
			for (uint32 j = 0; j < n; ++j) x[j] = (code >> j) & 1;
			auto const ax = apply_matrix(a, x);
			solutions += ax == b;
			kernel += ax == std::vector<bool>(rows);
		}
		uint32 const r = atl::rank(a);
		REQUIRE(kernel == uint64(1) << (n - r));
		auto const s = atl::solve_linear_system(a, b);
		REQUIRE(s.is_solvable == (solutions > 0));
		REQUIRE(s.is_unique() == (solutions == 1));
		REQUIRE(s.null_space.rows() == n - r);
		if (s.is_solvable) REQUIRE(apply_matrix(a, s.particular) == b);
		for (uint32 k = 0; k < s.null_space.rows(); ++k) {
			std::vector<bool> v(n);
			for (uint32 j = 0; j < n; ++j) v[j] = s.null_space.test(k, j);
			REQUIRE(apply_matrix(a, v) == std::vector<bool>(rows));
		}
		REQUIRE(atl::rank(s.null_space) == s.null_space.rows());
	}
}

TEST_CASE("bit_matrix: wide systems across word boundaries", "[bit_matrix]") {
	std::mt19937_64 rng(21);
	for (uint32 const n : {63u, 64u, 65u, 127u, 128u, 129u, 500u}) {
		for (int round = 0; round < 5; ++round) {
			uint32 const rows = uint32(n / 2 + rng() % n);
			auto const a = random_bits(rows, n, rng, 50);
			std::vector<bool> x(n);
			for (uint32 j = 0; j < n; ++j) x[j] = rng() & 1;
			auto const b = apply_matrix(a, x);
			auto const s = atl::solve_linear_system(a, b);
			REQUIRE(s.is_solvable);
			REQUIRE(apply_matrix(a, s.particular) == b);
			REQUIRE(s.null_space.rows() == n - atl::rank(a));
			for (uint32 k = 0; k < s.null_space.rows(); ++k) {
				std::vector<bool> v(n);
				for (uint32 j = 0; j < n; ++j) v[j] = s.null_space.test(k, j);
				REQUIRE(apply_matrix(a, v) == std::vector<bool>(rows));
			}
		}
	}
}

TEST_CASE("bit_matrix: asserted preconditions", "[bit_matrix]") {
	CHECK(atl_test::aborts([] { (void)atl::bit_matrix(2, 2).test(0, 2); }));
	CHECK(atl_test::aborts([] { atl::bit_matrix(2, 2).set(2, 0); }));
	CHECK(atl_test::aborts([] { (void)atl::solve_linear_system(atl::bit_matrix(2, 2), std::vector<bool>(1)); }));
}

TEST_CASE("bit_matrix: packed word width and zero-column shapes", "[bit_matrix]") {
	atl::bit_matrix a(1, 129);
	a.row_words(0)[0] = (std::numeric_limits<uint64>::max)();
	a.set(0, 64);
	a.set(0, 128);
	CHECK(std::popcount(a.row_words(0)[0]) == 64);
	CHECK(a.test(0, 63));
	CHECK(a.test(0, 64));
	CHECK(a.test(0, 128));
	CHECK(atl::rank(a) == 1);
	auto const consistent = atl::solve_linear_system(atl::bit_matrix(2, 0), {false, false});
	CHECK(consistent.is_unique());
	CHECK(consistent.particular.empty());
	CHECK(!atl::solve_linear_system(atl::bit_matrix(2, 0), {false, true}).is_solvable);
	CHECK(atl_test::aborts([] { (void)atl::solve_linear_system(atl::bit_matrix(0, (std::numeric_limits<uint32>::max)()), {}); }));
	[]<typename Size>() {
		if constexpr (std::numeric_limits<Size>::digits == 32)
			CHECK(atl_test::aborts([] { (void)atl::bit_matrix((std::numeric_limits<uint32>::max)(), 129); }));
	}.template operator()<std::size_t>();
}
