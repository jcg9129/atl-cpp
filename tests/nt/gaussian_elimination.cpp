#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/gaussian_elimination.hpp>
#include <atl/nt/matrix.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/int128.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <numeric>
#include <random>
#include <span>
#include <vector>
#include "../expect_abort.hpp"

using atl::int128;
using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	using mint = atl::mod_int32<998244353>;

	static_assert(atl::exact_field<mint>);
	static_assert(!atl::exact_field<int>);
	static_assert(!atl::exact_field<double>);

	template<atl::modular_integer Z>
	atl::matrix<Z> random_matrix(uint32 rows, uint32 columns, std::mt19937_64 &rng, uint64 range) {
		atl::matrix<Z> m(rows, columns);
		for (auto &x : m.entries()) x = Z(rng() % range);
		return m;
	}

	// Leibniz expansion over int128 (entries small): the determinant oracle
	int128 leibniz(std::vector<std::vector<int64>> const &a) {
		std::size_t const n = a.size();
		std::vector<std::size_t> perm(n);
		std::iota(perm.begin(), perm.end(), 0);
		int128 total = 0;
		do {
			int128 term = 1;
			for (std::size_t i = 0; i < n; ++i) term *= a[i][perm[i]];
			int inversions = 0;
			for (std::size_t i = 0; i < n; ++i)
				for (std::size_t j = i + 1; j < n; ++j) inversions += perm[i] > perm[j];
			total += inversions % 2 ? -term : term;
		} while (std::ranges::next_permutation(perm).found);
		return total;
	}

	template<atl::modular_integer Z>
	std::vector<std::vector<int64>> as_integers(atl::matrix<Z> const &m) {
		std::vector<std::vector<int64>> out(m.rows(), std::vector<int64>(m.columns()));
		for (uint32 i = 0; i < m.rows(); ++i)
			for (uint32 j = 0; j < m.columns(); ++j) out[i][j] = int64(m(i, j).value());
		return out;
	}

	template<typename Z>
		requires(atl::exact_field<Z> || std::floating_point<Z>)
	std::vector<Z> apply_matrix(atl::matrix<Z> const &a, std::vector<Z> const &x) {
		return a * std::span<Z const>(x);
	}

	// every x in GF(p)^n: the number of solutions of A x = b
	template<atl::modular_integer Z>
	uint64 count_solutions(atl::matrix<Z> const &a, std::vector<Z> const &b) {
		uint32 const n = a.columns();
		uint64 const p = Z::mod();
		uint64 total = 1, count = 0;
		for (uint32 i = 0; i < n; ++i) total *= p;
		std::vector<Z> x(n);
		for (uint64 code = 0; code < total; ++code) {
			uint64 c = code;
			for (uint32 i = 0; i < n; ++i) x[i] = Z(c % p), c /= p;
			count += apply_matrix(a, x) == b;
		}
		return count;
	}
}

TEST_CASE("gaussian elimination: determinant vs Leibniz", "[gaussian_elimination]") {
	std::mt19937_64 rng(10);
	REQUIRE(atl::determinant(atl::matrix<mint>()) == mint(1));
	for (int round = 0; round < 400; ++round) {
		uint32 const n = uint32(1 + rng() % 6);
		uint64 const range = round % 2 ? 3 : 1000; // small ranges hit singular matrices
		auto const a = random_matrix<mint>(n, n, rng, range);
		int128 const exact = leibniz(as_integers(a));
		int128 const mod = mint::mod();
		REQUIRE(atl::determinant(a).value() == uint64(((exact % mod) + mod) % mod));
		REQUIRE(atl::determinant_any_modulus(a) == atl::determinant(a));
	}
}

TEST_CASE("gaussian elimination: determinant over any modulus vs Leibniz", "[gaussian_elimination]") {
	std::mt19937_64 rng(11);
	auto check = [&]<atl::modular_integer Z>(Z, int rounds) {
		for (int round = 0; round < rounds; ++round) {
			uint32 const n = uint32(1 + rng() % 6);
			auto const a = random_matrix<Z>(n, n, rng, round % 2 ? 4 : Z::mod());
			auto ints = as_integers(a);
			// reduce entries so the int128 Leibniz sum cannot overflow
			for (auto &row : ints)
				for (auto &x : row) x %= 1'000'003;
			atl::matrix<Z> reduced(n, n);
			for (uint32 i = 0; i < n; ++i)
				for (uint32 j = 0; j < n; ++j) reduced(i, j) = Z(ints[i][j]);
			int128 const exact = leibniz(ints), mod = int128(Z::mod());
			REQUIRE(uint64(atl::determinant_any_modulus(reduced).value()) == uint64(((exact % mod) + mod) % mod));
		}
	};
	check(atl::mod_int32<1>{}, 20);
	check(atl::mod_int32<2>{}, 100);
	check(atl::mod_int32<4>{}, 200);
	check(atl::mod_int32<12>{}, 200);
	check(atl::mod_int32<1u << 31>{}, 200);
	check(atl::mod_int32<1'000'000'000>{}, 200);
	check(atl::mod_int64<1'000'000'000'000'000'000ull>{}, 200);
	using dyn = atl::dynamic_mod_int<>;
	dyn::set_mod(360);
	check(dyn{}, 200);

	dyn::set_mod(1'000'000'007);
	check(dyn{}, 100);
}

TEST_CASE("gaussian elimination: rank, solve and null space vs enumeration over GF(p)",
		  "[gaussian_elimination]") {
	std::mt19937_64 rng(12);
	auto check = [&]<atl::modular_integer Z>(Z, int rounds) {
		for (int round = 0; round < rounds; ++round) {
			uint32 const rows = uint32(rng() % 5), n = uint32(rng() % 5);
			auto const a = random_matrix<Z>(rows, n, rng, Z::mod());
			std::vector<Z> b(rows);
			for (auto &x : b) x = Z(rng() % Z::mod());
			if (round % 3 == 0) b = apply_matrix(a, std::vector<Z>(n, Z(1)));
			uint64 const solutions = count_solutions(a, b), kernel = count_solutions(a, std::vector<Z>(rows));
			uint32 const r = atl::rank(a);
			REQUIRE(r <= (std::min)(rows, n));
			uint64 expected_kernel = 1;
			for (uint32 i = r; i < n; ++i) expected_kernel *= Z::mod();
			REQUIRE(kernel == expected_kernel);

			auto const s = atl::solve_linear_system(a, b);
			REQUIRE(s.is_solvable == (solutions > 0));
			REQUIRE(s.null_space.rows() == n - r);
			REQUIRE(s.null_space.columns() == n);
			REQUIRE(s.is_unique() == (solutions == 1));
			if (s.is_solvable)
				REQUIRE(apply_matrix(a, s.particular) == b);
			else
				REQUIRE(s.particular.empty());
			for (uint32 k = 0; k < s.null_space.rows(); ++k) {
				std::vector<Z> v(s.null_space.row(k).begin(), s.null_space.row(k).end());
				REQUIRE(apply_matrix(a, v) == std::vector<Z>(rows));
			}
			REQUIRE(atl::rank(s.null_space) == s.null_space.rows());
		}
	};
	check(atl::mod_int32<2>{}, 500);
	check(atl::mod_int32<3>{}, 500);
	check(atl::mod_int32<5>{}, 300);
}

TEST_CASE("gaussian elimination: inverse", "[gaussian_elimination]") {
	std::mt19937_64 rng(13);
	REQUIRE(atl::inverse(atl::matrix<mint>()).has_value());
	for (int round = 0; round < 300; ++round) {
		uint32 const n = uint32(1 + rng() % 30);
		auto const a = random_matrix<mint>(n, n, rng, round % 3 ? mint::mod() : 2);
		auto const inv = atl::inverse(a);
		REQUIRE(inv.has_value() == !(atl::determinant(a) == mint(0)));
		if (inv) {
			REQUIRE(a * *inv == atl::matrix<mint>::identity(n));
			REQUIRE(*inv * a == atl::matrix<mint>::identity(n));
		}
	}

	uint32 const n = 200;
	auto const a = random_matrix<mint>(n, n, rng, mint::mod());
	std::vector<mint> x(n);
	for (auto &v : x) v = mint(rng());
	auto const s = atl::solve_linear_system(a, apply_matrix(a, x));
	REQUIRE(s.is_unique());
	REQUIRE(s.particular == x);
}

TEST_CASE("gaussian elimination: floating point with partial pivoting", "[gaussian_elimination]") {
	std::mt19937_64 rng(14);
	std::uniform_real_distribution<double> unit(-1, 1);
	for (int round = 0; round < 200; ++round) {
		uint32 const n = uint32(1 + rng() % 12);
		atl::matrix<double> a(n, n);
		for (auto &v : a.entries()) v = unit(rng);
		std::vector<double> x(n);
		for (auto &v : x) v = unit(rng);
		auto const s = atl::solve_linear_system(a, apply_matrix(a, x), 1e-12);
		REQUIRE(s.is_unique());
		for (uint32 i = 0; i < n; ++i) REQUIRE(std::abs(s.particular[i] - x[i]) < 1e-6);
		auto const inv = atl::inverse(a, 1e-12);
		REQUIRE(inv.has_value());
		auto const product = a * *inv;
		for (uint32 i = 0; i < n; ++i)
			for (uint32 j = 0; j < n; ++j) REQUIRE(std::abs(product(i, j) - (i == j)) < 1e-6);

		REQUIRE(std::abs(atl::determinant(a * a) - atl::determinant(a) * atl::determinant(a)) < 1e-9 * (1 + std::abs(atl::determinant(a * a))));

		uint32 const r = uint32(rng() % (n + 1)), m = uint32(1 + rng() % 12);
		atl::matrix<double> left(n, r), right(r, m);
		for (auto &v : left.entries()) v = unit(rng);
		for (auto &v : right.entries()) v = unit(rng);
		auto const low = left * right;
		REQUIRE(atl::rank(low, 1e-9) == (std::min)({r, n, m}));
		auto const ls = atl::solve_linear_system(low, std::vector<double>(n, 0.0), 1e-9);
		REQUIRE(ls.is_solvable);
		REQUIRE(ls.null_space.rows() == m - (std::min)({r, n, m}));
	}
	// pivoting matters: a zero leading entry, and a tiny one
	atl::matrix<double> const tricky{{1e-20, 1}, {1, 1}};
	auto const s = atl::solve_linear_system(tricky, {1.0, 2.0}, 1e-15);
	REQUIRE(std::abs(s.particular[0] - 1) < 1e-12);
	REQUIRE(std::abs(s.particular[1] - 1) < 1e-12);
	REQUIRE(std::abs(atl::determinant(atl::matrix<double>{{0, 1}, {1, 0}}) + 1.0) < 1e-12);
	REQUIRE(!atl::inverse(atl::matrix<double>{{1, 2}, {2, 4}}, 1e-12).has_value());
	auto const inconsistent = atl::solve_linear_system(atl::matrix<double>{{1, 2}, {2, 4}}, {1.0, 3.0}, 1e-12);
	REQUIRE(!inconsistent.is_solvable);
}

TEST_CASE("gaussian elimination: asserted preconditions", "[gaussian_elimination]") {
	CHECK(atl_test::aborts([] { (void)atl::determinant(atl::matrix<mint>(2, 3)); }));
	CHECK(atl_test::aborts([] { (void)atl::determinant_any_modulus(atl::matrix<mint>(2, 3)); }));
	CHECK(atl_test::aborts([] { (void)atl::inverse(atl::matrix<mint>(3, 2)); }));
	CHECK(atl_test::aborts([] { (void)atl::solve_linear_system(atl::matrix<mint>(2, 2), std::vector<mint>(3)); }));
	// a composite modulus is not a field: the first non-unit pivot aborts in inverse()
	CHECK(atl_test::aborts([] {
		using z4 = atl::mod_int32<4>;
		(void)atl::determinant(atl::matrix<z4>{{2, 1}, {0, 1}});
	}));
}

TEST_CASE("gaussian elimination: full-width modular residues", "[gaussian_elimination]") {
	using prime = atl::mod_int64<18446744073709551557ull>;
	atl::matrix<prime> const a{{-1, -2}, {-3, -4}};
	CHECK(atl::determinant(a) == prime(-2));
	CHECK(atl::determinant_any_modulus(a) == prime(-2));
	auto const inv = atl::inverse(a);
	REQUIRE(inv);
	CHECK(*inv * a == atl::matrix<prime>::identity(2));
	using composite = atl::mod_int64<18446744073709551614ull>;
	CHECK(atl::determinant_any_modulus(atl::matrix<composite>{{-1, -2}, {-3, -4}}) == composite(-2));
	using wide = atl::mod_int<atl::uint128, atl::uint128(1) << 100>;
	atl::matrix<wide> const w{{wide((atl::uint128(1) << 90) + 7), 1}, {3, 1}};
	CHECK(atl::determinant_any_modulus(w) == wide((atl::uint128(1) << 90) + 4));
}
