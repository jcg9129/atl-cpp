#include <atl/experimental/nt/linear_recurrence_variants.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/linear_recurrence.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;
namespace ex = atl::experimental;

namespace {
	using mint = atl::mod_int32<998244353>;

	template<atl::modular_integer Z>
	std::vector<Z> random_vector(std::size_t n, std::mt19937_64 &rng) {
		std::vector<Z> v(n);
		for (auto &x : v) x = Z(rng());
		return v;
	}

	template<atl::modular_integer Z>
	bool satisfies(std::vector<Z> const &s, std::vector<Z> const &c) {
		for (std::size_t i = c.size(); i < s.size(); ++i) {
			Z sum = Z(0);
			for (std::size_t j = 0; j < c.size(); ++j) sum += c[j] * s[i - 1 - j];
			if (!(sum == s[i])) return false;
		}
		return true;
	}

	// the shortest length by enumerating every coefficient vector over GF(p)
	template<atl::modular_integer Z>
	std::size_t brute_shortest(std::vector<Z> const &s) {
		uint64 const p = Z::mod();
		for (std::size_t length = 0;; ++length) {
			uint64 total = 1;
			for (std::size_t i = 0; i < length; ++i) total *= p;
			std::vector<Z> c(length);
			for (uint64 code = 0; code < total; ++code) {
				uint64 x = code;
				for (auto &v : c) v = Z(x % p), x /= p;
				if (satisfies(s, c)) return length;
			}
		}
	}
}

TEST_CASE("berlekamp_massey: shortest length vs enumeration over GF(2), GF(3)", "[linear_recurrence]") {
	std::mt19937_64 rng(40);
	auto check = [&]<atl::modular_integer Z>(Z) {
		for (int round = 0; round < 400; ++round) {
			std::size_t const n = rng() % 9;
			std::vector<Z> s(n);
			for (auto &x : s) x = Z(rng() % Z::mod());
			auto const c = atl::berlekamp_massey(s);
			REQUIRE(satisfies(s, c));
			REQUIRE(c.size() == brute_shortest(s));
		}
	};
	check(atl::mod_int32<2>{});
	check(atl::mod_int32<3>{});
	REQUIRE(atl::berlekamp_massey(std::vector<mint>{}).empty());
	REQUIRE(atl::berlekamp_massey(std::vector<mint>(5)).empty());
	REQUIRE(atl::berlekamp_massey(std::vector<mint>{0, 0, 0, 1}).size() == 4);
}

TEST_CASE("berlekamp_massey: recovers a generating recurrence from 2d terms", "[linear_recurrence]") {
	std::mt19937_64 rng(41);
	for (int round = 0; round < 200; ++round) {
		std::size_t const d = rng() % 40;
		auto const c = random_vector<mint>(d, rng);
		auto s = random_vector<mint>(d, rng);
		for (std::size_t i = d; i < 2 * d + 30; ++i) {
			mint next = 0;
			for (std::size_t j = 0; j < d; ++j) next += c[j] * s[i - 1 - j];
			s.push_back(next);
		}
		std::vector<mint> const prefix(s.begin(), s.begin() + std::ptrdiff_t(2 * d));
		auto const found = atl::berlekamp_massey(prefix);
		REQUIRE(found.size() <= d);
		REQUIRE(satisfies(s, found));
	}

	REQUIRE(atl::berlekamp_massey(std::vector<mint>{1, 1, 2, 3, 5, 8}) == std::vector<mint>{1, 1});

	using dyn = atl::dynamic_mod_int<>;
	dyn::set_mod(1'000'000'007);
	std::vector<dyn> powers{1, 3, 9, 27, 81};
	REQUIRE(atl::berlekamp_massey(powers) == std::vector<dyn>{3});
}

TEST_CASE("linear_recurrence_term: every contender vs stepping", "[linear_recurrence]") {
	std::mt19937_64 rng(42);
	for (int round = 0; round < 300; ++round) {
		std::size_t const d = rng() % 70; // crosses convolution's naive cutoff at 48
		auto const c = random_vector<mint>(d, rng);
		auto const a = random_vector<mint>(d + rng() % 3, rng);
		uint64 const k = rng() % 400;
		mint const expected = ex::linear_recurrence_term_by_stepping(c, a, k);
		REQUIRE(atl::linear_recurrence_term(c, a, k) == expected);
		REQUIRE(ex::linear_recurrence_term_by_schoolbook_kitamasa(c, a, k) == expected);
		REQUIRE(ex::linear_recurrence_term_by_convolution_kitamasa(c, a, k) == expected);
		REQUIRE(ex::linear_recurrence_term_by_convolution_bostan_mori(c, a, k) == expected);
		REQUIRE(ex::linear_recurrence_term_by_transform_bostan_mori(c, a, k) == expected);
		REQUIRE(ex::linear_recurrence_term_by_schoolbook_bostan_mori(c, a, k) == expected);
		if (d <= 12) REQUIRE(ex::linear_recurrence_term_by_matrix_power(c, a, k) == expected);
	}
}

TEST_CASE("linear_recurrence_term: huge k, contenders agree", "[linear_recurrence]") {
	std::mt19937_64 rng(43);
	for (std::size_t const d : {1, 2, 10, 47, 48, 49, 200, 1000}) {
		auto const c = random_vector<mint>(d, rng);
		auto const a = random_vector<mint>(d, rng);
		for (uint64 const k : {uint64(1'000'000'000'000'000'000ull), uint64(-1), uint64(1) << 63}) {
			mint const library = atl::linear_recurrence_term(c, a, k);
			REQUIRE(ex::linear_recurrence_term_by_schoolbook_kitamasa(c, a, k) == library);
			REQUIRE(ex::linear_recurrence_term_by_convolution_kitamasa(c, a, k) == library);
			REQUIRE(ex::linear_recurrence_term_by_convolution_bostan_mori(c, a, k) == library);
			REQUIRE(ex::linear_recurrence_term_by_transform_bostan_mori(c, a, k) == library);
			if (d <= 200) REQUIRE(ex::linear_recurrence_term_by_schoolbook_bostan_mori(c, a, k) == library);
			if (d <= 10) REQUIRE(ex::linear_recurrence_term_by_matrix_power(c, a, k) == library);
		}
	}
	// Fibonacci mod p at 1e18, against the doubling identities via matrix power
	std::vector<mint> const fc{1, 1}, fa{0, 1};
	atl::matrix<mint> const q{{1, 1}, {1, 0}};
	REQUIRE(atl::linear_recurrence_term(fc, fa, 1'000'000'000'000'000'000ull) == atl::power(q, 1'000'000'000'000'000'000ull)(0, 1));
}

TEST_CASE("linear_recurrence_term: other moduli and edge cases", "[linear_recurrence]") {
	std::mt19937_64 rng(44);
	using prime = atl::mod_int32<1'000'000'007>; // not NTT-friendly: convolution's FFT tier
	using dyn = atl::dynamic_mod_int<>;
	dyn::set_mod(1'000'000'009);
	for (int round = 0; round < 60; ++round) {
		std::size_t const d = rng() % 120;
		uint64 const k = rng() % 600;
		auto const c = random_vector<prime>(d, rng), a = random_vector<prime>(d, rng);
		REQUIRE(atl::linear_recurrence_term(c, a, k) == ex::linear_recurrence_term_by_stepping(c, a, k));
		auto const cd = random_vector<dyn>(d % 30, rng), ad = random_vector<dyn>(d % 30, rng);
		REQUIRE(atl::linear_recurrence_term(cd, ad, k) == ex::linear_recurrence_term_by_stepping(cd, ad, k));
	}

	REQUIRE(atl::linear_recurrence_term(std::vector<mint>{}, std::vector<mint>{}, 5) == mint(0));
	REQUIRE(atl::linear_recurrence_term(std::vector<mint>{1, 1}, std::vector<mint>{4, 7}, 1) == mint(7));
	REQUIRE(atl::linear_recurrence_term(std::vector<mint>{1, 1}, std::vector<mint>{4, 7}, 0) == mint(4));
}

TEST_CASE("rational_series_coefficient vs series division", "[linear_recurrence]") {
	std::mt19937_64 rng(45);
	for (int round = 0; round < 300; ++round) {
		std::size_t const np = rng() % 60, nq = 1 + rng() % 60;
		auto const p = random_vector<mint>(np, rng);
		auto q = random_vector<mint>(nq, rng);
		if (q[0] == mint(0)) q[0] = mint(2);
		uint64 const k = rng() % 200;

		std::vector<mint> f(std::size_t(k) + 1);
		mint const inv = q[0].inverse();
		for (uint64 i = 0; i <= k; ++i) {
			mint x = i < np ? p[std::size_t(i)] : mint(0);
			for (uint64 j = 1; j < nq && j <= i; ++j) x -= q[std::size_t(j)] * f[std::size_t(i - j)];
			f[std::size_t(i)] = x * inv;
		}
		REQUIRE(atl::rational_series_coefficient(p, q, k) == f[std::size_t(k)]);
	}

	REQUIRE(atl::rational_series_coefficient(std::vector<mint>{1}, std::vector<mint>{1, -1}, uint64(-1)) == mint(1));
}

TEST_CASE("linear recurrence: asserted preconditions", "[linear_recurrence]") {
	CHECK(atl_test::aborts([] {
		(void)atl::rational_series_coefficient(std::vector<mint>{1}, std::vector<mint>{0, 1}, 3);
	}));
	CHECK(atl_test::aborts([] { (void)atl::rational_series_coefficient(std::vector<mint>{1}, std::vector<mint>{}, 3); }));
	CHECK(atl_test::aborts([] {
		(void)atl::linear_recurrence_term(std::vector<mint>{1, 1}, std::vector<mint>{1}, 3);
	}));
}

TEST_CASE("linear recurrence: order-64 periodic transform route", "[linear_recurrence]") {
	std::vector<mint> c(64), a(64);
	c[63] = mint(1);
	for (int i = 0; i < 64; ++i) a[std::size_t(i)] = mint(i);
	CHECK(atl::linear_recurrence_term(c, a, 1000u) == mint(1000 % 64));
}
