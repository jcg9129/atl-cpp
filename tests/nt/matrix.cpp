#include <atl/experimental/nt/matrix_variants.hpp>
#include <atl/nt/matrix.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/power.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/int128.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <limits>
#include <random>
#include <span>
#include <vector>
#include "../expect_abort.hpp"

using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	using mint = atl::mod_int32<998244353>;

	template<atl::modular_integer Z>
	atl::matrix<Z> random_matrix(uint32 rows, uint32 columns, std::mt19937_64 &rng) {
		atl::matrix<Z> m(rows, columns);
		for (auto &x : m.entries()) x = Z(rng());
		return m;
	}

	// Independent oracle: plain residues, one % per term.
	template<atl::modular_integer Z>
	std::vector<uint64> reference_product(atl::matrix<Z> const &a, atl::matrix<Z> const &b) {
		uint64 const mod = Z::mod();
		std::vector<uint64> c(std::size_t(a.rows()) * b.columns());
		for (uint32 i = 0; i < a.rows(); ++i)
			for (uint32 j = 0; j < b.columns(); ++j) {
				uint64 sum = 0;
				for (uint32 k = 0; k < a.columns(); ++k)
					sum = uint64((atl::uint128(sum) + atl::uint128(a(i, k).value()) * b(k, j).value()) % mod);
				c[std::size_t(i) * b.columns() + j] = sum;
			}
		return c;
	}

	template<atl::modular_integer Z>
	bool matches(atl::matrix<Z> const &c, std::vector<uint64> const &expected) {
		for (std::size_t i = 0; i < expected.size(); ++i)
			if (c.entries()[i].value() != expected[i]) return false;
		return true;
	}

	template<typename Semiring>
	concept powerable_matrix = atl::semiring<Semiring> && requires(atl::matrix<typename Semiring::value_type, Semiring> m) {
		atl::power(m, uint64(3));
	};
}

static_assert(atl::semiring<atl::plus_times_semiring<mint>>);
static_assert(atl::semiring<atl::min_plus_semiring<int64>>);
static_assert(atl::semiring<atl::max_min_semiring<int>>);
static_assert(!atl::semiring<int>);
static_assert(powerable_matrix<atl::min_plus_semiring<int64>>);
static_assert(powerable_matrix<atl::plus_times_semiring<mint>>);
static_assert(atl::detail::lazy_matrix_product<mint, atl::plus_times_semiring<mint>>);
static_assert(atl::detail::lazy_matrix_product<atl::mod_int32<4294967291u>, atl::plus_times_semiring<atl::mod_int32<4294967291u>>>);
static_assert(!atl::detail::lazy_matrix_product<atl::mod_int64<(1ull << 61) - 1>,
												atl::plus_times_semiring<atl::mod_int64<(1ull << 61) - 1>>>);

TEST_CASE("matrix: construction, access, transpose, identity", "[matrix]") {
	atl::matrix<int64> m{{1, 2, 3}, {4, 5, 6}};
	REQUIRE(m.rows() == 2);
	REQUIRE(m.columns() == 3);
	REQUIRE(m(1, 2) == 6);
	REQUIRE(std::ranges::equal(m.row(1), std::vector<int64>{4, 5, 6}));
	REQUIRE(m.transpose() == atl::matrix<int64>{{1, 4}, {2, 5}, {3, 6}});
	REQUIRE(atl::matrix<int64>::identity(2) == atl::matrix<int64>{{1, 0}, {0, 1}});
	REQUIRE(m * atl::matrix<int64>::identity(3) == m);
	REQUIRE(m + m == atl::matrix<int64>{{2, 4, 6}, {8, 10, 12}});
	std::vector<int64> const v{1, 0, -1};
	REQUIRE(m * std::span<int64 const>(v) == std::vector<int64>{-2, -2});
	atl::matrix<int64> empty;
	REQUIRE(empty.rows() == 0);
	REQUIRE(empty * empty == empty);
	REQUIRE(atl::power(atl::matrix<int64>(0, 0), 5u) == empty);

	REQUIRE(atl::matrix<int64>(2, 0) * atl::matrix<int64>(0, 3) == atl::matrix<int64>(2, 3));
}

TEST_CASE("matrix: modular products against an independent oracle", "[matrix]") {
	std::mt19937_64 rng(1);
	for (int round = 0; round < 200; ++round) {
		uint32 const r = uint32(rng() % 40), k = uint32(rng() % 40), c = uint32(rng() % 40);
		auto const a = random_matrix<mint>(r, k, rng), b = random_matrix<mint>(k, c, rng);
		REQUIRE(matches(a * b, reference_product(a, b)));
	}
	// moduli at the edges of the lazy kernel: tiny, at 2^32, and above it (generic path)
	using tiny = atl::mod_int32<2>;
	using wide = atl::mod_int32<4294967291u>;
	using huge = atl::mod_int64<18446744073709551557ull>;
	for (int round = 0; round < 30; ++round) {
		uint32 const n = uint32(1 + rng() % 70);
		auto const a2 = random_matrix<tiny>(n, n, rng), b2 = random_matrix<tiny>(n, n, rng);
		REQUIRE(matches(a2 * b2, reference_product(a2, b2)));
		auto const aw = random_matrix<wide>(n, n, rng), bw = random_matrix<wide>(n, n, rng);
		REQUIRE(matches(aw * bw, reference_product(aw, bw)));
		auto const ah = random_matrix<huge>(n, n, rng), bh = random_matrix<huge>(n, n, rng);
		REQUIRE(matches(ah * bh, reference_product(ah, bh)));
	}
	// all entries mod - 1 over a long inner dimension: the accumulator worst case
	for (uint32 const n : {17u, 18u, 19u, 37u, 100u}) {
		atl::matrix<mint> a(n, n, mint(-1));
		REQUIRE(matches(a * a, reference_product(a, a)));
		atl::matrix<wide> w(n, n, wide(-1));
		REQUIRE(matches(w * w, reference_product(w, w)));
	}
}

TEST_CASE("matrix: product variants agree with the oracle", "[matrix]") {
	std::mt19937_64 rng(2);
	namespace ex = atl::experimental;
	for (uint32 const n : {1u, 2u, 17u, 63u, 64u, 255u, 257u, 300u}) {
		auto const a = random_matrix<mint>(n, n, rng), b = random_matrix<mint>(n, n, rng);
		std::span<mint const> const sa = a.entries(), sb = b.entries();
		auto const oracle = ex::matrix_product_by_montgomery_ikj<mint>(sa, sb, n);
		REQUIRE(std::ranges::equal((a * b).entries(), oracle));
		REQUIRE(ex::matrix_product_by_montgomery_transpose<mint>(sa, sb, n) == oracle);
		REQUIRE(ex::matrix_product_by_lazy_transpose<mint>(sa, sb, n) == oracle);
		REQUIRE(ex::matrix_product_by_lazy_tiles<mint>(sa, sb, n) == oracle);
		REQUIRE(ex::matrix_product_by_lazy_subtract<mint>(sa, sb, n) == oracle);
	}
}

TEST_CASE("matrix: power over the ring and over min-plus / max-min", "[matrix]") {
	std::mt19937_64 rng(3);

	atl::matrix<mint> fib{{1, 1}, {1, 0}};
	REQUIRE(atl::power(fib, 10u)(0, 1) == mint(55));
	REQUIRE(atl::power(fib, 0u) == atl::matrix<mint>::identity(2));
	for (int round = 0; round < 50; ++round) {
		uint32 const n = uint32(1 + rng() % 8);
		auto const a = random_matrix<mint>(n, n, rng);
		uint64 const e = rng() % 20;
		auto slow = atl::matrix<mint>::identity(n);
		for (uint64 i = 0; i < e; ++i) slow = slow * a;
		REQUIRE(atl::power(a, e) == slow);
	}

	using shortest = atl::min_plus_semiring<int64>;
	using widest = atl::max_min_semiring<int>;
	int64 const inf = shortest::zero();
	for (int round = 0; round < 100; ++round) {
		uint32 const n = uint32(1 + rng() % 6);
		atl::matrix<int64, shortest> w(n, n);
		atl::matrix<int, widest> cap(n, n);
		for (uint32 i = 0; i < n; ++i)
			for (uint32 j = 0; j < n; ++j)
				if (rng() % 3) w(i, j) = int64(rng() % 100), cap(i, j) = int(rng() % 100);
		uint32 const k = uint32(rng() % 7);
		// DP over walks with exactly k edges
		std::vector<std::vector<int64>> best(n, std::vector<int64>(n, inf));
		std::vector<std::vector<int>> wide(n, std::vector<int>(n, widest::zero()));
		for (uint32 i = 0; i < n; ++i) best[i][i] = 0, wide[i][i] = widest::one();
		for (uint32 step = 0; step < k; ++step) {
			auto nb = std::vector<std::vector<int64>>(n, std::vector<int64>(n, inf));
			auto nw = std::vector<std::vector<int>>(n, std::vector<int>(n, widest::zero()));
			for (uint32 i = 0; i < n; ++i)
				for (uint32 m = 0; m < n; ++m)
					for (uint32 j = 0; j < n; ++j) {
						if (best[i][m] != inf && w(m, j) != inf) nb[i][j] = (std::min)(nb[i][j], best[i][m] + w(m, j));
						nw[i][j] = (std::max)(nw[i][j], (std::min)(wide[i][m], cap(m, j)));
					}
			best = nb, wide = nw;
		}
		auto const pw = atl::power(w, k);
		auto const pc = atl::power(cap, k);
		for (uint32 i = 0; i < n; ++i)
			for (uint32 j = 0; j < n; ++j) {
				REQUIRE(pw(i, j) == best[i][j]);
				REQUIRE(pc(i, j) == wide[i][j]);
			}
	}

	atl::matrix<int64, atl::max_plus_semiring<int64>> lp{{0, 5}, {atl::max_plus_semiring<int64>::zero(), 0}};
	REQUIRE(atl::power(lp, 2u)(0, 1) == 5);
}

TEST_CASE("matrix: asserted preconditions", "[matrix]") {
	CHECK(atl_test::aborts([] { (void)(atl::matrix<int64>(2, 3) * atl::matrix<int64>(2, 3)); }));
	CHECK(atl_test::aborts([] {
		atl::matrix<int64> a(2, 3);
		a += atl::matrix<int64>(3, 2);
	}));
	CHECK(atl_test::aborts([] { (void)atl::power(atl::matrix<int64>(2, 3), 2u); }));
	CHECK(atl_test::aborts([] { (void)atl::matrix<int64>(2, 2)(2, 0); }));
	CHECK(atl_test::aborts([] { (void)atl::matrix<int64>{{1, 2}, {3}}; }));
	CHECK(atl_test::aborts([] {
		std::vector<int64> v(3);
		(void)(atl::matrix<int64>(2, 2) * std::span<int64 const>(v));
	}));
	[]<typename Size>() {
		if constexpr (std::numeric_limits<Size>::digits > 32)
			CHECK(atl_test::aborts([] { (void)atl::matrix<int64>(Size(uint64(1) << 32), 0); }));
	}.template operator()<std::size_t>();
	[]<typename Size>() {
		if constexpr (std::numeric_limits<Size>::digits == 32)
			CHECK(atl_test::aborts([] { (void)atl::matrix<int>(65536, 65536); }));
	}.template operator()<std::size_t>();
}
