#include <atl/ds/difference_fenwick.hpp>
#include <atl/ds/dual_segment_tree.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <functional>
#include <limits>
#include <list>
#include <random>
#include <type_traits>
#include <vector>
#include "../expect_abort.hpp"

using ll = long long;
using atl::uint64;

/** @brief Certify the public Fenwick wrapper against the commutative dual tree and a brute-force array. */
template<atl::binary_operation<ll> Op, atl::inverse_operation<ll> Inverse, typename Gen>
	requires std::is_invocable_r_v<ll, Gen &, std::mt19937_64 &>
static void difference_fenwick_vs_oracle(std::mt19937_64 &rng, Gen gen, Op op, Inverse inverse) {
	for (int n : {0, 1, 2, 3, 7, 8, 9, 64, 1 + int(rng() % 200)}) {
		bool const is_from_range = rng() % 2;
		std::vector<ll> a(static_cast<std::size_t>(n), ll(7));
		if (is_from_range)
			for (auto &x : a) x = gen(rng);
		atl::dual_segment_tree dual(atl::commutative_tags, a, ll(0), op, op);
		atl::difference_fenwick<ll, Op, Inverse> fen =
			is_from_range ? atl::difference_fenwick<ll, Op, Inverse>(a, ll(0), op, inverse)
						  : atl::difference_fenwick<ll, Op, Inverse>(a.size(), ll(7), ll(0), op, inverse);
		for (int step = 0; step < 300 && n > 0; ++step) {
			if (rng() % 2) {
				int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
				ll const f = gen(rng);
				fen.apply(std::size_t(l), std::size_t(r), f);
				dual.apply(std::size_t(l), std::size_t(r), f);
				for (int j = l; j < r; ++j) a[std::size_t(j)] = op(a[std::size_t(j)], f);
			} else if (rng() % 4 == 0) {
				int const i = int(rng() % n);
				ll const f = gen(rng);
				fen.apply(std::size_t(i), f);
				dual.apply(std::size_t(i), f);
				a[std::size_t(i)] = op(a[std::size_t(i)], f);
			} else {
				int const i = int(rng() % n);
				REQUIRE(fen.at(std::size_t(i)) == a[std::size_t(i)]);
				REQUIRE(dual.at(std::size_t(i)) == a[std::size_t(i)]);
			}
		}
		REQUIRE(fen.size() == std::size_t(n));
		for (int i = 0; i < n; ++i) REQUIRE(fen.at(std::size_t(i)) == a[std::size_t(i)]);
	}
}

TEST_CASE("difference_fenwick: matches the commutative dual tree (add, xor)", "[dual_segment_tree][difference_fenwick]") {
	auto seed = GENERATE(take(20, random(uint64(0), std::numeric_limits<uint64>::max())));
	std::mt19937_64 rng(seed);
	difference_fenwick_vs_oracle(rng, [](std::mt19937_64 &g) { return ll(g() % 2001) - 1000; }, std::plus<ll>{}, std::negate<ll>{});
	difference_fenwick_vs_oracle(rng, [](std::mt19937_64 &g) { return ll(g() % 1024); }, std::bit_xor<ll>{}, std::identity{});
}

namespace {
	struct masked_xor {
		int mask;

		constexpr int operator()(int a, int b) const { return a ^ b ^ mask; }
	};

	struct shifted_add {
		int identity;

		constexpr int operator()(int a, int b) const { return a + b - identity; }
	};

	struct shifted_inverse {
		int identity;

		constexpr int operator()(int value) const { return 2 * identity - value; }
	};

	template<typename Op, typename Inverse>
	concept accepts_operations = requires { typename atl::difference_fenwick<int, Op, Inverse>; };

	static_assert(accepts_operations<std::plus<>, std::negate<>>);
	static_assert(!accepts_operations<std::negate<>, std::negate<>>);
	static_assert(!accepts_operations<std::plus<>, std::plus<>>);
	using tree = atl::difference_fenwick<int, std::plus<>, std::negate<>>;
	static_assert(!std::constructible_from<tree, std::vector<std::vector<int>> const &, int>);
}

TEST_CASE("difference_fenwick: boundaries, const ranges and stateful nonzero identity", "[difference_fenwick]") {
	for (std::size_t n : {0u, 1u, 2u, 3u, 7u, 8u, 9u, 63u, 64u, 65u}) {
		atl::difference_fenwick<int, masked_xor, std::identity> filled(n, 3, 7, masked_xor{7});
		CHECK(filled.size() == n);
		CHECK(filled.empty() == (n == 0));
		filled.apply(n, n, 4);
		filled.apply(0, n, 2);
		std::vector<int> const initial(n, 3);
		atl::difference_fenwick<int, masked_xor, std::identity> ranged(initial, 7, masked_xor{7});
		ranged.apply(0, n, 2);
		CHECK(ranged.size() == n);
		for (std::size_t i = 0; i < n; ++i) {
			CHECK(filled.at(i) == (3 ^ 2 ^ 7));
			CHECK(ranged.at(i) == filled.at(i));
		}
		if (n != 0) {
			filled.apply(n - 1, 2);
			CHECK(std::as_const(filled).at(n - 1) == 3);
		}
	}
	std::list<int> const initial{2, 4, 8};
	tree from_list(initial, 0);
	from_list.apply(0, 3, 5);
	CHECK(from_list.at(0) == 7);
	CHECK(from_list.at(2) == 13);
	atl::difference_fenwick<int, shifted_add, shifted_inverse> shifted(initial, 7, shifted_add{7}, shifted_inverse{7});
	shifted.apply(1, 2, 10);
	CHECK(shifted.at(0) == 2);
	CHECK(shifted.at(1) == 7);
	CHECK(shifted.at(2) == 8);
}

TEST_CASE("difference_fenwick: asserted bounds and narrowing", "[difference_fenwick]") {
	CHECK(atl_test::aborts([] { tree(2, 0, 0).apply(2, 1, 3); }));
	CHECK(atl_test::aborts([] { tree(2, 0, 0).apply(0, 3, 3); }));
	CHECK(atl_test::aborts([] { tree(2, 0, 0).apply(2, 3); }));
	CHECK(atl_test::aborts([] { (void)tree(0, 0, 0).at(0); }));
	CHECK(atl_test::aborts([] {
		std::array const values{std::numeric_limits<long long>::max()};
		(void)tree(values, 0);
	}));
}

TEST_CASE("difference_fenwick: an inverse with a wider result type", "[difference_fenwick]") {
	auto const op = []<typename T>(T a, T b)
		requires std::same_as<T, int>
	{ return a + b; };
	auto const inverse = [](int x) -> long long { return -1LL * x; };
	atl::difference_fenwick<int, decltype(op), decltype(inverse)> fen(std::array{1, 2}, 0, op, inverse);
	fen.apply(0, 1, 5);
	REQUIRE(fen.at(0) == 6);
	REQUIRE(fen.at(1) == 2);
}
