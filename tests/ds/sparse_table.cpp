#include <atl/ds/direct_rmq.hpp>
#include <atl/ds/sparse_table.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <utility>
#include <vector>
#include "move_probe.hpp"

using atl::disjoint_sparse_table;
using atl::sparse_table;

namespace {
	// non-commutative associative op: affine maps x -> ax + b over Z_1009,
	// composed left-to-right
	struct affine {
		std::uint32_t a = 1, b = 0;
		friend bool operator==(affine, affine) = default;
	};

	constexpr affine compose(affine f, affine g) { // apply f, then g
		return {g.a * f.a % 1009, (g.a * f.b + g.b) % 1009};
	}
}

TEST_CASE("sparse_table: all idempotent ops match brute folds on all ranges", "[ds]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	// Keep the same one-block sizes without exposing a fixed allocation bound to GCC's speculative RMQ-path analysis.
	std::size_t const n = GENERATE(take(1, random(std::size_t(1), std::size_t(64))));
	std::vector<std::uint64_t> v(n);
	for (auto &x : v) x = rng() % 1000;

	auto const check = [&]<atl::binary_operation<std::uint64_t> Op>(Op op) {
		sparse_table const st(v, op);
		REQUIRE(st.size() == n);
		for (std::size_t l = 0; l < n; ++l) {
			auto acc = v[l];
			for (std::size_t r = l + 1; r <= n; ++r) {
				if (r > l + 1) acc = op(acc, v[r - 1]);
				REQUIRE(st.fold(l, r) == acc);
			}
		}
	};
	check([](std::uint64_t a, std::uint64_t b) { return std::min(a, b); });
	check([](std::uint64_t a, std::uint64_t b) { return std::max(a, b); });
	check([](std::uint64_t a, std::uint64_t b) { return std::gcd(a, b); });
	check([](std::uint64_t a, std::uint64_t b) { return a | b; });

	// min agrees with direct_rmq
	sparse_table const st(v, [](std::uint64_t a, std::uint64_t b) { return std::min(a, b); });
	atl::direct_rmq const rmq(v);
	for (int probe = 0; probe < 200; ++probe) {
		std::size_t const l = rng() % n, r = l + 1 + rng() % (n - l);
		REQUIRE(st.fold(l, r) == rmq.fold(l, r));
	}
}

TEST_CASE("disjoint_sparse_table: any associative op, order preserved", "[ds]") {
	auto seed = GENERATE(take(30, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 64;

	SECTION("sums and minima") {
		std::vector<std::uint64_t> v(n);
		for (auto &x : v) x = rng() % 1'000'000;
		disjoint_sparse_table const sum(v, [](std::uint64_t a, std::uint64_t b) { return a + b; });
		disjoint_sparse_table const mn(v, [](std::uint64_t a, std::uint64_t b) { return std::min(a, b); });
		for (std::size_t l = 0; l < n; ++l) {
			std::uint64_t s = 0, m = std::numeric_limits<std::uint64_t>::max();
			for (std::size_t r = l + 1; r <= n; ++r) {
				s += v[r - 1];
				m = std::min(m, v[r - 1]);
				REQUIRE(sum.fold(l, r) == s);
				REQUIRE(mn.fold(l, r) == m);
			}
		}
	}
	SECTION("non-commutative composition") {
		std::vector<affine> f(n);
		for (auto &x : f) x = {std::uint32_t(1 + rng() % 1008), std::uint32_t(rng() % 1009)};
		disjoint_sparse_table const dst(f, [](affine a, affine b) { return compose(a, b); });
		for (std::size_t l = 0; l < n; ++l) {
			affine acc = f[l];
			for (std::size_t r = l + 1; r <= n; ++r) {
				if (r > l + 1) acc = compose(acc, f[r - 1]);
				REQUIRE(dst.fold(l, r) == acc);
			}
		}
	}
}

TEST_CASE("sparse tables and direct_rmq: n around 1e5 against brute and prefix oracles", "[ds]") {
	auto seed = GENERATE(take(3, random(std::uint64_t(0), std::numeric_limits<std::uint64_t>::max())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 100'000 + rng() % 40'000;
	std::vector<std::uint64_t> v(n);
	// small values, so equal minima (the leftmost-position contract) are frequent
	for (auto &x : v) x = rng() % 5000;
	std::vector<std::uint64_t> prefix(n + 1, 0);
	for (std::size_t i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + v[i];

	auto const min_op = [](std::uint64_t a, std::uint64_t b) { return std::min(a, b); };
	sparse_table const st(v, min_op);
	disjoint_sparse_table const dmin(v, min_op);
	disjoint_sparse_table const dsum(v, [](std::uint64_t a, std::uint64_t b) { return a + b; });
	sparse_table const gcd(v, [](std::uint64_t a, std::uint64_t b) { return std::gcd(a, b); });
	atl::direct_rmq const rmq(v);

	for (int q = 0; q < 3000; ++q) {
		// a third short (inside or across one 64-block), the rest of any length
		std::size_t const len = q % 3 == 0 ? 1 + rng() % 200 : 1 + rng() % n;
		std::size_t const l = rng() % (n - len + 1), r = l + len;
		auto const it = std::ranges::min_element(v.begin() + std::ptrdiff_t(l), v.begin() + std::ptrdiff_t(r));
		REQUIRE(st.fold(l, r) == *it);
		REQUIRE(dmin.fold(l, r) == *it);
		REQUIRE(rmq.fold(l, r) == *it);
		REQUIRE(rmq.position(l, r) == std::size_t(it - v.begin()));
		REQUIRE(dsum.fold(l, r) == prefix[r] - prefix[l]);
		if (len <= 200) { // brute gcd only on short ranges, to bound the run
			std::uint64_t g = 0;
			for (std::size_t i = l; i < r; ++i) g = std::gcd(g, v[i]);
			REQUIRE(gcd.fold(l, r) == g);
		}
	}
}

TEST_CASE("sparse tables: edges", "[ds]") {
	sparse_table<int, std::plus<>> const empty_st{};
	REQUIRE(empty_st.empty());
	disjoint_sparse_table<int, std::plus<>> const empty_dst{};
	REQUIRE(empty_dst.empty());

	std::vector<int> const one{42};
	sparse_table const st(one, [](int a, int b) { return std::min(a, b); });
	REQUIRE(st.fold(0, 1) == 42);
	disjoint_sparse_table const dst(one, std::plus<>{});
	REQUIRE(dst.fold(0, 1) == 42);

	std::vector<int> const two{7, 3};
	disjoint_sparse_table const d2(two, std::minus<>{}); // not associative, but 2 elements: order check
	REQUIRE(d2.fold(0, 2) == 4);
	REQUIRE(d2.fold(0, 1) == 7);
	REQUIRE(d2.fold(1, 2) == 3);
}

TEST_CASE("sparse tables: row 0 moves only out of an owning rvalue range", "[ds]") {
	using atl_test::move_probe;
	// The sentinel sits last and exceeds every other value, so min never yields
	// it and neither padding nor the higher rows touch it: only row 0 does.
	auto const make = [] { return std::vector<move_probe>{1, 2, 3, move_probe::sentinel}; };
	auto const min_op = [](move_probe const &a, move_probe const &b) { return move_probe(std::min(a.v, b.v)); };
	auto const check = [&]<template<typename, typename> typename Table>() {
		using table = Table<move_probe, decltype(min_op)>;
		auto owning = make();
		move_probe::reset();
		table const owned(std::move(owning), min_op);
		CHECK(move_probe::sentinel_copies == 0);
		CHECK(owned.fold(3, 4).v == move_probe::sentinel);

		auto source = make();
		move_probe::reset();
		table const spanned{std::span(source), min_op};
		CHECK(move_probe::sentinel_copies == 1);
		CHECK(source[3].v == move_probe::sentinel);
		auto view = source | std::views::transform([](move_probe &x) -> move_probe && { return std::move(x); });
		move_probe::reset();
		table const viewed(std::move(view), min_op);
		CHECK(move_probe::sentinel_copies == 1);
		CHECK(source[3].v == move_probe::sentinel);
		CHECK(viewed.fold(3, 4).v == move_probe::sentinel);
	};
	check.template operator()<sparse_table>();
	check.template operator()<disjoint_sparse_table>();
}
