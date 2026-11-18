#include <atl/ds/binary_indexed_tree.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <functional>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <vector>
#include "../expect_abort.hpp"
#include "move_probe.hpp"

using atl::uint64;

TEST_CASE("Binary Indexed Tree small tests", "[binary_indexed_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), std::numeric_limits<atl::uint64>::max())));

	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

	int const MAX_SIZE = 16;
	int const NUM_QUERIES = 2048;
	int const MAX_VALUE = 64;

	std::uniform_int_distribution<> dist(1, MAX_SIZE - 1);
	int size = dist(rng);

	atl::binary_indexed_tree<int> bit(size); // default std::plus, identity 0
	std::vector<int> values(size);

	std::uniform_int_distribution<> pos_dist(0, size - 1);
	std::uniform_int_distribution<> value_dist(-MAX_VALUE, MAX_VALUE);

	for (int i = 0; i < NUM_QUERIES; ++i) {
		auto pos = pos_dist(rng);
		auto value = value_dist(rng);
		bit.update(value, pos);
		values[pos] += value;

		auto expected = 0;
		for (int j = 0; j < size; ++j) {
			auto found = bit.prefix_fold(j);
			REQUIRE(found == expected);
			expected += values[j];
		}
		REQUIRE(bit.prefix_fold(size) == expected);
	}
}

TEST_CASE("Binary Indexed Tree small test with multiplication", "[binary_indexed_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<atl::uint64>::max)())));
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

	int const MAX_SIZE = 16;
	int const NUM_QUERIES = 2048;
	constexpr int product_test_modulus = 1000000007;
	using Z = atl::mod_int32<product_test_modulus>;
	std::uniform_int_distribution<> dist(1, MAX_SIZE - 1);
	int size = dist(rng);

	atl::binary_indexed_tree<Z, 1, std::multiplies<>> bit(size, Z(1));
	std::vector<Z> values(size, Z(1));
	std::uniform_int_distribution<> pos_dist(0, size - 1);
	std::uniform_int_distribution<> value_dist(0, product_test_modulus - 1);
	for (int i = 0; i < NUM_QUERIES; ++i) {
		auto pos = pos_dist(rng);
		auto value = value_dist(rng);
		bit.update(value, pos);
		values[pos] *= value;

		auto expected = Z(1);
		for (int j = 0; j < size; ++j) {
			auto found = bit.prefix_fold(j);
			REQUIRE(found == expected);
			expected *= values[j];
		}
		REQUIRE(bit.prefix_fold(size) == expected);
	}
}

TEST_CASE("Binary Indexed Tree prefix max via a lambda", "[binary_indexed_tree]") {
	int const neg_inf = std::numeric_limits<int>::min();
	auto tree = atl::binary_indexed_tree(std::size_t(6), neg_inf,
										 [](int a, int b) { return std::max(a, b); });

	std::vector<int> values(6, neg_inf);
	auto set = [&](std::size_t pos, int v) {
		tree.update(v, pos);
		values[pos] = std::max(values[pos], v);
	};

	set(2, 5);
	set(4, 3);
	set(1, 8);
	set(4, 9);

	for (std::size_t j = 0; j <= 6; ++j) {
		int expected = neg_inf;
		for (std::size_t i = 0; i < j; ++i) expected = std::max(expected, values[i]);
		REQUIRE(tree.prefix_fold(j) == expected);
	}
}

TEST_CASE("Binary Indexed Tree 2D prefix sums", "[binary_indexed_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), std::numeric_limits<atl::uint64>::max())));
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

	std::size_t const R = 7, C = 9;
	auto bit = atl::make_binary_indexed_tree(0LL, R, C); // Dim deduced from the size count
	std::vector<std::vector<long long>> grid(R, std::vector<long long>(C, 0));

	std::uniform_int_distribution<int> row(0, R - 1), col(0, C - 1), val(-100, 100);

	for (int t = 0; t < 400; ++t) {
		std::size_t x = row(rng), y = col(rng);
		long long v = val(rng);
		bit.update(v, x, y);
		grid[x][y] += v;

		std::size_t px = row(rng) + 1, py = col(rng) + 1; // prefix bounds in [1, R] x [1, C]
		long long expected = 0;
		for (std::size_t i = 0; i < px; ++i)
			for (std::size_t j = 0; j < py; ++j)
				expected += grid[i][j];
		REQUIRE(bit.prefix_fold(px, py) == expected);
	}
}

TEST_CASE("Binary Indexed Tree built from initial values", "[binary_indexed_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), std::numeric_limits<atl::uint64>::max())));
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
	std::uniform_int_distribution<int> vd(-50, 50);

	SECTION("1D from a range") {
		std::size_t const n = 20;
		std::vector<long long> a(n);
		for (auto &x : a) x = vd(rng);
		auto bit = atl::make_binary_indexed_tree_from(a); // Dim=1 deduced
		for (std::size_t k = 0; k <= n; ++k) {
			long long expected = 0;
			for (std::size_t i = 0; i < k; ++i) expected += a[i];
			REQUIRE(bit.prefix_fold(k) == expected);
		}
	}

	SECTION("2D from a flat row-major range") {
		std::size_t const R = 6, C = 8;
		std::vector<long long> flat(R * C);
		for (auto &x : flat) x = vd(rng);
		auto bit = atl::make_binary_indexed_tree_from(flat, R, C); // Dim=2 deduced
		std::uniform_int_distribution<std::size_t> pr(0, R), pc(0, C);
		for (int t = 0; t < 60; ++t) {
			std::size_t px = pr(rng), py = pc(rng);
			long long expected = 0;
			for (std::size_t i = 0; i < px; ++i)
				for (std::size_t j = 0; j < py; ++j)
					expected += flat[i * C + j];
			REQUIRE(bit.prefix_fold(px, py) == expected);
		}
	}

	SECTION("2D from a nested range (dimension deduced from nesting depth)") {
		std::size_t const R = 6, C = 8;
		std::vector<std::vector<long long>> grid(R, std::vector<long long>(C));
		for (auto &row : grid)
			for (auto &x : row) x = vd(rng);
		auto bit = atl::make_binary_indexed_tree_from(grid); // T and Dim=2 both deduced
		std::uniform_int_distribution<std::size_t> pr(0, R), pc(0, C);
		for (int t = 0; t < 60; ++t) {
			std::size_t px = pr(rng), py = pc(rng);
			long long expected = 0;
			for (std::size_t i = 0; i < px; ++i)
				for (std::size_t j = 0; j < py; ++j)
					expected += grid[i][j];
			REQUIRE(bit.prefix_fold(px, py) == expected);
		}
	}

	SECTION("nested range via CTAD") {
		std::vector<std::vector<int>> grid{{1, 2, 3}, {4, 5, 6}};
		atl::binary_indexed_tree bit(grid); // deduces binary_indexed_tree<int, 2>
		REQUIRE(bit.prefix_fold(2, 3) == 21);
		REQUIRE(bit.prefix_fold(1, 2) == 3);
	}

	SECTION("3D from a nested range") {
		std::size_t const A = 4, B = 5, C = 6;
		std::vector cube(A, std::vector(B, std::vector<long long>(C)));
		for (auto &plane : cube)
			for (auto &row : plane)
				for (auto &x : row) x = vd(rng);
		auto bit = atl::make_binary_indexed_tree_from(cube); // T and Dim=3 both deduced
		std::uniform_int_distribution<std::size_t> pa(0, A), pb(0, B), pc(0, C);
		for (int t = 0; t < 60; ++t) {
			std::size_t px = pa(rng), py = pb(rng), pz = pc(rng);
			long long expected = 0;
			for (std::size_t i = 0; i < px; ++i)
				for (std::size_t j = 0; j < py; ++j)
					for (std::size_t k = 0; k < pz; ++k)
						expected += cube[i][j][k];
			REQUIRE(bit.prefix_fold(px, py, pz) == expected);
		}
	}
}

// update/query indices must be integers: a floating-point index is rejected at
// compile time instead of being silently truncated.
template<typename Tree, typename Index>
concept bit_indexable = requires(Tree t, Index i) {
	t.update(1, i);
	t.prefix_fold(i);
};
static_assert(bit_indexable<atl::binary_indexed_tree<int>, int>);
static_assert(bit_indexable<atl::binary_indexed_tree<int>, atl::uint64>);
static_assert(!bit_indexable<atl::binary_indexed_tree<int>, double>);

TEST_CASE("Binary Indexed Tree: rvalue range moves its elements in", "[binary_indexed_tree]") {
	using atl_test::move_probe;
	auto op = [](move_probe const &a, move_probe const &b) { return move_probe(a.v + b.v); };
	using tree = atl::binary_indexed_tree<move_probe, 1, decltype(op)>;

	std::vector<move_probe> vals(4, move_probe(move_probe::sentinel));
	move_probe::reset();
	tree copied(vals, move_probe(0), op);
	REQUIRE(move_probe::sentinel_copies == 4); // lvalue range: one copy per element

	move_probe::reset();
	tree moved(std::move(vals), move_probe(0), op);
	REQUIRE(move_probe::sentinel_copies == 0); // rvalue range: moved, never copied
}

namespace {
	template<typename Op, typename Values>
	concept builds_without_identity = requires(Values const &values) {
		atl::make_binary_indexed_tree_from<Op>(values);
	};
}

TEST_CASE("Binary Indexed Tree: a non-sum tree from values needs its identity", "[binary_indexed_tree]") {
	// regression: make_binary_indexed_tree_from<Op>(values) let the caller
	// pick Op but always used T{} as the identity, so a product tree was
	// seeded with 0 and every fold returned 0
	static_assert(builds_without_identity<std::plus<>, std::vector<long long>>);
	static_assert(!builds_without_identity<std::multiplies<>, std::vector<long long>>);

	std::vector<long long> const values{2, 3, 4, 5};
	auto const product = atl::make_binary_indexed_tree_from<std::multiplies<>>(1LL, values);
	long long expected = 1;
	REQUIRE(product.prefix_fold(0) == 1);
	for (std::size_t i = 0; i < values.size(); ++i) {
		expected *= values[i];
		REQUIRE(product.prefix_fold(i + 1) == expected);
	}
	// the identity converts to the value type, and the flat form takes sizes last
	std::vector<long long> const flat{1, 2, 3, 4, 5, 6};
	auto const grid = atl::make_binary_indexed_tree_from<std::multiplies<>>(1, flat, 2, 3);
	REQUIRE(grid.prefix_fold(2, 3) == 720);
	REQUIRE(grid.prefix_fold(1, 2) == 2);
}

TEST_CASE("Binary Indexed Tree: bad extents trip the entry asserts", "[binary_indexed_tree]") {
	// a negative extent would wrap to a huge size_t, an overflowing product to a small one
	CHECK(atl_test::aborts([] { (void)atl::make_binary_indexed_tree(0, 3, -1); }));
	std::vector<int> const flat{1, 2};
	CHECK(atl_test::aborts([&] { (void)atl::make_binary_indexed_tree_from(flat, -2); }));
	std::size_t const huge = std::size_t(1) << (std::numeric_limits<std::size_t>::digits / 2 + 1);
	CHECK(atl_test::aborts([&] { (void)atl::binary_indexed_tree<int, 2>({huge, huge}); }));
}

TEST_CASE("Binary Indexed Tree: max_right prefix descent vs linear scan", "[binary_indexed_tree]") {
	std::mt19937 rng(20261014);
	for (int round = 0; round < 200; ++round) {
		std::size_t const n = rng() % 40;
		std::vector<long long> vals(n);
		for (auto &v : vals) v = rng() % 10;
		atl::binary_indexed_tree<long long> bit(vals);
		REQUIRE(bit.size() == n);
		for (int q = 0; q < 10; ++q) {
			long long const k = rng() % 200;
			auto const pred = [k](long long s) { return s <= k; };
			std::size_t expected = 0;
			for (long long s = 0; expected < n && pred(s + vals[expected]);) s += vals[expected++];
			REQUIRE(bit.max_right(pred) == expected);
		}
	}
}

TEST_CASE("Binary Indexed Tree: max_right with a non-sum monoid", "[binary_indexed_tree]") {
	// prefix max stays below a threshold
	auto mx = [](int a, int b) { return std::max(a, b); };
	std::vector<int> vals{3, 1, 4, 1, 5, 9, 2, 6};
	atl::binary_indexed_tree<int, 1, decltype(mx)> bit(vals, std::numeric_limits<int>::min(), mx);
	REQUIRE(bit.max_right([](int m) { return m < 5; }) == 4);
	REQUIRE(bit.max_right([](int m) { return m < 3; }) == 0);
	REQUIRE(bit.max_right([](int m) { return m < 10; }) == 8);
}

TEST_CASE("Binary Indexed Tree: sizes of a 2-D tree", "[binary_indexed_tree]") {
	atl::binary_indexed_tree<int, 2> bit({3, 5});
	REQUIRE(bit.sizes() == std::array<std::size_t, 2>{3, 5});
}

TEST_CASE("Binary Indexed Tree: rvalue borrowed or view ranges leave the source intact", "[binary_indexed_tree]") {
	using atl_test::move_probe;
	auto op = [](move_probe const &a, move_probe const &b) { return move_probe(a.v + b.v); };
	using tree = atl::binary_indexed_tree<move_probe, 1, decltype(op)>;
	auto const intact = [](std::vector<move_probe> const &v) {
		return std::ranges::all_of(v, [](move_probe const &p) { return p.v == move_probe::sentinel; });
	};
	std::vector<move_probe> vals(4, move_probe(move_probe::sentinel));

	move_probe::reset();
	tree from_span(std::span<move_probe>(vals), move_probe(0), op);
	REQUIRE(move_probe::sentinel_copies == 4);
	REQUIRE(intact(vals));

	move_probe::reset();
	tree from_view(vals | std::views::transform([](move_probe &p) -> move_probe & { return p; }), move_probe(0), op);
	REQUIRE(move_probe::sentinel_copies == 4);
	REQUIRE(intact(vals));

	// nested: an owning outer vector of spans must not move through the spans
	std::vector<std::span<move_probe>> rows{std::span<move_probe>(vals)};
	using tree2 = atl::binary_indexed_tree<move_probe, 2, decltype(op)>;
	move_probe::reset();
	tree2 nested(std::move(rows), move_probe(0), op);
	REQUIRE(move_probe::sentinel_copies == 4);
	REQUIRE(intact(vals));
}

namespace {
	struct residue {
		unsigned value = 0;
		constexpr residue() = default;

		constexpr residue(unsigned x) : value(x % 17) {}

		friend constexpr residue operator*(residue a, residue b) { return residue(a.value * b.value); }

		friend constexpr bool operator==(residue, residue) = default;
	};
}

TEST_CASE("Fenwick multiplication accepts a class ring element", "[binary_indexed_tree]") {
	std::array const values{residue(3), residue(5), residue(7), residue(2)};
	atl::binary_indexed_tree<residue, 1, std::multiplies<>> tree(values, residue(1));
	REQUIRE(tree.prefix_fold(0) == residue(1));
	tree.update(residue(4), 2);
	residue product(1);
	for (std::size_t i = 0; i < values.size(); ++i) {
		product = product * values[i] * residue(i == 2 ? 4u : 1u);
		REQUIRE(tree.prefix_fold(i + 1) == product);
	}
}

TEST_CASE("Fenwick nested prvalue ranges own their yielded rows", "[binary_indexed_tree]") {
	using atl_test::move_probe;
	auto rows = std::views::iota(0, 1) | std::views::transform([](int) {
					std::vector<move_probe> row;
					row.reserve(2);
					row.emplace_back(1);
					row.emplace_back(move_probe::sentinel);
					return row;
				});
	auto const add = [](move_probe const &x, move_probe const &y) { return move_probe(x.v + y.v); };
	move_probe::reset();
	atl::binary_indexed_tree<move_probe, 2, decltype(add)> tree(std::move(rows), move_probe(0), add);
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(move_probe::sentinel_moves == 1);
	REQUIRE(tree.prefix_fold(1, 2).v == move_probe::sentinel + 1);
}

TEST_CASE("Fenwick validates every prefix endpoint even in an empty box", "[binary_indexed_tree]") {
	CHECK(atl_test::aborts([] {
		atl::binary_indexed_tree<int, 2> tree({3, 3});
		(void)tree.prefix_fold(0, 4);
	}));
	CHECK(atl_test::aborts([] {
		atl::binary_indexed_tree<int, 2> tree({3, 3});
		(void)tree.prefix_fold(0, -1);
	}));
}
