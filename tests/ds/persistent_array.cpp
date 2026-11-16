#include <atl/ds/persistent_array.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"
#include "move_probe.hpp"

using atl::uint64;

TEST_CASE("persistent_array: full persistence, branch off any version", "[persistent_array]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

	std::size_t const n = 300;
	std::uniform_int_distribution<int> vd(-1000, 1000);
	std::vector<int> initial(n);
	for (auto &x : initial) x = vd(rng);

	// Keep a reference copy of every version's full contents; branch new versions
	// off randomly chosen existing ones to exercise full (not just linear) persistence.
	std::vector<atl::persistent_array<int>> versions;
	std::vector<std::vector<int>> reference;
	versions.emplace_back(initial);
	reference.push_back(initial);

	std::uniform_int_distribution<std::size_t> pos(0, n - 1);
	for (int step = 0; step < 400; ++step) {
		std::size_t parent = std::uniform_int_distribution<std::size_t>(0, versions.size() - 1)(rng);
		std::size_t i = pos(rng);
		int v = vd(rng);
		versions.push_back(versions[parent].set(i, v));
		auto child = reference[parent];
		child[i] = v;
		reference.push_back(child);
	}

	for (std::size_t k = 0; k < versions.size(); ++k) {
		REQUIRE(versions[k].size() == n);
		for (std::size_t i = 0; i < n; ++i) REQUIRE(versions[k][i] == reference[k][i]);
	}
}

TEST_CASE("semi_persistent_array: linear history reads every past version", "[persistent_array]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

	std::size_t const n = 300;
	std::uniform_int_distribution<int> vd(-1000, 1000);
	std::vector<int> initial(n);
	for (auto &x : initial) x = vd(rng);

	std::vector<atl::semi_persistent_array<int>> versions;
	std::vector<std::vector<int>> reference;
	versions.emplace_back(initial);
	reference.push_back(initial);

	std::uniform_int_distribution<std::size_t> pos(0, n - 1);
	auto current = versions.front();
	for (int step = 0; step < 400; ++step) {
		std::size_t i = pos(rng);
		int v = vd(rng);
		current = current.set(i, v); // partial persistence: only extend the newest
		versions.push_back(current);
		auto next = reference.back();
		next[i] = v;
		reference.push_back(next);
	}

	for (std::size_t k = 0; k < versions.size(); ++k)
		for (std::size_t i = 0; i < n; ++i) REQUIRE(versions[k][i] == reference[k][i]);
}

TEMPLATE_TEST_CASE("persistent arrays agree across branching factors", "[persistent_array]",
				   (atl::persistent_array<long long, 2>), (atl::persistent_array<long long, 4>),
				   (atl::persistent_array<long long, 16>), (atl::persistent_array<long long, 64>)) {
	auto seed = GENERATE(take(2, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

	std::size_t const n = GENERATE(1, 15, 16, 17, 260, 1000);
	std::vector<long long> initial(n);
	std::uniform_int_distribution<long long> vd(-1'000'000, 1'000'000);
	for (auto &x : initial) x = vd(rng);

	TestType base(initial);
	std::vector<long long> ref = initial;
	std::uniform_int_distribution<std::size_t> pos(0, n - 1);
	auto cur = base;
	for (int step = 0; step < 200; ++step) {
		std::size_t i = pos(rng);
		long long v = vd(rng);
		cur = cur.set(i, v);
		ref[i] = v;
	}
	REQUIRE(cur.size() == n);
	for (std::size_t i = 0; i < n; ++i) REQUIRE(cur[i] == ref[i]);

	for (std::size_t i = 0; i < n; ++i) REQUIRE(base[i] == initial[i]);
}

TEST_CASE("persistent_array: cheap value-copy shares the arena", "[persistent_array]") {
	std::vector<int> initial{1, 2, 3, 4, 5};
	atl::persistent_array<int> a(initial); // CTAD-style value type
	auto b = a;							   // copy is a shared_ptr bump
	auto c = a.set(2, 99);
	REQUIRE(a[2] == 3);
	REQUIRE(b[2] == 3);
	REQUIRE(c[2] == 99);
	REQUIRE(c[0] == 1);
}

// A copy/move-counting element, tracking only operations on a sentinel value so
// the sibling copies done by path-copying don't pollute the count.
TEMPLATE_TEST_CASE("set moves the passed value into place, never copies it", "[persistent_array]",
				   atl::persistent_array<atl_test::move_probe>, atl::semi_persistent_array<atl_test::move_probe>) {
	using atl_test::move_probe;
	std::vector<move_probe> initial;
	for (int i = 0; i < 100; ++i) initial.emplace_back(i);
	TestType a(initial);

	move_probe::sentinel_copies = 0;
	move_probe::sentinel_moves = 0;
	auto b = a.set(50, move_probe(move_probe::sentinel)); // rvalue argument

	REQUIRE(b[50].v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies == 0); // the value was never copied
	REQUIRE(move_probe::sentinel_moves >= 1);
}

TEMPLATE_TEST_CASE("size constructor fills n copies", "[persistent_array]", atl::persistent_array<int>,
				   atl::semi_persistent_array<int>) {
	TestType a(5); // 5 default (0) elements
	REQUIRE(a.size() == 5);
	for (std::size_t i = 0; i < 5; ++i) REQUIRE(a[i] == 0);

	TestType b(4, 7); // 4 copies of 7
	REQUIRE(b.size() == 4);
	for (std::size_t i = 0; i < 4; ++i) REQUIRE(b[i] == 7);
}

TEMPLATE_TEST_CASE("persistent arrays: rvalue range build moves, never copies", "[persistent_array]",
				   atl::persistent_array<atl_test::move_probe>, atl::semi_persistent_array<atl_test::move_probe>) {
	using atl_test::move_probe;
	std::vector<move_probe> vals(100, move_probe(move_probe::sentinel));
	move_probe::reset();
	TestType copied(vals);
	auto const lvalue_copies = move_probe::sentinel_copies;
	REQUIRE(lvalue_copies >= 100); // lvalue range: at least one copy per element

	move_probe::reset();
	TestType moved(std::move(vals));
	REQUIRE(move_probe::sentinel_copies == 0); // rvalue range: moved all the way in
	REQUIRE(moved[42].v == move_probe::sentinel);
}

TEMPLATE_TEST_CASE("persistent arrays: an rvalue span is copied, leaving the source intact", "[persistent_array]",
				   atl::persistent_array<atl_test::move_probe>, atl::semi_persistent_array<atl_test::move_probe>) {
	using atl_test::move_probe;
	std::vector<move_probe> vals(40, move_probe(move_probe::sentinel));
	TestType from_span{std::span<move_probe>(vals)};
	REQUIRE(std::ranges::all_of(vals, [](move_probe const &p) { return p.v == move_probe::sentinel; }));
	REQUIRE(from_span[39].v == move_probe::sentinel);
}

TEST_CASE("persistent_array: iteration matches operator[] on every version", "[persistent_array]") {
	static_assert(std::forward_iterator<atl::persistent_array<int>::iterator>);
	REQUIRE(std::ranges::empty(atl::persistent_array<int>()));
	for (std::size_t n : {1, 15, 16, 17, 256, 1000}) {
		std::mt19937_64 rng(n);
		std::vector<atl::persistent_array<int, 4>> versions{atl::persistent_array<int, 4>(std::views::iota(0, int(n)))};
		for (int step = 0; step < 50; ++step)
			versions.push_back(versions[rng() % versions.size()].set(rng() % n, -step));
		for (auto const &v : versions) {
			std::vector<int> want;
			for (std::size_t i = 0; i < v.size(); ++i) want.push_back(v[i]);
			REQUIRE(std::ranges::equal(v, want));
		}
	}
}

TEST_CASE("persistent_array: rvalue set matches the lvalue one and leaves other versions intact", "[persistent_array]") {
	using array = atl::persistent_array<int>;
	static_assert(requires { static_cast<array (array::*)(std::size_t, int) const &>(&array::set); });
	static_assert(requires { static_cast<array (array::*)(std::size_t, int) &&>(&array::set); });

	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + rng() % 299;
	auto same = [](array const &a, std::vector<int> const &ref) {
		return a.size() == ref.size() && std::ranges::equal(a, ref);
	};
	std::vector<int> ref(n);
	for (auto &x : ref) x = int(rng());
	array moved(ref), copied(ref);
	std::vector<std::pair<array, std::vector<int>>> kept;
	for (int step = 0; step < 1500; ++step) {
		if (rng() % 40 == 0) kept.emplace_back(moved, ref);
		std::size_t i = rng() % n;
		int v = int(rng());
		moved = std::move(moved).set(i, v);
		copied = copied.set(i, v);
		ref[i] = v;
		REQUIRE(same(moved, ref));
		REQUIRE(same(copied, ref));
	}
	for (auto const &[a, r] : kept) REQUIRE(same(a, r));
	array a(ref);
	auto b = std::move(a).set(0, 7);
	a = b;
	REQUIRE(a[0] == 7);
	auto c = std::move(a).set(1, 8);
	REQUIRE(c[1] == 8);
}

TEST_CASE("semi_persistent_array: rvalue set matches the lvalue one and keeps past versions readable", "[persistent_array]") {
	using array = atl::semi_persistent_array<int>;
	static_assert(requires { static_cast<array (array::*)(std::size_t, int) const &>(&array::set); });
	static_assert(requires { static_cast<array (array::*)(std::size_t, int) &&>(&array::set); });

	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + rng() % 299;
	auto same = [](array const &a, std::vector<int> const &ref) {
		if (a.size() != ref.size()) return false;
		for (std::size_t i = 0; i < ref.size(); ++i)
			if (a[i] != ref[i]) return false;
		return true;
	};
	std::vector<int> ref(n);
	for (auto &x : ref) x = int(rng());
	array moved(ref), copied(ref);
	std::vector<std::pair<array, std::vector<int>>> kept;
	for (int step = 0; step < 1500; ++step) {
		if (rng() % 40 == 0) kept.emplace_back(moved, ref);
		std::size_t i = rng() % n;
		int v = int(rng());
		moved = std::move(moved).set(i, v);
		copied = copied.set(i, v);
		ref[i] = v;
		REQUIRE(same(moved, ref));
		REQUIRE(same(copied, ref));
	}
	for (auto const &[a, r] : kept) REQUIRE(same(a, r));
	array a(ref);
	auto b = std::move(a).set(0, 7);
	a = b;
	REQUIRE(a[0] == 7);
}

TEST_CASE("persistent arrays: checked update and reservation bounds", "[persistent_array]") {
	CHECK(atl_test::aborts([] {
		atl::persistent_array<int> a(1, 0);
		(void)a.set(1, 2);
	}));
	CHECK(atl_test::aborts([] {
		atl::semi_persistent_array<int> a(2, 0);
		auto const newest = a.set(0, 1);
		(void)newest;
		(void)a.set(1, 2);
	}));
	CHECK(atl_test::aborts([] {
		atl::persistent_array<int> a(300, 0);
		a.reserve((std::numeric_limits<std::size_t>::max)());
	}));
	CHECK(atl_test::aborts([] {
		atl::detail::persistent_arena<int> arena;
		arena.reserve_extra((std::numeric_limits<std::size_t>::max)());
	}));
}
