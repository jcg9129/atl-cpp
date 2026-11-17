#include <atl/ds/dual_segment_tree.hpp>
#include <atl/ds/lazy_segment_tree.hpp>
#include <atl/ds/segment_tree.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <vector>
#include "move_probe.hpp"

#include <cstddef>

using atl::uint64;

// Deduction from the identity must reject narrowing the range's values, while
// still allowing deliberate widening. (Dependent context so a failed deduction
// yields false instead of a hard error.)
template<typename Value, typename Identity>
concept seg_tree_deducible =
	requires(std::vector<Value> &v, Identity id) { atl::segment_tree(v, id, std::plus<>{}); };
static_assert(!seg_tree_deducible<long long, int>);
static_assert(seg_tree_deducible<int, long long>);

TEST_CASE("segment_tree: sum vs brute force (set/query/max_right/min_left)", "[segment_tree]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	int const n = 1 + int(rng() % 300);
	std::vector<long long> a(n);
	for (auto &x : a) x = int(rng() % 100);
	atl::segment_tree seg(a, 0LL, std::plus<>{});
	for (int i = 0; i < 4000; ++i) {
		switch (rng() % 4) {
		case 0: {
			int const j = int(rng() % n);
			long long const v = int(rng() % 100);
			seg.set(j, v);
			a[j] = v;
			break;
		}
		case 1: {
			int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
			REQUIRE(seg.fold(l, r) == std::accumulate(a.begin() + l, a.begin() + r, 0LL));
			break;
		}
		case 2: { // farthest r from l whose prefix sum stays <= t
			int const l = int(rng() % (n + 1));
			long long const t = rng() % 1000;
			std::size_t const got = seg.max_right(l, [&](long long s) { return s <= t; });
			long long acc = 0;
			std::size_t want = std::size_t(l);
			for (std::size_t j = l; j < std::size_t(n); ++j) {
				if (acc + a[j] > t) break;
				acc += a[j];
				want = j + 1;
			}
			REQUIRE(got == want);
			break;
		}
		default: {
			int const r = int(rng() % (n + 1));
			long long const t = rng() % 1000;
			std::size_t const got = seg.min_left(r, [&](long long s) { return s <= t; });
			long long acc = 0;
			std::size_t want = std::size_t(r);
			for (std::size_t j = r; j-- > 0;) {
				if (acc + a[j] > t) break;
				acc += a[j];
				want = j;
			}
			REQUIRE(got == want);
		}
		}
	}
	REQUIRE(seg.fold() == std::accumulate(a.begin(), a.end(), 0LL));
}

TEST_CASE("segment_tree: min monoid", "[segment_tree]") {
	std::vector<int> const a{5, 3, 8, 1, 9, 2};
	atl::segment_tree seg(a, (std::numeric_limits<int>::max)(), [](int x, int y) { return (std::min)(x, y); });
	REQUIRE(seg.fold(0, 6) == 1);
	REQUIRE(seg.fold(0, 3) == 3);
	REQUIRE(seg.fold(4, 6) == 2);
	seg.set(3, 10);
	REQUIRE(seg.fold(0, 6) == 2);
	REQUIRE(seg.fold(2, 4) == 8);
}

TEST_CASE("segment_tree: non-commutative (concatenation)", "[segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	int const n = 1 + int(rng() % 60);
	std::string s;
	for (int i = 0; i < n; ++i) s.push_back(char('a' + rng() % 4));
	auto cat = [](std::string const &x, std::string const &y) { return x + y; };
	std::vector<std::string> init;
	for (char c : s) init.emplace_back(1, c);
	atl::segment_tree seg(init, std::string{}, cat);
	for (int step = 0; step < 200; ++step) {
		int const i = int(rng() % n);
		char const c = char('a' + rng() % 4);
		seg.set(i, std::string(1, c));
		s[i] = c;
		int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
		REQUIRE(seg.fold(l, r) == s.substr(l, r - l));
		std::string fe;
		seg.for_each_segment(l, r, [&](std::string const &v) { fe += v; });
		REQUIRE(fe == s.substr(l, r - l));
		REQUIRE(seg.at(i) == std::string(1, s[i]));
	}
}

TEST_CASE("segment_tree: edge cases", "[segment_tree]") {
	atl::segment_tree one(std::size_t(1), 0, std::plus<>{});
	one.set(0, 42);
	REQUIRE(one.fold(0, 1) == 42);
	REQUIRE(one.fold() == 42);
	REQUIRE(one.at(0) == 42);

	atl::segment_tree seg(std::vector<int>{1, 2, 3, 4}, 0, std::plus<>{});
	REQUIRE(seg.fold(2, 2) == 0); // empty range -> identity
	REQUIRE(seg.fold() == 10);
	REQUIRE(seg.size() == 4);
	REQUIRE_FALSE(seg.empty());

	atl::segment_tree zeros(std::size_t(5), 0, std::plus<>{}); // 5 copies of the identity
	REQUIRE(zeros.fold() == 0);
	zeros.set(2, 9);
	REQUIRE(zeros.fold() == 9);

	atl::segment_tree filled(std::vector<int>(5, 7), 0, std::plus<>{}); // 5 copies of 7 via a range
	REQUIRE(filled.fold() == 35);
}

TEST_CASE("segment_tree: rvalue range moves its elements in", "[segment_tree]") {
	using atl_test::move_probe;
	auto op = [](move_probe const &a, move_probe const &b) { return move_probe(a.v + b.v); };

	std::vector<move_probe> vals(4, move_probe(move_probe::sentinel));
	move_probe::reset();
	atl::segment_tree copied(vals, move_probe(0), op);
	REQUIRE(move_probe::sentinel_copies == 4); // lvalue range: one copy per element

	move_probe::reset();
	atl::segment_tree moved(std::move(vals), move_probe(0), op);
	REQUIRE(move_probe::sentinel_copies == 0); // rvalue range: moved, never copied
	REQUIRE(moved.at(2).v == move_probe::sentinel);
	REQUIRE(moved.fold().v == 4 * move_probe::sentinel);
}

TEST_CASE("segment_tree: borrowed ranges and views are copied from, never moved from", "[segment_tree]") {
	using atl_test::move_probe;
	auto op = [](move_probe const &a, move_probe const &b) { return move_probe(a.v + b.v); };
	std::vector<move_probe> vals(4, move_probe(move_probe::sentinel));

	move_probe::reset();
	atl::segment_tree<move_probe, decltype(op)> from_span(std::span<move_probe>(vals), move_probe(0), op);
	REQUIRE(move_probe::sentinel_copies == 4); // one copy per element, the source keeps its values

	// a sized, non-borrowed view handing out lvalue references into vals
	auto const view = vals | std::views::transform([](move_probe &p) -> move_probe & { return p; });
	static_assert(!std::ranges::borrowed_range<decltype(view)>);
	move_probe::reset();
	atl::segment_tree<move_probe, decltype(op)> from_view(std::move(view), move_probe(0), op);
	REQUIRE(move_probe::sentinel_copies == 4); // one copy per element, the source keeps its values
	REQUIRE(std::ranges::all_of(vals, [](move_probe const &p) { return p.v == move_probe::sentinel; }));
	REQUIRE(from_span.fold().v == 4 * move_probe::sentinel);
	REQUIRE(from_view.fold().v == 4 * move_probe::sentinel);
}

TEST_CASE("segment_tree: update vs set, against brute force", "[segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	int const n = 1 + int(rng() % 200);
	std::vector<std::string> a(n);
	auto const concat = [](std::string const &x, std::string const &y) { return x + y; };
	atl::segment_tree seg(a, std::string(), concat);
	for (int step = 0; step < 2000; ++step) {
		int const i = int(rng() % n);
		char const c = char('a' + rng() % 26);
		if (rng() % 2) {
			seg.update(std::size_t(i), [&](std::string const &s) { return s.size() < 4 ? s + c : std::string(1, c); });
			a[i] = a[i].size() < 4 ? a[i] + c : std::string(1, c);
		} else {
			seg.set(std::size_t(i), std::string(1, c));
			a[i] = std::string(1, c);
		}
		int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
		std::string want;
		for (int j = l; j < r; ++j) want += a[j];
		REQUIRE(seg.fold(std::size_t(l), std::size_t(r)) == want);
	}
}

TEST_CASE("segment_tree: default construction and defaulted operation", "[segment_tree]") {
	atl::segment_tree<long long, std::plus<>> seg;
	REQUIRE(seg.empty());
	REQUIRE(seg.fold() == 0);
	REQUIRE(seg.fold(0, 0) == 0);
	seg = atl::segment_tree<long long, std::plus<>>(std::vector<long long>{1, 2, 3}, 0);
	REQUIRE(seg.fold() == 6);
	seg = atl::segment_tree<long long, std::plus<>>(5, 0);
	seg.update(4, [](long long x) { return x + 7; });
	REQUIRE(seg.fold(3, 5) == 7);
	int const offset = 0;
	auto const capturing = [offset](int x, int y) { return x + y + offset; };
	static_assert(!std::default_initializable<atl::segment_tree<int, decltype(capturing)>>);
}

TEST_CASE("segment_tree: point transformations run exactly once", "[segment_tree]") {
	atl::segment_tree<int, std::plus<>> library(std::size_t(9), 0);
	int calls = 0;
	auto transform = [&calls, serial = 0](int value) mutable { ++calls; return value + ++serial; };
	library.update(0, transform);
	REQUIRE(calls == 1);
	REQUIRE(library.at(0) == 1);
	library.update(8, transform);
	REQUIRE(calls == 2);
	REQUIRE(library.at(8) == 2);
	library.update(4, transform);
	REQUIRE(calls == 3);
	REQUIRE(library.at(4) == 3);
	library.update(0, transform);
	REQUIRE(calls == 4);
	REQUIRE(library.at(0) == 5);
	library.update(8, transform);
	REQUIRE(calls == 5);
	REQUIRE(library.at(8) == 7);
}

#include "../expect_abort.hpp"

TEST_CASE("segment family: oversized dense geometry aborts before allocation", "[segment_family]") {
	std::size_t const count = (std::numeric_limits<std::size_t>::max)();
	REQUIRE(atl_test::aborts([count] { atl::segment_tree<int, std::plus<>> library(count, 0); }));
	REQUIRE(atl_test::aborts([count] { atl::lazy_segment_tree<int, int, std::plus<>, std::plus<>, std::plus<>> library(count, 0, 0); }));
	REQUIRE(atl_test::aborts([count] { atl::dual_segment_tree<int, int, std::plus<>, std::plus<>> library(count, 0, 0); }));
}
