#include <atl/ds/lazy_segment_tree.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <utility>
#include <vector>
#include "move_probe.hpp"

#include <cstddef>

using atl::uint64;
using ll = long long;

namespace {
	auto const min_op = [](ll x, ll y) { return (std::min)(x, y); };
	auto const add_map = [](ll f, ll s) { return s + f; };
	auto const add_compose = [](ll f, ll g) { return f + g; };
}

// Deduction from the element identity must reject narrowing the range's values,
// while still allowing deliberate widening. (Dependent context so a failed
// deduction yields false instead of a hard error.)
template<typename Value, typename Identity>
concept lazy_seg_tree_deducible = requires(std::vector<Value> &v, Identity id) {
	atl::lazy_segment_tree(v, id, ll(0), min_op, add_map, add_compose);
};
static_assert(!lazy_seg_tree_deducible<long long, int>);
static_assert(lazy_seg_tree_deducible<long long, long long>);

TEST_CASE("lazy_segment_tree: range add / range min vs brute force", "[lazy_segment_tree]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	int const n = 1 + int(rng() % 200);
	std::vector<ll> a(n);
	for (auto &x : a) x = ll(rng() % 1000);
	atl::lazy_segment_tree st(a, (std::numeric_limits<ll>::max)(), ll(0), min_op, add_map, add_compose);
	for (int step = 0; step < 2000; ++step) {
		int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
		switch (rng() % 6) {
		case 0: {
			ll const f = ll(rng() % 100) - 50;
			st.apply(l, r, f);
			for (int j = l; j < r; ++j) a[j] += f;
			break;
		}
		case 1: {
			ll want = (std::numeric_limits<ll>::max)();
			for (int j = l; j < r; ++j) want = (std::min)(want, a[j]);
			REQUIRE(st.fold(l, r) == want);
			break;
		}
		case 2: {
			int const i = int(rng() % n);
			ll const v = ll(rng() % 1000);
			st.set(i, v);
			a[i] = v;
			break;
		}
		case 3: {
			int const i = int(rng() % n);
			REQUIRE(st.at(i) == a[i]);
			st.apply(i, ll(7));
			a[i] += 7;
			break;
		}
		case 4: { // farthest r' from l with min of [l, r') >= t
			ll const t = ll(rng() % 1000);
			std::size_t const got = st.max_right(l, [&](ll s) { return s >= t; });
			std::size_t want = std::size_t(l);
			while (want < std::size_t(n) && a[want] >= t) ++want;
			REQUIRE(got == want);
			break;
		}
		default: { // earliest l' below r with min of [l', r) >= t
			ll const t = ll(rng() % 1000);
			std::size_t const got = st.min_left(r, [&](ll s) { return s >= t; });
			std::size_t want = std::size_t(r);
			while (want > 0 && a[want - 1] >= t) --want;
			REQUIRE(got == want);
		}
		}
	}
	REQUIRE(st.fold() == *std::min_element(a.begin(), a.end()));
}

TEST_CASE("lazy_segment_tree: range add / range sum (width in S)", "[lazy_segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	int const n = 1 + int(rng() % 150);
	using node = std::pair<ll, ll>; // (sum, width)
	std::vector<node> init;
	std::vector<ll> a(n);
	for (int i = 0; i < n; ++i) {
		a[i] = ll(rng() % 100);
		init.push_back({a[i], 1});
	}
	atl::lazy_segment_tree st(
		init, node{0, 0}, ll(0), [](node const &x, node const &y) { return node{x.first + y.first, x.second + y.second}; },
		[](ll f, node const &s) { return node{s.first + f * s.second, s.second}; }, add_compose);
	for (int step = 0; step < 1000; ++step) {
		int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
		if (rng() % 2) {
			ll const f = ll(rng() % 50);
			st.apply(l, r, f);
			for (int j = l; j < r; ++j) a[j] += f;
		} else {
			REQUIRE(st.fold(l, r).first == std::accumulate(a.begin() + l, a.begin() + r, ll(0)));
		}
	}
}

TEST_CASE("lazy_segment_tree: range assign / range sum (non-commutative compose)", "[lazy_segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	int const n = 1 + int(rng() % 150);
	constexpr ll no_tag = -1;
	using node = std::pair<ll, ll>; // (sum, width)
	std::vector<node> init;
	std::vector<ll> a(n);
	for (int i = 0; i < n; ++i) {
		a[i] = ll(rng() % 100);
		init.push_back({a[i], 1});
	}
	atl::lazy_segment_tree st(
		init, node{0, 0}, no_tag,
		[](node const &x, node const &y) { return node{x.first + y.first, x.second + y.second}; },
		[](ll f, node const &s) { return f == no_tag ? s : node{f * s.second, s.second}; },
		[](ll f, ll g) { return f == no_tag ? g : f; }); // newer assignment wins
	for (int step = 0; step < 1000; ++step) {
		int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
		if (rng() % 2) {
			ll const f = ll(rng() % 100);
			st.apply(l, r, f);
			for (int j = l; j < r; ++j) a[j] = f;
		} else {
			REQUIRE(st.fold(l, r).first == std::accumulate(a.begin() + l, a.begin() + r, ll(0)));
		}
	}
}

TEST_CASE("lazy_segment_tree: edge cases", "[lazy_segment_tree]") {
	atl::lazy_segment_tree one(std::size_t(1), ll(0), ll(0), std::plus<>{}, add_map, add_compose);
	one.apply(0, 1, ll(5));
	one.apply(std::size_t(0), ll(2));
	REQUIRE(one.fold(0, 1) == 7);
	REQUIRE(one.at(0) == 7);
	one.set(0, 42);
	REQUIRE(one.fold() == 42);

	atl::lazy_segment_tree st(std::vector<ll>{1, 2, 3, 4}, ll(0), ll(0), std::plus<>{}, add_map, add_compose);
	REQUIRE(st.fold(2, 2) == 0); // empty range -> identity
	st.apply(1, 3, ll(0));		 // identity tag is a no-op
	REQUIRE(st.fold() == 10);
	REQUIRE(st.size() == 4);
	REQUIRE_FALSE(st.empty());

	// add-on-sum needs the width inside S — a plain sum would break the
	// distributivity law mapping(f, op(a, b)) == op(mapping(f, a), mapping(f, b)).
	using node = std::pair<ll, ll>; // (sum, width)
	atl::lazy_segment_tree zeros(
		std::vector<node>(5, node{0, 1}), node{0, 0}, ll(0),
		[](node const &x, node const &y) { return node{x.first + y.first, x.second + y.second}; },
		[](ll f, node const &s) { return node{s.first + f * s.second, s.second}; }, add_compose);
	zeros.apply(0, 5, ll(3)); // 5 zeros, +3 each
	REQUIRE(zeros.fold().first == 15);
	REQUIRE(zeros.fold(1, 3).first == 6);
}

TEST_CASE("lazy_segment_tree: borrowed ranges and views are copied from, owning rvalues moved", "[lazy_segment_tree]") {
	using atl_test::move_probe;
	auto op = [](move_probe const &a, move_probe const &b) { return move_probe(a.v + b.v); };
	auto map = [](int f, move_probe const &s) { return move_probe(s.v + f); };
	auto add = [](int f, int g) { return f + g; };
	using tree = atl::lazy_segment_tree<move_probe, int, decltype(op), decltype(map), decltype(add)>;
	std::vector<move_probe> vals(4, move_probe(move_probe::sentinel));

	move_probe::reset();
	tree from_span(std::span<move_probe>(vals), move_probe(0), 0, op, map, add);
	REQUIRE(move_probe::sentinel_copies == 4);
	auto const view = vals | std::views::transform([](move_probe &p) -> move_probe & { return p; });
	move_probe::reset();
	tree from_view(std::move(view), move_probe(0), 0, op, map, add);
	REQUIRE(move_probe::sentinel_copies == 4);
	REQUIRE(std::ranges::all_of(vals, [](move_probe const &p) { return p.v == move_probe::sentinel; }));

	move_probe::reset();
	tree moved(std::move(vals), move_probe(0), 0, op, map, add);
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(moved.at(1).v == move_probe::sentinel);
}

TEST_CASE("lazy_segment_tree: default construction and defaulted callables", "[lazy_segment_tree]") {
	auto min_of = [](ll x, ll y) { return (std::min)(x, y); };
	auto map = [](ll f, ll s) { return s + f; };
	auto add = [](ll f, ll g) { return f + g; };
	using tree = atl::lazy_segment_tree<ll, ll, decltype(min_of), decltype(map), decltype(add)>;
	tree st;
	REQUIRE(st.empty());
	REQUIRE(st.fold(0, 0) == 0);
	ll const inf = (std::numeric_limits<ll>::max)() / 2;
	st = tree(std::vector<ll>{5, 3, 8}, inf, 0);
	st.apply(0, 2, 10);
	REQUIRE(st.fold() == 8);
	st = tree(4, inf, 0);
	REQUIRE(st.fold() == inf);
}

// max_right / min_left under a non-commutative op. The aggregate carries the first
// and last element of its range, and the predicates demand they match the query's own
// boundary element, so a swapped operand order (op(node, sm) for op(sm, node)) fails.
TEST_CASE("lazy_segment_tree: max_right / min_left under a non-commutative op", "[lazy_segment_tree]") {
	struct node {
		ll sum = 0, width = 0, first = 0, last = 0;
	};

	auto const op = [](node const &x, node const &y) {
		if (x.width == 0) return y;
		if (y.width == 0) return x;
		return node{x.sum + y.sum, x.width + y.width, x.first, y.last};
	};
	auto const map = [](ll f, node const &s) {
		return s.width == 0 ? s : node{s.sum + f * s.width, s.width, s.first + f, s.last + f};
	};
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	int const n = 1 + int(rng() % 200);
	std::vector<ll> a(n);
	std::vector<node> init;
	for (auto &x : a) {
		x = ll(rng() % 50);
		init.push_back({x, 1, x, x});
	}
	atl::lazy_segment_tree st(init, node{}, ll(0), op, map, add_compose);
	for (int step = 0; step < 2000; ++step) {
		int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
		ll const bound = ll(rng() % 2000);
		switch (rng() % 3) {
		case 0: {
			ll const f = ll(rng() % 10);
			st.apply(l, r, f);
			for (int j = l; j < r; ++j) a[j] += f;
			break;
		}
		case 1: {
			// largest r' with sum(a[l, r')) <= bound
			int expect = l;
			for (ll s = 0; expect < n && s + a[expect] <= bound; ++expect) s += a[expect];
			auto const pred = [&](node const &s) { return s.width == 0 || (s.sum <= bound && s.first == a[l]); };
			REQUIRE(st.max_right(l, pred) == std::size_t(expect));
			break;
		}
		default: {
			// smallest l' with sum(a[l', r)) <= bound
			int expect = r;
			for (ll s = 0; expect > 0 && s + a[expect - 1] <= bound; --expect) s += a[expect - 1];
			auto const pred = [&](node const &s) { return s.width == 0 || (s.sum <= bound && s.last == a[r - 1]); };
			REQUIRE(st.min_left(r, pred) == std::size_t(expect));
			break;
		}
		}
	}
}

// the empty tree.
TEST_CASE("lazy_segment_tree: empty tree", "[lazy_segment_tree]") {
	atl::lazy_segment_tree st(std::vector<ll>{}, ll(0), ll(0), std::plus<>{}, add_map, add_compose);
	REQUIRE(st.empty());
	REQUIRE(st.size() == 0);
	REQUIRE(st.fold(0, 0) == 0);
	REQUIRE(st.fold() == 0);
	st.apply(0, 0, ll(5));
	REQUIRE(st.max_right(0, [](ll) { return true; }) == 0);
	REQUIRE(st.min_left(0, [](ll) { return true; }) == 0);
}
