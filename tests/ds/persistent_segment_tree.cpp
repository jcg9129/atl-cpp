#include <atl/ds/persistent_segment_tree.hpp>
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
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"
#include "move_probe.hpp"

using atl::uint32;
using atl::uint64;
using ll = long long;

TEST_CASE("persistent_segment_tree: version DAG vs brute force", "[persistent_segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 50; // deliberately not a power of two

	std::vector<ll> base(n);
	for (auto &v : base) v = ll(rng() % 1000);
	std::vector vs{atl::persistent_segment_tree(base, ll(0), std::plus<>{})};
	std::vector<std::vector<ll>> refs{base};

	for (int step = 0; step < 400; ++step) {
		std::size_t const k = rng() % vs.size();
		auto const &ref = refs[k];
		switch (rng() % 5) {
		case 4: { // paste [l, r) from version s onto version k
			std::size_t const s = rng() % vs.size();
			std::size_t const l = rng() % (n + 1), r = l + rng() % (n - l + 1);
			vs.push_back(vs[k].copy_range(vs[s], l, r));
			refs.push_back(ref);
			for (std::size_t j = l; j < r; ++j) refs.back()[j] = refs[s][j];
			break;
		}
		case 0: {
			std::size_t const i = rng() % n;
			ll const v = ll(rng() % 1000);
			vs.push_back(vs[k].set(i, v));
			refs.push_back(ref);
			refs.back()[i] = v;
			break;
		}
		case 1: {
			std::size_t const i = rng() % n;
			ll const d = ll(rng() % 100);
			vs.push_back(vs[k].update(i, [d](ll const &old) { return old + d; }));
			refs.push_back(ref);
			refs.back()[i] += d;
			break;
		}
		case 2: {
			std::size_t const i = rng() % n;
			REQUIRE(vs[k].at(i) == ref[i]);
			break;
		}
		default: {
			std::size_t const l = rng() % (n + 1), r = l + rng() % (n - l + 1);
			ll want = 0;
			for (std::size_t j = l; j < r; ++j) want += ref[j];
			REQUIRE(vs[k].fold(l, r) == want);
		}
		}
	}
	for (std::size_t k = 0; k < vs.size(); ++k) { // every version intact at the end
		ll want = 0;
		for (ll const x : refs[k]) want += x;
		REQUIRE(vs[k].fold() == want);
	}
}

TEST_CASE("persistent_segment_tree: descend as Chairman-tree kth smallest", "[persistent_segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 60, m = 1 + rng() % 30; // values in [0, m)

	std::vector<std::size_t> a(n);
	for (auto &v : a) v = rng() % m;

	// prefix count versions v_0 (all-zero) .. v_n over the value domain
	std::vector vs{atl::persistent_segment_tree(m, uint32(0), std::plus<>{})};
	for (std::size_t i = 0; i < n; ++i)
		vs.push_back(vs.back().update(a[i], [](uint32 const &c) { return c + 1; }));

	for (int step = 0; step < 200; ++step) {
		std::size_t const l = rng() % n, r = l + 1 + rng() % (n - l);
		std::size_t const k = rng() % (r - l);
		std::vector<std::size_t> slice(a.begin() + long(l), a.begin() + long(r));
		std::ranges::sort(slice);
		std::size_t got = atl::persistent_segment_tree<uint32, std::plus<>>::descend(
			vs[l], vs[r], [k = k](uint32 const &cl, uint32 const &cr) mutable {
				uint32 const c = cr - cl;
				if (k < c) return true;
				k -= c;
				return false;
			});
		REQUIRE(got == slice[k]);
	}
}

TEST_CASE("persistent_segment_tree: fill constructor, empty, edges", "[persistent_segment_tree]") {
	atl::persistent_segment_tree filled(std::vector<ll>(5, 7), ll(0), std::plus<>{});
	REQUIRE(filled.fold() == 35);
	REQUIRE(filled.at(4) == 7);
	auto v2 = filled.set(2, ll(0));
	REQUIRE(v2.fold(1, 4) == 14);
	REQUIRE(filled.fold(1, 4) == 21); // old version untouched
	REQUIRE(v2.fold(2, 2) == 0);

	atl::persistent_segment_tree empty(std::size_t(0), ll(0), std::plus<>{});
	REQUIRE(empty.empty());
	REQUIRE(empty.fold() == 0);
}

TEST_CASE("persistent_segment_tree: default construction, then assignment", "[persistent_segment_tree]") {
	atl::persistent_segment_tree<ll, std::plus<>> t;
	REQUIRE(t.empty());
	REQUIRE(t.fold() == 0);
	t = atl::persistent_segment_tree<ll, std::plus<>>(std::vector<ll>{1, 2, 3}, 0);
	REQUIRE(t.fold() == 6);
	REQUIRE(atl::persistent_segment_tree<ll, std::plus<>>(4, 0).fold(0, 4) == 0); // defaulted Op
}

// at returns by value: no reference into the shared arena can dangle.
static_assert(std::is_same_v<decltype(std::declval<atl::persistent_segment_tree<ll, std::plus<>> const &>().at(0)), ll>);

// update requires a callable on `T const &` whose result converts to `T`.
template<typename Update>
concept pst_updatable = requires(atl::persistent_segment_tree<ll, std::plus<>> const &t, Update f) { t.update(0, f); };
static_assert(pst_updatable<ll (*)(ll const &)>);
static_assert(!pst_updatable<void (*)(ll const &)>);
static_assert(!pst_updatable<ll (*)(ll &)>);

TEST_CASE("persistent_segment_tree: copy_range edges and arena sharing", "[persistent_segment_tree]") {
	atl::persistent_segment_tree a(std::vector<ll>{1, 2, 3, 4, 5}, ll(0), std::plus<>{});
	auto b = a.set(0, 10).set(4, 50);
	REQUIRE(a.copy_range(b, 0, 5).fold() == b.fold()); // whole range: the source
	REQUIRE(a.copy_range(b, 2, 2).fold() == a.fold()); // empty range: unchanged
	auto c = a.copy_range(b, 3, 5);
	REQUIRE(c.at(0) == 1);
	REQUIRE(c.at(4) == 50);
	REQUIRE(c.fold() == 1 + 2 + 3 + 4 + 50);
	std::size_t const before = a.arena_size();
	(void)a.copy_range(b, 0, 5);
	REQUIRE(a.arena_size() == before); // full-range paste adopts the source root
}

TEST_CASE("persistent_segment_tree: range constructor forwarding rule", "[persistent_segment_tree]") {
	using atl_test::move_probe;
	auto const op = [](move_probe const &a, move_probe const &b) { return move_probe(a.v + b.v); };
	std::vector<move_probe> vals(8, move_probe(move_probe::sentinel));

	move_probe::reset();
	atl::persistent_segment_tree copied(vals, move_probe(0), op);
	REQUIRE(move_probe::sentinel_copies >= 8); // lvalue range: copied

	move_probe::reset();
	atl::persistent_segment_tree spanned{std::span(vals), move_probe(0), op};
	REQUIRE(move_probe::sentinel_copies >= 8); // borrowed rvalue: copied, the source stays intact
	REQUIRE(std::ranges::all_of(vals, [](move_probe const &p) { return p.v == move_probe::sentinel; }));

	// a sized, non-borrowed rvalue view over the caller's container: copied too
	atl::persistent_segment_tree transformed(vals | std::views::transform([](move_probe &p) -> move_probe & { return p; }), move_probe(0), op);
	REQUIRE(std::ranges::all_of(vals, [](move_probe const &p) { return p.v == move_probe::sentinel; }));
	REQUIRE(transformed.at(5).v == move_probe::sentinel);

	move_probe::reset();
	atl::persistent_segment_tree moved(std::move(vals), move_probe(0), op);
	REQUIRE(move_probe::sentinel_copies == 0); // owning rvalue range: moved into the leaves
	REQUIRE(moved.at(3).v == move_probe::sentinel);
}

TEST_CASE("persistent_segment_tree: rvalue overloads match the lvalue ones and leave other versions intact", "[persistent_segment_tree]") {
	using tree = decltype(atl::persistent_segment_tree(std::vector<ll>{}, ll(0), std::plus<>{}));
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, ll) const &>(&tree::set); });
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, ll) &&>(&tree::set); });
	static_assert(requires { static_cast<tree (tree::*)(tree const &, std::size_t, std::size_t) const &>(&tree::copy_range); });
	static_assert(requires { static_cast<tree (tree::*)(tree const &, std::size_t, std::size_t) &&>(&tree::copy_range); });
	static_assert(requires(tree t) { { std::move(t).update(0, [](ll x) { return x; }) } -> std::same_as<tree>; });

	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 50;
	std::vector<ll> ref(n);
	for (auto &v : ref) v = ll(rng() % 1000);
	auto same = [n](tree const &t, std::vector<ll> const &r) {
		for (std::size_t i = 0; i < n; ++i)
			if (t.at(i) != r[i]) return false;
		return t.fold(0, n) == std::accumulate(r.begin(), r.end(), ll(0));
	};
	tree moved(ref, ll(0), std::plus<>{});
	tree copied = moved;
	std::vector<std::pair<tree, std::vector<ll>>> kept;
	for (int step = 0; step < 600; ++step) {
		if (rng() % 20 == 0) kept.emplace_back(moved, ref);
		std::size_t const i = rng() % n;
		ll const v = ll(rng() % 1000);
		switch (rng() % 3) {
		case 0:
			moved = std::move(moved).set(i, v);
			copied = copied.set(i, v);
			ref[i] = v;
			break;
		case 1:
			moved = std::move(moved).update(i, [v](ll x) { return x + v; });
			copied = copied.update(i, [v](ll x) { return x + v; });
			ref[i] += v;
			break;
		default: {
			auto const &[src, src_ref] = kept.empty() ? std::pair{moved, ref} : kept[rng() % kept.size()];
			std::size_t const l = rng() % (n + 1), r = l + rng() % (n - l + 1);
			moved = std::move(moved).copy_range(src, l, r);
			copied = copied.copy_range(src, l, r);
			for (std::size_t j = l; j < r; ++j) ref[j] = src_ref[j];
		}
		}
		REQUIRE(same(moved, ref));
		REQUIRE(same(copied, ref));
	}
	for (auto const &[t, r] : kept) REQUIRE(same(t, r));
	// self-source paste on an rvalue falls back to the copying path
	moved = std::move(moved).copy_range(moved, 0, n);
	REQUIRE(same(moved, ref));
	tree a = moved;
	auto b = std::move(a).set(0, 5);
	a = b; // assignment into a moved-from tree
	REQUIRE(a.at(0) == 5);
	auto c = std::move(a).set(0, 6);
	REQUIRE(c.at(0) == 6);
}

TEST_CASE("persistent segment tree rejects overflowing reserve estimates", "[persistent_segment_tree]") {
	REQUIRE(atl_test::aborts([] {
		atl::persistent_segment_tree<int, std::plus<>> tree(2, 0);
		tree.reserve((std::numeric_limits<std::size_t>::max)());
	}));
}
