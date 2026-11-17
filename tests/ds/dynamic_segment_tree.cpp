#include <atl/ds/dynamic_segment_tree.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <cstddef>

#include <utility>

using atl::uint64;
using ll = long long;

TEST_CASE("dynamic_segment_tree: vs brute force, non-commutative op", "[dynamic_segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	constexpr ll mod = 1'000'003;
	std::size_t const n = 1 + rng() % 60;

	// affine composite in sequence order: op(x, y) = y ∘ x — order-sensitive,
	// so left/right mixups cannot survive the test
	struct aff {
		ll a, b;
	};

	auto op = [](aff const &x, aff const &y) { return aff{y.a * x.a % mod, (y.a * x.b + y.b) % mod}; };
	atl::dynamic_segment_tree t(n, aff{1, 0}, op);
	std::vector<aff> ref(n, {1, 0});

	for (int step = 0; step < 500; ++step) {
		switch (rng() % 4) {
		case 0: {
			std::size_t const i = rng() % n;
			aff const f{1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
			t.set(i, f);
			ref[i] = f;
			break;
		}
		case 1: {
			std::size_t const i = rng() % n;
			ll const d = ll(rng() % mod);
			t.update(i, [&](aff const &old) { return aff{old.a, (old.b + d) % mod}; });
			ref[i].b = (ref[i].b + d) % mod;
			break;
		}
		case 2: {
			std::size_t const i = rng() % n;
			REQUIRE(t.at(i).a == ref[i].a);
			REQUIRE(t.at(i).b == ref[i].b);
			break;
		}
		default: {
			std::size_t const l = rng() % (n + 1), r = l + rng() % (n - l + 1);
			ll const x = ll(rng() % mod);
			ll want = x;
			for (std::size_t j = l; j < r; ++j) want = (ref[j].a * want + ref[j].b) % mod;
			aff const f = t.fold(l, r);
			REQUIRE((f.a * x + f.b) % mod == want);
		}
		}
	}
}

TEST_CASE("dynamic_segment_tree: huge domain stays sparse", "[dynamic_segment_tree]") {
	std::size_t const n = std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 2);
	atl::dynamic_segment_tree t(n, ll(0), std::plus<>{});
	REQUIRE(t.node_count() == 1); // just the null sentinel

	std::mt19937_64 rng(7);
	std::vector<std::pair<std::size_t, ll>> touched;
	for (int i = 0; i < 500; ++i) {
		std::size_t const p = rng() % n;
		ll const v = ll(rng() % 1000);
		t.set(p, v);
		touched.emplace_back(p, v);
	}
	REQUIRE(t.node_count() <= 1 + 500 * 63);

	// the final value of each touched position (later sets win)
	ll total = 0;
	std::vector<std::pair<std::size_t, ll>> fin;
	for (auto it = touched.rbegin(); it != touched.rend(); ++it) {
		bool seen = false;
		for (auto &[p, v] : fin) seen |= p == it->first;
		if (!seen) fin.push_back(*it);
	}
	for (auto &[p, v] : fin) total += v;
	REQUIRE(t.fold() == total);
	REQUIRE(t.fold(0, n) == total);
	REQUIRE(t.at(fin.front().first) == fin.front().second);
	REQUIRE(t.at(123) == 0); // untouched position reads the identity
}

TEST_CASE("dynamic_segment_tree: max_right / min_left vs brute force", "[dynamic_segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = rng() % 70;
	atl::dynamic_segment_tree t(n, ll(0), std::plus<>{});
	std::vector<ll> ref(n, 0);
	for (int step = 0; step < 400; ++step) {
		if (n > 0 && rng() % 3 == 0) {
			std::size_t const i = rng() % n;
			ll const v = ll(rng() % 10);
			t.set(i, v);
			ref[i] = v;
			continue;
		}
		ll const cap = ll(rng() % 40);
		auto const pred = [cap](ll s) { return s <= cap; };
		std::size_t const l = rng() % (n + 1);
		std::size_t want_r = l;
		for (ll s = 0; want_r < n && s + ref[want_r] <= cap; ++want_r) s += ref[want_r];
		REQUIRE(t.max_right(l, pred) == want_r);
		std::size_t const r = rng() % (n + 1);
		std::size_t want_l = r;
		for (ll s = 0; want_l > 0 && s + ref[want_l - 1] <= cap; --want_l) s += ref[want_l - 1];
		REQUIRE(t.min_left(r, pred) == want_l);
	}
}

TEST_CASE("dynamic_segment_tree: search over a huge sparse domain", "[dynamic_segment_tree]") {
	std::size_t const n = std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 4);
	atl::dynamic_segment_tree t(n, ll(0), std::plus<>{});
	t.set(5, 1);
	t.set(n / 2, 1);
	t.set(n - 1, 1);
	auto const at_most = [](ll k) { return [k](ll s) { return s <= k; }; };
	REQUIRE(t.max_right(0, at_most(0)) == 5);
	REQUIRE(t.max_right(0, at_most(1)) == n / 2);
	REQUIRE(t.max_right(6, at_most(1)) == n - 1);
	REQUIRE(t.max_right(0, at_most(3)) == n);
	REQUIRE(t.min_left(n, at_most(0)) == n);
	REQUIRE(t.min_left(n, at_most(1)) == n / 2 + 1);
	REQUIRE(t.min_left(n - 1, at_most(1)) == 6);
	REQUIRE(t.min_left(n, at_most(3)) == 0);
}

TEST_CASE("dynamic_segment_tree: default construction and defaulted op", "[dynamic_segment_tree]") {
	atl::dynamic_segment_tree<ll, std::plus<>> t;
	REQUIRE(t.empty());
	REQUIRE(t.size() == 0);
	REQUIRE(t.max_right(0, [](ll) { return true; }) == 0);
	REQUIRE(t.min_left(0, [](ll) { return true; }) == 0);
	t = atl::dynamic_segment_tree<ll, std::plus<>>(10, 0);
	REQUIRE_FALSE(t.empty());
	t.set(3, 4);
	t.update(3, [](ll v) { return v * 2; });
	REQUIRE(t.fold(0, 10) == 8);
}

TEST_CASE("dynamic_segment_tree: point transformations run exactly once", "[dynamic_segment_tree]") {
	atl::dynamic_segment_tree<int, std::plus<>> library(std::size_t(9), 0);
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

TEST_CASE("dynamic_segment_tree: overflowing reserve estimate aborts", "[dynamic_segment_tree]") {
	REQUIRE(atl_test::aborts([] {
		atl::dynamic_segment_tree<int, std::plus<>> library(8, 0);
		library.reserve((std::numeric_limits<std::size_t>::max)());
	}));
}

TEST_CASE("dynamic_segment_tree: suffix boundaries can equal size_t maximum", "[dynamic_segment_tree]") {
	std::size_t const n = (std::numeric_limits<std::size_t>::max)();
	atl::dynamic_segment_tree<int, std::plus<>> library(n, 0);
	library.set(n - 1, 1);
	REQUIRE(library.min_left(n, [](int sum) { return sum == 0; }) == n);
	REQUIRE(library.max_right(n - 1, [](int sum) { return sum == 0; }) == n - 1);
	REQUIRE(library.min_left(n - 1, [](int sum) { return sum == 0; }) == 0);
	library.set(0, 2);
	REQUIRE(library.min_left(n, [](int sum) { return sum <= 1; }) == 1);
}
