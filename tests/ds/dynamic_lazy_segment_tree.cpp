#include <atl/ds/dynamic_lazy_segment_tree.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <functional>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <cstddef>

using atl::uint64;
using ll = long long;

namespace {
	constexpr ll mod = 1'000'003;

	struct item {
		ll s, w;
	};

	struct aff {
		ll a, b;

		bool operator==(aff const &) const = default;
	};

	auto const op = [](item const &x, item const &y) { return item{(x.s + y.s) % mod, x.w + y.w}; };
	auto const mp = [](aff const &f, item const &s) { return item{(f.a * s.s + f.b * s.w) % mod, s.w}; };
	auto const cp = [](aff const &f, aff const &g) { return aff{f.a * g.a % mod, (f.a * g.b + f.b) % mod}; };
	auto const fill = [](std::size_t w) { return item{0, ll(w)}; };

	/** @brief Whether the class constraints accept `Fill` as the default-fold callable. */
	template<typename Fill>
	constexpr bool accepts_fill = requires {
		typename atl::dynamic_lazy_segment_tree<item, aff, decltype(op), decltype(mp), decltype(cp), Fill>;
	};

	// fill is only ever invoked through a const reference, so both arities are
	// checked as `Fill const &`: a mutable (caching) callable is rejected at the
	// class boundary instead of failing inside the body
	static_assert(accepts_fill<decltype(fill)>);
	static_assert(accepts_fill<decltype([](std::size_t lo, std::size_t hi) { return item{0, ll(hi - lo)}; })>);
	static_assert(!accepts_fill<decltype([n = 0](std::size_t w) mutable { return item{++n, ll(w)}; })>);
	static_assert(atl::fill_function<decltype(fill), item>);
	static_assert(!atl::fill_function<decltype([](std::size_t, std::size_t, std::size_t) { return item{}; }), item>);
}

TEST_CASE("dynamic_lazy_segment_tree: vs brute force", "[dynamic_lazy_segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 60;

	atl::dynamic_lazy_segment_tree t(n, item{0, 0}, aff{1, 0}, op, mp, cp, fill);
	std::vector<ll> ref(n, 0);

	for (int step = 0; step < 400; ++step) {
		switch (rng() % 5) {
		case 4: {
			std::size_t const i = rng() % n;
			aff const f{1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
			t.apply(i, f);
			ref[i] = (f.a * ref[i] + f.b) % mod;
			break;
		}
		case 0: {
			std::size_t const l = rng() % (n + 1), r = l + rng() % (n - l + 1);
			aff const f{1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
			t.apply(l, r, f);
			for (std::size_t j = l; j < r; ++j) ref[j] = (f.a * ref[j] + f.b) % mod;
			break;
		}
		case 1: {
			std::size_t const i = rng() % n;
			ll const v = ll(rng() % mod);
			t.set(i, item{v, 1});
			ref[i] = v;
			break;
		}
		case 2: {
			std::size_t const i = rng() % n;
			REQUIRE(t.at(i).s == ref[i]);
			break;
		}
		default: {
			std::size_t const l = rng() % (n + 1), r = l + rng() % (n - l + 1);
			ll want = 0;
			for (std::size_t j = l; j < r; ++j) want = (want + ref[j]) % mod;
			REQUIRE(t.fold(l, r).s == want);
		}
		}
	}
	ll total = 0;
	for (ll const v : ref) total = (total + v) % mod;
	REQUIRE(t.fold().s == total);
}

TEST_CASE("dynamic_lazy_segment_tree: huge domain stays sparse", "[dynamic_lazy_segment_tree]") {
	std::size_t const n = std::size_t((std::min)(1'000'000'000'000ull, static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)())));
	atl::dynamic_lazy_segment_tree t(n, item{0, 0}, aff{1, 0}, op, mp, cp, fill);

	t.apply(0, n, {1, 5}); // += 5 everywhere: one node touched
	REQUIRE(t.node_count() <= 3);
	REQUIRE(t.fold().s == 5 * ll(n % mod) % mod);
	REQUIRE(t.fold(17, 20).s == 15);
	REQUIRE(t.at(n - 1).s == 5);

	t.apply(1000, 1'000'000'000ull, {2, 1}); // deep partial range
	REQUIRE(t.fold(999, 1001).s == (5 + 11) % mod);
	REQUIRE(t.fold(0, 1000).s == 5000 % mod);
	std::size_t const inner = 1'000'000'000ull - 1000;
	REQUIRE(t.fold(1000, 1'000'000'000ull).s == 11 * ll(inner % mod) % mod);
	REQUIRE(t.node_count() <= 1 + 4 * 41);

	t.set(n - 1, item{7, 1});
	REQUIRE(t.at(n - 1).s == 7);
	REQUIRE(t.fold(n - 2, n).s == 12);
}

TEST_CASE("dynamic_lazy_segment_tree: tag type without operator==", "[dynamic_lazy_segment_tree]") {
	struct opaque {
		ll a, b; // deliberately not equality-comparable: the always-live path
	};

	auto mp2 = [](opaque const &f, item const &s) { return item{(f.a * s.s + f.b * s.w) % mod, s.w}; };
	auto cp2 = [](opaque const &f, opaque const &g) {
		return opaque{f.a * g.a % mod, (f.a * g.b + f.b) % mod};
	};
	std::mt19937_64 rng(5);
	std::size_t const n = 40;
	atl::dynamic_lazy_segment_tree t(n, item{0, 0}, opaque{1, 0}, op, mp2, cp2, fill);
	std::vector<ll> ref(n, 0);
	for (int step = 0; step < 200; ++step) {
		std::size_t const l = rng() % (n + 1), r = l + rng() % (n - l + 1);
		if (step % 2 == 0) {
			opaque const f{1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
			t.apply(l, r, f);
			for (std::size_t j = l; j < r; ++j) ref[j] = (f.a * ref[j] + f.b) % mod;
		} else {
			ll want = 0;
			for (std::size_t j = l; j < r; ++j) want = (want + ref[j]) % mod;
			REQUIRE(t.fold(l, r).s == want);
		}
	}
}

TEST_CASE("dynamic_lazy_segment_tree: position-aware fill (periodic defaults)", "[dynamic_lazy_segment_tree]") {
	std::mt19937_64 rng(9);
	std::size_t const n = 7, k = 5, N = n * k;
	std::vector<ll> base(n);
	for (auto &v : base) v = ll(rng() % 100);

	auto op = [](ll x, ll y) { return x < y ? x : y; };
	auto mp = [](ll f, ll s) { return f ? f : s; }; // assign, 0 = identity
	auto cp = [](ll f, ll g) { return f ? f : g; };
	auto fill = [&](std::size_t lo, std::size_t hi) {
		ll m = (std::numeric_limits<ll>::max)();
		for (std::size_t j = lo; j < hi; ++j) m = (std::min)(m, base[j % n]);
		return m;
	};
	atl::dynamic_lazy_segment_tree t(N, (std::numeric_limits<ll>::max)(), ll(0), op, mp, cp, fill);

	std::vector<ll> ref;
	for (std::size_t c = 0; c < k; ++c) ref.insert(ref.end(), base.begin(), base.end());
	for (int step = 0; step < 300; ++step) {
		std::size_t const l = rng() % N, r = l + 1 + rng() % (N - l);
		if (step % 2 == 0) {
			ll const x = 1 + ll(rng() % 100);
			t.apply(l, r, x);
			for (std::size_t j = l; j < r; ++j) ref[j] = x;
		} else {
			ll want = (std::numeric_limits<ll>::max)();
			for (std::size_t j = l; j < r; ++j) want = (std::min)(want, ref[j]);
			REQUIRE(t.fold(l, r) == want);
		}
	}
}

TEST_CASE("dynamic_lazy_segment_tree: max_right vs brute", "[dynamic_lazy_segment_tree]") {
	std::mt19937_64 rng(3);
	std::size_t const n = 200;
	// a huge implicit domain, but exercise max_right over the touched prefix:
	// leaf = present flag {cnt, w}; pred = "prefix fully present" -> first absent
	auto op = [](item const &x, item const &y) { return item{x.s + y.s, x.w + y.w}; };
	auto mp = [](aff const &f, item const &s) { return item{f.a ? s.w : (f.b ? 0 : s.s), s.w}; };
	auto cp = [](aff const &f, aff const &g) { return f.a || f.b ? f : g; }; // assign overrides
	auto fill = [](std::size_t w) { return item{0, ll(w)}; };
	// aff here reused as an assign tag: {a=set-to-1, b=set-to-0}, {0,0}=identity
	atl::dynamic_lazy_segment_tree t(n, item{0, 0}, aff{0, 0}, op, mp, cp, fill);
	std::vector<int> ref(n, 0);

	auto mex = [&] {
		std::size_t i = 0;
		while (i < n && ref[i]) ++i;
		return i;
	};
	for (int step = 0; step < 400; ++step) {
		std::size_t const l = rng() % n, r = l + 1 + rng() % (n - l);
		bool const add = rng() % 2;
		t.apply(l, r, add ? aff{1, 0} : aff{0, 1});
		for (std::size_t j = l; j < r; ++j) ref[j] = add;
		std::size_t const got = t.max_right(0, [](item const &s) { return s.s == s.w; });
		REQUIRE(got == mex());
	}
}

TEST_CASE("dynamic_lazy_segment_tree: default construction and defaulted callables", "[dynamic_lazy_segment_tree]") {
	using tree = atl::dynamic_lazy_segment_tree<item, aff, std::remove_cvref_t<decltype(op)>, std::remove_cvref_t<decltype(mp)>,
												std::remove_cvref_t<decltype(cp)>, std::remove_cvref_t<decltype(fill)>>;
	tree t;
	REQUIRE(t.empty());
	REQUIRE(t.size() == 0);
	t = tree(8, item{0, 0}, aff{1, 0});
	REQUIRE_FALSE(t.empty());
	t.apply(0, 8, aff{1, 2});
	t.apply(3, aff{3, 0});
	REQUIRE(t.at(3).s == 6);
	REQUIRE(t.fold(0, 8).s == 2 * 7 + 6);
}

namespace {
	struct counted_width_fill {
		int *calls;

		int operator()(std::size_t width) const {
			++*calls;
			return int(width) * 5;
		}
	};

	struct counted_position_fill {
		int *calls;

		int operator()(std::size_t lo, std::size_t hi) const {
			++*calls;
			return int(hi - lo) * 7;
		}
	};

	struct counted_both_fill {
		int *width_calls, *position_calls;

		int operator()(std::size_t width) const {
			++*width_calls;
			return int(width) * 11;
		}

		int operator()(std::size_t lo, std::size_t hi) const {
			++*position_calls;
			return int(hi - lo) * 3;
		}
	};
}

TEST_CASE("dynamic_lazy_segment_tree: fill selection follows invocable position precedence", "[dynamic_lazy_segment_tree]") {
	int width_calls = 0, position_calls = 0;
	auto mapping = [](int, int value) { return value; };
	atl::dynamic_lazy_segment_tree width(std::size_t(8), 0, 0, std::plus<>{}, mapping, std::plus<>{}, counted_width_fill{&width_calls});
	REQUIRE(width.at(0) == 5);
	REQUIRE(width.at(7) == 5);
	REQUIRE(width_calls > 0);
	atl::dynamic_lazy_segment_tree position(std::size_t(8), 0, 0, std::plus<>{}, mapping, std::plus<>{}, counted_position_fill{&position_calls});
	REQUIRE(position.at(0) == 7);
	REQUIRE(position.at(7) == 7);
	REQUIRE(position_calls > 0);
	width_calls = position_calls = 0;
	atl::dynamic_lazy_segment_tree both(std::size_t(8), 0, 0, std::plus<>{}, mapping, std::plus<>{}, counted_both_fill{&width_calls, &position_calls});
	REQUIRE(both.at(0) == 3);
	REQUIRE(both.at(7) == 3);
	REQUIRE(both.fold(2, 6) == 12);
	both.set(0, 9);
	REQUIRE(both.fold() == 30);
	REQUIRE(width_calls == 0);
	REQUIRE(position_calls > 0);
}

#include "../expect_abort.hpp"

TEST_CASE("dynamic_lazy_segment_tree: overflowing reserve estimate aborts", "[dynamic_lazy_segment_tree]") {
	REQUIRE(atl_test::aborts([] {
		atl::dynamic_lazy_segment_tree library(8, item{0, 0}, aff{1, 0}, op, mp, cp, fill);
		library.reserve((std::numeric_limits<std::size_t>::max)());
	}));
}
