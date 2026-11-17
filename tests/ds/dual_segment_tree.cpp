#include <atl/ds/dual_segment_tree.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <functional>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>
#include "move_probe.hpp"

#include <cstddef>

using atl::uint64;
using ll = long long;

namespace {
	struct affine {
		ll b, c;
	};

	auto const affine_map = [](affine const &f, ll s) { return f.b * s + f.c; };
	auto const affine_compose = [](affine const &f, affine const &g) { return affine{f.b * g.b, f.b * g.c + f.c}; };
}

// S is deduced from the range's elements.
static_assert(std::is_same_v<decltype(atl::dual_segment_tree(std::declval<std::vector<ll> &>(), affine{1, 0},
															 affine_map, affine_compose))::value_type,
							 ll>);

TEST_CASE("dual_segment_tree: range affine / point get vs brute force", "[dual_segment_tree]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	int const n = 1 + int(rng() % 200);
	std::vector<ll> a(n);
	for (auto &x : a) x = ll(rng() % 100);
	atl::dual_segment_tree st(a, affine{1, 0}, affine_map, affine_compose);
	for (int step = 0; step < 2000; ++step) {
		switch (rng() % 5) {
		case 0: { // range affine (small coefficients keep the brute values exact)
			int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
			affine const f{ll(rng() % 3), ll(rng() % 50) - 25};
			st.apply(std::size_t(l), std::size_t(r), f);
			for (int j = l; j < r; ++j) a[j] = f.b * a[j] + f.c;
			break;
		}
		case 1: {
			int const i = int(rng() % n);
			REQUIRE(st.at(std::size_t(i)) == a[i]);
			break;
		}
		case 2: {
			int const i = int(rng() % n);
			ll const v = ll(rng() % 100);
			st.set(std::size_t(i), v);
			a[i] = v;
			break;
		}
		default: { // point apply
			int const i = int(rng() % n);
			affine const f{ll(rng() % 3), ll(rng() % 50) - 25};
			st.apply(std::size_t(i), f);
			a[i] = f.b * a[i] + f.c;
		}
		}
		// values drift; rein them in so ll never overflows
		if (step % 50 == 49)
			for (int j = 0; j < n; ++j) {
				st.set(std::size_t(j), a[j] % 1000);
				a[j] %= 1000;
			}
	}
	for (int i = 0; i < n; ++i) REQUIRE(st.at(std::size_t(i)) == a[i]);
}

TEST_CASE("dual_segment_tree: non-commutative assign tags", "[dual_segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	int const n = 1 + int(rng() % 150);
	constexpr ll no_tag = -1;
	std::vector<ll> a(n);
	for (auto &x : a) x = ll(rng() % 100);
	atl::dual_segment_tree st(
		a, no_tag, [](ll f, ll s) { return f == no_tag ? s : f; },
		[](ll f, ll g) { return f == no_tag ? g : f; }); // newer assignment wins
	for (int step = 0; step < 1500; ++step) {
		if (rng() % 2) {
			int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
			ll const f = ll(rng() % 100);
			st.apply(std::size_t(l), std::size_t(r), f);
			for (int j = l; j < r; ++j) a[j] = f;
		} else {
			int const i = int(rng() % n);
			REQUIRE(st.at(std::size_t(i)) == a[i]);
		}
	}
}

TEST_CASE("dual_segment_tree: edge cases", "[dual_segment_tree]") {
	atl::dual_segment_tree one(std::size_t(1), ll(5), affine{1, 0}, affine_map, affine_compose);
	one.apply(0, 1, affine{2, 1}); // 5 -> 11
	one.apply(std::size_t(0), affine{1, 4});
	REQUIRE(one.at(0) == 15);
	one.set(0, 42);
	REQUIRE(one.at(0) == 42);

	atl::dual_segment_tree st(std::vector<ll>{1, 2, 3, 4}, affine{1, 0}, affine_map, affine_compose);
	st.apply(2, 2, affine{100, 100}); // empty range: no-op
	st.apply(0, 4, affine{1, 0});	  // identity tag: no-op
	for (int i = 0; i < 4; ++i) REQUIRE(st.at(std::size_t(i)) == i + 1);
	st.apply(1, 3, affine{10, 0});
	REQUIRE(st.at(0) == 1);
	REQUIRE(st.at(1) == 20);
	REQUIRE(st.at(2) == 30);
	REQUIRE(st.at(3) == 4);
	REQUIRE(st.size() == 4);
	REQUIRE_FALSE(st.empty());
}

TEST_CASE("dual_segment_tree: rvalue range moves its values in", "[dual_segment_tree]") {
	using atl_test::move_probe;
	auto const probe_map = [](int f, move_probe const &s) { return move_probe(s.v + f); };
	auto const add = [](int f, int g) { return f + g; };

	std::vector<move_probe> vals(4, move_probe(move_probe::sentinel));
	move_probe::reset();
	atl::dual_segment_tree copied(vals, 0, probe_map, add);
	REQUIRE(move_probe::sentinel_copies == 4); // lvalue range: one copy per element

	move_probe::reset();
	atl::dual_segment_tree moved(std::move(vals), 0, probe_map, add);
	REQUIRE(move_probe::sentinel_copies == 0); // rvalue range: moved, never copied
	REQUIRE(moved.at(2).v == move_probe::sentinel);
}

TEST_CASE("dual_segment_tree: borrowed ranges and views are copied from", "[dual_segment_tree]") {
	using atl_test::move_probe;
	auto probe_map = [](int f, move_probe const &s) { return move_probe(s.v + f); };
	auto add = [](int f, int g) { return f + g; };
	using tree = atl::dual_segment_tree<move_probe, int, decltype(probe_map), decltype(add)>;
	std::vector<move_probe> vals(4, move_probe(move_probe::sentinel));

	move_probe::reset();
	tree from_span(std::span<move_probe>(vals), 0, probe_map, add);
	REQUIRE(move_probe::sentinel_copies == 4);
	auto const view = vals | std::views::transform([](move_probe &p) -> move_probe & { return p; });
	move_probe::reset();
	tree from_view(std::move(view), 0, probe_map, add);
	REQUIRE(move_probe::sentinel_copies == 4);
	REQUIRE(std::ranges::all_of(vals, [](move_probe const &p) { return p.v == move_probe::sentinel; }));
	REQUIRE(from_view.at(3).v == move_probe::sentinel);
}

TEST_CASE("dual_segment_tree: values() matches at() after random applies", "[dual_segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	int const n = int(rng() % 150);
	std::vector<ll> a(n);
	for (auto &x : a) x = ll(rng() % 100);
	atl::dual_segment_tree st(a, affine{1, 0}, affine_map, affine_compose);
	for (int step = 0; step < 300; ++step) {
		int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
		// b in {0, 1} keeps every value bounded; b = 0 exercises the non-commutative compose
		affine const f{ll(rng() % 4 != 0), ll(rng() % 50) - 25};
		st.apply(std::size_t(l), std::size_t(r), f);
		for (int j = l; j < r; ++j) a[j] = f.b * a[j] + f.c;
		if (step % 7 == 6 && n > 0) {
			int const i = int(rng() % n);
			st.set(std::size_t(i), a[i] = ll(rng() % 100));
		}
		if (step % 10 == 0) {
			std::vector<ll> got = st.values();
			REQUIRE(got.size() == std::size_t(n));
			for (int j = 0; j < n; ++j) REQUIRE(got[j] == st.at(std::size_t(j)));
			REQUIRE(got == a);
		}
	}
}

TEST_CASE("dual_segment_tree: default construction and defaulted callables", "[dual_segment_tree]") {
	auto add_map = [](ll f, ll s) { return s + f; };
	auto add = [](ll f, ll g) { return f + g; };
	using tree = atl::dual_segment_tree<ll, ll, decltype(add_map), decltype(add)>;
	tree st;
	REQUIRE(st.empty());
	REQUIRE(st.values().empty());
	st = tree(std::vector<ll>{1, 2, 3}, 0);
	st.apply(0, 2, 10);
	REQUIRE(st.values() == std::vector<ll>{11, 12, 3});
	st = tree(4, 5, 0);
	st.apply(1, 4, 1);
	REQUIRE(st.values() == std::vector<ll>{5, 6, 6, 6});
}

namespace {
	/**
	 * @brief Random range / point applies, point sets and point gets of a commutative-mode
	 *        tree against a plain array, after checking that the tags commute on the
	 *        drawn samples (the mode's precondition).
	 */
	template<typename Draw, atl::action<ll, ll> Map, atl::binary_operation<ll> Compose>
		requires std::is_invocable_r_v<ll, Draw &, std::mt19937_64 &>
	void commutative_vs_brute(std::mt19937_64 &rng, Draw draw, ll tag_identity, Map map, Compose compose) {
		for (int t = 0; t < 1000; ++t) {
			ll const f = draw(rng), g = draw(rng);
			REQUIRE(compose(f, g) == compose(g, f));
		}
		int const n = 1 + int(rng() % 200);
		std::vector<ll> a(n);
		for (auto &x : a) x = ll(rng() % 100);
		atl::dual_segment_tree st(atl::commutative_tags, a, tag_identity, map, compose);
		static_assert(std::is_same_v<decltype(st), atl::dual_segment_tree<ll, ll, Map, Compose, atl::commutative_tags_t>>);
		for (int step = 0; step < 3000; ++step) {
			switch (rng() % 5) {
			case 0:
			case 1: {
				int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
				ll const f = draw(rng);
				st.apply(std::size_t(l), std::size_t(r), f);
				for (int j = l; j < r; ++j) a[j] = map(f, a[j]);
				break;
			}
			case 2: {
				int const i = int(rng() % n);
				ll const f = draw(rng);
				st.apply(std::size_t(i), f);
				a[i] = map(f, a[i]);
				break;
			}
			case 3: {
				int const i = int(rng() % n);
				st.set(std::size_t(i), a[i] = ll(rng() % 100));
				break;
			}
			default: {
				int const i = int(rng() % n);
				REQUIRE(st.at(std::size_t(i)) == a[i]);
			}
			}
			if (step % 100 == 99) REQUIRE(st.values() == a);
		}
		for (int i = 0; i < n; ++i) REQUIRE(st.at(std::size_t(i)) == a[i]);
	}
}

TEST_CASE("dual_segment_tree: commutative mode vs brute force (add, xor, chmin, chmax)", "[dual_segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto const add = [](ll f, ll s) { return f + s; };
	commutative_vs_brute(rng, [](std::mt19937_64 &g) { return ll(g() % 2001) - 1000; }, 0, add, add);
	auto const bit_xor = [](ll f, ll s) { return f ^ s; };
	commutative_vs_brute(rng, [](std::mt19937_64 &g) { return ll(g() % 1024); }, 0, bit_xor, bit_xor);
	auto const chmin = [](ll f, ll s) { return (std::min)(f, s); };
	commutative_vs_brute(rng, [](std::mt19937_64 &g) { return ll(g() % 120); }, (std::numeric_limits<ll>::max)(), chmin, chmin);
	auto const chmax = [](ll f, ll s) { return (std::max)(f, s); };
	commutative_vs_brute(rng, [](std::mt19937_64 &g) { return ll(g() % 120) - 10; }, (std::numeric_limits<ll>::min)(), chmax, chmax);
}

TEST_CASE("dual_segment_tree: commutative mode, fill constructor and explicit spelling", "[dual_segment_tree]") {
	auto add = [](ll f, ll s) { return f + s; };
	using tree = atl::dual_segment_tree<ll, ll, decltype(add), decltype(add), atl::commutative_tags_t>;
	atl::dual_segment_tree st(atl::commutative_tags, std::size_t(5), ll(1), ll(0), add, add);
	static_assert(std::is_same_v<decltype(st), tree>);
	st.apply(1, 4, 10);
	st.apply(std::size_t(2), 5);
	st.apply(0, 3, -1);
	REQUIRE(st.values() == std::vector<ll>{0, 10, 15, 11, 1});
	tree empty;
	REQUIRE(empty.empty());
	tree from_range(std::vector<ll>{4, 5}, 0); // the untagged constructors serve the mode spelled in the type
	from_range.apply(0, 2, 1);
	REQUIRE(from_range.values() == std::vector<ll>{5, 6});
	// the tagged constructors exist only in the commutative mode
	static_assert(!std::is_constructible_v<atl::dual_segment_tree<ll, ll, decltype(add), decltype(add)>,
										   atl::commutative_tags_t, std::size_t, ll, ll>);
	static_assert(std::is_constructible_v<tree, atl::commutative_tags_t, std::size_t, ll, ll>);
}
