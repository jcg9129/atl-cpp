#include <atl/ds/persistent_lazy_segment_tree.hpp>
#include <atl/experimental/ds/persistent_lazy_segment_tree_variants.hpp>
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
#include "move_probe.hpp"

using atl::uint64;
using ll = long long;

namespace {
	constexpr ll mod = 1'000'003; // short_values prime: b*a + c stays well inside ll

	struct node {
		ll sum, width;
	};

	struct affine {
		ll b, c;
	};

	auto const sum_op = [](node const &x, node const &y) { return node{(x.sum + y.sum) % mod, x.width + y.width}; };
	auto const affine_map = [](affine const &f, node const &s) {
		return node{(f.b * s.sum + f.c * s.width) % mod, s.width};
	};
	auto const affine_compose = [](affine const &f, affine const &g) {
		return affine{f.b * g.b % mod, (f.b * g.c + f.c) % mod};
	};
}

// Deduction from the element identity must reject narrowing the range's values.
template<typename Value, typename Identity>
concept plst_deducible = requires(std::vector<Value> &v, Identity id) {
	atl::persistent_lazy_segment_tree(v, id, affine{1, 0}, [](Identity a, Identity) { return a; }, [](affine const &, Identity s) { return s; }, affine_compose);
};
static_assert(!plst_deducible<long long, int>);
static_assert(plst_deducible<long long, long long>);

// The commutative range guide carries the same narrowing check.
template<typename Value, typename Identity>
concept plst_commutative_deducible = requires(std::vector<Value> &v, Identity id) {
	atl::persistent_lazy_segment_tree(atl::commutative_tags, v, id, ll{0}, [](Identity a, Identity) { return a; }, [](ll, Identity s) { return s; }, [](ll f, ll g) { return f + g; });
};
static_assert(!plst_commutative_deducible<long long, int>);
static_assert(plst_commutative_deducible<long long, long long>);

TEST_CASE("persistent_lazy_segment_tree: version DAG vs brute force", "[persistent_lazy_segment_tree]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	int const n = 1 + int(rng() % 60);

	std::vector<node> init;
	std::vector<ll> base(n);
	for (auto &x : base) {
		x = ll(rng() % mod);
		init.push_back({x, 1});
	}
	using tree = decltype(atl::persistent_lazy_segment_tree(init, node{0, 0}, affine{1, 0}, sum_op, affine_map,
															affine_compose));
	std::vector<tree> versions{
		atl::persistent_lazy_segment_tree(init, node{0, 0}, affine{1, 0}, sum_op, affine_map, affine_compose)};
	std::vector<std::vector<ll>> ref{base};

	for (int step = 0; step < 300; ++step) {
		std::size_t const k = rng() % versions.size();
		int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
		switch (rng() % 5) {
		case 4: { // branch: point update on version k
			int const i = int(rng() % n);
			ll const d = ll(rng() % mod);
			versions.push_back(versions[k].update(i, [d](node const &old) { return node{(old.sum + d) % mod, old.width}; }));
			ref.push_back(ref[k]);
			ref.back()[i] = (ref.back()[i] + d) % mod;
			break;
		}
		case 0: { // branch: range affine on version k
			affine const f{ll(rng() % mod), ll(rng() % mod)};
			versions.push_back(versions[k].apply(l, r, f));
			ref.push_back(ref[k]);
			for (int j = l; j < r; ++j) ref.back()[j] = (f.b * ref.back()[j] + f.c) % mod;
			break;
		}
		case 1: { // branch: paste [l, r) from version s onto version k
			std::size_t const s = rng() % versions.size();
			versions.push_back(versions[k].copy_range(versions[s], l, r));
			ref.push_back(ref[k]);
			for (int j = l; j < r; ++j) ref.back()[j] = ref[s][j];
			break;
		}
		case 2: { // branch: point set on version k
			int const i = int(rng() % n);
			ll const v = ll(rng() % mod);
			versions.push_back(versions[k].set(i, node{v, 1}));
			ref.push_back(ref[k]);
			ref.back()[i] = v;
			break;
		}
		default: { // query version k (creates nothing)
			ll want = 0;
			for (int j = l; j < r; ++j) want = (want + ref[k][j]) % mod;
			REQUIRE(versions[k].fold(l, r).sum == want);
			int const i = int(rng() % n);
			REQUIRE(versions[k].at(i).sum == ref[k][i]);
		}
		}
	}
	// every version is still intact at the end
	for (std::size_t k = 0; k < versions.size(); ++k) {
		ll want = 0;
		for (ll x : ref[k]) want = (want + x) % mod;
		REQUIRE(versions[k].fold().sum == want);
		REQUIRE(versions[k].fold(0, n).sum == want);
	}
}

TEST_CASE("persistent_lazy_segment_tree: edge cases", "[persistent_lazy_segment_tree]") {
	auto one = atl::persistent_lazy_segment_tree(std::vector<node>{{7, 1}}, node{0, 0}, affine{1, 0}, sum_op,
												 affine_map, affine_compose);
	auto two = one.apply(0, 1, affine{2, 5}); // 7 -> 19
	REQUIRE(one.at(0).sum == 7);			  // the old version is untouched
	REQUIRE(two.at(0).sum == 19);
	REQUIRE(two.fold(0, 0).sum == 0); // empty range -> identity

	// copy_range of the full range is the source; of an empty range, the base
	auto three = one.copy_range(two, 0, 1);
	REQUIRE(three.at(0).sum == 19);
	auto four = two.copy_range(one, 0, 0);
	REQUIRE(four.at(0).sum == 19);

	// n identity elements via the fill constructor
	auto zeros = atl::persistent_lazy_segment_tree(std::size_t(5), node{0, 1}, affine{1, 0}, sum_op, affine_map,
												   affine_compose);
	auto bumped = zeros.apply(1, 4, affine{1, 3});
	REQUIRE(bumped.fold(0, 5).sum == 9);
	REQUIRE(zeros.fold(0, 5).sum == 0);
}

TEST_CASE("persistent_lazy_segment_tree: rvalue range moves its elements in", "[persistent_lazy_segment_tree]") {
	using atl_test::move_probe;
	auto const op = [](move_probe const &a, move_probe const &b) { return move_probe(a.v + b.v); };
	auto const map = [](int f, move_probe const &s) { return move_probe(s.v + f); };
	auto const compose = [](int f, int g) { return f + g; };

	std::vector<move_probe> vals(8, move_probe(move_probe::sentinel));
	move_probe::reset();
	atl::persistent_lazy_segment_tree copied(vals, move_probe(0), 0, op, map, compose);
	REQUIRE(move_probe::sentinel_copies >= 8); // lvalue range: at least one copy per element

	move_probe::reset();
	atl::persistent_lazy_segment_tree moved(std::move(vals), move_probe(0), 0, op, map, compose);
	REQUIRE(move_probe::sentinel_copies == 0); // rvalue range: moved all the way into the leaves
	REQUIRE(moved.at(3).v == move_probe::sentinel);
}

TEST_CASE("persistent_lazy_segment_tree: default construction, defaulted callables, arena_size",
		  "[persistent_lazy_segment_tree]") {
	using tree = atl::persistent_lazy_segment_tree<ll, ll, std::plus<>, std::plus<>, std::plus<>>; // point operations only here (plus is not a width-aware action)
	tree t;
	REQUIRE(t.empty());
	REQUIRE(t.fold() == 0);
	t = tree(std::vector<ll>{1, 2, 3}, 0, 0); // defaulted Op/Mapping/Compose
	REQUIRE(t.fold() == 6);
	std::size_t const before = t.arena_size();
	auto u = t.update(1, [](ll const &x) { return x * 10; });
	REQUIRE(u.at(1) == 20);
	REQUIRE(t.at(1) == 2);
	REQUIRE(t.arena_size() > before); // the arena is shared: the new path shows in both versions
	REQUIRE(u.arena_size() == t.arena_size());
}

TEST_CASE("persistent_lazy_segment_tree: borrowed or view ranges are copied", "[persistent_lazy_segment_tree]") {
	using atl_test::move_probe;
	auto const op = [](move_probe const &a, move_probe const &b) { return move_probe(a.v + b.v); };
	auto const map = [](int f, move_probe const &s) { return move_probe(s.v + f); };
	auto const compose = [](int f, int g) { return f + g; };
	std::vector<move_probe> vals(8, move_probe(move_probe::sentinel));
	auto const intact = [&] {
		return std::ranges::all_of(vals, [](move_probe const &p) { return p.v == move_probe::sentinel; });
	};
	atl::persistent_lazy_segment_tree spanned{std::span(vals), move_probe(0), 0, op, map, compose};
	REQUIRE(intact());
	atl::persistent_lazy_segment_tree strided(vals | std::views::stride(1), move_probe(0), 0, op, map, compose);
	REQUIRE(intact());
	REQUIRE(strided.at(2).v == move_probe::sentinel);
}

TEST_CASE("persistent_lazy_segment_tree: rvalue overloads match the lvalue ones and leave other versions intact", "[persistent_lazy_segment_tree]") {
	std::vector<node> init;
	using tree = decltype(atl::persistent_lazy_segment_tree(init, node{0, 0}, affine{1, 0}, sum_op, affine_map,
															affine_compose));
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, std::size_t, affine const &) const &>(&tree::apply); });
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, std::size_t, affine const &) &&>(&tree::apply); });
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, node) const &>(&tree::set); });
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, node) &&>(&tree::set); });
	static_assert(requires { static_cast<tree (tree::*)(tree const &, std::size_t, std::size_t) const &>(&tree::copy_range); });
	static_assert(requires { static_cast<tree (tree::*)(tree const &, std::size_t, std::size_t) &&>(&tree::copy_range); });
	static_assert(requires(tree t) { { std::move(t).update(0, [](node const &x) { return x; }) } -> std::same_as<tree>; });

	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 60;
	std::vector<ll> ref(n);
	for (auto &x : ref) {
		x = ll(rng() % mod);
		init.push_back({x, 1});
	}
	auto same = [n](tree const &t, std::vector<ll> const &r) {
		for (std::size_t i = 0; i < n; ++i)
			if (t.at(i).sum != r[i]) return false;
		return true;
	};
	tree moved(init, node{0, 0}, affine{1, 0}, sum_op, affine_map, affine_compose);
	tree copied = moved;
	std::vector<std::pair<tree, std::vector<ll>>> kept;
	for (int step = 0; step < 400; ++step) {
		if (rng() % 20 == 0) kept.emplace_back(moved, ref);
		std::size_t const l = rng() % (n + 1), r = l + rng() % (n - l + 1), i = rng() % n;
		switch (rng() % 4) {
		case 0: {
			affine const f{ll(rng() % mod), ll(rng() % mod)};
			moved = std::move(moved).apply(l, r, f);
			copied = copied.apply(l, r, f);
			for (std::size_t j = l; j < r; ++j) ref[j] = (f.b * ref[j] + f.c) % mod;
			break;
		}
		case 1: {
			ll const v = ll(rng() % mod);
			moved = std::move(moved).set(i, node{v, 1});
			copied = copied.set(i, node{v, 1});
			ref[i] = v;
			break;
		}
		case 2: {
			ll const d = ll(rng() % mod);
			auto const bump = [d](node const &old) { return node{(old.sum + d) % mod, old.width}; };
			moved = std::move(moved).update(i, bump);
			copied = copied.update(i, bump);
			ref[i] = (ref[i] + d) % mod;
			break;
		}
		default: {
			auto const &[src, src_ref] = kept.empty() ? std::pair{moved, ref} : kept[rng() % kept.size()];
			moved = std::move(moved).copy_range(src, l, r);
			copied = copied.copy_range(src, l, r);
			for (std::size_t j = l; j < r; ++j) ref[j] = src_ref[j];
		}
		}
		REQUIRE(same(moved, ref));
		REQUIRE(same(copied, ref));
	}
	for (auto const &[t, r] : kept) REQUIRE(same(t, r));
	moved = std::move(moved).copy_range(moved, 0, n); // self-source falls back to the copying path
	REQUIRE(same(moved, ref));
	tree a = moved;
	auto b = std::move(a).set(0, node{5, 1});
	a = b; // assignment into a moved-from tree
	REQUIRE(a.at(0).sum == 5);
	auto c = std::move(a).apply(0, 1, affine{1, 1});
	REQUIRE(c.at(0).sum == 6);
}

namespace {
	auto const add_map = [](ll f, node const &s) { return node{(s.sum + f * s.width) % mod, s.width}; };
	auto const add_compose = [](ll f, ll g) { return (f + g) % mod; };
}

TEST_CASE("persistent_lazy_segment_tree: commutative mode, version DAG vs brute force (add tags)",
		  "[persistent_lazy_segment_tree]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (int t = 0; t < 1000; ++t) { // the mode's precondition, on the tags drawn below
		ll const f = ll(rng() % mod), g = ll(rng() % mod);
		REQUIRE(add_compose(f, g) == add_compose(g, f));
	}
	int const n = 1 + int(rng() % 60);
	std::vector<node> init;
	std::vector<ll> base(n);
	for (auto &x : base) {
		x = ll(rng() % mod);
		init.push_back({x, 1});
	}
	using tree = atl::persistent_lazy_segment_tree<node, ll, std::decay_t<decltype(sum_op)>, std::decay_t<decltype(add_map)>,
												   std::decay_t<decltype(add_compose)>, atl::commutative_tags_t>;
	tree const origin(atl::commutative_tags, init, node{0, 0}, ll(0), sum_op, add_map, add_compose);
	static_assert(std::is_same_v<decltype(atl::persistent_lazy_segment_tree(atl::commutative_tags, init, node{0, 0},
																			ll(0), sum_op, add_map, add_compose)),
								 tree>);
	std::vector<tree> versions{origin};
	std::vector<std::vector<ll>> ref{base};
	for (int step = 0; step < 400; ++step) {
		std::size_t const k = rng() % versions.size();
		int const l = int(rng() % (n + 1)), r = l + int(rng() % (n - l + 1));
		switch (rng() % 6) {
		case 0:
		case 1: { // branch: range add on version k (the permanent-tag path)
			ll const f = ll(rng() % mod);
			versions.push_back(versions[k].apply(l, r, f));
			ref.push_back(ref[k]);
			for (int j = l; j < r; ++j) ref.back()[j] = (ref.back()[j] + f) % mod;
			break;
		}
		case 2: { // branch: paste [l, r) from version s (settles tags over permanent ones)
			std::size_t const s = rng() % versions.size();
			versions.push_back(versions[k].copy_range(versions[s], l, r));
			ref.push_back(ref[k]);
			for (int j = l; j < r; ++j) ref.back()[j] = ref[s][j];
			break;
		}
		case 3: { // branch: point set, then update, on version k
			int const i = int(rng() % n);
			ll const v = ll(rng() % mod), d = ll(rng() % mod);
			versions.push_back(versions[k].set(i, node{v, 1}).update(i, [d](node const &old) {
				return node{(old.sum + d) % mod, old.width};
			}));
			ref.push_back(ref[k]);
			ref.back()[i] = (v + d) % mod;
			break;
		}
		default: {
			ll want = 0;
			for (int j = l; j < r; ++j) want = (want + ref[k][j]) % mod;
			REQUIRE(versions[k].fold(l, r).sum == want);
			int const i = int(rng() % n);
			REQUIRE(versions[k].at(i).sum == ref[k][i]);
		}
		}
	}
	for (std::size_t k = 0; k < versions.size(); ++k) {
		ll want = 0;
		for (ll x : ref[k]) want = (want + x) % mod;
		REQUIRE(versions[k].fold().sum == want);
		for (int i = 0; i < n; ++i) REQUIRE(versions[k].at(i).sum == ref[k][i]);
	}
}

TEST_CASE("persistent_lazy_segment_tree: commutative mode, chmin and chmax over max, and the node count",
		  "[persistent_lazy_segment_tree]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto const max_op = [](ll a, ll b) { return (std::max)(a, b); };
	auto const chmin = [](ll f, ll s) { return (std::min)(f, s); };
	auto const chmax = [](ll f, ll s) { return (std::max)(f, s); };
	for (int t = 0; t < 1000; ++t) {
		ll const f = ll(rng() % 1000), g = ll(rng() % 1000);
		REQUIRE(chmin(f, g) == chmin(g, f));
		REQUIRE(chmax(f, g) == chmax(g, f));
	}
	std::size_t const n = 1 + rng() % 80;
	std::vector<ll> ref(n);
	for (auto &x : ref) x = ll(rng() % 1000);
	constexpr ll low = (std::numeric_limits<ll>::min)(), high = (std::numeric_limits<ll>::max)();
	atl::persistent_lazy_segment_tree mins(atl::commutative_tags, ref, low, high, max_op, chmin, chmin);
	atl::persistent_lazy_segment_tree maxs(atl::commutative_tags, ref, low, low, max_op, chmax, chmax);
	atl::persistent_lazy_segment_tree plain(ref, low, high, max_op, chmin, chmin); // the default mode, for the count
	std::vector<ll> ref_max = ref;
	for (int step = 0; step < 400; ++step) {
		std::size_t const l = rng() % (n + 1), r = l + rng() % (n - l + 1);
		ll const f = ll(rng() % 1000);
		std::size_t const before = mins.arena_size(), plain_before = plain.arena_size();
		mins = std::move(mins).apply(l, r, f);
		plain = std::move(plain).apply(l, r, f);
		// one new node per visited node, never more than the settling path copy
		REQUIRE(mins.arena_size() - before <= plain.arena_size() - plain_before);
		maxs = std::move(maxs).apply(l, r, f);
		for (std::size_t j = l; j < r; ++j) ref[j] = (std::min)(ref[j], f), ref_max[j] = (std::max)(ref_max[j], f);
		std::size_t const ql = rng() % (n + 1), qr = ql + rng() % (n - ql + 1);
		ll want_min = low, want_max = low;
		for (std::size_t j = ql; j < qr; ++j) want_min = (std::max)(want_min, ref[j]), want_max = (std::max)(want_max, ref_max[j]);
		REQUIRE(mins.fold(ql, qr) == want_min);
		REQUIRE(maxs.fold(ql, qr) == want_max);
		REQUIRE(plain.fold(ql, qr) == want_min);
	}
	for (std::size_t i = 0; i < n; ++i) {
		REQUIRE(mins.at(i) == ref[i]);
		REQUIRE(maxs.at(i) == ref_max[i]);
	}
}

namespace {
	/** @brief A (sum, width) element counting its copies and moves, so an arena reallocation shows as a spike. */
	struct counted {
		static inline std::size_t transfers = 0;
		ll sum = 0, width = 0;

		counted() = default;

		counted(ll s, ll w) : sum(s), width(w) {}

		counted(counted const &o) : sum(o.sum), width(o.width) { ++transfers; }

		counted(counted &&o) noexcept : sum(o.sum), width(o.width) { ++transfers; }

		counted &operator=(counted const &o) {
			sum = o.sum, width = o.width;
			++transfers;
			return *this;
		}

		counted &operator=(counted &&o) noexcept {
			sum = o.sum, width = o.width;
			++transfers;
			return *this;
		}
	};
}

TEST_CASE("persistent_lazy_segment_tree: the commutative reserve covers an apply-only workload",
		  "[persistent_lazy_segment_tree]") {
	auto const op = [](counted const &x, counted const &y) { return counted(x.sum + y.sum, x.width + y.width); };
	auto const map = [](ll f, counted const &s) { return counted(s.sum + f * s.width, s.width); };
	auto const add = [](ll f, ll g) { return f + g; };
	std::size_t const n = 4096, ops = 3000;
	std::mt19937_64 rng(7);
	std::vector<std::pair<std::size_t, std::size_t>> ranges(ops);
	for (auto &[l, r] : ranges) {
		l = rng() % n, r = rng() % n;
		if (l > r) std::swap(l, r);
		++r;
	}
	// largest per-apply copy/move count, against the arena size before that apply: a
	// reallocation transfers every existing node, far above what one apply moves
	auto const worst = [&](bool const is_reserved) {
		atl::persistent_lazy_segment_tree t(atl::commutative_tags, std::size_t(n), counted(1, 1), ll(0), op, map, add);
		if (is_reserved) t.reserve(ops);
		bool is_spike = false;
		for (auto const &[l, r] : ranges) {
			std::size_t const before = t.arena_size();
			counted::transfers = 0;
			t = std::move(t).apply(l, r, ll(rng() % 100));
			is_spike |= counted::transfers >= before / 2;
		}
		return is_spike;
	};
	REQUIRE(worst(true) == false);
	REQUIRE(worst(false) == true); // control: unreserved, the arena does reallocate (the probe sees it)
}

TEST_CASE("persistent_lazy_segment_tree: the commutative reserve covers set, update and copy_range too",
		  "[persistent_lazy_segment_tree]") {
	auto const op = [](counted const &x, counted const &y) { return counted(x.sum + y.sum, x.width + y.width); };
	auto const map = [](ll f, counted const &s) { return counted(s.sum + f * s.width, s.width); };
	auto const add = [](ll f, ll g) { return f + g; };
	std::size_t const n = 4096, ops = 3000;
	std::mt19937_64 rng(11);
	atl::persistent_lazy_segment_tree t(atl::commutative_tags, std::size_t(n), counted(1, 1), ll(0), op, map, add);
	auto other = t.apply(0, n, ll(5));
	t.reserve(ops);
	bool is_spike = false;
	for (std::size_t i = 0; i < ops; ++i) {
		std::size_t l = rng() % n, r = rng() % n;
		if (l > r) std::swap(l, r);
		++r;
		std::size_t const before = t.arena_size();
		counted::transfers = 0;
		switch (i % 4) {
		case 0:
			t = std::move(t).apply(l, r, ll(rng() % 100));
			break;
		case 1:
			t = std::move(t).set(l, counted(3, 1));
			break;
		case 2:
			t = std::move(t).update(l, [](counted const &x) { return counted(x.sum + 1, x.width); });
			break;
		default:
			t = std::move(t).copy_range(other, l, r);
			break;
		}
		is_spike |= counted::transfers >= before / 2;
	}
	REQUIRE(is_spike == false);
}

TEST_CASE("persistent_lazy_segment_tree: the default reserve covers the worst-case apply",
		  "[persistent_lazy_segment_tree]") {
	auto const op = [](counted const &x, counted const &y) { return counted(x.sum + y.sum, x.width + y.width); };
	auto const map = [](ll f, counted const &s) { return counted(s.sum + f * s.width, s.width); };
	auto const add = [](ll f, ll g) { return f + g; };
	std::size_t const n = 4096, levels = 12, ops = 3000;
	// [1, n - 1) on a power-of-two n keeps both boundary paths partial down to the
	// leaves: every level settles 4 child copies and rebuilds or retags 4 of them
	auto const worst = [&](bool const is_reserved) {
		atl::persistent_lazy_segment_tree t(std::size_t(n), counted(1, 1), ll(0), op, map, add);
		auto const other = t.apply(0, n, ll(5));
		if (is_reserved) t.reserve(ops);
		std::size_t most_nodes = 0;
		bool is_spike = false;
		for (std::size_t i = 0; i < ops; ++i) {
			std::size_t const before = t.arena_size();
			counted::transfers = 0;
			if (i % 8 == 7)
				t = std::move(t).copy_range(other, 1, n - 1);
			else
				t = std::move(t).apply(1, n - 1, ll(1));
			most_nodes = (std::max)(most_nodes, t.arena_size() - before);
			is_spike |= counted::transfers >= before / 2;
		}
		return std::pair{most_nodes, is_spike};
	};
	auto const [most_nodes, is_spike] = worst(true);
	REQUIRE(most_nodes > 6 * (levels + 1)); // the old 6-per-level bound was too low
	REQUIRE(most_nodes <= 8 * (levels + 1));
	REQUIRE(is_spike == false);
	REQUIRE(worst(false).second == true); // control: unreserved, the probe sees a reallocation
}

// Partial-copy contender: the variant (lean and whole-node partial copies, default and commutative
// mode) against the library class over random version DAGs; every version stays queryable and
// the arenas must stay the same size (the contender changes the copy, not the node count).
template<typename Library, typename Variant, typename MakeTree, typename MakeTag>
	requires std::invocable<MakeTag &, std::mt19937_64 &> && requires(MakeTree make, std::vector<node> const &init, Library const &a, Variant const &b, std::invoke_result_t<MakeTag &, std::mt19937_64 &> tag) {
		{ make.template operator()<Library>(init) } -> std::same_as<Library>;
		{ make.template operator()<Variant>(init) } -> std::same_as<Variant>;
		{ a.apply(0, 1, tag) } -> std::same_as<Library>;
		{ b.apply(0, 1, tag) } -> std::same_as<Variant>;
		a.fold(0, 1);
		b.fold(0, 1);
		a.set(0, node{});
		b.set(0, node{});
	}
static void variant_vs_library(std::mt19937_64 &rng, MakeTree make_tree, MakeTag make_tag) {
	for (int n : {1, 2, 3, 5, 8, 17, 1 + int(rng() % 80)}) {
		std::vector<node> init(static_cast<std::size_t>(n));
		for (auto &x : init) x = node{ll(rng() % 100), 1};
		std::vector<Library> ship{make_tree.template operator()<Library>(init)};
		std::vector<Variant> var{make_tree.template operator()<Variant>(init)};
		for (int step = 0; step < 150; ++step) {
			std::size_t const base = rng() % ship.size();
			std::size_t const l = rng() % (std::size_t(n) + 1), r = l + rng() % (std::size_t(n) - l + 1);
			switch (rng() % 3) {
			case 0: {
				auto const f = make_tag(rng);
				ship.push_back(ship[base].apply(l, r, f));
				var.push_back(var[base].apply(l, r, f));
				break;
			}
			case 1: {
				node const x{ll(rng() % 100), 1};
				std::size_t const i = rng() % std::size_t(n);
				ship.push_back(ship[base].set(i, x));
				var.push_back(var[base].set(i, x));
				break;
			}
			default: {
				auto const a = ship[base].fold(l, r), b = var[base].fold(l, r);
				REQUIRE(a.sum == b.sum);
				REQUIRE(a.width == b.width);
			}
			}
		}
		REQUIRE(ship.back().arena_size() == var.back().arena_size());
		for (std::size_t v = 0; v < ship.size(); ++v)
			for (std::size_t i = 0; i < std::size_t(n); ++i)
				REQUIRE(ship[v].fold(i, i + 1).sum == var[v].fold(i, i + 1).sum);
	}
}

TEST_CASE("persistent_lazy_segment_tree_by_partial_copy: lean and whole-node copies match the library class",
		  "[persistent_lazy_segment_tree][variant]") {
	auto seed = GENERATE(take(20, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	namespace ex = atl::experimental;
	using op_t = decltype(sum_op);
	using map_t = decltype(affine_map);
	using comp_t = decltype(affine_compose);
	auto const make_affine = []<typename Tree>(std::vector<node> const &init)
		requires std::constructible_from<Tree, std::vector<node> const &, node, affine, decltype(sum_op), decltype(affine_map), decltype(affine_compose)>
	{
		return Tree(init, node{0, 0}, affine{1, 0}, sum_op, affine_map, affine_compose);
	};
	auto const affine_tag = [](std::mt19937_64 &g) { return affine{ll(g() % 5), ll(g() % 50)}; };
	variant_vs_library<atl::persistent_lazy_segment_tree<node, affine, op_t, map_t, comp_t>,
					   ex::persistent_lazy_segment_tree_by_partial_copy<node, affine, op_t, map_t, comp_t, void, true>>(
		rng, make_affine, affine_tag);
	variant_vs_library<atl::persistent_lazy_segment_tree<node, affine, op_t, map_t, comp_t>,
					   ex::persistent_lazy_segment_tree_by_partial_copy<node, affine, op_t, map_t, comp_t, void, false>>(
		rng, make_affine, affine_tag);

	auto const add_map = [](ll f, node const &s) { return node{(s.sum + f * s.width) % mod, s.width}; };
	auto const add = [](ll f, ll g) { return f + g; };
	using add_map_t = decltype(add_map);
	using add_t = decltype(add);
	auto const make_add = [&]<typename Tree>(std::vector<node> const &init) {
		return Tree(init, node{0, 0}, ll(0), sum_op, add_map, add);
	};
	auto const add_tag = [](std::mt19937_64 &g) { return ll(g() % 100); };
	variant_vs_library<atl::persistent_lazy_segment_tree<node, ll, op_t, add_map_t, add_t, atl::commutative_tags_t>,
					   ex::persistent_lazy_segment_tree_by_partial_copy<node, ll, op_t, add_map_t, add_t, atl::commutative_tags_t, true>>(
		rng, make_add, add_tag);
	variant_vs_library<atl::persistent_lazy_segment_tree<node, ll, op_t, add_map_t, add_t, atl::commutative_tags_t>,
					   ex::persistent_lazy_segment_tree_by_partial_copy<node, ll, op_t, add_map_t, add_t, atl::commutative_tags_t, false>>(
		rng, make_add, add_tag);
}
