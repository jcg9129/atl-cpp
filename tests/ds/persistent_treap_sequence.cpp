#include <cstdio> // PROBE
#include <atl/ds/persistent_treap_sequence.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <concepts>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <sstream>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint64;
using ll = long long;

TEST_CASE("persistent_treap_sequence: version DAG vs brute force", "[persistent_treap_sequence]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	constexpr ll mod = 1'000'003;

	struct node {
		ll s, w;
	};

	struct aff {
		ll a, b;
	};

	auto op = [](node const &x, node const &y) { return node{(x.s + y.s) % mod, x.w + y.w}; };
	auto mp = [](aff const &f, node const &s) { return node{(f.a * s.s + f.b * s.w) % mod, s.w}; };
	auto cp = [](aff const &f, aff const &g) { return aff{f.a * g.a % mod, (f.a * g.b + f.b) % mod}; };

	std::vector<node> init;
	std::vector<ll> base;
	for (int i = 0; i < 40; ++i) {
		ll const v = ll(rng() % mod);
		base.push_back(v);
		init.push_back({v, 1});
	}
	using tree = decltype(atl::persistent_treap_sequence(init, node{0, 0}, aff{1, 0}, op, mp, cp));
	std::vector<tree> vs{atl::persistent_treap_sequence(init, node{0, 0}, aff{1, 0}, op, mp, cp)};
	std::vector<std::vector<ll>> refs{base};

	for (int step = 0; step < 300; ++step) {
		std::size_t const k = rng() % vs.size();
		auto const &ref = refs[k];
		switch (rng() % 6) {
		case 0: {
			std::size_t const i = rng() % (ref.size() + 1);
			ll const v = ll(rng() % mod);
			vs.push_back(vs[k].insert(i, {v, 1}));
			refs.push_back(ref);
			refs.back().insert(refs.back().begin() + long(i), v);
			break;
		}
		case 1:
			if (!ref.empty()) {
				std::size_t const i = rng() % ref.size();
				vs.push_back(vs[k].erase(i));
				refs.push_back(ref);
				refs.back().erase(refs.back().begin() + long(i));
			}
			break;
		case 2: {
			std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
			aff const f{1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
			vs.push_back(vs[k].apply(l, r, f));
			refs.push_back(ref);
			for (std::size_t j = l; j < r; ++j) refs.back()[j] = (f.a * refs.back()[j] + f.b) % mod;
			break;
		}
		case 3: { // split one version, concat with another — versions interact
			std::size_t const k2 = rng() % vs.size();
			std::size_t const i = rng() % (ref.size() + 1);
			auto const [a, b] = vs[k].split(i);
			vs.push_back(a.concat(vs[k2]));
			std::vector<ll> want(refs[k].begin(), refs[k].begin() + long(i));
			want.insert(want.end(), refs[k2].begin(), refs[k2].end());
			refs.push_back(std::move(want));
			break;
		}
		case 4:
			if (!ref.empty()) {
				std::size_t const i = rng() % ref.size();
				ll const v = ll(rng() % mod);
				vs.push_back(vs[k].set(i, {v, 1}));
				refs.push_back(ref);
				refs.back()[i] = v;
			}
			break;
		default: {
			if (!ref.empty()) {
				std::size_t const i = rng() % ref.size();
				REQUIRE(vs[k].at(i).s == ref[i]);
			}
			std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
			ll want = 0;
			for (std::size_t j = l; j < r; ++j) want = (want + ref[j]) % mod;
			REQUIRE(vs[k].fold(l, r).s == want);
		}
		}
	}
	for (std::size_t k = 0; k < vs.size(); ++k) { // every version intact at the end
		REQUIRE(vs[k].size() == refs[k].size());
		ll want = 0;
		for (ll const x : refs[k]) want = (want + x) % mod;
		REQUIRE(vs[k].fold().s == want);
		// for_each composes pending tags in order; compacted keeps the elements
		std::vector<ll> seen;
		vs[k].for_each([&](node const &x) { seen.push_back(x.s); });
		REQUIRE(seen == refs[k]);
		auto const compact = vs[k].compacted();
		REQUIRE(compact.arena_size() == refs[k].size() + 1);
		REQUIRE(compact.fold().s == want);
		for (std::size_t i = 0; i < refs[k].size(); i += 7) REQUIRE(compact.at(i).s == refs[k][i]);
	}
}

TEST_CASE("persistent_treap_sequence: plain and fold tiers, handle semantics", "[persistent_treap_sequence]") {
	atl::persistent_treap_sequence a(std::vector<int>{1, 2, 3});
	auto b = a.insert(1, 9); // 1 9 2 3
	REQUIRE(a.size() == 3);	 // old version untouched
	REQUIRE(b.at(1) == 9);
	REQUIRE(a.at(1) == 2);
	auto c = b; // O(1) handle copy: same version
	REQUIRE(c.at(1) == 9);

	atl::persistent_treap_sequence s(std::vector<ll>{1, 2, 3, 4}, ll(0), std::plus<>{});
	REQUIRE(s.fold(1, 3) == 5);
	REQUIRE(s.fold() == 10);
	auto s2 = s.push_back(5).push_front(0);
	REQUIRE(s2.fold() == 15);
	REQUIRE(s.fold() == 10);
	auto const [l, r] = s2.split(3);
	REQUIRE(l.size() == 3);
	REQUIRE(r.size() == 3);
	REQUIRE(l.concat(r).fold() == s2.fold());
}

TEST_CASE("persistent_treap_sequence: structural end-splits under self-concat", "[persistent_treap_sequence]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	// end-splits use no size arithmetic, so they stay exact when logical sizes
	// saturate; here we verify them against a reference at verifiable sizes,
	// with heavy self-concat aliasing.
	std::vector<ll> ref{ll(rng() % 100)};
	atl::persistent_treap_sequence cur(std::vector<ll>(ref), ll(0), std::plus<>{});
	for (int step = 0; step < 200; ++step) {
		switch (rng() % 3) {
		case 0:
			if (ref.size() < (1u << 12)) { // self-concat: doubles, shares every node
				cur = cur.concat(cur);
				auto const copy = ref;
				ref.insert(ref.end(), copy.begin(), copy.end());
			}
			break;
		case 1:
			if (ref.size() >= 2) {
				auto const [first, rest] = cur.split_first();
				REQUIRE(first.size() == 1);
				REQUIRE(first.fold() == ref.front());
				REQUIRE(rest.fold() == std::accumulate(ref.begin() + 1, ref.end(), ll(0)));
				cur = rest;
				ref.erase(ref.begin());
			}
			break;
		default:
			if (ref.size() >= 2) {
				auto const [rest, last] = cur.split_last();
				REQUIRE(last.size() == 1);
				REQUIRE(last.fold() == ref.back());
				REQUIRE(rest.fold() == std::accumulate(ref.begin(), ref.end() - 1, ll(0)));
				cur = rest;
				ref.pop_back();
			}
		}
		REQUIRE(cur.fold() == std::accumulate(ref.begin(), ref.end(), ll(0)));
	}
}

TEST_CASE("persistent_treap_sequence: saturated giants allocate O(1) amortized per op", "[persistent_treap_sequence]") {
	// Once sizes saturate, joins switch to the spine-pinning merge and
	// end-splits rebuild their copied path balanced; without either, the
	// adversarial patterns below allocate thousands of nodes per op (the
	// blowup this test pins down). All-ones values keep wrapped uint64 sums
	// checkable at any logical size: a version's sum is its logical size
	// mod 2^64.
	auto const add = [](uint64 a, uint64 b) { return a + b; };
	atl::persistent_treap_sequence base(std::vector<uint64>(8, 1), uint64(0), add);

	auto a = base;
	uint64 sa = 8;
	for (int i = 0; i < 40; ++i) { // 8 * 2^40 > 2^32: saturated
		a = a.concat(a);
		sa *= 2;
	}
	REQUIRE(a.fold() == sa);
	auto b = a;
	uint64 sb = sa;

	std::mt19937_64 rng(1);
	std::size_t const before = a.arena_size();
	int const ops = 2000;
	for (int i = 0; i < ops; ++i) {
		switch (rng() % 6) {
		case 0:
			a = a.concat(b), sa += sb;
			break;
		case 1:
			b = b.concat(a), sb += sa;
			break;
		case 2:
			a = a.concat(a), sa *= 2;
			break;
		case 3:
			b = base.concat(b), sb += 8;
			break;
		case 4: {
			auto const [first, rest] = a.split_first();
			REQUIRE(first.fold() == 1);
			a = rest;
			sa -= 1;
			break;
		}
		default: {
			auto const [rest, last] = b.split_last();
			REQUIRE(last.fold() == 1);
			b = rest;
			sb -= 1;
		}
		}
		REQUIRE(a.fold() == sa);
		REQUIRE(b.fold() == sb);
	}
	REQUIRE(a.arena_size() - before < 100 * std::size_t(ops));
}

TEST_CASE("persistent_treap_sequence: tags tier with equality-comparable F", "[persistent_treap_sequence]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	constexpr ll mod = 1'000'003;

	struct node {
		ll s, w;
	};

	// A comparable tag enables the identity-skip fast paths in push and the
	// accumulator-carrying split/merge/apply — this mirrors the version-DAG
	// test through those branches.
	struct aff {
		ll a, b;

		bool operator==(aff const &) const = default;
	};

	auto op = [](node const &x, node const &y) { return node{(x.s + y.s) % mod, x.w + y.w}; };
	auto mp = [](aff const &f, node const &s) { return node{(f.a * s.s + f.b * s.w) % mod, s.w}; };
	auto cp = [](aff const &f, aff const &g) { return aff{f.a * g.a % mod, (f.a * g.b + f.b) % mod}; };

	std::vector<node> init;
	std::vector<ll> base;
	for (int i = 0; i < 32; ++i) {
		ll const v = ll(rng() % mod);
		base.push_back(v);
		init.push_back({v, 1});
	}
	std::vector vs{atl::persistent_treap_sequence(init, node{0, 0}, aff{1, 0}, op, mp, cp)};
	std::vector<std::vector<ll>> refs{base};

	for (int step = 0; step < 300; ++step) {
		std::size_t const k = rng() % vs.size();
		auto const &ref = refs[k];
		switch (rng() % 5) {
		case 0: {
			std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
			aff const f{1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
			vs.push_back(vs[k].apply(l, r, f));
			refs.push_back(ref);
			for (std::size_t j = l; j < r; ++j) refs.back()[j] = (f.a * refs.back()[j] + f.b) % mod;
			break;
		}
		case 1: { // cross-version range copy: split both, stitch
			std::size_t const k2 = rng() % vs.size();
			std::size_t const n2 = (std::min)(ref.size(), refs[k2].size());
			std::size_t const l = rng() % (n2 + 1), r = l + rng() % (n2 - l + 1);
			auto const [kl, kmr] = vs[k].split(l);
			auto const [sl, smr] = vs[k2].split(l);
			auto const [sm, sr] = smr.split(r - l);
			vs.push_back(kl.concat(sm).concat(kmr.split(r - l).second));
			refs.push_back(ref);
			for (std::size_t j = l; j < r; ++j) refs.back()[j] = refs[k2][j];
			break;
		}
		case 2:
			if (!ref.empty()) {
				auto const [first, rest] = vs[k].split_first();
				REQUIRE(first.at(0).s == ref.front());
				vs.push_back(rest);
				refs.emplace_back(ref.begin() + 1, ref.end());
			}
			break;
		case 3:
			if (!ref.empty()) {
				auto const [rest, last] = vs[k].split_last();
				REQUIRE(last.at(0).s == ref.back());
				vs.push_back(rest);
				refs.emplace_back(ref.begin(), ref.end() - 1);
			}
			break;
		default: {
			if (!ref.empty()) {
				std::size_t const i = rng() % ref.size();
				REQUIRE(vs[k].at(i).s == ref[i]);
			}
			std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
			ll want = 0;
			for (std::size_t j = l; j < r; ++j) want = (want + ref[j]) % mod;
			REQUIRE(vs[k].fold(l, r).s == want);
		}
		}
	}
	for (std::size_t k = 0; k < vs.size(); ++k) {
		REQUIRE(vs[k].size() == refs[k].size());
		ll want = 0;
		for (ll const x : refs[k]) want = (want + x) % mod;
		REQUIRE(vs[k].fold().s == want);
	}
}

TEST_CASE("persistent_treap_sequence: for_each and compacted on the plain tier", "[persistent_treap_sequence]") {
	atl::persistent_treap_sequence<int> v(std::vector{1, 2, 3});
	for (int i = 0; i < 200; ++i) v = v.push_back(i + 4).erase(0).push_front(1); // a long linear history
	std::vector<int> seen;
	v.for_each([&](int x) { seen.push_back(x); });
	std::vector<int> by_index;
	for (std::size_t i = 0; i < v.size(); ++i) by_index.push_back(v.at(i));
	REQUIRE(seen == by_index);
	auto const c = v.compacted();
	REQUIRE(c.arena_size() == v.size() + 1);
	REQUIRE(c.arena_size() < v.arena_size());
	std::vector<int> after;
	c.for_each([&](int x) { after.push_back(x); });
	REQUIRE(after == seen);
	REQUIRE(c.push_back(9).at(c.size()) == 9); // the compacted version is a normal version
	atl::persistent_treap_sequence<int> empty;
	REQUIRE(empty.compacted().empty());
}

TEST_CASE("persistent_treap_sequence: positional ops assert on a saturated size", "[persistent_treap_sequence]") {
	{ // PROBE scope
	atl::persistent_treap_sequence<int> v(std::vector{0, 1, 2, 3, 4, 5, 6, 7});
	for (int i = 0; i < 40; ++i) v = v.concat(v);
	REQUIRE(v.size() == (std::numeric_limits<atl::uint32>::max)());
	CHECK(atl_test::aborts([&] { (void)v.insert(1, 9); }));
	CHECK(atl_test::aborts([&] { (void)v.at(5); }));
	CHECK(atl_test::aborts([&] { (void)v.erase(5); }));
	CHECK(atl_test::aborts([&] { (void)v.set(5, 1); }));
	CHECK(atl_test::aborts([&] { (void)v.split(5); }));
	CHECK(atl_test::aborts([&] { (void)v.compacted(); }));
	REQUIRE(v.split_first().first.at(0) == 0); // structural end-splits stay valid
	std::fprintf(stderr, "PROBE saturated body done\n"); std::fflush(stderr); // PROBE
	} // PROBE scope
	std::fprintf(stderr, "PROBE saturated v destroyed\n"); std::fflush(stderr); // PROBE
}

TEST_CASE("persistent_treap_sequence: rvalue overloads match the lvalue ones and leave other versions intact", "[persistent_treap_sequence]") {
	std::fprintf(stderr, "PROBE rvalue start\n"); std::fflush(stderr); // PROBE
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	constexpr ll mod = 1'000'003;

	struct node {
		ll s, w;
	};

	struct aff {
		ll a, b;
	};

	auto op = [](node const &x, node const &y) { return node{(x.s + y.s) % mod, x.w + y.w}; };
	auto mp = [](aff const &f, node const &s) { return node{(f.a * s.s + f.b * s.w) % mod, s.w}; };
	auto cp = [](aff const &f, aff const &g) { return aff{f.a * g.a % mod, (f.a * g.b + f.b) % mod}; };

	std::vector<node> init;
	std::vector<ll> ref;
	for (int i = 0; i < 30; ++i) {
		ll const v = ll(rng() % mod);
		ref.push_back(v);
		init.push_back({v, 1});
	}
	using tree = decltype(atl::persistent_treap_sequence(init, node{0, 0}, aff{1, 0}, op, mp, cp));
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, node) const &>(&tree::insert); });
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, node) &&>(&tree::insert); });
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, std::size_t) const &>(&tree::erase); });
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, std::size_t) &&>(&tree::erase); });
	static_assert(requires { static_cast<tree (tree::*)(std::size_t) const &>(&tree::erase); });
	static_assert(requires { static_cast<tree (tree::*)(std::size_t) &&>(&tree::erase); });
	static_assert(requires { static_cast<tree (tree::*)(node) const &>(&tree::push_back); });
	static_assert(requires { static_cast<tree (tree::*)(node) &&>(&tree::push_back); });
	static_assert(requires { static_cast<tree (tree::*)(node) const &>(&tree::push_front); });
	static_assert(requires { static_cast<tree (tree::*)(node) &&>(&tree::push_front); });
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, node) const &>(&tree::set); });
	static_assert(requires { static_cast<tree (tree::*)(std::size_t, node) &&>(&tree::set); });
	static_assert(requires { static_cast<tree (tree::*)(tree const &) const &>(&tree::concat); });
	static_assert(requires { static_cast<tree (tree::*)(tree const &) &&>(&tree::concat); });
	static_assert(requires(tree t) { { std::move(t).apply(0, 0, aff{1, 0}) } -> std::same_as<tree>; });
	static_assert(requires(tree t) { { std::move(t).split(0) } -> std::same_as<std::pair<tree, tree>>; });
	static_assert(requires(tree t) { { std::move(t).split_first() } -> std::same_as<std::pair<tree, tree>>; });
	static_assert(requires(tree t) { { std::move(t).split_last() } -> std::same_as<std::pair<tree, tree>>; });

	auto same = [](tree const &t, std::vector<ll> const &r) {
		if (t.size() != r.size()) return false;
		std::vector<ll> seen;
		t.for_each([&](node const &x) { seen.push_back(x.s); });
		return seen == r;
	};
	tree moved(init, node{0, 0}, aff{1, 0}, op, mp, cp);
	tree copied = moved;
	std::vector<std::pair<tree, std::vector<ll>>> kept;
	for (int step = 0; step < 250; ++step) {
		if (rng() % 15 == 0) kept.emplace_back(moved, ref);
		ll const v = ll(rng() % mod);
		std::size_t const sz = ref.size();
		switch (sz == 0 ? rng() % 3 : rng() % 12) {
		case 0: {
			std::size_t const i = rng() % (sz + 1);
			moved = std::move(moved).insert(i, {v, 1});
			copied = copied.insert(i, {v, 1});
			ref.insert(ref.begin() + long(i), v);
			break;
		}
		case 1:
			moved = std::move(moved).push_back({v, 1});
			copied = copied.push_back({v, 1});
			ref.push_back(v);
			break;
		case 2:
			moved = std::move(moved).push_front({v, 1});
			copied = copied.push_front({v, 1});
			ref.insert(ref.begin(), v);
			break;
		case 3: {
			std::size_t const l = rng() % (sz + 1), r = l + rng() % (sz - l + 1);
			moved = std::move(moved).erase(l, r);
			copied = copied.erase(l, r);
			ref.erase(ref.begin() + long(l), ref.begin() + long(r));
			break;
		}
		case 4: {
			std::size_t const i = rng() % sz;
			moved = std::move(moved).erase(i);
			copied = copied.erase(i);
			ref.erase(ref.begin() + long(i));
			break;
		}
		case 5: {
			std::size_t const i = rng() % sz;
			moved = std::move(moved).set(i, {v, 1});
			copied = copied.set(i, {v, 1});
			ref[i] = v;
			break;
		}
		case 6: {
			std::size_t const l = rng() % (sz + 1), r = l + rng() % (sz - l + 1);
			aff const f{1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
			moved = std::move(moved).apply(l, r, f);
			copied = copied.apply(l, r, f);
			for (std::size_t j = l; j < r; ++j) ref[j] = (f.a * ref[j] + f.b) % mod;
			break;
		}
		case 7: { // split then glue back: the contents are unchanged
			auto [a, b] = std::move(moved).split(rng() % (sz + 1));
			moved = std::move(a).concat(b);
			break;
		}
		case 8: {
			auto [a, b] = std::move(moved).split_first();
			moved = std::move(a).concat(b);
			break;
		}
		case 9: {
			auto [a, b] = std::move(moved).split_last();
			moved = std::move(a).concat(b);
			break;
		}
		default: { // concat with a version of the same arena (or with itself)
			if (kept.empty() || ref.size() > 300) break;
			auto const &[other, other_ref] = kept[rng() % kept.size()];
			moved = std::move(moved).concat(other);
			copied = copied.concat(other);
			ref.insert(ref.end(), other_ref.begin(), other_ref.end());
		}
		}
		REQUIRE(same(moved, ref));
		REQUIRE(same(copied, ref));
	}
	for (auto const &[t, r] : kept) REQUIRE(same(t, r));
	tree a = moved;
	auto b = std::move(a).push_back({7, 1});
	a = b; // assignment into a moved-from sequence
	REQUIRE(a.size() == ref.size() + 1);
	auto c = std::move(a).erase(0);
	REQUIRE(c.size() == ref.size());
}

TEST_CASE("persistent_treap_sequence: sized input ranges can have move-only iterators", "[persistent_treap_sequence]") {
	std::fprintf(stderr, "PROBE moveonly start\n"); std::fflush(stderr); // PROBE
	std::istringstream source("1 2 3 4 5");
	auto input = std::views::istream<int>(source);
	static_assert(!std::copy_constructible<decltype(input.begin())>);
	auto values = std::ranges::subrange(input.begin(), std::default_sentinel, std::size_t(5));
	std::fprintf(stderr, "PROBE moveonly before build\n"); std::fflush(stderr); // PROBE
	atl::persistent_treap_sequence<int> tree(std::move(values));
	std::fprintf(stderr, "PROBE moveonly built\n"); std::fflush(stderr); // PROBE
	REQUIRE(tree.size() == 5);
	for (std::size_t i = 0; i < tree.size(); ++i) REQUIRE(tree.at(i) == int(i + 1));
}
