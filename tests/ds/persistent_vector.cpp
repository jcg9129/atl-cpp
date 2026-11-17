#include <atl/ds/persistent_vector.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <iterator>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <vector>
#include "../expect_abort.hpp"
#include "move_probe.hpp"

using atl::int64;
using atl::uint64;

// Reference-checked model: random push_back/pop_back/set, verifying every
// retained version against its snapshot.
TEST_CASE("persistent_vector: full persistence, branch off any version", "[persistent_vector]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);

	std::vector<atl::persistent_vector<int>> versions;
	std::vector<std::vector<int>> reference;
	std::size_t const n0 = rng() % 40;
	std::vector<int> initial(n0);
	for (auto &x : initial) x = int(rng());
	versions.emplace_back(initial);
	reference.push_back(initial);

	for (int step = 0; step < 1500; ++step) {
		std::size_t p = rng() % versions.size();
		auto base = versions[p];
		auto ref = reference[p];
		int op = rng() % 3;
		if (op == 0) {
			int v = int(rng());
			versions.push_back(base.push_back(v));
			ref.push_back(v);
		} else if (op == 1 && !ref.empty()) {
			versions.push_back(base.pop_back());
			ref.pop_back();
		} else if (!ref.empty()) {
			std::size_t i = rng() % ref.size();
			int v = int(rng());
			versions.push_back(base.set(i, v));
			ref[i] = v;
		} else
			continue;
		reference.push_back(ref);
	}
	for (std::size_t k = 0; k < versions.size(); ++k) {
		REQUIRE(versions[k].size() == reference[k].size());
		for (std::size_t i = 0; i < reference[k].size(); ++i) REQUIRE(versions[k][i] == reference[k][i]);
	}
}

TEST_CASE("semi_persistent_vector: linear history reads every past version", "[persistent_vector]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);

	std::size_t const n0 = rng() % 60;
	std::vector<long long> initial(n0);
	for (auto &x : initial) x = (long long)rng();
	std::vector<atl::semi_persistent_vector<long long>> versions;
	std::vector<std::vector<long long>> reference;
	versions.emplace_back(initial);
	reference.push_back(initial);

	auto cur = versions.front();
	auto curref = initial;
	for (int step = 0; step < 2000; ++step) {
		int op = rng() % 3;
		if (op == 0) {
			long long v = (long long)rng();
			cur = cur.push_back(v);
			curref.push_back(v);
		} else if (op == 1 && !curref.empty()) {
			cur = cur.pop_back();
			curref.pop_back();
		} else if (!curref.empty()) {
			std::size_t i = rng() % curref.size();
			long long v = (long long)rng();
			cur = cur.set(i, v);
			curref[i] = v;
		} else
			continue;
		versions.push_back(cur);
		reference.push_back(curref);
	}
	for (std::size_t k = 0; k < versions.size(); ++k) {
		REQUIRE(versions[k].size() == reference[k].size());
		for (std::size_t i = 0; i < reference[k].size(); ++i) REQUIRE(versions[k][i] == reference[k][i]);
	}
}

TEMPLATE_TEST_CASE("growable vectors: bulk build == push_back, across radices", "[persistent_vector]",
				   (atl::persistent_vector<int, 2>), (atl::persistent_vector<int, 4>),
				   (atl::persistent_vector<int, 16>), (atl::persistent_vector<int, 32>)) {
	auto seed = GENERATE(take(2, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = GENERATE(0, 1, 15, 16, 17, 33, 260, 5000);

	std::vector<int> data(n);
	for (auto &x : data) x = int(rng());

	TestType bulk(data); // O(n) build
	TestType incremental;
	for (int x : data) incremental = incremental.push_back(x);

	REQUIRE(bulk.size() == n);
	REQUIRE(incremental.size() == n);
	for (std::size_t i = 0; i < n; ++i) {
		REQUIRE(bulk[i] == data[i]);
		REQUIRE(incremental[i] == data[i]);
	}
	if (n > 0) {
		REQUIRE(bulk.back() == data[n - 1]);
		REQUIRE(bulk.front() == data[0]);
	}
}

TEST_CASE("persistent_vector: concat / slice / insert vs std::vector, branching", "[persistent_vector]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);

	std::vector<atl::persistent_vector<int>> vs;
	std::vector<std::vector<int>> refs;
	for (int i = 0; i < 5; ++i) {
		std::size_t n = rng() % 50;
		std::vector<int> r(n);
		for (auto &x : r) x = int(rng());
		vs.emplace_back(r);
		refs.push_back(r);
	}

	for (int step = 0; step < 1500; ++step) {
		std::size_t p = rng() % vs.size();
		auto base = vs[p];
		auto ref = refs[p];
		switch (rng() % 5) {
		case 0: { // concat with another version
			std::size_t q = rng() % vs.size();
			vs.push_back(base.concat(vs[q]));
			ref.insert(ref.end(), refs[q].begin(), refs[q].end());
			break;
		}
		case 1: { // slice
			std::size_t from = rng() % (ref.size() + 1), to = rng() % (ref.size() + 1);
			if (from > to) std::swap(from, to);
			vs.push_back(base.slice(from, to));
			ref = std::vector<int>(ref.begin() + from, ref.begin() + to);
			break;
		}
		case 2: { // insert
			std::size_t i = rng() % (ref.size() + 1);
			int x = int(rng());
			vs.push_back(base.insert(i, x));
			ref.insert(ref.begin() + i, x);
			break;
		}
		case 3: { // push_back
			int x = int(rng());
			vs.push_back(base.push_back(x));
			ref.push_back(x);
			break;
		}
		default: // pop_back
			if (ref.empty()) continue;
			vs.push_back(base.pop_back());
			ref.pop_back();
			break;
		}
		refs.push_back(ref);
	}

	for (std::size_t k = 0; k < vs.size(); ++k) {
		REQUIRE(vs[k].size() == refs[k].size());
		for (std::size_t i = 0; i < refs[k].size(); ++i) REQUIRE(vs[k][i] == refs[k][i]);
		REQUIRE(std::ranges::equal(vs[k], refs[k])); // iteration over relaxed (concat/slice) tries too
		REQUIRE(vs[k] == atl::persistent_vector<int>(refs[k]));
	}
}

TEST_CASE("persistent_vector: concat across independent arenas", "[persistent_vector]") {
	atl::persistent_vector<int> a(std::vector<int>{1, 2, 3});
	atl::persistent_vector<int> b(std::vector<int>{4, 5}); // different arena
	auto c = a.concat(b);
	REQUIRE(c.size() == 5);
	int expected[] = {1, 2, 3, 4, 5};
	for (std::size_t i = 0; i < 5; ++i) REQUIRE(c[i] == expected[i]);
	REQUIRE(a.size() == 3); // operands untouched
	REQUIRE(b.size() == 2);
}

TEST_CASE("persistent_vector: value copy shares the arena, versions independent", "[persistent_vector]") {
	atl::persistent_vector<int> a(std::vector<int>{1, 2, 3});
	auto b = a.push_back(4);
	auto c = b.set(0, 99);
	REQUIRE(a.size() == 3);
	REQUIRE(b.size() == 4);
	REQUIRE(c.size() == 4);
	REQUIRE(a[0] == 1);
	REQUIRE(b[3] == 4);
	REQUIRE(c[0] == 99);
	REQUIRE(b[0] == 1);
	auto d = c.pop_back().pop_back();
	REQUIRE(d.size() == 2);
	REQUIRE(d.back() == 2);
}

TEMPLATE_TEST_CASE("vector set/push_back move the passed value, never copy it", "[persistent_vector]",
				   atl::persistent_vector<atl_test::move_probe>, atl::semi_persistent_vector<atl_test::move_probe>) {
	using atl_test::move_probe;
	std::vector<move_probe> initial;
	for (int i = 0; i < 100; ++i) initial.emplace_back(i);

	{ // set moves into its slot
		TestType a(initial);
		move_probe::reset();
		auto b = a.set(50, move_probe(move_probe::sentinel));
		REQUIRE(b[50].v == move_probe::sentinel);
		REQUIRE(move_probe::sentinel_copies == 0);
		REQUIRE(move_probe::sentinel_moves >= 1);
	}
	{ // push_back moves into the tail (fresh instance: semi extends only the newest)
		TestType a(initial);
		move_probe::reset();
		auto b = a.push_back(move_probe(move_probe::sentinel));
		REQUIRE(b.back().v == move_probe::sentinel);
		REQUIRE(move_probe::sentinel_copies == 0);
		REQUIRE(move_probe::sentinel_moves >= 1);
	}
	{ // emplace_back constructs in place, never copies the value
		TestType a(initial);
		move_probe::reset();
		auto b = a.emplace_back(move_probe::sentinel);
		REQUIRE(b.back().v == move_probe::sentinel);
		REQUIRE(move_probe::sentinel_copies == 0);
	}
}

TEMPLATE_TEST_CASE("growable vectors: rvalue range build elides the per-element copy", "[persistent_vector]",
				   atl::persistent_vector<atl_test::move_probe>, atl::semi_persistent_vector<atl_test::move_probe>) {
	using atl_test::move_probe;
	// Arena-internal copies (e.g. pulling the tail leaf) happen either way; an
	// rvalue range must eliminate exactly the one-copy-per-element of the input.
	std::vector<move_probe> vals(100, move_probe(move_probe::sentinel));
	move_probe::reset();
	TestType copied(vals);
	auto const lvalue_copies = move_probe::sentinel_copies;

	move_probe::reset();
	TestType moved(std::move(vals));
	REQUIRE(lvalue_copies - move_probe::sentinel_copies == 100);
	REQUIRE(moved[42].v == move_probe::sentinel);
}

TEST_CASE("persistent_vector: deep-tree insert-anywhere (shift regression)", "[persistent_vector]") {
	// Repeated inserts concat-grow the RRB tree past height 8, where the digit
	// arithmetic (radix::bits * height >= 32) once overflowed a 32-bit shift and
	// walked off a node's child array. Large n is required to reach that height.
	constexpr std::size_t n = 1u << 20;
	std::mt19937_64 rng(42);
	std::vector<int64> ref(n);
	for (std::size_t i = 0; i < n; ++i) ref[i] = int64(i);
	atl::persistent_vector<int64> v(ref);
	for (int op = 0; op < 1500; ++op) {
		std::size_t const i = std::size_t(rng()) % (ref.size() + 1);
		int64 const x = int64(10'000'000 + op);
		v = v.insert(i, x);
		ref.insert(ref.begin() + long(i), x);
		REQUIRE(v.size() == ref.size());
		REQUIRE(v[i] == x);
		if (op % 250 == 0)
			for (int probe = 0; probe < 100; ++probe) {
				std::size_t const j = std::size_t(rng()) % ref.size();
				REQUIRE(v[j] == ref[j]);
			}
	}
	for (std::size_t j = 0; j < ref.size(); j += 997) REQUIRE(v[j] == ref[j]);
}

TEST_CASE("persistent_vector: the 32-bit size limit is asserted, not wrapped", "[persistent_vector]") {
	// regression: v = v.concat(v) doubles the size in O(log n) nodes, and the
	// 32nd doubling of a single element wrapped the size to 0 silently
	atl::persistent_vector<int> v(std::vector<int>{1});
	for (int doubling = 0; doubling < 31; ++doubling) v = v.concat(v);
	REQUIRE(v.size() == (std::size_t(1) << 31));
	REQUIRE(v[0] == 1);
	REQUIRE(v[(std::size_t(1) << 31) - 1] == 1);
	REQUIRE(atl_test::aborts([&] { (void)v.concat(v); }));
}

TEST_CASE("persistent_vector: iteration, equality and reserve", "[persistent_vector]") {
	static_assert(std::forward_iterator<atl::persistent_vector<int>::iterator>);
	atl::persistent_vector<int> empty;
	REQUIRE(empty.begin() == empty.end());
	atl::persistent_vector<int, 4> v;
	v.reserve(1000);
	for (int i = 0; i < 1000; ++i) v = v.push_back(i);
	REQUIRE(std::ranges::equal(v, std::views::iota(0, 1000)));
	auto const w = v.set(500, -1);
	REQUIRE_FALSE(v == w);
	REQUIRE(v == v.set(500, -1).set(500, 500)); // same elements, different nodes
	REQUIRE_FALSE(v == v.pop_back());
	REQUIRE(*std::ranges::find(w, -1) == -1);
}

TEMPLATE_TEST_CASE("growable vectors: an rvalue span is copied, leaving the source intact", "[persistent_vector]",
				   atl::persistent_vector<atl_test::move_probe>, atl::semi_persistent_vector<atl_test::move_probe>) {
	using atl_test::move_probe;
	std::vector<move_probe> vals(40, move_probe(move_probe::sentinel));
	TestType from_span{std::span<move_probe>(vals)};
	REQUIRE(std::ranges::all_of(vals, [](move_probe const &p) { return p.v == move_probe::sentinel; }));
	REQUIRE(from_span[39].v == move_probe::sentinel);
}

TEST_CASE("persistent_vector: rvalue overloads match the lvalue ones and leave other versions intact", "[persistent_vector]") {
	using vec = atl::persistent_vector<int>;
	static_assert(requires { static_cast<vec (vec::*)(int) const &>(&vec::push_back); });
	static_assert(requires { static_cast<vec (vec::*)(int) &&>(&vec::push_back); });
	static_assert(requires { static_cast<vec (vec::*)() const &>(&vec::pop_back); });
	static_assert(requires { static_cast<vec (vec::*)() &&>(&vec::pop_back); });
	static_assert(requires { static_cast<vec (vec::*)(std::size_t, int) const &>(&vec::set); });
	static_assert(requires { static_cast<vec (vec::*)(std::size_t, int) &&>(&vec::set); });
	static_assert(requires { static_cast<vec (vec::*)(std::size_t, int) const &>(&vec::insert); });
	static_assert(requires { static_cast<vec (vec::*)(std::size_t, int) &&>(&vec::insert); });
	static_assert(requires { static_cast<vec (vec::*)(std::size_t, std::size_t) const &>(&vec::slice); });
	static_assert(requires { static_cast<vec (vec::*)(std::size_t, std::size_t) &&>(&vec::slice); });
	static_assert(requires { static_cast<vec (vec::*)(vec const &) const &>(&vec::concat); });
	static_assert(requires { static_cast<vec (vec::*)(vec const &) &&>(&vec::concat); });
	static_assert(requires(vec v) { { std::move(v).emplace_back(1) } -> std::same_as<vec>; });
	static_assert(requires(vec v) { { std::move(v).emplace(0, 1) } -> std::same_as<vec>; });

	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto same = [](vec const &v, std::vector<int> const &ref) {
		return v.size() == ref.size() && std::ranges::equal(v, ref);
	};
	vec moved, copied;
	std::vector<int> ref;
	std::vector<std::pair<vec, std::vector<int>>> kept; // share moved's pool; must stay intact
	for (int step = 0; step < 1200; ++step) {
		if (rng() % 40 == 0) kept.emplace_back(moved, ref);
		int v = int(rng());
		switch (ref.empty() ? rng() % 3 * 3 : rng() % 9) {
		case 0:
			moved = std::move(moved).push_back(v);
			copied = copied.push_back(v);
			ref.push_back(v);
			break;
		case 3:
			moved = std::move(moved).emplace_back(v);
			copied = copied.emplace_back(v);
			ref.push_back(v);
			break;
		case 6: {
			std::size_t i = rng() % (ref.size() + 1);
			moved = std::move(moved).insert(i, v);
			copied = copied.insert(i, v);
			ref.insert(ref.begin() + long(i), v);
			break;
		}
		case 1:
			moved = std::move(moved).pop_back();
			copied = copied.pop_back();
			ref.pop_back();
			break;
		case 2: {
			std::size_t i = rng() % ref.size();
			moved = std::move(moved).set(i, v);
			copied = copied.set(i, v);
			ref[i] = v;
			break;
		}
		case 4: {
			std::size_t i = rng() % (ref.size() + 1);
			moved = std::move(moved).emplace(i, v);
			copied = copied.emplace(i, v);
			ref.insert(ref.begin() + long(i), v);
			break;
		}
		case 5: {
			std::size_t from = rng() % (ref.size() + 1), to = from + rng() % (ref.size() - from + 1);
			moved = std::move(moved).slice(from, to);
			copied = copied.slice(from, to);
			ref = std::vector<int>(ref.begin() + long(from), ref.begin() + long(to));
			break;
		}
		default: // concat a kept (shared-pool) version, or ourselves
			if (!kept.empty() && rng() % 2 && ref.size() + kept.back().second.size() < 2000) {
				auto const &[other, other_ref] = kept[rng() % kept.size()];
				moved = std::move(moved).concat(other);
				copied = copied.concat(other);
				ref.insert(ref.end(), other_ref.begin(), other_ref.end());
			} else if (ref.size() < 500) {
				std::vector<int> twice = ref;
				twice.insert(twice.end(), ref.begin(), ref.end());
				moved = std::move(moved).concat(moved);
				copied = copied.concat(copied);
				ref = std::move(twice);
			}
		}
		REQUIRE(same(moved, ref));
		REQUIRE(same(copied, ref));
	}
	for (auto const &[w, r] : kept) REQUIRE(same(w, r));
}

TEST_CASE("persistent_vector: a moved-from vector can be assigned to and destroyed", "[persistent_vector]") {
	atl::persistent_vector<int> a = atl::persistent_vector<int>().push_back(1);
	auto b = std::move(a).push_back(2);
	a = b; // assignment into a moved-from object
	REQUIRE(a.size() == 2);
	auto c = std::move(a).slice(0, 1);
	REQUIRE(c.size() == 1);
	REQUIRE(b.size() == 2);
}

TEST_CASE("semi_persistent_vector: rvalue overloads match the lvalue ones and keep past versions readable", "[persistent_vector]") {
	using vec = atl::semi_persistent_vector<int>;
	static_assert(requires { static_cast<vec (vec::*)(int) const &>(&vec::push_back); });
	static_assert(requires { static_cast<vec (vec::*)(int) &&>(&vec::push_back); });
	static_assert(requires { static_cast<vec (vec::*)() const &>(&vec::pop_back); });
	static_assert(requires { static_cast<vec (vec::*)() &&>(&vec::pop_back); });
	static_assert(requires { static_cast<vec (vec::*)(std::size_t, int) const &>(&vec::set); });
	static_assert(requires { static_cast<vec (vec::*)(std::size_t, int) &&>(&vec::set); });
	static_assert(requires(vec v) { { std::move(v).emplace_back(1) } -> std::same_as<vec>; });

	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto same = [](vec const &v, std::vector<int> const &ref) {
		if (v.size() != ref.size()) return false;
		for (std::size_t i = 0; i < ref.size(); ++i)
			if (v[i] != ref[i]) return false;
		return true;
	};
	vec moved, copied;
	std::vector<int> ref;
	std::vector<std::pair<vec, std::vector<int>>> kept;
	for (int step = 0; step < 1200; ++step) {
		if (rng() % 40 == 0) kept.emplace_back(moved, ref);
		int v = int(rng());
		switch (ref.empty() ? 0 : rng() % 4) {
		case 0:
			moved = std::move(moved).push_back(v);
			copied = copied.push_back(v);
			ref.push_back(v);
			break;
		case 1:
			moved = std::move(moved).emplace_back(v);
			copied = copied.emplace_back(v);
			ref.push_back(v);
			break;
		case 2:
			moved = std::move(moved).pop_back();
			copied = copied.pop_back();
			ref.pop_back();
			break;
		default: {
			std::size_t i = rng() % ref.size();
			moved = std::move(moved).set(i, v);
			copied = copied.set(i, v);
			ref[i] = v;
		}
		}
		REQUIRE(same(moved, ref));
		REQUIRE(same(copied, ref));
	}
	for (auto const &[w, r] : kept) REQUIRE(same(w, r));
	vec a = vec().push_back(1);
	auto b = std::move(a).push_back(2);
	a = b; // assign into a moved-from vector
	REQUIRE(a.size() == 2);
}
