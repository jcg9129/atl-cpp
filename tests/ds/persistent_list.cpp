#include <atl/ds/persistent_list.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <deque>
#include <iterator>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <vector>
#include "move_probe.hpp"

using atl::uint64;

// Front-oriented model: index 0 is the front. Mirror with a std::deque.
TEST_CASE("persistent_list: full persistence, branch off any version", "[persistent_list]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);

	std::vector<atl::persistent_list<int>> versions;
	std::vector<std::deque<int>> reference;
	versions.emplace_back();
	reference.emplace_back();

	for (int step = 0; step < 1500; ++step) {
		std::size_t p = rng() % versions.size();
		auto base = versions[p];
		auto ref = reference[p];
		int op = rng() % 3;
		if (op == 0) {
			int v = int(rng());
			versions.push_back(base.push_front(v));
			ref.push_front(v);
		} else if (op == 1 && !ref.empty()) {
			versions.push_back(base.pop_front());
			ref.pop_front();
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
		if (!reference[k].empty()) REQUIRE(versions[k].front() == reference[k].front());
		for (std::size_t i = 0; i < reference[k].size(); ++i) REQUIRE(versions[k][i] == reference[k][i]);
	}
}

TEST_CASE("persistent_list: build from a range keeps order (index 0 == first)", "[persistent_list]") {
	std::vector<int> data{10, 20, 30, 40, 50};
	atl::persistent_list list(data); // CTAD
	REQUIRE(list.size() == 5);
	for (std::size_t i = 0; i < data.size(); ++i) REQUIRE(list[i] == data[i]);
	REQUIRE(list.front() == 10);

	auto pushed = list.push_front(5);
	REQUIRE(pushed.front() == 5);
	REQUIRE(pushed[1] == 10);
	REQUIRE(list.front() == 10); // original untouched
}

TEST_CASE("list push_front/set move the passed value, never copy it", "[persistent_list]") {
	using atl_test::move_probe;
	std::vector<move_probe> initial;
	for (int i = 0; i < 100; ++i) initial.emplace_back(i);
	atl::persistent_list<move_probe> a(initial);

	move_probe::reset();
	auto b = a.push_front(move_probe(move_probe::sentinel));
	REQUIRE(b.front().v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(move_probe::sentinel_moves >= 1);

	move_probe::reset();
	auto c = a.set(30, move_probe(move_probe::sentinel));
	REQUIRE(c[30].v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(move_probe::sentinel_moves >= 1);

	move_probe::reset();
	auto d = a.emplace_front(move_probe::sentinel);
	REQUIRE(d.front().v == move_probe::sentinel);
	REQUIRE(move_probe::sentinel_copies == 0);
}

TEST_CASE("persistent_list: rvalue range build moves, never copies", "[persistent_list]") {
	using atl_test::move_probe;
	std::vector<move_probe> vals(100, move_probe(move_probe::sentinel));
	move_probe::reset();
	atl::persistent_list<move_probe> copied(vals);
	REQUIRE(move_probe::sentinel_copies >= 100);

	move_probe::reset();
	atl::persistent_list<move_probe> moved(std::move(vals));
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(moved[42].v == move_probe::sentinel);
}

TEST_CASE("persistent_list: an rvalue span is copied, leaving the source intact", "[persistent_list]") {
	using atl_test::move_probe;
	std::vector<move_probe> vals(10, move_probe(move_probe::sentinel));
	atl::persistent_list<move_probe> from_span{std::span<move_probe>(vals)};
	REQUIRE(std::ranges::all_of(vals, [](move_probe const &p) { return p.v == move_probe::sentinel; }));
	REQUIRE(from_span[9].v == move_probe::sentinel);
}

TEST_CASE("persistent_list: iteration matches operator[] on branched versions", "[persistent_list]") {
	static_assert(std::forward_iterator<atl::persistent_list<int>::iterator>);
	std::mt19937_64 rng(7);
	std::vector<atl::persistent_list<int>> versions{atl::persistent_list<int>()};
	for (int step = 0; step < 400; ++step) {
		auto const &base = versions[rng() % versions.size()];
		int const op = int(rng() % 4);
		if (op < 2 || base.empty())
			versions.push_back(base.push_front(int(rng() % 1000)));
		else if (op == 2)
			versions.push_back(base.pop_front());
		else
			versions.push_back(base.set(rng() % base.size(), -step));
	}
	for (auto const &v : versions) {
		std::vector<int> want;
		for (std::size_t i = 0; i < v.size(); ++i) want.push_back(v[i]);
		std::vector<int> got(v.begin(), v.end());
		REQUIRE(got == want);
		REQUIRE(std::ranges::equal(v, want));
	}
	atl::persistent_list<int> big(std::views::iota(0, 1000));
	big.reserve(10);
	REQUIRE(std::ranges::equal(big, std::views::iota(0, 1000)));
	REQUIRE(*big.push_front(-1).begin() == -1);
}

TEST_CASE("persistent_list: rvalue overloads match the lvalue ones and leave other versions intact", "[persistent_list]") {
	using list = atl::persistent_list<int>;
	static_assert(requires { static_cast<list (list::*)(int) const &>(&list::push_front); });
	static_assert(requires { static_cast<list (list::*)(int) &&>(&list::push_front); });
	static_assert(requires { static_cast<list (list::*)() const &>(&list::pop_front); });
	static_assert(requires { static_cast<list (list::*)() &&>(&list::pop_front); });
	static_assert(requires { static_cast<list (list::*)(std::size_t, int) const &>(&list::set); });
	static_assert(requires { static_cast<list (list::*)(std::size_t, int) &&>(&list::set); });
	static_assert(requires(list l) { { std::move(l).emplace_front(1) } -> std::same_as<list>; });

	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto same = [](list const &l, std::deque<int> const &ref) {
		if (l.size() != ref.size()) return false;
		for (std::size_t i = 0; i < ref.size(); ++i)
			if (l[i] != ref[i]) return false;
		return true;
	};

	list moved, copied;
	std::deque<int> ref;
	std::vector<std::pair<list, std::deque<int>>> kept; // share moved's pool; must stay intact
	for (int step = 0; step < 1500; ++step) {
		if (rng() % 40 == 0) kept.emplace_back(moved, ref);
		int op = int(rng() % 4);
		int v = int(rng());
		if (op == 0) {
			moved = std::move(moved).push_front(v);
			copied = copied.push_front(v);
			ref.push_front(v);
		} else if (op == 1) {
			moved = std::move(moved).emplace_front(v);
			copied = copied.emplace_front(v);
			ref.push_front(v);
		} else if (op == 2 && !ref.empty()) {
			moved = std::move(moved).pop_front();
			copied = copied.pop_front();
			ref.pop_front();
		} else if (!ref.empty()) {
			std::size_t i = rng() % ref.size();
			moved = std::move(moved).set(i, v);
			copied = copied.set(i, v);
			ref[i] = v;
		}
		REQUIRE(same(moved, ref));
		REQUIRE(same(copied, ref));
	}
	for (auto const &[l, r] : kept) REQUIRE(same(l, r));
}

TEST_CASE("persistent_list: a moved-from list can be assigned to and destroyed", "[persistent_list]") {
	atl::persistent_list<int> a = atl::persistent_list<int>().push_front(1);
	auto b = std::move(a).push_front(2);
	a = b; // assignment into a moved-from object
	REQUIRE(a.size() == 2);
	auto c = std::move(a).pop_front();
	REQUIRE(c.size() == 1);
	REQUIRE(b.size() == 2);
	// `c`'s source is destroyed moved-from at scope end, after `b` and `c` still hold the pool
}

TEST_CASE("persistent list iterator arrow ignores element address operators", "[persistent_list]") {
	struct element {
		int value = 0;

		element const *operator&() const { return nullptr; }
	};

	atl::persistent_list<element> empty;
	auto const list = empty.push_front(element{7});
	REQUIRE(list.begin().operator->() != nullptr);
	REQUIRE(list.begin()->value == 7);
}
