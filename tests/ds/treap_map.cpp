#include <atl/ds/detail/heterogeneous_key.hpp>
#include <atl/ds/splay_map.hpp>
#include <atl/ds/treap_map.hpp>
#include <atl/experimental/ds/splay_map_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"
#include "move_probe.hpp"

using atl::uint64;
using ll = long long;

/**
 * @file
 * @brief Every oracle test below runs on treap_map, splay_map and the splay
 *        contenders against one certifying oracle, std::map / std::set, with
 *        the same seeds, so parity between the classes holds by construction.
 */
namespace {
	inline constexpr auto sum_values = [](ll a, ll b) { return a + b; };
	inline constexpr auto concatenate_values = [](std::string const &a, std::string const &b) { return a + b; };

	struct treap_kind {
		template<std::semiregular Key, typename Value = void, typename Compare = std::ranges::less, typename Op = void>
		using map = atl::treap_map<Key, Value, Compare, Op>;
	};

	struct splay_kind {
		template<std::semiregular Key, typename Value = void, typename Compare = std::ranges::less, typename Op = void>
		using map = atl::splay_map<Key, Value, Compare, Op>;
	};

	template<typename Engine, atl::detail::splay_map_layout Layout>
	struct splay_variant_kind {
		template<std::semiregular Key, typename Value = void, typename Compare = std::ranges::less, typename Op = void>
		using map = atl::experimental::splay_map_by_strategy<Engine, Layout, Key, Value, Compare, Op>;
	};

	using top_down = atl::experimental::splay_by_top_down;
	using bottom_up = atl::experimental::splay_by_bottom_up;
	using map_kinds = std::tuple<treap_kind, splay_kind, splay_variant_kind<top_down, atl::splay_layout_interleaved>,
								 splay_variant_kind<bottom_up, atl::splay_layout_split>,
								 splay_variant_kind<bottom_up, atl::splay_layout_interleaved>>;
	using splay_kinds = std::tuple<splay_kind, splay_variant_kind<top_down, atl::splay_layout_interleaved>,
								   splay_variant_kind<bottom_up, atl::splay_layout_split>,
								   splay_variant_kind<bottom_up, atl::splay_layout_interleaved>>;
}

TEMPLATE_LIST_TEST_CASE("ordered set: matches std::set with order statistics", "[treap_map][splay_map]", map_kinds) {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::set<int> ref;
	typename TestType::template map<int> t;
	for (int step = 0; step < 1200; ++step) {
		int const k = int(rng() % 500);
		switch (rng() % 3) {
		case 0:
			REQUIRE(t.insert(k) == ref.insert(k).second);
			break;
		case 1:
			REQUIRE(t.erase(k) == (ref.erase(k) > 0));
			break;
		default:
			REQUIRE(t.contains(k) == (ref.count(k) > 0));
		}
		REQUIRE(t.size() == ref.size());
		int const q = int(rng() % 500);
		REQUIRE(t.rank(q) == std::size_t(std::distance(ref.begin(), ref.lower_bound(q))));
		if (!ref.empty()) {
			std::size_t const i = rng() % ref.size();
			REQUIRE(t.kth(i).key == *std::next(ref.begin(), long(i)));
			auto const it = ref.lower_bound(q);
			auto const got = t.next(q);
			REQUIRE(got.has_value() == (it != ref.end()));
			if (got) REQUIRE(got->key == *it);
			auto const it2 = ref.upper_bound(q);
			auto const got2 = t.prev(q);
			REQUIRE(got2.has_value() == (it2 != ref.begin()));
			if (got2) REQUIRE(got2->key == *std::prev(it2));
		}
	}
}

TEMPLATE_LIST_TEST_CASE("ordered map: values, folds over key ranges", "[treap_map][splay_map]", map_kinds) {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto plus_ = sum_values;
	std::map<int, ll> ref;
	typename TestType::template map<int, ll, std::ranges::less, decltype(plus_)> t(0LL, plus_);
	for (int step = 0; step < 800; ++step) {
		int const k = int(rng() % 300);
		ll const v = ll(rng() % 1000);
		switch (rng() % 4) {
		case 0:
			REQUIRE(t.insert(k, v) == ref.emplace(k, v).second);
			break;
		case 1:
			t.insert_or_assign(k, v);
			ref[k] = v;
			break;
		case 2:
			REQUIRE(t.erase(k) == (ref.erase(k) > 0));
			break;
		default:
			if (ref.count(k)) REQUIRE(t.at(k) == ref[k]);
		}
		int lo = int(rng() % 300), hi = int(rng() % 300);
		if (lo > hi) std::swap(lo, hi);
		ll want = 0;
		for (auto it = ref.lower_bound(lo); it != ref.end() && it->first < hi; ++it) want += it->second;
		REQUIRE(t.fold(lo, hi) == want);
	}
	ll total = 0;
	for (auto const &[k, v] : ref) total += v;
	REQUIRE(t.fold() == total);
}

TEST_CASE("treap_map: merge_union keeps this map's values on duplicates", "[treap_map]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto plus_ = sum_values;
	std::map<int, ll> ra, rb;
	atl::treap_map<int, ll, std::ranges::less, decltype(plus_)> ta(0LL, plus_);
	auto tb = ta.sibling(); // shares the pool, as merge_union requires
	for (int i = 0; i < 400; ++i) {
		int const k = int(rng() % 600);
		ll const v = ll(rng() % 1000);
		if (rng() % 2) {
			ta.insert_or_assign(k, v);
			ra[k] = v;
		} else {
			tb.insert_or_assign(k, v);
			rb[k] = v;
		}
	}
	ta.merge_union(std::move(tb));
	std::map<int, ll> want = rb;
	for (auto const &[k, v] : ra) want[k] = v;
	REQUIRE(ta.size() == want.size());
	auto it = want.begin();
	ta.for_each([&](int k, ll const &v) {
		REQUIRE(it->first == k);
		REQUIRE(it->second == v);
		++it;
	});
	ll total = 0;
	for (auto const &[k, v] : want) total += v;
	REQUIRE(ta.fold() == total);
}

TEMPLATE_LIST_TEST_CASE("ordered map: custom compare and edge cases", "[treap_map][splay_map]", map_kinds) {
	typename TestType::template map<int, void, std::ranges::greater> desc;
	desc.insert(1);
	desc.insert(5);
	desc.insert(3);
	REQUIRE(desc.kth(0).key == 5);
	REQUIRE(desc.kth(2).key == 1);
	REQUIRE(desc.rank(4) == 1); // Under greater, only key 5 precedes 4.

	typename TestType::template map<int> empty;
	REQUIRE(empty.empty());
	REQUIRE_FALSE(empty.contains(0));
	REQUIRE(empty.rank(7) == 0);
	REQUIRE_FALSE(empty.next(0).has_value());
	REQUIRE_FALSE(empty.prev(0).has_value());
	empty.insert(4);
	REQUIRE(empty.erase(4));
	REQUIRE(empty.empty());
}

namespace {
	struct smaller_of {
		long long operator()(long long const a, long long const b) const { return a < b ? a : b; }
	};
}

TEMPLATE_LIST_TEST_CASE("ordered map: a fold-tier map cannot be built without its identity", "[treap_map][splay_map]", map_kinds) {
	// A minimum fold needs an explicit identity: value-initialized zero would corrupt positive minima.
	using min_map = typename TestType::template map<int, long long, std::ranges::less, smaller_of>;
	static_assert(!std::default_initializable<min_map>);
	static_assert(!std::constructible_from<min_map, std::ranges::less>);
	static_assert(std::default_initializable<typename TestType::template map<int, long long>>);

	long long const top = (std::numeric_limits<long long>::max)();
	min_map m(top, smaller_of{});
	REQUIRE(m.fold() == top);
	m.insert(1, 5LL);
	m.insert(2, 9LL);
	m.insert(7, 3LL);
	REQUIRE(m.fold() == 3);
	REQUIRE(m.fold(1, 3) == 5);
	REQUIRE(m.fold(2, 3) == 9);
	REQUIRE(m.fold(3, 7) == top);
}

TEMPLATE_LIST_TEST_CASE("ordered map: value and identity arguments convert like plain parameters", "[treap_map][splay_map]", map_kinds) {
	auto plus_ = sum_values;
	typename TestType::template map<ll, ll, std::ranges::less, decltype(plus_)> t(0, plus_);
	REQUIRE(t.insert(1, 5));
	REQUIRE_FALSE(t.insert_or_assign(1, 7));
	REQUIRE(t.at(1) == 7);
	REQUIRE(t.fold() == 7);
}

TEMPLATE_LIST_TEST_CASE("ordered map: find, update and entry-returning kth/next/prev match std::map", "[treap_map][splay_map]", map_kinds) {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto plus_ = sum_values;
	std::map<int, ll> ref;
	typename TestType::template map<int, ll, std::ranges::less, decltype(plus_)> t(0LL, plus_);
	for (int step = 0; step < 1500; ++step) {
		int const k = int(rng() % 300);
		ll const d = ll(rng() % 1000);
		switch (rng() % 3) {
		case 0:
			REQUIRE(t.update(k, [d](ll v) { return v + d; }) == (ref[k] += d));
			break;
		case 1:
			REQUIRE(t.erase(k) == (ref.erase(k) > 0));
			break;
		default: {
			ll const *got = t.find(k);
			auto const it = ref.find(k);
			REQUIRE((got != nullptr) == (it != ref.end()));
			if (got) REQUIRE(*got == it->second);
		}
		}
		REQUIRE(t.size() == ref.size());
		ll total = 0;
		for (auto const &[key, v] : ref) total += v;
		REQUIRE(t.fold() == total);
		int const q = int(rng() % 300);
		auto const nx = t.next(q);
		auto const it = ref.lower_bound(q);
		REQUIRE(nx.has_value() == (it != ref.end()));
		if (nx) {
			REQUIRE(nx->key == it->first);
			REQUIRE(nx->value == it->second);
		}
		auto const pv = t.prev(q);
		auto const it2 = ref.upper_bound(q);
		REQUIRE(pv.has_value() == (it2 != ref.begin()));
		if (pv) {
			REQUIRE(pv->key == std::prev(it2)->first);
			REQUIRE(pv->value == std::prev(it2)->second);
		}
		if (!ref.empty()) {
			std::size_t const i = rng() % ref.size();
			auto const e = t.kth(i);
			auto const want = std::next(ref.begin(), long(i));
			REQUIRE(e.key == want->first);
			REQUIRE(e.value == want->second);
		}
	}
	typename TestType::template map<int, ll> plain;
	REQUIRE(plain.update(3, [](ll v) { return v + 1; }) == 1);
	REQUIRE(plain.update(3, [](ll v) { return v * 10; }) == 10);
	REQUIRE(plain.find(4) == nullptr);
}

TEMPLATE_LIST_TEST_CASE("ordered map: key-range split, erase and for_each match std::map", "[treap_map][splay_map]", map_kinds) {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::map<int, int> ref;
	typename TestType::template map<int, int> t;
	for (int step = 0; step < 600; ++step) {
		int const k = int(rng() % 1000);
		t.insert(k, k * 3);
		ref.emplace(k, k * 3);
		int lo = int(rng() % 1000), hi = int(rng() % 1000);
		if (rng() % 2 == 0) {
			std::vector<std::pair<int, int>> got, want;
			t.for_each(lo, hi, [&](int key, int v) { got.emplace_back(key, v); });
			for (auto it = ref.lower_bound(lo); it != ref.end() && it->first < hi; ++it) want.emplace_back(*it);
			REQUIRE(got == want);
		}
		if (rng() % 8 == 0) {
			if (lo > hi) std::swap(lo, hi);
			std::size_t const removed = t.erase(lo, hi);
			std::size_t want = 0;
			for (auto it = ref.lower_bound(lo); it != ref.end() && it->first < hi;) {
				it = ref.erase(it);
				++want;
			}
			REQUIRE(removed == want);
		}
		REQUIRE(t.size() == ref.size());
	}
	REQUIRE(t.erase(10, 5) == 0);
	int const cut = 500;
	auto upper = t.split(cut);
	std::size_t const below = std::size_t(std::distance(ref.begin(), ref.lower_bound(cut)));
	REQUIRE(t.size() == below);
	REQUIRE(upper.size() == ref.size() - below);
	if (!upper.empty()) REQUIRE(upper.kth(0).key >= cut);
	if (!t.empty()) REQUIRE(t.kth(t.size() - 1).key < cut);
	if constexpr (requires { t.merge_union(std::move(upper)); }) {
		t.merge_union(std::move(upper));
	} else {
		upper.for_each([&](int key, int v) { t.insert(key, v); });
	}
	REQUIRE(t.size() == ref.size());
	auto it = ref.begin();
	t.for_each([&](int key, int v) {
		REQUIRE(key == it->first);
		REQUIRE(v == it->second);
		++it;
	});
}

TEMPLATE_LIST_TEST_CASE("ordered map: sorted_unique build and reserve", "[treap_map][splay_map]", map_kinds) {
	std::vector<int> keys;
	for (int i = 0; i < 2000; i += 3) keys.push_back(i);
	typename TestType::template map<int> s(atl::sorted_unique, keys);
	REQUIRE(s.size() == keys.size());
	for (std::size_t i = 0; i < keys.size(); i += 37) REQUIRE(s.kth(i).key == keys[i]);
	REQUIRE(s.rank(1000) == std::size_t(std::ranges::lower_bound(keys, 1000) - keys.begin()));
	s.insert(1);
	REQUIRE(s.contains(1));
	REQUIRE(s.size() == keys.size() + 1);

	auto plus_ = sum_values;
	std::vector<std::pair<int, ll>> entries{{1, 10}, {4, 20}, {9, 30}};
	typename TestType::template map<int, ll, std::ranges::less, decltype(plus_)> m(atl::sorted_unique, entries, 0LL, plus_);
	REQUIRE(m.fold() == 60);
	REQUIRE(m.fold(2, 10) == 50);
	REQUIRE(m.at(4) == 20);

	typename TestType::template map<int, void, std::ranges::greater> desc(atl::sorted_unique, std::vector{9, 5, 2});
	REQUIRE(desc.kth(0).key == 9);
	typename TestType::template map<int> empty(atl::sorted_unique, std::vector<int>{});
	REQUIRE(empty.empty());

	typename TestType::template map<int> r;
	r.reserve(100);
	for (int i = 0; i < 100; ++i) r.insert(i);
	REQUIRE(r.size() == 100);

	CHECK(atl_test::aborts([] { typename TestType::template map<int> bad(atl::sorted_unique, std::vector{1, 3, 3}); }));
	CHECK(atl_test::aborts([] { typename TestType::template map<int> bad(atl::sorted_unique, std::vector{4, 2}); }));
}

TEMPLATE_LIST_TEST_CASE("ordered map: sorted_unique build moves out of an owning rvalue range only", "[treap_map][splay_map]", map_kinds) {
	using atl_test::move_probe;
	std::vector<std::pair<int, move_probe>> entries;
	entries.emplace_back(1, move_probe(move_probe::sentinel));
	move_probe::reset();
	typename TestType::template map<int, move_probe> copied(atl::sorted_unique, entries);
	REQUIRE(move_probe::sentinel_copies == 1);
	REQUIRE(entries[0].second.v == move_probe::sentinel);
	move_probe::reset();
	typename TestType::template map<int, move_probe> moved(atl::sorted_unique, std::move(entries));
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(moved.at(1).v == move_probe::sentinel);
}

namespace {
	struct string_less {
		using is_transparent = void;

		bool operator()(std::string_view a, std::string_view b) const { return a < b; }
	};

	struct opaque_less {
		bool operator()(std::string const &a, std::string const &b) const { return a < b; }
	};
}

// mixed signed/unsigned arithmetic compares by the usual conversions (-1 < 5u is false): never a direct probe
static_assert(!atl::detail::heterogeneous_key_for<atl::int32, atl::uint64, std::ranges::less>);
static_assert(!atl::detail::heterogeneous_key_for<atl::uint64, atl::int32, std::ranges::less>);
static_assert(atl::detail::heterogeneous_key_for<std::string_view, std::string, string_less>);
static_assert(atl::detail::ordered_lookup_key<atl::int32, atl::uint64, std::ranges::less>); // converts to Key

TEMPLATE_LIST_TEST_CASE("ordered map: heterogeneous lookup with a transparent comparator", "[treap_map][splay_map]", map_kinds) {
	typename TestType::template map<std::string, int, string_less> m;
	m.insert("apple", 1);
	m.insert("pear", 2);
	m.insert("plum", 3);
	std::string_view const key = "pear";
	REQUIRE(m.contains(key));
	REQUIRE(m.at(key) == 2);
	REQUIRE(*m.find(std::string_view("plum")) == 3);
	REQUIRE(m.find(std::string_view("fig")) == nullptr);
	REQUIRE(m.rank(std::string_view("peach")) == 1);
	REQUIRE(m.next(std::string_view("peach"))->key == "pear");
	REQUIRE(m.prev(std::string_view("pz"))->key == "plum");
	int visited = 0;
	m.for_each(std::string_view("b"), std::string_view("pf"), [&](std::string const &, int) { ++visited; });
	REQUIRE(visited == 1);
	REQUIRE(m.erase(std::string_view("apple")));
	REQUIRE(m.erase(std::string_view("p"), std::string_view("pl")) == 1);
	REQUIRE(m.size() == 1);

	// a non-transparent comparator still takes convertible arguments, built into a Key once
	typename TestType::template map<std::string, void, opaque_less> s;
	s.insert("a");
	REQUIRE(s.contains("a"));
	REQUIRE_FALSE(s.contains("b"));
	// mixed arithmetic arguments convert to Key rather than compare across signedness
	typename TestType::template map<int> ints;
	ints.insert(-1);
	REQUIRE(ints.contains(-1L));
	REQUIRE(ints.rank(0U) == 1);
}

TEMPLATE_LIST_TEST_CASE("ordered map: a non-commutative fold keeps key order through every operation", "[treap_map][splay_map]",
						map_kinds) {
	auto seed = GENERATE(take(3, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	auto concat = concatenate_values;
	std::map<int, std::string> ref;
	typename TestType::template map<int, std::string, std::ranges::less, decltype(concat)> t(std::string{}, concat);
	auto range_fold = [&](int lo, int hi) {
		std::string want;
		for (auto it = ref.lower_bound(lo); it != ref.end() && it->first < hi; ++it) want += it->second;
		return want;
	};
	for (int step = 0; step < 1500; ++step) {
		int const k = int(rng() % 200);
		std::string const v(1, char('a' + rng() % 26));
		int const lo = int(rng() % 210) - 5, hi = int(rng() % 210) - 5;
		switch (rng() % 6) {
		case 0:
			REQUIRE(t.insert(k, v) == ref.emplace(k, v).second);
			break;
		case 1:
			REQUIRE(t.insert_or_assign(k, v) == !ref.contains(k));
			ref[k] = v;
			break;
		case 2:
			REQUIRE(t.update(k, [&v](std::string const &s) { return s + v; }) == (ref[k] += v));
			break;
		case 3:
			REQUIRE(t.erase(k) == (ref.erase(k) > 0));
			break;
		case 4: {
			if (rng() % 4 != 0) break;
			std::size_t const removed = t.erase(lo, hi);
			std::size_t want = 0;
			for (auto it = ref.lower_bound(lo); lo < hi && it != ref.end() && it->first < hi; ++want) it = ref.erase(it);
			REQUIRE(removed == want);
			break;
		}
		default:
			if (rng() % 2)
				(void)t.next(k);
			else
				(void)t.prev(k);
		}
		REQUIRE(t.fold(lo, hi) == range_fold(lo, hi));
		REQUIRE(t.fold() == range_fold((std::numeric_limits<int>::min)(), (std::numeric_limits<int>::max)()));
		REQUIRE(t.size() == ref.size());
	}
}

TEMPLATE_LIST_TEST_CASE("ordered map: next/prev chains and range visits match std::set", "[treap_map][splay_map]", map_kinds) {
	auto seed = GENERATE(take(3, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::set<int> ref;
	typename TestType::template map<int> t;
	for (int i = 0; i < 3000; ++i) {
		int const k = int(rng() % 10'000);
		REQUIRE(t.insert(k) == ref.insert(k).second);
	}
	for (int chain = 0; chain < 20; ++chain) {
		int k = int(rng() % 10'000);
		for (int step = 0; step < 200; ++step) {
			auto const got = t.next(k);
			auto const it = ref.lower_bound(k);
			REQUIRE(got.has_value() == (it != ref.end()));
			if (!got) break;
			REQUIRE(got->key == *it);
			k = got->key + 1;
		}
		k = int(rng() % 10'000);
		for (int step = 0; step < 200; ++step) {
			auto const got = t.prev(k);
			auto const it = ref.upper_bound(k);
			REQUIRE(got.has_value() == (it != ref.begin()));
			if (!got) break;
			REQUIRE(got->key == *std::prev(it));
			k = got->key - 1;
		}
		int const lo = int(rng() % 10'100) - 50, hi = int(rng() % 10'100) - 50;
		std::vector<int> got, want;
		t.for_each(lo, hi, [&](int key) { got.push_back(key); });
		for (auto it = ref.lower_bound(lo); lo < hi && it != ref.end() && *it < hi; ++it) want.push_back(*it);
		REQUIRE(got == want);
		REQUIRE(t.rank(lo) == std::size_t(std::distance(ref.begin(), ref.lower_bound(lo))));
	}
	std::vector<int> all;
	t.for_each([&](int key) { all.push_back(key); });
	REQUIRE(std::ranges::equal(all, ref));
}

TEMPLATE_LIST_TEST_CASE("splay_map: deep paths, copies, siblings and split halves", "[splay_map]", splay_kinds) {
	constexpr int n = 1'000'000;
	typename TestType::template map<int> t;
	for (int i = 0; i < n; ++i) REQUIRE(t.insert(i)); // ascending inserts leave a left path of depth n
	REQUIRE(t.contains(0));							  // one O(n) splay: nothing may recurse
	REQUIRE(t.kth(n / 2).key == n / 2);
	REQUIRE(t.rank(n) == std::size_t(n));
	auto copy = t;
	REQUIRE(copy.erase(5));
	REQUIRE(t.contains(5));
	REQUIRE(copy.size() == std::size_t(n - 1));
	long long sum = 0;
	copy.for_each([&](int k) { sum += k; });
	REQUIRE(sum == ll(n) * (n - 1) / 2 - 5);
	auto upper = t.split(n - 10);
	REQUIRE(upper.size() == 10);
	REQUIRE(t.size() == std::size_t(n - 10));
	REQUIRE(upper.kth(0).key == n - 10);
	auto other = t.sibling();
	REQUIRE(other.empty());
	REQUIRE(other.insert(-1));
	REQUIRE(t.erase(0, n) == std::size_t(n - 10));
	REQUIRE(t.empty());
	t = std::move(upper);
	REQUIRE(t.size() == 10);
	REQUIRE(t.erase(n - 5, n + 5) == 5);
	REQUIRE(t.size() == 5);
}

TEMPLATE_LIST_TEST_CASE("splay_map: preconditions are asserted", "[splay_map]", splay_kinds) {
	using map = typename TestType::template map<int, int>;
	CHECK(atl_test::aborts([] {
		map m;
		m.insert(1, 2);
		(void)m.kth(1);
	}));
	CHECK(atl_test::aborts([] {
		map m;
		m.insert(1, 2);
		(void)m.at(3);
	}));
	CHECK(atl_test::aborts([] {
		map m;
		map moved = std::move(m);
		m.insert(1, 2); // a moved-from map only takes destruction or assignment
	}));
}

TEMPLATE_LIST_TEST_CASE("ordered map: update invokes its callable once on hits and misses", "[treap_map][splay_map]", map_kinds) {
	typename TestType::template map<int, int, std::ranges::less, std::plus<>> values(0, std::plus<>{});
	int calls = 0;
	auto update_value = [&calls](int const &value) mutable { ++calls; return value + 3; };
	for (int i = 0; i < 80; ++i) {
		int const key = i % 13;
		int const expected = (i / 13 + 1) * 3;
		REQUIRE(values.update(key, std::ref(update_value)) == expected);
		REQUIRE(calls == i + 1);
	}
	REQUIRE(values.size() == 13);
	REQUIRE(values.fold() == 240);
	typename TestType::template map<int, int> plain;
	REQUIRE(plain.update(1, update_value) == 3);
	REQUIRE(calls == 81);
	REQUIRE(plain.update(1, update_value) == 6);
	REQUIRE(calls == 82);
}

TEMPLATE_LIST_TEST_CASE("ordered set: singleton folds survive erase and node reuse", "[treap_map][splay_map]", map_kinds) {
	typename TestType::template map<int, void, std::ranges::less, std::plus<>> values(0, std::plus<>{});
	REQUIRE(values.insert(7));
	REQUIRE(values.fold() == 7);
	REQUIRE(values.erase(7));
	REQUIRE(values.fold() == 0);
	REQUIRE(values.insert(11));
	REQUIRE(values.fold() == 11);
	values.insert(3);
	REQUIRE(values.fold() == 14);
	REQUIRE(values.fold(3, 11) == 3);
}

TEMPLATE_LIST_TEST_CASE("ordered map: reserve rejects unrepresentable capacity", "[treap_map][splay_map]", map_kinds) {
	REQUIRE(atl_test::aborts([] {
		typename TestType::template map<int, int> values;
		values.reserve((std::numeric_limits<std::size_t>::max)());
	}));
}
