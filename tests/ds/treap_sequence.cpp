#include <atl/ds/splay_sequence.hpp>
#include <atl/ds/treap_sequence.hpp>
#include <atl/experimental/ds/splay_engine_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>
#include "move_probe.hpp"

using atl::uint64;
using ll = long long;

/** @file
 * @brief One vector oracle and shared seeds check all sequence engines and layouts.
 */
namespace {
	struct treap_kind {
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using sequence = atl::treap_sequence<T, Op, F, Mapping, Compose>;
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using reversible = atl::reversible_treap_sequence<T, Op, F, Mapping, Compose>;
	};

	struct splay_kind {
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using sequence = atl::splay_sequence<T, Op, F, Mapping, Compose>;
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using reversible = atl::reversible_splay_sequence<T, Op, F, Mapping, Compose>;
	};

	template<atl::experimental::splay_shape Shape, atl::detail::splay_layout Layout>
	struct splay_variant_kind {
		using engine = atl::experimental::splay_engine_by_strategy<Shape, atl::experimental::splay_path_push::ancestor_stack>;
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using sequence = atl::experimental::splay_sequence_by_strategy<engine, Layout, T, Op, F, Mapping, Compose>;
		template<std::semiregular T, typename Op = void, typename F = void, typename Mapping = void, typename Compose = void>
		using reversible = atl::experimental::splay_sequence_by_strategy<engine, Layout, T, Op, F, Mapping, Compose, true>;
	};

	using shape = atl::experimental::splay_shape;
	using sequence_kinds =
		std::tuple<treap_kind, splay_kind, splay_variant_kind<shape::bottom_up_repush, atl::detail::splay_layout_interleaved>,
				   splay_variant_kind<shape::bottom_up, atl::detail::splay_layout_interleaved>,
				   splay_variant_kind<shape::semi, atl::detail::splay_layout_interleaved>,
				   splay_variant_kind<shape::bottom_up, atl::detail::splay_layout_split>,
				   splay_variant_kind<shape::top_down, atl::detail::splay_layout_columnar>>;
}

// Deduction from the identity must reject narrowing the range's values.
template<typename Value, typename Identity>
concept treap_deducible =
	requires(std::vector<Value> &v, Identity id) { atl::treap_sequence(v, id, std::plus<>{}); };
static_assert(!treap_deducible<long long, int>);
static_assert(treap_deducible<long long, long long>);
static_assert(std::is_same_v<decltype(atl::treap_sequence(std::declval<std::vector<int> &>()))::value_type, int>);

template<typename Value, typename Identity>
concept splay_deducible =
	requires(std::vector<Value> &v, Identity id) { atl::splay_sequence(v, id, std::plus<>{}); };
static_assert(!splay_deducible<long long, int>);
static_assert(splay_deducible<long long, long long>);
static_assert(std::is_same_v<decltype(atl::splay_sequence(std::declval<std::vector<int> &>())), atl::splay_sequence<int>>);
static_assert(std::is_same_v<decltype(atl::splay_sequence(std::size_t(3), 1LL, std::plus<>{})), atl::splay_sequence<long long, std::plus<>>>);
static_assert(!std::is_default_constructible_v<atl::splay_sequence<long long, std::plus<>>>);

TEMPLATE_LIST_TEST_CASE("sequence: rope surgery vs std::vector", "[treap_sequence][splay_sequence]", sequence_kinds) {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<int> ref;
	typename TestType::template sequence<int> t;
	for (int step = 0; step < 1500; ++step) {
		switch (rng() % 7) {
		case 0: {
			std::size_t const i = rng() % (ref.size() + 1);
			int const v = int(rng() % 1000);
			t.insert(i, v);
			ref.insert(ref.begin() + long(i), v);
			break;
		}
		case 1:
			if (!ref.empty()) {
				std::size_t const i = rng() % ref.size();
				t.erase(i);
				ref.erase(ref.begin() + long(i));
			}
			break;
		case 2: {
			int const v = int(rng() % 1000);
			t.push_back(v);
			ref.push_back(v);
			break;
		}
		case 3: {
			int const v = int(rng() % 1000);
			t.push_front(v);
			ref.insert(ref.begin(), v);
			break;
		}
		case 4:
			if (!ref.empty()) {
				std::size_t const i = rng() % ref.size();
				int const v = int(rng() % 1000);
				t.set(i, v);
				ref[i] = v;
			}
			break;
		case 5: {
			std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
			t.erase(l, r);
			ref.erase(ref.begin() + long(l), ref.begin() + long(r));
			break;
		}
		default: {
			std::size_t const i = rng() % (ref.size() + 1);
			auto suffix = t.split(i);
			REQUIRE(t.size() == i);
			REQUIRE(suffix.size() == ref.size() - i);
			t.concat(std::move(suffix));
		}
		}
		REQUIRE(t.size() == ref.size());
		if (!ref.empty()) {
			std::size_t const i = rng() % ref.size();
			REQUIRE(t.at(i) == ref[i]);
		}
	}
	std::vector<int> flat;
	t.for_each([&](int v) { flat.push_back(v); });
	REQUIRE(flat == ref);
}

TEMPLATE_LIST_TEST_CASE("sequence: folds and max_right under surgery", "[treap_sequence][splay_sequence]", sequence_kinds) {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<ll> ref;
	typename TestType::template sequence<ll, std::plus<>> t(std::vector<ll>{}, ll(0), std::plus<>{});
	for (int step = 0; step < 1000; ++step) {
		if (std::size_t const i = rng() % (ref.size() + 1); rng() % 2) {
			ll const v = ll(rng() % 100);
			t.insert(i, v);
			ref.insert(ref.begin() + long(i), v);
		} else if (!ref.empty()) {
			std::size_t const j = rng() % ref.size();
			t.erase(j);
			ref.erase(ref.begin() + long(j));
		}
		std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
		REQUIRE(t.fold(l, r) == std::accumulate(ref.begin() + long(l), ref.begin() + long(r), ll(0)));
		ll const cap = ll(rng() % 2000);
		std::size_t const got = t.max_right(l, [&](ll s) { return s <= cap; });
		std::size_t want = l;
		ll acc = 0;
		while (want < ref.size() && acc + ref[want] <= cap) acc += ref[want], ++want;
		REQUIRE(got == want);
	}
	REQUIRE(t.fold() == std::accumulate(ref.begin(), ref.end(), ll(0)));
}

TEMPLATE_LIST_TEST_CASE("sequence: lazy tags with surgery interleaved", "[treap_sequence][splay_sequence]", sequence_kinds) {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	constexpr ll mod = 1'000'003;

	struct node {
		ll s, w;
	};

	struct aff {
		ll b, c;
	};

	auto const op = [](node const &x, node const &y) { return node{(x.s + y.s) % mod, x.w + y.w}; };
	auto const mp = [](aff const &f, node const &s) { return node{(f.b * s.s + f.c * s.w) % mod, s.w}; };
	auto const cp = [](aff const &f, aff const &g) { return aff{f.b * g.b % mod, (f.b * g.c + f.c) % mod}; };

	std::vector<ll> ref;
	typename TestType::template sequence<node, std::remove_const_t<decltype(op)>, aff, std::remove_const_t<decltype(mp)>,
										 std::remove_const_t<decltype(cp)>>
		t(std::vector<node>{}, node{0, 0}, aff{1, 0}, op, mp, cp);
	for (int step = 0; step < 1000; ++step) {
		switch (rng() % 4) {
		case 0: {
			std::size_t const i = rng() % (ref.size() + 1);
			ll const v = ll(rng() % mod);
			t.insert(i, {v, 1});
			ref.insert(ref.begin() + long(i), v);
			break;
		}
		case 1: {
			std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
			aff const f{ll(rng() % mod), ll(rng() % mod)};
			t.apply(l, r, f);
			for (std::size_t j = l; j < r; ++j) ref[j] = (f.b * ref[j] + f.c) % mod;
			break;
		}
		case 2:
			if (!ref.empty()) {
				std::size_t const j = rng() % ref.size();
				t.erase(j);
				ref.erase(ref.begin() + long(j));
			}
			break;
		default:
			if (!ref.empty()) {
				std::size_t const j = rng() % ref.size();
				REQUIRE(t.at(j).s == ref[j]); // const read under pending tags
			}
		}
		std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
		ll want = 0;
		for (std::size_t j = l; j < r; ++j) want = (want + ref[j]) % mod;
		REQUIRE(t.fold(l, r).s == want);
	}
}

TEMPLATE_LIST_TEST_CASE("sequence: copies are independent, moves are cheap", "[treap_sequence][splay_sequence]", sequence_kinds) {
	using sequence = typename TestType::template sequence<int>;
	sequence t(std::vector<int>{1, 2, 3, 4, 5});
	sequence copy = t;
	copy.set(2, 99);
	REQUIRE(t.at(2) == 3);
	REQUIRE(copy.at(2) == 99);
	copy.concat(t.split(3)); // both share the pool, so concat across them works
	REQUIRE(copy.size() == 7);
	REQUIRE(t.size() == 3);
	REQUIRE(copy.at(5) == 4);

	sequence moved = std::move(copy);
	REQUIRE(moved.size() == 7);
	REQUIRE(moved.at(2) == 99);
}

TEMPLATE_LIST_TEST_CASE("sequence: values are moved in, never copied", "[treap_sequence][splay_sequence]", sequence_kinds) {
	using atl_test::move_probe;
	typename TestType::template sequence<move_probe> t;
	move_probe::reset();
	t.push_back(move_probe(move_probe::sentinel));
	t.insert(0, move_probe(move_probe::sentinel));
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(t.at(0).v == move_probe::sentinel);
}

TEMPLATE_LIST_TEST_CASE("sequence: emplace builds in place with at most one move", "[treap_sequence][splay_sequence]", sequence_kinds) {
	using atl_test::move_probe;
	using probe_sequence = typename TestType::template sequence<atl_test::move_probe>;
	static_assert(std::constructible_from<probe_sequence, std::size_t, atl_test::move_probe>);
	probe_sequence t(4, move_probe(0));
	t.clear(); // recycled nodes: no node-array growth moving stored probes below
	move_probe::reset();
	t.emplace_back(move_probe::sentinel);
	t.emplace_front(move_probe::sentinel);
	t.emplace(1, 5);
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(move_probe::sentinel_moves <= 2); // one per emplace of a sentinel
	REQUIRE(t.size() == 3);
	REQUIRE(t.at(0).v == move_probe::sentinel);
	REQUIRE(t.at(1).v == 5);
	REQUIRE(t.at(2).v == move_probe::sentinel);

	typename TestType::template sequence<std::vector<int>> rows;
	rows.emplace_back(3, 7);
	for (int i = 0; i < 64; ++i) rows.emplace_back(rows.at(0)); // aliases the pool across its growth
	REQUIRE(rows.size() == 65);
	REQUIRE(rows.at(64) == std::vector<int>(3, 7));
}

TEMPLATE_LIST_TEST_CASE("sequence: erase and clear release heap-owning elements", "[treap_sequence][splay_sequence]", sequence_kinds) {
	auto const owner = std::make_shared<int>(42);
	typename TestType::template sequence<std::shared_ptr<int>> t;
	for (int i = 0; i < 8; ++i) t.push_back(owner);
	REQUIRE(owner.use_count() == 9);
	t.erase(2, 5);
	REQUIRE(owner.use_count() == 6);
	t.clear();
	REQUIRE(owner.use_count() == 1);
	t.push_back(owner);
	REQUIRE(t.at(0) == owner);
}

TEMPLATE_LIST_TEST_CASE("sequence: range construction moves only from owning rvalues", "[treap_sequence][splay_sequence]", sequence_kinds) {
	using atl_test::move_probe;
	using probe_sequence = typename TestType::template sequence<atl_test::move_probe>;
	static_assert(std::constructible_from<probe_sequence, std::vector<atl_test::move_probe>>);
	static_assert(std::constructible_from<probe_sequence, std::span<atl_test::move_probe>>);
	std::vector<move_probe> vals(4, move_probe(move_probe::sentinel));
	move_probe::reset();
	probe_sequence from_span{std::span<move_probe>(vals)};
	REQUIRE(move_probe::sentinel_copies == 4);
	auto const view = vals | std::views::transform([](move_probe &p) -> move_probe & { return p; });
	move_probe::reset();
	probe_sequence from_view(std::move(view));
	REQUIRE(move_probe::sentinel_copies == 4);
	REQUIRE(std::ranges::all_of(vals, [](move_probe const &p) { return p.v == move_probe::sentinel; }));

	move_probe::reset();
	probe_sequence moved(std::move(vals));
	REQUIRE(move_probe::sentinel_copies == 0);
	REQUIRE(moved.at(3).v == move_probe::sentinel);
}

TEMPLATE_LIST_TEST_CASE("sequence: edge cases", "[treap_sequence][splay_sequence]", sequence_kinds) {
	typename TestType::template sequence<int> empty;
	REQUIRE(empty.empty());
	auto suffix = empty.split(0);
	REQUIRE(suffix.empty());
	empty.concat(std::move(suffix));

	typename TestType::template sequence<int> one(std::size_t(1), 7);
	REQUIRE(one.at(0) == 7);
	one.erase(0);
	REQUIRE(one.empty());
	one.push_back(9);
	REQUIRE(one.at(0) == 9);

	typename TestType::template sequence<ll, std::plus<>> sums(std::vector<ll>{1, 2, 3}, ll(0), std::plus<>{});
	REQUIRE(sums.fold(1, 1) == 0);
	REQUIRE(sums.fold() == 6);
	sums.clear();
	REQUIRE(sums.empty());
	REQUIRE(sums.fold() == 0);
}

TEMPLATE_LIST_TEST_CASE("reversible sequence: surgery + reverse vs std::vector", "[treap_sequence][splay_sequence]", sequence_kinds) {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::vector<int> ref;
	typename TestType::template reversible<int> t;
	for (int step = 0; step < 1200; ++step) {
		switch (rng() % 5) {
		case 0: {
			std::size_t const i = rng() % (ref.size() + 1);
			int const v = int(rng() % 1000);
			t.insert(i, v);
			ref.insert(ref.begin() + long(i), v);
			break;
		}
		case 1:
			if (!ref.empty()) {
				std::size_t const i = rng() % ref.size();
				t.erase(i);
				ref.erase(ref.begin() + long(i));
			}
			break;
		case 2: {
			std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
			t.reverse(l, r);
			std::ranges::reverse(ref.begin() + long(l), ref.begin() + long(r));
			break;
		}
		case 3:
			if (!ref.empty()) {
				std::size_t const i = rng() % ref.size();
				int const v = int(rng() % 1000);
				t.set(i, v);
				ref[i] = v;
			}
			break;
		default: {
			auto suffix = t.split(rng() % (ref.size() + 1));
			t.concat(std::move(suffix));
		}
		}
		if (!ref.empty()) {
			std::size_t const i = rng() % ref.size();
			REQUIRE(t.at(i) == ref[i]);
		}
	}
	std::vector<int> flat;
	t.for_each([&](int v) { flat.push_back(v); });
	REQUIRE(flat == ref);
}

TEMPLATE_LIST_TEST_CASE("reversible sequence: non-commutative folds stay ordered", "[treap_sequence][splay_sequence]", sequence_kinds) {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	constexpr ll mod = 1'000'003;

	struct aff {
		ll a, b;
	};

	auto op = [](aff const &x, aff const &y) { return aff{y.a * x.a % mod, (y.a * x.b + y.b) % mod}; };
	std::vector<aff> ref;
	typename TestType::template reversible<aff, decltype(op)> t(std::vector<aff>{}, aff{1, 0}, op);
	for (int step = 0; step < 800; ++step) {
		switch (rng() % 3) {
		case 0: {
			std::size_t const i = rng() % (ref.size() + 1);
			aff const v{1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
			t.insert(i, v);
			ref.insert(ref.begin() + long(i), v);
			break;
		}
		case 1: {
			std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
			t.reverse(l, r);
			std::ranges::reverse(ref.begin() + long(l), ref.begin() + long(r));
			break;
		}
		default:
			if (!ref.empty()) {
				std::size_t const j = rng() % ref.size();
				t.erase(j);
				ref.erase(ref.begin() + long(j));
			}
		}
		std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
		aff want{1, 0};
		for (std::size_t j = l; j < r; ++j) want = op(want, ref[j]);
		aff const got = t.fold(l, r);
		REQUIRE(got.a == want.a);
		REQUIRE(got.b == want.b);
	}
}

TEMPLATE_LIST_TEST_CASE("reversible sequence: tags and reversals interleaved", "[treap_sequence][splay_sequence]", sequence_kinds) {
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

	std::vector<ll> ref;
	typename TestType::template reversible<node, decltype(op), aff, decltype(mp), decltype(cp)> t(
		std::vector<node>{}, node{0, 0}, aff{1, 0}, op, mp, cp);
	for (int step = 0; step < 800; ++step) {
		switch (rng() % 5) {
		case 0: {
			std::size_t const i = rng() % (ref.size() + 1);
			ll const v = ll(rng() % mod);
			t.insert(i, {v, 1});
			ref.insert(ref.begin() + long(i), v);
			break;
		}
		case 1: {
			std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
			t.reverse(l, r);
			std::ranges::reverse(ref.begin() + long(l), ref.begin() + long(r));
			break;
		}
		case 2: {
			std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
			aff const f{1 + ll(rng() % (mod - 1)), ll(rng() % mod)};
			t.apply(l, r, f);
			for (std::size_t j = l; j < r; ++j) ref[j] = (f.a * ref[j] + f.b) % mod;
			break;
		}
		case 3:
			if (!ref.empty()) {
				std::size_t const j = rng() % ref.size();
				t.erase(j);
				ref.erase(ref.begin() + long(j));
			}
			break;
		default:
			if (!ref.empty()) {
				std::size_t const j = rng() % ref.size();
				REQUIRE(t.at(j).s == ref[j]);
			}
		}
		std::size_t const l = rng() % (ref.size() + 1), r = l + rng() % (ref.size() - l + 1);
		ll want = 0;
		for (std::size_t j = l; j < r; ++j) want = (want + ref[j]) % mod;
		REQUIRE(t.fold(l, r).s == want);
	}
}
