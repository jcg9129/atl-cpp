#include <atl/ds/splay_engine.hpp>
#include <atl/ds/splay_sequence.hpp>
#include <atl/experimental/ds/splay_engine_variants.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <bit>
#include <catch2/catch.hpp>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;
using ll = long long;

namespace {
	constexpr ll mod = 1'000'003;

	/** @brief Affine composition makes reversal order observable. */
	struct affine {
		ll a = 1, b = 0;

		bool operator==(affine const &) const = default;
	};

	affine then(affine const &f, affine const &g) { return {g.a * f.a % mod, (g.a * f.b + g.b) % mod}; }

	/**
	 * The test-only store: the docs' reversible rope, interleaved, with a
	 * non-commutative fold, plus the hooks every push strategy needs.
	 */
	struct test_rope {
		struct node {
			uint32 child[2]{};
			uint32 parent = 0, size = 0;
			affine value, aggregate, reversed;
			bool is_marked = false;
		};

		std::vector<node> nodes = std::vector<node>(1);
		std::vector<uint32> scratch;

		uint32 child(uint32 x, bool d) const { return nodes[x].child[d]; }

		uint32 parent(uint32 x) const { return nodes[x].parent; }

		uint32 size(uint32 x) const { return nodes[x].size; }

		void set_child(uint32 x, bool d, uint32 c) { nodes[x].child[d] = c; }

		void set_parent(uint32 x, uint32 p) { nodes[x].parent = p; }

		void pull(uint32 x) {
			node &n = nodes[x];
			node const &l = nodes[n.child[0]], &r = nodes[n.child[1]];
			n.size = l.size + 1 + r.size;
			n.aggregate = then(then(l.aggregate, n.value), r.aggregate);
			n.reversed = then(then(r.reversed, n.value), l.reversed);
		}

		bool is_reversed(uint32 x) const { return nodes[x].is_marked; }

		void flip(uint32 x) {
			if (x == 0) return;
			nodes[x].is_marked = !nodes[x].is_marked;
			std::swap(nodes[x].aggregate, nodes[x].reversed);
		}

		void clear_reversed(uint32 x) { nodes[x].is_marked = false; }

		void push(uint32 x) {
			node &n = nodes[x];
			if (!n.is_marked) return;
			std::swap(n.child[0], n.child[1]);
			flip(n.child[0]);
			flip(n.child[1]);
			n.is_marked = false;
		}

		std::vector<uint32> &splay_scratch() { return scratch; }

		affine const &aggregate(uint32 x) const { return nodes[x].aggregate; }

		affine const &value(uint32 x) const { return nodes[x].value; }

		uint32 make(affine v) {
			nodes.push_back({{0, 0}, 0, 1, v, v, v, false});
			return uint32(nodes.size() - 1);
		}
	};

	struct logical {
		uint32 size;
		affine forward, backward;
	};

	/**
	 * Invariant checker (with the std::vector of ids in each test, the
	 * certifying oracle for every engine contender): parent/child agreement, sizes, both aggregates
	 * against the subtree's logical order (pending marks applied), and the
	 * null sentinel untouched. Returns the subtree's logical summary.
	 */
	logical check_subtree(test_rope const &s, uint32 x, uint32 expected_parent, bool &is_ok) {
		if (x == 0) return {0, {}, {}};
		test_rope::node const &n = s.nodes[x];
		if (n.parent != expected_parent) is_ok = false;
		logical const l = check_subtree(s, n.child[0], x, is_ok), r = check_subtree(s, n.child[1], x, is_ok);
		logical out{l.size + 1 + r.size, then(then(l.forward, n.value), r.forward), then(then(r.backward, n.value), l.backward)};
		if (n.is_marked) std::swap(out.forward, out.backward);
		if (n.size != out.size || !(n.aggregate == out.forward) || !(n.reversed == out.backward)) is_ok = false;
		return out;
	}

	bool invariants_hold(test_rope const &s, uint32 root, uint32 root_parent = 0) {
		test_rope::node const &null = s.nodes[0];
		bool is_ok = null.child[0] == 0 && null.child[1] == 0 && null.parent == 0 && null.size == 0
				  && null.aggregate == affine{} && null.reversed == affine{} && null.value == affine{} && !null.is_marked;
		check_subtree(s, root, root_parent, is_ok);
		return is_ok;
	}

	template<typename Engine, atl::splay_store Store>
		requires requires(Store &store) { Engine::for_each(store, uint32{}, [](uint32) {}); }
	std::vector<uint32> in_order(Store &s, uint32 root) {
		std::vector<uint32> out;
		Engine::for_each(s, root, [&](uint32 x) { out.push_back(x); });
		return out;
	}

	using atl::experimental::splay_engine_by_strategy;
	using atl::experimental::splay_path_push;
	using atl::experimental::splay_shape;
	using engines = std::tuple<atl::splay_engine, splay_engine_by_strategy<splay_shape::bottom_up_repush, splay_path_push::ancestor_stack>,
							   splay_engine_by_strategy<splay_shape::bottom_up, splay_path_push::ancestor_stack>,
							   splay_engine_by_strategy<splay_shape::semi, splay_path_push::ancestor_stack>,
							   splay_engine_by_strategy<splay_shape::bottom_up, splay_path_push::store_buffer>,
							   splay_engine_by_strategy<splay_shape::bottom_up, splay_path_push::pointer_reversal>,
							   splay_engine_by_strategy<splay_shape::bottom_up, splay_path_push::parity_walk>,
							   splay_engine_by_strategy<splay_shape::top_down, splay_path_push::parity_walk>>;
}

static_assert(atl::sized_splay_store<test_rope>);
static_assert(atl::sized_splay_store<atl::sized_splay_links>);
static_assert(atl::splay_store<atl::splay_links> && !atl::sized_splay_store<atl::splay_links>);
static_assert(sizeof(atl::sized_splay_links::link) == 16);

TEMPLATE_LIST_TEST_CASE("splay_engine: every operation vs std::vector, invariants after each", "[splay_engine]", engines) {
	using E = TestType;
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	test_rope s;
	std::vector<uint32> ref; // node ids in sequence order
	uint32 root = 0;
	auto const random_affine = [&] { return affine{1 + ll(rng() % (mod - 1)), ll(rng() % mod)}; };
	for (int step = 0; step < 1500; ++step) {
		std::size_t const n = ref.size();
		switch (rng() % 9) {
		case 0:
		case 1: {
			std::size_t const k = rng() % (n + 1);
			uint32 const x = s.make(random_affine());
			root = E::insert_at(s, root, uint32(k), x);
			REQUIRE(root == x);
			ref.insert(ref.begin() + long(k), x);
			break;
		}
		case 2:
			if (n > 0) {
				std::size_t const k = rng() % n;
				root = E::erase(s, ref[k]);
				REQUIRE(s.parent(ref[k]) == 0);
				REQUIRE(s.size(ref[k]) == 1);
				ref.erase(ref.begin() + long(k));
			}
			break;
		case 3:
			if (n > 0) {
				uint32 const x = ref[rng() % n];
				E::splay(s, x);
				root = x;
			}
			break;
		case 4:
			if (n > 0) {
				std::size_t const k = rng() % n;
				root = E::splay_at(s, root, uint32(k));
				REQUIRE(root == ref[k]);
			}
			break;
		case 5: {
			std::size_t const l = rng() % (n + 1), r = l + rng() % (n - l + 1);
			auto const [a, bc] = E::split_at(s, root, uint32(l));
			auto const [b, c] = E::split_at(s, bc, uint32(r - l));
			REQUIRE(s.size(a) == l);
			REQUIRE(s.size(b) == r - l);
			s.flip(b);
			root = E::join(s, E::join(s, a, b), c);
			std::ranges::reverse(ref.begin() + long(l), ref.begin() + long(r));
			break;
		}
		case 6:
			if (n > 0) {
				std::size_t const k = rng() % n;
				bool const is_before = rng() % 2;
				auto const [a, b] = is_before ? E::split_before(s, ref[k]) : E::split_after(s, ref[k]);
				REQUIRE(s.size(a) == (is_before ? k : k + 1));
				REQUIRE(invariants_hold(s, a));
				REQUIRE(invariants_hold(s, b));
				root = E::join(s, a, b);
			}
			break;
		case 7: // descend with a stateful probe: the k-th node from the right
			if (n > 0) {
				std::size_t const k = rng() % n;
				uint32 remaining = uint32(k);
				root = E::descend(s, root, [&](uint32 x) {
					uint32 const right_size = s.size(s.child(x, 1));
					if (remaining < right_size) return atl::splay_step::right;
					if (remaining == right_size) return atl::splay_step::stop;
					remaining -= right_size + 1;
					return atl::splay_step::left;
				});
				REQUIRE(root == ref[n - 1 - k]);
			}
			break;
		default: {
			REQUIRE(in_order<E>(s, root) == ref);
		}
		}
		REQUIRE(s.size(root) == ref.size());
		REQUIRE(invariants_hold(s, root));
		affine want;
		for (uint32 const x : ref) want = then(want, s.nodes[x].value);
		REQUIRE(s.nodes[root].aggregate == want);
	}
	REQUIRE(in_order<E>(s, root) == ref);
}

TEMPLATE_LIST_TEST_CASE("splay_engine: max_right and a lower-bound probe on a links-mixin store", "[splay_engine]", engines) {
	using E = TestType;
	auto seed = GENERATE(take(3, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	atl::experimental::reversible_fold_rope s(ll(0), std::plus<>{});
	std::vector<ll> ref;
	uint32 root = 0;
	for (int step = 0; step < 600; ++step) {
		std::size_t const k = rng() % (ref.size() + 1);
		ll const v = ll(rng() % 100);
		root = E::insert_at(s, root, uint32(k), s.make(v));
		ref.insert(ref.begin() + long(k), v);
		ll const cap = ll(rng() % 3000);
		auto const [top, length] = E::max_right(s, root, ll(0), std::plus<>{}, [&](ll sum) { return sum <= cap; });
		root = top;
		std::size_t want = 0;
		for (ll acc = 0; want < ref.size() && acc + ref[want] <= cap;) acc += ref[want++];
		REQUIRE(length == want);
		REQUIRE(s.parent(root) == 0);
	}
	// sort through the rope, then search: first element >= key
	std::vector<ll> sorted = ref;
	std::ranges::sort(sorted);
	atl::experimental::reversible_fold_rope t(ll(0), std::plus<>{});
	uint32 sorted_root = 0;
	for (ll const v : sorted) sorted_root = E::join(t, sorted_root, t.make(v));
	for (int q = 0; q < 200; ++q) {
		ll const key = ll(rng() % 110);
		uint32 candidate = 0;
		sorted_root = E::descend(t, sorted_root, [&](uint32 x) {
			if (t.value(x) >= key) {
				candidate = x;
				return atl::splay_step::left;
			}
			return atl::splay_step::right;
		});
		auto const it = std::ranges::lower_bound(sorted, key);
		if (it == sorted.end()) {
			REQUIRE(candidate == 0);
		} else {
			REQUIRE(candidate != 0);
			REQUIRE(t.value(candidate) == *it);
			E::splay(t, candidate);
			sorted_root = candidate;
			REQUIRE(t.size(t.child(candidate, 0)) == std::size_t(it - sorted.begin()));
		}
	}
}

TEMPLATE_LIST_TEST_CASE("splay_engine: a 1e6-node path needs no recursion", "[splay_engine]", engines) {
	using E = TestType;
	constexpr uint32 n = 1'000'000;
	test_rope s;
	s.nodes.reserve(n + 1);
	uint32 root = 0;
	for (uint32 i = 0; i < n; ++i) root = E::insert_at(s, root, 0, s.make({1, i % mod})); // a right path
	REQUIRE(s.size(root) == n);
	s.flip(root);	// a pending reversal at the top of the path
	E::splay(s, 1); // the deepest node: the push pass walks the whole path
	REQUIRE(s.parent(1) == 0);
	REQUIRE(s.size(1) == n);
	root = 1;
	uint32 count = 0;
	E::for_each(s, root, [&](uint32) { ++count; });
	REQUIRE(count == n);
	root = E::splay_at(s, root, n / 2);
	REQUIRE(s.size(root) == n);
}

TEST_CASE("splay_sequence: a 1e6-element path, copied and destroyed without recursion", "[splay_engine]") {
	constexpr std::size_t n = 1'000'000;
	atl::reversible_splay_sequence<int> t;
	for (std::size_t i = 0; i < n; ++i) t.push_front(int(i)); // each push hangs the old root right: a path
	t.reverse(0, n);										  // pending at the root
	REQUIRE(t.at(n - 1) == int(n - 1));						  // the deepest node
	atl::reversible_splay_sequence<int> copy = t;
	REQUIRE(copy.at(0) == 0);
	long long sum = 0;
	t.for_each([&](int v) { sum += v; });
	REQUIRE(sum == (long long)(n) * (n - 1) / 2);
	t.clear();
	REQUIRE(t.empty());
}

TEST_CASE("splay_engine: rotate and edge cases", "[splay_engine]") {
	test_rope s;
	uint32 const a = s.make({2, 1}), b = s.make({3, 4});
	uint32 root = atl::splay_engine::join(s, a, b);
	REQUIRE(root == a);
	atl::splay_engine::rotate(s, b);
	REQUIRE(s.parent(b) == 0);
	REQUIRE(s.child(b, 0) == a);
	REQUIRE(invariants_hold(s, b));
	REQUIRE(atl::splay_engine::join(s, 0, 0) == 0);
	REQUIRE(atl::splay_engine::descend(s, 0, [](uint32) { return atl::splay_step::stop; }) == 0);
	auto const [l, r] = atl::splay_engine::split_at(s, b, 0);
	REQUIRE((l == 0 && r == b));
	auto const [l2, r2] = atl::splay_engine::split_at(s, b, 2);
	REQUIRE((l2 == b && r2 == 0));
	REQUIRE(atl::splay_engine::erase(s, a) == b);
	REQUIRE(atl::splay_engine::erase(s, b) == 0);
	auto const [top, length] = atl::splay_engine::max_right(s, 0, affine{}, then, [](affine const &) { return true; });
	REQUIRE((top == 0 && length == 0));
}

TEST_CASE("splay_engine: contract violations abort", "[splay_engine]") {
	test_rope s;
	uint32 const root = atl::splay_engine::insert_at(s, 0, 0, s.make({1, 1}));
	CHECK(atl_test::aborts([&] { atl::splay_engine::splay(s, 0); }));
	CHECK(atl_test::aborts([&] { (void)atl::splay_engine::splay_at(s, root, 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::splay_engine::split_at(s, root, 2); }));
	uint32 const other = atl::splay_engine::insert_at(s, root, 1, s.make({1, 2}));
	CHECK(atl_test::aborts([&] { (void)atl::splay_engine::insert_at(s, 0, 0, other); }));
	CHECK(atl_test::aborts([&] { atl::splay_engine::rotate(s, other); }));
	atl::splay_sequence<int> seq(std::vector<int>{1, 2, 3});
	CHECK(atl_test::aborts([&] { (void)seq.at(3); }));
	CHECK(atl_test::aborts([&] { seq.erase(2, 4); }));
}

namespace docs_example {
	using atl::int64;

	// Reversal of a sum aggregate needs only a structural mark; the sum is order-independent.
	struct sum_rope {
		struct node {
			uint32 child[2]{}, parent = 0, size = 0;
			int64 value = 0, sum = 0;
			bool is_reversed = false;
		};

		std::vector<node> nodes{node{}}; // id 0 = null: size 0, sum 0

		uint32 child(uint32 x, bool d) const { return nodes[x].child[d]; }

		uint32 parent(uint32 x) const { return nodes[x].parent; }

		uint32 size(uint32 x) const { return nodes[x].size; }

		void set_child(uint32 x, bool d, uint32 c) { nodes[x].child[d] = c; }

		void set_parent(uint32 x, uint32 p) { nodes[x].parent = p; }

		void pull(uint32 x) {
			auto &n = nodes[x];
			auto const &l = nodes[n.child[0]], &r = nodes[n.child[1]];
			n.size = l.size + 1 + r.size;
			n.sum = l.sum + n.value + r.sum;
		}

		void push(uint32 x) {
			auto &n = nodes[x];
			if (!n.is_reversed) return;
			std::swap(n.child[0], n.child[1]);
			for (uint32 c : n.child)
				if (c) nodes[c].is_reversed ^= true;
			n.is_reversed = false;
		}

		uint32 make(int64 v) {
			nodes.push_back({{0, 0}, 0, 1, v, v, false});
			return uint32(nodes.size() - 1);
		}
	};
}

TEST_CASE("splay_engine: the docs example", "[splay_engine]") {
	docs_example::sum_rope rope;
	uint32 root = 0;
	for (atl::int64 v : {1, 2, 3, 4, 5}) root = atl::splay_engine::join(rope, root, rope.make(v));
	auto [a, bc] = atl::splay_engine::split_at(rope, root, 1);
	auto [b, c] = atl::splay_engine::split_at(rope, bc, 3);
	rope.nodes[b].is_reversed ^= true;
	root = atl::splay_engine::join(rope, atl::splay_engine::join(rope, a, b), c);
	std::vector<atl::int64> order;
	atl::splay_engine::for_each(rope, root, [&](uint32 x) { order.push_back(rope.nodes[x].value); });
	REQUIRE(order == std::vector<atl::int64>{1, 4, 3, 2, 5});
	REQUIRE(rope.nodes[root].sum == 15);
}

TEST_CASE("splay_engine: build_balanced", "[splay_engine]") {
	for (uint32 count : {0u, 1u, 2u, 7u, 8u, 1000u}) {
		docs_example::sum_rope rope;
		std::vector<uint32> ids;
		for (uint32 i = 0; i < count; ++i) ids.push_back(rope.make(atl::int64(i)));
		std::ranges::reverse(ids); // in-order is the order of ids, not of creation
		uint32 const root = atl::splay_engine::build_balanced(rope, ids);
		if (count == 0) {
			CHECK(root == 0);
			continue;
		}
		CHECK(rope.parent(root) == 0);
		CHECK(rope.size(root) == count);
		CHECK(rope.nodes[root].sum == atl::int64(count) * (count - 1) / 2);
		std::vector<uint32> order;
		atl::splay_engine::for_each(rope, root, [&](uint32 x) { order.push_back(x); });
		CHECK(order == ids);
		uint32 depth = 0; // perfectly balanced: depth floor(log2 count)
		for (uint32 x = ids.front(); x != root; x = rope.parent(x)) ++depth;
		CHECK(depth <= uint32(std::bit_width(count)) - 1);
		CHECK(atl::splay_engine::splay_at(rope, root, count / 2) == ids[count / 2]);
	}
}

TEST_CASE("splay_engine: insertion checks a detached parent and reserve checks its sentinel", "[splay_engine]") {
	REQUIRE(atl_test::aborts([] {
		atl::sized_splay_links store;
		auto const root = store.add_node(), node = store.add_node();
		store.set_parent(node, root);
		static_cast<void>(atl::splay_engine::insert_at(store, root, 0, node));
	}));
	REQUIRE(atl_test::aborts([] {
		atl::splay_links store;
		store.reserve((std::numeric_limits<std::size_t>::max)());
	}));
}
