#include <atl/ds/euler_tour_tree.hpp>
#include <atl/experimental/ds/euler_tour_tree_variants.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;
using ll = long long;

namespace {
	constexpr ll mod = 1'000'003;
	constexpr ll minus_infinity = (std::numeric_limits<ll>::min)();

	struct max_op {
		constexpr ll operator()(ll a, ll b) const { return (std::max)(a, b); }
	};

	/** @brief Sum with a width (the vertex count), so range tags see how many vertices they cover. */
	struct sum_count {
		ll sum = 0, count = 0;

		bool operator==(sum_count const &) const = default;
	};

	struct sum_op {
		constexpr sum_count operator()(sum_count const &x, sum_count const &y) const {
			return {(x.sum + y.sum) % mod, x.count + y.count};
		}
	};

	struct add_mapping {
		constexpr sum_count operator()(ll c, sum_count const &x) const { return {(x.sum + c * x.count) % mod, x.count}; }
	};

	/** @brief x -> a x + b mod p: tags that do not commute, so compose order is pinned. */
	struct affine {
		ll a = 1, b = 0;

		bool operator==(affine const &) const = default;
	};

	struct affine_mapping {
		constexpr sum_count operator()(affine const &f, sum_count const &x) const {
			return {(f.a * x.sum + f.b * x.count) % mod, x.count};
		}
	};

	/** @brief "g first, then f". */
	struct affine_compose {
		constexpr affine operator()(affine const &f, affine const &g) const { return {f.a * g.a % mod, (f.a * g.b + f.b) % mod}; }
	};

	/** @brief Max with a width: assignment must leave the arcs' identity alone. */
	struct max_count {
		ll max = minus_infinity, count = 0;

		bool operator==(max_count const &) const = default;
	};

	struct max_count_op {
		constexpr max_count operator()(max_count const &x, max_count const &y) const {
			return {(std::max)(x.max, y.max), x.count + y.count};
		}
	};

	struct assign_tag {
		bool is_set = false;
		ll c = 0;

		bool operator==(assign_tag const &) const = default;
	};

	struct assign_mapping {
		constexpr max_count operator()(assign_tag const &t, max_count const &x) const {
			return t.is_set && x.count > 0 ? max_count{t.c, x.count} : x;
		}
	};

	struct assign_compose {
		constexpr assign_tag operator()(assign_tag const &f, assign_tag const &g) const { return f.is_set ? f : g; }
	};

	/** @brief The certifying oracle: adjacency lists and per-vertex values; every query a DFS. */
	template<std::copyable V>
	struct naive_forest {
		std::vector<std::vector<uint32>> adj;
		std::vector<V> val;

		naive_forest(uint32 n, V fill) : adj(n), val(n, fill) {}

		/** @brief Vertices reachable from v without stepping onto `blocked`. */
		std::vector<uint32> reach(uint32 v, uint32 blocked) const {
			std::vector<uint32> out{v}, stack{v};
			std::vector<bool> seen(adj.size());
			seen[v] = true;
			if (blocked < adj.size()) seen[blocked] = true;
			while (!stack.empty()) {
				uint32 const x = stack.back();
				stack.pop_back();
				for (uint32 const y : adj[x])
					if (!seen[y]) {
						seen[y] = true;
						out.push_back(y);
						stack.push_back(y);
					}
			}
			return out;
		}

		std::vector<uint32> component(uint32 v) const { return reach(v, uint32(-1)); }

		bool connected(uint32 u, uint32 v) const { return std::ranges::contains(component(u), v); }

		bool has_edge(uint32 u, uint32 v) const { return std::ranges::contains(adj[u], v); }

		void link(uint32 u, uint32 v) {
			adj[u].push_back(v);
			adj[v].push_back(u);
		}

		void cut(uint32 u, uint32 v) {
			std::erase(adj[u], v);
			std::erase(adj[v], u);
		}

		template<atl::binary_operation<V> Op>
		V fold(std::vector<uint32> const &vs, V identity, Op const &op) const {
			V acc = identity;
			for (uint32 const x : vs) acc = op(acc, val[x]);
			return acc;
		}
	};

	/** @brief Random edge of the oracle (u, v) or {n, n} when there is none. */
	template<std::copyable V>
	std::pair<uint32, uint32> random_edge(naive_forest<V> const &nf, std::mt19937_64 &rng) {
		uint32 const n = uint32(nf.adj.size());
		std::vector<std::pair<uint32, uint32>> edges;
		for (uint32 u = 0; u < n; ++u)
			for (uint32 const v : nf.adj[u]) edges.emplace_back(u, v);
		if (edges.empty()) return {n, n};
		return edges[rng() % edges.size()];
	}

	/**
	 * @brief One random walk of mixed operations against the oracle. @p identity and
	 * @p op define the fold; @p map applies a tag to a value; @p rv draws a
	 * vertex value, @p rt a tag.
	 */
	template<bool HasValue, bool HasTags, typename Ett, std::copyable V, atl::binary_operation<V> Op, typename Map,
			 std::invocable RandomValue, std::invocable RandomTag>
		requires requires(Ett &t, uint32 v) {
			t.link(v, v);
			t.cut(v, v);
			{ t.connected(v, v) } -> std::same_as<bool>;
			{ t.component_size(v) } -> std::convertible_to<std::size_t>;
		} && std::regular_invocable<Map const &, std::invoke_result_t<RandomTag &>, V const &>
	void drive(Ett &t, naive_forest<V> &nf, std::mt19937_64 &rng, int ops, V identity, Op const &op, Map const &map,
			   RandomValue &&rv, RandomTag &&rt) {
		uint32 const n = uint32(nf.adj.size());
		auto vertex = [&] { return uint32(rng() % n); };
		for (int step = 0; step < ops; ++step) {
			switch (rng() % 10) {
			case 0:
			case 1: { // link (bias towards trees that grow)
				uint32 const u = vertex(), v = vertex();
				if (!nf.connected(u, v)) {
					t.link(u, v);
					nf.link(u, v);
				}
				break;
			}
			case 2: {
				auto const [u, v] = random_edge(nf, rng);
				if (u < n) {
					if (rng() % 2)
						t.cut(u, v);
					else
						t.cut(v, u);
					nf.cut(u, v);
				}
				break;
			}
			case 3: {
				uint32 const u = vertex(), v = vertex();
				REQUIRE(t.connected(u, v) == nf.connected(u, v));
				break;
			}
			case 4: {
				uint32 const v = vertex();
				REQUIRE(t.component_size(v) == nf.component(v).size());
				break;
			}
			case 5:
				if constexpr (HasValue) {
					uint32 const v = vertex();
					REQUIRE(t.component_fold(v) == nf.fold(nf.component(v), identity, op));
				}
				break;
			case 6:
				if constexpr (HasValue) {
					auto const [v, p] = random_edge(nf, rng);
					if (v < n) REQUIRE(t.subtree_fold(v, p) == nf.fold(nf.reach(v, p), identity, op));
				}
				break;
			case 7:
				if constexpr (HasTags) {
					auto const f = rt();
					uint32 const v = vertex();
					t.component_apply(v, f);
					for (uint32 const x : nf.component(v)) nf.val[x] = map(f, nf.val[x]);
				}
				break;
			case 8:
				if constexpr (HasTags) {
					auto const [v, p] = random_edge(nf, rng);
					if (v < n) {
						auto const f = rt();
						t.subtree_apply(v, p, f);
						for (uint32 const x : nf.reach(v, p)) nf.val[x] = map(f, nf.val[x]);
					}
				}
				break;
			default:
				if constexpr (HasValue) {
					uint32 const v = vertex();
					REQUIRE(t.at(v) == nf.val[v]);
					V const a = rv();
					if (rng() % 2) {
						t.set(v, a);
						nf.val[v] = a;
					} else {
						t.update(v, [&](V const &old) { return op(old, a); });
						nf.val[v] = op(nf.val[v], a);
					}
				}
				break;
			}
		}
		for (uint32 v = 0; v < n; ++v) {
			REQUIRE(t.component_size(v) == nf.component(v).size());
			if constexpr (HasValue) REQUIRE(t.at(v) == nf.val[v]);
		}
	}

	/** @brief The library treap-store class: its type and CTAD construction. */
	struct treap_kind {
		template<typename Value = void, typename Op = void, typename Tag = void, typename Mapping = void, typename Compose = void>
		using type = atl::euler_tour_tree<Value, Op, Tag, Mapping, Compose>;

		template<typename... Args>
			requires requires(Args &&...forwarded) { atl::euler_tour_tree(std::forward<Args>(forwarded)...); }
		static auto make(Args &&...args) {
			return atl::euler_tour_tree(std::forward<Args>(args)...);
		}
	};

	/** @brief The experimental splay-engine loser: same API, same deduction. */
	struct splay_kind {
		template<typename Value = void, typename Op = void, typename Tag = void, typename Mapping = void, typename Compose = void>
		using type = atl::experimental::euler_tour_tree_by_splay<Value, Op, Tag, Mapping, Compose>;

		template<typename... Args>
			requires requires(Args &&...forwarded) { atl::experimental::euler_tour_tree_by_splay(std::forward<Args>(forwarded)...); }
		static auto make(Args &&...args) {
			return atl::experimental::euler_tour_tree_by_splay(std::forward<Args>(args)...);
		}
	};

	auto no_map = [](auto const &, auto const &x) { return x; };
	auto no_tag = [] { return 0; };
} // namespace

TEMPLATE_TEST_CASE("euler_tour_tree: every tier against the naive forest", "[euler_tour_tree]", treap_kind, splay_kind) {
	// plain blocks, not SECTIONs: Catch runs a section once per test-case run, which would skip later seeds
	for (uint64 seed = 1; seed <= 60; ++seed) {
		std::mt19937_64 rng(seed);
		uint32 const n = 1 + uint32(rng() % 14);
		int const ops = 2000;
		auto draw_value = [&] { return ll(rng() % 100); };
		{ // structure only
			auto t = TestType::make(n);
			naive_forest<int> nf(n, 0);
			drive<false, false>(t, nf, rng, ops, 0, std::plus<>{}, no_map, no_tag, no_tag);
		}
		{ // max fold, set / update
			std::vector<ll> init(n);
			for (ll &x : init) x = draw_value();
			auto t = TestType::make(init, minus_infinity, max_op{});
			naive_forest<ll> nf(n, 0);
			nf.val = init;
			drive<true, false>(t, nf, rng, ops, minus_infinity, max_op{}, no_map, draw_value, no_tag);
		}
		{ // range add on sum with width
			auto t = TestType::make(std::vector<sum_count>(n, {0, 1}), sum_count{}, 0LL, sum_op{}, add_mapping{}, std::plus<>{});
			naive_forest<sum_count> nf(n, {0, 1});
			drive<true, true>(t, nf, rng, ops, sum_count{}, sum_op{}, add_mapping{}, [&] { return sum_count{draw_value(), 1}; }, [&] { return ll(rng() % 50); });
		}
		{ // affine tags on sum with width (non-commuting tags)
			auto t = TestType::make(std::vector<sum_count>(n, {1, 1}), sum_count{}, affine{}, sum_op{}, affine_mapping{},
									affine_compose{});
			naive_forest<sum_count> nf(n, {1, 1});
			drive<true, true>(t, nf, rng, ops, sum_count{}, sum_op{}, affine_mapping{}, [&] { return sum_count{draw_value(), 1}; }, [&] { return affine{ll(rng() % mod), ll(rng() % mod)}; });
		}
		{ // assign on max with width
			auto t = TestType::make(std::vector<max_count>(n, {0, 1}), max_count{}, assign_tag{}, max_count_op{}, assign_mapping{},
									assign_compose{});
			naive_forest<max_count> nf(n, {0, 1});
			drive<true, true>(t, nf, rng, ops, max_count{}, max_count_op{}, assign_mapping{}, [&] { return max_count{draw_value(), 1}; }, [&] { return assign_tag{rng() % 4 != 0, draw_value()}; });
		}
	}
}

TEMPLATE_TEST_CASE("euler_tour_tree: assign_parents against the naive forest", "[euler_tour_tree]", treap_kind, splay_kind) {
	constexpr uint32 none = atl::graph::none;
	for (uint64 seed = 1; seed <= 40; ++seed) {
		std::mt19937_64 rng(seed);
		uint32 const n = 1 + uint32(rng() % 30);
		auto t = TestType::make(std::vector<sum_count>(n, {1, 1}), sum_count{}, affine{}, sum_op{}, affine_mapping{},
								affine_compose{});
		naive_forest<sum_count> nf(n, {1, 1});
		if (seed % 2 == 0) { // churn to no edges first: shuffled free ids, tags left on isolated vertices
			std::vector<std::pair<uint32, uint32>> edges;
			for (uint32 v = 1; v < n; ++v) {
				edges.emplace_back(v, uint32(rng() % v));
				t.link(edges.back().first, edges.back().second);
				nf.link(edges.back().first, edges.back().second);
			}
			t.component_apply(0, affine{2, 1});
			for (sum_count &x : nf.val) x = affine_mapping{}(affine{2, 1}, x);
			std::ranges::shuffle(edges, rng);
			for (auto const &[u, v] : edges) {
				t.cut(u, v);
				nf.cut(u, v);
				t.component_apply(u, affine{3, 0});
				for (uint32 const x : nf.component(u)) nf.val[x] = affine_mapping{}(affine{3, 0}, nf.val[x]);
			}
		}
		std::vector<uint32> parents(n, none);
		for (uint32 v = 1; v < n; ++v)
			if (rng() % 5 != 0) parents[v] = uint32(rng() % v);
		if (n > 1 && rng() % 3 == 0) parents[0] = 0; // a self-parent marks a root too
		t.assign_parents(parents);
		for (uint32 v = 0; v < n; ++v)
			if (parents[v] != none && parents[v] != v) nf.link(v, parents[v]);
		drive<true, true>(t, nf, rng, 1500, sum_count{}, sum_op{}, affine_mapping{}, [&] { return sum_count{ll(rng() % 100), 1}; }, [&] { return affine{ll(rng() % mod), ll(rng() % mod)}; });
	}
}

TEMPLATE_TEST_CASE("euler_tour_tree: assign_parents after link / cut churn", "[euler_tour_tree]", treap_kind, splay_kind) {
	constexpr uint32 none = atl::graph::none;
	auto t = TestType::make(std::vector<sum_count>(5, {1, 1}), sum_count{}, 0LL, sum_op{}, add_mapping{}, std::plus<>{});
	t.link(0, 1);
	t.link(2, 3);
	t.link(1, 4);
	t.cut(2, 3);			 // frees an id out of order
	t.component_apply(2, 5); // a tag on an isolated vertex
	t.cut(0, 1);
	t.cut(4, 1);
	t.assign_parents(std::vector<uint32>{none, 0, 1, 1, 3});
	CHECK(t.component_size(0) == 5);
	CHECK(t.subtree_fold(1, 0) == sum_count{1 + 6 + 1 + 1, 4});
	CHECK(t.subtree_fold(3, 1) == sum_count{2, 2});
	t.cut(1, 3);
	t.link(3, 0); // every id still usable
	CHECK(t.subtree_fold(0, 3) == sum_count{1 + 1 + 6, 3});
}

TEMPLATE_TEST_CASE("euler_tour_tree: larger forests (deep trees, arc reuse)", "[euler_tour_tree]", treap_kind, splay_kind) {
	for (uint64 seed = 1; seed <= 3; ++seed) {
		std::mt19937_64 rng(seed);
		uint32 const n = 300;
		auto t = TestType::make(std::vector<sum_count>(n, {0, 1}), sum_count{}, 0LL, sum_op{}, add_mapping{}, std::plus<>{});
		naive_forest<sum_count> nf(n, {0, 1});
		for (uint32 v = 1; v < n; ++v) { // a long path first: deep tours
			t.link(v - 1, v);
			nf.link(v - 1, v);
		}
		drive<true, true>(t, nf, rng, 6000, sum_count{}, sum_op{}, add_mapping{}, [&] { return sum_count{ll(rng() % 100), 1}; }, [&] { return ll(rng() % 50); });
	}
}

TEMPLATE_TEST_CASE("euler_tour_tree: a 1e6-vertex path (deep tours, no deep recursion)", "[euler_tour_tree]", treap_kind, splay_kind) {
	uint32 const n = 1'000'000;
	auto t = TestType::make(std::vector<sum_count>(n, {1, 1}), sum_count{}, 0LL, sum_op{}, add_mapping{}, std::plus<>{});
	for (uint32 v = 1; v < n; ++v) t.link(v, v - 1);
	CHECK(t.component_size(0) == n);
	t.subtree_apply(n / 2, n / 2 - 1, 2); // the upper half
	CHECK(t.subtree_fold(n / 2 - 1, n / 2) == sum_count{n / 2, n / 2});
	CHECK(t.component_fold(n - 1) == sum_count{(ll(n / 2) + ll(n / 2) * 3) % mod, n});
	t.cut(n / 2, n / 2 - 1);
	CHECK(t.component_size(0) == n / 2);
	CHECK(!t.connected(0, n - 1));
}

TEMPLATE_TEST_CASE("euler_tour_tree: deduction, small cases, empty forest", "[euler_tour_tree]", treap_kind, splay_kind) {
	auto empty = TestType::make();
	static_assert(std::is_same_v<decltype(empty), typename TestType::template type<>>);
	CHECK(empty.empty());
	CHECK(empty.size() == 0);

	auto one = TestType::make(1);
	CHECK(one.connected(0, 0));
	CHECK(one.component_size(0) == 1);

	auto sums = TestType::make(std::vector<int>{1, 2, 3}, 0LL, std::plus<>{});
	static_assert(std::is_same_v<decltype(sums), typename TestType::template type<ll, std::plus<>>>);
	sums.link(0, 1);
	sums.link(2, 1);
	CHECK(sums.component_fold(0) == 6);
	CHECK(sums.subtree_fold(1, 0) == 5);
	CHECK(sums.subtree_fold(0, 1) == 1);
	CHECK(sums.at(2) == 3);
	sums.cut(1, 2);
	CHECK(sums.component_fold(2) == 3);
	CHECK(sums.component_fold(1) == 3);

	auto sized = TestType::make(4, sum_count{}, 0LL, sum_op{}, add_mapping{}, std::plus<>{});
	static_assert(std::is_same_v<decltype(sized), typename TestType::template type<sum_count, sum_op, ll, add_mapping, std::plus<>>>);
	for (uint32 v = 0; v < 4; ++v) sized.set(v, sum_count{v, 1});
	sized.link(0, 1);
	sized.link(1, 2);
	sized.link(1, 3);
	sized.subtree_apply(1, 0, 10); // 1, 2, 3
	CHECK(sized.component_fold(0) == sum_count{6 + 30, 4});
	CHECK(sized.at(0) == sum_count{0, 1});
	CHECK(sized.at(3) == sum_count{13, 1});
	sized.component_apply(2, 1);
	CHECK(sized.subtree_fold(0, 1) == sum_count{1, 1});
}

TEMPLATE_TEST_CASE("euler_tour_tree: preconditions abort", "[euler_tour_tree]", treap_kind, splay_kind) {
	auto make = [] {
		auto t = TestType::make(4, 0LL, std::plus<>{});
		t.link(0, 1);
		t.link(1, 2);
		return t;
	};
	CHECK(atl_test::aborts([&] { make().link(0, 2); }));									  // same tree
	CHECK(atl_test::aborts([&] { make().link(3, 3); }));									  // same vertex
	CHECK(atl_test::aborts([&] { make().cut(0, 2); }));										  // not an edge
	CHECK(atl_test::aborts([&] { make().cut(0, 3); }));										  // across trees
	CHECK(atl_test::aborts([&] { (void)make().subtree_fold(0, 2); }));						  // not an edge
	CHECK(atl_test::aborts([&] { (void)make().component_size(4); }));						  // out of range
	CHECK(atl_test::aborts([&] { (void)make().connected(0, 4); }));							  // out of range
	CHECK(atl_test::aborts([&] { make().set(9, 1LL); }));									  // out of range
	CHECK(atl_test::aborts([&] { make().assign_parents(std::vector<uint32>{4, 0, 1, 2}); })); // edges exist
	CHECK(atl_test::aborts([] {
		auto t = TestType::make(3);
		t.assign_parents(std::vector<uint32>{1, 2, 0}); // a cycle
	}));
	CHECK(atl_test::aborts([] {
		auto t = TestType::make(3);
		t.assign_parents(std::vector<uint32>{1, 2}); // too short
	}));
	CHECK(!atl_test::aborts([&] { make().cut(2, 1); }));				// either orientation
	CHECK(!atl_test::aborts([&] { (void)make().subtree_fold(1, 0); })); // an edge
}

TEST_CASE("Euler tour node counts fit their 32-bit links", "[euler_tour_tree]") {
	constexpr std::size_t too_many = std::size_t(atl::graph::none) / 3 + 1;
	REQUIRE(atl_test::aborts([] { atl::euler_tour_tree tree{too_many}; }));
	REQUIRE(atl_test::aborts([] { atl::experimental::euler_tour_tree_by_splay<> tree{too_many}; }));
}
