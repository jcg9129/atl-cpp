#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/subtree_link_cut_tree.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;
using ll = long long;

namespace {
	constexpr uint32 none = atl::graph::none;
	constexpr ll prime = 1'000'000'007;

	/** @brief A non-trivial commutative group: pairs under componentwise addition mod p. */
	struct mod_pair {
		ll a = 0, b = 0;

		bool operator==(mod_pair const &) const = default;
	};

	struct mod_pair_add {
		constexpr mod_pair operator()(mod_pair const &x, mod_pair const &y) const {
			return {(x.a + y.a) % prime, (x.b + y.b) % prime};
		}
	};

	struct mod_pair_negate {
		constexpr mod_pair operator()(mod_pair const &x) const { return {(prime - x.a) % prime, (prime - x.b) % prime}; }
	};

	struct identity_inverse {
		constexpr uint64 operator()(uint64 x) const { return x; }
	};

	/**
	 * @brief The certifying oracle: a parent array with explicit roots, values per
	 * vertex; make_root reverses the root path; subtrees by walking every
	 * vertex's ancestors. O(n) per structural op, O(n * depth) per subtree query.
	 */
	template<std::copyable Value, atl::binary_operation<Value> Op>
	struct naive_forest {
		std::vector<uint32> par;
		std::vector<Value> val;
		Value identity;
		Op op;

		naive_forest(std::vector<Value> values, Value id, Op o) : par(values.size(), none), val(std::move(values)), identity(std::move(id)), op(std::move(o)) {}

		uint32 root(uint32 v) const {
			while (par[v] != none) v = par[v];
			return v;
		}

		bool connected(uint32 u, uint32 v) const { return root(u) == root(v); }

		uint32 depth(uint32 v) const {
			uint32 d = 0;
			for (; par[v] != none; v = par[v]) ++d;
			return d;
		}

		void make_root(uint32 v) {
			uint32 prev = none;
			while (v != none) {
				uint32 const next = par[v];
				par[v] = prev;
				prev = v;
				v = next;
			}
		}

		void link(uint32 u, uint32 v) {
			make_root(u);
			par[u] = v;
		}

		bool is_ancestor(uint32 a, uint32 w) const {
			for (; w != none; w = par[w])
				if (w == a) return true;
			return false;
		}

		uint32 lca(uint32 u, uint32 v) const {
			if (!connected(u, v)) return none;
			for (uint32 x = v;; x = par[x])
				if (is_ancestor(x, u)) return x;
		}

		std::vector<uint32> path(uint32 u, uint32 v) const {
			uint32 const w = lca(u, v);
			std::vector<uint32> up, down;
			for (uint32 x = u; x != w; x = par[x]) up.push_back(x);
			up.push_back(w);
			for (uint32 x = v; x != w; x = par[x]) down.push_back(x);
			up.insert(up.end(), down.rbegin(), down.rend());
			return up;
		}

		Value fold(uint32 u, uint32 v) const {
			Value acc = identity;
			for (uint32 const x : path(u, v)) acc = op(acc, val[x]);
			return acc;
		}

		/** @brief The fold and size of v's subtree under the current root. */
		std::pair<Value, uint32> subtree(uint32 v) const {
			Value acc = identity;
			uint32 count = 0;
			for (uint32 w = 0; w < par.size(); ++w)
				if (is_ancestor(v, w)) acc = op(acc, val[w]), ++count;
			return {acc, count};
		}

		std::pair<Value, uint32> component(uint32 v) const {
			Value acc = identity;
			uint32 count = 0;
			for (uint32 w = 0; w < par.size(); ++w)
				if (connected(v, w)) acc = op(acc, val[w]), ++count;
			return {acc, count};
		}

		std::vector<uint32> non_roots() const {
			std::vector<uint32> r;
			for (uint32 v = 0; v < par.size(); ++v)
				if (par[v] != none) r.push_back(v);
			return r;
		}
	};

	/** @brief A random forest as a parent array (roughly a quarter of the vertices roots). */
	std::vector<uint32> random_parents(uint32 n, std::mt19937_64 &rng) {
		std::vector<uint32> order(n);
		for (uint32 i = 0; i < n; ++i) order[i] = i;
		std::ranges::shuffle(order, rng);
		std::vector<uint32> parents(n, none);
		for (uint32 i = 1; i < n; ++i)
			if (rng() % 4 != 0) parents[order[i]] = order[rng() % i];
		return parents;
	}

	/** @brief Random mixed operations on @p t against @p nf: every member, make_root interleaved. */
	template<typename Tree, std::copyable Value, atl::binary_operation<Value> Op, atl::inverse_operation<Value> Inverse,
			 std::invocable<std::mt19937_64 &> RandomValue>
	void drive(Tree &t, naive_forest<Value, Op> &nf, std::mt19937_64 &rng, int ops, Inverse const &inverse, RandomValue &&rv) {
		uint32 const n = uint32(nf.par.size());
		auto pick = [&] { return uint32(rng() % n); };
		for (int i = 0; i < ops; ++i) {
			uint32 const u = pick(), v = pick();
			switch (rng() % 16) {
			case 0:
			case 1:
				if (!nf.connected(u, v)) {
					t.link(u, v);
					nf.link(u, v);
				} else {
					REQUIRE(t.connected(u, v));
				}
				break;
			case 2: {
				auto const children = nf.non_roots();
				if (children.empty()) break;
				uint32 const c = children[rng() % children.size()], p = nf.par[c];
				switch (rng() % 3) {
				case 0:
					t.cut(c, p);
					break;
				case 1:
					t.cut(p, c);
					break;
				default:
					t.cut_from_parent(c);
				}
				nf.par[c] = none;
				break;
			}
			case 3:
				t.make_root(u);
				nf.make_root(u);
				break;
			case 4:
				switch (rng() % 4) {
				case 0:
					REQUIRE(t.connected(u, v) == nf.connected(u, v));
					break;
				case 1:
					REQUIRE(t.root(u) == nf.root(u));
					break;
				case 2:
					REQUIRE(t.parent(u) == nf.par[u]);
					break;
				default:
					REQUIRE(t.lca(u, v) == nf.lca(u, v));
				}
				break;
			case 5:
				REQUIRE(t.depth(u) == nf.depth(u));
				break;
			case 6:
			case 7: {
				Value const x = rv(rng);
				if (rng() % 2) {
					t.set(u, x);
					nf.val[u] = x;
				} else {
					auto const f = [&](Value const &a) { return nf.op(a, x); };
					t.update(u, f);
					nf.val[u] = f(nf.val[u]);
				}
				break;
			}
			case 8:
				REQUIRE(t.at(u) == nf.val[u]);
				break;
			case 9:
				if (nf.connected(u, v)) REQUIRE(t.fold(u, v) == nf.fold(u, v));
				break;
			case 10:
			case 11: {
				auto const [fold, count] = nf.subtree(u);
				if (rng() % 2)
					REQUIRE(t.subtree_fold(u) == fold);
				else
					REQUIRE(t.subtree_size(u) == count);
				break;
			}
			case 12:
			case 13: {
				// both sides of a random edge, either orientation of the query
				auto const children = nf.non_roots();
				if (children.empty()) break;
				uint32 const c = children[rng() % children.size()], p = nf.par[c];
				auto const [below, below_count] = nf.subtree(c);
				auto const [all, all_count] = nf.component(c);
				Value const above = nf.op(all, inverse(below));
				REQUIRE(t.subtree_fold(c, p) == below);
				REQUIRE(t.subtree_size(c, p) == below_count);
				REQUIRE(t.subtree_fold(p, c) == above);
				REQUIRE(t.subtree_size(p, c) == all_count - below_count);
				break;
			}
			case 14:
				REQUIRE(t.component_size(u) == nf.component(u).second);
				break;
			default:
				REQUIRE(t.distance(u, v) == (nf.connected(u, v) ? uint32(nf.path(u, v).size() - 1) : none));
			}
		}
		for (uint32 v = 0; v < n; ++v) {
			REQUIRE(t.parent(v) == nf.par[v]);
			REQUIRE(t.subtree_fold(v) == nf.subtree(v).first);
			REQUIRE(t.subtree_size(v) == nf.subtree(v).second);
			REQUIRE(t.component_size(v) == nf.component(v).second);
		}
	}

	/** @brief One group on the library class and the split / columnar layouts. */
	template<std::copyable Value, atl::binary_operation<Value> Op, atl::inverse_operation<Value> Inverse,
			 std::invocable<std::mt19937_64 &> RandomValue>
	void run_group(int seeds, int ops, Value identity, Op op, Inverse inverse, RandomValue rv) {
		using atl::detail::splay_layout_columnar;
		using atl::detail::splay_layout_split;
		for (int seed = 0; seed < seeds; ++seed) {
			std::mt19937_64 rng(uint64(seed) * 104'729 + 3);
			uint32 const n = 1 + uint32(rng() % 120);
			std::vector<Value> values(n);
			for (auto &x : values) x = rv(rng);
			auto const parents = random_parents(n, rng);
			bool const is_prebuilt = seed % 2 == 0;
			auto const start = [&](auto &t) {
				naive_forest<Value, Op> nf(values, identity, op);
				if (is_prebuilt) {
					t.assign_parents(parents);
					nf.par = parents;
				}
				return nf;
			};
			{
				atl::subtree_link_cut_tree<Value, Op, Inverse> t(values, identity, op, inverse);
				auto nf = start(t);
				drive(t, nf, rng, ops, inverse, rv);
			}
			{
				atl::detail::subtree_link_cut_tree_impl<splay_layout_split, Value, Op, Inverse> t(values, identity, op, inverse);
				auto nf = start(t);
				drive(t, nf, rng, ops, inverse, rv);
			}
			{
				atl::detail::subtree_link_cut_tree_impl<splay_layout_columnar, Value, Op, Inverse> t(values, identity, op, inverse);
				auto nf = start(t);
				drive(t, nf, rng, ops, inverse, rv);
			}
		}
	}
}

TEST_CASE("subtree_link_cut_tree: sum group against the naive forest", "[subtree_link_cut_tree]") {
	run_group(40, 2500, 0LL, std::plus<>{}, std::negate<>{}, [](std::mt19937_64 &r) { return ll(r() % 2001) - 1000; });
}

TEST_CASE("subtree_link_cut_tree: xor group against the naive forest", "[subtree_link_cut_tree]") {
	run_group(40, 2500, uint64(0), std::bit_xor<>{}, identity_inverse{}, [](std::mt19937_64 &r) { return uint64(r()); });
}

TEST_CASE("subtree_link_cut_tree: mod-p pair group against the naive forest", "[subtree_link_cut_tree]") {
	run_group(40, 2500, mod_pair{}, mod_pair_add{}, mod_pair_negate{},
			  [](std::mt19937_64 &r) { return mod_pair{ll(r() % prime), ll(r() % prime)}; });
}

TEST_CASE("subtree_link_cut_tree: deduction, small cases, empty forest", "[subtree_link_cut_tree]") {
	atl::subtree_link_cut_tree t(5, 0LL, std::plus<>{}, std::negate<>{});
	static_assert(std::is_same_v<decltype(t), atl::subtree_link_cut_tree<ll, std::plus<>, std::negate<>>>);
	atl::subtree_link_cut_tree r(std::vector<int>{1, 2, 3}, 0LL, std::plus<>{}, std::negate<>{});
	static_assert(std::is_same_v<decltype(r), atl::subtree_link_cut_tree<ll, std::plus<>, std::negate<>>>);
	REQUIRE(t.size() == 5);
	REQUIRE(t.component_size(3) == 1);
	REQUIRE(t.subtree_fold(3) == 0);
	// 1 - 0 - 2 rooted at 0 (each link hangs u's tree below v's root), 3 alone
	r.link(1, 0);
	r.link(2, 0);
	REQUIRE(r.root(1) == 0);
	REQUIRE(r.subtree_fold(0) == 6);
	REQUIRE(r.subtree_fold(1) == 2);
	REQUIRE(r.subtree_size(0) == 3);
	REQUIRE(r.subtree_fold(0, 1) == 4); // the side of {0, 1} holding 0
	REQUIRE(r.subtree_size(0, 2) == 2);
	r.make_root(1);
	REQUIRE(r.subtree_fold(0) == 4);
	REQUIRE(r.subtree_fold(0, 1) == 4); // root changes nothing for the edge form
	r.update(2, [](ll a) { return a + 10; });
	REQUIRE(r.subtree_fold(1) == 16);
	REQUIRE(r.fold(1, 2) == 16);
	r.cut(0, 2);
	REQUIRE(r.component_size(2) == 1);
	REQUIRE(r.component_size(1) == 2);

	atl::subtree_link_cut_tree<ll, std::plus<>, std::negate<>> empty;
	REQUIRE(empty.empty());
}

TEST_CASE("subtree_link_cut_tree: a 1e6-vertex path (no recursion, deep splays)", "[subtree_link_cut_tree]") {
	constexpr uint32 n = 1'000'000;
	atl::subtree_link_cut_tree t(std::vector<ll>(n, 1), 0LL, std::plus<>{}, std::negate<>{});
	std::vector<uint32> parents(n);
	parents[0] = none;
	for (uint32 v = 1; v < n; ++v) parents[v] = v - 1;
	t.assign_parents(parents);
	REQUIRE(t.subtree_size(0) == n);
	REQUIRE(t.subtree_fold(n / 2) == n - n / 2);
	REQUIRE(t.subtree_size(n / 2, n / 2 + 1) == n / 2 + 1);
	t.make_root(n - 1);
	REQUIRE(t.subtree_fold(n / 2) == n / 2 + 1);
	REQUIRE(t.component_size(0) == n);
	t.cut_from_parent(n / 2);
	REQUIRE(t.component_size(0) == n / 2 + 1);
}

TEST_CASE("subtree_link_cut_tree: preconditions abort", "[subtree_link_cut_tree]") {
	auto make = [] {
		atl::subtree_link_cut_tree t(4, 0LL, std::plus<>{}, std::negate<>{});
		t.link(0, 1);
		t.link(2, 1);
		return t;
	};
	CHECK(atl_test::aborts([&] { (void)make().subtree_fold(0, 2); })); // not an edge
	CHECK(atl_test::aborts([&] { (void)make().subtree_size(3, 0); })); // not an edge (other tree)
	CHECK(atl_test::aborts([&] { (void)make().subtree_fold(1, 1); })); // not an edge
	CHECK(atl_test::aborts([&] { make().link(0, 2); }));			   // same tree
	CHECK(atl_test::aborts([&] { (void)make().subtree_fold(4); }));	   // out of range
	CHECK(atl_test::aborts([&] { (void)make().component_size(9); }));
	CHECK(atl_test::aborts([&] { make().cut(0, 2); }));
}
