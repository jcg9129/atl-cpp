#include <atl/ds/lazy_segment_tree.hpp>
#include <atl/experimental/graph/lca_variants.hpp>
#include <atl/graph/lca.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	struct brute {
		std::vector<uint32> parent, depth, root;

		explicit brute(std::vector<uint32> const &par) : parent(par) {
			depth.assign(par.size(), 0);
			root.resize(par.size());
			for (uint32 v = 0; v < par.size(); ++v) root[v] = v;
			for (uint32 v = 1; v < par.size(); ++v)
				if (par[v] != g::none) {
					depth[v] = depth[par[v]] + 1;
					root[v] = root[par[v]];
				}
		}

		uint32 lca(uint32 u, uint32 v) const {
			if (root[u] != root[v]) return g::none;
			while (u != v) {
				if (depth[u] < depth[v]) std::swap(u, v);
				u = parent[u];
			}
			return u;
		}
	};

	std::vector<std::vector<uint32>> adjacency(std::vector<uint32> const &parent) {
		std::vector<std::vector<uint32>> adj(parent.size());
		for (uint32 v = 0; v < parent.size(); ++v)
			if (parent[v] != g::none) {
				adj[parent[v]].push_back(v);
				adj[v].push_back(parent[v]);
			}
		return adj;
	}
}

static_assert(g::path_segment_event<decltype([](uint32, uint32, bool) {})>);
static_assert(!g::path_segment_event<decltype([](uint32, uint32) {})>);
static_assert(g::vertex_range<std::vector<uint32>>);
static_assert(g::vertex_range<std::vector<int>>);
static_assert(!g::vertex_range<std::vector<std::pair<uint32, uint32>>>);
static_assert(g::vertex_pair_range<std::vector<std::pair<uint32, uint32>>>);
static_assert(g::vertex_pair_range<std::vector<std::array<uint32, 2>>>);
static_assert(!g::vertex_pair_range<std::vector<uint32>>);
static_assert(!g::vertex_pair_range<std::vector<std::tuple<uint32, uint32, uint32>>>);

TEST_CASE("lca: random forests, every query type against a brute", "[graph]") {
	auto seed = GENERATE(take(12, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 40;
	std::vector<uint32> parent(n, g::none);
	for (uint32 v = 1; v < n; ++v)
		if (rng() % 5) parent[v] = uint32(rng() % v);
	auto const adj = adjacency(parent);
	brute const b(parent);

	g::lca anc(adj);
	g::lca from_parents(parent);
	std::vector<std::pair<uint32, uint32>> queries;
	for (int q = 0; q < 60; ++q) {
		uint32 const u = uint32(rng() % n), v = uint32(rng() % n);
		queries.push_back({u, v});
		uint32 const want = b.lca(u, v);
		REQUIRE(anc(u, v) == want);
		REQUIRE(from_parents(u, v) == want);
		REQUIRE(anc.depth(u) == b.depth[u]);
		REQUIRE(anc.is_ancestor(u, v) == (want == u));
		REQUIRE(anc.distance(u, v)
				== (want == g::none ? g::none : b.depth[u] + b.depth[v] - 2 * b.depth[want]));
		uint32 const k = uint32(rng() % 6);
		uint32 climbed = u;
		for (uint32 i = 0; i < k && climbed != g::none; ++i) climbed = parent[climbed];
		REQUIRE(anc.ancestor(u, k) == (k > b.depth[u] ? g::none : climbed));
		REQUIRE(anc.parent(u) == parent[u]);
		REQUIRE(from_parents.parent(u) == parent[u]);
		REQUIRE(anc.root(u) == b.root[u]);
		if (want == g::none) REQUIRE(anc.path_vertex(u, v, 0) == g::none);
	}
	auto const offline = g::offline_lca(adj, 0, queries);
	for (std::size_t i = 0; i < queries.size(); ++i) {
		auto const [u, v] = queries[i];
		bool const reachable = b.root[u] == b.root[0] && b.root[v] == b.root[0];
		REQUIRE(offline[i] == (reachable ? b.lca(u, v) : g::none));
	}
	// the rootless form roots every tree at its smallest vertex, as the brute
	// (parents precede children) and the rootless lca do: full agreement
	auto const forest = g::offline_lca(adj, queries);
	for (std::size_t i = 0; i < queries.size(); ++i) REQUIRE(forest[i] == b.lca(queries[i].first, queries[i].second));
}

TEST_CASE("lca: experimental engines agree with the brute on trees", "[graph]") {
	auto seed = GENERATE(take(12, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	// sizes from a single vertex up to a few hundred, so FCB's blocks and
	// direct_rmq's 64-wide words are crossed
	std::size_t const n = 1 + rng() % (rng() % 2 ? 8 : 400);
	std::vector<uint32> parent(n, g::none);
	bool const is_path = rng() % 4 == 0;
	for (uint32 v = 1; v < n; ++v) parent[v] = is_path ? v - 1 : uint32(rng() % v);
	auto const adj = adjacency(parent);
	brute const b(parent);

	ex::lca_by_binary_lifting bl(adj, 0);
	ex::lca_by_head_jumping hld(adj, 0);
	ex::lca_by_euler_sparse_table es(adj, 0);
	ex::lca_by_euler_direct_rmq er_source(adj, 0);
	ex::lca_by_euler_direct_rmq const er = std::move(er_source); // the rmq view must follow the buffer
	ex::lca_by_preorder_sparse_table ps(adj, 0);
	ex::lca_by_preorder_direct_rmq pr_source(adj, 0);
	ex::lca_by_preorder_direct_rmq const pr = std::move(pr_source);
	ex::lca_by_adjacency_preorder sv(adj, 0);
	ex::lca_by_farach_colton_bender fcb(adj, 0);
	for (int q = 0; q < 300; ++q) {
		uint32 const u = uint32(rng() % n), v = uint32(rng() % n);
		uint32 const want = b.lca(u, v);
		REQUIRE(bl(u, v) == want);
		REQUIRE(hld(u, v) == want);
		REQUIRE(es(u, v) == want);
		REQUIRE(er(u, v) == want);
		REQUIRE(ps(u, v) == want);
		REQUIRE(pr(u, v) == want);
		REQUIRE(sv(u, v) == want);
		REQUIRE(fcb(u, v) == want);
	}
	for (uint32 v = 0; v < n; ++v) {
		uint32 const want = b.depth[v];
		REQUIRE(bl.depth(v) == want);
		REQUIRE(hld.depth(v) == want);
		REQUIRE(es.depth(v) == want);
		REQUIRE(er.depth(v) == want);
		REQUIRE(ps.depth(v) == want);
		REQUIRE(pr.depth(v) == want);
		REQUIRE(sv.depth(v) == want);
		REQUIRE(fcb.depth(v) == want);
	}
}

TEST_CASE("lca: canonical", "[graph]") {
	//        0
	//       / \.
	//      1   2
	//     / \   \.
	//    3   4   5
	std::vector<uint32> parent{g::none, 0, 0, 1, 1, 2};
	g::lca anc(parent);
	REQUIRE(anc(3, 4) == 1);
	REQUIRE(anc(3, 5) == 0);
	REQUIRE(anc(3, 3) == 3);
	REQUIRE(anc(1, 3) == 1);
	REQUIRE(anc.distance(3, 5) == 4);
	REQUIRE(anc.is_ancestor(0, 5));
	REQUIRE(!anc.is_ancestor(1, 5));
	REQUIRE(anc.ancestor(3, 2) == 0);
	REQUIRE(anc.ancestor(3, 3) == g::none);
	REQUIRE(anc.path_vertex(3, 5, 0) == 3);
	REQUIRE(anc.path_vertex(3, 5, 2) == 0);
	REQUIRE(anc.path_vertex(3, 5, 4) == 5);
	REQUIRE(anc.path_vertex(3, 5, 5) == g::none);
	REQUIRE(anc.exit_time(1) - anc.entry_time(1) == 3); // subtree sizes via the preorder
	REQUIRE(anc.exit_time(0) - anc.entry_time(0) == 6);

	// two components
	std::vector<uint32> two{g::none, 0, g::none, 2};
	g::lca forest(two);
	REQUIRE(forest(1, 3) == g::none);
	REQUIRE(forest.distance(1, 3) == g::none);
	REQUIRE(forest(1, 0) == 0);
	REQUIRE(forest.root(3) == 2);

	// single vertex
	g::lca one(std::vector<uint32>{g::none});
	REQUIRE(one(0, 0) == 0);
	REQUIRE(one.ancestor(0, 0) == 0);
	REQUIRE(one.ancestor(0, 1) == g::none);
}

TEST_CASE("lca: path decomposition and path_vertex reconstruct the exact path", "[graph]") {
	auto seed = GENERATE(take(10, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 60;
	bool const is_forest = rng() % 2;
	std::vector<uint32> parent(n, g::none);
	for (uint32 v = 1; v < n; ++v)
		if (!is_forest || rng() % 5) parent[v] = uint32(rng() % v);
	brute const b(parent);
	g::lca anc(parent);
	g::lca const from_graph(adjacency(parent));

	for (int q = 0; q < 80; ++q) {
		uint32 const u = uint32(rng() % n), v = uint32(rng() % n);
		uint32 const w = b.lca(u, v);
		if (w == g::none) {
			REQUIRE(anc.path_vertex(u, v, 0) == g::none);
			continue;
		}
		std::vector<uint32> want;
		for (uint32 x = u; x != w; x = parent[x]) want.push_back(x);
		want.push_back(w);
		std::vector<uint32> tail;
		for (uint32 x = v; x != w; x = parent[x]) tail.push_back(x);
		want.insert(want.end(), tail.rbegin(), tail.rend());
		for (uint32 k = 0; k < want.size(); ++k) {
			REQUIRE(anc.path_vertex(u, v, k) == want[k]);
			REQUIRE(from_graph.path_vertex(u, v, k) == want[k]);
		}
		REQUIRE(anc.path_vertex(u, v, uint32(want.size())) == g::none);

		// segments in path order + reversed flag => the exact vertex list
		std::vector<uint32> got;
		anc.path(u, v, [&](uint32 l, uint32 r, bool reversed) {
			REQUIRE(l < r);
			if (reversed)
				for (uint32 i = r; i-- > l;) got.push_back(anc.order()[i]);
			else
				for (uint32 i = l; i < r; ++i) got.push_back(anc.order()[i]);
		});
		REQUIRE(got == want);

		// includes_top = false drops exactly the LCA
		got.clear();
		anc.path(
			u, v,
			[&](uint32 l, uint32 r, bool reversed) {
				if (reversed)
					for (uint32 i = r; i-- > l;) got.push_back(anc.order()[i]);
				else
					for (uint32 i = l; i < r; ++i) got.push_back(anc.order()[i]);
			},
			false);
		std::vector<uint32> want_edges = want;
		want_edges.erase(std::ranges::find(want_edges, w));
		REQUIRE(got == want_edges);
	}
}

TEST_CASE("lca: path/subtree composition over a lazy segment tree", "[graph]") {
	// range add over a sum monoid needs width-aware items
	struct item {
		int64 sum;
		uint32 width;
	};

	auto const op = [](item x, item y) { return item{x.sum + y.sum, x.width + y.width}; };
	auto const map = [](int64 f, item s) { return item{s.sum + f * s.width, s.width}; };
	auto const compose = [](int64 f, int64 g) { return f + g; };

	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + rng() % 40;
	std::vector<uint32> parent(n, g::none);
	for (uint32 v = 1; v < n; ++v) parent[v] = uint32(rng() % v);
	brute const b(parent);
	g::lca anc(parent);

	// value[v] lives at preorder slot entry_time(v): path add + path sum + subtree
	// sum all on ONE structure — the point of the heavy-first order
	std::vector<int64> value(n, 0);
	std::vector<item> slots(n, item{0, 1});
	atl::lazy_segment_tree seg(slots, item{0, 0}, int64(0), op, map, compose);

	auto const on_path = [&]<typename Callback>(uint32 u, uint32 v, Callback const &per_vertex)
		requires std::invocable<Callback const &, uint32>
	{
		uint32 const w = b.lca(u, v);
		for (uint32 x = u; x != w; x = parent[x]) per_vertex(x);
		per_vertex(w);
		for (uint32 x = v; x != w; x = parent[x]) per_vertex(x);
	};
	for (int step = 0; step < 300; ++step) {
		uint32 const u = uint32(rng() % n), v = uint32(rng() % n);
		if (rng() % 2) { // path += delta
			int64 const delta = int64(rng() % 20) - 10;
			anc.path(u, v, [&](uint32 l, uint32 r, bool) { seg.apply(l, r, delta); });
			on_path(u, v, [&](uint32 x) { value[x] += delta; });
		} else if (rng() % 2) { // path sum
			int64 sum = 0;
			anc.path(u, v, [&](uint32 l, uint32 r, bool) { sum += seg.fold(l, r).sum; });
			int64 want = 0;
			on_path(u, v, [&](uint32 x) { want += value[x]; });
			REQUIRE(sum == want);
		} else { // subtree sum via [entry_time, exit_time) on the same structure
			int64 want = 0;
			for (uint32 x = 0; x < n; ++x)
				if (anc.is_ancestor(u, x)) want += value[x];
			REQUIRE(seg.fold(anc.entry_time(u), anc.exit_time(u)).sum == want);
		}
	}
}

TEST_CASE("lca: a rooted build covers every component of a forest", "[graph]") {
	// regression: lca(g, root) documented forests as supported but left the
	// vertices outside root's component without a parent or a root, which
	// read out of bounds while building
	auto seed = GENERATE(take(30, random(uint64(0), ~uint64(0))));
	std::mt19937_64 rng(seed);
	uint32 const n = uint32(2 + rng() % 40);
	std::vector<std::vector<uint32>> adj(n);
	for (uint32 v = 1; v < n; ++v)
		if (rng() % 4 != 0) { // otherwise v starts a new component
			uint32 const u = uint32(rng() % v);
			adj[u].push_back(v);
			adj[v].push_back(u);
		}
	uint32 const root = uint32(rng() % n);
	// oracle: root's component hangs from root, every other from its smallest vertex
	std::vector<uint32> parent(n, g::none), depth(n, 0), top(n, g::none);
	auto const hang = [&](uint32 const from) {
		std::vector<uint32> queue{from};
		top[from] = from;
		for (std::size_t head = 0; head < queue.size(); ++head)
			for (uint32 const w : adj[queue[head]])
				if (top[w] == g::none) {
					top[w] = from;
					parent[w] = queue[head];
					depth[w] = depth[queue[head]] + 1;
					queue.push_back(w);
				}
	};
	hang(root);
	for (uint32 v = 0; v < n; ++v)
		if (top[v] == g::none) hang(v);
	g::lca const anc(adj, root);
	for (uint32 u = 0; u < n; ++u)
		for (uint32 v = 0; v < n; ++v) {
			uint32 expected = g::none;
			if (top[u] == top[v]) {
				uint32 a = u, b = v;
				while (depth[a] > depth[b]) a = parent[a];
				while (depth[b] > depth[a]) b = parent[b];
				while (a != b) a = parent[a], b = parent[b];
				expected = a;
			}
			REQUIRE(anc(u, v) == expected);
		}
}

TEST_CASE("lca asserts vertex domains and parent-forest structure", "[graph]") {
	REQUIRE(atl_test::aborts([] { g::lca tree{std::vector<uint32>{}}; (void)tree; }));
	REQUIRE(atl_test::aborts([] { g::lca tree{std::array<uint32, 2>{1, 0}}; (void)tree; }));
	REQUIRE(atl_test::aborts([] {
		g::lca const tree{std::array<uint32, 1>{g::none}};
		(void)tree.depth(1);
	}));
	REQUIRE(atl_test::aborts([] {
		g::lca const tree{std::array<uint32, 2>{g::none, g::none}};
		tree.path(0, 1, [](uint32, uint32, bool) {});
	}));
}
