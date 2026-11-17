#include <atl/ds/detail/dynamic_forest.hpp>
#include <atl/ds/top_clusters.hpp>
#include <atl/ds/top_tree.hpp>
#include <atl/experimental/ds/top_tree_variants.hpp>
#include <atl/types/algebraic.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/vertex.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
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
	namespace ex = atl::experimental;
	namespace tc = atl::top_clusters;

	constexpr uint32 none = atl::graph::none;
	constexpr ll mod = 1'000'003;

	/** @brief x -> a x + b mod p; `then(f, g)` is "f, then g", so path composites depend on the order. */
	struct affine {
		ll a = 1, b = 0;

		bool operator==(affine const &) const = default;
	};

	constexpr affine then(affine const &f, affine const &g) { return {g.a * f.a % mod, (g.a * f.b + g.b) % mod}; }

	constexpr ll eval(affine const &f, ll x) { return (f.a * x + f.b) % mod; }

	/** @brief The generic test algebra. A vertex: a map f (composed along paths, not commutative) and a weight w (summed over spines and hanging parts, changed by affine tags that do not commute either). */
	struct vw {
		affine f;
		ll w = 0;

		bool operator==(vw const &) const = default;
	};

	/** @brief A path cluster: the spine composites both ways, spine and hanging weight sums and counts. */
	struct cluster {
		affine down, up;
		ll spine_sum = 0, spine_count = 0, light_sum = 0, light_count = 0;

		bool operator==(cluster const &) const = default;
	};

	/** @brief A hanging set: weight sum and vertex count. */
	struct hanging {
		ll sum = 0, count = 0;

		bool operator==(hanging const &) const = default;
	};

	struct compress_op {
		constexpr cluster operator()(cluster const &a, cluster const &b) const {
			return {then(a.down, b.down), then(b.up, a.up), (a.spine_sum + b.spine_sum) % mod,
					a.spine_count + b.spine_count, (a.light_sum + b.light_sum) % mod, a.light_count + b.light_count};
		}
	};

	struct rake_op {
		constexpr hanging operator()(hanging const &a, hanging const &b) const { return {(a.sum + b.sum) % mod, a.count + b.count}; }
	};

	struct add_vertex_op {
		constexpr cluster operator()(hanging const &light, vw const &v) const { return {v.f, v.f, v.w, 1, light.sum, light.count}; }
	};

	struct add_edge_op {
		constexpr hanging operator()(cluster const &c) const {
			return {(c.spine_sum + c.light_sum) % mod, c.spine_count + c.light_count};
		}
	};

	struct reverse_op {
		constexpr cluster operator()(cluster const &c) const {
			return {c.up, c.down, c.spine_sum, c.spine_count, c.light_sum, c.light_count};
		}
	};

	struct mapping_op {
		constexpr vw operator()(affine const &t, vw const &v) const { return {v.f, eval(t, v.w)}; }
	};

	struct map_path_op {
		constexpr cluster operator()(affine const &spine, affine const &light, cluster c) const {
			c.spine_sum = (spine.a * c.spine_sum + spine.b * c.spine_count) % mod;
			c.light_sum = (light.a * c.light_sum + light.b * c.light_count) % mod;
			return c;
		}
	};

	struct map_point_op {
		constexpr hanging operator()(affine const &t, hanging const &p) const { return {(t.a * p.sum + t.b * p.count) % mod, p.count}; }
	};

	/** @brief "g first, then f". */
	struct compose_op {
		constexpr affine operator()(affine const &f, affine const &g) const { return then(g, f); }
	};

	/** @brief A tree type over the generic algebra: `type<Reverse, ValuePlacement>`. */
	template<template<typename...> typename Class>
	struct generic_kind {
		template<typename Reverse, typename ValuePlacement>
		using type = Class<vw, cluster, hanging, compress_op, rake_op, add_vertex_op, add_edge_op, Reverse, affine, mapping_op,
						   map_path_op, map_point_op, compose_op, ValuePlacement>;
	};

	template<typename... Args>
	using library_class = atl::top_tree<Args...>;

	/**
	 * The certifying oracle over nodes (on the edge tier: vertices, then one
	 * node per edge id): a parent array with explicit roots, make_root by
	 * reversing the root path, regions by walks. O(n) or O(n^2) per query.
	 */
	template<std::copyable Value>
	struct naive_forest {
		std::vector<uint32> par;
		std::vector<Value> val;

		naive_forest(uint32 nodes, Value fill) : par(nodes, none), val(nodes, fill) {}

		[[nodiscard]] uint32 root(uint32 v) const {
			while (par[v] != none) v = par[v];
			return v;
		}

		[[nodiscard]] bool connected(uint32 u, uint32 v) const { return root(u) == root(v); }

		[[nodiscard]] uint32 depth(uint32 v) const {
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

		[[nodiscard]] uint32 lca(uint32 u, uint32 v) const {
			if (!connected(u, v)) return none;
			std::vector<char> mark(par.size(), 0);
			for (uint32 x = u; x != none; x = par[x]) mark[x] = 1;
			for (uint32 x = v;; x = par[x])
				if (mark[x]) return x;
		}

		/** @brief The u -> v path's nodes in order. @pre connected */
		[[nodiscard]] std::vector<uint32> path(uint32 u, uint32 v) const {
			uint32 const w = lca(u, v);
			std::vector<uint32> up, down;
			for (uint32 x = u; x != w; x = par[x]) up.push_back(x);
			up.push_back(w);
			for (uint32 x = v; x != w; x = par[x]) down.push_back(x);
			up.insert(up.end(), down.rbegin(), down.rend());
			return up;
		}

		[[nodiscard]] std::vector<std::vector<uint32>> adjacency() const {
			std::vector<std::vector<uint32>> adj(par.size());
			for (uint32 v = 0; v < par.size(); ++v)
				if (par[v] != none) {
					adj[v].push_back(par[v]);
					adj[par[v]].push_back(v);
				}
			return adj;
		}

		/** @brief Nodes reachable from v without entering `blocked` (none: the whole component), BFS order. */
		[[nodiscard]] std::vector<uint32> region(uint32 v, uint32 blocked) const {
			auto const adj = adjacency();
			std::vector<char> seen(par.size(), 0);
			std::vector<uint32> order{v};
			seen[v] = 1;
			if (blocked != none) seen[blocked] = 1;
			for (std::size_t i = 0; i < order.size(); ++i)
				for (uint32 const w : adj[order[i]])
					if (!seen[w]) {
						seen[w] = 1;
						order.push_back(w);
					}
			return order;
		}

		/** @brief v's subtree under the current root. */
		[[nodiscard]] std::vector<uint32> subtree(uint32 v) const { return region(v, par[v]); }
	};

	/** @brief The cluster callables of an algebra, for the canonical-decomposition oracle. */
	template<std::copyable Path, std::copyable Point, atl::binary_operation<Path> Compress, atl::binary_operation<Point> Rake, typename AddVertex, typename AddEdge>
		requires atl::top_edge_operation<AddEdge, Path, Point>
	struct algebra {
		Point point_identity{};
		Compress compress{};
		Rake rake{};
		AddVertex add_vertex{};
		AddEdge add_edge{};
	};

	/**
	 * The algebra oracle: the user's callables run on a canonical decomposition.
	 * Oriented from `from`, every node's subtree is `add_vertex(rake of its
	 * children's add_edge(subtree), value)`; a spine s_1..s_k is the left fold of
	 * compress over each s_i with every child but s_{i+1} raked in.
	 */
	template<std::copyable Value, typename Alg>
		requires requires(Alg const &a, Value const &value) { a.add_vertex(a.point_identity, value); }
	struct canonical {
		naive_forest<Value> const &nf;
		Alg const &alg;

		using path_type = std::invoke_result_t<decltype(Alg::add_vertex) const &, decltype(Alg::point_identity) const &, Value const &>;

		/** @brief Subtree clusters and parents, oriented from `from`. */
		std::pair<std::vector<path_type>, std::vector<uint32>> oriented(uint32 from) const {
			auto const order = nf.region(from, none);
			auto const adj = nf.adjacency();
			std::vector<uint32> parent(nf.par.size(), none);
			for (uint32 const x : order)
				for (uint32 const w : adj[x])
					if (w != parent[x]) parent[w] = x;
			std::vector<path_type> sub(nf.par.size());
			for (std::size_t i = order.size(); i-- > 0;) {
				uint32 const x = order[i];
				auto point = alg.point_identity;
				for (uint32 const w : adj[x])
					if (w != parent[x]) point = alg.rake(point, alg.add_edge(sub[w]));
				sub[x] = alg.add_vertex(point, nf.val[x]);
			}
			return {std::move(sub), std::move(parent)};
		}

		path_type fold(std::vector<uint32> const &spine) const {
			auto const [sub, parent] = oriented(spine[0]);
			auto const adj = nf.adjacency();
			path_type result{};
			for (std::size_t i = 0; i < spine.size(); ++i) {
				uint32 const x = spine[i], next = i + 1 < spine.size() ? spine[i + 1] : none;
				auto point = alg.point_identity;
				for (uint32 const w : adj[x])
					if (w != parent[x] && w != next) point = alg.rake(point, alg.add_edge(sub[w]));
				path_type const own = alg.add_vertex(point, nf.val[x]);
				result = i == 0 ? own : alg.compress(result, own);
			}
			return result;
		}

		path_type subtree(uint32 from, uint32 v) const { return oriented(from).first[v]; }
	};

	/** @brief Ground truth for the generic algebra from a spine and the region holding it. */
	cluster truth(naive_forest<vw> const &nf, std::vector<uint32> const &spine, std::vector<uint32> const &nodes) {
		cluster c;
		std::vector<char> on_spine(nf.par.size(), 0);
		for (uint32 const x : spine) {
			c.down = then(c.down, nf.val[x].f);
			c.up = then(nf.val[x].f, c.up);
			c.spine_sum = (c.spine_sum + nf.val[x].w) % mod;
			++c.spine_count;
			on_spine[x] = 1;
		}
		for (uint32 const x : nodes)
			if (!on_spine[x]) {
				c.light_sum = (c.light_sum + nf.val[x].w) % mod;
				++c.light_count;
			}
		return c;
	}

	affine random_affine(std::mt19937_64 &rng) { return {ll(rng() % (mod - 1)) + 1, ll(rng() % mod)}; }

	vw random_vw(std::mt19937_64 &rng) { return {random_affine(rng), ll(rng() % mod)}; }

	/** @brief A random existing edge of the naive forest's first n nodes' trees as (child node, parent node), or none. */
	std::pair<uint32, uint32> random_edge(naive_forest<vw> const &nf, std::mt19937_64 &rng) {
		std::vector<uint32> children;
		for (uint32 x = 0; x < nf.par.size(); ++x)
			if (nf.par[x] != none) children.push_back(x);
		if (children.empty()) return {none, none};
		uint32 const c = children[rng() % children.size()];
		return {c, nf.par[c]};
	}

	/**
	 * Drive a generic-algebra tree against both oracles: every structural and
	 * query member, path and subtree tags interleaved, re-rooting in between.
	 * On the edge tier the naive forest holds n vertices then n - 1 edge nodes
	 * (node n + e for edge id e).
	 */
	template<typename Tree>
	void drive_generic(uint32 n, int ops, uint64 seed) {
		constexpr bool is_edge = std::same_as<typename Tree::value_type, vw> && requires(Tree t) { t.edge_count(); };
		std::mt19937_64 rng(seed);
		std::vector<vw> values(n);
		for (auto &v : values) v = random_vw(rng);
		uint32 const nodes = is_edge ? 2 * n - 1 : n;
		naive_forest<vw> nf(nodes, vw{});
		for (uint32 v = 0; v < n; ++v) nf.val[v] = values[v];
		Tree t = [&] {
			if constexpr (Tree::has_reverse_callable_for_test)
				return Tree(values, cluster{}, hanging{}, compress_op{}, rake_op{}, add_vertex_op{}, add_edge_op{}, reverse_op{},
							affine{}, mapping_op{}, map_path_op{}, map_point_op{}, compose_op{});
			else
				return Tree(values, cluster{}, hanging{}, compress_op{}, rake_op{}, add_vertex_op{}, add_edge_op{}, affine{},
							mapping_op{}, map_path_op{}, map_point_op{}, compose_op{});
		}();
		algebra<cluster, hanging, compress_op, rake_op, add_vertex_op, add_edge_op> const alg;
		canonical<vw, decltype(alg)> const oracle{nf, alg};
		auto const vertex_of_node = [&](uint32 x) { return x; };
		auto const check_cluster = [&](cluster const &got, std::vector<uint32> const &spine, std::vector<uint32> const &region,
									   cluster const &canon) {
			REQUIRE(got == truth(nf, spine, region));
			REQUIRE(got == canon);
		};
		// edge tier: the node between v and its neighbor vertex p
		auto const between = [&](uint32 v, uint32 p) -> uint32 {
			if constexpr (is_edge) {
				uint32 const e = t.edge(v, p);
				return e == none ? none : n + e;
			} else {
				return nf.par[v] == p || nf.par[p] == v ? p : none;
			}
		};
		for (int step = 0; step < ops; ++step) {
			uint32 const u = uint32(rng() % n), v = uint32(rng() % n);
			int const kind = int(rng() % 16);
			if (kind <= 2) { // link
				if (nf.connected(u, v)) continue;
				if constexpr (is_edge) {
					vw const w = random_vw(rng);
					uint32 const e = t.link(u, v, w);
					REQUIRE(e < n - 1);
					REQUIRE(nf.par[n + e] == none);
					nf.make_root(u);
					nf.val[n + e] = w;
					nf.par[u] = n + e;
					nf.par[n + e] = v;
				} else {
					t.link(u, v);
					nf.make_root(u);
					nf.par[u] = v;
				}
			} else if (kind == 3) { // cut
				auto [c, p] = random_edge(nf, rng);
				if (c == none) continue;
				if constexpr (is_edge) {
					uint32 const enode = c >= n ? c : p;
					uint32 const e = enode - n, below = c >= n ? none : c;
					uint32 const a = below != none ? below : [&] {
						for (uint32 x = 0; x < n; ++x)
							if (nf.par[x] == enode) return x;
						return none;
					}();
					uint32 const b = nf.par[enode];
					if (rng() % 2)
						t.cut(e);
					else
						t.cut(a, b);
					nf.par[a] = none;
					nf.par[enode] = none;
				} else {
					if (rng() % 2)
						t.cut(c, p);
					else
						t.cut(p, c);
					nf.par[c] = none;
				}
			} else if (kind == 4) { // make_root
				t.make_root(v);
				nf.make_root(v);
			} else if (kind == 5) { // cut_from_parent
				if (nf.par[v] == none) continue;
				t.cut_from_parent(v);
				if constexpr (is_edge) {
					uint32 const enode = nf.par[v];
					nf.par[v] = none;
					nf.par[enode] = none;
				} else {
					nf.par[v] = none;
				}
			} else if (kind == 6) { // fold
				if (!nf.connected(u, v)) continue;
				uint32 const saved_root = nf.root(u);
				auto const spine = nf.path(u, v);
				check_cluster(t.fold(u, v), spine, nf.region(u, none), oracle.fold(spine));
				REQUIRE(t.root(u) == saved_root);
			} else if (kind == 7) { // subtree folds
				uint32 const r = nf.root(v);
				check_cluster(t.subtree_fold(v), {v}, nf.subtree(v), oracle.subtree(r, v));
				auto const [c, p] = random_edge(nf, rng);
				if (c == none) continue;
				// an edge-side fold from either end, on the vertex endpoints
				uint32 a = c, b = p;
				if constexpr (is_edge) {
					uint32 const enode = c >= n ? c : p;
					a = none;
					for (uint32 x = 0; x < n; ++x)
						if (nf.par[x] == enode) a = x;
					b = nf.par[enode];
				}
				if (rng() % 2) std::swap(a, b);
				uint32 const blocker = between(a, b);
				check_cluster(t.subtree_fold(a, b), {a}, nf.region(a, blocker), oracle.subtree(blocker, a));
				REQUIRE(t.root(a) == nf.root(a));
			} else if (kind == 8) { // component fold
				uint32 const r = nf.root(v);
				auto const spine = nf.path(r, v);
				check_cluster(t.component_fold(v), spine, nf.region(v, none), oracle.fold(spine));
			} else if (kind == 9) { // path tag
				if (!nf.connected(u, v)) continue;
				affine const f = random_affine(rng);
				t.apply(u, v, f);
				for (uint32 const x : nf.path(u, v)) nf.val[x].w = eval(f, nf.val[x].w);
			} else if (kind == 10) { // subtree tags
				affine const f = random_affine(rng);
				if (rng() % 2) {
					t.subtree_apply(v, f);
					for (uint32 const x : nf.subtree(v)) nf.val[x].w = eval(f, nf.val[x].w);
				} else {
					auto const [c, p] = random_edge(nf, rng);
					if (c == none) continue;
					uint32 a = c, b = p;
					if constexpr (is_edge) {
						uint32 const enode = c >= n ? c : p;
						for (uint32 x = 0; x < n; ++x)
							if (nf.par[x] == enode) a = x;
						b = nf.par[enode];
					}
					if (rng() % 2) std::swap(a, b);
					t.subtree_apply(a, b, f);
					for (uint32 const x : nf.region(a, between(a, b))) nf.val[x].w = eval(f, nf.val[x].w);
				}
			} else if (kind == 11) { // component tag
				affine const f = random_affine(rng);
				t.component_apply(v, f);
				for (uint32 const x : nf.region(v, none)) nf.val[x].w = eval(f, nf.val[x].w);
			} else if (kind == 12) { // values
				vw const x = random_vw(rng);
				if constexpr (is_edge) {
					if (rng() % 2) {
						t.set_vertex(v, x);
						nf.val[v] = x;
					} else if (t.edge_count() > 0) {
						auto const [c, p] = random_edge(nf, rng);
						uint32 const enode = c >= n ? c : p;
						t.update(enode - n, [&](vw const &) { return x; });
						nf.val[enode] = x;
					}
					REQUIRE(t.at_vertex(v) == nf.val[v]);
				} else {
					if (rng() % 2)
						t.set(v, x);
					else
						t.update(v, [&](vw const &) { return x; });
					nf.val[v] = x;
					REQUIRE(t.at(v) == x);
				}
			} else { // structure queries
				REQUIRE(t.connected(u, v) == nf.connected(u, v));
				REQUIRE(t.root(v) == vertex_of_node(nf.root(v)));
				uint32 const above = nf.par[v];
				uint32 const expected_parent = above == none ? none : is_edge ? nf.par[above]
																			  : above;
				REQUIRE(t.parent(v) == expected_parent);
				REQUIRE(t.depth(v) == nf.depth(v) / (is_edge ? 2 : 1));
				uint32 const w = nf.lca(u, v);
				REQUIRE(t.lca(u, v) == w);
				REQUIRE(t.distance(u, v) == (w == none ? none : uint32(nf.path(u, v).size() - 1) / (is_edge ? 2 : 1)));
			}
		}
	}

	/** @brief Marks a generic tree type with the slot its constructors need. */
	template<typename Base, bool HasReverseCallable>
	struct generic_tree : Base {
		static constexpr bool has_reverse_callable_for_test = HasReverseCallable;
		using Base::Base;
	};

	template<typename Kind, atl::value_placement ValuePlacement>
		requires requires { typename Kind::template type<void, ValuePlacement>; typename Kind::template type<reverse_op, ValuePlacement>; }
	void run_generic_kind(uint32 n, int ops, int seeds) {
		using plain = generic_tree<typename Kind::template type<void, ValuePlacement>, false>;
		using reversing = generic_tree<typename Kind::template type<reverse_op, ValuePlacement>, true>;
		for (int seed = 0; seed < seeds; ++seed) {
			drive_generic<plain>(n, ops, uint64(seed) * 7 + 1);
			drive_generic<reversing>(n, ops, uint64(seed) * 7 + 3);
		}
	}
}

TEST_CASE("top_tree: generic algebra against the naive forest and the algebra oracle", "[top_tree]") {
	for (uint32 const n : {1u, 2u, 3u, 8u, 40u}) run_generic_kind<generic_kind<library_class>, void>(n, 1500, 4);
}

TEST_CASE("top_tree: edge tier against the subdivided naive forest", "[top_tree]") {
	for (uint32 const n : {1u, 2u, 3u, 8u, 30u}) run_generic_kind<generic_kind<library_class>, atl::edge_values_t>(n, 1500, 4);
}

TEST_CASE("top_tree contenders against both oracles", "[top_tree]") {
	run_generic_kind<generic_kind<ex::top_tree_by_interleaved_layout>, void>(30, 1500, 2);
	run_generic_kind<generic_kind<ex::top_tree_by_per_splay_push>, void>(30, 1500, 2);
	run_generic_kind<generic_kind<ex::top_tree_by_rake_append>, void>(30, 1500, 2);
	run_generic_kind<generic_kind<ex::top_tree_by_rake_append>, atl::edge_values_t>(20, 1500, 2);
	for (uint32 const n : {1u, 2u, 3u, 8u, 30u}) run_generic_kind<generic_kind<ex::top_tree_by_edge_leaves>, atl::edge_values_t>(n, 1500, 3);
	run_generic_kind<generic_kind<ex::top_tree_by_rerooting>, void>(30, 1500, 2);
	run_generic_kind<generic_kind<ex::top_tree_by_rerooting>, atl::edge_values_t>(20, 1500, 2);
	run_generic_kind<generic_kind<ex::top_tree_by_cached_vertex_cluster>, void>(30, 1500, 3);
	run_generic_kind<generic_kind<ex::top_tree_by_cached_vertex_cluster>, atl::edge_values_t>(20, 1500, 2);
}

namespace {
	/** @brief Sum / extremes ground truth over a set of vertices. */
	struct brute_extremes {
		ll sum = 0, min = (std::numeric_limits<ll>::max)(), max = std::numeric_limits<ll>::lowest();
		std::size_t count = 0;
	};

	brute_extremes over(naive_forest<ll> const &nf, std::vector<uint32> const &nodes) {
		brute_extremes b;
		for (uint32 const x : nodes) {
			b.sum += nf.val[x];
			b.min = (std::min)(b.min, nf.val[x]);
			b.max = (std::max)(b.max, nf.val[x]);
			++b.count;
		}
		return b;
	}

	template<typename Set>
	using set_algebra = algebra<typename Set::path_type, typename Set::point_type, typename Set::compress, typename Set::rake,
								typename Set::add_vertex, typename Set::add_edge>;

	/** @brief The summing sets (sum, extremes) against brute force and the algebra oracle. */
	template<typename Tree, typename Set>
	void drive_summing(uint32 n, int ops, uint64 seed) {
		constexpr bool has_extremes = tc::detail::extremes_family<typename Set::path_type>;
		std::mt19937_64 rng(seed);
		std::vector<ll> values(n);
		for (auto &x : values) x = ll(rng() % 2001) - 1000;
		naive_forest<ll> nf(n, 0);
		nf.val = values;
		Tree t(values);
		set_algebra<Set> const alg;
		canonical<ll, decltype(alg)> const oracle{nf, alg};
		for (int step = 0; step < ops; ++step) {
			uint32 const u = uint32(rng() % n), v = uint32(rng() % n);
			ll const x = ll(rng() % 2001) - 1000;
			int const kind = int(rng() % 12);
			if (kind <= 2) {
				if (nf.connected(u, v)) continue;
				t.link(u, v);
				nf.make_root(u);
				nf.par[u] = v;
			} else if (kind == 3) {
				if (nf.par[v] == none) continue;
				if (rng() % 2)
					t.cut(v, nf.par[v]);
				else
					t.cut_from_parent(v);
				nf.par[v] = none;
			} else if (kind == 4) {
				t.make_root(v);
				nf.make_root(v);
			} else if (kind == 5) {
				if (!nf.connected(u, v)) continue;
				auto const spine = nf.path(u, v);
				auto const b = over(nf, spine);
				REQUIRE(t.path_sum(u, v) == b.sum);
				if constexpr (has_extremes) {
					REQUIRE(t.path_min(u, v) == b.min);
					REQUIRE(t.path_max(u, v) == b.max);
				}
				REQUIRE(t.fold(u, v) == oracle.fold(spine));
			} else if (kind == 6) {
				auto const b = over(nf, nf.subtree(v));
				REQUIRE(t.subtree_sum(v) == b.sum);
				REQUIRE(t.subtree_size(v) == b.count);
				if constexpr (has_extremes) {
					REQUIRE(t.subtree_min(v) == b.min);
					REQUIRE(t.subtree_max(v) == b.max);
				}
				REQUIRE(t.subtree_fold(v) == oracle.subtree(nf.root(v), v));
				if (nf.par[v] != none) {
					uint32 const p = nf.par[v];
					auto const side = over(nf, nf.region(p, v));
					REQUIRE(t.subtree_sum(p, v) == side.sum);
					REQUIRE(t.subtree_size(p, v) == side.count);
					if constexpr (has_extremes) REQUIRE(t.subtree_min(p, v) == side.min);
					if constexpr (has_extremes) REQUIRE(t.subtree_max(p, v) == side.max);
					REQUIRE(t.subtree_fold(p, v) == oracle.subtree(v, p));
				}
			} else if (kind == 7) {
				auto const b = over(nf, nf.region(v, none));
				REQUIRE(t.component_sum(v) == b.sum);
				REQUIRE(t.component_size(v) == b.count);
				if constexpr (has_extremes) {
					REQUIRE(t.component_min(v) == b.min);
					REQUIRE(t.component_max(v) == b.max);
				}
			} else if (kind == 8) {
				if (!nf.connected(u, v)) continue;
				bool const is_assign = has_extremes && rng() % 2;
				if constexpr (has_extremes)
					if (is_assign) t.path_assign(u, v, x);
				if (!is_assign) t.path_add(u, v, x);
				for (uint32 const y : nf.path(u, v)) nf.val[y] = is_assign ? x : nf.val[y] + x;
			} else if (kind == 9) {
				bool const is_assign = has_extremes && rng() % 2;
				if (rng() % 2 || nf.par[v] == none) {
					if constexpr (has_extremes)
						if (is_assign) t.subtree_assign(v, x);
					if (!is_assign) t.subtree_add(v, x);
					for (uint32 const y : nf.subtree(v)) nf.val[y] = is_assign ? x : nf.val[y] + x;
				} else {
					uint32 const p = nf.par[v];
					if constexpr (has_extremes)
						if (is_assign) t.subtree_assign(p, v, x);
					if (!is_assign) t.subtree_add(p, v, x);
					for (uint32 const y : nf.region(p, v)) nf.val[y] = is_assign ? x : nf.val[y] + x;
				}
			} else if (kind == 10) {
				t.set(v, x);
				nf.val[v] = x;
			} else {
				REQUIRE(t.at(v) == nf.val[v]);
				REQUIRE(t.connected(u, v) == nf.connected(u, v));
			}
		}
	}

	/** @brief Distances from `from` to every node of its component (node weights: edges carry lengths, vertices 0). */
	template<typename Value>
	std::vector<ll> distances(naive_forest<Value> const &nf, uint32 from, auto const &length) {
		auto const adj = nf.adjacency();
		std::vector<ll> d(nf.par.size(), -1);
		std::vector<uint32> order{from};
		d[from] = length(nf.val[from]);
		for (std::size_t i = 0; i < order.size(); ++i)
			for (uint32 const w : adj[order[i]])
				if (d[w] < 0) {
					d[w] = d[order[i]] + length(nf.val[w]);
					order.push_back(w);
				}
		return d;
	}

	/** @brief The edge-tier distance sets (diameter, marked distance) against brute force. */
	template<typename Set, typename Tree = atl::detail::top_tree_of_set<Set, atl::edge_values_t>>
	void drive_distance(uint32 n, int ops, uint64 seed) {
		constexpr bool is_diameter = tc::detail::diameter_family<typename Set::path_type>;
		using value = typename Set::value_type;
		using tree = Tree;
		std::mt19937_64 rng(seed);
		naive_forest<value> nf(2 * n - 1, value{});
		if constexpr (is_diameter)
			for (uint32 v = 0; v < n; ++v) nf.val[v] = value::of_vertex(v);
		tree t(n);
		set_algebra<Set> const alg;
		canonical<value, decltype(alg)> const oracle{nf, alg};
		auto const length = [](value const &x) { return x.length; };
		for (int step = 0; step < ops; ++step) {
			uint32 const u = uint32(rng() % n), v = uint32(rng() % n);
			ll const w = ll(rng() % 100) + 1;
			int const kind = int(rng() % 9);
			if (kind <= 2) {
				if (nf.connected(u, v)) continue;
				uint32 const e = t.link(u, v, w);
				nf.make_root(u);
				nf.val[n + e] = value(w);
				nf.par[u] = n + e;
				nf.par[n + e] = v;
			} else if (kind == 3) {
				if (nf.par[v] == none) continue;
				uint32 const enode = nf.par[v];
				t.cut(enode - n);
				nf.par[v] = none;
				nf.par[enode] = none;
			} else if (kind == 4) {
				t.make_root(v);
				nf.make_root(v);
			} else if (kind == 5) {
				if (nf.par[v] == none) continue;
				uint32 const enode = nf.par[v];
				t.set(enode - n, w);
				nf.val[enode] = value(w);
			} else if (kind == 6) {
				if constexpr (!is_diameter) {
					bool const is_marked = rng() % 2;
					t.set_marked(v, is_marked);
					nf.val[v].is_marked = is_marked;
					REQUIRE(t.is_marked(v) == is_marked);
				}
			} else if (kind == 7) {
				if (!nf.connected(u, v)) continue;
				auto const spine = nf.path(u, v);
				auto const got = t.fold(u, v);
				auto const canon = oracle.fold(spine);
				if constexpr (is_diameter) {
					REQUIRE(t.path_length(u, v) == distances(nf, u, length)[v]);
					REQUIRE(got.length == canon.length);
					REQUIRE(got.diameter.length == canon.diameter.length);
					REQUIRE(got.from_top.distance == canon.from_top.distance);
					REQUIRE(got.from_bottom.distance == canon.from_bottom.distance);
				} else {
					REQUIRE(got == canon);
				}
			} else {
				auto const component = nf.region(v, none);
				auto const from_v = distances(nf, v, length);
				if constexpr (is_diameter) {
					ll best = 0, farthest_distance = 0;
					for (uint32 const a : component) {
						if (a >= n) continue;
						farthest_distance = (std::max)(farthest_distance, from_v[a]);
						auto const d = distances(nf, a, length);
						for (uint32 const b : component)
							if (b < n) best = (std::max)(best, d[b]);
					}
					REQUIRE(t.diameter(v) == best);
					auto const [a, b] = t.diameter_endpoints(v);
					REQUIRE((a < n && b < n && nf.connected(a, v) && nf.connected(b, v)));
					REQUIRE(distances(nf, a, length)[b] == best);
					REQUIRE(t.eccentricity(v) == farthest_distance);
					auto const f = t.farthest(v);
					REQUIRE(f.distance == farthest_distance);
					REQUIRE((f.vertex < n && from_v[f.vertex] == farthest_distance));
				} else {
					ll sum = 0;
					std::size_t count = 0;
					for (uint32 const a : component)
						if (a < n && nf.val[a].is_marked) {
							sum += from_v[a];
							++count;
						}
					REQUIRE(t.marked_distance_sum(v) == sum);
					REQUIRE(t.marked_count(v) == count);
				}
			}
		}
	}
}

TEST_CASE("top_clusters::sum against brute force and the algebra oracle", "[top_tree]") {
	for (int seed = 0; seed < 6; ++seed)
		for (uint32 const n : {1u, 5u, 40u}) drive_summing<atl::sum_top_tree<ll>, tc::sum<ll>>(n, 1500, uint64(seed) * 11 + n);
}

TEST_CASE("top_clusters::extremes against brute force (mixed path/subtree operations)", "[top_tree]") {
	for (int seed = 0; seed < 6; ++seed)
		for (uint32 const n : {1u, 5u, 40u})
			drive_summing<atl::extremes_top_tree<ll>, tc::extremes<ll>>(n, 1500, uint64(seed) * 13 + n);
}

TEST_CASE("top_clusters::diameter against all-pairs distances", "[top_tree]") {
	for (int seed = 0; seed < 6; ++seed)
		for (uint32 const n : {1u, 4u, 25u}) drive_distance<tc::diameter<ll>>(n, 1200, uint64(seed) * 17 + n);
}

TEST_CASE("top_clusters::marked_distance against brute-force distance sums", "[top_tree]") {
	for (int seed = 0; seed < 6; ++seed)
		for (uint32 const n : {1u, 4u, 25u}) drive_distance<tc::marked_distance<ll>>(n, 1200, uint64(seed) * 19 + n);
}

namespace {
	/** @brief (edge leaves) over a ready-made edge-tier set. */
	template<typename Set>
	using edge_leaf_tree =
		ex::top_tree_by_edge_leaves<typename Set::value_type, typename Set::path_type, typename Set::point_type, typename Set::compress,
									typename Set::rake, typename Set::add_vertex, typename Set::add_edge, typename Set::reverse,
									typename Set::tag_type, typename Set::mapping, typename Set::map_path, typename Set::map_point,
									typename Set::compose, atl::edge_values_t>;
}

TEST_CASE("top_tree (edge leaves): distance sets against brute force", "[top_tree]") {
	for (int seed = 0; seed < 4; ++seed)
		for (uint32 const n : {1u, 4u, 25u}) {
			drive_distance<tc::diameter<ll>, edge_leaf_tree<tc::diameter<ll>>>(n, 1200, uint64(seed) * 23 + n);
			drive_distance<tc::marked_distance<ll>, edge_leaf_tree<tc::marked_distance<ll>>>(n, 1200, uint64(seed) * 29 + n);
		}
}

TEST_CASE("top_tree: assign_parents matches linking, both tiers", "[top_tree]") {
	std::mt19937_64 rng(5);
	for (uint32 const n : {1u, 2u, 7u, 60u}) {
		std::vector<uint32> parents(n, none);
		std::vector<ll> weights(n, 0);
		for (uint32 v = 1; v < n; ++v)
			if (rng() % 5) {
				parents[v] = uint32(rng() % v);
				weights[v] = ll(rng() % 50) + 1;
			}
		std::vector<ll> values(n);
		for (auto &x : values) x = ll(rng() % 100);
		atl::extremes_top_tree<ll> built(values), linked(values);
		built.assign_parents(parents);
		atl::diameter_top_tree<ll> dbuilt(n), dlinked(n);
		dbuilt.assign_parents(parents, weights);
		edge_leaf_tree<tc::diameter<ll>> leaves(n);
		leaves.assign_parents(parents, weights);
		for (uint32 v = 0; v < n; ++v)
			if (parents[v] != none) {
				linked.link(v, parents[v]);
				dlinked.link(v, parents[v], weights[v]);
			}
		for (uint32 v = 0; v < n; ++v) {
			REQUIRE(built.root(v) == linked.root(v));
			REQUIRE(built.parent(v) == linked.parent(v));
			REQUIRE(built.subtree_sum(v) == linked.subtree_sum(v));
			REQUIRE(built.subtree_min(v) == linked.subtree_min(v));
			REQUIRE(built.subtree_size(v) == linked.subtree_size(v));
			REQUIRE(dbuilt.diameter(v) == dlinked.diameter(v));
			REQUIRE(dbuilt.eccentricity(v) == dlinked.eccentricity(v));
			REQUIRE(dbuilt.parent(v) == dlinked.parent(v));
			REQUIRE(leaves.diameter(v) == dlinked.diameter(v));
			REQUIRE(leaves.eccentricity(v) == dlinked.eccentricity(v));
			REQUIRE(leaves.parent(v) == dlinked.parent(v));
			REQUIRE(leaves.depth(v) == dlinked.depth(v));
			if (parents[v] != none) REQUIRE(dbuilt.edge(v, parents[v]) != none);
		}
		REQUIRE(dbuilt.edge_count() == dlinked.edge_count());
	}
}

TEST_CASE("top_tree: deduction, lambdas, commutative mode, small cases", "[top_tree]") {
	auto compress = [](ll a, ll b) { return a + b; };
	auto rake = [](ll a, ll b) { return a + b; };
	auto add_vertex = [](ll light, ll v) { return light + v; };
	auto add_edge = [](ll child) { return child; };
	atl::top_tree t(3, ll(1), ll(0), ll(0), compress, rake, add_vertex, add_edge);
	static_assert(std::same_as<decltype(t)::path_type, ll>);
	t.link(0, 1);
	t.link(2, 1);
	CHECK(t.component_fold(0) == 3);
	CHECK(t.subtree_fold(1) == 3);
	CHECK(t.fold(0, 2) == 3);

	atl::top_tree c(atl::commutative_fold, std::vector<ll>{1, 2, 3}, ll(0), ll(0), compress, rake, add_vertex, add_edge);
	static_assert(std::same_as<decltype(c), atl::top_tree<ll, ll, ll, decltype(compress), decltype(rake), decltype(add_vertex),
														  decltype(add_edge), atl::commutative_fold_t>>);
	c.link(0, 1);
	c.link(1, 2);
	CHECK(c.subtree_fold(0) == 1);
	CHECK(c.component_fold(2) == 6);

	atl::top_tree e(atl::edge_values, 3, ll(0), ll(0), ll(0), compress, rake, add_vertex, add_edge);
	static_assert(std::same_as<decltype(e), atl::top_tree<ll, ll, ll, decltype(compress), decltype(rake), decltype(add_vertex),
														  decltype(add_edge), void, void, void, void, void, void,
														  atl::edge_values_t>>);
	uint32 const e01 = e.link(0, 1, 5), e12 = e.link(1, 2, 7);
	CHECK(e.fold(0, 2) == 12);
	CHECK(e.edge(1, 0) == e01);
	CHECK(e.edge_count() == 2);
	e.cut(e01);
	CHECK(e.edge(0, 1) == none);
	CHECK(e.component_fold(2) == 7);
	CHECK(e.at(e12) == 7);

	atl::sum_top_tree<ll> empty;
	CHECK(empty.empty());
	CHECK(empty.size() == 0);
	atl::sum_top_tree<ll> one(1);
	CHECK(one.path_sum(0, 0) == 0);
	one.path_add(0, 0, 4);
	CHECK(one.subtree_sum(0) == 4);
	CHECK(one.root(0) == 0);
	CHECK(one.parent(0) == none);
	CHECK(one.lca(0, 0) == 0);
	CHECK(one.distance(0, 0) == 0);
	atl::diameter_top_tree<ll> lone(1);
	CHECK(lone.diameter(0) == 0);
	CHECK(lone.diameter_endpoints(0) == std::pair<uint32, uint32>{0, 0});
	CHECK(lone.farthest(0).vertex == 0);
}

TEST_CASE("top_tree: a 1e6-vertex path and star (no recursion, deep compress and huge rake trees)", "[top_tree]") {
	constexpr uint32 n = 1'000'000;
	{
		atl::sum_top_tree<ll> t(n);
		std::vector<uint32> parents(n);
		for (uint32 v = 0; v < n; ++v) parents[v] = v == 0 ? none : v - 1;
		t.assign_parents(parents);
		t.path_add(0, n - 1, 1);
		CHECK(t.path_sum(n / 3, n - 1) == ll(n - n / 3));
		t.subtree_add(n / 2, 2);
		CHECK(t.subtree_sum(n / 2) == 3 * ll(n - n / 2));
		CHECK(t.depth(n - 1) == n - 1);
		t.make_root(n - 1);
		CHECK(t.subtree_size(n / 2) == n / 2 + 1);
		CHECK(t.lca(0, n / 2) == n / 2);
	}
	{
		atl::sum_top_tree<ll> t(n);
		for (uint32 v = 0; v + 1 < n; ++v) t.link(v + 1, v); // a path built by links
		CHECK(t.component_size(0) == n);
		CHECK(t.distance(0, n - 1) == n - 1);
	}
	{
		std::vector<uint32> parents(n);
		std::vector<ll> weights(n, 1);
		for (uint32 v = 0; v < n; ++v) parents[v] = v == 0 ? none : v - 1;
		atl::diameter_top_tree<ll> t(n);
		edge_leaf_tree<tc::diameter<ll>> leaves(n);
		t.assign_parents(parents, weights);
		leaves.assign_parents(parents, weights);
		CHECK(t.diameter(n / 2) == ll(n - 1));
		CHECK(leaves.diameter(n / 2) == ll(n - 1));
		leaves.set(leaves.edge(n - 1, n - 2), 5);
		CHECK(leaves.eccentricity(0) == ll(n - 2) + 5);
		CHECK(leaves.farthest(n - 1).vertex == 0);
		leaves.cut(n / 2, n / 2 + 1);
		CHECK(leaves.diameter(n - 1) == ll(n - n / 2 - 3) + 5);
	}
	{
		atl::extremes_top_tree<ll> t(n);
		std::vector<uint32> parents(n, 0);
		parents[0] = none;
		t.assign_parents(parents);
		for (uint32 v = 1; v < n; v += 99'991) t.subtree_add(v, ll(v));
		t.path_assign(5, 7, -3);
		CHECK(t.component_min(0) == -3);
		CHECK(t.subtree_size(0) == n);
		t.make_root(n - 1);
		CHECK(t.subtree_max(0, n - 1) == 99'991 * 10 + 1);
		t.cut(0, 5);
		CHECK(t.component_size(5) == 1);
	}
}

TEST_CASE("top_tree: preconditions abort", "[top_tree]") {
	auto const make = [] {
		atl::sum_top_tree<ll> t(4);
		t.link(0, 1);
		t.link(1, 2);
		return t;
	};
	CHECK(atl_test::aborts([&] { (void)make().connected(4, 0); }));
	CHECK(atl_test::aborts([&] { make().link(0, 2); }));
	CHECK(atl_test::aborts([&] { make().cut(0, 2); }));
	CHECK(atl_test::aborts([&] { make().cut_from_parent(make().root(0)); }));
	CHECK(atl_test::aborts([&] {
		auto t = make();
		t.cut_from_parent(t.root(0));
	}));
	CHECK(atl_test::aborts([&] { (void)make().path_sum(0, 3); }));
	CHECK(atl_test::aborts([&] { make().path_add(3, 0, 1); }));
	CHECK(atl_test::aborts([&] { (void)make().subtree_sum(0, 2); }));
	CHECK(atl_test::aborts([&] { make().subtree_add(3, 0, 1); }));
	CHECK(atl_test::aborts([&] {
		auto t = make();
		t.assign_parents(std::vector<uint32>{none, none, none, none});
	}));
	CHECK(atl_test::aborts([&] {
		atl::sum_top_tree<ll> t(3);
		t.assign_parents(std::vector<uint32>{1, 2, 0});
	}));
	CHECK(atl_test::aborts([&] {
		atl::sum_top_tree<ll> t(3);
		t.assign_parents(std::vector<uint32>{none, 0});
	}));
	CHECK(atl_test::aborts([] {
		atl::diameter_top_tree<ll> t(3);
		uint32 const e = t.link(0, 1, 2);
		t.cut(e);
		t.cut(e);
	}));
	CHECK(atl_test::aborts([] {
		atl::diameter_top_tree<ll> t(3);
		t.link(0, 1, 2);
		t.link(1, 0, 2);
	}));
	CHECK(atl_test::aborts([] {
		atl::diameter_top_tree<ll> t(3);
		t.link(0, 1, 2);
		t.cut(0, 2);
	}));
	CHECK(atl_test::aborts([] {
		atl::diameter_top_tree<ll> t(3);
		(void)t.diameter(3);
	}));
	CHECK(atl_test::aborts([] {
		edge_leaf_tree<tc::diameter<ll>> t(3);
		uint32 const e = t.link(0, 1, 2);
		t.cut(e);
		t.set(e, 4);
	}));
	CHECK(atl_test::aborts([] {
		edge_leaf_tree<tc::diameter<ll>> t(3);
		t.link(0, 1, 2);
		t.cut_from_parent(t.root(0));
	}));
	CHECK(atl_test::aborts([] {
		edge_leaf_tree<tc::diameter<ll>> t(3);
		t.link(0, 1, 2);
		t.link(2, 1, 2);
		(void)t.path_length(0, 2);
		t.link(0, 2, 1);
	}));
}
