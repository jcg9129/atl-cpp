#include <atl/ds/static_vector.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/traversal.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

namespace g = atl::graph;
using atl::uint32;
using atl::uint64;

namespace {
	constexpr atl::uint32 none = (std::numeric_limits<atl::uint32>::max)();

	/** @brief Reference BFS distances over a plain adjacency structure. */
	std::vector<atl::uint32> brute_dist(std::vector<std::vector<g::vertex_id>> const &adj, g::vertex_id s) {
		std::vector<atl::uint32> dist(adj.size(), none);
		std::queue<g::vertex_id> q;
		dist[s] = 0;
		q.push(s);
		while (!q.empty()) {
			auto const u = q.front();
			q.pop();
			for (auto const v : adj[u])
				if (dist[v] == none) {
					dist[v] = dist[u] + 1;
					q.push(v);
				}
		}
		return dist;
	}
}

TEST_CASE("bfs view: visits once, in depth order, with true distances", "[graph]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 60);
	std::size_t const m = std::size_t(rng() % (3 * n));
	std::vector<std::vector<g::vertex_id>> adj(n);
	std::vector<g::arc<>> arcs;
	for (std::size_t i = 0; i < m; ++i) {
		auto const a = g::vertex_id(rng() % n), b = g::vertex_id(rng() % n);
		adj[a].push_back(b);
		arcs.push_back({a, b});
	}
	auto const s = g::vertex_id(rng() % n);
	auto const want = brute_dist(adj, s);

	auto check = [&]<g::adjacency_graph Graph>(Graph const &graph) {
		std::vector<atl::uint32> seen_at(n, none);
		atl::uint32 last_depth = 0;
		for (auto const &[v, parent, depth] : g::bfs(graph, s)) {
			REQUIRE(seen_at[v] == none);
			seen_at[v] = depth;
			REQUIRE(depth >= last_depth);
			last_depth = depth;
			REQUIRE(want[v] == depth);
			if (v != s) {
				REQUIRE(seen_at[parent] != none);
				REQUIRE(seen_at[parent] + 1 == depth);
			}
		}
		for (std::size_t v = 0; v < n; ++v) REQUIRE(seen_at[v] == want[v]);
	};
	check(adj);
	check(g::csr(n, arcs));
}

TEST_CASE("dfs view: preorder over every reachable vertex", "[graph]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 60);
	std::size_t const m = std::size_t(rng() % (3 * n));
	std::vector<std::vector<g::vertex_id>> adj(n);
	for (std::size_t i = 0; i < m; ++i) adj[std::size_t(rng() % n)].push_back(g::vertex_id(rng() % n));
	auto const s = g::vertex_id(rng() % n);
	auto const reach = brute_dist(adj, s);

	std::vector<atl::uint32> depth_of(n, none);
	for (auto const &[v, parent, depth] : g::dfs(adj, s)) {
		REQUIRE(depth_of[v] == none);
		REQUIRE(reach[v] != none);
		if (v != s) {
			REQUIRE(depth_of[parent] != none);
			REQUIRE(depth_of[parent] + 1 == depth);
		}
		depth_of[v] = depth;
	}
	for (std::size_t v = 0; v < n; ++v) REQUIRE((depth_of[v] != none) == (reach[v] != none));
}

TEST_CASE("traversal over an implicit grid graph", "[graph]") {
	// 4-neighbor grid, never materialized; BFS depths must be Manhattan-like
	constexpr int W = 7, H = 5;
	auto grid = g::implicit_graph(W * H, [](g::vertex_id u) {
		atl::static_vector<g::vertex_id, 4> out;
		int const x = int(u) % W, y = int(u) / W;
		if (x > 0) out.push_back(u - 1);
		if (x + 1 < W) out.push_back(u + 1);
		if (y > 0) out.push_back(u - W);
		if (y + 1 < H) out.push_back(u + W);
		return out;
	});

	std::size_t visited = 0;
	for (auto const &[v, parent, depth] : g::bfs(grid, 0)) {
		REQUIRE(depth == atl::uint32(int(v) % W + int(v) / W)); // Manhattan distance from (0,0)
		++visited;
	}
	REQUIRE(visited == W * H);

	std::size_t dfs_visited = 0;
	for ([[maybe_unused]] auto const &v : g::dfs(grid, 0)) ++dfs_visited;
	REQUIRE(dfs_visited == W * H);
}

TEST_CASE("bfs view composes with range adaptors", "[graph]") {
	std::vector<std::vector<g::vertex_id>> adj{{1, 2}, {3}, {3}, {}};
	auto got = g::bfs(adj, 0) | std::views::transform([](g::visit const &v) { return v.depth; })
			 | std::ranges::to<std::vector>();
	REQUIRE(got == std::vector<atl::uint32>{0, 1, 1, 2});
}

namespace traversal_fixture {
	/** @brief Random-access edge range whose dereferences are counted. */
	struct counted_edges {
		g::vertex_id const *data;
		std::size_t count;
		std::size_t *reads;

		struct iterator {
			using value_type = g::vertex_id;
			using difference_type = std::ptrdiff_t;

			g::vertex_id const *p = nullptr;
			std::size_t *reads = nullptr;

			g::vertex_id operator*() const {
				++*reads;
				return *p;
			}

			g::vertex_id operator[](difference_type k) const {
				++*reads;
				return p[k];
			}

			iterator &operator++() { return ++p, *this; }

			iterator operator++(int) { return {p++, reads}; }

			iterator &operator--() { return --p, *this; }

			iterator operator--(int) { return {p--, reads}; }

			iterator &operator+=(difference_type k) { return p += k, *this; }

			iterator &operator-=(difference_type k) { return p -= k, *this; }

			friend iterator operator+(iterator a, difference_type k) { return {a.p + k, a.reads}; }

			friend iterator operator+(difference_type k, iterator a) { return {a.p + k, a.reads}; }

			friend iterator operator-(iterator a, difference_type k) { return {a.p - k, a.reads}; }

			friend difference_type operator-(iterator a, iterator b) { return a.p - b.p; }

			friend bool operator==(iterator a, iterator b) { return a.p == b.p; }

			friend auto operator<=>(iterator a, iterator b) { return a.p <=> b.p; }
		};

		iterator begin() const { return {data, reads}; }

		iterator end() const { return {data + count, reads}; }

		std::size_t size() const { return count; }
	};

	static_assert(std::ranges::random_access_range<counted_edges>);

	/** @brief A star: vertex 0 points at every leaf; edge reads are observable. */
	struct counted_star {
		std::vector<g::vertex_id> leaves;
		mutable std::size_t reads = 0;

		std::size_t vertex_count() const { return leaves.size() + 1; }

		counted_edges out_edges(g::vertex_id u) const {
			return u == 0 ? counted_edges{leaves.data(), leaves.size(), &reads}
						  : counted_edges{nullptr, 0, &reads};
		}
	};
}

using traversal_fixture::counted_star;

TEST_CASE("traversal views examine only the edges they need", "[graph]") {
	counted_star star;
	for (g::vertex_id v = 1; v <= 100'000; ++v) star.leaves.push_back(v);

	std::size_t taken = 0;
	for (auto const &v : g::bfs(star, 0) | std::views::take(3)) ++taken, (void)v;
	REQUIRE(taken == 3);
	REQUIRE(star.reads <= 8); // discovering 2 leaves must not scan the other 99'998

	star.reads = 0;
	taken = 0;
	for (auto const &v : g::dfs(star, 0) | std::views::take(3)) ++taken, (void)v;
	REQUIRE(taken == 3);
	REQUIRE(star.reads <= 8);

	star.reads = 0; // full traversal still reads every edge exactly once
	std::size_t visited = 0;
	for ([[maybe_unused]] auto const &v : g::bfs(star, 0)) ++visited;
	REQUIRE(visited == 100'001);
	REQUIRE(star.reads == 100'000);
}

namespace {
	/** @brief Reference recursive DFS computing preorder intervals and parents. */
	struct brute_dfs {
		std::vector<std::vector<g::vertex_id>> const &adj;
		std::vector<atl::uint32> pos, end, parent;
		atl::uint32 timer = 0;

		explicit brute_dfs(std::vector<std::vector<g::vertex_id>> const &a)
			: adj(a), pos(a.size(), g::none), end(a.size(), g::none), parent(a.size(), g::none) {
			for (g::vertex_id v = 0; v < adj.size(); ++v)
				if (pos[v] == g::none) {
					parent[v] = v;
					run(v);
				}
		}

		void run(g::vertex_id v) {
			pos[v] = timer++;
			for (auto const w : adj[v])
				if (pos[w] == g::none) {
					parent[w] = v;
					run(w);
				}
			end[v] = timer;
		}
	};
}

TEST_CASE("dfs_visit: enter/exit events reproduce recursive preorder intervals", "[graph]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 60);
	std::vector<std::vector<g::vertex_id>> adj(n);
	for (std::size_t i = 0, m = std::size_t(rng() % (3 * n)); i < m; ++i)
		adj[std::size_t(rng() % n)].push_back(g::vertex_id(rng() % n));
	brute_dfs const want(adj);

	std::vector<atl::uint32> pos(n, g::none), end(n, g::none), parent(n, g::none);
	atl::uint32 timer = 0;
	std::size_t seen_edges = 0;
	g::dfs_visit(
		adj,
		[&](g::vertex_id v, g::vertex_id p, atl::uint32 k, atl::uint32) {
			parent[v] = p;
			pos[v] = timer++;
			if (p != v) REQUIRE(adj[p][k] == v);
		},
		[&](g::vertex_id u, atl::uint32 k, g::vertex_id v) {
			REQUIRE(adj[u][k] == v);
			++seen_edges;
		},
		[&](g::vertex_id v, g::vertex_id) { end[v] = timer; });

	REQUIRE(pos == want.pos);
	REQUIRE(end == want.end);
	REQUIRE(parent == want.parent);
	std::size_t total_edges = 0, roots = 0;
	for (auto const &es : adj) total_edges += es.size();
	for (g::vertex_id v = 0; v < n; ++v) roots += parent[v] == v;
	// every edge fires exactly one event: tree edges enter, the rest seen_edge
	REQUIRE(seen_edges == total_edges - (n - roots));
}

TEST_CASE("dfs_visit: subtree sums via a single exit lambda", "[graph]") {
	// star + chain: subtree sizes are easy to state
	std::vector<std::vector<g::vertex_id>> adj{{1, 2}, {3, 4}, {}, {}, {}};
	std::vector<int> size(5, 1);
	g::dfs_visit(adj, 0, g::skip, g::skip, [&](g::vertex_id v, g::vertex_id p) {
		if (p != v) size[p] += size[v];
	});
	REQUIRE(size == std::vector<int>{5, 3, 1, 1, 1});
}

TEST_CASE("events adapter: roles inferred from arity, any order, gaps skipped", "[graph]") {
	std::vector<std::vector<g::vertex_id>> adj{{1}, {2}, {}};
	std::vector<g::vertex_id> pre, post;
	auto enter = [&](g::vertex_id v, g::vertex_id, atl::uint32, atl::uint32) { pre.push_back(v); };
	auto exit = [&](g::vertex_id v, g::vertex_id) { post.push_back(v); };

	g::dfs_visit(adj, 0, g::events(exit, enter));
	REQUIRE(pre == std::vector<g::vertex_id>{0, 1, 2});
	REQUIRE(post == std::vector<g::vertex_id>{2, 1, 0});

	pre.clear();
	g::dfs_visit(adj, 0, g::events(enter));
	REQUIRE(pre.size() == 3);

	// compile-time rejections: no matching role, ambiguous catch-alls, and
	// duplicate roles (checked through a template so failure is SFINAE, not
	// a hard error)
	auto constexpr accepts = []<typename... Fs>(Fs... fs) { return requires { g::events(fs...); }; };
	static_assert(!accepts([](int *) {}));
	static_assert(!accepts([](auto &&...) {}));
	static_assert(!accepts([](g::vertex_id, g::vertex_id) {}, [](g::vertex_id, g::vertex_id) {}));
	static_assert(accepts([](g::vertex_id, g::vertex_id) {}));
}

TEST_CASE("bfs view: multi-source shares one frontier", "[graph]") {
	// path 0-1-2-3-4-5, sources {0, 5}: depths are distance to nearest source
	std::vector<std::vector<g::vertex_id>> adj(6);
	for (g::vertex_id v = 0; v + 1 < 6; ++v) {
		adj[v].push_back(v + 1);
		adj[v + 1].push_back(v);
	}
	std::vector<atl::uint32> depth(6, g::none), root(6, g::none);
	for (auto const &[v, parent, d] : g::bfs(adj, std::vector<g::vertex_id>{0, 5})) {
		depth[v] = d;
		root[v] = parent == v ? v : root[parent];
	}
	REQUIRE(depth == std::vector<atl::uint32>{0, 1, 2, 2, 1, 0});
	REQUIRE(root[1] == 0);
	REQUIRE(root[4] == 5);
}

template<>
struct atl::graph::graph_traits<std::pair<atl::graph::vertex_id const, int>> {
	static vertex_id target(std::pair<vertex_id const, int> const &e) { return e.first; }

	static int weight(std::pair<vertex_id const, int> const &e) { return e.second; }
};

TEST_CASE("set- and map-backed adjacency traverse via stable iterators", "[graph]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 50);
	std::vector<std::set<g::vertex_id>> adj(n);
	for (std::size_t i = 0, m = std::size_t(rng() % (3 * n)); i < m; ++i)
		adj[std::size_t(rng() % n)].insert(g::vertex_id(rng() % n));

	// Preserve the set's iteration order so the traversal orders are comparable.
	std::vector<std::vector<g::vertex_id>> flat(n);
	for (std::size_t v = 0; v < n; ++v) flat[v].assign(adj[v].begin(), adj[v].end());

	std::vector<g::vertex_id> pre_set, pre_flat;
	std::size_t seen_set = 0, seen_flat = 0;
	g::dfs_visit(adj, [&](g::vertex_id v, g::vertex_id, atl::uint32, atl::uint32) { pre_set.push_back(v); }, [&](g::vertex_id, atl::uint32, g::vertex_id) { ++seen_set; }, g::skip);
	g::dfs_visit(flat, [&](g::vertex_id v, g::vertex_id, atl::uint32, atl::uint32) { pre_flat.push_back(v); }, [&](g::vertex_id, atl::uint32, g::vertex_id) { ++seen_flat; }, g::skip);
	REQUIRE(pre_set == pre_flat);
	REQUIRE(seen_set == seen_flat);

	auto const want = brute_dist(flat, 0);
	std::vector<atl::uint32> got(n, none);
	for (auto const &[v, parent, depth] : g::bfs(adj, 0)) got[v] = depth;
	REQUIRE(got == want);
}

TEST_CASE("map-backed weighted adjacency via graph_traits", "[graph]") {
	std::vector<std::map<g::vertex_id, int>> adj(4);
	adj[0][1] = 10;
	adj[0][2] = 20;
	adj[1][3] = 30;
	static_assert(g::weighted_adjacency_graph<decltype(adj)>);

	std::size_t visited = 0;
	int weight_sum = 0;
	for (auto const &[v, parent, depth] : g::dfs(adj, 0)) ++visited, (void)v;
	REQUIRE(visited == 4);
	g::dfs_visit(adj, 0, [&](g::vertex_id v, g::vertex_id p, atl::uint32 k, atl::uint32) {
					 if (p != v) weight_sum += g::weight(*std::next(adj[p].begin(), k)); }, g::skip, g::skip);
	REQUIRE(weight_sum == 60); // every tree edge's weight, resolved via its k
}

TEST_CASE("epoch markers: the small-side component race", "[graph]") {
	// two components: A = tiny cycle (5), B = long path (10'000). Race BFS
	// from one vertex of each in lockstep; the tiny side exhausts first after
	// O(min) steps — with reused markers, no O(n) work per race.
	std::size_t const nA = 5, nB = 10'000, n = nA + nB;
	std::vector<std::vector<g::vertex_id>> adj(n);
	for (g::vertex_id v = 0; v < nA; ++v) {
		adj[v].push_back((v + 1) % nA);
		adj[(v + 1) % nA].push_back(v);
	}
	for (g::vertex_id v = nA; v + 1 < n; ++v) {
		adj[v].push_back(v + 1);
		adj[v + 1].push_back(v);
	}

	g::epoch_marker marks_a(n), marks_b(n);
	for (int race = 0; race < 1000; ++race) {
		marks_a.next_run();
		marks_b.next_run();
		auto va = g::bfs(adj, 0, marks_a);
		auto vb = g::bfs(adj, g::vertex_id(nA), marks_b);
		auto a = va.begin();
		auto b = vb.begin();
		std::size_t steps = 0;
		for (;;) { // lockstep: one step each; first exhausted side is smaller
			if (a == std::default_sentinel) break;
			if (b == std::default_sentinel) break;
			++a;
			++b;
			++steps;
		}
		REQUIRE(a == std::default_sentinel); // A (size 5) always loses the race
		REQUIRE(b != std::default_sentinel);
		REQUIRE(steps <= nA + 1); // O(min component), not O(n)
	}
}

TEST_CASE("hash marker: implicit graph far too large for dense flags", "[graph]") {
	// 2^32 - 1 states, the whole vertex_id range; component around the source
	// is 6 vertices. dense bit flags would need 512 MiB — hash marks need O(6).
	auto huge = g::implicit_graph((std::numeric_limits<g::vertex_id>::max)(), [](g::vertex_id u) {
		atl::static_vector<g::vertex_id, 2> out;
		if (u % 8 != 5) out.push_back(u + 1);
		if (u % 8 != 0) out.push_back(u - 1);
		return out;
	});
	g::hash_marker marks;
	std::size_t visited = 0;
	for (auto const &[v, parent, depth] : g::bfs(huge, 8, marks)) {
		REQUIRE(v >= 8);
		REQUIRE(v <= 13);
		++visited;
	}
	REQUIRE(visited == 6);

	marks.clear();
	std::size_t dfs_visited = 0;
	g::dfs_visit(huge, 8, marks, [&](g::vertex_id, g::vertex_id, atl::uint32, atl::uint32) { ++dfs_visited; }, g::skip, g::skip);
	REQUIRE(dfs_visited == 6);
}

TEST_CASE("bfs: an out-of-range source is asserted, whatever the marker", "[graph]") {
	std::vector<std::vector<g::vertex_id>> const adj{{1}, {0}};
	g::hash_marker marks;
	REQUIRE(atl_test::aborts([&] { (void)g::bfs(adj, 2, marks).begin(); }));
	REQUIRE(atl_test::aborts([&] { (void)g::bfs(adj, std::vector<g::vertex_id>{0, 5}).begin(); }));
}

TEST_CASE("shared marker across runs: component labelling", "[graph]") {
	// three components; one dense marker shared across dfs_visit calls makes
	// later calls skip what earlier ones claimed
	std::vector<std::vector<g::vertex_id>> adj(7);
	adj[0] = {1};
	adj[1] = {0};
	adj[2] = {3, 4};
	adj[3] = {2};
	adj[4] = {2};
	// 5, 6 isolated except 5 -> 6
	adj[5] = {6};
	g::dense_marker marks(7);
	std::vector<int> component(7, -1);
	int c = 0;
	for (g::vertex_id s = 0; s < 7; ++s) {
		bool has_root = false;
		g::dfs_visit(adj, s, marks, [&](g::vertex_id v, g::vertex_id, atl::uint32, atl::uint32) {
						 component[v] = c;
						 has_root = true; }, g::skip, g::skip);
		if (has_root) ++c; // sources already claimed produce no events at all
	}
	REQUIRE(c == 3);
	REQUIRE(component == std::vector<int>{0, 0, 1, 1, 1, 2, 2});
}

TEST_CASE("bfs_visit: eager internal iteration matches the view", "[graph]") {
	auto seed = GENERATE(take(4, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 60);
	std::vector<std::vector<g::vertex_id>> adj(n);
	for (std::size_t i = 0, m = std::size_t(rng() % (3 * n)); i < m; ++i)
		adj[std::size_t(rng() % n)].push_back(g::vertex_id(rng() % n));
	auto const s = g::vertex_id(rng() % n);

	std::vector<g::visit> from_view, from_visit;
	for (auto const &v : g::bfs(adj, s)) from_view.push_back(v);
	g::bfs_visit(adj, s, [&](g::vertex_id v, g::vertex_id p, atl::uint32 d) {
		from_visit.push_back({v, p, d});
	});
	REQUIRE(from_visit.size() == from_view.size());
	for (std::size_t i = 0; i < from_view.size(); ++i) {
		REQUIRE(from_visit[i].vertex == from_view[i].vertex);
		REQUIRE(from_visit[i].parent == from_view[i].parent);
		REQUIRE(from_visit[i].depth == from_view[i].depth);
	}

	g::epoch_marker marks(n);
	std::size_t count = 0;
	g::bfs_visit(adj, std::vector<g::vertex_id>{s}, marks,
				 [&](g::vertex_id, g::vertex_id, atl::uint32) { ++count; });
	REQUIRE(count == from_view.size());
	g::bfs_visit(adj, std::vector<g::vertex_id>{s}, marks,
				 [&](g::vertex_id, g::vertex_id, atl::uint32) { ++count; });
	REQUIRE(count == from_view.size()); // marker retained: second run silent
}

namespace {
	// recursive reference for dfs_fold: returns {sum of w over subtree, size},
	// and accumulates cross-subtree ordered-pair count through each vertex.
	std::array<atl::int64, 2> fold_ref(std::vector<std::vector<uint32>> const &adj, std::vector<int> const &w, uint32 u,
									   uint32 p, atl::int64 &pairs) {
		std::array<atl::int64, 2> acc{w[u], 1};
		for (uint32 c : adj[u]) {
			if (c == p) continue;
			auto ch = fold_ref(adj, w, c, u, pairs);
			pairs += ch[1] * acc[1];
			acc[0] += ch[0];
			acc[1] += ch[1];
		}
		return acc;
	}
}

TEST_CASE("dfs_fold: matches a recursive DFS on random trees", "[graph]") {
	auto seed = GENERATE(take(12, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + std::size_t(rng() % 50);
	std::vector<std::vector<uint32>> adj(n);
	for (uint32 v = 1; v < n; ++v) {
		uint32 const p = uint32(rng() % v);
		adj[p].push_back(v);
		adj[v].push_back(p);
	}
	std::vector<int> w(n);
	for (auto &x : w) x = int(rng() % 20) - 5;

	atl::int64 ref_pairs = 0;
	auto const ref = fold_ref(adj, w, 0, g::none, ref_pairs);

	using Acc = std::array<atl::int64, 2>;
	atl::int64 pairs = 0;
	auto const got = g::dfs_fold(
		adj, 0, [&](g::visit u) { return Acc{w[u.vertex], 1}; },
		[&](Acc &acc, Acc child, g::visit, g::visit, uint32 const &) {
			pairs += child[1] * acc[1];
			acc[0] += child[0];
			acc[1] += child[1];
		});
	REQUIRE(got == ref);
	REQUIRE(pairs == ref_pairs);
	REQUIRE(got[1] == atl::int64(n));
}

TEST_CASE("dfs_fold: edge elements, finalize, depth, deep trees", "[graph]") {
	// weighted root-distance sum via the edge element (element, not index, so
	// this also exercises a weighted adjacency)
	std::vector<std::vector<g::edge<int>>> wt(5);
	auto link = [&](uint32 a, uint32 b, int c) {
		wt[a].push_back({b, c});
		wt[b].push_back({a, c});
	};
	link(0, 1, 3);
	link(0, 2, 5);
	link(1, 3, 2);
	link(1, 4, 4); // dists 0,3,5,5,7 -> sum 20
	using A = std::pair<atl::int64, atl::int64>;
	auto const r = g::dfs_fold(
		wt, 0, [](g::visit) { return A{0, 1}; },
		[](A &acc, A child, g::visit, g::visit, g::edge<int> const &e) {
			acc.first += child.first + atl::int64(e.weight) * child.second;
			acc.second += child.second;
		});
	REQUIRE(r.first == 20);
	REQUIRE(r.second == 5);

	std::vector<std::vector<uint32>> star(4);
	for (uint32 v = 1; v < 4; ++v) {
		star[0].push_back(v);
		star[v].push_back(0);
	}
	int finals = 0;
	uint32 max_depth = 0;
	(void)g::dfs_fold(
		star, 0, [](g::visit) { return 0; },
		[](int &, int, g::visit, g::visit, uint32 const &) {},
		[&](int &, g::visit u) {
			++finals;
			max_depth = (std::max)(max_depth, u.depth);
		});
	REQUIRE(finals == 4);
	REQUIRE(max_depth == 1);

	// a long path folds iteratively (would overflow a recursive DFS)
	std::size_t const deep = 100000;
	std::vector<std::vector<uint32>> path(deep);
	for (uint32 v = 1; v < deep; ++v) {
		path[v - 1].push_back(v);
		path[v].push_back(v - 1);
	}
	auto const size = g::dfs_fold(
		path, 0, [](g::visit) { return atl::int64(1); },
		[](atl::int64 &a, atl::int64 c, g::visit, g::visit, uint32 const &) { a += c; });
	REQUIRE(size == atl::int64(deep));
}

TEST_CASE("dfs_fold: single vertex", "[graph]") {
	std::vector<std::vector<uint32>> one(1);
	int folds = 0, finals = 0;
	auto const r = g::dfs_fold(
		one, 0, [](g::visit u) { return int(u.vertex) + 100; },
		[&](int &, int, g::visit, g::visit, uint32 const &) { ++folds; },
		[&](int &, g::visit) { ++finals; });
	REQUIRE(r == 100);
	REQUIRE(folds == 0);
	REQUIRE(finals == 1);
}

namespace {
	struct exit_probe {
		int *copies, *moves;

		exit_probe(int *c, int *m) : copies(c), moves(m) {}

		exit_probe(exit_probe const &o) : copies(o.copies), moves(o.moves) { ++*copies; }

		exit_probe(exit_probe &&o) noexcept : copies(o.copies), moves(o.moves) { ++*moves; }

		void operator()(g::vertex_id, g::vertex_id) const {}
	};
}

TEST_CASE("events: rvalue callables are moved into the bundle, lvalues copied once", "[graph]") {
	std::vector<std::vector<uint32>> adj(2);
	adj[0] = {1};
	adj[1] = {0};

	int copies = 0, moves = 0;
	g::dfs_visit(adj, 0, g::events(exit_probe(&copies, &moves)));
	REQUIRE(copies == 0);

	copies = moves = 0;
	exit_probe kept(&copies, &moves);
	g::dfs_visit(adj, 0, g::events(kept));
	REQUIRE(copies == 1);
}

TEST_CASE("traversal: sentinel sources and out-of-range marker queries are asserted", "[graph]") {
	std::vector<std::vector<g::vertex_id>> const rows(1);
	g::epoch_marker marker(1);
	REQUIRE(atl_test::aborts([&] { (void)g::bfs(rows, g::none).begin(); }));
	REQUIRE(atl_test::aborts([&] { (void)g::bfs(rows, g::none, marker).begin(); }));
	REQUIRE(atl_test::aborts([&] { (void)marker.visited(1); }));
	g::dense_marker dense(1);
	REQUIRE(atl_test::aborts([&] { (void)dense.visited(1); }));
}
