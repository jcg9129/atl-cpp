#include <atl/experimental/graph/centroid_variants.hpp>
#include <atl/graph/centroid_bisect.hpp>
#include <atl/graph/centroid_decomposition.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/lca.hpp>
#include <atl/graph/ternarization.hpp>
#include <atl/graph/virtual_tree.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::int64;
using atl::uint32;
using atl::uint64;

static_assert(g::centroid_event<decltype([](g::vertex_id, g::vertex_id,
											std::span<g::centroid_vertex const>) {})>);
static_assert(!g::centroid_event<decltype([](g::vertex_id) {})>);

namespace {
	/** @brief An adjacency element the engine cannot store (no default constructor). */
	struct pinned_edge {
		g::vertex_id to;

		explicit pinned_edge(g::vertex_id v) : to(v) {}
	};

	inline constexpr auto edge_probe = [](g::vertex_id, g::vertex_id, std::span<g::centroid_vertex const>,
										  std::span<pinned_edge const>) {};

	template<typename Graph, typename Callback>
	concept is_visitable = requires(Graph const &graph, Callback callback) { g::centroid_visit(graph, callback); };
}

static_assert(g::centroid_edge_event<decltype([](g::vertex_id, g::vertex_id, std::span<g::centroid_vertex const>,
												 std::span<uint32 const>) {}),
									 uint32>);
static_assert(!g::centroid_event<decltype(edge_probe)>);
static_assert(!g::centroid_edge_event<decltype(edge_probe), pinned_edge>);
// rejected at the constraint, not inside the engine
static_assert(!is_visitable<std::vector<std::vector<pinned_edge>>, decltype(edge_probe)>);

namespace {
	std::vector<std::vector<uint32>> random_forest(std::mt19937_64 &rng, std::size_t n,
												   bool forest) {
		std::vector<std::vector<uint32>> adj(n);
		for (uint32 v = 1; v < n; ++v) {
			if (forest && rng() % 6 == 0) continue;
			uint32 const p = uint32(rng() % v);
			adj[p].push_back(v);
			adj[v].push_back(p);
		}
		return adj;
	}

	std::vector<std::vector<int>> distances(std::vector<std::vector<uint32>> const &adj) {
		std::size_t const n = adj.size();
		std::vector<std::vector<int>> dist(n, std::vector<int>(n, -1));
		for (uint32 s = 0; s < n; ++s) {
			std::vector<uint32> queue{s};
			dist[s][s] = 0;
			for (std::size_t at = 0; at < queue.size(); ++at)
				for (uint32 w : adj[queue[at]])
					if (dist[s][w] < 0) {
						dist[s][w] = dist[s][queue[at]] + 1;
						queue.push_back(w);
					}
		}
		return dist;
	}
}

TEST_CASE("centroid_visit: every pair exactly once, depths and branches true", "[graph]") {
	auto seed = GENERATE(take(10, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 40;
	auto const adj = random_forest(rng, n, rng() % 3 == 0);
	auto const dist = distances(adj);

	auto certify = [&]<typename Visit>(Visit visit)
		requires std::invocable<Visit &, decltype(adj) &, std::function<void(uint32, uint32, std::span<g::centroid_vertex const>)>>
	{
		std::vector<std::vector<int>> covered(n, std::vector<int>(n, 0));
		std::size_t total = 0;
		// the most recent earlier event holding v, and the smallest component
		// size among all earlier events holding v
		std::vector<uint32> last(n, g::none);
		std::vector<std::size_t> last_size(n, 0), smallest(n, std::size_t(-1));
		visit(adj, [&](uint32 c, uint32 parent_center, std::span<g::centroid_vertex const> comp) {
			REQUIRE(comp[0].vertex == c);
			REQUIRE(comp[0].depth == 0);
			// parent_center: the weakest stronger separator holding c, i.e. the
			// smallest earlier component that contains it
			REQUIRE(parent_center == last[c]);
			if (last[c] != g::none) REQUIRE(last_size[c] == smallest[c]);
			for (std::size_t i = 1; i < comp.size(); ++i) {
				uint32 const v = comp[i].vertex;
				last[v] = c;
				last_size[v] = comp.size();
				smallest[v] = (std::min)(smallest[v], comp.size());
			}
			total += comp.size();
			for (std::size_t i = 1; i < comp.size(); ++i) {
				auto const &[v, p, depth, branch] = comp[i];
				REQUIRE(dist[c][v] == int(depth));
				REQUIRE(dist[c][p] == int(depth) - 1);
				REQUIRE(branch != g::none);
			}
			for (std::size_t i = 0; i < comp.size(); ++i)
				for (std::size_t j = 0; j < i; ++j)
					if (comp[i].branch != comp[j].branch)
						++covered[(std::min)(comp[i].vertex, comp[j].vertex)]
								 [(std::max)(comp[i].vertex, comp[j].vertex)];
		});
		for (uint32 u = 0; u < n; ++u)
			for (uint32 v = u + 1; v < n; ++v)
				REQUIRE(covered[u][v] == (dist[u][v] >= 0 ? 1 : 0));
		REQUIRE(total <= n * std::bit_width(n)); // n * (floor(log2 n) + 1)
	};
	static_assert(!std::invocable<decltype(certify), int>);
	certify([]<g::adjacency_graph Graph, g::centroid_event Event>(Graph const &t, Event f) { g::centroid_visit(t, f); });
	certify([]<g::detail::scannable_graph Graph, g::centroid_event Event>(Graph const &t, Event f) { ex::centroid_visit_by_subtree_sizes(t, f); });
	certify([]<g::detail::scannable_graph Graph, g::centroid_event Event>(Graph const &t, Event f) { ex::centroid_visit_by_reused_sizes(t, f); });
}

TEST_CASE("centroid_decomposition: the separator tree covers every pair", "[graph]") {
	auto seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 2 + rng() % 40;
	auto const adj = random_forest(rng, n, rng() % 2 == 0);
	auto const dist = distances(adj);
	auto const parent = g::centroid_decomposition(adj);
	REQUIRE(ex::centroid_decomposition_by_component_sweep(adj) == parent);

	auto const ancestors = [&](uint32 v) {
		std::vector<uint32> chain{v};
		while (parent[chain.back()] != g::none) chain.push_back(parent[chain.back()]);
		return chain;
	};
	for (uint32 u = 0; u < n; ++u)
		for (uint32 v = u + 1; v < n; ++v) {
			if (dist[u][v] < 0) continue; // different trees of the forest
			// some common separator-tree ancestor must lie on the u-v path
			auto const au = ancestors(u), av = ancestors(v);
			bool found = false;
			for (uint32 a : au)
				for (uint32 b : av)
					if (a == b && dist[u][a] + dist[a][v] == dist[u][v]) found = true;
			REQUIRE(found);
		}
}

TEST_CASE("centroid_decomposition: the direct separator tree matches the sweep on larger forests", "[graph]") {
	auto seed = GENERATE(take(6, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 3000;
	auto adj = random_forest(rng, n, rng() % 2 == 0);
	if (rng() % 2) // a long path segment, to deepen the ranking
		for (uint32 v = 1; v < n; v += 2)
			if (adj[v].empty()) {
				adj[v].push_back(v - 1);
				adj[v - 1].push_back(v);
			}
	REQUIRE(ex::centroid_decomposition_by_component_sweep(adj) == g::centroid_decomposition(adj));
	std::vector<std::vector<uint32>> path(n);
	for (uint32 v = 1; v < n; ++v) {
		path[v - 1].push_back(v);
		path[v].push_back(v - 1);
	}
	REQUIRE(ex::centroid_decomposition_by_component_sweep(path) == g::centroid_decomposition(path));
}

TEST_CASE("centroid: canonical", "[graph]") {
	// path of 7: every event component is an interval around its separator
	std::vector<std::vector<uint32>> path(7);
	for (uint32 v = 1; v < 7; ++v) {
		path[v - 1].push_back(v);
		path[v].push_back(v - 1);
	}
	int64 pairs_by_dist[7] = {};
	g::centroid_visit(path, [&](uint32, uint32, std::span<g::centroid_vertex const> comp) {
		for (std::size_t i = 0; i < comp.size(); ++i)
			for (std::size_t j = 0; j < i; ++j)
				if (comp[i].branch != comp[j].branch)
					++pairs_by_dist[comp[i].depth + comp[j].depth];
	});
	for (int d = 1; d < 7; ++d) REQUIRE(pairs_by_dist[d] == 7 - d);

	// single vertex
	std::vector<std::vector<uint32>> one(1);
	std::size_t events = 0;
	g::centroid_visit(one, [&](uint32 c, uint32 up, std::span<g::centroid_vertex const> comp) {
		++events;
		REQUIRE(c == 0);
		REQUIRE(up == g::none);
		REQUIRE(comp.size() == 1);
	});
	REQUIRE(events == 1);
}

TEST_CASE("centroid_visit: parallel edge span for weighted top-down values", "[graph]") {
	auto seed = GENERATE(take(10, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 30;
	std::vector<std::vector<g::edge<int>>> adj(n);
	for (uint32 v = 1; v < n; ++v) {
		uint32 const p = uint32(rng() % v);
		adj[p].push_back({v, int(rng() % 5) + 1});
		adj[v].push_back({p, adj[p].back().weight});
	}
	int const W = int(rng() % 20) + 1;

	// brute: all-pairs weighted distance
	int64 brute = 0;
	for (uint32 s = 0; s < n; ++s) {
		std::vector<int64> dist(n, -1);
		std::vector<uint32> st{s};
		dist[s] = 0;
		while (!st.empty()) {
			uint32 const u = st.back();
			st.pop_back();
			for (auto const &e : adj[u])
				if (dist[e.to] < 0) {
					dist[e.to] = dist[u] + e.weight;
					st.push_back(e.to);
				}
		}
		for (uint32 t = s + 1; t < n; ++t)
			if (dist[t] == W) ++brute;
	}

	int64 ans = 0;
	std::vector<int64> wdist(n), seen;
	g::centroid_visit(adj, [&](uint32 c, uint32, std::span<g::centroid_vertex const> comp,
							   std::span<g::edge<int> const> edges) {
		REQUIRE(comp.size() == edges.size());
		wdist[c] = 0;
		for (std::size_t i = 1; i < comp.size(); ++i)
			wdist[comp[i].vertex] = wdist[comp[i].parent] + edges[i].weight; // top-down
		seen.assign(W + 1, 0);
		seen[0] = 1;
		std::size_t i = 1;
		while (i < comp.size()) {
			std::size_t const from = i;
			uint32 const br = comp[i].branch;
			for (; i < comp.size() && comp[i].branch == br; ++i)
				if (wdist[comp[i].vertex] <= W) ans += seen[std::size_t(W - wdist[comp[i].vertex])];
			for (std::size_t j = from; j < i; ++j)
				if (wdist[comp[j].vertex] <= W) ++seen[std::size_t(wdist[comp[j].vertex])];
		}
	});
	REQUIRE(ans == brute);
}

TEST_CASE("centroid_visit: 3-arg and 4-arg callbacks both accepted", "[graph]") {
	std::vector<std::vector<uint32>> path(4);
	for (uint32 v = 1; v < 4; ++v) {
		path[v - 1].push_back(v);
		path[v].push_back(v - 1);
	}
	std::size_t three = 0, four = 0;
	g::centroid_visit(path, [&](uint32, uint32, std::span<g::centroid_vertex const> c) {
		three += c.size();
	});
	g::centroid_visit(path, [&](uint32, uint32, std::span<g::centroid_vertex const> c,
								std::span<uint32 const> e) {
		REQUIRE(c.size() == e.size());
		four += c.size();
	});
	REQUIRE(three == four); // same components, edge span merely added
}

TEST_CASE("tree algorithms use heap traversal storage on a million-vertex path", "[graph][deep]") {
	constexpr std::size_t vertices = 1'000'000;
	std::vector<g::arc<>> arcs;
	arcs.reserve(vertices - 1);
	for (g::vertex_id v = 1; v < vertices; ++v) arcs.push_back({v - 1, v});
	auto const tree = g::undirected_csr(vertices, arcs);
	arcs.clear();
	arcs.shrink_to_fit();
	{
		g::lca const ancestors(tree, 0);
		REQUIRE(ancestors.distance(0, g::vertex_id(vertices - 1)) == vertices - 1);
		auto const result = g::virtual_tree_of(ancestors, std::array<g::vertex_id, 2>{0, g::vertex_id(vertices - 1)});
		REQUIRE(result.vertices.size() == 2);
	}
	REQUIRE(g::centroid_decomposition(tree).size() == vertices);
	std::size_t emitted = 0;
	g::centroid_visit(tree, [&](g::vertex_id, g::vertex_id, std::span<g::centroid_vertex const> comp) { emitted += comp.size(); });
	REQUIRE(emitted <= vertices * 20);
	atl::uint64 pairs = 0;
	g::centroid_bisect(tree, [&](g::vertex_id, std::span<g::visit const> a, std::span<g::visit const> b) {
		pairs += atl::uint64(a.size()) * b.size();
	});
	REQUIRE(pairs == atl::uint64(vertices) * (vertices - 1) / 2 - (vertices - 1));
	REQUIRE(g::ternarization_of(tree, 0).tree.vertex_count() == vertices);
}
