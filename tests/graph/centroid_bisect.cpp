#include <atl/experimental/graph/centroid_bisect_variants.hpp>
#include <atl/graph/centroid_bisect.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
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

// the two-span callback is accepted; a lone-argument one is not
static_assert(
	std::invocable<decltype([](g::vertex_id, std::span<g::visit const>, std::span<g::visit const>) {}) &,
				   g::vertex_id, std::span<g::visit const>, std::span<g::visit const>>);

// the weighted (edge-span) callback needs a default-constructible, copyable
// edge element: a type without a default constructor is rejected at the
// constraint rather than inside the engine
namespace {
	struct pinned_edge {
		g::vertex_id to;

		explicit pinned_edge(g::vertex_id v) : to(v) {}
	};

	using pinned_graph = std::vector<std::vector<pinned_edge>>;
	inline constexpr auto weighted_probe = [](g::vertex_id, std::span<g::visit const>, std::span<g::visit const>,
											  std::span<pinned_edge const>, std::span<pinned_edge const>) {};
	inline constexpr auto hop_probe = [](g::vertex_id, std::span<g::visit const>, std::span<g::visit const>) {};
}

template<typename Graph, typename Callback>
concept is_bisectable = requires(Graph const &graph, Callback callback) { g::centroid_bisect(graph, callback); };

static_assert(!is_bisectable<pinned_graph, decltype(weighted_probe)>);
static_assert(is_bisectable<pinned_graph, decltype(hop_probe)>);

static_assert(g::centroid_bisect_event<decltype(hop_probe)>);
static_assert(!g::centroid_bisect_event<decltype(weighted_probe)>);
static_assert(!g::weighted_centroid_bisect_event<decltype(weighted_probe), pinned_graph>);
static_assert(g::weighted_centroid_bisect_event<
			  decltype([](g::vertex_id, std::span<g::visit const>, std::span<g::visit const>,
						  std::span<g::edge<int> const>, std::span<g::edge<int> const>) {}),
			  std::vector<std::vector<g::edge<int>>>>);
static_assert(!g::centroid_bisect_event<decltype([](g::vertex_id) {})>);

namespace {
	std::vector<std::vector<uint32>> random_forest(std::mt19937_64 &rng, std::size_t n,
												   bool forest) {
		std::vector<std::vector<uint32>> adj(n);
		for (uint32 v = 1; v < n; ++v) {
			if (forest && rng() % 6 == 0) continue; // start a new component
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

TEST_CASE("centroid_bisect: distance>=2 pairs straddle the two groups exactly once", "[graph]") {
	auto seed = GENERATE(take(30, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 45;
	auto const adj = random_forest(rng, n, rng() % 3 == 0);
	auto const dist = distances(adj);

	// runs the same certification through the library two-span API and the
	// experimental single-span variant — they must agree
	auto certify = [&]<typename Bisect>(Bisect bisect)
		requires std::invocable<Bisect &, decltype(adj) &, std::function<void(uint32, std::span<g::visit const>, std::span<g::visit const>)>>
	{
		std::vector<std::vector<int>> covered(n, std::vector<int>(n, 0));
		bisect(adj, [&](uint32 c, std::span<g::visit const> a, std::span<g::visit const> b) {
			auto check_group = [&](std::span<g::visit const> grp) {
				for (auto const &x : grp) {
					REQUIRE(x.vertex != c);							// center is in neither group
					REQUIRE(dist[c][x.vertex] == int(x.depth));		// depth = hops from center
					REQUIRE(dist[c][x.parent] == int(x.depth) - 1); // parent one hop closer
				}
			};
			check_group(a);
			check_group(b);
			for (auto const &x : a)
				for (auto const &y : b) {
					// a straddling pair's path runs through the center
					REQUIRE(dist[x.vertex][y.vertex] == int(x.depth) + int(y.depth));
					++covered[(std::min)(x.vertex, y.vertex)][(std::max)(x.vertex, y.vertex)];
				}
		});
		for (uint32 u = 0; u < n; ++u)
			for (uint32 v = u + 1; v < n; ++v)
				REQUIRE(covered[u][v] == (dist[u][v] >= 2 ? 1 : 0));
	};

	static_assert(!std::invocable<decltype(certify), int>);
	certify([]<g::adjacency_graph Graph, g::centroid_bisect_event Event>(Graph const &t, Event f) { g::centroid_bisect(t, f); });
	certify([]<g::adjacency_graph Graph, g::centroid_bisect_event Event>(Graph const &t, Event f) {
		// adapt the single-span variant to the (center, a, b) shape
		ex::centroid_bisect_by_single_span(t, [&](uint32 c, std::span<g::visit const> comp, std::size_t n0) {
			f(c, comp.subspan(1, n0), comp.subspan(1 + n0));
		});
	});
}

TEST_CASE("centroid_bisect: weighted edge-spans give true weighted depths", "[graph]") {
	auto seed = GENERATE(take(30, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 40;
	bool const is_forest = rng() % 2 == 0;
	std::vector<std::vector<g::edge<int>>> adj(n);
	for (uint32 v = 1; v < n; ++v) {
		if (is_forest && rng() % 6 == 0) continue; // start a new component
		uint32 const p = uint32(rng() % v);
		int const w = int(rng() % 5) + 1;
		adj[p].push_back({v, w});
		adj[v].push_back({p, w});
	}
	int const W = int(rng() % 30) + 1;

	// brute: all-pairs weighted distance count at exactly W
	int64 brute = 0;
	for (uint32 s = 0; s < n; ++s) {
		std::vector<int64> d(n, -1);
		std::vector<uint32> st{s};
		d[s] = 0;
		while (!st.empty()) {
			uint32 const u = st.back();
			st.pop_back();
			for (auto const &e : adj[u])
				if (d[e.to] < 0) {
					d[e.to] = d[u] + e.weight;
					st.push_back(e.to);
				}
		}
		for (uint32 t = s + 1; t < n; ++t)
			if (d[t] == W) ++brute;
	}
	// bisect skips distance-1 (adjacent) pairs, so add the single edges of weight W
	int64 adjacent = 0;
	for (uint32 u = 0; u < n; ++u)
		for (auto const &e : adj[u])
			if (e.to > u && e.weight == W) ++adjacent;

	int64 ans = 0;
	std::vector<int64> wdist(n);
	g::centroid_bisect(adj, [&](uint32 c, std::span<g::visit const> a, std::span<g::visit const> b,
								std::span<g::edge<int> const> ea, std::span<g::edge<int> const> eb) {
		auto fold = [&](std::span<g::visit const> grp, std::span<g::edge<int> const> e) {
			wdist[c] = 0;
			for (std::size_t i = 0; i < grp.size(); ++i)
				wdist[grp[i].vertex] = wdist[grp[i].parent] + e[i].weight; // parent seen first
		};
		fold(a, ea);
		fold(b, eb);
		for (auto const &x : a)
			for (auto const &y : b)
				if (wdist[x.vertex] + wdist[y.vertex] == W) ++ans;
	});
	REQUIRE(ans + adjacent == brute);

	// the single-span variant's weighted form: one fold over the whole span,
	// edges[i] parallel to component[i] and edges[0] the center's placeholder
	int64 split = 0;
	ex::centroid_bisect_by_single_span(adj, [&](uint32 c, std::span<g::visit const> comp, std::size_t n0,
												std::span<g::edge<int> const> edges) {
		REQUIRE(comp.size() == edges.size());
		REQUIRE(comp[0].vertex == c);
		wdist[c] = 0;
		for (std::size_t i = 1; i < comp.size(); ++i)
			wdist[comp[i].vertex] = wdist[comp[i].parent] + edges[i].weight;
		for (std::size_t i = 1; i <= n0; ++i)
			for (std::size_t j = n0 + 1; j < comp.size(); ++j)
				if (wdist[comp[i].vertex] + wdist[comp[j].vertex] == W) ++split;
	});
	REQUIRE(split + adjacent == brute);
}

TEST_CASE("centroid_bisect: each group holds at least a third of the branch mass", "[graph]") {
	// per event the component is center + a + b; neither group may fall below
	// a third of |a| + |b|, which is what bounds every vertex to O(log n) events
	auto check = []<g::adjacency_graph Graph>(Graph const &adj) {
		std::size_t events = 0;
		g::centroid_bisect(adj, [&](uint32, std::span<g::visit const> a, std::span<g::visit const> b) {
			++events;
			REQUIRE(3 * (std::min)(a.size(), b.size()) >= a.size() + b.size());
		});
		return events;
	};
	// a spider with legs of 1, h and h hops, the short leg first in the
	// adjacency: first-fit up to half the mass would put the short leg alone
	// in one group
	for (uint32 const h : {2u, 5u, 40u}) {
		std::vector<std::vector<uint32>> spider(2 + 2 * h);
		auto link = [&](uint32 u, uint32 v) {
			spider[u].push_back(v);
			spider[v].push_back(u);
		};
		link(0, 1);
		for (uint32 leg = 0; leg < 2; ++leg)
			for (uint32 k = 0; k < h; ++k) {
				uint32 const v = 2 + leg * h + k;
				link(k == 0 ? 0 : v - 1, v);
			}
		REQUIRE(check(spider) > 0);
	}
	auto seed = GENERATE(take(20, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::size_t const n = 1 + rng() % 2000;
	check(random_forest(rng, n, rng() % 3 == 0));
}

TEST_CASE("centroid_bisect: canonical", "[graph]") {
	// path of 7: cross pairs by distance = 7 - d for d >= 2 (d == 1 never emitted)
	std::vector<std::vector<uint32>> path(7);
	for (uint32 v = 1; v < 7; ++v) {
		path[v - 1].push_back(v);
		path[v].push_back(v - 1);
	}
	int64 by_dist[8] = {};
	g::centroid_bisect(path, [&](uint32, std::span<g::visit const> a, std::span<g::visit const> b) {
		for (auto const &x : a)
			for (auto const &y : b) ++by_dist[x.depth + y.depth];
	});
	for (int d = 2; d < 7; ++d) REQUIRE(by_dist[d] == 7 - d);
	REQUIRE(by_dist[1] == 0); // adjacent pairs are the caller's to seed

	// single vertex and an edge: no distance->=2 pair, so no cross pairs at all
	for (std::size_t k : {std::size_t(1), std::size_t(2)}) {
		std::vector<std::vector<uint32>> g(k);
		if (k == 2) {
			g[0].push_back(1);
			g[1].push_back(0);
		}
		int64 cross = 0;
		g::centroid_bisect(g, [&](uint32, std::span<g::visit const> a, std::span<g::visit const> b) {
			cross += int64(a.size()) * int64(b.size());
		});
		REQUIRE(cross == 0);
	}
}

namespace {
	using split_tree = std::vector<std::vector<uint32>>;
	using split_plain = decltype([](g::vertex_id, std::span<g::visit const>, std::size_t) {});
	using split_weighted =
		decltype([](g::vertex_id, std::span<g::visit const>, std::size_t, std::span<uint32 const>) {});
	using split_wrong = decltype([](g::vertex_id) {});

	// the requires-expression must be dependent, or an unsatisfied one is a
	// hard error rather than false
	template<typename F>
	concept is_split_callback = requires(split_tree const &t, F f) { ex::centroid_bisect_by_single_span(t, f); };
}

static_assert(is_split_callback<split_plain>);
static_assert(is_split_callback<split_weighted>);
static_assert(!is_split_callback<split_wrong>);
