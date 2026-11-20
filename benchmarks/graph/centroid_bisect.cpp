/**
 * @file
 * @brief Balanced-bisection engine throughput on random, star, caterpillar, path,
 * broom and spider trees. The library uses path reversal and thirds balance;
 * the CSR/BFS contenders use first-fit coloring and either cursor or bucket CSR.
 * A shared loop sums emitted mass; untimed pair counts certify agreement even
 * when the partitions have different mass. Hop and parent-edge forms are timed.
 * Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
 * -O2 / -O2 -march=native, 2026-10-15: random n=2e5 hops 47.8 / 48.3 ms
 * for the library and 110 / 112 ms by CSR BFS. Full tables: docs/centroid-decomposition.md.
 */
#include <atl/experimental/graph/centroid_bisect_variants.hpp>
#include <atl/graph/centroid_bisect.hpp>
#include <atl/graph/csr.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/counting_sort.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace g = atl::graph;
using atl::uint32;
using atl::uint64;

namespace {
	using namespace atl::graph;
	namespace graph = atl::graph;

	/**
	 * @brief Balanced bisection by CSR and BFS:
	 *        children CSR + centroid-rooted BFS per level, first-fit colouring
	 *        up to half the mass, two layout sweeps; the CSR from hand-rolled
	 *        counting or from `bucket_by_key` behind a switch.
	 */
	template<bool WantEdges, bool IsBucketByKey, g::adjacency_graph Graph, typename Emit>
		requires std::invocable<Emit &, g::vertex_id, std::span<g::visit const>, std::size_t, std::span<g::edge_t<Graph> const>>
			  && (!WantEdges || (std::copyable<g::edge_t<Graph>> && std::default_initializable<g::edge_t<Graph>>))
	void centroid_bisect_by_csr_bfs(Graph const &g, Emit &&emit) {
		using Edge = std::ranges::range_value_t<out_edges_t<Graph>>;
		std::size_t const n = graph::vertex_count(g);
		if (n == 0) return;

		// A subtree to process, held as a rooted PARENT ARRAY in LOCAL
		// indices [0, |vertices|), parent preceding child so `parent[i] < i`
		// (local 0 is the root, its own parent). Splitting rebuilds the two
		// child parent-arrays from THIS one — half the footprint of an
		// induced CSR, and a high-degree center's degree still splits
		// across the children, so it never goes quadratic.
		struct frame {
			std::vector<vertex_id> vertices; // local index -> global vertex id
			std::vector<uint32> parent;		 // i's parent, as a local index
			std::vector<Edge> parent_edge;	 // edge element to that parent (WantEdges)
		};

		std::vector<uint32> subtree_size(n), child_offset(n + 2), child_list(n), child_cursor(n);
		std::vector<uint32> rooted_parent(n), rooted_depth(n), group(n), new_local(n);
		std::vector<Edge> rooted_edge;
		if constexpr (WantEdges) rooted_edge.resize(n);
		std::vector<uint32> order;
		order.reserve(n);
		std::vector<visit> component;
		std::vector<Edge> component_edges;

		std::vector<frame> stack, freelist;
		auto const take_frame = [&]() -> frame {
			if (freelist.empty()) return frame{};
			frame f = std::move(freelist.back());
			freelist.pop_back();
			f.vertices.clear();
			f.parent.clear();
			if constexpr (WantEdges) f.parent_edge.clear();
			return f;
		};

		// seed one frame per connected component as a parent array (the only
		// pass over the original graph); BFS keeps parent[i] < i
		{
			dense_marker seen(n);
			for (uint32 s = 0; s < n; ++s) {
				if (seen.visited(s)) continue;
				frame f;
				f.vertices.push_back(s);
				f.parent.push_back(0);
				if constexpr (WantEdges) f.parent_edge.push_back(Edge{});
				seen.try_visit(s);
				for (std::size_t i = 0; i < f.vertices.size(); ++i)
					for (auto const &e : graph::out_edges(g, f.vertices[i])) {
						uint32 const w = graph::target(e);
						if (!seen.try_visit(w)) continue;
						f.vertices.push_back(w);
						f.parent.push_back(uint32(i));
						if constexpr (WantEdges) f.parent_edge.push_back(e);
					}
				stack.push_back(std::move(f));
			}
		}

		while (!stack.empty()) {
			frame fr = std::move(stack.back());
			stack.pop_back();
			std::size_t const m = fr.vertices.size();
			if (m <= 2) { // no pair at distance >= 2
				freelist.push_back(std::move(fr));
				continue;
			}
			auto const &parent = fr.parent;

			// subtree sizes (children precede parents: parent[i] < i) and the
			// centroid = the deepest vertex with 2*subtree_size >= m (argmin
			// subtree_size over that set; they form a root-to-centroid chain)
			for (std::size_t i = 0; i < m; ++i) subtree_size[i] = 1;
			for (std::size_t i = m; i-- > 1;) subtree_size[parent[i]] += subtree_size[i];
			uint32 c = 0, best_size = uint32(m) + 1;
			for (uint32 i = 0; i < m; ++i)
				if (std::size_t(subtree_size[i]) * 2 >= m && subtree_size[i] < best_size)
					best_size = subtree_size[i], c = i;

			// directed children CSR (m-1 arcs) to walk out from the centroid
			if constexpr (IsBucketByKey) {
				// items are the vertices, keyed by their parent; the root goes
				// to an extra last bucket that nothing reads
				atl::detail::bucket_by_key(
					m + 1, [&](std::size_t i) { return i == 0 ? m : std::size_t(parent[i]); },
					std::span<uint32>(child_offset.data(), m + 2), std::span<uint32>(child_list.data(), m));
			} else {
				for (std::size_t i = 0; i <= m; ++i) child_offset[i] = 0;
				for (std::size_t i = 1; i < m; ++i) ++child_offset[parent[i] + 1];
				for (std::size_t i = 0; i < m; ++i) child_offset[i + 1] += child_offset[i];
				for (std::size_t i = 0; i < m; ++i) child_cursor[i] = child_offset[i];
				for (uint32 i = 1; i < m; ++i) child_list[child_cursor[parent[i]]++] = i;
			}

			// re-root at the centroid: c-rooted BFS order, parent, hop depth,
			// parent edge — each vertex's neighbours are its children (child_list)
			// plus its csr_bfs parent (parent[u]), minus where we came from
			order.clear();
			rooted_parent[c] = c;
			rooted_depth[c] = 0;
			order.push_back(c);
			for (std::size_t i = 0; i < order.size(); ++i) {
				uint32 const u = order[i];
				for (uint32 k = child_offset[u]; k < child_offset[u + 1]; ++k) {
					uint32 const w = child_list[k];
					if (w == rooted_parent[u]) continue;
					rooted_parent[w] = u;
					rooted_depth[w] = rooted_depth[u] + 1;
					if constexpr (WantEdges) rooted_edge[w] = fr.parent_edge[w];
					order.push_back(w);
				}
				uint32 const p0 = parent[u];
				if (u != 0 && p0 != rooted_parent[u]) {
					rooted_parent[p0] = u;
					rooted_depth[p0] = rooted_depth[u] + 1;
					if constexpr (WantEdges) rooted_edge[p0] = fr.parent_edge[u];
					order.push_back(p0);
				}
			}
			// 2-color the centroid's branches balanced: fill group 0 up to
			// half the mass, the rest is group 1 (both nonempty for m >= 3).
			// Branch sizes come straight from the 0-rooted `subtree_size` —
			// a c-rooted child of c keeps its subtree, the up-branch through
			// parent[c] holds the rest (m - subtree_size[c]) — so no
			// c-rooted size recompute is needed. Colours propagate down the
			// c-rooted order from each branch root
			std::size_t const half = (m - 1) / 2;
			uint32 const up = (c == 0) ? none : parent[c]; // up-branch root
			std::size_t take = 0;
			for (std::size_t i = 1; i < m; ++i) {
				uint32 const u = order[i];
				if (rooted_parent[u] == c) {
					std::size_t const bs = (u == up) ? std::size_t(m) - subtree_size[c] : subtree_size[u];
					if (take + bs <= half)
						group[u] = 0, take += bs;
					else
						group[u] = 1;
				} else
					group[u] = group[rooted_parent[u]];
			}

			// lay out component = [center, group0, group1] AND build the two child
			// parent-arrays (group + center) in the SAME two sweeps
			component.clear();
			component.push_back({fr.vertices[c], fr.vertices[c], 0});
			if constexpr (WantEdges) {
				component_edges.clear();
				component_edges.push_back(Edge{});
			}
			frame child[2] = {take_frame(), take_frame()};
			for (auto &ch : child) {
				ch.vertices.push_back(fr.vertices[c]);
				ch.parent.push_back(0);
				if constexpr (WantEdges) ch.parent_edge.push_back(Edge{});
			}
			new_local[c] = 0;
			std::size_t n0 = 0;
			for (uint32 pass = 0; pass < 2; ++pass) {
				frame &ch = child[pass];
				for (std::size_t i = 1; i < m; ++i) {
					uint32 const u = order[i];
					if (group[u] != pass) continue;
					component.push_back({fr.vertices[u], fr.vertices[rooted_parent[u]], rooted_depth[u]});
					new_local[u] = uint32(ch.vertices.size());
					ch.vertices.push_back(fr.vertices[u]);
					ch.parent.push_back(new_local[rooted_parent[u]]); // rooted_parent[u] kept, earlier in sweep
					if constexpr (WantEdges) {
						component_edges.push_back(rooted_edge[u]);
						ch.parent_edge.push_back(rooted_edge[u]);
					}
				}
				if (pass == 0) n0 = component.size() - 1;
			}
			if constexpr (WantEdges)
				emit(fr.vertices[c], std::span<visit const>(component), n0,
					 std::span<Edge const>(component_edges));
			else
				emit(fr.vertices[c], std::span<visit const>(component), n0,
					 std::span<Edge const>{});
			stack.push_back(std::move(child[0]));
			stack.push_back(std::move(child[1]));
			freelist.push_back(std::move(fr)); // recycle after children built
		}
	}

	std::vector<std::pair<std::string, std::vector<uint32>>> shapes(std::size_t n) {
		std::mt19937 rng(1);
		std::vector<std::pair<std::string, std::vector<uint32>>> out;
		std::vector<uint32> parent(n, 0);
		for (uint32 v = 1; v < n; ++v) parent[v] = uint32(rng() % v);
		out.emplace_back("random", parent);
		for (uint32 v = 1; v < n; ++v) parent[v] = 0;
		out.emplace_back("star", parent);
		for (uint32 v = 1; v < n; ++v) parent[v] = v % 2 ? v - 1 : v - 2;
		out.emplace_back("caterpillar", parent);
		for (uint32 v = 1; v < n; ++v) parent[v] = v - 1;
		out.emplace_back("path", parent);
		for (uint32 v = 1; v < n; ++v) parent[v] = v < n / 2 ? v - 1 : uint32(n / 2 - 1);
		out.emplace_back("broom", parent);
		// spider: legs of 1, h, h hops repeated recursively would need a
		// construction; one level of 1, h, h exposes the first-fit split
		uint32 const h = uint32((n - 2) / 2);
		for (uint32 v = 1; v < n; ++v) parent[v] = v == 1 || v == 2 || v == 2 + h ? 0 : v - 1;
		out.emplace_back("spider 1,h,h", parent);
		return out;
	}

	template<g::adjacency_edge Edge>
		requires std::same_as<Edge, g::vertex_id> || std::same_as<Edge, g::edge<int>>
	csr<Edge> build(std::vector<uint32> const &parent) {
		std::mt19937 rng(7);
		if constexpr (std::is_same_v<Edge, vertex_id>) {
			std::vector<arc<>> edges;
			for (uint32 v = 1; v < parent.size(); ++v) edges.push_back({parent[v], v});
			return undirected_csr(parent.size(), edges);
		} else {
			std::vector<arc<int>> edges;
			for (uint32 v = 1; v < parent.size(); ++v) edges.push_back({parent[v], v, int(rng() % 100)});
			return undirected_csr(parent.size(), edges);
		}
	}

	/** @brief Engine work with a trivial callback: the emitted mass, so nothing is dead. */
	template<bool WantEdges, typename Engine, g::adjacency_graph Graph>
		requires std::invocable<Engine &, Graph const &, std::function<void(g::vertex_id, std::span<g::visit const>, std::size_t, std::span<g::edge_t<Graph> const>)>>
	uint64 run(Engine engine, Graph const &t) {
		uint64 acc = 0;
		engine(t, [&](vertex_id, std::span<visit const> comp, std::size_t n0, std::span<g::edge_t<Graph> const>) { acc += comp.size() + n0; });
		return acc;
	}
}

TEST_CASE("centroid_bisect engine contenders", "[centroid_bisect]") {
	auto const by_csr_bfs = []<g::adjacency_graph Graph, typename Emit>(Graph const &t, Emit &&f)
		requires std::invocable<Emit &, g::vertex_id, std::span<g::visit const>, std::size_t, std::span<g::edge_t<Graph> const>>
	{ centroid_bisect_by_csr_bfs<false, false>(t, f); };
	auto const by_bucket_csr_bfs = []<g::adjacency_graph Graph, typename Emit>(Graph const &t, Emit &&f)
		requires std::invocable<Emit &, g::vertex_id, std::span<g::visit const>, std::size_t, std::span<g::edge_t<Graph> const>>
	{ centroid_bisect_by_csr_bfs<false, true>(t, f); };
	auto const library = []<g::adjacency_graph Graph, typename Emit>(Graph const &t, Emit &&f)
		requires std::invocable<Emit &, g::vertex_id, std::span<g::visit const>, std::size_t, std::span<g::edge_t<Graph> const>>
	{ g::detail::centroid_bisect_engine<false>(t, f); };
	auto const by_csr_bfs_weighted = []<g::adjacency_graph Graph, typename Emit>(Graph const &t, Emit &&f)
		requires std::invocable<Emit &, g::vertex_id, std::span<g::visit const>, std::size_t, std::span<g::edge_t<Graph> const>>
	{ centroid_bisect_by_csr_bfs<true, false>(t, f); };
	auto const library_weighted = []<g::adjacency_graph Graph, typename Emit>(Graph const &t, Emit &&f)
		requires std::invocable<Emit &, g::vertex_id, std::span<g::visit const>, std::size_t, std::span<g::edge_t<Graph> const>>
	{ g::detail::centroid_bisect_engine<true>(t, f); };
	for (std::size_t const n : {std::size_t(200'000), std::size_t(1'000'000)})
		for (auto const &[name, parent] : shapes(n)) {
			bool const is_large = n > 200'000;
			if (is_large && name != "random" && name != "star") continue;
			std::string const label = name + (is_large ? " 1e6" : " 2e5");
			auto const tree = build<vertex_id>(parent);
			auto const pair_count = [&]<typename Engine>(Engine engine)
				requires std::invocable<Engine &, decltype(tree) &, std::function<void(vertex_id, std::span<visit const>, std::size_t, std::span<vertex_id const>)>>
			{
				uint64 pairs = 0;
				engine(tree, [&](vertex_id, std::span<visit const> comp, std::size_t n0, std::span<vertex_id const>) {
					pairs += uint64(n0) * (comp.size() - 1 - n0);
				});
				return pairs;
			};
			uint64 const want_pairs = uint64(n) * (n - 1) / 2 - (n - 1);
			REQUIRE(pair_count(library) == want_pairs);
			REQUIRE(pair_count(by_csr_bfs) == want_pairs);
			REQUIRE(pair_count(by_bucket_csr_bfs) == want_pairs);
			REQUIRE(run<false>(by_bucket_csr_bfs, tree) == run<false>(by_csr_bfs, tree));
			BENCHMARK(label + " hop | library") { return run<false>(library, tree); };
			BENCHMARK(label + " hop | by CSR BFS") { return run<false>(by_csr_bfs, tree); };
			if (is_large) continue;
			BENCHMARK(label + " hop | by bucket CSR BFS") { return run<false>(by_bucket_csr_bfs, tree); };
			if (name != "random" && name != "caterpillar") continue;
			auto const weighted = build<edge<int>>(parent);
			auto const weighted_pairs = [&]<typename Engine>(Engine engine)
				requires std::invocable<Engine &, decltype(weighted) &, std::function<void(vertex_id, std::span<visit const>, std::size_t, std::span<edge<int> const>)>>
			{
				uint64 pairs = 0;
				engine(weighted, [&](vertex_id, std::span<visit const> comp, std::size_t n0, std::span<edge<int> const> edges) {
					REQUIRE(edges.size() == comp.size());
					pairs += uint64(n0) * (comp.size() - 1 - n0);
				});
				return pairs;
			};
			REQUIRE(weighted_pairs(library_weighted) == want_pairs);
			REQUIRE(weighted_pairs(by_csr_bfs_weighted) == want_pairs);

			BENCHMARK(label + " weighted | library") { return run<true>(library_weighted, weighted); };
			BENCHMARK(label + " weighted | by CSR BFS") { return run<true>(by_csr_bfs_weighted, weighted); };
		}
}
