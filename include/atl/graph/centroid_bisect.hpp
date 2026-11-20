#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::graph {
	/**
	 * @brief The hop-distance @ref atl::graph::centroid_bisect callback:
	 *        `f(center, group_a, group_b)`, each group a `span<visit const>`
	 *        (`depth` = hops from the center, parent preceding child).
	 */
	template<typename F>
	concept centroid_bisect_event = std::invocable<F &, vertex_id, std::span<visit const>, std::span<visit const>>;

	/**
	 * @brief The weighted @ref atl::graph::centroid_bisect callback over graph @p Graph:
	 *        `f(center, group_a, group_b, edges_a, edges_b)`, the edge spans
	 *        parallel to the groups and holding each vertex's parent-edge
	 *        element — stored by value, so that element must be
	 *        default-initializable and copyable.
	 */
	template<typename F, typename Graph>
	concept weighted_centroid_bisect_event =
		adjacency_graph<Graph> && std::default_initializable<edge_t<Graph>> && std::copyable<edge_t<Graph>>
		&& std::constructible_from<edge_t<Graph>, std::remove_reference_t<std::ranges::range_reference_t<out_edges_t<Graph>>> const &>
		&& requires(edge_t<Graph> &stored, std::remove_reference_t<std::ranges::range_reference_t<out_edges_t<Graph>>> const &element) { stored = element; }
		&& std::invocable<F &, vertex_id, std::span<visit const>, std::span<visit const>,
						  std::span<edge_t<Graph> const>, std::span<edge_t<Graph> const>>;

	namespace detail {
		/**
		 * @brief The mode-1 (balanced-bisection) centroid engine. For each
		 *        centroid it 2-colors the branches into two groups of at least
		 *        a third of the component each and calls `emit(center, component, n0, edges)` once,
		 *        where `component` is `[center, group0..., group1...]` (parent
		 *        preceding child within each group), `n0 == |group0|`, and `edges`
		 *        is parallel to `component` carrying each vertex's parent-edge
		 *        element (empty unless @p WantEdges). It schedules child frames for
		 *        `group0 + center` and `group1 + center`.
		 *
		 * Every unordered pair at distance >= 2 straddles the two groups in
		 * exactly one event (distance-0/1 pairs never appear — adjacent pairs
		 * bottom out at size-2 subtrees); across the two groups a straddling
		 * pair has distance `depth_a + depth_b`. Works on the original tree —
		 * no ternarization, no fake vertices. Forest-safe (each component
		 * seeds its own recursion). Iterative: an explicit frame stack holds
		 * O(height) vertex lists, so deep trees cannot overflow. Re-rooting a
		 * frame at its centroid reverses parent pointers on the centroid-to-root
		 * path alone, so a level is a size pass, a path walk and one layout
		 * sweep, with no children CSR and no BFS (measured: see
		 * docs/centroid-decomposition.md).
		 * @complexity O(n log n) engine time (each vertex in O(log n) events:
		 * a child frame holds at most two thirds of its parent's branches),
		 * O(n) working memory including pending frames and recycled lists
		 */
		template<bool WantEdges, adjacency_graph Graph, typename Emit>
			requires std::invocable<Emit &, vertex_id, std::span<visit const>, std::size_t,
									std::span<edge_t<Graph> const>>
				  && (!WantEdges || (std::default_initializable<edge_t<Graph>> && std::copyable<edge_t<Graph>> && std::constructible_from<edge_t<Graph>, std::remove_reference_t<std::ranges::range_reference_t<out_edges_t<Graph>>> const &> && requires(edge_t<Graph> &stored, std::remove_reference_t<std::ranges::range_reference_t<out_edges_t<Graph>>> const &element) { stored = element; }))
		constexpr void centroid_bisect_engine(Graph const &g, Emit &&emit) {
			using Edge = std::ranges::range_value_t<out_edges_t<Graph>>;
			std::size_t const n = graph::vertex_count(g);
			atl_assert(n < (uint64(1) << 31));
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

			std::vector<uint32> subtree_size(n), rooted_depth(n), group(n), new_local(n);
			std::vector<uint8> is_on_path(n, 0);
			std::vector<uint32> path, kids;
			std::vector<visit> component;
			std::vector<Edge> component_edges;

			std::vector<frame> stack, freelist;
			auto const take_frame = [&]() -> frame {
				if (freelist.empty()) return frame{};
				frame f = std::move(freelist.back());
				freelist.pop_back();
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
							if constexpr (WantEdges) f.parent_edge.emplace_back(e);
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

				// the branches: the centroid's children (indices above c, since
				// parent[i] < i) keep their subtrees; the up-branch through
				// parent[c] holds the rest, m - subtree_size[c]
				kids.clear();
				for (uint32 i = c + 1; i < m; ++i)
					if (parent[i] == c) kids.push_back(i);
				std::size_t const branch_mass = m - 1;
				std::size_t const up_size = c == 0 ? 0 : m - subtree_size[c];
				// 2-colour the branches so both groups hold at least a third of
				// the mass (every branch is at most half of m, c being a centroid):
				// a branch of a third or more is group 0 alone; otherwise group 0
				// takes branches until it reaches a third, staying below two thirds
				auto const is_third = [&](std::size_t mass) { return mass >= branch_mass / 3 + (branch_mass % 3 != 0); };
				uint32 up_group = 1;
				std::size_t take = 0;
				if (is_third(up_size)) {
					up_group = 0, take = up_size;
					for (uint32 const k : kids) group[k] = 1;
				} else {
					auto const big = std::ranges::find_if(kids, [&](uint32 k) { return is_third(subtree_size[k]); });
					if (big != kids.end()) {
						for (uint32 const k : kids) group[k] = 1;
						group[*big] = 0, take = subtree_size[*big];
					} else {
						if (up_size > 0) up_group = 0, take = up_size;
						for (uint32 const k : kids)
							if (is_third(take))
								group[k] = 1;
							else
								group[k] = 0, take += subtree_size[k];
					}
				}

				// re-rooting at c reverses parent pointers on the c-to-0 path
				// alone: walk it for the path vertices' depths and parent edges
				rooted_depth[c] = 0;
				path.clear();
				for (uint32 x = c, d = 1; x != 0; ++d) {
					uint32 const up = parent[x];
					path.push_back(up);
					rooted_depth[up] = d;
					is_on_path[up] = 1;
					group[up] = up_group;
					x = up;
				}

				// lay out component = [center, group0, group1] AND the two child
				// parent-arrays (group + center) in one sweep with a write cursor
				// per group: path vertices first (each after its new parent), then
				// the rest in index order (each after its unchanged parent)
				std::size_t const group_size[2] = {take, branch_mass - take};
				component.resize(m);
				component[0] = {fr.vertices[c], fr.vertices[c], 0};
				if constexpr (WantEdges) {
					component_edges.resize(m);
					component_edges[0] = Edge{};
				}
				frame child[2] = {take_frame(), take_frame()};
				std::size_t at[2] = {1, 1 + take}, local[2] = {1, 1};
				for (uint32 side = 0; side < 2; ++side) {
					frame &ch = child[side];
					ch.vertices.resize(1 + group_size[side]);
					ch.parent.resize(1 + group_size[side]);
					ch.vertices[0] = fr.vertices[c];
					ch.parent[0] = 0;
					if constexpr (WantEdges) {
						ch.parent_edge.resize(1 + group_size[side]);
						ch.parent_edge[0] = Edge{};
					}
				}
				new_local[c] = 0;
				auto const place = [&](uint32 u, uint32 rooted_parent, uint32 side, Edge const *edge) {
					component[at[side]++] = {fr.vertices[u], fr.vertices[rooted_parent], rooted_depth[u]};
					frame &ch = child[side];
					std::size_t const l = local[side]++;
					new_local[u] = uint32(l);
					ch.vertices[l] = fr.vertices[u];
					ch.parent[l] = new_local[rooted_parent];
					if constexpr (WantEdges) {
						component_edges[at[side] - 1] = *edge;
						ch.parent_edge[l] = *edge;
					}
				};
				for (std::size_t k = 0, below = c; k < path.size(); below = path[k++])
					// the edge below path[k] now hangs path[k] under `below`
					place(path[k], uint32(below), up_group, WantEdges ? &fr.parent_edge[below] : nullptr);
				for (uint32 i = 1; i < m; ++i) {
					if (i == c || is_on_path[i]) continue;
					uint32 const p = parent[i];
					rooted_depth[i] = rooted_depth[p] + 1;
					if (p != c) group[i] = group[p];
					place(i, p, group[i], WantEdges ? &fr.parent_edge[i] : nullptr);
				}
				for (uint32 const x : path) is_on_path[x] = 0;
				std::size_t const n0 = take;
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
	}

	/**
	 * @brief Balanced-bisection (mode-1) centroid decomposition: the natural
	 *        engine for aggregating over all tree paths with a *balanced*
	 *        combine, e.g. one convolution of path-length histograms per
	 *        event. For each centroid it hands the callback the two
	 *        size-balanced branch groups whose cross pairs pass through that
	 *        center:
	 *
	 *            f(center, group_a, group_b) // hops
	 *            f(center, group_a, group_b, edges_a, edges_b) // weighted
	 *
	 * `group_a`/`group_b` are `span<visit const>` (neither contains the
	 * center; `depth` counts hops from it, `parent` precedes its child); a straddling pair `(x in a, y in b)` is at hop
	 * distance `x.depth + y.depth`. The five-argument form additionally passes
	 * the parallel parent-edge spans, so a caller can fold weighted depth
	 * top-down (seed `w[center] = 0`, then `w[v] = w[v.parent] + weight`).
	 * Edge elements retain their initial BFS orientation when a local frame is
	 * rerooted; only symmetric edge payloads, such as weights, can be used unchanged.
	 *
	 * Contract: every unordered pair at distance >= 2 straddles the two groups
	 * in **exactly one** event; distance-0 (self) and distance-1 (adjacent)
	 * pairs never appear and are the caller's to seed. Runs on the original
	 * tree (no ternarization); forest-safe. The spans live in shared scratch —
	 * valid only for the duration of the callback. The weighted form needs a
	 * default-constructible, copyable adjacency element (the engine stores
	 * each vertex's parent edge by value).
	 *
	 * Precedence: a callback modeling both @ref atl::graph::weighted_centroid_bisect_event
	 * and @ref atl::graph::centroid_bisect_event (say a generic lambda with defaulted
	 * trailing parameters) takes the weighted form and pays for the edge
	 * scratch.
	 * @pre @p g is a forest given as a symmetric adjacency (not asserted: the
	 *      engine seeds from a marker BFS, so a cyclic or asymmetric @p g is
	 *      silently treated as its BFS spanning forest)
	 * @tparam Graph Adjacency graph of a forest.
	 * @tparam Event Hop or parent-edge callback, selected by invocability.
	 * @param g Forest whose distance-at-least-two pairs are partitioned.
	 * @param f Receives a center and two temporary groups.
	 * @complexity O(n log n) engine time and total event size over O(n)
	 * events, excluding callback work
	 */
	template<adjacency_graph Graph, typename Event>
		requires centroid_bisect_event<Event> || weighted_centroid_bisect_event<Event, Graph>
	constexpr void centroid_bisect(Graph const &g, Event f) {
		using Edge = edge_t<Graph>;
		constexpr bool is_weighted = weighted_centroid_bisect_event<Event, Graph>;
		detail::centroid_bisect_engine<is_weighted>(
			g, [&](vertex_id c, std::span<visit const> component, std::size_t n0,
				   std::span<Edge const> ec) {
				auto const a = component.subspan(1, n0);
				auto const b = component.subspan(1 + n0);
				if constexpr (is_weighted)
					f(c, a, b, ec.subspan(1, n0), ec.subspan(1 + n0));
				else
					f(c, a, b);
			});
	}

}
