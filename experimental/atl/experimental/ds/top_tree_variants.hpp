#pragma once

/**
 * @file
 * @brief Race losers of @ref atl::top_tree (vertex-cluster: compress and
 *        rake splay trees), every one the full class over the shared
 *        implementation with one axis changed:
 *  - node layout: split links/payload (library) vs interleaved;
 *  - splaying after the one top-down chain push: without the engine's
 *    per-tree push pass (the public `splay_engine::splay_pushed`, library)
 *    vs `splay_engine::splay`, which re-walks each tree's path;
 *  - a new light child: inserted at the rake root (library) vs joined at
 *    the end of the rake tree;
 *  - the edge-cluster shape on the edge tier: edges are
 *    compress-tree leaves instead of edge-as-vertex compress nodes;
 *  - path and edge-side operations: two accesses around the lca with the
 *    root side read as a hanging child (library) vs re-rooting and back;
 *  - a cached one-vertex cluster per compress node vs recomputing it in
 *    every pull (library).
 * (library reversed `Path` vs a user `Reverse`) is the public `Reverse`
 * slot itself, raced on the diameter and affine shapes directly.
 *
 * benchmarks/ds/top_tree.cpp, Xeon Platinum 8168, GCC 15.2,
 * RelWithDebInfo, assertions off, 2026-10-25, plain -O2 and -march=native:
 * per-splay push and edge leaves lose the path-heavy workloads (see aliases).
 * Interleaved layout, rake append, rerooting and cached vertex clusters need
 * an x86-64 race; the benchmark includes every contender.
 */

#include <atl/ds/detail/splay_layout.hpp>
#include <atl/ds/top_tree.hpp>
#include <atl/types/alias.hpp>
#include <concepts>

namespace atl::experimental {
	/**
	 * @brief @ref atl::top_tree with the layout, the chain-push splay
	 *        and the rake insertion chosen at compile time (the template
	 *        arguments after them as in the library class).
	 */
	template<atl::detail::splay_layout Layout, bool IsChainPushed, bool IsRakeRootInsert, bool IsEdgeLeaf, bool IsEvertForm,
			 bool IsOwnCached, typename Value,
			 typename Path, typename Point, typename Compress, typename Rake, typename AddVertex, typename AddEdge,
			 typename Reverse = void, typename Tag = void, typename Mapping = void, typename MapPath = void,
			 typename MapPoint = void, typename Compose = void, typename ValuePlacement = void>
		requires atl::detail::top_tree_algebra<Value, Path, Point, Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping,
											   MapPath, MapPoint, Compose, ValuePlacement>
			  && (!IsEdgeLeaf || std::same_as<ValuePlacement, atl::edge_values_t>)
	class top_tree_by_configuration
		: public atl::detail::top_tree_impl<Layout, IsChainPushed, IsRakeRootInsert, IsEdgeLeaf, IsEvertForm, IsOwnCached, Value,
											Path, Point,
											Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath, MapPoint, Compose,
											ValuePlacement> {
		using base = atl::detail::top_tree_impl<Layout, IsChainPushed, IsRakeRootInsert, IsEdgeLeaf, IsEvertForm, IsOwnCached, Value,
												Path, Point,
												Compress, Rake, AddVertex, AddEdge, Reverse, Tag, Mapping, MapPath, MapPoint,
												Compose, ValuePlacement>;

	public:
		using base::base;
	};

	/**
	 * @brief The interleaved layout (one record per node). Within noise or
	 *        behind the split layout in local aarch64 runs; not in the x86-64
	 *        decision record.
	 */
	template<typename... Args>
	using top_tree_by_interleaved_layout = top_tree_by_configuration<atl::splay_layout_interleaved, true, true, false, false, false, Args...>;

	/**
	 * @brief Re-push each splay tree's ancestor path after the cross-tree push.
	 *        benchmarks/ds/top_tree.cpp, Xeon Platinum 8168, GCC 15.2,
	 *        RelWithDebInfo, assertions off, 2026-10-25, N=Q=5e5:
	 *        path sums take 975 ms with -O2 versus library 864 ms;
	 *        -O2 -march=native takes 975 versus 824 ms.
	 */
	template<typename... Args>
	using top_tree_by_per_splay_push = top_tree_by_configuration<atl::splay_layout_split, false, true, false, false, false, Args...>;

	/**
	 * @brief A new light child joins at the end of the rake tree. Within
	 *        noise or behind in local aarch64 runs;
	 *        not in the x86-64 decision record.
	 */
	template<typename... Args>
	using top_tree_by_rake_append = top_tree_by_configuration<atl::splay_layout_split, true, false, false, false, false, Args...>;

	/**
	 * @brief Edge tier only: the edge-cluster shape. Edges are
	 *        leaves of the compress trees (base clusters), vertices the internal
	 *        compress nodes (a compress at vertex m is
	 *        `compress(compress(A, add_vertex(light(m), m)), B)`), so only vertices
	 *        splay and a path's splay tree holds half the nodes of the
	 *        edge-as-vertex representation; an edge update first raises its lower endpoint
	 *        (one or two accesses) instead of accessing the edge itself. Rake trees
	 *        stay node-oriented. On the vertex tier the edge leaves would
	 *        hold identities and add n - 1 leaves without additional information.
	 *
	 * Lost (ds-top_tree, 2026-10-25, Xeon Platinum 8168, -march=native,
	 * N=Q=5e5; library edge-as-vertex first): weighted diameter on a path 1826
	 * vs 2489 ms, caterpillar 2204 vs 2540, random 2952 vs 3053;
	 * it wins only edge-replacement churn (4311 vs 4114) and marked-vertex
	 * distance sums (1719 vs 1617), in those workloads. Plain -O2 agrees (path 1806 vs
	 * 2458, caterpillar 2160 vs 2486). The library uses edge-as-vertex storage;
	 * edge weights use `edge_values_t`.
	 */
	template<typename... Args>
	using top_tree_by_edge_leaves = top_tree_by_configuration<atl::splay_layout_split, true, true, true, false, false, Args...>;

	/**
	 * @brief Path and edge-side operations by re-rooting at one end and
	 *        back (find_root, evert, access, evert: four accesses), against the library two accesses around the lca.
	 *        Within noise or behind in local aarch64 runs; not in the x86-64
	 *        decision record.
	 */
	template<typename... Args>
	using top_tree_by_rerooting = top_tree_by_configuration<atl::splay_layout_split, true, true, false, true, false, Args...>;

	/**
	 * @brief Each compress node caches its one-vertex cluster
	 *        `add_vertex(point(mid), value)`, refreshed where its value or rake
	 *        root changes and mapped by tags, so a pull neither reads the rake
	 *        pool nor calls `add_vertex` (one more `Path` per node). Within
	 *        noise or behind in local aarch64 runs; not in the x86-64 decision
	 *        record.
	 */
	template<typename... Args>
	using top_tree_by_cached_vertex_cluster = top_tree_by_configuration<atl::splay_layout_split, true, true, false, false, true, Args...>;
}
