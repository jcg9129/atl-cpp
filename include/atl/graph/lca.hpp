#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::graph {
	namespace detail {
		/** @brief Convert a tree id without implicit narrowing; -1 also denotes none in signed parent ranges. */
		template<typename Value>
			requires std::convertible_to<Value, vertex_id>
		[[nodiscard]] constexpr vertex_id tree_vertex_id(Value &&value) {
			using Raw = std::remove_cvref_t<Value>;
			if constexpr (std::integral<Raw>) {
				if constexpr (std::signed_integral<Raw>) {
					atl_assert(value >= -1 && (value == -1 || uint64(value) <= none));
				} else {
					atl_assert(uint64(value) <= none);
				}
			}
			return vertex_id(std::forward<Value>(value));
		}
	}

	/**
	 * @brief The path-decomposition event of @ref atl::graph::lca::path, one call per
	 * contiguous preorder segment — `f(l, r, reversed)`, `[l, r)`
	 * positions, `reversed` = the path walks the segment backwards.
	 */
	template<typename F>
	concept path_segment_event = std::invocable<F &, uint32, uint32, bool>;

	/**
	 * @brief Lowest common ancestors on a rooted tree or forest, plus the
	 *        tree queries that fall out of the same preprocessing: depth,
	 *        distance, ancestry tests, k-th ancestors and path jumps.
	 *
	 * Construct from an undirected adjacency graph (with a root, or rootless
	 * for a forest — every component gets rooted at its smallest vertex), or
	 * from a parent array (`parents[root] == none` or the root itself).
	 * LCA, distance and path_vertex return `none` across components;
	 * is_ancestor returns false and path requires one component.
	 *
	 * Schieber-Vishkin labels answer LCA in O(1); heavy-path chains answer
	 * ancestor and path_vertex in O(log n). The exposed heavy-first preorder
	 * makes subtrees half-open intervals and heavy chains contiguous.
	 * See docs/lca.md for the algorithm and layout comparisons.
	 * @pre The input is a nonempty forest with fewer than 2^31 vertices.
	 * Graphs are symmetric and acyclic (documented only); parent arrays must
	 * form a forest (asserted). Every query vertex is below size() (asserted).
	 * @complexity O(n) build and storage for a forest. Traversals use heap
	 * buffers, with no recursion per vertex. Allocation failures propagate.
	 */
	class lca {
		std::vector<uint32> parent_, depth_, head_, root_;
		std::vector<uint32> entry_time_, exit_time_, order_; // heavy-first preorder

		/**
		 * @brief The Schieber-Vishkin pair of a vertex: `label` is its path
		 *        label (the paths partition the tree, indexed like a segment
		 *        tree), `ascendant` the bitset of path labels above it. A query
		 *        always reads both, so they share one array entry: one cache
		 *        miss per endpoint instead of two (measured in
		 *        benchmarks/graph/lca.cpp, the layout rows).
		 */
		struct sv_label {
			uint32 label, ascendant;
		};

		std::vector<sv_label> sv_;
		std::vector<uint32> path_head_; // path label -> its topmost vertex
		bool is_single_tree_ = true;	// one component: skip the per-query root check

		/**
		 * @brief Where the root path of @p x enters the Schieber-Vishkin path
		 *        labelled @p target, whose lowest set bit is @p j: @p x itself
		 *        when it already lies on that path, else the parent of the
		 *        head of the deepest path above @p x labelled below bit @p j.
		 */
		[[nodiscard]] constexpr vertex_id path_entry(vertex_id x, uint32 target, uint32 j) const {
			auto const [x_label, x_ascendant] = sv_[x];
			if (x_label == target) return x;
			// the deepest path label above x strictly below bit j
			uint32 const k = uint32(std::bit_width(x_ascendant & ((1u << j) - 1))) - 1;
			uint32 const label = ((x_label >> k) | 1u) << k;
			return parent_[path_head_[label]];
		}

		/**
		 * @brief The parent-array build: children in heavy-first CSR order, a
		 *        BFS for depths and roots, one explicit-stack DFS emitting the
		 *        preorder and chain heads, then the Schieber-Vishkin labels.
		 *        Graph constructors reuse their rooting BFS through build().
		 * @pre `n < 2^31` (asserted): labels are 1-indexed preorder positions
		 *      and the query's `1u << (j + 1)` must stay below 32 bits.
		 */
		constexpr void build_from_parents(std::vector<uint32> const &roots) {
			std::size_t const n = parent_.size();
			atl_assert(n < std::size_t(1) << 31);
			is_single_tree_ = roots.size() == 1;
			depth_.assign(n, 0);
			head_.resize(n);
			root_.resize(n);
			entry_time_.resize(n);
			exit_time_.resize(n);
			order_.clear();
			order_.reserve(n);

			std::vector<uint32> offset(n + 1, 0), child(n - roots.size());
			for (std::size_t v = 0; v < n; ++v)
				if (parent_[v] != none) ++offset[parent_[v] + 1];
			for (std::size_t v = 0; v < n; ++v) offset[v + 1] += offset[v];
			{
				std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
				for (std::size_t v = 0; v < n; ++v)
					if (parent_[v] != none) child[cursor[parent_[v]]++] = uint32(v);
			}
			std::vector<uint32> queue;
			queue.reserve(n);
			for (uint32 const r : roots) {
				root_[r] = r;
				queue.push_back(r);
			}
			for (std::size_t at = 0; at < queue.size(); ++at) {
				uint32 const v = queue[at];
				for (uint32 a = offset[v]; a < offset[v + 1]; ++a) {
					uint32 const w = child[a];
					depth_[w] = depth_[v] + 1;
					root_[w] = root_[v];
					queue.push_back(w);
				}
			}
			atl_assert(queue.size() == n); // parents must form a forest

			std::vector<uint32> size(n, 1);
			for (std::size_t i = n; i-- > 0;) {
				uint32 const v = queue[i];
				uint32 heavy = offset[v];
				for (uint32 a = offset[v]; a < offset[v + 1]; ++a) {
					size[v] += size[child[a]];
					if (size[child[a]] > size[child[heavy]]) heavy = a;
				}
				if (offset[v] != offset[v + 1]) std::swap(child[offset[v]], child[heavy]);
			}

			std::vector<uint32> at(n);
			std::vector<uint32> stack;
			for (uint32 const r : roots) {
				head_[r] = r;
				stack.push_back(r);
				at[r] = offset[r];
				entry_time_[r] = uint32(order_.size());
				order_.push_back(r);
				while (!stack.empty()) {
					uint32 const v = stack.back();
					if (at[v] == offset[v + 1]) {
						exit_time_[v] = uint32(order_.size());
						stack.pop_back();
						continue;
					}
					uint32 const w = child[at[v]++];
					head_[w] = at[v] - 1 == offset[v] ? head_[v] : w;
					at[w] = offset[w];
					entry_time_[w] = uint32(order_.size());
					order_.push_back(w);
					stack.push_back(w);
				}
			}

			// Schieber-Vishkin labels on the 1-indexed preorder: a path label
			// is the number with the most trailing zeros in
			// [entry_time+1, exit_time] (identical along path segments);
			// ascendants accumulate path-label low bits down root paths
			// (preorder = parents first)
			sv_.resize(n);
			path_head_.assign(n + 1, 0);
			for (std::size_t v = 0; v < n; ++v) {
				uint32 const split = uint32(std::bit_width(entry_time_[v] ^ exit_time_[v]));
				sv_[v].label = exit_time_[v] >> (split - 1) << (split - 1);
			}
			for (uint32 const v : order_) {
				uint32 const label = sv_[v].label, low = label & (~label + 1);
				if (parent_[v] == none) {
					sv_[v].ascendant = low;
					path_head_[label] = v;
					continue;
				}
				sv_[v].ascendant = sv_[parent_[v]].ascendant | low;
				if (label != sv_[parent_[v]].label) path_head_[label] = v;
			}
		}

		/**
		 * @brief The graph-constructor build, after the BFS: `order_` holds a BFS order of the
		 *        whole forest (each tree contiguous, parents first) and
		 *        `parent_`, `depth_` and `root_` are filled. One backward sweep
		 *        gives subtree sizes and heavy children; one forward sweep
		 *        places every vertex in the heavy-first preorder (the heavy
		 *        child right after its parent, light children at a per-parent
		 *        cursor advanced by their subtree sizes) and computes chain
		 *        heads and the Schieber-Vishkin labels on the way, since all
		 *        of them need only the parent's values. No children lists, no
		 *        DFS.
		 * @pre `n < 2^31` (asserted): labels are 1-indexed preorder positions
		 *      and the query's `1u << (j + 1)` must stay below 32 bits.
		 */
		constexpr void build(std::size_t tree_count) {
			std::size_t const n = parent_.size();
			atl_assert(n < std::size_t(1) << 31);
			atl_assert(order_.size() == n); // parents must form a forest
			is_single_tree_ = tree_count == 1;

			std::vector<uint32> size(n, 1), heavy(n, none);
			for (std::size_t i = n; i-- > 0;) {
				uint32 const v = order_[i], p = parent_[v];
				if (p == none) continue;
				size[p] += size[v];
				if (heavy[p] == none || size[v] > size[heavy[p]]) heavy[p] = v;
			}

			// heavy-first preorder slots top-down. Schieber-Vishkin labels on
			// the 1-indexed preorder: a path label is the number with the most
			// trailing zeros in [entry_time+1, exit_time] (identical along path
			// segments); ascendants accumulate path-label low bits down root
			// paths
			head_.resize(n);
			entry_time_.resize(n);
			exit_time_.resize(n);
			sv_.resize(n);
			path_head_.assign(n + 1, 0);
			std::vector<uint32> preorder(n), cursor = std::move(size); // size[v] read prior to overwriting cursor[v] is written
			uint32 next_tree = 0;
			for (uint32 const v : order_) {
				uint32 const p = parent_[v], subtree = cursor[v];
				uint32 entry = 0;
				if (p == none) {
					entry = next_tree;
					next_tree += subtree;
					head_[v] = v;
				} else if (heavy[p] == v) {
					entry = entry_time_[p] + 1;
					head_[v] = head_[p];
				} else {
					entry = cursor[p];
					cursor[p] += subtree;
					head_[v] = v;
				}
				entry_time_[v] = entry;
				exit_time_[v] = entry + subtree;
				preorder[entry] = v;
				cursor[v] = entry + 1 + (heavy[v] == none ? 0 : cursor[heavy[v]]);

				uint32 const split = uint32(std::bit_width(entry ^ (entry + subtree)));
				uint32 const label = (entry + subtree) >> (split - 1) << (split - 1);
				uint32 const low = label & (~label + 1);
				if (p == none) {
					sv_[v] = {label, low};
					path_head_[label] = v;
				} else {
					sv_[v] = {label, sv_[p].ascendant | low};
					if (label != sv_[p].label) path_head_[label] = v;
				}
			}
			order_ = std::move(preorder);
		}

		/**
		 * @brief Breadth-first search from @p r over the vertices no tree has
		 *        claimed yet (`root_[w] == none`), appending to `order_` and
		 *        filling `parent_`, `depth_` and `root_`.
		 */
		template<adjacency_graph Graph>
		constexpr void grow(Graph const &g, uint32 r) {
			std::size_t const n = parent_.size();
			root_[r] = r;
			std::size_t head = order_.size();
			order_.push_back(r);
			for (; head < order_.size(); ++head) {
				uint32 const u = order_[head];
				for (auto const &e : graph::out_edges(g, u)) {
					vertex_id const w = graph::target(e);
					atl_assert(w < n);
					if (root_[w] != none) continue;
					root_[w] = r;
					parent_[w] = u;
					depth_[w] = depth_[u] + 1;
					order_.push_back(w);
				}
			}
		}

		/**
		 * @brief Roots the component of @p root (when given) there and every
		 *        other component at its smallest vertex, by BFS.
		 * @return The number of trees.
		 */
		template<adjacency_graph Graph>
		constexpr std::size_t adopt(Graph const &g, vertex_id root) {
			std::size_t const n = graph::vertex_count(g);
			atl_assert(n > 0 && n < std::size_t(1) << 31);
			parent_.assign(n, none);
			depth_.assign(n, 0);
			root_.assign(n, none);
			order_.clear();
			order_.reserve(n);
			std::size_t trees = 0;
			if (root != none) {
				atl_assert(root < n);
				grow(g, root);
				++trees;
			}
			for (uint32 v = 0; v < n; ++v)
				if (root_[v] == none) {
					grow(g, v);
					++trees;
				}
			return trees;
		}

	public:
		/**
		 * @brief Rooted tree. In a forest, @p root's component is rooted at
		 * @p root and every other component at its smallest vertex, so
		 * queries inside any component work and queries across two
		 * components answer `none`.
		 * @tparam Graph Adjacency graph of the forest.
		 * @param g Forest to index without retaining a reference.
		 * @param root Root of its component; must be valid.
		 */
		template<adjacency_graph Graph>
		constexpr lca(Graph const &g, vertex_id root) {
			atl_assert(root != none);
			build(adopt(g, root));
		}

		/** @brief Whole forest: every component rooted at its smallest vertex.
		 * @tparam Graph Adjacency graph of the forest.
		 * @param g Forest to index without retaining a reference.
		 */
		template<adjacency_graph Graph>
		constexpr explicit lca(Graph const &g) {
			build(adopt(g, none));
		}

		/** @brief From a parent array: `parents[v] == none` (or `== v`) marks roots.
		 * @tparam Parents Const-iterable input range with references convertible to vertex_id.
		 * @param parents Parent ids; signed -1 also denotes a root.
		 */
		template<typename Parents>
			requires vertex_range<Parents const> && std::convertible_to<std::ranges::range_reference_t<Parents const>, vertex_id>
		constexpr explicit lca(Parents const &parents) {
			for (auto &&p : parents) parent_.push_back(detail::tree_vertex_id(p));
			std::size_t const n = parent_.size();
			atl_assert(n > 0 && n < std::size_t(1) << 31);
			for (std::size_t v = 0; v < n; ++v)
				if (parent_[v] == v)
					parent_[v] = none;
				else
					atl_assert(parent_[v] == none || parent_[v] < n);
			std::vector<uint32> roots;
			for (std::size_t v = 0; v < n; ++v)
				if (parent_[v] == none) roots.push_back(uint32(v));
			build_from_parents(roots);
		}

		/** @brief Number of indexed vertices. @complexity O(1) */
		constexpr std::size_t size() const { return parent_.size(); }

		/**
		 * @brief The lowest common ancestor, `none` when u and v are in different
		 * components. O(1).
		 */
		[[nodiscard]] constexpr vertex_id operator()(vertex_id u, vertex_id v) const {
			atl_assert(u < size() && v < size());
			if (!is_single_tree_ && root_[u] != root_[v]) return none;
			auto const [u_label, u_ascendant] = sv_[u];
			auto const [v_label, v_ascendant] = sv_[v];
			if (u_label != v_label) {
				// the common path's inlabel: the lowest shared ascendant bit
				// at or above the inlabels' split bit
				uint32 const split = uint32(std::bit_width(u_label ^ v_label)) - 1;
				uint32 const shared = u_ascendant & v_ascendant & ~((1u << split) - 1);
				uint32 const j = uint32(std::countr_zero(shared));
				uint32 const target = ((u_label >> (j + 1)) << (j + 1)) | (1u << j);
				u = path_entry(u, target, j);
				v = path_entry(v, target, j);
			}
			return depth_[u] < depth_[v] ? u : v;
		}

		/** @brief Hop distance from the component root. @pre v < size(). @complexity O(1) */
		[[nodiscard]] constexpr uint32 depth(vertex_id v) const {
			atl_assert(v < size());
			return depth_[v];
		}

		/** @brief Parent vertex, or none at a component root. @pre v < size(). @complexity O(1) */
		[[nodiscard]] constexpr vertex_id parent(vertex_id v) const {
			atl_assert(v < size());
			return parent_[v];
		}

		/** @brief Root of the vertex's component. @pre v < size(). @complexity O(1) */
		[[nodiscard]] constexpr vertex_id root(vertex_id v) const {
			atl_assert(v < size());
			return root_[v];
		}

		/** @brief Edges on the u-v path, `none` across components. O(1). */
		[[nodiscard]] constexpr uint32 distance(vertex_id u, vertex_id v) const {
			vertex_id const w = (*this)(u, v);
			return w == none ? none : depth_[u] + depth_[v] - 2 * depth_[w];
		}

		/**
		 * @brief Is @p a on the root-to-@p b path? (Every vertex is its own
		 * ancestor.) O(1).
		 */
		[[nodiscard]] constexpr bool is_ancestor(vertex_id a, vertex_id b) const {
			atl_assert(a < size() && b < size());
			return entry_time_[a] <= entry_time_[b] && entry_time_[b] < exit_time_[a];
		}

		/**
		 * @brief The k-th ancestor of v (`ancestor(v, 0) == v`), `none` past the
		 * root. O(log n) chain hops; inside a chain the heavy-first order is
		 * depth-contiguous, so the last step is one array lookup.
		 */
		[[nodiscard]] constexpr vertex_id ancestor(vertex_id v, uint32 k) const {
			atl_assert(v < size());
			if (k > depth_[v]) return none;
			while (k > depth_[v] - depth_[head_[v]]) {
				k -= depth_[v] - depth_[head_[v]] + 1;
				v = parent_[head_[v]];
			}
			return order_[entry_time_[v] - k];
		}

		/**
		 * @brief The k-th vertex on the u -> v path (`path_vertex(u, v, 0) == u`), `none`
		 * past v or across components. O(log n).
		 */
		[[nodiscard]] constexpr vertex_id path_vertex(vertex_id u, vertex_id v, uint32 k) const {
			vertex_id const w = (*this)(u, v);
			if (w == none) return none;
			uint32 const du = depth_[u] - depth_[w], dv = depth_[v] - depth_[w];
			if (k <= du) return ancestor(u, k);
			if (k > du + dv) return none;
			return ancestor(v, du + dv - k);
		}

		/**
		 * @brief Decompose the u -> v path into its contiguous heavy-first
		 * preorder intervals: `f(l, r, reversed)` per segment, `[l, r)`
		 * half-open positions, segments delivered **in path order** from u
		 * to v. `reversed` means the path walks that segment from position
		 * r - 1 down to l (the u-side of the path runs against the
		 * preorder) — commutative consumers ignore it, non-commutative ones
		 * query a mirrored structure. `includes_top = false` omits the topmost
		 * vertex (the LCA) — the mode for edge values stored at their child
		 * endpoint's slot. Empty segments are never emitted.
		 *
		 * This is the composition primitive for path updates/aggregates:
		 * point a range structure at the preorder (`a[entry_time(v)] = value(v)`)
		 * and every path or subtree operation is O(log n) range operations
		 * on it — subtree ops directly via [entry_time, exit_time), path ops through
		 * here. @pre u and v are in the same component (asserted).
		 * @tparam F Invocable as f(uint32, uint32, bool), including reference wrappers.
		 * @param u First vertex of the path.
		 * @param v Last vertex of the path.
		 * @param f Receives each nonempty half-open segment and its direction.
		 * @param includes_top Whether to emit the LCA vertex.
		 * @complexity O(log n) segments
		 */
		template<path_segment_event F>
		constexpr void path(vertex_id u, vertex_id v, F f, bool includes_top = true) const {
			atl_assert(u < size() && v < size());
			atl_assert(is_single_tree_ || root_[u] == root_[v]);
			// u-side segments emit immediately (encounter order = path
			// order); v-side segments buffer and flush backwards. One side
			// crosses at most log2(n) light edges, so the buffer is tiny.
			std::pair<uint32, uint32> tail[33]{};
			std::size_t tails = 0;
			while (head_[u] != head_[v]) {
				if (depth_[head_[u]] >= depth_[head_[v]]) {
					f(std::uint32_t{entry_time_[head_[u]]}, std::uint32_t{entry_time_[u] + 1}, true);
					u = parent_[head_[u]];
				} else {
					atl_assert(tails < 33);
					tail[tails++] = {entry_time_[head_[v]], entry_time_[v] + 1};
					v = parent_[head_[v]];
				}
			}
			// the shared chain: whichever endpoint is shallower is the LCA
			if (depth_[u] <= depth_[v]) {
				uint32 const l = entry_time_[u] + !includes_top;
				if (l < entry_time_[v] + 1) f(std::uint32_t{l}, std::uint32_t{entry_time_[v] + 1}, false);
			} else {
				uint32 const l = entry_time_[v] + !includes_top;
				if (l < entry_time_[u] + 1) f(std::uint32_t{l}, std::uint32_t{entry_time_[u] + 1}, true);
			}
			while (tails != 0) {
				--tails;
				f(std::uint32_t{tail[tails].first}, std::uint32_t{tail[tails].second}, false);
			}
		}

		/**
		 * @brief Heavy-first preorder hooks for structures composed on top:
		 * subtree(v) = order positions [entry_time(v), exit_time(v)), chains contiguous.
		 */
		[[nodiscard]] constexpr uint32 entry_time(vertex_id v) const {
			atl_assert(v < size());
			return entry_time_[v];
		}

		/** @brief Exclusive end of the vertex's subtree in order(). @pre v < size(). @complexity O(1) */
		[[nodiscard]] constexpr uint32 exit_time(vertex_id v) const {
			atl_assert(v < size());
			return exit_time_[v];
		}

		/** @brief Heavy-first preorder; borrowed until assignment or destruction. @complexity O(1) */
		[[nodiscard]] constexpr std::vector<uint32> const &order() const { return order_; }
	};

	namespace detail {
		/**
		 * @brief The Tarjan offline-LCA engine behind both @ref atl::graph::offline_lca
		 *        entries. A query is answered when its second endpoint is
		 *        entered and the first is already marked with the same tree:
		 *        rooted runs mark entered vertices 1 (one tree, unvisited 0);
		 *        rootless runs mark each vertex with its tree's root, since
		 *        `parent[]` keeps finished trees collapsed onto their roots and
		 *        a cross-tree hit must not read one.
		 * @param root The single DFS root when @p IsRooted, unused otherwise.
		 */
		template<bool IsRooted, detail::scannable_graph Graph, vertex_pair_range Queries>
			requires vertex_pair_range<Queries const>
				  && requires(std::remove_reference_t<std::ranges::range_reference_t<Queries const>> const &q) {
						 { std::get<0>(q) } -> std::convertible_to<vertex_id>;
						 { std::get<1>(q) } -> std::convertible_to<vertex_id>;
					 }
		[[nodiscard]] constexpr std::vector<vertex_id> offline_lca_run(Graph const &g, vertex_id root, Queries const &queries) {
			std::size_t const n = graph::vertex_count(g);
			auto const query_count = std::ranges::size(queries);
			atl_assert(query_count <= none / 2);
			std::size_t const m = std::size_t(query_count);
			atl_assert(n <= none && n < (std::numeric_limits<std::size_t>::max)()
					   && m <= none / 2);

			std::vector<uint32> offset(n + 1, 0), slot(2 * m), which(2 * m);
			for (auto const &[first, second] : queries) {
				vertex_id const u = detail::tree_vertex_id(first), v = detail::tree_vertex_id(second);
				atl_assert(u < n && v < n);
				++offset[u + 1];
				++offset[v + 1];
			}
			for (std::size_t v = 0; v < n; ++v) offset[v + 1] += offset[v];
			{
				std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
				uint32 i = 0;
				for (auto const &[first, second] : queries) {
					vertex_id const u = detail::tree_vertex_id(first), v = detail::tree_vertex_id(second);
					slot[cursor[u]] = v;
					which[cursor[u]++] = i;
					slot[cursor[v]] = u;
					which[cursor[v]++] = i;
					++i;
				}
			}

			// Directed union keeps the representative equal to the lowest unfinished
			// ancestor, avoiding the extra ancestor map needed by arbitrary-root DSU.
			using mark_type = std::conditional_t<IsRooted, uint8, uint32>;
			std::vector<uint32> parent(n);
			std::vector<mark_type> tree(n);
			if constexpr (!IsRooted) std::ranges::fill(tree, none);
			std::vector<vertex_id> answer(m, none);
			for (std::size_t v = 0; v < n; ++v) parent[v] = uint32(v);
			auto const find = [&](uint32 v) {
				while (parent[v] != v) v = parent[v] = parent[parent[v]];
				return v;
			};
			auto const enter = [&](vertex_id v, vertex_id p, uint32, uint32) {
				if constexpr (IsRooted)
					tree[v] = 1;
				else
					tree[v] = p == v ? v : tree[p];
				// marked ahead of the scan, so (v, v) answers itself
				for (uint32 a = offset[v]; a < offset[v + 1]; ++a)
					if (tree[slot[a]] == tree[v]) answer[which[a]] = find(slot[a]);
			};
			auto const exit = [&](vertex_id v, vertex_id p) { parent[v] = p; };
			if constexpr (IsRooted)
				graph::dfs_visit(g, root, enter, graph::skip, exit);
			else
				graph::dfs_visit(g, enter, graph::skip, exit);
			return answer;
		}
	}

	/**
	 * @brief Offline LCA (Tarjan): answer a whole batch of queries during a
	 *        single DFS — finished subtrees collapse into their roots via
	 *        DSU, and when a query's second endpoint is reached, the first
	 *        endpoint's representative is the answer. No structure survives:
	 *        prefer this when all queries are known up front and @ref atl::graph::lca
	 *        would be built for one pass. A query with an endpoint
	 *        unreachable from @p root answers `none` — including one whose
	 *        endpoints share another component, where @ref atl::graph::lca would answer
	 *        (the rootless overload answers those).
	 * @pre The graph is a symmetric forest (documented only); `n <= none`,
	 * `n + 1` fits size_t and `q <= none / 2`. Every query endpoint is a vertex of
	 *      @p g (asserted).
	 * @complexity O((n + q) alpha) time, O(n + q) space
	 * @tparam Graph Resumable adjacency graph.
	 * @tparam Queries Const sized forward range of vertex pairs.
	 * @param g Forest to traverse.
	 * @param queries Query pairs in answer order.
	 * @return One ancestor id per query, or none for an unanswerable pair.
	 * @param root Root of the only component traversed.
	 */
	template<detail::scannable_graph Graph, vertex_pair_range Queries>
		requires vertex_pair_range<Queries const>
			  && requires(std::remove_reference_t<std::ranges::range_reference_t<Queries const>> const &q) {
					 { std::get<0>(q) } -> std::convertible_to<vertex_id>;
					 { std::get<1>(q) } -> std::convertible_to<vertex_id>;
				 }
	[[nodiscard]] constexpr std::vector<vertex_id> offline_lca(Graph const &g, vertex_id root, Queries const &queries) {
		atl_assert(root < graph::vertex_count(g));
		return detail::offline_lca_run<true>(g, root, queries);
	}

	/**
	 * @brief Offline LCA over a whole forest: every tree is rooted at its
	 *        lowest-numbered vertex (the all-components DFS), so it agrees
	 *        with @ref atl::graph::lca built on the same forest. Queries whose endpoints
	 *        lie in different trees answer `none`.
	 * @pre The graph is a symmetric forest (documented only); `n <= none`,
	 * `n + 1` fits size_t and `q <= none / 2`. Every query endpoint is a vertex of
	 *      @p g (asserted).
	 * @complexity O((n + q) alpha) time, O(n + q) space
	 * @tparam Graph Resumable adjacency graph.
	 * @tparam Queries Const sized forward range of vertex pairs.
	 * @param g Forest to traverse.
	 * @param queries Query pairs in answer order.
	 * @return One ancestor id per query, or none for an unanswerable pair.
	 */
	template<detail::scannable_graph Graph, vertex_pair_range Queries>
		requires vertex_pair_range<Queries const>
			  && requires(std::remove_reference_t<std::ranges::range_reference_t<Queries const>> const &q) {
					 { std::get<0>(q) } -> std::convertible_to<vertex_id>;
					 { std::get<1>(q) } -> std::convertible_to<vertex_id>;
				 }
	[[nodiscard]] constexpr std::vector<vertex_id> offline_lca(Graph const &g, Queries const &queries) {
		return detail::offline_lca_run<false>(g, vertex_id(0), queries);
	}
}
