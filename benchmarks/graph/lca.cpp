/**
 * @file
 * @brief Build and query latency for binary lifting, head jumping, Euler/preorder RMQ,
 * block RMQ, adjacency-preorder labels, children-CSR/DFS builds and label layouts.
 * Shared query loops measure throughput on random, path, star, caterpillar,
 * binary and forest shapes, plus n=4e6. Untimed queries check agreement.
 * Xeon Platinum 8280 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
 * -O2 / -O2 -march=native, 2026-10-15: random 5e5 library queries
 * 13.3 / 8.9 ms; by binary lifting 37.0 / 37.0 ms. Full tables: docs/lca.md.
 */
#include <atl/experimental/graph/lca_variants.hpp>
#include <atl/graph/lca.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/counting_sort.hpp>
#include <algorithm>
#include <bit>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl::experimental::children_dfs {
	using namespace atl::graph;

	/** @brief Build child CSR, compute a second BFS and emit heavy-first order with an explicit DFS stack. */
	class lca_by_children_dfs {
		std::vector<uint32> parent_, depth_, head_, root_;
		std::vector<uint32> entry_time_, exit_time_, order_; // heavy-first preorder
		// Schieber-Vishkin over the preorder: path_label_ partitions the
		// tree into segment-tree-indexed paths, ascendant_ collects the path
		// labels above a vertex as a bitset, path_head_ maps a path label to
		// its topmost vertex
		std::vector<uint32> path_label_, ascendant_, path_head_;
		bool is_single_tree_ = true; // one component: skip the per-query root check

		constexpr vertex_id path_entry(vertex_id x, uint32 target, uint32 j) const {
			if (path_label_[x] == target) return x;
			// the deepest path label above x strictly below bit j
			uint32 const k = uint32(std::bit_width(ascendant_[x] & ((1u << j) - 1))) - 1;
			uint32 const label = ((path_label_[x] >> k) | 1u) << k;
			return parent_[path_head_[label]];
		}

		constexpr void build(std::vector<uint32> const &roots) {
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
			path_label_.resize(n);
			ascendant_.resize(n);
			path_head_.assign(n + 1, 0);
			for (std::size_t v = 0; v < n; ++v) {
				uint32 const split = uint32(std::bit_width(entry_time_[v] ^ exit_time_[v]));
				path_label_[v] = exit_time_[v] >> (split - 1) << (split - 1);
			}
			for (uint32 const v : order_) {
				uint32 const low = path_label_[v] & (~path_label_[v] + 1);
				if (parent_[v] == none) {
					ascendant_[v] = low;
					path_head_[path_label_[v]] = v;
					continue;
				}
				ascendant_[v] = ascendant_[parent_[v]] | low;
				if (path_label_[v] != path_label_[parent_[v]]) path_head_[path_label_[v]] = v;
			}
		}

		template<atl::graph::detail::scannable_graph Graph>
		constexpr std::vector<uint32> adopt(Graph const &g, std::vector<uint32> roots) {
			std::size_t const n = graph::vertex_count(g);
			atl_assert(n > 0);
			parent_.assign(n, none);
			dense_marker marks(n);
			if (roots.empty())
				for (uint32 v = 0; v < n; ++v) {
					// rootless: every component claims its smallest vertex
					bool is_fresh = true;
					for (auto const &[w, p, d] : graph::bfs(g, v, marks)) {
						if (is_fresh) roots.push_back(v);
						is_fresh = false;
						if (w != v) parent_[w] = p;
					}
				}
			else {
				for (uint32 const r : roots)
					for (auto const &[w, p, d] : graph::bfs(g, r, marks))
						if (w != r) parent_[w] = p;
				// components the given roots do not reach are rooted at their
				// smallest vertex, as in the rootless form: every vertex must
				// belong to some rooted tree when build() runs
				for (uint32 v = 0; v < n; ++v) {
					if (marks.visited(v)) continue;
					roots.push_back(v);
					for (auto const &[w, p, d] : graph::bfs(g, v, marks))
						if (w != v) parent_[w] = p;
				}
			}
			return roots;
		}

	public:
		/** @brief Rooted tree. In a forest, @p root's component is rooted at
		 * @p root and every other component at its smallest vertex, so
		 * queries inside any component work and queries across two
		 * components answer `none`. */
		template<atl::graph::detail::scannable_graph Graph>
		constexpr lca_by_children_dfs(Graph const &g, vertex_id root) {
			build(adopt(g, {root}));
		}

		/** @brief Whole forest: every component rooted at its smallest vertex. */
		template<atl::graph::detail::scannable_graph Graph>
		constexpr explicit lca_by_children_dfs(Graph const &g) {
			build(adopt(g, {}));
		}

		/** @brief From a parent array: `parents[v] == none` (or `== v`) marks roots. */
		template<typename Parents>
			requires vertex_range<Parents const> && std::convertible_to<std::ranges::range_reference_t<Parents const>, vertex_id>
		constexpr explicit lca_by_children_dfs(Parents const &parents) {
			for (auto &&p : parents) parent_.push_back(atl::graph::detail::tree_vertex_id(p));
			std::vector<uint32> roots;
			for (std::size_t v = 0; v < parent_.size(); ++v)
				if (parent_[v] == none || parent_[v] == v) {
					parent_[v] = none;
					roots.push_back(uint32(v));
				} else {
					atl_assert(parent_[v] < parent_.size());
				}
			atl_assert(!parent_.empty());
			build(roots);
		}

		constexpr std::size_t size() const { return parent_.size(); }

		/** @brief The lowest common ancestor, `none` when u and v are in different
		 * components. O(1). */
		constexpr vertex_id operator()(vertex_id u, vertex_id v) const {
			if (!is_single_tree_ && root_[u] != root_[v]) return none;
			if (path_label_[u] != path_label_[v]) {
				// the common path's inlabel: the lowest shared ascendant bit
				// at or above the inlabels' split bit
				uint32 const split = uint32(std::bit_width(path_label_[u] ^ path_label_[v])) - 1;
				uint32 const shared = ascendant_[u] & ascendant_[v] & ~((1u << split) - 1);
				uint32 const j = uint32(std::countr_zero(shared));
				uint32 const target = ((path_label_[u] >> (j + 1)) << (j + 1)) | (1u << j);
				u = path_entry(u, target, j);
				v = path_entry(v, target, j);
			}
			return depth_[u] < depth_[v] ? u : v;
		}
	};

	template<bool IsRooted, atl::graph::detail::scannable_graph Graph, vertex_pair_range Queries>
	constexpr std::vector<vertex_id> offline_lca_by_query_buckets(Graph const &g, vertex_id root, Queries const &queries) {
		std::size_t const n = graph::vertex_count(g);
		std::size_t const m = std::ranges::size(queries);
		atl_assert(n <= none && n < (std::numeric_limits<std::size_t>::max)() && m <= none / 2);
		std::vector<uint32> offset(n + 1, 0), slot(2 * m), which(2 * m);
		for (auto const &[u, v] : queries) {
			atl_assert(vertex_id(u) < n && vertex_id(v) < n);
			++offset[u + 1];
			++offset[v + 1];
		}
		for (std::size_t v = 0; v < n; ++v) offset[v + 1] += offset[v];
		{
			std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
			uint32 i = 0;
			for (auto const &[u, v] : queries) {
				slot[cursor[u]] = v;
				which[cursor[u]++] = i;
				slot[cursor[v]] = u;
				which[cursor[v]++] = i;
				++i;
			}
		}

		// deliberately not atl::dsu: Tarjan's union is *directed* (a finished
		// subtree hangs under its parent, so the representative IS the lowest
		// unfinished ancestor), which fuses away the anc[] indirection an
		// arbitrary-representative dsu would need — measured at-or-ahead of
		// dsu + anc[] on every benchmark shape (10-20% on three of four)
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

namespace g = atl::graph;
namespace ex = atl::experimental;
using atl::uint32;
using atl::uint64;

namespace {
	/**
	 * @brief The library Schieber-Vishkin query over three memory layouts,
	 *        rebuilt from a `g::lca`'s public preorder: @p Layout 0
	 *        keeps separate label/ascendant arrays, 1 interleaves
	 *        `{path_label, ascendant}`, 2 folds `depth` in as well.
	 */
	template<int Layout>
		requires(Layout >= 0 && Layout <= 2)
	class lca_by_label_layout {
		struct pair_entry {
			uint32 label, ascendant;
		};

		struct triple_entry {
			uint32 label, ascendant, depth;
		};

		std::vector<uint32> label_, ascendant_, depth_, parent_, path_head_, root_;
		std::vector<pair_entry> pair_;
		std::vector<triple_entry> triple_;
		bool is_forest_ = false;

		uint32 label(uint32 x) const {
			if constexpr (Layout == 0)
				return label_[x];
			else if constexpr (Layout == 1)
				return pair_[x].label;
			else
				return triple_[x].label;
		}

		uint32 ascendant(uint32 x) const {
			if constexpr (Layout == 0)
				return ascendant_[x];
			else if constexpr (Layout == 1)
				return pair_[x].ascendant;
			else
				return triple_[x].ascendant;
		}

		uint32 depth(uint32 x) const {
			if constexpr (Layout == 2)
				return triple_[x].depth;
			else
				return depth_[x];
		}

		uint32 path_entry(uint32 x, uint32 target, uint32 j) const {
			uint32 const lx = label(x);
			if (lx == target) return x;
			uint32 const k = uint32(std::bit_width(ascendant(x) & ((1u << j) - 1))) - 1;
			return parent_[path_head_[((lx >> k) | 1u) << k]];
		}

	public:
		explicit lca_by_label_layout(g::lca const &a) {
			std::size_t const n = a.size();
			label_.resize(n);
			ascendant_.resize(n);
			depth_.resize(n);
			parent_.resize(n);
			root_.resize(n);
			path_head_.assign(n + 1, 0);
			for (uint32 v = 0; v < n; ++v) {
				uint32 const split = uint32(std::bit_width(a.entry_time(v) ^ a.exit_time(v)));
				label_[v] = a.exit_time(v) >> (split - 1) << (split - 1);
				depth_[v] = a.depth(v);
				parent_[v] = a.parent(v);
				root_[v] = a.root(v);
				is_forest_ |= root_[v] != root_[0];
			}
			for (uint32 const v : a.order()) {
				uint32 const low = label_[v] & (~label_[v] + 1), p = parent_[v];
				ascendant_[v] = p == g::none ? low : ascendant_[p] | low;
				if (p == g::none || label_[v] != label_[p]) path_head_[label_[v]] = v;
			}
			if constexpr (Layout == 1)
				for (uint32 v = 0; v < n; ++v) pair_.push_back({label_[v], ascendant_[v]});
			if constexpr (Layout == 2)
				for (uint32 v = 0; v < n; ++v) triple_.push_back({label_[v], ascendant_[v], depth_[v]});
			if constexpr (Layout != 0) {
				label_ = {};
				ascendant_ = {};
			}
			if constexpr (Layout == 2) depth_ = {};
		}

		uint32 operator()(uint32 u, uint32 v) const {
			if (is_forest_ && root_[u] != root_[v]) return g::none;
			uint32 const lu = label(u), lv = label(v);
			if (lu != lv) {
				uint32 const split = uint32(std::bit_width(lu ^ lv)) - 1;
				uint32 const shared = ascendant(u) & ascendant(v) & ~((1u << split) - 1);
				uint32 const j = uint32(std::countr_zero(shared));
				uint32 const target = ((lu >> (j + 1)) << (j + 1)) | (1u << j);
				u = path_entry(u, target, j);
				v = path_entry(v, target, j);
			}
			return depth(u) < depth(v) ? u : v;
		}
	};

	/** @brief The adjacency lists of a parent array (`none` marks roots). */
	std::vector<std::vector<uint32>> adjacency(std::vector<uint32> const &parent) {
		std::vector<std::vector<uint32>> adj(parent.size());
		for (uint32 v = 0; v < parent.size(); ++v)
			if (parent[v] != g::none) {
				adj[parent[v]].push_back(v);
				adj[v].push_back(parent[v]);
			}
		return adj;
	}

	std::vector<std::pair<uint32, uint32>> random_queries(std::mt19937 &rng, std::size_t n, std::size_t q) {
		std::vector<std::pair<uint32, uint32>> queries(q);
		for (auto &[u, v] : queries) {
			u = uint32(rng() % n);
			v = uint32(rng() % n);
		}
		return queries;
	}

	template<typename Query>
		requires std::is_invocable_r_v<uint32, Query const &, uint32, uint32>
	uint64 throughput(Query const &lca, std::vector<std::pair<uint32, uint32>> const &queries) {
		uint64 sum = 0;
		for (auto const &[u, v] : queries) sum += lca(u, v);
		return sum;
	}

	/** @brief Compare the graph build with children-CSR/DFS and label-layout alternatives. */
	void library_rows(std::string const &name, std::vector<uint32> const &parent,
					  std::vector<std::vector<uint32>> const &adj,
					  std::vector<std::pair<uint32, uint32>> const &queries) {
		g::lca const sh(adj);
		ex::children_dfs::lca_by_children_dfs const comparison(adj);
		lca_by_label_layout<0> const soa(sh);
		lca_by_label_layout<1> const aos(sh);
		lca_by_label_layout<2> const aos_depth(sh);
		auto const offline = g::offline_lca(adj, queries);
		auto const offline_by_children_dfs = ex::children_dfs::offline_lca_by_query_buckets<false>(adj, 0, queries);
		for (std::size_t i = 0; i < 1000; ++i) {
			auto const [u, v] = queries[i];
			uint32 const want = comparison(u, v);
			REQUIRE(sh(u, v) == want);
			REQUIRE(soa(u, v) == want);
			REQUIRE(aos(u, v) == want);
			REQUIRE(aos_depth(u, v) == want);
			REQUIRE(offline[i] == want);
			REQUIRE(offline_by_children_dfs[i] == want);
		}

		BENCHMARK(name + " | build SV (library)") { return g::lca(adj)(0, 1); };
		BENCHMARK(name + " | build SV (by children DFS)") { return ex::children_dfs::lca_by_children_dfs(adj)(0, 1); };
		BENCHMARK(name + " | build SV rooted (library)") { return g::lca(adj, 0)(0, 1); };
		BENCHMARK(name + " | build SV rooted (by children DFS)") { return ex::children_dfs::lca_by_children_dfs(adj, 0)(0, 1); };
		BENCHMARK(name + " | build SV from parents (library)") { return g::lca(parent)(0, 1); };
		BENCHMARK(name + " | build SV from parents (by children DFS)") { return ex::children_dfs::lca_by_children_dfs(parent)(0, 1); };
		BENCHMARK(name + " | children CSR hand-rolled") {
			std::size_t const n = parent.size();
			std::vector<uint32> offset(n + 1, 0), child(n);
			for (std::size_t v = 0; v < n; ++v)
				if (parent[v] != g::none) ++offset[parent[v] + 1];
			for (std::size_t v = 0; v < n; ++v) offset[v + 1] += offset[v];
			std::vector<uint32> cursor(offset.begin(), offset.end() - 1);
			for (std::size_t v = 0; v < n; ++v)
				if (parent[v] != g::none) child[cursor[parent[v]]++] = uint32(v);
			return child[n / 2] + offset[n / 2];
		};
		BENCHMARK(name + " | children CSR bucket_by_key") {
			std::size_t const n = parent.size();
			std::vector<uint32> offset(n + 2), child(n);
			atl::detail::bucket_by_key(
				n + 1, [&](std::size_t v) { return parent[v] == g::none ? n : std::size_t(parent[v]); }, offset,
				child);
			return child[n / 2] + offset[n / 2];
		};
		BENCHMARK(name + " | 5e5 queries SV (library)") { return throughput(sh, queries); };
		BENCHMARK(name + " | 5e5 queries SV layout by separate labels") { return throughput(soa, queries); };
		BENCHMARK(name + " | 5e5 queries SV by label pairs") { return throughput(aos, queries); };
		BENCHMARK(name + " | 5e5 queries SV by label depth triples") {
			return throughput(aos_depth, queries);
		};
		BENCHMARK(name + " | offline batch (Tarjan, library)") { return g::offline_lca(adj, queries).back(); };
		BENCHMARK(name + " | offline batch (Tarjan, by query buckets)") {
			return ex::children_dfs::offline_lca_by_query_buckets<false>(adj, 0, queries).back();
		};
	}
}

TEST_CASE("lca contenders", "[lca]") {
	std::mt19937 rng(1);
	std::size_t const n = 500'000, q = 500'000;

	auto race = [&](char const *label, std::vector<uint32> const &parent) {
		std::string const name = label;
		auto const adj = adjacency(parent);
		auto const queries = random_queries(rng, parent.size(), q);

		ex::lca_by_binary_lifting bl(adj, 0);
		ex::lca_by_head_jumping hld(adj, 0);
		ex::lca_by_euler_sparse_table es(adj, 0);
		ex::lca_by_euler_direct_rmq er(adj, 0);
		ex::lca_by_preorder_sparse_table ps(adj, 0);
		ex::lca_by_preorder_direct_rmq pr(adj, 0);
		ex::lca_by_adjacency_preorder sv(adj, 0);
		ex::lca_by_farach_colton_bender fcb(adj, 0);
		g::lca sh(adj);
		for (std::size_t i = 0; i < 1000; ++i) {
			auto const [u, v] = queries[i];
			uint32 const want = bl(u, v);
			REQUIRE(hld(u, v) == want);
			REQUIRE(es(u, v) == want);
			REQUIRE(er(u, v) == want);
			REQUIRE(ps(u, v) == want);
			REQUIRE(pr(u, v) == want);
			REQUIRE(sv(u, v) == want);
			REQUIRE(sh(u, v) == want);
			REQUIRE(fcb(u, v) == want);
		}

		BENCHMARK(name + " | build binary lifting") { return ex::lca_by_binary_lifting(adj, 0).depth(1); };
		BENCHMARK(name + " | build HLD") { return ex::lca_by_head_jumping(adj, 0).depth(1); };
		BENCHMARK(name + " | build Euler+sparse") { return ex::lca_by_euler_sparse_table(adj, 0)(0, 1); };
		BENCHMARK(name + " | build preorder+sparse") { return ex::lca_by_preorder_sparse_table(adj, 0)(0, 1); };
		BENCHMARK(name + " | build FCB") { return ex::lca_by_farach_colton_bender(adj, 0)(0, 1); };
		BENCHMARK(name + " | build Euler+direct_rmq") { return ex::lca_by_euler_direct_rmq(adj, 0)(0, 1); };
		BENCHMARK(name + " | build preorder+direct_rmq") { return ex::lca_by_preorder_direct_rmq(adj, 0)(0, 1); };
		BENCHMARK(name + " | build Schieber-Vishkin") { return ex::lca_by_adjacency_preorder(adj, 0)(0, 1); };
		BENCHMARK(name + " | 5e5 queries binary lifting") { return throughput(bl, queries); };
		BENCHMARK(name + " | 5e5 queries HLD") { return throughput(hld, queries); };
		BENCHMARK(name + " | 5e5 queries Euler+sparse") { return throughput(es, queries); };
		BENCHMARK(name + " | 5e5 queries preorder+sparse") { return throughput(ps, queries); };
		BENCHMARK(name + " | 5e5 queries FCB") { return throughput(fcb, queries); };
		BENCHMARK(name + " | 5e5 queries Euler+direct_rmq") { return throughput(er, queries); };
		BENCHMARK(name + " | 5e5 queries preorder+direct_rmq") { return throughput(pr, queries); };
		BENCHMARK(name + " | 5e5 queries Schieber-Vishkin") { return throughput(sv, queries); };
		library_rows(name, parent, adj, queries);

		std::vector<uint32> ks(q);
		for (std::size_t i = 0; i < q; ++i)
			ks[i] = uint32(rng() % (sh.distance(queries[i].first, queries[i].second) + 1));
		BENCHMARK(name + " | 5e5 ancestor (library)") {
			uint64 sum = 0;
			for (std::size_t i = 0; i < q; ++i) {
				uint32 const u = queries[i].first;
				sum += sh.ancestor(u, ks[i] % (sh.depth(u) + 1));
			}
			return sum;
		};
		BENCHMARK(name + " | 5e5 path_vertex (library)") {
			uint64 sum = 0;
			for (std::size_t i = 0; i < q; ++i) sum += sh.path_vertex(queries[i].first, queries[i].second, ks[i]);
			return sum;
		};
		BENCHMARK(name + " | 5e5 path segments (library)") {
			uint64 segments = 0;
			for (auto const &[u, v] : queries) sh.path(u, v, [&](uint32, uint32, bool) { ++segments; });
			return segments;
		};
	};

	{
		std::vector<uint32> parent(n, g::none);
		for (uint32 v = 1; v < n; ++v) parent[v] = uint32(rng() % v);
		race("random recursive", parent);
	}
	{
		std::vector<uint32> parent(n, g::none);
		for (uint32 v = 1; v < n; ++v) parent[v] = v - 1;
		race("path", parent);
	}
	{
		std::vector<uint32> parent(n, 0); // star: everything hangs off the root
		parent[0] = g::none;
		race("star", parent);
	}
	{
		std::vector<uint32> parent(n, g::none); // caterpillar: spine + a leg per vertex
		for (uint32 v = 1; v < n; ++v) parent[v] = v % 2 ? v - 1 : v - 2;
		race("caterpillar", parent);
	}
	{
		std::vector<uint32> parent(n, g::none); // complete binary tree
		for (uint32 v = 1; v < n; ++v) parent[v] = (v - 1) / 2;
		race("binary tree", parent);
	}
}

TEST_CASE("lca library engine: forests and a large tree", "[lca]") {
	std::mt19937 rng(2);
	std::size_t const n = 500'000, q = 500'000;
	auto rows = [&](char const *label, std::vector<uint32> const &parent) {
		library_rows(label, parent, adjacency(parent), random_queries(rng, parent.size(), q));
	};
	rows("forest of singletons", std::vector<uint32>(n, g::none));
	{
		std::vector<uint32> parent(n, g::none);
		for (uint32 v = 1; v < n; v += 2) parent[v] = v - 1;
		rows("forest of 2-vertex trees", parent);
	}
	{
		std::vector<uint32> parent(n, g::none); // 1000 random recursive trees
		for (uint32 v = 1000; v < n; ++v) parent[v] = uint32(rng() % v);
		rows("forest of 1000 random trees", parent);
	}
	{
		std::size_t const big = 4'000'000;
		std::vector<uint32> parent(big, g::none);
		for (uint32 v = 1; v < big; ++v) parent[v] = uint32(rng() % v);
		rows("random recursive n=4e6", parent);
	}
}
