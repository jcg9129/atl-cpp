#pragma once
#include <atl/ds/direct_rmq.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/graph/traversal.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

/**
 * @file
 * @brief LCA contenders that lost the race to the library Schieber-Vishkin
 *        engine (atl/graph/lca.hpp).
 *
 * Shared contract, deliberately smaller than the library class's: construct
 * from (connected tree, root), query `operator()(u, v)` and `depth(v)`. The
 * rootless-forest and parent-array constructors, `size()` and the `none`
 * answer across components are not mirrored: lifting six engines onto that
 * constructor set costs far more than the race needs, since every shape it
 * times is one rooted tree. Every engine is
 * checked against a brute-force parent walk in tests/graph/lca.cpp (n = 1 to
 * 400, so FCB and direct_rmq blocks are crossed) and against binary lifting
 * in the benchmark binary.
 */
namespace atl::experimental {
	/**
	 * @brief Binary lifting stores each power-of-two ancestor and lifts endpoints together.
	 * Random-tree query throughput (n=q=500000): 37.0 / 37.0 ms versus library
	 * 13.3 / 8.9 ms, plain / native. Xeon Platinum 8280 @ 2.70 GHz,
	 * GCC 15.2.0, RelWithDebInfo, assertions off, -O2 / -O2 -march=native,
	 * 2026-10-15, benchmarks/graph/lca.cpp. Full shape comparisons: docs/lca.md.
	 * @pre A connected symmetric tree, root < n < 2^31; valid query vertices.
	 * @complexity O(n log n) build and storage; O(log n) query.
	 */
	class lca_by_binary_lifting {
		std::size_t n_ = 0;
		uint32 levels_ = 1;
		std::vector<uint32> depth_;
		std::vector<uint32> up_; // up_[k * n + v]

	public:
		/** @brief Build the comparison index. @pre Connected tree and root < n < 2^31. */
		template<graph::detail::scannable_graph Graph>
		constexpr lca_by_binary_lifting(Graph const &g, graph::vertex_id root) : n_(graph::vertex_count(g)) {
			atl_assert(root < n_ && n_ < (uint64(1) << 31)); // uint32 depths, tour positions, labels
			depth_.assign(n_, 0);
			std::vector<uint32> parent(n_, root);
			uint32 deepest = 0;
			for (auto const &[v, p, d] : graph::bfs(g, root)) {
				parent[v] = p;
				depth_[v] = d;
				deepest = (std::max)(deepest, d);
			}
			levels_ = std::max<uint32>(1, std::bit_width(deepest));
			atl_assert(n_ <= (std::numeric_limits<std::size_t>::max)() / levels_);
			up_.resize(std::size_t(levels_) * n_);
			std::ranges::copy(parent, up_.begin());
			for (uint32 k = 1; k < levels_; ++k)
				for (std::size_t v = 0; v < n_; ++v)
					up_[k * n_ + v] = up_[(k - 1) * n_ + up_[(k - 1) * n_ + v]];
		}

		/** @brief Hop distance from the root. @pre v is a valid vertex. */
		[[nodiscard]] constexpr uint32 depth(graph::vertex_id v) const { return depth_[v]; }

		/** @brief Lowest common ancestor of valid vertices. */
		[[nodiscard]] constexpr graph::vertex_id operator()(graph::vertex_id u, graph::vertex_id v) const {
			if (depth_[u] < depth_[v]) std::swap(u, v);
			for (uint32 diff = depth_[u] - depth_[v]; diff; diff &= diff - 1)
				u = up_[std::size_t(std::countr_zero(diff)) * n_ + u];
			if (u == v) return u;
			for (uint32 k = levels_; k-- > 0;)
				if (up_[k * n_ + u] != up_[k * n_ + v]) {
					u = up_[k * n_ + u];
					v = up_[k * n_ + v];
				}
			return up_[u];
		}
	};

	/**
	 * @brief Jump heavy-chain heads until endpoints share a chain.
	 * Random-tree query throughput (n=q=500000): 67.8 / 68.6 ms versus library
	 * 13.3 / 8.9 ms, plain / native. Xeon Platinum 8280 @ 2.70 GHz,
	 * GCC 15.2.0, RelWithDebInfo, assertions off, -O2 / -O2 -march=native,
	 * 2026-10-15, benchmarks/graph/lca.cpp. Full shape comparisons: docs/lca.md.
	 * @pre A connected symmetric tree, root < n < 2^31; valid query vertices.
	 * @complexity O(n) build and storage; O(log n) query.
	 */
	class lca_by_head_jumping {
		std::vector<uint32> parent_, depth_, head_;

	public:
		/** @brief Build the comparison index. @pre Connected tree and root < n < 2^31. */
		template<graph::detail::scannable_graph Graph>
		constexpr lca_by_head_jumping(Graph const &g, graph::vertex_id root) {
			std::size_t const n = graph::vertex_count(g);
			atl_assert(root < n && n < (uint64(1) << 31)); // uint32 depths, tour positions, labels
			parent_.assign(n, root);
			depth_.assign(n, 0);
			head_.assign(n, root);
			std::vector<uint32> order;
			order.reserve(n);
			for (auto const &[v, p, d] : graph::bfs(g, root)) {
				parent_[v] = p;
				depth_[v] = d;
				order.push_back(v);
			}
			std::vector<uint32> size(n, 1), heavy(n, graph::none);
			for (std::size_t i = order.size(); i-- > 0;) {
				uint32 const v = order[i];
				if (v == root) continue;
				uint32 const p = parent_[v];
				size[p] += size[v];
				if (heavy[p] == graph::none || size[v] > size[heavy[p]]) heavy[p] = v;
			}
			for (uint32 const v : order)
				head_[v] = (v != root && heavy[parent_[v]] == v) ? head_[parent_[v]] : v;
		}

		/** @brief Hop distance from the root. @pre v is a valid vertex. */
		[[nodiscard]] constexpr uint32 depth(graph::vertex_id v) const { return depth_[v]; }

		/** @brief Lowest common ancestor of valid vertices. */
		[[nodiscard]] constexpr graph::vertex_id operator()(graph::vertex_id u, graph::vertex_id v) const {
			while (head_[u] != head_[v]) {
				if (depth_[head_[u]] >= depth_[head_[v]])
					u = parent_[head_[u]];
				else
					v = parent_[head_[v]];
			}
			return depth_[u] < depth_[v] ? u : v;
		}
	};

	/**
	 * @brief Sparse-table minima of packed depth and position over an Euler tour.
	 * Random-tree query throughput (n=q=500000): 16.4 / 15.5 ms versus library
	 * 13.3 / 8.9 ms, plain / native. Xeon Platinum 8280 @ 2.70 GHz,
	 * GCC 15.2.0, RelWithDebInfo, assertions off, -O2 / -O2 -march=native,
	 * 2026-10-15, benchmarks/graph/lca.cpp. Full shape comparisons: docs/lca.md.
	 * @pre A connected symmetric tree, root < n < 2^31; valid query vertices.
	 * @complexity O(n log n) build and storage; O(1) query.
	 */
	class lca_by_euler_sparse_table {
		std::vector<uint32> first_, tour_, depth_;
		std::vector<uint64> table_; // levels concatenated
		std::size_t len_ = 0;

		/** @brief Tour entry keyed for `std::min`: depth in the high word,
		 *         position in the low word, so ties break to the leftmost. */
		[[nodiscard]] static constexpr uint64 pack(uint32 depth, uint32 pos) { return (uint64(depth) << 32) | pos; }

	public:
		/** @brief Build the comparison index. @pre Connected tree and root < n < 2^31. */
		template<graph::detail::scannable_graph Graph>
		constexpr lca_by_euler_sparse_table(Graph const &g, graph::vertex_id root) {
			std::size_t const n = graph::vertex_count(g);
			atl_assert(root < n && n < (uint64(1) << 31)); // uint32 depths, tour positions, labels
			first_.assign(n, 0);
			tour_.reserve(2 * n);
			depth_.assign(n, 0);
			graph::dfs_visit(
				g, root,
				[&](graph::vertex_id v, graph::vertex_id, uint32, uint32 d) {
					depth_[v] = d;
					first_[v] = uint32(tour_.size());
					tour_.push_back(v);
				},
				graph::skip,
				[&](graph::vertex_id, graph::vertex_id p) { tour_.push_back(p); });
			if (tour_.size() > 1) tour_.pop_back(); // the root re-entry after its exit
			len_ = tour_.size();
			uint32 const levels = uint32(std::bit_width(len_ > 1 ? len_ - 1 : 1));
			atl_assert(len_ <= (std::numeric_limits<std::size_t>::max)() / levels);
			table_.resize(std::size_t(levels) * len_);
			for (std::size_t i = 0; i < len_; ++i) table_[i] = pack(depth_[tour_[i]], uint32(i));
			for (uint32 k = 1; k < levels; ++k)
				for (std::size_t i = 0; i + (std::size_t(1) << k) <= len_; ++i)
					table_[k * len_ + i] = (std::min)(table_[(k - 1) * len_ + i],
													  table_[(k - 1) * len_ + i + (1u << (k - 1))]);
		}

		/** @brief Hop distance from the root. @pre v is a valid vertex. */
		[[nodiscard]] constexpr uint32 depth(graph::vertex_id v) const { return depth_[v]; }

		/** @brief Lowest common ancestor of valid vertices. */
		[[nodiscard]] constexpr graph::vertex_id operator()(graph::vertex_id u, graph::vertex_id v) const {
			auto [l, r] = std::minmax(first_[u], first_[v]);
			uint32 const k = std::bit_width(uint32(r - l + 1)) - 1;
			uint64 const best = (std::min)(table_[std::size_t(k) * len_ + l],
										   table_[std::size_t(k) * len_ + r + 1 - (1u << k)]);
			return tour_[uint32(best)];
		}
	};

	/**
	 * @brief Word-mask RMQ over Euler-tour depths, borrowing owned depth storage.
	 * Random-tree query throughput (n=q=500000): 27.0 / 25.4 ms versus library
	 * 13.3 / 8.9 ms, plain / native. Xeon Platinum 8280 @ 2.70 GHz,
	 * GCC 15.2.0, RelWithDebInfo, assertions off, -O2 / -O2 -march=native,
	 * 2026-10-15, benchmarks/graph/lca.cpp. Full shape comparisons: docs/lca.md.
	 * @pre A connected symmetric tree, root < n < 2^31; valid query vertices.
	 * @complexity O(n) build and storage; O(1) query.
	 */
	class lca_by_euler_direct_rmq {
		std::vector<uint32> first_, tour_, tdepth_, depth_;
		// Views tdepth_'s heap buffer: a move carries the buffer along, so
		// moves are safe; a copy would leave the view on the source's buffer.
		direct_rmq<uint32> rmq_;

	public:
		/** @brief Build the comparison index. @pre Connected tree and root < n < 2^31. */
		template<graph::detail::scannable_graph Graph>
		constexpr lca_by_euler_direct_rmq(Graph const &g, graph::vertex_id root) {
			std::size_t const n = graph::vertex_count(g);
			atl_assert(root < n && n < (uint64(1) << 31)); // uint32 depths, tour positions, labels
			first_.assign(n, 0);
			tour_.reserve(2 * n);
			tdepth_.reserve(2 * n);
			depth_.assign(n, 0);
			graph::dfs_visit(
				g, root,
				[&](graph::vertex_id v, graph::vertex_id, uint32, uint32 d) {
					depth_[v] = d;
					first_[v] = uint32(tour_.size());
					tour_.push_back(v);
					tdepth_.push_back(d);
				},
				graph::skip,
				[&](graph::vertex_id, graph::vertex_id p) {
					tour_.push_back(p);
					tdepth_.push_back(depth_[p]);
				});
			if (tour_.size() > 1) {
				tour_.pop_back();
				tdepth_.pop_back();
			}
			rmq_ = direct_rmq<uint32>(tdepth_);
		}

		/** @brief Preserve the borrowed RMQ buffer ownership rule. */
		lca_by_euler_direct_rmq(lca_by_euler_direct_rmq const &) = delete;
		/** @brief Transfer ownership; copying a borrowed RMQ buffer is disabled. */
		lca_by_euler_direct_rmq &operator=(lca_by_euler_direct_rmq const &) = delete;
		/** @brief Preserve the borrowed RMQ buffer ownership rule. */
		lca_by_euler_direct_rmq(lca_by_euler_direct_rmq &&) noexcept = default;
		/** @brief Transfer ownership; copying a borrowed RMQ buffer is disabled. */
		lca_by_euler_direct_rmq &operator=(lca_by_euler_direct_rmq &&) noexcept = default;

		/** @brief Hop distance from the root. @pre v is a valid vertex. */
		[[nodiscard]] constexpr uint32 depth(graph::vertex_id v) const { return depth_[v]; }

		/** @brief Lowest common ancestor of valid vertices. */
		[[nodiscard]] constexpr graph::vertex_id operator()(graph::vertex_id u, graph::vertex_id v) const {
			auto const [l, r] = std::minmax(first_[u], first_[v]);
			return tour_[rmq_.position(l, r + 1)];
		}
	};

	/**
	 * @brief Sparse-table minima of parent entry times over an n-entry preorder.
	 * Random-tree query throughput (n=q=500000): 10.1 / 9.2 ms versus library
	 * 13.3 / 8.9 ms, plain / native. Xeon Platinum 8280 @ 2.70 GHz,
	 * GCC 15.2.0, RelWithDebInfo, assertions off, -O2 / -O2 -march=native,
	 * 2026-10-15, benchmarks/graph/lca.cpp. Full shape comparisons: docs/lca.md.
	 * @pre A connected symmetric tree, root < n < 2^31; valid query vertices.
	 * @complexity O(n log n) build and storage; O(1) query.
	 */
	class lca_by_preorder_sparse_table {
		std::vector<uint32> tin_, order_, depth_;
		std::vector<uint32> table_; // levels concatenated, len_ entries each
		std::size_t len_ = 0;

	public:
		/** @brief Build the comparison index. @pre Connected tree and root < n < 2^31. */
		template<graph::detail::scannable_graph Graph>
		constexpr lca_by_preorder_sparse_table(Graph const &g, graph::vertex_id root) {
			std::size_t const n = graph::vertex_count(g);
			atl_assert(root < n && n < (uint64(1) << 31)); // uint32 depths and preorder numbers
			tin_.assign(n, 0);
			depth_.assign(n, 0);
			order_.reserve(n);
			std::vector<uint32> parent_tin;
			parent_tin.reserve(n);
			graph::dfs_visit(
				g, root,
				[&](graph::vertex_id v, graph::vertex_id p, uint32, uint32 d) {
					depth_[v] = d;
					tin_[v] = uint32(order_.size());
					order_.push_back(v);
					parent_tin.push_back(tin_[p]);
				},
				graph::skip, graph::skip);
			len_ = order_.size();
			uint32 const levels = uint32(std::bit_width(len_));
			atl_assert(len_ <= (std::numeric_limits<std::size_t>::max)() / levels);
			table_.resize(std::size_t(levels) * len_);
			std::ranges::copy(parent_tin, table_.begin());
			for (uint32 k = 1; k < levels; ++k)
				for (std::size_t i = 0; i + (std::size_t(1) << k) <= len_; ++i)
					table_[k * len_ + i] =
						(std::min)(table_[(k - 1) * len_ + i], table_[(k - 1) * len_ + i + (std::size_t(1) << (k - 1))]);
		}

		/** @brief Hop distance from the root. @pre v is a valid vertex. */
		[[nodiscard]] constexpr uint32 depth(graph::vertex_id v) const { return depth_[v]; }

		/** @brief Lowest common ancestor of valid vertices. */
		[[nodiscard]] constexpr graph::vertex_id operator()(graph::vertex_id u, graph::vertex_id v) const {
			if (u == v) return u;
			// positions (tin[u], tin[v]]
			uint32 const l = (std::min)(tin_[u], tin_[v]) + 1, r = (std::max)(tin_[u], tin_[v]);
			uint32 const k = uint32(std::bit_width(r - l + 1)) - 1;
			return order_[(std::min)(table_[std::size_t(k) * len_ + l], table_[std::size_t(k) * len_ + r + 1 - (1u << k)])];
		}
	};

	/**
	 * @brief Word-mask RMQ over parent entry times in preorder.
	 * Random-tree query throughput (n=q=500000): 21.8 / 20.0 ms versus library
	 * 13.3 / 8.9 ms, plain / native. Xeon Platinum 8280 @ 2.70 GHz,
	 * GCC 15.2.0, RelWithDebInfo, assertions off, -O2 / -O2 -march=native,
	 * 2026-10-15, benchmarks/graph/lca.cpp. Full shape comparisons: docs/lca.md.
	 * @pre A connected symmetric tree, root < n < 2^31; valid query vertices.
	 * @complexity O(n) build and storage; O(1) query.
	 */
	class lca_by_preorder_direct_rmq {
		std::vector<uint32> tin_, order_, parent_tin_, depth_;
		// Views parent_tin_'s heap buffer: a move carries the buffer along, so
		// moves are safe; a copy would leave the view on the source's buffer.
		direct_rmq<uint32> rmq_;

	public:
		/** @brief Build the comparison index. @pre Connected tree and root < n < 2^31. */
		template<graph::detail::scannable_graph Graph>
		constexpr lca_by_preorder_direct_rmq(Graph const &g, graph::vertex_id root) {
			std::size_t const n = graph::vertex_count(g);
			atl_assert(root < n && n < (uint64(1) << 31)); // uint32 depths and preorder numbers
			tin_.assign(n, 0);
			depth_.assign(n, 0);
			order_.reserve(n);
			parent_tin_.reserve(n);
			graph::dfs_visit(
				g, root,
				[&](graph::vertex_id v, graph::vertex_id p, uint32, uint32 d) {
					depth_[v] = d;
					tin_[v] = uint32(order_.size());
					order_.push_back(v);
					parent_tin_.push_back(tin_[p]);
				},
				graph::skip, graph::skip);
			rmq_ = direct_rmq<uint32>(parent_tin_);
		}

		/** @brief Preserve the borrowed RMQ buffer ownership rule. */
		lca_by_preorder_direct_rmq(lca_by_preorder_direct_rmq const &) = delete;
		/** @brief Transfer ownership; copying a borrowed RMQ buffer is disabled. */
		lca_by_preorder_direct_rmq &operator=(lca_by_preorder_direct_rmq const &) = delete;
		/** @brief Preserve the borrowed RMQ buffer ownership rule. */
		lca_by_preorder_direct_rmq(lca_by_preorder_direct_rmq &&) noexcept = default;
		/** @brief Transfer ownership; copying a borrowed RMQ buffer is disabled. */
		lca_by_preorder_direct_rmq &operator=(lca_by_preorder_direct_rmq &&) noexcept = default;

		/** @brief Hop distance from the root. @pre v is a valid vertex. */
		[[nodiscard]] constexpr uint32 depth(graph::vertex_id v) const { return depth_[v]; }

		/** @brief Lowest common ancestor of valid vertices. */
		[[nodiscard]] constexpr graph::vertex_id operator()(graph::vertex_id u, graph::vertex_id v) const {
			if (u == v) return u;
			auto const [l, r] = std::minmax(tin_[u], tin_[v]);
			return order_[parent_tin_[rmq_.position(l + 1, r + 1)]];
		}
	};

	/**
	 * @brief Schieber-Vishkin labels over adjacency-order preorder without heavy chains.
	 * Random-tree query throughput (n=q=500000): 13.2 / 9.3 ms versus library
	 * 13.3 / 8.9 ms, plain / native. Xeon Platinum 8280 @ 2.70 GHz,
	 * GCC 15.2.0, RelWithDebInfo, assertions off, -O2 / -O2 -march=native,
	 * 2026-10-15, benchmarks/graph/lca.cpp. Full shape comparisons: docs/lca.md.
	 * @pre A connected symmetric tree, root < n < 2^31; valid query vertices.
	 * @complexity O(n) build and storage; O(1) query.
	 */
	class lca_by_adjacency_preorder {
		std::vector<uint32> parent_, depth_, pre_, path_label_, ascendant_, head_;

		/** @brief The vertex where x's root path enters the inlabel path
		 *         `target`, whose lowest set bit is j. */
		[[nodiscard]] constexpr graph::vertex_id path_entry(graph::vertex_id x, uint32 target, uint32 j) const {
			if (path_label_[x] == target) return x;
			// the deepest inlabel above x strictly below bit j
			uint32 const k = uint32(std::bit_width(ascendant_[x] & ((1u << j) - 1))) - 1;
			uint32 const label = ((path_label_[x] >> k) | 1u) << k;
			return parent_[head_[label]];
		}

	public:
		/** @brief Build the comparison index. @pre Connected tree and root < n < 2^31. */
		template<graph::detail::scannable_graph Graph>
		constexpr lca_by_adjacency_preorder(Graph const &g, graph::vertex_id root) {
			std::size_t const n = graph::vertex_count(g);
			atl_assert(root < n && n < (uint64(1) << 31)); // uint32 depths, tour positions, labels
			parent_.assign(n, root);
			depth_.assign(n, 0);
			pre_.assign(n, 0);
			path_label_.assign(n, 0);
			ascendant_.assign(n, 0);
			head_.assign(n + 1, root);
			std::vector<uint32> last(n, 0); // preorder of the last subtree vertex
			uint32 timer = 0;
			graph::dfs_visit(
				g, root,
				[&](graph::vertex_id v, graph::vertex_id p, uint32, uint32 d) {
					parent_[v] = p;
					depth_[v] = d;
					pre_[v] = ++timer; // 1-indexed: trailing-zero arithmetic
				},
				graph::skip, [&](graph::vertex_id v, graph::vertex_id) { last[v] = timer; });
			// inlabel: the number with the most trailing zeros in
			// [pre, last]; identical for a vertex and the child continuing
			// its path
			for (std::size_t v = 0; v < n; ++v) {
				uint32 const split = std::bit_width((pre_[v] - 1) ^ last[v]);
				path_label_[v] = last[v] >> (split - 1) << (split - 1);
			}
			// ascendants accumulate inlabel low-bits along root paths;
			// heads invert inlabel -> topmost vertex (preorder = parents
			// precede children, so one forward pass suffices)
			std::vector<uint32> order(n);
			for (std::size_t v = 0; v < n; ++v) order[pre_[v] - 1] = uint32(v);
			ascendant_[root] = path_label_[root] & (~path_label_[root] + 1);
			head_[path_label_[root]] = root;
			for (uint32 const v : order) {
				if (v == root) continue;
				uint32 const low = path_label_[v] & (~path_label_[v] + 1);
				ascendant_[v] = ascendant_[parent_[v]] | low;
				if (path_label_[v] != path_label_[parent_[v]]) head_[path_label_[v]] = v;
			}
		}

		/** @brief Hop distance from the root. @pre v is a valid vertex. */
		[[nodiscard]] constexpr uint32 depth(graph::vertex_id v) const { return depth_[v]; }

		/** @brief Lowest common ancestor of valid vertices. */
		[[nodiscard]] constexpr graph::vertex_id operator()(graph::vertex_id u, graph::vertex_id v) const {
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

	/**
	 * @brief Euler-tour block signatures, within-block argmins and a sparse table of blocks.
	 * Random-tree query throughput (n=q=500000): 24.9 / 24.2 ms versus library
	 * 13.3 / 8.9 ms, plain / native. Xeon Platinum 8280 @ 2.70 GHz,
	 * GCC 15.2.0, RelWithDebInfo, assertions off, -O2 / -O2 -march=native,
	 * 2026-10-15, benchmarks/graph/lca.cpp. Full shape comparisons: docs/lca.md.
	 * @pre A connected symmetric tree, root < n < 2^31; valid query vertices.
	 * @complexity O(n) build and storage; O(1) query.
	 */
	class lca_by_farach_colton_bender {
		std::vector<uint32> first_, tour_, tdepth_, depth_, mask_;
		std::vector<uint64> blocks_;
		std::vector<uint8> in_;
		std::size_t len_ = 0, nb_ = 0;
		uint32 b_ = 1;

		/** @brief Tour entry keyed for `std::min`: depth in the high word,
		 *         position in the low word, so ties break to the leftmost. */
		[[nodiscard]] static constexpr uint64 pack(uint32 depth, uint32 pos) { return (uint64(depth) << 32) | pos; }

		/** @brief Tour position of the minimum over offsets [l, r] of one
		 *         block. @pre l <= r < b_ */
		[[nodiscard]] constexpr uint32 in_block(std::size_t block, uint32 l, uint32 r) const {
			// the table stores argmin - l (an offset from the range start)
			return uint32(block * b_) + l + in_[(std::size_t(mask_[block]) * b_ + l) * b_ + r];
		}

	public:
		/** @brief Build the comparison index. @pre Connected tree and root < n < 2^31. */
		template<graph::detail::scannable_graph Graph>
		constexpr lca_by_farach_colton_bender(Graph const &g, graph::vertex_id root) {
			std::size_t const n = graph::vertex_count(g);
			atl_assert(root < n && n < (uint64(1) << 31)); // uint32 depths, tour positions, labels
			first_.assign(n, 0);
			tour_.reserve(2 * n);
			tdepth_.reserve(2 * n);
			depth_.assign(n, 0);
			graph::dfs_visit(
				g, root,
				[&](graph::vertex_id v, graph::vertex_id, uint32, uint32 d) {
					depth_[v] = d;
					first_[v] = uint32(tour_.size());
					tour_.push_back(v);
					tdepth_.push_back(d);
				},
				graph::skip,
				[&](graph::vertex_id, graph::vertex_id p) {
					tour_.push_back(p);
					tdepth_.push_back(depth_[p]);
				});
			if (tour_.size() > 1) {
				tour_.pop_back();
				tdepth_.pop_back();
			}
			len_ = tour_.size();
			b_ = std::max<uint32>(1, uint32(std::bit_width(len_)) / 2);
			nb_ = len_ / b_ + (len_ % b_ != 0);
			mask_.assign(nb_, 0);
			std::vector<uint64> base(nb_);
			for (std::size_t blk = 0; blk < nb_; ++blk) {
				std::size_t const lo = blk * b_, hi = (std::min)(lo + b_, len_);
				uint64 best = pack(tdepth_[lo], uint32(lo));
				for (std::size_t i = lo + 1; i < hi; ++i) {
					if (tdepth_[i] > tdepth_[i - 1]) mask_[blk] |= 1u << (i - lo - 1);
					best = (std::min)(best, pack(tdepth_[i], uint32(i)));
				}
				base[blk] = best;
			}
			uint32 const levels = uint32(std::bit_width(nb_ > 1 ? nb_ - 1 : 1));
			atl_assert(nb_ <= (std::numeric_limits<std::size_t>::max)() / levels);
			blocks_.resize(std::size_t(levels) * nb_);
			std::ranges::copy(base, blocks_.begin());
			for (uint32 k = 1; k < levels; ++k)
				for (std::size_t i = 0; i + (std::size_t(1) << k) <= nb_; ++i)
					blocks_[k * nb_ + i] =
						(std::min)(blocks_[(k - 1) * nb_ + i],
								   blocks_[(k - 1) * nb_ + i + (std::size_t(1) << (k - 1))]);
			in_.assign((std::size_t(1) << (b_ - 1)) * b_ * b_, 0);
			for (uint32 m = 0; m < (1u << (b_ - 1)); ++m)
				for (uint32 l = 0; l < b_; ++l) {
					int cur = 0, best = 0;
					uint32 arg = l;
					for (uint32 r = l; r < b_; ++r) {
						if (r > l) {
							cur += (m >> (r - 1)) & 1 ? 1 : -1;
							if (cur < best) {
								best = cur;
								arg = r;
							}
						}
						in_[(std::size_t(m) * b_ + l) * b_ + r] = uint8(arg - l);
					}
				}
		}

		/** @brief Hop distance from the root. @pre v is a valid vertex. */
		[[nodiscard]] constexpr uint32 depth(graph::vertex_id v) const { return depth_[v]; }

		/** @brief Lowest common ancestor of valid vertices. */
		[[nodiscard]] constexpr graph::vertex_id operator()(graph::vertex_id u, graph::vertex_id v) const {
			auto const [l, r] = std::minmax(first_[u], first_[v]);
			std::size_t const bl = l / b_, br = r / b_;
			if (bl == br) return tour_[in_block(bl, l % b_, r % b_)];
			uint32 best = in_block(bl, l % b_, b_ - 1);
			uint32 const right = in_block(br, 0, r % b_);
			if (tdepth_[right] < tdepth_[best]) best = right;
			if (bl + 1 < br) {
				uint32 const k = uint32(std::bit_width(br - bl - 1)) - 1;
				uint64 const mid =
					(std::min)(blocks_[std::size_t(k) * nb_ + bl + 1],
							   blocks_[std::size_t(k) * nb_ + br - (std::size_t(1) << k)]);
				if (uint32(mid >> 32) < tdepth_[best]) best = uint32(mid);
			}
			return tour_[best];
		}
	};
}
