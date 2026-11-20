#pragma once
#include <atl/graph/concepts.hpp>
#include <atl/graph/lca.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace atl::graph {
	/**
	 * @brief A virtual tree (auxiliary tree): the key vertices plus the LCA of
	 *        every two of them, each vertex hung under its nearest kept
	 *        ancestor. At most `2k - 1` vertices for `k` distinct keys.
	 *
	 * `vertices` is sorted by the lca's `entry_time` (its heavy-first
	 * preorder), so parents come comparison children, every virtual subtree is a
	 * contiguous run, and a backward pass is a bottom-up DP order. Edge
	 * lengths are left to the caller: `tree.depth(v) - tree.depth(p)`, or a
	 * weighted depth of its own.
	 */
	struct virtual_tree {
		std::vector<vertex_id> vertices; ///< kept vertices in the lca's preorder (parents precede children)
		std::vector<uint32> parent;		 ///< local index of the parent, `none` for a root
	};

	namespace detail {
		/**
		 * @brief A monotone-stack entry: `rank << 32 | vertex` and the head of
		 *        its finished children's list (indices into `out.parent`).
		 */
		struct virtual_tree_frame {
			uint64 key;
			uint32 children;
		};

		/**
		 * @brief The monotone-stack sweep over keys packed `entry_time << 32 | v`
		 *        and sorted, swept from the LARGEST entry time down; at most
		 *        k - 1 LCA queries.
		 *
		 * The stack is one root path. Popping is a postorder whose siblings
		 * leave in decreasing entry time, so its reverse is exactly the lca's
		 * preorder: vertices are written by pop index and reversed once at the
		 * end. A popped vertex's parent is still on the stack with no index
		 * yet, so popped children wait in a list threaded through `out.parent`
		 * (each entry the next sibling until the parent pops and overwrites it
		 * with its own index): no memory beyond the result.
		 *
		 * Frames are ordered by depth or entry time; both order a root path.
		 * Depth-ranked frames avoid a fresh entry-time load for an LCA result.
		 */
		constexpr void virtual_tree_sweep(lca const &tree, std::span<uint64 const> sorted,
										  std::vector<virtual_tree_frame> &stack, bool is_depth_ranked,
										  virtual_tree &out) {
			std::size_t const k = sorted.size();
			out.vertices.resize(k == 0 ? 0 : 2 * k - 1);
			out.parent.resize(out.vertices.size());
			stack.clear();
			auto const frame_key = [&](uint64 entry_key) {
				return is_depth_ranked ? uint64(tree.depth(vertex_id(entry_key))) << 32 | uint32(entry_key) : entry_key;
			};
			uint32 popped = 0;
			auto const pop = [&] {
				uint32 const e = popped++;
				auto const [key, children] = stack.back();
				stack.pop_back();
				out.vertices[e] = vertex_id(key);
				for (uint32 c = children; c != none;) {
					uint32 const next = out.parent[c];
					out.parent[c] = e;
					c = next;
				}
				return e;
			};
			auto const pop_under_top = [&] {
				uint32 const e = pop();
				out.parent[e] = stack.back().children;
				stack.back().children = e;
			};
			auto const flush = [&] {
				while (stack.size() > 1) pop_under_top();
				if (!stack.empty()) out.parent[pop()] = none;
			};
			for (std::size_t i = k; i-- > 0;) {
				vertex_id const v = vertex_id(sorted[i]);
				if (stack.empty()) {
					stack.push_back({frame_key(sorted[i]), none});
					continue;
				}
				vertex_id const above = vertex_id(stack.back().key);
				if (v == above) continue; // a duplicate key
				vertex_id const w = tree(above, v);
				if (w == none) { // another tree of the forest
					flush();
					stack.push_back({frame_key(sorted[i]), none});
					continue;
				}
				uint64 const w_key = uint64(is_depth_ranked ? tree.depth(w) : tree.entry_time(w)) << 32 | w;
				while (stack.size() > 1 && stack[stack.size() - 2].key >= w_key) pop_under_top();
				if (stack.back().key != w_key) {
					uint32 const e = pop();
					out.parent[e] = none;
					stack.push_back({w_key, e});
				}
				if (w != v) stack.push_back({frame_key(sorted[i]), none});
			}
			flush();
			uint32 const m = popped;
			out.vertices.resize(m);
			out.parent.resize(m);
			std::ranges::reverse(out.vertices);
			std::ranges::reverse(out.parent);
			for (uint32 &p : out.parent) p = p == none ? none : m - 1 - p;
		}

		/**
		 * @brief Sort packed keys `entry_time << 32 | v` by setting one bit per entry time in an
		 *        `n / 8`-byte bitmap and reading the set bits back in order (countr_zero per word),
		 *        each rebuilt through `tree.order()`; duplicates collapse, which the sweep skips anyway.
		 *
		 * O(k + n / 64) with sequential word reads and no element buffer, against the counting sort's
		 * O(k + n) count pass: it replaces counting from n / 128 keys. Reading clears each word, so the bitmap is zero again.
		 * @pre @p bits is all zero (asserted, O(n / 64))
		 */
		constexpr void virtual_tree_bitmap_sort(lca const &tree, std::vector<uint64> &keys, std::vector<uint64> &bits) {
			atl_assert(std::ranges::all_of(bits, [](uint64 word) { return word == 0; }));
			bits.resize((tree.size() + 63) / 64);
			for (uint64 const key : keys) {
				atl_assert((key >> 32) < tree.size());
				uint32 const t = uint32(key >> 32);
				bits[t >> 6] |= uint64(1) << (t & 63);
			}
			std::vector<uint32> const &order = tree.order();
			keys.clear();
			for (std::size_t word = 0; word < bits.size(); ++word) {
				for (uint64 b = std::exchange(bits[word], 0); b != 0; b &= b - 1) {
					std::size_t const entry = word * 64 + std::size_t(std::countr_zero(b));
					atl_assert(entry < tree.size() && entry <= (std::numeric_limits<uint32>::max)());
					uint32 const t = uint32(entry);
					keys.push_back(uint64(t) << 32 | order[t]);
				}
			}
		}

		/** @brief Scratch reused across queries: sorted keys, stack and a zeroed bitmap. */
		struct virtual_tree_scratch {
			std::vector<uint64> sorted;
			std::vector<virtual_tree_frame> stack;
			std::vector<uint64> bits; ///< Zero between bitmap sorts.
		};

		inline virtual_tree_scratch &virtual_tree_buffers() {
			static thread_local virtual_tree_scratch s;
			return s;
		}

		/**
		 * @brief Gather, sort, sweep. Up to 32 keys from a sized range sort in
		 *        a stack buffer; from `128k >= n` the sort uses a zeroed bitmap
		 *        and from `64k >= n` frames are
		 *        depth-ranked. Both thresholds measured at n = 5e5.
		 */
		template<vertex_range R>
			requires std::convertible_to<std::ranges::range_reference_t<R>, vertex_id>
		constexpr void virtual_tree_build(lca const &tree, R &&keys, virtual_tree_scratch &s, virtual_tree &out) {
			std::size_t const n = tree.size();
			constexpr std::size_t stack_key_capacity = 32;
			if constexpr (std::ranges::sized_range<R>) {
				auto const key_count = std::ranges::size(keys);
				atl_assert(key_count < (uint64(1) << 31));
				std::size_t const k = std::size_t(key_count);
				if (k <= stack_key_capacity) {
					std::array<uint64, stack_key_capacity> sorted{};
					std::size_t i = 0;
					for (auto &&key : keys) {
						vertex_id const v = detail::tree_vertex_id(key);
						atl_assert(v < n);
						sorted[i++] = uint64(tree.entry_time(v)) << 32 | v;
					}
					std::ranges::sort(sorted.begin(), sorted.begin() + std::ptrdiff_t(k));
					virtual_tree_sweep(tree, std::span<uint64 const>(sorted.data(), k), s.stack, false, out);
					return;
				}
			}
			s.sorted.clear();
			if constexpr (std::ranges::sized_range<R>) s.sorted.reserve(std::size_t(std::ranges::size(keys)));
			for (auto &&key : keys) {
				vertex_id const v = detail::tree_vertex_id(key);
				atl_assert(v < n);
				s.sorted.push_back(uint64(tree.entry_time(v)) << 32 | v);
			}
			std::size_t const k = s.sorted.size();
			atl_assert(k < std::size_t(1) << 31);
			if (k >= n / 128 + (n % 128 != 0))
				virtual_tree_bitmap_sort(tree, s.sorted, s.bits);
			else
				std::ranges::sort(s.sorted);
			virtual_tree_sweep(tree, s.sorted, s.stack, k >= n / 64 + (n % 64 != 0), out);
		}
	}

	/**
	 * @brief Rebuild @p out as the virtual tree of @p keys, reusing its
	 *        storage across queries.
	 * @tparam R Input range with references convertible to vertex_id.
	 * @param tree Preprocessed nonempty forest.
	 * @param keys Input range of valid vertex ids; duplicates are accepted.
	 * @param out Result arrays, replaced by this call.
	 *
	 * Keys may repeat and may span several trees of a forest (each gives its
	 * own root). An empty key range produces empty result arrays. The scratch is thread-local and reused (and iterating
	 * @p keys must not itself build a virtual tree), so once `out` and the
	 * scratch (including the bitmap) have grown a refill allocates nothing.
	 * @pre Input iteration must not invoke virtual_tree_of or virtual_tree_clear
	 * on the same thread, and keys must not alias out storage.
	 * Every key `< tree.size()` and fewer than 2^31 keys (both asserted)
	 * @complexity O(k log k) and k - 1 LCA queries; O(k + n / 64) once `128k >= n`
	 * (except the sized stack-buffer path for at most 32 keys).
	 */
	template<vertex_range R>
		requires std::convertible_to<std::ranges::range_reference_t<R>, vertex_id>
	constexpr void virtual_tree_of(lca const &tree, R &&keys, virtual_tree &out) {
		if consteval {
			detail::virtual_tree_scratch s;
			detail::virtual_tree_build(tree, std::forward<R>(keys), s, out);
		} else {
			detail::virtual_tree_build(tree, std::forward<R>(keys), detail::virtual_tree_buffers(), out);
		}
	}

	/**
	 * @brief The virtual tree of @p keys; use the refilling overload to reuse storage.
	 * @tparam R Input range with references convertible to vertex_id.
	 * @param tree Preprocessed forest.
	 * @param keys Input vertex ids, with duplicates allowed.
	 * @return Owning vertex ids and local parent indices.
	 * @pre Input iteration must not invoke virtual_tree_of or virtual_tree_clear
	 * on the same thread, and keys must not alias out storage.
	 * Every key `< tree.size()` and fewer than 2^31 keys (both asserted)
	 */
	template<vertex_range R>
		requires std::convertible_to<std::ranges::range_reference_t<R>, vertex_id>
	[[nodiscard]] constexpr virtual_tree virtual_tree_of(lca const &tree, R &&keys) {
		virtual_tree out;
		virtual_tree_of(tree, std::forward<R>(keys), out);
		return out;
	}

	/**
	 * @brief Release the reusable virtual-tree scratch on the current thread (frees
	 *        the sorted-key, stack and bitmap buffers). Rarely needed: only to reclaim
	 *        memory.
	 * @note Not constexpr (thread-local scratch).
	 */
	inline void virtual_tree_clear() { detail::virtual_tree_buffers() = detail::virtual_tree_scratch{}; }
}
