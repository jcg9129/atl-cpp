#pragma once
#include <atl/ds/sparse_table.hpp>
#include <atl/graph/concepts.hpp>
#include <atl/graph/lca.hpp>
#include <atl/graph/virtual_tree.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/counting_sort.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Virtual-tree builders raced against the library one
 *        (atl/graph/virtual_tree.hpp).
 *
 * Every builder works in a key space chosen by a key policy (packed
 * `entry_time << 32 | v`, bare entry times, or vertex ids sorted through a
 * projection), so the race separates the algorithm from the sort key and the
 * LCA backend. Shared contract: the library result, vertices in the lca's
 * preorder with local parent indices. Cross-checked against a brute force in
 * tests/graph/virtual_tree.cpp.
 *
 * Key-sort measurements: Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo,
 * assertions off, -O2 / -O2 -march=native, 2026-11-01. At n=500000,
 * 64 queries of n/128 keys take 32.0 / 30.2 ms with bitmap sorting,
 * 53.3 / 49.0 with counting and 38.9 / 38.6 with comparison sorting.
 * Other construction and backend contenders are not measured yet in the
 * supplied publication records. See docs/virtual-tree.md.
 */
namespace atl::experimental {
	/**
	 * @brief The preorder sparse-table LCA engine (see `lca_by_preorder_sparse_table` in
	 *        lca_variants.hpp) re-keyed on an `atl::graph::lca`'s heavy-first
	 *        preorder and answering in entry-time space: for entry times
	 *        `a < b` the LCA is the parent of the shallowest vertex at
	 *        positions `(a, b]`, whose entry time is the minimum of
	 *        `entry_time(parent) + 1` there. A root stores 0, so a range that
	 *        crosses into another tree of a forest folds to 0 and the answer
	 *        `0 - 1` is exactly `none`.
	 *
	 * The table trades O(n log n) storage for queries in entry-time space.
	 * Backend comparison: not measured yet.
	 * @complexity O(n log n) build and space, O(1) query
	 */
	class lca_by_entry_sparse_table {
		struct min_op {
			[[nodiscard]] constexpr uint32 operator()(uint32 a, uint32 b) const { return a < b ? a : b; }
		};

		graph::lca const *tree_;
		sparse_table<uint32, min_op> table_;

		/** @brief Position t holds `entry_time(parent(order[t])) + 1`, 0 at a root. */
		[[nodiscard]] static constexpr std::vector<uint32> parent_entries(graph::lca const &tree) {
			std::vector<uint32> values(tree.size());
			for (std::size_t t = 0; t < values.size(); ++t) {
				graph::vertex_id const p = tree.parent(tree.order()[t]);
				values[t] = p == graph::none ? 0 : tree.entry_time(p) + 1;
			}
			return values;
		}

	public:
		/** @brief Builds the table over @p tree's preorder; @p tree must outlive this engine. */
		constexpr explicit lca_by_entry_sparse_table(graph::lca const &tree) : tree_(&tree), table_(parent_entries(tree)) {}

		/** @brief Borrowed tree whose preorder the table indexes. */
		[[nodiscard]] constexpr graph::lca const &tree() const { return *tree_; }

		/**
		 * @brief The entry time of the LCA of the vertices entered at @p a and
		 *        @p b, `none` across trees.
		 * @pre `a != b`
		 */
		[[nodiscard]] constexpr uint32 lca_entry(uint32 a, uint32 b) const {
			uint32 const lo = a < b ? a : b, hi = a < b ? b : a;
			return table_.fold(lo + 1, std::size_t(hi) + 1) - 1;
		}
	};

	/** @brief @p is_counting, except that constant evaluation always comparison-sorts (`counting_sort` is not constexpr). */
	[[nodiscard]] constexpr bool use_counting_sort(bool is_counting) {
		if consteval {
			return false;
		} else {
			return is_counting;
		}
	}

	/**
	 * @brief A key space for the builders: `size` is the vertex count, `key_of`/`vertex_of` convert, `entry`
	 *        orders (the lca's entry time), `join` is the LCA of two distinct
	 *        keys (`none_key` across trees), `sort` orders a key span by entry.
	 */
	template<typename Policy>
	concept virtual_tree_key_policy = requires { typename Policy::key_type; }
								   && std::unsigned_integral<typename Policy::key_type> && (!std::same_as<typename Policy::key_type, bool>) && requires(Policy const &p, typename Policy::key_type k, graph::vertex_id v, std::span<typename Policy::key_type> keys, bool is_counting) {
										  { Policy::none_key } -> std::convertible_to<typename Policy::key_type>;
										  { p.size() } -> std::same_as<std::size_t>;
										  { p.key_of(v) } -> std::same_as<typename Policy::key_type>;
										  { p.vertex_of(k) } -> std::same_as<graph::vertex_id>;
										  { p.entry(k) } -> std::same_as<uint32>;
										  { p.join(k, k) } -> std::same_as<typename Policy::key_type>;
										  p.sort(keys, is_counting);
									  };

	/**
	 * @brief Packed keys `entry_time << 32 | v`: the comparator is a plain
	 *        integer compare, and the vertex rides along for the LCA call.
	 */
	struct packed_entry_keys {
		using key_type = uint64; ///< Packed entry time and vertex id.
		/** @brief Key sentinel for disconnected roots. */
		static constexpr key_type none_key = ~uint64(0);
		graph::lca const &tree; ///< Borrowed LCA; must outlive the policy.

		/** @brief Number of vertices in the borrowed tree. */
		constexpr std::size_t size() const { return tree.size(); }

		/** @brief Encode a valid vertex as a sortable key. */
		[[nodiscard]] constexpr key_type key_of(graph::vertex_id v) const { return uint64(tree.entry_time(v)) << 32 | v; }

		/** @brief Decode a valid key to its vertex. */
		[[nodiscard]] constexpr graph::vertex_id vertex_of(key_type k) const { return graph::vertex_id(k); }

		/** @brief Entry time of the key. */
		[[nodiscard]] constexpr uint32 entry(key_type k) const { return uint32(k >> 32); }

		/** @brief The LCA of two distinct keys, `none_key` across trees. */
		[[nodiscard]] constexpr key_type join(key_type a, key_type b) const {
			graph::vertex_id const w = tree(graph::vertex_id(a), graph::vertex_id(b));
			return w == graph::none ? none_key : key_of(w);
		}

		/** @brief Sort by entry time; is_counting requests a runtime counting sort. */
		constexpr void sort(std::span<key_type> keys, bool is_counting) const {
			if (use_counting_sort(is_counting))
				counting_sort(keys, [](key_type k) { return uint32(k >> 32); });
			else
				std::ranges::sort(keys);
		}
	};

	/**
	 * @brief Vertex keys sorted through an `entry_time` projection (an
	 *        indirect load per compare).
	 *        Projection-sort comparison: not measured yet.
	 */
	struct projected_keys {
		using key_type = uint32; ///< Vertex id or entry time selected by this policy.
		/** @brief Key sentinel for disconnected roots. */
		static constexpr key_type none_key = graph::none;
		graph::lca const &tree; ///< Borrowed LCA; must outlive the policy.

		/** @brief Number of vertices in the borrowed tree. */
		constexpr std::size_t size() const { return tree.size(); }

		/** @brief Encode a valid vertex as a sortable key. */
		[[nodiscard]] constexpr key_type key_of(graph::vertex_id v) const { return v; }

		/** @brief Decode a valid key to its vertex. */
		[[nodiscard]] constexpr graph::vertex_id vertex_of(key_type k) const { return k; }

		/** @brief Entry time of the key. */
		[[nodiscard]] constexpr uint32 entry(key_type k) const { return tree.entry_time(k); }

		/** @brief The LCA of two distinct keys, `none_key` across trees. */
		[[nodiscard]] constexpr key_type join(key_type a, key_type b) const { return tree(a, b); }

		/** @brief Sort by entry time; is_counting requests a runtime counting sort. */
		constexpr void sort(std::span<key_type> keys, bool is_counting) const {
			auto const by_entry = [this](key_type v) { return tree.entry_time(v); };
			if (use_counting_sort(is_counting))
				counting_sort(keys, by_entry);
			else
				std::ranges::sort(keys, {}, by_entry);
		}
	};

	/**
	 * @brief Bare entry times on the Schieber-Vishkin backend: the cheapest
	 *        sort, paid for by an order() load per endpoint.
	 * Comparison with packed keys: not measured yet.
	 */
	struct entry_keys {
		using key_type = uint32; ///< Vertex id or entry time selected by this policy.
		/** @brief Key sentinel for disconnected roots. */
		static constexpr key_type none_key = graph::none;
		graph::lca const &tree; ///< Borrowed LCA; must outlive the policy.

		/** @brief Number of vertices in the borrowed tree. */
		constexpr std::size_t size() const { return tree.size(); }

		/** @brief Encode a valid vertex as a sortable key. */
		[[nodiscard]] constexpr key_type key_of(graph::vertex_id v) const { return tree.entry_time(v); }

		/** @brief Decode a valid key to its vertex. */
		[[nodiscard]] constexpr graph::vertex_id vertex_of(key_type k) const { return tree.order()[k]; }

		/** @brief Entry time of the key. */
		[[nodiscard]] constexpr uint32 entry(key_type k) const { return k; }

		/** @brief The LCA of two distinct keys, `none_key` across trees. */
		[[nodiscard]] constexpr key_type join(key_type a, key_type b) const {
			graph::vertex_id const w = tree(tree.order()[a], tree.order()[b]);
			return w == graph::none ? none_key : tree.entry_time(w);
		}

		/** @brief Sort by entry time; is_counting requests a runtime counting sort. */
		constexpr void sort(std::span<key_type> keys, bool is_counting) const {
			if (use_counting_sort(is_counting))
				counting_sort(keys);
			else
				std::ranges::sort(keys);
		}
	};

	/**
	 * @brief Bare entry times on the preorder sparse table: the LCA never
	 *        leaves entry-time space. See @ref atl::experimental::lca_by_entry_sparse_table for
	 *        the race.
	 */
	struct sparse_entry_keys {
		using key_type = uint32; ///< Vertex id or entry time selected by this policy.
		/** @brief Key sentinel for disconnected roots. */
		static constexpr key_type none_key = graph::none;
		lca_by_entry_sparse_table const &engine; ///< Borrowed table; must outlive the policy.

		/** @brief Number of vertices in the borrowed tree. */
		constexpr std::size_t size() const { return engine.tree().size(); }

		/** @brief Encode a valid vertex as a sortable key. */
		[[nodiscard]] constexpr key_type key_of(graph::vertex_id v) const { return engine.tree().entry_time(v); }

		/** @brief Decode a valid key to its vertex. */
		[[nodiscard]] constexpr graph::vertex_id vertex_of(key_type k) const { return engine.tree().order()[k]; }

		/** @brief Entry time of the key. */
		[[nodiscard]] constexpr uint32 entry(key_type k) const { return k; }

		/** @brief The LCA of two distinct keys, `none_key` across trees. */
		[[nodiscard]] constexpr key_type join(key_type a, key_type b) const { return engine.lca_entry(a, b); }

		/** @brief Sort by entry time; is_counting requests a runtime counting sort. */
		constexpr void sort(std::span<key_type> keys, bool is_counting) const {
			if (use_counting_sort(is_counting))
				counting_sort(keys);
			else
				std::ranges::sort(keys);
		}
	};

	/** @brief Reusable scratch of the builders. */
	template<std::unsigned_integral Key>
	struct virtual_tree_workspace {
		/** @brief A stack entry: the key and the head of its finished children's list. */
		struct frame {
			Key key;		 ///< Policy key.
			uint32 children; ///< Pending child-list head or local result index, selected by the algorithm.
		};

		std::vector<Key> keys;	  ///< Gathered sort keys.
		std::vector<frame> stack; ///< Reusable root-path stack.
	};

	/**
	 * @brief One-sort construction: one sort, then one monotone stack swept from the
	 *        LARGEST entry time down, at most k - 1 joins. Popping is a postorder
	 *        whose sibling order is decreasing entry time, so its reverse is the
	 *        lca's preorder: vertices are written by pop index and reversed at the
	 *        end. A popped vertex's parent is still on the stack, its index
	 *        unknown, so children wait in a list threaded through `out.parent`
	 *        (each entry the next sibling until the parent pops and overwrites it
	 *        with its own index): no extra memory. The same algorithm as the library,
	 *        generic over the key policy so the race can vary keys and backend;
	 *        the library `detail::virtual_tree_sweep` is its packed-key form.
	 * @pre `w.keys` sorted by entry (duplicates allowed)
	 */
	template<virtual_tree_key_policy Policy>
	constexpr void virtual_tree_by_stack_sweep(Policy const &p, virtual_tree_workspace<typename Policy::key_type> &w, graph::virtual_tree &out) {
		using key_type = typename Policy::key_type;
		std::size_t const k = w.keys.size();
		out.vertices.resize(k == 0 ? 0 : 2 * k - 1);
		out.parent.resize(out.vertices.size());
		auto &stack = w.stack;
		stack.clear();
		uint32 popped = 0;
		auto const pop = [&] {
			uint32 const e = popped++;
			auto const [key, children] = stack.back();
			stack.pop_back();
			out.vertices[e] = p.vertex_of(key);
			for (uint32 c = children; c != graph::none;) {
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
			if (!stack.empty()) out.parent[pop()] = graph::none;
		};
		for (std::size_t i = k; i-- > 0;) {
			key_type const v = w.keys[i];
			if (stack.empty()) {
				stack.push_back({v, graph::none});
				continue;
			}
			if (v == stack.back().key) continue;
			key_type const l = p.join(stack.back().key, v);
			if (l == Policy::none_key) {
				flush();
				stack.push_back({v, graph::none});
				continue;
			}
			uint32 const l_entry = p.entry(l);
			while (stack.size() > 1 && p.entry(stack[stack.size() - 2].key) >= l_entry) pop_under_top();
			if (stack.back().key != l) {
				uint32 const e = pop();
				out.parent[e] = graph::none;
				stack.push_back({l, e});
			}
			if (l != v) stack.push_back({v, graph::none});
		}
		flush();
		uint32 const m = popped;
		out.vertices.resize(m);
		out.parent.resize(m);
		std::ranges::reverse(out.vertices);
		std::ranges::reverse(out.parent);
		for (uint32 &q : out.parent) q = q == graph::none ? graph::none : m - 1 - q;
	}

	/**
	 * @brief One-sort construction with depth comparisons (Schieber-Vishkin backend only):
	 *        packed keys for the sort, then frames hold `{vertex, depth}`. The
	 *        stack is one root path, so depth orders it as well as entry time
	 *        does, and the depth of a join's answer is on a cache line the query
	 *        just read, where its entry time is a fresh miss on the query's
	 *        critical path; the keys' own depths are independent loads. The
	 *        This isolates frame ranking from sorting.
	 *        Frame-rank comparison: not measured yet.
	 * @pre `keys` sorted (duplicates allowed)
	 */
	constexpr void virtual_tree_by_depth_sweep(graph::lca const &tree, std::span<uint64 const> keys,
											   std::vector<virtual_tree_workspace<uint64>::frame> &stack, graph::virtual_tree &out) {
		std::size_t const k = keys.size();
		out.vertices.resize(k == 0 ? 0 : 2 * k - 1);
		out.parent.resize(out.vertices.size());
		stack.clear();
		uint32 popped = 0;
		// frame.key = depth << 32 | vertex
		auto const pop = [&] {
			uint32 const e = popped++;
			auto const [key, children] = stack.back();
			stack.pop_back();
			out.vertices[e] = graph::vertex_id(key);
			for (uint32 c = children; c != graph::none;) {
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
			if (!stack.empty()) out.parent[pop()] = graph::none;
		};
		auto const frame_key = [&](graph::vertex_id v) { return uint64(tree.depth(v)) << 32 | v; };
		for (std::size_t i = k; i-- > 0;) {
			graph::vertex_id const v = graph::vertex_id(keys[i]);
			if (stack.empty()) {
				stack.push_back({frame_key(v), graph::none});
				continue;
			}
			graph::vertex_id const top = graph::vertex_id(stack.back().key);
			if (v == top) continue;
			graph::vertex_id const l = tree(top, v);
			if (l == graph::none) {
				flush();
				stack.push_back({frame_key(v), graph::none});
				continue;
			}
			uint64 const l_key = frame_key(l);
			while (stack.size() > 1 && stack[stack.size() - 2].key >= l_key) pop_under_top();
			if (stack.back().key != l_key) {
				uint32 const e = pop();
				out.parent[e] = graph::none;
				stack.push_back({l_key, e});
			}
			if (l != v) stack.push_back({frame_key(v), graph::none});
		}
		flush();
		uint32 const m = popped;
		out.vertices.resize(m);
		out.parent.resize(m);
		std::ranges::reverse(out.vertices);
		std::ranges::reverse(out.parent);
		for (uint32 &q : out.parent) q = q == graph::none ? graph::none : m - 1 - q;
	}

	/**
	 * @brief One-sort construction over a raw frame array indexed by a height, packed keys
	 *        on the Schieber-Vishkin backend. Raw-array comparison: not measured yet.
	 * @param stack Room for `2 * keys.size()` frames.
	 * @pre `keys` sorted (duplicates allowed)
	 */
	constexpr void virtual_tree_by_raw_array_sweep(graph::lca const &tree, std::span<uint64 const> keys,
												   virtual_tree_workspace<uint64>::frame *stack, graph::virtual_tree &out) {
		std::size_t const k = keys.size();
		out.vertices.resize(k == 0 ? 0 : 2 * k - 1);
		out.parent.resize(out.vertices.size());
		std::size_t top = 0; // stack[0 .. top)
		uint32 popped = 0;
		auto const pop = [&](std::size_t &height) {
			uint32 const e = popped++;
			auto const [key, children] = stack[--height];
			out.vertices[e] = graph::vertex_id(key);
			for (uint32 c = children; c != graph::none;) {
				uint32 const next = out.parent[c];
				out.parent[c] = e;
				c = next;
			}
			return e;
		};
		auto const pop_under_top = [&](std::size_t &height) {
			uint32 const e = pop(height);
			out.parent[e] = stack[height - 1].children;
			stack[height - 1].children = e;
		};
		auto const flush = [&] {
			while (top > 1) pop_under_top(top);
			if (top == 1) out.parent[pop(top)] = graph::none;
		};
		for (std::size_t i = k; i-- > 0;) {
			uint64 const v = keys[i];
			if (top == 0) {
				stack[top++] = {v, graph::none};
				continue;
			}
			graph::vertex_id const above = graph::vertex_id(stack[top - 1].key);
			if (graph::vertex_id(v) == above) continue;
			graph::vertex_id const w = tree(above, graph::vertex_id(v));
			if (w == graph::none) {
				flush();
				stack[top++] = {v, graph::none};
				continue;
			}
			uint64 const w_key = uint64(tree.entry_time(w)) << 32 | w;
			while (top > 1 && stack[top - 2].key >= w_key) pop_under_top(top);
			if (stack[top - 1].key != w_key) {
				uint32 const e = pop(top);
				out.parent[e] = graph::none;
				stack[top++] = {w_key, e};
			}
			if (w != graph::vertex_id(v)) stack[top++] = {v, graph::none};
		}
		flush();
		uint32 const m = popped;
		out.vertices.resize(m);
		out.parent.resize(m);
		std::ranges::reverse(out.vertices);
		std::ranges::reverse(out.parent);
		for (uint32 &q : out.parent) q = q == graph::none ? graph::none : m - 1 - q;
	}

	/** @brief The raw-array contender: packed sort, then @ref atl::experimental::virtual_tree_by_raw_array_sweep. */
	template<graph::vertex_range R>
		requires std::convertible_to<std::ranges::range_reference_t<R>, graph::vertex_id>
	constexpr void virtual_tree_by_raw_array(graph::lca const &tree, R &&keys, virtual_tree_workspace<uint64> &w,
											 graph::virtual_tree &out) {
		packed_entry_keys const p{tree};
		w.keys.clear();
		for (auto &&key : keys) {
			graph::vertex_id const v = graph::detail::tree_vertex_id(key);
			atl_assert(v < tree.size());
			w.keys.push_back(p.key_of(v));
		}
		atl_assert(w.keys.size() < std::size_t(1) << 31);
		p.sort(w.keys, false);
		if (w.stack.size() < 2 * w.keys.size()) w.stack.resize(2 * w.keys.size());
		virtual_tree_by_raw_array_sweep(tree, w.keys, w.stack.data(), out);
	}

	/** @brief The depth-comparing contender: packed sort, then @ref atl::experimental::virtual_tree_by_depth_sweep. */
	template<graph::vertex_range R>
		requires std::convertible_to<std::ranges::range_reference_t<R>, graph::vertex_id>
	constexpr void virtual_tree_by_depth(graph::lca const &tree, R &&keys, virtual_tree_workspace<uint64> &w,
										 graph::virtual_tree &out, bool is_counting = false) {
		packed_entry_keys const p{tree};
		w.keys.clear();
		for (auto &&key : keys) {
			graph::vertex_id const v = graph::detail::tree_vertex_id(key);
			atl_assert(v < tree.size());
			w.keys.push_back(p.key_of(v));
		}
		atl_assert(w.keys.size() < std::size_t(1) << 31);
		p.sort(w.keys, is_counting);
		virtual_tree_by_depth_sweep(tree, w.keys, w.stack, out);
	}

	/**
	 * @brief Two-sort construction: sort, append the joins of adjacent keys, sort and
	 *        deduplicate again; the parent of each vertex is the join with its
	 *        predecessor, located on a root-path stack of entry times.
	 *        At most 2k - 2 joins and a second sort of up to 2k - 1 keys.
	 *        Comparison with a single sweep: not measured yet.
	 * @pre `w.keys` sorted by entry (duplicates allowed)
	 */
	template<virtual_tree_key_policy Policy>
	constexpr void virtual_tree_by_two_sort(Policy const &p, virtual_tree_workspace<typename Policy::key_type> &w, graph::virtual_tree &out,
											bool is_counting) {
		using key_type = typename Policy::key_type;
		auto &keys = w.keys;
		keys.erase(std::ranges::unique(keys).begin(), keys.end());
		std::size_t const k = keys.size();
		for (std::size_t i = 0; i + 1 < k; ++i) {
			key_type const l = p.join(keys[i], keys[i + 1]);
			if (l != Policy::none_key) keys.push_back(l);
		}
		p.sort(keys, is_counting);
		keys.erase(std::ranges::unique(keys).begin(), keys.end());
		std::size_t const m = keys.size();
		out.vertices.resize(m);
		out.parent.resize(m);
		auto &stack = w.stack; // frame.children holds the local index here
		stack.clear();
		for (std::size_t i = 0; i < m; ++i) {
			out.vertices[i] = p.vertex_of(keys[i]);
			key_type const l = i == 0 ? Policy::none_key : p.join(keys[i - 1], keys[i]);
			if (l == Policy::none_key) {
				stack.clear();
				out.parent[i] = graph::none;
			} else {
				while (stack.back().key != l) stack.pop_back();
				out.parent[i] = stack.back().children;
			}
			stack.push_back({keys[i], uint32(i)});
		}
	}

	/** @brief The two algorithms, for the race. */
	enum class virtual_tree_algorithm {
		stack_sweep, ///< One sort and a monotone root-path stack.
		two_sort	 ///< Add adjacent LCAs, sort again, then attach parents.
	};

	/**
	 * @brief A contender: sort the keys of @p keys in @p Policy's key space
	 *        (counting sort when @p is_counting), then build with @p Algorithm,
	 *        refilling @p out and @p w.
	 */
	template<virtual_tree_algorithm Algorithm, virtual_tree_key_policy Policy, graph::vertex_range R>
		requires std::convertible_to<std::ranges::range_reference_t<R>, graph::vertex_id>
			  && (Algorithm == virtual_tree_algorithm::stack_sweep || Algorithm == virtual_tree_algorithm::two_sort)
	constexpr void virtual_tree_by_policy(Policy const &p, R &&keys, virtual_tree_workspace<typename Policy::key_type> &w,
										  graph::virtual_tree &out, bool is_counting = false) {
		w.keys.clear();
		for (auto &&key : keys) {
			graph::vertex_id const v = graph::detail::tree_vertex_id(key);
			atl_assert(v < p.size());
			w.keys.push_back(p.key_of(v));
		}
		atl_assert(w.keys.size() < std::size_t(1) << 31);
		p.sort(w.keys, is_counting);
		if constexpr (Algorithm == virtual_tree_algorithm::stack_sweep)
			virtual_tree_by_stack_sweep(p, w, out);
		else
			virtual_tree_by_two_sort(p, w, out, is_counting);
	}

	/**
	 * @brief Key-sort alternatives: n/32 counting dispatch, pure counting, and comparison.
	 * At n=500000, k=n/128 (64 queries), bitmap takes 32.0 / 30.2 ms versus
	 * counting 53.3 / 49.0 and comparison 38.9 / 38.6; at k=n/2 the n/32 dispatch takes 20.3 / 20.0 versus bitmap 15.7 / 14.9 ms.
	 * Records: graph-virtual_tree, 2026-11-01, Xeon 8280, -O2 / native.
	 * Comparison sorting is used below n/128; n/128 .. n/1024 crossover unmeasured.
	 */
	enum class virtual_tree_sort {
		counting_dispatch, ///< Comparison below ceil(n/32), counting otherwise.
		counting,		   ///< Runtime counting sort over the entry-time domain.
		comparison		   ///< Comparison sort of packed keys.
	};

	/**
	 * @brief The library `virtual_tree_of` (same fixed stack-buffer path, same depth-ranked frames
	 *        from `64k >= n`) with the selected alternative key sort chosen by @p method.
	 *        Constant evaluation never takes the counting sort, as in the library build.
	 * @pre every key `< tree.size()` and fewer than 2^31 keys (both asserted)
	 */
	template<graph::vertex_range R>
		requires std::convertible_to<std::ranges::range_reference_t<R>, graph::vertex_id>
	constexpr void virtual_tree_by_sort(graph::lca const &tree, R &&keys, virtual_tree_sort method,
										graph::detail::virtual_tree_scratch &s, graph::virtual_tree &out) {
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
					graph::vertex_id const v = graph::detail::tree_vertex_id(key);
					atl_assert(v < n);
					sorted[i++] = uint64(tree.entry_time(v)) << 32 | v;
				}
				std::ranges::sort(sorted.begin(), sorted.begin() + std::ptrdiff_t(k));
				graph::detail::virtual_tree_sweep(tree, std::span<uint64 const>(sorted.data(), k), s.stack, false,
												  out);
				return;
			}
		}
		std::vector<uint64> &sorted = s.sorted;
		sorted.clear();
		if constexpr (std::ranges::sized_range<R>) sorted.reserve(std::size_t(std::ranges::size(keys)));
		for (auto &&key : keys) {
			graph::vertex_id const v = graph::detail::tree_vertex_id(key);
			atl_assert(v < n);
			sorted.push_back(uint64(tree.entry_time(v)) << 32 | v);
		}
		std::size_t const k = sorted.size();
		atl_assert(k < std::size_t(1) << 31);
		if (method == virtual_tree_sort::counting
			|| (method == virtual_tree_sort::counting_dispatch && k >= n / 32 + (n % 32 != 0))) {
			if consteval {
				std::ranges::sort(sorted);
			} else {
				counting_sort(sorted, [](uint64 key) { return uint32(key >> 32); });
			}
		} else {
			std::ranges::sort(sorted);
		}
		graph::detail::virtual_tree_sweep(tree, sorted, s.stack, k >= n / 64 + (n % 64 != 0), out);
	}
}
