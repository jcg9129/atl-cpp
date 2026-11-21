#pragma once
#include <atl/ds/detail/hashed_child_pool.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/strings/trie.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

/** @brief Benchmark-local construction contenders with the public query contracts. */
namespace atl::baseline {
	/**
	 * @brief Ukkonen suffix tree using separate pool lookups and local sealing scratch.
	 * @note Not constexpr (the child pool uses a process hash seed). See the matching
	 * public component for bounds and operation contracts. Traversals do not
	 * recurse per character or node. Full results: docs/suffix-tree.md.
	 * Intel Xeon Platinum 8358 @ 2.60 GHz, GCC 15.2.0, RelWithDebInfo,
	 * 2026-10-15, -O2 -g -DNDEBUG -std=c++23, optionally -march=native.
	 * Random26 1e6 word: library 575/584 ms, by separate lookups 601/611 ms.
	 */
	template<std::size_t Alphabet>
		requires(Alphabet <= (uint64(1) << 32))
	class suffix_tree_by_separate_lookups {
		static constexpr uint32 nil = uint32(-1);
		static constexpr uint32 open = uint32(-1); // edge end sentinel while a word grows

		/** @brief A transition entry: a `detail::sibling_node` (the pool manages
		 * `symbol`/`left`/`right`) carrying the explicit `target` node.
		 */
		struct transition {
			uint32 symbol = 0;
			uint32 left = nil, right = nil;
			uint32 target = nil;
		};

		struct node {
			uint32 edge_l = 0;
			uint32 edge_r = 0;
			uint32 parent = nil;
			uint32 link = nil;
			uint32 children = nil;
			uint32 length = 0;		 // |path label|; kept exact once the edge closes
			uint32 suffix_count = 0; // sealed suffixes ending exactly here
			uint32 first_suffix = nil;
		};

		::atl::detail::hashed_child_pool<transition> pool_;
		std::vector<node> nodes_;
		std::vector<uint32> text_;
		std::vector<uint32> word_begin_{0};
		std::vector<uint32> open_nodes_;
		uint32 active_node_ = 0, active_edge_ = 0, active_len_ = 0, remainder_ = 0;
		uint32 need_link_ = nil;

		[[nodiscard]] uint32 child(uint32 v, uint32 sym) const {
			uint32 const t = pool_.find(nodes_[v].children, sym);
			return t == nil ? nil : pool_[t].target;
		}

		void set_child(uint32 v, uint32 sym, uint32 target) {
			auto const placed = pool_.emplace(nodes_[v].children, sym);
			nodes_[v].children = placed.root;
			pool_[placed.node].target = target;
		}

		uint32 new_node(uint32 l, uint32 r, uint32 parent) {
			nodes_.push_back({l, r, parent, nil, nil, 0, 0, nil});
			atl_assert(nodes_.size() < (uint64(1) << 32)); // ids are uint32, nil reserved
			return uint32(nodes_.size() - 1);
		}

		[[nodiscard]] uint32 edge_length(uint32 v) const {
			return (nodes_[v].edge_r == open ? uint32(text_.size()) : nodes_[v].edge_r) - nodes_[v].edge_l;
		}

		void add_link(uint32 v) {
			if (need_link_ != nil) nodes_[need_link_].link = v;
			need_link_ = v;
		}

		/** @brief After consuming one pending suffix, hop to the next-shorter one.
		 */
		void hop() {
			--remainder_;
			if (active_node_ == 0 && active_len_ > 0) {
				--active_len_;
				active_edge_ = uint32(text_.size()) - remainder_;
			} else if (active_node_ != 0) {
				active_node_ = nodes_[active_node_].link != nil ? nodes_[active_node_].link : 0;
			}
		}

		/** @brief Split the active edge following active_len_ symbols; returns the new
		 * internal node (its edge is closed, so its length is exact).
		 */
		uint32 split_active() {
			uint32 const next = child(active_node_, text_[active_edge_]);
			uint32 const mid = new_node(nodes_[next].edge_l, nodes_[next].edge_l + active_len_, active_node_);
			nodes_[mid].length = nodes_[active_node_].length + active_len_;
			set_child(active_node_, text_[active_edge_], mid);
			nodes_[next].edge_l += active_len_;
			nodes_[next].parent = mid;
			set_child(mid, text_[nodes_[next].edge_l], next);
			return mid;
		}

	public:
		/** @brief A handle to a node. Stable for the tree's lifetime; the root is 0.
		 * Ids are creation order — NOT topological, see @ref atl::baseline::suffix_tree_by_separate_lookups::nodes_by_length.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		using node_id = uint32;

		/** @brief One sealed suffix: word @p word starting at @p offset.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		struct suffix_ref {
			uint32 word, offset;
		};

		/** @brief A node's incoming edge label: @p length symbols of word @p word
		 * starting at @p offset.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		struct edge_ref {
			uint32 word, offset, length;
		};

		/** @brief The empty tree: just the root.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		suffix_tree_by_separate_lookups() { nodes_.push_back({0, 0, nil, nil, nil, 0, 0, nil}); }

		/**
		 * @brief Append one symbol to the word in progress (one Ukkonen
		 *        extension).
		 * @return Length of the longest suffix of the current word that also
		 *         occurs at an other position (in any word, or other in
		 *         this one) — the online longest repeated suffix.
		 * @complexity Expected amortized O(log(degree + 1))
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		template<std::integral Sym>
		uint32 extend(Sym sym) {
			atl_assert(atl::detail::symbol_below(sym, Alphabet)); // in [0, Alphabet)
			uint32 const c = uint32(sym);
			atl_assert(text_.size() < (uint64(1) << 32) - 2);
			uint32 const pos = uint32(text_.size());
			text_.push_back(c);
			atl_assert(text_.size() < (uint64(1) << 32)); // positions are uint32
			need_link_ = nil;
			++remainder_;
			while (remainder_ > 0) {
				if (active_len_ == 0) active_edge_ = pos;
				if (uint32 const next = child(active_node_, text_[active_edge_]); next == nil) {
					uint32 const leaf = new_node(pos, open, active_node_);
					open_nodes_.push_back(leaf);
					set_child(active_node_, text_[active_edge_], leaf);
					add_link(active_node_);
				} else {
					if (active_len_ >= edge_length(next)) { // canonicalize: walk down
						active_edge_ += edge_length(next);
						active_len_ -= edge_length(next);
						active_node_ = next;
						continue;
					}
					if (text_[nodes_[next].edge_l + active_len_] == c) { // already present
						++active_len_;
						add_link(active_node_);
						break;
					}
					uint32 const mid = split_active();
					uint32 const leaf = new_node(pos, open, mid);
					open_nodes_.push_back(leaf);
					set_child(mid, c, leaf);
					add_link(mid);
				}
				hop();
			}
			return nodes_[active_node_].length + active_len_;
		}

		/**
		 * @brief Seal the word in progress — every suffix of it becomes
		 *        explicit, marked at the node where it ends (@ref
		 *        suffix_count; edges are split where a suffix ends mid-edge,
		 *        with no terminator stored) — and reset for the next word.
		 * @return The finished word's index.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		uint32 finish_word() {
			atl_assert(word_begin_.size() < (uint64(1) << 32) - 1);
			uint32 const word_end = uint32(text_.size());
			uint32 const w = uint32(word_begin_.size() - 1);
			uint32 const word_len = word_end - word_begin_.back();
			// which node each of this word's suffixes ends at, by offset
			std::vector<uint32> end_node(word_len, nil);
			// the virtual-terminator pass: it matches nothing, so every
			// pending suffix ends at a node (marking it) or mid-edge
			// (splitting first); the suffix-link hops mirror extend's
			need_link_ = nil;
			while (remainder_ > 0) {
				uint32 v = active_node_;
				if (active_len_ != 0) {
					uint32 const next = child(active_node_, text_[active_edge_]);
					if (active_len_ >= edge_length(next)) { // canonicalize
						active_edge_ += edge_length(next);
						active_len_ -= edge_length(next);
						active_node_ = next;
						continue;
					}
					v = split_active();
				}
				atl_assert(nodes_[v].suffix_count < (std::numeric_limits<uint32>::max)());
				++nodes_[v].suffix_count;
				if (nodes_[v].first_suffix == nil) nodes_[v].first_suffix = word_end - remainder_;
				end_node[word_len - remainder_] = v;
				add_link(v);
				hop();
			}
			// close this word's leaf edges and freeze their lengths; each
			// leaf is the (unique-extension) suffix from its recorded start
			for (uint32 const leaf : open_nodes_) {
				nodes_[leaf].edge_r = word_end;
				nodes_[leaf].length = nodes_[nodes_[leaf].parent].length + edge_length(leaf);
				atl_assert(nodes_[leaf].suffix_count < (std::numeric_limits<uint32>::max)());
				++nodes_[leaf].suffix_count;
				if (nodes_[leaf].first_suffix == nil)
					nodes_[leaf].first_suffix = word_end - nodes_[leaf].length;
				end_node[word_len - nodes_[leaf].length] = leaf;
			}
			// complete the suffix links: the node of suffix k links to the
			// node of suffix k+1 (its path minus the first symbol). Leaves
			// need this — a later word's active point can walk INTO a closed
			// leaf, and the hop following it follows the leaf's link, a case
			// single-word Ukkonen never has.
			for (uint32 k = 0; k < word_len; ++k) {
				uint32 const v = end_node[k];
				atl_assert(v != nil); // every suffix ends somewhere
				if (nodes_[v].link == nil) nodes_[v].link = k + 1 < word_len ? end_node[k + 1] : 0;
			}
			open_nodes_.clear();
			word_begin_.push_back(word_end);
			active_node_ = active_edge_ = active_len_ = remainder_ = 0;
			need_link_ = nil;
			return w;
		}

		/**
		 * @brief Add one whole word — @ref atl::baseline::suffix_tree_by_separate_lookups::extend each projected symbol, then
		 *        @ref atl::baseline::suffix_tree_by_separate_lookups::finish_word.
		 * @return The word's index.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		uint32 insert(R &&word, Proj proj = {}) {
			for (auto &&e : atl::detail::input_elements(word)) extend(std::invoke(proj, std::forward<decltype(e)>(e)));
			return finish_word();
		}

		/** @brief The root — the empty path (id 0).
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] node_id root() const { return 0; }

		/** @brief @p v's parent in the tree. @pre `v != root()`
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] node_id parent(node_id v) const {
			atl_assert(v != 0 && v < nodes_.size());
			return nodes_[v].parent;
		}

		/** @brief The node whose path label is @p v's minus its first symbol.
		 * Present for every internal node and every sealed suffix-end node
		 * (leaves included); `std::nullopt` only for a still-open leaf of
		 * the word in progress.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] std::optional<node_id> suffix_link(node_id v) const {
			atl_assert(v != 0 && v < nodes_.size());
			return nodes_[v].link == nil ? std::nullopt : std::optional<node_id>(nodes_[v].link);
		}

		/** @brief |path label| of @p v. O(1); a still-open leaf's label grows with
		 * the word in progress.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] uint32 length(node_id v) const {
			atl_assert(v < nodes_.size());
			if (nodes_[v].edge_r != open) return nodes_[v].length;
			return nodes_[nodes_[v].parent].length + edge_length(v);
		}

		/** @brief The child whose edge label starts with @p sym, or `std::nullopt`.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		template<std::integral Sym>
		[[nodiscard]] std::optional<node_id> walk(node_id v, Sym sym) const {
			atl_assert(v < nodes_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return std::nullopt;
			uint32 const c = child(v, uint32(sym));
			return c == nil ? std::nullopt : std::optional<node_id>(c);
		}

		/** @brief Visit every child of @p v, in unspecified (hashed) order.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		template<typename Visit>
			requires std::invocable<Visit &, node_id>
		void for_each_child(node_id v, Visit visit) const {
			atl_assert(v < nodes_.size());
			pool_.for_each(nodes_[v].children, [&](uint32 t) { std::invoke(visit, pool_[t].target); });
		}

		/**
		 * @p v's incoming edge label, as a slice of a word. @pre `v != root()`
		 * @complexity O(log(word_count() + 1)), a binary search for the word
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] edge_ref edge(node_id v) const {
			atl_assert(v != 0 && v < nodes_.size());
			uint32 const w = word_of(nodes_[v].edge_l);
			return {w, nodes_[v].edge_l - word_begin_[w], edge_length(v)};
		}

		/** @brief The stored (projected) symbols of finished word @p w.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] std::span<uint32 const> word(uint32 w) const {
			atl_assert(std::size_t(w) < word_begin_.size() - 1);
			return std::span<uint32 const>(text_).subspan(word_begin_[w], word_begin_[w + 1] - word_begin_[w]);
		}

		/** @brief Sealed suffixes ending exactly at @p v (with multiplicity across
		 * duplicate words).
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] uint64 suffix_count(node_id v) const {
			atl_assert(v < nodes_.size());
			return nodes_[v].suffix_count;
		}

		/**
		 * @brief One suffix ending at @p v. @pre `suffix_count(v) > 0`
		 * @complexity O(log(word_count() + 1)), a binary search for the word
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] suffix_ref suffix(node_id v) const {
			atl_assert(v < nodes_.size() && nodes_[v].suffix_count > 0);
			uint32 const w = word_of(nodes_[v].first_suffix);
			return {w, nodes_[v].first_suffix - word_begin_[w]};
		}

		/**
		 * @brief Occurrences of each node's path label across all sealed
		 *        words: the subtree sums of @ref atl::baseline::suffix_tree_by_separate_lookups::suffix_count, one pass over
		 *        @ref atl::baseline::suffix_tree_by_separate_lookups::nodes_by_length in reverse. The root entry is the
		 *        total suffix count.
		 * @complexity O(node_count() + longest word length)
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] std::vector<uint64> occurrence_counts() const {
			std::vector<uint64> occ(nodes_.size());
			for (uint32 v = 0; v < uint32(nodes_.size()); ++v) occ[v] = nodes_[v].suffix_count;
			auto const order = nodes_by_length();
			for (auto i = order.size(); i-- > 1;) occ[nodes_[order[i]].parent] += occ[order[i]];
			return occ;
		}

		/**
		 * @brief All nodes in increasing @ref atl::baseline::suffix_tree_by_separate_lookups::length order — a topological
		 *        order of the tree (parents strictly shorter), which creation
		 *        order is NOT (edge splits insert parents late).
		 * @complexity O(node_count() + longest word length), with heap-backed counting sort
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		[[nodiscard]] std::vector<node_id> nodes_by_length() const {
			uint32 max_len = 0;
			for (uint32 v = 0; v < uint32(nodes_.size()); ++v) max_len = (std::max)(max_len, length(v));
			atl_assert(uint64(max_len) + 2 <= (std::numeric_limits<std::size_t>::max)());
			std::vector<uint32> bucket(std::size_t(max_len) + 2, 0);
			for (uint32 v = 0; v < uint32(nodes_.size()); ++v) ++bucket[std::size_t(length(v)) + 1];
			for (std::size_t l = 1; l < bucket.size(); ++l) bucket[l] += bucket[l - 1];
			std::vector<node_id> order(nodes_.size());
			for (uint32 v = 0; v < uint32(nodes_.size()); ++v) order[bucket[length(v)]++] = v;
			return order;
		}

		/** @brief Number of nodes, root included.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		std::size_t node_count() const { return nodes_.size(); }

		/** @brief Number of finished words.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		std::size_t word_count() const { return word_begin_.size() - 1; }

		/** @brief Total symbols fed (finished words + the word in progress).
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		std::size_t size() const { return text_.size(); }

		/** @brief Pre-allocate for @p symbols total input symbols.
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		void reserve(std::size_t symbols) {
			atl_assert(symbols <= ((std::numeric_limits<std::size_t>::max)() - 1) / 2);
			atl_assert(uint64(symbols) <= ((uint64(1) << 32) - 2) / 2);
			nodes_.reserve(2 * symbols + 1);
			pool_.reserve(2 * symbols);
			text_.reserve(symbols);
		}

		/** @brief Back to the empty tree (memory kept).
		 * @note Not constexpr (the child pool uses a process hash seed).
		 */
		void clear() {
			pool_.clear();
			nodes_.clear();
			nodes_.push_back({0, 0, nil, nil, nil, 0, 0, nil});
			text_.clear();
			word_begin_.assign(1, 0);
			open_nodes_.clear();
			active_node_ = active_edge_ = active_len_ = remainder_ = 0;
			need_link_ = nil;
		}

	private:
		[[nodiscard]] uint32 word_of(uint32 pos) const {
			auto const it = std::ranges::upper_bound(word_begin_, pos);
			return uint32(it - word_begin_.begin()) - 1;
		}
	};
}
