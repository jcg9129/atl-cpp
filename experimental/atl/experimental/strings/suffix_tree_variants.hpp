#pragma once
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

/**
 * @file
 * @brief The child-storage contender that lost the suffix-tree race to the
 *        library `atl::suffix_tree` (transitions in a hashed child pool).
 */
/** @brief Alternative string algorithms retained for comparison and certification. */
namespace atl::experimental {
	/**
	 * @brief The (generalized) suffix tree with the classic first-child /
	 *        next-sibling list for transitions: the same online Ukkonen
	 *        construction, sealing and interface as @ref atl::suffix_tree,
	 *        only the child storage differs.
	 *
	 * A node's children form a singly linked list threaded through the
	 * nodes themselves (no separate transition entries); a child is found by
	 * scanning the list and comparing the first symbol of each edge label in
	 * the retained text. New children are pushed at the head, so
	 * @ref atl::experimental::suffix_tree_by_sibling_list::for_each_child visits them newest first. Node ids follow the same
	 * creation order as the library tree, which makes the two differentially
	 * testable node by node.
	 *
	 * RACE RESULT (why this lives in experimental), building one random
	 * 10^6-symbol word (measured using benchmarks/strings/suffix_tree.cpp,
	 * 2026-10-15, Xeon Platinum 8358, x86-64; plain -O2 / -march=native):
	 * the list wins only sigma = 2 (269 / 277 ms vs the pool's 399 / 404);
	 * the pool wins sigma = 26 (560 / 577 ms vs 1206 / 1218) and is the only
	 * viable option for sparse alphabets (sigma = 10^6: pool 672 / 691 ms;
	 * the list is not timed there, every root lookup is an O(degree) scan).
	 * Kept as the known tiny-alphabet lever and for re-measurement
	 * (benchmarks/strings/suffix_tree.cpp).
	 *
	 * @tparam Alphabet Exclusive upper bound on symbols; checked on extend.
	 */
	template<std::size_t Alphabet>
		requires(Alphabet <= (uint64(1) << 32))
	class suffix_tree_by_sibling_list {
		static constexpr uint32 nil = uint32(-1);
		static constexpr uint32 open = uint32(-1); // edge end sentinel while a word grows

		/** @brief A node: its incoming edge label, tree and suffix links, its
		 *         place in its parent's child list, and the sealing counts. */
		struct node {
			uint32 edge_l = 0;
			uint32 edge_r = 0;
			uint32 parent = nil;
			uint32 link = nil;		   // suffix link
			uint32 first_child = nil;  // head of the child list
			uint32 next_sibling = nil; // next entry in the parent's child list
			uint32 length = 0;		   // |path label|; kept exact once the edge closes
			uint32 suffix_count = 0;   // sealed suffixes ending exactly here
			uint32 first_suffix = nil;
		};

		std::vector<node> nodes_;
		std::vector<uint32> text_;
		std::vector<uint32> word_begin_{0};
		std::vector<uint32> open_nodes_;
		uint32 active_node_ = 0, active_edge_ = 0, active_len_ = 0, remainder_ = 0;
		uint32 need_link_ = nil;

		/** @brief The child of @p v whose edge label starts with @p sym, or
		 *         `nil`; an O(degree) list scan. */
		[[nodiscard]] constexpr uint32 child(uint32 v, uint32 sym) const {
			for (uint32 c = nodes_[v].first_child; c != nil; c = nodes_[c].next_sibling)
				if (text_[nodes_[c].edge_l] == sym) return c;
			return nil;
		}

		/** @brief Push @p c at the head of @p v's child list. */
		constexpr void add_child(uint32 v, uint32 c) {
			nodes_[c].next_sibling = nodes_[v].first_child;
			nodes_[v].first_child = c;
		}

		/** @brief Put @p fresh into @p old's slot of @p v's child list and
		 *         unlink @p old. @pre @p old is a child of @p v */
		constexpr void replace_child(uint32 v, uint32 old, uint32 fresh) {
			uint32 *slot = &nodes_[v].first_child;
			while (*slot != old) slot = &nodes_[*slot].next_sibling;
			*slot = fresh;
			nodes_[fresh].next_sibling = nodes_[old].next_sibling;
			nodes_[old].next_sibling = nil;
		}

		/** @brief Append a node with edge label `[l, r)` under @p parent
		 *         (not yet linked into its child list); returns its id. */
		constexpr uint32 new_node(uint32 l, uint32 r, uint32 parent) {
			nodes_.push_back({l, r, parent, nil, nil, nil, 0, 0, nil});
			atl_assert(nodes_.size() < (uint64(1) << 32)); // ids are uint32, nil reserved
			return uint32(nodes_.size() - 1);
		}

		/** @brief The length of @p v's incoming edge label, an open edge
		 *         reaching the end of the text. */
		[[nodiscard]] constexpr uint32 edge_length(uint32 v) const {
			return (nodes_[v].edge_r == open ? uint32(text_.size()) : nodes_[v].edge_r) - nodes_[v].edge_l;
		}

		/** @brief Point the pending suffix link at @p v, which becomes the
		 *         next node awaiting one. */
		constexpr void add_link(uint32 v) {
			if (need_link_ != nil) nodes_[need_link_].link = v;
			need_link_ = v;
		}

		/** @brief After consuming one pending suffix, hop to the next-shorter
		 *         one. */
		constexpr void hop() {
			--remainder_;
			if (active_node_ == 0 && active_len_ > 0) {
				--active_len_;
				active_edge_ = uint32(text_.size()) - remainder_;
			} else if (active_node_ != 0) {
				active_node_ = nodes_[active_node_].link != nil ? nodes_[active_node_].link : 0;
			}
		}

		/** @brief Split the active edge following `active_len_` symbols; returns
		 *         the new internal node (its edge is closed, so its length is
		 *         exact). */
		constexpr uint32 split_active() {
			uint32 const next = child(active_node_, text_[active_edge_]);
			uint32 const mid = new_node(nodes_[next].edge_l, nodes_[next].edge_l + active_len_, active_node_);
			nodes_[mid].length = nodes_[active_node_].length + active_len_;
			replace_child(active_node_, next, mid);
			nodes_[next].edge_l += active_len_;
			nodes_[next].parent = mid;
			add_child(mid, next);
			return mid;
		}

		[[nodiscard]] constexpr uint32 word_of(uint32 pos) const {
			auto const it = std::ranges::upper_bound(word_begin_, pos);
			return uint32(it - word_begin_.begin()) - 1;
		}

	public:
		/** @brief A handle to a node; the root is 0. Ids are creation order,
		 *         not topological (see @ref atl::experimental::suffix_tree_by_sibling_list::nodes_by_length).
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		using node_id = uint32;

		/** @brief One sealed suffix: word @p word starting at @p offset.
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		struct suffix_ref {
			uint32 word, offset;
		};

		/** @brief A node's incoming edge label: @p length symbols of word
		 *         @p word starting at @p offset.
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		struct edge_ref {
			uint32 word, offset, length;
		};

		/** @brief The empty tree: just the root.
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		constexpr suffix_tree_by_sibling_list() { nodes_.push_back({0, 0, nil, nil, nil, nil, 0, 0, nil}); }

		/**
		 * @brief Append one symbol to the word in progress (one Ukkonen
		 *        extension).
		 * @return Length of the longest suffix of the current word that also
		 *         occurs at an other position.
		 * @complexity amortized O(degree) — the child-list scans
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		template<std::integral Sym>
		constexpr uint32 extend(Sym sym) {
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
					add_child(active_node_, leaf);
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
					add_child(mid, leaf);
					add_link(mid);
				}
				hop();
			}
			return nodes_[active_node_].length + active_len_;
		}

		/**
		 * @brief Seal the word in progress — every suffix of it becomes
		 *        explicit, marked at the node where it ends — and reset for
		 *        the next word.
		 * @return The finished word's index.
		 * @complexity O(word length · degree)
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		constexpr uint32 finish_word() {
			atl_assert(word_begin_.size() < (uint64(1) << 32) - 1);
			uint32 const word_end = uint32(text_.size());
			uint32 const w = uint32(word_begin_.size() - 1);
			uint32 const word_len = word_end - word_begin_.back();
			std::vector<uint32> end_node(word_len, nil);
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
			for (uint32 const leaf : open_nodes_) {
				nodes_[leaf].edge_r = word_end;
				nodes_[leaf].length = nodes_[nodes_[leaf].parent].length + edge_length(leaf);
				atl_assert(nodes_[leaf].suffix_count < (std::numeric_limits<uint32>::max)());
				++nodes_[leaf].suffix_count;
				if (nodes_[leaf].first_suffix == nil) nodes_[leaf].first_suffix = word_end - nodes_[leaf].length;
				end_node[word_len - nodes_[leaf].length] = leaf;
			}
			// leaves need links too: a later word's active point can walk
			// into a closed leaf and hop along its link
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
		 * @brief Add one whole word — @ref atl::experimental::suffix_tree_by_sibling_list::extend each projected symbol, then
		 *        @ref atl::experimental::suffix_tree_by_sibling_list::finish_word.
		 * @return The word's index.
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		constexpr uint32 insert(R &&word, Proj proj = {}) {
			for (auto &&e : atl::detail::input_elements(word)) extend(std::invoke(proj, std::forward<decltype(e)>(e)));
			return finish_word();
		}

		/** @brief The root — the empty path (id 0).
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		[[nodiscard]] constexpr node_id root() const { return 0; }

		/** @brief @p v's parent in the tree. @pre `v != root()`
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		[[nodiscard]] constexpr node_id parent(node_id v) const {
			atl_assert(v != 0 && v < nodes_.size());
			return nodes_[v].parent;
		}

		/** @brief The node whose path label is @p v's minus its first symbol;
		 *         `std::nullopt` only for a still-open leaf. @pre `v != root()`
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		[[nodiscard]] constexpr std::optional<node_id> suffix_link(node_id v) const {
			atl_assert(v != 0 && v < nodes_.size());
			return nodes_[v].link == nil ? std::nullopt : std::optional<node_id>(nodes_[v].link);
		}

		/** @brief |path label| of @p v; a still-open leaf's grows with the
		 *         word in progress.
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		[[nodiscard]] constexpr uint32 length(node_id v) const {
			atl_assert(v < nodes_.size());
			if (nodes_[v].edge_r != open) return nodes_[v].length;
			return nodes_[nodes_[v].parent].length + edge_length(v);
		}

		/** @brief The child whose edge label starts with @p sym, or
		 *         `std::nullopt`. @complexity O(degree)
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		template<std::integral Sym>
		[[nodiscard]] constexpr std::optional<node_id> walk(node_id v, Sym sym) const {
			atl_assert(v < nodes_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return std::nullopt;
			uint32 const c = child(v, uint32(sym));
			return c == nil ? std::nullopt : std::optional<node_id>(c);
		}

		/** @brief Visit every child of @p v, newest first.
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		template<typename Visit>
			requires std::invocable<Visit &, node_id>
		constexpr void for_each_child(node_id v, Visit visit) const {
			atl_assert(v < nodes_.size());
			for (uint32 c = nodes_[v].first_child; c != nil; c = nodes_[c].next_sibling) visit(c);
		}

		/**
		 * @brief @p v's incoming edge label, as a slice of a word.
		 * @pre `v != root()`
		 * @complexity O(log(word_count() + 1)), a binary search for the word
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		constexpr edge_ref edge(node_id v) const {
			atl_assert(v != 0 && v < nodes_.size());
			uint32 const w = word_of(nodes_[v].edge_l);
			return {w, nodes_[v].edge_l - word_begin_[w], edge_length(v)};
		}

		/** @brief The stored (projected) symbols of finished word @p w.
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		[[nodiscard]] constexpr std::span<uint32 const> word(uint32 w) const {
			atl_assert(std::size_t(w) < word_begin_.size() - 1);
			return std::span<uint32 const>(text_).subspan(word_begin_[w], word_begin_[w + 1] - word_begin_[w]);
		}

		/** @brief Sealed suffixes ending exactly at @p v.
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		[[nodiscard]] constexpr uint64 suffix_count(node_id v) const {
			atl_assert(v < nodes_.size());
			return nodes_[v].suffix_count;
		}

		/**
		 * @brief One suffix ending at @p v. @pre `suffix_count(v) > 0`
		 * @complexity O(log(word_count() + 1)), a binary search for the word
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		constexpr suffix_ref suffix(node_id v) const {
			atl_assert(v < nodes_.size() && nodes_[v].suffix_count > 0);
			uint32 const w = word_of(nodes_[v].first_suffix);
			return {w, nodes_[v].first_suffix - word_begin_[w]};
		}

		/**
		 * @brief Occurrences of each node's path label across all sealed
		 *        words: the subtree sums of @ref atl::experimental::suffix_tree_by_sibling_list::suffix_count.
		 * @complexity O(node_count() + longest word length)
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		constexpr std::vector<uint64> occurrence_counts() const {
			std::vector<uint64> occ(nodes_.size());
			for (uint32 v = 0; v < uint32(nodes_.size()); ++v) occ[v] = nodes_[v].suffix_count;
			auto const order = nodes_by_length();
			for (auto i = order.size(); i-- > 1;) occ[nodes_[order[i]].parent] += occ[order[i]];
			return occ;
		}

		/**
		 * @brief All nodes in increasing @ref atl::experimental::suffix_tree_by_sibling_list::length order — a topological
		 *        order of the tree, which creation order is not.
		 * @complexity O(node_count() + longest word length), with heap-backed counting sort
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		constexpr std::vector<node_id> nodes_by_length() const {
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
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		constexpr std::size_t node_count() const { return nodes_.size(); }

		/** @brief Number of finished words.
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		constexpr std::size_t word_count() const { return word_begin_.size() - 1; }

		/** @brief Total symbols fed (finished words + the word in progress).
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		constexpr std::size_t size() const { return text_.size(); }

		/** @brief Pre-allocate for @p symbols total input symbols.
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		constexpr void reserve(std::size_t symbols) {
			atl_assert(symbols <= ((std::numeric_limits<std::size_t>::max)() - 1) / 2);
			atl_assert(uint64(symbols) <= ((uint64(1) << 32) - 2) / 2);
			nodes_.reserve(2 * symbols + 1);
			text_.reserve(symbols);
		}

		/** @brief Back to the empty tree (memory kept).
		 * @note Not constexpr (this contender mirrors the runtime-only tree interface).
		 */
		constexpr void clear() {
			nodes_.clear();
			nodes_.push_back({0, 0, nil, nil, nil, nil, 0, 0, nil});
			text_.clear();
			word_begin_.assign(1, 0);
			open_nodes_.clear();
			active_node_ = active_edge_ = active_len_ = remainder_ = 0;
			need_link_ = nil;
		}
	};
}
