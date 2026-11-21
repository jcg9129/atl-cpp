#pragma once
#include <atl/ds/detail/hashed_child_pool.hpp>
#include <atl/strings/dense_transitions.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/strings/trie.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <atl/utils/counting_sort.hpp>
#include <atl/utils/prefetch.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief The Aho–Corasick automaton over the symbol alphabet
	 *        `[0, Alphabet)`: the trie of a pattern collection plus suffix
	 *        links, frozen at construction, laid out for a large sparse
	 *        alphabet. Scan any text through @ref atl::aho_corasick automata::step and every pattern
	 *        occurrence is found — @ref atl::aho_corasick automata::match_count per position in O(1).
	 *
	 * Nodes are the distinct pattern prefixes, ids in first-touch creation
	 * order (insert order over the words, then depth). @ref atl::aho_corasick automata::suffix_link
	 * points to the node of the longest proper suffix that is itself a
	 * pattern prefix; links can point to YOUNGER ids, so link-tree DPs
	 * iterate @ref atl::aho_corasick automata::nodes_by_depth (BFS order, parents preceding children in
	 * both trees) — unlike the palindromic_tree, creation order is NOT topological
	 * here.
	 *
	 * The node fuses into a shared hashed child-pool entry like the
	 * trie's (~40 B/node plus 8 B/node and 8 B/pattern of side arrays, at
	 * most total pattern length + 1 nodes). There are
	 * no per-node transition tables: @ref atl::aho_corasick automata::step resolves a missing edge by
	 * walking suffix links, which is amortized O(log(1 + degree)) expected per symbol over a
	 * forward text scan (the KMP argument) and what keeps `Alphabet` up to
	 * 10^6 free.
	 *
	 * Patterns form a multiset: duplicates count with multiplicity in
	 * @ref atl::aho_corasick automata::word_count / @ref atl::aho_corasick automata::match_count. The empty pattern ends at the
	 * root and is counted as matching at every position.
	 *
	 * Match enumeration is output-sensitive: at scan state `v`, the patterns
	 * ending at the current position are @ref atl::aho_corasick automata::words_at of `v` (when
	 * @ref atl::aho_corasick automata::word_count is nonzero) and of each node down its
	 * @ref atl::aho_corasick automata::output_link chain — O(1 + matches), never a walk over suffix
	 * links that end no pattern.
	 *
	 * @tparam Alphabet Exclusive upper bound on symbols; checked on build.
	 *         Text symbols outside the alphabet are legal in @ref atl::aho_corasick automata::step and
	 *         simply reset the scan to the root.
	 *
	 * @pre Pattern count fits uint32; a new node requires fewer than 2^32 - 1
	 *       existing nodes; projected pattern symbols lie in `[0, Alphabet)` (asserted).
	 * @note Not constexpr (captures the process hash seed). Construct from main or later.
	 *       Hash lookup bounds are expected, with linear worst-case sibling height.
	 *       All traversal uses loops or heap stacks, without recursion per symbol or node.
	 *       Signed characters retain their numeric value: project to unsigned char for bytes.
	 *       Pattern ranges may be non-contiguous and single-pass. Allocation/projection
	 *       exceptions propagate. Spans remain valid until assignment or destruction;
	 *       moving transfers their storage. A moved-from automaton may only be assigned
	 *       or destroyed.
	 */
	template<std::size_t Alphabet>
		requires(Alphabet <= (uint64(1) << 32))
	class aho_corasick {
		static constexpr uint32 nil = uint32(-1);

		/**
		 * @brief A fused pool entry: the pool manages `symbol`/`left`/`right` (the
		 *        parent's sibling BST); the rest is the prefix's payload.
		 */
		struct node {
			uint32 symbol = 0;
			uint32 left = nil, right = nil;
			uint32 children = nil;
			uint32 parent = nil; // this prefix minus its last symbol; nil at the root
			uint32 link = nil;	 // longest proper suffix that is a pattern prefix
			uint32 output = nil; // nearest proper suffix ending a pattern; nil if none
			uint32 length = 0;
			uint32 words = 0;
			uint32 matches = 0;
		};

		detail::hashed_child_pool<node> pool_;
		std::vector<uint32> word_node_;	  // end node of each inserted pattern
		std::vector<uint32> by_depth_;	  // BFS order; the topological order for DPs
		std::vector<uint32> word_offset_; // node -> first entry of its run in word_ids_
		std::vector<uint32> word_ids_;	  // pattern ids bucketed by end node, ascending per node

		[[nodiscard]] uint32 child(uint32 v, uint32 sym) const { return pool_.find(pool_[v].children, sym); }

		void bucket_words() {
			atl_assert(std::size_t(pool_.size()) < word_offset_.max_size());
			word_offset_.resize(std::size_t(pool_.size()) + 1);
			word_ids_.resize(word_node_.size());
			detail::bucket_by_key(
				pool_.size(), [&](std::size_t const i) { return word_node_[i]; }, word_offset_, word_ids_);
		}

		/**
		 * @brief Fill @p next (root row zeroed, `node_count() × sigma`) with the
		 *        transition function, for @ref atl::aho_corasick automata::transition_table.
		 *
		 * BFS order: suffix_link(v) is shallower, so its row is final when v
		 * copies it; v's children then overwrite their own columns. The ctor
		 * enqueued each node's children as one run, in the order the nodes
		 * themselves are dequeued, so one cursor over @ref atl::aho_corasick automata::by_depth_ yields
		 * them without touching the sibling BSTs. Rows are indexed by id, not
		 * BFS position, so on a large table every node costs a few cache
		 * misses; @p IsPrefetched requests them a few nodes ahead.
		 */
		template<bool IsPrefetched>
		void fill_transitions(uint32 *const next, std::size_t const sigma) const {
			constexpr std::size_t ahead = 8;
			std::size_t const count = by_depth_.size();
			std::size_t next_child = 1;
			for (std::size_t i = 0; i < count; ++i) {
				if constexpr (IsPrefetched) {
					if (i + 2 * ahead < count) atl::prefetch(&pool_[by_depth_[i + 2 * ahead]]);
					if (i + ahead < count) {
						uint32 const w = by_depth_[i + ahead];
						atl::prefetch(next + std::size_t(w) * sigma);
						atl::prefetch(next + std::size_t(pool_[w].link) * sigma);
					}
					if (next_child + ahead < count) atl::prefetch(&pool_[by_depth_[next_child + ahead]]);
				}
				uint32 const v = by_depth_[i];
				uint32 *const row = next + std::size_t(v) * sigma;
				if (v != 0) {
					uint32 const *const fallback = next + std::size_t(pool_[v].link) * sigma;
					std::ranges::copy(fallback, fallback + sigma, row);
				}
				for (; next_child < count; ++next_child) {
					node const &u = pool_[by_depth_[next_child]];
					if (u.parent != v) break;
					if (u.symbol < sigma) row[u.symbol] = by_depth_[next_child];
				}
			}
		}

	public:
		/**
		 * @brief A handle to a node — a distinct pattern prefix. The root (empty
		 *        prefix) is id 0; ids are creation order and stable.
		 */
		using node_id = uint32;

		/**
		 * @brief Build from @p words, each element a pattern whose symbols
		 *        are seen through @p proj. The whole automaton — trie,
		 *        suffix links, match counts — is frozen here.
		 * @complexity O(n + L·log(1 + degree)) expected, for n patterns and L symbols
		 * @note Not constexpr (runtime node storage and hashing).
		 * @tparam R An input range of words.
		 * @tparam Proj Lvalue-invocable projection returning an integral symbol.
		 * @param words The projected pattern collection.
		 * @param proj Symbol projection.
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word_range<R, Proj>
		explicit aho_corasick(R &&words, Proj proj = {}) {
			pool_.add();
			if constexpr (std::ranges::sized_range<R>) {
				auto const count = std::ranges::size(words);
				atl_assert(count <= nil);
				word_node_.reserve(std::size_t(count));
			}
			for (auto &&w : atl::detail::input_elements(words)) {
				atl_assert(word_node_.size() < nil);
				uint32 v = 0;
				for (auto &&e : atl::detail::input_elements(w)) {
					auto const s = std::invoke(proj, std::forward<decltype(e)>(e));
					atl_assert(atl::detail::symbol_below(s, Alphabet)); // in [0, Alphabet)
					auto const placed = pool_.emplace(pool_[v].children, uint32(s));
					pool_[v].children = placed.root;
					if (placed.was_inserted) {
						pool_[placed.node].parent = v;
						pool_[placed.node].length = pool_[v].length + 1;
					}
					v = placed.node;
				}
				++pool_[v].words;
				word_node_.push_back(v);
			}
			// suffix links + match counts, BFS: parents (in both trees) are
			// settled in increasing depth, so step() below only ever
			// touches finished nodes
			by_depth_.reserve(pool_.size());
			by_depth_.push_back(0);
			for (std::size_t i = 0; i < by_depth_.size(); ++i) {
				uint32 const v = by_depth_[i];
				if (v != 0) {
					uint32 const link = pool_[v].link;
					pool_[v].matches = pool_[v].words + pool_[link].matches;
					pool_[v].output = pool_[link].words != 0 ? link : pool_[link].output;
				} else {
					pool_[v].matches = pool_[v].words;
				}
				pool_.for_each(pool_[v].children, [&](uint32 u) {
					pool_[u].link = v == 0 ? 0 : step(pool_[v].link, pool_[u].symbol);
					by_depth_.push_back(u);
				});
			}
			bucket_words();
		}

		/**
		 * @brief The automaton of no patterns: just the root, at which every
		 *        @ref atl::aho_corasick automata::step stays.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		aho_corasick() : by_depth_{0}, word_offset_{0, 0} { pool_.add(); }

		/** @brief The root — the empty prefix (id 0), where a text scan starts.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] node_id root() const { return 0; }

		/**
		 * @brief The matching transition: the state reached by reading @p sym in
		 *        state @p v — the longest suffix of the extended text that is
		 *        a pattern prefix. Total (never fails): a symbol leading
		 *        nowhere, including one outside the alphabet, resets toward
		 *        the root.
		 * @complexity Amortized O(log degree) per symbol over a forward scan.
		 *             A single call is worst case O(length(v)) hops, each
		 *             expected O(log degree): it walks the suffix-link chain
		 *             with one child lookup per hop. Tabulating every
		 *             (state, symbol) pair through step therefore costs up to
		 *             O(nodes · depth · sigma), e.g. Θ(k² σ) on the pattern a^k;
		 *             @ref atl::aho_corasick automata::transition_table builds the same table in
		 *             O(nodes · sigma).
		 * @note Not constexpr (runtime node storage and hashing).
		 * @tparam Sym Integral symbol type.
		 * @param sym Numeric edge label; out-of-alphabet values miss or reset.
		 * @pre The node handle belongs to this object (asserted).
		 */
		template<std::integral Sym>
		[[nodiscard]] node_id step(node_id v, Sym sym) const {
			atl_assert(v < pool_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return 0;
			uint32 const c = uint32(sym);
			for (;;) {
				if (uint32 const t = child(v, c); t != nil) return t;
				if (v == 0) return 0;
				v = pool_[v].link;
			}
		}

		/**
		 * @brief The full matching transition function restricted to the
		 *        symbols `[0, sigma)`: `table(v, c) == step(v, c)` for every
		 *        node `v` and `c < sigma`. Pattern symbols `>= sigma` are
		 *        simply unreachable through the table.
		 * @pre `1 <= sigma <= Alphabet`; the `node_count() × sigma` table is
		 *      addressable (asserted)
		 * @complexity O(node_count · sigma) time and 4·node_count·sigma
		 *             bytes, independent of pattern depth
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] dense_transitions transition_table(std::size_t sigma) const {
			atl_assert(sigma >= 1 && uint64(sigma) <= uint64(Alphabet));
			std::size_t const states = pool_.size();
			constexpr std::size_t max_cells = (std::numeric_limits<std::size_t>::max)() / sizeof(uint32);
			atl_assert(sigma <= max_cells / states);
			auto next = std::make_unique_for_overwrite<uint32[]>(states * sigma);
			// Every non-root row is copied in full from its completed suffix row.
			std::ranges::fill_n(next.get(), sigma, uint32(0));
			// Large tables use lookahead to overlap row and payload cache misses.
			if (states * sigma >= (std::size_t(1) << 18))
				fill_transitions<true>(next.get(), sigma);
			else
				fill_transitions<false>(next.get(), sigma);
			return detail::dense_transitions_builder::adopt(std::move(next), states * sigma, sigma);
		}

		/**
		 * @brief @ref atl::aho_corasick automata::transition_table over the whole alphabet: `table(v, c) ==
		 *        step(v, c)` for every node and every `c < Alphabet`.
		 * @pre `Alphabet > 0` (a constraint: a table needs at least one column).
		 * @complexity O(node_count · Alphabet) time and memory
		 */
		[[nodiscard]] dense_transitions transition_table() const
			requires(Alphabet > 0)
		{
			return transition_table(Alphabet);
		}

		/**
		 * @brief The pure trie edge: the node of @p v's prefix extended by
		 *        @p sym, or `std::nullopt` if that is not a pattern prefix
		 *        (unlike @ref atl::aho_corasick automata::step, no suffix-link fallback).
		 * @complexity O(log degree) expected
		 * @note Not constexpr (runtime node storage and hashing).
		 * @tparam Sym Integral symbol type.
		 * @param sym Numeric edge label; out-of-alphabet values miss or reset.
		 * @pre The node handle belongs to this object (asserted).
		 */
		template<std::integral Sym>
		[[nodiscard]] std::optional<node_id> walk(node_id v, Sym sym) const {
			atl_assert(v < pool_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return std::nullopt;
			uint32 const c = child(v, uint32(sym));
			return c == nil ? std::nullopt : std::optional<node_id>(c);
		}

		/** @brief The length of @p v's prefix (its trie depth); 0 at the root.
		 * @note Not constexpr (runtime node storage and hashing).
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] uint32 length(node_id v) const {
			atl_assert(v < pool_.size());
			return pool_[v].length;
		}

		/**
		 * @brief The output (dictionary-suffix) link: the node of the longest
		 *        PROPER suffix of @p v's prefix at which some pattern ends, or
		 *        `std::nullopt` if none does (always `std::nullopt` at the
		 *        root). Chaining it from a scan state visits exactly the
		 *        nodes whose patterns end at the current position.
		 * @note Not constexpr (runtime node storage and hashing).
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] std::optional<node_id> output_link(node_id v) const {
			atl_assert(v < pool_.size());
			return pool_[v].output == nil ? std::nullopt : std::optional<node_id>(pool_[v].output);
		}

		/**
		 * @brief The ids (insertion indices) of the patterns ending exactly at
		 *        @p v's prefix, ascending; `size() == word_count(v)`.
		 * @note Not constexpr (runtime node storage and hashing).
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] std::span<uint32 const> words_at(node_id v) const {
			atl_assert(v < pool_.size());
			return std::span<uint32 const>(word_ids_).subspan(word_offset_[v], word_offset_[v + 1] - word_offset_[v]);
		}

		/** @brief Patterns (with multiplicity) ending exactly at @p v's prefix.
		 * @note Not constexpr (runtime node storage and hashing).
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] std::size_t word_count(node_id v) const {
			atl_assert(v < pool_.size());
			return pool_[v].words;
		}

		/**
		 * @brief Patterns (with multiplicity) ending at @p v's prefix or at any of
		 *        its suffixes — the number of pattern occurrences ending at the
		 *        current position when @p v is the scan state. O(1): precomputed
		 *        down the suffix-link chain at construction.
		 * @note Not constexpr (runtime node storage and hashing).
		 * @pre The node handle belongs to this object (asserted).
		 */
		[[nodiscard]] std::size_t match_count(node_id v) const {
			atl_assert(v < pool_.size());
			return pool_[v].matches;
		}

		/**
		 * @brief The node of the longest proper suffix of @p v's prefix that is
		 *        also a pattern prefix (the root if only the empty one is).
		 * @pre `v != root()`
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] node_id suffix_link(node_id v) const {
			atl_assert(v != 0 && v < pool_.size());
			return pool_[v].link;
		}

		/**
		 * @brief @p v's prefix minus its last symbol.
		 * @pre `v != root()`
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] node_id parent(node_id v) const {
			atl_assert(v != 0 && v < pool_.size());
			return pool_[v].parent;
		}

		/**
		 * @brief The last symbol of @p v's prefix (its trie edge).
		 * @pre `v != root()`
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] uint32 symbol(node_id v) const {
			atl_assert(v != 0 && v < pool_.size());
			return pool_[v].symbol;
		}

		/** @brief The end node of the @p i-th inserted pattern (insertion order).
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] node_id word_node(std::size_t i) const {
			atl_assert(i < word_node_.size());
			return word_node_[i];
		}

		/**
		 * @brief All nodes in BFS order — root first, parents preceding children in
		 *        BOTH the trie and the suffix-link tree. The iteration link-tree
		 *        DPs want (e.g. per-pattern occurrence counting: seed scan states,
		 *        sweep this in reverse adding into suffix_link targets).
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] std::span<node_id const> nodes_by_depth() const { return by_depth_; }

		/** @brief Number of inserted patterns (with multiplicity).
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		std::size_t size() const { return word_node_.size(); }

		/** @brief Whether no pattern was inserted.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		bool empty() const { return word_node_.empty(); }

		/** @brief Number of nodes — distinct pattern prefixes, root included.
		 * @note Not constexpr (runtime node storage and hashing).
		 */
		[[nodiscard]] std::size_t node_count() const { return pool_.size(); }
	};
}
